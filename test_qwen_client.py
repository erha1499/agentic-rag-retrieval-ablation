"""零网络、零费用自检：价格、预算、账本与失败处理。"""

from concurrent.futures import ThreadPoolExecutor
import base64
import json
import math
from http.client import IncompleteRead, RemoteDisconnected
import os
from pathlib import Path
import tempfile
import threading
import struct
from unittest.mock import patch

from qwen_client import (BudgetError, CHAT_MODEL, Client, EMBED_MODEL,
                         RERANK_MODEL, UnknownUsageError, price)


def client_at(directory, cap=100):
    return Client(Path(directory), cap_cny=cap)


def raises(kind, callback):
    try:
        callback()
    except kind:
        return
    raise AssertionError("预期异常 " + kind.__name__)


def test_price():
    assert price(CHAT_MODEL, {"prompt_tokens": 128000, "completion_tokens": 1000}) == 0.1044
    assert math.isclose(price(CHAT_MODEL, {"prompt_tokens": 128001, "completion_tokens": 1000}), 0.3272024)
    assert math.isclose(price(CHAT_MODEL, {"prompt_tokens": 256001, "completion_tokens": 1000}), 1.2768048)
    assert price(EMBED_MODEL, {"total_tokens": 4}) == 0.000002
    assert price(RERANK_MODEL, {"total_tokens": 47}) == 0.0000235
    assert price('qwen3.7-text-embedding', {'prompt_tokens': 4}) == 0.000002
    assert price('qwen3.7-text-rerank', {'total_tokens': 47}) == 0.0000235
    assert price('text-embedding-v4', {'prompt_tokens': 4}) == 0.000002
    assert price('qwen3-rerank', {'total_tokens': 47}) == 0.0000235
    assert price(CHAT_MODEL, {}) is None
    assert price(CHAT_MODEL, None) is None
    assert price(CHAT_MODEL, {"prompt_tokens": True, "completion_tokens": 0}) is None


def test_success_and_probes():
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory)
        client.record_probe(CHAT_MODEL, {"prompt_tokens": 11, "completion_tokens": 1}, "probe-chat")
        client.record_probe(CHAT_MODEL, {"prompt_tokens": 11, "completion_tokens": 1}, "probe-chat")
        client.record_probe(EMBED_MODEL, {"total_tokens": 4}, "probe-embed")
        client.record_probe(RERANK_MODEL, {"total_tokens": 47}, "probe-rerank")
        assert client.summary()["calls"] == 3
        fake = {"usage": {"prompt_tokens": 11, "completion_tokens": 1},
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        with patch.object(client, "_send", return_value=(200, fake, "safe-id")):
            assert client.chat([{"role": "user", "content": "hello"}], [], "mock") == fake
        state = client.summary()
        assert state["calls"] == 4 and state["pending_calls"] == 0
        assert state["reserved_upper_cost_cny"] == 0
        assert abs(state["reported_cost_cny"] - 0.0000471) < 1e-12
        ledger = client.ledger.read_text()
        assert "TEST_SECRET_NEVER_LOG" not in ledger and "hello" not in ledger
        assert "private-workspace" not in ledger


def test_budget_and_threads():
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory, cap=0.15)
        gate = threading.Barrier(2)
        payload = {"model": CHAT_MODEL, "max_tokens": 2048}
        def reserve_once():
            gate.wait()
            try:
                client._reserve(CHAT_MODEL, payload, "parallel", 0)
                return True
            except BudgetError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sum(pool.map(lambda _: reserve_once(), range(2))) == 1
        assert client.summary()["pending_calls"] == 1
        reopened = client_at(directory, cap=1000)
        assert reopened.summary()["cap_cny"] == 0.15
        raises(BudgetError, lambda: reopened._reserve(CHAT_MODEL, payload, "later", 0))


def test_unknown_and_retries():
    answer = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    successful = {**answer, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory)
        with patch.object(client, "_send", return_value=(200, answer, "unknown")):
            assert client.chat([], [], "missing-usage") == answer
        assert client.summary()["unknown_calls"] == 1
        assert client.summary()["reported_cost_cny"] == 0
        assert client.summary()["reserved_upper_cost_cny"] > 0
        with patch.object(client, "_send", return_value=(200, successful, "later")) as send:
            client.chat([], [], "bounded-unknown-can-continue")
            assert send.call_count == 1
        state=client.summary()
        assert 0 < state['cost_lower_bound_cny'] < state['cost_upper_bound_cny'] < state['cap_cny']
        assert state['cost_upper_bound_cny'] == state['reported_cost_cny']+state['unknown_upper_bound_cny']
        assert state['pending_upper_bound_cny']==0 and state['unknown_calls']==1
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory)
        responses = [(429, {"error": {"code": "Throttling"}}, None),
                     (503, {"code": "ServiceUnavailable"}, None),
                     (200, successful, "final")]
        with patch.object(client, "_send", side_effect=responses) as send, patch("qwen_client.time.sleep") as sleep:
            client.chat([], [], "retry")
            assert send.call_count == 3
            assert [args.args[0] for args in sleep.call_args_list]==[1,2]
        assert client.summary()["unreported_calls"] == 1
        assert client.summary()["unknown_calls"] == 1
        calls = [json.loads(line) for line in client.ledger.read_text().splitlines()
                 if json.loads(line)["record_type"] == "api_call"]
        assert [c["retry_index"] for c in calls] == [0, 1, 2]
        assert [c['retry_delay_seconds'] for c in calls]==[1,2,0]
        assert calls[0]["public_price_cost_cny"] is None
        assert calls[1]['error_code']=='ServiceUnavailable'
    with tempfile.TemporaryDirectory() as directory:
        client=client_at(directory)
        limited=(429,{'error':{'code':'insufficient_quota'}},'tpm-limited')
        with patch.object(client,'_send',side_effect=[limited,limited,(200,successful,'recovered')]) as send, patch('qwen_client.time.sleep') as sleep:
            assert client.chat([],[],'tpm-backoff')==successful
            assert send.call_count==3 and [args.args[0] for args in sleep.call_args_list]==[35,70]
        calls=[r for r in map(json.loads,client.ledger.read_text().splitlines()) if r['record_type']=='api_call']
        assert [r['retry_delay_seconds'] for r in calls]==[35,70,0]
        assert calls[0]['usage_status']=='unreported' and calls[0]['public_price_cost_cny'] is None
        assert calls[0]['error_code']=='insufficient_quota'
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory)
        with patch.object(client, "_send", side_effect=TimeoutError("private-workspace TEST_SECRET_NEVER_LOG")) as send, patch('qwen_client.time.sleep'):
            raises(RuntimeError, lambda: client.chat([], [], "timeout"))
            assert send.call_count==3
        assert client.summary()['unknown_calls']==3
        assert "TEST_SECRET_NEVER_LOG" not in client.ledger.read_text()
        assert "private-workspace" not in client.ledger.read_text()
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory)
        with patch.object(client, "_send", side_effect=[(500, {'error':{'code':'InternalError'}}, None), (200,successful,'recovered')]) as send, patch('qwen_client.time.sleep'):
            assert client.chat([], [], "ambiguous-500") == successful
            assert send.call_count==2
        assert client.summary()["unknown_calls"] == 1 and client.summary()['cost_upper_bound_cny']>client.summary()['cost_lower_bound_cny']
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory,cap=0.25)
        with patch.object(client, '_send', return_value=(500,{},None)) as send, patch('qwen_client.time.sleep'):
            raises(BudgetError,lambda:client.chat([],[],'retry-cap'))
            assert send.call_count==2
        assert client.summary()['cost_upper_bound_cny']<=0.25
        assert client.summary()['unknown_calls']==2
    with tempfile.TemporaryDirectory() as directory:
        client=client_at(directory)
        with client._locked():
            client._append({'record_type':'api_call','reservation_id':None,'model':CHAT_MODEL,'label':'unbounded',
                'usage_status':'unknown','usage':None,'public_price_cost_cny':None,'reserved_upper_cost_cny':None})
        with patch.object(client,'_send') as send:
            raises(UnknownUsageError,lambda:client.chat([],[],'blocked-unbounded'))
            assert send.call_count==0
    with tempfile.TemporaryDirectory() as directory:
        client=client_at(directory)
        with patch.object(client,'_send',return_value=(200,{'choices':[]},'bad-schema')):
            raises(ValueError,lambda:client.chat([],[],'bad-schema'))
        assert client.summary()['unknown_calls']==1
    for interrupted in (IncompleteRead(b'TEST_SECRET_NEVER_LOG',100),
                        RemoteDisconnected('private-workspace TEST_SECRET_NEVER_LOG')):
        with tempfile.TemporaryDirectory() as directory:
            client=client_at(directory)
            with patch.object(client,'_send',side_effect=[interrupted,(200,successful,'read-recovered')]) as send, patch('qwen_client.time.sleep'):
                assert client.chat([],[],'interrupted-read')==successful
                assert send.call_count==2
            state=client.summary()
            assert state['unknown_calls']==1 and state['unknown_upper_bound_cny']>0
            assert state['cost_upper_bound_cny']==state['reported_cost_cny']+state['unknown_upper_bound_cny']
            assert 'TEST_SECRET_NEVER_LOG' not in client.ledger.read_text()
            assert 'private-workspace' not in client.ledger.read_text()


def test_vectors_and_rerank():
    with tempfile.TemporaryDirectory() as directory:
        client = client_at(directory)
        def send_embedding(url, payload):
            assert len(payload["input"]) <= 20
            assert payload['model']=='qwen3.7-text-embedding' and payload['encoding_format']=='base64'
            return (200, {"data": [{"index": i, "embedding": [float(i)] * 512}
                                    for i in reversed(range(len(payload["input"])))],
                          "usage": {"total_tokens": 12}}, "embedding")
        with patch.object(client, "_send", side_effect=send_embedding):
            vectors = client.embed(["text"] * 21, "batch-float-compatible")
            assert len(vectors) == 21 and vectors[19][0] == 19 and vectors[20][0] == 0
        encoded=base64.b64encode(struct.pack('<512f',*([0.25]*512))).decode('ascii')
        valid={'data':[{'index':0,'embedding':encoded}],'usage':{'total_tokens':12}}
        with patch.object(client,'_send',return_value=(200,valid,'base64-ok')):
            assert client.embed(['text'],'base64-ok')==[[0.25]*512]
        invalid_values=['%%%not-base64%%%',base64.b64encode(b'\x00'*2047).decode('ascii'),
            base64.b64encode(struct.pack('<512f',*([float('nan')]*512))).decode('ascii')]
        for malformed in invalid_values:
            invalid={'data':[{'index':0,'embedding':malformed}],'usage':{'total_tokens':12}}
            with patch.object(client,'_send',return_value=(200,invalid,'base64-invalid')):
                raises(ValueError,lambda:client.embed(['text'],'base64-invalid'))
        response = {"output": {"results": [{"index": 1, "relevance_score": 0.9},
                                             {"index": 0, "relevance_score": 0.2}]},
                    "usage": {"total_tokens": 3}}
        with patch.object(client, "_send", return_value=(200, response, "rerank")):
            assert client.rerank("query", ["one", "two"], "rank")[0]["index"] == 1


if __name__ == "__main__":
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "TEST_SECRET_NEVER_LOG",
        "DASHSCOPE_BASE_URL": "https://private-workspace.example/compatible-mode/v1"}):
        test_price()
        test_success_and_probes()
        test_budget_and_threads()
        test_unknown_and_retries()
        test_vectors_and_rerank()
    print("Qwen client 自检通过；未进行网络调用或产生 API 费用。")
