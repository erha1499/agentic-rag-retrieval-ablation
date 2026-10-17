"""Qwen 实验接口及可恢复的保守费用账本；不保存密钥、提示词或向量。"""

import base64
import binascii
import contextlib
import fcntl
from http.client import IncompleteRead, RemoteDisconnected
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


CHAT_MODEL = "qwen-plus-2025-12-01"
EMBED_MODEL = "qwen3.7-text-embedding"
RERANK_MODEL = "qwen3.7-text-rerank"
_THREAD_LOCK = threading.RLock()


class BudgetError(RuntimeError):
    pass


class UnknownUsageError(RuntimeError):
    pass


def price(model, usage):
    """缺失或无效用量返回 None，绝不把缺失用量当作零。"""
    def count(name):
        value = usage.get(name)
        return value if type(value) is int and value >= 0 else None

    if not isinstance(usage, dict):
        return None
    if model == CHAT_MODEL:
        incoming, outgoing = count("prompt_tokens"), count("completion_tokens")
        if incoming is None or outgoing is None:
            return None
        # 价格表中的 K 按 1000 计算，临界区间保守采用较高价格。
        input_rate, output_rate = ((0.8, 2) if incoming <= 128000 else
                                   (2.4, 20) if incoming <= 256000 else (4.8, 48))
        return (incoming * input_rate + outgoing * output_rate) / 1_000_000
    if model in ("text-embedding-v4", "qwen3-rerank",
                 "qwen3.7-text-embedding", "qwen3.7-text-rerank"):
        incoming = count("prompt_tokens")
        if incoming is None:
            incoming = count("total_tokens")
        return None if incoming is None else incoming * 0.5 / 1_000_000
    raise ValueError("未知模型")


class Client:
    def __init__(self, root: Path, cap_cny=3):
        self.root = Path(root)
        self.cap_cny = float(cap_cny)
        if not math.isfinite(self.cap_cny) or self.cap_cny <= 0:
            raise ValueError("预算必须为正数")
        self.ledger = self.root / "runs" / "api_calls.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("DASHSCOPE_API_KEY", "")
        self.base = os.environ.get("DASHSCOPE_BASE_URL", "").rstrip("/")
        if not self.key or not self.base:
            raise ValueError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_BASE_URL")
        parsed = urllib.parse.urlsplit(self.base)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("模型地址必须是无用户名密码的 HTTPS 地址")
        self.rerank_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc,
            "/api/v1/services/rerank/text-rerank/text-rerank", "", ""))
        with self._locked():
            records = self._records()
            if not any(r["record_type"] == "budget" for r in records):
                self._append({"record_type": "budget", "cap_cny": self.cap_cny})

    @contextlib.contextmanager
    def _locked(self):
        with _THREAD_LOCK:
            with self.ledger.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _records(self):
        if not self.ledger.exists():
            return []
        try:
            records = [json.loads(line) for line in self.ledger.read_text().splitlines() if line]
            if any(not isinstance(r, dict) or r.get("record_type") not in
                   ("budget", "reservation", "api_call") for r in records):
                raise ValueError("无效账本事件")
            return records
        except (ValueError, OSError) as error:
            raise BudgetError("账本不完整，需核验后才能继续付费") from error

    def _append(self, record):
        record = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        with self.ledger.open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def _summary(self, records):
        pending = {}
        calls = []
        cap = self.cap_cny
        for record in records:
            kind = record["record_type"]
            if kind == "budget":
                if type(record.get("cap_cny")) not in (int, float) or not math.isfinite(record["cap_cny"]) or record["cap_cny"] <= 0:
                    raise BudgetError("账本预算无效，需核验后才能继续付费")
                cap = min(cap, record["cap_cny"])
            elif kind == "reservation":
                pending[record["reservation_id"]] = record
            elif kind == "api_call":
                reported = record.get("public_price_cost_cny")
                if reported is not None and (type(reported) not in (int, float) or not math.isfinite(reported) or reported < 0):
                    raise BudgetError("账本已报费用无效，需核验后才能继续付费")
                if record.get("usage_status") not in ("reported", "unreported", "unknown") or (record.get("usage_status") == "reported" and reported is None):
                    raise BudgetError("账本用量状态无效，需核验后才能继续付费")
                pending.pop(record.get("reservation_id"), None)
                calls.append(record)
        unknown = [r for r in calls if r["usage_status"] == "unknown"]
        for record in [*pending.values(), *unknown]:
            upper = record.get("reserved_upper_cost_cny")
            if type(upper) not in (int, float) or not math.isfinite(upper) or upper <= 0:
                raise UnknownUsageError("未知或未完成请求缺少有效费用上界，需核验后才能继续付费")
        cost = sum(r["public_price_cost_cny"] for r in calls
                   if r["public_price_cost_cny"] is not None)
        pending_upper = sum(r["reserved_upper_cost_cny"] for r in pending.values())
        unknown_upper = sum(r["reserved_upper_cost_cny"] for r in unknown)
        held = pending_upper + unknown_upper
        models = {}
        for record in calls:
            model = models.setdefault(record["model"], {
                "calls": 0, "reported_cost_cny": 0, "unreported_calls": 0,
                "unknown_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0})
            model["calls"] += 1
            if record["public_price_cost_cny"] is not None:
                model["reported_cost_cny"] += record["public_price_cost_cny"]
            else:
                model[record["usage_status"] + "_calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = (record.get("usage") or {}).get(key)
                if type(value) is int and value >= 0:
                    model[key] += value
        return {"cap_cny": cap, "calls": len(calls), "reported_cost_cny": cost,
                "reserved_upper_cost_cny": held, "remaining_cny": cap - cost - held,
                "cost_lower_bound_cny": cost, "cost_upper_bound_cny": cost + held,
                "unknown_upper_bound_cny": unknown_upper,
                "pending_upper_bound_cny": pending_upper,
                "unknown_calls": len(unknown),
                "unreported_calls": sum(r["usage_status"] == "unreported" for r in calls),
                "pending_calls": len(pending), "by_model": models}

    def summary(self):
        with self._locked():
            return self._summary(self._records())

    def _reserve(self, model, payload, label, retry_index):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # UTF-8 字节上界加协议余量；按最高聊天档位预留，结算按实际用量。
        tokens = len(encoded) + 1024
        if model == CHAT_MODEL:
            upper = (tokens * 4.8 + payload["max_tokens"] * 48) / 1_000_000
        elif model in (RERANK_MODEL, 'qwen3-rerank', 'qwen3.7-text-rerank'):
            data = payload["input"]
            tokens = sum(len((data["query"] + doc).encode("utf-8")) + 1024
                         for doc in data["documents"])
            upper = tokens * 0.5 / 1_000_000
        else:
            upper = tokens * 0.5 / 1_000_000
        reservation = {"record_type": "reservation", "reservation_id": uuid.uuid4().hex,
                       "model": model, "label": label, "retry_index": retry_index,
                       "reserved_upper_cost_cny": upper}
        with self._locked():
            state = self._summary(self._records())
            if upper > state["remaining_cny"]:
                raise BudgetError("新增请求的保守费用预留将超过预算")
            self._append(reservation)
        return reservation

    def _send(self, url, payload):
        request = urllib.request.Request(url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.key,
                     "Content-Type": "application/json"}, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read()
            status = response.status
            request_id = response.headers.get("x-request-id")
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            data = {}
        return status, data if isinstance(data, dict) else {}, request_id

    def _call(self, url, payload, label):
        model = payload["model"]
        for retry in range(3):
            reservation = self._reserve(model, payload, label, retry)
            start = time.monotonic()
            status, data, request_id, failure, transport_retry = None, {}, None, None, False
            try:
                status, data, request_id = self._send(url, payload)
            except Exception as error:
                # 异常文本可能含地址或认证信息，只记录异常类型。
                failure = type(error).__name__
                transport_retry = isinstance(error, (TimeoutError, urllib.error.URLError,
                                                    ConnectionError, IncompleteRead, RemoteDisconnected))
            usage = data.get("usage")
            cost = price(model, usage)
            error = data.get("error")
            code = error.get("code", "") if isinstance(error, dict) else data.get("code", "")
            logged_code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", code) else None
            if logged_code and self.key in logged_code:
                logged_code = "redacted"
            refused = status is not None and (400 <= status < 500 and status != 408)
            usage_status = "reported" if cost is not None else "unreported" if refused else "unknown"
            will_retry = (status in (429, 500, 502, 503, 504) or transport_retry) and retry < 2
            retry_delay = ((35 if status == 429 and code == 'insufficient_quota' else 1) * 2 ** retry
                           if will_retry else 0)
            record = {**reservation, "record_type": "api_call", "http_status": status,
                      "request_id": request_id or data.get("request_id") or data.get("id"),
                      "usage": usage, "usage_status": usage_status,
                      "latency_seconds": time.monotonic() - start,
                      "public_price_cost_cny": cost, "error_type": failure,
                      "error_code": logged_code, "retry_delay_seconds": retry_delay}
            with self._locked():
                self._append(record)
            if status is not None and 200 <= status < 300:
                return data
            if will_retry:
                time.sleep(retry_delay)
                continue
            raise RuntimeError("Qwen 请求失败，HTTP " + str(status))

    def chat(self, messages, tools, label, max_tokens=2048):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise ValueError("无效输出 token 上限")
        payload = {"model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens,
                   "enable_thinking": False, "temperature": 0}
        if tools:
            payload["tools"] = tools
        response = self._call(self.base + "/chat/completions", payload, label)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict) or not isinstance(choices[0].get("message"), dict):
            raise ValueError("聊天响应缺少有效 choices/message")
        return response

    def embed(self, texts, label):
        if not isinstance(texts, list) or any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("向量输入必须是非空文本列表")
        result = []
        for start in range(0, len(texts), 20):
            batch = texts[start:start + 20]
            response = self._call(self.base + "/embeddings",
                {"model": EMBED_MODEL, "input": batch, "dimensions": 512,
                 "encoding_format": "base64"}, label)
            vectors = sorted(response.get("data", []), key=lambda item: item["index"])
            if [v["index"] for v in vectors] != list(range(len(batch))):
                raise ValueError("向量响应序号缺失或重复")
            for vector in vectors:
                values = vector.get("embedding")
                if isinstance(values, str):
                    try:
                        raw = base64.b64decode(values, validate=True)
                    except (ValueError, binascii.Error) as error:
                        raise ValueError("向量 base64 编码无效") from error
                    if len(raw) != 512 * 4:
                        raise ValueError("向量 base64 二进制长度不是 512 个 float32")
                    values = list(struct.unpack("<512f", raw))
                if not isinstance(values, list) or len(values) != 512 or any(
                        type(v) not in (float, int) or not math.isfinite(v) for v in values):
                    raise ValueError("向量响应不是 512 维有限数值")
                result.append(values)
        return result

    def rerank(self, query, documents, label):
        if not isinstance(query, str) or not query or not documents or any(
                not isinstance(t, str) or not t for t in documents):
            raise ValueError("精排需要非空查询与文档列表")
        response = self._call(self.rerank_url,
            {"model": RERANK_MODEL, "input": {"query": query, "documents": documents},
             "parameters": {"top_n": len(documents), "return_documents": False}}, label)
        results = response.get("output", {}).get("results", response.get("results", []))
        if len(results) != len(documents) or {r.get("index") for r in results} != set(range(len(documents))):
            raise ValueError("精排响应序号缺失或重复")
        if any(type(r.get("relevance_score")) not in (int, float) or
               not math.isfinite(r["relevance_score"]) for r in results):
            raise ValueError("精排分数无效")
        return [{"index": r["index"], "relevance_score": r["relevance_score"]} for r in results]

    def record_probe(self, model, usage, label):
        """补录本轮已发生的成功探测；相同标签与模型不会重复入账。"""
        cost = price(model, usage)
        if cost is None:
            raise ValueError("探测补录必须有明确用量")
        with self._locked():
            records = self._records()
            if any(r.get("label") == label and r.get("model") == model for r in records):
                return
            self._append({"record_type": "api_call", "reservation_id": None,
                "model": model, "label": label, "request_id": None, "http_status": 200,
                "usage": usage, "usage_status": "reported", "latency_seconds": None,
                "public_price_cost_cny": cost, "reserved_upper_cost_cny": 0,
                "retry_index": 0, "provenance": "previous_live_probe_usage"})
