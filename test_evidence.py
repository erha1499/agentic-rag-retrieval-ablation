"""零网络检验：区间、已读引用、单次修复与预算停止。"""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import run_experiment as runner
from qwen_client import BudgetError, UnknownUsageError


def answer(text, finish='stop'):
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': finish}]}


def read_call(start=2, end=3):
    return {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'read-1', 'type': 'function', 'function': {'name': 'read_file',
         'arguments': json.dumps({'path': 'docs/sample.txt', 'start_line': start, 'end_line': end})}}]},
        'finish_reason': 'tool_calls'}]}


def grep_call():
    return {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'grep-1', 'type': 'function', 'function': {'name': 'grep',
         'arguments': json.dumps({'pattern': 'rule one', 'scope': 'doc', 'context_lines': 0})}}]},
        'finish_reason': 'tool_calls'}]}


def skill_call():
    return {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'skill-1', 'type': 'function', 'function': {'name': 'load_skill',
         'arguments': json.dumps({'name': '检索流程'})}}]}, 'finish_reason': 'tool_calls'}]}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, tools, **kwargs):
        self.requests.append(copy.deepcopy((messages, tools, kwargs)))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


CONFIG = {'model': 'fake', 'max_model_calls': 8, 'max_completion_tokens': 1200,
          'time_limit_seconds': 180, 'development_time_limit_seconds': 180, 'budget_cap_cny': 3}
CASE = {'id': 'C01', 'kind': 'doc', 'question': '只根据本地材料回答。',
        'required_facts': ['隐藏金标准不得传给模型']}


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    corpus = root / 'original-corpus'
    (corpus / 'docs').mkdir(parents=True); (corpus / 'code').mkdir()
    data = root / 'data'; data.mkdir()
    (data / 'corpus').symlink_to(corpus, target_is_directory=True)
    (corpus / 'docs/sample.txt').write_text('heading\nrule one\nrule two\nend\n')
    (corpus / 'docs/large.txt').write_text(('line-' + 'x' * 400 + '\n') * 40)
    (corpus / 'docs/escape.txt').symlink_to('/etc/passwd')
    skill_file = root / runner.SKILLS['检索流程']
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text('---\nname: 检索流程\ndescription: 查阅资料时选择检索方法。\n---\n\n技能正文只在加载后出现。')
    (root / 'prompts').mkdir()
    (root / 'prompts/系统提示词.md').write_text('你是数据库文档助手。')
    tools = runner.Tools(root)
    skill = tools.call('load_skill', {'name': '检索流程'}, label='offline')
    assert '技能正文只在加载后出现' in skill['text']
    assert not runner.returned_windows('load_skill', skill)
    initial = runner.system_prompt(root)
    assert '查阅资料时选择检索方法' in initial and '技能正文只在加载后出现' not in initial
    for name in ('未注册', '../检索流程', '/etc/passwd', None):
        try:
            tools.call('load_skill', {'name': name}, label='offline')
        except ValueError:
            pass
        else:
            raise AssertionError('未拒绝非法技能名')
    original_skill = skill_file.read_text()
    (root / 'secret.md').write_text('不能作为技能读取的资料')
    skill_file.unlink()
    skill_file.symlink_to(root / 'secret.md')
    try:
        tools.call('load_skill', {'name': '检索流程'}, label='offline')
    except ValueError:
        pass
    else:
        raise AssertionError('技能文件通过符号链接越界')
    skill_file.unlink()
    skill_file.write_text(original_skill)
    local = tools.read_file('docs/sample.txt', 2, 3)
    assert local['text'] == '2: rule one\n3: rule two' and local['read_mode'] == 'range'
    assert local['start_line'] == 2 and local['end_line'] == 3
    assert tools.read_file('docs/sample.txt')['read_mode'] == 'whole'
    capped = tools.read_file('docs/large.txt')
    assert capped.get('error') and 'text' not in capped
    assert len(tools.read_file('docs/large.txt', 1, 10)['text']) < runner.READ_CHAR_LIMIT
    for args in [('docs/sample.txt', 0, 1), ('docs/sample.txt', 3, 2),
                 ('docs/sample.txt', 1, 5), ('docs/sample.txt', True, 2),
                 ('../secret', None, None), ('docs/escape.txt', None, None),
                 ('/etc/passwd', None, None)]:
        try:
            tools.read_file(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('边界未拒绝：' + str(args))
    listing = tools.grep('rule', files_only=True)
    assert listing['files'] == ['docs/sample.txt']
    assert listing['file_info'] == [{'path': 'docs/sample.txt', 'chars': 30, 'total_lines': 4}]
    assert not runner.returned_windows('read_file', capped)
    assert not runner.returned_windows('grep', listing)
    rows = tools.grep('rule one', context_lines=0)
    assert runner.returned_windows('grep', rows) == [('docs/sample.txt', 2, 2)]
    assert runner.returned_windows('read_file', local) == [('docs/sample.txt', 2, 3)]
    assert runner.returned_windows('rag_search', {'results': [
        {'path': 'docs/sample.txt', 'start_line': 2, 'end_line': 3, 'text': 'rule one\nrule two'}
    ]}) == [('docs/sample.txt', 2, 3)]

    windows = [('docs/sample.txt', 2, 2), ('docs/sample.txt', 3, 3)]
    assert runner.validate_citations('规则【docs/sample.txt:L2-L3】', windows)['passed']
    single = runner.validate_citations('规则【docs/sample.txt:L2】', windows)
    assert single['passed'] and single['citations'] == [
        {'path': 'docs/sample.txt', 'start_line': 2, 'end_line': 2}]
    assert not runner.validate_citations('【docs/sample.txt:L1】', windows)['passed']
    assert not runner.validate_citations('【docs/sample.txt:L0】', windows)['passed']
    for text in ('未引用', '【docs/sample.txt:L1-L3】', '【docs/sample.txt:L0-L2】',
                 '【docs/sample.txt:L3-L2】', '【docs/sample.txt:2-3】',
                 '【docs/../sample.txt:L2-L3】', '【code/missing.c:L2-L3】'):
        assert not runner.validate_citations(text, windows)['passed'], text
    assert not runner.validate_citations('【docs/sample.txt:L2-L4】', windows)['passed']
    assert not runner.validate_citations('【docs/sample.txt:L2-L4】',
        [('docs/sample.txt', 2, 2), ('docs/sample.txt', 4, 4)])['passed']

    with patch.object(runner, 'ROOT', root):
        good = '规则一和规则二。【docs/sample.txt:L2-L3】'
        client = FakeClient([read_call(), answer('错误行号【docs/sample.txt:L1-L3】'), answer(good)])
        result = runner.execute(CASE, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'complete' and result['repair_count'] == 1
        assert len(result['answer_attempts']) == 2
        assert result['range_reads'] == 1 and result['whole_reads'] == result['full_file_read_chars'] == 0
        assert result['executed'] and client.requests[-1][1] == []
        assert '隐藏金标准' not in json.dumps(client.requests, ensure_ascii=False)

        case = {**CASE, 'id': 'C02'}
        client = FakeClient([answer('先验回答'), answer('仍然没有证据')])
        result = runner.execute(case, 'B', 'test', client, None, CONFIG)
        assert result['status'] == 'evidence_validation_failed' and result['repair_count'] == 1
        assert len(client.requests) == len(result['answer_attempts']) == 2
        assert client.requests[0][0][0]['content'] == runner.SYSTEM
        assert client.requests[1][1] and result['repair_mode'] == 'retrieve_evidence'
        assert '先调用检索工具取证' in client.requests[1][0][-1]['content']

        client = FakeClient([read_call(), answer(good, 'length'), answer(good)])
        result = runner.execute({**CASE, 'id': 'C03'}, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'complete' and result['repair_count'] == 1
        assert result['answer_attempts'][0]['finish_reason'] == 'length'

        client = FakeClient([read_call(), answer(good, 'length'), answer(good, 'length')])
        result = runner.execute({**CASE, 'id': 'C04'}, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'incomplete_answer' and len(client.requests) == 3
        assert result['repair_count'] == 1

        client = FakeClient([read_call(), answer(good)])
        result = runner.execute({**CASE, 'id': 'C05'}, 'A', 'test', client, None,
                                {**CONFIG, 'max_model_calls': 2})
        assert result['status'] == 'complete' and client.requests[-1][1] == []

        client = FakeClient([BudgetError('预算不足，未发请求')])
        result = runner.execute({**CASE, 'id': 'C06'}, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'budget_blocked' and not result['executed']
        assert result['stop_scheduling'] and result['api_calls'] == 0

        client = FakeClient([read_call(), UnknownUsageError('需要核验用量上界')])
        result = runner.execute({**CASE, 'id': 'C07'}, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'budget_blocked' and result['executed']
        assert result['error_type'] == 'UnknownUsageError'

        client = FakeClient([read_call(1, 4), answer('规则【docs/sample.txt:L1-L4】')])
        result = runner.execute({**CASE, 'id': 'C08'}, 'A', 'test', client, None, CONFIG)
        assert result['whole_reads'] == 1 and result['range_reads'] == 0
        assert result['full_file_read_chars'] > 0

        client = FakeClient([answer('先验回答'), grep_call(), answer('规则一。【docs/sample.txt:L2-L2】')])
        result = runner.execute({**CASE, 'id': 'C09'}, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'complete' and result['repair_count'] == 1
        assert result['repair_mode'] == 'retrieve_evidence'
        assert result['grep_calls'] == 1 and result['received_windows'] == [('docs/sample.txt', 2, 2)]
        assert client.requests[1][1] and len(client.requests) == 3
        assert len(result['answer_attempts']) == 2

        client = FakeClient([answer('先验回答'), answer('仍无证据')])
        result = runner.execute({**CASE, 'id': 'C10'}, 'A', 'test', client, None,
                                {**CONFIG, 'max_model_calls': 2})
        assert result['status'] == 'evidence_validation_failed' and result['repair_count'] == 1
        assert len(client.requests) == 2 and not client.requests[-1][1]
        assert '已经收到的证据' in client.requests[-1][0][-1]['content']

        arm_requests = {}
        for arm in ('A', 'B'):
            client = FakeClient([skill_call(), read_call(), answer(good)])
            result = runner.execute({**CASE, 'id': 'SKILL_' + arm}, arm, 'test', client, None, CONFIG)
            assert result['status'] == 'complete' and result['skill_calls'] == 1
            assert result['loaded_skills'] == ['检索流程'] and result['tool_calls'] == 2
            assert result['received_windows'] == [('docs/sample.txt', 2, 3)]
            assert '技能正文只在加载后出现' not in client.requests[0][0][0]['content']
            assert '技能正文只在加载后出现' in client.requests[1][0][-1]['content']
            arm_requests[arm] = client.requests[0]
        assert arm_requests['A'][0] == arm_requests['B'][0]
        a_tools = arm_requests['A'][1]
        b_tools = arm_requests['B'][1]
        assert a_tools == b_tools[:-1]
        assert [t['function']['name'] for t in a_tools] == ['load_skill', 'grep', 'read_file']
        assert b_tools[-1]['function']['name'] == 'rag_search'

        client = FakeClient([skill_call(), answer('只读技能不能当作证据'), answer('仍无语料证据')])
        result = runner.execute({**CASE, 'id': 'SKILL_ONLY'}, 'A', 'test', client, None, CONFIG)
        assert result['status'] == 'evidence_validation_failed' and result['received_windows'] == []
        assert result['skill_calls'] == 1 and result['repair_mode'] == 'retrieve_evidence'

    cases = [{**CASE, 'id': f'S{i}'} for i in range(3)]
    calls = []
    def blocked(case, arm, *args):
        calls.append((case['id'], arm))
        assert set(case) == {'id', 'kind', 'question'}
        return {'status': 'budget_blocked' if len(calls) == 3 else 'complete'}
    with patch.object(runner, 'execute', blocked):
        results = runner.schedule(cases, 'test', None, None, CONFIG)
    assert len(results) == 3 and calls == [('S0', 'A'), ('S0', 'B'), ('S1', 'B')]
    try:
        runner.select_cases(cases, ['MISSING'])
    except ValueError:
        pass
    else:
        raise AssertionError('未知题号没有拒绝')
    assert runner.select_cases(cases, ['S1']) == [cases[1]]
print('技能加载、A/B隔离、区间读取、已读引用、一次共享修复与预算停止检查通过；未发起网络请求')
