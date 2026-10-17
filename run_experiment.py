"""冻结语料上的场景检索工具消融；串行运行并保留失败与原始答案。"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from qwen_client import Client, BudgetError, UnknownUsageError, CHAT_MODEL
from retrieval import Retriever

ROOT = Path(__file__).resolve().parent
SYSTEM = (ROOT / 'prompts/检索流程.md').read_text()
READ_CHAR_LIMIT = 12000

GREP = {'type': 'function', 'function': {'name': 'grep', 'description':
    '对本地语料做正则检索；代码校验与实现细节优先使用。命中返回逐行原文；files_only返回路径及文件大小、行数。',
    'parameters': {'type': 'object', 'properties': {
        'pattern': {'type': 'string', 'description': 'ripgrep正则；可用|组合术语'},
        'scope': {'type': 'string', 'enum': ['all', 'doc', 'code']},
        'path_glob': {'type': 'string', 'description': '可选，如**/nodeLimit.c或docs/*isolation*'},
        'context_lines': {'type': 'integer', 'minimum': 0, 'maximum': 20},
        'max_matches': {'type': 'integer', 'minimum': 1, 'maximum': 200},
        'files_only': {'type': 'boolean'}, 'case_sensitive': {'type': 'boolean'}},
        'required': ['pattern']}}}
READ = {'type': 'function', 'function': {'name': 'read_file', 'description':
    '读取已定位文件，返回逐行原文。小文件可省略行范围读取全文；大文件先grep，再用start_line/end_line读取局部。返回最多12000字符，超出报错且不截断。',
    'parameters': {'type': 'object', 'properties': {
        'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1},
        'end_line': {'type': 'integer', 'minimum': 1}}, 'required': ['path']}}}
RAG = {'type': 'function', 'function': {'name': 'rag_search', 'description':
    '当文档量大、问题只需局部语义且术语不确定时，混合检索并精排局部父块。全局概览不能仅凭top-k片段判断覆盖完整；代码精确条件仍须grep核验。',
    'parameters': {'type': 'object', 'properties': {
        'query': {'type': 'string', 'maxLength': 2000},
        'scope': {'type': 'string', 'enum': ['all', 'doc', 'code']},
        'top_k': {'type': 'integer', 'enum': [3, 6, 10]}}, 'required': ['query']}}}


class Tools:
    def __init__(self, root, retriever=None):
        self.root = Path(root).resolve()
        # 本轮语料根目录是复用已核验语料的链接；文件仍必须留在其真实根目录内。
        self.corpus = (self.root / 'data/corpus').resolve()
        self.retriever = retriever

    def member(self, path):
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError('path必须是语料内的相对路径')
        candidate = (self.corpus / path).resolve()
        if not candidate.is_relative_to(self.corpus) or not candidate.is_file():
            raise ValueError('路径必须是语料内存在的文件，不能越界')
        return candidate

    def call(self, name, args, label):
        if not isinstance(args, dict):
            raise ValueError('工具参数必须是对象')
        if name == 'read_file':
            return self.read_file(**args)
        if name == 'grep':
            return self.grep(**args)
        if name == 'rag_search' and self.retriever:
            return self.retriever.search(**args, label=label)
        raise ValueError('工具不可用：' + name)

    def read_file(self, path, start_line=None, end_line=None):
        candidate = self.member(path)
        raw = candidate.read_text()
        lines = raw.splitlines()
        for value in (start_line, end_line):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError('行号必须是正整数')
        lo = 1 if start_line is None else start_line
        hi = len(lines) if end_line is None else end_line
        if lines and not 1 <= lo <= hi <= len(lines):
            raise ValueError(f'行范围必须满足1 <= start_line <= end_line <= {len(lines)}')
        if not lines and (start_line is not None or end_line is not None):
            raise ValueError('空文件没有可读取的行范围')
        text = '\n'.join(f'{i}: {lines[i-1]}' for i in range(lo, hi + 1))
        metadata = {'path': candidate.relative_to(self.corpus).as_posix(),
                    'total_lines': len(lines), 'file_chars': len(raw),
                    'start_line': lo, 'end_line': hi}
        if len(text) > READ_CHAR_LIMIT:
            return {**metadata, 'error': '返回将超过12000字符；请先grep定位，再缩小start_line/end_line范围。没有返回任何文件正文。'}
        return {**metadata, 'read_mode': 'whole' if lo == 1 and hi == len(lines) else 'range',
                'truncated': False, 'text': text}

    def grep(self, pattern, scope='all', path_glob='', context_lines=3, max_matches=50,
             files_only=False, case_sensitive=False):
        if not isinstance(pattern, str) or len(pattern) > 1000 or not pattern:
            raise ValueError('需要非空且不超过1000字符的正则')
        if (scope not in ('all', 'doc', 'code') or type(context_lines) is not int
                or not 0 <= context_lines <= 20 or type(max_matches) is not int
                or not 1 <= max_matches <= 200 or not isinstance(path_glob, str)
                or len(path_glob) > 500 or type(files_only) is not bool
                or type(case_sensitive) is not bool):
            raise ValueError('grep参数无效')
        cmd = (['rg', '--files-with-matches', '--sort', 'path'] if files_only else
               ['rg', '--json', '--sort', 'path', '--max-count', str(max_matches + 1), '-C', str(context_lines)])
        if not case_sensitive:
            cmd.append('-i')
        if path_glob:
            cmd += ['--glob', path_glob]
        cmd += ['--', pattern, *{'all': ['docs', 'code'], 'doc': ['docs'], 'code': ['code']}[scope]]
        process = subprocess.run(cmd, cwd=self.corpus, text=True, capture_output=True, timeout=15)
        if process.returncode not in (0, 1):
            return {'error': process.stderr[:2000], 'matches': []}
        if files_only:
            paths = process.stdout.splitlines()
            info = []
            for path in paths[:max_matches]:
                raw = self.member(path).read_text()
                info.append({'path': path, 'chars': len(raw), 'total_lines': len(raw.splitlines())})
            return {'files': paths[:max_matches], 'file_info': info,
                    'file_count': min(len(paths), max_matches), 'truncated': len(paths) > max_matches}
        matches = []; contexts = []; count = 0; truncated = False
        for line in process.stdout.splitlines():
            event = json.loads(line); data = event['data']
            if event['type'] not in ('match', 'context'):
                continue
            item = {'path': data['path'].get('text', ''), 'line': data.get('line_number'),
                    'text': data['lines'].get('text', '').rstrip('\n')}
            if event['type'] == 'match':
                count += 1
                if count > max_matches:
                    truncated = True
                    break
                matches.append(item)
            elif count <= max_matches:
                contexts.append(item)
        wanted = {(m['path'], i) for m in matches
                  for i in range(max(1, m['line'] - context_lines), m['line'] + context_lines + 1)}
        items = {(m['path'], m['line']): m for m in contexts if (m['path'], m['line']) in wanted}
        items.update({(m['path'], m['line']): m for m in matches})
        return {'lines': [items[key] for key in sorted(items)],
                'match_count': len(matches), 'truncated': truncated}


def returned_windows(name, value):
    """只记录实际成功返回的正文；文件名列表和失败读取不构成已读证据。"""
    if value.get('error'):
        return []
    if name == 'grep':
        return [(row['path'], row['line'], row['line']) for row in value.get('lines', [])]
    if name == 'read_file' and value.get('text'):
        return [(value['path'], value['start_line'], value['end_line'])]
    if name == 'rag_search':
        return [(row['path'], row['start_line'], row['end_line'])
                for row in value.get('results', []) if row.get('text')]
    return []


def validate_citations(answer, windows):
    """仅校验引用位置已读，不把位置匹配当作事实或语义支持的判断。"""
    citations = []; errors = []
    for content in re.findall(r'【([^】]+)】', answer or ''):
        if not content.startswith(('docs/', 'code/')):
            continue
        match = re.fullmatch(r'((?:docs|code)/[^:\s]+):L(\d+)(?:-L(\d+))?', content)
        if not match:
            errors.append('引用格式无效：' + content)
            continue
        path, lo, hi = match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))
        citation = {'path': path, 'start_line': lo, 'end_line': hi}
        citations.append(citation)
        if lo < 1 or hi < lo or '..' in Path(path).parts:
            errors.append('引用范围无效：' + content)
            continue
        cursor = lo
        for _, start, end in sorted((w for w in windows if w[0] == path), key=lambda w: w[1]):
            if start > cursor:
                break
            if end >= cursor:
                cursor = end + 1
            if cursor > hi:
                break
        if cursor <= hi:
            errors.append('引用包含未收到的行：' + content)
    if not citations:
        errors.append('缺少固定格式引用；至少引用一处实际取得的正文证据')
    return {'passed': bool(citations) and not errors, 'citations': citations, 'errors': errors,
            'scope': '只验证位置已读，不验证内容正确或语义支持'}


def executed_api_calls(client, label, fallback=0):
    """BudgetError可能发生在发出请求前；真实账本是是否已执行的依据。"""
    ledger = getattr(client, 'ledger', None)
    if ledger is None:
        return fallback
    return sum(record.get('record_type') == 'api_call' and
               str(record.get('label', '')).startswith(label + ':')
               for record in (json.loads(line) for line in Path(ledger).read_text().splitlines() if line))


def execute(case, arm, split, client, retriever, config):
    config = dict(config)
    if split == 'dev':
        config['time_limit_seconds'] = config.get('development_time_limit_seconds', config['time_limit_seconds'])
    output = ROOT / 'runs' / split / f"{case['id']}_{arm}.json"
    output.parent.mkdir(exist_ok=True, parents=True)
    if output.exists():
        prior = json.loads(output.read_text())
        if prior.get('status') != 'running':
            return prior
        raise RuntimeError('存在未核验的中断任务：' + str(output))
    label = f'{split}:{case["id"]}:{arm}'
    tools = Tools(ROOT, retriever if arm == 'B' else None)
    schemas = [GREP, READ] + ([RAG] if arm == 'B' else [])
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': case['question']}]
    result = {'id': case['id'], 'arm': arm, 'split': split, 'kind': case['kind'], 'question': case['question'],
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'limits': {key: config[key] for key in ('max_model_calls', 'max_completion_tokens', 'time_limit_seconds')},
              'allowed_tools': [schema['function']['name'] for schema in schemas],
              'status': 'running', 'model': config['model'], 'events': [], 'final_answer': None,
              'tool_calls': 0, 'grep_calls': 0, 'read_calls': 0, 'rag_calls': 0,
              'whole_reads': 0, 'range_reads': 0, 'tool_return_chars': 0, 'full_file_read_chars': 0,
              'received_windows': [], 'answer_attempts': [], 'repair_count': 0,
              'model_responses': 0, 'executed': False}
    started = time.monotonic(); repair_pending = None

    def save():
        result['seconds'] = round(time.monotonic() - started, 3)
        result['messages'] = messages
        result['last_saved_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2)); temp.replace(output)

    save()
    try:
        for step in range(config['max_model_calls']):
            if time.monotonic() - started > config['time_limit_seconds']:
                result['status'] = 'timeout'
                break
            final_turn = step == config['max_model_calls'] - 1 or repair_pending == 'answer_only'
            if step == config['max_model_calls'] - 1 and repair_pending != 'answer_only':
                messages.append({'role': 'user', 'content': '已达到本题调用上限。只能依据已经收到的证据简洁作答，并使用固定格式行号引用；证据不足请明确说明，不得新增推测。'})
            offered = [] if final_turn else schemas
            before = time.monotonic()
            response = client.chat(messages, offered, label=label + f':step{step}',
                                   max_tokens=config['max_completion_tokens'])
            result['model_responses'] += 1
            choice = response['choices'][0]; message = choice['message']
            messages.append({key: message[key] for key in ('role', 'content', 'tool_calls') if key in message})
            result['events'].append({'type': 'model', 'step': step, 'seconds': round(time.monotonic() - before, 3),
                'offered_tools': [schema['function']['name'] for schema in offered],
                'usage': response.get('usage'), 'finish_reason': choice.get('finish_reason'),
                'model': response.get('model'), 'request_id': response.get('id')})
            calls = message.get('tool_calls') or []
            late = time.monotonic() - started > config['time_limit_seconds']
            if not calls:
                answer = message.get('content') or ''
                validation = validate_citations(answer, result['received_windows'])
                result['answer_attempts'].append({'step': step, 'answer': answer,
                    'finish_reason': choice.get('finish_reason'), 'evidence_validation': validation})
                result['final_answer'] = answer
                result['evidence_validation'] = validation
                incomplete = not answer or choice.get('finish_reason') == 'length'
                result['status'] = 'incomplete_answer' if incomplete else (
                    'complete' if validation['passed'] else 'evidence_validation_failed')
                if late:
                    result['status'] = 'timeout_late_answer'
                    break
                if (incomplete or not validation['passed']) and not result['repair_count'] and step + 1 < config['max_model_calls']:
                    result['repair_count'] = 1
                    repair_pending = 'answer_only' if result['received_windows'] else 'retrieve_evidence'
                    result['repair_mode'] = repair_pending
                    issues = '；'.join(validation['errors'][:4])
                    instruction = (
                        '仅有这一次修正机会，不再检索。' if repair_pending == 'answer_only' else
                        '这是唯一一次反馈。目前尚未取得任何正文证据。先调用检索工具取证，再根据实际返回的正文作答；不要直接重写先验回答。')
                    messages.append({'role': 'user', 'content':
                        instruction + '请压缩为300—600中文字以内，只保留回答所需的已取证结论；'
                        '每个关键结论使用【相对路径:L起始行-L结束行】格式，范围必须全部来自实际收到的正文。'
                        '没有证据的内容删除并说明不足，不能编造引用。' + ('上次输出被截断。' if incomplete else '') + issues})
                    result['status'] = 'running'
                    save()
                    continue
                break
            if not offered:
                result['status'] = 'incomplete_answer'
                result['error'] = '最终回答轮仍返回工具调用；未执行这些调用'
                break
            if late:
                result['status'] = 'timeout'
                break
            for call in calls:
                if time.monotonic() - started > config['time_limit_seconds']:
                    result['status'] = 'timeout'
                    break
                name = call['function']['name']; raw = call['function']['arguments']; before = time.monotonic()
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    value = tools.call(name, args, label=label + f':tool{result["tool_calls"]}')
                except (BudgetError, UnknownUsageError):
                    raise
                except Exception as error:
                    value = {'error': str(error)[:2000]}
                text = json.dumps(value, ensure_ascii=False)
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': text})
                result['tool_calls'] += 1; result['tool_return_chars'] += len(text)
                windows = returned_windows(name, value)
                result['received_windows'].extend(windows)
                if name == 'grep':
                    result['grep_calls'] += 1
                elif name == 'read_file':
                    result['read_calls'] += 1
                    if not value.get('error'):
                        if value['read_mode'] == 'whole':
                            result['whole_reads'] += 1
                            result['full_file_read_chars'] += len(value['text'])
                        else:
                            result['range_reads'] += 1
                elif name == 'rag_search':
                    result['rag_calls'] += 1
                result['events'].append({'type': 'tool', 'name': name, 'arguments': raw,
                    'seconds': round(time.monotonic() - before, 3), 'returned_chars': len(text),
                    'error': value.get('error'), 'received_windows': windows})
                save()
                if time.monotonic() - started > config['time_limit_seconds']:
                    result['status'] = 'timeout'
                    break
            if result['status'] == 'timeout':
                break
        if result['status'] == 'running':
            result['status'] = 'call_limit'
    except (BudgetError, UnknownUsageError) as error:
        result['status'] = 'budget_blocked'
        result['stop_scheduling'] = True
        result['error_type'] = error.__class__.__name__; result['error'] = str(error)[:2000]
    except Exception as error:
        result['status'] = 'error'; result['error_type'] = error.__class__.__name__; result['error'] = str(error)[:2000]
    result['api_calls'] = executed_api_calls(client, label, result['model_responses'])
    result['executed'] = result['api_calls'] > 0
    save()
    print(json.dumps({key: result[key] for key in ('id', 'arm', 'status', 'executed', 'seconds', 'tool_calls', 'rag_calls')}, ensure_ascii=False), flush=True)
    return result


def schedule(cases, split, client, retriever, config):
    """每题两组相邻，AB/BA交错；预算阻断后不再创建后续任务。"""
    completed = []
    for index, case in enumerate(cases):
        public = {key: case[key] for key in ('id', 'question', 'kind')}
        for arm in (['A', 'B'] if index % 2 == 0 else ['B', 'A']):
            result = execute(public, arm, split, client, retriever, config)
            completed.append(result)
            if result.get('stop_scheduling') or result.get('status') == 'budget_blocked':
                return completed
    return completed


def select_cases(cases, ids):
    if ids is None:
        return cases
    if not ids:
        raise ValueError('--ids至少提供一个题号')
    unknown = set(ids) - {case['id'] for case in cases}
    if unknown:
        raise ValueError('未知题号：' + ', '.join(sorted(unknown)))
    return [case for case in cases if case['id'] in ids]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('split', choices=['dev', 'test'])
    parser.add_argument('--workers', type=int, default=1); parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    if args.workers != 1:
        parser.error('本轮预算收紧，仅允许--workers 1')
    config = json.loads((ROOT / 'experiment_config.json').read_text())
    if config['model'] != CHAT_MODEL:
        raise ValueError('配置模型与客户端模型不一致')
    if not 0 < config['budget_cap_cny'] <= 3:
        raise ValueError('本轮新增调用的预算必须大于0且不超过3元')
    cases = select_cases(json.loads((ROOT / f'data/cases_{args.split}.json').read_text()), args.ids)
    client = Client(ROOT, cap_cny=config['budget_cap_cny']); retriever = Retriever(ROOT, client)
    results = schedule(cases, args.split, client, retriever, config)
    print(json.dumps({'scheduled_tasks': len(results), 'api_summary': client.summary()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
