"""汇总小样本消融的完整分母、逐事实评审与供应方费用区间。"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics

from experiment_state import bind_run
from prepare_review import ROOT, TERMINAL_EXCLUSIONS, ledger_records, prepare, read_json, review_directory, task_records, was_executed
from qwen_client import CHAT_MODEL, Client

OUTPUT = ROOT / 'outputs'
NON_TERMINAL = TERMINAL_EXCLUSIONS | {'in_flight'}


def cost_stats(records, cap=3):
    # 复用唯一计费账本实现；不初始化需要密钥的网络客户端。
    client = object.__new__(Client)
    client.cap_cny = cap
    summary = client._summary(records)
    calls = [row for row in records if row.get('record_type') == 'api_call']
    summary.update({
        'api_calls': len(calls),
        'http_failed_calls': sum(row.get('http_status') != 200 for row in calls),
        'chat_prompt_tokens': sum((row.get('usage') or {}).get('prompt_tokens', 0)
                                  for row in calls if row.get('model') == CHAT_MODEL),
        'chat_completion_tokens': sum((row.get('usage') or {}).get('completion_tokens', 0)
                                      for row in calls if row.get('model') == CHAT_MODEL)})
    return summary


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * p
    lo = int(point)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (point - lo)


def validate_review(review, case, answer):
    facts = review.get('fact_results', [])
    has_answer = isinstance(answer, str) and bool(answer.strip())
    if bool(review.get('assessable')) != has_answer:
        raise ValueError('评审可评分状态与实际答案不一致')
    for key in ('assessable', 'source_requirement_met', 'citation_supported', 'strict_pass'):
        if type(review.get(key)) is not bool:
            raise ValueError('总体评分字段须为布尔值')
    if not isinstance(review.get('material_errors'), list):
        raise ValueError('实质错误列表缺失')
    if not has_answer and any(review[key] for key in ('source_requirement_met', 'citation_supported', 'strict_pass')):
        raise ValueError('无答案不能通过来源或引用检查')
    if len(facts) != (len(case['required_facts']) if has_answer else 0):
        raise ValueError('逐事实评分数量与金标准不一致')
    if facts and [row.get('index') for row in facts] != list(range(1, len(facts) + 1)):
        raise ValueError('逐事实评分编号须连续且从1开始')
    for fact in facts:
        if any(type(fact.get(key)) is not bool for key in ('covered', 'correct', 'evidence_supported')):
            raise ValueError('逐事实评分字段须为布尔值')
        if fact['evidence_supported'] and not (fact['covered'] and fact['correct']):
            raise ValueError('错误或未覆盖事实不能计入有支持覆盖')
    if review.get('gold_issue') and not review.get('gold_issue_resolved'):
        raise ValueError('发现未审裁的金标准问题')
    if review.get('strict_pass') and (not has_answer or review.get('material_errors') or
            not review.get('source_requirement_met') or not review.get('citation_supported') or
            not all(f['covered'] and f['correct'] and f['evidence_supported'] for f in facts)):
        raise ValueError('严格通过与事实或引用评分相矛盾')
    return facts


def public_text(value):
    text = str(value or '')
    key = os.environ.get('DASHSCOPE_API_KEY')
    if key:
        text = text.replace(key, '[密钥已隐去]')
    text = re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b', '[密钥已隐去]', text)
    return re.sub(r'/(?:Users|home|private|tmp|var|Volumes)/[^\s<>【】]+', '[本机路径已隐去]', text)


def tool_counts(run):
    whole = ranged = 0
    for event in run.get('events', []):
        if event.get('type') != 'tool' or event.get('name') != 'read_file':
            continue
        arguments = event.get('arguments', {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if 'start_line' in arguments or 'end_line' in arguments:
            ranged += 1
        else:
            whole += 1
    return whole, ranged


def aggregate(rows):
    executed = [row for row in rows if row['已执行']]
    timed = [row['耗时秒'] for row in executed if row['耗时秒'] is not None]
    reviewed = [row for row in executed if row['已评审']]
    finalized = bool(executed) and all(row['已评审'] and row['运行状态'] not in NON_TERMINAL for row in executed)
    def median(values):
        return statistics.median(values) if values else None
    return {
        'planned': len(rows), 'executed': len(executed),
        'completed': sum(row['运行状态'] == 'complete' for row in executed),
        'assessable': sum(row['可评分'] for row in executed), 'reviewed': len(reviewed),
        'pending_review': sum(row['可评分'] and not row['已评审'] for row in executed),
        'strict_passed': sum(row['严格通过'] is True for row in executed),
        'strict_task_success_rate': (sum(row['严格通过'] is True for row in executed) / len(executed)
                                     if finalized else None),
        'strict_planned_success_rate': (sum(row['严格通过'] is True for row in rows) / len(rows)
                                       if finalized else None),
        'planned_fact_count': sum(row['必要事实数'] for row in rows),
        'assessed_fact_count': sum(row['必要事实数'] for row in reviewed if row['可评分']),
        'correct_covered_facts': sum(row['覆盖且正确事实数'] or 0 for row in reviewed),
        'supported_covered_facts': sum(row['有支持覆盖事实数'] or 0 for row in reviewed),
        'chat_prompt_tokens': sum(row['模型输入token'] for row in rows),
        'chat_completion_tokens': sum(row['模型输出token'] for row in rows),
        'online_cost_lower_cny': sum(row['在线费用下界元'] for row in rows),
        'online_cost_upper_cny': sum(row['在线费用上界元'] for row in rows),
        'unknown_calls': sum(row['费用未知请求'] for row in rows),
        'pending_calls': sum(row['在途请求'] for row in rows),
        'cost_median_lower_cny': median([row['在线费用下界元'] for row in executed]),
        'cost_median_upper_cny': median([row['在线费用上界元'] for row in executed]),
        'cost_p95_lower_cny': percentile([row['在线费用下界元'] for row in executed], .95),
        'cost_p95_upper_cny': percentile([row['在线费用上界元'] for row in executed], .95),
        'latency_median_seconds': median(timed), 'latency_p95_seconds': percentile(timed, .95),
        'whole_file_read_calls': sum(row['全文读取次数'] for row in rows),
        'range_read_calls': sum(row['范围读取次数'] for row in rows),
        'grep_calls': sum(row['grep次数'] for row in rows),
        'skill_calls': sum(row['Skill加载次数'] for row in rows),
        'rag_calls': sum(row['RAG次数'] for row in rows),
        'tasks_using_rag': sum(row['RAG次数'] > 0 for row in executed),
        'tool_return_chars': sum(row['工具返回字符数'] for row in rows)}


def paired_outcomes(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault(row['题号'], {})[row['组别']] = row
    result = {'planned_pairs': len(by_case), 'both_executed': 0, 'both_terminal': 0,
              'reviewed_pairs': 0, 'incomplete_pairs': 0, 'both_pass': 0,
              'A_only_pass': 0, 'B_only_pass': 0, 'neither_pass': 0}
    for pair in by_case.values():
        a, b = pair['A'], pair['B']
        both = a['已执行'] and b['已执行']
        result['both_executed'] += bool(both)
        terminal = both and a['运行状态'] not in NON_TERMINAL and b['运行状态'] not in NON_TERMINAL
        result['both_terminal'] += bool(terminal)
        if not terminal:
            result['incomplete_pairs'] += 1
        if not terminal or not (a['已评审'] and b['已评审']):
            continue
        result['reviewed_pairs'] += 1
        ap, bp = a['严格通过'] is True, b['严格通过'] is True
        result['both_pass' if ap and bp else 'A_only_pass' if ap else 'B_only_pass' if bp else 'neither_pass'] += 1
    return result


def summarize(root=ROOT, split='test', output=OUTPUT):
    root = Path(root)
    fingerprint = bind_run(root)
    config = read_json(root / 'experiment_config.json')
    cases = {case['id']: case for case in read_json(root / f'data/cases_{split}.json')}
    directory = review_directory(root, split)
    if list(directory.glob('review_*.json')):
        try:
            preparation = read_json(directory / 'preparation_manifest.json')
        except (OSError, ValueError) as error:
            raise ValueError('已有评分但评审准备清单缺失或无效，不能确定原始盲评映射') from error
        packets = preparation.get('packets') if isinstance(preparation, dict) else None
        if type(packets) is not int or not 1 <= packets <= 3:
            raise ValueError('已有评分的原始评审包数量无效，不能重建未知盲评映射')
        prepare(root, split, packets=packets)
    mapping = read_json(directory / 'private_mapping.json') if (directory / 'private_mapping.json').exists() else {}
    review_by_run = {}
    for path in sorted(directory.glob('review_*.json')):
        for review in read_json(path):
            blind_id = review['blind_id']
            if blind_id not in mapping:
                raise ValueError('评审包含映射外的答案')
            identity = mapping[blind_id]
            key = (identity['id'], identity['arm'])
            if key in review_by_run:
                raise ValueError('同一答案存在重复评审')
            review_by_run[key] = review
    records = ledger_records(root)
    rows = []
    for case_id, case in sorted(cases.items()):
        for arm in ('A', 'B'):
            path = root / f'runs/{split}/{case_id}_{arm}.json'
            run = read_json(path) if path.exists() else {}
            task_events = task_records(records, split, case_id, arm)
            executed = was_executed(task_events)
            review = review_by_run.get((case_id, arm))
            if review is not None and not executed:
                raise ValueError('未发出API请求的任务不能冒充已执行来评分')
            answer = run.get('final_answer')
            facts = validate_review(review, case, answer) if review else []
            costs = cost_stats(task_events, config['budget_cap_cny'])
            whole, ranged = tool_counts(run)
            status = 'in_flight' if costs['pending_calls'] else run.get('status', 'not_started')
            passed = bool(review['strict_pass'] and status == 'complete') if review and status not in NON_TERMINAL else None
            rows.append({
                '题号': case_id, '组别': arm, '问题': public_text(case['question']),
                '题型': case['kind'], '难度': case.get('difficulty', '未标注'),
                '场景': case.get('scenario', '未标注'),
                '预期行为': case.get('expected_action', 'answer'), '运行状态': status,
                '执行情况': ('已确认执行，仍有在途请求' if executed and costs['pending_calls'] else
                             '在途，执行情况未知' if costs['pending_calls'] else
                             '已确认执行' if executed else '未确认执行'),
                '已执行': executed, '可评分': bool(executed and isinstance(answer, str) and answer.strip()),
                '已评审': review is not None, '严格通过': passed,
                '覆盖且正确事实数': sum(f['covered'] and f['correct'] for f in facts) if review else None,
                '有支持覆盖事实数': sum(f['covered'] and f['correct'] and f['evidence_supported'] for f in facts) if review else None,
                '必要事实数': len(case['required_facts']),
                '证据支持': review.get('citation_supported') if review else None,
                '模型输入token': costs['chat_prompt_tokens'], '模型输出token': costs['chat_completion_tokens'],
                '在线费用下界元': costs['cost_lower_bound_cny'], '在线费用上界元': costs['cost_upper_bound_cny'],
                '费用未知请求': costs['unknown_calls'], '在途请求': costs['pending_calls'],
                'HTTP失败请求': costs['http_failed_calls'], '耗时秒': run.get('seconds'),
                '工具调用数': run.get('tool_calls', 0), 'grep次数': run.get('grep_calls', 0),
                'Skill加载次数': sum(event.get('type') == 'tool' and event.get('name') == 'load_skill'
                                  for event in run.get('events', [])),
                '全文读取次数': whole, '范围读取次数': ranged, 'RAG次数': run.get('rag_calls', 0),
                '工具返回字符数': run.get('tool_return_chars', 0),
                '评审理由': public_text(review.get('reason')) if review else '',
                '最终回答': public_text(answer)})
    def grouped(field):
        return {value: {arm: aggregate([row for row in rows if row[field] == value and row['组别'] == arm])
                        for arm in ('A', 'B')} for value in sorted({row[field] for row in rows})}
    terminal = all(row['已执行'] and row['运行状态'] not in NON_TERMINAL for row in rows)
    fully_reviewed = terminal and all(row['已评审'] for row in rows)
    budget_stopped = any('budget' in row['运行状态'].lower() for row in rows)
    api_total = cost_stats(records, config['budget_cap_cny'])
    result = {
        'split': split, 'freeze_sha256': fingerprint,
        'state': ('in_flight' if api_total['pending_calls'] else
                  'not_started' if not any(row['已执行'] for row in rows) else
                  'complete' if fully_reviewed else 'pending_review' if terminal else 'partial'),
        'budget_stopped': budget_stopped, 'planned_questions': len(cases), 'planned_tasks': len(rows),
        'executed_tasks': sum(row['已执行'] for row in rows),
        'by_arm': {arm: aggregate([row for row in rows if row['组别'] == arm]) for arm in ('A', 'B')},
        'by_kind': grouped('题型'), 'by_difficulty': grouped('难度'), 'by_scenario': grouped('场景'),
        'paired': paired_outcomes(rows), 'api_total': api_total,
        'cost_by_phase': {phase: cost_stats([r for r in records if r.get('label', '').startswith(prefix)], config['budget_cap_cny'])
                          for phase, prefix in [('development', 'dev:'), ('formal', 'test:')]},
        'limitations': ['少量题目单次配对，不能声称跨数据库或统计显著的普遍收益',
                       '隐藏组别代理评审及根代理抽核，未经独立人工复核',
                       '费用按供应方用量乘公开价，未核对账单、优惠或缓存抵扣',
                       '复用已有索引，本轮费用不含前轮建库；本地计算与评审订阅消耗未折算人民币']}
    destination = root / 'results/当前实验' / ('正式' if split == 'test' else '开发')
    destination.mkdir(parents=True, exist_ok=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = '正式' if split == 'test' else '开发'
    for path in (destination / '统计汇总.json', output / f'{name}统计汇总.json'):
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for path in (destination / '逐题结果.csv', output / f'{name}逐题结果.csv'):
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['题号', '组别'])
            writer.writeheader()
            writer.writerows(rows)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'scored_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    args = parser.parse_args()
    print(json.dumps(summarize(split=args.split), ensure_ascii=False, indent=2))
