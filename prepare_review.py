"""按实际执行结果生成隐藏组别评审包，支持预算停止后的部分实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from experiment_state import bind_run

ROOT = Path(__file__).resolve().parent
TERMINAL_EXCLUSIONS = {'running', 'pending', 'not_started'}


def read_json(path):
    return json.loads(Path(path).read_text())


def ledger_records(root):
    path = Path(root) / 'runs/api_calls.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def task_records(records, split, case_id, arm):
    prefix = f'{split}:{case_id}:{arm}:'
    return [row for row in records if row.get('label', '').startswith(prefix)]


def was_executed(records):
    return any(row.get('record_type') == 'api_call' for row in records)


def received_windows(run):
    """只记录实际返回的行；total_lines 是文件长度，不是读取范围。"""
    windows = {}
    def add(path, lo, hi):
        if isinstance(path, str) and type(lo) is int and type(hi) is int and 1 <= lo <= hi:
            windows.setdefault(path, []).append([lo, hi])
    for message in run.get('messages', []):
        if message.get('role') != 'tool':
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError, KeyError):
            continue
        if not isinstance(value, dict) or value.get('error'):
            continue
        if all(key in value for key in ('path', 'start_line', 'end_line')):
            add(value['path'], value['start_line'], value['end_line'])
        elif 'path' in value and isinstance(value.get('text'), str):
            # shortcut: 兼容旧全文工具按实际编号行重建窗口，不凭总行数推断已读。
            for line in value['text'].splitlines():
                number, separator, _ = line.partition(': ')
                if separator and number.isdigit():
                    add(value['path'], int(number), int(number))
        for row in value.get('lines', []):
            if isinstance(row, dict):
                add(row.get('path'), row.get('line'), row.get('line'))
        for row in value.get('results', []):
            if isinstance(row, dict):
                add(row.get('path'), row.get('start_line'), row.get('end_line'))
    merged = {}
    for path, spans in windows.items():
        output = []
        for lo, hi in sorted(spans):
            if output and lo <= output[-1][1] + 1:
                output[-1][1] = max(output[-1][1], hi)
            else:
                output.append([lo, hi])
        merged[path] = output
    return merged


def review_directory(root, split):
    return Path(root) / 'runs' / ('review' if split == 'test' else 'review_dev')


def prepare(root=ROOT, split='test', packets=None):
    root = Path(root)
    fingerprint = bind_run(root)
    cases = {case['id']: case for case in read_json(root / f'data/cases_{split}.json')}
    records = ledger_records(root)
    runs, states = [], []
    for case_id, case in sorted(cases.items()):
        for arm in ('A', 'B'):
            path = root / f'runs/{split}/{case_id}_{arm}.json'
            run = read_json(path) if path.exists() else {}
            executed = was_executed(task_records(records, split, case_id, arm))
            status = run.get('status', 'not_started')
            eligible = executed and bool(run) and status not in TERMINAL_EXCLUSIONS
            states.append({'id': case_id, 'arm': arm, 'status': status,
                           'executed': executed, 'included_for_review': eligible})
            if not eligible:
                continue
            windows = received_windows(run)
            evidence = []
            for reference in case.get('evidence', []):
                relative = Path(reference['path'])
                candidate = (root / 'data/corpus' / relative).resolve()
                if relative.is_absolute() or not candidate.is_relative_to((root / 'data/corpus').resolve()):
                    raise ValueError('金标准证据路径超出语料目录')
                lines = candidate.read_text().splitlines()
                lo, hi = reference['start_line'], reference['end_line']
                if not 1 <= lo <= hi <= len(lines):
                    raise ValueError(f'金标准证据行号越界：{case_id}')
                evidence.append({**reference,
                    'text': '\n'.join(f'{i}: {lines[i - 1]}' for i in range(lo, hi + 1)),
                    'received_overlap': [[max(lo, a), min(hi, b)] for a, b in windows.get(reference['path'], [])
                                         if a <= hi and b >= lo]})
            entry = {'question': case['question'], 'kind': case['kind'],
                     'difficulty': case.get('difficulty'), 'scenario': case.get('scenario'),
                     'expected_action': case.get('expected_action', 'answer'),
                     'required_facts': case['required_facts'], 'reference_evidence': evidence,
                     'reference_notes': case.get('reference_notes', []),
                     'reference_notes_policy': '仅供理解与核验证据，不作为额外必答项；按required_facts评分。',
                     'answer': run.get('final_answer'), 'run_status': status,
                     'received_windows': windows}
            runs.append((case_id, arm, entry))
    random.Random(20261017).shuffle(runs)
    packet_count = min(len(runs), packets or min(3, max(1, (len(runs) + 3) // 4))) if runs else 0
    if packet_count > 3 or packet_count < 0:
        raise ValueError('评审包数量须为1至3')
    output_packets = [[] for _ in range(packet_count)]
    mapping = {}
    for index, (case_id, arm, entry) in enumerate(runs, 1):
        blind_id = f'E{index:03d}'
        entry['blind_id'] = blind_id
        output_packets[(index - 1) % packet_count].append(entry)
        mapping[blind_id] = {'id': case_id, 'arm': arm}
    directory = review_directory(root, split)
    directory.mkdir(parents=True, exist_ok=True)
    mapping_path = directory / 'private_mapping.json'
    if list(directory.glob('review_*.json')):
        changed = not mapping_path.exists() or read_json(mapping_path) != mapping
        for index, packet in enumerate(output_packets, 1):
            path = directory / f'packet_{index}.json'
            changed = changed or not path.exists() or read_json(path) != packet
        if changed:
            raise ValueError('已有评分且盲评材料发生变化；请先归档本次评分，避免错配')
    for stale in directory.glob('packet_*.json'):
        if stale.name != 'packet_hashes.json':
            stale.unlink()
    for index, packet in enumerate(output_packets, 1):
        (directory / f'packet_{index}.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in directory.glob('packet_*.json') if path.name != 'packet_hashes.json'}
    (directory / 'packet_hashes.json').write_text(json.dumps(hashes, indent=2))
    manifest = {'split': split, 'freeze_sha256': fingerprint, 'planned_tasks': len(cases) * 2,
                'executed_tasks': sum(row['executed'] for row in states),
                'answers_for_review': len(runs), 'packets': packet_count, 'states': states}
    (directory / 'preparation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--packets', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    print(json.dumps(prepare(split=args.split, packets=args.packets), ensure_ascii=False, indent=2))
