"""评审/汇总离线回归：没有网络调用，不读真实实验答案。"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment_state import bind_run
from prepare_review import prepare, received_windows
from qwen_client import CHAT_MODEL
from summarize import cost_stats, public_text, summarize, validate_review


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def freeze(root, names):
    manifest = {'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in names}}
    write_json(root / 'freeze_manifest.json', manifest)
    return manifest


def review(blind_id, answer=True, passed=True):
    return {'blind_id': blind_id, 'assessable': answer,
            'fact_results': [{'index': 1, 'covered': passed, 'correct': passed,
                             'evidence_supported': passed, 'comment': '离线固定样例'}] if answer else [],
            'material_errors': [], 'source_requirement_met': passed if answer else False,
            'citation_supported': passed if answer else False, 'strict_pass': passed if answer else False,
            'reason': '离线固定样例', 'evidence_paths_checked': ['docs/demo.txt']}


class EvaluationTests(unittest.TestCase):
    def test_ranges_are_actual_not_file_length(self):
        values = [
            {'path': 'docs/demo.txt', 'start_line': 11, 'end_line': 12, 'total_lines': 900, 'text': '11: x\n12: y'},
            {'lines': [{'path': 'docs/demo.txt', 'line': 13, 'text': 'z'}]},
            {'results': [{'path': 'docs/demo.txt', 'start_line': 20, 'end_line': 25}]},
            {'path': 'docs/legacy.txt', 'total_lines': 900, 'text': '3: only returned line'},
            {'files': ['docs/unread.txt']},
            {'error': 'failure', 'path': 'docs/fail.txt', 'start_line': 1, 'end_line': 900}]
        run = {'messages': [{'role': 'tool', 'content': json.dumps(value)} for value in values]}
        self.assertEqual(received_windows(run), {'docs/demo.txt': [[11, 13], [20, 25]], 'docs/legacy.txt': [[3, 3]]})

    def test_strict_invariants(self):
        case = {'required_facts': ['必要事实']}
        good = review('E001')
        validate_review(good, case, '有答案')
        invalid = copy.deepcopy(good)
        invalid['fact_results'][0]['evidence_supported'] = False
        with self.assertRaises(ValueError):
            validate_review(invalid, case, '有答案')
        invalid = copy.deepcopy(good)
        invalid['material_errors'] = ['重要额外错误']
        with self.assertRaises(ValueError):
            validate_review(invalid, case, '有答案')
        with self.assertRaises(ValueError):
            validate_review(good, case, '')
        validate_review(review('E002', answer=False), case, None)

    def test_pending_and_unknown_are_not_zero(self):
        data = [
            {'record_type': 'budget', 'cap_cny': 3},
            {'record_type': 'reservation', 'reservation_id': 'pending', 'reserved_upper_cost_cny': .2},
            {'record_type': 'api_call', 'reservation_id': 'unknown', 'model': CHAT_MODEL,
             'label': 'test:X:A:step0', 'http_status': None, 'usage_status': 'unknown',
             'public_price_cost_cny': None, 'reserved_upper_cost_cny': .3, 'usage': None}]
        result = cost_stats(data)
        self.assertEqual(result['cost_lower_bound_cny'], 0)
        self.assertAlmostEqual(result['cost_upper_bound_cny'], .5)
        self.assertEqual(result['pending_calls'], 1)
        self.assertEqual(result['unknown_calls'], 1)

    def test_pending_tasks_wait_for_settlement_before_final_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'data/cases_test.json', [
                {'id': 'Q01', 'question': '自然问题', 'kind': 'doc', 'required_facts': ['事实']}])
            write_json(root / 'experiment_config.json', {'budget_cap_cny': 3})
            freeze(root, ['data/cases_test.json', 'experiment_config.json'])
            bind_run(root)
            ledger = root / 'runs/api_calls.jsonl'
            pending = {'record_type': 'reservation', 'reservation_id': 'first',
                       'label': 'test:Q01:A:step0', 'reserved_upper_cost_cny': .2}
            records = [pending]

            def snapshot():
                ledger.write_text('\n'.join(json.dumps(row) for row in records))
                result = summarize(root, output=root / 'outputs')
                rows = json.loads((root / 'runs/review/scored_rows.json').read_text())
                return result, rows[0]

            result, row = snapshot()
            self.assertEqual(result['state'], 'in_flight')
            self.assertEqual(result['executed_tasks'], 0)
            self.assertEqual(result['by_arm']['A']['executed'], 0)
            self.assertEqual(result['by_arm']['A']['pending_calls'], 1)
            self.assertAlmostEqual(result['api_total']['cost_upper_bound_cny'], .2)
            self.assertEqual(result['paired']['both_executed'], 0)
            self.assertEqual(result['paired']['both_terminal'], 0)
            self.assertEqual(row['运行状态'], 'in_flight')
            self.assertFalse(row['已执行'])
            self.assertIn('未知', row['执行情况'])

            call = {'record_type': 'api_call', 'reservation_id': 'first', 'model': CHAT_MODEL,
                    'label': 'test:Q01:A:step0', 'http_status': 200, 'usage_status': 'reported',
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 2},
                    'public_price_cost_cny': .01, 'reserved_upper_cost_cny': .2}
            records.extend([call, {**call, 'reservation_id': 'other', 'label': 'test:Q01:B:step0'}])
            for arm in ('A', 'B'):
                write_json(root / f'runs/test/Q01_{arm}.json', {
                    'id': 'Q01', 'arm': arm, 'status': 'complete', 'final_answer': '答案'})
            snapshot()
            prepare(root, packets=1)
            mapping = json.loads((root / 'runs/review/private_mapping.json').read_text())
            write_json(root / 'runs/review/review_1.json', [review(key) for key in mapping])
            result, row = snapshot()
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['api_total']['pending_calls'], 0)
            self.assertAlmostEqual(result['api_total']['cost_upper_bound_cny'], .02)
            self.assertEqual(result['paired']['both_pass'], 1)

            records.append({**pending, 'reservation_id': 'next', 'label': 'test:Q01:A:step1'})
            result, row = snapshot()
            self.assertEqual(result['state'], 'in_flight')
            self.assertEqual(result['executed_tasks'], 2)
            self.assertEqual(result['by_arm']['A']['completed'], 0)
            self.assertIsNone(result['by_arm']['A']['strict_task_success_rate'])
            self.assertIsNone(result['by_arm']['A']['strict_planned_success_rate'])
            self.assertIsNone(row['严格通过'])
            self.assertEqual(result['paired']['both_executed'], 1)
            self.assertEqual(result['paired']['both_terminal'], 0)
            self.assertEqual(result['paired']['reviewed_pairs'], 0)
            records.append({**call, 'reservation_id': 'next', 'label': 'test:Q01:A:step1'})
            result, row = snapshot()
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['paired']['both_pass'], 1)
            self.assertEqual(result['by_arm']['A']['strict_task_success_rate'], 1)

    def test_unassigned_reservation_is_still_in_flight(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'data/cases_test.json', [
                {'id': 'Q01', 'question': '自然问题', 'kind': 'doc', 'required_facts': ['事实']}])
            write_json(root / 'experiment_config.json', {'budget_cap_cny': 3})
            freeze(root, ['data/cases_test.json', 'experiment_config.json'])
            bind_run(root)
            (root / 'runs/api_calls.jsonl').write_text(json.dumps({
                'record_type': 'reservation', 'reservation_id': 'unassigned',
                'label': 'index:0:20', 'reserved_upper_cost_cny': .2}))
            result = summarize(root, output=root / 'outputs')
            self.assertEqual(result['state'], 'in_flight')
            self.assertEqual(result['executed_tasks'], 0)
            self.assertEqual(result['api_total']['pending_calls'], 1)
            self.assertAlmostEqual(result['api_total']['cost_upper_bound_cny'], .2)
            self.assertEqual(result['by_arm']['A']['pending_calls'], 0)

    def test_partial_denominators_and_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = [{'id': f'N{i:02d}', 'question': '离线题', 'kind': kind,
                      'difficulty': difficulty, 'scenario': '离线场景', 'expected_action': 'answer',
                      'reference_notes': ['延伸说明不是额外必答项'],
                      'required_facts': ['事实'], 'evidence': [{'path': 'docs/demo.txt', 'start_line': 2, 'end_line': 3}]}
                     for i, kind, difficulty in [(1, 'doc', 'easy'), (2, 'code', 'hard')]]
            write_json(root / 'data/cases_test.json', cases)
            write_json(root / 'experiment_config.json', {'budget_cap_cny': 3})
            corpus = root / 'data/corpus/docs/demo.txt'
            corpus.parent.mkdir(parents=True)
            corpus.write_text('one\ntwo\nthree\nfour')
            freeze(root, ['data/cases_test.json', 'experiment_config.json'])
            bind_run(root)
            ledger = [{'record_type': 'budget', 'cap_cny': 3}]
            for cid, arm, status, answer in [('N01', 'A', 'complete', '答案'),
                                             ('N01', 'B', 'incomplete_answer', '截断答案'),
                                             ('N02', 'A', 'complete', '答案')]:
                write_json(root / f'runs/test/{cid}_{arm}.json', {
                    'id': cid, 'arm': arm, 'status': status, 'final_answer': answer, 'seconds': 2,
                    'messages': [{'role': 'tool', 'content': json.dumps({'path': 'docs/demo.txt',
                                  'start_line': 2, 'end_line': 3, 'total_lines': 4, 'text': '2: two\n3: three'})}],
                    'events': [{'type': 'tool', 'name': 'read_file', 'arguments': {'path': 'docs/demo.txt', 'start_line': 2, 'end_line': 3}}]})
                ledger.append({'record_type': 'api_call', 'reservation_id': f'{cid}{arm}',
                               'model': CHAT_MODEL, 'label': f'test:{cid}:{arm}:step0', 'http_status': 200,
                               'usage_status': 'reported', 'usage': {'prompt_tokens': 10, 'completion_tokens': 2},
                               'public_price_cost_cny': .01, 'reserved_upper_cost_cny': .1})
            # 存在运行文件但没有API，仍然未执行。
            write_json(root / 'runs/test/N02_B.json', {'id': 'N02', 'arm': 'B', 'status': 'budget_blocked'})
            (root / 'runs/api_calls.jsonl').write_text('\n'.join(json.dumps(row) for row in ledger))
            manifest = prepare(root, packets=2)
            self.assertEqual(manifest['planned_tasks'], 4)
            self.assertEqual(manifest['executed_tasks'], 3)
            self.assertEqual(manifest['answers_for_review'], 3)
            self.assertEqual(manifest['packets'], 2)
            mapping = json.loads((root / 'runs/review/private_mapping.json').read_text())
            self.assertNotIn({'id': 'N02', 'arm': 'B'}, mapping.values())
            packet = json.loads((root / 'runs/review/packet_1.json').read_text())
            self.assertEqual(packet[0]['reference_notes'], ['延伸说明不是额外必答项'])
            self.assertEqual(packet[0]['required_facts'], ['事实'])
            self.assertIn('不作为额外必答项', packet[0]['reference_notes_policy'])
            pending = summarize(root, output=root / 'outputs')
            self.assertIsNone(pending['by_arm']['A']['strict_task_success_rate'])
            write_json(root / 'runs/review/review_1.json', [review(blind_id) for blind_id in mapping])
            result = summarize(root, output=root / 'outputs')
            self.assertEqual(result['state'], 'partial')
            self.assertTrue(result['budget_stopped'])
            self.assertEqual(result['by_arm']['A']['planned'], 2)
            self.assertEqual(result['by_arm']['B']['executed'], 1)
            self.assertEqual(result['by_arm']['A']['strict_passed'], 2)
            self.assertEqual(result['by_arm']['B']['strict_passed'], 0)  # 截断内容可正确，任务仍未完成。
            self.assertEqual(result['by_arm']['B']['supported_covered_facts'], 1)
            self.assertEqual(result['by_arm']['A']['range_read_calls'], 2)
            self.assertEqual(result['paired']['reviewed_pairs'], 1)
            self.assertEqual(result['paired']['A_only_pass'], 1)
            self.assertEqual(result['paired']['incomplete_pairs'], 1)
            self.assertAlmostEqual(result['api_total']['cost_upper_bound_cny'], .03)
            with (root / 'results/当前实验/正式/逐题结果.csv').open(encoding='utf-8-sig') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            preparation_path = root / 'runs/review/preparation_manifest.json'
            preparation = json.loads(preparation_path.read_text())
            preparation_path.unlink()
            with self.assertRaisesRegex(ValueError, '评审准备清单缺失或无效'):
                summarize(root, output=root / 'outputs')
            for packets in [0, True, 4]:
                write_json(preparation_path, {**preparation, 'packets': packets})
                with self.assertRaisesRegex(ValueError, '原始评审包数量无效'):
                    summarize(root, output=root / 'outputs')
            write_json(preparation_path, preparation)
            changed_run = root / 'runs/test/N01_A.json'
            changed = json.loads(changed_run.read_text())
            changed['final_answer'] = '评分后被改变的答案'
            write_json(changed_run, changed)
            with self.assertRaisesRegex(ValueError, '盲评材料发生变化'):
                summarize(root, output=root / 'outputs')
            with self.assertRaises(ValueError):
                prepare(root)

    def test_frozen_question_and_system_cannot_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ['data/cases_test.json', 'prompts/系统提示词.md']
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('冻结内容')
            manifest = freeze(root, names)
            fingerprint = bind_run(root)
            self.assertEqual(bind_run(root), fingerprint)
            for name in names:
                with self.subTest(name=name):
                    path = root / name
                    original = path.read_bytes()
                    path.write_text('已修改')
                    with self.assertRaisesRegex(ValueError, '冻结文件已改变'):
                        bind_run(root)
                    path.write_bytes(original)
            manifest['revision'] = '不同清单'
            write_json(root / 'freeze_manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, '运行目录与当前冻结清单不一致'):
                bind_run(root)

    def test_existing_unbound_runs_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'frozen.txt').write_text('冻结')
            freeze(root, ['frozen.txt'])
            old = root / 'runs/api_calls.jsonl'
            old.parent.mkdir()
            old.write_text('旧账本必须原样保留')
            with self.assertRaisesRegex(ValueError, '已有未绑定的数据'):
                bind_run(root)
            self.assertEqual(old.read_text(), '旧账本必须原样保留')
            self.assertFalse((root / 'runs/实验标识.json').exists())

    def test_corpus_body_and_inventory_are_frozen(self):
        for change in ('body', 'missing', 'extra'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                corpus = root / 'local_history/corpus'
                body = corpus / 'docs/demo.txt'
                body.parent.mkdir(parents=True)
                body.write_text('原始正文')
                (root / 'data').mkdir()
                (root / 'data/corpus').symlink_to(corpus, target_is_directory=True)
                row = {'path': 'docs/demo.txt', 'sha256': hashlib.sha256(body.read_bytes()).hexdigest()}
                (root / 'data/files.jsonl').write_text(json.dumps(row) + '\n')
                freeze(root, ['data/files.jsonl'])
                fingerprint = bind_run(root)
                self.assertEqual(bind_run(root), fingerprint)
                if change == 'body':
                    body.write_text('改变正文')
                elif change == 'missing':
                    body.unlink()
                else:
                    (corpus / 'docs/extra.txt').write_text('额外正文')
                with self.assertRaisesRegex(ValueError, '语料正文已改变|语料文件集合与冻结清单不一致'):
                    bind_run(root)
                self.assertEqual(json.loads((root / 'runs/实验标识.json').read_text())['freeze_sha256'], fingerprint)

    def test_corpus_manifest_rejects_duplicate_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = root / 'data/corpus/docs/demo.txt'
            body.parent.mkdir(parents=True)
            body.write_text('正文')
            row = {'path': 'docs/demo.txt', 'sha256': hashlib.sha256(body.read_bytes()).hexdigest()}
            bad_rows = [[row, row],
                        [{**row, 'path': '../outside.txt'}],
                        [{**row, 'path': str(body)}],
                        [{**row, 'path': 'docs/./demo.txt'}],
                        [{**row, 'path': 'docs//demo.txt'}],
                        [{**row, 'sha256': 'invalid'}], []]
            for rows in bad_rows:
                with self.subTest(rows=rows):
                    (root / 'data/files.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in rows))
                    freeze(root, ['data/files.jsonl'])
                    with self.assertRaises(ValueError):
                        bind_run(root)
                    self.assertFalse((root / 'runs').exists())
            (root / 'data/files.jsonl').write_text(json.dumps(row) + '\n')
            (root / 'other.txt').write_text('冻结')
            freeze(root, ['other.txt'])
            with self.assertRaisesRegex(ValueError, '语料文件清单必须列入冻结清单'):
                bind_run(root)

    def test_corpus_links_cannot_escape(self):
        for link in ('file', 'directory', 'root'):
            with self.subTest(link=link), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                root = base / 'project'
                (root / 'data/corpus/docs').mkdir(parents=True)
                outside = base / 'outside'
                outside.mkdir()
                body = outside / 'demo.txt'
                body.write_text('不能读取的外部正文')
                corpus = root / 'data/corpus'
                if link == 'file':
                    (corpus / 'docs/demo.txt').symlink_to(body)
                elif link == 'directory':
                    (corpus / 'docs/linked').symlink_to(outside, target_is_directory=True)
                else:
                    (corpus / 'docs').rmdir()
                    corpus.rmdir()
                    corpus.symlink_to(outside, target_is_directory=True)
                row = {'path': 'docs/demo.txt', 'sha256': hashlib.sha256(body.read_bytes()).hexdigest()}
                (root / 'data/files.jsonl').write_text(json.dumps(row) + '\n')
                freeze(root, ['data/files.jsonl'])
                with self.assertRaisesRegex(ValueError, '越出'):
                    bind_run(root)
                self.assertFalse((root / 'runs').exists())

    def test_frozen_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'project'
            root.mkdir()
            outside = Path(folder) / 'outside.txt'
            outside.write_text('外部文件')
            (root / 'link.txt').symlink_to(outside)
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            for name in ['../outside.txt', str(outside), 'link.txt']:
                with self.subTest(name=name):
                    write_json(root / 'freeze_manifest.json', {'files': {name: digest}})
                    with self.assertRaises(ValueError):
                        bind_run(root)
                    self.assertFalse((root / 'runs').exists())

    def test_no_tasks_means_no_execution_cost_or_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'data/cases_test.json', [
                {'id': 'Q01', 'question': '自然问题', 'kind': 'doc', 'required_facts': ['事实']}])
            write_json(root / 'experiment_config.json', {'budget_cap_cny': 3})
            freeze(root, ['data/cases_test.json', 'experiment_config.json'])
            manifest = prepare(root)
            self.assertEqual(manifest['executed_tasks'], 0)
            self.assertEqual(manifest['answers_for_review'], 0)
            result = summarize(root, output=root / 'outputs')
            self.assertEqual(result['state'], 'not_started')
            self.assertEqual(result['executed_tasks'], 0)
            self.assertEqual(result['api_total']['api_calls'], 0)
            self.assertEqual(result['api_total']['cost_upper_bound_cny'], 0)
            for arm in ('A', 'B'):
                self.assertEqual(result['by_arm'][arm]['reviewed'], 0)
                self.assertIsNone(result['by_arm'][arm]['strict_task_success_rate'])
            self.assertFalse((root / 'results/改进轮').exists())
            self.assertTrue((root / 'outputs/正式统计汇总.json').exists())

    def test_public_text(self):
        value = public_text('本地 /Users/demo/secret/file.txt 以及 sk-abcdefghijklmnop')
        self.assertNotIn('/Users/', value)
        self.assertNotIn('sk-', value)


if __name__ == '__main__':
    unittest.main()
