"""独立验证原文保留、父子行号、覆盖范围与全部 HTML 代码示例。"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

from build_corpus import self_test


def main():
    self_test()
    data = Path(__file__).resolve().parent / 'data'
    corpus = data / 'corpus'
    files = [json.loads(line) for line in (data / 'files.jsonl').read_text().splitlines()]
    chunks = [json.loads(line) for line in (data / 'chunks.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk['path']].append(chunk)
    assert len(groups) == len(files)
    assert len({c['id'] for c in chunks}) == len(chunks)
    for file in files:
        raw = (corpus / file['path']).read_bytes()
        lines = raw.decode().splitlines(keepends=True)
        assert hashlib.sha256(raw).hexdigest() == file['sha256']
        ordered = sorted(groups[file['path']], key=lambda c: c['start_line'])
        assert ''.join(c['text'] for c in ordered) == raw.decode()
        assert ordered[0]['start_line'] == 1 and ordered[-1]['end_line'] == len(lines)
        for c in ordered:
            assert c['text'] == ''.join(lines[c['start_line'] - 1:c['end_line']])
            assert c['parent_start_line'] <= c['start_line'] <= c['end_line'] <= c['parent_end_line']
            assert len(''.join(lines[c['parent_start_line'] - 1:c['parent_end_line']])) <= 6000
        if file['kind'] == 'code':
            original = data / 'raw/source/postgresql-17.0' / file['path'].removeprefix('code/')
            assert raw == original.read_bytes()
    pre_count = 0
    for html in (data / 'raw/docs').rglob('*.html'):
        soup = BeautifulSoup(html.read_text(), 'html.parser')
        output = (corpus / 'docs' / (html.stem + '.txt')).read_text()
        for pre in soup.find_all('pre'):
            pre_count += 1
            code = pre.get_text().rstrip()
            assert not code or code in output, html
    result = {'status': 'passed', 'files_checked': len(files), 'chunks_checked': len(chunks),
              'complete_chunk_coverage': True, 'source_bytes_unchanged': True,
              'all_html_pre_blocks_preserved': pre_count, 'child_exact_line_slice': True,
              'parents_at_most_6000_chars': True, 'child_chars_max': max(len(c['text']) for c in chunks),
              'source_archive_official_sha256': (data / 'downloads/postgresql-17.0.tar.bz2.sha256').read_text().split()[0]}
    (data / 'corpus_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
