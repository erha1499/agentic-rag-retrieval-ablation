#!/usr/bin/env python3
"""从 PostgreSQL 17.0 官方发行包构造共享文本语料与稳定的父子索引。"""
import argparse
import hashlib
import io
import json
import re
import shutil
import statistics
import tarfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
BASE_URL = 'https://ftp.postgresql.org/pub/source/v17.0/'
SOURCE_NAME = 'postgresql-17.0.tar.bz2'
DOCS_NAME = 'postgresql-17.0-docs.tar.gz'
VERSION = '17.0'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError(f'不安全的归档路径：{name}')
    return p


def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            safe_member(m.name)
            if not (m.isfile() or m.isdir()):
                raise ValueError(f'不接受符号链接或特殊归档成员：{m.name}')
        tf.extractall(target, members=members, filter='data')


def download(name):
    p = DATA / 'downloads' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with urlopen(BASE_URL + name, timeout=120) as r, p.with_suffix(p.suffix + '.part').open('wb') as f:
            shutil.copyfileobj(r, f)
        p.with_suffix(p.suffix + '.part').replace(p)
    return p


def inline(node):
    return re.sub(r'\s+', ' ', node.get_text('')).strip()


def doc_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = inline(soup.title) if soup.title else ''
    body = soup.body or soup
    for el in body.select('.navheader, .navfooter, .toc, script, style, hr'):
        el.decompose()
    blocks = []

    def emit(s, wrap=True):
        if s.strip():
            if wrap:
                s = '\n'.join(textwrap.wrap(s.strip(), width=110, break_long_words=False, break_on_hyphens=False))
            blocks.append(s.rstrip())

    def visit(node):
        if isinstance(node, NavigableString):
            if node.strip():
                emit(re.sub(r'\s+', ' ', str(node)).strip())
            return
        if not isinstance(node, Tag):
            return
        if node.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            emit('#' * int(node.name[1]) + ' ' + inline(node))
        elif node.name == 'pre':
            emit(node.get_text(), wrap=False)
        elif node.name == 'table':
            for tr in node.find_all('tr'):
                cells = tr.find_all(['th', 'td'], recursive=False)
                if cells:
                    values = []
                    contains_pre = False
                    for cell in cells:
                        saved = []
                        for n, pre in enumerate(cell.find_all('pre')):
                            token = f'PRESERVEDCODEBLOCK{n}TOKEN'
                            saved.append((token, pre.get_text()))
                            pre.replace_with(NavigableString(token))
                        value = inline(cell)
                        for token, code in saved:
                            value = value.replace(token, '\n' + code.rstrip() + '\n')
                        values.append(value)
                        contains_pre |= bool(saved)
                    emit(' | '.join(values), wrap=not contains_pre)
        elif node.name in ['p', 'dt']:
            emit(inline(node))
        elif node.name == 'li':
            pending = []
            def flush():
                if pending:
                    emit('- ' + re.sub(r'\s+', ' ', ''.join(pending)).strip())
                    pending.clear()
            for child in node.children:
                if isinstance(child, Tag) and child.name in ['p', 'pre', 'table', 'ul', 'ol', 'dl', 'div']:
                    flush()
                    visit(child)
                else:
                    pending.append(str(child) if isinstance(child, NavigableString) else child.get_text(''))
            flush()
        else:
            for child in node.children:
                visit(child)

    visit(body)
    return title, '\n\n'.join(blocks).strip() + '\n'


def core_file(p):
    if p.suffix not in {'.c', '.h', '.y', '.l'}:
        return False
    # shortcut: 保留官方手写核心代码；若研究第三方实现，需要另设对照语料。
    excluded_parts = {'test', 'tests', 'regress', 'vendor', 'generated', 'libstemmer', 'regex'}
    if any(part in excluded_parts for part in p.parts):
        return False
    first = p.read_text(errors='replace')[:5000].lower()
    return not any(s in first for s in ['do not edit', 'automatically generated', 'file auto-generated', 'a bison parser', 'a lexical scanner generated'])


def split_ranges(lines, target, maximum):
    result = []
    start = 0
    while start < len(lines):
        end = start
        total = 0
        boundaries = []
        while end < len(lines):
            n = len(lines[end])
            # shortcut: 单个超长原文行不拆行；若接近模型输入上限，再采用列位置索引。
            if total + n > maximum and end > start:
                break
            total += n
            end += 1
            if not lines[end - 1].strip() or lines[end - 1].startswith('}'):
                boundaries.append((end, total))
            if total >= target:
                break
        if end < len(lines) and boundaries:
            candidates = [(e, n) for e, n in boundaries if n >= target * .6]
            if candidates:
                end = min(candidates, key=lambda x: abs(x[1] - target))[0]
        assert end > start
        result.append((start, end))
        start = end
    return result


def make_chunks(rel, kind, title, text):
    lines = text.splitlines(keepends=True)
    ranges = split_ranges(lines, 1200 if kind == 'doc' else 1500, 1800)
    chunks = []
    parent_groups = []
    group = []
    size = 0
    for start, end in ranges:
        chunk_size = sum(map(len, lines[start:end]))
        if group and size + chunk_size > 5800:
            parent_groups.append(group)
            group, size = [], 0
        group.append((start, end))
        size += chunk_size
    if group:
        parent_groups.append(group)
    for group in parent_groups:
        ps, pe = group[0][0], group[-1][1]
        for start, end in group:
            value = ''.join(lines[start:end])
            cid = sha256((rel + '\n' + str(start + 1) + ':' + str(end) + '\n' + value).encode())[:24]
            chunks.append(dict(id=cid, path=rel, kind=kind, start_line=start + 1, end_line=end, text=value,
                               parent_start_line=ps + 1, parent_end_line=pe, title=title))
    return chunks


def distribution(values):
    v = sorted(values)
    if not v:
        return {}
    return {'min': v[0], 'p25': v[int((len(v)-1)*.25)], 'median': statistics.median(v),
            'p75': v[int((len(v)-1)*.75)], 'p90': v[int((len(v)-1)*.9)], 'max': v[-1], 'sum': sum(v)}


def self_test():
    assert safe_member('postgresql-17.0/src/backend/foo.c').name == 'foo.c'
    for s in ['../x', '/tmp/x', 'safe/../../x', r'safe\evil']:
        try:
            safe_member(s)
        except ValueError:
            pass
        else:
            raise AssertionError(s)
    bad_tar = io.BytesIO()
    with tarfile.open(fileobj=bad_tar, mode='w') as tf:
        info = tarfile.TarInfo('../escape')
        info.size = 1
        tf.addfile(info, io.BytesIO(b'x'))
    with tarfile.open(fileobj=io.BytesIO(bad_tar.getvalue())) as tf:
        try:
            safe_member(tf.getmembers()[0].name)
        except ValueError:
            pass
        else:
            raise AssertionError('归档穿越未拒绝')
    title, text = doc_text('<html><head><title>T</title></head><body><div class="navheader">NAV</div><h1>Heading</h1><p>hello <code>world</code>.</p><pre>a\n  b</pre><table><tr><td>x</td><td>y</td></tr></table></body></html>')
    assert title == 'T' and 'NAV' not in text and 'hello world.' in text and 'a\n  b' in text and 'x | y' in text
    _, nested = doc_text('<body><ul><li><p>example</p><pre>a\n  b</pre><p>after</p></li></ul></body>')
    assert 'a\n  b' in nested and 'after' in nested
    _, table_code = doc_text('<body><table><tr><td>result<pre>a\n  b</pre></td><td>note</td></tr></table></body>')
    assert 'a\n  b' in table_code and 'result' in table_code and 'note' in table_code
    sample = '\n'.join('line ' + str(n) + ' x' * 30 for n in range(400)) + '\n'
    chunks = make_chunks('test.txt', 'code', 'test', sample)
    for c in chunks:
        ls = sample.splitlines(keepends=True)
        assert c['text'] == ''.join(ls[c['start_line']-1:c['end_line']])
        assert c['parent_start_line'] <= c['start_line'] <= c['end_line'] <= c['parent_end_line']
        assert len(''.join(ls[c['parent_start_line']-1:c['parent_end_line']])) <= 6000
    assert chunks == make_chunks('test.txt', 'code', 'test', sample)
    source = DATA / 'raw/source/postgresql-17.0'
    if source.exists():
        assert core_file(source / 'src/include/nodes/nodes.h')
        assert core_file(source / 'src/backend/utils/cache/relcache.c')
        assert not core_file(source / 'src/include/common/unicode_case_table.h')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print('构建器自检通过')
        return
    if any((DATA/name).is_symlink() for name in ('raw','corpus','files.jsonl','chunks.jsonl')):
        raise RuntimeError('共享冻结语料只读；重建须使用独立目录')
    archives = {name: download(name) for name in [SOURCE_NAME, DOCS_NAME]}
    checksum = download(SOURCE_NAME + '.sha256').read_text().split()[0]
    actual = sha256(archives[SOURCE_NAME].read_bytes())
    assert actual == checksum, '源码归档官方 SHA256 校验失败'
    for name, archive in archives.items():
        target = DATA / 'raw' / ('source' if name == SOURCE_NAME else 'docs')
        if not target.exists():
            extract(archive, target)
    corpus = DATA / 'corpus'
    corpus.mkdir(parents=True, exist_ok=True)
    files = []
    chunks = []
    for html in sorted((DATA / 'raw' / 'docs').rglob('*.html')):
        title, text = doc_text(html.read_text())
        rel = 'docs/' + html.stem + '.txt'
        out = corpus / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        files.append(dict(path=rel, kind='doc', url='https://www.postgresql.org/docs/17/' + html.name, title=title,
                          sha256=sha256(text.encode()), lines=len(text.splitlines()), chars=len(text), words=len(text.split())))
        chunks.extend(make_chunks(rel, 'doc', title, text))
    source = DATA / 'raw' / 'source' / 'postgresql-17.0'
    excluded_code = []
    for directory in ['src/backend', 'src/include']:
        for p in sorted((source / directory).rglob('*')):
            if not p.is_file() or p.suffix not in {'.c', '.h', '.y', '.l'}:
                continue
            if not core_file(p):
                excluded_code.append(p.relative_to(source).as_posix())
                continue
            rel_src = p.relative_to(source).as_posix()
            rel = 'code/' + rel_src
            content = p.read_bytes()
            text = content.decode('utf-8')
            out = corpus / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
            files.append(dict(path=rel, kind='code', url='https://github.com/postgres/postgres/blob/REL_17_0/' + rel_src,
                              title=rel_src, sha256=sha256(content), lines=len(text.splitlines()), chars=len(text), words=len(text.split())))
            chunks.extend(make_chunks(rel, 'code', rel_src, text))
    assert len({f['path'] for f in files}) == len(files)
    assert len({c['id'] for c in chunks}) == len(chunks)
    for f in files:
        data = (corpus / f['path']).read_bytes()
        assert sha256(data) == f['sha256']
    texts = {f['path']: (corpus / f['path']).read_text().splitlines(keepends=True) for f in files}
    for c in chunks:
        ls = texts[c['path']]
        assert c['text'] == ''.join(ls[c['start_line']-1:c['end_line']])
        assert c['parent_start_line'] <= c['start_line'] <= c['end_line'] <= c['parent_end_line']
        assert len(''.join(ls[c['parent_start_line']-1:c['parent_end_line']])) <= 6000
    for name, rows in [('files.jsonl', files), ('chunks.jsonl', chunks)]:
        with (DATA / name).open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    docs = [f for f in files if f['kind'] == 'doc']
    code = [f for f in files if f['kind'] == 'code']
    manifest = {
        'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'archives': [{ 'url': BASE_URL + name, 'bytes': path.stat().st_size, 'sha256': sha256(path.read_bytes()),
                       'official_sha256_verified': name == SOURCE_NAME,
                       'official_checksum_url': BASE_URL + SOURCE_NAME + '.sha256' if name == SOURCE_NAME else None}
                      for name, path in archives.items()],
        'scope': {'docs': '完整官方 HTML 手册，每个自然 HTML 页面转为文本，去除导航、目录导航、脚本与样式',
                  'code': 'src/backend 和 src/include 的手写 .c/.h/.y/.l；排除测试、第三方 regex/libstemmer 及标为自动生成的文件',
                  'same_corpus_for_both_arms': True, 'source_lines_include_comments_and_blank': True,
                  'doc_page_urls_may_follow_17_minor_updates': True, 'authoritative_snapshot': '本地经哈希记录的17.0发行包',
                  'excluded_code_files': excluded_code},
        'files': len(files), 'doc_files': len(docs), 'code_files': len(code),
        'doc_chars': distribution([f['chars'] for f in docs]), 'doc_words': distribution([f['words'] for f in docs]),
        'doc_pages_at_least_3000_words': sum(f['words'] >= 3000 for f in docs),
        'code_lines': distribution([f['lines'] for f in code]), 'code_chars': sum(f['chars'] for f in code),
        'chunk_count': len(chunks), 'doc_chunks': sum(c['kind'] == 'doc' for c in chunks),
        'code_chunks': sum(c['kind'] == 'code' for c in chunks),
        'chunk_chars': distribution([len(c['text']) for c in chunks]),
        'chunking': {'doc_target_chars': 1200, 'code_target_chars': 1500, 'child_target_max_chars': 1800, 'child_observed_max_chars': max(len(c['text']) for c in chunks),
                     'parent_max_chars': 6000, 'parent_target_max_chars': 5800,
                     'boundaries': '空行、代码顶层右花括号附近；过大函数按行切分，父块是相邻子块构成的局部原文'},
        'validation': {'safe_archive_paths': 'passed', 'stable_ids': 'passed', 'child_exact_line_slice': 'passed',
                       'parent_contains_child': 'passed', 'parent_max_6000_chars': 'passed', 'all_file_hashes': 'passed'},
        'files_manifest_sha256': sha256((DATA / 'files.jsonl').read_bytes()),
        'chunks_sha256': sha256((DATA / 'chunks.jsonl').read_bytes())}
    json_write(DATA / 'corpus_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['files', 'doc_files', 'code_files', 'doc_pages_at_least_3000_words', 'chunk_count', 'code_lines', 'doc_words']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
