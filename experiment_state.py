"""将当前运行目录绑定到经过文件校验的冻结清单。"""
import hashlib
import json
from pathlib import Path
import re


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus(root, frozen_files):
    manifest = root / 'data/files.jsonl'
    if 'data/files.jsonl' not in frozen_files:
        if manifest.exists() or manifest.is_symlink():
            raise ValueError('语料文件清单必须列入冻结清单')
        return
    corpus = (root / 'data/corpus').resolve()
    if not corpus.is_relative_to(root) or not corpus.is_dir():
        raise ValueError('语料目录不存在或越出项目目录')
    expected = {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        name = row.get('path') if isinstance(row, dict) else None
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or any(part in ('.', '..') for part in name.split('/'))
                or Path(name).as_posix() != name or '\\' in name):
            raise ValueError('语料文件必须使用规范的安全相对路径')
        if name in expected:
            raise ValueError('语料文件清单包含重复路径：' + name)
        checksum = row.get('sha256')
        if not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
            raise ValueError('语料文件SHA256无效：' + name)
        expected[name] = checksum
    if not expected:
        raise ValueError('语料文件清单不能为空')
    actual = set()
    for path in corpus.rglob('*'):
        name = path.relative_to(corpus).as_posix()
        if not path.resolve().is_relative_to(corpus):
            raise ValueError('语料文件越出语料目录：' + name)
        # rglob不遍历目录链接，拒绝它们以免漏检文件或陷入循环。
        if path.is_symlink() and path.is_dir():
            raise ValueError('语料内部目录不能使用符号链接：' + name)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('语料包含无效文件：' + name)
        actual.add(name)
    if actual != set(expected):
        raise ValueError('语料文件集合与冻结清单不一致')
    for name, checksum in expected.items():
        if file_sha256(corpus / name) != checksum:
            raise ValueError('语料正文已改变：' + name)


def bind_run(root):
    root = Path(root).resolve()
    manifest_path = root / 'freeze_manifest.json'
    if not manifest_path.resolve().is_relative_to(root):
        raise ValueError('冻结清单不能越出项目目录')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError('冻结清单必须包含非空files映射')
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or '..' in Path(name).parts):
            raise ValueError('冻结文件必须使用安全的项目相对路径')
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('冻结文件不存在或越出项目目录：' + name)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('冻结文件SHA256无效：' + name)
        if file_sha256(path) != expected:
            raise ValueError('冻结文件已改变：' + name)
    verify_corpus(root, files)
    fingerprint = hashlib.sha256(raw).hexdigest()
    runs = root / 'runs'
    if not runs.resolve().is_relative_to(root) or (runs.exists() and not runs.is_dir()):
        raise ValueError('运行目录必须位于项目内')
    marker = runs / '实验标识.json'
    if marker.is_symlink():
        raise ValueError('实验标识不能使用符号链接')
    if marker.exists():
        identity = json.loads(marker.read_text())
        if not isinstance(identity, dict) or identity.get('freeze_sha256') != fingerprint:
            raise ValueError('运行目录与当前冻结清单不一致；不能复用旧轨迹或账本')
    else:
        if runs.exists() and any(runs.iterdir()):
            raise ValueError('运行目录已有未绑定的数据；请先归档并核验旧轨迹与账本')
        runs.mkdir(exist_ok=True)
        # 独占创建，避免并发绑定覆盖已有实验归属。
        with marker.open('x') as handle:
            json.dump({'freeze_sha256': fingerprint}, handle, ensure_ascii=False, indent=2)
    return fingerprint
