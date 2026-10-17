"""Shared local corpus, FTS5/BM25, dense retrieval, RRF and Qwen reranking."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time

import numpy as np

from qwen_client import Client, EMBED_MODEL


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_chunks(root):
    with (root / 'data/chunks.jsonl').open() as f:
        return [json.loads(line) for line in f]


def embed_text(chunk):
    return f"{chunk['path']}\n{chunk['title']}\n{chunk['text']}"


def build_index(root, workers=1):
    index_dir = root / 'data/index'
    index_dir.mkdir(exist_ok=True)
    chunks = load_chunks(root)
    fingerprint = digest(root / 'data/chunks.jsonl')
    config = {'chunks_sha256': fingerprint, 'model': EMBED_MODEL,
              'dimensions': 512, 'count': len(chunks), 'batch_size':20}
    config_path = index_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing embedding cache belongs to a different corpus/configuration')
    config_path.write_text(json.dumps(config, indent=2))
    conn = sqlite3.connect(index_dir / 'keyword.sqlite')
    conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(path,title,text)')
    if conn.execute('SELECT count(*) FROM passages').fetchone()[0] != len(chunks):
        conn.execute('DELETE FROM passages')
        conn.executemany('INSERT INTO passages(rowid,path,title,text) VALUES (?,?,?,?)',
                         ((i+1,c['path'],c['title'],c['text']) for i,c in enumerate(chunks)))
        conn.commit()
    conn.close()
    client = Client(root)
    batches = [(start, min(start+20, len(chunks))) for start in range(0,len(chunks),20)]
    def batch(item):
        start,end = item
        path = index_dir / f'batch-{start:06d}.npy'
        if path.exists():
            arr = np.load(path)
            if arr.shape == (end-start,512) and np.isfinite(arr).all():
                return False
            raise RuntimeError(f'Invalid cached embedding batch: {path.name}')
        time.sleep(0.6)
        vectors = client.embed([embed_text(c) for c in chunks[start:end]], label=f'index:{start}:{end}')
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape != (end-start,512) or not np.isfinite(arr).all():
            raise RuntimeError(f'Invalid embedding response for {start}:{end}')
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.save(f,arr)
        temp.replace(path)
        return True
    started = time.monotonic()
    completed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(batch,item) for item in batches]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append({'type':type(error).__name__,'message':str(error)[:300]})
            completed += 1
            if completed % 100 == 0 or completed == len(batches):
                print(json.dumps({'index_batches':completed,'total_batches':len(batches),
                                  'failed_batches':len(failures),'seconds':round(time.monotonic()-started,1)},ensure_ascii=False),flush=True)
    if failures:
        (index_dir/'failed_batches.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
        raise RuntimeError(f'{len(failures)} embedding batches failed; cached successful batches preserved')
    matrix = np.concatenate([np.load(index_dir / f'batch-{start:06d}.npy') for start,end in batches])
    norms = np.linalg.norm(matrix,axis=1,keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError('Zero embedding vector')
    matrix /= norms
    np.save(index_dir / 'vectors.npy', matrix)
    (index_dir / 'manifest.json').write_text(json.dumps({**config,'vector_sha256':digest(index_dir/'vectors.npy'),
        'seconds':time.monotonic()-started,'cost_snapshot':client.summary()},ensure_ascii=False,indent=2))


class Retriever:
    def __init__(self, root, client):
        self.root = Path(root)
        self.corpus = self.root / 'data/corpus'
        self.client = client
        self.chunks = load_chunks(self.root)
        config = json.loads((self.root/'data/index/config.json').read_text())
        if config['chunks_sha256'] != digest(self.root/'data/chunks.jsonl'):
            raise RuntimeError('Corpus changed after indexing')
        if config['model'] != EMBED_MODEL or config['dimensions'] != 512:
            raise RuntimeError('Embedding model or dimensions differ between corpus and query')
        self.vectors = np.load(self.root/'data/index/vectors.npy',mmap_mode='r')
        if self.vectors.shape != (len(self.chunks),512):
            raise RuntimeError('Vector/chunk count mismatch')
        self.kind = np.array([c['kind'] for c in self.chunks])
        self._local = threading.local()
    def keyword(self,query,scope):
        terms = re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[\u4e00-\u9fff]+|\d+',query)
        stop = {'the','a','an','of','in','to','and','is','are','for','how','does','what','postgresql'}
        terms = [t for t in dict.fromkeys(terms) if t.lower() not in stop][:24]
        if not terms:
            return []
        match = ' OR '.join('"'+t.replace('"','""')+'"' for t in terms)
        if not hasattr(self._local,'conn'):
            self._local.conn = sqlite3.connect(self.root/'data/index/keyword.sqlite')
        prefix = '%' if scope == 'all' else ('docs/%' if scope == 'doc' else 'code/%')
        rows = self._local.conn.execute('SELECT rowid FROM passages WHERE passages MATCH ? AND path LIKE ? ORDER BY bm25(passages,3,2,1) LIMIT 30',(match,prefix)).fetchall()
        return [i-1 for (i,) in rows]
    def search(self,query,scope='all',top_k=6,label='retrieval'):
        if not isinstance(query,str) or not query.strip() or len(query)>2000:
            raise ValueError('query must be nonempty and <=2000 characters')
        if scope not in ('all','doc','code') or top_k not in (3,6,10):
            raise ValueError('Invalid scope or top_k')
        start = time.monotonic()
        q = np.asarray(self.client.embed([query],label=label+':query')[0],dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0 or not np.isfinite(q).all():
            raise ValueError('Invalid query vector')
        scores = self.vectors @ (q/norm)
        if scope != 'all':
            scores[self.kind != scope] = -np.inf
        n = min(30,int(np.isfinite(scores).sum()))
        dense = np.argpartition(scores,-n)[-n:] if n else np.array([],dtype=int)
        dense = sorted(map(int,dense),key=lambda i:(-float(scores[i]),i))
        lexical = self.keyword(query,scope)
        rrf = {}
        for ranking in (dense,lexical):
            for rank,i in enumerate(ranking,1):
                rrf[i] = rrf.get(i,0)+1/(60+rank)
        candidates = sorted(rrf,key=lambda i:(-rrf[i],i))[:30]
        if not candidates:
            return {'results':[],'candidate_count':0}
        reranked = self.client.rerank(query,[embed_text(self.chunks[i]) for i in candidates],label=label+':rerank')
        selected=[]; windows=[]; total_chars=0; omitted=0
        for item in reranked:
            c = self.chunks[candidates[item['index']]]
            lo,hi = c['parent_start_line'],c['parent_end_line']
            if any(p==c['path'] and a<=lo and hi<=b for p,a,b in windows):
                continue
            lines=(self.corpus/c['path']).read_text().splitlines()
            text='\n'.join(lines[lo-1:hi])
            if total_chars+len(text)>24000:
                omitted+=1
                continue
            selected.append({'path':c['path'],'kind':c['kind'],'start_line':lo,'end_line':hi,
                'matched_start_line':c['start_line'],'matched_end_line':c['end_line'],
                'score':item['relevance_score'],'text':text})
            windows.append((c['path'],lo,hi)); total_chars+=len(text)
            if len(selected)>=top_k:
                break
        return {'results':selected,'candidate_count':len(candidates),'dense_count':len(dense),
            'keyword_count':len(lexical),'returned_chars':total_chars,'omitted_for_return_budget':omitted,
            'seconds':round(time.monotonic()-start,3)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1)
    args=p.parse_args();build_index(Path(__file__).resolve().parent,args.workers)
