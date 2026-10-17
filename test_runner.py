"""Zero-network checks for corpus tools, trust boundaries and BM25."""
import json
from pathlib import Path
import sqlite3
import tempfile
from run_experiment import Tools


with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);d=root/'data/corpus/docs';d.mkdir(parents=True)
    (root/'data/corpus/code').mkdir()
    long='x'*3000+' transaction'
    (d/'sample.txt').write_text('heading\ntransaction begins\n'+long+'\nend\n')
    tools=Tools(root)
    result=tools.grep('transaction',context_lines=1,max_matches=50)
    assert result['match_count']==2 and not result['truncated']
    assert any(long==r['text'] for r in result['lines'])
    assert len({(r['path'],r['line']) for r in result['lines']})==len(result['lines'])
    whole=tools.read_file('docs/sample.txt')
    assert whole['total_lines']==4 and long in whole['text'] and not whole['truncated']
    try:tools.read_file('../../../../etc/passwd')
    except ValueError:pass
    else:raise AssertionError('Path escape accepted')
    (d/'link.txt').symlink_to('/etc/passwd')
    try:tools.read_file('docs/link.txt')
    except ValueError:pass
    else:raise AssertionError('Symlink escape accepted')
    assert tools.grep('begins',files_only=True)['files']==['docs/sample.txt']
    (d/'a.txt').write_text('needle\n'*5)
    (d/'b.txt').write_text('needle\n')
    (d/'c.txt').write_text('needle\n')
    capped=tools.grep('needle',max_matches=2,context_lines=0)
    assert capped['match_count']==2 and capped['truncated']
    listed=tools.grep('needle',max_matches=2,files_only=True,context_lines=0)
    assert listed['files']==['docs/a.txt','docs/b.txt'] and listed['file_count']==2 and listed['truncated']
    conn=sqlite3.connect(':memory:')
    conn.execute('CREATE VIRTUAL TABLE passages USING fts5(path,title,text)')
    conn.execute('INSERT INTO passages VALUES (?,?,?)',('docs/sample.txt','transaction','isolation begins'))
    assert conn.execute('SELECT rowid FROM passages WHERE passages MATCH ? ORDER BY bm25(passages,3,2,1)',('"transaction" OR "isolation"',)).fetchone()==(1,)
print('Runner tools and FTS5 checks passed')


# 小型本地索引只用假向量；不创建 Client，也不请求网络。
import hashlib
import numpy as np
from retrieval import Retriever


class FakeRetrievalClient:
    def __init__(self):
        self.embedded=[];self.reranked=[]
    def embed(self,texts,label):
        self.embedded.append((texts,label))
        return [[1.0]+[0.0]*511 for _ in texts]
    def rerank(self,query,documents,label):
        self.reranked.append((query,documents,label))
        return [{'index':i,'relevance_score':1.0-i*0.01} for i in range(len(documents))]


with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);corpus=root/'data/corpus';index=root/'data/index'
    (corpus/'docs').mkdir(parents=True);(corpus/'code').mkdir();index.mkdir()
    chunks=[]
    for i in range(35):
        path=f'code/file{i:02d}.c';text='alpha\ncode detail'
        (corpus/path).write_text(text)
        chunks.append({'path':path,'title':'code','kind':'code','text':text,
            'start_line':1,'end_line':2,'parent_start_line':1,'parent_end_line':2})
    (corpus/'docs/shared.txt').write_text('alpha\nspecific detail\nalpha second\nlast detail')
    for start,text in [(1,'alpha\nspecific detail'),(3,'alpha second\nlast detail')]:
        chunks.append({'path':'docs/shared.txt','title':'manual','kind':'doc','text':text,
            'start_line':start,'end_line':start+1,'parent_start_line':1,'parent_end_line':4})
    (corpus/'docs/other.txt').write_text('beta\nunrelated')
    chunks.append({'path':'docs/other.txt','title':'manual','kind':'doc','text':'beta\nunrelated',
        'start_line':1,'end_line':2,'parent_start_line':1,'parent_end_line':2})
    chunk_file=root/'data/chunks.jsonl'
    chunk_file.write_text(''.join(json.dumps(c)+'\n' for c in chunks))
    from qwen_client import EMBED_MODEL
    (index/'config.json').write_text(json.dumps({'chunks_sha256':hashlib.sha256(chunk_file.read_bytes()).hexdigest(),
                                               'model':EMBED_MODEL,'dimensions':512}))
    matrix=np.zeros((len(chunks),512),dtype=np.float32);matrix[:35,0]=1
    matrix[35,:2]=[0.8,0.6];matrix[36,:2]=[0.6,0.8];matrix[37,1]=1
    np.save(index/'vectors.npy',matrix)
    conn=sqlite3.connect(index/'keyword.sqlite')
    conn.execute('CREATE VIRTUAL TABLE passages USING fts5(path,title,text)')
    conn.executemany('INSERT INTO passages(rowid,path,title,text) VALUES (?,?,?,?)',
        [(i+1,c['path'],c['title'],c['text']) for i,c in enumerate(chunks)])
    conn.commit();conn.close()
    fake=FakeRetrievalClient();retriever=Retriever(root,fake)
    # 35 个更强的源码命中不能占用 doc scope 的 SQL LIMIT 30 名额。
    assert set(retriever.keyword('alpha','doc'))=={35,36}
    assert len(retriever.keyword('alpha','code'))==30
    found=retriever.search('alpha',scope='doc',top_k=3,label='mock-doc')
    assert found['dense_count']==3 and found['keyword_count']==2 and found['candidate_count']==3
    assert len(found['results'])==2
    assert found['results'][0]['path']=='docs/shared.txt'
    assert found['results'][0]['text']=='alpha\nspecific detail\nalpha second\nlast detail'
    assert (found['results'][0]['start_line'],found['results'][0]['end_line'])==(1,4)
    assert all(r['kind']=='doc' for r in found['results'])
    # 双路 RRF 的前两项都优于仅 dense 命中的 other，并只向 rerank 发送子块。
    sent=fake.reranked[0][1]
    assert len(sent)==3 and sent[0].startswith('docs/shared.txt\n') and sent[-1].startswith('docs/other.txt\n')
    assert len(fake.embedded)==1 and fake.embedded[0][0]==['alpha']
    # 部分重叠父块仍可能包含新证据，不能仅因重叠就整块丢弃。
    retriever.chunks[35]['parent_end_line']=3
    retriever.chunks[36]['parent_start_line']=2
    overlapping=retriever.search('alpha',scope='doc',top_k=3,label='mock-overlap')
    shared=[r for r in overlapping['results'] if r['path']=='docs/shared.txt']
    assert len(shared)==2 and any(r['end_line']==4 for r in shared)
    code_found=retriever.search('alpha',scope='code',top_k=3,label='mock-code')
    assert len(code_found['results'])==3 and all(r['kind']=='code' for r in code_found['results'])
    assert code_found['returned_chars']==sum(len(r['text']) for r in code_found['results'])
print('Retriever scope, RRF, rerank input and parent deduplication mock checks passed; no API calls')
