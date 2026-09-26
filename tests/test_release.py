import json
from collections import Counter
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from src.paths import ROOT
from src.behavior.protocol import parse,branch,normalize

def load(name):return json.loads((ROOT/name).read_text())

def test_eight_model_release_scope():
    from src.analyze import COHORT,ORDER,NAMES
    expected={'gpt55','gemini25flash','gemma31b','qwen27b','ministral14b',
              'sonnet5','deepseek_flash','qwen35b'}
    assert set(COHORT)==set(ORDER)==set(NAMES)==expected
    assert set(load('configs/models.json')['models'])==expected
    assert set(load('configs/paper_scope.json')['behavior_models'])==expected
    assert {r['model'] for r in load('reference/paper_values.json')['behavior']}==expected
    assert set(load('configs/runtime_versions.json'))<=expected
    files=list((ROOT/'data/behavior').glob('*.jsonl'))
    assert {p.stem for p in files}==expected
    protocol={r['run_id']:{t['item_id']:t for t in r['trials']}
              for r in load('configs/protocol.json')['runs']}
    for path in files:
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows)==30*76
        for row in rows:
            assert row['trial']==protocol[row['run_id']][row['trial']['item_id']]
    trials=pd.read_csv(ROOT/'data/behavior/trials.csv')
    assert set(trials.model)==expected and len(trials)==8*30*76
    assert trials.groupby(['model','run_id','phase']).size().groupby('phase').unique().to_dict()=={
        'tag':np.array([64]),'tg':np.array([4]),'conflict':np.array([8])}

def test_retained_local_generation_settings():
    from src.behavior.model_client import LocalClient as QwenClient
    from src.behavior.local_client_extended import LocalClient as MinistralClient
    config=load('configs/models.json')['models'];runtime=load('configs/runtime_versions.json')
    for name,cls in [('qwen35b',QwenClient),('ministral14b',MinistralClient)]:
        client=cls.__new__(cls);client.config=config[name]
        client.max_tokens=config[name]['run_settings']['max_new_tokens']
        client.backend=config[name]['backend']
        assert client.settings()==runtime[name]['generation']

def test_gemma_cli_installs_recorded_attention(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    from src.behavior import run,attention_core
    calls=[]
    class Client:
        def __init__(self,model,max_tokens):
            calls.append(('client',model['label'],max_tokens))
    monkeypatch.setitem(sys.modules,'src.behavior.local_client_core',SimpleNamespace(LocalClient=Client))
    monkeypatch.setattr(run,'checkpoint_fingerprint',lambda path:{})
    monkeypatch.setattr(run,'versions',lambda:{})
    monkeypatch.setattr(attention_core,'install',lambda client,messages:calls.append(('attention',len(messages))))
    monkeypatch.setattr(run,'run_one',lambda client,out,settings,protocol,r:calls.append(('run',r['run_id'])))
    monkeypatch.setattr(sys,'argv',['behavior','--model','gemma31b','--model-path',str(tmp_path),
                                  '--out',str(tmp_path/'outputs'),'--runs','0'])
    run.main()
    assert calls==[('client','Gemma-4-31B-it',4096),('attention',2),('run',0)]

@pytest.mark.parametrize('rid',range(30))
def test_protocol(rid):
    run=load('configs/protocol.json')['runs'][rid];trials=run['trials']
    assert len(trials)==76
    tag=[r for r in trials if r['phase']=='tag'];assert len(tag)==64
    ids=sorted({r['participant_id'] for r in tag});assert len(ids)==4
    truth=[]
    for participant in ids:
        rows=[r for r in tag if r['participant_id']==participant]
        assert len(rows)==16
        assert sum(r['advice_side_wins'] for r in rows)==8
        truth.append(sum(r['report_is_truthful'] for r in rows))
    assert sorted(truth)==[4,4,12,12]
    pairs=Counter((r['high_participant_id'],r['low_participant_id']) for r in trials if r['phase']=='conflict')
    assert sorted(pairs.values())==[2,2,2,2]

def test_json_and_independent_branch():
    parent=[{'role':'user','content':'feedback'}]
    a=branch(parent,'task A');b=branch(parent,'task B')
    a[0]['content']='changed';assert parent[0]['content']=='feedback' and 'task B' in b[0]['content']
    assert parse('{"allocation":5,"reason":"history"}','tg')['allocation']==5
    assert parse(normalize({'choice':'left','reason':'report'}),'tag')['choice']=='left'
    with pytest.raises(ValueError):parse('{"allocation":11,"reason":"test"}','tg')
    with pytest.raises(ValueError):parse('{"choice":"left","choice":"right","reason":"x"}','tag')

def test_lexicon_and_split():
    lex=load('configs/lexicon.json');assert len(lex['V_H'])==14 and len(lex['V_L'])==11
    assert not set(lex['V_H'])&set(lex['V_L'])
    pos=pd.read_csv(ROOT/'data/verbalizer/pos_filter.csv').set_index('word')
    assert pos.loc[lex['V_H']+lex['V_L'],'eligible'].all()
    split=load('configs/mechanism.json')['split']
    assert split['development_runs']==list(range(20)) and split['heldout_runs']==list(range(20,30))

def test_shared_direction_reconstruction():
    metadata=pd.DataFrame(load('data/probe/selected_feature_metadata.json'))
    with np.load(ROOT/'data/probe/selected_features.npz') as z:x=z['direction_query_id'].astype(float)
    mask=metadata.run_id<20;x=x[mask];labels=metadata.loc[mask,'label'].to_numpy()
    vector=x[labels==1].mean(0)-x[labels==0].mean(0);vector/=np.linalg.norm(vector)
    with np.load(ROOT/'configs/steering_direction.npz') as z:
        assert np.allclose(vector,z['query_id__43__mean'],atol=1e-10)
        assert np.isclose(np.std(x@vector,ddof=1),z['query_id__43__mean__sd'],atol=1e-10)
    for m in [1,4]:
        perturb=vector*2/np.sqrt(m)
        assert np.isclose(m*np.dot(perturb,perturb),4)

def test_dose_records_and_scoring():
    lex=load('configs/lexicon.json')
    records=[json.loads(s) for s in (ROOT/'data/intervention/dose_records.jsonl').read_text().splitlines()]
    assert len(records)==1240
    assert sum(r['task']['condition']=='full_id_span_steer' for r in records)==840
    for r in records:
        t=r['task'];assert 20<=t['run_id']<30 and t['layer']==43
        assert len(t['endpoint']['query_id_span'])==4
        for field in ['baseline','changed']:
            value=r[field]
            if t['readout']=='verbalizer':
                p=value['word_logp']
                score=logsumexp([p[w] for w in lex['V_H']])-np.log(14)-logsumexp([p[w] for w in lex['V_L']])+np.log(11)
            elif t['readout']=='tg':score=np.asarray(value['candidate_probability'])@np.arange(11)
            else:
                i=value['candidates'].index(t['advisor_id']);p=value['candidate_logp'];score=p[i]-p[1-i]
            assert np.isclose(score,value['score'],atol=1e-9)
        assert np.isclose(r['changed']['score']-r['baseline']['score'],r['delta'],atol=1e-10)
        if t['alpha']==0:assert r['delta']==0 and r['kl']==0

def test_patch_alignment():
    tasks=load('configs/patch_tasks.json');assert len(tasks)==60
    assert len({t['pair_id'] for t in tasks})==10
    for task in tasks:
        assert task['source_label']!=task['endpoint']['label']
        assert task['source_run']!=task['run_id']
        assert task['endpoint']['advisor_id']==task['advisor_id']

def test_metadata_baselines_refit():
    from src.mechanism.probe_helpers import metadata_model
    meta=pd.DataFrame(load('data/probe/selected_feature_metadata.json'))
    from sklearn.metrics import roc_auc_score
    for recent,expected in [(False,.82),(True,.9625)]:
        model=metadata_model(meta[meta.run_id<20],recent)
        values=model.predict_proba(meta[meta.run_id>=20])[:,1]
        assert abs(roc_auc_score(meta.loc[meta.run_id>=20,'label'],values)-expected)<1e-9

def test_runner_without_model_or_network(tmp_path):
    from src.behavior.runner import run_one
    protocol=load('configs/protocol.json');run=protocol['runs'][0]
    config=load('configs/models.json');config.update(config['models']['qwen27b']['run_settings']);config['run_ids']=[0]
    class Client:
        release_identity={'kind':'synthetic-offline-test','revision':1}
        def __init__(self):self.index=0;self.lengths=[]
        def complete(self,messages,seed):
            t=run['trials'][self.index];self.index+=1;self.lengths.append(len(messages))
            field={'tag':'choice','tg':'allocation','conflict':'chosen_id'}[t['phase']]
            value='left' if t['phase']=='tag' else 5 if t['phase']=='tg' else t['participant_a_id']
            return dict(content=json.dumps({field:value,'reason':'test'}),non_thinking_ok=True,finish_status='stop',input_tokens=1,output_tokens=1)
    client=Client();run_one(client,tmp_path,config,protocol,run)
    rows=[json.loads(s) for s in (tmp_path/'raw/run_000.jsonl').read_text().splitlines()]
    assert len(rows)==76 and client.index==76
    assert len({r['parent_history_hash'] for r in rows[64:]})==1
    assert client.lengths[64:]==[130]*12
    run_one(client,tmp_path,config,protocol,run)
    assert client.index==76
