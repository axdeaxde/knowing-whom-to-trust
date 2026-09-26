import copy
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.paths import ROOT
from src.repro import (atomic_json,atomic_npz,bind_run,activation_complete,
                       file_hash,validate_checkpoint,validate_runtime)
from src.select_lexicon import validate_scores
from src.validation import check_interventions,load_tasks


def load(name):
    return json.loads((ROOT/name).read_text())


@pytest.fixture
def scores():
    return pd.read_csv(ROOT/'data/verbalizer/development_words.csv')


@pytest.mark.parametrize('failure',['template','query','word','duplicate','label','nan','probability'])
def test_reject_incomplete_word_design(scores,failure):
    d=scores.copy()
    if failure=='template':d=d[d.template_id=='template_1']
    elif failure=='query':d=d[~((d.run_id==0)&(d.advisor_id=='Participant B6J')&(d.template_id=='template_1'))]
    elif failure=='word':d=d.iloc[1:]
    elif failure=='duplicate':d=pd.concat([d,d.iloc[:1]])
    elif failure=='label':d.loc[0,'label']=1-d.loc[0,'label']
    elif failure=='nan':d.loc[0,'logp']=np.nan
    else:d.loc[0,'termination_probability']=1.5
    with pytest.raises(ValueError):validate_scores(d)


def test_complete_word_design(scores):
    assert len(validate_scores(scores))==13600


def test_atomic_write_preserves_previous_on_failure(tmp_path):
    path=tmp_path/'record.json';atomic_json(path,{'valid':1})
    with pytest.raises(ValueError):atomic_json(path,{'invalid':float('nan')})
    assert json.loads(path.read_text())=={'valid':1}
    assert list(tmp_path.iterdir())==[path]


def test_fingerprint_change_and_missing_manifest(tmp_path):
    out=tmp_path/'run';a={'model':'test','seed':1}
    assert bind_run(out,a)==bind_run(out,a)
    with pytest.raises(ValueError,match='fingerprint'):bind_run(out,dict(a,seed=2))
    orphan=tmp_path/'orphan';orphan.mkdir();(orphan/'old.json').write_text('{}')
    with pytest.raises(ValueError,match='no run manifest'):bind_run(orphan,a)


def test_partial_activation_pair_recovery(tmp_path):
    rows=[r for r in load('data/probe/query_metadata.json') if r['run_id']==0]
    path=tmp_path/'run_000.npz';fp='synthetic-test'
    atomic_npz(path,history=np.zeros((4,64,5120),dtype=np.float16),
               query=np.zeros((8,2,64,5120),dtype=np.float16))
    assert not activation_complete(path,rows,fp)
    atomic_json(path.with_suffix('.json'),dict(run_fingerprint=fp,queries=rows,npz_sha256=file_hash(path),
        history_advisors=sorted({r['advisor_id'] for r in rows})))
    assert activation_complete(path,rows,fp)
    with pytest.raises(ValueError,match='fingerprint'):activation_complete(path,rows,'other-runtime')
    path.write_bytes(b'interrupted-write')
    assert not activation_complete(path,rows,fp)


def test_runtime_and_checkpoint_validation():
    expected=load('configs/mechanism_runtime.json')
    engine=SimpleNamespace(manifest=lambda:dict(expected))
    validate_runtime(engine,expected)
    changed=dict(expected,chat_template_hash='changed')
    with pytest.raises(ValueError,match='chat_template'):validate_runtime(engine,changed)
    with pytest.raises(ValueError,match='file mismatch'):validate_checkpoint({'config.json':'a'},{'config.json':'b'})


def test_span_is_verified(monkeypatch):
    from src.mechanism import run
    endpoint=load('configs/endpoints.json')[0]
    monkeypatch.setattr(run,'messages',lambda *a,**k:[])
    class Tokenizer:
        def __init__(self,wrong=False):self.wrong=wrong
        def __call__(self,text,**kw):
            start=text.index(endpoint['advisor_id']);count=endpoint['input_tokens']
            offsets=[(0,0)]*count;positions=range(4) if self.wrong else endpoint['query_id_span']
            for i in positions:offsets[i]=(start,start+len(endpoint['advisor_id']))
            return dict(offset_mapping=offsets,input_ids=list(range(count)))
        def decode(self,*args,**kw):return endpoint['advisor_id']
    engine=SimpleNamespace(render=lambda *a,**k:endpoint['prompt'],tokenizer=Tokenizer())
    assert run.span(engine,endpoint)==endpoint['query_id_span']
    engine.tokenizer=Tokenizer(True)
    with pytest.raises(ValueError,match='Frozen query'):run.span(engine,endpoint)


def test_baseline_guard_and_incomplete_json(tmp_path):
    from src.mechanism.run import baseline_guard
    baseline_guard(tmp_path,'endpoint',{'score':1.0},'same')
    baseline_guard(tmp_path,'endpoint',{'score':1.0},'same')
    with pytest.raises(ValueError,match='baseline drift'):baseline_guard(tmp_path,'endpoint',{'score':2.0},'same')
    from src.repro import read_completed_json
    p=tmp_path/'broken.json';p.write_text('{"task":')
    with pytest.raises(ValueError,match='Incomplete result'):read_completed_json(p)


@pytest.fixture
def behavior_run(tmp_path):
    from src.behavior.runner import run_one
    p=load('configs/protocol.json');c=load('configs/models.json')
    c.update(c['models']['qwen27b']['run_settings']);c['run_ids']=[0]
    class Client:
        release_identity={'kind':'synthetic-test-client','version':1}
        def __init__(self):self.calls=0
        def complete(self,messages,seed):
            t=p['runs'][0]['trials'][self.calls];self.calls+=1
            field={'tag':'choice','tg':'allocation','conflict':'chosen_id'}[t['phase']]
            value='left' if t['phase']=='tag' else 5 if t['phase']=='tg' else t['participant_a_id']
            return dict(content=json.dumps({field:value,'reason':'synthetic test'}),
                        non_thinking_ok=True,finish_status='stop',input_tokens=1,output_tokens=1)
    client=Client();out=tmp_path/'behavior'
    run_one(client,out,c,p,p['runs'][0])
    return client,out,c,p


def test_behavior_resume_rejects_changed_parameters(behavior_run):
    from src.behavior.runner import run_one
    client,out,c,p=behavior_run
    changed=dict(c,max_new_tokens=c['max_new_tokens']+1)
    with pytest.raises(ValueError,match='fingerprint'):run_one(client,out,changed,p,p['runs'][0])
    assert client.calls==76
    run_one(client,out,c,p,p['runs'][0]);assert client.calls==76


def test_new_behavior_analysis_and_plot(behavior_run,tmp_path):
    from src.analyze_rerun import behavioral,plot_new
    _,out,_,_=behavior_run
    tables,_=behavioral(out,[0]);assert tables['behavior_run_metrics'].iloc[0].tg_gap==0
    plot_new(tables,tmp_path/'new_analysis')
    assert (tmp_path/'new_analysis/figures/new_behavior.png').is_file()
    row=out/'raw/run_000.jsonl';row.write_text('\n'.join(row.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='Incomplete'):behavioral(out,[0])


def test_new_dose_analysis_does_not_read_frozen_tables(tmp_path):
    from src.analyze_rerun import mechanistic,plot_new
    out=tmp_path/'mechanism';fp=bind_run(out,dict(kind='mechanism',runtime='synthetic-fixture'))
    records=[json.loads(x) for x in (ROOT/'data/intervention/dose_records.jsonl').read_text().splitlines()]
    for r in records:
        if r['task']['run_id']==20:
            atomic_json(out/'dose'/(r['task']['task_id']+'.json'),dict(r,run_fingerprint=fp))
    tables,_=mechanistic(out,[20],['dose'])
    assert len(tables['dose_records'])==124
    assert len(tables['dose_summary'][tables['dose_summary'].condition=='full_id_span_steer'])==21
    plot_new(tables,tmp_path/'new_analysis')
    assert (tmp_path/'new_analysis/figures/new_dose.png').is_file()
    first=next((out/'dose').glob('*.json'));first.unlink()
    with pytest.raises(ValueError,match='Incomplete stage'):mechanistic(out,[20],['dose'])


def test_missing_manifest_never_reports_success(tmp_path):
    from src.verify import verify_checksums
    with pytest.raises(ValueError,match='missing'):verify_checksums(tmp_path)
    p=tmp_path/'file.txt';p.write_text('payload')
    (tmp_path/'MANIFEST.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  file.txt\n')
    assert verify_checksums(tmp_path)==1
    p.write_text('changed')
    with pytest.raises(ValueError,match='altered'):verify_checksums(tmp_path)


def test_missing_auxiliary_outputs_rejected(tmp_path,monkeypatch):
    import src.verify as verify
    monkeypatch.setattr(verify,'OUTPUT',tmp_path)
    with pytest.raises(ValueError,match='Incomplete offline'):verify.require_outputs()


@pytest.mark.parametrize('stage,name',[('dose','dose'),('patch','patch'),
                         ('composed-query','composed_query'),('direct-query','direct_query')])
def test_all_saved_intervention_arithmetic(stage,name):
    rows=[json.loads(x) for x in (ROOT/f'data/intervention/{name}_records.jsonl').read_text().splitlines()]
    assert check_interventions(rows,load_tasks(stage))==len(rows)
    bad=copy.deepcopy(rows);bad[0]['oriented']+=1
    with pytest.raises(ValueError,match='Oriented'):check_interventions(bad,load_tasks(stage))


def test_mechanism_resume_runtime_change_is_rejected(tmp_path,monkeypatch):
    from src.mechanism import run
    engine=SimpleNamespace(release_identity={'runtime':'test-1'})
    args=SimpleNamespace(out=tmp_path/'mechanism')
    run.bind_engine(engine,args)
    monkeypatch.setitem(sys.modules,'src.mechanism.engine',SimpleNamespace(distribution_kl=lambda *a:0))
    engine.release_identity={'runtime':'test-2'}
    with pytest.raises(ValueError,match='fingerprint'):run.run_interventions(engine,args)


def test_all_tokenizer_dependencies_affect_fingerprint(monkeypatch):
    import src.repro as repro
    for dependency in ['mistral-common','sentencepiece','tiktoken','tokenizers']:
        monkeypatch.setattr(repro.importlib.metadata,'version',lambda name:'old')
        first=repro.versions()
        monkeypatch.setattr(repro.importlib.metadata,'version',
                            lambda name,dependency=dependency:'new' if name==dependency else 'old')
        second=repro.versions()
        assert first!=second and dependency in second


def test_zero_dose_tolerance_is_consistent():
    from scipy.special import logsumexp
    from src.validation import ZERO_DOSE_TOLERANCE
    rows=[json.loads(x) for x in (ROOT/'data/intervention/dose_records.jsonl').read_text().splitlines()]
    r=copy.deepcopy(next(r for r in rows if r['task']['alpha']==0 and r['task']['readout']=='verbalizer'))
    lex=load('configs/lexicon.json')
    for w in lex['V_H']:r['changed']['word_logp'][w]+=1e-6
    p=r['changed']['word_logp']
    r['changed']['score']=(logsumexp([p[w] for w in lex['V_H']])-np.log(len(lex['V_H']))-
                           logsumexp([p[w] for w in lex['V_L']])+np.log(len(lex['V_L'])))
    r['delta']=r['changed']['score']-r['baseline']['score'];r['oriented']=0;r['kl']=1e-13
    assert abs(r['delta'])<ZERO_DOSE_TOLERANCE
    assert check_interventions([r],[r['task']])==1
