"""Recompute manuscript statistics from anonymous record-level inputs on CPU."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
from scipy.stats import rankdata
from .paths import ROOT, DATA, OUTPUT

COHORT=['gpt55','gemini25flash','gemma31b','qwen27b','ministral14b','sonnet5','deepseek_flash','qwen35b']
ORDER=COHORT
NAMES={'gpt55':'GPT-5.5','gemini25flash':'Gemini 2.5 Flash','gemma31b':'Gemma-4-31B-it',
 'qwen27b':'Qwen3.5-27B','ministral14b':'Ministral-3-14B','sonnet5':'Claude Sonnet 5',
 'deepseek_flash':'DeepSeek v4.1 Flash','qwen35b':'Qwen3.5-35B-A3B'}
SEED=20260925

def read(path):
    return json.loads(Path(path).read_text())

def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def save(name, data):
    frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    (OUTPUT/'tables').mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUTPUT/'tables'/f'{name}.csv',index=False)
    return frame

def ci(values):
    values=np.asarray(values,dtype=float)
    assert len(values) and np.isfinite(values).all()
    rng=np.random.default_rng(SEED)
    means=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return float(values.mean()), *map(float,np.quantile(means,[.025,.975]))

def auc(y,p):
    y,p=np.asarray(y),np.asarray(p)
    n1,n0=np.sum(y==1),np.sum(y==0)
    assert n1 and n0
    return float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def auc_ci(frame,field='prediction'):
    grouped=[g for _,g in frame.groupby('run_id')]
    labels=np.stack([g.label.to_numpy() for g in grouped])
    values=np.stack([g[field].to_numpy() for g in grouped])
    rng=np.random.default_rng(SEED)
    draws=rng.integers(len(grouped),size=(10000,len(grouped)))
    boots=[auc(labels[d].ravel(),values[d].ravel()) for d in draws]
    return auc(frame.label,frame[field]), *map(float,np.quantile(boots,[.025,.975]))

def behavior():
    trials=pd.read_csv(DATA/'behavior/trials.csv')
    assert set(trials.model)==set(COHORT) and len(trials)==len(COHORT)*30*76
    runs=[]
    for (model,rid),rows in trials.groupby(['model','run_id']):
        assert rows.phase.value_counts().to_dict()=={'tag':64,'conflict':8,'tg':4}
        tg=rows[rows.phase=='tg']; conflict=rows[rows.phase=='conflict']
        high=tg[tg.label.isin(['high_honesty','high'])].allocation.mean()
        low=tg[tg.label.isin(['low_honesty','low'])].allocation.mean()
        assert np.isfinite([high,low]).all(),tg.label.tolist()
        runs.append(dict(model=model,run_id=rid,tg_high=high,tg_low=low,tg_gap=high-low,
            conflict_high_rate=float((conflict.chosen_id==conflict.high_participant_id).mean())))
    runs=save('all_behavior_run_metrics',runs)
    summary=[]
    for model in ORDER:
        g=runs[runs.model==model];assert len(g)==30
        gap,lo,hi=ci(g.tg_gap);rate,rlo,rhi=ci(g.conflict_high_rate)
        summary.append(dict(model=model,label=NAMES[model],runs=30,tg_high=g.tg_high.mean(),tg_low=g.tg_low.mean(),
            tg_gap=gap,tg_low_ci=lo,tg_high_ci=hi,conflict_rate=rate,conflict_low_ci=rlo,conflict_high_ci=rhi))
    summary=save('all_behavior_summary',summary)
    save('behavior_summary',summary[summary.model.isin(COHORT)])
    save('behavior_run_metrics',runs[runs.model.isin(COHORT)])

def verbalizer():
    lex=read(ROOT/'configs/lexicon.json')
    rows=[]
    for r in read(DATA/'verbalizer/evaluation_baselines.json'):
        lp=r['measurement']['word_logp']
        score=logsumexp([lp[w] for w in lex['V_H']])-np.log(len(lex['V_H']))-logsumexp([lp[w] for w in lex['V_L']])+np.log(len(lex['V_L']))
        assert np.isclose(score,r['measurement']['score'],atol=1e-10)
        rows.append(dict(run_id=r['run_id'],advisor_id=r['advisor_id'],label=r['label'],readout='verbalizer',baseline=score))
    data=save('baseline_all_readouts',rows)
    g=data.groupby(['run_id','label']).baseline.mean().unstack()
    gap,lo,hi=ci(g[1]-g[0]);a,alo,ahi=auc_ci(data,'baseline')
    save('verbalizer_summary',[dict(auc=a,auc_low=alo,auc_high=ahi,high_low_gap=gap,gap_low=lo,gap_high=hi,
        high_mean=data[data.label==1].baseline.mean(),low_mean=data[data.label==0].baseline.mean(),n=len(data),independent_runs=10)])
    save('publication_word_effects_clustered',[dict(word=w,group=group) for group in ['H','L'] for w in lex['V_'+group]])

def probes():
    predictions=pd.read_csv(DATA/'probe/all_predictions.csv')
    original=pd.read_csv(DATA/'probe/reference_summary.csv').set_index('position')
    selection=read(ROOT/'configs/probe_selection.json')['selection']
    coefficients=read(ROOT/'configs/probe_coefficients.json')
    meta=pd.DataFrame(read(DATA/'probe/selected_feature_metadata.json'))
    with np.load(DATA/'probe/selected_features.npz',allow_pickle=False) as arrays:
        for pos in ['history_feedback_end','query_id','query_end']:
            params=coefficients[pos]
            for feature in [pos]+([pos+'_template1_transfer'] if pos!='history_feedback_end' else []):
                x=arrays[feature].astype(float)
                p=expit(((x-np.array(params['mean']))/np.array(params['scale']))@np.array(params['coef'])+params['intercept'])
                expected=predictions[predictions.position==feature].set_index(['run_id','advisor_id']).prediction
                for i,r in meta.iterrows():
                    if r.run_id>=20:assert np.isclose(p[i],expected.loc[(r.run_id,r.advisor_id)],atol=1e-9)
    rows=[]
    for pos,part in predictions.groupby('position',sort=False):
        a,lo,hi=auc_ci(part)
        assert abs(a-original.loc[pos,'auc'])<1e-10,(pos,a)
        rows.append(dict(position=pos,auc=a,ci_low=lo,ci_high=hi,
            logistic_layer=selection[pos]['logistic']['layer'] if pos in selection else np.nan))
    save('probe_summary_recomputed_uniform_bootstrap',rows)
    # The manuscript's metadata/alternate intervals retain their original seeded
    # bootstrap. Reproduce that exact RNG sequence, including permutation draws.
    from .mechanism.probe_helpers import evaluate_scores
    rng=np.random.default_rng(20260923)
    legacy={}
    for pos in ['history_feedback_end','query_id','query_id_template1_transfer','query_end','query_end_template1_transfer','metadata_basic','metadata_recent4']:
        part=predictions[predictions.position==pos].sort_values(['run_id','advisor_id'])
        legacy[pos]=evaluate_scores(part,part.prediction.to_numpy(),rng)
    frame=pd.DataFrame(rows).set_index('position')
    for pos in legacy:
        if pos not in selection:
            for k in ['ci_low','ci_high']:
                assert np.isclose(legacy[pos][k],original.loc[pos,k],atol=1e-10),(pos,k,legacy[pos][k],original.loc[pos,k])
                frame.loc[pos,k]=legacy[pos][k]
    save('probe_summary',frame.reset_index())

def flatten(records):
    rows=[]
    for r in records:
        t=r['task'];e=t['endpoint']
        row={k:t.get(k) for k in ['task_id','condition','task_type','readout','run_id','advisor_id','alpha','random_index','pair_id','source_run','source_label']}
        row.update(endpoint_key=e['key'],label=e.get('label'),baseline=r['baseline']['score'],changed=r['changed']['score'],
            delta=r['delta'],oriented=r['oriented'],kl=r['kl'],relative_l2=r['audit'].get('relative_l2',0),
            baseline_mass=r['baseline'].get('candidate_mass'),changed_mass=r['changed'].get('candidate_mass'))
        assert np.isclose(row['delta'],row['changed']-row['baseline'],atol=1e-10)
        rows.append(row)
    return pd.DataFrame(rows)

def interventions():
    raw=jsonl(DATA/'intervention/dose_records.jsonl')
    data=save('all_tasks',flatten(raw));main=data[data.condition=='full_id_span_steer']
    assert len(data)==1240 and len(main)==840
    assert main.groupby('endpoint_key').baseline.nunique().eq(1).all()
    assert main[main.alpha==0].delta.eq(0).all()
    summaries=[];words=[]
    for (task,alpha),g in main.groupby(['readout','alpha']):
        effect,lo,hi=ci(g.groupby('run_id').delta.mean())
        summaries.append(dict(readout=task,alpha=alpha,mean_delta=effect,ci_low=lo,ci_high=hi,
            mean_probability_change=(expit(g.changed)-expit(g.baseline)).mean() if task=='conflict' else np.nan,
            mean_kl=g.kl.mean(),mean_relative_l2=g.relative_l2.mean()))
    save('dose_summary',summaries);save('steering_run_clustered',summaries)
    characterization=[]
    for task,g in main.groupby('readout'):
        means=g.groupby('alpha').delta.mean().sort_index()
        fit=np.polyval(np.polyfit(means.index,means.to_numpy(),1),means.index)
        r2=1-float(np.square(means.to_numpy()-fit).sum()/np.square(means-means.mean()).sum())
        endpoint_curves=g.pivot(index='endpoint_key',columns='alpha',values='delta').sort_index(axis=1)
        history_curves=g.groupby(['run_id','alpha']).delta.mean().unstack().sort_index(axis=1)
        oriented=g[g.alpha.abs()==4].oriented.mean()
        characterization.append(dict(readout=task,mean_curve_r2=r2,magnitude4_oriented=oriented,
            monotonic_endpoints=int((np.diff(endpoint_curves,axis=1)>=-1e-12).all(axis=1).sum()),
            endpoint_count=len(endpoint_curves),monotonic_histories=int((np.diff(history_curves,axis=1)>=-1e-12).all(axis=1).sum())))
    save('dose_characterization',characterization)
    lex=read(ROOT/'configs/lexicon.json')
    for r in raw:
        t=r['task']
        if t['condition']!='full_id_span_steer' or t['readout']!='verbalizer':continue
        for w in lex['V_H']+lex['V_L']:
            words.append(dict(run_id=t['run_id'],advisor_id=t['advisor_id'],alpha=t['alpha'],word=w,group='H' if w in lex['V_H'] else 'L',
                 baseline_logp=r['baseline']['word_logp'][w],changed_logp=r['changed']['word_logp'][w],
                 delta_logp=r['changed']['word_logp'][w]-r['baseline']['word_logp'][w]))
    words=save('word_level_rows',words)
    save('word_level_summary',words.groupby(['group','word','alpha']).delta_logp.mean().reset_index())
    save('publication_word_effects_clustered',words.groupby(['group','word','alpha']).delta_logp.mean().reset_index().rename(columns={'delta_logp':'mean'}))
    save('word_intervention_records',words)
    patch=flatten(jsonl(DATA/'intervention/patch_records.jsonl'))
    save('intervention_records',pd.concat([main,patch],ignore_index=True))
    save('patch_descriptive',patch.groupby('readout').agg(mean_oriented=('oriented','mean'),mean_kl=('kl','mean'),mean_relative_l2=('relative_l2','mean')).reset_index())
    save('patch_by_direction',patch.groupby(['readout','source_label']).delta.mean().reset_index())
    random=data[data.condition=='random_id_span_steer']
    paired=random.merge(main[['run_id','advisor_id','readout','alpha','oriented']],on=['run_id','advisor_id','readout','alpha'],suffixes=('_random','_main'),validate='many_to_one')
    paired['magnitude']=paired.alpha.abs()
    save('random_paired_rows',paired)
    save('random_comparison',paired.groupby(['readout','magnitude','random_index'])[['oriented_random','oriented_main']].mean().reset_index())
    spill=flatten(jsonl(DATA/'intervention/composed_query_records.jsonl'))
    save('composed_query_records',spill)
    save('composed_query_summary',[dict(mean_oriented=spill.oriented.mean(),rows=len(spill),independent_runs=spill.run_id.nunique())])
    direct=flatten(jsonl(DATA/'intervention/direct_query_records.jsonl'))
    save('direct_query_records',direct)
    save('direct_query_summary',[dict(mean_oriented=direct.oriented.mean(),rows=len(direct),independent_runs=direct.run_id.nunique())])
    references=[]
    for task,part in main[main.alpha==0].groupby('readout'):
        g=part.groupby(['run_id','label']).baseline.mean().unstack()
        mean,lo,hi=ci(g[1]-g[0])
        references.append(dict(readout=task,high_low_gap=mean,ci_low=lo,ci_high=hi))
    save('natural_gap_reference',references)
    zero=main[main.alpha==0][['endpoint_key','readout','baseline']].drop_duplicates()
    patch_base=patch[['endpoint_key','baseline']].drop_duplicates()
    batch=patch_base.merge(zero,on='endpoint_key',suffixes=('_patch','_dose'),validate='one_to_one')
    batch['patch_minus_dose_baseline']=batch.baseline_patch-batch.baseline_dose
    save('baseline_batch_comparison',batch)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-probes',action='store_true');args=parser.parse_args()
    behavior();verbalizer();interventions()
    if not args.skip_probes:probes()
    print('Recomputed behavioral, description, probe, dose, patching and control tables in outputs/tables.')

if __name__=='__main__':main()
