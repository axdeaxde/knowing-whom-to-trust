"""Verify release checksums and compare freshly recomputed values with the paper."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import ROOT, OUTPUT
from .validation import check_interventions, load_tasks

REQUIRED_TABLES=[
    'all_behavior_run_metrics','all_behavior_summary','behavior_summary','behavior_run_metrics',
    'baseline_all_readouts','verbalizer_summary','probe_summary','probe_summary_recomputed_uniform_bootstrap',
    'all_tasks','dose_summary','steering_run_clustered','dose_characterization',
    'word_level_rows','word_level_summary','publication_word_effects_clustered','word_intervention_records',
    'intervention_records','patch_descriptive','patch_by_direction','random_paired_rows','random_comparison',
    'composed_query_records','composed_query_summary','direct_query_records','direct_query_summary',
    'natural_gap_reference','baseline_batch_comparison','lexicon_selection_recomputed']

def verify_checksums(root):
    root=Path(root)
    hashes=root/'MANIFEST.sha256'
    if not hashes.is_file():
        raise ValueError('MANIFEST.sha256 is missing; integrity cannot be verified.')
    lines=hashes.read_text().splitlines()
    if not lines:raise ValueError('Empty checksum manifest.')
    seen=set()
    for line in lines:
        expected,name=line.split('  ',1);relative=Path(name);path=root/relative
        if (relative.is_absolute() or '..' in relative.parts or name in seen or
                path.is_symlink() or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError('Unsafe or duplicate checksum entry.')
        seen.add(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or altered packaged file: '+name)
    return len(seen)

def require_outputs():
    paths=[OUTPUT/'tables'/f'{n}.csv' for n in REQUIRED_TABLES]+[OUTPUT/'tables/lexicon_recomputed.json']
    missing=[str(p.relative_to(OUTPUT)) for p in paths if not p.is_file()]
    if missing:raise ValueError('Incomplete offline workflow; missing: '+', '.join(missing))

def verify_auxiliary():
    from .analyze import flatten, jsonl, ci
    from .select_lexicon import validate_scores, select
    checked=0
    def compare(name,expected,keys):
        nonlocal checked
        actual=pd.read_csv(OUTPUT/'tables'/f'{name}.csv')
        columns=list(expected.columns)
        if set(actual.columns)!=set(columns):raise ValueError('Table schema mismatch: '+name)
        actual=actual[columns].sort_values(keys).reset_index(drop=True)
        expected=expected.sort_values(keys).reset_index(drop=True)
        actual=actual.where(pd.notna(actual),np.nan)
        expected=expected.where(pd.notna(expected),np.nan)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-8,atol=1e-9)
        checked+=len(expected)
    lex=json.loads((ROOT/'configs/lexicon.json').read_text())
    if json.loads((OUTPUT/'tables/lexicon_recomputed.json').read_text())!={k:lex[k] for k in ['V_H','V_L']}:
        raise ValueError('Recomputed lexicon differs from the frozen selection.')
    words=pd.read_csv(ROOT/'data/verbalizer/development_words.csv')
    validate_scores(words)
    selection=select(words,json.loads((ROOT/'configs/mechanism.json').read_text()))
    compare('lexicon_selection_recomputed',selection,['word'])
    stages={'dose':'dose','patch':'patch','composed-query':'composed_query','direct-query':'direct_query'}
    records={};frames={}
    for stage,name in stages.items():
        rows=jsonl(ROOT/f'data/intervention/{name}_records.jsonl')
        checked+=check_interventions(rows,load_tasks(stage))
        records[stage]=rows;frames[stage]=flatten(rows)
    main=frames['dose'][frames['dose'].condition=='full_id_span_steer']
    random=frames['dose'][frames['dose'].condition=='random_id_span_steer']
    paired=random.merge(main[['run_id','advisor_id','readout','alpha','oriented']],
        on=['run_id','advisor_id','readout','alpha'],suffixes=('_random','_main'),validate='many_to_one')
    paired['magnitude']=paired.alpha.abs()
    compare('random_paired_rows',paired,['task_id'])
    comparison=paired.groupby(['readout','magnitude','random_index'])[['oriented_random','oriented_main']].mean().reset_index()
    compare('random_comparison',comparison,['readout','magnitude','random_index'])
    for stage,name in [('composed-query','composed_query'),('direct-query','direct_query')]:
        f=frames[stage]
        compare(name+'_records',f,['task_id'])
        expected=pd.DataFrame([dict(mean_oriented=f.oriented.mean(),rows=len(f),independent_runs=f.run_id.nunique())])
        compare(name+'_summary',expected,['rows'])
    references=[];characterization=[];word_rows=[]
    for task,g in main.groupby('readout'):
        zero=g[g.alpha==0].groupby(['run_id','label']).baseline.mean().unstack()
        mean,lo,hi=ci(zero[1]-zero[0])
        references.append(dict(readout=task,high_low_gap=mean,ci_low=lo,ci_high=hi))
        means=g.groupby('alpha').delta.mean().sort_index()
        fit=np.polyval(np.polyfit(means.index,means.to_numpy(),1),means.index)
        r2=1-float(np.square(means.to_numpy()-fit).sum()/np.square(means-means.mean()).sum())
        endpoints=g.pivot(index='endpoint_key',columns='alpha',values='delta').sort_index(axis=1)
        histories=g.groupby(['run_id','alpha']).delta.mean().unstack().sort_index(axis=1)
        characterization.append(dict(readout=task,mean_curve_r2=r2,magnitude4_oriented=g[g.alpha.abs()==4].oriented.mean(),
            monotonic_endpoints=int((np.diff(endpoints,axis=1)>=-1e-12).all(axis=1).sum()),endpoint_count=len(endpoints),
            monotonic_histories=int((np.diff(histories,axis=1)>=-1e-12).all(axis=1).sum())))
    compare('natural_gap_reference',pd.DataFrame(references),['readout'])
    compare('dose_characterization',pd.DataFrame(characterization),['readout'])
    for r in records['dose']:
        t=r['task']
        if t['condition']!='full_id_span_steer' or t['readout']!='verbalizer':continue
        for w in lex['V_H']+lex['V_L']:
            word_rows.append(dict(group='H' if w in lex['V_H'] else 'L',word=w,alpha=t['alpha'],
                delta_logp=r['changed']['word_logp'][w]-r['baseline']['word_logp'][w]))
    expected=pd.DataFrame(word_rows).groupby(['group','word','alpha']).delta_logp.mean().reset_index()
    compare('word_level_summary',expected,['group','word','alpha'])
    zero=main[main.alpha==0][['endpoint_key','readout','baseline']].drop_duplicates()
    pb=frames['patch'][['endpoint_key','baseline']].drop_duplicates()
    batch=pb.merge(zero,on='endpoint_key',suffixes=('_patch','_dose'),validate='one_to_one')
    batch['patch_minus_dose_baseline']=batch.baseline_patch-batch.baseline_dose
    compare('baseline_batch_comparison',batch,['endpoint_key'])
    return checked

def main():
    p=argparse.ArgumentParser();p.add_argument('--checksums-only',action='store_true');args=p.parse_args()
    checksum_count=verify_checksums(ROOT)
    if args.checksums_only:
        print('All packaged file checksums match.');return
    require_outputs()
    ref=json.loads((ROOT/'reference/paper_values.json').read_text());count=0
    for key,file,indices,rename in [
        ('behavior','all_behavior_summary',['model'],{'conflict_high_rate':'conflict_rate'}),
        ('dose','dose_summary',['readout','alpha'],{}),('probe','probe_summary',['position'],{}),
        ('patch','patch_descriptive',['readout'],{})]:
        actual=pd.read_csv(OUTPUT/'tables'/f'{file}.csv').set_index(indices)
        for row in ref[key]:
            index=tuple(row[k] for k in indices) if len(indices)>1 else row[indices[0]]
            for col,value in row.items():
                if col in indices:continue
                assert np.isclose(actual.loc[index,rename.get(col,col)],value,rtol=1e-8,atol=1e-9),(key,index,col,value,actual.loc[index,rename.get(col,col)])
                count+=1
    v=pd.read_csv(OUTPUT/'tables/verbalizer_summary.csv').iloc[0]
    for k in ['auc','auc_low','auc_high','high_low_gap','gap_low','gap_high','high_mean','low_mean']:
        assert np.isclose(v[k],ref['description'][k],atol=1e-9),(k,v[k],ref['description'][k]);count+=1
    audits=json.loads((ROOT/'data/validation/identity_hook.json').read_text())
    assert all(c['cleanup'] and c['cache_branch_order_equal'] and c['deep_copy_alias_free'] for c in audits)
    assert all(r['max_abs_logp_error']==0 for c in audits for r in c['identity_and_zero'])
    auxiliary=verify_auxiliary()
    summary=dict(numeric_checks=count,auxiliary_rows_checked=auxiliary,checksums=checksum_count,
                 identity_checks=sum(len(c['identity_and_zero']) for c in audits),passed=True)
    (OUTPUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':main()
