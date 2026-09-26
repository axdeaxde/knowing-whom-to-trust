"""Recompute the frozen complete-adjective selection from development scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import ROOT, DATA, OUTPUT
from .repro import read_completed_json, digest

def validate_scores(data):
    metadata=json.loads((ROOT/'data/probe/query_metadata.json').read_text())
    queries={(r['run_id'],r['advisor_id'],r['template_id']):r['label']
             for r in metadata if r['run_id']<20}
    words=set(pd.read_csv(ROOT/'data/verbalizer/candidate_selection.csv').word)
    required={'run_id','advisor_id','template_id','label','word','logp','termination_probability'}
    if not required.issubset(data.columns):
        raise ValueError('Missing score columns: '+str(sorted(required-set(data.columns))))
    keys=['run_id','advisor_id','template_id','word']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate query/word scores.')
    actual=set(map(tuple,data[['run_id','advisor_id','template_id']].drop_duplicates().to_numpy()))
    if actual!=set(queries):
        raise ValueError(f'Incomplete query design: missing={sorted(set(queries)-actual)}, '
                         f'unexpected={sorted(actual-set(queries))}')
    for key,g in data.groupby(keys[:3]):
        if set(g.word)!=words or len(g)!=len(words):
            raise ValueError(f'Candidate set mismatch for {key}.')
        if not g.label.eq(queries[key]).all():
            raise ValueError(f'Label mismatch for {key}.')
    values=data[['logp','termination_probability']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:,0]>1e-5).any() or
            (values[:,1]<0).any() or (values[:,1]>1.00001).any()):
        raise ValueError('Scores contain non-finite or invalid probabilities.')
    return data

def select(data,cfg):
    data=data.copy()
    data['rank']=data.groupby(['run_id','advisor_id','template_id']).logp.rank(ascending=False,method='min')
    rng=np.random.default_rng(cfg['seed'])
    bootstrap=rng.integers(20,size=(cfg['selection']['bootstrap_repeats'],20))
    criteria=cfg['selection'];rows=[]
    for word,part in data.groupby('word'):
        diff=part.groupby(['template_id','run_id','label']).logp.mean().unstack()
        diff=(diff[1]-diff[0]).unstack(0)
        gap=diff.mean(axis=1).to_numpy();mean=float(gap.mean());sign=1 if mean>0 else -1
        sampled=gap[bootstrap].mean(axis=1)
        endpoints=part.groupby(['run_id','advisor_id','label']).logp.mean().reset_index()
        hi,lo=[endpoints.loc[endpoints.label==v,'logp'].to_numpy() for v in [1,0]]
        effect=(hi.mean()-lo.mean())/max(np.sqrt((hi.var(ddof=1)+lo.var(ddof=1))/2),1e-12)
        stability=float(np.mean(sampled*sign>0));termination=float(part.termination_probability.median())
        coverage=float(part.groupby('template_id')['rank'].apply(lambda x:(x<=20).mean()).min())
        selected=all([all(diff[t].mean()*sign>0 for t in diff),abs(effect)>=criteria['min_abs_cohens_d'],
            stability>=criteria['bootstrap_sign_rate'],termination>=criteria['median_boundary_probability'],coverage>=criteria['min_template_top20_rate']])
        rows.append(dict(word=word,group='H' if sign>0 else 'L',difference=mean,cohens_d=effect,
            bootstrap_sign_rate=stability,median_termination_probability=termination,top20_coverage_min=coverage,selected=selected))
    return pd.DataFrame(rows).sort_values('difference',ascending=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scores-dir',type=Path)
    p.add_argument('--output-root',type=Path,default=OUTPUT);a=p.parse_args()
    protected=[ROOT/'data',ROOT/'src',ROOT/'configs',ROOT/'reference',ROOT/'tests']
    if a.output_root.resolve()==ROOT or any(a.output_root.resolve().is_relative_to(p) for p in protected):
        p.error('Do not write analysis results into the supplied data.')
    if a.scores_dir:
        manifest=read_completed_json(a.scores_dir.parent/'run_manifest.json')
        if not manifest or manifest.get('fingerprint')!=digest(manifest.get('specification')):
            raise ValueError('Missing or invalid word-score runtime manifest.')
        rows=[]
        for path in sorted(a.scores_dir.glob('*.json')):
            d=json.loads(path.read_text());m=d['metadata']
            if d.get('run_fingerprint')!=manifest['fingerprint']:
                raise ValueError('Mixed runtime fingerprints in word scores.')
            assert m['run_id']<20
            for r in d['scores']:
                rows.append({k:m[k] for k in ['run_id','advisor_id','label','template_id']}|
                    {k:r[k] for k in ['word','logp','termination_probability']})
        data=pd.DataFrame(rows)
    else:data=pd.read_csv(DATA/'verbalizer/development_words.csv')
    validate_scores(data)
    cfg=json.loads((ROOT/'configs/mechanism.json').read_text())
    result=select(data,cfg);out=a.output_root/'tables';out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'lexicon_selection_recomputed.csv',index=False)
    lex={f'V_{group}':result.loc[result.selected & (result.group==group),'word'].tolist() for group in ['H','L']}
    if not a.scores_dir:
        frozen=json.loads((ROOT/'configs/lexicon.json').read_text())
        assert all(lex[k]==frozen[k] for k in lex)
    (out/'lexicon_recomputed.json').write_text(json.dumps(lex,indent=2)+'\n')
    print({k:len(v) for k,v in lex.items()})

if __name__=='__main__':main()
