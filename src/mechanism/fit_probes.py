"""Fit grouped logistic probes from regenerated activations; derive the L43 vector."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ..paths import ROOT, OUTPUT
from . import probe_helpers as h
from ..repro import activation_complete, read_completed_json

def features(folder):
    folder=Path(folder)
    manifest=read_completed_json(folder.parent/'run_manifest.json')
    if manifest is None:
        raise ValueError('Activation run manifest is missing; use a new extraction output root.')
    query_meta=json.loads((ROOT/'data/probe/query_metadata.json').read_text())
    meta=[];features={k:[] for k in ['history_feedback_end','query_id','query_end','query_id_template1_transfer','query_end_template1_transfer']}
    for rid in range(30):
        path=folder/f'run_{rid:03d}.npz'
        expected=[r for r in query_meta if r['run_id']==rid]
        if not activation_complete(path,expected,manifest['fingerprint']):
            raise ValueError(f'Incomplete activation pair for run {rid}; resume extraction first.')
        m=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            for advisor in m['history_advisors']:
                i=next(i for i,r in enumerate(m['queries']) if r['advisor_id']==advisor and r['template_id']=='template_2')
                j=next(i for i,r in enumerate(m['queries']) if r['advisor_id']==advisor and r['template_id']=='template_1')
                meta.append(m['queries'][i]);features['history_feedback_end'].append(z['history'][m['history_advisors'].index(advisor)])
                for pos,index in [('query_id',0),('query_end',1)]:
                    features[pos].append(z['query'][i,index]);features[pos+'_template1_transfer'].append(z['query'][j,index])
    return pd.DataFrame(meta),{k:np.stack(v) for k,v in features.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--activations',type=Path,required=True)
    p.add_argument('--out',type=Path,default=OUTPUT/'fitted_probes');p.add_argument('--ridge',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'configs/mechanism.json').read_text())
    meta,x=features(a.activations);dev=(meta.run_id<20).to_numpy();test=~dev
    predictions=[];selection={};coefficients={}
    with threadpool_limits(limits=2):
        for pos in ['history_feedback_end','query_id','query_end']:
            m=meta.loc[dev].reset_index(drop=True)
            choice=h.select(m,x[pos][dev],cfg,a.out/f'{pos}_selection.csv')
            clf=h.fit(x[pos][dev,choice['layer']].astype(float),m.label.to_numpy(),choice['C'])
            scale,model=clf.steps[0][1],clf.steps[1][1]
            coefficients[pos]=dict(mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),layer=choice['layer'])
            selection[pos]=choice
            for name in [pos]+([pos+'_template1_transfer'] if pos!='history_feedback_end' else []):
                probabilities=clf.predict_proba(x[name][test,choice['layer']].astype(float))[:,1]
                for r,value in zip(meta.loc[test].to_dict('records'),probabilities):
                    predictions.append({k:r[k] for k in ['run_id','advisor_id','label']}|dict(position=name,prediction=float(value)))
        for recent in [False,True]:
            clf=h.metadata_model(meta.loc[dev],recent)
            for r,value in zip(meta.loc[test].to_dict('records'),clf.predict_proba(meta.loc[test])[:,1]):
                predictions.append({k:r[k] for k in ['run_id','advisor_id','label']}|dict(position='metadata_recent4' if recent else 'metadata_basic',prediction=float(value)))
        if a.ridge:
            score=pd.read_csv(ROOT/'data/verbalizer/development_words.csv')
            from scipy.special import logsumexp
            lex=json.loads((ROOT/'configs/lexicon.json').read_text())
            score=score[score.template_id=='template_2'].pivot(index=['run_id','advisor_id'],columns='word',values='logp')
            y=pd.Series(logsumexp(score[lex['V_H']],axis=1)-np.log(len(lex['V_H']))-logsumexp(score[lex['V_L']],axis=1)+np.log(len(lex['V_L'])),index=score.index)
            m=meta.loc[dev];target=y.loc[pd.MultiIndex.from_frame(m[['run_id','advisor_id']])].to_numpy()
            for pos in ['history_feedback_end','query_id','query_end']:
                rows=[]
                def scan(layers):
                    for layer in layers:
                        for alpha in cfg['probe']['ridge_grid']:
                            pred=np.zeros(len(m));values=x[pos][dev,layer].astype(float)
                            for train,valid in GroupKFold(5).split(values,target,m.run_id):
                                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='lsqr')).fit(values[train],target[train])
                                pred[valid]=model.predict(values[valid])
                            rows.append(dict(layer=layer,alpha=alpha,spearman=float(spearmanr(target,pred).statistic)))
                scan(cfg['probe']['coarse_layers']);best=max(rows,key=lambda r:(r['spearman'],-r['layer'],-r['alpha']))
                scan(range(max(0,best['layer']-3),min(64,best['layer']+4)))
                pd.DataFrame(rows).to_csv(a.out/f'{pos}_ridge_selection.csv',index=False)
    matrix=x['query_id'][dev,43].astype(float);labels=meta.loc[dev,'label'].to_numpy()
    direction=matrix[labels==1].mean(0)-matrix[labels==0].mean(0);direction/=np.linalg.norm(direction)
    np.savez(a.out/'steering_direction.npz',query_id__43__mean=direction,query_id__43__mean__sd=np.std(matrix@direction,ddof=1))
    pd.DataFrame(predictions).to_csv(a.out/'predictions.csv',index=False)
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2));(a.out/'coefficients.json').write_text(json.dumps(coefficients))
    print('Fitted classifiers and direction saved to',a.out)

if __name__=='__main__':main()
