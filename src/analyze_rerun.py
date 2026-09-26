"""Aggregate and plot newly generated results, without using frozen observations."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .paths import ROOT
from .repro import digest, read_completed_json, atomic_json
from .analyze import ci, flatten
from .validation import load_tasks, check_interventions
from .behavior.protocol import parse, normalize


def runtime_manifest(root):
    m=read_completed_json(root/'run_manifest.json')
    if not m or m.get('fingerprint')!=digest(m.get('specification')):
        raise ValueError('Missing or invalid rerun manifest.')
    return m


def behavioral(root,runs):
    manifest=runtime_manifest(root)
    if manifest['specification'].get('kind')!='behavior':
        raise ValueError('Not a behavior rerun root.')
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    tables=[]
    for rid in runs:
        path=root/'raw'/f'run_{rid:03d}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        trials=protocol['runs'][rid]['trials']
        expected={f"run_{rid:03d}_{t['item_id']}":t for t in trials}
        if len(rows)!=76 or {r['task_id'] for r in rows}!=set(expected):
            raise ValueError(f'Incomplete/duplicate behavior session: {rid}')
        high=[];low=[];choices=[]
        for r in rows:
            t=r['trial']
            if t!=expected[r['task_id']] or r.get('run_fingerprint')!=manifest['fingerprint']:
                raise ValueError('Mixed configuration or schedule in behavior data.')
            parsed=parse(normalize(r['parsed']),t['phase'],True,
                         [t.get('participant_a_id'),t.get('participant_b_id')])
            if t['phase']=='tg':
                (high if t['advisor_class'] in ['high','high_honesty'] else low).append(parsed['allocation'])
            elif t['phase']=='conflict':
                choices.append(parsed['chosen_id']==t['high_participant_id'])
        if len(high)!=2 or len(low)!=2 or len(choices)!=8:
            raise ValueError('Post-test group balance changed.')
        tables.append(dict(run_id=rid,tg_high=np.mean(high),tg_low=np.mean(low),
                           tg_gap=np.mean(high)-np.mean(low),conflict_high_rate=np.mean(choices)))
    frame=pd.DataFrame(tables)
    summary=[]
    for metric in ['tg_high','tg_low','tg_gap','conflict_high_rate']:
        mean,lo,hi=ci(frame[metric])
        summary.append(dict(metric=metric,mean=mean,ci_low=lo,ci_high=hi,n_runs=len(runs)))
    return {'behavior_run_metrics':frame,'behavior_summary':pd.DataFrame(summary)},manifest


def mechanistic(root,runs,stages):
    manifest=runtime_manifest(root)
    if manifest['specification'].get('kind')!='mechanism':
        raise ValueError('Not a mechanism rerun root.')
    tables={}
    for stage in stages:
        tasks=[t for t in load_tasks(stage) if t['run_id'] in runs]
        if not tasks:
            raise ValueError(f'No prescribed tasks for stage {stage} and runs {runs}.')
        records=[]
        for t in tasks:
            p=root/stage/(t['task_id']+'.json')
            r=read_completed_json(p)
            if r is None:
                raise ValueError(f'Incomplete stage {stage}: missing {t["task_id"]}')
            if r.get('run_fingerprint')!=manifest['fingerprint']:
                raise ValueError('Mixed runtime fingerprints in new intervention results.')
            records.append(r)
        check_interventions(records,tasks)
        data=flatten(records);name=stage.replace('-','_')
        tables[name+'_records']=data
        if stage=='dose':
            summaries=[]
            for (condition,readout,alpha),g in data.groupby(['condition','readout','alpha']):
                # A random-vector index is a control identity, not an extra history.
                for index,part in (g.groupby('random_index') if condition=='random_id_span_steer'
                                   else [(None,g)]):
                    mean,lo,hi=ci(part.groupby('run_id').delta.mean())
                    summaries.append(dict(condition=condition,readout=readout,alpha=alpha,
                        random_index=index,mean_delta=mean,ci_low=lo,ci_high=hi,
                        n_runs=part.run_id.nunique()))
            tables['dose_summary']=pd.DataFrame(summaries)
        elif stage=='patch':
            tables['patch_summary']=data.groupby(['readout','source_label']).agg(
                mean_delta=('delta','mean'),mean_oriented=('oriented','mean'),
                rows=('delta','size'),recipient_histories=('run_id','nunique')).reset_index()
        else:
            tables[name+'_summary']=pd.DataFrame([dict(mean_oriented=data.oriented.mean(),
                rows=len(data),n_runs=data.run_id.nunique())])
    return tables,manifest


def plot_new(tables,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'mathtext.fontset':'cm','pdf.fonttype':42})
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True)
    if 'behavior_run_metrics' in tables:
        d=tables['behavior_run_metrics']
        fig,axes=plt.subplots(1,2,figsize=(6,2.6),layout='constrained')
        axes[0].boxplot([d.tg_high,d.tg_low],tick_labels=['75% history','25% history'])
        axes[0].set(ylabel='Transfer (points)',ylim=(-.5,10.5))
        axes[1].hist(d.conflict_high_rate,bins=np.arange(-.0625,1.1,.125),color='#276B91')
        axes[1].set(xlabel='High-history choice rate',ylabel='Histories')
        fig.savefig(folder/'new_behavior.pdf');fig.savefig(folder/'new_behavior.png',dpi=200)
        plt.close(fig)
    if 'dose_summary' in tables:
        d=tables['dose_summary'];d=d[d.condition=='full_id_span_steer']
        tasks=['verbalizer','tg','conflict'];fig,axes=plt.subplots(1,3,figsize=(7,2.6),layout='constrained')
        for ax,task,label in zip(axes,tasks,['Description score','Expected transfer','Choice log odds']):
            g=d[d.readout==task].sort_values('alpha')
            ax.plot(g.alpha,g.mean_delta,color='#276B91')
            ax.fill_between(g.alpha,g.ci_low,g.ci_high,color='#276B91',alpha=.25)
            ax.axhline(0,color='gray',linewidth=.6)
            ax.set(xlabel='Steering strength',ylabel=label+' change')
        fig.savefig(folder/'new_dose.pdf');fig.savefig(folder/'new_dose.png',dpi=200)
        plt.close(fig)
    if 'patch_records' in tables:
        d=tables['patch_records'];fig,axes=plt.subplots(1,3,figsize=(7,2.6),layout='constrained')
        for ax,task in zip(axes,['verbalizer','tg','conflict']):
            g=d[d.readout==task]
            ax.boxplot([g[g.source_label==1].delta,g[g.source_label==0].delta],
                       tick_labels=['75 to 25','25 to 75'])
            ax.axhline(0,color='gray',linewidth=.6);ax.set(title=task,ylabel='Score change')
        fig.savefig(folder/'new_patch.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',required=True,choices=['behavior','mechanism'])
    p.add_argument('--input-root',required=True,type=Path)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--runs',help='Explicit comma-separated subset; default is the full prescribed split')
    p.add_argument('--stages',default='dose,patch,composed-query,direct-query')
    p.add_argument('--plot',action='store_true')
    a=p.parse_args();source=a.input_root.resolve();out=a.output_root.resolve()
    protected=[ROOT/'data',ROOT/'src',ROOT/'configs',ROOT/'reference',ROOT/'tests']
    if (out==ROOT or out==source or out.is_relative_to(source) or
            any(out.is_relative_to(p) for p in protected) or out==ROOT/'outputs'):
        p.error('Use a separate new output directory, not the inputs or frozen-analysis outputs.')
    if out.exists() and any(out.iterdir()):
        p.error('Analysis output directory is nonempty; choose a new directory.')
    runs=sorted(set(map(int,a.runs.split(',')))) if a.runs else list(range(30) if a.kind=='behavior' else range(20,30))
    valid=set(range(30) if a.kind=='behavior' else range(20,30))
    if not runs or not set(runs)<=valid:p.error('Requested runs outside the prescribed split.')
    if len(runs)<2:warnings.warn('One history: bootstrap intervals are degenerate, not evidence of precision.')
    if a.kind=='behavior':tables,manifest=behavioral(source,runs)
    else:
        stages=a.stages.split(',')
        if not set(stages)<={'dose','patch','composed-query','direct-query'}:p.error('Unknown stage.')
        tables,manifest=mechanistic(source,runs,stages)
    out.mkdir(parents=True,exist_ok=True);(out/'tables').mkdir()
    for name,frame in tables.items():frame.to_csv(out/'tables'/f'{name}.csv',index=False)
    atomic_json(out/'analysis_provenance.json',dict(source='new_model_outputs',
        run_fingerprint=manifest['fingerprint'],runs=runs,kind=a.kind,stages=a.stages if a.kind=='mechanism' else None))
    if a.plot:plot_new(tables,out)
    print(f'Analyzed new {a.kind} outputs only: {out}')


if __name__=='__main__':main()
