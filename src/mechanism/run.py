"""Rerun the frozen readouts/interventions; no development sweep is required."""
import argparse
import json
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
from ..paths import ROOT, OUTPUT
from ..repro import (atomic_json, atomic_npz, read_completed_json, bind_run,
    package_signature, require_runtime, digest, file_hash, directory_lock,
    same_measurement, activation_complete, checkpoint_fingerprint,
    validate_checkpoint, validate_runtime, versions)
from ..validation import ZERO_DOSE_TOLERANCE

def read(path):return json.loads(Path(path).read_text())

@lru_cache(maxsize=30)
def history(run_id):
    from .common import build_history
    return build_history(ROOT/f'data/behavior/qwen27b/raw/run_{run_id:03d}.jsonl').messages

def messages(endpoint,run_id=None):
    from .engine import branch
    return branch(history(endpoint['run_id'] if run_id is None else run_id),endpoint['prompt'])

def span(engine,endpoint,run_id=None,intervention_id=None):
    text=engine.render(messages(endpoint,run_id))
    start=text.rindex(endpoint['prompt'])
    advisor=intervention_id or endpoint['advisor_id']
    if intervention_id:marker=advisor;offset=0
    elif endpoint['readout']=='tg':marker='Receiver: '+advisor;offset=len('Receiver: ')
    elif endpoint['readout']=='conflict':marker=advisor+' recommends';offset=0
    else:marker=advisor;offset=0
    char=start+endpoint['prompt'].index(marker)+offset
    encoded=engine.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=encoded['offset_mapping']
    positions=[i for i,(a,b) in enumerate(offsets) if a<char+len(advisor) and b>char]
    if not positions or len(positions)!=4:
        raise ValueError('Identity span must contain the frozen four tokens.')
    decoded=engine.tokenizer.decode([encoded['input_ids'][i] for i in positions],
                                     skip_special_tokens=False).strip()
    if decoded!=advisor:
        raise ValueError('Identity span does not decode to the intended participant.')
    original_run=run_id is None or run_id==endpoint['run_id']
    if original_run and 'query_id_span' in endpoint and not intervention_id:
        if positions!=endpoint['query_id_span'] or len(offsets)!=endpoint['input_tokens']:
            raise ValueError('Frozen query positions or prompt length changed.')
    if 'prompt_sha256' in endpoint and digest(endpoint['prompt'])!=endpoint['prompt_sha256']:
        raise ValueError('Frozen endpoint prompt hash mismatch.')
    return positions

def bind_engine(engine,args):
    return bind_run(args.out, dict(kind='mechanism',runtime=require_runtime(engine),
                                  package_signature=package_signature()))

def baseline_guard(out,key,measurement,fingerprint):
    folder=out/'baselines';folder.mkdir(parents=True,exist_ok=True)
    name=digest(key);path=folder/(name+'.json')
    with directory_lock(folder/(name+'.lock')):
        prior=read_completed_json(path)
        if prior is not None:
            if (prior['run_fingerprint']!=fingerprint or
                    not same_measurement(prior['measurement'],measurement)):
                raise ValueError('Same-endpoint baseline drift across executions. '
                                 'Do not mix batches; use a new output root.')
        else:atomic_json(path,dict(key=key,measurement=measurement,run_fingerprint=fingerprint))

def measure(engine,state,endpoint,paths,lexicon):
    if endpoint['readout']=='verbalizer':
        from .complete_answers import score_complete
        from .engine import set_score
        values=score_complete(engine,state,paths)
        return dict(score=set_score(values,lexicon),word_logp={r['word']:r['logp'] for r in values},
            candidate_mass=float(sum(np.exp(r['logp']) for r in values)),metric='S_complete_answer')
    from .decision import decision_score
    raw=ROOT/f"data/behavior/qwen27b/raw/run_{endpoint['run_id']:03d}.jsonl"
    trials=[json.loads(s)['trial'] for s in raw.read_text().splitlines()]
    trial=next(t for t in trials if t['phase']==endpoint['readout'] and t['trial_id']==endpoint['trial_id'])
    choices=[str(i) for i in range(11)] if endpoint['readout']=='tg' else [trial['participant_a_id'],trial['participant_b_id']]
    return decision_score(engine,state,dict(candidates=choices,target=endpoint['advisor_id']),endpoint['readout'])

def run_interventions(engine,args):
    from .engine import distribution_kl
    fingerprint=bind_engine(engine,args)
    cfgname={'dose':'dose_tasks','patch':'patch_tasks','composed-query':'composed_query_tasks','direct-query':'direct_query_tasks'}[args.stage]
    tasks=read(ROOT/f'configs/{cfgname}.json')
    if args.readout:tasks=[t for t in tasks if t['readout']==args.readout]
    if args.runs:tasks=[t for t in tasks if t['run_id'] in args.runs]
    tasks=[t for i,t in enumerate(tasks) if i%args.shards==args.shard]
    if args.limit:tasks=tasks[:args.limit]
    lex=read(ROOT/'configs/lexicon.json');words=set(lex['V_H']+lex['V_L'])
    paths=read(ROOT/'configs/answer_paths.json')
    paths['paths']=[p for p in paths['paths'] if p['word'] in words]
    paths['word_prefixes']=[p for p in paths['word_prefixes'] if p['word'] in words]
    with np.load(ROOT/'configs/steering_direction.npz') as z:
        direction=z['query_id__43__mean'].copy();scale=float(z['query_id__43__mean__sd'])
    cache={}
    for i,t in enumerate(tasks):
        dest=args.out/args.stage/f"{t['task_id']}.json"
        if dest.exists():
            prior=read_completed_json(dest)
            if prior.get('task')!=t or prior.get('run_fingerprint')!=fingerprint:
                raise ValueError('Cached task/configuration mismatch.')
            if not {'baseline','changed','delta','oriented','kl','audit'}.issubset(prior):
                raise ValueError('Incomplete result. Move it aside and rerun this task.')
            continue
        e=t['endpoint'];key=e['key'];suffix=e.get('suffix','')
        positions=span(engine,e,intervention_id=t.get('intervention_advisor_id'))
        if key not in cache:
            state=engine.prefill(messages(e),suffix=suffix)
            if 'decision_input_tokens' in e:assert state['length']==e['decision_input_tokens']
            cache[key]=dict(measurement=measure(engine,state,e,paths,lex),logp=state['logp'].cpu().numpy().copy())
            baseline_guard(args.out,key,cache[key]['measurement'],fingerprint)
            del state
        base=cache[key]
        if t['task_type']=='patch':
            source=engine.prefill(messages(e,t['source_run']),capture={'layers':[t['layer']],
                'positions':span(engine,e,t['source_run'])})
            vectors=source['captured'][t['layer']].copy();del source
            assert len(vectors)==len(positions)
            edit=dict(layer=t['layer'],positions=positions,vectors=vectors,kind='patch')
        elif t['alpha']==0:edit=None
        else:
            unit=direction.copy()
            if t['condition']=='random_id_span_steer':
                rng=np.random.default_rng(t['random_seed']+t['layer']*1000+t['random_index'])
                unit=rng.normal(size=len(unit)).astype(np.float32);unit/=np.linalg.norm(unit)
            vector=unit*scale*t['alpha']/np.sqrt(len(positions))
            edit=dict(layer=t['layer'],positions=positions,vectors=[vector]*len(positions),units=[unit]*len(positions),kind='steer')
        changed=engine.prefill(messages(e),edit=edit,suffix=suffix)
        value=measure(engine,changed,e,paths,lex)
        delta=float(value['score']-base['measurement']['score'])
        sign=2*t['source_label']-1 if t['task_type']=='patch' else np.sign(t['alpha'])
        row=dict(task=t,baseline=base['measurement'],changed=value,delta=delta,oriented=float(delta*sign),
                 kl=distribution_kl(base['logp'],changed['logp'].cpu().numpy()),audit=changed['audit'],
                 input_tokens=changed['length'],run_fingerprint=fingerprint)
        if t.get('alpha')==0:
            assert (abs(delta)<=ZERO_DOSE_TOLERANCE and
                    np.max(np.abs(base['logp']-changed['logp'].cpu().numpy()))<=ZERO_DOSE_TOLERANCE)
        atomic_json(dest,row)
        print(f'{args.stage}: {i+1}/{len(tasks)} {t["task_id"]}',flush=True);del changed

def run_extract(engine,args):
    from .engine import branch
    fingerprint=bind_engine(engine,args)
    meta=read(ROOT/'data/probe/query_metadata.json')
    cfg=read(ROOT/'configs/mechanism.json')
    runs=args.runs or list(range(30))
    for rid in runs:
        if rid%args.shards!=args.shard:continue
        dest=args.out/'activations'/f'run_{rid:03d}.npz'
        rows=[r for r in meta if r['run_id']==rid]
        if activation_complete(dest,rows,fingerprint):continue
        advisors=sorted({r['advisor_id'] for r in rows})
        history_positions=[next(r['checkpoint_position'] for r in rows if r['advisor_id']==a) for a in advisors]
        layers=list(range(64))
        saved={};handles=[]
        from .common import primary_tensor
        try:
            for layer,module in enumerate(engine.layers):
                def capture(module,args,output,layer=layer):
                    saved[layer]=primary_tensor(output)[0,history_positions].detach().float().cpu().numpy().astype(np.float16)
                handles.append(module.register_forward_hook(capture))
            inputs,_=engine.encode(engine.render(history(rid),add_generation_prompt=False))
            engine.final_logits(inputs)
        finally:
            for h in handles:h.remove()
        h=np.stack([saved[k] for k in layers],axis=1);queries=[]
        for r in rows:
            prompt=cfg['templates'][r['template_id']].format(advisor_id=r['advisor_id'])
            check=dict(r,prompt=prompt,readout='verbalizer')
            query_span=span(engine,check)
            if query_span[-1]!=r['query_id_position'] or r['query_end_position']!=r['input_tokens']-1:
                raise ValueError('Frozen extraction query position changed.')
            state=engine.prefill(branch(history(rid),prompt),capture={'layers':layers,
                'positions':[r['query_id_position'],r['query_end_position']]})
            assert state['length']==r['input_tokens']
            queries.append(np.stack([state['captured'][k] for k in layers],axis=1).astype(np.float16));del state
        dest.parent.mkdir(parents=True,exist_ok=True)
        atomic_npz(dest,history=h,query=np.stack(queries))
        atomic_json(dest.with_suffix('.json'),dict(history_advisors=advisors,queries=rows,
            run_fingerprint=fingerprint,npz_sha256=file_hash(dest)))
        print('Extracted',rid,flush=True)

def run_discovery(engine,args):
    from .engine import branch
    from .candidate_search import Search
    fingerprint=bind_engine(engine,args)
    cfg=read(ROOT/'configs/mechanism.json');search=Search(engine)
    meta=read(ROOT/'data/probe/query_metadata.json')
    rows=[r for r in meta if r['run_id']<20]
    if args.runs:rows=[r for r in rows if r['run_id'] in args.runs]
    rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard]
    if args.limit:rows=rows[:args.limit]
    for r in rows:
        key=f"r{r['run_id']:03d}_{r['advisor_id'].split()[-1]}_{r['template_id']}"
        dest=args.out/'discovery'/f'{key}.json'
        if dest.exists():
            prior=read_completed_json(dest)
            if (prior.get('query')!=r or prior.get('run_fingerprint')!=fingerprint or
                    not {'candidates','natural','suffix_forwards'}.issubset(prior)):
                raise ValueError('Discovery task/configuration mismatch.')
            continue
        prompt=cfg['templates'][r['template_id']].format(advisor_id=r['advisor_id'])
        state=engine.prefill(branch(history(r['run_id']),prompt))
        seed=read(ROOT/'configs/discovery_seeds.json')[key]
        result=search.run(state,seed)
        atomic_json(dest,dict(query=r,run_fingerprint=fingerprint,**result))
        del state;print('Discovered',key,flush=True)

def run_word_scores(engine,args):
    from .engine import branch
    from .complete_answers import score_complete
    fingerprint=bind_engine(engine,args)
    cfg=read(ROOT/'configs/mechanism.json');manifest=read(ROOT/'configs/answer_paths.json')
    meta=read(ROOT/'data/probe/query_metadata.json')
    rows=[r for r in meta if r['run_id']<20]
    if args.runs:rows=[r for r in rows if r['run_id'] in args.runs]
    rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard]
    if args.limit:rows=rows[:args.limit]
    for r in rows:
        key=f"r{r['run_id']:03d}_{r['advisor_id'].split()[-1]}_{r['template_id']}"
        dest=args.out/'word_scores'/f'{key}.json'
        if dest.exists():
            prior=read_completed_json(dest)
            expected_words={p['word'] for p in manifest['paths']}
            if (prior.get('metadata')!=r or prior.get('run_fingerprint')!=fingerprint or
                    len(prior.get('scores',[]))!=len(expected_words) or
                    {s['word'] for s in prior['scores']}!=expected_words):
                raise ValueError('Word-score task incomplete or configuration changed.')
            continue
        prompt=cfg['templates'][r['template_id']].format(advisor_id=r['advisor_id'])
        state=engine.prefill(branch(history(r['run_id']),prompt))
        scores=score_complete(engine,state,manifest)
        atomic_json(dest,dict(metadata=r,scores=scores,run_fingerprint=fingerprint))
        del state;print('Scored',key,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dose','patch','composed-query','direct-query','extract','discover','word-scores'])
    p.add_argument('--model-path',required=True);p.add_argument('--runs');p.add_argument('--readout',choices=['verbalizer','tg','conflict'])
    p.add_argument('--out',type=Path,default=OUTPUT/'rerun_mechanism');p.add_argument('--limit',type=int)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();assert 0<=a.shard<a.shards
    a.runs=[int(x) for x in a.runs.split(',')] if a.runs else None
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.manual_seed(20260924);torch.use_deterministic_algorithms(True)
    from .engine import Engine
    cfg=read(ROOT/'configs/mechanism.json');cfg['model']['path']=a.model_path
    checkpoint=checkpoint_fingerprint(a.model_path)
    validate_checkpoint(checkpoint,read(ROOT/'configs/runtime_versions.json')['qwen27b']['config_hashes'])
    engine=Engine(cfg)
    runtime=validate_runtime(engine,read(ROOT/'configs/mechanism_runtime.json'))
    engine.release_identity=dict(runtime=runtime,checkpoint_files=checkpoint,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        software=versions(),cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        deterministic_algorithms=True,cublas_workspace_config=':4096:8',attention='sdpa')
    if a.stage=='extract':run_extract(engine,a)
    elif a.stage=='discover':run_discovery(engine,a)
    elif a.stage=='word-scores':run_word_scores(engine,a)
    else:run_interventions(engine,a)

if __name__=='__main__':main()
