"""Rerun the frozen 76-trial protocol with a local model or explicit API access."""
import argparse
import json
import os
import urllib.request
from pathlib import Path
from ..paths import ROOT, OUTPUT
from .runner import run_one
from .api_formats import build_request, unpack, NoRedirect
from .model_client import has_active_thought
from ..repro import checkpoint_fingerprint, digest, versions


class ApiClient:
    is_api=True
    def __init__(self, model, base_url, max_tokens, max_requests):
        self.c=model;self.base=base_url.rstrip('/');self.max_tokens=max_tokens
        self.key=os.environ['EXPERIMENT_API_KEY']
        self.remaining=max_requests
        self.opener=urllib.request.build_opener(NoRedirect())

    def complete_request(self,messages,task_id,attempt):
        if self.remaining<=0:raise RuntimeError('API request limit reached')
        self.remaining-=1
        if self.c['backend']=='deepseek_api':
            path='/chat/completions'
            payload=dict(self.c['request_settings'],messages=messages)
        else:
            path,payload=build_request(self.c,messages,self.max_tokens)
        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'}
        if self.c['backend']=='anthropic_api':
            headers.update({'x-api-key':self.key,'anthropic-version':'2023-06-01'})
        req=urllib.request.Request(self.base+path,data=json.dumps(payload).encode(),headers=headers)
        # Do not log credentials, HTTP headers or provider error bodies.
        try:
            with self.opener.open(req,timeout=240) as response:body=json.load(response)
        except Exception as exc:
            raise RuntimeError('API transport failed: '+type(exc).__name__) from None
        if self.c['backend']=='deepseek_api':
            normalized=dict(self.c,backend='openai_api')
            result=unpack(normalized,body)
        else:result=unpack(self.c,body)
        expected=self.c.get('expected_response_model',self.c['api_model'])
        if result['response_model']!=expected:raise RuntimeError('Returned model identifier changed')
        assert not result['context_truncated']
        return dict(content=result['content'],raw_response=result['content'],response_model=result['response_model'],
            input_tokens=result['input_tokens'],output_tokens=result['output_tokens'],finish_status=result['finish_status'],
            non_thinking_ok=not result['thinking'] and not has_active_thought(result['content']),
            sampling_seed=None,sampling_seed_applied=False,generation_config=payload.get('generationConfig',self.c['request_settings']))


def main():
    p=argparse.ArgumentParser()
    config=json.loads((ROOT/'configs/models.json').read_text())
    p.add_argument('--model',required=True,choices=sorted(config['models']));p.add_argument('--model-path');p.add_argument('--gpus',default='0,1')
    p.add_argument('--runs',default='0');p.add_argument('--api-base-url');p.add_argument('--allow-paid-api',action='store_true')
    p.add_argument('--max-api-requests',type=int,default=100);p.add_argument('--out',type=Path,default=OUTPUT/'rerun_behavior')
    a=p.parse_args();config=json.loads((ROOT/'configs/models.json').read_text());protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    model=dict(config['models'][a.model]);settings=dict(config,**model['run_settings'])
    runs=[int(x) for x in a.runs.split(',')];assert all(0<=r<30 for r in runs)
    settings['run_ids']=runs
    out=a.out/a.model;out.mkdir(parents=True,exist_ok=True)
    if model['backend'].endswith('_api'):
        if not a.allow_paid_api or not a.api_base_url:p.error('API runs require --allow-paid-api and --api-base-url')
        client=ApiClient(model,a.api_base_url,settings['max_new_tokens'],a.max_api_requests)
        client.release_identity=dict(model=a.model,software=versions(),
            service_root_sha256=digest(a.api_base_url.rstrip('/')),model_settings=model)
    else:
        if not a.model_path:p.error('--model-path is required for local inference')
        os.environ['CUDA_VISIBLE_DEVICES']=a.gpus
        model.update(model_path=a.model_path,gpus=a.gpus)
        weights=checkpoint_fingerprint(a.model_path)
        import importlib
        adapter=model['adapter']
        module='model_client' if adapter=='model_client' else 'local_client_'+adapter
        LocalClient=importlib.import_module('src.behavior.'+module).LocalClient
        client=LocalClient(model,settings['max_new_tokens'])
        client.release_identity=dict(model=a.model,software=versions(),checkpoint_files=weights,
            model_settings={k:v for k,v in model.items() if k!='model_path'})
        if a.model=='gemma31b':
            from .attention_core import install
            messages=[{'role':'system','content':protocol['system']},{'role':'user','content':protocol['tag_intro']+'\n\n'+protocol['runs'][0]['trials'][0]['prompt']}]
            install(client,messages)
    for rid in runs:run_one(client,out,settings,protocol,protocol['runs'][rid])

if __name__=='__main__':main()
