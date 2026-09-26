"""Data completeness and score checks shared by frozen and regenerated inputs."""
import json
import numpy as np
from scipy.special import logsumexp, softmax
from .paths import ROOT
from .repro import same_measurement

ZERO_DOSE_TOLERANCE=1e-5


def load_tasks(stage):
    name={'dose':'dose','patch':'patch','composed-query':'composed_query',
          'direct-query':'direct_query'}[stage]
    return json.loads((ROOT/f'configs/{name}_tasks.json').read_text())


def check_interventions(records, tasks):
    expected={t['task_id']:t for t in tasks}
    actual=[r['task']['task_id'] for r in records]
    if len(set(actual))!=len(actual) or set(actual)!=set(expected):
        raise ValueError('Missing, duplicate or unexpected intervention task IDs.')
    lex=json.loads((ROOT/'configs/lexicon.json').read_text())
    baselines={}
    for r in records:
        t=r['task']
        if t!=expected[t['task_id']]:
            raise ValueError('Task specification differs from the frozen manifest.')
        for name in ['baseline','changed']:
            v=r[name]
            if t['readout']=='verbalizer':
                p=v['word_logp']
                score=(logsumexp([p[w] for w in lex['V_H']])-np.log(len(lex['V_H']))-
                       logsumexp([p[w] for w in lex['V_L']])+np.log(len(lex['V_L'])))
            elif t['readout']=='tg':
                if v['candidates']!=[str(i) for i in range(11)]:
                    raise ValueError('Allocation candidate order changed.')
                probabilities=softmax(v['candidate_logp'])
                if not np.allclose(probabilities,v['candidate_probability'],rtol=0,atol=1e-9):
                    raise ValueError('Candidate normalization mismatch.')
                score=probabilities @ np.arange(11)
            else:
                i=v['candidates'].index(t.get('advisor_id',t['endpoint']['advisor_id']))
                score=v['candidate_logp'][i]-v['candidate_logp'][1-i]
            if not np.isfinite(score) or not np.isclose(score,v['score'],rtol=0,atol=1e-9):
                raise ValueError('Readout score does not match candidate probabilities.')
        delta=r['changed']['score']-r['baseline']['score']
        sign=2*t['source_label']-1 if t['task_type']=='patch' else np.sign(t['alpha'])
        if not np.isclose(delta,r['delta'],rtol=0,atol=1e-9):
            raise ValueError('Changed-minus-own-baseline mismatch.')
        if not np.isclose(delta*sign,r['oriented'],rtol=0,atol=1e-9):
            raise ValueError('Oriented intervention sign mismatch.')
        if not np.isfinite(r['kl']) or r['kl'] < -1e-8:
            raise ValueError('Invalid saved KL.')
        if t.get('alpha')==0 and (abs(delta)>ZERO_DOSE_TOLERANCE or abs(r['kl'])>ZERO_DOSE_TOLERANCE):
            raise ValueError('Zero dose changes the readout.')
        key=t['endpoint']['key']
        if key in baselines and not same_measurement(baselines[key],r['baseline']):
            raise ValueError('Different baselines for the same endpoint within one stage.')
        baselines[key]=r['baseline']
    return len(records)
