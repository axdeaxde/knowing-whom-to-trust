"""Candidate-normalized TG expectation and conflict log odds."""
import numpy as np
from scipy.special import logsumexp, softmax

def decision_score(engine, base, info, kind):
    paths = []
    for text in info['candidates']:
        for prefix in ('', ' ') if kind == 'tg' else ('',):
            form = prefix + text
            paths.append(dict(word=text, form=form, ids=engine.tokenizer.encode(form, add_special_tokens=False)))
    boundaries = engine.number_boundaries if kind == 'tg' else engine.string_boundaries
    scored = engine.score(base, paths, boundary_ids=boundaries)
    bytext = {r['word']: r['logp'] for r in scored}
    ordered = np.array([bytext[c] for c in info['candidates']])
    prob = softmax(ordered)
    mass = float(np.exp(logsumexp(ordered)))
    assert 0 < mass < 1.00001, ('JSON value paths do not form coherent disjoint events', kind, mass)
    if kind == 'tg':
        metric = float(prob @ np.arange(11))
    else:
        index = info['candidates'].index(info['target'])
        metric = float(ordered[index] - ordered[1 - index])
    return dict(score=metric, candidate_logp=ordered.tolist(), candidate_probability=prob.tolist(), candidate_mass=mass, candidates=info['candidates'], scoring_version='json_prefix_tree_v3', value_scores=scored, metric='expected_allocation' if kind == 'tg' else 'target_choice_log_odds')
