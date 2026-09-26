"""Disjoint complete-answer events with shared prefix conditionals."""
from collections import defaultdict

import numpy as np
import torch
from scipy.special import logsumexp


def build_paths(tokenizer, words, suffixes, max_word_tokens=4):
    paths, prefixes = {}, {}
    for word in sorted(words):
        for form in (word, word.capitalize(), ' ' + word, ' ' + word.capitalize()):
            ids = tuple(tokenizer.encode(form, add_special_tokens=False))
            if not 0 < len(ids) <= max_word_tokens:
                continue
            owner = prefixes.setdefault(ids, word)
            assert owner == word, (word, owner, ids)
            for suffix in suffixes:
                body = form + suffix
                body_ids = tuple(tokenizer.encode(body, add_special_tokens=False))
                # This guarantees that conditional termination has a coherent denominator.
                assert body_ids[:len(ids)] == ids, ('Merged word boundary', body, ids, body_ids)
                assert tokenizer.decode(body_ids, clean_up_tokenization_spaces=False) == body
                row = dict(word=word, form=form, suffix=suffix, ids=list(body_ids), word_ids=list(ids))
                if body_ids in paths:
                    assert paths[body_ids]['word'] == word
                paths[body_ids] = row
    return list(paths.values()), [dict(word=w, ids=list(ids)) for ids, w in prefixes.items()]


def compile_trie(paths):
    children = {(): set()}
    terminal_owner = {}
    for row in paths:
        ids = tuple(row['ids'])
        if ids in terminal_owner:
            assert terminal_owner[ids] == row['word']
        terminal_owner[ids] = row['word']
        for i, token in enumerate(ids):
            children.setdefault(ids[:i], set()).add(token)
            children.setdefault(ids[:i+1], set())
    return children, terminal_owner


def evaluate_nodes(engine, base, paths, stop_ids):
    children, terminal_owner = compile_trie(paths)
    stop_ids = sorted(set(stop_ids))
    assert stop_ids and not any(set(row['ids']) & set(stop_ids) for row in paths)
    nodes = {}

    def save(prefix, logp):
        values = dict(edges={token: float(logp[token]) for token in children[prefix]})
        if prefix in terminal_owner:
            values['stop_logp'] = float(torch.logsumexp(logp[stop_ids], 0))
        nodes[prefix] = values

    save((), base['logp'])
    batch = engine.config['scoring']['batch_size']
    for depth in range(1, max(map(len, children)) + 1):
        layer = sorted(prefix for prefix in children if len(prefix) == depth)
        for start in range(0, len(layer), batch):
            chunk = layer[start:start+batch]
            lp = engine.continuation(base, chunk, last_only=True)
            for j, prefix in enumerate(chunk):
                save(prefix, lp[j, -1])
            del lp
    return nodes


def path_logp(nodes, ids):
    ids = tuple(ids)
    return sum(nodes[ids[:i]]['edges'][token] for i, token in enumerate(ids))


def aggregate_paths(nodes, paths, word_prefixes):
    events, word_values, seen = defaultdict(list), defaultdict(list), set()
    for row in paths:
        ids = tuple(row['ids'])
        if ids in seen:
            continue
        seen.add(ids)
        events[row['word']].append(dict(form=row['form'], suffix=row['suffix'], ids=list(ids),
                                       logp=path_logp(nodes, ids) + nodes[ids]['stop_logp']))
    for row in word_prefixes:
        word_values[row['word']].append(path_logp(nodes, row['ids']))
    result = []
    for word in sorted(events):
        complete = float(logsumexp([row['logp'] for row in events[word]]))
        prefix = float(logsumexp(word_values[word]))
        probability = float(np.exp(complete-prefix))
        assert 0 <= probability <= 1.00001, (word, probability)
        result.append(dict(word=word, logp=complete, prefix_logp=prefix,
                           termination_probability=probability, events=events[word]))
    mass = float(np.exp(logsumexp([row['logp'] for row in result])))
    assert 0 < mass <= 1.00001, ('Complete events not disjoint', mass)
    return result


def score_complete(engine, base, manifest):
    nodes = evaluate_nodes(engine, base, manifest['paths'], manifest['stop_ids'])
    return aggregate_paths(nodes, manifest['paths'], manifest['word_prefixes'])


def audit_complete_prefixes(engine, base, path, stop_ids, tolerance):
    """Compare matched one-position output heads, including rare long endings."""
    ids = path['ids']
    cached_values, full_values, components = [], [], []
    for depth in range(len(ids) + 1):
        cached = base['logp'] if depth == 0 else engine.continuation(base, [ids[:depth]], last_only=True)[0, -1]
        joined = torch.cat([base['input_ids'], torch.tensor([ids[:depth]], dtype=torch.long, device=engine.device)], 1)
        with torch.inference_mode():
            output = engine.model(input_ids=joined, use_cache=False, logits_to_keep=1)
        full = torch.log_softmax(output.logits[0, -1].float(), -1)
        indices = [ids[depth]] if depth < len(ids) else stop_ids
        a, b = (float(torch.logsumexp(lp[indices], 0)) for lp in (cached, full))
        cached_values.append(a); full_values.append(b)
        components.append(dict(depth=depth, cached_logp=a, full_logp=b, absolute_error=abs(a-b)))
    a, b = sum(cached_values), sum(full_values)
    error = abs(a-b)
    assert error < tolerance, (path, error, components)
    return dict(path=path, cached_logp=a, full_logp=b, absolute_logp_error=error,
                absolute_probability_error=abs(float(np.exp(a))-float(np.exp(b))), components=components,
                scope='Single-prefix cached/full numerical check; bf16 paths need not be bitwise identical.')
