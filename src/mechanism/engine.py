"""Sparse residual capture and cache-safe continuation scoring."""
import copy
import re
from contextlib import contextmanager
import numpy as np
import torch
from scipy.special import logsumexp
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from .common import ModelAdapter, primary_tensor, replace_primary
WORD_RE=re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")

def branch(messages, text):
    messages = copy.deepcopy(messages)
    assert messages[-1]['role'] == 'user'
    messages[-1]['content'] += '\n\n' + text
    return messages


def word_paths(tokenizer, words, max_tokens=4):
    paths = []
    for word in words:
        seen = set()
        for form in (word, word.capitalize(), ' ' + word, ' ' + word.capitalize()):
            ids = tuple(tokenizer.encode(form, add_special_tokens=False))
            if ids not in seen and 0 < len(ids) <= max_tokens:
                paths.append(dict(word=word, form=form, ids=list(ids)))
                seen.add(ids)
    return paths


def set_score(scores, lexicon):
    values = {r['word']: r['logp'] for r in scores}
    high = logsumexp([values[w] for w in lexicon['V_H']]) - np.log(len(lexicon['V_H']))
    low = logsumexp([values[w] for w in lexicon['V_L']]) - np.log(len(lexicon['V_L']))
    return float(high - low)


def distribution_kl(base_logp, changed_logp):
    a, b = (np.asarray(base_logp, dtype=np.float64), np.asarray(changed_logp, dtype=np.float64))
    a -= logsumexp(a)
    b -= logsumexp(b)
    return max(0.0, float(np.sum(np.exp(a) * (a - b))))


class Engine(ModelAdapter):

    def __init__(self, cfg):
        super().__init__(cfg)
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.add(self.tokenizer.eos_token_id)
        vocabulary = [self.tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(self.tokenizer))]
        self.boundaries = [i for i, text in enumerate(vocabulary) if i in self.eos or re.match('^[\\s.!?,;:]', text)]
        self.boundary_set = set(self.boundaries)
        self.number_boundaries = [i for i, text in enumerate(vocabulary) if re.match('^\\s*[,}]', text)]
        self.string_boundaries = [i for i, text in enumerate(vocabulary) if text.startswith('"')]

    @contextmanager
    def hook(self, edit):
        handles = []
        audit = {}
        try:
            if edit is not None:
                positions = [int(edit['position'])] if 'position' in edit else [int(x) for x in edit['positions']]
                vectors = [edit['vector']] if 'vector' in edit else edit['vectors']
                assert len(positions) == len(vectors)

                def modify(module, args, output):
                    h = primary_tensor(output)
                    changed = h.clone()
                    if any((position >= h.shape[1] for position in positions)):
                        return output
                    total_before = 0.0
                    total_delta = 0.0
                    projections = []
                    targets = []
                    for position, raw_vector in zip(positions, vectors):
                        before = h[0, position].detach().clone()
                        vector = torch.as_tensor(raw_vector, dtype=h.dtype, device=h.device)
                        target = vector if edit['kind'] == 'patch' else before + vector
                        changed[0, position] = target
                        delta = target.float() - before.float()
                        total_before += float(before.float().square().sum())
                        total_delta += float(delta.square().sum())
                        targets.append(target)
                        if edit.get('units') is not None:
                            unit = torch.as_tensor(edit['units'][len(projections)], device=h.device, dtype=torch.float32)
                            projections.append(float(delta @ unit))
                    audit.update(relative_l2=float(np.sqrt(total_delta) / max(np.sqrt(total_before), 1e-12)), achieved_l2=float(np.sqrt(total_delta)), calls=len(positions))
                    if edit.get('unit') is not None:
                        unit = torch.as_tensor(edit['unit'], device=h.device, dtype=torch.float32)
                        delta = targets[0].float() - h[0, positions[0]].detach().float()
                        audit['achieved_projection'] = float(delta @ unit)
                        audit['target_projection'] = float(np.asarray(edit['vector']) @ np.asarray(edit['unit']))
                    if projections:
                        audit['achieved_projections'] = projections
                        audit['target_projections'] = [float(np.asarray(vector) @ np.asarray(unit)) for vector, unit in zip(vectors, edit['units'])]
                    return replace_primary(output, changed)
                handles.append(self.layers[edit['layer']].register_forward_hook(modify))
            yield audit
        finally:
            for handle in handles:
                handle.remove()

    def prefill(self, messages, edit=None, suffix='', capture=None):
        text = self.render(messages) + suffix
        inputs, _ = self.encode(text)
        captured, handles = ({}, [])
        try:
            if capture:
                for layer in capture['layers']:

                    def save(module, args, output, layer=layer):
                        captured[layer] = primary_tensor(output)[0, capture['positions']].detach().float().cpu().numpy()
                    handles.append(self.layers[layer].register_forward_hook(save))
            with self.hook(edit) as audit, torch.inference_mode():
                result = self.model(**inputs, use_cache=True, logits_to_keep=1)
            logp = torch.log_softmax(result.logits[0, -1].float(), dim=-1)
            return dict(cache=result.past_key_values, logp=logp, length=inputs['input_ids'].shape[1], input_ids=inputs['input_ids'], audit=dict(audit), captured=captured)
        finally:
            for handle in handles:
                handle.remove()

    def continuation(self, base, sequences, last_only=False):
        assert len({len(x) for x in sequences}) == 1
        ids = torch.tensor(sequences, dtype=torch.long, device=self.device)
        cache = copy.deepcopy(base['cache'])
        if len(sequences) > 1:
            indices = torch.zeros(len(sequences), device=self.device, dtype=torch.long)
            for layer in cache.layers:
                layer.reorder_cache(indices)
        mask = torch.ones((len(sequences), base['length'] + ids.shape[1]), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1 if last_only else ids.shape[1])
            lp = torch.log_softmax(out.logits.float(), -1)
        assert base['cache'].get_seq_length() == base['length']
        return lp

    def score(self, base, paths, boundary_ids=None):
        if boundary_ids is not None:
            return self.score_prefix_tree(base, paths, boundary_ids)
        rows = []
        boundaries = self.boundaries if boundary_ids is None else boundary_ids
        boundary_set = set(boundaries)
        batch_size = self.config['scoring']['batch_size']
        for length in sorted({len(p['ids']) for p in paths}):
            group = [p for p in paths if len(p['ids']) == length]
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                lp = self.continuation(base, [p['ids'] for p in chunk])
                for j, p in enumerate(chunk):
                    ids = p['ids']
                    value = float(base['logp'][ids[0]])
                    if length > 1:
                        value += float(lp[j, torch.arange(length - 1, device=lp.device), ids[1:]].sum())
                    end = float(torch.logsumexp(lp[j, -1, boundaries], 0))
                    rows.append(dict(p, sequence_logp=value, logp=value + end, boundary_probability=float(np.exp(end)), boundary_top1=int(lp[j, -1].argmax()) in boundary_set))
                del lp
        result = []
        for word in sorted({p['word'] for p in paths}):
            group = [r for r in rows if r['word'] == word]
            total = logsumexp([r['logp'] for r in group])
            prefix = logsumexp([r['sequence_logp'] for r in group])
            best = max(group, key=lambda r: r['sequence_logp'])
            result.append(dict(word=word, logp=float(total), boundary_probability=float(np.exp(total - prefix)), boundary_top1=best['boundary_top1'], paths=group))
        return result

    def score_prefix_tree(self, base, paths, boundary_ids):
        """Share each conditional distribution across overlapping JSON value paths."""
        children = {(): set()}
        terminals = set()
        for path in paths:
            ids = tuple(path['ids'])
            terminals.add(ids)
            for i, token in enumerate(ids):
                children.setdefault(ids[:i], set()).add(token)
                children.setdefault(ids[:i + 1], set())
        nodes = {}
        boundary_set = set(boundary_ids)

        def save(prefix, logp):
            nodes[prefix] = dict(edges={token: float(logp[token]) for token in children[prefix]})
            if prefix in terminals:
                nodes[prefix].update(end=float(torch.logsumexp(logp[boundary_ids], 0)), top1=int(logp.argmax()) in boundary_set)
        save((), base['logp'])
        batch_size = self.config['scoring']['batch_size']
        for length in range(1, max(map(len, children)) + 1):
            prefixes = sorted((p for p in children if len(p) == length))
            for start in range(0, len(prefixes), batch_size):
                chunk = prefixes[start:start + batch_size]
                logp = self.continuation(base, chunk)
                for i, prefix in enumerate(chunk):
                    save(prefix, logp[i, -1])
                del logp
        rows = []
        for path in paths:
            ids = tuple(path['ids'])
            sequence = sum((nodes[ids[:i]]['edges'][token] for i, token in enumerate(ids)))
            end = nodes[ids]['end']
            rows.append(dict(path, sequence_logp=sequence, logp=sequence + end, boundary_probability=float(np.exp(end)), boundary_top1=nodes[ids]['top1']))
        result = []
        for word in sorted({p['word'] for p in paths}):
            group = [row for row in rows if row['word'] == word]
            total = logsumexp([r['logp'] for r in group])
            prefix = logsumexp([r['sequence_logp'] for r in group])
            best = max(group, key=lambda r: r['sequence_logp'])
            result.append(dict(word=word, logp=float(total), boundary_probability=float(np.exp(total - prefix)), boundary_top1=best['boundary_top1'], paths=group))
        return result

    def sample(self, base, seed, max_tokens=8):
        cfg = self.config['generation']
        generator = torch.Generator(device='cpu').manual_seed(seed)
        warpers = [TemperatureLogitsWarper(cfg['temperature']), TopKLogitsWarper(cfg['top_k']), TopPLogitsWarper(cfg['top_p'])]
        prefix = []
        for _ in range(max_tokens):
            lp = base['logp'].cpu() if not prefix else self.continuation(base, [prefix])[0, -1].cpu()
            scores = lp.clone()[None]
            ids = torch.tensor([prefix], dtype=torch.long)
            if prefix:
                scores[0, sorted(set(prefix))] -= cfg['presence_penalty']
            for warper in warpers:
                scores = warper(ids, scores)
            token = int(torch.multinomial(torch.softmax(scores[0], -1), 1, generator=generator))
            prefix.append(token)
            if token in self.eos:
                break
        text = self.tokenizer.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return dict(text=text, token_ids=prefix, seed=seed, stopped=prefix[-1] in self.eos, generation_config=cfg)

    def audit_cache(self, base, path, boundary_ids=None):
        boundaries = self.boundaries if boundary_ids is None else boundary_ids
        ids = path['ids']
        cached = self.continuation(base, [ids])[0]
        joined = torch.cat([base['input_ids'], torch.tensor([ids], device=self.device)], dim=1)
        with torch.inference_mode():
            out = self.model(input_ids=joined, use_cache=False, logits_to_keep=len(ids) + 1)
        full = torch.log_softmax(out.logits[0].float(), -1)

        def value(first, suffix):
            val = float(first[ids[0]])
            val += sum((float(suffix[j - 1, ids[j]]) for j in range(1, len(ids))))
            return val + float(torch.logsumexp(suffix[-1, boundaries], 0))
        error = abs(value(base['logp'], cached) - value(full[0], full[1:]))
        assert error < 0.2, (path, error)
        return dict(path=path, cached_full_word_abs_logp_error=error, first_logp_max_abs_error=float((base['logp'] - full[0]).abs().max()))
