"""Label-blind complete-word candidate search."""
import heapq
import re
import torch
from wordfreq import top_n_list
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from .engine import WORD_RE, word_paths

class Search:

    def __init__(self, engine):
        self.e = engine
        self.cfg = engine.config['search']
        self.english = {w.lower() for w in top_n_list('en', self.cfg['wordlist_size']) if WORD_RE.fullmatch(w)}
        self.trie = {}
        words = [w for w in top_n_list('en', self.cfg['trie_size']) if WORD_RE.fullmatch(w) and 2 <= len(w) <= 24]
        for p in word_paths(engine.tokenizer, sorted(set(words))):
            node = self.trie
            for token in p['ids']:
                node = node.setdefault(token, {})
            node[-1] = p['word'].lower()

    def text(self, ids):
        return self.e.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def complete(self, ids):
        eos = next((j for j, token in enumerate(ids) if token in self.e.eos), None)
        text = self.text(ids if eos is None else ids[:eos])
        if eos is not None:
            word = text.strip()
            complete = bool(WORD_RE.fullmatch(word))
        else:
            match = re.fullmatch("\\s*([A-Za-z]+(?:[-'][A-Za-z]+)*)(?:\\s+|[.!?,;:]+)[\\s.!?,;:]*", text)
            word = match.group(1) if match else ''
            complete = bool(match)
        return word.lower() if complete and word.lower() in self.english else None

    def run(self, base, seed):
        cache = {(): base['logp'].cpu()}

        def probs(prefix):
            if prefix not in cache:
                cache[prefix] = self.e.continuation(base, [list(prefix)])[0, -1].cpu()
            return cache[prefix]
        found = []

        def retain(ids, method, score=None):
            word = self.complete(ids)
            if word and len(self.e.tokenizer.encode(word, add_special_tokens=False)) <= self.cfg['max_word_tokens']:
                found.append(dict(word=word, method=method, ids=list(ids), text=self.text(ids), search_logp=score))
                return True
            return False
        depth = self.cfg['max_word_tokens'] + 1
        audits = []
        generator = torch.Generator(device='cpu').manual_seed(seed)
        gen = self.e.config['generation']
        warpers = [TemperatureLogitsWarper(gen['temperature']), TopKLogitsWarper(gen['top_k']), TopPLogitsWarper(gen['top_p'])]
        for _ in range(self.cfg['natural_samples']):
            prefix = ()
            for j in range(gen['max_new_tokens']):
                scores = probs(prefix).clone()[None]
                tokens = torch.tensor([prefix], dtype=torch.long)
                if prefix:
                    scores[0, sorted(set(prefix))] -= gen['presence_penalty']
                for warper in warpers:
                    scores = warper(tokens, scores)
                token = int(torch.multinomial(torch.softmax(scores[0], -1), 1, generator=generator))
                prefix += (token,)
                if token in self.e.eos:
                    break
            word = self.complete(prefix)
            audits.append(dict(text=self.text(prefix), ids=list(prefix), single_word=word is not None, word=word))
            retain(prefix, 'natural_sampling')
        for method in ('beam', 'topk_recursive'):
            frontier = [((), 0.0)]
            width = self.cfg['beam_width'] if method == 'beam' else self.cfg['recursive_frontier']
            for _ in range(depth):
                children = []
                for prefix, score in frontier:
                    values, ids = torch.topk(probs(prefix), self.cfg['branch_top_k'])
                    for value, token in zip(values.tolist(), ids.tolist()):
                        child, total = (prefix + (token,), score + value)
                        if retain(child, method, total):
                            continue
                        if token not in self.e.eos and (not self.text(child).strip() or WORD_RE.fullmatch(self.text(child).lstrip())):
                            children.append((child, total))
                frontier = sorted(children, key=lambda p: -p[1])[:width]
                if not frontier:
                    break
        heap = [(0.0, ())]
        for _ in range(self.cfg['best_first_expansions']):
            if not heap:
                break
            neg, prefix = heapq.heappop(heap)
            if prefix and retain(prefix, 'best_first', -neg):
                continue
            if len(prefix) >= depth:
                continue
            values, ids = torch.topk(probs(prefix), self.cfg['branch_top_k'])
            for value, token in zip(values.tolist(), ids.tolist()):
                child = prefix + (token,)
                text = self.text(child).lstrip()
                if self.complete(child) or (token not in self.e.eos and (not text or WORD_RE.fullmatch(text))):
                    heapq.heappush(heap, (neg - value, child))
        frontier = [((), 0.0, self.trie)]
        for _ in range(self.cfg['max_word_tokens'] + 1):
            children = []
            for prefix, score, node in frontier:
                lp = probs(prefix)
                if -1 in node:
                    eos = max(self.e.eos, key=lambda t: float(lp[t]))
                    retain(prefix + (eos,), 'trie', score + float(lp[eos]))
                allowed = [t for t in node if t != -1]
                for token in sorted(allowed, key=lambda t: -float(lp[t]))[:self.cfg['beam_width']]:
                    children.append((prefix + (token,), score + float(lp[token]), node[token]))
            frontier = sorted(children, key=lambda p: -p[1])[:self.cfg['beam_width']]
            if not frontier:
                break
        return dict(candidates=found, natural=audits, suffix_forwards=len(cache) - 1)
