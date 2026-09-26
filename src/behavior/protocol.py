"""JSON validation and independent post-test branching."""
import copy
import hashlib
import json
import re

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse(raw, kind, reason=True, ids=None):
    match = re.fullmatch('```(?:json)?[ \\t]*\\r?\\n(.*?)\\r?\\n```', raw.strip(), re.S | re.I)

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate key')
            value[key] = item
        return value
    value = json.loads(match.group(1) if match else raw, object_pairs_hook=unique)
    field = {'tag': 'choice', 'tg': 'allocation', 'conflict': 'chosen_id'}[kind]
    if not isinstance(value, dict) or set(value) != {field, 'reason'}:
        raise ValueError('Expected decision and reason fields')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Expected nonempty reason string')
    if field == 'allocation':
        if type(value[field]) is not int or not 0 <= value[field] <= 10:
            raise ValueError('Invalid allocation')
    elif field == 'choice':
        if type(value[field]) is not str or value[field] not in ('left', 'right'):
            raise ValueError('Invalid choice')
    elif type(value[field]) is not str or value[field] not in ids:
        raise ValueError('Invalid participant ID')
    return value


def normalize(decision):
    return json.dumps(decision, ensure_ascii=False, separators=(',', ':'))


def branch(parent, suffix):
    assert parent[-1]['role'] == 'user'
    request = copy.deepcopy(parent)
    request[-1]['content'] += '\n\n' + suffix
    return request
