"""Sequential interaction; post-tests branch independently."""
import copy
import hashlib
import os
import time
from datetime import datetime, timezone
from .protocol import branch, digest, normalize, parse
from .storage import read, rows, save, append
from ..repro import bind_run, digest as stable_digest, package_signature, require_runtime

def sampling_seed(task_id, attempt, base):
    return (base + int(hashlib.sha256(f'{task_id}:{attempt}'.encode()).hexdigest()[:8], 16)) % 2 ** 31


def decide(client, out, config, messages, task_id, trial):
    temporary, failures, attempt_files = (copy.deepcopy(messages), [], [])
    for attempt in range(config['max_parse_attempts']):
        path = out / 'attempts' / f'{task_id}_{attempt}.json'
        seed = sampling_seed(task_id, attempt, config['sampling_seed_base'])
        if path.exists():
            record = read(path)
            if record.get('run_fingerprint') != client.active_run_fingerprint:
                raise ValueError('Cached attempt belongs to another runtime.')
            assert record['messages'] == temporary and record['seed'] == (None if getattr(client, 'is_api', False) else seed)
        else:
            response = client.complete_request(temporary, task_id, attempt) if getattr(client, 'is_api', False) else client.complete(temporary, seed)
            record = dict(task_id=task_id, attempt=attempt, seed=None if getattr(client, 'is_api', False) else seed, messages=temporary, finished_utc=datetime.now(timezone.utc).isoformat(), response=response,
                          run_fingerprint=client.active_run_fingerprint)
            save(path, record)
        response = record['response']
        assert response['non_thinking_ok'], 'Thinking contamination'
        attempt_files.append(str(path.relative_to(out)))
        try:
            if response.get('finish_status', 'stop') != 'stop':
                raise ValueError('Response was truncated')
            decision = parse(response['content'], trial['phase'], True, [trial.get('participant_a_id'), trial.get('participant_b_id')])
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(str(exc))
            if response['content']:
                temporary = temporary + [{'role': 'assistant', 'content': response['content']}]
            temporary = temporary + [{'role': 'user', 'content': trial['repair']}]
            continue
        return dict(response, parsed=decision, assistant_history_text=normalize(decision), base_messages_hash=digest(messages), attempt_files=attempt_files, parse_retry_count=attempt, parse_failures=failures, finished_utc=record['finished_utc'], reason_first=next(iter(decision)) == 'reason')
    raise RuntimeError('JSON format retries exhausted: ' + task_id)


def run_one(client, out, config, protocol, run):
    fingerprint = bind_run(out, dict(kind='behavior', runtime=require_runtime(client),
        config={k:v for k,v in config.items() if k!='run_ids'},
        protocol_sha256=stable_digest(protocol), package_signature=package_signature()))
    client.active_run_fingerprint = fingerprint
    rid = run['run_id']
    raw = out / 'raw' / f'run_{rid:03d}.jsonl'
    entries = rows(raw)
    done = {r['task_id']: r for r in entries}
    assert len(done) == len(entries)
    start = time.time()
    expected_rows = 76 * len(config['run_ids'])

    def progress(status, last=None):
        save(out / 'progress' / f'run_{rid:03d}.json', dict(status=status, run_id=rid, rows_this_run=len(done), total_rows=sum((len(rows(p)) for p in (out / 'raw').glob('*.jsonl'))), expected_total=expected_rows, pid=os.getpid(), last=last, elapsed_run_seconds=time.time() - start, updated_utc=datetime.now(timezone.utc).isoformat()))

    def trial_step(t, messages, parent=None):
        task_id = f"run_{rid:03d}_{t['item_id']}"
        if task_id in done:
            row = done[task_id]
            if row.get('run_fingerprint') != fingerprint:
                raise ValueError('Existing behavior row belongs to another runtime.')
            assert row['base_messages_hash'] == digest(messages) and row['trial'] == t
            assert row['parent_history_hash'] == parent
        else:
            result = decide(client, out, config, messages, task_id, t)
            row = dict(result, task_id=task_id, run_id=rid, phase=t['phase'], trial=t,
                       parent_history_hash=parent, run_fingerprint=fingerprint)
            append(raw, row)
            done[task_id] = row
        progress('running', task_id)
        return row
    history = [{'role': 'system', 'content': protocol['system']}]
    previous = protocol['tag_intro']
    progress('running')
    try:
        for t in run['trials'][:64]:
            messages = history + [{'role': 'user', 'content': previous + '\n\n' + t['prompt']}]
            row = trial_step(t, messages)
            history = messages + [{'role': 'assistant', 'content': row['assistant_history_text']}]
            previous = t['feedback'][row['parsed']['choice']]
            if t['trial_id'] % 8 == 0:
                print(f"run {rid}: TAG {t['trial_id']}/64; input={row['input_tokens']}; output={row['output_tokens']}", flush=True)
        parent = history + [{'role': 'user', 'content': previous}]
        parent_hash = digest(parent)
        save(out / 'histories' / f'run_{rid:03d}.json', dict(messages=parent, history_hash=parent_hash))
        for t in run['trials'][64:]:
            trial_step(t, branch(parent, t['prompt']), parent_hash)
            assert digest(parent) == parent_hash
        assert len(done) == 76
        progress('run_complete')
        print(f'run {rid}: COMPLETE 76/76', flush=True)
    except Exception as exc:
        progress('failed:' + type(exc).__name__)
        raise
