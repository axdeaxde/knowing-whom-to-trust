"""Portable provenance, atomic completion and strict input validation."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from .paths import ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_completed_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        value=json.loads(path.read_text())
        if not isinstance(value,dict):
            raise ValueError('Expected a completed JSON object.')
        return value
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Incomplete result {path.name}. Move this file aside and rerun; '
                         'completed tasks can be retained.') from exc


@contextmanager
def directory_lock(path, timeout=30):
    path = Path(path)
    started = time.monotonic()
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() - started > timeout:
                raise RuntimeError(f'Active or stale lock: {path.name}. Stop competing workers '
                                   'before removing a stale lock.')
            time.sleep(.05)
    try:
        yield
    finally:
        path.rmdir()


def bind_run(out, specification):
    """All shards/stages sharing an output root must agree on scientific inputs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'run_manifest.json'
    identity = dict(schema_version=1, fingerprint=digest(specification), specification=specification)
    with directory_lock(out / '.manifest.lock'):
        prior = read_completed_json(manifest)
        if prior is not None:
            if prior != identity:
                raise ValueError('Run fingerprint mismatch. Use a new output directory; '
                                 'do not combine different models, inputs or settings.')
        else:
            existing = [p for p in out.iterdir() if p.name != '.manifest.lock']
            if existing:
                raise ValueError('Existing output has no run manifest. Use a new output directory '
                                 'or archive the old output; it cannot be safely resumed.')
            atomic_json(manifest, identity)
    return identity['fingerprint']


def package_signature():
    paths = []
    for folder in ['configs', 'src', 'data/behavior/qwen27b/raw', 'data/probe']:
        paths += [p for p in (ROOT / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)})


def versions():
    names = ['torch', 'transformers', 'accelerate', 'numpy', 'tokenizers',
             'scipy', 'scikit-learn', 'wordfreq', 'nltk', 'mistral-common',
             'sentencepiece', 'safetensors', 'protobuf', 'tiktoken']
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not-installed'
    return result


def checkpoint_fingerprint(path):
    """Hash actual checkpoint/tokenizer files, not a private directory name."""
    root = Path(path)
    patterns = ['*.json', '*.jinja', '*.model', '*.tiktoken', '*.safetensors', '*.bin',
                'vocab.*', 'merges.*', '*.py']
    files = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    if not files or not any(p.suffix in ['.safetensors', '.bin'] for p in files):
        raise ValueError('No local checkpoint weight files found.')
    return {p.name: file_hash(p) for p in files}


def require_runtime(owner):
    identity = getattr(owner, 'release_identity', None)
    if not isinstance(identity, dict) or not identity:
        raise ValueError('Missing runtime identity. Use the documented CLI; tests must '
                         'provide an explicit synthetic identity.')
    return identity


def validate_runtime(engine, expected):
    actual = engine.manifest()
    for name in ['layer_count', 'hidden_size', 'vocab_size', 'chat_template_hash',
                 'torch_version', 'transformers_version']:
        if actual.get(name) != expected[name]:
            raise ValueError(f'Frozen runtime mismatch: {name}. Use the recorded environment.')
    # Historical model_config_hash can contain the old local _name_or_path. The
    # new resume identity uses file content hashes and never stores that path.
    return {k: actual[k] for k in ['layer_count', 'hidden_size', 'vocab_size',
                                  'chat_template_hash', 'model_config_hash',
                                  'torch_version', 'transformers_version']}


def validate_checkpoint(files, expected):
    for name, value in expected.items():
        if files.get(name) != value:
            raise ValueError(f'Frozen checkpoint/tokenizer file mismatch: {name}')


def same_measurement(a, b, atol=2e-5):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(
            same_measurement(a[k], b[k], atol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a)==len(b) and all(
            same_measurement(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return isinstance(b, (int, float)) and bool(np.isclose(a,b,rtol=0,atol=atol))
    return a == b


def activation_complete(path, metadata, fingerprint):
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if not path.exists() or not sidecar.exists():
        return False
    try:
        saved = json.loads(sidecar.read_text())
        if not isinstance(saved,dict):
            return False
        if saved.get('run_fingerprint') != fingerprint:
            raise ValueError('Activation fingerprint mismatch.')
        if saved.get('queries') != metadata:
            raise ValueError('Activation query metadata mismatch.')
        if saved.get('history_advisors') != sorted({r['advisor_id'] for r in metadata}):
            return False
        if saved.get('npz_sha256') != file_hash(path):
            return False
        with np.load(path, allow_pickle=False) as z:
            return (set(z.files) == {'history', 'query'} and
                    z['history'].shape == (4, 64, 5120) and
                    z['query'].shape == (len(metadata), 2, 64, 5120) and
                    all(np.isfinite(z[k]).all() for k in z.files))
    except (OSError, EOFError, json.JSONDecodeError, zipfile.BadZipFile):
        return False
