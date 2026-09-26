import json
import os
import threading
from pathlib import Path
from ..repro import atomic_json

LOCK = threading.RLock()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with LOCK:
        atomic_json(path, value)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL row: {path}:{index}")
        result.append(json.loads(line))
    return result
