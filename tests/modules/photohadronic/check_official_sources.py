#!/usr/bin/env python3
"""Check the restored physics files against the pinned upstream source hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
reference = Path(__file__).resolve().parent / 'reference'
manifest = json.loads((reference / 'official.json').read_text())
for name, digest in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    frozen = reference / name
    if frozen.exists():
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == digest, str(frozen)
print('Exact official source match:', manifest['commit'], len(manifest['sha256']))
