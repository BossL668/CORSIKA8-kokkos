"""Read-only source/binary snapshot; writes only a fresh validation directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = Path(__file__).resolve().parents[2]
a.output.mkdir(parents=True, exist_ok=False)
def run(*argv):
    r = subprocess.run(argv, cwd=source, capture_output=True, text=True)
    return dict(code=r.returncode, stdout=r.stdout, stderr=r.stderr)

snapshot = dict(head=run('git', 'rev-parse', 'HEAD'), status=run('git', 'status', '--short'),
                production=run('systemctl', '--user', 'status',
                               'c8-beta5-uhe100-cuda-persistent.service', '--no-pager'))
(a.output/'source_before.patch').write_text(run('git', 'diff', '--binary')['stdout'])
snapshot['files'] = {}
for root in [source/'applications', source/'src', source/'corsika', source/'tests',
             source.parent/'install', source.parent/'build/cuda-openmp/applications']:
    for f in root.rglob('*'):
        if f.is_file() and not f.is_symlink() and (f.suffix in ('.cpp','.hpp','.inl','.cu','.txt','.py')
                                                  or f.name == 'c8_air_shower'):
            snapshot['files'][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
for backend in ('cuda', 'openmp', 'cuda-openmp'):
    exe = source.parent/f'install/{backend}/bin/c8_air_shower'
    if exe.is_file():
        snapshot[f'help_{backend}'] = run(str(exe), '--help')
(a.output/'baseline.json').write_text(json.dumps(snapshot, indent=2))
print(a.output)
