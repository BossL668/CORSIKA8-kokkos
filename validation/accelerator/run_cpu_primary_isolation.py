#!/usr/bin/env python3
"""Guarded fixed-input OpenMP runtime/workspace isolation, not a shower campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'cache', 'guard', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--count', type=int, default=32768)
    p.add_argument('--energy-mev', type=float, default=10.)
    p.add_argument('--affinity', help='Exact inherited logical CPU list, e.g. 382-511')
    p.add_argument('--rss-limit-gib', type=float, default=6.)
    p.add_argument('--libraries', default='')
    p.add_argument('--modes', nargs='+', default=['direct', 'prepared', 'borrowed', 'idle-cuda'],
                   choices=['direct', 'prepared', 'borrowed', 'idle-cuda', 'host-pump'])
    a = p.parse_args()
    if a.affinity:
        lo, hi = map(int, a.affinity.split('-'))
        if sorted(os.sched_getaffinity(0)) != list(range(lo, hi+1)):
            raise RuntimeError('launch this tool under the requested exact taskset')
    binary, guard = a.binary.resolve(), a.guard.resolve()
    before = {str(x): sha(x) for x in (binary, guard)}
    help_text = subprocess.check_output([sys.executable, str(guard), '--help'], text=True)
    if '--require-exclusive-gpu' not in help_text:
        raise RuntimeError('guard has no GPU exclusivity check')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OMP_THREAD_LIMIT=str(a.threads),
               OMP_PROC_BIND='spread', OMP_PLACES='threads', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if a.libraries:
        env['LD_LIBRARY_PATH'] = a.libraries
    if 'KOKKOS_TOOLS_LIBS' in env or 'KOKKOS_PROFILE_LIBRARY' in env:
        raise RuntimeError('uninstrumented comparison must not inherit a profiling plugin')
    state = dict(complete=False, scope='fixed EM fixture; not a hadronic shower; see per-result driver scope',
                 includes_cooperative_pump='host-pump' in a.modes,
                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                 hashes=before, affinity=sorted(os.sched_getaffinity(0)), records=[])
    save(out/'STATUS.json', state)
    reference = {}
    try:
        # Both orders and explicit per-process warm-up; never discard slow results.
        for radio in (0, 1):
            for order in range(2):
                for mode in (a.modes if order==0 else list(reversed(a.modes))):
                    label = 'radio%d-order%d-%s' % (radio, order, mode)
                    state['phase'] = label
                    save(out/'STATUS.json', state)
                    if any(sha(Path(x)) != h for x, h in before.items()):
                        raise RuntimeError('frozen binary/guard changed')
                    cmd = [str(binary), str(a.cache.resolve()), mode, str(a.threads),
                           str(a.count), str(a.energy_mev), str(radio), '3']
                    subprocess.run([sys.executable, str(guard), '--output', str(out/label),
                                    '--timeout', '300', '--rss-limit-gib', str(a.rss_limit_gib),
                                    '--sample-resources', '--sample-threads', '--require-exclusive-gpu',
                                    '--', *cmd], cwd=out, env=env, check=True)
                    g = json.loads((out/label/'summary.json').read_text())
                    lines = (out/label/'command.log').read_text().splitlines()
                    results = [json.loads(x[len('ISOLATION_RESULT '):]) for x in lines
                               if x.startswith('ISOLATION_RESULT ')]
                    if len(results)!=1 or not results[0]['complete']:
                        raise RuntimeError('missing completed fixture report')
                    result = results[0]
                    if result['gpu_transport_steps'] != 0 or len(result['runs']) != 3:
                        raise RuntimeError('unexpected work/completion')
                    fields = ('input_sha256', 'physics_arrays_terminal_sha256', 'call_sequence_sha256',
                              'steps', 'photon_calls', 'lepton_calls', 'photon_waves', 'lepton_waves',
                              'energy_relative_residual', 'terminal_count')
                    for run in result['runs']:
                        identity = {k: run[k] for k in fields}
                        if radio not in reference:
                            reference[radio] = identity
                        if identity != reference[radio]:
                            raise RuntimeError('same physical input/call/output check failed: '+label)
                    state['records'].append(dict(label=label, radio=radio, order=order, mode=mode,
                        exact_input_call_output_match=True, guard=g, result=result,
                        warm_driver_median_s=statistics.median(x['driver_seconds'] for x in result['runs'] if x['warm'])))
                    save(out/'STATUS.json', state)
        state.update(complete=True, phase='complete-fixed-input-isolation', reference=reference)
    except BaseException as e:
        state.update(complete=False, error=repr(e))
        raise
    finally:
        save(out/'STATUS.json', state)


if __name__ == '__main__':
    main()
