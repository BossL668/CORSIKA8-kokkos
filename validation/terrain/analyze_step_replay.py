#!/usr/bin/env python3
"""Separate local same-input errors from free-running shower drift; never relax gates."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

CASES = ['photon_up', 'photon_down', 'electron_rock', 'positron_rock']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(folder):
    counts = json.loads((folder/'steps.counts.json').read_text())
    summary = yaml.safe_load((folder/'shower/terrain_run.yaml').read_text())
    resources = json.loads((folder/'guard/resources.json').read_text())
    groups = {}
    with (folder/'steps.csv').open() as f:
        for row in csv.DictReader(f):
            key = row['mode']+'/'+row['stage']
            group = groups.setdefault(key, dict(count=0, above_gate=0, integer_differences=0,
                uniform_or_key_field_differences=0, photon_split_field_differences=0,
                other_rng_field_differences=0, max_abs=0., max_ulp=0, first=row, fields=Counter(), field_metrics={}))
            group['count'] += 1
            group['above_gate'] += row['within_gate'] == '0'
            group['fields'][row['field']] += 1
            is_rng = ('uniform' in row['field'] or 'draw_id' in row['field'] or 'random_process' in row['field'])
            group['uniform_or_key_field_differences'] += is_rng
            # For Compton/photoelectric, split_uniform is a derived loss
            # quantile, not a new Philox draw. Never label all such fields RNG.
            split = row['stage']=='photon_final' and row['field']=='split_uniform'
            group['photon_split_field_differences'] += split
            group['other_rng_field_differences'] += is_rng and not split
            if row['ulp'] == 'integer':
                group['integer_differences'] += 1
            else:
                a, b = float(row['host']), float(row['device'])
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError('nonfinite differing audit field: '+str(row))
                group['max_abs'] = max(group['max_abs'], abs(a-b))
                group['max_ulp'] = max(group['max_ulp'], int(row['ulp']))
                m = group['field_metrics'].setdefault(row['field'], dict(max_abs=0.,max_relative=0.,max_ulp=0))
                m['max_abs'] = max(m['max_abs'],abs(a-b))
                m['max_relative'] = max(m['max_relative'],abs(a-b)/max(abs(a),abs(b),1.e-300))
                m['max_ulp'] = max(m['max_ulp'],int(row['ulp']))
                if row['within_gate'] == '0':
                    group.setdefault('first_above_gate', row)
    if sum(g['count'] for g in groups.values()) != counts['bit_differences'] or sum(
            g['above_gate'] for g in groups.values()) != counts['strict_failures']:
        raise ValueError('audit CSV/count mismatch: '+str(folder))
    if counts['compared_step_pairs'] <= 0 or counts['fields'] <= 0:
        raise ValueError('empty replay is not acceptance')
    return dict(counts=counts, groups=groups, shower_complete=summary.get('complete', False),
                resources_passed=resources['returncode']==0 and resources['stop_reason'] is None,
                max_rss_kib=max(s.get('rss_kib', 0) for s in resources['samples']),
                max_gpu_increment_mib=max((s.get('gpu_delta_mib', 0) for s in resources['samples']), default=0),
                material_tables=summary.get('material_tables'),
                portable_host_oracle_not_scalar_proposal=True)


def endpoint_errors(left, right):
    # Rows are only comparable until discrete labels cease to agree. The
    # legacy trajectory CSV has no history IDs: do not call this a tree proof.
    errors = []
    with left.open() as lf, right.open() as rf:
        lrows, rrows = csv.DictReader(lf), csv.DictReader(rf)
        for a, b in zip(lrows, rrows):
            if any(a[k]!=b[k] for k in ['step','pdg','medium']):
                break
            errors.append(math.sqrt(sum((float(a[k])-float(b[k]))**2 for k in ['x1_m','y1_m','z1_m'])))
    return errors


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    args = p.parse_args()
    result = dict(scope='strict local replay and drift diagnosis; NOT ensemble or interface-radio acceptance',
                  tolerance=dict(relative=1e-10, absolute=1e-12), variants={}, baseline_track_bytes={})
    for variant in ['openmp','cuda','cuda_nocontract']:
        runs = result['variants'][variant] = {}
        for case in CASES:
            folder = args.root/variant/case
            if not (folder/'steps.counts.json').exists():
                runs[case] = dict(incomplete=True)
                continue
            runs[case] = audit(folder)
            if variant != 'cuda_nocontract':
                result['baseline_track_bytes'][variant+'/'+case] = (
                    digest(folder/'shower/terrain/tracks.csv') ==
                    digest(args.previous/(variant+'_igrf')/case/'terrain/tracks.csv'))
    result['all_runs_complete'] = all(
        not r.get('incomplete') and r['shower_complete'] and r['resources_passed']
        for cases in result['variants'].values() for r in cases.values())
    result['material_banks_equal'] = all(
        result['variants']['openmp'][case].get('material_tables') ==
        result['variants'][variant][case].get('material_tables')
        for case in CASES for variant in ['cuda','cuda_nocontract'])
    result['all_strict_local_fields_passed'] = result['all_runs_complete'] and all(
        r['counts']['strict_failures']==0 for cases in result['variants'].values() for r in cases.values())
    (args.root/'step_replay_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax, case in zip(axs.flat, CASES):
        omp = args.root/'openmp'/case/'shower/terrain/tracks.csv'
        for variant,label in [('cuda','CUDA default'),('cuda_nocontract','CUDA diagnostic: contraction off')]:
            cuda = args.root/variant/case/'shower/terrain/tracks.csv'
            if not omp.exists() or not cuda.exists():
                continue
            errors = endpoint_errors(omp,cuda)
            # Exact zeros are omitted on the log scale, never replaced by an artificial tolerance.
            x,y = zip(*[(i+1,v) for i,v in enumerate(errors) if v>0]) if any(errors) else ([],[])
            ax.plot(x,y,lw=.7,label=label)
        ax.set(title=case,xlabel='Ordered track row (not a history ID)',ylabel='Endpoint separation [m]',yscale='log')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.root/'trajectory_drift.png',dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='variants'},indent=2))


if __name__=='__main__':
    main()
