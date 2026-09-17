#!/usr/bin/env python3
"""Check completed same-build single/dual pilots; never control simulations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from accept_adaptive_service_uhe import load, reserve, scan_parquet
from summarize_adaptive_service_pilot import paired_timing_summary
from accept_adaptive_same_binary_pair import check_guard
from run_adaptive_cooperative_acceptance import check_coalesced_fallbacks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--require-exclusive-gpu', action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    state=json.loads((root/'STATUS.json').read_text())
    assert state['complete'] and all(r['complete'] for r in state['records'])
    assert json.loads((root/'CORRECTNESS_GATES.json').read_text())['passed']
    provenance=json.loads((root/'PROVENANCE.json').read_text())
    binary=root/'binaries/c8_air_shower'
    sha=hashlib.sha256()
    with binary.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):sha.update(block)
    assert sha.hexdigest()==provenance['binary_sha256']
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    pairs={};checks={}
    for r in state['records']:
        reserve()
        label=f"{r['family']}-{r['seed']}-{r['mode']}"
        path=root/label
        guard=json.loads((root/(label+'-guard/summary.json')).read_text())
        exclusive_checked=check_guard(guard,args.require_exclusive_gpu)
        assert guard['command'][0]==str(binary)
        g=load(path/'gpu_em/summary.yaml')['shower_0'];s=g['statistics']
        event=load(path/'summary.yaml')
        timing=load(path/'simulation_timing/summary.yaml')['shower_0']
        assert event['showers']==1 and event['seed']==r['seed'] and event['end time']
        assert g['complete'] and timing['closed']
        assert s['gpu_particles']==s['profile']['steps']==r['transport_records']
        assert s['queue_overflows']==s['profile']['fixed_point_overflows']==s['radio']['fixed_point_overflows']==0
        assert s['profile']['invalid_records']==s['proposal_native']['inverse_failures']==0
        assert s['cross_species']['final_pending_photons']==s['cross_species']['final_pending_leptons']==0
        assert s['process_registry']['accepted']
        a=s['accelerator'];c=a.get('cooperative',{})
        if r['mode']=='adaptive':
            assert a['backend']=='cuda-openmp' and a['host_threads']==20
            assert c['scheduling_policy']==provenance['policy']
            assert c['subshower_cuda_submissions']==c['subshower_cuda_commits']
            assert sum(c['adaptive'][e][k]['transport_records'] for e in ('cuda','openmp')
                       for k in ('photon','lepton'))==s['gpu_particles']
            if 'specified_fallback_batch_limit' in c:
                check_coalesced_fallbacks(s)
        elif r['mode']=='cuda':
            assert a['backend']=='cuda' and a['host_threads']==1 and not a['openmp']
        else:
            raise ValueError('This acceptance expects CUDA/adaptive pairs')
        cmd=guard['command'];assert len(cmd)%2==1
        flags=dict(zip(cmd[1::2],cmd[2::2]))
        for key in ('-f','--kokkos-execution','--kokkos-num-threads','--kokkos-cooperative-policy'):
            flags.pop(key,None)
        cfg=load(path/'gpu_em/config.yaml')
        identity=dict(flags=flags,primary=load(path/'primary/summary.yaml'),
            environment=cfg['environment'],magnetic_constant=cfg['magnetic_rigidity_GeV_per_T_m'],
            table={k:s['proposal_native'][k] for k in ('proposal_version','cubic_interpolation_version','table_sha256','aux_sha256')},
            thinning={k:s['thinning'][k] for k in ('em_fraction','maximum_weight','automatic_maximum_weight')},
            radio={alg:load(path/alg/'config.yaml') for alg in ('CoREAS','ZHS')})
        pairs.setdefault((r['family'],r['seed']),{})[r['mode']]=identity
        scans={str(p.relative_to(path)):scan_parquet(p) for p in sorted(path.rglob('*.parquet'))}
        assert len(scans)==7
        profile=pq.read_table(path/'profile/profile.parquet',use_threads=False).to_pandas()
        assert np.all(np.diff(profile['X'])>0)
        assert all(np.all(profile[k]>=0) for k in profile if k not in ('X','shower'))
        for p in (path/'interaction_hist').glob('*.npz'):
            with np.load(p,allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        checks[label]=dict(complete=True,files=scans,actual_backend=a['backend'],
            actual_threads=a['host_threads'],elapsed_s=guard['elapsed_s'],
            exclusive_gpu_checked=exclusive_checked,
            gpu_observation_failures=guard.get('gpu_observation_failures',0),
            peak_rss_GiB=guard['peak_tree_rss_bytes']/2**30,
            minimum_available_GiB=guard['minimum_available_bytes']/2**30,
            energy_ledger_complete=s['energy_ledger']['complete_coverage'],
            energy_ledger_accepted=s['energy_ledger']['accepted'])
    for pair in pairs.values():assert set(pair)=={'cuda','adaptive'} and pair['cuda']==pair['adaptive']
    summary=paired_timing_summary(state['records'])
    out=root/'performance_5seeds'
    out.mkdir(exist_ok=True)
    result=dict(operational_checks_passed=True,physical_settings_matched=True,
        binary_sha256=sha.hexdigest(),events=checks,paired_timing=summary,
        exclusive_gpu_checked=all(x['exclusive_gpu_checked'] for x in checks.values()),
        full_physics_acceptance=False,full_energy_closure_certified=False,
        performance_certified=False)
    (out/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':11,'savefig.dpi':180})
    for family in sorted({r['family'] for r in state['records']}):
        rows=[r for r in state['records'] if r['family']==family]
        seeds=sorted({r['seed'] for r in rows})
        lookup={(r['seed'],r['mode']):r for r in rows}
        fig,ax=plt.subplots(figsize=(10,4.8))
        for dx,mode,color in ((-.19,'cuda','#0072B2'),(.19,'adaptive','#D55E00')):
            values=[lookup[seed,mode]['process_s'] for seed in seeds]
            bars=ax.bar(np.arange(len(seeds))+dx,values,width=.36,color=color,label=mode)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set_xticks(np.arange(len(seeds)),[str(s) for s in seeds])
        ax.set_xlabel('Initial seed (dynamic scheduling allows different shower trees)')
        ax.set_ylabel('End-to-end time [s]');ax.grid(axis='y',alpha=.2)
        ax.margins(y=.18);ax.legend()
        ax.set_title(provenance['policy']+' / '+family+'\nSame executable, full radio, RTX 4060; adaptive uses 20 CPU threads')
        fig.tight_layout();fig.savefig(out/(family+'_time_comparison.png'));plt.close(fig)
    print(json.dumps(dict(output=str(out),checks_passed=True,paired_timing=summary),indent=2))


if __name__=='__main__':main()
