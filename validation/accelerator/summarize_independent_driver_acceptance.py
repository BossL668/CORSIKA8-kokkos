#!/usr/bin/env python3
"""Verify the bounded independent-driver tests without inferring speed-up."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--regression', default='air-regression-v2')
    p.add_argument('--backend', default='backend20-warm')
    p.add_argument('--overlap', default='overlap-figure')
    p.add_argument('--report', default='BOUNDED_ACCEPTANCE.json')
    p.add_argument('--lifecycle', default='independent-N32')
    p.add_argument('--require-subshowers', action='store_true',
                   help='Require the independent queue protocol, not the older paired driver')
    a=p.parse_args();r=a.root.resolve()
    load=lambda p:json.loads(p.read_text())
    comparisons={mode:load(r/a.regression/(mode+'-comparison.json'))
                 for mode in ('proposal','cuda','openmp')}
    assert all(x['pass'] for x in comparisons.values())
    status=load(r/a.regression/'status.json')
    assert status['cuda-trace-exact'] and status['openmp-trace-exact']
    backend=load(r/a.backend/'summary.json')
    assert backend['pass']
    timeline=load(r/a.overlap/'timeline.json')
    assert timeline['overlap_ns']>0 and timeline['dropped_records']==0
    event=r/a.regression/a.lifecycle
    gpu=yaml.safe_load((event/'gpu_em/summary.yaml').read_text())
    timing=yaml.safe_load((event/'simulation_timing/summary.yaml').read_text())
    assert len(gpu)==len(timing)==32
    assert all(x['complete'] for x in gpu.values())
    assert all(x['closed'] for x in timing.values())
    stats=[x['statistics'] for x in gpu.values()]
    assert all(x['queue_overflows']==x['profile']['fixed_point_overflows']==
               x['radio']['fixed_point_overflows']==0 for x in stats)
    assert all(x['accelerator']['cooperative']['independent_drivers'] for x in stats)
    cooperative=[x['accelerator']['cooperative'] for x in stats]
    if a.require_subshowers:
        assert all(x['independent_subshowers'] for x in cooperative)
        assert all(x['subshower_cuda_submissions']==x['subshower_cuda_commits']
                   for x in cooperative), 'uncommitted GPU epoch at event close'
        assert all(x['cuda_input_particles']>0 and x['openmp_input_particles']>0
                   for x in cooperative), 'fixture did not exercise both endpoints'
    for f in event.rglob('*.npz'):
        with np.load(f) as data:
            for k in data.files:
                if data[k].dtype.kind in 'fc': assert np.isfinite(data[k]).all(), str(f)
    storage=[x['workspace_bytes']+x['accelerator']['cooperative']['openmp_workspace_bytes']
             for x in stats]
    assert len(set(storage))==1, 'resident workspace grew between events'
    guard=load(r/a.regression/(a.lifecycle+'-guard/summary.json'))
    assert guard['pass'] and guard['minimum_available_bytes']>=4*1024**3
    report=dict(pass_bounded_correctness=True,
        single_endpoint_arrays_equal={k:len(v['arrays']) for k,v in comparisons.items()},
        single_endpoint_traces_exact=True,closed_events=32,
        independent_subshowers_required=a.require_subshowers,
        maximum_host_epochs_per_cuda_job=max(
            x.get('maximum_host_epochs_per_cuda_job',0) for x in cooperative),
        host_queue_peak_bytes=max(
            x.get('subshower_host_queue_peak_bytes',0) for x in cooperative),
        measured_kernel_overlap_ms=timeline['overlap_ns']/1e6,
        workspace_bytes=storage[0],peak_N32_RSS_MiB=guard['peak_tree_rss_bytes']/2**20,
        large_sample_acceptance=False,performance_acceptance=False,
        scope='Tested local CUDA+OpenMP 20 threads, Kokkos 4.7.03; not a general thread-safety or speed guarantee.')
    (r/a.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
