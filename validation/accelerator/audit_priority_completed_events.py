#!/usr/bin/env python3
"""Re-read completed pilot outputs; never promote incomplete energy coverage."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from run_priority_endpoints_acceptance import check_output, save


def physics_identity(path, command):
    # Remove only host/device allocation and filesystem/seed differences;
    # retain all cuts, thinning, models, memory and wavefront settings.
    skip={'-f','-s','--antenna-file','--kokkos-execution','--kokkos-num-threads',
          '--kokkos-device'}
    args=[];i=1
    while i<len(command):
        if command[i] in skip:i+=2
        else:args.append(command[i]);i+=1
    config=yaml.safe_load((path/'gpu_em/config.yaml').read_text())
    environment=dict(config['environment']);environment.pop('antenna_file')
    radio={name:yaml.safe_load((path/name/'config.yaml').read_text())
           for name in ('CoREAS','ZHS')}
    primary=yaml.safe_load((path/'primary/summary.yaml').read_text())['shower_0']
    data=dict(args=args,environment=environment,primary=primary,radio=radio,
        magnetic_rigidity=config['magnetic_rigidity_GeV_per_T_m'],
        scalar_transport_constants_version=config['scalar_transport_constants_version'])
    data['sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return data


def check_termination(stats):
    assert stats['cross_species']['final_pending_photons']==0
    assert stats['cross_species']['final_pending_leptons']==0
    assert stats['deferred_cpu_fallbacks_queued']==stats['deferred_cpu_fallbacks_flushed']
    assert stats['proposal_native']['inverse_failures']==0
    assert stats['queue_overflows']==0
    assert stats['profile']['fixed_point_overflows']==stats['radio']['fixed_point_overflows']==0
    return stats['energy_ledger']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args()
    state=json.loads((a.run/'STATUS.json').read_text())
    records=[]
    for record in state['records']:
        path=a.run/record['label']
        guard=json.loads((a.run/(record['label']+'-guard/summary.json')).read_text())
        assert guard['pass'], record['label']+' did not finish successfully'
        # Re-read arrays instead of trusting the earlier CHECK.json flag.
        output=check_output(path,1,record['mode'])
        shower=yaml.safe_load((path/'gpu_em/summary.yaml').read_text())['shower_0']
        ledger=check_termination(shower['statistics'])
        records.append(dict(label=record['label'],mode=record['mode'],
            family=record['family'],seed=record['seed'],
            physics=physics_identity(path,guard['command']),
            physical_arrays=output['physical_arrays'],termination_verified=True,
            energy_ledger=ledger,
            full_energy_closure_certified=bool(ledger['complete_coverage'] and ledger['accepted'])))
    for family in {r['family'] for r in records}:
        assert len({r['physics']['sha256'] for r in records if r['family']==family})==1, \
            'primary/geometry/radio/CLI physical conditions differ within '+family
    result=dict(all_planned_runs_complete=state['complete'],records=records,
        completed_outputs_verified=True,
        scope='output integrity and empty queues, not ensemble equivalence or full hadronic energy closure')
    save(a.run/'COMPLETED_OUTPUT_AUDIT.json',result)
    print('PASS: re-read {} completed events, finite outputs, empty queues and flushed fallbacks'.format(len(records)))


if __name__=='__main__':
    main()
