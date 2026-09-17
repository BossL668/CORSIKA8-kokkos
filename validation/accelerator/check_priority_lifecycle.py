#!/usr/bin/env python3
"""Post-run N=32 cache/workspace audit; fixed diagnostic bounds, not a leak proof."""
import argparse
import json
from pathlib import Path
import yaml


def correctness_source(run, identity=None):
    """Revalidate a pinned reuse receipt; legacy runs still read their own files."""
    run = Path(run)
    receipt_path = run/'CORRECTNESS_REUSE.json'
    if not receipt_path.exists():
        if identity is not None and identity.get('correctness_reused_from'):
            raise ValueError('correctness reuse provenance exists without its receipt')
        return run, None
    if identity is None:
        identity = json.loads((run/'PROVENANCE.json').read_text())
    receipt = json.loads(receipt_path.read_text())
    reference = Path(receipt['reference_run']).resolve(strict=True)
    if identity.get('correctness_reused_from') != str(reference):
        raise ValueError('correctness reference differs from current provenance')
    from run_priority_endpoints_acceptance import validate_correctness_reuse
    verified = validate_correctness_reuse(reference, identity)
    if verified != receipt:
        raise ValueError('correctness reuse receipt no longer matches validated source hashes')
    return reference, verified


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);a=p.parse_args();result={}
    source, reuse = correctness_source(a.run)
    for mode in ('cuda-openmp','openmp-cuda'):
        path=source/('N32-'+mode)
        data=yaml.safe_load((path/'gpu_em/summary.yaml').read_text())
        assert set(data)=={f'shower_{i}' for i in range(32)}
        rows=[]
        for i in range(32):
            s=data[f'shower_{i}']['statistics'];c=s['accelerator']['cooperative']
            assert data[f'shower_{i}']['complete']
            assert s['backend_lifecycle']['reused']==(i>0)
            assert s['backend_lifecycle']['shower_ordinal']==i+1
            assert c['subshower_cuda_submissions']==c['subshower_cuda_commits']
            assert s['queue_overflows']==s['profile']['fixed_point_overflows']==s['radio']['fixed_point_overflows']==0
            assert s['proposal_native']['inverse_failures']==0
            rows.append(dict(event=i,gpu_workspace_bytes=s['workspace_bytes'],
                host_workspace_bytes=c['openmp_workspace_bytes'],
                table_hash=s['proposal_native']['table_sha256'],
                auxiliary_hash=s['proposal_native']['aux_sha256'],
                profile_shard_bytes=c.get('host_profile_shard_bytes',0)))
        for key in ('gpu_workspace_bytes','host_workspace_bytes','table_hash','auxiliary_hash','profile_shard_bytes'):
            assert len({r[key] for r in rows})==1, mode+' changed '+key
        guard=json.loads((source/('N32-'+mode+'-guard/summary.json')).read_text())
        assert guard['pass'] and guard['minimum_available_bytes']>=4*2**30
        result[mode]=dict(pass_=True,events=rows,guard=guard,
            scope='bounded fixed-workspace N=32; not proof for unbounded production or all energies')
        if reuse is not None:
            result[mode].update(source_run=str(source), correctness_reused=True)
    (a.run/'N32_LIFECYCLE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: both N=32 modes reused tables/backend, constant workspaces, finite completed output and memory floor')


if __name__=='__main__':main()
