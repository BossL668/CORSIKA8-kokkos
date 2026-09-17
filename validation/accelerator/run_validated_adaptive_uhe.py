#!/usr/bin/env python3
"""One guarded UHE job using the exact frozen binary of a completed pilot.

No production install, physics changes, replacement output or automatic retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def uhe_command(original, binary, output, mode, seed):
    """Change execution only; preserve all physical CLI arguments verbatim."""
    command=list(original)
    command[0]=str(binary)
    dual=mode!='cuda'
    if '--kokkos-cooperative-policy' in command:
        i=command.index('--kokkos-cooperative-policy');del command[i:i+2]
    settings=[('-N',1),('-s',seed),('-f',output),
        ('--kokkos-execution','cuda-openmp' if dual else 'cuda'),
        ('--kokkos-num-threads',20 if dual else 1),('--gpu-memory-fraction',.7),
        ('--gpu-min-batch',4096),('--hadronic-workers',1)]
    if dual:settings.append(('--kokkos-cooperative-policy',mode))
    for flag,value in settings:
        if flag in command:command[command.index(flag)+1]=str(value)
        else:command += [flag,str(value)]
    for flag,expected in (('-E',1e8),('-z',47),('-a',180),('--emthin',1e-6)):
        assert float(command[command.index(flag)+1])==expected, flag+' physical setting mismatch'
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-unit', required=True)
    p.add_argument('--mode',choices=('cuda','adaptive','legacy'),default='adaptive')
    p.add_argument('--seed',type=int,default=2026110001)
    p.add_argument('--source-root',type=Path,
                   help='Frozen source supplying physics data and the guard; defaults to this source')
    p.add_argument('--guard-script',type=Path,
                   help='Explicit diagnostic guard; does not change frozen physics data')
    p.add_argument('--require-exclusive-gpu',action='store_true')
    p.add_argument('--require-paired-acceptance',action='store_true')
    a=p.parse_args()
    while subprocess.run(['systemctl','--user','is-active','--quiet',a.wait_unit]).returncode==0:
        time.sleep(2)
    code=subprocess.check_output(['systemctl','--user','show',a.wait_unit,
        '-p','ExecMainStatus','--value'],text=True).strip()
    assert code=='0', 'pilot process did not pass'
    state=json.loads((a.pilot/'STATUS.json').read_text())
    assert state['complete'] and all(r['complete'] for r in state['records']), 'pilot incomplete'
    assert json.loads((a.pilot/'CORRECTNESS_GATES.json').read_text())['passed']
    if a.require_paired_acceptance:
        accepted=json.loads((a.pilot/'performance_5seeds/ACCEPTANCE.json').read_text())
        assert accepted['operational_checks_passed'] and accepted['physical_settings_matched']
        pair=accepted['paired_timing']['Fe100TeV/cuda_vs_adaptive']
        assert pair['paired_seeds']>=5
    assert subprocess.run(['pgrep','-x','c8_air_shower'],stdout=subprocess.DEVNULL).returncode==1, 'other shower active'
    frozen=json.loads((a.pilot/'PROVENANCE.json').read_text())
    binary=a.pilot/'binaries/c8_air_shower'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==frozen['binary_sha256']
    command=json.loads(Path(frozen['proton_command']).read_text())['command']
    source=(a.source_root or Path(__file__).resolve().parents[2]).resolve()
    guard_script=(a.guard_script or source/'validation/accelerator/run_overlap_guarded.py').resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    label=f'proton100PeV-{a.seed}-{a.mode}'
    command=uhe_command(command,binary,a.output/label,a.mode,a.seed)
    provenance=dict(
        pilot=str(a.pilot),binary_sha256=frozen['binary_sha256'],policy=frozen['policy'],
        mode=a.mode,command=command,complete=False,phase='running',source=str(source),
        guard_script=str(guard_script),guard_sha256=hashlib.sha256(guard_script.read_bytes()).hexdigest(),
        exclusive_gpu_required=a.require_exclusive_gpu)
    def save():
        temporary=a.output/'PROVENANCE.tmp'
        temporary.write_text(json.dumps(provenance,indent=2)+'\n')
        temporary.replace(a.output/'PROVENANCE.json')
    save()
    threads='1' if a.mode=='cuda' else '20'
    env=dict(os.environ,FLUPRO='/home/yuhanglu/fluka',CORSIKA_DATA=str(source/'modules/data'),
        OMP_NUM_THREADS=threads,OMP_THREAD_LIMIT=threads,OMP_PROC_BIND='false',OMP_PLACES='cores',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        guard_command=[sys.executable,str(guard_script),
            '--output',str(a.output/(label+'-guard')),'--timeout','7200','--rss-limit-gib','5',
            '--sample-resources','--sample-threads']
        if a.require_exclusive_gpu:guard_command+=['--require-exclusive-gpu']
        subprocess.run([*guard_command,'--',*command],env=env,cwd=source,check=True)
        guard=json.loads((a.output/(label+'-guard/summary.json')).read_text())
        assert guard['pass'] and guard['returncode']==0
        provenance.update(complete=True,phase='guard-complete-acceptance-required',
                          process_s=guard['elapsed_s'],acceptance_complete=False,
                          performance_valid=guard.get('performance_valid',False))
        if a.require_exclusive_gpu and not guard['performance_valid']:
            # A telemetry gap does not erase a physically completed event.
            # Strict timing acceptance must reject it; no automatic retry.
            provenance['phase']='simulation-complete-performance-invalid'
    except BaseException as error:
        provenance.update(complete=False,phase='failed',error=repr(error))
        raise
    finally:
        save()
    print('UHE simulation ended; physics/output/performance acceptance still required.',flush=True)


if __name__=='__main__':
    main()
