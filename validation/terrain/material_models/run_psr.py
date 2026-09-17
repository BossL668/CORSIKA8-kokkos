#!/usr/bin/env python3
"""Remote-only material admission, native LPM oracles and resident radio runs."""
import argparse, copy, importlib.util, json, os, pathlib, subprocess, time, types
import yaml

ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913')
OLD=ROOT.parent/'beta5_doublebang_radio_audit_20260912'
spec=importlib.util.spec_from_file_location('owned_run',ROOT/'source/validation/terrain/doublebang_radio/run_psr.py')
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h); h.ROOT=ROOT
MODELS={'silica':'S0_SILICA_TRANSPORT_PROXY','limestone':'L0_LIMESTONE_CALCITE_PROXY','granite':'G0_GRANITE_REFERENCE'}

def set_process_placement(backend):
    # Both backends use exactly CPUs 0..255. Rotate the CUDA host team's
    # placement so concurrent serial model initialization does not share CPU 0.
    # Inject only the subprocess launcher of our imported validation helper;
    # the shared helper and all physical application modules stay untouched.
    def launch(*args,**kwargs):
        env=dict(kwargs['env'])
        if backend=='cuda':
            env['OMP_PLACES']=','.join('{%d}'%((i+128)%256) for i in range(256))
        kwargs['env']=env
        command=args[0]; output=pathlib.Path(command[command.index('--output')+1])
        (output.parent/'placement.json').write_text(json.dumps({k:env[k] for k in ['OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND']},indent=2))
        return subprocess.Popen(*args,**kwargs)
    h.subprocess=types.SimpleNamespace(Popen=launch,check_output=subprocess.check_output,
        STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired)

def select_cache(label):
    private=ROOT/'table-cache'/label
    if private.exists():
        # CORSIKA_DATA also resolves QGSJet and geomagnetic datasets. Isolate
        # only writable PROPOSAL tables; retain the complete physical data tree.
        for source in (ROOT/'source/modules/data').iterdir():
            destination=private/source.name
            if source.name!='PROPOSAL' and not destination.exists():
                destination.symlink_to(source,target_is_directory=source.is_dir())
        os.environ['CORSIKA_DATA']=str(private)
    else: os.environ.pop('CORSIKA_DATA',None)

def wait_prewarm(label):
    marker=ROOT/(label+'-prewarm.pid')
    if not marker.exists(): return
    pid=int(marker.read_text()); start=time.monotonic()
    while True:
        try: command=pathlib.Path('/proc/%d/cmdline'%pid).read_bytes()
        except FileNotFoundError: break
        if b'testMaterialOracle' not in command: break
        if time.monotonic()-start>7200: raise RuntimeError('material prewarm timeout: '+label)
        time.sleep(2.)
    path=ROOT/(label+'-prewarm.csv')
    expected={'limestone':360,'granite':1200}[label]
    if not path.exists() or len(path.read_text().splitlines())!=expected+1:
        raise RuntimeError('incomplete prewarm oracle: '+label)

def wait_openmp(label):
    start=time.monotonic()
    status=ROOT/'runs'/('openmp_'+label+'_on')/'status.json'
    while True:
        if status.exists():
            try: result=json.loads(status.read_text())
            except json.JSONDecodeError: result={}
            if result.get('complete') and result.get('returncode')==0: return
            if result.get('returncode') is not None: raise RuntimeError('OpenMP reference failed: '+label)
        if time.monotonic()-start>7200: raise RuntimeError('OpenMP reference timeout: '+label)
        time.sleep(2.)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--backend',choices=['openmp','cuda'],required=True)
    a=p.parse_args(); backend=a.backend
    set_process_placement(backend)
    if not (ROOT/('TESTS_'+backend.upper()+'_PASSED')).exists():
        raise RuntimeError('Current build/test gates must pass before physics runs')
    if not (ROOT/'FLUKA_TARGETS_PASSED').exists():
        raise RuntimeError('Expanded FLUKA target comparison must pass before physics runs')
    results=[]
    for label,model in MODELS.items():
        wait_prewarm(label)
        if backend=='cuda': wait_openmp(label) # no concurrent writes to cold scalar tables
        select_cache(label)
        if backend=='openmp':
            command=['taskset','-c','0-255',str(ROOT/'build-openmp/tests/modules/testMaterialOracle'),
                     model,str(ROOT/'auxiliary'),str(ROOT/(label+'_lpm.csv'))]
            with (ROOT/(label+'_oracle.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        scene=yaml.safe_load((OLD/'scene_control_True.yaml').read_text())
        g=scene['geometry']
        for key in ['rock_density_g_cm3','rock_refractive_index','rock_hadronic_target_approximation','attenuation_length_m']:
            g.pop(key,None)
        g['material']={'preset':model}
        scene['radio'].pop('rock_attenuation_length_m',None)
        scene['provenance']['material_source']='documentation/materials.yaml; native resolved card in run output'
        pair=[]
        for enabled in [False,True]:
            scene['radio']['enabled']=enabled
            scene_path=ROOT/(label+('_on' if enabled else '_off')+'.yaml')
            scene_path.write_text(yaml.safe_dump(scene,sort_keys=False))
            cmd=json.loads((OLD/'runs/openmp_electron_on/command.json').read_text())
            cmd[3]=str(ROOT/('build-'+backend)/'applications/c8_terrain_cascade')
            for key,value in {'--scene':scene_path,'--energy-GeV':10,'--aux-cache':ROOT/'auxiliary',
                              '--emthin':0,'--max-weight':1,'--track-row-limit':1000000,
                              '--transport-step-limit':1000000}.items(): h.setarg(cmd,key,value)
            i=cmd.index('--position-m'); cmd[i+1:i+4]=['0','0','-.2']
            folder,s=h.run(backend+'_'+label+('_on' if enabled else '_off'),cmd,backend,1800,reuse_complete=True)
            check=h.checks(folder,s,enabled)
            resolved=s['resolved_material']; actual=next(x for x in s['proposal_materials'] if x['name']==model)
            assert abs(actual['density_g_cm3']*1000-resolved['transport']['density_kg_m3'])<1.e-9
            assert actual['I_eV']==resolved['transport']['ionisation']['I_eV'] and actual['photon_pair_LPM']
            assert actual['components']==len(resolved['transport']['nuclei'])
            if enabled:
                for algorithm in ['CoREAS','ZHS']:
                    optical=json.loads((folder/'output/radio'/algorithm/'config.json').read_text())['media'][1]
                    assert optical['index']==resolved['radio']['refractive_index']
                    assert optical['attenuation_length_m']==resolved['radio']['field_attenuation_length_m']
                (ROOT/(label+'_resolved.yaml')).write_text(yaml.safe_dump(resolved,sort_keys=False))
            hashes={f.name:h.digest(f) for f in (folder/'output/terrain').glob('*.csv')}
            pair.append(dict(folder=str(folder),checks=check,hashes=hashes,proposal_material=actual,
                             material_tables=s['material_tables'],radio=s.get('radio_result')))
        identical=pair[0]['hashes']==pair[1]['hashes']
        if not identical: raise RuntimeError('radio changed transport: '+label)
        results.append(dict(material=model,radio_does_not_change_transport=identical,pair=pair))
        (ROOT/('materials_'+backend+'.json')).write_text(json.dumps(results,indent=2))
    # Exercise the newly material-aware hadron continuous process with actual
    # FLUKA/QGSJet transport. This is a bounded integration check, not a spectrum.
    for label in MODELS:
        select_cache(label)
        cmd=json.loads((ROOT/'runs'/(backend+'_'+label+'_off')/'command.json').read_text())
        for k,v in {'--primary':'proton','--energy-GeV':100,'--seed':1703}.items(): h.setarg(cmd,k,v)
        folder,s=h.run(backend+'_'+label+'_proton',cmd,backend,1800,reuse_complete=True)
        h.checks(folder,s,False)
    # A pure CPU run also exercises the material construction with original
    # scalar EM transport; no resident scheduler participates in this control.
    if backend=='openmp':
        select_cache('limestone')
        cmd=json.loads((ROOT/'runs/openmp_limestone_off/command.json').read_text())
        h.setarg(cmd,'--em-backend','proposal')
        folder,s=h.run('cpu_limestone',cmd,'openmp',1800,reuse_complete=True)
        assert s['complete'] and s['diagnostics']['material_mismatches']==0
        assert abs(s['energy_ledger']['unexplained_over_initial'])<1.e-8
    (ROOT/('MATERIAL_RUNS_'+backend.upper()+'_PASSED')).touch()

if __name__=='__main__': main()
