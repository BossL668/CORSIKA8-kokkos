#!/usr/bin/env python3
"""Record the exact audited snapshot, not the unrelated dirty workspace diff."""
import difflib,hashlib,json,pathlib,shutil
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def main():
    before=json.loads((ROOT/'source_before.json').read_text());after={}
    for folder in ['applications','corsika','src','tests','cmake','modules','validation']:
        for p in (ROOT/'source'/folder).rglob('*'):
            if p.is_file() and p.suffix in ['.hpp','.cpp','.h','.c','.cmake','.py','.sh','.txt']:
                after[str(p.relative_to(ROOT/'source'))]=digest(p)
    changed=[p for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)]
    allowed_production=['corsika/modules/radio/interface/','src/radio/interface/','src/transport/InterfaceEmSession.cpp','applications/detail/mountain/TerrainAcceleratedRun.hpp','applications/detail/mountain/TerrainShowerOutput.hpp','corsika/modules/transport/detail/InterfaceEmStep.hpp','corsika/modules/transport/detail/InterfaceMagneticPolicy.hpp','corsika/geometry/terrain/TerrainMagneticAccuracy.hpp','corsika/modules/terrain/TerrainMagneticTracking.hpp']
    unexpected=[p for p in changed if p.startswith(('applications/','corsika/','src/')) and not any(p.startswith(a) for a in allowed_production)]
    files=dict(source_before=before,source_after=after)
    out=ROOT/'report';out.mkdir(exist_ok=True)
    (out/'source_hashes.json').write_text(json.dumps(files,indent=2))
    result=dict(changed_paths=changed,unexpected_production_changes=unexpected,
        checks=dict(only_owned_interface_production_changes=not unexpected),
        binaries={mode:digest(ROOT/('build-'+mode)/'applications/c8_terrain_cascade') for mode in ['openmp','cuda']})
    (out/'provenance_check.json').write_text(json.dumps(result,indent=2))
    patch=[]
    for name in changed:
        if not name.startswith(('applications/','corsika/','src/','tests/')):continue
        old=ROOT/'baseline/source'/name;new=ROOT/'source'/name
        patch.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],
            new.read_text().splitlines(True) if new.exists() else [],fromfile='a/'+name,tofile='b/'+name))
    (out/'audited_changes.patch').write_text(''.join(patch))
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    for name in ['controls_openmp.json','controls_cuda.json','residency_summary.json',
                 'bvh-real-dem-OPENMP.json','bvh-real-dem-CUDA.json',
                 'source_roundoff_examples.json','source_roundoff_inventory.json',
                 'tau_energy_current.csv','tau_spin_current.csv',
                 'tests-openmp.log','tests-cuda.log',
                 'magnetic-accuracy-OPENMP.csv','magnetic-accuracy-CUDA.csv',
                 'scene_radio.yaml']:
        file=ROOT/name
        if file.exists():shutil.copy2(file,evidence/name)
    for preset in ['precision','run_resolution']:
        file=ROOT/('optical_oracle_'+preset)/'coreas_summary.json'
        if file.exists():shutil.copy2(file,evidence/('optical_'+preset+'.json'))
    for folder in (ROOT/'runs').glob('*_seed*'):
        for name in ['status.json','checks.json']:
            if (folder/name).exists():shutil.copy2(folder/name,evidence/(folder.name+'_'+name))
        if (folder/'checks.json').exists():
            shutil.copy2(folder/'output/terrain_run.yaml',evidence/(folder.name+'_terrain_run.yaml'))
            for algorithm in ['CoREAS','ZHS']:
                shutil.copy2(folder/'output/radio'/algorithm/'config.json',
                             evidence/(folder.name+'_'+algorithm+'_config.json'))
    manifest=dict(remote_root=str(ROOT),
        raw_events={p.name:str(p/'output') for p in sorted((ROOT/'runs').glob('*_seed*'))
                    if (p/'checks.json').exists()},
        failed_runs=str(ROOT/'failed_runs'),
        source_snapshot=str(ROOT/'source'),
        note='Full transport CSV files and unfiltered radio outputs remain on PSR; report figures use an ideal 50-100 MHz bandpass.')
    (out/'remote_artifacts.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result))
    if unexpected:raise RuntimeError('Unexpected production snapshot mutation')
if __name__=='__main__':main()
