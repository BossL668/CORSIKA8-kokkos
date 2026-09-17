#!/usr/bin/env python3
"""Isolated 300 PeV / 1 EeV SiO2 campaign; simulation and checks only on PSR."""
import argparse
import copy
import csv
import datetime
import fcntl
import importlib.util
import json
import os
import signal
import pathlib
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
import types

os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import yaml

BASE = pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results')
REFERENCE = BASE / 'beta5_material_doublebang_openmp256_20260913'
MATERIALS = BASE / 'beta5_material_models_20260913'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def setarg(command, key, value):
    command[command.index(key) + 1] = str(value)


def prepare(root):
    if (root / 'campaign.json').exists():
        raise RuntimeError('Existing campaign inputs are immutable')
    assert (REFERENCE / 'ALL_NINE_PASSED').exists()
    assert json.loads((REFERENCE / 'report/acceptance.json').read_text())['accepted_count'] == 9
    previous = json.loads((REFERENCE / 'campaign.json').read_text())
    helper = module('previous_driver', REFERENCE / 'code/run_psr.py')
    for row in previous['files'] + previous['runtime_libraries']:
        assert helper.digest(row['path']) == row['sha256'], row['path']
    for name in ['bundle', 'scenes', 'report', 'preflight']:
        (root / name).mkdir()
    assert (root / 'build/BUILT').exists()
    for filename in ['owned_run.py', 'material_source_manifest.json',
                     'material.md', 'materials.yaml', 'silica_SiO2.yaml', 'silica_SiO2_expected.yaml']:
        shutil.copy2(REFERENCE / 'bundle' / filename, root / 'bundle' / filename)
    shutil.copy2(root / 'build/c8_terrain_cascade', root / 'bundle/c8_terrain_cascade')
    # Adapt only this campaign's analysis reader to compressed CSV, and record
    # actual transported vertex daughters while scanning all tracks once.
    owned = root / 'bundle/owned_run.py'
    text = owned.read_text().replace('import numpy as np', 'import numpy as np\nimport gzip')
    text = text.replace('with (folder/"output/terrain/tracks.csv").open() as stream:',
        'with (gzip.open(folder/"output/terrain/tracks.csv.gz","rt") if (folder/"output/terrain/tracks.csv.gz").exists() else (folder/"output/terrain/tracks.csv").open()) as stream:')
    text = text.replace('        charged=0\n',
        '        charged=0\n        wanted={int(d["history_id"]) for v in s["neutrino"].get("interactions",[])+s["tau"].get("decays",[]) for d in v["daughters"]}\n        witnessed={}\n')
    text = text.replace('                if pid==15: tau_tracks.append(t)',
        '                if int(t["history_id"]) in wanted:\n                    key=t["history_id"]\n                    if key not in witnessed: witnessed[key]=dict(first=t,steps=0)\n                    witnessed[key]["steps"]+=1\n                if pid==15: tau_tracks.append(t)')
    text = text.replace('        fft=[]\n',
        '        (folder/"vertex_daughter_tracks.json").write_text(json.dumps(witnessed,indent=2))\n        fft=[]\n')
    owned.write_text(text)
    air_source = MATERIALS / 'source/applications/c8_air_shower.cpp'
    shutil.copy2(air_source, root / 'bundle/air_defaults_reference.cpp')
    save(root / 'bundle/air_defaults.json', dict(emcut_GeV=.0005, hadcut_GeV=.3, mucut_GeV=.3, taucut_GeV=.3,
        air_default_emthin=1e-6, requested_emthin=1e-5,
        maximum_weight_formula='0.5 * emthin * primary_energy_GeV', source_sha256=helper.digest(air_source)))
    shutil.copy2(REFERENCE / 'code/run_psr.py', root / 'bundle/material_driver.py')
    shutil.copy2(REFERENCE / 'code/geometry_figures_psr.py', root / 'bundle/geometry_helpers.py')
    shutil.copy2(REFERENCE / 'report/results.json', root / 'bundle/baseline_results.json')
    shutil.copy2(REFERENCE / 'report/source_components/contributions.json', root / 'bundle/baseline_components.json')
    scene = yaml.safe_load((REFERENCE / 'scenes/silica_SiO2.yaml').read_text())
    (root / 'bundle/baseline_scene.yaml').write_text(yaml.safe_dump(scene, sort_keys=False))
    scene['geometry']['material_file'] = str(root / 'bundle/silica_SiO2.yaml')
    scene['radio'].update(samples=524288, memory_MiB=4096)
    scene['provenance']['campaign_note'] = 'Higher-energy pilot. Original DEM and stations; 500 us transport, 2048 us reception. Natural CC/NC and TAUOLA. No forced vertices or decay selection.'
    scene_file = root / 'scenes/silica_SiO2.yaml'
    scene_file.write_text(yaml.safe_dump(scene, sort_keys=False))
    cases = []
    seed_lists = {}
    for energy in [300000000, 1000000000]:
        with (root / 'build' / ('seeds_%d.csv' % energy)).open() as stream:
            candidates = list(csv.DictReader(stream))
        seeds = [3605, 946, 158]
        for candidate in candidates:
            seed = int(candidate['seed'])
            if seed not in seeds:
                seeds.append(seed)
            if len(seeds) == 30:
                break
        assert len(seeds) == 30
        seed_lists[energy] = seeds
        shutil.copy2(root / 'build' / ('seeds_%d.csv' % energy), root / 'bundle' / ('seed_prefilter_%d.csv' % energy))
    # Interleave energies: obtain the first 3x/10x comparison early, retaining
    # every event regardless of its realized decay channel or signal strength.
    for rank in range(30):
        for energy in [300000000, 1000000000]:
            seed = seed_lists[energy][rank]
            original = next(c for c in previous['cases'] if c['material'] == 'silica_SiO2' and c['seed'] == (seed if seed in [158,946,3605] else 3605))
            tag = 'openmp_silica_SiO2_%dPeV_seed%d' % (energy // 1000000, seed)
            command = list(original['command'])
            command[3] = str(root / 'bundle/c8_terrain_cascade')
            changes = {'--scene': scene_file, '--output': root / 'runs' / tag / 'output',
                       '--energy-GeV': energy, '--transport-window-ns': 500000,
                       '--seed': seed, '--emcut-GeV': .0005, '--hadcut-GeV': .3, '--mucut-GeV': .3,
                       '--emthin': 1e-5, '--max-weight': int(.5e-5 * energy),
                       '--device-memory-MiB': 8192, '--track-row-limit': 1000000000,
                       '--transport-step-limit': 1000000000}
            for k, v in changes.items():
                setarg(command, k, v)
            allowed = {3} | {command.index(k) + 1 for k in changes}
            assert len(command) == len(original['command'])
            differences = [dict(index=i, previous=a, current=b) for i, (a, b) in
                           enumerate(zip(original['command'], command)) if a != b]
            assert all(x['index'] in allowed for x in differences)
            command += ['--compress-terrain-csv']
            cases.append(dict(tag=tag, material='silica_SiO2', label='Silica (SiO2)', seed=seed,
                              energy_GeV=energy, command=command, cache=original['cache'],
                              baseline_tag=original['tag'] if seed in [158,946,3605] else None,
                              command_changes=differences, added_flags=['--compress-terrain-csv']))
    files = [dict(path=str(p), sha256=helper.digest(p)) for directory in ['bundle', 'scenes', 'code']
             for p in sorted((root / directory).iterdir()) if p.is_file()]
    mesh = pathlib.Path(scene['geometry']['mesh_path'])
    files.append(dict(path=str(mesh), sha256=helper.digest(mesh)))
    manifest = dict(created_utc=now(), host=socket.gethostname(), cases=cases, files=files,
                    binary_sha256=helper.digest(root / 'bundle/c8_terrain_cascade'), original_binary_sha256=previous['binary_sha256'], runtime_libraries=previous['runtime_libraries'],
                    baseline_root=str(REFERENCE), physical_cores=256, transport_window_ns=500000,
                    radio_samples=524288, radio_sample_rate_Hz=256e6,
                    target_doublebangs_per_energy=3, maximum_candidates_per_energy=30,
                    seed_prefilter_scope='Approximate cascade Philox interaction-depth draws in original first rock chord. No full shower/decay claim; final qualification uses actual genealogy and transported daughters. Selection is not an unbiased event-rate sample.',
                    comparison_note='Old reference uses emthin=.01, EM cut=.1 GeV, hadron cut=10 GeV and 100 us. New runs use emthin=1e-5, air-default cuts and 500 us. Comparisons cannot isolate primary energy.')
    save(root / 'campaign.json', manifest)
    save(root / 'progress.json', dict(state='prepared', completed=[], total=len(cases), updated_utc=now()))
    print('Prepared up to 60 candidates; require three verified double bangs per energy', flush=True)


def preflight(root, helper, owned):
    folder = root / 'preflight'
    if (folder / 'PASSED').exists():
        return
    build = MATERIALS / 'build-openmp/tests/modules'
    original = MATERIALS / 'source/validation/terrain/material_models/MaterialOracle.cpp'
    text = original.read_text()
    old = 'for(double energy:{1.e3,1.e6,1.e9,1.e11})'
    assert text.count(old) == 1
    source = folder / 'MaterialOracleEeV.cpp'
    source.write_text(text.replace(old, 'for(double energy:{1.e3,1.e6,1.e9,1.e11,3.e11,1.e12})'))
    flags = {}
    for line in (build / 'CMakeFiles/testMaterialOracle.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            flags[k] = shlex.split(v)
    link = shlex.split((build / 'CMakeFiles/testMaterialOracle.dir/link.txt').read_text())
    obj = folder / 'MaterialOracleEeV.o'
    binary = folder / 'MaterialOracleEeV'
    compile_cmd = [link[0]] + flags['CXX_DEFINES'] + flags['CXX_INCLUDES'] + flags['CXX_FLAGS']
    compile_cmd += ['-c', str(source), '-o', str(obj)]
    index = next(i for i, x in enumerate(link) if x.endswith('MaterialOracle.cpp.o'))
    link[index] = str(obj)
    link[link.index('-o') + 1] = str(binary)
    command = ['taskset', '-c', '0-255', str(binary), 'S0_SILICA_TRANSPORT_PROXY',
               str(MATERIALS / 'auxiliary'), str(folder / 'silica_lpm_to_1EeV.csv')]
    save(folder / 'commands.json', [compile_cmd, link, command])
    with (folder / 'build_and_oracle.log').open('w') as log:
        subprocess.run(compile_cmd, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        subprocess.run(link, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    # Oracle throws on any nonfinite or mismatched native/portable LPM result.
    import csv
    with (folder / 'silica_lpm_to_1EeV.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 360 and max(float(r['energy_MeV']) for r in rows) == 1e12
    save(folder / 'result.json', dict(passed=True, points=len(rows), maximum_energy_GeV=1e9,
        maximum_relative_error=max(float(r['relative_error']) for r in rows),
        original_source_sha256=helper.digest(original), oracle_source_sha256=helper.digest(source),
        scope='Native versus portable bremsstrahlung and pair-production LPM factors in SiO2; not an independent physical validation of PROPOSAL or a full shower convergence check.'))
    # Transport is deterministic and must be byte-identical after decompression.
    # Radio uses parallel atomic additions: compare finite numerical arrays,
    # allowing double-precision reduction-order roundoff, not arbitrary bytes.
    import gzip
    import hashlib
    scene = yaml.safe_load((MATERIALS / 'silica_on.yaml').read_text())
    scene['radio'].update(samples=32768, memory_MiB=256)
    scene_file = folder / 'scene_compression.yaml'
    scene_file.write_text(yaml.safe_dump(scene, sort_keys=False))
    baseline_command = json.loads((MATERIALS / 'runs/openmp_silica_on/command.json').read_text())
    outputs, folders = [], []
    for compressed in [False, True]:
        cmd = list(baseline_command)
        cmd[3] = str(root / 'bundle/c8_terrain_cascade') if compressed else str(REFERENCE / 'bundle/c8_terrain_cascade')
        for key, value in {'--scene': scene_file, '--emcut-GeV': .0005, '--hadcut-GeV': .3,
                           '--mucut-GeV': .3, '--emthin': 1e-5, '--max-weight': 1,
                           '--track-row-limit': 10000000, '--transport-step-limit': 10000000}.items():
            setarg(cmd, key, value)
        if compressed:
            cmd += ['--compress-terrain-csv']
        target, summary = owned.run('compression_control_' + ('gzip' if compressed else 'plain'), cmd, 'openmp', 4 * 3600, reuse_complete=True)
        owned.checks(target, summary, True)
        folders.append(target)
        hashes = {}
        for name in ['tracks.csv', 'deposits.csv', 'window_survivors.csv']:
            path = target / 'output/terrain' / (name + ('.gz' if compressed else ''))
            digest = hashlib.sha256()
            with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
                for chunk in iter(lambda: stream.read(1048576), b''):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
        outputs.append(hashes)
    assert outputs[0] == outputs[1], 'Decompressed terrain CSV differs from accepted transport'
    import numpy as np
    import pandas as pd
    radio_comparison = {}
    def compare(name, a, b):
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name
        norm = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-100)
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-100)
        l2 = float(np.linalg.norm(a-b) / norm)
        maximum = float(np.max(np.abs(a-b)) / scale)
        radio_comparison[name] = dict(relative_l2=l2, maximum_error_over_global_maximum=maximum,
            byte_identical=np.array_equal(a,b), tolerance=1e-12)
        assert l2 < 1e-12 and maximum < 1e-12, (name, radio_comparison[name])
    for alg in ['CoREAS', 'ZHS']:
        first, second = [f / 'output/radio' / alg for f in folders]
        for path in first.glob('*.bin'):
            compare(alg + '/' + path.name, np.fromfile(path,dtype=np.float64),
                    np.fromfile(second/path.name,dtype=np.float64))
        for name in ['field.csv','spectrum.csv']:
            a,b = [pd.read_csv(p/name,float_precision='round_trip') for p in [first,second]]
            columns = [c for c in a.columns if c.startswith(('Ex','Ey','Ez'))]
            identity = [c for c in a.columns if c not in columns]
            assert a[identity].equals(b[identity]), (alg,name,'observer/sample identity')
            compare(alg+'/'+name,a[columns].to_numpy(),b[columns].to_numpy())
    save(folder / 'compression_equivalence.json', dict(passed=True, decompressed_csv_hashes=outputs[0],
        radio_comparison=radio_comparison,
        scope='10 GeV electron; every decompressed transport CSV byte is identical. Moment, field and spectrum arrays agree to 1e-12 relative/global-scaled tolerance; atomic radio reduction is not bitwise deterministic. No physics implementation changed.'))
    (folder / 'PASSED').touch()


def execute(root):
    lock = (root / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / 'campaign.json').read_text())
    helper = module('material_driver', root / 'bundle/material_driver.py')
    for row in manifest['files'] + manifest['runtime_libraries']:
        assert helper.digest(row['path']) == row['sha256'], row['path']
    owned = module('owned_run', root / 'bundle/owned_run.py')
    owned.ROOT = root
    class DiskAuditedProcess(helper.AuditedProcess):
        def poll(self):
            result = super().poll()
            if result is None:
                requested = getattr(self, '_disk_stop_requested', None)
                if requested is None and shutil.disk_usage(root).free < 30 * 1024**3:
                    self._disk_stop_requested = time.monotonic()
                    save(root / 'disk_guard.json', dict(pid=self.pid, requested_utc=now(),
                         reason='Free disk below 30 GiB', signal='SIGTERM'))
                    try:
                        os.killpg(self.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif requested is not None and time.monotonic() - requested >= 10:
                    # This executable has been observed to survive SIGTERM.
                    # A disk safeguard must not depend on graceful shutdown.
                    if not getattr(self, '_disk_kill_sent', False):
                        self._disk_kill_sent = True
                        save(root / 'disk_guard.json', dict(pid=self.pid, kill_sent_utc=now(),
                             reason='Free disk below 30 GiB; SIGTERM grace expired', signal='SIGKILL'))
                        try:
                            os.killpg(self.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            return result
    owned.subprocess = types.SimpleNamespace(Popen=DiskAuditedProcess,
        check_output=subprocess.check_output, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    completed, current = [], 'LPM_preflight'
    try:
        save(root / 'progress.json', dict(state='preflight', current=current, completed=completed,
             total=len(manifest['cases']), updated_utc=now()))
        os.environ['CORSIKA_DATA'] = manifest['cases'][0]['cache']
        preflight(root, helper, owned)
        expected = yaml.safe_load((root / 'bundle/silica_SiO2_expected.yaml').read_text())
        classifier = module('classify_doublebang', root / 'code/classify_psr.py')
        qualified = {300000000: [], 1000000000: []}
        for case in manifest['cases']:
            energy = case['energy_GeV']
            if len(qualified[energy]) >= manifest['target_doublebangs_per_energy']:
                continue
            current = case['tag']
            if shutil.disk_usage(root).free < 100 * 1024**3:
                raise RuntimeError('Less than 100 GiB free before next event; retain all completed outputs')
            save(root / 'progress.json', dict(state='running', current=current, completed=completed,
                 total=len(manifest['cases']), qualified=qualified, updated_utc=now()))
            print('START', current, now(), flush=True)
            folder, summary = owned.run(current, list(case['command']), 'openmp', 7 * 24 * 3600, reuse_complete=True)
            save(root / 'progress.json', dict(state='auditing', current=current, completed=completed,
                 total=len(manifest['cases']), updated_utc=now()))
            owned.checks(folder, summary, True)
            actual = summary['resolved_material']
            assert actual['transport'] == expected['transport']
            assert actual['radio'] == expected['radio']
            assert actual['magnetic_field_enu_T'] == expected['magnetic_field_enu_T']
            assert summary['energy_GeV'] == case['energy_GeV']
            assert summary['transport_window_ns'] == manifest['transport_window_ns']
            assert summary['emcut_GeV'] == .0005 and summary['hadcut_GeV'] == .3 and summary['mucut_GeV'] == .3
            assert summary['emthin'] == 1e-5 and summary['max_weight'] == int(.5e-5 * energy)
            assert all(v['photon_pair_LPM'] for v in summary['proposal_materials'])
            for algorithm in ['CoREAS', 'ZHS']:
                config = json.loads((folder / 'output/radio' / algorithm / 'config.json').read_text())
                assert config['samples'] == 524288 and config['sample_rate_Hz'] == 256e6
                assert config['media'][1]['index'] == actual['radio']['refractive_index']
                assert config['media'][1]['attenuation_length_m'] == actual['radio']['field_attenuation_length_m']
            assert summary['accelerator']['execution_space'] == 'OpenMP'
            assert summary['accelerator']['execution_concurrency'] == 256
            assert summary['radio_result']['execution_space'] == 'OpenMP'
            assert json.loads((folder / 'affinity.json').read_text())['verified']
            save(folder / 'material_checks.json', dict(passed=True, material=actual, primary_energy_GeV=case['energy_GeV'], physical_cores=256))
            classification = classifier.classify(folder, summary)
            if classification['qualified_doublebang']:
                qualified[energy].append(current)
            completed.append(current)
            save(root / 'progress.json', dict(state='analyzing', current=current, completed=completed,
                 total=len(manifest['cases']), qualified=qualified, updated_utc=now()))
            subprocess.run([sys.executable, str(root / 'code/analyze_psr.py'), '--root', str(root)], check=True)
        target_met = all(len(v) >= manifest['target_doublebangs_per_energy'] for v in qualified.values())
        save(root / 'progress.json', dict(state='complete' if target_met else 'candidate_limit_reached',
             completed=completed, qualified=qualified, total=len(manifest['cases']), target_met=target_met, updated_utc=now()))
        if target_met:
            (root / 'DOUBLEBANG_TARGET_MET').touch()
    except BaseException as error:
        save(root / 'progress.json', dict(state='failed', current=current, completed=completed,
             total=len(manifest['cases']), qualified=locals().get('qualified', {}), error=str(error), traceback=traceback.format_exc(), updated_utc=now()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('PSR only: no local simulation, build, or numerical checks')
    if args.prepare:
        prepare(args.root.resolve())
    elif args.preflight_only:
        root=args.root.resolve()
        manifest=json.loads((root/'campaign.json').read_text())
        helper=module('material_driver',root/'bundle/material_driver.py')
        for row in manifest['files']+manifest['runtime_libraries']:
            assert helper.digest(row['path'])==row['sha256'],row['path']
        owned=module('owned_run',root/'bundle/owned_run.py');owned.ROOT=root
        owned.subprocess=types.SimpleNamespace(Popen=helper.AuditedProcess,check_output=subprocess.check_output,
            STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired)
        os.environ['CORSIKA_DATA']=manifest['cases'][0]['cache']
        preflight(root,helper,owned)
    else:
        execute(args.root.resolve())


if __name__ == '__main__':
    main()
