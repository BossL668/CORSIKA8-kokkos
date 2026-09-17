#!/usr/bin/env python3
"""PSR-only replay of the three saved 100 PeV nu_tau inputs in three materials."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import traceback
import types

import yaml

BASE = pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results')
MATERIALS = BASE / 'beta5_material_models_20260913'
PREVIOUS = BASE / 'beta5_doublebang_radio_audit_20260912'
BINARY_HASH = '320b84f6a200f5954de8a1f13b2c450280fb7483a7e22c433819073c83621ab5'
MODELS = [
    ('silica_SiO2', 'silica', 'Silica (SiO2)'),
    ('limestone_CaCO3', 'limestone', 'Calcite / limestone (CaCO3)'),
    ('granite_H_C_O_Na_Mg_Al_Si_K_Ca_Fe', 'granite',
     'Granite (H, C, O, Na, Mg, Al, Si, K, Ca, Fe)'),
]


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def setarg(command, key, value):
    command[command.index(key) + 1] = str(value)


def prepare(root):
    if (root / 'campaign.json').exists():
        raise RuntimeError('Campaign already prepared; inputs are immutable')
    for marker in ['TESTS_OPENMP_PASSED', 'MATERIAL_RUNS_OPENMP_PASSED', 'FLUKA_TARGETS_PASSED']:
        if not (MATERIALS / marker).exists():
            raise RuntimeError('Missing validation gate: ' + marker)
    source_manifest = json.loads((MATERIALS / 'report/source_manifest.json').read_text())
    for row in source_manifest['files']:
        if digest(MATERIALS / 'source' / row['path']) != row['sha256']:
            raise RuntimeError('Validated source changed: ' + row['path'])
    binary = MATERIALS / 'build-openmp/applications/c8_terrain_cascade'
    if digest(binary) != BINARY_HASH:
        raise RuntimeError('Validated executable changed')
    bundle = root / 'bundle'
    bundle.mkdir()
    (root / 'scenes').mkdir()
    (root / 'report').mkdir()
    shutil.copy2(binary, bundle / binary.name)
    shutil.copy2(MATERIALS / 'report/source_manifest.json', bundle / 'material_source_manifest.json')
    shutil.copy2(MATERIALS / 'source/validation/terrain/doublebang_radio/run_psr.py', bundle / 'owned_run.py')
    for filename in ['material.md', 'materials.yaml']:
        shutil.copy2(MATERIALS / 'source/documentation' / filename, bundle / filename)
    original = yaml.safe_load((PREVIOUS / 'scene_radio.yaml').read_text())
    shutil.copy2(PREVIOUS / 'scene_radio.yaml', bundle / 'previous_scene.yaml')
    mesh = pathlib.Path(original['geometry']['mesh_path'])
    if digest(mesh) != original['provenance']['mesh_sha256']:
        raise RuntimeError('Original terrain mesh changed')
    cases = []
    for tag, short, label in MODELS:
        card = bundle / (tag + '.yaml')
        shutil.copy2(MATERIALS / 'source/configs/mountain/materials' / (short + '.yaml'), card)
        shutil.copy2(MATERIALS / (short + '_resolved.yaml'), bundle / (tag + '_expected.yaml'))
        scene = copy.deepcopy(original)
        for key in ['material', 'rock_density_g_cm3', 'rock_refractive_index',
                    'rock_hadronic_target_approximation', 'attenuation_length_m']:
            scene['geometry'].pop(key, None)
        scene['geometry']['material_file'] = str(card)
        scene['radio'].pop('rock_attenuation_length_m', None)
        assert scene['radio']['enabled'] is True
        scene['provenance']['radio_note'] = 'Three original stations; material radio constants from supplied materials.yaml, not site measurements.'
        scene['provenance']['material_description'] = label
        scene_file = root / 'scenes' / (tag + '.yaml')
        scene_file.write_text(yaml.safe_dump(scene, sort_keys=False))
        cache = MATERIALS / 'table-cache' / short
        if not cache.exists():
            cache = MATERIALS / 'source/modules/data'
        for seed in [158, 946, 3605]:
            reference = PREVIOUS / 'runs' / ('openmp_seed' + str(seed)) / 'status.json'
            old_status = json.loads(reference.read_text())
            assert old_status['complete'] and old_status['returncode'] == 0
            command = list(old_status['command'])
            original_command = list(command)
            name = 'openmp_' + tag + '_seed' + str(seed)
            command[3] = str(bundle / binary.name)
            setarg(command, '--scene', scene_file)
            setarg(command, '--output', root / 'runs' / name / 'output')
            setarg(command, '--aux-cache', MATERIALS / 'auxiliary')
            # Only file locations change; all physics and scheduling arguments
            # are inherited from the last successful 256-core radio replay.
            differences = [dict(index=i, previous=x, current=y)
                           for i, (x, y) in enumerate(zip(original_command, command)) if x != y]
            allowed = {3} | {command.index(k) + 1 for k in ['--scene', '--output', '--aux-cache']}
            assert all(d['index'] in allowed for d in differences)
            assert command[2] == '0-255' and command[command.index('--threads') + 1] == '256'
            cases.append(dict(tag=name, material=tag, label=label, seed=seed, command=command,
                              cache=str(cache), reference=str(reference), command_changes=differences))
    files = [dict(path=str(p), sha256=digest(p)) for p in sorted(bundle.iterdir()) if p.is_file()]
    files += [dict(path=str(p), sha256=digest(p)) for p in sorted((root / 'scenes').iterdir())]
    files += [dict(path=str(mesh), sha256=digest(mesh))]
    # Archive the resolved runtime libraries as identities, without rebuilding.
    ldd = subprocess.check_output(['ldd', str(bundle / binary.name)], text=True)
    (bundle / 'ldd.txt').write_text(ldd)
    if 'not found' in ldd:
        raise RuntimeError('Missing runtime dependency')
    libraries = []
    for line in ldd.splitlines():
        if ' => /' in line:
            path = pathlib.Path(line.split(' => ', 1)[1].split(' (', 1)[0])
            libraries.append(dict(path=str(path), sha256=digest(path)))
    manifest = dict(created_utc=now(), host=socket.gethostname(), cases=cases, files=files,
                    runtime_libraries=libraries, binary_sha256=BINARY_HASH,
                    physical_cores=256, execution='sequential OpenMP resident queue; CoREAS + ZHS',
                    material_reference=str(MATERIALS), source_reference=str(PREVIOUS))
    write_json(root / 'campaign.json', manifest)
    write_json(root / 'progress.json', dict(state='prepared', completed=[], total=len(cases), updated_utc=now()))
    print('Prepared %d immutable cases in %s' % (len(cases), root), flush=True)


class AuditedProcess(subprocess.Popen):
    """Record actual OpenMP worker affinity from /proc once all workers exist."""
    def __init__(self, command, **kwargs):
        self.folder = pathlib.Path(command[command.index('--output') + 1]).parent
        self.affinity_recorded = False
        write_json(self.folder / 'environment.json', {
            k: kwargs['env'].get(k) for k in ['OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
                                             'CORSIKA_DATA', 'LD_LIBRARY_PATH', 'FLUPRO']})
        super().__init__(command, **kwargs)

    def poll(self):
        result = super().poll()
        if result is None and not self.affinity_recorded:
            try:
                tids = sorted(int(p.name) for p in pathlib.Path('/proc/%d/task' % self.pid).iterdir())
                affinity = {str(tid): sorted(os.sched_getaffinity(tid)) for tid in tids}
                # Runtime I/O/CUDA-context service threads can coexist with the
                # 256-member OpenMP team, without taking additional CPU cores.
                good = len(tids) >= 256 and all(len(v) == 1 for v in affinity.values())
                good = good and {v[0] for v in affinity.values()} == set(range(256))
                if good:
                    write_json(self.folder / 'affinity.json', dict(pid=self.pid, verified=True,
                               observed_threads=len(tids), physical_cores=256, workers=affinity, utc=now()))
                    self.affinity_recorded = True
            except (OSError, ProcessLookupError):
                pass
        return result


def execute(root):
    lock = (root / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / 'campaign.json').read_text())
    for row in manifest['files'] + manifest['runtime_libraries']:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Input/runtime changed: ' + row['path'])
    spec = importlib.util.spec_from_file_location('owned_run', root / 'bundle/owned_run.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = root
    helper.subprocess = types.SimpleNamespace(Popen=AuditedProcess, check_output=subprocess.check_output,
        STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    completed = []
    current = None
    try:
        for case in manifest['cases']:
            current = case['tag']
            if shutil.disk_usage(root).free < 50 * 1024**3:
                raise RuntimeError('Less than 50 GiB available before next event')
            os.environ['CORSIKA_DATA'] = case['cache']
            write_json(root / 'progress.json', dict(state='running', current=current, completed=completed,
                total=len(manifest['cases']), updated_utc=now()))
            print('START ' + current + ' ' + now(), flush=True)
            folder, summary = helper.run(current, list(case['command']), 'openmp', 24 * 3600, reuse_complete=True)
            helper.checks(folder, summary, True)
            expected = yaml.safe_load((root / 'bundle' / (case['material'] + '_expected.yaml')).read_text())
            actual = summary['resolved_material']
            assert actual['transport'] == expected['transport']
            assert actual['radio']['refractive_index'] == expected['radio']['refractive_index']
            assert actual['radio']['field_attenuation_length_m'] == expected['radio']['field_attenuation_length_m']
            assert actual['magnetic_field_enu_T'] == expected['magnetic_field_enu_T']
            native = next(p for p in summary['proposal_materials'] if p['name'] == actual['id'])
            assert native['photon_pair_LPM'] and native['I_eV'] == actual['transport']['ionisation']['I_eV']
            assert abs(native['density_g_cm3'] * 1000 - actual['transport']['density_kg_m3']) < 1e-9
            for algorithm in ['CoREAS', 'ZHS']:
                optical = json.loads((folder / 'output/radio' / algorithm / 'config.json').read_text())['media'][1]
                assert optical['index'] == actual['radio']['refractive_index']
                assert optical['attenuation_length_m'] == actual['radio']['field_attenuation_length_m']
            assert summary['accelerator']['execution_space'] == 'OpenMP'
            assert summary['accelerator']['execution_concurrency'] == 256
            assert summary['radio_result']['execution_space'] == 'OpenMP'
            assert json.loads((folder / 'affinity.json').read_text())['verified']
            write_json(folder / 'material_checks.json', dict(passed=True, material=actual,
                proposal=native, actual_OpenMP_threads=256, actual_physical_cores=256))
            completed.append(current)
            write_json(root / 'progress.json', dict(state='analyzing', current=current, completed=completed,
                total=len(manifest['cases']), updated_utc=now()))
            subprocess.run([sys.executable, str(root / 'code/analyze_psr.py'), '--root', str(root)], check=True)
        write_json(root / 'progress.json', dict(state='complete', completed=completed,
            total=len(manifest['cases']), updated_utc=now()))
        (root / 'ALL_NINE_PASSED').touch()
    except BaseException as error:
        write_json(root / 'progress.json', dict(state='failed', current=current, completed=completed,
            total=len(manifest['cases']), error=str(error), traceback=traceback.format_exc(), updated_utc=now()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025' or not BASE.is_dir():
        raise RuntimeError('Run only on PSR; no local computation')
    if args.prepare:
        prepare(args.root.resolve())
    else:
        execute(args.root.resolve())


if __name__ == '__main__':
    main()
