#!/usr/bin/env python3
"""Isolated PSR-only rebuild of the independent interface libraries/application."""
import argparse
import hashlib
import json
import pathlib
import shlex
import shutil
import socket
import subprocess
import time

OLD = pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913')
NVCC = '/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/bin/nvcc'
CXX = '/home/yuhanglu/miniconda/envs/corsika_venv/bin/g++'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    p.add_argument('--mode', choices=['openmp', 'cuda'], required=True)
    p.add_argument('--tests-only', action='store_true')
    p.add_argument('--app-only', action='store_true')
    a = p.parse_args()
    assert socket.gethostname() == 'psrpku2025', 'builds/tests belong on PSR'
    source = a.root / 'source'
    for original in (OLD / 'source/applications/detail/mountain').rglob('*'):
        if original.is_file():
            destination = source / original.relative_to(OLD / 'source')
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
    out = a.root / ('build-' + a.mode)
    out.mkdir(parents=True, exist_ok=True)
    old = OLD / ('build-' + a.mode)
    commands = []
    invocation = str(time.time_ns())
    def run(cmd, cwd, label):
        commands.append({'label': label, 'cwd': str(cwd), 'argv': cmd})
        (out / 'commands.json').write_text(json.dumps(commands, indent=2))
        (out / ('commands_' + invocation + '.json')).write_text(json.dumps(commands, indent=2))
        print(label, flush=True)
        with (out / (label + '.log')).open('w') as log:
            subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    def compile_file(target, relative, output, cuda=False):
        directory = old / target
        flags = {}
        for line in (directory / 'flags.make').read_text().splitlines():
            if ' = ' in line:
                k, v = line.split(' = ', 1)
                flags[k] = shlex.split(v)
        lang = 'CUDA' if cuda else 'CXX'
        compiler = [NVCC, '-forward-unknown-to-host-compiler', '-ccbin=' + CXX] if cuda else [CXX]
        cmd = compiler + flags[lang + '_DEFINES'] + ['-I' + str(source), '-I' + str(OLD / 'source/applications')] + flags[lang + '_INCLUDES'] + flags[lang + '_FLAGS']
        if cuda:
            cmd += ['-x', 'cu']
        cmd += ['-c', str(source / relative), '-o', str(output)]
        run(cmd, directory.parent.parent, output.stem)
    app_target = pathlib.Path('applications/CMakeFiles/c8_terrain_cascade.dir')
    replacements = {}
    if not a.tests_only:
        for target, relative, name, cuda in [
            ('src/transport/CMakeFiles/CORSIKA8InterfaceEm.dir', 'src/transport/InterfaceEmSession.cpp', 'CORSIKA8InterfaceEm', True),
            ('src/radio/interface/CMakeFiles/CORSIKA8InterfaceRadio.dir', 'src/radio/interface/Output.cpp', 'CORSIKA8InterfaceRadio', False)]:
            obj = out / (name + '.o')
            archive = out / ('lib' + name + '.a')
            if not a.app_only:
                compile_file(target, relative, obj, cuda)
                run(['/usr/bin/ar', 'rcs', str(archive), str(obj)], out, name + '-archive')
            replacements[archive.name] = str(archive)
        obj = out / 'c8_terrain_cascade.o'
        compile_file(app_target, 'applications/c8_terrain_cascade.cpp', obj)
        link = shlex.split((old / app_target / 'link.txt').read_text())
        link = [replacements.get(pathlib.Path(x).name, x) for x in link]
        link[next(i for i, x in enumerate(link) if x.endswith('c8_terrain_cascade.cpp.o'))] = str(obj)
        link[link.index('-o') + 1] = str(out / 'c8_terrain_cascade')
        run(link, old / 'applications', 'link-application')
    test = source / 'validation/terrain/dem_boundary/check_geometry.cpp'
    if test.exists() and not a.app_only:
        obj = out / 'check_geometry.o'
        target = pathlib.Path('tests/accelerator/CMakeFiles/testKokkosInterfaceQueue.dir')
        compile_file(target, str(test.relative_to(source)), obj, True)
        loader = source / 'validation/terrain/dem_boundary/MeshLoader.cpp'
        shutil.copy2(OLD / 'source/tests/accelerator/InterfaceRadioBvhMeshLoader.cpp', loader)
        loader_obj = out / 'MeshLoader.o'
        compile_file(app_target, str(loader.relative_to(source)), loader_obj)
        link = shlex.split((old / target / 'link.txt').read_text())
        link[next(i for i, x in enumerate(link) if x.endswith('testKokkosInterfaceQueue.cpp.o'))] = str(obj)
        link[link.index('-o') + 1] = str(out / 'check_geometry')
        link.insert(link.index('-o'), str(loader_obj))
        run(link, old / 'tests/accelerator', 'link-geometry')
    (out / 'BUILT.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.name in ['c8_terrain_cascade', 'check_geometry']}, indent=2))

if __name__ == '__main__':
    main()
