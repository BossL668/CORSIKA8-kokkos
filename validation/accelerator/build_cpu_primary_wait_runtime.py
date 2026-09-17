#!/usr/bin/env python3
"""Compile a validation-only CUDA wait test using frozen runtime-target flags.

No CMake invocation, cache/object modification, installation or execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    build = stage / 'build'
    target = 'testKokkosCooperativeRuntime'
    base = build / 'tests/accelerator/CMakeFiles' / (target + '.dir')
    flags = dict(line.split(' = ', 1) for line in (base / 'flags.make').read_text().splitlines()
                 if ' = ' in line)
    cache = dict(line.split('=', 1) for line in (build / 'CMakeCache.txt').read_text().splitlines()
                 if line and not line.startswith(('#', '//')) and '=' in line)
    compilers = [value for key, value in cache.items() if key.split(':')[0] == 'CMAKE_CUDA_COMPILER']
    if len(compilers) != 1:
        raise ValueError('frozen CUDA compiler is not uniquely specified')
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archived = output / source.name
    shutil.copy2(source, archived)
    obj = output / 'wait_runtime.o'
    binary = output / 'cpu_primary_blocking_wait_runtime'
    compile_command = [compilers[0], '-x', 'cu', *shlex.split(flags['CUDA_DEFINES']),
                       *shlex.split(flags['CUDA_INCLUDES']), *shlex.split(flags['CUDA_FLAGS']),
                       '-c', str(archived), '-o', str(obj)]
    link = shlex.split((base / 'link.txt').read_text())
    indices = [i for i, value in enumerate(link) if value.endswith('/' + target + '.cpp.o')]
    if len(indices) != 1:
        raise ValueError('frozen runtime object is not uniquely specified')
    link[indices[0]] = str(obj)
    link[link.index('-o') + 1] = str(binary)
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    archived_source=str(archived), frozen_stage=str(stage),
                    compile_command=compile_command, link_command=link,
                    writes_production=False, complete=False)
    path = output / 'BUILD.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run(compile_command, cwd=build / 'tests/accelerator', check=True)
    subprocess.run(link, cwd=build / 'tests/accelerator', check=True)
    manifest.update(complete=True, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(binary)


if __name__ == '__main__':
    main()
