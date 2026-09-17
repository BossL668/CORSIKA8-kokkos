#!/usr/bin/env python3
"""Link an opt-in diagnostic against existing frozen test libraries/fixture.

Never invokes CMake, rewrites a cached build, or modifies production objects.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    stage = a.stage.resolve()
    build = stage/'build'
    base = build/'tests/accelerator/CMakeFiles/testKokkosCooperativeBackend.dir'
    defs = dict(line.split(' = ', 1) for line in (base/'flags.make').read_text().splitlines() if ' = ' in line)
    link = shlex.split((base/'link.txt').read_text())
    source = a.source.resolve()
    assert source.is_file()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    archived_source = out/source.name
    shutil.copy2(source, archived_source)
    obj = out/'isolation.o'
    binary = out/'cpu_primary_backend_isolation'
    compile_cmd = [link[0], *shlex.split(defs['CXX_DEFINES']), *shlex.split(defs['CXX_INCLUDES']),
                   *shlex.split(defs['CXX_FLAGS']), '-I'+str(stage/'source/tests/accelerator'),
                   '-c', str(archived_source), '-o', str(obj)]
    index = next(i for i, x in enumerate(link) if x.endswith('/testKokkosCooperativeBackend.cpp.o'))
    assert sum(x.endswith('/testKokkosCooperativeBackend.cpp.o') for x in link) == 1
    link[index] = str(obj)
    link[link.index('-o')+1] = str(binary)
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    archived_source=str(archived_source),
                    compile_command=compile_cmd, link_command=link, frozen_stage=str(stage),
                    writes_production=False, complete=False)
    path = out/'BUILD.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run(compile_cmd, cwd=build/'tests/accelerator', check=True)
    subprocess.run(link, cwd=build/'tests/accelerator', check=True)
    manifest.update(complete=True, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(binary)


if __name__ == '__main__':
    main()
