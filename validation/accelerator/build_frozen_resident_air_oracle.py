#!/usr/bin/env python3
"""Link isolated frozen resident instances into the existing air application.

The overlay is prepared separately and never edits production source. It calls
the frozen pre-refactor monolithic loops with CURRENT physics constants,
tables, runtime, writers and application. This is a single-endpoint oracle,
NOT a cooperative backend and NOT an installable production binary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    a = p.parse_args()
    build, overlay = a.build.resolve(), a.overlay.resolve()
    directory = build / "src/accelerator/em/kokkos"
    flags = (directory / "CMakeFiles/CORSIKA8KokkosEm.dir/flags.make").read_text()
    compiler = re.search(r"# compile CUDA with (.+)", flags)[1]
    options = []
    for name in ("CUDA_DEFINES", "CUDA_INCLUDES", "CUDA_FLAGS"):
        options += shlex.split(re.search(r"^" + name + r" = (.*)$", flags, re.M)[1])
    commands, objects = [], []
    for mode in ("Cuda", "OpenMP"):
        obj = overlay / (mode + ".o")
        if obj.exists():
            raise RuntimeError("Refusing to overwrite diagnostic object")
        cmd = [compiler, "-forward-unknown-to-host-compiler"] + options + [
            "-x", "cu", "-c", str(overlay / (mode + ".cu")), "-o", str(obj)]
        commands.append({"cwd": str(directory), "command": cmd})
        subprocess.run(cmd, cwd=directory, check=True)
        objects.append(str(obj))
    # Direct object definitions resolve factory symbols before the existing
    # static backend archive. Its normal instance objects are not extracted.
    command = shlex.split((build / "applications/CMakeFiles/c8_air_shower.dir/link.txt").read_text())
    binary = overlay / "c8_air_shower_frozen_resident_SINGLE_ENDPOINT_ONLY"
    if binary.exists():
        raise RuntimeError("Refusing to overwrite diagnostic executable")
    command[command.index("-o") + 1] = str(binary)
    command[1:1] = objects
    commands.append({"cwd": str(build / "applications"), "command": command})
    subprocess.run(command, cwd=build / "applications", check=True)
    with (overlay / "build.json").open("x") as f:
        json.dump({"commands": commands, "binary": str(binary),
                   "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                   "restriction": "single endpoint only; frozen resident loops, current common physics"}, f, indent=2)


if __name__ == "__main__":
    main()
