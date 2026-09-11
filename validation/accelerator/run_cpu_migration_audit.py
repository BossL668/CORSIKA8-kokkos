#!/usr/bin/env python3
"""Run bounded CPU-reference tests, separating live oracles from self-comparisons.

Uses existing independent diagnostic builds. Never builds, installs, regenerates
production tables, signals an existing task, or overwrites an earlier result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--portable-self-test", action="store_true")
    parser.add_argument("--live-oracles", action="store_true",
                        help="also preserve strict final-state differences and Epair CDF tests")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    original = source.parents[1] / "corsika-21cma" / "corsika"
    reference = source.parents[1] / "corsika-21cma-cuda-beta2" / "corsika8_gpu_refactor_beta2"
    protected_air = source / "applications/c8_air_shower.cpp"
    entries = []
    for root in (source / "corsika/accelerator", source / "src/accelerator"):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                entries.append(dict(path=str(path.relative_to(source)), sha256=sha256(path)))
    scalar_files = [
        "framework/core/PhysicalConstants.hpp", "framework/core/PhysicalUnits.hpp",
        "detail/modules/tracking/TrackingLeapFrogCurved.inl",
        "detail/media/BaseExponential.inl", "detail/media/SlidingPlanarExponential.inl",
        "detail/modules/proposal/ContinuousProcess.inl",
        "detail/modules/proposal/InteractionModel.inl",
        "detail/modules/proposal/ProposalProcessBase.inl",
        "detail/modules/thinning/EMThinning.inl", "detail/modules/ParticleCut.inl",
        "detail/modules/radio/CoREAS.inl", "detail/modules/radio/ZHS.inl",
        "detail/modules/radio/RadioProcess.inl",
        "detail/modules/radio/observers/TimeDomainObserver.inl",
    ]
    references = {}
    for relative in scalar_files:
        values = {}
        for name, root in (("beta5", source), ("original_local", original), ("beta2", reference)):
            path = root / "corsika" / relative
            values[name] = sha256(path) if path.is_file() else None
        references[relative] = values
    manifest = dict(
        build=str(args.build.resolve()), gpu=args.gpu, source=str(source),
        source_hashes=entries, scalar_reference_hashes=references,
        air_source_before=sha256(protected_air),
        scope="bounded module tests; NOT full shower-tree or million-point acceptance",
        tests=[],
    )
    for name in ("CMakeCache.txt", "tests/accelerator/CMakeFiles/testKokkosCpuTransportAlignment.dir/flags.make"):
        path = args.build / name
        if path.is_file():
            (args.output / ("build_" + path.name)).write_text(path.read_text())
    cases = [
        ("testKokkosCpuTransportAlignment", "live scalar ContinuousProcess/ParticleCut/stack plus RNG-domain check", []),
        ("testKokkosCpuDepositionAlignment", "live scalar energy-deposition writer and output contracts", []),
        ("testKokkosScalarThinningAlignment", "live scalar EMThinning, 200000 inputs", []),
        ("testKokkosScalarRadioAlignment", "live current scalar CoREAS/ZHS and observer; not frozen old observer", []),
    ]
    if args.portable_self_test:
        cases.append(("testKokkosProposalNativeTable", "portable host/device self-comparison, not live final-state oracle", ["4096"]))
    if args.live_oracles:
        cases += [
            ("testKokkosLiveFinalStateOracle", "live PROPOSAL conditional final states; strict numerical failure is NOT automatically a physical distribution failure", []),
            ("testKokkosLiveFinalStateOracle__epair_cdf", "24 cells, 65536 samples per cell, live PROPOSAL rho quantile reference; NOT a full process/energy matrix", ["--epair-cdf"]),
        ]
    env = dict(os.environ, OMP_NUM_THREADS="1" if args.gpu else "2", OMP_PROC_BIND="false",
               OPENBLAS_NUM_THREADS="1")
    guard = source / "validation/terrain/run_guarded_diagnostic.py"
    failed = False
    for name, scope, extra in cases:
        binary = args.build.resolve() / "tests/accelerator" / name.split("__", 1)[0]
        item = dict(name=name, scope=scope, binary=str(binary))
        if not binary.is_file():
            item.update(status="missing", returncode=None)
            failed = True
        else:
            item["binary_sha256"] = sha256(binary)
            command = [sys.executable, str(guard), "--output", str(args.output / name), "--timeout", "240"]
            if args.gpu:
                command.append("--gpu")
            command += ["--", str(binary)] + extra
            completed = subprocess.run(command, env=env, cwd=args.build.resolve())
            item.update(status="pass" if completed.returncode == 0 else "fail", returncode=completed.returncode)
            failed |= completed.returncode != 0
        manifest["tests"].append(item)
        manifest["air_source_after"] = sha256(protected_air)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(name, item["status"], flush=True)
    if manifest["air_source_after"] != manifest["air_source_before"]:
        raise RuntimeError("air source changed during audit; investigate external edits")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
