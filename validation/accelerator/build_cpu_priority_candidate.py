#!/usr/bin/env python3
"""Build an isolated CPU-primary candidate from a frozen per-host stage.

Default: print a read-only plan. Add --execute to copy frozen source, overlay
only the allowlisted scheduler/test files, configure a fresh build and compile.
No install, simulation, Conan invocation, cache editing or production restart.
Run this script ON the intended host: its frozen CMakeCache supplies that host's
compilers, CUDA architecture, patched PROPOSAL/Kokkos toolchain and dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import time


OVERLAY_ALLOWLIST = (
    "applications/detail/air_shower_kokkos/KokkosShowerReport.hpp",
    "corsika/accelerator/em/detail/CpuPrimarySubshowerPolicy.hpp",
    "corsika/accelerator/em/detail/IndependentSubshowerPump.hpp",
    "src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp",
    "tests/accelerator/testIndependentSubshowerPump.cpp",
    "tests/accelerator/testKokkosCooperativeBackend.cpp",
)
BLOCKING_WAIT_OVERLAY_ALLOWLIST = (
    "corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp",
    "corsika/accelerator/em/kokkos/CudaCompletionTicket.hpp",
    "corsika/accelerator/em/detail/KokkosBackendInstance.hpp",
    "src/accelerator/em/kokkos/KokkosBackendInstance.inl",
    "corsika/accelerator/em/common/Types.hpp",
)
CHECKPOINT_WAIT_OVERLAY_ALLOWLIST = (
    "corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp",
    "corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp",
    "corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp",
)
BLOCKING_WAIT_STATISTICS_ADDITION = (
    "    // CPU-primary helper only: event-scoped blocking waits, never global CUDA\n"
    "    // scheduling flags. Durations overlap OpenMP work and are not kernel time.\n"
    "    bool auxiliary_blocking_wait_enabled{};\n"
    "    std::uint64_t auxiliary_blocking_wait_calls{};\n"
    "    double auxiliary_blocking_wait_ms{};\n")
TARGET_ALLOWLIST = (
    "c8_air_shower", "testIndependentSubshowerPump", "testAdaptiveSubshowerControl",
    "testCooperativeDriverAffinity", "testKokkosCooperativeBackend",
    "testKokkosCooperativeRuntime", "testCooperativeScheduling",
    "testCooperativeScheduleJournal", "testIndependentEndpointDriver",
    "kokkos_em_proposal_native_decision_oracle",
)
DEFAULT_TARGETS = TARGET_ALLOWLIST[:5]
CACHE_KEYS = (
    "CMAKE_BUILD_TYPE", "CMAKE_TOOLCHAIN_FILE", "CONAN_CMAKE_DIR",
    "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER",
    "CMAKE_Fortran_COMPILER", "CMAKE_MAKE_PROGRAM", "CMAKE_CUDA_ARCHITECTURES",
    "CORSIKA_ENABLE_KOKKOS", "CORSIKA_KOKKOS_BACKEND",
    "CORSIKA_KOKKOS_ARCHITECTURE", "CORSIKA_KOKKOS_CUDA_ARCHITECTURES",
    "CORSIKA_BUILD_MOUNTAIN_APPLICATION", "C8_COOPERATIVE_BACKEND_CUPTI",
    "C8_TAUOLA_PREFIX", "USE_Pythia8_C8", "Pythia8_PREFIX", "WITH_FLUKA",
    # A prebuilt CORSIKA Pythia install uses include/corsika_modules and
    # lib/corsika, not the upstream SYSTEM layout inferred from PREFIX.
    "Pythia8_Pythia_h_LOC", "Pythia8_INCLUDE_DIR", "Pythia8_LIBRARY",
    "Pythia8_DATA_DIR",
)
for _language in ("C", "CXX", "CUDA", "Fortran", "EXE_LINKER", "MODULE_LINKER",
                  "SHARED_LINKER", "STATIC_LINKER"):
    CACHE_KEYS += ("CMAKE_" + _language + "_FLAGS", "CMAKE_" + _language + "_FLAGS_RELEASE")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_entries(path):
    entries = {}
    for line in path.read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        if ":" in key_type:
            key, kind = key_type.split(":", 1)
            entries[key] = (kind, value)
    return entries


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def contains(parent, child):
    return parent == child or parent in child.parents


def copy_overlay(source, destination, candidate_source):
    """Make only a freshly copied regular file writable, never the snapshot.

    copytree preserves read-only snapshot modes. Refuse linked files,
    escaping parent directories and hardlinks before changing a mode.
    """
    source, destination, candidate_source = map(Path, (source, destination, candidate_source))
    if not contains(candidate_source.resolve(), destination.parent.resolve()):
        raise ValueError('overlay destination escaped candidate source')
    info = destination.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('overlay destination must be a private regular copy')
    destination.chmod(info.st_mode | stat.S_IWUSR)
    shutil.copy2(source, destination)


def selected_overlay_files(requested, include_blocking_wait=False, include_checkpoint_wait=False):
    if any(path not in OVERLAY_ALLOWLIST for path in requested):
        raise ValueError("unapproved base overlay")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate base overlay")
    if include_checkpoint_wait and not include_blocking_wait:
        raise ValueError('checkpoint waits require the explicit blocking-wait overlay')
    return (tuple(requested) + (BLOCKING_WAIT_OVERLAY_ALLOWLIST if include_blocking_wait else ())
            + (CHECKPOINT_WAIT_OVERLAY_ALLOWLIST if include_checkpoint_wait else ()))


def validate_checkpoint_overlay(relative, before, after):
    """Only the reviewed wait plumbing may change in these shared physics headers."""
    if before == after:
        return 'unchanged'
    if relative.endswith('KokkosWavefrontQueue.hpp'):
        replacements = (
            ('#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>\n',
             '#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>\n'
             '#include <corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp>\n'),
            ('    std::vector<gpu::em::EmParticleState> download(\n'
             '        ExecutionSpace const& execution) const {',
             '    std::vector<gpu::em::EmParticleState> download(\n'
             '        ExecutionSpace const& execution,\n'
             '        ResidentExecutionWait<ExecutionSpace>* const blocking_wait = nullptr) const {'),
            ('      ordered_execution.fence("download Kokkos EM wavefront");',
             '      // Only CPU-primary\'s explicit blocking mode changes this boundary.\n'
             '      // In particular, never introduce a progress callback/reentrant host-work\n'
             '      // point here for GPU-primary or other existing callers. The queue\'s\n'
             '      // owning stream, not the compatibility argument, orders pack and copy.\n'
             '      auto* const waiter = blocking_wait && blocking_wait->blockingEnabled()\n'
             '                               ? blocking_wait : nullptr;\n'
             '      waitResidentExecution(\n'
             '          ordered_execution, "download Kokkos EM wavefront", waiter);'),
        )
    elif relative.endswith(('KokkosResidentPhotonCascade.hpp', 'KokkosResidentLeptonCascade.hpp')):
        species = 'photons' if relative.endswith('KokkosResidentPhotonCascade.hpp') else 'leptons'
        replacements = ((f'      result.remaining_{species} = queue.download(execution);',
                         f'      result.remaining_{species} = queue.download(execution, cooperative_wait);'),)
    else:
        raise ValueError('unapproved checkpoint overlay')
    expected = before
    for old, new in replacements:
        if expected.count(old) != 1:
            raise ValueError('frozen checkpoint context is not uniquely matched')
        expected = expected.replace(old, new, 1)
    if expected != after:
        raise ValueError('checkpoint overlay contains changes beyond approved wait plumbing')
    return 'exact blocking-only owning-stream checkpoint wait plumbing'


def validate_statistics_overlay(before, after):
    if before == after:
        return 'unchanged'
    if after.count(BLOCKING_WAIT_STATISTICS_ADDITION) != 1 or after.replace(
            BLOCKING_WAIT_STATISTICS_ADDITION, '', 1) != before:
        raise ValueError('Types.hpp overlay contains changes beyond three approved wait-statistics fields')
    return 'exact three wait-statistics fields and their two-line explanatory comment only'


def require_new_stage(candidate, frozen, overlay):
    if candidate.exists() or candidate.is_symlink():
        raise ValueError("candidate stage already exists; choose a new unused directory")
    if not candidate.parent.is_dir():
        raise ValueError("candidate parent must already exist")
    for protected in (frozen, overlay):
        if contains(protected, candidate) or contains(candidate, protected):
            raise ValueError("candidate must not contain or be inside frozen/overlay source")


def inventory(root):
    """Hash regular source files; preserve but do not traverse data symlink."""
    rows = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                if relative != "modules/data" or not path.resolve().is_dir():
                    raise ValueError("unapproved source symlink: " + relative)
                rows[relative] = {"symlink": str(path.resolve()),
                                  "read_only_reused_data": True,
                                  "cmake_sha256": digest(path / "CMakeLists.txt")}
            elif path.is_file():
                rows[relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return rows


def make_plan(args):
    frozen = args.frozen_stage.resolve(strict=True)
    overlay = args.overlay_source.resolve(strict=True)
    # Reject dangling symlinks before resolving the target path.
    if args.candidate_stage.is_symlink():
        raise ValueError("candidate cannot be a symlink")
    candidate = args.candidate_stage.resolve()
    require_new_stage(candidate, frozen, overlay)
    entries = cache_entries(frozen / "build/CMakeCache.txt")
    get = lambda key: entries[key][1]
    if Path(get("CMAKE_HOME_DIRECTORY")).resolve() != frozen / "source":
        raise ValueError("frozen cache does not belong to frozen source")
    for key, expected in (("CMAKE_BUILD_TYPE", "Release"),
                          ("CORSIKA_KOKKOS_BACKEND", "CUDA_OPENMP"),
                          ("CORSIKA_ENABLE_KOKKOS", "ON"),
                          ("CORSIKA_BUILD_MOUNTAIN_APPLICATION", "OFF"),
                          ("USE_Pythia8_C8", "SYSTEM"), ("WITH_FLUKA", "ON")):
        if get(key) != expected:
            raise ValueError("unexpected frozen configuration: " + key)
    for key in ("CMAKE_COMMAND", "CMAKE_TOOLCHAIN_FILE", "CMAKE_C_COMPILER",
                "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_Fortran_COMPILER"):
        if not Path(get(key)).is_file():
            raise ValueError("missing host tool/dependency: " + key)
    for key in ("CONAN_CMAKE_DIR", "C8_TAUOLA_PREFIX", "Pythia8_PREFIX"):
        if not Path(get(key)).is_dir():
            raise ValueError("missing host dependency directory: " + key)
    fluka = args.flupro.resolve(strict=True)
    if not (fluka / "libflukahp.a").is_file():
        raise ValueError("FLUPRO lacks libflukahp.a")
    overlays = []
    checkpoint_wait = getattr(args, 'include_cpu_primary_checkpoint_wait', False)
    selected_overlays = selected_overlay_files(
        args.overlay, args.include_cpu_primary_blocking_wait, checkpoint_wait)
    for relative in selected_overlays:
        path = overlay / relative
        if path.is_symlink() or not path.is_file() or not contains(overlay, path.resolve()):
            raise ValueError("overlay must be a regular file inside overlay source: " + relative)
        old = frozen / "source" / relative
        if not old.is_file() or old.is_symlink():
            raise ValueError("overlay must replace an existing frozen file: " + relative)
        row = {"path": relative, "before_sha256": digest(old), "after_sha256": digest(path)}
        if relative == "corsika/accelerator/em/common/Types.hpp":
            row['restricted_diff_validation'] = validate_statistics_overlay(old.read_text(), path.read_text())
        if relative in CHECKPOINT_WAIT_OVERLAY_ALLOWLIST:
            row['restricted_diff_validation'] = validate_checkpoint_overlay(
                relative, old.read_text(), path.read_text())
        overlays.append(row)
    cmake = get("CMAKE_COMMAND")
    source, build = candidate / "source", candidate / "build"
    selected = {key: entries[key] for key in CACHE_KEYS if key in entries}
    # Cached effective flags avoid changes caused by a different activated shell.
    # Compiler paths are supplied without resolving symlinks (Conda drivers may
    # use argv[0] to find their sysroot). No CMakeCache/build object is copied.
    configure = [cmake, "-S", str(source), "-B", str(build),
                 "-G", get("CMAKE_GENERATOR")]
    configure += ["-D{}:{}={}".format(key, kind if kind != "UNINITIALIZED" else "STRING", value)
                  for key, (kind, value) in selected.items()]
    configure += ["-DCMAKE_INSTALL_PREFIX:PATH=" + str(candidate / "install-unused"),
                  "-DC8_FLUKALIB:FILEPATH=" + str(fluka / "libflukahp.a")]
    return dict(schema=1, frozen_stage=str(frozen), overlay_source=str(overlay),
                candidate_stage=str(candidate), frozen_cache_sha256=digest(frozen / "build/CMakeCache.txt"),
                overlay_allowlist=list(selected_overlay_files(
                    OVERLAY_ALLOWLIST, args.include_cpu_primary_blocking_wait, checkpoint_wait)),
                overlays=overlays,
                cpu_primary_blocking_wait_overlay=args.include_cpu_primary_blocking_wait,
                cpu_primary_checkpoint_wait_overlay=checkpoint_wait,
                reused_cache_entries=selected, configure_command=configure,
                build_command=[cmake, "--build", str(build), "--parallel", str(args.jobs),
                               "--target", *args.targets],
                flupro=str(fluka), jobs=args.jobs, targets=list(args.targets),
                minimum_available_gib=args.memory_floor_gib,
                maximum_build_tree_rss_gib=args.max_build_rss_gib,
                timeout_seconds=args.timeout, installed=False,
                simulations_started=False, complete=False)


def guarded_command(command, cwd, log_path, env, args):
    import psutil
    if psutil.virtual_memory().available < args.memory_floor_gib * 1024**3:
        raise RuntimeError("available memory below required floor before build")
    result = dict(command=command, peak_tree_rss_bytes=0, failure=None)
    started = time.monotonic()
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    members = [process] + process.children(recursive=True)
                except psutil.NoSuchProcess:
                    members = []
                rss = 0
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                result["peak_tree_rss_bytes"] = max(result["peak_tree_rss_bytes"], rss)
                if psutil.virtual_memory().available < args.memory_floor_gib * 1024**3:
                    result["failure"] = "available memory below floor"
                elif rss > args.max_build_rss_gib * 1024**3:
                    result["failure"] = "build tree RSS limit"
                elif time.monotonic() - started > args.timeout:
                    result["failure"] = "build timeout"
                if result["failure"]:
                    break
                time.sleep(0.5)
        finally:
            if child.poll() is None:
                # Only this script's newly created child process group.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            result["returncode"] = child.wait()
    result["seconds"] = time.monotonic() - started
    save(log_path.with_suffix(".json"), result)
    if result["returncode"] != 0 or result["failure"]:
        raise RuntimeError("candidate command failed; inspect " + str(log_path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-stage", type=Path, required=True)
    parser.add_argument("--overlay-source", type=Path, required=True)
    parser.add_argument("--candidate-stage", type=Path, required=True)
    parser.add_argument("--flupro", type=Path, required=True)
    parser.add_argument("--overlay", nargs="+", choices=OVERLAY_ALLOWLIST,
                        default=OVERLAY_ALLOWLIST)
    parser.add_argument("--include-cpu-primary-blocking-wait", action="store_true",
                        help="Opt in to five reviewed v5 wait/instance/statistics files in addition to the six-file default")
    parser.add_argument("--include-cpu-primary-checkpoint-wait", action="store_true",
                        help="Also permit only the exact reviewed blocking-only final queue-download plumbing")
    parser.add_argument("--targets", nargs="+", choices=TARGET_ALLOWLIST, default=DEFAULT_TARGETS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--memory-floor-gib", type=float, default=4)
    parser.add_argument("--max-build-rss-gib", type=float, default=8)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= len(os.sched_getaffinity(0)):
        parser.error("jobs must be between 1 and available affinity CPU count")
    if min(args.memory_floor_gib, args.max_build_rss_gib, args.timeout) <= 0:
        parser.error("memory and timeout limits must be positive")
    if len(set(args.overlay)) != len(args.overlay) or len(set(args.targets)) != len(args.targets):
        parser.error("duplicate overlay or target")
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    frozen, overlay, candidate = map(Path, (plan["frozen_stage"], plan["overlay_source"], plan["candidate_stage"]))
    before = inventory(frozen / "source")
    candidate.mkdir(exist_ok=False)  # Atomic claim: never adopt an existing stage.
    save(candidate / "BUILD.json", plan)
    save(candidate / "FROZEN_SOURCE_INVENTORY.json", before)
    shutil.copy2(frozen / "build/CMakeCache.txt", candidate / "FROZEN_CMakeCache.txt")
    shutil.copytree(frozen / "source", candidate / "source", symlinks=True)
    for row in plan["overlays"]:
        source = overlay / row["path"]
        if digest(source) != row["after_sha256"]:
            raise RuntimeError("overlay changed while staging: " + row["path"])
        copy_overlay(source, candidate / "source" / row["path"], candidate / "source")
    after = inventory(candidate / "source")
    changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
    unexpected = sorted(set(changed) - {r["path"] for r in plan["overlays"]})
    save(candidate / "SOURCE_BOUNDARY_AUDIT.json", dict(changed=changed, unexpected=unexpected,
                                                       pass_=not unexpected))
    if unexpected:
        raise RuntimeError("unexpected source change while staging")
    save(candidate / "CANDIDATE_SOURCE_INVENTORY.json", after)
    env = dict(os.environ, FLUPRO=plan["flupro"])
    compiler_dirs = [str(Path(plan["reused_cache_entries"][key][1]).parent)
                     for key in ("CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_Fortran_COMPILER")]
    env["PATH"] = os.pathsep.join(dict.fromkeys(compiler_dirs)) + os.pathsep + env.get("PATH", "")
    for name in ("configure", "build"):
        plan[name + "_result"] = guarded_command(plan[name + "_command"], candidate,
                                                  candidate / (name + ".log"), env, args)
        save(candidate / "BUILD.json", plan)
    plan["binaries"] = {str(path.relative_to(candidate)): digest(path)
                         for target in args.targets
                         for path in (candidate / "build/applications" / target,
                                      candidate / "build/tests/accelerator" / target)
                         if path.is_file()}
    plan["complete"] = True
    save(candidate / "BUILD.json", plan)
    print("Build completed without install or simulation: " + str(candidate), flush=True)


if __name__ == "__main__":
    main()
