#!/usr/bin/env python3
"""Linux/WSL beta5 dispatcher. No Kokkos or GPU library is loaded by this process.

Installed as <root>/bin/c8_air_shower, with independent <root>/<backend>/bin/
executables. Selection happens before exec, never during a shower.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys


BACKENDS = ("cuda", "hip", "sycl", "openmp")
PROBE_TIMEOUT = 30
HOST_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class LaunchError(RuntimeError):
    pass


def parser():
    result = argparse.ArgumentParser(
        description="beta5: one entry point, independent Kokkos executables",
        allow_abbrev=False, add_help=False,
        epilog="Other arguments are passed unchanged to c8_air_shower. "
               "Use --backend openmp -- --help for the application's help.")
    result.add_argument("--backend", choices=("auto",) + BACKENDS, default="auto")
    result.add_argument("--list-backends", action="store_true",
                        help="list installed backends without initializing hardware")
    result.add_argument("--check-backends", action="store_true",
                        help="probe installed backends; initialize devices in child processes")
    result.add_argument("--dry-run", action="store_true",
                        help="probe and print the selected command as JSON, without a shower")
    result.add_argument("--launcher-help", "-h", "--help", action="store_true")
    return result


def option_value(arguments, name):
    """Inspect only our safety-relevant application options; never re-tokenize."""
    found = []
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                raise LaunchError(f"{name} requires a value")
            found.append(arguments[index + 1])
        elif argument.startswith(name + "="):
            found.append(argument.split("=", 1)[1])
    if len(found) > 1:
        raise LaunchError(f"duplicate {name}; specify it once")
    return found[0] if found else None


def integer_option(arguments, name, default=0):
    value = option_value(arguments, name)
    try:
        number = int(value) if value is not None else default
    except ValueError as error:
        raise LaunchError(f"{name} requires an integer") from error
    if number < 0:
        raise LaunchError(f"{name} cannot be negative")
    return number


def selected_devices(arguments):
    """Public NVIDIA selection; old worker ordinals remain an independent option."""
    result = []
    seen = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        key, equals, value = argument.partition("=")
        if key in ("--device", "--devices"):
            if seen:
                raise LaunchError("specify --device once; do not combine it with --devices")
            seen = True
            values = [value] if equals else []
            while index + 1 < len(arguments) and not arguments[index + 1].startswith("-"):
                index += 1
                values.append(arguments[index])
            for token in values:
                for item in token.split(","):
                    item = item.strip()
                    if item and all(c in "0123456789" for c in item):
                        item = str(int(item))
                        if int(item) > 2147483647:
                            raise LaunchError("GPU index is too large")
                    elif not (item.startswith("GPU-") and len(item) > 4 and
                              all(c.isascii() and (c.isalnum() or c == "-") for c in item)):
                        raise LaunchError("--device expects GPU indices or UUIDs, without empty entries")
                    if item in result:
                        raise LaunchError("duplicate GPU ID: " + item)
                    result.append(item)
            if not result or len(result) > 255:
                raise LaunchError("--device requires 1..255 distinct GPU IDs")
        index += 1
    if seen and option_value(arguments, "--kokkos-device") is not None:
        raise LaunchError("do not combine --device/--devices with --kokkos-device")
    return result


def resolve_requested_gpus(ids):
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        raise LaunchError("--device requires nvidia-smi to resolve physical GPU IDs")
    result = subprocess.run([executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise LaunchError("NVIDIA GPU query failed")
    aliases = {}
    for line in result.stdout.splitlines():
        index, comma, uuid = line.partition(",")
        if comma and index.strip().isdigit() and uuid.strip().startswith("GPU-"):
            aliases[index.strip()] = aliases[uuid.strip()] = uuid.strip()
    if any(item not in aliases for item in ids):
        raise LaunchError("unknown NVIDIA GPU in --device")
    uuids = [aliases[item] for item in ids]
    if len(set(uuids)) != len(uuids):
        raise LaunchError("GPU IDs refer to the same device")
    return uuids


def application_arguments(arguments):
    # The launcher is the accelerated convenience entry point. Direct binaries
    # keep their scalar defaults. Explicit scalar reference requests are honored.
    em = option_value(arguments, "--em-backend")
    radio = option_value(arguments, "--radio-backend")
    if em is None:
        em = "proposal" if radio == "cpu" else "kokkos"
    if radio is None:
        radio = "cpu" if em == "proposal" else "kokkos"
    if (em, radio) not in (("proposal", "cpu"), ("kokkos", "kokkos"), ("egs4", "kokkos"), ("egs4", "cpu")):
        raise LaunchError("EM/radio must be kokkos/kokkos, egs4/kokkos, egs4/cpu or proposal/cpu")
    source = option_value(arguments, "--gpu-physics-source")
    if source not in (None, "proposal-native"):
        raise LaunchError("beta5 supports only proposal-native")
    if integer_option(arguments, "--hadronic-workers", 1) > 1:
        raise LaunchError("shower-internal hadronic workers are not supported by this launcher")
    if option_value(arguments, "--hadronic-backend") not in (None, "scalar"):
        raise LaunchError("use --hadronic-backend scalar")
    added = []
    if option_value(arguments, "--em-backend") is None:
        added += ["--em-backend", em]
    if option_value(arguments, "--radio-backend") is None:
        added += ["--radio-backend", radio]
    return added + arguments, em == "proposal"


def inventory(root):
    records = {}
    for backend in BACKENDS:
        prefix = root / backend
        record = {"backend": backend, "installed": False,
                  "executable": str(prefix / "bin/c8_air_shower"),
                  "probe": str(prefix / "bin/kokkos_backend_probe")}
        try:
            manifest = json.loads((prefix / "share/corsika/backend.json").read_text())
            if (manifest.get("schema") != 1 or manifest.get("backend") != backend or
                    manifest.get("physics_source") != "proposal-native"):
                raise ValueError("invalid backend manifest")
            for key in ("executable", "probe"):
                path = Path(record[key])
                if not path.is_file() or not os.access(path, os.X_OK):
                    raise ValueError(f"missing executable: {path.name}")
            record.update(installed=True, build=manifest)
        except (OSError, ValueError, AttributeError) as error:
            record["reason"] = str(error)
        records[backend] = record
    return records


def child_environment(backend, arguments, probe=False):
    environment = dict(os.environ)
    if backend != "openmp" or probe:
        # Probes use one host thread; GPU showers also use serial host scheduling.
        for name in HOST_THREAD_ENV:
            environment[name] = "1"
        environment["OMP_THREAD_LIMIT"] = "1"
    else:
        threads = integer_option(arguments, "--kokkos-num-threads")
        if threads:
            environment["OMP_NUM_THREADS"] = str(threads)
    return environment


def no_core_dump():
    # An unavailable driver may make Kokkos abort. Do not leave multi-GB cores.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def probe(record, arguments, timeout=PROBE_TIMEOUT):
    result = dict(record)
    result["available"] = False
    if not record["installed"]:
        return result
    command = [record["probe"], "--values", "1024", "--threads", "1"]
    try:
        environment = child_environment(record["backend"], arguments, probe=True)
        ids = selected_devices(arguments)
        if ids:
            if record["backend"] != "cuda":
                raise LaunchError("--device requires the CUDA backend")
            environment["CUDA_VISIBLE_DEVICES"] = resolve_requested_gpus(ids)[0]
            command += ["--device", "0"]
        elif record["backend"] != "openmp":
            command += ["--device", str(integer_option(arguments, "--kokkos-device"))]
        completed = subprocess.run(
            command, capture_output=True, text=True, errors="replace", timeout=timeout,
            env=environment,
            preexec_fn=no_core_dump)
        if completed.returncode:
            message = (completed.stderr or completed.stdout).strip()[-1500:]
            raise LaunchError(f"probe exited {completed.returncode}: {message}")
        fields = dict(line.split("=", 1) for line in completed.stdout.splitlines()
                      if "=" in line)
        expected = {"backend": record["backend"], "scan_valid": "true",
                    "queue_stable_order": "true", "queue_roundtrip_exact": "true",
                    "memory_query_valid": "true",
                    "gpu": "false" if record["backend"] == "openmp" else "true",
                    "openmp": "true" if record["backend"] == "openmp" else "false"}
        if any(fields.get(key) != value for key, value in expected.items()):
            raise LaunchError("probe backend, runtime isolation or primitive validation mismatch")
        result.update(available=True, device=fields)
    except (OSError, subprocess.TimeoutExpired, LaunchError) as error:
        result["reason"] = str(error)
    return result


def select_backend(requested, records, arguments, scalar=False):
    ids = selected_devices(arguments)
    device_requested = bool(ids) or option_value(arguments, "--kokkos-device") is not None
    if ids:
        if requested not in ("auto", "cuda") or scalar:
            raise LaunchError("--device requires accelerated CUDA execution")
        requested = "cuda"
    threads = integer_option(arguments, "--kokkos-num-threads")
    if requested == "openmp" and device_requested:
        raise LaunchError("--kokkos-device cannot be used with --backend openmp")
    if requested not in ("auto", "openmp") and threads > 1:
        raise LaunchError("GPU backends reject --kokkos-num-threads > 1")
    if requested == "auto" and (threads > 1 or scalar):
        if device_requested:
            raise LaunchError("auto CPU selection conflicts with --kokkos-device")
        requested = "openmp"
    if requested != "auto":
        checked = probe(records[requested], arguments)
        if not checked["available"]:
            raise LaunchError(f"{requested} unavailable: {checked.get('reason', 'probe failed')}; "
                              "no backend substitution was performed")
        return checked
    checked = [probe(records[name], arguments) for name in BACKENDS if name != "openmp"]
    available = [item for item in checked if item["available"]]
    if len(available) > 1:
        raise LaunchError("multiple GPU backends are usable; select --backend explicitly: " +
                          ", ".join(item["backend"] for item in available))
    if available:
        return available[0]
    if device_requested:
        raise LaunchError("no usable GPU backend for --kokkos-device; refusing CPU substitution")
    for item in checked:
        if item["installed"]:
            print(f"[c8 launcher] {item['backend']} unavailable: {item.get('reason')}",
                  file=sys.stderr)
    host = probe(records["openmp"], arguments)
    if not host["available"]:
        raise LaunchError("no usable installed backend; build/install a matching backend first; "
                          f"OpenMP: {host.get('reason')}")
    print("[c8 launcher] no usable installed GPU backend; selecting OpenMP before startup",
          file=sys.stderr)
    return host


def main(argv=None, root=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    # '--' separates launcher parsing from application help/arguments.
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    option_value(arguments[:separator], "--backend")
    options, forwarded = parser().parse_known_args(arguments[:separator])
    forwarded += arguments[separator + 1:]
    if options.launcher_help or not arguments:
        parser().print_help()
        return 0
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    records = inventory(root)
    if options.list_backends:
        print(json.dumps({"install_root": str(root), "backends": records}, indent=2))
        return 0
    if options.check_backends:
        names = BACKENDS if options.backend == "auto" else (options.backend,)
        checked = {name: probe(records[name], forwarded) for name in names}
        print(json.dumps(checked, indent=2))
        return 0 if any(item["available"] for item in checked.values()) else 2
    forwarded, scalar = application_arguments(forwarded)
    selected = select_backend(options.backend, records, forwarded, scalar)
    environment = child_environment(selected["backend"], forwarded)
    command = [selected["executable"]] + forwarded
    if options.dry_run:
        print(json.dumps({"requested_backend": options.backend, "selected": selected,
                          "argv": command,
                          "thread_environment": {key: environment.get(key) for key in
                                                 HOST_THREAD_ENV + ("OMP_THREAD_LIMIT",)}},
                         indent=2))
        return 0
    print(f"[c8 launcher] backend={selected['backend']}; executable={command[0]}",
          file=sys.stderr, flush=True)
    # No shell, no argument rewriting, no supervising retry/fallback loop.
    # exec preserves PID/signals and the shower's exit code.
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (LaunchError, OSError) as error:
        print(f"[c8 launcher] ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)
