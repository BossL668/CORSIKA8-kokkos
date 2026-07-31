#!/usr/bin/env python3
"""Run a resumable, CPU-pinned scalar CORSIKA 8 ensemble.

This launcher is intentionally independent of the CUDA acceptance driver.  It
creates one CORSIKA output library per shower, records immutable executable and
input identities, pins every long-lived worker to one logical CPU, and can
resume only outputs whose command and provenance match exactly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import queue
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
PROVENANCE_FILENAME = "validation_provenance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent scalar CORSIKA 8 showers on a fixed CPU set."
        )
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--antenna-file", type=Path, required=True)
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=130)
    parser.add_argument("--cpu-list", default="120-249")
    parser.add_argument("--seed-start", type=int, default=10100051)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--energy-gev", type=float, default=1.0e5)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--shower-core-x-m", type=float, default=0.0)
    parser.add_argument("--shower-core-y-m", type=float, default=0.0)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument("--em-cut-gev", type=float, default=0.5e-3)
    parser.add_argument("--em-thinning", type=float, default=1.0e-6)
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.0,
        help=(
            "Explicit --max-weight. Zero preserves the original executable's "
            "automatic value by omitting the option."
        ),
    )
    parser.add_argument("--flupro", type=Path, default=Path("/home/yuhanglu/fluka"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only completed shards with exactly matching provenance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the immutable configuration without running.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    status = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def parse_cpu_list(specification: str) -> list[int]:
    result: list[int] = []
    for field in specification.split(","):
        field = field.strip()
        if not field:
            raise ValueError("empty field in --cpu-list")
        if "-" in field:
            lower_text, upper_text = field.split("-", 1)
            lower = int(lower_text)
            upper = int(upper_text)
            if lower < 0 or upper < lower:
                raise ValueError(f"invalid CPU range: {field}")
            result.extend(range(lower, upper + 1))
        else:
            value = int(field)
            if value < 0:
                raise ValueError(f"invalid CPU ID: {value}")
            result.append(value)
    if len(result) != len(set(result)):
        raise ValueError("--cpu-list contains duplicate CPU IDs")
    return result


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as destination:
        destination.write(json_bytes(value))
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("xb") as destination:
        destination.write(json_bytes(value))
        destination.flush()
        os.fsync(destination.fileno())


def c8_command(
    args: argparse.Namespace,
    output: Path,
    seed: int,
) -> list[str]:
    command = [
        str(args.executable),
        "-p",
        str(args.primary_pdg),
        "-E",
        f"{args.energy_gev:.17g}",
        "-N",
        "1",
        "-f",
        str(output),
        "--seed",
        str(seed),
        "--zenith",
        f"{args.zenith_deg:.17g}",
        "--azimuth",
        f"{args.azimuth_deg:.17g}",
        "--shower-core-x",
        f"{args.shower_core_x_m:.17g}",
        "--shower-core-y",
        f"{args.shower_core_y_m:.17g}",
        "--ring",
        str(args.ring),
        "--antenna-file",
        str(args.antenna_file),
        "--emcut",
        f"{args.em_cut_gev:.17g}",
        "--emthin",
        f"{args.em_thinning:.17g}",
        "--hadcut",
        f"{args.had_cut_gev:.17g}",
        "--mucut",
        f"{args.mu_cut_gev:.17g}",
        "--taucut",
        f"{args.tau_cut_gev:.17g}",
        "--verbosity",
        "warn",
    ]
    if args.maximum_weight > 0.0:
        insertion = command.index("--hadcut")
        command[insertion:insertion] = [
            "--max-weight",
            f"{args.maximum_weight:.17g}",
        ]
    return command


def immutable_configuration(
    args: argparse.Namespace,
    cpus: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "executable": artifact_identity(args.executable),
        "antenna_file": artifact_identity(args.antenna_file),
        "runner": artifact_identity(Path(__file__)),
        "events": args.events,
        "jobs": args.jobs,
        "cpus": cpus[: args.jobs],
        "seed_start": args.seed_start,
        "physics": {
            "primary_pdg": args.primary_pdg,
            "energy_GeV": args.energy_gev,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "shower_core_x_m": args.shower_core_x_m,
            "shower_core_y_m": args.shower_core_y_m,
            "ring": args.ring,
            "em_cut_GeV": args.em_cut_gev,
            "em_thinning": args.em_thinning,
            "had_cut_GeV": args.had_cut_gev,
            "mu_cut_GeV": args.mu_cut_gev,
            "tau_cut_GeV": args.tau_cut_gev,
            "maximum_weight": args.maximum_weight,
            "maximum_weight_cli_omitted": args.maximum_weight == 0.0,
            "em_backend": "legacy_scalar_proposal",
            "radio_backend": "cpu_coreas_and_zhs",
        },
        "flupro": artifact_identity(args.flupro / "libflukahp.a"),
        "single_thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def expected_provenance(
    configuration: dict[str, Any],
    command: list[str],
    seed: int,
    cpu: int,
) -> dict[str, Any]:
    encoded_command = json.dumps(
        command,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "proposal",
        "seed": seed,
        "events": 1,
        "cpu": cpu,
        "executable": configuration["executable"],
        "antenna_file": configuration["antenna_file"],
        "flupro": configuration["flupro"],
        "runner": configuration["runner"],
        "physics": configuration["physics"],
        "command": command,
        "command_sha256": hashlib.sha256(encoded_command).hexdigest(),
    }


def validate_completed_shard(
    output: Path,
    command: list[str],
    provenance: dict[str, Any],
    *,
    require_provenance: bool,
) -> None:
    required = [
        output / "summary.yaml",
        output / "config.yaml",
        output / "profile" / "config.yaml",
        output / "particles" / "summary.yaml",
        output / "CoREAS" / "summary.yaml",
        output / "ZHS" / "summary.yaml",
        output / "simulation_timing" / "summary.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("missing completed-shard artifacts: " + ", ".join(missing))

    summary = load_yaml_mapping(output / "summary.yaml")
    if int(summary.get("showers", -1)) != 1:
        raise ValueError(f"unexpected shower count in {output / 'summary.yaml'}")
    if int(summary.get("seed", -1)) != int(provenance["seed"]):
        raise ValueError(f"unexpected seed in {output / 'summary.yaml'}")

    config = load_yaml_mapping(output / "config.yaml")
    observed_command = shlex.split(str(config.get("args", "")))
    if observed_command != command:
        raise ValueError(f"recorded CORSIKA command differs in {output / 'config.yaml'}")

    timing = load_yaml_mapping(output / "simulation_timing" / "summary.yaml")
    shower_timing = timing.get("shower_0")
    if not isinstance(shower_timing, dict):
        raise ValueError(f"missing shower_0 timing in {output}")
    wall_ms = float(shower_timing.get("wall_time_ms", -1.0))
    if (
        wall_ms <= 0.0
        or shower_timing.get("closed") is not True
        or shower_timing.get("status") != "closed"
    ):
        raise ValueError(f"invalid simulation timing in {output}")

    if require_provenance:
        path = output / PROVENANCE_FILENAME
        if not path.is_file():
            raise ValueError(f"missing runner provenance: {path}")
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != provenance:
            raise ValueError(f"runner provenance differs: {path}")


def validate_args(args: argparse.Namespace, cpus: list[int]) -> None:
    args.executable = args.executable.resolve()
    args.output_root = args.output_root.resolve()
    args.antenna_file = args.antenna_file.resolve()
    args.flupro = args.flupro.resolve()
    if not args.executable.is_file() or not os.access(args.executable, os.X_OK):
        raise ValueError(f"executable is not runnable: {args.executable}")
    if not args.antenna_file.is_file():
        raise ValueError(f"antenna file is missing: {args.antenna_file}")
    if not (args.flupro / "libflukahp.a").is_file():
        raise ValueError(f"FLUKA library is missing below: {args.flupro}")
    if args.events <= 0 or args.jobs <= 0:
        raise ValueError("--events and --jobs must be positive")
    if args.jobs > args.events or args.jobs > len(cpus):
        raise ValueError("--jobs cannot exceed the event or CPU count")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative")
    if args.energy_gev <= 0.0:
        raise ValueError("--energy-gev must be positive")
    if any(
        value <= 0.0
        for value in (
            args.em_cut_gev,
            args.had_cut_gev,
            args.mu_cut_gev,
            args.tau_cut_gev,
        )
    ):
        raise ValueError("all particle cuts must be positive")
    if not 0.0 <= args.em_thinning <= 1.0:
        raise ValueError("--em-thinning must be in [0, 1]")
    if args.maximum_weight < 0.0:
        raise ValueError("--maximum-weight must be non-negative")
    available = os.sched_getaffinity(0)
    unavailable = [cpu for cpu in cpus[: args.jobs] if cpu not in available]
    if unavailable:
        raise ValueError(f"requested CPUs are outside process affinity: {unavailable}")


class EnsembleRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        cpus: list[int],
        configuration: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        self.args = args
        self.cpus = cpus[: args.jobs]
        self.configuration = configuration
        self.manifest = manifest
        self.manifest_lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.active: dict[int, subprocess.Popen[Any]] = {}
        self.stop = threading.Event()
        self.failures: list[dict[str, Any]] = []

    def update_manifest(self, status: str | None = None) -> None:
        with self.manifest_lock:
            if status is not None:
                self.manifest["status"] = status
            self.manifest["updated_utc"] = utc_now()
            self.manifest["completed"] = len(self.manifest["completed_indices"])
            self.manifest["failed"] = len(self.failures)
            self.manifest["failures"] = list(self.failures)
            atomic_write_json(
                self.args.output_root / "run_manifest.json",
                self.manifest,
            )

    def terminate_active(self) -> None:
        with self.process_lock:
            active = list(self.active.values())
        for process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if all(process.poll() is not None for process in active):
                return
            time.sleep(0.1)
        for process in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def quarantine_incomplete(self, output: Path) -> Path:
        destination_root = self.args.output_root / "failed_attempts"
        destination_root.mkdir(exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = destination_root / f"{output.name}.{suffix}"
        sequence = 0
        while destination.exists():
            sequence += 1
            destination = destination_root / f"{output.name}.{suffix}.{sequence}"
        output.rename(destination)
        return destination

    def run_one(self, index: int, cpu: int) -> None:
        output = self.args.output_root / f"proposal_shard_{index:03d}"
        seed = self.args.seed_start + index
        command = c8_command(self.args, output, seed)
        provenance = expected_provenance(
            self.configuration,
            command,
            seed,
            cpu,
        )
        if output.exists():
            try:
                validate_completed_shard(
                    output,
                    command,
                    provenance,
                    require_provenance=True,
                )
                return
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.quarantine_incomplete(output)

        attempt_logs = self.args.output_root / "attempt_logs"
        attempt_logs.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt_log = attempt_logs / f"proposal_shard_{index:03d}.{stamp}.log"
        launch_command = [
            "/usr/bin/time",
            "-v",
            "taskset",
            "-c",
            str(cpu),
            *command,
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "FLUPRO": str(self.args.flupro),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        started_utc = utc_now()
        started = time.perf_counter()
        with attempt_log.open("xb") as log:
            process = subprocess.Popen(
                launch_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            with self.process_lock:
                self.active[index] = process
            return_code = process.wait()
            with self.process_lock:
                self.active.pop(index, None)
        elapsed = time.perf_counter() - started
        if return_code != 0:
            quarantined = (
                str(self.quarantine_incomplete(output))
                if output.exists()
                else None
            )
            raise RuntimeError(
                f"shard {index} exited {return_code}; log={attempt_log}; "
                f"quarantined={quarantined}"
            )

        validate_completed_shard(
            output,
            command,
            provenance,
            require_provenance=False,
        )
        write_json_exclusive(output / PROVENANCE_FILENAME, provenance)
        result = {
            "schema_version": SCHEMA_VERSION,
            "index": index,
            "seed": seed,
            "cpu": cpu,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "runner_wall_seconds": elapsed,
            "return_code": return_code,
            "attempt_log": str(attempt_log),
        }
        write_json_exclusive(output / "runner_result.json", result)
        shutil.copy2(
            attempt_log,
            self.args.output_root / f"proposal_shard_{index:03d}.log",
        )

    def worker(self, cpu: int, pending: queue.Queue[int]) -> None:
        while not self.stop.is_set():
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            try:
                self.run_one(index, cpu)
                with self.manifest_lock:
                    if index not in self.manifest["completed_indices"]:
                        self.manifest["completed_indices"].append(index)
                        self.manifest["completed_indices"].sort()
                self.update_manifest()
            except Exception as error:  # noqa: BLE001 - preserve remote batch state
                with self.manifest_lock:
                    self.failures.append(
                        {
                            "index": index,
                            "cpu": cpu,
                            "error": str(error),
                            "time_utc": utc_now(),
                        }
                    )
                self.update_manifest("running_with_failures")
            finally:
                pending.task_done()

    def run(self, pending: list[int]) -> int:
        work: queue.Queue[int] = queue.Queue()
        for index in pending:
            work.put(index)
        active_cpus = self.cpus[: min(len(self.cpus), len(pending))]
        self.update_manifest("running")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(active_cpus)
        ) as executor:
            futures = [
                executor.submit(self.worker, cpu, work)
                for cpu in active_cpus
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        if self.failures:
            self.update_manifest("failed")
            return 2
        self.update_manifest("complete")
        return 0


def acquire_lock(output_root: Path) -> Path:
    lock = output_root / ".runner.lock"
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_utc": utc_now(),
    }
    try:
        write_json_exclusive(lock, payload)
    except FileExistsError:
        observed = json.loads(lock.read_text(encoding="utf-8"))
        same_host = observed.get("hostname") == socket.gethostname()
        pid = int(observed.get("pid", -1))
        alive = same_host and Path(f"/proc/{pid}").exists()
        if alive:
            raise RuntimeError(f"another ensemble runner is active: {observed}")
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lock.rename(output_root / f".runner.lock.stale.{suffix}")
        write_json_exclusive(lock, payload)
    return lock


def main() -> int:
    args = parse_args()
    cpus = parse_cpu_list(args.cpu_list)
    validate_args(args, cpus)
    configuration = immutable_configuration(args, cpus)

    if args.dry_run:
        print(json.dumps(configuration, indent=2, sort_keys=True))
        return 0

    manifest_path = args.output_root / "run_manifest.json"
    if args.output_root.exists() and not args.resume:
        raise ValueError(
            f"output root exists; pass --resume to inspect it: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=args.resume)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("immutable_configuration") != configuration:
            raise ValueError(
                "existing run manifest differs from the requested configuration"
            )
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
            },
            "immutable_configuration": configuration,
            "completed_indices": [],
            "completed": 0,
            "failed": 0,
            "failures": [],
        }
        atomic_write_json(manifest_path, manifest)

    lock = acquire_lock(args.output_root)
    runner = EnsembleRunner(args, cpus, configuration, manifest)

    def handle_signal(signum: int, _frame: Any) -> None:
        runner.stop.set()
        runner.update_manifest(f"interrupted_by_signal_{signum}")
        runner.terminate_active()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    pending: list[int] = []
    try:
        for index in range(args.events):
            output = args.output_root / f"proposal_shard_{index:03d}"
            command = c8_command(args, output, args.seed_start + index)
            provenance_path = output / PROVENANCE_FILENAME
            try:
                observed_provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                cpu = int(observed_provenance["cpu"])
                if cpu not in runner.cpus:
                    raise ValueError(f"recorded CPU is outside requested set: {cpu}")
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
                cpu = runner.cpus[index % len(runner.cpus)]
            provenance = expected_provenance(
                configuration,
                command,
                args.seed_start + index,
                cpu,
            )
            if output.exists():
                try:
                    validate_completed_shard(
                        output,
                        command,
                        provenance,
                        require_provenance=True,
                    )
                    if index not in manifest["completed_indices"]:
                        manifest["completed_indices"].append(index)
                    continue
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            pending.append(index)
        manifest["completed_indices"].sort()
        runner.update_manifest("running" if pending else "complete")
        if not pending:
            return 0
        return runner.run(pending)
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
