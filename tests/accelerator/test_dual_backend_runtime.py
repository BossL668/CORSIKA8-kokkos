"""One ELF, both execution spaces; also verify fail-closed initialization."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = []
    for backend, threads, hidden, success in [
        ("cuda", 1, False, True), ("openmp", 1, False, True),
        ("openmp", 2, False, True), ("openmp", 4, False, True),
        ("cuda", 2, False, False), ("hip", 1, False, False),
        ("openmp", 2, True, False), ("cuda", 1, True, False),
    ]:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                   OMP_PROC_BIND="spread", OMP_PLACES="cores")
        if hidden:
            env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [str(args.probe.resolve()), "--backend", backend,
             "--threads", str(threads), "--values", "1048576"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=60, preexec_fn=no_core)
        record = {"backend": backend, "threads": threads, "hidden_gpu": hidden,
                  "expected_success": success, "code": result.returncode,
                  "output": result.stdout}
        assert (result.returncode == 0) == success, record
        if success:
            fields = dict(line.split("=", 1) for line in result.stdout.splitlines()
                          if "=" in line)
            assert fields["backend"] == backend, record
            for field in ("scan_valid", "queue_stable_order", "queue_roundtrip_exact", "memory_query_valid"):
                assert fields[field] == "true", record
            assert fields["checksum"] == "366504574976", record
            if backend == "openmp":
                assert int(fields["concurrency"]) == threads, record
                assert fields["gpu"] == "false", record
        elif hidden:
            assert "requires a working NVIDIA device" in result.stdout, record
        elif backend == "hip":
            # Cooperative execution modes are also advertised by the current
            # dual runtime; HIP must still be rejected, without falling back.
            assert "This dual Kokkos binary supports cuda, openmp, cuda-openmp or openmp-cuda" in result.stdout, record
        else:
            assert "at most one host thread" in result.stdout, record
        record["pass"] = True
        report.append(record)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: 8 dual-backend runtime cases (CUDA, OpenMP 1/2/4, rejection gates)")


if __name__ == "__main__":
    main()
