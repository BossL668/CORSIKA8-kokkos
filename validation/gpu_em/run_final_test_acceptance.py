#!/usr/bin/env python3
"""Run the final C++/CUDA and Python regression suites fail-closed."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--minimum-ctest-count", type=int, default=1)
    parser.add_argument("--minimum-python-count", type=int, default=1)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def parse_ctest_count(text: str) -> int | None:
    match = re.search(r"out of\s+(\d+)", text)
    return int(match.group(1)) if match else None


def parse_unittest_count(text: str) -> int | None:
    match = re.search(r"Ran\s+(\d+)\s+tests?", text)
    return int(match.group(1)) if match else None


def run_and_record(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": command,
        "cwd": str(cwd),
        "return_code": completed.returncode,
        "log": str(log_path),
    }


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve()
    build = args.build_root.resolve()
    output = args.output.resolve()
    if not source.is_dir() or not build.is_dir():
        raise ValueError("source and build roots must be directories")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing report: {output}")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    ctest_log = output.with_name(output.stem + "_ctest.log")
    python_log = output.with_name(output.stem + "_python.log")
    ctest = run_and_record(
        [
            "ctest",
            "--test-dir",
            str(build),
            "--output-on-failure",
            "--no-tests=error",
            "-j",
            str(args.jobs),
        ],
        cwd=source,
        environment=environment,
        log_path=ctest_log,
    )
    python = run_and_record(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "validation/gpu_em/tests",
            "-p",
            "test_*.py",
        ],
        cwd=source,
        environment=environment,
        log_path=python_log,
    )
    ctest_text = ctest_log.read_text(encoding="utf-8")
    python_text = python_log.read_text(encoding="utf-8")
    ctest_count = parse_ctest_count(ctest_text)
    python_count = parse_unittest_count(python_text)
    ctest.update(
        {
            "test_count": ctest_count,
            "minimum_test_count": args.minimum_ctest_count,
            "passed": bool(
                ctest["return_code"] == 0
                and ctest_count is not None
                and ctest_count >= args.minimum_ctest_count
            ),
        }
    )
    python.update(
        {
            "test_count": python_count,
            "minimum_test_count": args.minimum_python_count,
            "passed": bool(
                python["return_code"] == 0
                and python_count is not None
                and python_count >= args.minimum_python_count
            ),
        }
    )
    passed = ctest["passed"] and python["passed"]
    report = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "environment": {
            "FLUPRO": environment.get("FLUPRO"),
            "FLUPRO_is_directory": bool(
                environment.get("FLUPRO")
                and Path(environment["FLUPRO"]).is_dir()
            ),
        },
        "ctest": ctest,
        "python_unittest": python,
    }
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if passed or not args.require_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"final test acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
