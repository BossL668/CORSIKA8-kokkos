#!/usr/bin/env python3
"""Run one isolated diagnostic with a 4 GiB available-memory floor.

Only the new child's process group may be terminated. Existing production
processes, services and the IDE are never signalled. Writes fresh logs and a
small summary, not an indefinitely growing in-memory monitor trace.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--rss-limit-gib", type=float, default=4)
    parser.add_argument("--sample-resources", action="store_true",
                        help="Append bounded-rate CPU/RSS/device-wide GPU telemetry")
    parser.add_argument("--sample-threads", action="store_true",
                        help="Also record per-thread CPU seconds and names (requires --sample-resources)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.sample_threads and not args.sample_resources:
        parser.error('--sample-threads requires --sample-resources')
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not all(math.isfinite(x) and x > 0 for x in (args.timeout, args.rss_limit_gib)):
        parser.error("requires a command and positive finite limits")
    args.output.mkdir(parents=True, exist_ok=False)
    floor = 4 * 1024**3
    minimum = psutil.virtual_memory().available
    report = dict(command=command, cwd=os.getcwd(), memory_floor_bytes=floor,
                  minimum_available_bytes=minimum, peak_tree_rss_bytes=0,
                  failure=None, returncode=None)
    start = time.monotonic()
    child = None
    telemetry = None
    next_sample = 0.
    try:
        if minimum < floor:
            report["failure"] = "less than 4 GiB available before starting"
        else:
            with (args.output / "command.log").open("xb") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                report["child_pid"] = child.pid
                root = psutil.Process(child.pid)
                if args.sample_resources:
                    telemetry = (args.output / "resources.jsonl").open("x")
                while child.poll() is None:
                    available = psutil.virtual_memory().available
                    report["minimum_available_bytes"] = min(report["minimum_available_bytes"], available)
                    try:
                        members = [root] + root.children(recursive=True)
                    except psutil.NoSuchProcess:
                        members = []
                    rss = 0
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    report["peak_tree_rss_bytes"] = max(report["peak_tree_rss_bytes"], rss)
                    elapsed = time.monotonic() - start
                    if telemetry is not None and elapsed >= next_sample:
                        sample = dict(elapsed_s=elapsed, rss_bytes=rss,
                                      available_bytes=available)
                        try:
                            sample.update(cpu_seconds=sum(root.cpu_times()[:2]),
                                          threads=root.num_threads())
                        except psutil.NoSuchProcess:
                            pass
                        if args.sample_threads:
                            try:
                                sample['thread_cpu_seconds'] = {
                                    str(t.id): t.user_time + t.system_time for t in root.threads()}
                                sample['thread_names'] = {}
                                for tid in sample['thread_cpu_seconds']:
                                    try:
                                        sample['thread_names'][tid] = Path(
                                            f'/proc/{child.pid}/task/{tid}/comm').read_text().strip()
                                    except OSError:
                                        pass
                            except psutil.NoSuchProcess:
                                pass
                        smi = shutil.which("nvidia-smi")
                        if smi is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
                            smi = "/usr/lib/wsl/lib/nvidia-smi"
                        if smi:
                            try:
                                raw = subprocess.check_output([smi, "--id=0",
                                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True, timeout=2)
                                used, total, util = map(float, raw.strip().split(","))
                                sample.update(device_used_mib=used, device_total_mib=total,
                                              device_util_percent=util)
                            except (OSError, ValueError, subprocess.SubprocessError):
                                sample["gpu_sample_unavailable"] = True
                        telemetry.write(json.dumps(sample) + "\n")
                        telemetry.flush()
                        next_sample = time.monotonic() - start + 1.
                    if available < floor:
                        report["failure"] = "available memory fell below 4 GiB"
                    elif rss > args.rss_limit_gib * 1024**3:
                        report["failure"] = "diagnostic process tree RSS limit"
                    elif time.monotonic() - start > args.timeout:
                        report["failure"] = "diagnostic timeout"
                    if report["failure"]:
                        break
                    time.sleep(.1)
    except BaseException as error:
        report["failure"] = str(error) or type(error).__name__
        raise
    finally:
        if telemetry is not None:
            telemetry.close()
        if child is not None:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            report["returncode"] = child.wait()
        report["elapsed_s"] = time.monotonic() - start
        report["pass"] = report["returncode"] == 0 and report["failure"] is None
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
