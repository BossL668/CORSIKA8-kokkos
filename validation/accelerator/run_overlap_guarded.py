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
import threading
import time

import psutil


def read_foreign_workflow_file(path):
    """Read bounded workflow-specific prefixes, never expand them into OS argv."""
    with Path(path).open(encoding='utf-8') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('foreign-workflow configuration exceeds 1 MiB')
    spec = json.loads(raw)
    if isinstance(spec, dict):
        if 'prefixes' in spec:
            prefixes = spec['prefixes']
        else:
            isolation = spec.get('isolation', {})
            if not isinstance(isolation, dict):
                raise ValueError('foreign-workflow isolation must be an object')
            prefixes = isolation.get('wait_workflow_path')
    else:
        prefixes = spec
    if not isinstance(prefixes, list) or not 1 <= len(prefixes) <= 64:
        raise ValueError('foreign-workflow file requires 1..64 absolute prefixes')
    for prefix in prefixes:
        if (not isinstance(prefix, str) or not 16 <= len(prefix) <= 4096 or
                not Path(prefix).is_absolute() or '\x00' in prefix or
                prefix.rstrip('/') in ('/', '/home', '/tmp', '/data', '/mnt',
                                       str(Path.home()))):
            raise ValueError('foreign-workflow prefixes must be long, specific absolute paths')
    return list(dict.fromkeys(prefixes))


def foreign_workflow_pids(prefixes, child_pid=None):
    """Match real argv payloads, excluding this guard's ancestry and child tree.

    AccessDenied, disappearing processes and other uncertain /proc observations
    propagate to the caller: they are not evidence of an idle machine. A known
    zombie has no executing workload. No bash/python/process-name exemption is
    used; a script path inside a shell -c argument is intentionally inspected.
    """
    if not prefixes:
        return []
    guard_pid = os.getpid()
    guard = psutil.Process(guard_pid)
    excluded = {guard_pid, *(p.pid for p in guard.parents())}
    own_roots = {guard_pid}
    if child_pid is not None:
        child = psutil.Process(child_pid)
        own_roots.add(child_pid)
        excluded.add(child_pid)
        excluded.update(p.pid for p in child.children(recursive=True))
    found = []
    for process in psutil.process_iter():
        if process.pid in excluded:
            continue
        if process.status() == psutil.STATUS_ZOMBIE:
            continue
        command = process.cmdline()
        if not all(isinstance(argument, str) for argument in command):
            raise ValueError('uncertain process command-line observation')
        if any(prefix in argument for argument in command for prefix in prefixes):
            # A descendant may have appeared after the first child snapshot.
            # Exclude only our OWN roots here, not every common ancestor/PID 1.
            if own_roots.intersection(p.pid for p in process.parents()):
                continue
            found.append(process.pid)
    return sorted(set(found))


def record_foreign_workflow_interference(report, output, elapsed, *, pids=(),
                                         error=None, starting=False):
    """Invalidate timing without signalling an already running physical event."""
    report['foreign_workflow_interference'] = True
    report['performance_valid'] = False
    merged = sorted(set(report['foreign_workflow_pids']) | set(pids))
    report['foreign_workflow_pid_observations'] += len(pids)
    report['foreign_workflow_pids_truncated'] |= len(merged) > 256
    report['foreign_workflow_pids'] = merged[:256]
    if error is not None:
        report['foreign_workflow_observation_failures'] += 1
        if len(report['foreign_workflow_error_samples']) < 16:
            report['foreign_workflow_error_samples'].append(
                dict(elapsed_s=elapsed, before_start=starting, error=str(error)))
    if starting:
        report['failure'] = ('foreign workflow observation uncertain before starting'
                             if error is not None else 'foreign workflow before starting')
    # This marker is sticky, including uncertainty or interference that has
    # disappeared by the final sample. No unbounded argv/process history is kept.
    marker = {key: value for key, value in report.items()
              if key.startswith('foreign_workflow') or key == 'performance_valid'}
    marker.update(schema=1, elapsed_s=elapsed, before_start=starting,
                  physical_event_killed_for_cpu_interference=False)
    try:
        (Path(output) / 'KNOWN_INTERFERENCE.json').write_text(
            json.dumps(marker, indent=2, allow_nan=False) + '\n')
    except OSError as write_error:
        # A report I/O problem is also not permission to kill valid physics.
        report['foreign_workflow_observation_failures'] += 1
        if len(report['foreign_workflow_error_samples']) < 16:
            report['foreign_workflow_error_samples'].append(dict(
                elapsed_s=elapsed, error='interference marker write failed: ' + str(write_error)))


def observe_foreign_workflows(report, prefixes, output, elapsed, *, child_pid=None,
                              starting=False):
    try:
        pids = foreign_workflow_pids(prefixes, child_pid)
    except (OSError, ValueError, psutil.Error) as error:
        record_foreign_workflow_interference(
            report, output, elapsed, error=error, starting=starting)
        return None
    report['foreign_workflow_checks'] += 1
    if pids:
        record_foreign_workflow_interference(
            report, output, elapsed, pids=pids, starting=starting)
    return pids


def compute_gpu_pids():
    smi=shutil.which('nvidia-smi')
    if smi is None and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        smi='/usr/lib/wsl/lib/nvidia-smi'
    if smi is None:raise FileNotFoundError('nvidia-smi unavailable')
    result=subprocess.check_output([smi,'--id=0','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=3)
    return {int(line.strip()) for line in result.splitlines() if line.strip()}


def observe_running_gpu(report, owned, elapsed):
    """Unknown telemetry is not an idle GPU, nor evidence of simulator failure.

    Preserve the running simulation on a transient external query failure, but
    permanently invalidate strict exclusive-GPU timing evidence for that run.
    Memory/RSS/timeout checks remain active, and real foreign work still fails.
    Keep only a bounded error sample; the counter is the authoritative total.
    """
    try:
        occupied = compute_gpu_pids()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report['gpu_observation_failures'] += 1
        if len(report['gpu_observation_error_samples']) < 16:
            report['gpu_observation_error_samples'].append(
                dict(elapsed_s=elapsed, error=str(error)))
        return None
    report['gpu_checks'] += 1
    foreign = occupied-owned
    live, unresolved = set(), set()
    for pid in foreign:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                unresolved.add(pid)
            else:
                live.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unresolved.add(pid)
    # WSL's NVML can briefly report a predecessor after wait() has reaped it.
    # A PID absent from this namespace is NOT proof of either a competing
    # Linux job or an idle GPU (it could also be host/namespace work). Keep the
    # simulation, mark the timing evidence unknown, never silently count it
    # as exclusive. A verified live competitor still stops only our child.
    if unresolved:
        report['gpu_observation_failures'] += 1
        if len(report['gpu_observation_error_samples']) < 16:
            report['gpu_observation_error_samples'].append(dict(
                elapsed_s=elapsed, error='NVML PID has no verifiable live process in this namespace',
                unresolved_gpu_pids=sorted(unresolved)))
    return live if live else (None if unresolved else set())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--rss-limit-gib", type=float, default=4)
    parser.add_argument("--sample-resources", action="store_true",
                        help="Append bounded-rate CPU/RSS/device-wide GPU telemetry")
    parser.add_argument("--sample-threads", action="store_true",
                        help="Also record per-thread CPU seconds and names (requires --sample-resources)")
    parser.add_argument('--require-exclusive-gpu',action='store_true',
                        help='Reject foreign GPU work; query failures before launch reject launch, during simulation invalidate timing without killing it; never signal foreign processes')
    parser.add_argument('--foreign-workflow-file', type=Path,
                        help='JSON workflow-prefix file: refuse occupied/uncertain startup; later CPU interference invalidates timing without stopping the event')
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
    if args.require_exclusive_gpu:
        report.update(exclusive_gpu_required=True,foreign_gpu_pids=[],gpu_checks=0,
                      performance_valid=False,gpu_observation_failures=0,
                      gpu_observation_error_samples=[])
    if args.foreign_workflow_file is not None:
        report.update(foreign_workflow_file=str(args.foreign_workflow_file),
                      foreign_workflow_checks=0, foreign_workflow_pids=[],
                      foreign_workflow_pid_observations=0,
                      foreign_workflow_pids_truncated=False,
                      foreign_workflow_observation_failures=0,
                      foreign_workflow_error_samples=[],
                      foreign_workflow_interference=False, performance_valid=False)
    start = time.monotonic()
    child = None
    telemetry = None
    next_sample = 0.
    next_gpu_check=0.
    next_workflow_check=0.
    workflow_prefixes = None
    waiter = None
    process_started = None
    process_finished = []
    try:
        if args.foreign_workflow_file is not None:
            try:
                workflow_prefixes = read_foreign_workflow_file(args.foreign_workflow_file)
            except (OSError, ValueError) as error:
                record_foreign_workflow_interference(
                    report, args.output, 0., error=error, starting=True)
            else:
                observe_foreign_workflows(report, workflow_prefixes, args.output, 0.,
                                          starting=True)
        if args.require_exclusive_gpu:
            occupied=compute_gpu_pids();report['gpu_checks']+=1
            if occupied:
                report['foreign_gpu_pids']=sorted(occupied)
                report['failure']='foreign GPU work before starting'
        if minimum < floor:
            report["failure"] = "less than 4 GiB available before starting"
        if report['failure'] is None:
            with (args.output / "command.log").open("xb") as log:
                process_started = time.monotonic()
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                # Resolve /proc before the waiter reaps a very short-lived child.
                root = psutil.Process(child.pid)
                # Independently timestamp wait(): slow external NVML queries must
                # not count as extra shower runtime after the child has exited.
                def reap():
                    child.wait()
                    process_finished.append(time.monotonic())
                waiter = threading.Thread(target=reap, daemon=True)
                waiter.start()
                report["child_pid"] = child.pid
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
                    if workflow_prefixes is not None and elapsed >= next_workflow_check:
                        observe_foreign_workflows(report, workflow_prefixes, args.output,
                                                  elapsed, child_pid=child.pid)
                        # No failure/break/signal on workflow matches or /proc
                        # uncertainty after launch; memory/time guards still apply.
                        next_workflow_check=time.monotonic()-start+5.
                    if args.require_exclusive_gpu and elapsed>=next_gpu_check:
                        foreign=observe_running_gpu(report,
                            {child.pid}|{p.pid for p in members},elapsed)
                        if foreign:
                            report['foreign_gpu_pids']=sorted(set(report['foreign_gpu_pids'])|foreign)
                            report['failure']='foreign GPU work during diagnostic'
                            break
                        next_gpu_check=time.monotonic()-start+5.
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
                                sample['thread_affinities'] = {}
                                for tid in sample['thread_cpu_seconds']:
                                    try:
                                        sample['thread_affinities'][tid] = sorted(os.sched_getaffinity(int(tid)))
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
            if waiter is not None:
                waiter.join()
                report['process_wall_s'] = process_finished[0] - process_started
                report['process_wall_semantics'] = 'Popen start to independent wait() completion; excludes preflight and final telemetry delay'
        report["elapsed_s"] = time.monotonic() - start
        report["pass"] = report["returncode"] == 0 and report["failure"] is None
        if args.require_exclusive_gpu or args.foreign_workflow_file is not None:
            valid = report['pass']
            if args.require_exclusive_gpu:
                valid = valid and not report['foreign_gpu_pids'] and report['gpu_observation_failures']==0
            if args.foreign_workflow_file is not None:
                valid = (valid and not report['foreign_workflow_interference'] and
                         report['foreign_workflow_observation_failures']==0 and
                         report['foreign_workflow_checks']>0)
            report['performance_valid'] = bool(valid)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
