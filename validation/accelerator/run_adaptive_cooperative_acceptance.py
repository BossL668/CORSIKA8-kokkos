#!/usr/bin/env python3
"""Isolated adaptive scheduling gates; never installs binaries or changes physics cuts."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import yaml


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def check_coalesced_fallbacks(stats):
    c = stats['accelerator']['cooperative']
    assert c['specified_fallback_batch_limit'] == 4096
    queued = stats['deferred_cpu_fallbacks_queued']
    assert queued == stats['deferred_cpu_fallbacks_flushed'], 'uncommitted specified completion'
    assert 0 <= stats['maximum_deferred_cpu_fallback_batch'] <= 4096, 'unbounded deferred queue'
    assert stats['deferred_fallback_scalar_expansion_rounds'] == 0
    total = sum(stats['cpu_fallbacks_by_reason_name'].values())
    assert total == sum(stats['cpu_fallbacks_by_process'].values())
    assert total == queued + stats['cpu_generic_fallbacks'], 'fallback lost or double-counted'
    assert stats['cpu_completed_selected_losses'] == stats['cpu_completed_native_selection_replays']
    assert stats['cpu_completed_selected_losses'] == stats['cpu_fallbacks_by_reason_name'].get('native_selection_replay', 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("before", "after", "fixture", "output", "data", "libraries", "antennas"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=130)
    parser.add_argument("--timing-seeds", type=int, default=2)
    parser.add_argument("--policy-version", default="adaptive-v2")
    parser.add_argument("--before-has-policy", action="store_true")
    parser.add_argument("--high-energy-seeds", type=int, default=0)
    parser.add_argument("--sample-threads", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true",
                        help="Reject timing contaminated by another GPU process; never stop that process")
    parser.add_argument("--expected-affinity", help="Inclusive logical CPU range, e.g. 382-511")
    parser.add_argument("--wait-build-unit")
    args = parser.parse_args()
    if args.expected_affinity:
        first, last = map(int, args.expected_affinity.split('-'))
        assert sorted(os.sched_getaffinity(0)) == list(range(first, last+1))
        assert last-first+1 == args.threads
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never mix partial attempts
    tools = Path(__file__).resolve().parent
    env = dict(os.environ, CORSIKA_DATA=str(args.data), LD_LIBRARY_PATH=str(args.libraries),
               OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND="spread",
               OMP_PLACES="threads", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", OMP_THREAD_LIMIT=str(args.threads),
               FLUPRO=os.environ.get("FLUPRO", "/home/yuhanglu/fluka"))
    def status(phase, **extra):
        save(out/"STATUS.json", dict(phase=phase, updated_unix=time.time(), **extra))
    status("waiting-for-build", complete=False)
    if args.wait_build_unit:
        while subprocess.run(['systemctl','--user','is-active','--quiet',args.wait_build_unit]).returncode == 0:
            time.sleep(2)
        code = subprocess.check_output(['systemctl','--user','show',args.wait_build_unit,
            '-p','ExecMainStatus','--value'], text=True).strip()
        assert code == '0', 'isolated build failed'
    save(out / "CONFIG.json", dict(arguments={k: str(v) for k, v in vars(args).items()},
        affinity=sorted(os.sched_getaffinity(0)),
        hashes={str(b): hashlib.sha256(b.read_bytes()).hexdigest()
                for b in (args.before, args.after, args.fixture)}))

    def run(label, command, timeout=900):
        if args.require_exclusive_gpu:
            from wait_gpu_idle_then_exec import gpu_processes
            assert not gpu_processes(), 'GPU is occupied before isolated test '+label
        status(label, complete=False)
        record = dict(command=list(map(str, command)), complete=False)
        save(out / (label + ".json"), record)
        started = time.monotonic()
        peak = 0
        samples = []
        next_threads = 0.
        next_exclusivity = 0.
        foreign_gpu_pids = set()
        with (out / (label + ".log")).open("w") as log, \
             (out / (label + ".telemetry.jsonl")).open("w") as telemetry:
            process = subprocess.Popen(command, cwd=out, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            finished = []
            exited = threading.Event()
            def wait_for_exit():
                process.wait()
                finished.append(time.monotonic())
                exited.set()
            waiter = threading.Thread(target=wait_for_exit, daemon=True)
            waiter.start()
            try:
                while process.poll() is None:
                    try:
                        fields = Path("/proc/%d/stat" % process.pid).read_text().rsplit(")", 1)[1].split()
                    except FileNotFoundError:
                        break
                    rss = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
                    peak = max(peak, rss)
                    free = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                                    if x.startswith("MemAvailable:"))) * 1024
                    elapsed = time.monotonic() - started
                    cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                    gpu = subprocess.run(["nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10).stdout.strip()
                    sample = dict(elapsed_s=elapsed, cpu_s=cpu, rss_bytes=rss,
                                  available_bytes=free, gpu=gpu)
                    if args.require_exclusive_gpu and elapsed >= next_exclusivity:
                        next_exclusivity = elapsed+5.
                        foreign = set(gpu_processes())-{process.pid}
                        foreign_gpu_pids.update(foreign)
                        sample['foreign_gpu_pids'] = sorted(foreign)
                    if args.sample_threads and elapsed >= next_threads:
                        # Stream telemetry to disk; never retain thread traces in RAM.
                        next_threads = elapsed+5.
                        sample['threads'] = {}
                        for task in Path('/proc/%d/task' % process.pid).glob('*'):
                            try:
                                tf = (task/'stat').read_text().rsplit(')',1)[1].split()
                                allowed = sorted(os.sched_getaffinity(int(task.name)))
                                sample['threads'][task.name] = dict(
                                    name=(task/'comm').read_text().strip(),
                                    cpu_s=(int(tf[11])+int(tf[12]))/os.sysconf('SC_CLK_TCK'),
                                    last_cpu=int(tf[36]), affinity=allowed)
                                if args.expected_affinity:
                                    assert set(allowed) <= set(range(first,last+1)), 'thread escaped CPU binding'
                            except (FileNotFoundError, ProcessLookupError):
                                continue
                        sample['gpu_state_csv'] = subprocess.run(['nvidia-smi',
                            '--query-gpu=pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem',
                            '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, timeout=10).stdout.strip()
                    telemetry.write(json.dumps(sample) + "\n")
                    telemetry.flush()
                    samples.append((elapsed, cpu))
                    if rss > 24 * 2**30 or free < 32 * 2**30:
                        raise RuntimeError("memory safety stop")
                    if elapsed > timeout:
                        raise RuntimeError("timeout safety stop")
                    exited.wait(.5)
                code = process.wait()
                waiter.join()
                record.update(returncode=code, process_s=finished[0]-started,
                              peak_rss_bytes=peak)
                record['gpu_exclusivity_checked'] = args.require_exclusive_gpu
                record['foreign_gpu_pids'] = sorted(foreign_gpu_pids)
                record['performance_valid'] = not foreign_gpu_pids
                if len(samples) > 1:
                    record["mean_cpu_percent"] = 100*(samples[-1][1]-samples[0][1])/(samples[-1][0]-samples[0][0])
                    record["peak_cpu_percent"] = max(100*(b[1]-a[1])/(b[0]-a[0])
                                                     for a, b in zip(samples, samples[1:]))
                if code:
                    raise RuntimeError("nonzero exit; see " + label + ".log")
                if foreign_gpu_pids:
                    raise RuntimeError("completed test had competing GPU processes; exclude this timing")
                record["complete"] = True
            except BaseException as error:
                record["error"] = repr(error)
                status(label+"-failed", complete=False, error=repr(error))
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                save(out / (label + ".json"), record)
                raise
        save(out / (label + ".json"), record)
        print(label, record["process_s"], flush=True)
        return record

    # Only the explicitly added policy option may change the existing help.
    help_text = []
    for i, binary in enumerate((args.before, args.after)):
        text = subprocess.run(["c8_air_shower", "--help"], executable=str(binary),
                              env=env, capture_output=True, check=True, text=True).stdout
        (out / ("help-%d.txt" % i)).write_text(text)
        # Adding a CLI option shifts the source location of the citation log.
        # Normalize that location only, not help text, option order or defaults.
        help_text.append(re.sub(
            r"(?m)^\[corsika:info    \(c8_air_shower\.cpp:\d+\)\] (?=Please cite)",
            "[corsika:info (source line)] ", text))
    after = help_text[1] if args.before_has_policy else re.sub(
        r"(?m)^  --kokkos-cooperative-policy[^\n]*\n(?: {4,}\S[^\n]*\n)*", "", help_text[1])
    diff = "".join(difflib.unified_diff(help_text[0].splitlines(True), after.splitlines(True)))
    (out / "help-unexpected.diff").write_text(diff)
    assert not diff, "unexpected CLI change, inspect help-unexpected.diff"

    def command(binary, label, mode, energy, seed, count=1, small=False, particle=2212):
        cmd = [str(binary), "-p", str(particle), "-E", str(energy), "-z", "47", "-a", "180",
               "-s", str(seed), "-N", str(count), "--emthin", "1e-6",
               "--antenna-file", str(args.antennas), "--verbosity", "warn", "-f", str(out/label)]
        if mode == "proposal":
            return cmd + ["--em-backend", "proposal", "--radio-backend", "cpu"]
        execution = "cuda-openmp" if mode in ("adaptive", "legacy") else mode
        cmd += ["--em-backend", "kokkos-proposal", "--radio-backend", "kokkos",
                "--kokkos-execution", execution, "--kokkos-num-threads",
                "1" if mode == "cuda" else str(args.threads),
                "--gpu-memory-fraction", ".1" if small else ".7",
                "--gpu-min-batch", "16" if small else "4096"]
        if small:
            cmd += ["--gpu-resident-batch-limit", "4096"]
        if mode == "adaptive":
            cmd += ["--kokkos-cooperative-policy", "adaptive"]
        return cmd

    for mode in ("proposal", "cuda", "openmp"):
        for phase, binary in (("before", args.before), ("after", args.after)):
            label = phase + "-" + mode
            cmd = command(binary, label, mode, 1, 26091021, count=2, small=True, particle=22)
            if mode != "proposal":
                cmd += ["--cuda-replay-trace", str(out/(label+"-trace.csv"))]
            run(label, cmd)
        subprocess.run([sys.executable, str(tools/"compare_backend_build_outputs.py"),
            str(out/("before-"+mode)), str(out/("after-"+mode)),
            "--report", str(out/(mode+"-comparison.json"))], check=True, env=env)
        if mode != "proposal":
            assert (out/("before-"+mode+"-trace.csv")).read_bytes() == (out/("after-"+mode+"-trace.csv")).read_bytes()

    # Invalid policy combinations must fail BEFORE starting a shower.
    for mode in ("proposal", "openmp", "cuda"):
        cmd = command(args.after, "invalid-"+mode, mode, 1, 1, small=True, particle=22)
        cmd += ["--kokkos-cooperative-policy", "adaptive"]
        check = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        assert check.returncode != 0 and "Adaptive scheduling requires" in check.stdout+check.stderr
        assert not (out/("invalid-"+mode)).exists()
    run("adaptive-em-radio-N2", [str(args.fixture), str(args.data/"PROPOSAL"),
                               str(args.threads), "subshowers", ".2", "adaptive"])
    lifecycle = run("adaptive-N32", command(args.after, "adaptive-N32", "adaptive", 100, 26091021,
                                          count=32, small=True, particle=22), timeout=1800)
    lifecycle.update(process_complete=True, complete=False)
    save(out/"adaptive-N32.json", lifecycle)
    summary = yaml.safe_load((out/"adaptive-N32/gpu_em/summary.yaml").read_text())
    assert len(summary) == 32
    shard_storage = None
    for event in summary.values():
        assert event["complete"]
        c = event["statistics"]["accelerator"]["cooperative"]
        assert c["scheduling_policy"] == args.policy_version
        assert c["subshower_cuda_submissions"] == c["subshower_cuda_commits"]
        if args.policy_version in ('adaptive-v11-host-profile-shards', 'adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
            current = (c['host_profile_shards'], c['host_profile_shard_bytes'])
            assert 1 < current[0] <= 256 and 0 < current[1] <= 16 * 2**20
            assert shard_storage is None or current == shard_storage, 'profile shard storage grew between showers'
            shard_storage = current
        if args.policy_version in ('adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
            check_coalesced_fallbacks(event['statistics'])
    lifecycle["complete"] = True
    save(out/"adaptive-N32.json", lifecycle)
    save(out/"CORRECTNESS_GATES.json", dict(passed=True, physics_statistics_accepted=False,
        note="Fixed-seed single ends, exact traces, EM energy closure and lifecycle; not ensemble acceptance."))

    timing = []
    def event(mode, energy, seed, label):
        record = run(label, command(args.after, label, mode, energy, seed),
                     timeout=43200 if energy >= 100000000 else 1800)
        record.update(process_complete=True, complete=False)
        save(out/(label+".json"), record)
        data = yaml.safe_load((out/label/"gpu_em/summary.yaml").read_text())["shower_0"]
        assert data["complete"]
        s = data["statistics"]
        assert s["queue_overflows"] == s["profile"]["fixed_point_overflows"] == s["radio"]["fixed_point_overflows"] == 0
        a = s["accelerator"]
        event_timing = yaml.safe_load((out/label/"simulation_timing/summary.yaml").read_text())["shower_0"]
        assert event_timing["closed"]
        record["shower_s"] = event_timing["wall_time_ms"]/1000
        assert a["host_threads"] == args.threads and a["openmp"]
        assert bool(a["gpu"]) == (mode == "adaptive")
        if mode == "adaptive":
            c = a["cooperative"]
            assert c["scheduling_policy"] == args.policy_version
            assert c["subshower_cuda_submissions"] == c["subshower_cuda_commits"] > 0
            assert c["subshower_openmp_epochs"] > 0
            if args.policy_version in ('adaptive-v4-bounded-batching','adaptive-v5-work-quantum','adaptive-v6-tail-grain','adaptive-v7-gpu-continuation','adaptive-v8-bounded-continuation','adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                steps = waves = jobs = 0
                for endpoint in ('cuda','openmp'):
                    ep = c['adaptive'][endpoint]
                    jobs += sum(ep['job_input_histogram_floor_log2'])
                    for kind in ('photon','lepton'):
                        steps += ep[kind]['transport_records']
                        waves += ep[kind]['resident_wavefronts']
                assert steps == s['gpu_particles'] == s['profile']['steps'], 'physical step accounting'
                assert waves == s['resident_photon_wavefronts']+s['resident_lepton_wavefronts'], 'wave accounting'
                assert jobs == c['subshower_cuda_commits']+c['subshower_openmp_epochs'], 'batch histogram accounting'
                if args.policy_version in ('adaptive-v5-work-quantum','adaptive-v6-tail-grain','adaptive-v7-gpu-continuation','adaptive-v8-bounded-continuation','adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                    reasons=sum(sum(c['adaptive'][e][k]['completion_reasons'])
                                for e in ('cuda','openmp') for k in ('photon','lepton'))
                    assert reasons==jobs, 'exclusive completion reason accounting'
                    if args.policy_version in ('adaptive-v7-gpu-continuation','adaptive-v8-bounded-continuation','adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                        cap=2 if args.policy_version=='adaptive-v7-gpu-continuation' else 16
                        assert c['cuda_completion_packets']+c['subshower_cuda_autonomous_continuations']==c['subshower_cuda_commits']
                        assert c['subshower_cuda_autonomous_continuations']<=(cap-1)*c['cuda_completion_packets']
                        assert c['cuda_completion_mailbox_capacity']==cap
                        assert c['cuda_completion_buffer_delay_ms']>=c['cuda_result_service_delay_ms']
                        if cap==16:
                            assert sum(c['cuda_continuation_stops'])==c['cuda_completion_packets']
                            assert c['maximum_cuda_packet_calls']<=cap
                            assert c['cuda_completion_retention_budget_bytes']==64*2**20
                        if args.policy_version in ('adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                            assert c['subshower_cuda_foreground_packets']<=c['cuda_completion_packets']
                            assert c['subshower_cuda_foreground_continuations']<=c['subshower_cuda_autonomous_continuations']
                if args.policy_version in ('adaptive-v11-host-profile-shards', 'adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                    assert 1 < c['host_profile_shards'] <= 256
                    assert 0 < c['host_profile_shard_bytes'] <= 16 * 2**20
                if args.policy_version in ('adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                    check_coalesced_fallbacks(s)
        for algorithm in ("CoREAS", "ZHS"):
            parquet = pq.ParquetFile(out/label/algorithm/"observers.parquet")
            assert parquet.metadata.num_rows > 0
            for batch in parquet.iter_batches(columns=["Ex", "Ey", "Ez"], use_threads=False):
                assert all(np.isfinite(col.to_numpy()).all() for col in batch.columns)
        record.update(mode=mode, energy_GeV=energy, seed=seed, accelerator=a,
                      transport_records=s['gpu_particles'], complete=True)
        save(out/(label+".json"), record)
        return record
    for mode in ("openmp", "adaptive"):
        event(mode, 1000, 2026110000, "warmup-"+mode)
    for i in range(args.timing_seeds):
        seed = 2026110001+i
        for mode in (("openmp", "adaptive") if i%2 == 0 else ("adaptive", "openmp")):
            timing.append(event(mode, 100000, seed, "100TeV-%d-%s" % (seed, mode)))
            save(out/"TIMING_PARTIAL.json", timing)
    medians = {mode: statistics.median(r["process_s"] for r in timing if r["mode"] == mode)
               for mode in ("openmp", "adaptive")}
    autonomous_calls = sum(r['accelerator'].get('cooperative', {}).get(
        'subshower_cuda_autonomous_continuations', 0) for r in timing)
    save(out/"PERFORMANCE_RESULT.json", dict(complete=True, timing=timing, median_s=medians,
         openmp_over_adaptive=medians["openmp"]/medians["adaptive"],
         autonomous_gpu_calls=autonomous_calls,
         autonomous_gpu_path_exercised=autonomous_calls>0,
         note="Pilot only; no 500-event physics acceptance or production recommendation."))
    high = []
    for i in range(args.high_energy_seeds):
        seed = 2026110001+i
        # Start the requested dual test first; no concurrent CPU/GPU comparator.
        for mode in (('adaptive','openmp') if i%2 == 0 else ('openmp','adaptive')):
            high.append(event(mode, 100000000, seed, '100PeV-%d-%s' % (seed,mode)))
            save(out/'TIMING_100PEV_PARTIAL.json', high)
    status('complete-pilot-not-ensemble-acceptance', complete=True)


if __name__ == "__main__":
    main()
