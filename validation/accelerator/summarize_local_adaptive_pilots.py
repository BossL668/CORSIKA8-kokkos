#!/usr/bin/env python3
"""Summarize completed local adaptive pilots without loading shower arrays.

Optionally follow an existing systemd test service. This observer never starts,
stops, resumes, or otherwise modifies simulation jobs.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import yaml


MODES = ("single-cuda", "legacy20", "adaptive20")
LABELS = {"single-cuda": "单 CUDA", "legacy20": "原双端20线程",
          "adaptive20": "自适应双端20线程"}


def write_atomic(path, contents):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def telemetry(path):
    first = last = None
    gpu_integral = duration = peak_gpu = peak_rss = 0.0
    previous_gpu = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            peak_rss = max(peak_rss, row.get("rss_bytes", 0))
            peak_gpu = max(peak_gpu, row.get("device_used_mib", 0))
            if "cpu_seconds" in row:
                if first is None:
                    first = row
                last = row
            if "device_util_percent" in row:
                if previous_gpu is not None:
                    dt = row["elapsed_s"] - previous_gpu["elapsed_s"]
                    gpu_integral += dt * (row["device_util_percent"] +
                                           previous_gpu["device_util_percent"]) / 2
                    duration += dt
                previous_gpu = row
    elapsed = last["elapsed_s"] - first["elapsed_s"] if first and last else 0
    return dict(average_logical_cores=(last["cpu_seconds"] - first["cpu_seconds"]) /
                elapsed if elapsed else None,
                sampled_gpu_util_percent=gpu_integral / duration if duration else None,
                sampled_device_peak_mib=peak_gpu, sampled_rss_peak_bytes=peak_rss)


def summarize(root):
    status = json.loads((root / "STATUS.json").read_text())
    records = []
    for record in status["records"]:
        if record.get("warmup") or not record.get("complete"):
            continue
        label = f"{record['family']}-{record['seed']}-{record['mode']}"
        stats = yaml.safe_load((root / label / "gpu_em/summary.yaml").read_text())[
            "shower_0"]["statistics"]
        row = {k: record[k] for k in ("family", "seed", "mode", "process_s", "shower_s")}
        row.update(telemetry(root / (label + "-guard/resources.jsonl")))
        row["peak_rss_bytes"] = record["monitor"]["peak_tree_rss_bytes"]
        row["minimum_available_bytes"] = record["monitor"]["minimum_available_bytes"]
        row["proposal_table_sha256"] = stats["proposal_native"]["table_sha256"]
        row["proposal_aux_sha256"] = stats["proposal_native"]["aux_sha256"]
        row["proposal_cache_all_hit"] = stats["proposal_native"]["proposal_cache_all_hit"]
        row["inverse_failures"] = stats["proposal_native"]["inverse_failures"]
        row["energy_ledger"] = stats["energy_ledger"]
        row["cooperative"] = stats["accelerator"].get("cooperative", {})
        records.append(row)
    groups = {}
    for family in ("Fe100TeV", "proton100PeV"):
        selected = [r for r in records if r["family"] == family]
        # Only common seeds enter three-mode aggregates. An unfinished slower
        # run must never silently select the faster completed subset.
        seeds = set.intersection(*({r["seed"] for r in selected if r["mode"] == mode}
                                   for mode in MODES))
        complete = [r for r in selected if r["seed"] in seeds]
        median = {mode: statistics.median(r["process_s"] for r in complete
                                         if r["mode"] == mode) for mode in MODES} if seeds else {}
        pairs = []
        for baseline in ("single-cuda", "legacy20"):
            common = ({r["seed"] for r in selected if r["mode"] == baseline} &
                      {r["seed"] for r in selected if r["mode"] == "adaptive20"})
            for seed in sorted(common):
                base = next(r for r in selected if r["seed"] == seed and r["mode"] == baseline)
                adapt = next(r for r in selected if r["seed"] == seed and r["mode"] == "adaptive20")
                pairs.append(dict(seed=seed, baseline=baseline,
                                  speed_ratio=base["process_s"] / adapt["process_s"],
                                  time_reduction_percent=100 * (1-adapt["process_s"]/base["process_s"])))
        groups[family] = dict(completed_counts={mode: sum(r["mode"] == mode for r in selected)
                                                for mode in MODES},
                              three_mode_common_seeds=sorted(seeds), median_process_s=median,
                              paired_comparisons=pairs)
    report = dict(phase=status["phase"], complete=status["complete"],
                  error=status.get("error"), groups=groups, records=records,
                  interpretation="pilot timing; no ensemble physics or hardware-kernel-overlap claim")
    write_atomic(root / "PERFORMANCE_SUMMARY.json",
                 json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# RTX4060＋20线程：自适应调度性能试跑", "",
             f"当前阶段：`{status['phase']}`。流水线完成：{status['complete']}。", "",
             "本表只计正式、完整退出的事件，不计预热。旧生产队列仍暂停；本报告不恢复任务。", "",
             "三种模式使用同一冻结二进制、同一物理输入与种子。双端动态调度不保证同一棵 shower 树，",
             "因此单个种子的时间比只能作为描述性结果，不能单独证明算法加速。", "",
             "## 参数与计时口径", "",
             "- Fe56：总能量100TeV，垂直，emthin=1e-6；其他设置沿用原Fe manifest。",
             "- 质子：100PeV（1e8GeV），theta=47°、phi=180°，emthin=1e-6；沿用高能命令。",
             "- 不改变max-weight、cut或物理表；完整CoREAS/ZHS；双端20线程，GPU显存预算70%。",
             "- 进程时间包含启动、上传、shower与输出，以及守护采样的少量误差；shower时间来自SimulationTiming。",
             "- CPU核数由进程累计CPU时间差/墙钟时间差得到，包含启动和输出阶段。",
             "- 显存与GPU利用率为设备级抽样，包含显示占用；不能当成精确kernel占用或实际重叠时间。", ""]
    for family, title in (("Fe100TeV", "100TeV铁核"), ("proton100PeV", "100PeV质子")):
        lines += ["## " + title, "", "| seed | 模式 | 进程时间(s) | shower时间(s) | 平均逻辑核数 | 峰值显存(MiB) | 峰值RSS(GiB) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in sorted((r for r in records if r["family"] == family),
                          key=lambda r: (r["seed"], MODES.index(r["mode"]))):
            cores = row["average_logical_cores"]
            lines.append(f"| {row['seed']} | {LABELS[row['mode']]} | {row['process_s']:.3f} | "
                         f"{row['shower_s']:.3f} | {cores:.2f} | {row['sampled_device_peak_mib']:.0f} | "
                         f"{row['peak_rss_bytes']/2**30:.2f} |")
        group = groups[family]
        lines += ["", "三模式均完成的共同种子数：" + str(len(group["three_mode_common_seeds"])) + "。"]
        median = group["median_process_s"]
        if median:
            lines += ["", "共同种子的进程中位数：" + "；".join(
                f"{LABELS[mode]} {median[mode]:.3f}s" for mode in MODES) + "。",
                f"自适应相对单CUDA：时间变化{100*(median['adaptive20']/median['single-cuda']-1):+.1f}%"
                f"，描述性加速比{median['single-cuda']/median['adaptive20']:.3f}。"]
        else:
            lines += ["暂不汇总三模式中位数，以免未完成事件造成选择偏差。"]
        lines += [""]
    lines += ["## 正确性范围和限制", "",
              "旧/新单端固定seed数组与decision trace、20线程自适应N=32及纯EM＋射电N=2门禁已通过。",
              "正式事件完成状态、提交/回收数量、队列/定点溢出、波形有限值已由运行器检查。",
              "Fe和proton包含强子过程，当前GPU energy_ledger标记complete_coverage=false；",
              "该局部账本不是完整强子事件能量守恒验收，不能把accepted=false忽略后宣布能量闭合通过，",
              "也不能把不完整账本的差额直接解释成新调度丢失了能量。仍需独立的完整账本/系综验收。", "",
              "Fe每模式5例、高能每模式1例均为性能试跑，不是500例物理统计验收。",
              "实际kernel重叠需要设备时间线；本次cooperative统计中的call-window交集不能代替它。", "",
              "完整命令、身份哈希和资源记录分别见每事件JSON、PROVENANCE.json与*-guard/resources.jsonl。", ""]
    if status.get("error"):
        lines += ["## 流水线错误", "", "```text", status["error"], "```", ""]
    write_atomic(root / "PERFORMANCE_REPORT_CN.md", "\n".join(lines))
    return status["complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--watch-service", help="Observe this existing user service; never mutate it")
    args = parser.parse_args()
    while True:
        done = summarize(args.output)
        if not args.watch_service or done:
            break
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                                 args.watch_service]).returncode == 0
        if not active:
            summarize(args.output)
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
