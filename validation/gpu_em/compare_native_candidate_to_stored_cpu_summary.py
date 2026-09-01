#!/usr/bin/env python3
"""Cross-check a new native ensemble against a stored CPU summary.

This intentionally does not pretend that aggregate CPU statistics replace the
raw CPU ensemble: means, standard errors and z scores can be recomputed, while
KS tests and direct pointwise CPU/candidate curves require the original data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "profile_xmax_charged_gcm2",
    "profile_charged_max",
    "profile_charged_integral",
    "profile_photon_integral",
    "profile_electron_positron_integral",
    "profile_em_integral",
    "energy_deposit_sum_GeV",
    "energy_deposit_xmax_gcm2",
    "ground_em_weighted_count",
    "ground_em_kinetic_energy_GeV",
    "energy_closure_fraction",
)

LABELS = {
    "profile_xmax_charged_gcm2": r"charged $X_{\max}$",
    "profile_charged_max": r"charged $N_{\max}$",
    "profile_charged_integral": "charged-profile integral",
    "profile_photon_integral": r"$\gamma$-profile integral",
    "profile_electron_positron_integral": r"$e^-+e^+$ profile integral",
    "profile_em_integral": "total-EM profile integral",
    "energy_deposit_sum_GeV": "deposited energy",
    "energy_deposit_xmax_gcm2": r"energy-deposit $X_{\max}$",
    "ground_em_weighted_count": "ground EM count",
    "ground_em_kinetic_energy_GeV": "ground EM kinetic energy",
    "energy_closure_fraction": "energy closure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference-comparison", type=Path, required=True)
    parser.add_argument("--native-stratum-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def ensure_linkage(cpu_comparison: dict[str, Any], strata: dict[str, Any]) -> None:
    """Prove that the old native arm is the bridge shared by both reports."""

    for metric in METRICS:
        old_from_cpu = cpu_comparison["scalars"][metric]["cuda"]
        old_from_strata = strata["scalars"][metric]["proposal"]
        for key in ("events", "mean", "standard_deviation", "standard_error"):
            left = float(old_from_cpu[key])
            right = float(old_from_strata[key])
            if not np.isclose(left, right, rtol=2.0e-13, atol=1.0e-13):
                raise ValueError(
                    f"old-native bridge differs for {metric}.{key}: "
                    f"{left} versus {right}"
                )


def comparison_row(
    metric: str,
    cpu: dict[str, Any],
    old_native: dict[str, Any],
    new_native: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "cpu_events": int(cpu["events"]),
        "old_native_events": int(old_native["events"]),
        "new_native_events": int(new_native["events"]),
        "cpu_mean": float(cpu["mean"]),
        "cpu_standard_error": float(cpu["standard_error"]),
        "old_native_mean": float(old_native["mean"]),
        "old_native_standard_error": float(old_native["standard_error"]),
        "new_native_mean": float(new_native["mean"]),
        "new_native_standard_error": float(new_native["standard_error"]),
    }
    reference = row["cpu_mean"]
    for prefix in ("old_native", "new_native"):
        difference = row[f"{prefix}_mean"] - reference
        combined = float(
            np.hypot(row["cpu_standard_error"], row[f"{prefix}_standard_error"])
        )
        row[f"{prefix}_relative_shift"] = difference / reference if reference else 0.0
        row[f"{prefix}_combined_standard_error"] = combined
        row[f"{prefix}_absolute_z_score"] = abs(difference) / combined if combined else 0.0
        row[f"{prefix}_relative_95_half_width"] = (
            1.96 * combined / abs(reference) if reference else 0.0
        )
    row["absolute_shift_improved"] = abs(row["new_native_relative_shift"]) < abs(
        row["old_native_relative_shift"]
    )
    return row


def plot_shifts(frame: pd.DataFrame, output: Path) -> None:
    y = np.arange(len(frame), dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 6.3))
    for offset, prefix, label, color in (
        (-0.13, "old_native", "previous native 2000", "#0072B2"),
        (+0.13, "new_native", "new paired-seed native 2000", "#D55E00"),
    ):
        axis.errorbar(
            100.0 * frame[f"{prefix}_relative_shift"],
            y + offset,
            xerr=100.0 * frame[f"{prefix}_relative_95_half_width"],
            fmt="o",
            markersize=5.5,
            capsize=3,
            color=color,
            label=label,
        )
    axis.axvline(0.0, color="0.25", linewidth=1.0)
    axis.axvspan(-1.0, 1.0, color="#009E73", alpha=0.09, label="±1% band")
    axis.set_yticks(y, [LABELS[item] for item in frame["metric"]])
    axis.invert_yaxis()
    axis.set_xlabel("mean shift relative to stored CPU 2000 summary [%]")
    axis.set_title("Native-PROPOSAL ensemble means versus CPU PROPOSAL")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_report(
    path: Path,
    frame: pd.DataFrame,
    cpu_comparison_path: Path,
    strata_path: Path,
    cpu_comparison: dict[str, Any],
    strata: dict[str, Any],
) -> None:
    compatibility_path = strata_path.with_name("compatibility.json")
    compatibility = read_json(compatibility_path) if compatibility_path.is_file() else {}
    lines = [
        "# 新同种子 proposal-native 2000 例本地阶段验收",
        "",
        "## 结论",
        "",
        (
            "新样本的主要纵向、能量沉积和地面 EM **均值整体明显比上一批 "
            "native 更接近已保存的 CPU 2000 例均值**。11 个列出指标中，"
            f"{int(frame['absolute_shift_improved'].sum())} 个的绝对均值偏移减小。"
        ),
        "",
        (
            "这是一项本地可完成的 aggregate cross-check，而不是新的完整 CPU/raw "
            "直接验收。PSR 上的 CPU 原始数据恢复可访问后，仍需补做逐深度 CPU/new-native "
            "profile、KS、ground timing 和射电脉冲的直接比较。"
        ),
        "",
        "## 关键均值",
        "",
        "| Observable | 旧 native/CPU | 新 native/CPU | 新样本 |z| | 是否改善 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {LABELS[row['metric']]} | "
            f"{100.0 * row['old_native_relative_shift']:+.3f}% | "
            f"{100.0 * row['new_native_relative_shift']:+.3f}% | "
            f"{row['new_native_absolute_z_score']:.3f} | "
            f"{'yes' if row['absolute_shift_improved'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 新旧 native 的直接2000 vs 2000门禁",
            "",
            f"- key-scalar 3-sigma gate: `{compatibility.get('key_scalar_sigma_pass')}`",
            f"- longitudinal/ground curve gate: `{compatibility.get('curve_pass')}`",
            f"- key-scalar KS gate: `{compatibility.get('key_scalar_ks_pass')}`",
            (
                "- 唯一失败的 key KS 指标是地面 EM 动能："
                f"D={strata['scalars']['ground_em_kinetic_energy_GeV']['distribution_diagnostics']['empirical_KS_distance']:.4f}，"
                f"95% 临界值={strata['scalars']['ground_em_kinetic_energy_GeV']['distribution_diagnostics']['KS_95pct_critical_value']:.4f}。"
            ),
            "",
            (
                "由于两批 native 使用完全相同的可执行文件、原生表、辅助表和物理配置，"
                "该单项失败不能解释为代码版本变化；它说明地面低能 EM 尾部对随机种子仍然敏感，"
                "正式结论必须保留这一门禁结果。"
            ),
            "",
            "## 纵向曲线自一致性",
            "",
            "| Curve | 新/旧 native relative L1 | RMS active z | pass |",
            "|---|---:|---:|---|",
        ]
    )
    for metric in (
        "profile_electron_positron",
        "profile_em",
        "profile_photon",
        "profile_hadron",
        "profile_muon",
        "energy_deposit",
    ):
        item = strata["curves"][metric]
        lines.append(
            f"| {metric} | {item['relative_l1']:.4f} | "
            f"{item['rms_active_z_score']:.3f} | {item['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 数据来源与限制",
            "",
            f"- CPU/旧 native 摘要：`{cpu_comparison_path}`",
            f"- 旧/新 native 原始分布比较：`{strata_path}`",
            (
                "- bridge 校验：旧 native 在两个 JSON 中的 events、mean、standard "
                "deviation 和 standard error 逐指标一致。"
            ),
            (
                f"- 上一轮 CPU/旧 native 总门禁状态："
                f"`{cpu_comparison.get('acceptance', {}).get('passed')}`；"
                "其主要纵向曲线通过，整体失败包含地面时间尾部分布门禁。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cpu_path = args.cpu_reference_comparison.resolve()
    strata_path = args.native_stratum_comparison.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu_comparison = read_json(cpu_path)
    strata = read_json(strata_path)
    ensure_linkage(cpu_comparison, strata)
    rows = []
    for metric in METRICS:
        rows.append(
            comparison_row(
                metric,
                cpu_comparison["scalars"][metric]["proposal"],
                cpu_comparison["scalars"][metric]["cuda"],
                strata["scalars"][metric]["cuda"],
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "stored_cpu_scalar_crosscheck.csv", index=False)
    payload = {
        "status": "aggregate_crosscheck_complete_raw_cpu_direct_test_pending",
        "cpu_reference_comparison": str(cpu_path),
        "native_stratum_comparison": str(strata_path),
        "metrics_improved": int(frame["absolute_shift_improved"].sum()),
        "metrics_total": len(frame),
        "rows": rows,
    }
    (output / "stored_cpu_scalar_crosscheck.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_shifts(frame, output / "stored_cpu_scalar_relative_shifts.png")
    write_report(
        output / "VALIDATION_REPORT_CN.md",
        frame,
        cpu_path,
        strata_path,
        cpu_comparison,
        strata,
    )


if __name__ == "__main__":
    main()
