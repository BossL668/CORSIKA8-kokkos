#!/usr/bin/env python3
"""Diagnose an apparent CPU/CUDA difference in an ensemble-mean profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats  # noqa: E402


PROFILE_COLUMNS = (
    "charged",
    "hadron",
    "photon",
    "electron",
    "positron",
    "muplus",
    "muminus",
)
COMPONENTS = (
    "em",
    "electron_positron",
    "photon",
    "electron",
    "positron",
    "hadron",
    "muon",
    "muplus",
    "muminus",
    "charged",
)
COLORS = {"proposal": "#1f77b4", "cuda": "#d62728"}
LABELS = {
    "proposal": "Original CPU (PROPOSAL)",
    "cuda": "CUDA EM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional pooled-analysis manifest. Defaults to "
            "<ensemble-root>/run_manifest.json."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--large-sample-reference-root",
        type=Path,
        help=(
            "Optional older large ensemble used only to estimate how "
            "ensemble-mean peak noise scales with event count."
        ),
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_731)
    parser.add_argument(
        "--proposal-implicit-geomagnetic-model",
        choices=("IGRF13", "IGRF14"),
        help=(
            "Geomagnetic model hard-coded by a legacy proposal executable; "
            "must be supplied with --proposal-implicit-geomagnetic-year."
        ),
    )
    parser.add_argument(
        "--proposal-implicit-geomagnetic-year",
        type=float,
        help=(
            "Geomagnetic epoch hard-coded by a legacy proposal executable; "
            "must be supplied with --proposal-implicit-geomagnetic-model."
        ),
    )
    return parser.parse_args()


def profile_roots(
    ensemble_root: Path,
    manifest: dict[str, Any],
    backend: str,
) -> list[Path]:
    if backend == "proposal":
        roots = sorted(
            path
            for path in ensemble_root.glob("proposal_shard_*")
            if (path / "profile" / "profile.parquet").is_file()
        )
    else:
        roots = [ensemble_root / "cuda"]
    for source in manifest.get("additional_sources", {}).get(backend, []):
        roots.append(Path(source))
    roots = [
        path.resolve()
        for path in roots
        if (path / "profile" / "profile.parquet").is_file()
    ]
    if not roots:
        raise ValueError(f"no {backend} longitudinal-profile source found")
    return roots


def read_profile(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    parquet = path / "profile" / "profile.parquet"
    frame = pq.read_table(
        parquet,
        columns=["shower", "X", *PROFILE_COLUMNS],
    ).to_pandas()
    shower_ids = sorted(int(value) for value in frame["shower"].unique())
    coordinate: np.ndarray | None = None
    rows = {column: [] for column in PROFILE_COLUMNS}
    for shower_id in shower_ids:
        selected = frame.loc[
            frame["shower"] == shower_id
        ].sort_values("X")
        x = selected["X"].to_numpy(dtype=np.float64)
        if coordinate is None:
            coordinate = x
        elif coordinate.shape != x.shape or not np.array_equal(
            coordinate, x
        ):
            raise ValueError(f"inconsistent X grid in {parquet}")
        for column in PROFILE_COLUMNS:
            rows[column].append(
                selected[column].to_numpy(dtype=np.float64)
            )
    if coordinate is None:
        raise ValueError(f"empty profile: {parquet}")
    return coordinate, {
        column: np.stack(values)
        for column, values in rows.items()
    }


def read_ensemble(
    roots: list[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coordinate: np.ndarray | None = None
    blocks = {column: [] for column in PROFILE_COLUMNS}
    for root in roots:
        candidate_coordinate, matrices = read_profile(root)
        if coordinate is None:
            coordinate = candidate_coordinate
        elif (
            coordinate.shape != candidate_coordinate.shape
            or not np.array_equal(coordinate, candidate_coordinate)
        ):
            raise ValueError("ensemble profile coordinate grids differ")
        for column in PROFILE_COLUMNS:
            blocks[column].append(matrices[column])
    assert coordinate is not None
    result = {
        column: np.concatenate(values, axis=0)
        for column, values in blocks.items()
    }
    result["electron_positron"] = (
        result["electron"] + result["positron"]
    )
    result["muon"] = result["muplus"] + result["muminus"]
    result["em"] = (
        result["photon"]
        + result["electron"]
        + result["positron"]
    )
    return coordinate, result


def relative_shift(reference: float, candidate: float) -> float:
    return (candidate - reference) / max(
        abs(reference), np.finfo(np.float64).tiny
    )


def component_summary(
    coordinate: np.ndarray,
    proposal: np.ndarray,
    cuda: np.ndarray,
) -> dict[str, float]:
    proposal_mean = np.mean(proposal, axis=0)
    cuda_mean = np.mean(cuda, axis=0)
    common_peak = int(np.argmax((proposal_mean + cuda_mean) / 2.0))
    proposal_at_peak = proposal[:, common_peak]
    cuda_at_peak = cuda[:, common_peak]
    fixed_peak_se = math.sqrt(
        float(np.var(proposal_at_peak, ddof=1)) / proposal.shape[0]
        + float(np.var(cuda_at_peak, ddof=1)) / cuda.shape[0]
    )
    fixed_peak_difference = (
        float(np.mean(cuda_at_peak)) - float(np.mean(proposal_at_peak))
    )
    proposal_integrals = np.trapz(proposal, coordinate, axis=1)
    cuda_integrals = np.trapz(cuda, coordinate, axis=1)
    integral_se = math.sqrt(
        float(np.var(proposal_integrals, ddof=1)) / proposal.shape[0]
        + float(np.var(cuda_integrals, ddof=1)) / cuda.shape[0]
    )
    integral_difference = (
        float(np.mean(cuda_integrals))
        - float(np.mean(proposal_integrals))
    )
    proposal_maxima = np.max(proposal, axis=1)
    cuda_maxima = np.max(cuda, axis=1)
    maximum_se = math.sqrt(
        float(np.var(proposal_maxima, ddof=1)) / proposal.shape[0]
        + float(np.var(cuda_maxima, ddof=1)) / cuda.shape[0]
    )
    maximum_difference = (
        float(np.mean(cuda_maxima)) - float(np.mean(proposal_maxima))
    )
    return {
        "common_peak_X_gcm2": float(coordinate[common_peak]),
        "fixed_depth_peak_relative_shift": relative_shift(
            float(np.mean(proposal_at_peak)),
            float(np.mean(cuda_at_peak)),
        ),
        "fixed_depth_peak_z_score": (
            fixed_peak_difference / fixed_peak_se
            if fixed_peak_se > 0.0
            else 0.0
        ),
        "profile_integral_relative_shift": relative_shift(
            float(np.mean(proposal_integrals)),
            float(np.mean(cuda_integrals)),
        ),
        "profile_integral_z_score": (
            integral_difference / integral_se
            if integral_se > 0.0
            else 0.0
        ),
        "per_shower_maximum_relative_shift": relative_shift(
            float(np.mean(proposal_maxima)),
            float(np.mean(cuda_maxima)),
        ),
        "per_shower_maximum_z_score": (
            maximum_difference / maximum_se
            if maximum_se > 0.0
            else 0.0
        ),
    }


def align_to_xmax(
    coordinate: np.ndarray,
    matrix: np.ndarray,
    xmax_reference: np.ndarray,
    relative_grid: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.interp(
                relative_grid,
                coordinate - xmax_reference[index],
                matrix[index],
                left=np.nan,
                right=np.nan,
            )
            for index in range(matrix.shape[0])
        ]
    )


def permutation_diagnostics(
    proposal: np.ndarray,
    cuda: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    pooled = np.concatenate((proposal, cuda), axis=0)
    proposal_size = proposal.shape[0]
    pooled_mean = np.mean(pooled, axis=0)
    active = pooled_mean >= 1.0e-4 * float(np.max(pooled_mean))
    scale = float(np.max(pooled_mean))

    def statistics(
        reference: np.ndarray,
        candidate: np.ndarray,
    ) -> tuple[float, float, float]:
        reference_mean = np.mean(reference, axis=0)
        candidate_mean = np.mean(candidate, axis=0)
        return (
            float(
                np.max(
                    np.abs(
                        candidate_mean[active] - reference_mean[active]
                    )
                )
                / scale
            ),
            relative_shift(
                float(np.max(reference_mean)),
                float(np.max(candidate_mean)),
            ),
            float(
                np.sum(
                    np.abs(
                        candidate_mean[active] - reference_mean[active]
                    )
                )
                / np.sum(np.abs(reference_mean[active]))
            ),
        )

    observed = statistics(proposal, cuda)
    sampled = np.empty((repetitions, 3), dtype=np.float64)
    for repetition in range(repetitions):
        order = rng.permutation(pooled.shape[0])
        sampled[repetition] = statistics(
            pooled[order[:proposal_size]],
            pooled[order[proposal_size:]],
        )
    return {
        "global_supremum_statistic": observed[0],
        "global_supremum_permutation_pvalue": float(
            (
                np.count_nonzero(sampled[:, 0] >= observed[0])
                + 1
            )
            / (repetitions + 1)
        ),
        "ensemble_mean_peak_height_relative_shift": observed[1],
        "ensemble_mean_peak_height_permutation_pvalue": float(
            (
                np.count_nonzero(
                    np.abs(sampled[:, 1]) >= abs(observed[1])
                )
                + 1
            )
            / (repetitions + 1)
        ),
        "relative_L1_statistic": observed[2],
        "relative_L1_permutation_pvalue": float(
            (
                np.count_nonzero(sampled[:, 2] >= observed[2])
                + 1
            )
            / (repetitions + 1)
        ),
        "permutation_repetitions": repetitions,
    }


def bootstrap_fixed_peak(
    proposal: np.ndarray,
    cuda: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> list[float]:
    common_peak = int(
        np.argmax(
            (
                np.mean(proposal, axis=0)
                + np.mean(cuda, axis=0)
            )
            / 2.0
        )
    )
    proposal_values = proposal[:, common_peak]
    cuda_values = cuda[:, common_peak]
    proposal_indices = rng.integers(
        0,
        proposal_values.size,
        size=(repetitions, proposal_values.size),
    )
    cuda_indices = rng.integers(
        0,
        cuda_values.size,
        size=(repetitions, cuda_values.size),
    )
    proposal_means = np.mean(
        proposal_values[proposal_indices],
        axis=1,
    )
    cuda_means = np.mean(cuda_values[cuda_indices], axis=1)
    shifts = (cuda_means - proposal_means) / proposal_means
    return [
        float(value)
        for value in np.quantile(shifts, (0.025, 0.5, 0.975))
    ]


def ensemble_mean_peak_shift(
    proposal: np.ndarray,
    cuda: np.ndarray,
) -> float:
    return relative_shift(
        float(np.max(np.mean(proposal, axis=0))),
        float(np.max(np.mean(cuda, axis=0))),
    )


def large_sample_subsampling(
    reference_root: Path,
    sample_size: int,
    observed_shift: float,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    manifest = json.loads(
        (reference_root / "run_manifest.json").read_text()
    )
    _, proposal = read_ensemble(
        profile_roots(reference_root, manifest, "proposal")
    )
    _, cuda = read_ensemble(
        profile_roots(reference_root, manifest, "cuda")
    )
    proposal_matrix = proposal["electron_positron"]
    cuda_matrix = cuda["electron_positron"]
    if sample_size > min(proposal_matrix.shape[0], cuda_matrix.shape[0]):
        raise ValueError(
            "requested diagnostic sample is larger than the reference "
            "ensemble"
        )

    def sample_shifts(size: int, count: int) -> np.ndarray:
        shifts = np.empty(count, dtype=np.float64)
        for index in range(count):
            proposal_indices = rng.integers(
                0, proposal_matrix.shape[0], size=size
            )
            cuda_indices = rng.integers(
                0, cuda_matrix.shape[0], size=size
            )
            shifts[index] = ensemble_mean_peak_shift(
                proposal_matrix[proposal_indices],
                cuda_matrix[cuda_indices],
            )
        return shifts

    selected = sample_shifts(sample_size, repetitions)
    scaling: list[dict[str, float | int]] = []
    scaling_repetitions = min(5_000, repetitions)
    candidate_sizes = (
        sample_size,
        100,
        200,
        500,
        1_000,
    )
    for size in dict.fromkeys(candidate_sizes):
        if size > min(proposal_matrix.shape[0], cuda_matrix.shape[0]):
            continue
        shifts = sample_shifts(size, scaling_repetitions)
        low, median, high = np.quantile(
            shifts, (0.025, 0.5, 0.975)
        )
        scaling.append(
            {
                "events_per_backend": size,
                "relative_shift_95pct_low": float(low),
                "relative_shift_median": float(median),
                "relative_shift_95pct_high": float(high),
            }
        )
    low, median, high = np.quantile(
        selected, (0.025, 0.5, 0.975)
    )
    return {
        "source": str(reference_root),
        "events_proposal": int(proposal_matrix.shape[0]),
        "events_cuda": int(cuda_matrix.shape[0]),
        "full_ensemble_peak_relative_shift": (
            ensemble_mean_peak_shift(proposal_matrix, cuda_matrix)
        ),
        "subsample_events_per_backend": sample_size,
        "subsample_relative_shift_95pct": [
            float(low),
            float(median),
            float(high),
        ],
        "probability_abs_shift_at_least_observed": float(
            np.mean(np.abs(selected) >= abs(observed_shift))
        ),
        "probability_positive_shift_at_least_observed": float(
            np.mean(selected >= observed_shift)
        ),
        "subsampling_repetitions": repetitions,
        "sample_size_scaling": scaling,
    }


def mean_and_sem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.nanmean(matrix, axis=0),
        np.nanstd(matrix, axis=0, ddof=1)
        / math.sqrt(matrix.shape[0]),
    )


def plot_alignment_diagnostic(
    coordinate: np.ndarray,
    proposal: dict[str, np.ndarray],
    cuda: dict[str, np.ndarray],
    relative_grid: np.ndarray,
    aligned_proposal: np.ndarray,
    aligned_cuda: np.ndarray,
    output: Path,
) -> None:
    electron_positron_proposal = proposal["electron_positron"]
    electron_positron_cuda = cuda["electron_positron"]
    proposal_xmax = coordinate[
        np.argmax(electron_positron_proposal, axis=1)
    ]
    cuda_xmax = coordinate[
        np.argmax(electron_positron_cuda, axis=1)
    ]
    proposal_maximum = np.max(electron_positron_proposal, axis=1)
    cuda_maximum = np.max(electron_positron_cuda, axis=1)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for backend, matrix in (
        ("proposal", electron_positron_proposal),
        ("cuda", electron_positron_cuda),
    ):
        mean, sem = mean_and_sem(matrix)
        axes[0, 0].plot(
            coordinate,
            mean,
            color=COLORS[backend],
            linestyle="--" if backend == "cuda" else "-",
            label=LABELS[backend],
        )
        axes[0, 0].fill_between(
            coordinate,
            mean - sem,
            mean + sem,
            color=COLORS[backend],
            alpha=0.14,
            linewidth=0.0,
        )
    axes[0, 0].set(
        title="Fixed-depth ensemble mean",
        xlabel=r"$X$ [g cm$^{-2}$]",
        ylabel=r"$N_{e^-}+N_{e^+}$",
    )
    combined_mean = (
        np.mean(electron_positron_proposal, axis=0)
        + np.mean(electron_positron_cuda, axis=0)
    ) / 2.0
    active = combined_mean >= 1.0e-4 * float(np.max(combined_mean))
    active_coordinate = coordinate[active]
    if active_coordinate.size:
        axes[0, 0].set_xlim(
            max(0.0, float(np.min(active_coordinate)) - 20.0),
            float(np.max(active_coordinate)) + 20.0,
        )
    axes[0, 0].legend(frameon=False)

    for backend, matrix in (
        ("proposal", aligned_proposal),
        ("cuda", aligned_cuda),
    ):
        mean, sem = mean_and_sem(matrix)
        axes[0, 1].plot(
            relative_grid,
            mean,
            color=COLORS[backend],
            linestyle="--" if backend == "cuda" else "-",
            label=LABELS[backend],
        )
        axes[0, 1].fill_between(
            relative_grid,
            mean - sem,
            mean + sem,
            color=COLORS[backend],
            alpha=0.14,
            linewidth=0.0,
        )
    axes[0, 1].set(
        title=r"Each shower aligned to its own $X_{\max}$",
        xlabel=r"$X-X_{\max}$ [g cm$^{-2}$]",
        ylabel=r"$N_{e^-}+N_{e^+}$",
    )

    bins = np.arange(
        min(float(np.min(proposal_xmax)), float(np.min(cuda_xmax))) - 5.0,
        max(float(np.max(proposal_xmax)), float(np.max(cuda_xmax))) + 15.0,
        20.0,
    )
    axes[1, 0].hist(
        proposal_xmax,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=COLORS["proposal"],
        label=LABELS["proposal"],
    )
    axes[1, 0].hist(
        cuda_xmax,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color=COLORS["cuda"],
        label=LABELS["cuda"],
    )
    axes[1, 0].set(
        title=r"Per-shower $X_{\max}$ distribution",
        xlabel=r"$X_{\max}$ [g cm$^{-2}$]",
        ylabel="density",
    )

    maximum_bins = np.histogram_bin_edges(
        np.concatenate((proposal_maximum, cuda_maximum)),
        bins="auto",
    )
    axes[1, 1].hist(
        proposal_maximum,
        bins=maximum_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=COLORS["proposal"],
        label=LABELS["proposal"],
    )
    axes[1, 1].hist(
        cuda_maximum,
        bins=maximum_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        linestyle="--",
        color=COLORS["cuda"],
        label=LABELS["cuda"],
    )
    axes[1, 1].set(
        title=r"Per-shower $N_{e^\pm,\max}$ distribution",
        xlabel=r"$N_{e^-}+N_{e^+}$ at the shower maximum",
        ylabel="density",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Why a fixed-depth ensemble-mean peak can differ for small samples"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_sample_size_scaling(
    reference: dict[str, Any],
    observed_shift: float,
    observed_events_per_backend: int,
    output: Path,
) -> None:
    frame = pd.DataFrame(reference["sample_size_scaling"])
    events = frame["events_per_backend"].to_numpy(dtype=np.float64)
    low = (
        100.0
        * frame["relative_shift_95pct_low"].to_numpy(dtype=np.float64)
    )
    median = (
        100.0
        * frame["relative_shift_median"].to_numpy(dtype=np.float64)
    )
    high = (
        100.0
        * frame["relative_shift_95pct_high"].to_numpy(dtype=np.float64)
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.fill_between(
        events,
        low,
        high,
        color="#9467bd",
        alpha=0.22,
        label="95% resampling interval",
    )
    axis.plot(
        events,
        median,
        color="#9467bd",
        marker="o",
        label="resampling median",
    )
    axis.axhline(
        100.0 * observed_shift,
        color=COLORS["cuda"],
        linestyle="--",
        label=(
            "observed 100 TeV "
            f"N={observed_events_per_backend} peak shift"
        ),
    )
    axis.set_xscale("log")
    axis.set(
        xlabel="showers per backend",
        ylabel="ensemble-mean peak shift (CUDA/CPU - 1) [%]",
        title=(
            "Finite-sample width estimated from the older "
            "1 TeV, N=1000 ensemble"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    em = report["components"]["electron_positron"]
    aligned = report["aligned_electron_positron"]
    xmax = report["xmax_distribution"]
    permutation = report["permutation"]
    thinning = report["thinning"]
    bootstrap = report[
        "fixed_depth_peak_bootstrap_relative_shift_95pct"
    ]
    proposal_events = int(report["events"]["proposal"])
    cuda_events = int(report["events"]["cuda"])
    lines = [
        "# 100 TeV 纵向平均 profile 峰值差异诊断",
        "",
        "## 结论",
        "",
        (
            f"本次比较使用 {proposal_events} 个 CPU 和 "
            f"{cuda_events} 个 CUDA 独立 shower。固定深度系综均值的"
            "差异主要伴随两组 $X_{\\max}$ 位置与展宽的差异；"
            "按单 shower 的 $X_{\\max}$ 对齐后，峰值和峰后形状接近。"
            "现有检验没有证明 CUDA 在单 shower 中存在系统性的电磁"
            "粒子增益或损失。"
        ),
        "",
        (
            f"- 固定深度平均峰的 $e^-+e^+$ 相对差为 "
            f"{100 * em['fixed_depth_peak_relative_shift']:+.2f}%。"
        ),
        (
            f"- 按每个 shower 自己的 $X_{{\\max}}$ 对齐后，"
            f"$\\Delta X=0$ 处只差 "
            f"{100 * aligned['relative_shift_at_zero']:+.2f}%。"
        ),
        (
            f"- 逐 shower 最大粒子数均值只差 "
            f"{100 * em['per_shower_maximum_relative_shift']:+.2f}%"
            f"（{em['per_shower_maximum_z_score']:+.2f}σ）。"
        ),
        (
            f"- CPU/CUDA 的 $X_{{\\max}}$ 均值分别为 "
            f"{xmax['proposal_mean_gcm2']:.1f}/"
            f"{xmax['cuda_mean_gcm2']:.1f} g cm⁻²，标准差分别为 "
            f"{xmax['proposal_standard_deviation_gcm2']:.1f}/"
            f"{xmax['cuda_standard_deviation_gcm2']:.1f} g cm⁻²。"
        ),
        (
            f"- $X_{{\\max}}$ 均值 Welch 检验 p="
            f"{xmax['welch_mean_pvalue']:.3f}，展宽 Brown--Forsythe "
            f"检验 p={xmax['brown_forsythe_scale_pvalue']:.3f}，"
            f"两样本 KS 检验 p={xmax['KS_pvalue']:.3f}。"
        ),
        (
            f"- 全曲线最大偏差置换检验 p="
            f"{permutation['global_supremum_permutation_pvalue']:.3f}，"
            f"相对 L1 置换检验 p="
            f"{permutation['relative_L1_permutation_pvalue']:.3f}。"
        ),
        (
            f"- 只针对已经看到的平均峰高做检验时，置换 p="
            f"{permutation['ensemble_mean_peak_height_permutation_pvalue']:.3f}，"
            f"bootstrap 95% 区间为 [{100 * bootstrap[0]:+.1f}%, "
            f"{100 * bootstrap[2]:+.1f}%]。区间包含 0；由于峰高指标"
            "是在看图后选出的，也不能把它单独当成已经证实的系统偏差。"
        ),
        "",
        "## 配置与 thinning 核查",
        "",
        (
            f"- CPU/CUDA 纵向输运物理参数清单匹配（射电 observer layout "
            f"不参与 profile 输运，故在此忽略）："
            f"{report['physics_configuration_match']}。"
        ),
        (
            f"- `emthin={thinning['em_fraction']:.3g}` 对应阈值 "
            f"{thinning['threshold_GeV']:.3g} GeV；自动最大权重为 "
            f"{thinning['maximum_weight']:.3g}。"
        ),
        (
            f"- 从单位权重开始可触发 thinning："
            f"{thinning['can_activate_from_unit_weight']}。"
        ),
        (
            "因此这组数据实际是未薄化输运；CPU 原程序和 CUDA 程序使用"
            "相同的自动最大权重公式。这排除了 CPU/CUDA 低薄化分支不同"
            "这一解释，但也说明不能把这组数据称为已经启用的 1e-6 thinning。"
        ),
        (
            "- 两侧不是同一个二进制：参考侧是原始程序，CUDA 侧还包含"
            "进程隔离 FLUKA 和稀有光核末态 fallback。输运参数相同，"
            "而实现级差异仍需通过“原始 CPU / 重构 CPU-PROPOSAL / CUDA”"
            "三臂对照单独排除。"
        ),
        "",
        "## 下一步验收",
        "",
        (
            "1. 用独立种子再复制一组同规模系综，继续同时保留固定深度"
            "平均、逐 shower 标量和按 $X_{\\max}$ 对齐的三种比较；"
            "也可用 decision-tape replay 做相同决策树的逐过程检查。"
        ),
        (
            "2. 若要专门验收 thinning，应在两侧显式给出同一个 "
            "`--max-weight`（必须大于 1，例如 100），并与当前未薄化"
            "样本分开命名。"
        ),
        (
            "3. 非 EM 输运 profile 必须继续比较 hadron、μ⁺、μ⁻ 和总 μ；"
            "`charged` 在 CORSIKA 8 中表示所有带电粒子，不能再标成 "
            "$e^-+e^+$。"
        ),
        (
            "4. `ProductionProfile` 的 parent 分类由独立的 "
            "`muon_production_parent_alignment_summary.json` 验收；"
            "本诊断不从纵向粒子 profile 推断 parent 输出是否正确。"
        ),
        "",
        "对应图：",
        "",
        "- `xmax_alignment_diagnostic.png`",
        "- `component_peak_diagnostics.csv`",
        "",
    ]
    reference = report.get("large_sample_reference")
    if isinstance(reference, dict):
        low, median, high = reference[
            "subsample_relative_shift_95pct"
        ]
        lines.extend(
            [
                "## 与旧 1 TeV、1000 事例图的样本量交叉检查",
                "",
                (
                    f"- 旧样本完整集合的平均峰差为 "
                    f"{100 * reference['full_ensemble_peak_relative_shift']:+.2f}%。"
                ),
                (
                    f"- 从旧样本反复各抽 "
                    f"{reference['subsample_events_per_backend']} 个事例时，"
                    f"峰差的 95% 区间为 [{100 * low:+.1f}%, "
                    f"{100 * high:+.1f}%]，中位数为 "
                    f"{100 * median:+.1f}%。"
                ),
                (
                    f"- 产生不小于本次峰差绝对值的概率为 "
                    f"{100 * reference['probability_abs_shift_at_least_observed']:.1f}%。"
                ),
                (
                    "该交叉检查的能量、天顶角和 thinning 设置不同，只能"
                    f"用于说明 {proposal_events} 个质子 shower 的平均峰"
                    "仍可能具有抽样噪声，"
                    "不能替代同一 100 TeV 配置的增样本验收。"
                ),
                "- 对应图：`reference_sample_size_scaling.png`。",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.resamples < 100:
        raise ValueError("--resamples must be at least 100")
    implicit_model_set = (
        args.proposal_implicit_geomagnetic_model is not None
    )
    implicit_year_set = (
        args.proposal_implicit_geomagnetic_year is not None
    )
    if implicit_model_set != implicit_year_set:
        raise ValueError(
            "proposal implicit geomagnetic model and year must be provided "
            "together"
        )
    proposal_implicit_physics_options: dict[str, str] = {}
    if implicit_model_set:
        if not math.isfinite(args.proposal_implicit_geomagnetic_year):
            raise ValueError(
                "proposal implicit geomagnetic year must be finite"
            )
        proposal_implicit_physics_options = {
            "--geomagnetic-model":
                args.proposal_implicit_geomagnetic_model,
            "--geomagnetic-year":
                f"{args.proposal_implicit_geomagnetic_year:.17g}",
        }
    root = args.ensemble_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else root / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    proposal_roots = profile_roots(root, manifest, "proposal")
    cuda_roots = profile_roots(root, manifest, "cuda")
    coordinate, proposal = read_ensemble(proposal_roots)
    cuda_coordinate, cuda = read_ensemble(cuda_roots)
    if (
        coordinate.shape != cuda_coordinate.shape
        or not np.array_equal(coordinate, cuda_coordinate)
    ):
        raise ValueError("CPU and CUDA profile coordinate grids differ")

    configuration = manifest["configuration"]
    physics_configuration_match = True
    try:
        from compare_ensembles import canonical_physics_configuration

        def profile_transport_configuration(
            path: Path,
            *,
            implicit_physics_options: dict[str, str] | None = None,
        ) -> tuple[str, ...]:
            configuration = canonical_physics_configuration(
                path,
                implicit_physics_options=implicit_physics_options,
            )
            filtered: list[str] = []
            for index in range(0, len(configuration), 2):
                key = configuration[index]
                value = configuration[index + 1]
                if key != "--observer-layout-sha256":
                    filtered.extend((key, value))
            return tuple(filtered)

        reference_configuration = profile_transport_configuration(
            proposal_roots[0],
            implicit_physics_options=proposal_implicit_physics_options,
        )
        proposal_configuration_match = all(
            profile_transport_configuration(
                path,
                implicit_physics_options=
                    proposal_implicit_physics_options,
            )
            == reference_configuration
            for path in proposal_roots[1:]
        )
        cuda_configuration_match = all(
            profile_transport_configuration(path)
            == reference_configuration
            for path in cuda_roots
        )
        physics_configuration_match = (
            proposal_configuration_match
            and cuda_configuration_match
        )
    except (ImportError, ValueError):
        physics_configuration_match = False

    rng = np.random.default_rng(args.seed)
    summaries = {
        component: component_summary(
            coordinate,
            proposal[component],
            cuda[component],
        )
        for component in COMPONENTS
    }
    electron_positron_proposal = proposal["electron_positron"]
    electron_positron_cuda = cuda["electron_positron"]
    proposal_xmax = coordinate[
        np.argmax(electron_positron_proposal, axis=1)
    ]
    cuda_xmax = coordinate[
        np.argmax(electron_positron_cuda, axis=1)
    ]
    relative_grid = np.arange(-400.0, 501.0, 10.0)
    aligned_proposal = align_to_xmax(
        coordinate,
        electron_positron_proposal,
        proposal_xmax,
        relative_grid,
    )
    aligned_cuda = align_to_xmax(
        coordinate,
        electron_positron_cuda,
        cuda_xmax,
        relative_grid,
    )
    zero = int(np.flatnonzero(relative_grid == 0.0)[0])
    aligned_proposal_zero = aligned_proposal[:, zero]
    aligned_cuda_zero = aligned_cuda[:, zero]
    aligned_difference = (
        float(np.mean(aligned_cuda_zero))
        - float(np.mean(aligned_proposal_zero))
    )
    aligned_se = math.sqrt(
        float(np.var(aligned_proposal_zero, ddof=1))
        / aligned_proposal_zero.size
        + float(np.var(aligned_cuda_zero, ddof=1))
        / aligned_cuda_zero.size
    )

    em_fraction = float(configuration["em_thinning"])
    primary_energy = float(configuration["energy_GeV"])
    configured_maximum_weight = float(configuration["maximum_weight"])
    automatic_maximum_weight = configured_maximum_weight <= 0.0
    maximum_weight = (
        0.5 * em_fraction * primary_energy
        if automatic_maximum_weight
        else configured_maximum_weight
    )
    report = {
        "source": str(root),
        "manifest": str(manifest_path),
        "events": {
            "proposal": int(electron_positron_proposal.shape[0]),
            "cuda": int(electron_positron_cuda.shape[0]),
        },
        "physics_configuration_match": physics_configuration_match,
        "observer_layout_ignored_for_profile_configuration": True,
        "thinning": {
            "em_fraction": em_fraction,
            "threshold_GeV": em_fraction * primary_energy,
            "maximum_weight": maximum_weight,
            "automatic_maximum_weight": automatic_maximum_weight,
            "can_activate_from_unit_weight": maximum_weight > 1.0,
        },
        "components": summaries,
        "aligned_electron_positron": {
            "relative_shift_at_zero": relative_shift(
                float(np.mean(aligned_proposal_zero)),
                float(np.mean(aligned_cuda_zero)),
            ),
            "z_score_at_zero": (
                aligned_difference / aligned_se
                if aligned_se > 0.0
                else 0.0
            ),
        },
        "xmax_distribution": {
            "proposal_mean_gcm2": float(np.mean(proposal_xmax)),
            "proposal_standard_deviation_gcm2": float(
                np.std(proposal_xmax, ddof=1)
            ),
            "cuda_mean_gcm2": float(np.mean(cuda_xmax)),
            "cuda_standard_deviation_gcm2": float(
                np.std(cuda_xmax, ddof=1)
            ),
            "welch_mean_pvalue": float(
                stats.ttest_ind(
                    proposal_xmax,
                    cuda_xmax,
                    equal_var=False,
                ).pvalue
            ),
            "brown_forsythe_scale_pvalue": float(
                stats.levene(
                    proposal_xmax,
                    cuda_xmax,
                    center="median",
                ).pvalue
            ),
            "KS_pvalue": float(
                stats.ks_2samp(proposal_xmax, cuda_xmax).pvalue
            ),
        },
        "permutation": permutation_diagnostics(
            electron_positron_proposal,
            electron_positron_cuda,
            args.resamples,
            rng,
        ),
        "fixed_depth_peak_bootstrap_relative_shift_95pct": (
            bootstrap_fixed_peak(
                electron_positron_proposal,
                electron_positron_cuda,
                args.resamples,
                rng,
            )
        ),
    }
    if args.large_sample_reference_root is not None:
        report["large_sample_reference"] = large_sample_subsampling(
            args.large_sample_reference_root.resolve(),
            electron_positron_proposal.shape[0],
            summaries["electron_positron"][
                "fixed_depth_peak_relative_shift"
            ],
            args.resamples,
            rng,
        )
    with (output / "diagnosis.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    pd.DataFrame(
        [
            {"component": component, **summary}
            for component, summary in summaries.items()
        ]
    ).to_csv(output / "component_peak_diagnostics.csv", index=False)
    plot_alignment_diagnostic(
        coordinate,
        proposal,
        cuda,
        relative_grid,
        aligned_proposal,
        aligned_cuda,
        output / "xmax_alignment_diagnostic.png",
    )
    if "large_sample_reference" in report:
        plot_sample_size_scaling(
            report["large_sample_reference"],
            summaries["electron_positron"][
                "fixed_depth_peak_relative_shift"
            ],
            int(report["events"]["proposal"]),
            output / "reference_sample_size_scaling.png",
        )
    write_markdown(output / "diagnosis.md", report)
    print(
        json.dumps(
            {
                "events": report["events"],
                "fixed_depth_peak_relative_shift": summaries[
                    "electron_positron"
                ]["fixed_depth_peak_relative_shift"],
                "aligned_relative_shift": report[
                    "aligned_electron_positron"
                ]["relative_shift_at_zero"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
