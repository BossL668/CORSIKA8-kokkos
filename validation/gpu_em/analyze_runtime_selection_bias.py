#!/usr/bin/env python3
"""Diagnose completion-time selection in an in-progress CPU ensemble.

The input is an existing ``compare_ensembles.py`` output.  Proposal rows are
joined to their per-shower wall times using the ordered source list recorded in
``comparison.json``.  This tool is diagnostic only: a correlation between
runtime and a shower observable explains why early completions are not an
unbiased ensemble, but it is never a CPU/CUDA acceptance test.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


DEFAULT_METRICS = (
    "profile_hadron_integral",
    "profile_muon_integral",
    "profile_em_integral",
    "profile_charged_integral",
    "profile_xmax_charged_gcm2",
    "ground_em_weighted_count",
)

METRIC_LABELS = {
    "profile_hadron_integral": "Hadron profile integral",
    "profile_muon_integral": "Muon profile integral",
    "profile_em_integral": "EM profile integral",
    "profile_charged_integral": "All-charged profile integral",
    "profile_xmax_charged_gcm2": r"Charged $X_{\max}$",
    "ground_em_weighted_count": "Ground EM weighted count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CPU runtime correlations in a completion-conditioned "
            "interim ensemble."
        )
    )
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        required=True,
        help="Output directory previously created by compare_ensembles.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        action="append",
        help="Observable to analyze; repeat to replace the default set.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required YAML file is missing: {path}")
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with path.open("r", encoding="utf-8") as source:
        return yaml.load(source, Loader=loader)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def shower_number(name: str) -> int:
    if not name.startswith("shower_"):
        raise ValueError(f"invalid shower timing key: {name}")
    return int(name.removeprefix("shower_"))


def command_seed(root: Path) -> int:
    config = read_yaml(root / "config.yaml")
    if not isinstance(config, dict) or not isinstance(config.get("args"), str):
        raise ValueError(f"invalid config.yaml in {root}")
    tokens = shlex.split(config["args"])
    try:
        index = tokens.index("--seed")
        seed = int(tokens[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing or invalid --seed in {root}") from error
    return seed


def source_timing_rows(root: Path) -> list[dict[str, Any]]:
    timing = read_yaml(root / "simulation_timing" / "summary.yaml")
    if not isinstance(timing, dict) or not timing:
        raise ValueError(f"invalid simulation timing summary in {root}")
    seed_start = command_seed(root)
    rows: list[dict[str, Any]] = []
    for name, record in sorted(
        timing.items(), key=lambda item: shower_number(str(item[0]))
    ):
        shower = shower_number(str(name))
        if (
            not isinstance(record, dict)
            or record.get("closed") is not True
            or record.get("status") != "closed"
        ):
            raise ValueError(f"incomplete timing record {name} in {root}")
        wall_time_ms = float(record.get("wall_time_ms", math.nan))
        if not math.isfinite(wall_time_ms) or wall_time_ms <= 0.0:
            raise ValueError(f"invalid wall_time_ms for {name} in {root}")
        rows.append(
            {
                "source": str(root),
                "source_shower": shower,
                "seed": seed_start + shower,
                "runtime_seconds": wall_time_ms / 1000.0,
                "runtime_hours": wall_time_ms / 3.6e6,
            }
        )
    expected = list(range(len(rows)))
    observed = [int(row["source_shower"]) for row in rows]
    if observed != expected:
        raise ValueError(
            f"timing shower IDs are not contiguous in {root}: {observed}"
        )
    return rows


def attach_proposal_timings(
    comparison: dict[str, Any], observables: pd.DataFrame
) -> pd.DataFrame:
    proposal = observables.loc[observables["backend"] == "proposal"].copy()
    metadata = comparison.get("proposal")
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("sources"), list
    ):
        raise ValueError("comparison.json lacks ordered proposal sources")
    timing_rows = [
        row
        for source in metadata["sources"]
        for row in source_timing_rows(Path(source))
    ]
    if len(timing_rows) != len(proposal):
        raise ValueError(
            "proposal timing/observable count mismatch: "
            f"{len(timing_rows)} versus {len(proposal)}"
        )
    proposal = proposal.reset_index(drop=True)
    timing = pd.DataFrame(timing_rows)
    result = pd.concat([timing, proposal], axis=1)
    if result["seed"].duplicated().any():
        raise ValueError("duplicate proposal seeds in runtime diagnostic")
    return result


def correlation_summary(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    runtime = frame["runtime_seconds"].to_numpy(dtype=np.float64)
    values = frame[metric].to_numpy(dtype=np.float64)
    if (
        runtime.size < 3
        or not np.isfinite(runtime).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"invalid samples for {metric}")
    pearson = stats.pearsonr(runtime, values)
    spearman = stats.spearmanr(runtime, values)
    count = min(10, max(1, runtime.size // 3))
    fastest = frame.nsmallest(count, "runtime_seconds")[metric]
    slowest = frame.nlargest(count, "runtime_seconds")[metric]
    mean = float(np.mean(values))
    scale = max(abs(mean), np.finfo(np.float64).tiny)
    return {
        "events": int(runtime.size),
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "tail_count": int(count),
        "fastest_tail_over_all_mean": float(fastest.mean() / scale),
        "slowest_tail_over_all_mean": float(slowest.mean() / scale),
    }


def plot_correlations(
    frame: pd.DataFrame,
    metrics: tuple[str, ...],
    summaries: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.0, 4.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    runtime_hours = frame["runtime_hours"].to_numpy(dtype=np.float64)
    for axis, metric in zip(axes.flat, metrics):
        values = frame[metric].to_numpy(dtype=np.float64)
        scale = max(abs(float(np.mean(values))), np.finfo(np.float64).tiny)
        normalized = values / scale
        axis.scatter(
            runtime_hours,
            normalized,
            c=runtime_hours,
            cmap="viridis",
            edgecolor="white",
            linewidth=0.5,
            s=42,
            alpha=0.9,
        )
        if np.ptp(runtime_hours) > 0.0:
            slope, intercept = np.polyfit(runtime_hours, normalized, 1)
            xline = np.linspace(runtime_hours.min(), runtime_hours.max(), 100)
            axis.plot(xline, intercept + slope * xline, color="tab:red", lw=1.8)
        axis.axhline(1.0, color="0.35", lw=1.0, ls="--")
        summary = summaries[metric]
        axis.text(
            0.03,
            0.97,
            (
                f"Pearson r={summary['pearson_r']:.3f}, "
                f"p={summary['pearson_pvalue']:.2g}\n"
                f"Spearman rho={summary['spearman_rho']:.3f}, "
                f"p={summary['spearman_pvalue']:.2g}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xlabel("Original CPU single-shower runtime [h]")
        axis.set_ylabel("Observable / completed-sample mean")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Runtime-conditioned original-CPU sample: diagnostic only\n"
        "Positive trends show why early completions are not an unbiased ensemble",
        fontsize=14,
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    ensemble_root = args.ensemble_root.resolve()
    output = args.output.resolve()
    if not ensemble_root.is_dir():
        raise ValueError(f"ensemble root is not a directory: {ensemble_root}")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS

    comparison = read_json(ensemble_root / "comparison.json")
    observables = pd.read_csv(ensemble_root / "per_shower_observables.csv")
    missing = set(metrics) - set(observables.columns)
    if missing:
        raise ValueError(f"observable table lacks metrics: {sorted(missing)}")
    frame = attach_proposal_timings(comparison, observables)
    summaries = {
        metric: correlation_summary(frame, metric) for metric in metrics
    }
    runtimes = frame["runtime_seconds"].to_numpy(dtype=np.float64)
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "source": str(ensemble_root),
        "events": int(len(frame)),
        "selection_warning": (
            "If the parent campaign is still running, this sample contains "
            "only completed CPU showers and is right-censored by runtime. "
            "It is not final CPU/CUDA acceptance evidence."
        ),
        "runtime_seconds": {
            "minimum": float(np.min(runtimes)),
            "median": float(np.median(runtimes)),
            "mean": float(np.mean(runtimes)),
            "maximum": float(np.max(runtimes)),
        },
        "correlations": summaries,
    }

    output.mkdir(parents=True)
    frame.to_csv(output / "runtime_conditioned_cpu_events.csv", index=False)
    with (output / "runtime_selection_bias.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    plot_correlations(
        frame,
        metrics,
        summaries,
        output / "runtime_selection_bias.png",
    )
    with (output / "README.md").open("w", encoding="utf-8") as destination:
        destination.write(
            "# Runtime-selection diagnostic\n\n"
            "This directory quantifies correlations between original scalar-CPU "
            "runtime and shower observables. It is diagnostic only. During an "
            "active campaign, completed events are right-censored by runtime and "
            "must not be treated as an unbiased CPU reference ensemble.\n"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        raise SystemExit(2)
