#!/usr/bin/env python3
"""Synthesize the complete ground-EM and radio validation evidence.

The report keeps three causal layers separate:

1. independent-shower ground-EM distributions;
2. radio settings and identical-track CPU/CUDA projection algorithms;
3. independent-shower end-to-end radio pulse distributions.

Passing layer 2 does not force layer 3 to pass, but it excludes radio setting
drift and the projection implementation as a direct cause of a layer-3 shift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


GROUND_METRICS = (
    "ground_em_weighted_count",
    "ground_em_kinetic_energy_GeV",
)
TWO_SAMPLE_TESTS = (
    "ks",
    "cramer_von_mises",
    "anderson_darling_k_sample",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required validation artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def interval_contains(interval: Any, target: float) -> bool:
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"invalid confidence interval: {interval!r}")
    low, high = (float(interval[0]), float(interval[1]))
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid confidence interval: {interval!r}")
    return low <= target <= high


def holm_bonferroni(
    p_values: dict[str, float], alpha: float
) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise alpha must lie in (0, 1)")
    ordered = sorted((float(value), name) for name, value in p_values.items())
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("two-sample p-values must be finite and in [0, 1]")
    rejected: list[str] = []
    decisions: list[dict[str, Any]] = []
    continue_rejecting = True
    count = len(ordered)
    for index, (value, name) in enumerate(ordered):
        threshold = alpha / (count - index)
        reject = bool(continue_rejecting and value <= threshold)
        if reject:
            rejected.append(name)
        else:
            continue_rejecting = False
        decisions.append(
            {
                "test": name,
                "p_value": value,
                "threshold": threshold,
                "rejected": reject,
            }
        )
    return {
        "method": "Holm-Bonferroni",
        "familywise_alpha": alpha,
        "family_size": count,
        "rejected": rejected,
        "decisions": decisions,
        "passed": not rejected,
    }


def ground_evidence(
    comparison: dict[str, Any],
    attribution: dict[str, Any],
    *,
    expected_events: int,
    alpha: float,
) -> dict[str, Any]:
    p_values: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for metric in GROUND_METRICS:
        scalar = comparison["scalars"][metric]
        detailed = attribution["ground"][metric]
        event_count_pass = bool(
            int(scalar["proposal"]["events"]) == expected_events
            and int(scalar["cuda"]["events"]) == expected_events
        )
        statistical_pass = bool(scalar.get("statistical_pass"))
        adjusted_interval = detailed["models"][
            "adjusted_xmax_and_em_integral"
        ]["bootstrap_shower_95pct"]
        adjusted_null_pass = interval_contains(adjusted_interval, 1.0)
        tail_fraction_pass: dict[str, bool] = {}
        tail_scale_pass: dict[str, bool] = {}
        for label in ("q05", "q16"):
            tail_fraction_pass[label] = interval_contains(
                detailed["low_tail_event_fractions"][label][
                    "bootstrap_shower_95pct"
                ],
                0.0,
            )
            tail_scale_pass[label] = interval_contains(
                detailed["low_tail_quantile_ratios"][label][
                    "bootstrap_shower_95pct"
                ],
                1.0,
            )
        for test in TWO_SAMPLE_TESTS:
            p_values[f"{metric}:{test}"] = float(
                detailed["tests"][test]["p_value"]
            )
        metrics[metric] = {
            "events_exact": event_count_pass,
            "comparison_statistical_pass": statistical_pass,
            "raw_relative_mean_difference": float(scalar["relative_difference"]),
            "raw_absolute_z_score": float(scalar["absolute_z_score"]),
            "adjusted_xmax_em_ratio": float(
                detailed["models"]["adjusted_xmax_and_em_integral"][
                    "cuda_over_proposal_ratio"
                ]
            ),
            "adjusted_xmax_em_interval": adjusted_interval,
            "adjusted_interval_contains_one": adjusted_null_pass,
            "low_tail_fraction_intervals_contain_zero": tail_fraction_pass,
            "low_tail_quantile_ratio_intervals_contain_one": tail_scale_pass,
        }
    shape_family = holm_bonferroni(p_values, alpha)
    for metric in GROUND_METRICS:
        item = metrics[metric]
        item["passed"] = bool(
            item["events_exact"]
            and item["comparison_statistical_pass"]
            and item["adjusted_interval_contains_one"]
            and all(item["low_tail_fraction_intervals_contain_zero"].values())
            and all(item["low_tail_quantile_ratio_intervals_contain_one"].values())
        )
    passed = bool(
        shape_family["passed"]
        and all(item["passed"] for item in metrics.values())
    )
    return {
        "status": (
            "no_detectable_difference" if passed else "difference_detected_or_unresolved"
        ),
        "passed": passed,
        "null_hypotheses": {
            "adjusted_ratio": 1.0,
            "low_tail_fraction_difference": 0.0,
            "low_tail_quantile_ratio": 1.0,
        },
        "metrics": metrics,
        "two_sample_shape_test_family": shape_family,
    }


def radio_evidence(
    configuration: dict[str, Any],
    waveform_oracle: dict[str, Any],
    pulse_oracle: dict[str, Any],
    independent_pulses: dict[str, Any],
    *,
    expected_events: int,
) -> dict[str, Any]:
    settings_pass = bool(
        configuration.get("status") == "equivalent"
        and int(configuration["events_by_backend"]["proposal"]) == expected_events
        and int(configuration["events_by_backend"]["cuda"]) == expected_events
    )
    waveform_pass = bool(
        waveform_oracle.get("status") == "passed"
        and waveform_oracle.get("transport_identity", {}).get("accepted") is True
        and all(
            waveform_oracle.get("algorithms", {}).get(algorithm, {}).get("accepted")
            is True
            for algorithm in ("CoREAS", "ZHS")
        )
    )
    pulse_feature_pass = bool(
        pulse_oracle.get("status") == "passed"
        and all(
            pulse_oracle.get("algorithms", {}).get(algorithm, {}).get("status")
            == "passed"
            for algorithm in ("CoREAS", "ZHS")
        )
    )
    independent_pass = bool(
        independent_pulses.get("acceptance", {}).get("passed") is True
    )
    projection_pass = waveform_pass and pulse_feature_pass
    direct_projection_cause_excluded = bool(settings_pass and projection_pass)
    if settings_pass and projection_pass and independent_pass:
        status = "end_to_end_statistically_consistent"
    elif direct_projection_cause_excluded:
        status = "end_to_end_difference_not_caused_by_radio_settings_or_projection"
    else:
        status = "radio_projection_or_settings_unresolved"
    return {
        "status": status,
        "passed": bool(settings_pass and projection_pass and independent_pass),
        "settings_equivalent": settings_pass,
        "identical_track_waveforms_passed": waveform_pass,
        "identical_track_pulse_features_passed": pulse_feature_pass,
        "independent_shower_pulse_distributions_passed": independent_pass,
        "settings_and_projection_excluded_as_direct_cause": (
            direct_projection_cause_excluded
        ),
        "independent_acceptance": independent_pulses.get("acceptance"),
    }


def summarize(
    finalization: dict[str, Any],
    comparison: dict[str, Any],
    attribution: dict[str, Any],
    configuration: dict[str, Any],
    waveform_oracle: dict[str, Any],
    pulse_oracle: dict[str, Any],
    independent_pulses: dict[str, Any],
    *,
    expected_events: int,
    alpha: float,
) -> dict[str, Any]:
    integrity_pass = bool(
        finalization.get("status") == "complete"
        and int(finalization.get("proposal_events", -1)) == expected_events
        and int(finalization.get("cuda_events", -1)) == expected_events
        and int(comparison.get("proposal", {}).get("events", -1)) == expected_events
        and int(comparison.get("cuda", {}).get("events", -1)) == expected_events
    )
    ground = ground_evidence(
        comparison,
        attribution,
        expected_events=expected_events,
        alpha=alpha,
    )
    radio = radio_evidence(
        configuration,
        waveform_oracle,
        pulse_oracle,
        independent_pulses,
        expected_events=expected_events,
    )
    shower_profiles_pass = bool(
        comparison.get("acceptance", {}).get("curve_pass") is True
        and all(
            item.get("statistical_pass") is True
            for item in comparison.get("scalars", {}).values()
            if item.get("acceptance_gate") is True
        )
    )
    passed = integrity_pass and shower_profiles_pass and ground["passed"] and radio["passed"]
    return {
        "schema_version": 1,
        "status": "passed" if passed else "needs_investigation",
        "expected_events_per_backend": expected_events,
        "data_integrity_passed": integrity_pass,
        "shower_profile_statistical_consistency_passed": shower_profiles_pass,
        "ground_em": ground,
        "radio": radio,
        "interpretation": (
            "A radio end-to-end failure with settings and identical-track "
            "projection both passing cannot be attributed directly to the "
            "CoREAS/ZHS CUDA projection. The remaining causes are upstream "
            "shower/track populations, finite independent-shower statistics, "
            "or the pulse-selection stage and must be diagnosed separately."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    ground = report["ground_em"]
    radio = report["radio"]
    lines = [
        "# Ground EM and radio validation conclusion",
        "",
        f"Overall status: **{report['status']}**",
        "",
        f"- Exact event/data integrity: {report['data_integrity_passed']}",
        f"- Shower profile statistical consistency: {report['shower_profile_statistical_consistency_passed']}",
        f"- Ground EM: {ground['status']}",
        f"- Radio: {radio['status']}",
        "",
        "## Ground EM",
        "",
        "| Metric | Raw relative mean difference | |z| | Xmax+EM adjusted ratio [95%] | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for metric, item in ground["metrics"].items():
        interval = item["adjusted_xmax_em_interval"]
        lines.append(
            f"| `{metric}` | {item['raw_relative_mean_difference']:+.4%} | "
            f"{item['raw_absolute_z_score']:.3f} | "
            f"{item['adjusted_xmax_em_ratio']:.4f} "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{'passed' if item['passed'] else 'needs investigation'} |"
        )
    lines.extend(
        (
            "",
            "## Radio causal layers",
            "",
            f"- Configuration equivalence: {radio['settings_equivalent']}",
            f"- Identical-track waveform projection: {radio['identical_track_waveforms_passed']}",
            f"- Identical-track pulse features: {radio['identical_track_pulse_features_passed']}",
            f"- Independent-shower pulse distributions: {radio['independent_shower_pulse_distributions_passed']}",
            "",
            report["interpretation"],
            "",
        )
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalization", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--ground-attribution", type=Path, required=True)
    parser.add_argument("--radio-configuration", type=Path, required=True)
    parser.add_argument("--radio-waveform-oracle", type=Path, required=True)
    parser.add_argument("--radio-pulse-oracle", type=Path, required=True)
    parser.add_argument("--independent-radio-pulses", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(
        read_json(args.finalization),
        read_json(args.comparison),
        read_json(args.ground_attribution),
        read_json(args.radio_configuration),
        read_json(args.radio_waveform_oracle),
        read_json(args.radio_pulse_oracle),
        read_json(args.independent_radio_pulses),
        expected_events=args.expected_events,
        alpha=args.familywise_alpha,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output / "ground_radio_validation_conclusion.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    atomic_write(args.output / "README.md", markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
