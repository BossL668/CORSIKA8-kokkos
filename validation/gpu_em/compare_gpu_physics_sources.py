#!/usr/bin/env python3
"""Compare independent c8emrt and proposal-native CUDA shower ensembles.

This is deliberately separate from ``compare_cuda_build_strata.py``.  A
build-stratum pooling audit must require one identical physics table, whereas
this comparison is specifically intended to test two different GPU physics
representations of the same requested shower configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .compare_ensembles import (
        DEFAULT_KEY_SCALAR_METRICS,
        compare_ensembles,
        concatenate_ensembles,
        extract_ensemble,
        infer_legacy_c8emrt_source,
        read_validation_provenance,
        read_yaml,
    )
except ImportError:  # Direct script execution from validation/gpu_em.
    from compare_ensembles import (
        DEFAULT_KEY_SCALAR_METRICS,
        compare_ensembles,
        concatenate_ensembles,
        extract_ensemble,
        infer_legacy_c8emrt_source,
        read_validation_provenance,
        read_yaml,
    )


REFERENCE_SOURCE = "c8emrt"
CANDIDATE_SOURCE = "proposal-native"
GROUND_DIAGNOSTICS = (
    "ground_muon_weighted_count",
    "ground_muon_kinetic_energy_GeV",
    "ground_hadron_weighted_count",
    "ground_hadron_kinetic_energy_GeV",
)
SELECTED_LOSS_REASON_NAMES = (
    "native_selection_replay",
    "inverse_cdf_unavailable",
    "loss_energy_out_of_range",
    "loss_quantile_out_of_range",
)
GENERIC_SCALAR_REASON_NAMES = (
    "unsupported_particle",
    "unsupported_medium",
    "unsupported_geometry",
)
PERMITTED_SPECIFIED_REASON_NAMES = (
    *SELECTED_LOSS_REASON_NAMES,
    "cpu_only_process",
    "epair_rejection_envelope_exceeded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, action="append", required=True,
        help="Completed c8emrt CUDA output; repeat for independent shards.",
    )
    parser.add_argument(
        "--candidate", type=Path, action="append", required=True,
        help=(
            "Completed proposal-native CUDA output; repeat for independent "
            "shards."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-events", type=int, default=2)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sigma-limit", type=float, default=3.0)
    parser.add_argument("--active-fraction", type=float, default=1.0e-4)
    parser.add_argument("--minimum-bin-pass-fraction", type=float, default=0.95)
    parser.add_argument(
        "--fail-on-acceptance",
        action="store_true",
        help="Return exit code 2 when the inherited ensemble gate fails.",
    )
    parser.add_argument(
        "--allow-legacy-c8emrt-source-inference",
        action="store_true",
        help=(
            "Explicitly permit strict c8emrt identity inference for reference "
            "outputs which predate gpu_physics_source metadata. Provenance, "
            "commands, table paths, and legacy table metadata must all agree."
        ),
    )
    return parser.parse_args()


def require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"invalid {label} SHA-256: {value!r}")
    return text


def require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def summarize_fallbacks(statistics: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "cpu_generic_fallbacks": 0,
        "cpu_completed_selected_losses": 0,
        "cpu_completed_native_selection_replays": 0,
        "cpu_specified_final_states": 0,
        "deferred_cpu_fallbacks_queued": 0,
        "deferred_cpu_fallbacks_flushed": 0,
    }
    by_reason: dict[str, int] = {}
    for shower in statistics:
        for key in counters:
            value = shower.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid fallback counter {key}={value!r}")
            counters[key] += value
        total_fallbacks = (
            int(shower["cpu_generic_fallbacks"])
            + int(shower["cpu_specified_final_states"])
        )
        reasons = shower.get("cpu_fallbacks_by_reason_name")
        if reasons is None and total_fallbacks == 0:
            reasons = {}
        if not isinstance(reasons, dict):
            raise ValueError("CPU fallback reason mapping is missing")
        process_names = shower.get("cpu_fallbacks_by_process_name")
        process_ids = shower.get("cpu_fallbacks_by_process")
        reason_ids = shower.get("cpu_fallbacks_by_reason")
        mappings = {
            "reason-name": reasons,
            "reason-id": reason_ids,
            "process-name": process_names,
            "process-id": process_ids,
        }
        for label, mapping in mappings.items():
            if mapping is None and total_fallbacks == 0:
                mapping = {}
            if not isinstance(mapping, dict):
                raise ValueError(f"CPU fallback {label} mapping is missing")
            mapping_total = 0
            for key, value in mapping.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"invalid CPU fallback {label} counter {key}={value!r}"
                    )
                mapping_total += value
            if mapping_total != total_fallbacks:
                raise ValueError(
                    f"CPU fallback {label} mapping is not closed: "
                    f"sum={mapping_total}, expected={total_fallbacks}"
                )
        for reason, value in reasons.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"invalid CPU fallback reason counter {reason}={value!r}"
                )
            by_reason[str(reason)] = by_reason.get(str(reason), 0) + value

        known_reasons = set(GENERIC_SCALAR_REASON_NAMES) | set(
            PERMITTED_SPECIFIED_REASON_NAMES
        )
        unknown_reasons = sorted(set(map(str, reasons)) - known_reasons)
        if unknown_reasons:
            raise ValueError(
                f"unpermitted CPU fallback reasons: {unknown_reasons}"
            )
        generic_shower = sum(
            reasons.get(reason, 0) for reason in GENERIC_SCALAR_REASON_NAMES
        )
        specified_shower = sum(
            reasons.get(reason, 0) for reason in PERMITTED_SPECIFIED_REASON_NAMES
        )
        if generic_shower != shower["cpu_generic_fallbacks"]:
            raise ValueError(
                "generic fallback count is not closed by declared scalar reasons"
            )
        if specified_shower != shower["cpu_specified_final_states"]:
            raise ValueError(
                "specified fallback count is not closed by permitted reasons"
            )
        if (
            shower["deferred_cpu_fallbacks_queued"]
            != shower["cpu_specified_final_states"]
            or shower["deferred_cpu_fallbacks_flushed"]
            != shower["cpu_specified_final_states"]
        ):
            raise ValueError(
                "specified/deferred CPU fallback counters are not closed"
            )
        selected_shower = sum(
            reasons.get(reason, 0) for reason in SELECTED_LOSS_REASON_NAMES
        )
        if selected_shower != shower["cpu_completed_selected_losses"]:
            raise ValueError(
                "selected-loss completion count is not closed by reason counters"
            )
        if reasons.get("native_selection_replay", 0) != shower[
            "cpu_completed_native_selection_replays"
        ]:
            raise ValueError(
                "native-selection replay count differs from reason map"
            )

    generic_by_reason = {
        reason: by_reason.get(reason, 0)
        for reason in GENERIC_SCALAR_REASON_NAMES
    }
    if counters["cpu_generic_fallbacks"] != sum(generic_by_reason.values()):
        raise ValueError(
            "generic fallback count is not closed by declared scalar reasons"
        )
    selected_by_reason = {
        reason: by_reason.get(reason, 0)
        for reason in SELECTED_LOSS_REASON_NAMES
    }
    if counters["cpu_completed_selected_losses"] != sum(
        selected_by_reason.values()
    ):
        raise ValueError(
            "selected-loss completion count is not closed by reason counters"
        )
    if counters["cpu_completed_native_selection_replays"] != by_reason.get(
        "native_selection_replay", 0
    ):
        raise ValueError("native-selection replay count differs from reason map")
    if counters["deferred_cpu_fallbacks_queued"] != counters[
        "deferred_cpu_fallbacks_flushed"
    ]:
        raise ValueError("deferred CPU fallback queue did not drain exactly")
    if counters["cpu_specified_final_states"] != counters[
        "deferred_cpu_fallbacks_flushed"
    ]:
        raise ValueError("specified final-state count differs from flushed fallbacks")
    return {
        **counters,
        "generic_by_reason": generic_by_reason,
        "selected_loss_by_reason": selected_by_reason,
        "by_reason_name": dict(sorted(by_reason.items())),
    }


def gpu_source_identity(
    root: Path,
    expected_source: str,
    *,
    allow_legacy_c8emrt_source_inference: bool = False,
) -> dict[str, Any]:
    """Read and cross-check the immutable GPU physics-source identity."""

    configuration = read_yaml(root / "gpu_em" / "config.yaml")
    observed_source = configuration.get("gpu_physics_source")
    legacy_inference: dict[str, Any] | None = None
    if observed_source is None and allow_legacy_c8emrt_source_inference:
        if expected_source != REFERENCE_SOURCE:
            raise ValueError(
                "legacy c8emrt source inference is only valid for the "
                "c8emrt reference"
            )
        legacy_inference = infer_legacy_c8emrt_source(root)
        observed_source = legacy_inference["inferred_source"]
    if observed_source != expected_source:
        raise ValueError(
            f"GPU physics source differs in {root}: expected "
            f"{expected_source!r}, observed {observed_source!r}"
        )
    provenance = read_validation_provenance(root, "cuda")
    assert provenance is not None
    executable_sha256 = require_sha256(
        provenance["executable"]["sha256"], "executable"
    )
    provenance_table_sha256 = require_sha256(
        provenance["table"]["sha256"], "provenance table"
    )

    summary = read_yaml(root / "gpu_em" / "summary.yaml")
    if not summary:
        raise ValueError(f"empty GPU summary in {root}")
    statistics: list[dict[str, Any]] = []
    for shower, record in summary.items():
        if not isinstance(record, dict) or not isinstance(
            record.get("statistics"), dict
        ):
            raise ValueError(f"invalid GPU statistics for {shower} in {root}")
        item = record["statistics"]
        if legacy_inference is not None:
            if "gpu_physics_source" in item:
                raise ValueError(
                    f"per-shower GPU source unexpectedly appears for legacy "
                    f"{shower} in {root}"
                )
            # This counter was introduced together with proposal-native.  Its
            # absence in a strictly inferred pre-native c8emrt record means
            # exactly zero; do not relax this for modern records.
            normalized_item = dict(item)
            normalized_item["cpu_completed_native_selection_replays"] = 0
            item = normalized_item
        elif item.get("gpu_physics_source") != expected_source:
            raise ValueError(
                f"per-shower GPU physics source differs for {shower} in {root}"
            )
        statistics.append(item)

    table_configuration = configuration.get("table")
    if not isinstance(table_configuration, dict):
        raise ValueError(f"missing GPU table configuration in {root}")

    identity: dict[str, Any] = {
        "root": str(root),
        "events": len(statistics),
        "gpu_physics_source": expected_source,
        "executable_sha256": executable_sha256,
        # c8emrt provenance normally hashes the serialized file, whereas its
        # config hashes canonical table content.  Record both and do not claim
        # that these two intentionally different hash domains are equal.
        "provenance_table_sha256": provenance_table_sha256,
        "fallbacks": summarize_fallbacks(statistics),
        "gpu_physics_source_inferred": legacy_inference is not None,
        "legacy_c8emrt_source_inference": legacy_inference,
    }
    if expected_source == REFERENCE_SOURCE:
        identity.update(
            {
                "physics_table_sha256": require_sha256(
                    table_configuration.get("sha256"), "c8emrt content"
                ),
                "proposal_version": str(
                    table_configuration.get("proposal_version", "unknown")
                ),
                "table_format_version": table_configuration.get(
                    "format_version"
                ),
                "auxiliary_sha256": None,
            }
        )
        return identity

    native_records: list[dict[str, Any]] = []
    for item in statistics:
        native = item.get("proposal_native")
        if not isinstance(native, dict):
            raise ValueError(f"missing proposal-native statistics in {root}")
        native_records.append(native)
    table_hashes = {
        require_sha256(item.get("table_sha256"), "proposal-native table")
        for item in native_records
    }
    auxiliary_hashes = {
        require_sha256(item.get("aux_sha256"), "proposal-native auxiliary")
        for item in native_records
    }
    node_counts = {int(item.get("node_count", -1)) for item in native_records}
    device_bytes = {int(item.get("device_bytes", -1)) for item in native_records}
    if len(table_hashes) != 1 or len(auxiliary_hashes) != 1:
        raise ValueError(f"proposal-native hashes vary between showers in {root}")
    if len(node_counts) != 1 or min(node_counts) <= 0:
        raise ValueError(f"proposal-native node count varies or is invalid in {root}")
    if len(device_bytes) != 1 or min(device_bytes) <= 0:
        raise ValueError(f"proposal-native device bytes vary or are invalid in {root}")
    identity.update(
        {
            "physics_table_sha256": next(iter(table_hashes)),
            "auxiliary_sha256": next(iter(auxiliary_hashes)),
            "proposal_version": str(
                table_configuration.get("proposal_version", "unknown")
            ),
            "cubic_interpolation_version": str(
                table_configuration.get(
                    "cubic_interpolation_version", "unknown"
                )
            ),
            "node_count": next(iter(node_counts)),
            "device_bytes": next(iter(device_bytes)),
            "proposal_cache_all_hit_events": sum(
                bool(item.get("proposal_cache_all_hit"))
                for item in native_records
            ),
            "aux_cache_hit_events": sum(
                bool(item.get("aux_cache_hit")) for item in native_records
            ),
            "inverse_failures": sum(
                int(item.get("inverse_failures", 0)) for item in native_records
            ),
        }
    )
    return identity


def stable_source_identity(identity: dict[str, Any]) -> tuple[Any, ...]:
    """Fields which must be identical when independent shards are pooled."""

    return (
        identity["gpu_physics_source"],
        identity["executable_sha256"],
        identity["provenance_table_sha256"],
        identity["physics_table_sha256"],
        identity.get("auxiliary_sha256"),
        identity.get("proposal_version"),
        identity.get("cubic_interpolation_version"),
        identity.get("table_format_version"),
        identity.get("node_count"),
        identity.get("gpu_physics_source_inferred", False),
    )


def require_uniform_identities(
    identities: list[dict[str, Any]], label: str
) -> None:
    reference = stable_source_identity(identities[0])
    for identity in identities[1:]:
        if stable_source_identity(identity) != reference:
            raise ValueError(
                f"{label} GPU physics-source identity differs between "
                f"{identities[0]['root']} and {identity['root']}"
            )


def signed_shift(result: dict[str, Any]) -> float | None:
    reference = float(result["proposal"]["mean"])
    if reference == 0.0:
        return 0.0 if float(result["difference"]) == 0.0 else None
    return float(result["difference"]) / abs(reference)


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.3f}%"


def markdown(report: dict[str, Any]) -> str:
    reference = report["physics_sources"]["reference"]["shards"][0]
    candidate = report["physics_sources"]["candidate"]["shards"][0]
    acceptance = report["acceptance"]
    reference_fallbacks = {
        key: sum(
            int(shard["fallbacks"].get(key, 0))
            for shard in report["physics_sources"]["reference"]["shards"]
        )
        for key in (
            "cpu_generic_fallbacks",
            "cpu_completed_selected_losses",
            "cpu_completed_native_selection_replays",
            "cpu_specified_final_states",
        )
    }
    candidate_fallbacks = {
        key: sum(
            int(shard["fallbacks"].get(key, 0))
            for shard in report["physics_sources"]["candidate"]["shards"]
        )
        for key in reference_fallbacks
    }
    lines = [
        "# c8emrt 与 proposal-native GPU 系综验收",
        "",
        "## 结论",
        "",
        (
            f"- 总状态：**{report['status']}**；标量门禁："
            f"{acceptance['scalar_pass']}；曲线门禁："
            f"{acceptance['curve_pass']}。"
        ),
        (
            f"- 独立 shower 数：c8emrt {report['reference']['events']}，"
            f"proposal-native {report['candidate']['events']}。"
        ),
        "- 两侧均通过 CUDA 完整性检查，并严格使用同一个可执行文件。",
        "- 两种 GPU 物理源的表哈希属于不同表示，不要求彼此相等。",
        (
            "- c8emrt source 身份：旧格式严格推断（已记录完整证据链）。"
            if reference.get("gpu_physics_source_inferred") is True
            else "- c8emrt source 身份：输出显式记录。"
        ),
        "",
        "## 身份与可重复性",
        "",
        "| 项目 | c8emrt | proposal-native |",
        "|---|---|---|",
        (
            f"| executable SHA-256 | `{reference['executable_sha256']}` | "
            f"`{candidate['executable_sha256']}` |"
        ),
        (
            f"| physics-table SHA-256 | `{reference['physics_table_sha256']}` | "
            f"`{candidate['physics_table_sha256']}` |"
        ),
        (
            f"| provenance table SHA-256 | "
            f"`{reference['provenance_table_sha256']}` | "
            f"`{candidate['provenance_table_sha256']}` |"
        ),
        (
            f"| auxiliary SHA-256 | n/a | "
            f"`{candidate['auxiliary_sha256']}` |"
        ),
        "",
        "## CPU completion 与显式回退（按物理源分列）",
        "",
        "| Counter | c8emrt | proposal-native |",
        "|---|---:|---:|",
        (
            "| generic scalar fallback | "
            f"{reference_fallbacks['cpu_generic_fallbacks']} | "
            f"{candidate_fallbacks['cpu_generic_fallbacks']} |"
        ),
        (
            "| completed selected loss | "
            f"{reference_fallbacks['cpu_completed_selected_losses']} | "
            f"{candidate_fallbacks['cpu_completed_selected_losses']} |"
        ),
        (
            "| native selection replay | "
            f"{reference_fallbacks['cpu_completed_native_selection_replays']} | "
            f"{candidate_fallbacks['cpu_completed_native_selection_replays']} |"
        ),
        (
            "| specified CPU final state | "
            f"{reference_fallbacks['cpu_specified_final_states']} | "
            f"{candidate_fallbacks['cpu_specified_final_states']} |"
        ),
        "",
        "## 关键 shower 标量",
        "",
        "| Observable | proposal-native/c8emrt 均值偏移 | |z| | 判定 |",
        "|---|---:|---:|---|",
    ]
    metrics = [
        *DEFAULT_KEY_SCALAR_METRICS,
        *(metric for metric in GROUND_DIAGNOSTICS if metric in report["scalars"]),
    ]
    for metric in metrics:
        result = report["scalars"][metric]
        lines.append(
            f"| `{metric}` | {format_percent(signed_shift(result))} | "
            f"{float(result['absolute_z_score']):.3f} | "
            f"{result['outcome']} |"
        )
    gated_curves = {
        name: value
        for name, value in report["curves"].items()
        if value.get("acceptance_gate") is True
    }
    failed_curves = [
        name for name, value in gated_curves.items() if value.get("passed") is not True
    ]
    lines.extend(
        [
            "",
            "## 纵向与地面曲线",
            "",
            f"- 正式曲线数：{len(gated_curves)}。",
            f"- 未通过曲线数：{len(failed_curves)}。",
            (
                "- 未通过项：" + ", ".join(f"`{name}`" for name in failed_curves)
                if failed_curves
                else "- 所有正式曲线均通过当前统计门禁。"
            ),
            "",
            "## 文件语义",
            "",
            (
                "底层通用比较器历史上使用 `proposal`/`cuda` 字段名。"
                "在本目录中它们严格映射为 `c8emrt` reference 与 "
                "`proposal-native` candidate；CSV 同时写出真实 "
                "`gpu_physics_source`，不得解释为 CPU/GPU 比较。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def validate_args(args: argparse.Namespace) -> None:
    if args.minimum_events < 2:
        raise ValueError("--minimum-events must be at least two")
    if not 0.0 < args.relative_tolerance < 1.0:
        raise ValueError("--relative-tolerance must be in (0, 1)")
    if not math.isfinite(args.sigma_limit) or args.sigma_limit <= 0.0:
        raise ValueError("--sigma-limit must be positive and finite")
    if not 0.0 <= args.active_fraction < 1.0:
        raise ValueError("--active-fraction must be in [0, 1)")
    if not 0.0 < args.minimum_bin_pass_fraction <= 1.0:
        raise ValueError("--minimum-bin-pass-fraction must be in (0, 1]")


def main() -> int:
    args = parse_args()
    validate_args(args)
    reference_paths = [
        require_directory(path, "c8emrt reference") for path in args.reference
    ]
    candidate_paths = [
        require_directory(path, "proposal-native candidate")
        for path in args.candidate
    ]
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")

    reference_identities = [
        gpu_source_identity(
            path,
            REFERENCE_SOURCE,
            allow_legacy_c8emrt_source_inference=(
                args.allow_legacy_c8emrt_source_inference
            ),
        )
        for path in reference_paths
    ]
    candidate_identities = [
        gpu_source_identity(path, CANDIDATE_SOURCE) for path in candidate_paths
    ]
    require_uniform_identities(reference_identities, REFERENCE_SOURCE)
    require_uniform_identities(candidate_identities, CANDIDATE_SOURCE)

    reference = concatenate_ensembles(
        REFERENCE_SOURCE,
        [
            extract_ensemble(
                REFERENCE_SOURCE,
                path,
                expect_gpu=True,
                permitted_generic_fallback_reasons=GENERIC_SCALAR_REASON_NAMES,
                allow_legacy_c8emrt_source_inference=(
                    args.allow_legacy_c8emrt_source_inference
                ),
            )
            for path in reference_paths
        ],
    )
    candidate = concatenate_ensembles(
        CANDIDATE_SOURCE,
        [
            extract_ensemble(
                CANDIDATE_SOURCE,
                path,
                expect_gpu=True,
                permitted_generic_fallback_reasons=GENERIC_SCALAR_REASON_NAMES,
            )
            for path in candidate_paths
        ],
    )
    if len(reference.showers) < args.minimum_events:
        raise ValueError("c8emrt reference has too few completed showers")
    if len(candidate.showers) < args.minimum_events:
        raise ValueError("proposal-native candidate has too few completed showers")
    if (
        reference.metadata["physics_configuration"]
        != candidate.metadata["physics_configuration"]
    ):
        raise ValueError("GPU physics sources use different shower configurations")
    reference_provenance = reference.metadata["provenance_fingerprint"]
    candidate_provenance = candidate.metadata["provenance_fingerprint"]
    if reference_provenance[2] != candidate_provenance[2]:
        raise ValueError("GPU physics sources use different executable SHA-256 values")

    report, curve_rows = compare_ensembles(
        reference,
        candidate,
        args.relative_tolerance,
        args.sigma_limit,
        args.active_fraction,
        args.minimum_bin_pass_fraction,
        DEFAULT_KEY_SCALAR_METRICS,
        allow_cross_build_reference=False,
    )
    physical_configuration = list(reference.metadata["physics_configuration"])
    report["comparison_semantics"] = {
        "reference": REFERENCE_SOURCE,
        "candidate": CANDIDATE_SOURCE,
        "legacy_schema_mapping": {
            "proposal": REFERENCE_SOURCE,
            "cuda": CANDIDATE_SOURCE,
        },
        "independent_shower_ensembles": True,
        "same_executable_required": True,
        "same_physical_configuration_required": True,
        "different_physics_table_hashes_allowed": True,
        "legacy_c8emrt_source_inference_opt_in": (
            args.allow_legacy_c8emrt_source_inference
        ),
        "legacy_c8emrt_source_inference_used": any(
            identity.get("gpu_physics_source_inferred") is True
            for identity in reference_identities
        ),
    }
    report["physics_configuration_sha256"] = hashlib.sha256(
        json.dumps(
            physical_configuration,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report["physics_sources"] = {
        "reference": {
            "source": REFERENCE_SOURCE,
            "shards": reference_identities,
        },
        "candidate": {
            "source": CANDIDATE_SOURCE,
            "shards": candidate_identities,
        },
    }
    # Source-neutral aliases make the top-level JSON self-explanatory while
    # retaining the inherited proposal/cuda keys for existing plot readers.
    report["reference"] = report["proposal"]
    report["candidate"] = report["cuda"]

    output.mkdir(parents=True)
    scalar_rows = pd.concat(
        [
            reference.scalars.assign(
                backend="proposal",
                backend_role="reference",
                gpu_physics_source=REFERENCE_SOURCE,
            ),
            candidate.scalars.assign(
                backend="cuda",
                backend_role="candidate",
                gpu_physics_source=CANDIDATE_SOURCE,
            ),
        ]
    ).reset_index()
    scalar_rows.to_csv(output / "per_shower_observables.csv", index=False)
    curve_rows = curve_rows.copy()
    for legacy, neutral in (
        ("proposal_mean", "reference_mean"),
        ("cuda_mean", "candidate_mean"),
        ("proposal_standard_error", "reference_standard_error"),
        ("cuda_standard_error", "candidate_standard_error"),
        ("proposal_events", "reference_events"),
        ("cuda_events", "candidate_events"),
    ):
        if legacy in curve_rows:
            curve_rows[neutral] = curve_rows[legacy]
    curve_rows.to_csv(output / "curve_comparison.csv", index=False)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "README_CN.md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "reference_events": len(reference.showers),
                "candidate_events": len(candidate.showers),
                "output": str(output),
            },
            indent=2,
        )
    )
    if args.fail_on_acceptance and report["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"GPU physics-source comparison failed: {error}")
        raise SystemExit(1)
