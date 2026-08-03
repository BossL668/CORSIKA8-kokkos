#!/usr/bin/env python3
"""Compare two CUDA batches generated from the same shower seed range.

The comparison is deliberately stricter than an ensemble test.  It validates
the normalized command and immutable configuration, then fingerprints the
logical values of every production Parquet output.  Parquet values are read in
bounded-size batches, so multi-million-row ground-particle files do not need to
be loaded into memory at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


PARQUET_ARTIFACTS = (
    "profile/profile.parquet",
    "production_profile/profile.parquet",
    "energyloss/dEdX.parquet",
    "interactions/interactions.parquet",
    "particles/particles.parquet",
    "CoREAS/observers.parquet",
    "ZHS/observers.parquet",
)

YAML_ARTIFACTS = (
    "gpu_em/config.yaml",
    "CoREAS/config.yaml",
    "ZHS/config.yaml",
    "primary/summary.yaml",
    "interactions/summary.yaml",
    "particles/summary.yaml",
    "production_profile/summary.yaml",
    "energyloss/summary.yaml",
    "CoREAS/summary.yaml",
    "ZHS/summary.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference batch root or its cuda output directory",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate batch root or its cuda output directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=65536)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_cuda_root(path: Path) -> Path:
    root = path.resolve()
    if (root / "cuda" / "summary.yaml").is_file():
        root = root / "cuda"
    if not (root / "summary.yaml").is_file():
        raise ValueError(f"not a complete CUDA output root: {root}")
    return root


def normalized_command(provenance: dict[str, Any]) -> list[str]:
    command = list(provenance.get("command", []))
    if not command:
        raise ValueError("validation provenance has no command")
    command[0] = "<executable>"
    try:
        output_index = command.index("-f") + 1
    except ValueError as error:
        raise ValueError("CUDA command has no -f output option") from error
    if output_index >= len(command):
        raise ValueError("CUDA command has an empty -f output option")
    command[output_index] = "<output>"
    return command


def schema_description(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def logical_parquet_fingerprint(
    path: Path,
    batch_rows: int = 65536,
) -> dict[str, Any]:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    column_hashes = []
    for field in schema:
        digest = hashlib.sha256()
        digest.update(field.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(field.type).encode("utf-8"))
        column_hashes.append(digest)

    rows_read = 0
    for batch in parquet.iter_batches(batch_size=batch_rows):
        rows_read += batch.num_rows
        for index, array in enumerate(batch.columns):
            if array.null_count:
                raise ValueError(
                    f"null values are not supported in strict artifact "
                    f"fingerprints: {path}, column={schema[index].name}"
                )
            values = array.to_numpy(zero_copy_only=False)
            if values.dtype.kind not in "biufc":
                raise ValueError(
                    f"unsupported Parquet type in {path}: "
                    f"{schema[index].name}={values.dtype}"
                )
            contiguous = np.ascontiguousarray(values)
            column_hashes[index].update(contiguous.view(np.uint8).tobytes())

    expected_rows = parquet.metadata.num_rows
    if rows_read != expected_rows:
        raise ValueError(
            f"Parquet row-count mismatch while reading {path}: "
            f"{rows_read} != {expected_rows}"
        )
    columns = {
        field.name: digest.hexdigest()
        for field, digest in zip(schema, column_hashes)
    }
    logical = hashlib.sha256(
        json.dumps(
            {
                "schema": schema_description(schema),
                "rows": expected_rows,
                "columns": columns,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "logical_sha256": logical,
        "rows": expected_rows,
        "row_groups": parquet.num_row_groups,
        "schema": schema_description(schema),
        "column_sha256": columns,
    }


def load_batch(root: Path) -> dict[str, Any]:
    summary = yaml.safe_load((root / "summary.yaml").read_text(encoding="utf-8"))
    provenance = json.loads(
        (root / "validation_provenance.json").read_text(encoding="utf-8")
    )
    configuration = yaml.safe_load(
        (root / "config.yaml").read_text(encoding="utf-8")
    )
    normalized_configuration = dict(configuration)
    normalized_configuration.pop("path", None)
    normalized_configuration.pop("args", None)
    return {
        "root": str(root),
        "showers": int(summary["showers"]),
        "seed": int(summary["seed"]),
        "executable_sha256": provenance["executable"]["sha256"],
        "table_sha256": provenance["table"]["sha256"],
        "runner_sha256": provenance["runner"]["sha256"],
        "normalized_command": normalized_command(provenance),
        "normalized_configuration": normalized_configuration,
    }


def compare_batches(
    reference_root: Path,
    candidate_root: Path,
    batch_rows: int,
) -> dict[str, Any]:
    reference = load_batch(reference_root)
    candidate = load_batch(candidate_root)
    identity_checks = {
        "seed_equal": reference["seed"] == candidate["seed"],
        "showers_equal": reference["showers"] == candidate["showers"],
        "command_equal_after_path_normalization": (
            reference["normalized_command"] == candidate["normalized_command"]
        ),
        "configuration_equal_after_path_normalization": (
            reference["normalized_configuration"]
            == candidate["normalized_configuration"]
        ),
        "table_sha256_equal": (
            reference["table_sha256"] == candidate["table_sha256"]
        ),
        "runner_sha256_equal": (
            reference["runner_sha256"] == candidate["runner_sha256"]
        ),
    }

    yaml_results: dict[str, Any] = {}
    for relative in YAML_ARTIFACTS:
        reference_path = reference_root / relative
        candidate_path = candidate_root / relative
        reference_value = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        candidate_value = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        yaml_results[relative] = {
            "equal": reference_value == candidate_value,
            "reference_sha256": sha256_file(reference_path),
            "candidate_sha256": sha256_file(candidate_path),
        }

    parquet_results: dict[str, Any] = {}
    for relative in PARQUET_ARTIFACTS:
        reference_fingerprint = logical_parquet_fingerprint(
            reference_root / relative, batch_rows
        )
        candidate_fingerprint = logical_parquet_fingerprint(
            candidate_root / relative, batch_rows
        )
        parquet_results[relative] = {
            "logical_values_equal": (
                reference_fingerprint["logical_sha256"]
                == candidate_fingerprint["logical_sha256"]
            ),
            "physical_files_equal": (
                reference_fingerprint["file_sha256"]
                == candidate_fingerprint["file_sha256"]
            ),
            "reference": reference_fingerprint,
            "candidate": candidate_fingerprint,
        }

    passed = (
        all(identity_checks.values())
        and all(item["equal"] for item in yaml_results.values())
        and all(
            item["logical_values_equal"] for item in parquet_results.values()
        )
    )
    return {
        "schema_version": 1,
        "purpose": "fixed-seed CUDA build-equivalence audit",
        "reference": reference,
        "candidate": candidate,
        "identity_checks": identity_checks,
        "yaml_artifacts": yaml_results,
        "parquet_artifacts": parquet_results,
        "passed": passed,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    reference = report["reference"]
    candidate = report["candidate"]
    lines = [
        "# Fixed-seed CUDA build-equivalence audit",
        "",
        f"- Seed range: `{reference['seed']}.."
        f"{reference['seed'] + reference['showers'] - 1}`",
        f"- Showers: {reference['showers']}",
        f"- Reference executable: `{reference['executable_sha256']}`",
        f"- Candidate executable: `{candidate['executable_sha256']}`",
        f"- Result: **{'passed' if report['passed'] else 'failed'}**",
        "",
        "| Artifact | Logical values equal | Physical file equal | Rows |",
        "|---|---|---|---:|",
    ]
    for relative, result in report["parquet_artifacts"].items():
        lines.append(
            f"| `{relative}` | {result['logical_values_equal']} | "
            f"{result['physical_files_equal']} | "
            f"{result['reference']['rows']} |"
        )
    lines.extend(["", "## Identity checks", ""])
    for name, passed in report["identity_checks"].items():
        lines.append(f"- `{name}`: {passed}")
    lines.extend(["", "## YAML configuration", ""])
    for relative, result in report["yaml_artifacts"].items():
        lines.append(f"- `{relative}`: {result['equal']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    reference = resolve_cuda_root(args.reference)
    candidate = resolve_cuda_root(args.candidate)
    report = compare_batches(reference, candidate, args.batch_rows)
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output / "README.md", report)
    print(json.dumps({
        "seed": report["reference"]["seed"],
        "showers": report["reference"]["showers"],
        "passed": report["passed"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
