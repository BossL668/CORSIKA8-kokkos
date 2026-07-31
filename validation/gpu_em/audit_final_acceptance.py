#!/usr/bin/env python3
"""Fail-closed audit of the evidence used for CUDA EM final acceptance.

The auditor does not run simulations and does not infer success from a file
name.  A YAML specification names every required artifact and the exact
machine-readable fields that constitute evidence.  Missing artifacts,
unresolved field paths, non-finite numbers, and unknown operators are failures.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class Artifact:
    requested_path: str
    path: Path
    format: str
    sha256: str
    size_bytes: int
    document: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a declarative requirement-to-evidence matrix and emit a "
            "single fail-closed final-acceptance report."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 unless every required requirement passes.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_document(path: Path, format_name: str) -> Any:
    if format_name == "json":
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    if format_name == "yaml":
        with path.open("r", encoding="utf-8") as source:
            return yaml.safe_load(source)
    if format_name == "text":
        return path.read_text(encoding="utf-8")
    if format_name == "file":
        return None
    raise ValueError(f"unsupported artifact format: {format_name}")


def load_artifact(spec_root: Path, evidence: dict[str, Any]) -> Artifact:
    requested = str(evidence.get("path", "")).strip()
    if not requested:
        raise ValueError("evidence.path is required")
    path = Path(requested)
    if not path.is_absolute():
        path = spec_root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"evidence file is missing: {path}")
    format_name = str(evidence.get("format", "")).strip().lower()
    if not format_name:
        if path.suffix == ".json":
            format_name = "json"
        elif path.suffix in (".yaml", ".yml"):
            format_name = "yaml"
        else:
            format_name = "file"
    return Artifact(
        requested_path=requested,
        path=path,
        format=format_name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        document=read_document(path, format_name),
    )


def _matching_values(value: Any, token: str) -> list[Any]:
    if isinstance(value, dict):
        if any(character in token for character in "*?["):
            keys = sorted(
                key
                for key in value
                if fnmatch.fnmatchcase(str(key), token)
            )
            return [value[key] for key in keys]
        return [value[token]] if token in value else []
    if isinstance(value, list):
        if token == "*":
            return list(value)
        try:
            index = int(token)
        except ValueError:
            return []
        return [value[index]] if -len(value) <= index < len(value) else []
    return []


def resolve_values(document: Any, field_path: str) -> list[Any]:
    if field_path in ("", "."):
        return [document]
    values = [document]
    for token in field_path.split("."):
        next_values: list[Any] = []
        for value in values:
            next_values.extend(_matching_values(value, token))
        values = next_values
        if not values:
            break
    return values


def finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not accepted as a numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"value is not finite: {value!r}")
    return result


def evaluate_one(value: Any, operator: str, check: dict[str, Any]) -> bool:
    expected = check.get("expected")
    if operator == "equals":
        return value == expected
    if operator == "not_equals":
        return value != expected
    if operator == "truthy":
        return value is True
    if operator == "falsy":
        return value is False
    if operator == "finite":
        finite_number(value)
        return True
    if operator == "less_equal":
        return finite_number(value) <= finite_number(expected)
    if operator == "greater_equal":
        return finite_number(value) >= finite_number(expected)
    if operator == "less":
        return finite_number(value) < finite_number(expected)
    if operator == "greater":
        return finite_number(value) > finite_number(expected)
    if operator == "absolute_less_equal":
        return abs(finite_number(value)) <= finite_number(expected)
    if operator == "between":
        if not isinstance(expected, list) or len(expected) != 2:
            raise ValueError("between expects [low, high]")
        number = finite_number(value)
        return finite_number(expected[0]) <= number <= finite_number(
            expected[1]
        )
    if operator == "interval_contains":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("interval_contains requires a two-element value")
        target = finite_number(expected)
        return finite_number(value[0]) <= target <= finite_number(value[1])
    if operator == "contains":
        return expected in value
    if operator == "matches":
        return re.search(str(expected), str(value)) is not None
    raise ValueError(f"unsupported check operator: {operator}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def evaluate_check(
    artifact: Artifact, check: dict[str, Any]
) -> dict[str, Any]:
    field_path = str(check.get("field", "")).strip()
    operator = str(check.get("operator", "")).strip()
    aggregate = str(check.get("aggregate", "all")).strip()
    if operator == "sha256_equals":
        values = [artifact.sha256]
        outcomes = [artifact.sha256 == str(check.get("expected", ""))]
    elif operator == "size_greater_equal":
        values = [artifact.size_bytes]
        outcomes = [
            artifact.size_bytes >= int(check.get("expected", 0))
        ]
    else:
        if artifact.document is None:
            raise ValueError(
                f"operator {operator} requires a parsed document"
            )
        values = resolve_values(artifact.document, field_path)
        if not values:
            raise ValueError(f"field path resolved no values: {field_path}")
        outcomes = [
            evaluate_one(value, operator, check) for value in values
        ]
    if aggregate == "all":
        passed = all(outcomes)
    elif aggregate == "any":
        passed = any(outcomes)
    else:
        raise ValueError(f"unsupported aggregate: {aggregate}")
    return {
        "name": str(check.get("name", field_path or operator)),
        "field": field_path,
        "operator": operator,
        "aggregate": aggregate,
        "expected": json_safe(check.get("expected")),
        "values": [json_safe(value) for value in values],
        "outcomes": outcomes,
        "passed": passed,
    }


def audit_requirement(
    spec_root: Path, requirement: dict[str, Any]
) -> dict[str, Any]:
    identifier = str(requirement.get("id", "")).strip()
    if not identifier:
        raise ValueError("every requirement needs a non-empty id")
    classification = str(
        requirement.get("classification", "required")
    ).strip()
    if classification not in ("required", "diagnostic"):
        raise ValueError(
            f"{identifier}: classification must be required or diagnostic"
        )
    result: dict[str, Any] = {
        "id": identifier,
        "title": str(requirement.get("title", identifier)),
        "classification": classification,
        "passed": False,
    }
    try:
        evidence = requirement.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be a mapping")
        artifact = load_artifact(spec_root, evidence)
        checks_spec = requirement.get("checks")
        if not isinstance(checks_spec, list) or not checks_spec:
            raise ValueError("at least one check is required")
        checks = [
            evaluate_check(artifact, check)
            for check in checks_spec
            if isinstance(check, dict)
        ]
        if len(checks) != len(checks_spec):
            raise ValueError("every check must be a mapping")
        result.update(
            {
                "artifact": {
                    "requested_path": artifact.requested_path,
                    "resolved_path": str(artifact.path),
                    "format": artifact.format,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                },
                "checks": checks,
                "passed": all(check["passed"] for check in checks),
            }
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def load_spec(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"acceptance specification is missing: {resolved}")
    with resolved.open("r", encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict):
        raise ValueError("acceptance specification must be a mapping")
    if document.get("schema_version") != 1:
        raise ValueError("acceptance specification schema_version must be 1")
    requirements = document.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("acceptance specification has no requirements")
    return document


def audit(spec_path: Path) -> dict[str, Any]:
    resolved = spec_path.resolve()
    specification = load_spec(resolved)
    requirements = [
        audit_requirement(resolved.parent, requirement)
        for requirement in specification["requirements"]
        if isinstance(requirement, dict)
    ]
    if len(requirements) != len(specification["requirements"]):
        raise ValueError("every requirement must be a mapping")
    identifiers = [requirement["id"] for requirement in requirements]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("requirement ids must be unique")
    required = [
        requirement
        for requirement in requirements
        if requirement["classification"] == "required"
    ]
    passed = bool(required) and all(
        requirement["passed"] for requirement in required
    )
    return {
        "schema_version": 1,
        "title": str(
            specification.get("title", "CUDA EM final acceptance")
        ),
        "specification": {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        },
        "status": "passed" if passed else "failed",
        "required_requirements": len(required),
        "passed_required_requirements": sum(
            requirement["passed"] for requirement in required
        ),
        "diagnostic_requirements": sum(
            requirement["classification"] == "diagnostic"
            for requirement in requirements
        ),
        "requirements": requirements,
    }


def main() -> int:
    args = parse_args()
    try:
        result = audit(args.spec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.require_pass and result["status"] != "passed":
            return 2
        return 0
    except Exception as error:
        print(f"final acceptance audit failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
