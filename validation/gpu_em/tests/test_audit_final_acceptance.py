import json
import tempfile
import unittest
from pathlib import Path

import yaml

from validation.gpu_em.audit_final_acceptance import audit


class FinalAcceptanceAuditTest(unittest.TestCase):
    def test_required_and_diagnostic_checks_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "records": {
                            "shower_0": {"complete": True, "error": 1e-5},
                            "shower_1": {"complete": True, "error": 2e-5},
                        },
                        "interval": [0.9, 1.1],
                    }
                ),
                encoding="utf-8",
            )
            spec = root / "acceptance.yaml"
            spec.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "requirements": [
                            {
                                "id": "required",
                                "evidence": {
                                    "path": "evidence.json",
                                    "format": "json",
                                },
                                "checks": [
                                    {
                                        "field": "status",
                                        "operator": "equals",
                                        "expected": "passed",
                                    },
                                    {
                                        "field": "records.shower_*.complete",
                                        "operator": "truthy",
                                    },
                                    {
                                        "field": "records.shower_*.error",
                                        "operator": "less_equal",
                                        "expected": 1e-4,
                                    },
                                    {
                                        "field": "interval",
                                        "operator": "interval_contains",
                                        "expected": 1.0,
                                    },
                                ],
                            },
                            {
                                "id": "diagnostic",
                                "classification": "diagnostic",
                                "evidence": {"path": "evidence.json"},
                                "checks": [
                                    {
                                        "field": "status",
                                        "operator": "equals",
                                        "expected": "not-passed",
                                    }
                                ],
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            result = audit(spec)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["passed_required_requirements"], 1)
            self.assertFalse(result["requirements"][1]["passed"])

    def test_missing_field_and_artifact_fail_required_requirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.yaml"
            evidence.write_text("value: 1\n", encoding="utf-8")
            spec = root / "acceptance.yaml"
            spec.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "requirements": [
                            {
                                "id": "missing-field",
                                "evidence": {"path": "evidence.yaml"},
                                "checks": [
                                    {
                                        "field": "absent",
                                        "operator": "equals",
                                        "expected": 1,
                                    }
                                ],
                            },
                            {
                                "id": "missing-file",
                                "evidence": {"path": "absent.json"},
                                "checks": [
                                    {
                                        "field": "status",
                                        "operator": "equals",
                                        "expected": "passed",
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = audit(spec)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["passed_required_requirements"], 0)
            self.assertIn("resolved no values", result["requirements"][0]["error"])
            self.assertIn("missing", result["requirements"][1]["error"])


if __name__ == "__main__":
    unittest.main()
