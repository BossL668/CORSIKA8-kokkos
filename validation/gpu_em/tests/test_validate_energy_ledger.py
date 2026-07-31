from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from validate_energy_ledger import validate_outputs  # noqa: E402


class EnergyLedgerValidationTests(unittest.TestCase):
    def write_output(
        self,
        root: Path,
        *,
        residual: float = 0.0,
        complete: bool = True,
        accepted: bool = True,
    ) -> None:
        source = 10.5
        terminal = source - residual
        relative = abs(residual) / source
        summary = {
            "shower_0": {
                "complete": True,
                "status": "complete",
                "statistics": {
                    "energy_ledger": {
                        "complete_coverage": complete,
                        "accepted": accepted,
                        "initial_total_GeV": 10.0,
                        "medium_rest_mass_input_GeV": 0.5,
                        "deposited_GeV": terminal - 1.0,
                        "cut_rest_mass_energy_GeV": 1.0,
                        "observed_total_energy_GeV": 0.0,
                        "escaped_total_energy_GeV": 0.0,
                        "source_GeV": source,
                        "terminal_GeV": terminal,
                        "residual_GeV": residual,
                        "relative_closure_error": relative,
                    }
                },
            }
        }
        directory = root / "gpu_em"
        directory.mkdir(parents=True)
        (directory / "summary.yaml").write_text(
            yaml.safe_dump(summary), encoding="utf-8"
        )

    def test_complete_ledger_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_output(root, residual=1.0e-6)
            result = validate_outputs([root], 1.0e-4, 1, True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["complete_events"], 1)

    def test_closure_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_output(
                root, residual=2.0e-3, accepted=False
            )
            result = validate_outputs([root], 1.0e-4, 1, True)
            self.assertEqual(result["status"], "failed")

    def test_partial_ledger_is_rejected_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_output(root, complete=False, accepted=False)
            result = validate_outputs([root], 1.0e-4, 1, True)
            self.assertEqual(result["status"], "failed")

    def test_stored_arithmetic_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_output(root)
            path = root / "gpu_em" / "summary.yaml"
            summary = yaml.safe_load(path.read_text(encoding="utf-8"))
            summary["shower_0"]["statistics"]["energy_ledger"][
                "terminal_GeV"
            ] += 0.1
            path.write_text(
                yaml.safe_dump(summary), encoding="utf-8"
            )
            result = validate_outputs([root], 1.0e-4, 1, True)
            self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
