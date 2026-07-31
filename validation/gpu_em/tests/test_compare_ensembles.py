#!/usr/bin/env python3

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from compare_ensembles import (  # noqa: E402
    DEFAULT_KEY_SCALAR_METRICS,
    Ensemble,
    canonical_physics_configuration,
    compare_scalar,
    compare_ensembles,
    concatenate_ensembles,
    provenance_fingerprint,
    quadratic_peak,
    read_validation_provenance,
    VALIDATION_PROVENANCE_FILENAME,
)


def make_ensemble(name: str, scale: float = 1.0) -> Ensemble:
    showers = (0, 1, 2, 3)
    scalar_values = {
        "profile_xmax_charged_gcm2": np.asarray([500.0, 510.0, 490.0, 505.0]),
        "profile_charged_max": np.asarray([100.0, 102.0, 98.0, 101.0]),
        "profile_charged_integral": np.asarray([1000.0, 1020.0, 980.0, 1010.0]),
        "profile_photon_integral": np.asarray([1100.0, 1120.0, 1080.0, 1110.0]),
        "energy_deposit_sum_GeV": np.asarray([900.0, 905.0, 895.0, 902.0]),
        "energy_deposit_xmax_gcm2": np.asarray([510.0, 515.0, 505.0, 512.0]),
        "energy_deposit_max_GeV": np.asarray([30.0, 31.0, 29.0, 30.5]),
        "ground_em_weighted_count": np.asarray([20.0, 21.0, 19.0, 20.5]),
        "ground_em_kinetic_energy_GeV": np.asarray([50.0, 52.0, 48.0, 51.0]),
        "ground_radius_mean_m": np.asarray([80.0, 82.0, 78.0, 81.0]),
        "ground_time_rms_s": np.asarray([1.0e-8, 1.1e-8, 0.9e-8, 1.05e-8]),
        "energy_closure_fraction": np.asarray([0.95, 0.96, 0.94, 0.955]),
    }
    scalars = pd.DataFrame(
        {key: value * scale for key, value in scalar_values.items()},
        index=pd.Index(showers, name="shower"),
    )
    coordinate = np.asarray([0.0, 10.0, 20.0])
    matrix = (
        np.asarray(
            [
                [0.0, 10.0, 1.0],
                [0.0, 10.2, 1.1],
                [0.0, 9.8, 0.9],
                [0.0, 10.1, 1.0],
            ]
        )
        * scale
    )
    histogram_edges = np.asarray([0.0, 1.0, 2.0, np.inf])
    histogram = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.2, 0.69, 0.11],
            [0.21, 0.69, 0.1],
            [0.19, 0.71, 0.1],
        ]
    )
    return Ensemble(
        name=name,
        root=Path(f"/{name}"),
        showers=showers,
        scalars=scalars,
        curves={"profile_charged": (coordinate, matrix)},
        histograms={"ground_radial_fraction": (histogram_edges, histogram)},
        metadata={"events": len(showers), "source": f"/{name}"},
    )


class EnsembleComparisonTest(unittest.TestCase):
    def test_validation_provenance_is_strict_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                ValueError,
                "required validation provenance is missing",
            ):
                read_validation_provenance(
                    root,
                    "proposal",
                )
            self.assertIsNone(
                read_validation_provenance(
                    root,
                    "proposal",
                    allow_legacy=True,
                )
            )

            provenance = {
                "schema_version": 1,
                "backend": "proposal",
                "executable": {
                    "sha256": "a" * 64,
                },
                "table": None,
            }
            with (
                root / VALIDATION_PROVENANCE_FILENAME
            ).open("w", encoding="utf-8") as destination:
                json.dump(provenance, destination)
            observed = read_validation_provenance(
                root,
                "proposal",
            )
            self.assertEqual(
                provenance_fingerprint(observed),
                (1, "proposal", "a" * 64, None),
            )

    def test_canonical_configuration_ignores_only_run_and_backend_controls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "proposal"
            cuda = root / "cuda"
            proposal.mkdir()
            cuda.mkdir()
            common = (
                "-p 11 -E 1000 --zenith 45 --emcut 0.0005 "
                "--emthin 0.0001 --max-weight 100"
            )
            with (proposal / "config.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "args": (
                            f"/first/c8 {common} -N 10 -f /first/output "
                            "--seed 1 --em-backend proposal --verbosity warn"
                        )
                    },
                    destination,
                )
            with (cuda / "config.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "args": (
                            f"/second/c8 {common} --nevent=20 "
                            "--filename=/second/output --seed=2 "
                            "--em-backend=cuda --gpu-device 0 "
                            "--gpu-table-cache /table --gpu-detailed-stage-timing"
                        )
                    },
                    destination,
                )
            self.assertEqual(
                canonical_physics_configuration(proposal),
                canonical_physics_configuration(cuda),
            )

            with (cuda / "config.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "args": (
                            f"/second/c8 {common} "
                            "--radio-sampling-rate-ghz 1 "
                            "--radio-window-duration-ns=400 "
                            "--radio-pretrigger-ns 10 "
                            "--gpu-radio-track-diagnostics"
                        )
                    },
                    destination,
                )
            self.assertEqual(
                canonical_physics_configuration(proposal),
                canonical_physics_configuration(cuda),
            )

            with (cuda / "config.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "args": (
                            f"/second/c8 {common} "
                            "--radio-sampling-rate-ghz 10"
                        )
                    },
                    destination,
                )
            self.assertNotEqual(
                canonical_physics_configuration(proposal),
                canonical_physics_configuration(cuda),
            )

            with (cuda / "config.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "args": (
                            f"/second/c8 {common.replace('-E 1000', '-E 2000')} "
                            "-N 20 -f /second/output --seed 2 "
                            "--em-backend cuda"
                        )
                    },
                    destination,
                )
            self.assertNotEqual(
                canonical_physics_configuration(proposal),
                canonical_physics_configuration(cuda),
            )

    def test_mismatched_physics_configuration_is_rejected(self) -> None:
        proposal = make_ensemble("proposal")
        cuda = make_ensemble("cuda")
        proposal.metadata["physics_configuration"] = (
            "--energy",
            "1000",
        )
        cuda.metadata["physics_configuration"] = (
            "--energy",
            "2000",
        )
        with self.assertRaisesRegex(
            ValueError,
            "physics configurations differ",
        ):
            compare_ensembles(
                proposal,
                cuda,
                relative_tolerance=0.01,
                sigma_limit=3.0,
                active_fraction=1.0e-4,
                minimum_bin_pass_fraction=0.95,
            )

    def test_mismatched_shard_configuration_is_rejected(self) -> None:
        first = make_ensemble("first")
        second = make_ensemble("second")
        first.metadata["physics_configuration"] = (
            "--energy",
            "1000",
        )
        second.metadata["physics_configuration"] = (
            "--energy",
            "2000",
        )
        with self.assertRaisesRegex(
            ValueError,
            "shard physics configurations differ",
        ):
            concatenate_ensembles(
                "proposal",
                [first, second],
            )

    def test_mismatched_executable_provenance_is_rejected(self) -> None:
        proposal = make_ensemble("proposal")
        cuda = make_ensemble("cuda")
        proposal.metadata["provenance_fingerprint"] = (
            1,
            "proposal",
            "a" * 64,
            None,
        )
        cuda.metadata["provenance_fingerprint"] = (
            1,
            "cuda",
            "b" * 64,
            "c" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "executable build provenance differs",
        ):
            compare_ensembles(
                proposal,
                cuda,
                relative_tolerance=0.01,
                sigma_limit=3.0,
                active_fraction=1.0e-4,
                minimum_bin_pass_fraction=0.95,
            )

    def test_explicit_cross_build_reference_is_permitted(self) -> None:
        proposal = make_ensemble("proposal")
        cuda = make_ensemble("cuda")
        proposal.metadata["provenance_fingerprint"] = (
            1,
            "proposal",
            "a" * 64,
            None,
        )
        cuda.metadata["provenance_fingerprint"] = (
            1,
            "cuda",
            "b" * 64,
            "c" * 64,
        )
        report, _ = compare_ensembles(
            proposal,
            cuda,
            relative_tolerance=0.01,
            sigma_limit=3.0,
            active_fraction=1.0e-4,
            minimum_bin_pass_fraction=0.95,
            allow_cross_build_reference=True,
        )
        self.assertEqual(report["status"], "passed")

    def test_mismatched_shard_table_provenance_is_rejected(self) -> None:
        first = make_ensemble("cuda_0")
        second = make_ensemble("cuda_1")
        first.metadata["provenance_fingerprint"] = (
            1,
            "cuda",
            "a" * 64,
            "b" * 64,
        )
        second.metadata["provenance_fingerprint"] = (
            1,
            "cuda",
            "a" * 64,
            "c" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "build/table provenance differs",
        ):
            concatenate_ensembles(
                "cuda",
                [first, second],
            )

    def test_default_gate_covers_original_nine_key_scalars(self) -> None:
        self.assertEqual(len(DEFAULT_KEY_SCALAR_METRICS), 9)
        self.assertIn(
            "ground_em_weighted_count",
            DEFAULT_KEY_SCALAR_METRICS,
        )
        self.assertIn(
            "ground_em_kinetic_energy_GeV",
            DEFAULT_KEY_SCALAR_METRICS,
        )

    def test_quadratic_peak(self) -> None:
        x = np.asarray([0.0, 1.0, 2.0])
        y = np.asarray([0.0, 1.0, 0.0])
        self.assertEqual(quadratic_peak(x, y), (1.0, 1.0))

    def test_identical_ensembles_pass(self) -> None:
        proposal = make_ensemble("proposal")
        cuda = make_ensemble("cuda")
        report, curve_rows = compare_ensembles(
            proposal,
            cuda,
            relative_tolerance=0.01,
            sigma_limit=3.0,
            active_fraction=1.0e-4,
            minimum_bin_pass_fraction=0.95,
        )
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["acceptance"]["passed"])
        self.assertFalse(curve_rows.empty)

    def test_identical_zero_metric_passes(self) -> None:
        result = compare_scalar(
            np.zeros(4),
            np.zeros(4),
            relative_tolerance=0.01,
            sigma_limit=3.0,
            gate=True,
        )
        self.assertEqual(result["relative_difference"], 0.0)
        self.assertEqual(result["outcome"], "passed")
        self.assertTrue(result["passed"])

    def test_noisy_relative_failure_is_reported_as_inconclusive(self) -> None:
        result = compare_scalar(
            np.asarray([0.0, 200.0]),
            np.asarray([2.0, 202.0]),
            relative_tolerance=0.01,
            sigma_limit=3.0,
            gate=True,
        )
        self.assertFalse(result["relative_pass"])
        self.assertTrue(result["statistical_pass"])
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["outcome"],
            "relative_threshold_failed_but_statistically_inconclusive",
        )
        self.assertGreater(
            result["sigma_scaled_relative_precision"],
            0.01,
        )
        self.assertIn("distribution_diagnostics", result)
        self.assertIn(
            "bootstrap_signed_relative_mean_shift_95pct",
            result["distribution_diagnostics"],
        )

    def test_scalar_distribution_diagnostics_identical_samples(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        result = compare_scalar(
            values,
            values,
            relative_tolerance=0.01,
            sigma_limit=3.0,
            gate=True,
            bootstrap_repetitions=100,
        )
        diagnostics = result["distribution_diagnostics"]
        self.assertEqual(diagnostics["empirical_KS_distance"], 0.0)
        self.assertEqual(diagnostics["relative_quantile_wasserstein"], 0.0)
        self.assertEqual(diagnostics["signed_relative_median_shift"], 0.0)

    def test_unequal_independent_sample_sizes_are_supported(self) -> None:
        proposal = make_ensemble("proposal")
        cuda = make_ensemble("cuda")
        cuda.showers = cuda.showers[:3]
        cuda.scalars = cuda.scalars.iloc[:3]
        cuda.curves = {
            name: (coordinate, matrix[:3])
            for name, (coordinate, matrix) in cuda.curves.items()
        }
        cuda.histograms = {
            name: (edges, matrix[:3])
            for name, (edges, matrix) in cuda.histograms.items()
        }
        cuda.metadata["events"] = 3
        report, _ = compare_ensembles(
            proposal,
            cuda,
            relative_tolerance=0.01,
            sigma_limit=3.0,
            active_fraction=1.0e-4,
            minimum_bin_pass_fraction=0.95,
        )
        self.assertEqual(report["proposal"]["events"], 4)
        self.assertEqual(report["cuda"]["events"], 3)
        self.assertIn(report["status"], ("passed", "failed"))

    def test_two_percent_bias_fails_key_metrics_and_curve(self) -> None:
        proposal = make_ensemble("proposal")
        cuda = make_ensemble("cuda", scale=1.02)
        report, _ = compare_ensembles(
            proposal,
            cuda,
            relative_tolerance=0.01,
            sigma_limit=3.0,
            active_fraction=1.0e-4,
            minimum_bin_pass_fraction=0.95,
        )
        self.assertEqual(report["status"], "failed")
        self.assertFalse(
            report["scalars"]["profile_charged_max"]["relative_pass"]
        )
        self.assertIn(
            report["scalars"]["profile_charged_max"]["outcome"],
            {
                "relative_threshold_failed_but_statistically_inconclusive",
                "relative_and_statistical_thresholds_failed",
            },
        )
        self.assertFalse(report["curves"]["profile_charged"]["relative_pass"])

    def test_shards_are_concatenated_on_the_event_axis(self) -> None:
        first = make_ensemble("proposal_0")
        second = make_ensemble("proposal_1")
        combined = concatenate_ensembles(
            "proposal",
            [first, second],
        )
        self.assertEqual(len(combined.showers), 8)
        self.assertEqual(combined.scalars.shape[0], 8)
        self.assertEqual(
            combined.curves["profile_charged"][1].shape,
            (8, 3),
        )
        self.assertEqual(combined.metadata["shards"], 2)


if __name__ == "__main__":
    unittest.main()
