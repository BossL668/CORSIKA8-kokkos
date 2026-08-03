#!/usr/bin/env python3

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_cpu_cuda_radio import (  # noqa: E402
    band_limited_waveform,
    compare_distributions,
    compare_track_diagnostics,
    json_compatible,
    paired_formalism_comparison,
    peak_aligned_unit_energy,
    radial_integral_contributions,
    read_track_diagnostics,
    unit_sum,
    vector_comparison,
    waveform_feature,
)


class CpuCudaRadioAnalysisTest(unittest.TestCase):
    def test_json_compatible_maps_undefined_single_event_statistics_to_null(
        self,
    ) -> None:
        value = {
            "finite": np.float64(1.5),
            "undefined": math.nan,
            "nested": [math.inf, np.asarray([2.0, -math.inf])],
        }
        converted = json_compatible(value)
        self.assertEqual(
            converted,
            {
                "finite": 1.5,
                "undefined": None,
                "nested": [None, [2.0, None]],
            },
        )
        json.dumps(converted, allow_nan=False)

    def test_band_limited_waveform_rejects_out_of_band_tone(self) -> None:
        sample_rate = 1.0e9
        time = np.arange(1000) / sample_rate
        field = np.column_stack(
            [
                np.sin(2.0 * np.pi * 50.0e6 * time),
                np.sin(2.0 * np.pi * 200.0e6 * time),
                np.zeros_like(time),
            ]
        )
        filtered = band_limited_waveform(
            field, 1.0 / sample_rate, 30.0, 80.0
        )
        self.assertGreater(np.linalg.norm(filtered[:, 0]), 1.0)
        self.assertLess(
            np.linalg.norm(filtered[:, 1]),
            np.linalg.norm(filtered[:, 0]) * 1.0e-12,
        )

    def test_peak_alignment_does_not_wrap_power_across_window(self) -> None:
        power = np.zeros(9)
        power[0] = 3.0
        power[1] = 1.0
        aligned = peak_aligned_unit_energy(power)
        self.assertEqual(int(np.argmax(aligned)), 4)
        self.assertEqual(aligned[0], 0.0)
        self.assertAlmostEqual(float(np.sum(aligned)), 1.0)

    def test_unit_sum_and_vector_comparison(self) -> None:
        normalized = unit_sum(np.asarray([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.sum(normalized)), 1.0)
        result = vector_comparison(normalized, normalized)
        self.assertEqual(result["relative_L1"], 0.0)
        self.assertAlmostEqual(result["cosine_similarity"], 1.0)

    def test_radial_integral_contributions_match_constant_footprint(self) -> None:
        radii = np.asarray([0.0, 1.0, 2.0])
        contributions = radial_integral_contributions(
            radii, np.ones_like(radii)
        )
        self.assertAlmostEqual(float(np.sum(contributions)), 4.0 * np.pi)

    def test_waveform_feature_reports_polarization_and_spectrum(self) -> None:
        sample_rate = 1.0e9
        time_s = np.arange(1000) / sample_rate
        field = np.column_stack(
            [
                np.sin(2.0 * np.pi * 50.0e6 * time_s),
                np.zeros_like(time_s),
                np.zeros_like(time_s),
            ]
        )
        feature, template, spectrum = waveform_feature(
            field, time_s * 1.0e9, 1.0, (30.0, 80.0)
        )
        self.assertAlmostEqual(feature["fluence_x_fraction"], 1.0)
        self.assertAlmostEqual(feature["fluence_y_fraction"], 0.0)
        self.assertAlmostEqual(feature["stokes_q_over_i"], 1.0)
        self.assertAlmostEqual(float(np.sum(template)), 1.0)
        self.assertAlmostEqual(float(np.sum(spectrum)), 1.0)

    def test_paired_formalism_comparison_uses_aggregate_definition(self) -> None:
        result = paired_formalism_comparison(
            {0: 9.0, 1: 18.0},
            {0: 10.0, 1: 20.0},
            bootstrap_repetitions=100,
            seed=1,
        )
        self.assertAlmostEqual(
            result["aggregate_radiation_energy_shift"], 0.1
        )
        self.assertAlmostEqual(result["mean_per_shower_shift"], 0.1)
        self.assertAlmostEqual(result["pearson_radiation_energy"], 1.0)

    def test_identical_distribution_has_zero_distance(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        result = compare_distributions(
            values,
            values,
            bootstrap_repetitions=1000,
            seed=1,
        )
        self.assertEqual(result["relative_mean_difference"], 0.0)
        self.assertEqual(result["KS_distance"], 0.0)
        self.assertEqual(result["relative_quantile_wasserstein"], 0.0)

    def test_bootstrap_relative_shift_preserves_tiny_physical_scale(self) -> None:
        reference = np.asarray([1.0, 1.1, 0.9, 1.05]) * 1.0e-28
        result = compare_distributions(
            reference,
            reference * 1.1,
            bootstrap_repetitions=1000,
            seed=2,
        )
        lower, upper = result[
            "bootstrap_signed_relative_mean_shift_95pct"
        ]
        self.assertGreater(lower, -0.1)
        self.assertLess(upper, 0.3)

    def test_signed_cancellation_distribution_remains_json_finite(self) -> None:
        result = compare_distributions(
            [-1.0, 1.0, -2.0, 2.0],
            [-1.1, 1.2, -2.1, 2.2],
            bootstrap_repetitions=1000,
            seed=3,
        )
        json.dumps(result, allow_nan=False)
        interval = result[
            "bootstrap_signed_relative_mean_shift_95pct"
        ]
        self.assertIsNotNone(interval)
        self.assertTrue(all(math.isfinite(value) for value in interval))

    def test_track_diagnostics_use_em_deposit_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            candidate = root / "candidate"
            fields = (
                "segment_count",
                "weighted_segment_count",
                "track_length_m",
                "weighted_track_length_m",
                "electron_weighted_track_length_m",
                "positron_weighted_track_length_m",
                "signed_charge_weighted_track_length_m",
                "energy_weighted_track_length_GeV_m",
                "maximum_segment_length_m",
            )
            for output, scale in ((reference, 1.0), (candidate, 2.0)):
                (output / "CoREAS").mkdir(parents=True)
                (output / "energyloss").mkdir()
                radio = {}
                energy = {}
                for shower in range(2):
                    key = f"shower_{shower}"
                    radio[key] = {
                        field: scale * (shower + 1) for field in fields
                    }
                    radio[key][
                        "weighted_track_length_by_kinetic_energy"
                    ] = {
                        "upper_edge_GeV": [0.001, "inf"],
                        "weighted_track_length_m": [
                            scale * (shower + 1),
                            scale * (shower + 2),
                        ],
                    }
                    energy[key] = {
                        "sum_dEdX": 10.0 * scale,
                        "sum_dEdX_em": 5.0 * scale,
                        "Xmax": 100.0,
                        "dEdXmax": 1.0,
                    }
                (output / "CoREAS" / "summary.yaml").write_text(
                    yaml.safe_dump(radio), encoding="utf-8"
                )
                (output / "energyloss" / "summary.yaml").write_text(
                    yaml.safe_dump(energy), encoding="utf-8"
                )
            result = compare_track_diagnostics(
                reference,
                candidate,
                "CoREAS",
                bootstrap_repetitions=100,
                seed=1,
            )
            self.assertEqual(
                result["fields"]["weighted_track_length_m"][
                    "signed_relative_shift_from_reference"
                ],
                1.0,
            )
            self.assertEqual(
                result["fields_per_electromagnetic_deposited_GeV"][
                    "weighted_track_length_m"
                ]["relative_mean_difference"],
                0.0,
            )
            self.assertEqual(
                result["weighted_track_length_by_kinetic_energy"][
                    "le_0.001_GeV"
                ]["per_electromagnetic_deposited_GeV"][
                    "relative_mean_difference"
                ],
                0.0,
            )

    def test_cuda_resident_and_scalar_track_diagnostics_are_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "CoREAS").mkdir()
            (output / "gpu_em").mkdir()
            scalar = {
                "shower_0": {
                    "segment_count": 2,
                    "weighted_segment_count": 3.0,
                    "track_length_m": 4.0,
                    "weighted_track_length_m": 5.0,
                    "electron_weighted_track_length_m": 4.0,
                    "positron_weighted_track_length_m": 1.0,
                    "signed_charge_weighted_track_length_m": -3.0,
                    "energy_weighted_track_length_GeV_m": 6.0,
                    "maximum_segment_length_m": 3.0,
                    "weighted_track_length_by_kinetic_energy": {
                        "upper_edge_GeV": [0.001, "inf"],
                        "weighted_track_length_m": [2.0, 3.0],
                    },
                }
            }
            gpu_radio = {
                "track_diagnostics_enabled": True,
                "segment_count": 7,
                "weighted_segment_count": 8.0,
                "track_length_m": 9.0,
                "weighted_track_length_m": 10.0,
                "electron_weighted_track_length_m": 6.0,
                "positron_weighted_track_length_m": 4.0,
                "signed_charge_weighted_track_length_m": -2.0,
                "energy_weighted_track_length_GeV_m": 11.0,
                "maximum_segment_length_m": 12.0,
                "weighted_track_length_by_kinetic_energy": {
                    "upper_edge_GeV": [0.001, "inf"],
                    "weighted_track_length_m": [4.0, 6.0],
                },
            }
            (output / "CoREAS" / "summary.yaml").write_text(
                yaml.safe_dump(scalar), encoding="utf-8"
            )
            (output / "gpu_em" / "summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "shower_0": {
                            "statistics": {"radio": gpu_radio}
                        }
                    }
                ),
                encoding="utf-8",
            )
            merged = read_track_diagnostics(output, "CoREAS")[
                "shower_0"
            ]
            self.assertEqual(merged["segment_count"], 9.0)
            self.assertEqual(merged["weighted_track_length_m"], 15.0)
            self.assertEqual(merged["maximum_segment_length_m"], 12.0)
            self.assertEqual(
                merged["weighted_track_length_by_kinetic_energy"][
                    "weighted_track_length_m"
                ],
                [6.0, 9.0],
            )


if __name__ == "__main__":
    unittest.main()
