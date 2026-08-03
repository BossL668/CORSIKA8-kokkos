import json
import tempfile
import unittest
from pathlib import Path

from validation.gpu_em.analyze_geomagnetic_pulse_distributions import (
    aggregate_by_shower,
    apply_reference_width_filter,
    backend_output_directories,
    build_acceptance,
)


def comparison(
    *,
    cpu_count=100,
    cuda_count=100,
    ratio=1.0,
    ci_low=0.98,
    ci_high=1.02,
    ks_p_value=0.5,
):
    return {
        "available": True,
        "cpu": {"count": cpu_count},
        "cuda": {"count": cuda_count},
        "cuda_over_cpu": {
            "ratio": ratio,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        },
        "ks_two_sample": {"p_value": ks_p_value},
    }


def shower_rows(count=100):
    return [
        {
            "backend": backend,
            "algorithm": algorithm,
        }
        for backend in ("legacy_proposal", "cuda")
        for algorithm in ("CoREAS", "ZHS")
        for _ in range(count)
    ]


class GeomagneticPulseAcceptanceTest(unittest.TestCase):
    def test_radius_is_part_of_shower_aggregation_key(self):
        rows = []
        for radius, amplitude in ((50.0, 2.0), (100.0, 8.0)):
            for observer in range(2):
                rows.append(
                    {
                        "backend": "cuda",
                        "algorithm": "CoREAS",
                        "shower": 0,
                        "observer": observer,
                        "radius_m": radius,
                        "geomagnetic_peak_abs_V_per_m": amplitude,
                        "pulse_width_ns": radius / 10.0,
                        "pulse_width_valid": True,
                        "pulse_width_filter_pass": True,
                    }
                )
        aggregated = aggregate_by_shower(rows)
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(
            [row["radius_m"] for row in aggregated], [50.0, 100.0]
        )
        self.assertTrue(
            all(
                abs(actual - expected) < 1.e-12
                for actual, expected in zip(
                    [
                        row["geomagnetic_amplitude_geomean_V_per_m"]
                        for row in aggregated
                    ],
                    [2.0, 8.0],
                )
            )
        )

    def test_width_filter_is_applied_independently_per_radius(self):
        rows = [
            {
                "backend": "cuda",
                "algorithm": "ZHS",
                "radius_m": radius,
                "pulse_width_ns": width,
                "pulse_width_valid": True,
            }
            for radius, width in ((50.0, 2.0), (50.0, 3.0), (100.0, 8.0))
        ]
        calls = []

        def keep_all(widths, _config):
            calls.append(widths.tolist())
            return [True] * len(widths)

        apply_reference_width_filter(
            rows,
            robust_pulse_width_mask=keep_all,
            filter_config=object(),
        )
        self.assertEqual(calls, [[2.0, 3.0], [8.0]])
        self.assertTrue(all(row["pulse_width_filter_pass"] for row in rows))

    def test_manifest_additional_sources_are_loaded_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            proposal = dataset / "proposal_shard_000"
            cuda = dataset / "cuda"
            old_proposal = root / "old_proposal"
            old_cuda = root / "old_cuda"
            for path in (proposal, cuda, old_proposal, old_cuda):
                path.mkdir(parents=True)
            (dataset / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "additional_sources": {
                            "proposal": [
                                str(old_proposal),
                                str(old_proposal),
                            ],
                            "cuda": [str(old_cuda)],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                backend_output_directories(
                    dataset, "legacy_proposal"
                ),
                [proposal.resolve(), old_proposal.resolve()],
            )
            self.assertEqual(
                backend_output_directories(dataset, "cuda"),
                [cuda.resolve(), old_cuda.resolve()],
            )

    def test_external_manifest_can_supply_cuda_without_direct_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            proposal = dataset / "proposal_shard_000"
            cuda = root / "cuda_source"
            manifest = root / "analysis_manifest.json"
            proposal.mkdir(parents=True)
            cuda.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "additional_sources": {
                            "proposal": [],
                            "cuda": [str(cuda)],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                backend_output_directories(
                    dataset, "legacy_proposal", manifest
                ),
                [proposal.resolve()],
            )
            self.assertEqual(
                backend_output_directories(dataset, "cuda", manifest),
                [cuda.resolve()],
            )

    def test_raw_pulse_requires_contained_intervals_and_coverage(self):
        comparisons = {
            algorithm: {
                "amplitude": comparison(),
                "width": comparison(cpu_count=90, cuda_count=85),
            }
            for algorithm in ("CoREAS", "ZHS")
        }
        result = build_acceptance(
            comparisons=comparisons,
            shower_rows=shower_rows(),
            band_MHz=None,
            relative_tolerance=0.1,
            minimum_ks_p_value=0.05,
            minimum_count=20,
            minimum_width_shower_fraction=0.8,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["algorithms"]["CoREAS"]["width"]["coverage_pass"])

        comparisons["ZHS"]["amplitude"] = comparison(
            ci_low=0.89, ci_high=1.01
        )
        result = build_acceptance(
            comparisons=comparisons,
            shower_rows=shower_rows(),
            band_MHz=None,
            relative_tolerance=0.1,
            minimum_ks_p_value=0.05,
            minimum_count=20,
            minimum_width_shower_fraction=0.8,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["algorithms"]["ZHS"]["amplitude"][
                "confidence_interval_contained"
            ]
        )

    def test_ideal_bandpass_is_diagnostic_only_for_square_width(self):
        comparisons = {
            algorithm: {
                "amplitude": comparison(),
                "width": comparison(),
            }
            for algorithm in ("CoREAS", "ZHS")
        }
        result = build_acceptance(
            comparisons=comparisons,
            shower_rows=shower_rows(),
            band_MHz=(30.0, 80.0),
            relative_tolerance=0.1,
            minimum_ks_p_value=0.05,
            minimum_count=20,
            minimum_width_shower_fraction=0.8,
        )
        self.assertFalse(result["raw_waveform_gate"])
        self.assertFalse(result["passed"])
        self.assertIn("diagnostic only", result["scientific_scope"])

    def test_shape_tests_control_familywise_error_with_holm(self):
        comparisons = {
            algorithm: {
                "amplitude": comparison(ks_p_value=0.1),
                "width": comparison(ks_p_value=0.1),
            }
            for algorithm in ("CoREAS", "ZHS")
        }
        comparisons["CoREAS"]["amplitude"] = comparison(
            ks_p_value=0.029
        )
        result = build_acceptance(
            comparisons=comparisons,
            shower_rows=shower_rows(),
            band_MHz=None,
            relative_tolerance=0.1,
            minimum_ks_p_value=0.05,
            minimum_count=20,
            minimum_width_shower_fraction=0.8,
        )
        self.assertTrue(result["passed"])
        coreas_amplitude = result["algorithms"]["CoREAS"]["amplitude"]
        self.assertFalse(coreas_amplitude["uncorrected_shape_pass"])
        self.assertFalse(coreas_amplitude["holm_bonferroni_rejected"])
        self.assertTrue(coreas_amplitude["shape_pass"])

        comparisons["CoREAS"]["amplitude"] = comparison(
            ks_p_value=0.005
        )
        result = build_acceptance(
            comparisons=comparisons,
            shower_rows=shower_rows(),
            band_MHz=None,
            relative_tolerance=0.1,
            minimum_ks_p_value=0.05,
            minimum_count=20,
            minimum_width_shower_fraction=0.8,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            result["algorithms"]["CoREAS"]["amplitude"][
                "holm_bonferroni_rejected"
            ]
        )


if __name__ == "__main__":
    unittest.main()
