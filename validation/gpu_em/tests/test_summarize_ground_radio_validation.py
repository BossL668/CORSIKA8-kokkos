from __future__ import annotations

import copy

from validation.gpu_em.summarize_ground_radio_validation import summarize


def fixtures() -> tuple[dict, ...]:
    finalization = {"status": "complete", "proposal_events": 500, "cuda_events": 500}
    scalar = {
        "proposal": {"events": 500},
        "cuda": {"events": 500},
        "relative_difference": 0.01,
        "absolute_z_score": 0.5,
        "statistical_pass": True,
        "acceptance_gate": True,
    }
    comparison = {
        "proposal": {"events": 500},
        "cuda": {"events": 500},
        "acceptance": {"curve_pass": True},
        "scalars": {
            "ground_em_weighted_count": copy.deepcopy(scalar),
            "ground_em_kinetic_energy_GeV": copy.deepcopy(scalar),
        },
    }
    metric = {
        "models": {
            "adjusted_xmax_and_em_integral": {
                "cuda_over_proposal_ratio": 1.0,
                "bootstrap_shower_95pct": [0.98, 1.02],
            }
        },
        "tests": {
            "ks": {"p_value": 0.8},
            "cramer_von_mises": {"p_value": 0.7},
            "anderson_darling_k_sample": {"p_value": 0.25},
        },
        "low_tail_event_fractions": {
            "q05": {"bootstrap_shower_95pct": [-0.02, 0.02]},
            "q16": {"bootstrap_shower_95pct": [-0.03, 0.03]},
        },
        "low_tail_quantile_ratios": {
            "q05": {"bootstrap_shower_95pct": [0.9, 1.1]},
            "q16": {"bootstrap_shower_95pct": [0.9, 1.1]},
        },
    }
    attribution = {
        "ground": {
            "ground_em_weighted_count": copy.deepcopy(metric),
            "ground_em_kinetic_energy_GeV": copy.deepcopy(metric),
        }
    }
    configuration = {
        "status": "equivalent",
        "events_by_backend": {"proposal": 500, "cuda": 500},
    }
    waveform = {
        "status": "passed",
        "transport_identity": {"accepted": True},
        "algorithms": {"CoREAS": {"accepted": True}, "ZHS": {"accepted": True}},
    }
    pulse_oracle = {
        "status": "passed",
        "algorithms": {"CoREAS": {"status": "passed"}, "ZHS": {"status": "passed"}},
    }
    independent = {"acceptance": {"passed": True}}
    return (
        finalization,
        comparison,
        attribution,
        configuration,
        waveform,
        pulse_oracle,
        independent,
    )


def call(values: tuple[dict, ...]) -> dict:
    return summarize(*values, expected_events=500, alpha=0.05)


def test_complete_consistent_evidence_passes() -> None:
    report = call(fixtures())
    assert report["status"] == "passed"
    assert report["ground_em"]["status"] == "no_detectable_difference"
    assert report["radio"]["status"] == "end_to_end_statistically_consistent"


def test_independent_radio_failure_does_not_blame_projection_or_settings() -> None:
    values = list(fixtures())
    values[-1] = {"acceptance": {"passed": False}}
    report = call(tuple(values))
    assert report["status"] == "needs_investigation"
    assert (
        report["radio"]["status"]
        == "end_to_end_difference_not_caused_by_radio_settings_or_projection"
    )
    assert report["radio"]["settings_and_projection_excluded_as_direct_cause"]


def test_ground_low_tail_scale_interval_excluding_one_fails() -> None:
    values = list(fixtures())
    values[2]["ground"]["ground_em_weighted_count"][
        "low_tail_quantile_ratios"
    ]["q05"]["bootstrap_shower_95pct"] = [0.80, 0.95]
    report = call(tuple(values))
    assert report["status"] == "needs_investigation"
    assert report["ground_em"]["status"] == "difference_detected_or_unresolved"
