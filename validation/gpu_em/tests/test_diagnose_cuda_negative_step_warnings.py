from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "diagnose_cuda_negative_step_warnings.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_cuda_negative_step_warnings", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_warning_sections_assigns_warnings_to_events() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        log = Path(temporary) / "run.log"
        log.write_text(
            "setup\n"
            + MODULE.EVENT_MARKER
            + " event zero\n"
            + "negative step length of l=-0.25 m\n"
            + "negative step length of l=-1.5e-2 m\n"
            + MODULE.EVENT_MARKER
            + " event one\n",
            encoding="utf-8",
        )
        result = MODULE.parse_warning_sections(log, 2)
        assert result[0]["negative_step_warnings"] == 2
        assert result[0]["negative_step_abs_length_sum_m"] == pytest.approx(0.265)
        assert result[1]["negative_step_warnings"] == 0


def test_parse_warning_sections_rejects_marker_count_difference() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        log = Path(temporary) / "run.log"
        log.write_text(MODULE.EVENT_MARKER + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="event-marker count differs"):
            MODULE.parse_warning_sections(log, 2)


def test_fit_models_recovers_adjusted_warning_effect() -> None:
    rng = np.random.default_rng(42)
    size = 400
    xmax = rng.normal(500.0, 45.0, size)
    em_integral = np.exp(rng.normal(12.0, 0.4, size))
    warning_rate = rng.uniform(1.0e-4, 1.2e-3, size)
    warning_z = (warning_rate - warning_rate.mean()) / warning_rate.std(ddof=0)
    response = np.exp(
        4.0
        + 0.003 * (xmax - xmax.mean())
        + 0.6 * (np.log(em_integral) - np.log(em_integral).mean())
        + 0.08 * warning_z
        + rng.normal(0.0, 0.01, size)
    )
    frame = pd.DataFrame(
        {
            "profile_xmax_charged_gcm2": xmax,
            "profile_em_integral": em_integral,
            "negative_step_warning_rate": warning_rate,
            "ground_em_weighted_count": response,
        }
    )
    result = MODULE.fit_models(frame, "ground_em_weighted_count")
    assert result["warning_log_coefficient"] == pytest.approx(0.08, abs=0.005)
    assert result["partial_r_squared"] > 0.95
