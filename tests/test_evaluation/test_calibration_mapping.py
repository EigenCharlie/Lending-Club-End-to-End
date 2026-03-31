from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.calibration_mapping import (
    apply_logit_intercept_shift,
    calibration_mapping_candidates_report,
    logit_intercept_shift,
)


def test_logit_intercept_shift_moves_mean_toward_observed_rate() -> None:
    y_true = np.r_[np.ones(250), np.zeros(750)]
    y_prob = np.full(1000, 0.12, dtype=float)
    delta = logit_intercept_shift(y_true, y_prob)
    shifted = apply_logit_intercept_shift(y_prob, delta)

    assert abs(float(shifted.mean()) - float(np.mean(y_true))) < abs(
        float(y_prob.mean()) - float(np.mean(y_true))
    )


def test_calibration_mapping_candidates_report_emits_sidecar_candidates() -> None:
    frame = pd.DataFrame(
        {
            "default_flag": np.r_[np.ones(300), np.zeros(900)],
            "pd_calibrated": np.r_[np.full(600, 0.10), np.full(600, 0.18)],
            "issue_quarter": ["2020Q1"] * 300
            + ["2020Q2"] * 300
            + ["2020Q3"] * 300
            + ["2020Q4"] * 300,
            "grade": ["A"] * 400 + ["B"] * 400 + ["C"] * 400,
        }
    )

    report = calibration_mapping_candidates_report(frame)

    assert not report.empty
    assert {"current_identity", "logit_intercept_shift", "isotonic_sidecar"}.issubset(
        set(report["candidate_id"].astype(str))
    )
    assert "abs_global_gap_bp" in report.columns
