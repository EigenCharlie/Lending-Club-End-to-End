from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.calibration_mapping import (
    apply_logit_intercept_shift,
    calibration_mapping_candidates_report,
    logit_intercept_shift,
    materialize_candidate_calibrator,
    temporal_otv_split,
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
    assert "stage_a_pass" in report.columns


def test_temporal_otv_split_orders_rows_without_leakage() -> None:
    frame = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(
                ["2020-01-01", "2020-01-15", "2020-02-01", "2020-02-15", "2020-03-01", "2020-03-15"]
            ),
            "default_flag": [0, 1, 0, 1, 0, 1],
            "pd_calibrated": [0.1, 0.2, 0.15, 0.25, 0.2, 0.3],
        }
    )

    adaptation, evaluation = temporal_otv_split(frame, min_eval_rows=2)

    assert adaptation["issue_d"].max() <= evaluation["issue_d"].min()
    assert len(adaptation) + len(evaluation) == len(frame)


def test_materialize_candidate_calibrator_builds_isotonic_sidecar() -> None:
    report = calibration_mapping_candidates_report(
        pd.DataFrame(
            {
                "default_flag": np.r_[np.ones(400), np.zeros(1200)],
                "pd_calibrated": np.r_[np.full(800, 0.12), np.full(800, 0.22)],
                "issue_quarter": ["2020Q1"] * 400
                + ["2020Q2"] * 400
                + ["2020Q3"] * 400
                + ["2020Q4"] * 400,
                "grade": ["A"] * 800 + ["B"] * 800,
            }
        )
    )
    isotonic_spec = report.loc[report["candidate_id"] == "isotonic_sidecar", "candidate_spec"].iloc[
        0
    ]
    calibrator = materialize_candidate_calibrator(isotonic_spec)
    preds = calibrator.transform(np.array([0.1, 0.2, 0.3], dtype=float))

    assert preds.shape == (3,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
