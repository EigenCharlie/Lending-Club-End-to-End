from __future__ import annotations

import pandas as pd

from src.evaluation.pd_validation_interpretation import (
    classify_overall_gap_materiality,
    summarize_slice_materiality,
    validation_interpretation_status,
)


def test_classify_overall_gap_materiality_assigns_low_band() -> None:
    out = classify_overall_gap_materiality(0.2200, 0.2165)
    assert out["materiality_band"] == "low"
    assert abs(float(out["gap_bp"])) < 50.0


def test_summarize_slice_materiality_counts_grade_and_band_breaches() -> None:
    grade = pd.DataFrame(
        {
            "observed_default_rate": [0.10, 0.30],
            "mean_predicted_pd": [0.07, 0.31],
        }
    )
    band = pd.DataFrame({"rate_gap": [0.002, 0.020, -0.018]})
    out = summarize_slice_materiality(grade, band)
    assert out["grade_material_breaches"] == 1
    assert out["band_material_breaches"] == 2


def test_validation_interpretation_escalates_to_warning_for_slice_breaches() -> None:
    quarter = pd.DataFrame({"abs_gap_bp": [80.0, 20.0], "rate_gap": [0.008, 0.002]})
    out = validation_interpretation_status(
        overall_backtesting={
            "observed_default_rate": 0.2198,
            "mean_predicted_pd": 0.2161,
            "predicted_pd_inside_jeffreys": False,
            "exact_binomial_p_value": 1e-6,
            "hl_p_value": 0.0,
        },
        slice_materiality={
            "grade_material_breaches": 2,
            "max_grade_gap_bp": 500.0,
            "band_material_breaches": 1,
            "max_band_gap_bp": 120.0,
        },
        quarter_report=quarter,
        rare_event={"max_decile_gap_bp": 150.0},
    )
    assert out["overall_pass"] is True
    assert out["severity"] == "warning"
