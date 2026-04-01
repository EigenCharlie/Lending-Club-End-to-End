from __future__ import annotations

from src.evaluation.model_shift import interpret_model_shift


def test_interpret_model_shift_distinguishes_structural_shift_only() -> None:
    out = interpret_model_shift(
        c2st_auc=0.66,
        c2st_materiality="high",
        score_psi=0.18,
        auc_delta=0.005,
        brier_increase=0.001,
        calibration_gap_delta=0.002,
        distribution_warning_ratio=0.25,
        score_psi_max=0.15,
        auc_delta_max=0.05,
        brier_increase_max=0.02,
        calibration_gap_delta_max=0.02,
    )

    assert out["shift_type"] == "structural_shift_only"
    assert out["governance_posture"] == "warning_only"


def test_interpret_model_shift_detects_mixed_shift() -> None:
    out = interpret_model_shift(
        c2st_auc=0.68,
        c2st_materiality="high",
        score_psi=0.20,
        auc_delta=0.08,
        brier_increase=0.04,
        calibration_gap_delta=0.05,
        distribution_warning_ratio=0.40,
        score_psi_max=0.15,
        auc_delta_max=0.05,
        brier_increase_max=0.02,
        calibration_gap_delta_max=0.02,
    )

    assert out["shift_type"] == "mixed_shift"
    assert out["governance_posture"] == "candidate_gate"
