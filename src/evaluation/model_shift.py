"""Interpretive helpers for structural shift and p-value semantics."""

from __future__ import annotations

from typing import Any


def interpret_model_shift(
    *,
    c2st_auc: float,
    c2st_materiality: str,
    score_psi: float,
    auc_delta: float,
    brier_increase: float,
    calibration_gap_delta: float,
    distribution_warning_ratio: float,
    score_psi_max: float,
    auc_delta_max: float,
    brier_increase_max: float,
    calibration_gap_delta_max: float,
) -> dict[str, Any]:
    """Distinguish structural shift from predictive degradation."""
    structural_level = "none"
    if c2st_auc >= 0.70 or score_psi >= max(score_psi_max * 1.5, 0.20):
        structural_level = "severe"
    elif c2st_auc >= 0.60 or score_psi >= score_psi_max:
        structural_level = "high"
    elif c2st_auc >= 0.55 or score_psi >= max(score_psi_max * 0.7, 0.10):
        structural_level = "moderate"
    elif c2st_auc >= 0.52 or distribution_warning_ratio > 0.05:
        structural_level = "low"

    predictive_level = "none"
    if (
        auc_delta >= auc_delta_max * 1.5
        or brier_increase >= brier_increase_max * 1.5
        or calibration_gap_delta >= calibration_gap_delta_max * 1.5
    ):
        predictive_level = "severe"
    elif (
        auc_delta > auc_delta_max
        or brier_increase > brier_increase_max
        or calibration_gap_delta > calibration_gap_delta_max
    ):
        predictive_level = "high"
    elif (
        auc_delta >= auc_delta_max * 0.75
        or brier_increase >= brier_increase_max * 0.75
        or calibration_gap_delta >= calibration_gap_delta_max * 0.75
    ):
        predictive_level = "moderate"
    elif auc_delta > 0.0 or brier_increase > 0.0 or calibration_gap_delta > 0.0:
        predictive_level = "low"

    if structural_level != "none" and predictive_level in {"none", "low"}:
        shift_type = "structural_shift_only"
    elif structural_level == "none" and predictive_level != "none":
        shift_type = "predictive_degradation"
    elif structural_level != "none" and predictive_level != "none":
        shift_type = "mixed_shift"
    else:
        shift_type = "stable"

    if shift_type == "mixed_shift" or predictive_level in {"high", "severe"}:
        governance_posture = "candidate_gate"
    elif shift_type == "structural_shift_only":
        governance_posture = "warning_only"
    else:
        governance_posture = "monitor"

    pvalue_note = (
        "Distribution p-values are informative, but they can trigger on small effects at large N; "
        "the governance posture should be anchored on materiality, predictive loss, and C2ST/score shift."
    )
    if shift_type == "structural_shift_only":
        pvalue_note = (
            "KS/CvM or C2ST detect structural population change without clear predictive degradation. "
            "Treat this as representativeness pressure and monitoring, not an automatic retraining gate."
        )
    elif shift_type == "predictive_degradation":
        pvalue_note = (
            "Predictive degradation is visible even without strong structural shift. "
            "Operational metrics dominate p-value interpretation in this case."
        )
    elif shift_type == "mixed_shift":
        pvalue_note = (
            "Both structural shift and predictive degradation are present. "
            "This combination deserves the strongest governance posture."
        )

    return {
        "shift_type": shift_type,
        "structural_shift_level": structural_level,
        "predictive_degradation_level": predictive_level,
        "governance_posture": governance_posture,
        "c2st_materiality": str(c2st_materiality),
        "pvalue_interpretation": pvalue_note,
    }
