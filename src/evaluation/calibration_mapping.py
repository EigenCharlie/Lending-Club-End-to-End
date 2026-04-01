"""Diagnostic-only calibration remapping helpers inspired by ADSFCR."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation.backtesting import hosmer_lemeshow_test
from src.evaluation.metrics import classification_metrics
from src.evaluation.pd_validation_interpretation import summarize_slice_materiality
from src.models.calibration import LogitShiftCalibrator, expected_calibration_error


def _clip_prob(y_prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)


def logit_intercept_shift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find a logit-space intercept shift that matches the observed base rate."""
    target = float(np.mean(np.asarray(y_true, dtype=float)))
    prob = _clip_prob(np.asarray(y_prob, dtype=float))
    logits = np.log(prob / (1.0 - prob))
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        shifted_mean = float((1.0 / (1.0 + np.exp(-(logits + mid)))).mean())
        if shifted_mean < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def apply_logit_intercept_shift(y_prob: np.ndarray, delta: float) -> np.ndarray:
    prob = _clip_prob(np.asarray(y_prob, dtype=float))
    logits = np.log(prob / (1.0 - prob))
    shifted = 1.0 / (1.0 + np.exp(-(logits + float(delta))))
    return np.clip(np.asarray(shifted, dtype=float), 1e-6, 1.0 - 1e-6)


def _grade_backtesting(frame: pd.DataFrame, *, target_col: str, score_col: str) -> pd.DataFrame:
    if "grade" not in frame.columns:
        return pd.DataFrame(columns=["observed_default_rate", "mean_predicted_pd"])
    rows: list[dict[str, Any]] = []
    for grade, grp in frame.groupby("grade", observed=True):
        if len(grp) < 100:
            continue
        rows.append(
            {
                "grade": str(grade),
                "observed_default_rate": float(
                    pd.to_numeric(grp[target_col], errors="coerce").mean()
                ),
                "mean_predicted_pd": float(pd.to_numeric(grp[score_col], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _band_backtesting(frame: pd.DataFrame, *, target_col: str, score_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["rate_gap"])
    band_df = frame[[target_col, score_col]].copy()
    band_df[target_col] = pd.to_numeric(band_df[target_col], errors="coerce")
    band_df[score_col] = pd.to_numeric(band_df[score_col], errors="coerce")
    band_df = band_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(band_df) < 200:
        return pd.DataFrame(columns=["rate_gap"])
    band_df["band"] = pd.qcut(band_df[score_col], q=10, labels=False, duplicates="drop")
    summary = (
        band_df.groupby("band", observed=True)
        .agg(
            observed_default_rate=(target_col, "mean"),
            mean_predicted_pd=(score_col, "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        return pd.DataFrame(columns=["rate_gap"])
    summary["rate_gap"] = summary["observed_default_rate"] - summary["mean_predicted_pd"]
    return summary


def _quarter_gap_report(
    frame: pd.DataFrame,
    *,
    target_col: str,
    score_col: str,
    min_rows_per_quarter: int = 100,
) -> pd.DataFrame:
    if "issue_quarter" in frame.columns:
        quarter = frame["issue_quarter"].astype("string")
    elif "issue_d" in frame.columns:
        quarter = (
            pd.to_datetime(frame["issue_d"], errors="coerce").dt.to_period("Q").astype("string")
        )
    else:
        return pd.DataFrame(columns=["issue_quarter", "n_obs", "rate_gap", "abs_gap_bp"])
    qframe = (
        pd.DataFrame(
            {
                "issue_quarter": quarter,
                "y_true": pd.to_numeric(frame[target_col], errors="coerce"),
                "score": pd.to_numeric(frame[score_col], errors="coerce"),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if qframe.empty:
        return pd.DataFrame(columns=["issue_quarter", "n_obs", "rate_gap", "abs_gap_bp"])
    report = (
        qframe.groupby("issue_quarter", observed=True)
        .agg(
            n_obs=("y_true", "size"),
            observed_default_rate=("y_true", "mean"),
            mean_predicted_pd=("score", "mean"),
        )
        .reset_index()
        .sort_values("issue_quarter")
    )
    report = report.loc[report["n_obs"] >= int(min_rows_per_quarter)].reset_index(drop=True)
    if report.empty:
        return report
    report["rate_gap"] = report["observed_default_rate"] - report["mean_predicted_pd"]
    report["abs_gap_bp"] = report["rate_gap"].abs() * 10_000.0
    return report


def evaluate_calibration_candidate(
    frame: pd.DataFrame,
    *,
    score_col: str,
    target_col: str = "default_flag",
) -> dict[str, Any]:
    eval_frame = frame.copy()
    eval_frame[target_col] = pd.to_numeric(eval_frame[target_col], errors="coerce")
    eval_frame[score_col] = pd.to_numeric(eval_frame[score_col], errors="coerce")
    eval_frame = eval_frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[target_col, score_col]
    )
    y_true = eval_frame[target_col].to_numpy(dtype=int)
    y_prob = eval_frame[score_col].to_numpy(dtype=float)
    if len(np.unique(y_true)) >= 2:
        metrics = classification_metrics(y_true, y_prob)
    else:
        metrics = {
            "auc_roc": 0.5,
            "brier_score": float(brier_score_loss(y_true, y_prob)) if len(y_true) else 0.0,
            "ece": float(expected_calibration_error(y_true, y_prob)) if len(y_true) else 0.0,
            "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])) if len(y_true) else 0.0,
        }
    hl = hosmer_lemeshow_test(y_true, y_prob, n_groups=10)
    quarter_report = _quarter_gap_report(eval_frame, target_col=target_col, score_col=score_col)
    slice_materiality = summarize_slice_materiality(
        _grade_backtesting(eval_frame, target_col=target_col, score_col=score_col),
        _band_backtesting(eval_frame, target_col=target_col, score_col=score_col),
    )
    max_quarter_gap_bp = (
        float(pd.to_numeric(quarter_report["abs_gap_bp"], errors="coerce").max())
        if not quarter_report.empty and "abs_gap_bp" in quarter_report.columns
        else 0.0
    )
    return {
        "n_eval": int(len(eval_frame)),
        "observed_default_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "mean_predicted_pd": float(np.mean(y_prob)) if len(y_prob) else 0.0,
        "global_gap_bp": float((np.mean(y_true) - np.mean(y_prob)) * 10_000.0)
        if len(y_prob)
        else 0.0,
        "abs_global_gap_bp": float(abs(np.mean(y_true) - np.mean(y_prob)) * 10_000.0)
        if len(y_prob)
        else 0.0,
        "brier_score": float(metrics["brier_score"]),
        "ece": float(metrics["ece"]),
        "auc_roc": float(metrics["auc_roc"]),
        "hl_p_value": float(hl["hl_p_value"]),
        "material_quarter_breaches": int(
            (
                pd.to_numeric(
                    quarter_report.get("abs_gap_bp", pd.Series(dtype=float)), errors="coerce"
                )
                >= 100.0
            ).sum()
        )
        if not quarter_report.empty
        else 0,
        "max_quarter_gap_bp": max_quarter_gap_bp,
        "grade_material_breaches": int(slice_materiality.get("grade_material_breaches", 0)),
        "band_material_breaches": int(slice_materiality.get("band_material_breaches", 0)),
        "max_grade_gap_bp": float(slice_materiality.get("max_grade_gap_bp", 0.0)),
        "max_band_gap_bp": float(slice_materiality.get("max_band_gap_bp", 0.0)),
    }


def temporal_otv_split(
    frame: pd.DataFrame,
    *,
    min_eval_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split OOT rows into adaptation and evaluation windows."""
    ordered = frame.copy()
    if "issue_d" in ordered.columns:
        ordered["__sort_date"] = pd.to_datetime(ordered["issue_d"], errors="coerce")
        ordered = ordered.sort_values("__sort_date", kind="stable").drop(columns="__sort_date")
    elif "issue_quarter" in ordered.columns:
        ordered = ordered.assign(__sort_q=ordered["issue_quarter"].astype(str)).sort_values(
            "__sort_q", kind="stable"
        )
        ordered = ordered.drop(columns="__sort_q")
    midpoint = max(int(math.floor(len(ordered) / 2)), 1)
    adaptation = ordered.iloc[:midpoint].copy()
    evaluation = ordered.iloc[midpoint:].copy()
    if len(evaluation) < min_eval_rows and len(ordered) >= min_eval_rows:
        evaluation = ordered.iloc[-min_eval_rows:].copy()
        adaptation = ordered.iloc[: max(len(ordered) - min_eval_rows, 1)].copy()
    return adaptation, evaluation


def evaluate_stage_a_gate(
    current_candidate: dict[str, Any],
    challenger_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether a shadow calibrator is promising enough for downstream validation."""
    gap_improvement_bp = float(current_candidate.get("abs_global_gap_bp", 0.0)) - float(
        challenger_candidate.get("abs_global_gap_bp", 0.0)
    )
    max_quarter_improvement_bp = float(current_candidate.get("max_quarter_gap_bp", 0.0)) - float(
        challenger_candidate.get("max_quarter_gap_bp", 0.0)
    )
    quarter_breaches_delta = int(challenger_candidate.get("material_quarter_breaches", 0)) - int(
        current_candidate.get("material_quarter_breaches", 0)
    )
    brier_delta = float(challenger_candidate.get("brier_score", 0.0)) - float(
        current_candidate.get("brier_score", 0.0)
    )
    ece_delta = float(challenger_candidate.get("ece", 0.0)) - float(
        current_candidate.get("ece", 0.0)
    )
    auc_delta = float(challenger_candidate.get("auc_roc", 0.0)) - float(
        current_candidate.get("auc_roc", 0.0)
    )

    checks = {
        "min_gap_improvement_bp": bool(gap_improvement_bp >= 10.0),
        "quarter_breaches_non_increasing": bool(quarter_breaches_delta <= 0),
        "max_quarter_gap_non_worsening": bool(max_quarter_improvement_bp >= 0.0),
        "brier_within_tolerance": bool(brier_delta <= 0.002),
        "ece_within_tolerance": bool(ece_delta <= 0.002),
        "auc_within_tolerance": bool(auc_delta >= -0.001),
    }
    return {
        "stage_a_pass": bool(all(checks.values())),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "deltas": {
            "abs_global_gap_improvement_bp": gap_improvement_bp,
            "max_quarter_gap_improvement_bp": max_quarter_improvement_bp,
            "material_quarter_breaches_delta": quarter_breaches_delta,
            "brier_delta": brier_delta,
            "ece_delta": ece_delta,
            "auc_delta": auc_delta,
        },
    }


def calibration_mapping_candidates_report(
    frame: pd.DataFrame,
    *,
    target_col: str = "default_flag",
    score_col: str = "pd_calibrated",
) -> pd.DataFrame:
    """Evaluate current calibration against sidecar remapping candidates."""
    base = frame.copy()
    base[target_col] = pd.to_numeric(base[target_col], errors="coerce")
    base[score_col] = pd.to_numeric(base[score_col], errors="coerce")
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col, score_col])
    if len(base) < 1_000:
        return pd.DataFrame()

    adaptation, evaluation = temporal_otv_split(base)
    candidates: list[dict[str, Any]] = []

    current_eval = evaluation.copy()
    current_eval["candidate_score"] = current_eval[score_col].astype(float)
    candidates.append(
        {
            "candidate_id": "current_identity",
            "fit_window_rows": int(len(adaptation)),
            "eval_window_rows": int(len(evaluation)),
            "fit_notes": "No remapping; current canonical calibrated PD.",
            "candidate_kind": "identity",
            "candidate_spec": {"type": "identity"},
            **evaluate_calibration_candidate(
                current_eval, score_col="candidate_score", target_col=target_col
            ),
        }
    )

    delta = logit_intercept_shift(
        adaptation[target_col].to_numpy(dtype=float),
        adaptation[score_col].to_numpy(dtype=float),
    )
    intercept_eval = evaluation.copy()
    intercept_eval["candidate_score"] = apply_logit_intercept_shift(
        intercept_eval[score_col].to_numpy(dtype=float),
        delta,
    )
    candidates.append(
        {
            "candidate_id": "logit_intercept_shift",
            "fit_window_rows": int(len(adaptation)),
            "eval_window_rows": int(len(evaluation)),
            "fit_notes": f"Delta fitted on earlier OOT window: {delta:.4f}",
            "candidate_kind": "calibrator_object",
            "candidate_spec": {"type": "logit_shift", "delta": float(delta)},
            "intercept_delta": float(delta),
            **evaluate_calibration_candidate(
                intercept_eval, score_col="candidate_score", target_col=target_col
            ),
        }
    )

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(
        adaptation[score_col].to_numpy(dtype=float),
        adaptation[target_col].to_numpy(dtype=float),
    )
    isotonic_eval = evaluation.copy()
    isotonic_eval["candidate_score"] = np.clip(
        np.asarray(iso.predict(isotonic_eval[score_col].to_numpy(dtype=float)), dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    candidates.append(
        {
            "candidate_id": "isotonic_sidecar",
            "fit_window_rows": int(len(adaptation)),
            "eval_window_rows": int(len(evaluation)),
            "fit_notes": "Monotone remap fitted on earlier OOT window only.",
            "candidate_kind": "calibrator_object",
            "candidate_spec": {
                "type": "isotonic",
                "x_thresholds": [float(x) for x in np.asarray(iso.X_thresholds_, dtype=float)],
                "y_thresholds": [float(y) for y in np.asarray(iso.y_thresholds_, dtype=float)],
                "y_min": 0.0,
                "y_max": 1.0,
            },
            **evaluate_calibration_candidate(
                isotonic_eval, score_col="candidate_score", target_col=target_col
            ),
        }
    )

    report = pd.DataFrame(candidates)
    report["stage_a_pass"] = False
    report["stage_a_reasons"] = [[] for _ in range(len(report))]
    report["stage_a_deltas"] = [{} for _ in range(len(report))]
    if not report.empty and (report["candidate_id"] == "current_identity").any():
        current = report.loc[report["candidate_id"] == "current_identity"].iloc[0].to_dict()
        for idx, row in report.iterrows():
            if str(row.get("candidate_id")) == "current_identity":
                continue
            evaluation = evaluate_stage_a_gate(current, row.to_dict())
            report.at[idx, "stage_a_pass"] = bool(evaluation["stage_a_pass"])
            report.at[idx, "stage_a_reasons"] = list(evaluation["reasons"])
            report.at[idx, "stage_a_deltas"] = dict(evaluation["deltas"])
    return report.sort_values(
        ["stage_a_pass", "abs_global_gap_bp", "material_quarter_breaches", "brier_score", "ece"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def materialize_candidate_calibrator(candidate_spec: dict[str, Any] | None) -> Any | None:
    """Build a serializable calibrator object from a persisted candidate spec."""
    spec = dict(candidate_spec or {})
    candidate_type = str(spec.get("type", "")).strip().lower()
    if candidate_type in {"", "identity"}:
        return None
    if candidate_type == "logit_shift":
        return LogitShiftCalibrator(float(spec["delta"]))
    if candidate_type == "isotonic":
        x_thresholds = np.asarray(spec.get("x_thresholds", []), dtype=float)
        y_thresholds = np.asarray(spec.get("y_thresholds", []), dtype=float)
        iso = IsotonicRegression(
            y_min=float(spec.get("y_min", 0.0)),
            y_max=float(spec.get("y_max", 1.0)),
            out_of_bounds="clip",
        )
        if x_thresholds.size == 0 or y_thresholds.size == 0:
            raise ValueError("Isotonic candidate spec requires non-empty thresholds.")
        iso.fit(x_thresholds, y_thresholds)
        return iso
    raise ValueError(f"Unsupported calibration mapping candidate type: {candidate_type}")
