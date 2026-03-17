"""Freeze a paper-grade protocol status from current canonical artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SCHEMA_VERSION = "2026-03-13.1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _derive_causal_rule_mask(df: pd.DataFrame, rule_name: str) -> np.ndarray:
    q85 = float(df["cate"].quantile(0.85))
    q90 = float(df["cate"].quantile(0.90))
    if rule_name == "high_only":
        return df["segment"].eq("high_sensitivity").to_numpy()
    if rule_name == "high_plus_medium_all":
        return df["segment"].isin(["high_sensitivity", "medium_sensitivity"]).to_numpy()
    if rule_name == "high_plus_medium_positive":
        return (
            df["segment"].eq("high_sensitivity")
            | (df["segment"].eq("medium_sensitivity") & (df["net_value"] > 0))
        ).to_numpy()
    if rule_name == "top15_cate":
        return (df["cate"] >= q85).to_numpy()
    if rule_name == "top10_cate":
        return (df["cate"] >= q90).to_numpy()
    return np.zeros(len(df), dtype=bool)


def _build_causal_uncertainty_diagnostic() -> dict[str, Any]:
    sim_path = DATA / "causal_policy_simulation.parquet"
    set_path = DATA / "pd_set_prediction_cases.parquet"
    intervals_path = DATA / "conformal_intervals_mondrian.parquet"
    causal_rule = _load_json(MODELS / "causal_policy_rule.json")
    if (
        not sim_path.exists()
        or not set_path.exists()
        or not intervals_path.exists()
        or not causal_rule
    ):
        return {"available": False, "reason": "missing_artifacts"}

    sim = pd.read_parquet(sim_path).reset_index(drop=True)
    sets = pd.read_parquet(set_path).reset_index(drop=True)
    intervals = pd.read_parquet(intervals_path).reset_index(drop=True)
    n = min(len(sim), len(sets), len(intervals))
    if n == 0:
        return {"available": False, "reason": "empty_artifacts"}
    sim = sim.iloc[:n].copy()
    sets = sets.iloc[:n].copy()
    intervals = intervals.iloc[:n].copy()
    sim["ambiguous"] = sets["ambiguous"].astype(int).to_numpy()
    sim["interval_width_90"] = (
        pd.to_numeric(intervals["pd_high_90"], errors="coerce")
        - pd.to_numeric(intervals["pd_low_90"], errors="coerce")
    ).to_numpy(dtype=float)
    rule_name = str(causal_rule.get("selected_rule", "")).strip()
    selected_mask = _derive_causal_rule_mask(sim, rule_name)
    sim["selected_by_policy"] = selected_mask.astype(int)

    summary = {
        "available": True,
        "selected_rule": rule_name,
        "selected_n": int(selected_mask.sum()),
        "selected_ambiguity_rate": float(sim.loc[selected_mask, "ambiguous"].mean())
        if selected_mask.any()
        else 0.0,
        "selected_avg_interval_width_90": float(sim.loc[selected_mask, "interval_width_90"].mean())
        if selected_mask.any()
        else 0.0,
        "overall_ambiguity_rate": float(sim["ambiguous"].mean()),
        "overall_avg_interval_width_90": float(sim["interval_width_90"].mean()),
    }
    out_path = DATA / "causal_uncertainty_diagnostic.parquet"
    sim[
        [
            "segment",
            "cate",
            "net_value",
            "ambiguous",
            "interval_width_90",
            "selected_by_policy",
        ]
    ].to_parquet(out_path, index=False)
    summary["artifact_path"] = str(out_path)
    return summary


def main() -> None:
    prior_payload = _load_json(MODELS / "paper_grade_protocol_status.json")
    conformal = _load_json(MODELS / "conformal_policy_status.json")
    conformal_sensitivity = _load_json(MODELS / "conformal_policy_sensitivity_status.json")
    conformal_variant = _load_json(MODELS / "conformal_variant_selection_status.json")
    time_series = _load_json(MODELS / "time_series_status.json")
    causal_rule = _load_json(MODELS / "causal_policy_rule.json")
    causal_oot = _load_json(MODELS / "causal_policy_oot_status.json")
    ab_default = _load_json(MODELS / "ab_simulation_status.json")
    ab_ambiguity = _load_json(MODELS / "ab_simulation_status_ambiguity_defer.json")
    ab_selective = _load_json(MODELS / "ab_simulation_status_selective_ambiguity_defer.json")
    set_prediction = _load_json(MODELS / "pd_set_prediction_status.json")
    governance = _load_json(MODELS / "governance_status.json")
    resolved_run_tag = resolve_run_tag(
        fallback_candidates=[
            conformal.get("run_tag"),
            governance.get("run_tag"),
            time_series.get("run_tag"),
            resolve_official_baseline_run_tag(),
        ],
        require_explicit=True,
    )
    final_run_tag = (
        str(os.environ.get("PAPER_GRADE_FINAL_RUN_TAG", "")).strip()
        or str(prior_payload.get("final_run_tag", "")).strip()
        or None
    )
    final_run_promoted = _coerce_bool(os.environ.get("PAPER_GRADE_FINAL_RUN_PROMOTED"))
    if final_run_promoted is None:
        final_run_promoted = prior_payload.get("final_run_promoted")
    final_rebuild_verified = _coerce_bool(os.environ.get("PAPER_GRADE_FINAL_REBUILD_VERIFIED"))
    if final_rebuild_verified is None:
        final_rebuild_verified = prior_payload.get("final_rebuild_verified")
    final_run_comparison_path = (
        str(os.environ.get("PAPER_GRADE_FINAL_RUN_COMPARISON_PATH", "")).strip()
        or str(prior_payload.get("final_run_comparison_path", "")).strip()
        or None
    )

    causal_uncertainty = _build_causal_uncertainty_diagnostic()

    sensitivity_results = (conformal_sensitivity.get("policy_sensitivity", {}) or {}).get(
        "results", []
    ) or []
    sensitivity_closure = next(
        (
            row
            for row in sensitivity_results
            if bool(row.get("methodological_justification_pass", False))
            and not (row.get("failing_non_statistical_checks", []) or [])
        ),
        None,
    )

    pd_conformal_closed = bool(
        conformal.get("strict_overall_pass", False)
        or (
            conformal.get("methodological_justification_pass", False)
            and not (conformal.get("failing_non_statistical_checks", []) or [])
        )
        or sensitivity_closure
    )
    time_series_decision = (time_series.get("final_interval_decision", {}) or {}).get(
        "status", "research_only"
    )
    causal_role = str(causal_rule.get("role", "insights_only")).strip() or "insights_only"
    causal_closed = causal_role == "insights_only" or bool(
        causal_rule.get("promotion_eligible", False)
    )
    ab_sidecar_available = bool(ab_ambiguity)
    ab_selective_available = bool(ab_selective)
    ab_default_no_regression = bool(
        (ab_default.get("no_regression", {}) or {}).get("passed", False)
    )
    ab_ambiguity_no_regression = bool(
        (ab_ambiguity.get("no_regression", {}) or {}).get("passed", False)
    )
    ab_selective_no_regression = bool(
        (ab_selective.get("no_regression", {}) or {}).get("passed", False)
    )
    ab_selective_cross_scenario_pass = bool(
        (ab_selective.get("cross_scenario_gate", {}) or {}).get("passed", False)
    )
    ab_selective_promoted = bool(ab_selective.get("promoted", False))
    set_prediction_promoted = bool(set_prediction.get("promoted", False))
    governance_summary = governance.get("summary", {}) or {}

    payload = {
        "pd_conformal": {
            "closed_for_paper_grade": pd_conformal_closed,
            "strict_overall_pass": bool(conformal.get("strict_overall_pass", False)),
            "canonical_methodological_justification_pass": bool(
                conformal.get("methodological_justification_pass", False)
            ),
            "sensitivity_methodological_closure_available": bool(sensitivity_closure),
            "sensitivity_methodological_closure_threshold": (
                float(sensitivity_closure.get("max_winkler_90"))
                if sensitivity_closure is not None
                else None
            ),
            "promotion_pass": bool(conformal_variant.get("promotion_pass", False)),
            "failing_non_statistical_checks": conformal.get("failing_non_statistical_checks", []),
            "failing_statistical_checks": conformal.get("failing_statistical_checks", []),
        },
        "time_series": {
            "decision": str(time_series_decision),
            "point_model": (time_series.get("point_champion", {}) or {}).get("model"),
            "interval_model": (time_series.get("interval_champion", {}) or {}).get("model"),
            "interval_promotable": bool(
                (time_series.get("interval_champion", {}) or {}).get("promotable", False)
            ),
            "adaptive_best_method": (time_series.get("adaptive_method_status", {}) or {}).get(
                "best_method"
            ),
        },
        "causal_cate": {
            "closed_for_paper_grade": causal_closed,
            "role": causal_role,
            "promotion_eligible": bool(causal_rule.get("promotion_eligible", False)),
            "selected_rule": causal_rule.get("selected_rule"),
            "selection_reason": causal_rule.get("selection_reason"),
            "oot_status": causal_oot,
            "uncertainty_diagnostic": causal_uncertainty,
            "limits_note": (
                "Current causal lane is insights_only until identification, continuous-treatment "
                "semantics and policy evaluation are strengthened."
            ),
        },
        "ab_evidence": {
            "default_scenario": ab_default,
            "ambiguity_defer_scenario_available": ab_sidecar_available,
            "ambiguity_defer_scenario": ab_ambiguity if ab_sidecar_available else {},
            "selective_ambiguity_defer_available": ab_selective_available,
            "selective_ambiguity_defer_scenario": ab_selective if ab_selective_available else {},
            "decision_rule": (
                "prefer selective_ambiguity_defer if it passes no_regression; "
                "fallback to baseline if neither defer scenario passes"
            ),
            "ab_artifacts_present": bool(ab_default),
            "ab_no_regression_pass": bool(ab_default_no_regression),
            "ab_gate_summary": {
                "default_passed_no_regression": bool(ab_default_no_regression),
                "ambiguity_defer_passed_no_regression": bool(ab_ambiguity_no_regression)
                if ab_sidecar_available
                else None,
                "selective_ambiguity_defer_passed_no_regression": bool(ab_selective_no_regression)
                if ab_selective_available
                else None,
                "selective_ambiguity_defer_cross_scenario_pass": bool(
                    ab_selective_cross_scenario_pass
                )
                if ab_selective_available
                else None,
                "selective_ambiguity_defer_promoted": bool(ab_selective_promoted)
                if ab_selective_available
                else None,
                "recommended_scenario": (
                    "selective_ambiguity_defer"
                    if ab_selective_available
                    and (ab_selective_no_regression or ab_selective_cross_scenario_pass)
                    else (
                        "ambiguity_defer"
                        if ab_sidecar_available and ab_ambiguity_no_regression
                        else "default"
                    )
                ),
                "gate_rule": "no_regression",
            },
        },
        "set_prediction": {
            "promoted": set_prediction_promoted,
            "status": set_prediction.get("status", "unknown"),
            "promotion_gate": set_prediction.get("promotion_gate", {}),
            "promotion_rationale": set_prediction.get("promotion_rationale", ""),
        },
        "governance": {
            "overall_pass": bool(governance.get("overall_pass", False)),
            "warnings": governance.get("warnings", {}),
            "contextualized_warning_note": (
                "Warnings are interpreted with materiality and temporal/subgroup reliability; "
                "the project does not claim exact conditional coverage."
            ),
            "summary": {
                "max_psi": governance_summary.get("max_psi"),
                "distribution_warning_ratio": governance_summary.get("distribution_warning_ratio"),
                "c2st_auc": governance_summary.get("c2st_auc"),
            },
        },
        "final_checks": {
            "pd_conformal": pd_conformal_closed,
            "time_series": bool(time_series_decision in {"promoted", "research_only"}),
            "causal_cate": causal_closed,
            "ab_evidence": bool(ab_default),
            "ab_no_regression": bool(ab_default_no_regression),
            "set_prediction_promoted": set_prediction_promoted,
            "selective_ambiguity_defer_pass": bool(
                ab_selective_available
                and (ab_selective_no_regression or ab_selective_cross_scenario_pass)
            ),
            "governance": bool(governance),
            "protocol_frozen": True,
        },
        "final_run_tag": final_run_tag,
        "final_run_promoted": final_run_promoted,
        "final_run_comparison_path": final_run_comparison_path,
        "final_rebuild_verified": final_rebuild_verified,
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }

    out_path = MODELS / "paper_grade_protocol_status.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Saved paper-grade protocol status: {}", out_path)


if __name__ == "__main__":
    main()
