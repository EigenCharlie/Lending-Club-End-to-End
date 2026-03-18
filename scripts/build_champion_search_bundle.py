"""Assemble a promotion-ready bundle from the latest champion-search artifacts."""

from __future__ import annotations

import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.baseline_registry import resolve_official_baseline_run_tag
from src.utils.threshold_semantics import load_threshold_semantics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
BASELINES = ROOT / "configs" / "baselines"
SCHEMA_VERSION = "2026-03-13.2"


def _meaningful_run_tag(*values: object) -> str:
    for value in values:
        candidate = str(value or "").strip()
        if candidate and candidate.lower() not in {"untracked", "unknown"}:
            return candidate
    return "untracked"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_upstream_baseline() -> str | None:
    return resolve_official_baseline_run_tag()


def main() -> None:
    training_record = _load_pickle(MODELS / "pd_training_record.pkl")
    model_comparison = _load_json(DATA / "model_comparison.json")
    conformal_status = _load_json(MODELS / "conformal_policy_status.json")
    conformal_method_registry = _load_json(MODELS / "conformal_method_registry.json")
    conformal_variant_status = _load_json(MODELS / "conformal_variant_selection_status.json")
    fairness_status = _load_json(MODELS / "fairness_audit_status.json")
    champion_policy = _load_json(MODELS / "champion_portfolio_policy.json")
    causal_effect = _load_json(MODELS / "causal_effect_status.json")
    causal_rule = _load_json(MODELS / "causal_policy_rule.json")
    causal_oot = _load_json(MODELS / "causal_policy_oot_status.json")
    time_series = _load_json(MODELS / "time_series_status.json")
    cate_status = _load_json(MODELS / "cate_portfolio_status.json")
    governance_status = _load_json(MODELS / "governance_status.json")
    pd_set_prediction = _load_json(MODELS / "pd_set_prediction_status.json")
    pd_rare_event = _load_json(MODELS / "pd_rare_event_calibration_status.json")
    pd_calibration_diagnostics = _load_json(MODELS / "pd_calibration_diagnostics.json")
    paper_grade_protocol = _load_json(MODELS / "paper_grade_protocol_status.json")
    survival_summary = _load_pickle(MODELS / "survival_summary.pkl")
    threshold_semantics = load_threshold_semantics()
    upstream_baseline = _resolve_upstream_baseline()

    run_tag = _meaningful_run_tag(
        os.environ.get("PIPELINE_RUN_TAG", ""),
        governance_status.get("run_tag", ""),
        champion_policy.get("run_tag", ""),
        threshold_semantics.get("run_tag", ""),
        upstream_baseline,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": run_tag,
        "pipeline_family": str(
            os.environ.get("PIPELINE_FAMILY", "champion_search") or "champion_search"
        ),
        "pipeline_profile": str(
            os.environ.get("PIPELINE_PROFILE", "champion_search_max") or "champion_search_max"
        ),
        "artifact_scope": "search",
        "promotion_state": "research_open",
        "writes_canonical_artifacts": False,
        "upstream_canonical_run_tag": upstream_baseline,
        "threshold_semantics": threshold_semantics,
        "pd": {
            "best_model": model_comparison.get("best_model"),
            "best_calibration": model_comparison.get("best_calibration"),
            "training_regime": training_record.get("training_regime", {}),
            "stable_core": training_record.get("stable_core", {}),
            "decision_threshold": training_record.get("decision_threshold", {}),
            "decision_threshold_semantics": {
                "pd_internal_selected_threshold": threshold_semantics.get(
                    "pd_internal_selected_threshold"
                ),
                "fairness_primary_threshold": threshold_semantics.get("fairness_primary_threshold"),
                "decision_policy_global_threshold": threshold_semantics.get(
                    "decision_policy_global_threshold"
                ),
            },
            "set_prediction": pd_set_prediction,
            "rare_event_calibration": pd_rare_event,
            "calibration_diagnostics": pd_calibration_diagnostics,
        },
        "conformal": {
            "policy_status": conformal_status,
            "method_registry": conformal_method_registry,
            "variant_selection": conformal_variant_status,
        },
        "fairness": fairness_status,
        "portfolio": champion_policy,
        "survival": survival_summary,
        "time_series": time_series,
        "causal": {
            "effect_status": causal_effect,
            "policy_rule": causal_rule,
            "oot_status": causal_oot,
            "cate_portfolio_status": cate_status,
        },
        "governance": governance_status,
        "paper_grade_protocol": paper_grade_protocol,
    }
    out_path = MODELS / "champion_search_bundle.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[champion_search_bundle] saved {out_path}")


if __name__ == "__main__":
    main()
