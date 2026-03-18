"""Export a frozen storytelling snapshot for demos/defense sessions.

Usage:
    uv run python scripts/export_storytelling_snapshot.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.threshold_semantics import load_threshold_semantics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_PATH = REPORTS_DIR / "storytelling_snapshot.json"
BASELINE_REGISTRY_PATH = (
    PROJECT_ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
)
SCHEMA_VERSION = "2026-03-13.2"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_meta(path: Path) -> dict[str, object]:
    exists = path.exists()
    if not exists:
        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": False,
            "updated_at_utc": None,
            "size_bytes": 0,
        }

    stat = path.stat()
    updated = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "updated_at_utc": updated,
        "size_bytes": int(stat.st_size),
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    pipeline_summary = _load_json(DATA_DIR / "pipeline_summary.json")
    conformal_status = _load_json(MODEL_DIR / "conformal_policy_status.json")
    conformal_sensitivity = _load_json(MODEL_DIR / "conformal_policy_sensitivity_status.json")
    conformal_variant_status = _load_json(MODEL_DIR / "conformal_variant_selection_status.json")
    conformal_method_registry = _load_json(MODEL_DIR / "conformal_method_registry.json")
    model_comparison = _load_json(DATA_DIR / "model_comparison.json")
    baseline_registry = _load_json(BASELINE_REGISTRY_PATH)
    threshold_semantics = load_threshold_semantics()
    fairness_status = _load_json(MODEL_DIR / "fairness_audit_status.json")
    time_series_status = _load_json(MODEL_DIR / "time_series_status.json")
    causal_status = _load_json(MODEL_DIR / "causal_policy_rule.json")
    pd_set_prediction = _load_json(MODEL_DIR / "pd_set_prediction_status.json")
    pd_rare_event = _load_json(MODEL_DIR / "pd_rare_event_calibration_status.json")
    official_run_tag = str(baseline_registry.get("official_run_tag", "")).strip()
    comparison_payload = (
        _load_json(REPORTS_DIR / "run_comparisons" / official_run_tag / "comparison.json")
        if official_run_tag
        else {}
    )

    required_files = {
        "pipeline_summary": DATA_DIR / "pipeline_summary.json",
        "model_comparison": DATA_DIR / "model_comparison.json",
        "conformal_policy_status": MODEL_DIR / "conformal_policy_status.json",
        "ifrs9_scenario_summary": DATA_DIR / "ifrs9_scenario_summary.parquet",
        "portfolio_robustness_summary": DATA_DIR / "portfolio_robustness_summary.parquet",
    }
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts for storytelling snapshot: " + ", ".join(sorted(missing))
        )

    ifrs9 = pd.read_parquet(required_files["ifrs9_scenario_summary"])
    robustness = pd.read_parquet(required_files["portfolio_robustness_summary"])

    pipeline = pipeline_summary.get("pipeline", {})
    final_metrics = model_comparison.get("final_test_metrics", {})

    baseline_row = ifrs9[ifrs9["scenario"] == "baseline"]
    severe_row = ifrs9[ifrs9["scenario"] == "severe"]
    baseline_ecl = _safe_float(baseline_row["total_ecl"].iloc[0]) if not baseline_row.empty else 0.0
    severe_ecl = _safe_float(severe_row["total_ecl"].iloc[0]) if not severe_row.empty else 0.0

    robust_row = robustness.sort_values("risk_tolerance").iloc[0] if not robustness.empty else None
    robust_price = _safe_float(robust_row["price_of_robustness"]) if robust_row is not None else 0.0
    robust_price_pct = (
        _safe_float(robust_row["price_of_robustness_pct"]) if robust_row is not None else 0.0
    )

    required_artifacts = [
        DATA_DIR / "pipeline_summary.json",
        MODEL_DIR / "conformal_results_mondrian.pkl",
        DATA_DIR / "conformal_intervals_mondrian.parquet",
        MODEL_DIR / "conformal_policy_status.json",
        DATA_DIR / "portfolio_robustness_summary.parquet",
        DATA_DIR / "ifrs9_scenario_summary.parquet",
    ]

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

    snapshot = {
        "snapshot_name": "storytelling_snapshot",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_scope": pipeline_summary.get("dataset_scope", "unknown"),
        "official_run_tag": official_run_tag or conformal_status.get("run_tag", "unknown"),
        "threshold_semantics": threshold_semantics,
        "conformal_promotion_pass": bool(
            conformal_variant_status.get(
                "promotion_pass",
                comparison_payload.get("conformal_promotion_pass", False),
            )
        ),
        "conformal_strict_policy_pass": bool(conformal_status.get("strict_overall_pass", False)),
        "conformal_methodological_justification_pass": bool(
            conformal_status.get("methodological_justification_pass", False)
        ),
        "conformal_methodological_closure_candidate": bool(sensitivity_closure),
        "conformal_methodological_closure_threshold": (
            float(sensitivity_closure.get("max_winkler_90"))
            if sensitivity_closure is not None
            else None
        ),
        "conformal_statistical_warning": bool(
            len(conformal_status.get("failing_statistical_checks", []) or []) > 0
            or (conformal_status.get("warning_alerts", 0) or 0) > 0
            or comparison_payload.get("conformal_statistical_warning", False)
        ),
        "time_series_interval_promotable": bool(
            (time_series_status.get("interval_champion", {}) or {}).get("promotable", False)
        ),
        "time_series_final_interval_decision": (
            (time_series_status.get("final_interval_decision", {}) or {}).get("status")
        ),
        "causal_status": {
            "selected_rule": causal_status.get("selected_rule"),
            "selection_reason": causal_status.get("selection_reason"),
            "run_tag": causal_status.get("run_tag"),
        },
        "ab_gate_mode": str(comparison_payload.get("ab_gate_mode") or "comparison_gate_unknown"),
        "method_registry_status": str(
            conformal_method_registry.get("libraries", {}).get("mapie", {}).get("status", "unknown")
        ),
        "headline_metrics": {
            "auc_oot": _safe_float(
                final_metrics.get("auc_roc"), _safe_float(pipeline.get("pd_auc"))
            ),
            "coverage_90": _safe_float(conformal_status.get("coverage_90")),
            "coverage_95": _safe_float(conformal_status.get("coverage_95")),
            "pd_pr_auc": _safe_float((pd_rare_event.get("global", {}) or {}).get("pr_auc")),
            "pd_ambiguity_rate": _safe_float(
                (pd_set_prediction.get("summary", {}) or {}).get("ambiguity_rate")
            ),
            "price_of_robustness": _safe_float(pipeline.get("price_of_robustness"), robust_price),
            "price_of_robustness_pct": robust_price_pct,
            "robust_return": _safe_float(pipeline.get("robust_return")),
            "nonrobust_return": _safe_float(pipeline.get("nonrobust_return")),
            "baseline_ecl": baseline_ecl,
            "severe_ecl": severe_ecl,
            "severe_uplift_pct": ((severe_ecl / baseline_ecl - 1.0) * 100.0)
            if baseline_ecl > 0
            else 0.0,
            "fairness_primary_threshold": _safe_float(
                threshold_semantics.get("fairness_primary_threshold"),
                _safe_float(fairness_status.get("primary_threshold")),
            ),
        },
        "artifact_health": [_artifact_meta(path) for path in required_artifacts],
        "notes": [
            "Snapshot con métricas para storytelling reproducible.",
            "Fuente conformal canónica: conformal_results_mondrian.pkl + conformal_intervals_mondrian.parquet.",
            "Threshold PD interno y threshold operativo de fairness se reportan por separado.",
            "Classification sets y rare-event calibration son sidecars diagnósticos; no forman parte del champion operativo actual.",
        ],
    }

    required_headline_keys = [
        "auc_oot",
        "coverage_90",
        "coverage_95",
        "baseline_ecl",
        "severe_ecl",
    ]
    missing_headline = [k for k in required_headline_keys if k not in snapshot["headline_metrics"]]
    if missing_headline:
        raise ValueError(
            "Storytelling snapshot missing headline keys: " + ", ".join(sorted(missing_headline))
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info(f"Saved storytelling snapshot: {OUT_PATH}")


if __name__ == "__main__":
    main()
