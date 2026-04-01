"""Build a frozen clean-baseline replay manifest from blessed artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.replay_manifest import (
    artifact_descriptor,
    resolve_manifest_path,
    save_replay_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_pickle(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "rb") as handle:
        return pickle.load(handle)


def _coerce_rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _snapshot_metrics(snapshot: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    metrics = snapshot.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    payload = metrics.get(key, {})
    return dict(payload) if isinstance(payload, dict) else {}


def _infer_pd_config_path_from_run(source_run_tag: str) -> str | None:
    status_dir = ROOT / "reports" / "run_logs" / source_run_tag / "status"
    payload = _load_json(status_dir / "core_data_pd.json")
    command = str(payload.get("command", "")).strip()
    if command:
        match = re.search(r"scripts/train_pd_model\.py\s+--config\s+(\S+)", command)
        if match:
            return str(match.group(1)).strip()
    return None


def build_manifest(
    *,
    source_run_tag: str,
    conformal_namespace: str,
    baseline_snapshot_path: str | Path | None = None,
    pd_config_path_override: str | None = None,
) -> dict[str, Any]:
    snapshot_path = (
        baseline_snapshot_path or f"reports/run_comparisons/{source_run_tag}/baseline_snapshot.json"
    )
    baseline_snapshot = _load_json(snapshot_path)
    pd_status = _load_json("models/pd_training_status.json")
    pd_record = _load_pickle("models/pd_training_record.pkl")
    pd_contract = _load_json("models/pd_model_contract.json")
    threshold_semantics = _load_json("models/threshold_semantics.json")
    decision_threshold = _load_json("models/decision_threshold.json")
    conformal_status = _load_json(
        f"models/conformal_gap/{conformal_namespace}/conformal_policy_status.json"
    )
    conformal_results = _load_pickle(
        f"models/conformal_gap/{conformal_namespace}/conformal_results_mondrian.pkl"
    )
    champion_policy = _load_json("models/champion_portfolio_policy.json")
    ab_status = _load_json("models/ab_simulation_status.json")
    selector_status = _load_json("models/champion_policy_selection_status.json")
    snapshot_model_comparison = _snapshot_metrics(baseline_snapshot, "model_comparison")
    _snapshot_metrics(baseline_snapshot, "pipeline_summary")
    snapshot_conformal = _snapshot_metrics(baseline_snapshot, "conformal_status")
    snapshot_ab = _snapshot_metrics(baseline_snapshot, "ab_simulation_status")

    seed_replay = dict(pd_record.get("seed_replay_report") or {})
    selected_params = (
        seed_replay.get("selected_params") or pd_record.get("optuna_best_params") or {}
    )
    calibration_report = dict(
        pd_record.get("calibration_selection_report")
        or snapshot_model_comparison.get("calibration_selection_report")
        or {}
    )
    final_metrics = dict(
        pd_record.get("final_test_metrics")
        or snapshot_model_comparison.get("final_test_metrics")
        or {}
    )
    tune_best = dict(conformal_results.get("tuning_90_best") or {})
    champion_selected_policy = dict(champion_policy.get("selected_policy") or {})
    snapshot_robust_policy = dict(
        ab_status.get("robust_policy") or snapshot_ab.get("robust_policy") or {}
    )
    selected_policy = champion_selected_policy or snapshot_robust_policy
    pd_config_path = (
        str(pd_config_path_override).strip()
        if str(pd_config_path_override or "").strip()
        else _infer_pd_config_path_from_run(source_run_tag)
        or str(pd_status.get("config_path", "configs/pd_model.champion.yaml"))
    )
    manifest = {
        "schema_version": "2026-03-26.2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run_tag": str(source_run_tag),
        "baseline_snapshot_path": _coerce_rel(snapshot_path),
        "notes": [
            "Frozen baseline manifest for deterministic replay.",
            "PD replay uses fixed params, fixed feature order and fixed calibration method.",
            "Conformal replay restores the blessed grade/no-shrink recovery artifacts.",
            "Portfolio confirmatory rebuild re-runs tradeoff + selector + AB with the champion policy search settings.",
        ],
        "pd": {
            "config_path": pd_config_path,
            "model_contract_path": "models/pd_model_contract.json",
            "feature_names": list(pd_contract.get("feature_names", [])),
            "categorical_features": list(pd_contract.get("categorical_features", [])),
            "selected_params": selected_params,
            "selected_calibration_method": str(
                calibration_report.get("selected_method", "venn_abers")
            ),
            "decision_threshold_artifact": decision_threshold,
            "threshold_semantics": threshold_semantics,
            "expectations": {
                "auc_roc": final_metrics.get("auc_roc"),
                "brier_score": final_metrics.get("brier_score"),
                "ece": final_metrics.get("ece"),
                "d2_brier_score": final_metrics.get("d2_brier_score"),
            },
            "tolerances": {
                "auc_roc": 0.0025,
                "brier_score": 0.003,
                "ece": 0.002,
                "d2_brier_score": 0.002,
            },
            "artifacts": {
                "status": artifact_descriptor("models/pd_training_status.json"),
                "record": artifact_descriptor("models/pd_training_record.pkl"),
                "contract": artifact_descriptor("models/pd_model_contract.json"),
            },
        },
        "conformal": {
            "source_namespace": str(conformal_namespace),
            "replay_mode": "restore_blessed_namespace",
            "source_artifacts": {
                "results": _coerce_rel(
                    f"models/conformal_gap/{conformal_namespace}/conformal_results_mondrian.pkl"
                ),
                "intervals": _coerce_rel(
                    f"data/processed/conformal_gap/{conformal_namespace}/conformal_intervals_mondrian.parquet"
                ),
                "group_metrics": _coerce_rel(
                    f"data/processed/conformal_gap/{conformal_namespace}/conformal_group_metrics_mondrian.parquet"
                ),
                "backtest_monthly": _coerce_rel(
                    f"data/processed/conformal_gap/{conformal_namespace}/conformal_backtest_monthly.parquet"
                ),
                "backtest_alerts": _coerce_rel(
                    f"data/processed/conformal_gap/{conformal_namespace}/conformal_backtest_alerts.parquet"
                ),
                "policy_status": _coerce_rel(
                    f"models/conformal_gap/{conformal_namespace}/conformal_policy_status.json"
                ),
                "width_status": _coerce_rel(
                    f"models/conformal_gap/{conformal_namespace}/pd_conformal_width_attribution_status.json"
                ),
            },
            "search_space": {
                "partition_candidates": [tune_best.get("partition", "grade")],
                "partition": str(tune_best.get("partition", "grade")),
                "alpha_target_90": float(tune_best.get("alpha_target_90", 0.1)),
                "alpha_used_90": float(tune_best.get("alpha_used_90", 0.09)),
                "scaled_scores": bool(tune_best.get("scaled_scores", True)),
                "min_group_size": int(tune_best.get("min_group_size", 100)),
                "min_group_coverage_target": float(
                    tune_best.get("min_group_coverage_target", 0.89)
                ),
                "group_coverage_floor_target_90": float(
                    tune_best.get("group_coverage_floor_target_90", 0.92)
                ),
                "coverage_guardband_90": float(tune_best.get("coverage_guardband_90", 0.005)),
                "min_group_guardband_90": float(tune_best.get("min_group_guardband_90", 0.005)),
                "max_width_budget_90": float(tune_best.get("max_width_budget_90", 0.9)),
                "shrinkback_enabled": False,
            },
            "expectations": {
                "overall_pass": snapshot_conformal.get(
                    "overall_pass", conformal_status.get("overall_pass")
                ),
                "methodological_justification_pass": snapshot_conformal.get(
                    "methodological_justification_pass",
                    conformal_status.get("methodological_justification_pass"),
                ),
                "coverage_90": snapshot_conformal.get(
                    "coverage_90", conformal_status.get("coverage_90")
                ),
                "coverage_95": snapshot_conformal.get(
                    "coverage_95", conformal_status.get("coverage_95")
                ),
                "avg_width_90": snapshot_conformal.get(
                    "avg_width_90", conformal_status.get("avg_width_90")
                ),
                "min_group_coverage_90": snapshot_conformal.get(
                    "min_group_coverage_90", conformal_status.get("min_group_coverage_90")
                ),
                "warning_alerts": snapshot_conformal.get(
                    "warning_alerts", conformal_status.get("warning_alerts")
                ),
                "critical_alerts": snapshot_conformal.get(
                    "critical_alerts", conformal_status.get("critical_alerts")
                ),
            },
        },
        "portfolio": {
            "config_path": "configs/optimization.yaml",
            "selection_universe_path": str(
                snapshot_ab.get(
                    "candidate_universe_path",
                    champion_policy.get(
                        "selection_universe_path",
                        "data/processed/champion_candidate_universe.parquet",
                    ),
                )
            ),
            "selected_policy": selected_policy,
            "selected_candidate": dict(selector_status.get("selected_candidate", {})),
            "selector_name": str(selector_status.get("selector_name", "economic_actual_ab_v3")),
            "decision_scenario": str(
                snapshot_ab.get(
                    "decision_scenario", champion_policy.get("decision_scenario", "baseline")
                )
            ),
            "solver_backend": str(
                ab_status.get("solver_backend", snapshot_ab.get("solver_backend", "cuopt"))
            ),
            "candidate_caps": {
                "portfolio_max_candidates": 150000,
                "tradeoff_max_candidates": int(
                    ab_status.get("n_candidates_used", snapshot_ab.get("n_candidates_used", 80000))
                ),
                "ab_max_candidates": int(
                    ab_status.get(
                        "max_candidates_requested",
                        snapshot_ab.get("max_candidates_requested", 150000),
                    )
                ),
            },
            "tradeoff_grid_profile": "balanced",
            "ab_gate": {
                "max_portfolio_pd": float(snapshot_ab.get("max_portfolio_pd_requested", 0.18)),
                "n_boot": int(
                    (ab_status.get("diagnostics") or snapshot_ab.get("diagnostics") or {}).get(
                        "n_boot", 5000
                    )
                ),
                "seed": int(
                    (ab_status.get("diagnostics") or snapshot_ab.get("diagnostics") or {}).get(
                        "seed", 42
                    )
                ),
                "no_regression_tolerance_pct": float(
                    (ab_status.get("no_regression") or snapshot_ab.get("no_regression") or {}).get(
                        "tolerance_pct_of_control", 0.05
                    )
                ),
            },
            "expectations": {
                "selection_outcome": champion_policy.get("selection_outcome", "robust_selected"),
                "diff_total_return": (
                    (ab_status.get("no_regression") or snapshot_ab.get("no_regression") or {}).get(
                        "diff_total_return"
                    )
                    or (champion_policy.get("economic_metrics") or {}).get("diff_total_return")
                ),
                "passed_no_regression": (
                    (ab_status.get("no_regression") or snapshot_ab.get("no_regression") or {}).get(
                        "passed"
                    )
                    if (
                        ab_status.get("no_regression") or snapshot_ab.get("no_regression") or {}
                    ).get("passed")
                    is not None
                    else (champion_policy.get("economic_metrics") or {}).get("passed_no_regression")
                ),
            },
            "artifacts": {
                "champion_policy": artifact_descriptor("models/champion_portfolio_policy.json"),
                "selector_status": artifact_descriptor(
                    "models/champion_policy_selection_status.json"
                ),
                "candidate_universe": artifact_descriptor(
                    champion_policy.get(
                        "selection_universe_path",
                        "data/processed/champion_candidate_universe.parquet",
                    )
                ),
            },
        },
        "causal": {
            "role": "insights_only",
            "promotion_state": "insights_only",
        },
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build clean baseline replay manifest.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--conformal-namespace",
        default="conformal_v3_grade_noshrink_2026_03_26",
    )
    parser.add_argument(
        "--baseline-snapshot",
        default="",
        help="Optional baseline snapshot JSON to source promoted expectations from.",
    )
    parser.add_argument(
        "--output",
        default=str(resolve_manifest_path()),
    )
    parser.add_argument(
        "--pd-config-path",
        default="",
        help="Optional explicit PD config path to freeze in the manifest.",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest(
        source_run_tag=str(args.run_tag).strip(),
        conformal_namespace=str(args.conformal_namespace).strip(),
        baseline_snapshot_path=str(args.baseline_snapshot).strip() or None,
        pd_config_path_override=str(args.pd_config_path).strip() or None,
    )
    path = save_replay_manifest(manifest, args.output)
    print(f"[manifest] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
