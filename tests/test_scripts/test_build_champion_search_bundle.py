"""Tests for scripts/build_champion_search_bundle.py."""

from __future__ import annotations

import json
import pickle

from scripts import build_champion_search_bundle as bundle_mod


def test_build_champion_search_bundle_carries_threshold_semantics_and_baseline(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    with open(model_dir / "pd_training_record.pkl", "wb") as f:
        pickle.dump(
            {
                "training_regime": {"mode": "recent_12q"},
                "stable_core": {"enabled": True},
                "decision_threshold": {"selected_threshold": 0.05},
            },
            f,
        )
    with open(model_dir / "survival_summary.pkl", "wb") as f:
        pickle.dump({"rsf_c_index_test": 0.68}, f)

    (data_dir / "model_comparison.json").write_text(
        json.dumps({"best_model": "CatBoost", "best_calibration": "Venn-Abers"}),
        encoding="utf-8",
    )
    for name, payload in {
        "conformal_policy_status.json": {"run_tag": "run-live"},
        "conformal_method_registry.json": {"libraries": {"mapie": {"status": "adopted"}}},
        "conformal_variant_selection_status.json": {"promotion_pass": False},
        "fairness_audit_status.json": {"run_tag": "run-live", "overall_pass": True},
        "champion_portfolio_policy.json": {"run_tag": "run-live", "policy_mode": "robust"},
        "causal_effect_status.json": {"run_tag": "run-live"},
        "causal_policy_rule.json": {"run_tag": "run-live"},
        "causal_policy_oot_status.json": {"run_tag": "run-live"},
        "time_series_status.json": {"run_tag": "run-live", "status": "warn"},
        "cate_portfolio_status.json": {"run_tag": "run-live"},
        "governance_status.json": {"run_tag": "run-live", "overall_pass": True},
        "pd_set_prediction_status.json": {"summary": {"ambiguity_rate": 0.5}},
        "pd_rare_event_calibration_status.json": {"global": {"pr_auc": 0.4}},
        "pd_calibration_diagnostics.json": {"selected_method": "venn_abers"},
        "paper_grade_protocol_status.json": {"pd_conformal": {"closed_for_paper_grade": True}},
        "threshold_semantics.json": {
            "run_tag": "run-live",
            "pd_internal_selected_threshold": 0.05,
            "fairness_primary_threshold": 0.35,
            "decision_policy_global_threshold": 0.35,
        },
    }.items():
        (model_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(bundle_mod, "ROOT", tmp_path)
    monkeypatch.setattr(bundle_mod, "DATA", data_dir)
    monkeypatch.setattr(bundle_mod, "MODELS", model_dir)
    monkeypatch.setattr(bundle_mod, "resolve_official_baseline_run_tag", lambda: "run-baseline")

    bundle_mod.main()

    payload = json.loads((model_dir / "champion_search_bundle.json").read_text(encoding="utf-8"))
    assert payload["run_tag"] == "run-live"
    assert payload["upstream_canonical_run_tag"] == "run-baseline"
    assert payload["threshold_semantics"]["fairness_primary_threshold"] == 0.35
    assert payload["pd"]["decision_threshold"]["selected_threshold"] == 0.05
    assert payload["pd"]["decision_threshold_semantics"]["decision_policy_global_threshold"] == 0.35
