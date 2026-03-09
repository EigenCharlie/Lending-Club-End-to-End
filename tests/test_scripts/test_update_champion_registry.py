"""Tests for scripts/update_champion_registry.py."""

from __future__ import annotations

import json
import pickle

from scripts import update_champion_registry as reg_mod


def test_update_champion_registry_aggregates_current_champion_state(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    with open(model_dir / "pd_training_record.pkl", "wb") as f:
        pickle.dump(
            {
                "training_regime": {"mode": "recent_12q"},
                "stable_core": {"enabled": True, "excluded_features": ["rev_utilization"]},
                "decision_threshold": {"selected_threshold": 0.4},
            },
            f,
        )

    (model_dir / "fairness_audit_status.json").write_text(
        json.dumps(
            {
                "run_tag": "run-champion",
                "overall_pass": True,
                "prediction_threshold": 0.4,
                "prediction_threshold_source": "decision_policy_artifact",
                "decision_policy": {"global_threshold": 0.4, "n_overrides": 1},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "governance_status.json").write_text(
        json.dumps(
            {
                "run_tag": "run-champion",
                "overall_pass": True,
                "summary": {"max_psi": 0.09, "c2st_auc": 0.58},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "champion_portfolio_policy.json").write_text(
        json.dumps(
            {
                "run_tag": "run-champion",
                "policy_mode": "blended_uncertainty",
                "gamma": 0.5,
                "ab_pass": True,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "cate_portfolio_status.json").write_text(
        json.dumps(
            {
                "promotion_eligible": False,
                "cate_policy_mode": "research_only_fallback",
                "fallback_applied": True,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "time_series_status.json").write_text(
        json.dumps({"run_tag": "run-champion", "status": "ok"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(reg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(reg_mod, "MODELS", model_dir)

    reg_mod.main()

    payload = json.loads((model_dir / "champion_registry.json").read_text(encoding="utf-8"))
    assert payload["run_tag"] == "run-champion"
    assert payload["pd"]["training_regime"]["mode"] == "recent_12q"
    assert payload["portfolio"]["policy_mode"] == "blended_uncertainty"
    assert payload["fairness"]["decision_policy"]["global_threshold"] == 0.4
    assert payload["cate"]["cate_policy_mode"] == "research_only_fallback"
    assert payload["governance"]["overall_pass"] is True
