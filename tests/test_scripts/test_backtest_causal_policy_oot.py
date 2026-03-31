"""Tests for scripts/backtest_causal_policy_oot.py."""

from __future__ import annotations

import json

import pandas as pd

from scripts import backtest_causal_policy_oot as backtest_mod


def test_backtest_causal_policy_oot_writes_status_with_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    train_rows = []
    sim_rows = []
    for month_idx, month in enumerate(months):
        for offset in range(2):
            loan_id = f"{month_idx}-{offset}"
            train_rows.append(
                {"id": loan_id, "issue_d": month, "grade": "A" if offset == 0 else "B"}
            )
            sim_rows.append(
                {
                    "id": loan_id,
                    "segment": "high_sensitivity" if offset == 0 else "medium_sensitivity",
                    "cate": 0.03 if offset == 0 else 0.015,
                    "recommended_action": "decrease_100bps" if offset == 0 else "decrease_50bps",
                    "policy_value_score": 100 + month_idx * 5 + offset,
                    "net_value": 100 + month_idx * 5 + offset,
                    "expected_loss_reduction": 150 + month_idx * 5,
                    "revenue_impact": -40,
                    "grade": "A" if offset == 0 else "B",
                }
            )

    pd.DataFrame(sim_rows).to_parquet(data_dir / "causal_policy_simulation.parquet", index=False)
    pd.DataFrame(train_rows).to_parquet(data_dir / "train_fe.parquet", index=False)
    (model_dir / "causal_policy_rule.json").write_text(
        json.dumps({"selected_rule": "discount_100_only", "run_tag": "run-causal-test"}),
        encoding="utf-8",
    )
    (model_dir / "causal_effect_status.json").write_text(
        json.dumps({"run_tag": "run-causal-test"}),
        encoding="utf-8",
    )

    backtest_mod.main(min_history_months=2)

    status = json.loads((model_dir / "causal_policy_oot_status.json").read_text(encoding="utf-8"))
    runtime_status = json.loads(
        (model_dir / "causal_policy_oot_runtime_status.json").read_text(encoding="utf-8")
    )
    backtest_df = pd.read_parquet(data_dir / "causal_policy_oot_backtest.parquet")

    assert status["run_tag"] == "run-causal-test"
    assert status["selected_rule_path"] == "models/causal_policy_rule.json"
    assert status["effect_status_path"] == "models/causal_effect_status.json"
    assert status["policy_value_method"] == "local_cate_discrete_grid"
    assert not backtest_df.empty
    assert runtime_status["state"] == "completed"
