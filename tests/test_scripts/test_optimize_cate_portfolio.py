"""Tests for scripts/optimize_cate_portfolio.py."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pandas as pd

from scripts import optimize_cate_portfolio as opt_mod


def test_optimize_cate_portfolio_blocks_when_policy_gate_is_not_validated(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "grade": ["A", "B", "C"],
            "loan_amnt": [5000, 6000, 7000],
            "int_rate": [10.0, 11.0, 12.0],
            "purpose": ["debt_consolidation"] * 3,
        }
    ).to_parquet(data_dir / "test_fe.parquet", index=False)
    pd.DataFrame(
        {
            "y_pred": [0.1, 0.2, 0.3],
            "pd_low_90": [0.08, 0.18, 0.28],
            "pd_high_90": [0.12, 0.22, 0.32],
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)
    pd.DataFrame({"id": ["1", "2"], "cate": [0.01, 0.02]}).to_parquet(
        data_dir / "cate_estimates_oot.parquet", index=False
    )
    pd.DataFrame({"grade": ["C"], "cate": [0.03]}).to_parquet(
        data_dir / "cate_estimates.parquet", index=False
    )
    (model_dir / "causal_effect_status.json").write_text(
        json.dumps({"run_tag": "run-causal-test"}),
        encoding="utf-8",
    )
    (model_dir / "causal_policy_rule.json").write_text(
        json.dumps({"run_tag": "run-causal-test", "policy_evaluation_consistent": False}),
        encoding="utf-8",
    )

    build_mock = Mock()
    monkeypatch.setattr(opt_mod, "build_cate_adjusted_portfolio", build_mock)

    opt_mod.main(max_candidates=0)

    status = json.loads((model_dir / "cate_portfolio_status.json").read_text(encoding="utf-8"))
    assert status["run_tag"] == "run-causal-test"
    assert status["cate_policy_mode"] == "research_blocked_by_policy_gate"
    assert status["promotion_state"] == "research_blocked_by_policy_gate"
    assert status["feasible_baseline"] is False
    assert status["feasible_adjusted"] is False
    assert status["constraint_binding_reason"] == "research_blocked_by_policy_gate"
    assert "blocked" in status["warning"].lower()
    assert status["promotion_eligible"] is False
    assert status["fallback_applied"] is True
    build_mock.assert_not_called()
