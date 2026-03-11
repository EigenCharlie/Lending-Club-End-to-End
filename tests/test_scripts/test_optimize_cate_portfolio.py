"""Tests for scripts/optimize_cate_portfolio.py."""

from __future__ import annotations

import json

import pandas as pd

from scripts import optimize_cate_portfolio as opt_mod


def test_optimize_cate_portfolio_prefers_oot_alignment_and_flags_infeasible_runs(
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

    monkeypatch.setattr(
        opt_mod,
        "build_cate_adjusted_portfolio",
        lambda **kwargs: {
            "baseline": {"objective_value": 0.0, "n_funded": 0},
            "cate_adjusted": {"objective_value": 0.0, "n_funded": 0},
            "comparison_df": pd.DataFrame(
                [
                    {"scenario": "baseline", "objective_value": 0.0, "n_funded": 0},
                    {"scenario": "cate_adjusted", "objective_value": 0.0, "n_funded": 0},
                ]
            ),
        },
    )

    opt_mod.main(max_candidates=0)

    status = json.loads((model_dir / "cate_portfolio_status.json").read_text(encoding="utf-8"))
    assert status["run_tag"] == "run-causal-test"
    assert status["alignment_strategy"] == "id_join_oot_cate_plus_grade_fallback"
    assert status["feasible_baseline"] is False
    assert status["feasible_adjusted"] is False
    assert (
        status["constraint_binding_reason"]
        == "no_feasible_loans_under_current_budget_or_pd_constraints"
    )
    assert "Integracion causal no utilizable" in status["warning"]
    assert status["promotion_eligible"] is False
    assert status["fallback_applied"] is True
    assert status["cate_policy_mode"] == "research_only_fallback"
