"""Unit tests for robust policy resolution in simulate_ab_test."""

from __future__ import annotations

import pandas as pd

from scripts import simulate_ab_test as ab_mod


def test_resolve_robust_policy_from_summary_prefers_closest_lower_tolerance(tmp_path) -> None:
    summary_path = tmp_path / "portfolio_robustness_summary.parquet"
    pd.DataFrame(
        [
            {
                "risk_tolerance": 0.06,
                "best_robust_lambda": 1.0,
                "best_robust_min_budget_utilization": 0.05,
                "best_robust_pd_cap_slack_penalty": 1.5,
                "best_robust_return": 100.0,
            },
            {
                "risk_tolerance": 0.10,
                "best_robust_lambda": 0.5,
                "best_robust_min_budget_utilization": 0.01,
                "best_robust_pd_cap_slack_penalty": 0.5,
                "best_robust_return": 120.0,
            },
            {
                "risk_tolerance": 0.12,
                "best_robust_lambda": 0.25,
                "best_robust_min_budget_utilization": 0.0,
                "best_robust_pd_cap_slack_penalty": 0.0,
                "best_robust_return": 150.0,
            },
        ]
    ).to_parquet(summary_path, index=False)

    policy = ab_mod._resolve_robust_policy(
        max_portfolio_pd=0.11,
        summary_path=str(summary_path),
        champion_policy_path=str(tmp_path / "missing_champion.json"),
    )

    assert policy["source"] == "portfolio_robustness_summary"
    assert policy["risk_tolerance"] == 0.10
    assert policy["uncertainty_aversion"] == 0.5


def test_resolve_robust_policy_falls_back_when_summary_missing(tmp_path) -> None:
    policy = ab_mod._resolve_robust_policy(
        max_portfolio_pd=0.09,
        summary_path=str(tmp_path / "missing.parquet"),
        champion_policy_path=str(tmp_path / "missing_champion.json"),
    )
    assert policy["source"] == "fallback_default"
    assert policy["risk_tolerance"] == 0.09


def test_resolve_robust_policy_prefers_champion_artifact(tmp_path) -> None:
    champion_path = tmp_path / "champion_portfolio_policy.json"
    champion_path.write_text(
        (
            '{"selected_policy":{"risk_tolerance":0.08,"uncertainty_aversion":0.25,'
            '"min_budget_utilization":0.05,"pd_cap_slack_penalty":1.5,'
            '"policy_mode":"blended_uncertainty","gamma":0.5}}'
        ),
        encoding="utf-8",
    )

    policy = ab_mod._resolve_robust_policy(
        max_portfolio_pd=0.11,
        summary_path=str(tmp_path / "missing.parquet"),
        champion_policy_path=str(champion_path),
    )

    assert policy["source"] == "champion_policy_artifact"
    assert policy["risk_tolerance"] == 0.08
    assert policy["policy_mode"] == "blended_uncertainty"
    assert policy["gamma"] == 0.5
