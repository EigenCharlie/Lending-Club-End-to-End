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

    assert policy["source"] == "champion_policy_artifact::promotion_first"
    assert policy["risk_tolerance"] == 0.08
    assert policy["policy_mode"] == "blended_uncertainty"
    assert policy["gamma"] == 0.5


def test_resolve_robust_policy_can_use_robustness_aware_artifact(tmp_path) -> None:
    champion_path = tmp_path / "champion_portfolio_policy.json"
    champion_path.write_text(
        (
            '{"selected_policy":{"risk_tolerance":0.08,"uncertainty_aversion":0.0,'
            '"min_budget_utilization":0.0,"pd_cap_slack_penalty":0.0,'
            '"policy_mode":"blended_uncertainty","gamma":0.0},'
            '"selected_policy_robustness_aware":{"risk_tolerance":0.10,'
            '"uncertainty_aversion":0.5,"min_budget_utilization":0.05,'
            '"pd_cap_slack_penalty":1.5,"policy_mode":"blended_uncertainty","gamma":0.5}}'
        ),
        encoding="utf-8",
    )

    policy = ab_mod._resolve_robust_policy(
        max_portfolio_pd=0.11,
        policy_selector="robustness_aware",
        summary_path=str(tmp_path / "missing.parquet"),
        champion_policy_path=str(champion_path),
    )

    assert policy["source"] == "champion_policy_artifact::robustness_aware"
    assert policy["risk_tolerance"] == 0.10
    assert policy["gamma"] == 0.5
    assert policy["uncertainty_aversion"] == 0.5


def test_resolve_robust_policy_can_use_guardrail_artifact(tmp_path) -> None:
    champion_path = tmp_path / "champion_portfolio_policy.json"
    champion_path.write_text(
        (
            '{"selected_policy":{"risk_tolerance":0.08,"uncertainty_aversion":0.0,'
            '"min_budget_utilization":0.0,"pd_cap_slack_penalty":0.0,'
            '"policy_mode":"blended_uncertainty","gamma":0.0},'
            '"selected_policy_guardrail_robustness":{"risk_tolerance":0.10,'
            '"uncertainty_aversion":0.0,"min_budget_utilization":0.0,'
            '"pd_cap_slack_penalty":0.0,"policy_mode":"blended_uncertainty","gamma":0.25}}'
        ),
        encoding="utf-8",
    )

    policy = ab_mod._resolve_robust_policy(
        max_portfolio_pd=0.11,
        policy_selector="guardrail_robustness",
        summary_path=str(tmp_path / "missing.parquet"),
        champion_policy_path=str(champion_path),
    )

    assert policy["source"] == "champion_policy_artifact::guardrail_robustness"
    assert policy["gamma"] == 0.25


def test_resolve_robust_policy_explicit_champion_only_requires_artifact(tmp_path) -> None:
    try:
        ab_mod._resolve_robust_policy(
            max_portfolio_pd=0.1,
            policy_selector="explicit_champion_only",
            summary_path=str(tmp_path / "missing.parquet"),
            champion_policy_path=str(tmp_path / "missing_champion.json"),
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for explicit_champion_only")


def test_apply_decision_scenario_baseline_is_noop() -> None:
    test_df = pd.DataFrame({"id": ["a", "b"], "default_flag": [0, 1]})
    intervals = pd.DataFrame({"id": ["a", "b"], "pd_high": [0.2, 0.7]})

    test_out, ints_out, meta = ab_mod._apply_decision_scenario(
        test_df,
        intervals,
        decision_scenario="baseline",
    )

    assert len(test_out) == 2
    assert len(ints_out) == 2
    assert meta["decision_scenario"] == "baseline"
    assert meta["rows_removed"] == 0


def test_apply_decision_scenario_ambiguity_defer_filters_ambiguous_ids(tmp_path) -> None:
    set_path = tmp_path / "pd_set_prediction_cases.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "ambiguous": [0, 1, 0],
        }
    ).to_parquet(set_path, index=False)

    test_df = pd.DataFrame({"id": ["a", "b", "c"], "default_flag": [0, 1, 0]})
    intervals = pd.DataFrame({"id": ["a", "b", "c"], "pd_high": [0.2, 0.7, 0.3]})

    test_out, ints_out, meta = ab_mod._apply_decision_scenario(
        test_df,
        intervals,
        decision_scenario="ambiguity_defer",
        set_prediction_path=str(set_path),
    )

    assert test_out["id"].tolist() == ["a", "c"]
    assert ints_out["id"].tolist() == ["a", "c"]
    assert meta["rows_removed"] == 1
    assert meta["rows_remaining"] == 2
