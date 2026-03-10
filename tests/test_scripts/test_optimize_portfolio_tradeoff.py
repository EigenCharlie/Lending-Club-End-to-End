from __future__ import annotations

import pandas as pd

from scripts.optimize_portfolio_tradeoff import _select_champion_policy


def test_select_champion_policy_exposes_dual_selectors() -> None:
    frontier = pd.DataFrame(
        [
            {
                "policy": "nonrobust",
                "risk_tolerance": 0.10,
                "policy_mode": "point_estimate",
                "gamma": 0.0,
                "uncertainty_aversion": 0.0,
                "realized_total_return": 100.0,
                "price_of_robustness": 0.0,
                "ab_pass": True,
                "n_funded": 10,
                "solver_backend": "cuopt",
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
                "pd_cap_slack": 0.0,
                "ab_diff_total_return": 0.0,
            },
            {
                "policy": "robust",
                "risk_tolerance": 0.10,
                "policy_mode": "blended_uncertainty",
                "gamma": 0.0,
                "uncertainty_aversion": 0.0,
                "realized_total_return": 120.0,
                "price_of_robustness": -10.0,
                "ab_pass": True,
                "n_funded": 11,
                "solver_backend": "cuopt",
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
                "pd_cap_slack": 0.0,
                "ab_diff_total_return": 20.0,
            },
            {
                "policy": "robust",
                "risk_tolerance": 0.10,
                "policy_mode": "blended_uncertainty",
                "gamma": 0.5,
                "uncertainty_aversion": 0.5,
                "realized_total_return": 95.0,
                "price_of_robustness": 5.0,
                "ab_pass": True,
                "n_funded": 9,
                "solver_backend": "cuopt",
                "min_budget_utilization": 0.05,
                "pd_cap_slack_penalty": 1.5,
                "pd_cap_slack": 0.0,
                "ab_diff_total_return": -5.0,
            },
        ]
    )

    frontier_out, selected, robust_selected, balanced_selected, guardrail_selected = (
        _select_champion_policy(frontier)
    )

    assert selected["gamma"] == 0.0
    assert robust_selected is not None
    assert robust_selected["gamma"] == 0.5
    assert balanced_selected is not None
    assert balanced_selected["gamma"] == 0.5
    assert guardrail_selected is not None
    assert guardrail_selected["gamma"] == 0.5
    assert frontier_out["selected_for_champion"].sum() == 1
    assert frontier_out["selected_for_robustness_aware"].sum() == 1
    assert frontier_out["selected_for_balanced_robustness"].sum() == 1
    assert frontier_out["selected_for_guardrail_robustness"].sum() == 1
