from __future__ import annotations

from scripts import search_monotonic_economic_promotion as econ_mod


def test_build_cross_scenario_gate_passes_when_selective_improves_nonrobust_and_keeps_robust_within_tolerance() -> (
    None
):
    gate = econ_mod._build_cross_scenario_gate(
        baseline_status={
            "metrics_a": {"total_return": 200_000.0},
            "metrics_b": {"total_return": 190_000.0},
        },
        selective_status={
            "metrics_a": {"total_return": 215_000.0},
            "metrics_b": {"total_return": 183_000.0},
        },
        tolerance_pct=0.05,
    )

    assert gate["nonrobust_improved"] is True
    assert gate["robust_within_tolerance"] is True
    assert gate["passed"] is True


def test_choose_overall_winner_allows_selective_to_displace_baseline_when_it_has_better_diff() -> (
    None
):
    baseline = {
        "scenario": "baseline",
        "selector": "explicit_champion_only",
        "passed_no_regression": True,
        "diff_total_return": 5_000.0,
        "funded_ratio": 1.0,
        "selector_realized_total_return": 100_000.0,
        "selector_n_funded": 200.0,
        "cross_scenario_gate": {"passed": False},
        "status": {"decision_scenario": "baseline", "no_regression": {"passed": True}},
    }
    selective = {
        "scenario": "selective_ambiguity_defer",
        "selector": "guardrail_robustness",
        "passed_no_regression": True,
        "diff_total_return": 9_000.0,
        "funded_ratio": 1.02,
        "selector_realized_total_return": 101_000.0,
        "selector_n_funded": 210.0,
        "cross_scenario_gate": {"passed": True},
        "status": {
            "decision_scenario": "selective_ambiguity_defer",
            "no_regression": {"passed": True},
            "cross_scenario_gate": {"passed": True},
        },
    }

    winner, scenario, blocker = econ_mod._choose_overall_winner(
        scenario_best={
            "baseline": baseline,
            "selective_ambiguity_defer": selective,
        }
    )

    assert winner == selective
    assert scenario == "selective_ambiguity_defer"
    assert blocker is None
