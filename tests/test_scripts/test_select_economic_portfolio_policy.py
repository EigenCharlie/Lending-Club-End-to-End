from __future__ import annotations

import json

import pandas as pd

from scripts import select_economic_portfolio_policy as sel_mod


def _write_common_inputs(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "int_rate": [10.0, 12.0, 9.0],
            "loan_amnt": [1000.0, 1000.0, 1000.0],
            "default_flag": [0, 0, 0],
        }
    ).to_parquet(data_dir / "test_fe.parquet", index=False)
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "pd_calibrated": [0.05, 0.08, 0.07],
            "pd_low": [0.04, 0.07, 0.06],
            "pd_high": [0.09, 0.11, 0.10],
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)
    pd.DataFrame({"id": ["a", "b", "c"], "sample_order": [0, 1, 2]}).to_parquet(
        data_dir / "champion_candidate_universe.parquet", index=False
    )
    frontier = pd.DataFrame(
        [
            {
                "policy": "robust",
                "policy_mode": "blended_uncertainty",
                "gamma": 0.0,
                "delta_cap_quantile": 1.0,
                "risk_tolerance": 0.1,
                "uncertainty_aversion": 0.0,
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
                "realized_total_return": 100.0,
                "price_of_robustness_pct": 0.0,
                "selected_for_champion": True,
                "selected_for_balanced_robustness": False,
                "selected_for_guardrail_robustness": False,
                "eligible_for_canonical_selection": True,
            },
            {
                "policy": "robust",
                "policy_mode": "capped_blended_uncertainty",
                "gamma": 0.1,
                "delta_cap_quantile": 0.75,
                "risk_tolerance": 0.1,
                "uncertainty_aversion": 0.1,
                "min_budget_utilization": 0.0,
                "pd_cap_slack_penalty": 0.0,
                "realized_total_return": 95.0,
                "price_of_robustness_pct": -5.0,
                "selected_for_champion": False,
                "selected_for_balanced_robustness": True,
                "selected_for_guardrail_robustness": True,
                "eligible_for_canonical_selection": True,
            },
        ]
    )
    frontier.to_parquet(data_dir / "portfolio_robustness_frontier.parquet", index=False)
    (model_dir / "portfolio_research_policy.json").write_text(
        json.dumps({"selected_policy": {"gamma": 0.0}}),
        encoding="utf-8",
    )
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "optimization.yaml").write_text(
        """
portfolio:
  total_budget: 1000
portfolio_selection:
  canonical_selector: economic_actual_ab_v2
  actual_ab_top_k: 20
  min_funded_ratio: 0.88
  min_total_allocated_ratio: 0.98
  min_breadth_score: 0.995
  breadth_weight_funded_ratio: 0.50
  breadth_weight_allocation_ratio: 0.30
  breadth_weight_allocation_similarity: 0.20
  max_price_of_robustness_pct: -15.0
  canonical_policy_modes: [blended_uncertainty, capped_blended_uncertainty]
""".strip(),
        encoding="utf-8",
    )


def _set_run_tag(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_RUN_TAG", "run-policy-test")


def test_selector_promotes_robust_candidate_when_one_passes(tmp_path, monkeypatch) -> None:
    _write_common_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_run_tag(monkeypatch)

    def fake_run_strategy(*, robust, robust_policy=None, **kwargs):
        _ = kwargs
        if not robust:
            return {
                "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 1000.0,
            }, None
        gamma = float((robust_policy or {}).get("gamma", 0.0))
        if gamma > 0:
            return {
                "allocation": {0: 0.98, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 1000.0,
            }, None
        return {
            "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
            "n_funded": 1,
            "total_allocated": 1000.0,
        }, None

    def fake_candidate_metrics(*, solution, **kwargs):
        _ = kwargs
        gamma_like = 0.98 if solution["allocation"][0] < 1.0 else 1.0
        total_return = 100.0 if gamma_like == 1.0 else 99.5
        return None, {
            "total_return": total_return,
            "n_funded": solution["n_funded"],
            "total_allocated": solution["total_allocated"],
            "avg_return_per_funded": total_return,
        }

    monkeypatch.setattr(sel_mod, "_run_strategy", fake_run_strategy)
    monkeypatch.setattr(sel_mod, "_candidate_metrics", fake_candidate_metrics)

    sel_mod.main(config_path="configs/optimization.yaml")

    payload = json.loads((tmp_path / "models" / "champion_portfolio_policy.json").read_text())
    status = json.loads((tmp_path / "models" / "champion_policy_selection_status.json").read_text())
    assert payload["selection_outcome"] == "robust_selected"
    assert payload["selection_stage"] == "economic_actual_ab_v2"
    assert payload["selected_policy"]["gamma"] == 0.1
    assert status["fallback_applied"] is False
    assert status["selected_candidate"]["breadth_score"] >= 0.93


def test_selector_falls_back_when_no_robust_candidate_passes(tmp_path, monkeypatch) -> None:
    _write_common_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_run_tag(monkeypatch)

    def fake_run_strategy(*, robust, robust_policy=None, **kwargs):
        _ = kwargs
        if not robust:
            return {
                "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 1000.0,
            }, None
        gamma = float((robust_policy or {}).get("gamma", 0.0))
        return {
            "allocation": {0: max(0.7, 1.0 - gamma), 1: 0.0, 2: 0.0},
            "n_funded": 1,
            "total_allocated": 1000.0,
        }, None

    def fake_candidate_metrics(*, solution, **kwargs):
        _ = kwargs
        total_return = 100.0 if solution["allocation"][0] == 1.0 else 80.0
        return None, {
            "total_return": total_return,
            "n_funded": solution["n_funded"],
            "total_allocated": solution["total_allocated"],
            "avg_return_per_funded": total_return,
        }

    monkeypatch.setattr(sel_mod, "_run_strategy", fake_run_strategy)
    monkeypatch.setattr(sel_mod, "_candidate_metrics", fake_candidate_metrics)

    sel_mod.main(config_path="configs/optimization.yaml")

    payload = json.loads((tmp_path / "models" / "champion_portfolio_policy.json").read_text())
    status = json.loads((tmp_path / "models" / "champion_policy_selection_status.json").read_text())
    assert payload["selection_outcome"] == "fallback_nonrobust"
    assert payload["selected_policy"]["gamma"] == 0.0
    assert status["fallback_applied"] is True


def test_selector_v2_prefers_breadth_aware_candidate(tmp_path, monkeypatch) -> None:
    _write_common_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_run_tag(monkeypatch)

    def fake_run_strategy(*, robust, robust_policy=None, **kwargs):
        _ = kwargs
        if not robust:
            return {
                "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 1000.0,
            }, None
        gamma = float((robust_policy or {}).get("gamma", 0.0))
        if gamma > 0:
            return {
                "allocation": {0: 0.99, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 990.0,
            }, None
        return {
            "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
            "n_funded": 1,
            "total_allocated": 1000.0,
        }, None

    def fake_candidate_metrics(*, solution, **kwargs):
        _ = kwargs
        total_allocated = float(solution["total_allocated"])
        total_return = 100.0 if total_allocated >= 999.0 else 108.0
        return None, {
            "total_return": total_return,
            "n_funded": solution["n_funded"],
            "total_allocated": total_allocated,
            "avg_return_per_funded": total_return,
        }

    monkeypatch.setattr(sel_mod, "_run_strategy", fake_run_strategy)
    monkeypatch.setattr(sel_mod, "_candidate_metrics", fake_candidate_metrics)

    sel_mod.main(config_path="configs/optimization.yaml")

    payload = json.loads((tmp_path / "models" / "champion_portfolio_policy.json").read_text())
    status = json.loads((tmp_path / "models" / "champion_policy_selection_status.json").read_text())
    assert payload["selection_outcome"] == "robust_selected"
    assert payload["selected_policy"]["gamma"] == 0.1
    assert status["selected_candidate"]["total_allocated_ratio"] == 0.99
    assert status["selected_candidate"]["breadth_score"] >= 0.93


def test_selector_v3_uses_ab_like_ranking(tmp_path, monkeypatch) -> None:
    _write_common_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_run_tag(monkeypatch)
    (tmp_path / "configs" / "optimization.yaml").write_text(
        """
portfolio:
  total_budget: 1000
portfolio_selection:
  canonical_selector: economic_actual_ab_v3
  actual_ab_top_k: 20
  ab_like_top_m: 2
  ab_like_bootstrap_n: 10
  ab_like_seed: 7
  min_funded_ratio: 0.88
  min_total_allocated_ratio: 0.98
  min_breadth_score: 0.93
  breadth_weight_funded_ratio: 0.50
  breadth_weight_allocation_ratio: 0.30
  breadth_weight_allocation_similarity: 0.20
  max_price_of_robustness_pct: -15.0
  canonical_policy_modes: [blended_uncertainty, capped_blended_uncertainty]
""".strip(),
        encoding="utf-8",
    )

    def fake_run_strategy(*, robust, robust_policy=None, **kwargs):
        _ = kwargs
        if not robust:
            return {
                "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 1000.0,
            }, None
        gamma = float((robust_policy or {}).get("gamma", 0.0))
        alloc0 = 0.99 if gamma > 0.09 else 0.98
        return {
            "allocation": {0: alloc0, 1: 0.0, 2: 0.0},
            "n_funded": 1,
            "total_allocated": 1000.0,
        }, None

    def fake_candidate_metrics(*, solution, **kwargs):
        _ = kwargs
        alloc0 = solution["allocation"][0]
        if abs(alloc0 - 0.99) < 1e-9:
            returns = pd.Series([100.0, 0.0, 0.0]).to_numpy()
            total_return = 100.0
        else:
            returns = pd.Series([120.0, -40.0, 0.0]).to_numpy()
            total_return = 80.0
        return returns, {
            "total_return": total_return,
            "n_funded": solution["n_funded"],
            "total_allocated": solution["total_allocated"],
            "avg_return_per_funded": total_return,
        }

    monkeypatch.setattr(sel_mod, "_run_strategy", fake_run_strategy)
    monkeypatch.setattr(sel_mod, "_candidate_metrics", fake_candidate_metrics)

    sel_mod.main(config_path="configs/optimization.yaml")

    payload = json.loads((tmp_path / "models" / "champion_portfolio_policy.json").read_text())
    status = json.loads((tmp_path / "models" / "champion_policy_selection_status.json").read_text())
    assert payload["selection_stage"] == "economic_actual_ab_v3"
    assert payload["selected_policy"]["gamma"] == 0.1
    assert status["selected_candidate"]["ab_like_passed_no_regression"] is True


def test_selector_v3_respects_breadth_hard_filters(tmp_path, monkeypatch) -> None:
    _write_common_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_run_tag(monkeypatch)
    (tmp_path / "configs" / "optimization.yaml").write_text(
        """
portfolio:
  total_budget: 1000
portfolio_selection:
  canonical_selector: economic_actual_ab_v3
  actual_ab_top_k: 20
  ab_like_top_m: 2
  ab_like_bootstrap_n: 10
  ab_like_seed: 7
  min_funded_ratio: 0.88
  min_total_allocated_ratio: 0.98
  min_breadth_score: 0.93
  breadth_weight_funded_ratio: 0.50
  breadth_weight_allocation_ratio: 0.30
  breadth_weight_allocation_similarity: 0.20
  max_price_of_robustness_pct: -15.0
  canonical_policy_modes: [blended_uncertainty, capped_blended_uncertainty]
""".strip(),
        encoding="utf-8",
    )

    def fake_run_strategy(*, robust, robust_policy=None, **kwargs):
        _ = kwargs
        if not robust:
            return {
                "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 1000.0,
            }, None
        gamma = float((robust_policy or {}).get("gamma", 0.0))
        if gamma > 0:
            return {
                "allocation": {0: 0.8, 1: 0.0, 2: 0.0},
                "n_funded": 1,
                "total_allocated": 900.0,
            }, None
        return {
            "allocation": {0: 1.0, 1: 0.0, 2: 0.0},
            "n_funded": 1,
            "total_allocated": 1000.0,
        }, None

    def fake_candidate_metrics(*, solution, **kwargs):
        _ = kwargs
        alloc0 = solution["allocation"][0]
        total_return = 120.0 if alloc0 < 1.0 else 100.0
        return pd.Series([total_return, 0.0, 0.0]).to_numpy(), {
            "total_return": total_return,
            "n_funded": solution["n_funded"],
            "total_allocated": solution["total_allocated"],
            "avg_return_per_funded": total_return,
        }

    monkeypatch.setattr(sel_mod, "_run_strategy", fake_run_strategy)
    monkeypatch.setattr(sel_mod, "_candidate_metrics", fake_candidate_metrics)

    sel_mod.main(config_path="configs/optimization.yaml")

    payload = json.loads((tmp_path / "models" / "champion_portfolio_policy.json").read_text())
    status = json.loads((tmp_path / "models" / "champion_policy_selection_status.json").read_text())
    assert payload["selection_outcome"] == "fallback_nonrobust"
    assert payload["selected_policy"]["gamma"] == 0.0
    assert status["fallback_applied"] is True
