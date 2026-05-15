"""Guardrails for the Paper 4 living-lab evidence pack."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

PAPER4_ROOT = Path("reports/paper_material/paper4")
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
BOOK_DIR = Path("book")

EXPECTED_PAPER1_CHAMPION_LABEL = "bound_aware_276k_economic_champion"
EXPECTED_PAPER1_CHAMPION_RETURN = 170464.5429284627
CURATED_PAPER4_PAGES = {
    "index.qmd",
    "19a-proposal-and-scope.qmd",
    "19b-current-assets-and-gaps.qmd",
    "19c-integrated-architecture.qmd",
    "19f-sequential-decision-framework.qmd",
    "19h-mvp-evidence-pack.qmd",
    "19i-regret-auditability-frontier.qmd",
    "19n-online-mdcp-fairness.qmd",
    "19t-multi-period-solver.qmd",
    "19ca-v38-final-synthesis.qmd",
}


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def _read_json(name: str) -> dict:
    return json.loads((STATUS_DIR / name).read_text(encoding="utf-8"))


def _registered_paper4_pages() -> list[str]:
    pages: list[str] = []
    for raw_line in (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("- chapters/19-paper-mega-extension/"):
            continue
        pages.append(Path(stripped.removeprefix("- ")).name)
    return pages


def test_paper4_source_manifest_exists() -> None:
    manifest_path = TABLE_DIR / "paper4_table0_source_manifest.csv"
    assert manifest_path.exists()

    manifest = pd.read_csv(manifest_path)
    expected_columns = {
        "artifact",
        "source_paper",
        "role",
        "status",
        "run_tag",
        "caveat",
        "path_exists",
    }
    assert expected_columns.issubset(manifest.columns)
    assert len(manifest) >= 8
    assert manifest["path_exists"].astype(bool).all()


def test_paper4_sdam_schema_exists() -> None:
    schema = _read_json("paper4_sequential_decision_schema.json")
    assert schema["schema_version"]

    elements = set(schema["elements"])
    assert {"R_t", "I_t", "B_t", "x_t", "W_t_plus_1", "S_M", "C_t", "X_pi"}.issubset(elements)

    post_state = _read_json("paper4_post_decision_state_schema.json")
    fields = {entry["name"] for entry in post_state["fields"]}
    assert {"funded_exposure", "capital_used", "budget_remaining", "stage_mix_proxy"}.issubset(
        fields
    )


def test_paper4_policy_class_registry_schema() -> None:
    registry = _read_csv("paper4_policy_class_registry.csv")
    expected_columns = {
        "policy_id",
        "source",
        "policy_class",
        "objective_type",
        "evaluation_mode",
        "decision_scope",
        "status",
        "is_paper1_champion",
        "source_artifact",
        "notes",
    }
    assert expected_columns.issubset(registry.columns)
    assert registry["policy_id"].is_unique
    assert "paper1_economic_champion" in set(registry["policy_id"])
    assert {"CFA", "PFA", "DLA", "VFA/CFA"}.intersection(set(registry["policy_class"]))
    assert {"implemented_frozen", "planned", "blocked_by_identification"}.issubset(
        set(registry["status"])
    )


def test_paper4_artifacts_do_not_override_paper1_champion() -> None:
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    paper4_status = _read_json("paper4_mvp_status.json")
    assert paper4_status["paper1_champion_protected"] is True
    assert paper4_status["paper4_final_promotion_created"] is False

    paper1_promotion = json.loads(
        Path("models/final_project_promotion.json").read_text(encoding="utf-8")
    )
    champion = paper1_promotion["final_champion"]
    assert champion["label"] == EXPECTED_PAPER1_CHAMPION_LABEL
    assert champion["realized_total_return"] == pytest.approx(EXPECTED_PAPER1_CHAMPION_RETURN)


def test_paper4_mvp_tables_have_policy_id() -> None:
    for name in (
        "paper4_table1_policy_ecl_scenario.csv",
        "paper4_table2_net_return_after_ecl.csv",
        "paper4_table3_tail_risk_oce_cvar_by_policy.csv",
        "paper4_table4_satisficing_screen.csv",
        "paper4_table10_policy_pairwise_differences.csv",
    ):
        table = _read_csv(name)
        assert "policy_id" in table.columns, name
        assert not table.empty, name
        assert table["policy_id"].notna().all(), name

    universe = _read_csv("paper4_policy_universe.csv")
    assert len(universe) == 45
    assert universe["policy_id"].is_unique


def test_paper4_evidence_parquets_exist() -> None:
    expectations = {
        "paper4_loan_level_policy_evidence.parquet": {
            "loan_id",
            "period",
            "policy_id",
            "ecl_proxy_lgd45",
        },
        "paper4_policy_level_evidence.parquet": {"policy_id", "realized_total_return", "gamma_cp"},
        "paper4_monthly_replay.parquet": {"month", "period", "policy_id", "coverage_90"},
        "online_conformal_coverage_regret.parquet": {"month", "coverage_regret_90_cum"},
    }

    for name, columns in expectations.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_extended_policy_loan_matrix_exists() -> None:
    matrix = pd.read_parquet(TABLE_DIR / "paper4_policy_loan_matrix.parquet")
    evidence = pd.read_parquet(TABLE_DIR / "paper4_policy_loan_level_evidence.parquet")
    status = _read_json("paper4_extended_lanes_status.json")

    assert not matrix.empty
    assert not evidence.empty
    assert {"policy_id", "loan_id", "funded_flag", "reconstruction_method"}.issubset(matrix.columns)
    assert evidence["policy_id"].nunique() == 45
    assert "policy-implied greedy proxies" in status["reconstruction_caveat"]
    assert evidence.loc[
        evidence["policy_id"].eq("paper1_economic_champion"), "is_champion_exact"
    ].all()


def test_paper4_ifrs9_full_and_monthly_replay_exist() -> None:
    ifrs9 = _read_csv("paper4_table12_ifrs9_policy_full_eval.csv")
    monthly = pd.read_parquet(TABLE_DIR / "paper4_monthly_policy_replay.parquet")

    assert {"policy_id", "scenario", "ecl", "provision", "net_return_after_ecl_full"}.issubset(
        ifrs9.columns
    )
    assert set(ifrs9["scenario"]) == {"baseline", "adverse", "severe"}
    assert ifrs9["policy_id"].nunique() == 45
    assert monthly["policy_id"].nunique() == 45
    assert {"month", "policy_id", "coverage_alpha01", "net_return_after_ecl"}.issubset(
        monthly.columns
    )


def test_paper4_diagnostic_selector_does_not_promote() -> None:
    selector = _read_csv("paper4_table14_ifrs9_tail_satisficing_selector.csv")
    config = _read_json("paper4_diagnostic_selector_config.json")

    assert not selector.empty
    assert "diagnostic_selector_rank" in selector.columns
    assert selector["diagnostic_selector_rank"].min() == 1
    assert config["mode"] == "diagnostic_no_promotion"
    assert config["promotion_json_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_p1_p2_lanes_exist() -> None:
    expected_csvs = {
        "paper4_policy_pairwise_bootstrap_ci.csv": {"policy_id", "p05_net_return_diff"},
        "paper4_robust_satisficing_policy_eval.csv": {
            "policy_id",
            "robust_satisficing_decision",
        },
        "paper4_mdcp_worst_source_coverage.csv": {
            "policy_id",
            "source_family",
            "worst_source_coverage_alpha01",
        },
        "paper4_fairness_constraint_screen.csv": {"policy_id", "fairness_proxy_pass"},
        "paper4_dla_toy_horizon_summary.csv": {"policy_id", "horizon_months"},
        "paper4_cate_policy_value_toy.csv": {"policy_id", "toy_loss_reduction_value"},
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_online_conformal_aci_replay.parquet": {"month", "recommended_alpha_next"},
        "paper4_online_conformal_grade_replay.parquet": {
            "grade",
            "recommended_alpha_next",
        },
        "paper4_dla_toy_state_replay.parquet": {"policy_id", "budget_start", "budget_end"},
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v2_priority_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_exact_limited_topk_policy_eval.csv": {
            "policy_id",
            "solver_status",
            "n_funded_exact_limited",
        },
        "paper4_exact_vs_proxy_comparison.csv": {
            "policy_id",
            "jaccard_exact_limited_vs_proxy",
            "status",
        },
        "paper4_ifrs9_v2_policy_summary.csv": {
            "policy_id",
            "scenario",
            "sicr_rule",
            "net_return_after_ecl_v2",
        },
        "paper4_selector_v2_results.csv": {
            "policy_id",
            "selector_v2_rank",
            "selector_v2_decision",
        },
        "paper4_online_conformal_coverage_regret_v2.csv": {
            "policy_id",
            "month",
            "coverage_regret_90_cum",
        },
        "paper4_regret_auditability_pareto_v2.csv": {
            "policy_id",
            "mean_regret_net_ecl",
            "auditability_score_v2",
        },
        "paper4_oce_cvar_constraint_grid.csv": {
            "cvar_cap",
            "oce_penalty",
            "selected_policy_id",
        },
        "paper4_multi_period_policy_path.csv": {
            "month",
            "chosen_policy_id",
            "budget_end",
        },
        "paper4_mdcp_worst_source_frontier.csv": {
            "policy_id",
            "source_id",
            "worst_source_coverage_online_90",
        },
        "paper4_cate_exact_topk_value.csv": {
            "policy_id",
            "cate_loss_reduction_value",
        },
        "paper4_claim_artifact_test_matrix.csv": {
            "priority",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_v2_policy_loan_evidence.parquet": {"loan_id", "policy_id", "exact_scope"},
        "paper4_ifrs9_lifetime_ecl_grid.parquet": {
            "loan_id",
            "policy_id",
            "ifrs9_stage_v2",
            "ecl_v2",
        },
        "paper4_online_conformal_intervals.parquet": {
            "loan_id",
            "covered_online_90",
            "online_source",
        },
        "paper4_regret_auditability_replay_v2.parquet": {
            "policy_id",
            "decision_regret_net_ecl",
            "auditability_score_v2",
        },
        "paper4_multi_period_solver_results.parquet": {
            "chosen_policy_id",
            "budget_start",
            "budget_end",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v2_does_not_promote_and_declares_exact_scope() -> None:
    status = _read_json("paper4_v2_priorities_status.json")
    exact = _read_json("paper4_exact_replay_topk_status.json")

    assert status["phase"] == "v2_priorities_1_to_10"
    assert status["mode"] == "diagnostic_no_promotion"
    assert status["paper1_champion_protected"] is True
    assert status["paper4_final_promotion_created"] is False
    assert status["priorities_completed"] == 10
    assert exact["promotion_json_created"] is False
    assert "not a full 276k-loan exact rerun" in exact["caveat"]
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    comparison = _read_csv("paper4_exact_vs_proxy_comparison.csv")
    assert "paper1_economic_champion" in set(comparison["policy_id"])
    champion = comparison.loc[comparison["policy_id"].eq("paper1_economic_champion")]
    assert champion["jaccard_exact_limited_vs_proxy"].iloc[0] == pytest.approx(1.0)
    assert (comparison["jaccard_exact_limited_vs_proxy"].between(0, 1)).all()


def test_paper4_v2_quarto_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_claim_artifact_test_matrix.csv")

    expected_pages = {
        "19p-exact-replay-topk.qmd",
        "19q-ifrs9-v2-lifetime-ecl.qmd",
        "19r-online-conformal-real.qmd",
        "19s-regret-auditability-replay.qmd",
        "19t-multi-period-solver.qmd",
        "19u-artifact-catalog-and-claims.qmd",
        "19v-tail-mdcp-causal-v2.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_next_wave_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_full_universe_topk_policy_eval.csv": {
            "policy_id",
            "solver_status",
            "full_n_funded",
        },
        "paper4_full_universe_vs_exact_limited_comparison.csv": {
            "policy_id",
            "jaccard_full_vs_exact_limited",
        },
        "paper4_full_universe_ifrs9_tail_eval.csv": {
            "policy_id",
            "scenario",
            "net_return_after_ecl_next_wave",
        },
        "paper4_online_conformal_method_search.csv": {
            "online_method",
            "coverage_online_mean",
            "coverage_online_min_month_policy",
        },
        "paper4_selector_gate_sensitivity_summary.csv": {
            "mdcp_threshold",
            "fairness_gap_threshold",
            "max_gate_pass",
        },
        "paper4_tail_penalty_lp_search.csv": {
            "tail_lp_policy_id",
            "tail_penalty",
            "solver_status",
        },
        "paper4_mdcp_source_family_search.csv": {
            "policy_id",
            "source_id",
            "worst_source_coverage",
        },
        "paper4_fairness_proxy_governance_grid.csv": {
            "policy_id",
            "source_id",
            "fairness_proxy_pass_25",
        },
        "paper4_causal_treatment_identification_grid.csv": {
            "treatment_id",
            "identification_gate",
            "promotion_allowed",
        },
        "paper4_multi_period_sample_path_search.csv": {
            "strategy",
            "scenario",
            "policy_path",
        },
        "paper4_nonpromoted_lane_dashboard.csv": {
            "lane",
            "current_status",
            "promotion_condition",
        },
        "paper4_next_wave_claim_artifact_matrix.csv": {
            "priority",
            "artifact",
            "quarto_page",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_full_universe_topk_allocations.parquet": {
            "loan_id",
            "policy_id",
            "allocation_fraction",
        },
        "paper4_full_universe_ifrs9_tail_grid.parquet": {
            "loan_id",
            "policy_id",
            "scenario",
            "ecl_next_wave",
        },
        "paper4_online_conformal_method_policy_month.parquet": {
            "policy_id",
            "month",
            "online_method",
        },
        "paper4_tail_penalty_lp_allocations.parquet": {
            "loan_id",
            "tail_lp_policy_id",
            "allocation_fraction",
        },
        "paper4_best_online_policy_month_replay.parquet": {
            "policy_id",
            "month",
            "coverage_online",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_next_wave_no_promotion_and_full_exact_scope() -> None:
    status = _read_json("paper4_next_wave_status.json")
    exact = _read_json("paper4_full_universe_exact_topk_status.json")

    assert status["phase"] == "next_wave_top10_plus_nonpromoted_lanes"
    assert status["mode"] == "searches_and_gates_no_promotion"
    assert status["paper1_champion_protected"] is True
    assert status["paper4_final_promotion_created"] is False
    assert status["priorities_completed"] == 10
    assert status["nonpromoted_lanes_dashboard_rows"] >= 10
    assert exact["candidate_pool_n"] == 276869
    assert exact["promotion_json_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    full_eval = _read_csv("paper4_full_universe_topk_policy_eval.csv")
    challengers = full_eval.loc[
        full_eval["policy_id"].ne("paper1_economic_champion"), "solver_status"
    ].astype(str)
    assert not challengers.empty
    assert challengers.str.contains("optimal", case=False).all()

    comparison = _read_csv("paper4_full_universe_vs_exact_limited_comparison.csv")
    assert comparison["jaccard_full_vs_exact_limited"].between(0, 1).all()
    assert comparison["jaccard_full_vs_exact_limited"].min() >= 0.98

    causal = _read_csv("paper4_causal_treatment_identification_grid.csv")
    assert not causal["promotion_allowed"].astype(bool).any()


def test_paper4_next_wave_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_next_wave_claim_artifact_matrix.csv")

    expected_pages = {
        "19w-next-wave-exact-topk.qmd",
        "19x-next-wave-method-searches.qmd",
        "19y-nonpromoted-lanes-and-causal-gates.qmd",
        "19z-next-wave-promotion-dashboard.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_v3_deepening_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_v3_full_universe_all_policy_eval.csv": {
            "policy_id",
            "solver_status",
            "full_realized_return",
            "return_delta_vs_champion",
        },
        "paper4_v3_ifrs9_cashflow_policy_summary.csv": {
            "policy_id",
            "scenario",
            "cashflow_ecl_v3",
            "net_return_after_cashflow_ecl_v3",
        },
        "paper4_v3_online_conformal_method_summary.csv": {
            "online_method_v3",
            "coverage_policy_month_mean",
            "coverage_policy_month_min",
            "coverage_source_month_min",
        },
        "paper4_v3_formal_cvar_lp_results.csv": {
            "cvar_policy_id",
            "formal_cvar_loss",
            "solver_status",
            "candidate_pool_n",
        },
        "paper4_v3_selector_results.csv": {
            "policy_id",
            "v3_decision",
            "v3_gate_online",
            "v3_gate_mdcp",
        },
        "paper4_v3_causal_high_rate_dossier.csv": {
            "treatment_id",
            "max_smd_ipw_att",
            "promotion_allowed",
        },
        "paper4_v3_fairness_attribute_audit.csv": {
            "attribute",
            "available_in_lending_club_artifacts",
            "usable_for_fair_lending_claim",
            "decision",
        },
        "paper4_v3_multiperiod_loan_level_summary.csv": {
            "strategy",
            "policy_id",
            "net_cash_result",
        },
        "paper4_v3_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_v3_full_universe_all_policy_allocations.parquet": {
            "loan_id",
            "policy_id",
            "allocation_fraction",
            "exact_scope",
        },
        "paper4_v3_ifrs9_cashflow_loan_month_grid.parquet": {
            "loan_id",
            "policy_id",
            "scenario",
            "ead_start",
            "ecl_contribution_v3",
        },
        "paper4_v3_online_conformal_intervals.parquet": {
            "loan_id",
            "online_method_v3",
            "covered_online_v3",
            "interval_width_online_v3",
        },
        "paper4_v3_multiperiod_loan_level_state.parquet": {
            "strategy",
            "month",
            "cash_budget_end",
            "outstanding_exposure_end",
        },
        "paper4_v3_multiperiod_loan_level_decisions.parquet": {
            "strategy",
            "decision_month",
            "loan_id",
            "funded_exposure",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v3_no_promotion_and_scope() -> None:
    status = _read_json("paper4_v3_deepening_status.json")
    exact = _read_json("paper4_v3_full_universe_all_policy_status.json")

    assert status["phase"] == "v3_deepening_eight_open_priorities"
    assert status["mode"] == "research_deepening_no_promotion"
    assert status["paper1_champion_protected"] is True
    assert status["paper4_final_promotion_created"] is False
    assert status["priorities_completed"] == 8
    assert status["full_exact_policy_count"] == 45
    assert status["full_exact_target_policy_count"] == 45
    assert exact["candidate_pool_n"] == 276869
    assert exact["completed_policy_count"] == 45
    assert exact["promotion_json_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    full_eval = _read_csv("paper4_v3_full_universe_all_policy_eval.csv")
    assert full_eval["policy_id"].nunique() == 45
    assert "paper1_economic_champion" in set(full_eval["policy_id"])
    challengers = full_eval.loc[
        full_eval["policy_id"].ne("paper1_economic_champion"), "solver_status"
    ].astype(str)
    assert challengers.str.contains("optimal", case=False).all()

    online = _read_csv("paper4_v3_online_conformal_method_summary.csv")
    assert online["coverage_policy_month_min"].min() < 0.80
    assert not online["promotion_candidate"].astype(bool).any()

    selector = _read_csv("paper4_v3_selector_results.csv")
    assert set(selector["v3_decision"]) == {"park", "protected_paper1_champion"}
    champion = selector.loc[selector["policy_id"].eq("paper1_economic_champion")]
    assert champion["v3_decision"].iloc[0] == "protected_paper1_champion"
    assert not selector["v3_gate_online"].astype(bool).any()

    causal = _read_csv("paper4_v3_causal_high_rate_dossier.csv")
    assert not causal["promotion_allowed"].astype(bool).any()


def test_paper4_v3_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v3_claim_artifact_matrix.csv")

    expected_pages = {
        "19aa-v3-full-exact-and-ifrs9-realistic.qmd",
        "19ab-v3-online-cvar-governance.qmd",
        "19ac-v3-causal-fairness-multiperiod.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_v4_open_priority_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_challenger_local_search.csv": {
            "policy_id",
            "risk_tolerance",
            "gamma",
            "uncertainty_aversion",
            "realized_return_proxy_lgd45",
        },
        "paper4_challenger_local_bootstrap.csv": {
            "policy_id",
            "paired_monthly_diff",
            "prob_diff_positive",
        },
        "paper4_online_conformal_v4_method_summary.csv": {
            "online_method_v4",
            "coverage_policy_month_min",
            "coverage_source_month_min",
            "gate_pass",
        },
        "paper4_ifrs9_v4_contractual_policy_summary.csv": {
            "policy_id",
            "scenario",
            "contractual_ecl_v4",
            "net_return_after_contractual_ecl_v4",
        },
        "paper4_cvar_oce_v4_constraint_frontier.csv": {
            "cvar_policy_id",
            "cvar_cap",
            "return_floor",
            "solver_status",
        },
        "paper4_selector_v4_results.csv": {
            "policy_id",
            "paper4_v4_decision",
            "selector_v4_score",
            "gate_online_global_method",
        },
        "paper4_sdam_v4_dynamic_solver_summary.csv": {
            "strategy",
            "net_cash_result",
            "total_deployed",
        },
        "paper4_causal_high_rate_v4_dossier.csv": {
            "treatment_id",
            "cate_policy_value_allowed",
            "decision",
        },
        "paper4_fairness_v4_proxy_strategy.csv": {"item", "decision"},
        "paper4_mdcp_v4_source_coverage.csv": {
            "policy_id",
            "source_id",
            "worst_source_coverage_v4",
        },
        "paper4_regret_auditability_v4_frontier.csv": {
            "policy_id",
            "regret_gross_v4",
            "auditability_score_v4",
        },
        "paper4_common_sample_path_pairwise_ci.csv": {
            "policy_id",
            "scenario",
            "mean_diff_vs_paper1",
            "prob_beats_paper1",
        },
        "paper4_v4_claim_artifact_matrix.csv": {
            "priority",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    local = _read_csv("paper4_challenger_local_search.csv")
    assert local["policy_id"].nunique() == 100

    ci = _read_csv("paper4_common_sample_path_pairwise_ci.csv")
    assert ci["policy_id"].nunique() >= 100
    assert ci["n_paths"].min() >= 300

    expected_parquets = {
        "paper4_challenger_local_allocations.parquet": {"loan_id", "policy_id"},
        "paper4_online_conformal_v4_policy_month.parquet": {
            "policy_id",
            "online_method_v4",
            "coverage_online_v4",
        },
        "paper4_ifrs9_v4_contractual_loan_level.parquet": {
            "loan_id",
            "policy_id",
            "scenario",
            "contractual_ecl_v4",
        },
        "paper4_common_sample_paths_v4.parquet": {
            "policy_id",
            "scenario",
            "path_id",
            "simulated_return",
        },
        "paper4_sdam_v4_dynamic_solver_state.parquet": {
            "strategy",
            "month",
            "cash_budget_end",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v4_keeps_paper1_frozen_and_uses_working_champion_scope() -> None:
    status = _read_json("paper4_v4_open_priorities_status.json")
    working = _read_json("paper4_v4_working_champion.json")
    selector = _read_csv("paper4_selector_v4_results.csv")
    online = _read_csv("paper4_online_conformal_v4_method_summary.csv")

    assert status["phase"] == "v4_eleven_open_priorities"
    assert status["mode"] == "paper4_working_champions_allowed_paper1_frozen"
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert status["priorities_completed"] == 11
    assert working["scope"] == "paper4_working_champion_only"
    assert working["does_not_modify_paper1"] is True
    assert working["decision"] in {"review_candidate", "promote_to_paper4_working_champion"}
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    best_online_passes = bool(online.iloc[0]["gate_pass"])
    if not best_online_passes:
        assert not selector["paper4_v4_decision"].eq("promote_to_paper4_working_champion").any()
        assert not selector["gate_online_global_method"].astype(bool).any()

    paper1_promotion = json.loads(
        Path("models/final_project_promotion.json").read_text(encoding="utf-8")
    )
    champion = paper1_promotion["final_champion"]
    assert champion["label"] == EXPECTED_PAPER1_CHAMPION_LABEL
    assert champion["realized_total_return"] == pytest.approx(EXPECTED_PAPER1_CHAMPION_RETURN)

    for figure in status["generated_figures"]:
        path = PAPER4_ROOT / "figures" / figure
        assert path.exists(), figure
        assert path.stat().st_size > 0, figure


def test_paper4_v4_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v4_claim_artifact_matrix.csv")
    manifest = _read_csv("paper4_table0_source_manifest.csv")

    expected_pages = {
        "19ad-v4-challenger-online-mdcp.qmd",
        "19ae-v4-ifrs9-cvar-selector.qmd",
        "19af-v4-sdam-causal-fairness-regret.qmd",
        "19ag-v4-sample-paths-working-champion.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert expected_pages.issubset(set(claims["quarto_page"]))
    assert manifest["artifact"].str.contains("paper4_v4", regex=False).any()
    assert manifest["path_exists"].astype(bool).all()


def test_paper4_v5_blocker_resolution_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_v5_online_source_month_search.csv": {
            "online_method_v5",
            "coverage_source_month_min",
            "avg_width_loan",
            "gate_pass_80",
            "efficiency_gate_width_98",
        },
        "paper4_v5_ifrs9_contractual_policy_summary.csv": {
            "policy_id",
            "scenario",
            "sicr_rule",
            "contractual_ecl_v5",
            "net_return_after_contractual_ecl_v5",
            "ecl_estimation_scope_v5",
        },
        "paper4_v5_sicr_rule_comparison.csv": {
            "scenario",
            "sicr_rule",
            "stage",
            "ecl",
            "ead",
        },
        "paper4_v5_cvar_topk_expanded_frontier.csv": {
            "policy_id",
            "cvar90_expected_loss_v5",
            "return_after_cvar90_v5",
            "pareto_cvar_return_v5",
        },
        "paper4_v5_mdcp_aware_search.csv": {
            "policy_id",
            "mdcp_aware_score_v5",
            "mdcp_aware_decision_v5",
            "v5_source_month_min",
        },
        "paper4_v5_selector_threshold_protocol.csv": {
            "threshold_id",
            "value",
            "committee_rationale",
        },
        "paper4_v5_sdam_realistic_transition_summary.csv": {
            "horizon_months",
            "strategy",
            "cumulative_losses_v5",
            "cumulative_expected_loss_v5",
        },
        "paper4_v5_dla_endogenous_policy_trace.csv": {
            "horizon_months",
            "chosen_action",
            "coverage_signal",
            "cumulative_expected_loss",
        },
        "paper4_v5_causal_cate_policy_value_screen.csv": {
            "share_ci_crosses_zero",
            "policy_value_allowed",
        },
        "paper4_v5_fairness_protocol_decision.csv": {"item", "decision"},
        "paper4_v5_correlated_sample_path_ci.csv": {
            "policy_id",
            "scenario",
            "prob_beats_paper1",
            "cvar90_simulated_loss",
        },
        "paper4_v5_spo_dfl_hybrid_screen.csv": {"hybrid_id", "status"},
        "paper4_v5_temporal_ecl_downstream_value.csv": {
            "lane",
            "coverage_gate",
            "downstream_value",
        },
        "paper4_v5_cqr_decision_aware_screen.csv": {
            "method",
            "promotion_allowed",
            "status",
        },
        "paper4_v5_claim_artifact_matrix.csv": {
            "priority",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_v5_online_source_month_funded_intervals.parquet": {
            "policy_id",
            "loan_id",
            "online_method_v5",
            "interval_width_online_v5",
        },
        "paper4_v5_ifrs9_servicing_panel.parquet": {
            "policy_id",
            "loan_id",
            "month_index",
            "ead_start_proxy",
            "loss_cash_proxy",
        },
        "paper4_v5_correlated_sample_paths.parquet": {
            "policy_id",
            "scenario",
            "path_id",
            "simulated_return",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v5_keeps_paper1_frozen_and_documents_blockers() -> None:
    status = _read_json("paper4_v5_blocker_resolution_status.json")
    online = _read_csv("paper4_v5_online_source_month_search.csv")
    cate = _read_csv("paper4_v5_causal_cate_policy_value_screen.csv")
    fairness = _read_csv("paper4_v5_fairness_protocol_decision.csv")

    assert status["phase"] == "v5_blocker_resolution_wave"
    assert status["mode"] == "paper4_living_lab_no_paper1_changes"
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert status["priorities_completed"] == 13
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    best = online.iloc[0]
    assert bool(best["gate_pass_80"]) is True
    if float(best["avg_width_loan"]) > 0.98:
        assert status["online_efficiency_blocker"] is True
        assert bool(best["operational_gate_pass"]) is False

    assert not cate["policy_value_allowed"].astype(bool).any()
    assert "no fair-lending legal claim" in set(fairness["decision"])


def test_paper4_v5_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v5_claim_artifact_matrix.csv")

    expected_pages = {
        "19ah-v5-online-ifrs9-sicr.qmd",
        "19ai-v5-tail-mdcp-selector-sdam.qmd",
        "19aj-v5-causal-fairness-paths-hybrids.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_v6_priority_resolution_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_v6_online_source_month_efficiency_search.csv": {
            "online_method_v6",
            "deployable_without_outcomes",
            "coverage_policy_month_min",
            "coverage_source_month_min",
            "avg_width_loan",
            "promotion_eligible",
        },
        "paper4_v6_sicr_calibration_grid.csv": {
            "scenario",
            "sicr_rule_v6",
            "mean_stage2_share_v6",
            "sicr_recommendation_v6",
        },
        "paper4_v6_contractual_data_audit.csv": {"source_path", "rows", "loan_status"},
        "paper4_v6_servicing_gap_register.csv": {
            "requirement",
            "available_in_current_artifacts",
            "decision",
        },
        "paper4_v6_contractual_ifrs9_readiness.csv": {
            "readiness_score",
            "claim_scope",
            "promotion_allowed",
        },
        "paper4_v6_mdcp_solver_summary.csv": {
            "policy_id",
            "solver_status",
            "solver_lane",
        },
        "paper4_v6_cvar_constraint_summary.csv": {
            "policy_id",
            "solver_status",
            "cvar_cap",
            "return_floor",
            "solver_lane",
        },
        "paper4_v6_dla_loan_level_summary.csv": {
            "policy_id",
            "horizon_months",
            "claim_scope",
        },
        "paper4_v6_correlated_sample_path_ci.csv": {
            "policy_id",
            "scenario",
            "prob_beats_reference_v6",
            "cvar90_simulated_loss_v6",
        },
        "paper4_v6_cate_gate.csv": {
            "policy_value_allowed",
            "blocking_reason",
        },
        "paper4_v6_fairness_protocol.csv": {"item", "decision"},
        "paper4_v6_spo_dfl_hybrid_summary.csv": {
            "policy_id",
            "hybrid_status",
            "auditability_constraint_retained",
        },
        "paper4_v6_claim_artifact_matrix.csv": {
            "priority",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_v6_online_source_month_intervals.parquet": {
            "policy_id",
            "loan_id",
            "online_method_v6",
            "interval_width_online_v6",
        },
        "paper4_v6_solver_allocations.parquet": {
            "policy_id",
            "loan_id",
            "funded_exposure",
        },
        "paper4_v6_dla_loan_level_decisions.parquet": {
            "policy_id",
            "decision_month",
            "loan_id",
            "funded_exposure",
        },
        "paper4_v6_correlated_sample_paths.parquet": {
            "policy_id",
            "scenario",
            "path_id",
            "simulated_return_v6",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v6_no_promotion_and_documents_oracle_boundary() -> None:
    status = _read_json("paper4_v6_priority_resolution_status.json")
    online = _read_csv("paper4_v6_online_source_month_efficiency_search.csv")
    readiness = _read_csv("paper4_v6_contractual_ifrs9_readiness.csv")
    cate = _read_csv("paper4_v6_cate_gate.csv")
    fairness = _read_csv("paper4_v6_fairness_protocol.csv")

    assert status["phase"] == "v6_priority_resolution_wave"
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert status["online_promotion_eligible"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    passing = online[online["gate_pass_80"].astype(bool)]
    if not passing.empty:
        assert not passing["deployable_without_outcomes"].astype(bool).any()
    assert not readiness["promotion_allowed"].astype(bool).any()
    assert not cate["policy_value_allowed"].astype(bool).any()
    assert "no fair-lending legal claim" in set(fairness["decision"])


def test_paper4_v6_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v6_claim_artifact_matrix.csv")

    expected_pages = {
        "19ak-v6-online-ifrs9-efficient.qmd",
        "19al-v6-solvers-dla-sample-paths.qmd",
        "19am-v6-causal-fairness-hybrids.qmd",
    }
    pending_page = "19an-v6-pending-backlog.qmd"
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page
    assert (BOOK_DIR / "chapters/19-paper-mega-extension" / pending_page).exists()
    assert pending_page not in set(_registered_paper4_pages())

    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_v7_resolution_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_v7_online_deployable_weak_cell_search.csv": {
            "online_method_v7",
            "method_family",
            "coverage_source_month_defended_min",
            "avg_width_loan",
            "gate_pass_80_defended",
            "promotion_eligible",
        },
        "paper4_v7_online_min_support_pooling.csv": {
            "online_method_v7",
            "gate_scope",
            "min_effective_sample_size",
            "pooled_small_cells",
        },
        "paper4_v7_mdcp_soft_penalty_solver_summary.csv": {
            "policy_id",
            "solver_status",
            "mdcp_gate_proxy_v7",
            "auditability_score_v7",
        },
        "paper4_v7_cvar_adaptive_frontier.csv": {
            "policy_id",
            "solver_status",
            "cvar_cap",
            "return_floor",
            "frontier_feasible_v7",
        },
        "paper4_v7_sicr_mrm_shortlist.csv": {
            "sicr_rule_v6",
            "mean_stage2_share_v6",
            "mrm_decision_v7",
        },
        "paper4_v7_ifrs9_contractual_build_plan.csv": {
            "ifrs9_contractual_component",
            "current_status_v7",
            "next_step_v7",
        },
        "paper4_v7_dla_capital_state_summary.csv": {
            "policy_id",
            "horizon_months",
            "cumulative_realized_loss",
            "cumulative_capital_used",
            "claim_scope",
        },
        "paper4_v7_sample_path_calibration_grid.csv": {
            "scenario",
            "macro_ar1_rho",
            "lgd_cycle_beta",
            "claim_scope_v7",
        },
        "paper4_v7_common_sample_path_ci.csv": {
            "policy_id",
            "scenario",
            "prob_beats_reference_v6",
            "n_paths",
        },
        "paper4_v7_causal_fairness_blocker_matrix.csv": {
            "lane",
            "current_result_v7",
            "promotion_allowed",
        },
        "paper4_v7_fairness_no_legal_claim_statement.csv": {
            "statement_id",
            "statement",
            "must_appear_in_quarto",
        },
        "paper4_v7_hybrid_solver_candidate_registry.csv": {
            "policy_id",
            "hybrid_family_v7",
            "hybrid_candidate_status_v7",
        },
        "paper4_v7_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_v7_online_deployable_intervals.parquet": {
            "policy_id",
            "loan_id",
            "online_method_v7",
            "interval_width_online_v7",
        },
        "paper4_v7_online_deployable_policy_month.parquet": {
            "online_method_v7",
            "policy_id",
            "standalone_gate_cell",
        },
        "paper4_v7_online_deployable_source_month.parquet": {
            "online_method_v7",
            "source_id",
            "coverage_online_v7",
            "standalone_gate_cell",
        },
        "paper4_v7_mdcp_soft_penalty_allocations.parquet": {
            "policy_id",
            "loan_id",
            "funded_exposure",
        },
        "paper4_v7_cvar_adaptive_allocations.parquet": {
            "policy_id",
            "loan_id",
            "funded_exposure",
        },
        "paper4_v7_dla_capital_state_decisions.parquet": {
            "policy_id",
            "decision_month",
            "loan_id",
            "funded_exposure",
            "capital_charge_v7",
        },
        "paper4_v7_common_sample_paths.parquet": {
            "policy_id",
            "scenario",
            "path_id",
            "simulated_return_v6",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v7_no_promotion_and_documents_efficiency_blocker() -> None:
    status = _read_json("paper4_v7_resolution_status.json")
    online = _read_csv("paper4_v7_online_deployable_weak_cell_search.csv")
    causal = _read_csv("paper4_v7_causal_fairness_blocker_matrix.csv")
    fairness = _read_csv("paper4_v7_fairness_no_legal_claim_statement.csv")
    dla = _read_csv("paper4_v7_dla_capital_state_summary.csv")

    assert status["phase"] == "v7_resolution_loop"
    assert status["mode"] == "paper4_living_lab_no_paper1_changes"
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert status["online_gate80_defended_deployable_exists"] is True
    assert status["online_promotion_eligible"] is False
    assert status["online_efficiency_blocker"] is True
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    best = online.iloc[0]
    assert bool(best["gate_pass_80_defended"]) is True
    assert float(best["coverage_source_month_defended_min"]) >= 0.80
    assert float(best["avg_width_loan"]) > 0.95
    assert not online["promotion_eligible"].astype(bool).any()
    assert not causal["promotion_allowed"].astype(bool).any()
    assert (
        fairness["statement"]
        .str.contains("does not make a fair-lending legal compliance claim")
        .any()
    )
    assert float(dla["cumulative_realized_loss"].iloc[0]) > 0


def test_paper4_v7_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v7_claim_artifact_matrix.csv")

    expected_pages = {
        "19ao-v7-online-mdcp-resolution.qmd",
        "19ap-v7-ifrs9-dla-causal-governance.qmd",
        "19aq-v7-pending-and-results.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert {
        "19ao-v7-online-mdcp-resolution.qmd",
        "19ap-v7-ifrs9-dla-causal-governance.qmd",
    }.issubset(set(claims["quarto_page"]))


def test_paper4_v8_resolution_wave_artifacts_exist() -> None:
    expected_csvs = {
        "paper4_v8_literature_to_method_registry.csv": {
            "source_id",
            "primary_source_url",
            "concept",
            "implemented_artifact",
            "caveat",
        },
        "paper4_v8_online_efficiency_frontier.csv": {
            "online_method_v8",
            "coverage_source_month_defended_min",
            "avg_width_loan",
            "gate_pass_80_defended",
            "promotion_eligible",
        },
        "paper4_v8_online_breakpoint_report.csv": {
            "breakpoint_id",
            "online_method_v8",
            "width_gap_to_0p95",
            "interpretation",
        },
        "paper4_v8_cvar_bisection_frontier.csv": {
            "policy_id",
            "tested_cvar_cap",
            "feasible_v8",
            "bisection_floor",
            "solver_lane_v8",
        },
        "paper4_v8_mdcp_family_cap_solver_summary.csv": {
            "policy_id",
            "solver_status",
            "mdcp_family_gate_v8",
            "auditability_score_v8",
        },
        "paper4_v8_selector_governance_results.csv": {
            "policy_id",
            "lane",
            "selector_decision_v8",
            "primary_blocker_v8",
        },
        "paper4_v8_selector_committee_memo.csv": {
            "threshold_id",
            "threshold_value",
            "committee_rationale",
            "decision_effect",
        },
        "paper4_v8_dla_value_function_summary.csv": {
            "policy_id",
            "final_state_value_proxy_v8",
            "cumulative_realized_loss",
            "total_funded_exposure",
        },
        "paper4_v8_dla_vs_static_path_comparison.csv": {
            "comparison_id",
            "delta_final_state_value",
            "delta_cumulative_realized_loss",
            "interpretation",
        },
        "paper4_v8_causal_dossier.csv": {
            "treatment_id",
            "cate_policy_value_allowed",
            "decision_v8",
            "required_to_unblock",
        },
        "paper4_v8_fairness_permanent_proxy_protocol.csv": {
            "protocol_item",
            "decision_v8",
            "allowed_claim",
            "blocked_claim",
        },
        "paper4_v8_dfl_surrogate_training.csv": {
            "policy_id",
            "decision_loss_proxy_v8",
            "training_status_v8",
        },
        "paper4_v8_blocker_dashboard.csv": {
            "blocker_id",
            "status_v8",
            "current_diagnosis",
            "next_action",
        },
        "paper4_v8_v6_v7_change_report.csv": {
            "metric",
            "v6_value",
            "v7_value",
            "v8_value",
        },
        "paper4_v8_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    expected_parquets = {
        "paper4_v8_online_selected_intervals.parquet": {
            "policy_id",
            "loan_id",
            "online_method_v8",
            "interval_width_online_v8",
        },
        "paper4_v8_online_policy_month.parquet": {
            "online_method_v8",
            "policy_id",
            "standalone_gate_cell",
        },
        "paper4_v8_online_source_month.parquet": {
            "online_method_v8",
            "source_id",
            "coverage_online_v8",
            "standalone_gate_cell",
        },
        "paper4_v8_cvar_bisection_allocations.parquet": {
            "policy_id",
            "loan_id",
            "funded_exposure",
        },
        "paper4_v8_mdcp_family_cap_allocations.parquet": {
            "policy_id",
            "loan_id",
            "funded_exposure",
        },
        "paper4_v8_dla_value_function_decisions.parquet": {
            "policy_id",
            "decision_month",
            "loan_id",
            "funded_exposure",
            "action_v8",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v8_no_promotion_and_blockers_are_explicit() -> None:
    status = _read_json("paper4_v8_resolution_status.json")
    online = _read_csv("paper4_v8_online_efficiency_frontier.csv")
    selector = _read_csv("paper4_v8_selector_governance_results.csv")
    fairness = _read_csv("paper4_v8_fairness_permanent_proxy_protocol.csv")
    causal = _read_csv("paper4_v8_causal_dossier.csv")
    dashboard = _read_csv("paper4_v8_blocker_dashboard.csv")

    assert status["phase"] == "v8_resolution_wave"
    assert status["mode"] == "paper4_living_lab_no_paper1_changes"
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["causal_policy_value_allowed"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    assert online["gate_pass_80_defended"].astype(bool).any()
    best = (
        online[online["gate_pass_80_defended"].astype(bool)].sort_values("avg_width_loan").iloc[0]
    )
    assert float(best["coverage_source_month_defended_min"]) >= 0.80
    assert status["selector_promote_count"] == int(
        selector["selector_decision_v8"].eq("promote_to_paper4_working_candidate_only").sum()
    )
    assert (
        fairness["decision_v8"].str.contains("no_fair_lending_legal_claim|proxy", regex=True).any()
    )
    assert not causal["cate_policy_value_allowed"].astype(bool).any()
    assert {"resolved", "near_resolved", "data_blocked", "theory_blocked", "active"}.intersection(
        set(dashboard["status_v8"])
    )


def test_paper4_v8_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v8_claim_artifact_matrix.csv")

    expected_pages = {
        "19ar-v8-method-foundations-and-online.qmd",
        "19as-v8-solvers-selector-dla.qmd",
        "19at-v8-causal-fairness-hybrids-dashboard.qmd",
        "19au-v8-results-and-pending.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page

    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_v9_online_goal_resolution() -> None:
    status = _read_json("paper4_v9_online_goal_status.json")
    frontier = _read_csv("paper4_v9_online_efficiency_frontier.csv")
    breakpoint = _read_csv("paper4_v9_online_breakpoint_report.csv")
    claims = _read_csv("paper4_v9_claim_artifact_matrix.csv")

    assert status["phase"] == "v9_online_goal_resolution"
    assert status["mode"] == "paper4_living_lab_no_paper1_changes"
    assert status["online_goal_achieved"] is True
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    best = frontier[frontier["goal_pass"].astype(bool)].sort_values("avg_width_loan").iloc[0]
    assert float(best["coverage_source_month_defended_min"]) >= 0.80
    assert float(best["coverage_policy_month_defended_min"]) >= 0.90
    assert float(best["avg_width_loan"]) <= 0.95
    assert status["online_best_method_v9"] == best["online_method_v9"]

    assert {"v9_v8_reference", "v9_best_goal_passing_width"}.issubset(
        set(breakpoint["breakpoint_id"])
    )
    assert "19av-v9-online-goal-resolution.qmd" in set(claims["quarto_page"])

    selected = pd.read_parquet(TABLE_DIR / "paper4_v9_online_selected_intervals.parquet")
    policy = pd.read_parquet(TABLE_DIR / "paper4_v9_online_policy_month.parquet")
    source = pd.read_parquet(TABLE_DIR / "paper4_v9_online_source_month.parquet")
    assert {"online_method_v9", "qhat_v9", "interval_width_online_v9"}.issubset(selected.columns)
    assert {"online_method_v9", "coverage_online_v9", "standalone_gate_cell"}.issubset(
        policy.columns
    )
    assert {"online_method_v9", "source_id", "coverage_online_v9"}.issubset(source.columns)


def test_paper4_v9_page_is_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    page = "19av-v9-online-goal-resolution.qmd"
    assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
    assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())


def test_paper4_v10_resolution_wave_artifacts_and_guardrails() -> None:
    status = _read_json("paper4_v10_resolution_status.json")
    assert status["phase"] == "v10_resolution_wave"
    assert status["mode"] == "paper4_living_lab_no_paper1_changes"
    assert status["online_goal_achieved"] is True
    assert status["online_robustness_all_pass"] is True
    assert float(status["online_best_source_month_defended_min"]) >= 0.80
    assert float(status["online_best_policy_month_defended_min"]) >= 0.90
    assert float(status["online_best_width"]) <= 0.95
    assert status["selector_promote_count"] == 0
    assert status["primary_remaining_blocker"] == "ifrs9_contractual_data_blocker"
    assert status["contractual_ifrs9_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    expected_csvs = {
        "paper4_v10_online_robustness_summary.csv": {
            "robustness_item",
            "source_month_defended_min",
            "policy_month_defended_min",
            "avg_width_loan",
            "pass_v10",
        },
        "paper4_v10_online_min_support_sensitivity.csv": {
            "online_method_v9",
            "min_support",
            "gate_source80_policy90_width95",
        },
        "paper4_v10_mdcp_empirical_cap_solver_summary.csv": {
            "policy_id",
            "solver_status",
            "mdcp_family_gate_v8",
            "empirical_coverage_calibrated_v10",
        },
        "paper4_v10_cvar_expanded_frontier.csv": {
            "policy_id",
            "tested_cvar_cap_v10",
            "feasible_v10",
            "non_dominated_v10",
            "solver_lane_v10",
        },
        "paper4_v10_dfl_decision_loss_training.csv": {
            "policy_id",
            "decision_loss_proxy_v10",
            "constraint_pd_pass",
            "constraint_mdcp_proxy_pass",
            "training_status_v10",
        },
        "paper4_v10_dla_rollout_comparison.csv": {
            "policy_id",
            "delta_state_value_vs_static",
            "delta_loss_vs_static",
        },
        "paper4_v10_sample_path_stress_ci.csv": {
            "policy_id",
            "mean_loss",
            "p95_loss",
            "mean_default_count",
        },
        "paper4_v10_ifrs9_proxy_readiness.csv": {
            "readiness_score",
            "contractual_ifrs9_claim_allowed",
            "claim_scope_v10",
        },
        "paper4_v10_causal_dossier.csv": {
            "cate_policy_value_allowed",
            "decision_v10",
            "required_to_unblock_v10",
        },
        "paper4_v10_fairness_proxy_protocol.csv": {
            "legal_fair_lending_claim_allowed",
            "allowed_scope_v10",
        },
        "paper4_v10_selector_rerun_with_v9_online.csv": {
            "policy_id",
            "lane",
            "selector_decision_v10",
            "primary_blocker_v10",
        },
        "paper4_v10_blocker_dashboard.csv": {
            "blocker_id",
            "status_v10",
            "selector_promote_count_v10",
        },
        "paper4_v10_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    online = _read_csv("paper4_v10_online_robustness_summary.csv")
    assert online["pass_v10"].astype(bool).all()

    selector = _read_csv("paper4_v10_selector_rerun_with_v9_online.csv")
    assert (
        not selector["selector_decision_v10"].eq("promote_to_paper4_working_candidate_only").any()
    )
    assert (
        selector["primary_blocker_v10"]
        .str.contains("ifrs9_contractual_data_blocker|solver_infeasible", regex=True)
        .all()
    )

    dfl = _read_csv("paper4_v10_dfl_decision_loss_training.csv")
    assert len(dfl) >= 100
    assert dfl["constraint_pd_pass"].astype(bool).any()
    assert dfl["constraint_mdcp_proxy_pass"].astype(bool).any()

    readiness = _read_csv("paper4_v10_ifrs9_proxy_readiness.csv")
    assert not readiness["contractual_ifrs9_claim_allowed"].astype(bool).any()
    assert float(readiness["readiness_score"].iloc[0]) < 0.75

    fairness = _read_csv("paper4_v10_fairness_proxy_protocol.csv")
    assert not fairness["legal_fair_lending_claim_allowed"].astype(bool).any()

    expected_parquets = {
        "paper4_v10_mdcp_empirical_cap_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v10_cvar_expanded_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v10_dfl_decision_loss_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v10_dla_rollout_decisions.parquet": {"policy_id", "path_id", "loan_id"},
        "paper4_v10_dla_rollout_trace.parquet": {"policy_id", "path_id", "state_value_proxy_v10"},
        "paper4_v10_sample_path_stress_paths.parquet": {
            "policy_id",
            "path_id",
            "portfolio_loss_v10",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v10_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v10_claim_artifact_matrix.csv")
    expected_pages = {
        "19aw-v10-online-selector-mdcp.qmd",
        "19ax-v10-solvers-dla-dfl.qmd",
        "19ay-v10-ifrs9-causal-fairness-dashboard.qmd",
    }
    for page in expected_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists(), page
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages()), page
    assert expected_pages.issubset(set(claims["quarto_page"]))


def test_paper4_v11_promising_lanes_artifacts_and_guardrails() -> None:
    status = _read_json("paper4_v11_promising_lanes_status.json")
    assert status["phase"] == "v11_promising_lanes"
    assert status["mode"] == "paper4_living_lab_no_paper1_changes"
    assert status["online_goal_achieved"] is True
    assert status["candidate_universe_source_v11"] == "base_full_universe"
    assert int(status["cvar_universe_n_v11"]) >= 276_000
    assert int(status["cvar_pool_n_v11"]) > 12_000
    assert int(status["cvar_feasible_count_v11"]) > 0
    assert int(status["cvar_non_dominated_count_v11"]) > 0
    assert int(status["spo_dfl_candidate_count_v11"]) >= 4
    assert int(status["sample_path_policy_count_v11"]) > 0
    assert int(status["working_candidate_count_v11"]) > 0
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["final_promotion_allowed"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    expected_csvs = {
        "paper4_v11_cvar_topk_warm_frontier.csv": {
            "policy_id",
            "feasible_v11",
            "non_dominated_v11",
            "pool_n_v11",
            "universe_n_v11",
            "pool_design_v11",
        },
        "paper4_v11_dla_adp_value_coefficients.csv": {
            "feature",
            "coefficient",
            "model_scope_v11",
        },
        "paper4_v11_dla_adp_comparison.csv": {
            "policy_id",
            "delta_state_value_vs_static",
            "adp_scope_v11",
        },
        "paper4_v11_spo_dfl_training_summary.csv": {
            "training_id",
            "top1000_target_precision",
            "claim_scope_v11",
        },
        "paper4_v11_spo_dfl_candidate_summary.csv": {
            "policy_id",
            "constraint_pd_pass",
            "constraint_mdcp_proxy_pass",
            "training_scope_v11",
        },
        "paper4_v11_sample_path_calibration_table.csv": {
            "period",
            "original_grade",
            "default_multiplier_v11",
            "support_status_v11",
        },
        "paper4_v11_sample_path_calibrated_ci.csv": {
            "policy_id",
            "mean_loss",
            "p95_loss",
            "mean_default_multiplier",
        },
        "paper4_v11_working_candidate_registry.csv": {
            "policy_id",
            "lane_v11",
            "working_candidate_score_v11",
            "registry_decision_v11",
            "final_promotion_allowed",
        },
        "paper4_v11_blocker_dashboard.csv": {
            "blocker_id",
            "status_v11",
            "current_diagnosis",
        },
        "paper4_v11_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    cvar = _read_csv("paper4_v11_cvar_topk_warm_frontier.csv")
    assert cvar["feasible_v11"].astype(bool).any()
    assert cvar["pool_design_v11"].str.contains("balanced_topk").any()

    spo = _read_csv("paper4_v11_spo_dfl_candidate_summary.csv")
    assert spo["constraint_pd_pass"].astype(bool).any()
    assert spo["constraint_mdcp_proxy_pass"].astype(bool).all()

    registry = _read_csv("paper4_v11_working_candidate_registry.csv")
    assert not registry["final_promotion_allowed"].astype(bool).any()
    assert registry["registry_decision_v11"].str.contains("candidate_for_v12|keep_as_lane").all()

    expected_parquets = {
        "paper4_v11_cvar_topk_warm_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v11_dla_adp_decisions.parquet": {"policy_id", "path_id", "loan_id"},
        "paper4_v11_dla_adp_trace.parquet": {"policy_id", "path_id", "state_value_proxy_v11"},
        "paper4_v11_spo_dfl_allocations.parquet": {"policy_id", "loan_id", "spo_target_score_v11"},
        "paper4_v11_sample_path_calibrated_paths.parquet": {
            "policy_id",
            "path_id",
            "portfolio_loss_v11",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v11_page_is_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v11_claim_artifact_matrix.csv")
    page = "19az-v11-promising-lanes.qmd"
    assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
    assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())
    assert page in set(claims["quarto_page"])


def test_paper4_v12_resolution_wave_artifacts_and_guardrails() -> None:
    status = _read_json("paper4_v12_status.json")
    champion = _read_json("paper4_v12_working_champion.json")

    assert status["phase"] == "v12_resolution_wave"
    assert status["mode"] == "paper4_working_champion_allowed_no_paper1_changes"
    assert status["online_goal_achieved"] is True
    assert int(status["candidate_universe_n_v12"]) >= 276_000
    assert int(status["cvar_pool_n_v12"]) > 18_000
    assert int(status["cvar_feasible_count_v12"]) > 0
    assert int(status["cvar_non_dominated_count_v12"]) > 0
    assert int(status["mdcp_optimal_count_v12"]) > 0
    assert int(status["spo_candidate_count_v12"]) >= 3
    assert int(status["sample_path_policy_count_v12"]) > 0
    assert int(status["working_candidate_count_v12"]) > 0
    assert status["working_champion_created_v12"] is True
    assert status["working_champion_policy_id_v12"] == champion["policy_id"]
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert champion["scope"] == "paper4_working_champion_only"
    assert champion["paper1_promotion_allowed"] is False
    assert champion["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    paper1_promotion = json.loads(
        Path("models/final_project_promotion.json").read_text(encoding="utf-8")
    )
    assert paper1_promotion["final_champion"]["label"] == EXPECTED_PAPER1_CHAMPION_LABEL
    assert paper1_promotion["final_champion"]["realized_total_return"] == pytest.approx(
        EXPECTED_PAPER1_CHAMPION_RETURN
    )

    expected_csvs = {
        "paper4_v12_artifact_audit.csv": {"artifact", "kind", "version_guess", "exists"},
        "paper4_v12_method_reference_registry.csv": {
            "method_lane",
            "url",
            "claim_boundary_v12",
        },
        "paper4_v12_cvar_column_generation_frontier.csv": {
            "policy_id",
            "feasible_v12",
            "non_dominated_v12",
            "pool_n_v12",
            "cap_relaxation_v12",
        },
        "paper4_v12_cvar_active_constraints.csv": {
            "policy_id",
            "constraint_name",
            "slack_v12",
            "active_v12",
        },
        "paper4_v12_mdcp_source_cap_rationale.csv": {
            "source_id",
            "mapped_cap",
            "empirical_cap_v12",
        },
        "paper4_v12_mdcp_source_cap_solver_summary.csv": {
            "policy_id",
            "solver_status",
            "auditability_score_v12",
        },
        "paper4_v12_dla_state_schema.csv": {"sdam_element", "definition_v12", "role"},
        "paper4_v12_dla_fitted_value_coefficients.csv": {
            "feature",
            "coefficient",
            "model_scope_v12",
        },
        "paper4_v12_dla_fvi_comparison.csv": {
            "policy_id",
            "delta_state_value_vs_static",
            "adp_scope_v12",
        },
        "paper4_v12_spo_plus_surrogate_training.csv": {
            "training_id",
            "split",
            "mean_decision_regret",
            "claim_scope_v12",
        },
        "paper4_v12_spo_plus_surrogate_candidates.csv": {
            "policy_id",
            "decision_regret_proxy_v12",
            "training_scope_v12",
        },
        "paper4_v12_sample_path_macro_calibrated_ci.csv": {
            "policy_id",
            "mean_loss",
            "p95_loss",
            "mean_default_multiplier",
        },
        "paper4_v12_sample_path_pairwise_champion_ci.csv": {
            "policy_id",
            "reference_policy_id",
            "n_common_paths",
        },
        "paper4_v12_ifrs9_sicr_sensitivity.csv": {
            "scenario",
            "stage2_share_v12",
            "contractual_ifrs9_claim_allowed",
        },
        "paper4_v12_causal_cate_dossier.csv": {
            "dossier_item",
            "policy_value_allowed_v12",
            "blocker_v12",
        },
        "paper4_v12_fairness_proxy_governance.csv": {
            "protocol_item",
            "legal_fair_lending_claim_allowed",
            "allowed_scope_v12",
        },
        "paper4_v12_working_candidate_registry.csv": {
            "policy_id",
            "lane_v12",
            "working_candidate_score_v12",
            "registry_decision_v12",
        },
        "paper4_v12_blocker_dashboard.csv": {
            "blocker_id",
            "status_v12",
            "current_diagnosis",
        },
        "paper4_v12_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    cvar = _read_csv("paper4_v12_cvar_column_generation_frontier.csv")
    assert cvar["feasible_v12"].astype(bool).any()
    assert cvar["non_dominated_v12"].astype(bool).any()

    registry = _read_csv("paper4_v12_working_candidate_registry.csv")
    assert registry["registry_rank_v12"].min() == 1
    assert registry.iloc[0]["registry_decision_v12"] == "paper4_working_champion"
    assert registry.iloc[0]["policy_id"] == champion["policy_id"]
    assert not registry["paper1_promotion_allowed"].astype(bool).any()
    assert not registry["paper4_final_promotion_allowed"].astype(bool).any()

    ifrs9 = _read_csv("paper4_v12_ifrs9_sicr_sensitivity.csv")
    assert not ifrs9["contractual_ifrs9_claim_allowed"].astype(bool).any()
    fairness = _read_csv("paper4_v12_fairness_proxy_governance.csv")
    assert not fairness["legal_fair_lending_claim_allowed"].astype(bool).any()

    expected_parquets = {
        "paper4_v12_cvar_column_generation_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v12_mdcp_source_cap_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v12_dla_fvi_decisions.parquet": {"policy_id", "path_id", "loan_id"},
        "paper4_v12_dla_fvi_trace.parquet": {"policy_id", "path_id", "state_value_proxy_v12"},
        "paper4_v12_spo_plus_surrogate_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v12_sample_path_macro_calibrated_paths.parquet": {
            "policy_id",
            "path_id",
            "portfolio_loss_v12",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v12_page_is_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v12_claim_artifact_matrix.csv")
    page = "19ba-v12-resolution-wave.qmd"
    assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
    assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())
    assert page in set(claims["quarto_page"])


def test_paper4_v13_resolution_wave_artifacts_and_guardrails() -> None:
    status = _read_json("paper4_v13_status.json")
    champion = _read_json("paper4_v13_working_champion.json")

    assert status["phase"] == "v13_resolution_wave"
    assert status["mode"] == "paper4_working_champion_allowed_no_paper1_changes"
    assert status["online_goal_achieved"] is True
    assert int(status["candidate_universe_n_v13"]) >= 276_000
    assert int(status["cvar_pool_n_v13"]) > 24_000
    assert int(status["cvar_feasible_count_v13"]) > 0
    assert int(status["cvar_non_dominated_count_v13"]) > 0
    assert int(status["mdcp_optimal_count_v13"]) > 0
    assert int(status["spo_candidate_count_v13"]) >= 3
    assert int(status["sample_path_policy_count_v13"]) > 0
    assert int(status["working_candidate_count_v13"]) > 0
    assert status["working_champion_created_v13"] is True
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert status["working_champion_policy_id_v13"] == champion["policy_id"]
    assert champion["scope"] == "paper4_working_champion_only"
    assert champion["paper1_promotion_allowed"] is False
    assert champion["paper4_final_promotion_created"] is False
    assert champion["contractual_ifrs9_claim_allowed"] is False
    assert champion["fair_lending_legal_claim_allowed"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    paper1_promotion = json.loads(
        Path("models/final_project_promotion.json").read_text(encoding="utf-8")
    )
    assert paper1_promotion["final_champion"]["label"] == EXPECTED_PAPER1_CHAMPION_LABEL
    assert paper1_promotion["final_champion"]["realized_total_return"] == pytest.approx(
        EXPECTED_PAPER1_CHAMPION_RETURN
    )

    expected_csvs = {
        "paper4_v13_v12_artifact_audit.csv": {
            "artifact",
            "source_version",
            "kind",
            "exists",
        },
        "paper4_v13_method_reference_registry.csv": {
            "method_lane",
            "primary_source",
            "url",
            "claim_boundary_v13",
        },
        "paper4_v13_cvar_stronger_decomposition_frontier.csv": {
            "policy_id",
            "feasible_v13",
            "non_dominated_v13",
            "pool_n_v13",
            "frontier_scope_v13",
            "exact_full_universe_claim_v13",
        },
        "paper4_v13_cvar_active_constraints.csv": {
            "policy_id",
            "constraint_name",
            "slack_v12",
            "active_v12",
        },
        "paper4_v13_mdcp_cap_regime_rationale.csv": {
            "source_id",
            "mapped_cap",
            "empirical_cap_v12",
            "version_v13",
        },
        "paper4_v13_mdcp_cap_regime_solver_summary.csv": {
            "policy_id",
            "solver_status",
            "cap_regime_scope_v13",
        },
        "paper4_v13_dla_fitted_value_coefficients.csv": {
            "feature",
            "coefficient",
            "version_v13",
        },
        "paper4_v13_dla_fvi_comparison.csv": {
            "policy_id",
            "delta_state_value_vs_static",
            "adp_scope_v12",
        },
        "paper4_v13_spo_decision_loss_report.csv": {
            "split",
            "mean_decision_regret",
            "differentiable_layer_implemented_v13",
            "claim_scope_v13",
        },
        "paper4_v13_spo_decision_loss_candidates.csv": {
            "policy_id",
            "decision_regret_proxy_v12",
            "training_scope_v12",
        },
        "paper4_v13_sample_path_macro_calibrated_ci.csv": {
            "policy_id",
            "mean_loss",
            "p95_loss",
            "mean_default_multiplier",
        },
        "paper4_v13_sample_path_pairwise_champion_ci.csv": {
            "policy_id",
            "reference_policy_id",
            "n_common_paths",
        },
        "paper4_v13_ifrs9_readiness.csv": {
            "readiness_item",
            "claim_scope_v12",
            "contractual_ifrs9_claim_allowed",
        },
        "paper4_v13_ifrs9_sicr_sensitivity.csv": {
            "scenario",
            "stage2_share_v12",
            "contractual_ifrs9_claim_allowed",
        },
        "paper4_v13_ifrs9_data_blocker_register.csv": {
            "required_item",
            "available_in_current_artifacts",
            "v13_decision",
        },
        "paper4_v13_causal_cate_dossier.csv": {
            "dossier_item",
            "policy_value_allowed_v12",
            "blocker_v12",
        },
        "paper4_v13_fairness_proxy_governance.csv": {
            "protocol_item",
            "legal_fair_lending_claim_allowed",
            "allowed_scope_v12",
        },
        "paper4_v13_working_candidate_registry.csv": {
            "policy_id",
            "lane_v13",
            "working_candidate_score_v13",
            "registry_decision_v13",
        },
        "paper4_v13_champion_stress_test.csv": {
            "policy_id",
            "reference_champion_v13",
            "stress_decision_v13",
        },
        "paper4_v13_blocker_dashboard.csv": {
            "blocker_id",
            "status_v13",
            "current_diagnosis",
        },
        "paper4_v13_claim_artifact_matrix.csv": {
            "priority",
            "claim_status",
            "artifact",
            "quarto_page",
            "caveat",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    cvar = _read_csv("paper4_v13_cvar_stronger_decomposition_frontier.csv")
    assert cvar["feasible_v13"].astype(bool).any()
    assert cvar["non_dominated_v13"].astype(bool).any()
    assert not cvar["exact_full_universe_claim_v13"].astype(bool).any()

    mdcp = _read_csv("paper4_v13_mdcp_cap_regime_solver_summary.csv")
    assert mdcp["solver_status"].astype(str).str.contains("optimal", case=False).any()

    registry = _read_csv("paper4_v13_working_candidate_registry.csv")
    assert registry["registry_rank_v13"].min() == 1
    assert registry.iloc[0]["registry_decision_v13"] == "paper4_working_champion"
    assert registry.iloc[0]["policy_id"] == champion["policy_id"]
    assert not registry["paper1_promotion_allowed"].astype(bool).any()
    assert not registry["paper4_final_promotion_allowed"].astype(bool).any()

    stress = _read_csv("paper4_v13_champion_stress_test.csv")
    assert champion["policy_id"] in set(stress["policy_id"])
    assert set(stress["reference_champion_v13"]) == {champion["policy_id"]}

    ifrs9 = _read_csv("paper4_v13_ifrs9_sicr_sensitivity.csv")
    assert not ifrs9["contractual_ifrs9_claim_allowed"].astype(bool).any()
    causal = _read_csv("paper4_v13_causal_cate_dossier.csv")
    assert not causal["policy_value_allowed_v12"].astype(bool).any()
    fairness = _read_csv("paper4_v13_fairness_proxy_governance.csv")
    assert not fairness["legal_fair_lending_claim_allowed"].astype(bool).any()

    expected_parquets = {
        "paper4_v13_cvar_stronger_decomposition_allocations.parquet": {
            "policy_id",
            "loan_id",
        },
        "paper4_v13_mdcp_cap_regime_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v13_dla_fvi_decisions.parquet": {"policy_id", "path_id", "loan_id"},
        "paper4_v13_dla_fvi_trace.parquet": {
            "policy_id",
            "path_id",
            "state_value_proxy_v12",
        },
        "paper4_v13_dla_representative_allocations.parquet": {
            "policy_id",
            "loan_id",
            "allocation_scope_v13",
        },
        "paper4_v13_spo_decision_loss_allocations.parquet": {"policy_id", "loan_id"},
        "paper4_v13_sample_path_macro_calibrated_paths.parquet": {
            "policy_id",
            "path_id",
            "portfolio_loss_v13",
        },
    }
    for name, columns in expected_parquets.items():
        table = pd.read_parquet(TABLE_DIR / name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v13_page_is_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v13_claim_artifact_matrix.csv")
    page = "19bb-v13-resolution-wave.qmd"
    assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
    assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())
    assert page in set(claims["quarto_page"])


def test_paper4_v14_powell_framing_artifacts_and_guardrails() -> None:
    status = _read_json("paper4_v14_status.json")

    assert status["phase"] == "v14_powell_framing_audit"
    assert status["mode"] == "powell_framing_governance_no_paper1_changes"
    assert status["powell_source_exists"] is True
    assert status["previous_working_champion_policy_id_v13"] == "v13_fvi_return_recovery"
    assert status["working_champion_policy_id_v14"] == "v13_fvi_return_recovery"
    assert status["champion_replaced_v14"] is False
    assert int(status["framing_lane_count_v14"]) >= 12
    assert int(status["metric_count_v14"]) >= 13
    assert int(status["decision_count_v14"]) >= 11
    assert int(status["uncertainty_class_count_v14"]) == 12
    assert int(status["uncertainty_form_count_v14"]) >= 10
    assert int(status["decision_metric_interaction_rows_v14"]) > 100
    assert int(status["uncertainty_metric_interaction_rows_v14"]) > 100
    assert int(status["decision_uncertainty_interaction_rows_v14"]) > 100
    assert int(status["policy_class_evidence_rows_v14"]) >= 5
    assert int(status["claim_count_v14"]) >= 13
    assert status["required_v14_artifacts_exist"] is True
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    paper1_promotion = json.loads(
        Path("models/final_project_promotion.json").read_text(encoding="utf-8")
    )
    assert paper1_promotion["final_champion"]["label"] == EXPECTED_PAPER1_CHAMPION_LABEL
    assert paper1_promotion["final_champion"]["realized_total_return"] == pytest.approx(
        EXPECTED_PAPER1_CHAMPION_RETURN
    )

    expected_csvs = {
        "paper4_v14_v1_v13_artifact_audit.csv": {
            "artifact",
            "artifact_kind",
            "version_guess",
            "exists",
            "v14_use",
        },
        "paper4_v14_powell_framing_audit.csv": {
            "lane_id",
            "lane_name",
            "powell_stage",
            "metric_ids",
            "controlled_decision",
            "uncertainty_classes",
            "policy_class",
            "evidence_artifact",
            "current_status",
            "allowed_claim",
            "caveat",
        },
        "paper4_v14_metric_pyramid.csv": {
            "metric_id",
            "pyramid_level",
            "metric_family",
            "metric_name",
            "powell_role",
            "deployability_gate",
        },
        "paper4_v14_objective_target_limit_registry.csv": {
            "metric_id",
            "powell_role",
            "target_or_limit",
            "strictness",
            "v14_governance_decision",
        },
        "paper4_v14_decision_inventory.csv": {
            "decision_id",
            "decision_name",
            "decision_type",
            "decision_maker",
            "frequency",
            "lag",
            "affected_state_variables",
            "policy_class",
        },
        "paper4_v14_uncertainty_taxonomy.csv": {
            "uncertainty_id",
            "powell_uncertainty_class",
            "paper4_example",
            "affected_metrics",
            "affected_decisions",
            "stage_readiness",
        },
        "paper4_v14_uncertainty_forms_sample_path_design.csv": {
            "form_id",
            "uncertainty_form",
            "lending_club_manifestation",
            "v14_sample_path_design_spec",
            "overclaim_guardrail",
        },
        "paper4_v14_decision_metric_interaction_matrix.csv": {
            "decision_id",
            "metric_id",
            "impact_code",
            "impact_score",
            "rationale",
        },
        "paper4_v14_uncertainty_metric_interaction_matrix.csv": {
            "uncertainty_id",
            "metric_id",
            "impact_code",
            "impact_score",
            "rationale",
        },
        "paper4_v14_decision_uncertainty_interaction_matrix.csv": {
            "decision_id",
            "uncertainty_id",
            "impact_code",
            "impact_score",
            "rationale",
        },
        "paper4_v14_policy_class_evidence_matrix.csv": {
            "policy_class",
            "paper4_examples",
            "minimum_evidence_artifact",
            "claim_boundary",
        },
        "paper4_v14_base_vs_lookahead_model_registry.csv": {
            "artifact",
            "lane",
            "model_role",
            "powell_element",
            "decision_use",
            "evaluation_use",
            "current_limit",
        },
        "paper4_v14_working_champion_powell_audit.csv": {
            "policy_id",
            "criterion_id",
            "powell_metric_family",
            "metric_id",
            "gate_status_v14",
            "decision_impact",
        },
        "paper4_v14_stage_readiness_dashboard.csv": {
            "lane_id",
            "stage1_framing_readiness",
            "stage2_modeling_readiness",
            "stage3_implementation_readiness",
            "status_v14",
            "blocker_category",
        },
        "paper4_v14_claim_artifact_matrix.csv": {
            "priority",
            "claim",
            "claim_status",
            "artifact",
            "quarto_page",
            "claim_boundary_v14",
            "blocker_if_any",
            "no_claim_contractual_ifrs9",
            "no_claim_cate_policy_value",
            "no_claim_fair_lending_legal",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    framing = _read_csv("paper4_v14_powell_framing_audit.csv")
    assert {"DLA/FVI", "IFRS9/SICR proxy", "fairness proxy governance"}.issubset(
        set(framing["lane_name"])
    )
    policy_classes_text = " ".join(framing["policy_class"].astype(str))
    assert "PFA" in policy_classes_text
    assert "CFA" in policy_classes_text

    metrics = _read_csv("paper4_v14_metric_pyramid.csv")
    assert {"base", "risk", "estimation", "implementation"}.issubset(set(metrics["metric_family"]))
    assert {"objective", "target", "limit"}.issubset(set(metrics["powell_role"]))

    uncertainties = _read_csv("paper4_v14_uncertainty_taxonomy.csv")
    assert len(uncertainties) == 12
    assert uncertainties["uncertainty_id"].is_unique

    decision_metric = _read_csv("paper4_v14_decision_metric_interaction_matrix.csv")
    assert set(decision_metric["impact_code"]).issubset({"H", "M", "L", "N"})
    assert decision_metric["impact_code"].eq("H").any()

    model_registry = _read_csv("paper4_v14_base_vs_lookahead_model_registry.csv")
    assert {"base_evaluation_model", "lookahead_policy_model"}.issubset(
        set(model_registry["model_role"])
    )

    champion_audit = _read_csv("paper4_v14_working_champion_powell_audit.csv")
    assert set(champion_audit["policy_id"]) == {"v13_fvi_return_recovery"}
    assert {"blocked", "pass_working"}.intersection(set(champion_audit["gate_status_v14"]))

    claims = _read_csv("paper4_v14_claim_artifact_matrix.csv")
    assert claims["quarto_page"].eq("19bc-v14-powell-framing-audit.qmd").all()
    assert claims["no_claim_contractual_ifrs9"].astype(bool).any()
    assert claims["no_claim_cate_policy_value"].astype(bool).any()
    assert claims["no_claim_fair_lending_legal"].astype(bool).any()


def test_paper4_v14_page_is_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    claims = _read_csv("paper4_v14_claim_artifact_matrix.csv")
    page = "19bc-v14-powell-framing-audit.qmd"
    assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
    assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())
    assert page in set(claims["quarto_page"])


def test_paper4_v15_dynamic_engine_artifacts_exist() -> None:
    status = _read_json("paper4_v15_status.json")
    assert status["phase"] == "v15_dynamic_stress_engine"
    assert int(status["dynamic_policy_count_v15"]) >= 10
    assert int(status["dynamic_path_count_v15"]) >= 16
    assert int(status["dynamic_trace_rows_v15"]) > 0
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False

    schema = _read_json("paper4_v15_dynamic_state_schema.json")
    state_fields = {field["name"] for field in schema["fields"]}
    assert {
        "cash",
        "outstanding_principal",
        "repayments",
        "defaults",
        "recoveries",
        "ECL",
        "budget_remaining",
        "wealth",
    }.issubset(state_fields)

    trace = pd.read_parquet(TABLE_DIR / "paper4_v15_dynamic_policy_trace.parquet")
    assert not trace.empty
    assert {
        "policy_id",
        "path_id",
        "month",
        "cash",
        "outstanding_principal",
        "funded_exposure",
        "repayments",
        "defaults",
        "recoveries",
        "losses",
        "ECL",
        "wealth",
        "no_temporal_leakage_flag",
    }.issubset(trace.columns)
    assert trace["no_temporal_leakage_flag"].astype(bool).all()

    summary = _read_csv("paper4_v15_dynamic_policy_summary.csv")
    assert {
        "policy_id",
        "dynamic_value_score_v15",
        "paper4_champion_score_v15",
        "dynamic_governance_gate_pass_v15",
        "registry_decision_v15",
    }.issubset(summary.columns)
    assert summary["registry_decision_v15"].eq("paper4_working_champion_candidate").sum() == 1


def test_paper4_v16_cvar_spo_and_champion_decomposition_exist() -> None:
    status = _read_json("paper4_v16_status.json")
    assert status["phase"] == "v16_cvar_spo_champion_decomposition"
    assert int(status["cvar_frontier_rows_v16"]) >= 10
    assert int(status["cvar_infeasibility_rows_v16"]) >= 1
    assert status["spo_differentiable_layer_implemented_v16"] is False
    assert status["paper4_final_promotion_created"] is False

    frontier = _read_csv("paper4_v16_cvar_full_or_colgen_frontier.csv")
    assert {"policy_id", "v16_regime_label", "exact_full_universe_claim_v16"}.issubset(
        frontier.columns
    )
    assert not frontier["exact_full_universe_claim_v16"].astype(bool).any()

    cert = _read_csv("paper4_v16_cvar_strict_infeasibility_certificate.csv")
    assert {"policy_id", "certificate_scope", "academic_interpretation"}.issubset(cert.columns)
    assert cert["certificate_scope"].str.contains("not mathematical Farkas").any()

    blockers = _read_csv("paper4_v16_spo_dependency_blockers.csv")
    assert {"package", "available_v16", "decision_v16", "blocker_detail"}.issubset(blockers.columns)
    assert {"cvxpy", "cvxpylayers", "torch"}.issubset(set(blockers["package"]))
    assert (
        not blockers.loc[
            blockers["package"].isin(["cvxpy", "cvxpylayers", "torch"]), "available_v16"
        ]
        .astype(bool)
        .any()
    )

    report = _read_csv("paper4_v16_spo_training_report.csv")
    assert {"split", "mean_decision_regret_proxy", "claim_scope_v16"}.issubset(report.columns)
    assert report["claim_scope_v16"].str.contains("not_formal_spo_plus").all()

    decomp = _read_csv("paper4_v16_champion_decomposition_summary.csv")
    assert {"champion_policy_id", "challenger_policy_id", "selection_bucket"}.issubset(
        decomp.columns
    )
    assert {"champion_only", "challenger_only"}.issubset(set(decomp["selection_bucket"]))


def test_paper4_v17_ifrs9_cate_fairness_claims_blocked() -> None:
    status = _read_json("paper4_v17_status.json")
    assert status["phase"] == "v17_ifrs9_causal_fairness_gates"
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper4_final_promotion_created"] is False

    readiness = _read_csv("paper4_v17_ifrs9_proxy_panel_readiness.csv")
    assert {"requirement", "available_for_contractual_ifrs9", "v17_decision"}.issubset(
        readiness.columns
    )
    assert not readiness["available_for_contractual_ifrs9"].astype(bool).all()

    cate = _read_csv("paper4_v17_cate_gate_report.csv")
    assert {"gate", "status_v17", "policy_value_allowed"}.issubset(cate.columns)
    assert not cate["policy_value_allowed"].astype(bool).any()

    fairness = _read_csv("paper4_v17_fairness_proxy_only_protocol.csv")
    assert {"protocol_item", "status_v17", "allowed_claim"}.issubset(fairness.columns)
    assert fairness["allowed_claim"].str.contains("no fair-lending legal claim").any()


def test_paper4_v18_academic_synthesis_and_working_champion_only() -> None:
    status = _read_json("paper4_v18_status.json")
    champion = _read_json("paper4_v18_working_champion.json")
    assert status["phase"] == "v18_academic_synthesis"
    assert status["working_champion_policy_id_v18"] == champion["policy_id"]
    assert champion["scope"] == "paper4_working_champion_only"
    assert champion["paper1_promotion_allowed"] is False
    assert champion["paper4_final_promotion_created"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    contributions = _read_csv("paper4_v18_academic_contribution_map.csv")
    assert {"finding", "primary_artifact", "publishability_class"}.issubset(contributions.columns)
    assert len(contributions) >= 8

    blockers = _read_csv("paper4_v18_blocker_dashboard.csv")
    assert {"blocker_id", "status_v18", "current_diagnosis", "next_action"}.issubset(
        blockers.columns
    )
    assert {"data_blocked", "theory_blocked", "prohibited_claim"}.issubset(
        set(blockers["status_v18"])
    )

    claims = _read_csv("paper4_v18_claim_artifact_matrix.csv")
    assert {"claim", "allowed", "artifact", "quarto_page", "boundary"}.issubset(claims.columns)
    assert (
        not claims.loc[claims["claim"].str.contains("contractual IFRS9"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("CATE policy value"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("fair-lending legal"), "allowed"]
        .astype(bool)
        .any()
    )


def test_paper4_v15_v18_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    for page in [
        "19bd-v15-dynamic-stress-engine.qmd",
        "19be-v16-cvar-spo-champion-decomposition.qmd",
        "19bf-v17-ifrs9-causal-fairness-gates.qmd",
        "19bg-v18-academic-synthesis.qmd",
    ]:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())


def test_paper4_v19_dynamic_engine_v2_artifacts_exist() -> None:
    status = _read_json("paper4_v19_status.json")
    assert status["phase"] == "v19_dynamic_engine_v2"
    assert int(status["dynamic_policy_count_v19"]) >= 10
    assert int(status["dynamic_path_count_v19"]) >= 64
    assert int(status["horizon_months_v19"]) >= 36
    assert float(status["no_temporal_leakage_min_rate_v19"]) == pytest.approx(1.0)
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False

    for name in [
        "paper4_v19_dynamic_state_schema.json",
        "paper4_v19_status.json",
    ]:
        assert (STATUS_DIR / name).exists(), name

    trace = pd.read_parquet(TABLE_DIR / "paper4_v19_dynamic_policy_trace.parquet")
    assert {
        "policy_id",
        "path_id",
        "month",
        "cash",
        "outstanding_principal",
        "defaults",
        "prepayments",
        "recoveries",
        "losses",
        "ECL",
        "wealth",
        "no_temporal_leakage_flag",
    }.issubset(trace.columns)
    assert trace["no_temporal_leakage_flag"].astype(bool).all()

    summary = _read_csv("paper4_v19_dynamic_policy_summary.csv")
    assert {"policy_id", "dynamic_value_score_v15", "claim_scope_v19"}.issubset(summary.columns)
    assert "paper1_economic_champion" in set(summary["policy_id"])

    for name, columns in {
        "paper4_v19_horizon_sensitivity.csv": {"policy_id", "horizon_months"},
        "paper4_v19_gate_sensitivity.csv": {"gate_variant_v19", "winning_policy_id"},
        "paper4_v19_policy_pairwise_common_path_ci.csv": {
            "policy_id",
            "reference_policy_id",
            "prob_higher_wealth",
        },
        "paper4_v19_path_family_sensitivity.csv": {
            "policy_id",
            "macro_regime_v15",
            "final_wealth_mean",
        },
    }.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v20_dla_cvar_spo_resolution_artifacts_exist() -> None:
    status = _read_json("paper4_v20_status.json")
    assert status["phase"] == "v20_dla_cvar_spo_resolution"
    assert int(status["dynamic_policy_count_v20"]) >= 17
    assert int(status["endogenous_dla_policy_count_v20"]) >= 3
    assert status["spo_formal_differentiable_claim_allowed"] is False
    assert status["cvar_full_universe_exact_claim_v20"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False

    decisions = pd.read_parquet(TABLE_DIR / "paper4_v20_endogenous_dla_decisions.parquet")
    assert {"policy_id", "loan_id", "funded_exposure"}.issubset(decisions.columns)
    assert decisions["policy_id"].str.contains("v20_dla").any()

    dla_summary = _read_csv("paper4_v20_endogenous_dla_policy_summary.csv")
    assert {"policy_id", "final_wealth_mean", "cumulative_losses_p95"}.issubset(dla_summary.columns)

    cvar = _read_csv("paper4_v20_cvar_oce_frontier_v2.csv")
    assert {"policy_id", "regime_v20", "full_universe_exact_claim_v20"}.issubset(cvar.columns)
    assert not cvar["full_universe_exact_claim_v20"].astype(bool).any()

    deps = _read_csv("paper4_v20_spo_dependency_blockers.csv")
    assert {"package", "available_v20", "decision_v20"}.issubset(deps.columns)
    assert {"cvxpy", "cvxpylayers", "torch"}.issubset(set(deps["package"]))

    report = _read_csv("paper4_v20_spo_training_report.csv")
    assert {"model_id", "split", "claim_scope_v20"}.issubset(report.columns)
    assert report["claim_scope_v20"].str.contains("not formal differentiable SPO").any()

    memo = _read_csv("paper4_v20_champion_decision_memo.csv")
    assert {
        "candidate_policy_id",
        "current_reference_policy_id",
        "working_champion_change_recommended_v20",
        "decision_scope",
    }.issubset(memo.columns)
    assert memo["decision_scope"].str.contains("Paper 4 working champion only").all()


def test_paper4_v21_ifrs9_cate_fairness_gates_blocked() -> None:
    status = _read_json("paper4_v21_status.json")
    assert status["phase"] == "v21_ifrs9_causal_fairness_gates"
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper4_final_promotion_created"] is False

    readiness = _read_csv("paper4_v21_ifrs9_readiness_matrix.csv")
    assert {"requirement", "contractual_ready", "contractual_claim_allowed"}.issubset(
        readiness.columns
    )
    assert not readiness["contractual_claim_allowed"].astype(bool).any()

    proxy = pd.read_parquet(TABLE_DIR / "paper4_v21_ifrs9_proxy_monthly_panel.parquet")
    assert {"policy_id", "scenario", "ecl_proxy_v21", "stage2_composite_v21"}.issubset(
        proxy.columns
    )

    cate = _read_csv("paper4_v21_cate_gate_report.csv")
    assert {"gate", "status_v21", "cate_policy_value_allowed"}.issubset(cate.columns)
    assert not cate["cate_policy_value_allowed"].astype(bool).any()

    flags = _read_csv("paper4_v21_no_legal_claim_flags.csv")
    assert {"claim_or_requirement", "allowed_v21", "status_v21"}.issubset(flags.columns)
    legal = flags.loc[flags["claim_or_requirement"].eq("fair_lending_legal_claim")]
    assert not legal["allowed_v21"].astype(bool).any()


def test_paper4_v22_synthesis_and_working_champion_only() -> None:
    status = _read_json("paper4_v22_status.json")
    champion = _read_json("paper4_v22_working_champion.json")
    assert status["phase"] == "v22_academic_synthesis"
    assert status["working_champion_policy_id_v22"] == champion["policy_id"]
    assert champion["scope"] == "paper4_working_champion_only"
    assert champion["paper1_promotion_allowed"] is False
    assert champion["paper4_final_promotion_created"] is False
    assert champion["fair_lending_legal_claim_allowed"] is False
    assert champion["contractual_ifrs9_claim_allowed"] is False
    assert champion["cate_policy_value_allowed"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    blockers = _read_csv("paper4_v22_blocker_dashboard.csv")
    assert {"blocker_id", "status_v22", "current_diagnosis", "next_action"}.issubset(
        blockers.columns
    )
    assert {"data_blocked", "theory_blocked", "prohibited_claim"}.issubset(
        set(blockers["status_v22"])
    )

    claims = _read_csv("paper4_v22_claim_artifact_matrix.csv")
    assert {
        "claim",
        "allowed",
        "artifact",
        "quarto_page",
        "artifact_exists",
        "no_claim_contractual_ifrs9",
        "no_claim_cate_policy_value",
        "no_claim_fair_lending_legal",
    }.issubset(claims.columns)
    assert claims["artifact_exists"].astype(bool).all()
    assert (
        not claims.loc[claims["claim"].str.contains("contractual IFRS9"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("CATE policy value"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("fair-lending legal"), "allowed"]
        .astype(bool)
        .any()
    )


def test_paper4_v19_v22_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    for page in [
        "19bh-v19-dynamic-engine-v2.qmd",
        "19bi-v20-dla-cvar-spo-resolution.qmd",
        "19bj-v21-ifrs9-causal-fairness-gates.qmd",
        "19bk-v22-academic-synthesis.qmd",
    ]:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())


def test_paper4_v23_dynamic_scale_and_paths_artifacts_exist() -> None:
    status = _read_json("paper4_v23_status.json")
    assert status["phase"] == "v23_dynamic_scale_and_paths"
    assert int(status["dynamic_path_count_v23"]) >= 128
    assert int(status["dynamic_trace_rows_v23"]) > 100_000
    assert float(status["no_temporal_leakage_min_rate_v23"]) == pytest.approx(1.0)
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False

    trace = pd.read_parquet(TABLE_DIR / "paper4_v23_dynamic_policy_trace.parquet")
    assert {"policy_id", "path_id", "month", "wealth", "no_temporal_leakage_flag"}.issubset(
        trace.columns
    )
    assert trace["no_temporal_leakage_flag"].astype(bool).all()

    for name, columns in {
        "paper4_v23_scale_convergence.csv": {"policy_id", "n_paths", "wealth_ci95_width"},
        "paper4_v23_ranking_stability.csv": {"comparison", "spearman_rank_correlation"},
        "paper4_v23_performance_report.csv": {"run_id", "runtime_seconds", "bottleneck"},
        "paper4_v23_path_calibration_diagnostics.csv": {
            "macro_regime_v15",
            "path_claim_boundary",
        },
    }.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name


def test_paper4_v24_dla_cvar_spo_upgrade_artifacts_exist() -> None:
    status = _read_json("paper4_v24_status.json")
    assert status["phase"] == "v24_dla_cvar_spo_upgrade"
    assert int(status["adp_policy_count_v24"]) >= 4
    assert int(status["adp_trace_rows_v24"]) > 0
    assert status["cvar_exact_full_universe_claim_v24"] is False
    assert status["spo_formal_differentiable_claim_allowed"] is False
    assert status["paper1_artifacts_modified"] is False
    assert status["paper4_final_promotion_created"] is False

    decisions = pd.read_parquet(TABLE_DIR / "paper4_v24_dla_adp_decisions.parquet")
    assert {"policy_id", "loan_id", "funded_exposure", "decision_priority_score_v15"}.issubset(
        decisions.columns
    )
    assert decisions["policy_id"].str.contains("v24_adp").any()

    summary = _read_csv("paper4_v24_dla_adp_dynamic_summary.csv")
    assert {"policy_id", "final_wealth_mean", "cumulative_losses_p95"}.issubset(summary.columns)

    cvar = _read_csv("paper4_v24_cvar_frontier_non_dominated.csv")
    assert {"policy_id", "full_universe_exact_claim_v24", "column_generation_claim_v24"}.issubset(
        cvar.columns
    )
    assert not cvar["full_universe_exact_claim_v24"].astype(bool).any()

    cert = _read_csv("paper4_v24_cvar_infeasibility_certificate_formalized.csv")
    assert {"certificate_type_v24", "mathematical_infeasibility_proof_claim"}.issubset(cert.columns)
    assert not cert["mathematical_infeasibility_proof_claim"].astype(bool).any()

    memo = _read_csv("paper4_v24_committee_profile_memos.csv")
    assert {"committee_profile", "winning_policy_id", "decision_scope"}.issubset(memo.columns)
    assert memo["decision_scope"].str.contains("Paper 4 committee memo only").all()


def test_paper4_v25_ifrs9_causal_fairness_upgrade_claims_blocked() -> None:
    status = _read_json("paper4_v25_status.json")
    assert status["phase"] == "v25_ifrs9_causal_fairness_upgrade"
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper4_final_promotion_created"] is False

    panel = pd.read_parquet(TABLE_DIR / "paper4_v25_ifrs9_proxy_cashflow_panel.parquet")
    assert {
        "policy_id",
        "loan_id",
        "month_index",
        "ead_start_proxy_v25",
        "ecl_proxy_v25",
        "claim_scope_v25",
    }.issubset(panel.columns)
    assert panel["claim_scope_v25"].str.contains("not contractual IFRS9").all()

    cate = _read_csv("paper4_v25_cate_gate_report.csv")
    assert {"gate", "status_v25", "cate_policy_value_allowed"}.issubset(cate.columns)
    assert not cate["cate_policy_value_allowed"].astype(bool).any()

    flags = _read_csv("paper4_v25_no_legal_claim_flags.csv")
    assert {"claim_or_requirement", "allowed_v25", "status_v25"}.issubset(flags.columns)
    assert (
        not flags.loc[flags["claim_or_requirement"].eq("fair_lending_legal_claim"), "allowed_v25"]
        .astype(bool)
        .any()
    )


def test_paper4_v26_registry_docs_synthesis_and_working_champion_only() -> None:
    status = _read_json("paper4_v26_status.json")
    champion = _read_json("paper4_v26_working_champion.json")
    assert status["phase"] == "v26_registry_docs_synthesis"
    assert int(status["artifact_registry_rows_v26"]) >= 50
    assert int(status["candidate_registry_rows_v26"]) >= 10
    assert status["all_claim_artifacts_exist_v26"] is True
    assert status["working_champion_policy_id_v26"] == champion["policy_id"]
    assert champion["scope"] == "paper4_working_champion_only"
    assert champion["paper1_promotion_allowed"] is False
    assert champion["paper4_final_promotion_created"] is False
    assert champion["contractual_ifrs9_claim_allowed"] is False
    assert champion["cate_policy_value_allowed"] is False
    assert champion["fair_lending_legal_claim_allowed"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    registry = _read_csv("paper4_v26_artifact_registry.csv")
    assert {"artifact", "version", "path", "sha16", "path_exists"}.issubset(registry.columns)
    assert registry["path_exists"].astype(bool).all()

    candidates = _read_csv("paper4_v26_candidate_registry.csv")
    assert {"policy_id", "decision_v26", "full_candidate_score_v26"}.issubset(candidates.columns)
    assert "v13_spo_regret_audit_guarded" in set(candidates["policy_id"])
    spo_decision = candidates.loc[
        candidates["policy_id"].eq("v13_spo_regret_audit_guarded"), "decision_v26"
    ].iloc[0]
    assert spo_decision == "serious_challenger"

    claims = _read_csv("paper4_v26_claim_artifact_matrix.csv")
    assert claims["artifact_exists"].astype(bool).all()
    assert (
        not claims.loc[claims["claim"].str.contains("contractual IFRS9"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("CATE policy value"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("fair-lending legal"), "allowed"]
        .astype(bool)
        .any()
    )


def test_paper4_v23_v26_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    for page in [
        "19bl-v23-dynamic-scale-paths.qmd",
        "19bm-v24-dla-cvar-spo-upgrade.qmd",
        "19bn-v25-ifrs9-causal-fairness-upgrade.qmd",
        "19bo-v26-registry-and-synthesis.qmd",
    ]:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())


def test_paper4_v27_dynamic_scale_champion_stress_artifacts_exist() -> None:
    status = _read_json("paper4_v27_status.json")
    assert status["phase"] == "v27_dynamic_scale_champion_stress"
    assert int(status["dynamic_path_count_v27"]) >= 256
    assert int(status["dynamic_trace_rows_v27"]) > 200_000
    assert status["no_temporal_leakage_min_rate_v27"] == pytest.approx(1.0)
    assert status["paper4_final_promotion_created"] is False

    trace = pd.read_parquet(TABLE_DIR / "paper4_v27_dynamic_policy_trace.parquet")
    assert {"policy_id", "path_id", "month", "wealth", "no_temporal_leakage_flag"}.issubset(
        trace.columns
    )
    assert trace["no_temporal_leakage_flag"].astype(bool).all()

    stress = _read_csv("paper4_v27_champion_vs_cvar_stress_memo.csv")
    assert {
        "reference_policy_id",
        "challenger_policy_id",
        "prob_challenger_beats_reference",
        "decision_v27",
        "decision_scope",
    }.issubset(stress.columns)
    assert stress["decision_scope"].str.contains("Paper 4 working champion only").all()


def test_paper4_v28_cvar_dla_spo_upgrade_claim_boundaries() -> None:
    status = _read_json("paper4_v28_status.json")
    assert status["phase"] == "v28_cvar_dla_spo_upgrade"
    assert int(status["adp_policy_count_v28"]) >= 4
    assert status["cvar_exact_full_universe_claim_v28"] is False
    assert status["spo_formal_differentiable_claim_allowed"] is False
    assert status["paper4_final_promotion_created"] is False

    cvar = _read_csv("paper4_v28_cvar_frontier_non_dominated.csv")
    assert {"policy_id", "exact_full_universe_claim_v28", "restricted_master_claim_v28"}.issubset(
        cvar.columns
    )
    assert not cvar["exact_full_universe_claim_v28"].astype(bool).any()

    deps = _read_csv("paper4_v28_spo_dependency_blockers.csv")
    assert {"package", "formal_differentiable_spo_claim_allowed"}.issubset(deps.columns)
    assert not deps["formal_differentiable_spo_claim_allowed"].astype(bool).any()

    cert = _read_csv("paper4_v28_cvar_infeasibility_certificate_formalized.csv")
    assert {"certificate_type_v28", "mathematical_infeasibility_proof_claim"}.issubset(cert.columns)
    assert not cert["mathematical_infeasibility_proof_claim"].astype(bool).any()


def test_paper4_v29_ifrs9_causal_fairness_source_claims_blocked() -> None:
    status = _read_json("paper4_v29_status.json")
    assert status["phase"] == "v29_ifrs9_causal_fairness_source_upgrade"
    assert status["ifrs9_contractual_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False
    assert status["paper4_final_promotion_created"] is False

    panel = pd.read_parquet(TABLE_DIR / "paper4_v29_ifrs9_proxy_cashflow_panel.parquet")
    assert {
        "policy_id",
        "scenario_v29",
        "ecl_proxy_v29",
        "contractual_ifrs9_claim_allowed",
    }.issubset(panel.columns)
    assert not panel["contractual_ifrs9_claim_allowed"].astype(bool).any()

    cate = _read_csv("paper4_v29_cate_gate_report.csv")
    assert {"gate", "cate_policy_value_allowed", "claim_boundary_v29"}.issubset(cate.columns)
    assert not cate["cate_policy_value_allowed"].astype(bool).any()

    flags = _read_csv("paper4_v29_no_legal_claim_flags.csv")
    assert "allowed_v29" in flags.columns
    assert (
        not flags.loc[flags["claim_or_requirement"].eq("fair_lending_legal_claim"), "allowed_v29"]
        .astype(bool)
        .any()
    )


def test_paper4_v30_registry_docs_synthesis_and_working_champion_only() -> None:
    status = _read_json("paper4_v30_status.json")
    champion = _read_json("paper4_v30_working_champion.json")
    assert status["phase"] == "v30_registry_docs_synthesis"
    assert int(status["artifact_registry_rows_v30"]) >= 100
    assert int(status["candidate_registry_rows_v30"]) >= 10
    assert status["all_claim_artifacts_exist_v30"] is True
    assert status["working_champion_policy_id_v30"] == champion["policy_id"]
    assert champion["scope"] == "paper4_working_champion_only"
    assert champion["paper1_promotion_allowed"] is False
    assert champion["paper4_final_promotion_created"] is False
    assert champion["contractual_ifrs9_claim_allowed"] is False
    assert champion["cate_policy_value_allowed"] is False
    assert champion["fair_lending_legal_claim_allowed"] is False
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()

    candidates = _read_csv("paper4_v30_candidate_registry.csv")
    assert {"policy_id", "decision_v30", "full_candidate_score_v30"}.issubset(candidates.columns)
    assert "v13_spo_regret_audit_guarded" in set(candidates["policy_id"])

    claims = _read_csv("paper4_v30_claim_artifact_matrix.csv")
    assert claims["artifact_exists"].astype(bool).all()
    assert (
        not claims.loc[claims["claim"].str.contains("contractual IFRS9"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("CATE policy value"), "allowed"]
        .astype(bool)
        .any()
    )
    assert (
        not claims.loc[claims["claim"].str.contains("fair-lending legal"), "allowed"]
        .astype(bool)
        .any()
    )


def test_paper4_v27_v30_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    for page in [
        "19bp-v27-dynamic-scale-and-champion-stress.qmd",
        "19bq-v28-cvar-dla-spo-upgrade.qmd",
        "19br-v29-ifrs9-causal-fairness-source.qmd",
        "19bs-v30-registry-and-synthesis.qmd",
    ]:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())


def test_paper4_v31_v38_full_project_wave_artifacts_exist() -> None:
    v31 = _read_json("paper4_v31_status.json")
    assert v31["phase"] == "v31_dynamic_512_stress"
    assert int(v31["dynamic_path_count_v31"]) >= 256
    assert v31["paper4_final_promotion_created"] is False
    trace = pd.read_parquet(TABLE_DIR / "paper4_v31_dynamic_policy_trace.parquet")
    assert {"policy_id", "path_id", "month", "wealth", "no_temporal_leakage_flag"}.issubset(
        trace.columns
    )
    assert trace["no_temporal_leakage_flag"].astype(bool).all()

    v32 = _read_json("paper4_v32_status.json")
    assert v32["phase"] == "v32_spo_environment_and_oracle_regret"
    assert v32["formal_differentiable_spo_claim_allowed"] is False
    deps = _read_csv("paper4_v32_spo_dependency_blockers.csv")
    assert {"package", "formal_differentiable_spo_claim_allowed"}.issubset(deps.columns)
    assert not deps["formal_differentiable_spo_claim_allowed"].astype(bool).any()

    v33 = _read_json("paper4_v33_status.json")
    assert v33["phase"] == "v33_cvar_certificate_full_universe"
    assert v33["exact_full_universe_claim_v33"] is False
    cvar_guard = _read_csv("paper4_v33_cvar_full_universe_feasibility_attempt.csv")
    assert {"exact_full_universe_claim", "claim_boundary"}.issubset(cvar_guard.columns)
    assert not cvar_guard["exact_full_universe_claim"].astype(bool).any()

    v34 = _read_json("paper4_v34_status.json")
    assert v34["phase"] == "v34_dla_redesign_failure_analysis"
    assert v34["bellman_exact_claim_allowed"] is False

    v35 = _read_json("paper4_v35_status.json")
    assert v35["phase"] == "v35_online_macro_validation"
    assert v35["external_forecast_validation_claim_allowed"] is False

    v36 = _read_json("paper4_v36_status.json")
    assert v36["phase"] == "v36_ifrs9_sicr_upgrade"
    assert v36["contractual_ifrs9_claim_allowed"] is False

    v37 = _read_json("paper4_v37_status.json")
    assert v37["phase"] == "v37_cate_fairness_protocol"
    assert v37["causal_policy_value_allowed"] is False
    assert v37["fair_lending_legal_claim"] is False

    global_status = json.loads(
        Path("reports/paper_material/global/status/global_v38_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert global_status["phase"] == "global_v38_project_synthesis"
    assert global_status["paper1_promotion_happened_v38"] is False
    assert global_status["paper4_final_promotion_created"] is False


def test_paper4_v31_v38_pages_are_registered() -> None:
    config = (BOOK_DIR / "_quarto.yml").read_text(encoding="utf-8")
    for page in [
        "19bt-v31-dynamic-512-stress.qmd",
        "19bu-v32-spo-environment-oracle.qmd",
        "19bv-v33-cvar-certificate.qmd",
        "19bw-v34-dla-redesign.qmd",
        "19bx-v35-online-macro-validation.qmd",
        "19by-v36-ifrs9-sicr-upgrade.qmd",
        "19bz-v37-cate-fairness-protocol.qmd",
        "19ca-v38-final-synthesis.qmd",
    ]:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()
        assert page in CURATED_PAPER4_PAGES or page not in set(_registered_paper4_pages())


def test_paper4_curated_quarto_surface_is_compact() -> None:
    registered = _registered_paper4_pages()
    assert len(registered) <= 12
    assert set(registered) == CURATED_PAPER4_PAGES

    historical_pages = {
        "19u-artifact-catalog-and-claims.qmd",
        "19bs-v30-registry-and-synthesis.qmd",
        "19bt-v31-dynamic-512-stress.qmd",
        "19bz-v37-cate-fairness-protocol.qmd",
    }
    assert historical_pages.isdisjoint(set(registered))
    for page in historical_pages:
        assert (BOOK_DIR / "chapters/19-paper-mega-extension" / page).exists()


def test_paper4_living_lab_notebook_exists_and_has_backlog() -> None:
    notebook = PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md"
    assert notebook.exists()
    text = notebook.read_text(encoding="utf-8")

    for heading in [
        "## Current State",
        "## What Enters Quarto",
        "## Consolidated Findings",
        "## Claim Boundaries",
        "## Implementable Backlog",
        "## Template For Future Waves",
    ]:
        assert heading in text

    backlog = _read_csv("paper4_living_lab_backlog.csv")
    assert {"horizon", "lane", "executable_item", "status"}.issubset(backlog.columns)
    assert {"immediate", "short", "medium", "long"}.issubset(set(backlog["horizon"]))


def test_paper4_current_official_claims_are_artifact_backed() -> None:
    findings = _read_csv("paper4_current_official_findings.csv")
    claims = _read_csv("paper4_current_claim_boundaries.csv")
    page_registry = _read_csv("paper4_quarto_page_registry.csv")
    status = _read_json("paper4_quarto_restructure_status.json")

    assert status["official_quarto_page_count"] <= 12
    assert status["paper4_final_promotion_created"] is False
    assert status["contractual_ifrs9_claim_allowed"] is False
    assert status["causal_policy_value_allowed"] is False
    assert status["fair_lending_legal_claim"] is False

    assert {"finding_id", "evidence_artifact", "quarto_page", "status"}.issubset(findings.columns)
    for artifact in findings["evidence_artifact"]:
        assert Path(artifact).exists(), artifact

    assert {"claim", "allowed", "evidence_artifact", "prohibited_claim_flag"}.issubset(
        claims.columns
    )
    for artifact in claims["evidence_artifact"]:
        assert Path(artifact).exists(), artifact

    forbidden = claims.loc[
        claims["claim"].str.contains("Contractual IFRS9|CATE policy value|Fair-lending", case=False)
    ]
    assert not forbidden["allowed"].astype(bool).any()

    rendered = page_registry.loc[page_registry["rendered_in_quarto"].astype(bool), "page"]
    assert set(rendered) == CURATED_PAPER4_PAGES
    assert page_registry["path_exists"].astype(bool).all()
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v39_v40_living_lab_artifacts_exist_and_do_not_expand_quarto() -> None:
    v39 = _read_json("paper4_v39_status.json")
    v40 = _read_json("paper4_v40_status.json")

    assert v39["phase"] == "v39_living_lab_execution_after_quarto_restructure"
    assert v40["phase"] == "v40_spo_dla_claim_boundary_and_publishability"
    assert v39["quarto_compact_guardrail_pass"] is True
    assert int(v39["official_quarto_page_count"]) <= 12
    assert v39["paper4_final_promotion_created"] is False
    assert v40["paper4_final_promotion_created"] is False
    assert v39["contractual_ifrs9_claim_allowed"] is False
    assert v39["causal_policy_value_allowed"] is False
    assert v39["fair_lending_legal_claim"] is False
    assert v40["formal_differentiable_spo_claim_allowed"] is False
    assert v40["bellman_exact_claim_allowed"] is False

    expected_csvs = {
        "paper4_v39_online_source_family_holdout.csv": {
            "validation_item",
            "gate_source80_policy90_width95_v39",
            "live_deployability_claim_allowed",
            "claim_boundary_v39",
        },
        "paper4_v39_dynamic_candidate_stress.csv": {
            "policy_id",
            "paired_wealth_robustness_gate_v39",
            "tail_challenger_gate_v39",
            "decision_v39",
        },
        "paper4_v39_cvar_certificate_delta.csv": {
            "policy_id",
            "exact_full_universe_claim_v39",
            "mathematical_infeasibility_proof_claim_v39",
            "claim_boundary_v39",
        },
        "paper4_v39_source_governance_caps.csv": {
            "policy_id",
            "source_family",
            "recommended_source_cap_v39",
            "fair_lending_legal_claim_allowed",
        },
        "paper4_v39_candidate_registry.csv": {
            "policy_id",
            "full_candidate_score_v39",
            "paper1_promotion_allowed_v39",
            "claim_boundary_v39",
        },
        "paper4_v39_ifrs9_sicr_proxy_update.csv": {
            "policy_id",
            "sicr_rule_v36",
            "contractual_ifrs9_claim_allowed",
            "claim_boundary_v39",
        },
        "paper4_v40_spo_dependency_audit.csv": {
            "package",
            "formal_differentiable_spo_claim_allowed",
            "decision_v40",
        },
        "paper4_v40_spo_oracle_regret_v4.csv": {
            "month",
            "regret_ratio_v40",
            "formal_differentiable_spo_claim_allowed",
            "claim_boundary_v40",
        },
        "paper4_v40_dla_rollout_reaudit.csv": {
            "policy_id",
            "bellman_exact_claim_allowed",
            "underperformance_diagnosis_v40",
            "claim_boundary_v40",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    online = _read_csv("paper4_v39_online_source_family_holdout.csv")
    assert not online["live_deployability_claim_allowed"].astype(bool).any()

    cvar = _read_csv("paper4_v39_cvar_certificate_delta.csv")
    assert not cvar["exact_full_universe_claim_v39"].astype(bool).any()
    assert not cvar["mathematical_infeasibility_proof_claim_v39"].astype(bool).any()

    fairness = _read_csv("paper4_v39_source_governance_caps.csv")
    assert not fairness["fair_lending_legal_claim_allowed"].astype(bool).any()

    registry = _read_csv("paper4_v39_candidate_registry.csv")
    assert not registry["paper1_promotion_allowed_v39"].astype(bool).any()

    for note_name in [
        "paper4_spo_isolated_env_repro.md",
        "paper4_sample_path_claim_boundary.md",
        "paper4_cate_identification_reaudit.md",
        "paper4_fairness_protocol_update.md",
        "paper4_publishability_focus_memo.md",
        "paper4_v39_v40_documentation_cleanup.md",
    ]:
        note_path = PAPER4_ROOT / "notes" / note_name
        assert note_path.exists(), note_name
        assert note_path.stat().st_size > 0, note_name

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v39-v40" in notebook
    assert "No new Quarto page is promoted from v39-v40" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v41_v44_living_lab_artifacts_exist_and_preserve_claim_boundaries() -> None:
    v41 = _read_json("paper4_v41_status.json")
    v42 = _read_json("paper4_v42_status.json")
    v43 = _read_json("paper4_v43_status.json")
    v44 = _read_json("paper4_v44_status.json")

    assert v41["phase"] == "v41_online_cvar_source_dynamic_gate"
    assert v42["phase"] == "v42_spo_dependency_and_oracle_regret_training"
    assert v43["phase"] == "v43_dla_ifrs9_samplepaths_cate_fairness"
    assert v44["phase"] == "v44_registry_publishability_docs"
    assert v41["strict_live_deployability_claim_allowed"] is False
    assert v41["exact_full_universe_cvar_claim_allowed"] is False
    assert v42["formal_differentiable_spo_claim_allowed"] is False
    assert v43["bellman_exact_claim_allowed"] is False
    assert v43["contractual_ifrs9_claim_allowed"] is False
    assert v43["cate_policy_value_allowed"] is False
    assert v43["fair_lending_legal_claim_allowed"] is False
    assert v44["paper1_promotion_allowed_v44"] is False
    assert v44["quarto_compact_guardrail_pass"] is True
    assert int(v44["official_quarto_page_count"]) <= 12

    expected_csvs = {
        "paper4_v41_online_method_grid.csv": {
            "source_family",
            "method_v41",
            "gate_source80_policy90_width95_v41",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v41_online_source_family_solver.csv": {
            "source_family",
            "decision_v41",
            "claim_boundary_v41",
        },
        "paper4_v41_source_governance_solver_caps.csv": {
            "policy_id",
            "source_family",
            "recommended_source_cap_v41",
            "fair_lending_legal_claim_allowed",
        },
        "paper4_v41_source_solver_feasibility.csv": {
            "policy_id",
            "source_family",
            "required_relaxation_v41",
            "solver_constraint_use_v41",
        },
        "paper4_v41_cvar_column_generation_v2.csv": {
            "policy_id",
            "pricing_score_v41",
            "claim_scope_v41",
        },
        "paper4_v41_cvar_strict_infeasibility_v2.csv": {
            "policy_id",
            "exact_full_universe_claim_v41",
            "claim_boundary_v41",
        },
        "paper4_v41_dynamic_rerun_gate.csv": {
            "policy_id",
            "new_candidate_screen_v41",
            "expensive_512_1024_rerun_action_v41",
        },
        "paper4_v42_spo_dependency_isolation.csv": {
            "package",
            "formal_differentiable_spo_claim_allowed",
            "decision_v42",
        },
        "paper4_v42_spo_training_report.csv": {
            "model_v42",
            "mae_candidate_to_oracle_ratio",
            "formal_differentiable_spo_claim_allowed",
        },
        "paper4_v42_spo_temporal_regret_validation.csv": {
            "month",
            "model_v42",
            "predicted_regret_v42",
            "claim_boundary_v42",
        },
        "paper4_v43_dla_adp_rollout_grid.csv": {
            "policy_id",
            "bellman_exact_claim_allowed",
            "claim_boundary_v43",
        },
        "paper4_v43_external_macro_registry.csv": {
            "series_id",
            "source_url",
            "fetch_success",
        },
        "paper4_v43_ifrs9_proxy_policy_summary.csv": {
            "policy_id",
            "total_ecl_proxy",
            "contractual_ifrs9_claim_allowed",
        },
        "paper4_v43_ifrs9_sicr_sensitivity.csv": {
            "policy_id",
            "sicr_rule_v36",
            "contractual_ifrs9_claim_allowed",
        },
        "paper4_v43_cate_treatment_search.csv": {
            "gate",
            "cate_policy_value_allowed",
            "claim_boundary_v43",
        },
        "paper4_v43_fairness_source_protocol.csv": {
            "source_family",
            "fair_lending_legal_claim_allowed",
            "claim_boundary_v43",
        },
        "paper4_v44_candidate_registry.csv": {
            "policy_id",
            "full_candidate_score_v44",
            "paper1_promotion_allowed_v44",
            "claim_boundary_v44",
        },
        "paper4_v44_champion_decomposition.csv": {
            "policy_id",
            "wealth_gap_vs_reference_v44",
            "economic_interpretation_v44",
        },
        "paper4_v44_claim_matrix.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    online = _read_csv("paper4_v41_online_method_grid.csv")
    assert not online["strict_live_deployability_claim_allowed"].astype(bool).any()

    cvar = _read_csv("paper4_v41_cvar_strict_infeasibility_v2.csv")
    assert not cvar["exact_full_universe_claim_v41"].astype(bool).any()

    spo = _read_csv("paper4_v42_spo_training_report.csv")
    assert not spo["formal_differentiable_spo_claim_allowed"].astype(bool).any()

    ifrs9 = _read_csv("paper4_v43_ifrs9_proxy_policy_summary.csv")
    assert not ifrs9["contractual_ifrs9_claim_allowed"].astype(bool).any()

    fairness = _read_csv("paper4_v43_fairness_source_protocol.csv")
    assert not fairness["fair_lending_legal_claim_allowed"].astype(bool).any()

    registry = _read_csv("paper4_v44_candidate_registry.csv")
    assert not registry["paper1_promotion_allowed_v44"].astype(bool).any()

    champion = _read_json("paper4_v44_working_champion.json")
    assert champion["paper4_working_only"] is True
    assert champion["paper1_promotion_allowed"] is False

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v41-v44" in notebook
    assert "Keep v41-v44 in the living notebook" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v45_v48_living_lab_artifacts_exist_and_preserve_claim_boundaries() -> None:
    v45 = _read_json("paper4_v45_status.json")
    v46 = _read_json("paper4_v46_status.json")
    v47 = _read_json("paper4_v47_status.json")
    v48 = _read_json("paper4_v48_status.json")

    assert v45["phase"] == "v45_online_cvar_source_solver"
    assert v46["phase"] == "v46_spo_dla_dynamic"
    assert v47["phase"] == "v47_ifrs9_cate_fairness_paths"
    assert v48["phase"] == "v48_registry_docs_guardrails"
    assert v45["strict_live_deployability_claim_allowed"] is False
    assert v45["exact_full_universe_cvar_claim_allowed"] is False
    assert v45["fair_lending_legal_claim_allowed"] is False
    assert v46["formal_differentiable_spo_claim_allowed"] is False
    assert v46["bellman_exact_claim_allowed"] is False
    assert v47["contractual_ifrs9_claim_allowed"] is False
    assert v47["cate_policy_value_allowed"] is False
    assert v47["fair_lending_legal_claim_allowed"] is False
    assert v48["paper1_promotion_allowed_v48"] is False
    assert v48["quarto_compact_guardrail_pass"] is True
    assert int(v48["official_quarto_page_count"]) <= 12

    expected_csvs = {
        "paper4_v45_online_source_family_direct_holdout.csv": {
            "source_family",
            "source_family_defended_min_v45",
            "policy_month_direct_min_v45",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v45_online_recalibration_grid.csv": {
            "source_family",
            "method_v45",
            "gate_source80_policy90_width95_v45",
            "claim_boundary_v45",
        },
        "paper4_v45_cvar_slack_lp_certificate.csv": {
            "policy_id",
            "required_cvar_slack_v45",
            "exact_lp_certificate_claim_allowed_v45",
            "claim_boundary_v45",
        },
        "paper4_v45_cvar_full_universe_attempt.csv": {
            "attempt_id",
            "full_universe_lp_executed_v45",
            "exact_full_universe_claim_v45",
            "blocker_v45",
        },
        "paper4_v45_mdcp_source_solver_frontier.csv": {
            "policy_id",
            "source_family",
            "source_solver_cap_pass_v45",
            "fair_lending_legal_claim_allowed",
        },
        "paper4_v46_spo_isolated_env_smoke_test.csv": {
            "package",
            "formal_differentiable_spo_claim_allowed",
            "decision_v46",
        },
        "paper4_v46_spo_loan_level_oracle_regret.csv": {
            "model_v46",
            "decision_regret_v46",
            "formal_differentiable_spo_claim_allowed",
            "claim_boundary_v46",
        },
        "paper4_v46_dla_common_path_replay.csv": {
            "policy_id",
            "candidate_family_v46",
            "bellman_exact_claim_allowed",
            "claim_boundary_v46",
        },
        "paper4_v46_focused_dynamic_1024_ci.csv": {
            "candidate_policy_id",
            "prob_higher_wealth_v46",
            "focused_1024_rerun_executed_v46",
            "claim_boundary_v46",
        },
        "paper4_v47_ifrs9_proxy_policy_summary.csv": {
            "policy_id",
            "scenario",
            "total_ecl_proxy",
            "contractual_ifrs9_claim_allowed",
        },
        "paper4_v47_sicr_robust_calibration.csv": {
            "policy_id",
            "sicr_rule_v47",
            "contractual_ifrs9_claim_allowed",
            "claim_boundary_v47",
        },
        "paper4_v47_sample_path_macro_alignment.csv": {
            "path_family_v19",
            "external_forecast_validation_claim_allowed",
            "claim_boundary_v47",
        },
        "paper4_v47_champion_case_studies.csv": {
            "policy_id",
            "selection_relation_v47",
            "economic_case_label_v47",
        },
        "paper4_v47_cate_treatment_outcome_search.csv": {
            "diagnostic_v47",
            "cate_policy_value_allowed",
            "claim_boundary_v47",
        },
        "paper4_v47_fairness_source_protocol.csv": {
            "policy_id",
            "support_gate_v47",
            "fair_lending_legal_claim_allowed",
        },
        "paper4_v48_candidate_registry.csv": {
            "policy_id",
            "full_governance_score_v48",
            "paper1_promotion_allowed_v48",
            "claim_boundary_v48",
        },
        "paper4_v48_claim_matrix.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    online = _read_csv("paper4_v45_online_source_family_direct_holdout.csv")
    assert not online["strict_live_deployability_claim_allowed"].astype(bool).any()

    cvar = _read_csv("paper4_v45_cvar_full_universe_attempt.csv")
    assert not cvar["exact_full_universe_claim_v45"].astype(bool).any()

    spo = _read_csv("paper4_v46_spo_loan_level_oracle_regret.csv")
    assert not spo["formal_differentiable_spo_claim_allowed"].astype(bool).any()

    ifrs9 = _read_csv("paper4_v47_ifrs9_proxy_policy_summary.csv")
    assert not ifrs9["contractual_ifrs9_claim_allowed"].astype(bool).any()

    cate = _read_csv("paper4_v47_cate_treatment_outcome_search.csv")
    assert not cate["cate_policy_value_allowed"].astype(bool).any()

    fairness = _read_csv("paper4_v47_fairness_source_protocol.csv")
    assert not fairness["fair_lending_legal_claim_allowed"].astype(bool).any()

    registry = _read_csv("paper4_v48_candidate_registry.csv")
    assert not registry["paper1_promotion_allowed_v48"].astype(bool).any()

    champion = _read_json("paper4_v48_working_champion.json")
    assert champion["paper4_working_only"] is True
    assert champion["paper1_promotion_allowed"] is False

    decomposition = pd.read_parquet(
        TABLE_DIR / "paper4_v47_champion_decomposition_loan_level.parquet"
    )
    assert not decomposition.empty
    assert {"policy_id", "selection_relation_v47", "tail_loss_proxy_v47"}.issubset(
        decomposition.columns
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v45-v48" in notebook
    assert "Keep v45-v48 in the living notebook" in notebook
    assert (PAPER4_ROOT / "notes" / "paper4_future_wave_template.md").exists()
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v49_v54_self_directed_loop_repairs_artifacts_and_claim_boundaries() -> None:
    v49 = _read_json("paper4_v49_status.json")
    v50 = _read_json("paper4_v50_status.json")
    v51 = _read_json("paper4_v51_status.json")
    v52 = _read_json("paper4_v52_status.json")
    v53 = _read_json("paper4_v53_status.json")
    v54 = _read_json("paper4_v54_status.json")

    assert v49["phase"] == "v49_loss_matrix_online_repair"
    assert v50["phase"] == "v50_cvar_source_lp_solver"
    assert v51["phase"] == "v51_spo_dla_books_sicr_cases"
    assert v52["phase"] == "v52_registry_backlog_claims_self_directed"
    assert v53["phase"] == "v53_expected_loss_cvar_repair_and_budget_books"
    assert v54["phase"] == "v54_dynamic_budget_capped_book_replay"
    assert v49["strict_live_deployability_claim_allowed"] is False
    assert v50["exact_full_universe_cvar_claim_allowed"] is False
    assert v51["formal_differentiable_spo_claim_allowed"] is False
    assert v51["bellman_exact_claim_allowed"] is False
    assert v51["contractual_ifrs9_claim_allowed"] is False
    assert v52["paper1_promotion_allowed_v52"] is False
    assert v53["v50_zero_cvar_artifact_repaired_v53"] is True
    assert v53["exact_full_universe_cvar_claim_allowed"] is False
    assert v53["paper1_promotion_allowed_v53"] is False
    assert v54["bellman_exact_claim_allowed"] is False
    assert v54["live_no_leakage_claim_allowed"] is False
    assert v54["paper1_promotion_allowed_v54"] is False

    expected_csvs = {
        "paper4_v49_online_qhat_recalibration_search.csv": {
            "source_family",
            "delta_qhat_v49",
            "gate_source80_policy90_width95_v49",
            "claim_boundary_v49",
        },
        "paper4_v50_cvar_source_lp_frontier.csv": {
            "regime_v50",
            "solver_success_v50",
            "scenario_loss_cvar90_v50",
            "exact_full_universe_claim_v50",
        },
        "paper4_v51_policy_scenario_replay.csv": {
            "policy_id",
            "mean_scenario_return_v51",
            "p95_scenario_loss_v51",
            "claim_boundary_v51",
        },
        "paper4_v52_claim_matrix.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
        "paper4_v53_expected_loss_frontier.csv": {
            "regime_v53",
            "solver_success_v53",
            "scenario_loss_cvar90_v53",
            "exact_full_universe_claim_v53",
        },
        "paper4_v53_binary_vs_expected_diagnostic.csv": {
            "diagnostic_id",
            "source_artifact",
            "interpretation",
            "claim_impact",
        },
        "paper4_v53_budget_capped_policy_replay.csv": {
            "policy_id",
            "funded_exposure_v53",
            "budget_capped_v53",
            "p95_scenario_loss_v53",
        },
        "paper4_v53_claim_matrix_delta.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
        "paper4_v54_dynamic_budget_capped_book_summary.csv": {
            "policy_id",
            "final_wealth_mean_v54",
            "final_loss_p95_v54",
            "live_no_leakage_claim_allowed_v54",
        },
        "paper4_v54_dynamic_champion_decision_memo.csv": {
            "memo_id",
            "best_policy_id",
            "working_champion_change_allowed",
            "claim_boundary",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    matrix_v49 = pd.read_parquet(TABLE_DIR / "paper4_v49_loan_scenario_loss_matrix.parquet")
    matrix_v53 = pd.read_parquet(TABLE_DIR / "paper4_v53_expected_loss_matrix.parquet")
    trace_v54 = pd.read_parquet(TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_trace.parquet")
    assert len(matrix_v49) >= 1_000_000
    assert len(matrix_v53) == len(matrix_v49)
    assert {"hybrid_loss_amount_v53", "hybrid_return_amount_v53"}.issubset(matrix_v53.columns)
    assert pd.to_numeric(matrix_v53["hybrid_loss_amount_v53"], errors="coerce").max() > 0
    assert not trace_v54.empty
    assert {"cash_v54", "outstanding_principal_v54", "wealth_v54"}.issubset(trace_v54.columns)

    v50_frontier = _read_csv("paper4_v50_cvar_source_lp_frontier.csv")
    v53_frontier = _read_csv("paper4_v53_expected_loss_frontier.csv")
    assert pd.to_numeric(v50_frontier["scenario_loss_cvar90_v50"], errors="coerce").max() == 0
    assert pd.to_numeric(v53_frontier["scenario_loss_cvar90_v53"], errors="coerce").min() > 0

    replay = _read_csv("paper4_v53_budget_capped_policy_replay.csv")
    assert replay["budget_capped_v53"].astype(bool).all()
    summary = _read_csv("paper4_v54_dynamic_budget_capped_book_summary.csv")
    assert not summary["live_no_leakage_claim_allowed_v54"].astype(bool).any()

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v49-v52" in notebook
    assert "Wave v53: Expected-Loss CVaR Repair" in notebook
    assert "Wave v54: Dynamic Replay Of Repaired Books" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v55_v61_unlock_loop_keeps_claim_boundaries() -> None:
    v55 = _read_json("paper4_v55_status.json")
    v56 = _read_json("paper4_v56_status.json")
    v57 = _read_json("paper4_v57_status.json")
    v58 = _read_json("paper4_v58_status.json")
    v59 = _read_json("paper4_v59_status.json")
    v60 = _read_json("paper4_v60_status.json")
    v61 = _read_json("paper4_v61_status.json")
    v62 = _read_json("paper4_v62_status.json")

    assert v55["phase"] == "v55_full_comparable_universe_lineage_unlock"
    assert v55["maximal_comparable_universe_rows_v55"] == 276_869
    assert v55["prediction_test_join_rate_v55"] == pytest.approx(1.0)
    assert v56["phase"] == "v56_expanded_restricted_master_cvar_source_solver"
    assert v56["restricted_master_columns_v56"] >= 36_000
    assert v57["phase"] == "v57_online_spo_dla_ifrs9_gate_updates"
    assert v58["phase"] == "v58_registry_claims_storage_notebook"
    assert v59["phase"] == "v59_adaptive_cvar_feasibility_frontier"
    assert v59["adaptive_feasible_rows_v59"] >= 1
    assert v60["phase"] == "v60_dynamic_stress_gate_for_v59_candidates"
    assert v61["phase"] == "v61_source_diversified_cvar_frontier"
    assert v62["phase"] == "v62_source_diversification_slack_certificate"

    for status in (v55, v56, v57, v58, v59, v60, v61, v62):
        assert status.get("paper4_final_promotion_created") is False

    assert v56["exact_full_universe_cvar_claim_allowed"] is False
    assert v57["strict_live_deployability_claim_allowed"] is False
    assert v57["formal_differentiable_spo_claim_allowed"] is False
    assert v57["bellman_exact_claim_allowed"] is False
    assert v57["contractual_ifrs9_claim_allowed"] is False
    assert v57["cate_policy_value_allowed"] is False
    assert v57["fair_lending_legal_claim_allowed"] is False
    assert v60["focused_512_or_1024_rerun_executed_v60"] is False
    assert v60["paper1_promotion_allowed_v60"] is False
    assert v61["exact_full_universe_cvar_claim_allowed"] is False
    assert v62["paper1_promotion_allowed_v62"] is False
    assert v62["max_required_cap_slack_share_v62"] > 0

    expected_csvs = {
        "paper4_v55_full_universe_lineage_audit.csv": {
            "lineage_item",
            "status_v55",
            "match_rows_v55",
            "claim_boundary_v55",
        },
        "paper4_v55_join_match_rate_table.csv": {
            "left_source",
            "right_source",
            "intersection_n",
            "left_match_rate",
        },
        "paper4_v56_cvar_full_comparable_frontier.csv": {
            "regime_v56",
            "solver_success_v56",
            "exact_full_universe_claim_v56",
            "claim_boundary_v56",
        },
        "paper4_v56_cvar_slack_certificate.csv": {
            "regime_v56",
            "required_cvar_slack_v56",
            "certificate_scope_v56",
        },
        "paper4_v57_online_source_family_direct_repair.csv": {
            "source_family",
            "gate_source80_policy90_width95_v57",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v57_spo_dependency_probe.csv": {
            "package",
            "formal_differentiable_spo_claim_allowed",
            "claim_boundary_v57",
        },
        "paper4_v57_ifrs9_sicr_proxy_panel_update.csv": {
            "sicr_rule_v57",
            "contractual_ifrs9_claim_allowed",
            "claim_boundary_v57",
        },
        "paper4_v58_candidate_registry.csv": {
            "policy_id",
            "full_governance_score_v58",
            "paper1_promotion_allowed_v58",
        },
        "paper4_v58_claim_matrix.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
        "paper4_v59_cvar_adaptive_feasible_frontier.csv": {
            "regime_v59",
            "solver_success_v59",
            "scenario_loss_cvar90_v59",
            "claim_boundary_v59",
        },
        "paper4_v60_dynamic_rerun_gate_memo.csv": {
            "memo_id",
            "focused_512_or_1024_rerun_executed_v60",
            "working_champion_change_allowed_v60",
        },
        "paper4_v61_source_diversified_frontier.csv": {
            "regime_v61",
            "solver_success_v61",
            "hard_family_caps_v61",
            "claim_boundary_v61",
        },
        "paper4_v61_blocker_dashboard.csv": {
            "lane",
            "status",
            "evidence_artifact",
            "next_unlock",
        },
        "paper4_v62_source_diversification_slack_certificate.csv": {
            "policy_id",
            "source_family",
            "required_cap_slack_share_v62",
            "certificate_scope_v62",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    universe = pd.read_parquet(
        TABLE_DIR / "paper4_v55_maximal_comparable_universe.parquet",
        columns=["loan_id", "pd_point", "qhat_v4", "grade"],
    )
    assert len(universe) == 276_869
    assert universe["loan_id"].is_unique
    assert universe["pd_point"].between(0, 1).all()

    lineage = _read_csv("paper4_v55_full_universe_lineage_audit.csv")
    assert (
        lineage.loc[
            lineage["lineage_item"].eq("exact_prediction_to_test_join"), "match_rows_v55"
        ].iloc[0]
        == 276_869
    )

    v56_frontier = _read_csv("paper4_v56_cvar_full_comparable_frontier.csv")
    assert not v56_frontier["exact_full_universe_claim_v56"].astype(bool).any()

    v57_online = _read_csv("paper4_v57_online_source_family_direct_repair.csv")
    assert not v57_online["strict_live_deployability_claim_allowed"].astype(bool).any()

    v58_claims = _read_csv("paper4_v58_claim_matrix.csv")
    prohibited = v58_claims.loc[v58_claims["claim_id"].str.contains("exact|online|spo|ifrs9")]
    assert not prohibited["allowed"].astype(bool).all()

    v60_memo = _read_csv("paper4_v60_dynamic_rerun_gate_memo.csv")
    assert not v60_memo["focused_512_or_1024_rerun_executed_v60"].astype(bool).any()
    assert not v60_memo["working_champion_change_allowed_v60"].astype(bool).any()

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Source-diversified CVaR challenger is feasible and promotable." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v55-v58" in notebook
    assert "Wave v59" in notebook
    assert "Wave v60" in notebook
    assert "Wave v61" in notebook
    assert "Wave v62" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v63_source_governance_repair_is_lab_only() -> None:
    status = _read_json("paper4_v63_status.json")

    assert status["phase"] == "v63_source_governance_repair_frontier"
    assert status["source_repair_frontier_rows_v63"] >= 4
    assert status["source_repair_success_rows_v63"] >= 1
    assert status["source_repair_book_rows_v63"] > 0
    assert status["dynamic_rerun_recommended_rows_v63"] == 0
    assert status["exact_full_universe_cvar_claim_allowed"] is False
    assert status["paper1_promotion_allowed_v63"] is False
    assert status["paper4_working_champion_changed_v63"] is False
    assert status["paper4_final_promotion_created"] is False

    expected_csvs = {
        "paper4_v63_source_repair_frontier.csv": {
            "policy_id",
            "source_repair_success_v63",
            "scenario_loss_cvar90_v63",
            "max_required_cap_slack_share_v63",
            "exact_full_universe_claim_v63",
            "paper1_promotion_allowed_v63",
            "claim_boundary_v63",
        },
        "paper4_v63_source_repair_concentration.csv": {
            "policy_id",
            "source_family",
            "top_exposure_share_v63",
            "target_cap_v63",
            "required_cap_slack_share_v63",
        },
        "paper4_v63_dynamic_gate_memo.csv": {
            "policy_id",
            "dynamic_512_or_1024_rerun_recommended_v63",
            "working_champion_change_allowed_v63",
            "rerun_decision_reason_v63",
        },
        "paper4_v63_claim_matrix_delta.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    books = pd.read_parquet(TABLE_DIR / "paper4_v63_source_repair_candidate_books.parquet")
    assert not books.empty
    assert {"policy_id_v63", "allocated_exposure_v63", "claim_boundary_v63"}.issubset(books.columns)

    frontier = _read_csv("paper4_v63_source_repair_frontier.csv")
    assert frontier["source_repair_success_v63"].astype(bool).any()
    assert not frontier["exact_full_universe_claim_v63"].astype(bool).any()
    assert not frontier["paper1_promotion_allowed_v63"].astype(bool).any()
    assert frontier["max_source_share_v63"].max() < 0.90

    concentration = _read_csv("paper4_v63_source_repair_concentration.csv")
    grade_top = concentration.loc[concentration["source_family"].eq("grade")]
    assert grade_top["top_exposure_share_v63"].max() < 0.90
    assert concentration["required_cap_slack_share_v63"].max() < 0.01

    gate = _read_csv("paper4_v63_dynamic_gate_memo.csv")
    assert not gate["dynamic_512_or_1024_rerun_recommended_v63"].astype(bool).any()
    assert not gate["working_champion_change_allowed_v63"].astype(bool).any()

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has heuristic source-governance repair candidates after v59-v62." in set(
        current_boundaries["claim"]
    )
    assert "A v63 source-repaired candidate replaces Paper Estrella." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v63: Source-Governance Repair Frontier" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v64_online_pseudo_unseen_stress_is_lab_only() -> None:
    status = _read_json("paper4_v64_status.json")

    assert status["phase"] == "v64_online_pseudo_unseen_stress"
    assert status["stress_grid_rows_v64"] >= 2_000
    assert status["method_split_summary_rows_v64"] >= 500
    assert status["winner_rows_v64"] == 7
    assert status["cell_summary_rows_v64"] > 0
    assert status["local_split_gate_pass_rows_v64"] > 0
    assert status["strict_all_split_pass_rows_v64"] == 0
    assert status["external_unseen_data_available_v64"] is False
    assert status["strict_live_deployability_claim_allowed_v64"] is False
    assert status["paper1_promotion_allowed_v64"] is False
    assert status["paper4_working_champion_changed_v64"] is False
    assert status["paper4_final_promotion_created"] is False

    expected_csvs = {
        "paper4_v64_online_input_audit.csv": {
            "audit_item_v64",
            "value_v64",
            "claim_boundary_v64",
        },
        "paper4_v64_online_pseudo_unseen_stress_grid.csv": {
            "source_family",
            "pseudo_unseen_split_v64",
            "source_month_defended_min_v64",
            "policy_month_defended_min_v64",
            "avg_width_loan_v64",
            "gate_source80_policy90_width95_v64",
            "strict_live_deployability_claim_allowed",
            "claim_boundary_v64",
        },
        "paper4_v64_online_method_split_summary.csv": {
            "source_family",
            "split_gate_pass_rows_v64",
            "all_splits_gate_pass_v64",
            "width_margin_to_0p95_v64",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v64_online_winner_memo.csv": {
            "source_family",
            "recommendation_v64",
            "all_splits_gate_pass_v64",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v64_claim_matrix_delta.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    grid = _read_csv("paper4_v64_online_pseudo_unseen_stress_grid.csv")
    assert grid["gate_source80_policy90_width95_v64"].astype(bool).any()
    assert not grid["strict_live_deployability_claim_allowed"].astype(bool).any()
    assert not grid["external_unseen_data_available_v64"].astype(bool).any()

    summary = _read_csv("paper4_v64_online_method_split_summary.csv")
    assert not summary["all_splits_gate_pass_v64"].astype(bool).any()
    assert summary["split_gate_pass_rows_v64"].max() < summary["evaluated_splits_v64"].max()

    winners = _read_csv("paper4_v64_online_winner_memo.csv")
    assert len(winners) == 7
    assert not winners["all_splits_gate_pass_v64"].astype(bool).any()
    assert winners["split_gate_pass_rows_v64"].max() == 3
    assert winners["max_avg_width_loan_v64"].min() > 0.95
    assert not winners["strict_live_deployability_claim_allowed"].astype(bool).any()

    cells = pd.read_parquet(TABLE_DIR / "paper4_v64_online_stress_cell_summary.parquet")
    assert not cells.empty
    assert {
        "source_family",
        "pseudo_unseen_split_v64",
        "coverage_v64",
        "avg_width_v64",
        "claim_boundary_v64",
    }.issubset(cells.columns)

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has pseudo-unseen online conformal stress diagnostics." in set(
        current_boundaries["claim"]
    )
    assert "Online conformal coverage is strictly live-deployable after v64." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v64: Online Pseudo-Unseen Stress" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v65_online_margin_repair_keeps_live_claim_blocked() -> None:
    status = _read_json("paper4_v65_status.json")

    assert status["phase"] == "v65_online_margin_repair"
    assert status["margin_grid_rows_v65"] >= 1_000
    assert status["margin_summary_rows_v65"] >= 250
    assert status["winner_rows_v65"] == 2
    assert status["cell_summary_rows_v65"] > 0
    assert status["all_split_gate_pass_rows_v65"] > 0
    assert status["families_with_all_split_pass_v65"] == 2
    assert status["best_width_margin_to_0p95_v65"] > 0
    assert status["external_unseen_data_available_v65"] is False
    assert status["strict_live_deployability_claim_allowed_v65"] is False
    assert status["paper1_promotion_allowed_v65"] is False
    assert status["paper4_working_champion_changed_v65"] is False
    assert status["paper4_final_promotion_created"] is False

    expected_csvs = {
        "paper4_v65_online_margin_repair_input_audit.csv": {
            "audit_item_v64",
            "value_v64",
            "claim_boundary_v65",
        },
        "paper4_v65_online_margin_repair_grid.csv": {
            "source_family",
            "pseudo_unseen_split_v65",
            "global_delta_v65",
            "source_month_defended_min_v65",
            "policy_month_defended_min_v65",
            "avg_width_loan_v65",
            "gate_source80_policy90_width95_v65",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v65_online_margin_repair_summary.csv": {
            "source_family",
            "all_splits_gate_pass_v65",
            "source_margin_to_0p80_v65",
            "policy_margin_to_0p90_v65",
            "width_margin_to_0p95_v65",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v65_online_margin_repair_winners.csv": {
            "source_family",
            "global_delta_v65",
            "recommendation_v65",
            "all_splits_gate_pass_v65",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v65_claim_matrix_delta.csv": {
            "claim_id",
            "allowed",
            "artifact",
            "boundary",
        },
    }
    for name, columns in expected_csvs.items():
        table = _read_csv(name)
        assert not table.empty, name
        assert columns.issubset(table.columns), name

    summary = _read_csv("paper4_v65_online_margin_repair_summary.csv")
    passing = summary.loc[summary["all_splits_gate_pass_v65"].astype(bool)]
    assert not passing.empty
    assert set(passing["source_family"].unique()) == {"period", "term"}
    assert passing["source_margin_to_0p80_v65"].min() > 0
    assert passing["policy_margin_to_0p90_v65"].min() >= 0
    assert passing["width_margin_to_0p95_v65"].min() > 0
    assert not passing["strict_live_deployability_claim_allowed"].astype(bool).any()

    winners = _read_csv("paper4_v65_online_margin_repair_winners.csv")
    assert set(winners["source_family"]) == {"period", "term"}
    assert winners["all_splits_gate_pass_v65"].astype(bool).all()
    assert winners["global_delta_v65"].min() == pytest.approx(0.012)
    assert winners["width_margin_to_0p95_v65"].min() > 0
    assert not winners["strict_live_deployability_claim_allowed"].astype(bool).any()

    cells = pd.read_parquet(TABLE_DIR / "paper4_v65_online_margin_repair_cells.parquet")
    assert not cells.empty
    assert {
        "source_family",
        "pseudo_unseen_split_v65",
        "coverage_v64",
        "avg_width_v64",
        "claim_boundary_v65",
    }.issubset(cells.columns)

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert (
        "Paper 4 has an internal pseudo-unseen online margin repair that passes all historical splits."
        in set(current_boundaries["claim"])
    )
    assert "v65 margin repair proves live online deployability." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v65: Online Margin Repair" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_quarto_chapter_renders() -> None:
    if shutil.which("quarto") is None:
        pytest.skip("quarto CLI is not installed")

    env = os.environ.copy()
    env["QUARTO_PYTHON"] = str((Path.cwd() / ".venv/bin/python").absolute())
    subprocess.run(
        [
            "quarto",
            "render",
            "chapters/19-paper-mega-extension",
            "--to",
            "html",
            "--execute-daemon-restart",
        ],
        cwd=BOOK_DIR,
        env=env,
        check=True,
        timeout=1200,
    )
