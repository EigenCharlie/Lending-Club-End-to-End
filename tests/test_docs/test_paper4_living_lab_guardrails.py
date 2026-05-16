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


def test_paper4_v66_external_holdout_protocol_is_not_validation() -> None:
    status = _read_json("paper4_v66_status.json")

    assert status["phase"] == "v66_external_holdout_protocol"
    assert status["frozen_method_rows_v66"] == 2
    assert status["required_schema_rows_v66"] >= 10
    assert status["gate_spec_rows_v66"] >= 7
    assert status["protocol_rows_v66"] >= 6
    assert status["leakage_check_rows_v66"] >= 7
    assert status["method_frozen_for_future_holdout_v66"] is True
    assert status["external_holdout_data_available_v66"] is False
    assert status["strict_live_deployability_claim_allowed_v66"] is False
    assert status["paper1_promotion_allowed_v66"] is False
    assert status["paper4_working_champion_changed_v66"] is False
    assert status["paper4_final_promotion_created"] is False

    expected_csvs = {
        "paper4_v66_frozen_method_manifest.csv": {
            "frozen_method_id_v66",
            "source_family",
            "selection_artifact_sha256_v66",
            "allow_parameter_changes_before_external_holdout_v66",
            "strict_live_deployability_claim_allowed",
        },
        "paper4_v66_required_holdout_schema.csv": {
            "column_name_v66",
            "required_v66",
            "validation_rule_v66",
        },
        "paper4_v66_holdout_gate_spec.csv": {
            "gate_id_v66",
            "threshold_v66",
            "operator_v66",
            "gate_required_for_live_claim_v66",
        },
        "paper4_v66_external_holdout_protocol.csv": {
            "protocol_step_v66",
            "action_v66",
            "locked_instruction_v66",
            "editable_after_v66_freeze",
        },
        "paper4_v66_leakage_prevention_checklist.csv": {
            "leakage_check_id_v66",
            "rule_v66",
            "failure_action_v66",
        },
        "paper4_v66_claim_matrix_delta.csv": {
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

    manifest = _read_csv("paper4_v66_frozen_method_manifest.csv")
    assert set(manifest["source_family"]) == {"period", "term"}
    assert manifest["global_delta_v65"].eq(0.012).all()
    assert not manifest["allow_parameter_changes_before_external_holdout_v66"].astype(bool).any()
    assert not manifest["strict_live_deployability_claim_allowed"].astype(bool).any()
    assert manifest["selection_artifact_sha256_v66"].str.len().eq(64).all()
    assert manifest["script_sha256_v66"].str.len().eq(64).all()

    schema = _read_csv("paper4_v66_required_holdout_schema.csv")
    assert {"loan_id", "issue_month", "y_true", "y_pred", "qhat_v9", "period", "term"}.issubset(
        set(schema["column_name_v66"])
    )
    assert schema["required_v66"].astype(bool).all()

    gates = _read_csv("paper4_v66_holdout_gate_spec.csv")
    assert {
        "source_month_coverage_min",
        "policy_month_coverage_min",
        "avg_interval_width",
    }.issubset(set(gates["gate_id_v66"]))
    assert gates["gate_required_for_live_claim_v66"].astype(bool).all()

    protocol = _read_csv("paper4_v66_external_holdout_protocol.csv")
    assert not protocol["editable_after_v66_freeze"].astype(bool).any()

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a frozen external-holdout protocol for v65 online candidates." in set(
        current_boundaries["claim"]
    )
    assert "v66 protocol itself validates live online deployment." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v66: External Holdout Protocol Freeze" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v67_external_holdout_scorer_blocks_without_data() -> None:
    status = _read_json("paper4_v67_status.json")

    assert status["phase"] == "v67_external_holdout_scorer"
    assert status["readiness_rows_v67"] >= 4
    assert status["scorecard_rows_v67"] == 2
    assert status["holdout_data_available_v67"] is False
    assert status["passing_methods_v67"] == 0
    assert status["strict_live_deployability_claim_allowed_v67"] is False
    assert status["paper1_promotion_allowed_v67"] is False
    assert status["paper4_working_champion_changed_v67"] is False
    assert status["paper4_final_promotion_created"] is False

    expected_csvs = {
        "paper4_v67_scorer_readiness.csv": {
            "readiness_item_v67",
            "pass_v67",
            "detail_v67",
            "claim_boundary_v67",
        },
        "paper4_v67_external_holdout_scorecard.csv": {
            "frozen_method_id_v66",
            "source_family",
            "holdout_data_available_v67",
            "all_gates_pass_v67",
            "strict_live_deployability_claim_allowed",
            "score_status_v67",
            "claim_boundary_v67",
        },
        "paper4_v67_claim_matrix_delta.csv": {
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

    readiness = _read_csv("paper4_v67_scorer_readiness.csv")
    readiness_map = dict(zip(readiness["readiness_item_v67"], readiness["pass_v67"], strict=False))
    assert readiness_map["frozen_manifest_exists"] is True
    assert readiness_map["selection_hash_matches_manifest"] is True
    assert readiness_map["holdout_file_available"] is False
    assert readiness_map["holdout_schema_complete"] is False

    scorecard = _read_csv("paper4_v67_external_holdout_scorecard.csv")
    assert set(scorecard["source_family"]) == {"period", "term"}
    assert not scorecard["holdout_data_available_v67"].astype(bool).any()
    assert not scorecard["all_gates_pass_v67"].astype(bool).any()
    assert not scorecard["strict_live_deployability_claim_allowed"].astype(bool).any()
    assert set(scorecard["score_status_v67"]) == {"blocked_missing_external_holdout_data"}

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has an executable frozen scorer for the v66 holdout protocol." in set(
        current_boundaries["claim"]
    )
    assert "v67 validates live online deployment." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v67: External Holdout Scorer" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v68_full_universe_pricing_screen_blocks_exact_claims() -> None:
    status = _read_json("paper4_v68_status.json")

    assert status["phase"] == "v68_full_universe_source_pricing_screen"
    assert status["screened_policy_rows_v68"] == 4
    assert status["candidate_screen_rows_v68"] >= 200
    assert status["benchmark_rows_v68"] == 4
    assert status["source_relief_rows_v68"] >= 24
    assert status["policies_with_unpriced_columns_v68"] == 4
    assert status["exact_dual_pricing_performed_v68"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v68"] is False
    assert status["paper1_promotion_allowed_v68"] is False
    assert status["paper4_working_champion_changed_v68"] is False
    assert status["paper4_final_promotion_created"] is False

    candidates = pd.read_parquet(TABLE_DIR / "paper4_v68_full_universe_candidate_screen.parquet")
    assert not candidates.empty
    assert {
        "policy_id_v68",
        "loan_id",
        "pricing_screen_score_v68",
        "source_relief_share_v68",
        "screen_scope_v68",
        "candidate_rank_v68",
    }.issubset(candidates.columns)
    assert candidates["policy_id_v68"].nunique() == 4
    assert candidates.groupby("policy_id_v68")["candidate_rank_v68"].max().eq(50).all()
    assert candidates["screen_scope_v68"].str.contains("not exact dual pricing").all()

    benchmark = _read_csv("paper4_v68_screen_vs_book_benchmark.csv")
    assert {
        "policy_id",
        "screened_universe_rows_v68",
        "out_of_book_rows_v68",
        "screen_detects_unpriced_columns_v68",
        "exact_full_universe_cvar_claim_allowed_v68",
        "claim_boundary_v68",
    }.issubset(benchmark.columns)
    assert benchmark["screened_universe_rows_v68"].eq(276869).all()
    assert benchmark["screen_detects_unpriced_columns_v68"].astype(bool).all()
    assert not benchmark["exact_full_universe_cvar_claim_allowed_v68"].astype(bool).any()
    assert benchmark["claim_boundary_v68"].str.contains("proxy screen only").all()

    relief = _read_csv("paper4_v68_source_relief_summary.csv")
    assert {
        "policy_id",
        "source_family",
        "top_candidate_share_not_top_source_v68",
        "claim_boundary_v68",
    }.issubset(relief.columns)
    assert relief["source_family"].nunique() >= 6
    assert relief["top_candidate_share_not_top_source_v68"].between(0, 1).all()

    claim_delta = _read_csv("paper4_v68_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v68_full_universe_proxy_screen_exists"]) is True
    assert bool(claim_map["v68_exact_full_universe_cvar_optimality"]) is False
    assert bool(claim_map["v68_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a full-universe source/pricing screen for v63 repair books." in set(
        current_boundaries["claim"]
    )
    assert "v68 proves exact full-universe CVaR optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v68: Full-Universe Source/Pricing Screen" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v69_restricted_master_expansion_is_protocol_only() -> None:
    status = _read_json("paper4_v69_status.json")

    assert status["phase"] == "v69_restricted_master_expansion_protocol"
    assert status["candidate_rows_v69"] == 200
    assert status["expanded_master_rows_v69"] == 1824
    assert status["swap_audit_rows_v69"] == 100
    assert status["protocol_rows_v69"] == 7
    assert status["policies_ready_for_restricted_solver_v69"] == 4
    assert status["positive_swap_score_rows_v69"] == 100
    assert status["restricted_master_pack_ready_v69"] is True
    assert status["exact_column_generation_certificate_v69"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v69"] is False
    assert status["paper1_promotion_allowed_v69"] is False
    assert status["paper4_working_champion_changed_v69"] is False
    assert status["paper4_final_promotion_created"] is False

    candidates = pd.read_parquet(
        TABLE_DIR / "paper4_v69_source_pricing_expansion_candidates.parquet"
    )
    assert not candidates.empty
    assert {
        "policy_id_v69",
        "source_policy_id_v63",
        "candidate_rank_v68",
        "pricing_screen_score_v69",
        "recommended_for_restricted_master_v69",
        "claim_boundary_v69",
    }.issubset(candidates.columns)
    assert candidates["policy_id_v69"].nunique() == 4
    assert candidates.groupby("policy_id_v69")["candidate_rank_v68"].max().eq(50).all()
    assert candidates["recommended_for_restricted_master_v69"].astype(bool).all()

    master = pd.read_parquet(TABLE_DIR / "paper4_v69_expanded_restricted_master.parquet")
    assert not master.empty
    assert {
        "policy_id_v69",
        "master_role_v69",
        "pricing_screen_score_v69",
        "exact_column_generation_certificate_v69",
        "claim_boundary_v69",
    }.issubset(master.columns)
    assert set(master["master_role_v69"]) == {"incumbent_v63_book", "v68_pricing_candidate"}
    assert not master["exact_column_generation_certificate_v69"].astype(bool).any()
    assert master["claim_boundary_v69"].str.contains("not exact full-universe").all()

    swaps = _read_csv("paper4_v69_candidate_swap_audit.csv")
    assert {
        "policy_id",
        "swap_rank_v69",
        "add_loan_id_v69",
        "drop_loan_id_v69",
        "delta_pricing_score_v69",
        "valid_budget_after_swap_v69",
        "claim_boundary_v69",
    }.issubset(swaps.columns)
    assert swaps["policy_id"].nunique() == 4
    assert swaps.groupby("policy_id")["swap_rank_v69"].max().eq(25).all()
    assert swaps["delta_pricing_score_v69"].gt(0).all()
    assert swaps["valid_budget_after_swap_v69"].astype(bool).all()
    assert swaps["claim_boundary_v69"].str.contains("heuristic").all()

    protocol = _read_csv("paper4_v69_exact_column_generation_protocol.csv")
    assert {
        "protocol_step_v69",
        "step_name_v69",
        "artifact_v69",
        "required_evidence_v69",
        "claim_if_missing_v69",
        "locked_v69",
    }.issubset(protocol.columns)
    assert protocol["locked_v69"].astype(bool).all()
    assert {
        "solve_expanded_restricted_master",
        "persist_duals",
        "price_omitted_universe",
        "iterate_negative_reduced_cost_columns",
    }.issubset(set(protocol["step_name_v69"]))

    claim_delta = _read_csv("paper4_v69_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v69_restricted_master_expansion_pack_exists"]) is True
    assert bool(claim_map["v69_exact_column_generation_certificate"]) is False
    assert bool(claim_map["v69_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert (
        "Paper 4 has a v69 restricted-master expansion pack for column-generation follow-up."
        in set(current_boundaries["claim"])
    )
    assert "v69 is an exact full-universe column-generation certificate." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v69: Restricted-Master Expansion Pack" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v70_restricted_master_solver_keeps_full_universe_claim_blocked() -> None:
    status = _read_json("paper4_v70_status.json")

    assert status["phase"] == "v70_restricted_master_continuous_lp_solver"
    assert status["frontier_rows_v70"] == 8
    assert status["successful_lp_rows_v70"] == 8
    assert status["allocation_rows_v70"] > 0
    assert status["scenario_rows_v70"] == 1024
    assert status["active_constraint_rows_v70"] >= 1000
    assert status["policies_with_successful_lp_v70"] == 4
    assert status["best_delta_return_vs_incumbent_v70"] > 0
    assert status["exact_restricted_master_lp_claim_allowed_v70"] is True
    assert status["exact_full_universe_cvar_claim_allowed_v70"] is False
    assert status["paper1_promotion_allowed_v70"] is False
    assert status["paper4_working_champion_changed_v70"] is False
    assert status["paper4_final_promotion_created"] is False

    frontier = _read_csv("paper4_v70_restricted_master_solver_frontier.csv")
    assert {
        "policy_id",
        "regime_v70",
        "solver_success_v70",
        "delta_return_vs_incumbent_v70",
        "candidate_allocation_share_v70",
        "exact_restricted_master_lp_v70",
        "exact_full_universe_cvar_claim_allowed_v70",
        "claim_boundary_v70",
    }.issubset(frontier.columns)
    assert frontier["policy_id"].nunique() == 4
    assert set(frontier["regime_v70"]) == {
        "incumbent_cvar_relaxed_source_lp",
        "target_source_cap_probe_lp",
    }
    assert frontier["solver_success_v70"].astype(bool).all()
    assert frontier["delta_return_vs_incumbent_v70"].gt(0).all()
    assert frontier["candidate_allocation_share_v70"].gt(0).all()
    assert frontier["exact_restricted_master_lp_v70"].astype(bool).all()
    assert not frontier["exact_full_universe_cvar_claim_allowed_v70"].astype(bool).any()
    assert frontier["claim_boundary_v70"].str.contains("no omitted-column pricing").all()

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v70_restricted_master_allocations.parquet")
    assert not allocations.empty
    assert {
        "policy_id",
        "regime_v70",
        "loan_id",
        "master_role_v69",
        "allocation_fraction_v70",
        "allocated_exposure_v70",
        "claim_boundary_v70",
    }.issubset(allocations.columns)
    assert allocations["allocation_fraction_v70"].between(0, 1).all()
    assert "v68_pricing_candidate" in set(allocations["master_role_v69"])
    assert allocations["claim_boundary_v70"].str.contains("not full-universe").all()

    scenarios = _read_csv("paper4_v70_restricted_master_scenario_losses.csv")
    assert {
        "policy_id",
        "regime_v70",
        "path_id",
        "scenario_loss_v70",
        "scenario_return_v70",
    }.issubset(scenarios.columns)
    assert scenarios.groupby(["policy_id", "regime_v70"])["path_id"].nunique().eq(128).all()

    active = _read_csv("paper4_v70_solver_active_constraints.csv")
    assert {
        "policy_id",
        "regime_v70",
        "constraint_type_v70",
        "lhs_v70",
        "rhs_v70",
        "slack_v70",
        "binding_v70",
        "claim_boundary_v70",
    }.issubset(active.columns)
    assert {"budget_lower", "cvar_cap", "source_share", "cvar_path_excess"}.issubset(
        set(active["constraint_type_v70"])
    )

    blockers = _read_csv("paper4_v70_solver_claim_blockers.csv")
    assert {
        "blocker_id_v70",
        "status_v70",
        "required_next_artifact_v70",
        "claim_boundary_v70",
    }.issubset(blockers.columns)
    assert "omitted_column_reduced_costs_missing" in set(blockers["blocker_id_v70"])
    assert "continuous_relaxation_not_whole_loan_milp" in set(blockers["blocker_id_v70"])

    claim_delta = _read_csv("paper4_v70_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v70_exact_restricted_master_continuous_lp"]) is True
    assert bool(claim_map["v70_exact_full_universe_cvar_optimality"]) is False
    assert bool(claim_map["v70_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has exact continuous LP solves over the v69 restricted master." in set(
        current_boundaries["claim"]
    )
    assert "v70 proves exact full-universe CVaR optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v70: Restricted-Master Continuous LP Solver" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v71_reduced_cost_screen_blocks_column_generation_termination() -> None:
    status = _read_json("paper4_v71_status.json")

    assert status["phase"] == "v71_full_universe_reduced_cost_screen"
    assert status["dual_rows_v71"] == 1410
    assert status["reduced_cost_rows_v71"] == 2211304
    assert status["summary_rows_v71"] == 8
    assert status["source_cap_dual_rows_v71"] == 48
    assert status["claim_blocker_rows_v71"] == 3
    assert status["improving_omitted_columns_v71"] == 5738
    assert status["policies_priced_v71"] == 4
    assert status["regime_rows_priced_v71"] == 8
    assert status["negative_reduced_cost_detected_v71"] is True
    assert status["full_universe_termination_claim_allowed_v71"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v71"] is False
    assert status["paper1_promotion_allowed_v71"] is False
    assert status["paper4_working_champion_changed_v71"] is False
    assert status["paper4_final_promotion_created"] is False

    reduced_costs = pd.read_parquet(TABLE_DIR / "paper4_v71_full_universe_reduced_costs.parquet")
    assert not reduced_costs.empty
    assert {
        "policy_id",
        "regime_v71",
        "loan_id",
        "minimization_reduced_cost_v71",
        "return_improvement_signal_v71",
        "improving_column_v71",
        "pricing_scope_v71",
        "claim_boundary_v71",
    }.issubset(reduced_costs.columns)
    assert reduced_costs["policy_id"].nunique() == 4
    assert set(reduced_costs["regime_v71"]) == {
        "incumbent_cvar_relaxed_source_lp",
        "target_source_cap_probe_lp",
    }
    assert (
        int(reduced_costs["improving_column_v71"].sum()) == status["improving_omitted_columns_v71"]
    )
    assert reduced_costs["minimization_reduced_cost_v71"].min() < 0
    assert reduced_costs["claim_boundary_v71"].str.contains("not full-universe termination").all()

    summary = _read_csv("paper4_v71_reduced_cost_summary.csv")
    assert {
        "policy_id",
        "regime_v71",
        "omitted_rows_priced_v71",
        "improving_columns_v71",
        "negative_reduced_cost_detected_v71",
        "column_generation_termination_certificate_v71",
        "exact_full_universe_cvar_claim_allowed_v71",
    }.issubset(summary.columns)
    assert summary["omitted_rows_priced_v71"].sum() == status["reduced_cost_rows_v71"]
    assert summary["improving_columns_v71"].sum() == status["improving_omitted_columns_v71"]
    assert summary["negative_reduced_cost_detected_v71"].astype(bool).any()
    assert not summary["column_generation_termination_certificate_v71"].astype(bool).any()
    assert not summary["exact_full_universe_cvar_claim_allowed_v71"].astype(bool).any()

    duals = _read_csv("paper4_v71_restricted_master_duals.csv")
    assert {
        "policy_id",
        "regime_v71",
        "constraint_index_v71",
        "constraint_type_v71",
        "marginal_v71",
        "binding_v71",
        "claim_boundary_v71",
    }.issubset(duals.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(duals["constraint_type_v71"])
    )
    assert duals["marginal_v71"].abs().gt(0).any()
    assert duals["claim_boundary_v71"].str.contains("not full-universe certificate").all()

    source_diag = _read_csv("paper4_v71_source_cap_dual_diagnostics.csv")
    assert {
        "policy_id",
        "regime_v71",
        "source_family",
        "missing_source_ids_v71",
        "source_constraint_scope_complete_v71",
        "claim_boundary_v71",
    }.issubset(source_diag.columns)
    assert source_diag["missing_source_ids_v71"].sum() > 0
    assert not source_diag["source_constraint_scope_complete_v71"].astype(bool).all()

    blockers = _read_csv("paper4_v71_claim_blockers.csv")
    assert {"blocker_id_v71", "blocking_v71", "evidence_count_v71", "claim_boundary_v71"}.issubset(
        blockers.columns
    )
    assert blockers["blocking_v71"].astype(bool).all()
    assert {
        "negative_reduced_cost_columns_detected",
        "source_constraint_scope_not_full_universe",
        "continuous_relaxation_not_whole_loan_milp",
    }.issubset(set(blockers["blocker_id_v71"]))

    claim_delta = _read_csv("paper4_v71_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v71_omitted_column_reduced_cost_screen"]) is True
    assert bool(claim_map["v71_full_universe_column_generation_termination"]) is False
    assert bool(claim_map["v71_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has v71 reduced-cost pricing for omitted v55 columns under v70 duals." in set(
        current_boundaries["claim"]
    )
    assert "v71 proves full-universe column-generation termination." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v71: Full-Universe Reduced-Cost Screen" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v72_column_generation_iteration_requires_repricing() -> None:
    status = _read_json("paper4_v72_status.json")

    assert status["phase"] == "v72_column_generation_iteration_1"
    assert status["candidate_rows_v72"] == 5738
    assert status["frontier_rows_v72"] == 5
    assert status["successful_iteration_rows_v72"] == 5
    assert status["allocation_rows_v72"] > 0
    assert status["scenario_rows_v72"] == 640
    assert status["active_constraint_rows_v72"] == 880
    assert status["comparison_rows_v72"] == 5
    assert status["best_delta_return_vs_v70_iteration_v72"] > 0
    assert status["v71_candidate_allocated_exposure_v72"] > 0
    assert status["reprice_after_iteration_performed_v72"] is False
    assert status["column_generation_termination_claim_allowed_v72"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v72"] is False
    assert status["paper1_promotion_allowed_v72"] is False
    assert status["paper4_working_champion_changed_v72"] is False
    assert status["paper4_final_promotion_created"] is False

    candidates = pd.read_parquet(TABLE_DIR / "paper4_v72_iteration_1_candidates.parquet")
    assert not candidates.empty
    assert {
        "policy_id",
        "regime_v71",
        "loan_id",
        "minimization_reduced_cost_v71",
        "master_role_v69",
        "claim_boundary_v69",
    }.issubset(candidates.columns)
    assert len(candidates) == status["candidate_rows_v72"]
    assert candidates["minimization_reduced_cost_v71"].lt(0).all()
    assert set(candidates["master_role_v69"]) == {"v71_negative_reduced_cost_column"}

    frontier = _read_csv("paper4_v72_iteration_1_frontier.csv")
    assert {
        "policy_id",
        "regime_v72",
        "solver_success_v72",
        "negative_reduced_cost_candidates_added_v72",
        "delta_return_vs_v70_iteration_v72",
        "v71_candidate_allocated_exposure_v72",
        "reprice_after_iteration_performed_v72",
        "column_generation_termination_claim_allowed_v72",
        "exact_full_universe_cvar_claim_allowed_v72",
        "claim_boundary_v72",
    }.issubset(frontier.columns)
    assert frontier["solver_success_v72"].astype(bool).all()
    assert (
        frontier["negative_reduced_cost_candidates_added_v72"].sum() == status["candidate_rows_v72"]
    )
    assert frontier["delta_return_vs_v70_iteration_v72"].gt(0).all()
    assert frontier["v71_candidate_allocated_exposure_v72"].gt(0).all()
    assert not frontier["reprice_after_iteration_performed_v72"].astype(bool).any()
    assert not frontier["column_generation_termination_claim_allowed_v72"].astype(bool).any()
    assert not frontier["exact_full_universe_cvar_claim_allowed_v72"].astype(bool).any()
    assert frontier["claim_boundary_v72"].str.contains("re-pricing still required").all()

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v72_iteration_1_allocations.parquet")
    assert not allocations.empty
    assert {
        "policy_id",
        "regime_v72",
        "loan_id",
        "master_role_v72",
        "allocation_fraction_v72",
        "allocated_exposure_v72",
        "claim_boundary_v72",
    }.issubset(allocations.columns)
    assert allocations["allocation_fraction_v72"].between(0, 1).all()
    assert "v71_negative_reduced_cost_column" in set(allocations["master_role_v72"])
    assert allocations["claim_boundary_v72"].str.contains("re-pricing still required").all()

    scenarios = _read_csv("paper4_v72_iteration_1_scenario_losses.csv")
    assert {
        "policy_id",
        "regime_v72",
        "path_id",
        "scenario_loss_v72",
        "scenario_return_v72",
    }.issubset(scenarios.columns)
    assert scenarios.groupby(["policy_id", "regime_v72"])["path_id"].nunique().eq(128).all()

    active = _read_csv("paper4_v72_iteration_1_active_constraints.csv")
    assert {
        "policy_id",
        "regime_v72",
        "constraint_type_v72",
        "lhs_v72",
        "rhs_v72",
        "slack_v72",
        "binding_v72",
    }.issubset(active.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(active["constraint_type_v72"])
    )

    comparison = _read_csv("paper4_v72_iteration_1_comparison.csv")
    assert {
        "policy_id",
        "regime_v72",
        "delta_return_vs_v70_iteration_v72",
        "delta_cvar90_vs_v70_iteration_v72",
        "column_generation_termination_claim_allowed_v72",
    }.issubset(comparison.columns)
    assert comparison["delta_return_vs_v70_iteration_v72"].gt(0).all()
    assert comparison["delta_cvar90_vs_v70_iteration_v72"].lt(0).all()
    assert not comparison["column_generation_termination_claim_allowed_v72"].astype(bool).any()

    blockers = _read_csv("paper4_v72_claim_blockers.csv")
    assert {
        "blocker_id_v72",
        "blocking_v72",
        "required_next_artifact_v72",
        "claim_boundary_v72",
    }.issubset(blockers.columns)
    assert blockers["blocking_v72"].astype(bool).all()
    assert {
        "post_iteration_repricing_missing",
        "source_constraint_scope_needs_reaudit",
        "continuous_relaxation_not_whole_loan_milp",
    }.issubset(set(blockers["blocker_id_v72"]))

    claim_delta = _read_csv("paper4_v72_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v72_column_generation_iteration_1_completed"]) is True
    assert bool(claim_map["v72_column_generation_converged"]) is False
    assert bool(claim_map["v72_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v72 first column-generation iteration over negative v71 columns." in set(
        current_boundaries["claim"]
    )
    assert "v72 proves column-generation convergence." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v72: Column-Generation Iteration 1" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v73_post_iteration_reprice_blocks_convergence() -> None:
    status = _read_json("paper4_v73_status.json")

    assert status["phase"] == "v73_reprice_after_column_generation_iteration_1"
    assert status["reprice_rows_v73"] == 1376398
    assert status["summary_rows_v73"] == 5
    assert status["dual_rows_v73"] == 880
    assert status["source_scope_rows_v73"] == 30
    assert status["claim_blocker_rows_v73"] == 3
    assert status["improving_columns_after_iteration_v73"] == 83328
    assert status["negative_reduced_cost_detected_v73"] is True
    assert status["source_scope_missing_ids_v73"] == 30
    assert status["post_iteration_reprice_performed_v73"] is True
    assert status["column_generation_termination_claim_allowed_v73"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v73"] is False
    assert status["paper1_promotion_allowed_v73"] is False
    assert status["paper4_working_champion_changed_v73"] is False
    assert status["paper4_final_promotion_created"] is False

    repriced = pd.read_parquet(TABLE_DIR / "paper4_v73_reprice_after_iteration_1.parquet")
    assert not repriced.empty
    assert {
        "policy_id",
        "regime_v73",
        "loan_id",
        "minimization_reduced_cost_v73",
        "return_improvement_signal_v73",
        "improving_column_v73",
        "post_iteration_reprice_v73",
        "claim_boundary_v73",
    }.issubset(repriced.columns)
    assert repriced["policy_id"].nunique() == 4
    assert (
        int(repriced["improving_column_v73"].sum())
        == status["improving_columns_after_iteration_v73"]
    )
    assert repriced["minimization_reduced_cost_v73"].min() < 0
    assert repriced["post_iteration_reprice_v73"].eq(1).all()
    assert repriced["claim_boundary_v73"].str.contains("termination allowed only").all()

    summary = _read_csv("paper4_v73_reprice_summary.csv")
    assert {
        "policy_id",
        "regime_v73",
        "omitted_rows_priced_v73",
        "improving_columns_v73",
        "negative_reduced_cost_detected_v73",
        "post_iteration_reprice_performed_v73",
        "column_generation_termination_claim_allowed_v73",
        "exact_full_universe_cvar_claim_allowed_v73",
    }.issubset(summary.columns)
    assert summary["omitted_rows_priced_v73"].sum() == status["reprice_rows_v73"]
    assert summary["improving_columns_v73"].sum() == status["improving_columns_after_iteration_v73"]
    assert summary["negative_reduced_cost_detected_v73"].astype(bool).all()
    assert summary["post_iteration_reprice_performed_v73"].astype(bool).all()
    assert not summary["column_generation_termination_claim_allowed_v73"].astype(bool).any()
    assert not summary["exact_full_universe_cvar_claim_allowed_v73"].astype(bool).any()

    duals = _read_csv("paper4_v73_restricted_master_duals.csv")
    assert {
        "policy_id",
        "regime_v73",
        "constraint_index_v73",
        "constraint_type_v73",
        "marginal_v73",
        "binding_v73",
        "post_iteration_reprice_v73",
        "claim_boundary_v73",
    }.issubset(duals.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(duals["constraint_type_v73"])
    )
    assert duals["marginal_v73"].abs().gt(0).any()
    assert duals["post_iteration_reprice_v73"].eq(1).all()
    assert duals["claim_boundary_v73"].str.contains("not full-universe certificate").all()

    source_scope = _read_csv("paper4_v73_source_scope_after_iteration.csv")
    assert {
        "policy_id",
        "regime_v73",
        "source_family",
        "missing_source_ids_v73",
        "source_constraint_scope_complete_v73",
        "claim_boundary_v73",
    }.issubset(source_scope.columns)
    assert source_scope["missing_source_ids_v73"].sum() == status["source_scope_missing_ids_v73"]
    assert not source_scope["source_constraint_scope_complete_v73"].astype(bool).all()

    blockers = _read_csv("paper4_v73_claim_blockers.csv")
    assert {"blocker_id_v73", "blocking_v73", "evidence_count_v73", "claim_boundary_v73"}.issubset(
        blockers.columns
    )
    assert blockers["blocking_v73"].astype(bool).all()
    assert {
        "negative_reduced_cost_columns_after_iteration_1",
        "source_scope_after_iteration_incomplete",
        "continuous_relaxation_not_whole_loan_milp",
    }.issubset(set(blockers["blocker_id_v73"]))

    claim_delta = _read_csv("paper4_v73_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v73_post_iteration_reprice_executed"]) is True
    assert bool(claim_map["v73_column_generation_converged"]) is False
    assert bool(claim_map["v73_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has v73 post-iteration re-pricing after v72 column generation." in set(
        current_boundaries["claim"]
    )
    assert "v73 proves column-generation convergence." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v73: Re-Price After Column-Generation Iteration 1" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v74_column_generation_iteration_2_requires_repricing() -> None:
    status = _read_json("paper4_v74_status.json")

    assert status["phase"] == "v74_column_generation_iteration_2"
    assert status["candidate_rows_v74"] == 83328
    assert status["frontier_rows_v74"] == 5
    assert status["successful_iteration_rows_v74"] == 5
    assert status["allocation_rows_v74"] == 1091
    assert status["scenario_rows_v74"] == 640
    assert status["active_constraint_rows_v74"] == 880
    assert status["comparison_rows_v74"] == 5
    assert status["best_delta_return_vs_v72_iteration_v74"] > 800
    assert status["v73_candidate_allocated_exposure_v74"] > 180000
    assert status["v71_previous_candidate_allocated_exposure_v74"] > 2700000
    assert status["reprice_after_iteration_performed_v74"] is False
    assert status["column_generation_termination_claim_allowed_v74"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v74"] is False
    assert status["paper1_promotion_allowed_v74"] is False
    assert status["paper4_working_champion_changed_v74"] is False
    assert status["paper4_final_promotion_created"] is False

    candidates = pd.read_parquet(TABLE_DIR / "paper4_v74_iteration_2_candidates.parquet")
    assert not candidates.empty
    assert {
        "policy_id",
        "regime_v73",
        "loan_id",
        "minimization_reduced_cost_v73",
        "return_improvement_signal_v73",
        "master_role_v69",
        "claim_boundary_v69",
    }.issubset(candidates.columns)
    assert len(candidates) == status["candidate_rows_v74"]
    assert candidates["minimization_reduced_cost_v73"].lt(0).all()
    assert set(candidates["master_role_v69"]) == {"v73_negative_reduced_cost_column"}
    assert candidates.groupby(["policy_id", "regime_v73"]).size().sum() == 83328

    frontier = _read_csv("paper4_v74_iteration_2_frontier.csv")
    assert {
        "policy_id",
        "regime_v74",
        "solver_success_v74",
        "negative_reduced_cost_candidates_added_v74",
        "delta_return_vs_v72_iteration_v74",
        "v73_candidate_allocated_exposure_v74",
        "v71_previous_candidate_allocated_exposure_v74",
        "reprice_after_iteration_performed_v74",
        "column_generation_termination_claim_allowed_v74",
        "exact_full_universe_cvar_claim_allowed_v74",
        "claim_boundary_v74",
    }.issubset(frontier.columns)
    assert frontier["solver_success_v74"].astype(bool).all()
    assert (
        frontier["negative_reduced_cost_candidates_added_v74"].sum() == status["candidate_rows_v74"]
    )
    assert frontier["delta_return_vs_v72_iteration_v74"].max() == pytest.approx(
        status["best_delta_return_vs_v72_iteration_v74"]
    )
    assert frontier["delta_return_vs_v72_iteration_v74"].max() > 800
    assert frontier["v73_candidate_allocated_exposure_v74"].sum() == pytest.approx(
        status["v73_candidate_allocated_exposure_v74"]
    )
    assert frontier["v73_candidate_allocated_exposure_v74"].sum() > 180000
    assert frontier["v71_previous_candidate_allocated_exposure_v74"].sum() > 2700000
    assert not frontier["reprice_after_iteration_performed_v74"].astype(bool).any()
    assert not frontier["column_generation_termination_claim_allowed_v74"].astype(bool).any()
    assert not frontier["exact_full_universe_cvar_claim_allowed_v74"].astype(bool).any()
    assert frontier["claim_boundary_v74"].str.contains("re-pricing still required").all()

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v74_iteration_2_allocations.parquet")
    assert not allocations.empty
    assert {
        "policy_id",
        "regime_v74",
        "loan_id",
        "master_role_v74",
        "allocation_fraction_v74",
        "allocated_exposure_v74",
        "claim_boundary_v74",
    }.issubset(allocations.columns)
    assert allocations["allocation_fraction_v74"].between(0, 1).all()
    assert {
        "v71_negative_reduced_cost_column",
        "v73_negative_reduced_cost_column",
    }.issubset(set(allocations["master_role_v74"]))
    assert allocations["claim_boundary_v74"].str.contains("re-pricing still required").all()

    scenarios = _read_csv("paper4_v74_iteration_2_scenario_losses.csv")
    assert {
        "policy_id",
        "regime_v74",
        "path_id",
        "scenario_loss_v74",
        "scenario_return_v74",
    }.issubset(scenarios.columns)
    assert scenarios.groupby(["policy_id", "regime_v74"])["path_id"].nunique().eq(128).all()

    active = _read_csv("paper4_v74_iteration_2_active_constraints.csv")
    assert {
        "policy_id",
        "regime_v74",
        "constraint_type_v74",
        "lhs_v74",
        "rhs_v74",
        "slack_v74",
        "binding_v74",
    }.issubset(active.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(active["constraint_type_v74"])
    )

    comparison = _read_csv("paper4_v74_iteration_2_comparison.csv")
    assert {
        "policy_id",
        "regime_v74",
        "delta_return_vs_v72_iteration_v74",
        "delta_cvar90_vs_v72_iteration_v74",
        "v73_candidate_allocated_exposure_v74",
        "column_generation_termination_claim_allowed_v74",
    }.issubset(comparison.columns)
    assert comparison["delta_return_vs_v72_iteration_v74"].max() > 800
    assert comparison["v73_candidate_allocated_exposure_v74"].sum() > 180000
    assert not comparison["column_generation_termination_claim_allowed_v74"].astype(bool).any()

    blockers = _read_csv("paper4_v74_claim_blockers.csv")
    assert {
        "blocker_id_v74",
        "blocking_v74",
        "required_next_artifact_v74",
        "claim_boundary_v74",
    }.issubset(blockers.columns)
    assert blockers["blocking_v74"].astype(bool).all()
    assert {
        "post_iteration_2_repricing_missing",
        "source_constraint_scope_needs_reaudit_after_iteration_2",
        "continuous_relaxation_not_whole_loan_milp",
    }.issubset(set(blockers["blocker_id_v74"]))

    claim_delta = _read_csv("paper4_v74_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v74_column_generation_iteration_2_completed"]) is True
    assert bool(claim_map["v74_column_generation_converged"]) is False
    assert bool(claim_map["v74_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v74 second column-generation iteration over negative v73 columns." in set(
        current_boundaries["claim"]
    )
    assert "v74 proves column-generation convergence." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v74: Column-Generation Iteration 2" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v75_post_iteration_2_reprice_blocks_convergence() -> None:
    status = _read_json("paper4_v75_status.json")
    v73_status = _read_json("paper4_v73_status.json")

    assert status["phase"] == "v75_reprice_after_column_generation_iteration_2"
    assert status["reprice_rows_v75"] == 1293070
    assert status["summary_rows_v75"] == 5
    assert status["dual_rows_v75"] == 880
    assert status["source_scope_rows_v75"] == 30
    assert status["claim_blocker_rows_v75"] == 3
    assert status["improving_columns_after_iteration_2_v75"] == 3897
    assert (
        status["improving_columns_after_iteration_2_v75"]
        < v73_status["improving_columns_after_iteration_v73"]
    )
    assert status["negative_reduced_cost_detected_v75"] is True
    assert status["source_scope_missing_ids_v75"] == 30
    assert status["post_iteration_reprice_performed_v75"] is True
    assert status["column_generation_termination_claim_allowed_v75"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v75"] is False
    assert status["paper1_promotion_allowed_v75"] is False
    assert status["paper4_working_champion_changed_v75"] is False
    assert status["paper4_final_promotion_created"] is False

    repriced = pd.read_parquet(TABLE_DIR / "paper4_v75_reprice_after_iteration_2.parquet")
    assert not repriced.empty
    assert {
        "policy_id",
        "regime_v75",
        "loan_id",
        "minimization_reduced_cost_v75",
        "return_improvement_signal_v75",
        "improving_column_v75",
        "post_iteration_reprice_v75",
        "claim_boundary_v75",
    }.issubset(repriced.columns)
    assert repriced["policy_id"].nunique() == 4
    assert (
        int(repriced["improving_column_v75"].sum())
        == status["improving_columns_after_iteration_2_v75"]
    )
    assert repriced["minimization_reduced_cost_v75"].min() < 0
    assert repriced["post_iteration_reprice_v75"].eq(1).all()
    assert repriced["claim_boundary_v75"].str.contains("termination allowed only").all()

    summary = _read_csv("paper4_v75_reprice_summary.csv")
    assert {
        "policy_id",
        "regime_v75",
        "omitted_rows_priced_v75",
        "improving_columns_v75",
        "negative_reduced_cost_detected_v75",
        "post_iteration_reprice_performed_v75",
        "column_generation_termination_claim_allowed_v75",
        "exact_full_universe_cvar_claim_allowed_v75",
    }.issubset(summary.columns)
    assert summary["omitted_rows_priced_v75"].sum() == status["reprice_rows_v75"]
    assert (
        summary["improving_columns_v75"].sum() == status["improving_columns_after_iteration_2_v75"]
    )
    assert summary["improving_columns_v75"].gt(0).sum() == 1
    assert summary["negative_reduced_cost_detected_v75"].astype(bool).sum() == 1
    assert summary["post_iteration_reprice_performed_v75"].astype(bool).all()
    assert not summary["column_generation_termination_claim_allowed_v75"].astype(bool).any()
    assert not summary["exact_full_universe_cvar_claim_allowed_v75"].astype(bool).any()

    duals = _read_csv("paper4_v75_restricted_master_duals.csv")
    assert {
        "policy_id",
        "regime_v75",
        "constraint_index_v75",
        "constraint_type_v75",
        "marginal_v75",
        "binding_v75",
        "post_iteration_reprice_v75",
        "claim_boundary_v75",
    }.issubset(duals.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(duals["constraint_type_v75"])
    )
    assert duals["marginal_v75"].abs().gt(0).any()
    assert duals["post_iteration_reprice_v75"].eq(1).all()
    assert duals["claim_boundary_v75"].str.contains("not full-universe certificate").all()

    source_scope = _read_csv("paper4_v75_source_scope_after_iteration_2.csv")
    assert {
        "policy_id",
        "regime_v75",
        "source_family",
        "missing_source_ids_v75",
        "source_constraint_scope_complete_v75",
        "claim_boundary_v75",
    }.issubset(source_scope.columns)
    assert source_scope["missing_source_ids_v75"].sum() == status["source_scope_missing_ids_v75"]
    assert not source_scope["source_constraint_scope_complete_v75"].astype(bool).all()

    blockers = _read_csv("paper4_v75_claim_blockers.csv")
    assert {"blocker_id_v75", "blocking_v75", "evidence_count_v75", "claim_boundary_v75"}.issubset(
        blockers.columns
    )
    assert blockers["blocking_v75"].astype(bool).all()
    assert {
        "negative_reduced_cost_columns_after_iteration_2",
        "source_scope_after_iteration_2_incomplete",
        "continuous_relaxation_not_whole_loan_milp",
    }.issubset(set(blockers["blocker_id_v75"]))

    claim_delta = _read_csv("paper4_v75_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v75_post_iteration_2_reprice_executed"]) is True
    assert bool(claim_map["v75_column_generation_converged"]) is False
    assert bool(claim_map["v75_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has v75 post-iteration-2 re-pricing after v74 column generation." in set(
        current_boundaries["claim"]
    )
    assert "v75 proves column-generation convergence." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v75: Re-Price After Column-Generation Iteration 2" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v76_column_generation_iteration_3_requires_repricing() -> None:
    status = _read_json("paper4_v76_status.json")

    assert status["phase"] == "v76_column_generation_iteration_3"
    assert status["candidate_rows_v76"] == 3897
    assert status["frontier_rows_v76"] == 1
    assert status["successful_iteration_rows_v76"] == 1
    assert status["allocation_rows_v76"] == 174
    assert status["scenario_rows_v76"] == 128
    assert status["active_constraint_rows_v76"] == 175
    assert status["comparison_rows_v76"] == 1
    assert abs(status["best_delta_return_vs_v74_iteration_v76"]) < 1e-6
    assert status["v75_candidate_allocated_exposure_v76"] == 0
    assert status["v73_previous_candidate_allocated_exposure_v76"] > 180000
    assert status["v71_previous_candidate_allocated_exposure_v76"] > 490000
    assert status["reprice_after_iteration_performed_v76"] is False
    assert status["column_generation_termination_claim_allowed_v76"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v76"] is False
    assert status["paper1_promotion_allowed_v76"] is False
    assert status["paper4_working_champion_changed_v76"] is False
    assert status["paper4_final_promotion_created"] is False

    candidates = pd.read_parquet(TABLE_DIR / "paper4_v76_iteration_3_candidates.parquet")
    assert not candidates.empty
    assert {
        "policy_id",
        "regime_v75",
        "loan_id",
        "minimization_reduced_cost_v75",
        "return_improvement_signal_v75",
        "master_role_v69",
        "claim_boundary_v69",
    }.issubset(candidates.columns)
    assert len(candidates) == status["candidate_rows_v76"]
    assert candidates["minimization_reduced_cost_v75"].lt(0).all()
    assert set(candidates["master_role_v69"]) == {"v75_negative_reduced_cost_column"}
    assert set(candidates["regime_v75"]) == {"incumbent_cvar_relaxed_source_lp"}

    frontier = _read_csv("paper4_v76_iteration_3_frontier.csv")
    assert {
        "policy_id",
        "regime_v76",
        "solver_success_v76",
        "negative_reduced_cost_candidates_added_v76",
        "delta_return_vs_v74_iteration_v76",
        "v75_candidate_allocated_exposure_v76",
        "v73_previous_candidate_allocated_exposure_v76",
        "v71_previous_candidate_allocated_exposure_v76",
        "reprice_after_iteration_performed_v76",
        "column_generation_termination_claim_allowed_v76",
        "exact_full_universe_cvar_claim_allowed_v76",
        "claim_boundary_v76",
    }.issubset(frontier.columns)
    assert frontier["solver_success_v76"].astype(bool).all()
    assert (
        frontier["negative_reduced_cost_candidates_added_v76"].sum() == status["candidate_rows_v76"]
    )
    assert frontier["delta_return_vs_v74_iteration_v76"].abs().max() < 1e-6
    assert frontier["v75_candidate_allocated_exposure_v76"].sum() == 0
    assert frontier["v73_previous_candidate_allocated_exposure_v76"].sum() > 180000
    assert frontier["v71_previous_candidate_allocated_exposure_v76"].sum() > 490000
    assert not frontier["reprice_after_iteration_performed_v76"].astype(bool).any()
    assert not frontier["column_generation_termination_claim_allowed_v76"].astype(bool).any()
    assert not frontier["exact_full_universe_cvar_claim_allowed_v76"].astype(bool).any()
    assert frontier["claim_boundary_v76"].str.contains("re-pricing still required").all()

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v76_iteration_3_allocations.parquet")
    assert not allocations.empty
    assert {
        "policy_id",
        "regime_v76",
        "loan_id",
        "master_role_v76",
        "allocation_fraction_v76",
        "allocated_exposure_v76",
        "claim_boundary_v76",
    }.issubset(allocations.columns)
    assert allocations["allocation_fraction_v76"].between(0, 1).all()
    assert "v75_negative_reduced_cost_column" not in set(allocations["master_role_v76"])
    assert {
        "v71_negative_reduced_cost_column",
        "v73_negative_reduced_cost_column",
    }.issubset(set(allocations["master_role_v76"]))
    assert allocations["claim_boundary_v76"].str.contains("re-pricing still required").all()

    scenarios = _read_csv("paper4_v76_iteration_3_scenario_losses.csv")
    assert {
        "policy_id",
        "regime_v76",
        "path_id",
        "scenario_loss_v76",
        "scenario_return_v76",
    }.issubset(scenarios.columns)
    assert scenarios.groupby(["policy_id", "regime_v76"])["path_id"].nunique().eq(128).all()

    active = _read_csv("paper4_v76_iteration_3_active_constraints.csv")
    assert {
        "policy_id",
        "regime_v76",
        "constraint_type_v76",
        "lhs_v76",
        "rhs_v76",
        "slack_v76",
        "binding_v76",
    }.issubset(active.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(active["constraint_type_v76"])
    )

    comparison = _read_csv("paper4_v76_iteration_3_comparison.csv")
    assert {
        "policy_id",
        "regime_v76",
        "delta_return_vs_v74_iteration_v76",
        "delta_cvar90_vs_v74_iteration_v76",
        "v75_candidate_allocated_exposure_v76",
        "column_generation_termination_claim_allowed_v76",
    }.issubset(comparison.columns)
    assert comparison["delta_return_vs_v74_iteration_v76"].abs().max() < 1e-6
    assert comparison["v75_candidate_allocated_exposure_v76"].sum() == 0
    assert not comparison["column_generation_termination_claim_allowed_v76"].astype(bool).any()

    blockers = _read_csv("paper4_v76_claim_blockers.csv")
    assert {
        "blocker_id_v76",
        "blocking_v76",
        "required_next_artifact_v76",
        "claim_boundary_v76",
    }.issubset(blockers.columns)
    assert blockers["blocking_v76"].astype(bool).all()
    assert {
        "post_iteration_3_repricing_missing",
        "source_constraint_scope_needs_reaudit_after_iteration_3",
        "continuous_relaxation_not_whole_loan_milp",
    }.issubset(set(blockers["blocker_id_v76"]))

    claim_delta = _read_csv("paper4_v76_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v76_column_generation_iteration_3_completed"]) is True
    assert bool(claim_map["v76_column_generation_converged"]) is False
    assert bool(claim_map["v76_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert (
        "Paper 4 has a v76 third column-generation iteration over remaining negative v75 columns."
        in set(current_boundaries["claim"])
    )
    assert "v76 proves column-generation convergence." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v76: Column-Generation Iteration 3" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v77_post_iteration_3_reprice_clears_pricing_only() -> None:
    status = _read_json("paper4_v77_status.json")

    assert status["phase"] == "v77_reprice_after_column_generation_iteration_3"
    assert status["reprice_rows_v77"] == 257954
    assert status["summary_rows_v77"] == 1
    assert status["dual_rows_v77"] == 175
    assert status["source_scope_rows_v77"] == 6
    assert status["claim_blocker_rows_v77"] == 3
    assert status["improving_columns_after_iteration_3_v77"] == 0
    assert status["negative_reduced_cost_detected_v77"] is False
    assert status["pricing_blocker_cleared_v77"] is True
    assert status["source_scope_missing_ids_v77"] == 7
    assert status["post_iteration_reprice_performed_v77"] is True
    assert status["column_generation_termination_claim_allowed_v77"] is False
    assert status["exact_full_universe_cvar_claim_allowed_v77"] is False
    assert status["paper1_promotion_allowed_v77"] is False
    assert status["paper4_working_champion_changed_v77"] is False
    assert status["paper4_final_promotion_created"] is False

    repriced = pd.read_parquet(TABLE_DIR / "paper4_v77_reprice_after_iteration_3.parquet")
    assert not repriced.empty
    assert {
        "policy_id",
        "regime_v77",
        "loan_id",
        "minimization_reduced_cost_v77",
        "return_improvement_signal_v77",
        "improving_column_v77",
        "post_iteration_reprice_v77",
        "claim_boundary_v77",
    }.issubset(repriced.columns)
    assert len(repriced) == status["reprice_rows_v77"]
    assert repriced["policy_id"].nunique() == 1
    assert repriced["regime_v77"].nunique() == 1
    assert int(repriced["improving_column_v77"].sum()) == 0
    assert repriced["minimization_reduced_cost_v77"].min() > 0
    assert repriced["post_iteration_reprice_v77"].eq(1).all()
    assert repriced["claim_boundary_v77"].str.contains("source and integrality blockers").all()

    summary = _read_csv("paper4_v77_reprice_summary.csv")
    assert {
        "policy_id",
        "regime_v77",
        "omitted_rows_priced_v77",
        "improving_columns_v77",
        "negative_reduced_cost_detected_v77",
        "post_iteration_reprice_performed_v77",
        "column_generation_termination_claim_allowed_v77",
        "exact_full_universe_cvar_claim_allowed_v77",
    }.issubset(summary.columns)
    assert summary["omitted_rows_priced_v77"].sum() == status["reprice_rows_v77"]
    assert summary["improving_columns_v77"].sum() == 0
    assert not summary["negative_reduced_cost_detected_v77"].astype(bool).any()
    assert summary["post_iteration_reprice_performed_v77"].astype(bool).all()
    assert not summary["column_generation_termination_claim_allowed_v77"].astype(bool).any()
    assert not summary["exact_full_universe_cvar_claim_allowed_v77"].astype(bool).any()

    duals = _read_csv("paper4_v77_restricted_master_duals.csv")
    assert {
        "policy_id",
        "regime_v77",
        "constraint_index_v77",
        "constraint_type_v77",
        "marginal_v77",
        "binding_v77",
        "post_iteration_reprice_v77",
        "claim_boundary_v77",
    }.issubset(duals.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(duals["constraint_type_v77"])
    )
    assert duals["marginal_v77"].abs().gt(0).any()
    assert duals["post_iteration_reprice_v77"].eq(1).all()
    assert duals["claim_boundary_v77"].str.contains("not full-universe certificate").all()

    source_scope = _read_csv("paper4_v77_source_scope_after_iteration_3.csv")
    assert {
        "policy_id",
        "regime_v77",
        "source_family",
        "missing_source_ids_v77",
        "source_constraint_scope_complete_v77",
        "claim_boundary_v77",
    }.issubset(source_scope.columns)
    assert source_scope["missing_source_ids_v77"].sum() == status["source_scope_missing_ids_v77"]
    assert not source_scope["source_constraint_scope_complete_v77"].astype(bool).all()

    blockers = _read_csv("paper4_v77_claim_blockers.csv")
    assert {"blocker_id_v77", "blocking_v77", "evidence_count_v77", "claim_boundary_v77"}.issubset(
        blockers.columns
    )
    blocker_map = dict(zip(blockers["blocker_id_v77"], blockers["blocking_v77"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v77"], blockers["evidence_count_v77"], strict=False)
    )
    assert bool(blocker_map["negative_reduced_cost_columns_after_iteration_3"]) is False
    assert int(evidence_map["negative_reduced_cost_columns_after_iteration_3"]) == 0
    assert bool(blocker_map["source_scope_after_iteration_3_incomplete"]) is True
    assert int(evidence_map["source_scope_after_iteration_3_incomplete"]) == 7
    assert bool(blocker_map["continuous_relaxation_not_whole_loan_milp"]) is True

    claim_delta = _read_csv("paper4_v77_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v77_post_iteration_3_reprice_executed"]) is True
    assert bool(claim_map["v77_pricing_blocker_cleared"]) is True
    assert bool(claim_map["v77_exact_full_universe_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has v77 post-iteration-3 re-pricing after v76 column generation." in set(
        current_boundaries["claim"]
    )
    assert "v77 proves exact full-universe CVaR optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v77: Re-Price After Column-Generation Iteration 3" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v78_source_scope_expanded_reprice_clears_source_scope_only() -> None:
    status = _read_json("paper4_v78_status.json")

    assert status["phase"] == "v78_source_scope_expanded_reprice"
    assert status["reprice_rows_v78"] == 257954
    assert status["summary_rows_v78"] == 1
    assert status["dual_rows_v78"] == 182
    assert status["source_scope_rows_v78"] == 6
    assert status["claim_blocker_rows_v78"] == 3
    assert status["improving_columns_v78"] == 0
    assert status["negative_reduced_cost_detected_v78"] is False
    assert status["pricing_blocker_cleared_v78"] is True
    assert status["missing_source_constraint_ids_v78"] == 0
    assert status["source_scope_blocker_cleared_v78"] is True
    assert status["exact_full_universe_cvar_claim_allowed_v78"] is False
    assert status["paper1_promotion_allowed_v78"] is False
    assert status["paper4_working_champion_changed_v78"] is False
    assert status["paper4_final_promotion_created"] is False

    repriced = pd.read_parquet(TABLE_DIR / "paper4_v78_source_scope_expanded_reprice.parquet")
    assert not repriced.empty
    assert {
        "policy_id",
        "regime_v78",
        "loan_id",
        "minimization_reduced_cost_v78",
        "return_improvement_signal_v78",
        "improving_column_v78",
        "source_scope_expanded_reprice_v78",
        "claim_boundary_v78",
    }.issubset(repriced.columns)
    assert len(repriced) == status["reprice_rows_v78"]
    assert int(repriced["improving_column_v78"].sum()) == 0
    assert repriced["minimization_reduced_cost_v78"].min() > 0
    assert repriced["source_scope_expanded_reprice_v78"].eq(1).all()
    assert repriced["claim_boundary_v78"].str.contains("integrality blocker remains").all()

    summary = _read_csv("paper4_v78_source_scope_expanded_summary.csv")
    assert {
        "policy_id",
        "regime_v78",
        "omitted_rows_priced_v78",
        "improving_columns_v78",
        "negative_reduced_cost_detected_v78",
        "source_scope_expanded_reprice_v78",
        "exact_full_universe_cvar_claim_allowed_v78",
    }.issubset(summary.columns)
    assert summary["omitted_rows_priced_v78"].sum() == status["reprice_rows_v78"]
    assert summary["improving_columns_v78"].sum() == 0
    assert not summary["negative_reduced_cost_detected_v78"].astype(bool).any()
    assert summary["source_scope_expanded_reprice_v78"].astype(bool).all()
    assert not summary["exact_full_universe_cvar_claim_allowed_v78"].astype(bool).any()

    duals = _read_csv("paper4_v78_source_scope_expanded_duals.csv")
    assert {
        "policy_id",
        "regime_v78",
        "constraint_index_v78",
        "constraint_type_v78",
        "marginal_v78",
        "source_scope_expanded_v78",
        "source_id_present_in_master_v78",
        "source_scope_expanded_reprice_v78",
        "claim_boundary_v78",
    }.issubset(duals.columns)
    assert {"budget_lower", "source_share", "cvar_path_excess"}.issubset(
        set(duals["constraint_type_v78"])
    )
    source_rows = duals.loc[duals["constraint_type_v78"].eq("source_share")]
    assert len(source_rows) == 51
    assert source_rows["source_scope_expanded_v78"].astype(bool).all()
    assert source_rows["source_id_present_in_master_v78"].eq(False).sum() == 7
    assert duals["marginal_v78"].abs().gt(0).any()
    assert duals["source_scope_expanded_reprice_v78"].eq(1).all()
    assert duals["claim_boundary_v78"].str.contains("not whole-loan certificate").all()

    source_scope = _read_csv("paper4_v78_source_scope_expanded_diagnostics.csv")
    assert {
        "policy_id",
        "regime_v78",
        "source_family",
        "universe_source_ids_v78",
        "constrained_source_ids_v78",
        "missing_source_constraint_ids_v78",
        "source_constraint_scope_complete_v78",
        "claim_boundary_v78",
    }.issubset(source_scope.columns)
    assert (
        source_scope["constrained_source_ids_v78"] == source_scope["universe_source_ids_v78"]
    ).all()
    assert source_scope["missing_source_constraint_ids_v78"].sum() == 0
    assert source_scope["source_constraint_scope_complete_v78"].astype(bool).all()

    blockers = _read_csv("paper4_v78_claim_blockers.csv")
    assert {"blocker_id_v78", "blocking_v78", "evidence_count_v78", "claim_boundary_v78"}.issubset(
        blockers.columns
    )
    blocker_map = dict(zip(blockers["blocker_id_v78"], blockers["blocking_v78"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v78"], blockers["evidence_count_v78"], strict=False)
    )
    assert bool(blocker_map["negative_reduced_cost_columns_after_full_source_scope"]) is False
    assert int(evidence_map["negative_reduced_cost_columns_after_full_source_scope"]) == 0
    assert bool(blocker_map["source_constraint_scope_incomplete"]) is False
    assert int(evidence_map["source_constraint_scope_incomplete"]) == 0
    assert bool(blocker_map["continuous_relaxation_not_whole_loan_milp"]) is True

    claim_delta = _read_csv("paper4_v78_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v78_full_source_scope_reprice_executed"]) is True
    assert bool(claim_map["v78_pricing_and_source_scope_cleared"]) is True
    assert bool(claim_map["v78_exact_full_universe_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has v78 full source-scope rows for the focused pricing check." in set(
        current_boundaries["claim"]
    )
    assert "v78 proves whole-loan full-universe optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v78: Source-Scope Expanded Re-Price" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v79_integrality_probe_is_support_only() -> None:
    status = _read_json("paper4_v79_status.json")

    assert status["phase"] == "v79_support_integrality_probe"
    assert status["summary_rows_v79"] == 2
    assert status["allocation_rows_v79"] == 174
    assert status["source_summary_rows_v79"] == 102
    assert status["claim_blocker_rows_v79"] == 3
    assert status["support_rows_v79"] == 174
    assert status["lp_fractional_rows_v79"] == 3
    assert status["milp_solver_success_v79"] is True
    assert status["milp_selected_rows_v79"] == 172
    assert status["milp_delta_return_vs_lp_v79"] == pytest.approx(-43.68935963552258)
    assert status["milp_delta_cvar90_vs_lp_v79"] == pytest.approx(277.15277429476555)
    assert status["milp_source_cap_violations_v79"] == 0
    assert status["whole_loan_full_universe_claim_allowed_v79"] is False
    assert status["paper1_promotion_allowed_v79"] is False
    assert status["paper4_working_champion_changed_v79"] is False
    assert status["paper4_final_promotion_created"] is False

    summary = _read_csv("paper4_v79_integrality_probe_summary.csv")
    assert {
        "portfolio_label_v79",
        "solver_success_v79",
        "support_rows_v79",
        "selected_rows_v79",
        "fractional_rows_v79",
        "portfolio_exposure_v79",
        "objective_return_v79",
        "scenario_loss_cvar90_v79",
        "delta_return_vs_lp_v79",
        "delta_cvar90_vs_lp_v79",
        "claim_boundary_v79",
    }.issubset(summary.columns)
    assert set(summary["portfolio_label_v79"]) == {
        "continuous_lp_reference",
        "support_binary_milp",
    }
    lp = summary.loc[summary["portfolio_label_v79"].eq("continuous_lp_reference")].iloc[0]
    milp_row = summary.loc[summary["portfolio_label_v79"].eq("support_binary_milp")].iloc[0]
    assert int(lp["fractional_rows_v79"]) == 3
    assert int(milp_row["fractional_rows_v79"]) == 0
    assert int(milp_row["selected_rows_v79"]) == status["milp_selected_rows_v79"]
    assert float(milp_row["portfolio_exposure_v79"]) == pytest.approx(845000.0)
    assert float(milp_row["delta_return_vs_lp_v79"]) == pytest.approx(
        status["milp_delta_return_vs_lp_v79"]
    )
    assert float(milp_row["delta_cvar90_vs_lp_v79"]) == pytest.approx(
        status["milp_delta_cvar90_vs_lp_v79"]
    )
    assert bool(milp_row["solver_success_v79"]) is True
    assert int(milp_row["source_cap_violations_v79"]) == 0
    assert summary["claim_boundary_v79"].str.contains("not full-pool or full-universe").all()

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v79_integrality_probe_allocations.parquet")
    assert not allocations.empty
    assert {
        "policy_id",
        "regime_v79",
        "loan_id",
        "allocation_fraction_v76",
        "support_binary_selected_v79",
        "support_binary_exposure_v79",
        "claim_boundary_v79",
    }.issubset(allocations.columns)
    assert set(allocations["support_binary_selected_v79"].unique()).issubset({0.0, 1.0})
    assert int(allocations["support_binary_selected_v79"].sum()) == status["milp_selected_rows_v79"]
    assert allocations["support_binary_exposure_v79"].sum() == pytest.approx(845000.0)
    assert (
        allocations.loc[allocations["support_binary_selected_v79"].eq(0), "loan_amnt"].sum() == 3500
    )
    assert allocations["claim_boundary_v79"].str.contains("support-restricted").all()

    source_summary = _read_csv("paper4_v79_integrality_probe_source_summary.csv")
    assert {
        "portfolio_label_v79",
        "source_family",
        "source_id",
        "cap_share_v79",
        "source_share_v79",
        "source_cap_violated_v79",
    }.issubset(source_summary.columns)
    assert set(source_summary["portfolio_label_v79"]) == {
        "continuous_lp_reference",
        "support_binary_milp",
    }
    assert not source_summary["source_cap_violated_v79"].astype(bool).any()

    blockers = _read_csv("paper4_v79_claim_blockers.csv")
    assert {"blocker_id_v79", "blocking_v79", "evidence_count_v79", "claim_boundary_v79"}.issubset(
        blockers.columns
    )
    blocker_map = dict(zip(blockers["blocker_id_v79"], blockers["blocking_v79"], strict=False))
    assert bool(blocker_map["support_integrality_gap_quantified"]) is False
    assert bool(blocker_map["full_pool_or_full_universe_milp_missing"]) is True
    assert bool(blocker_map["paper_estrella_or_final_promotion_not_allowed"]) is True

    claim_delta = _read_csv("paper4_v79_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v79_support_integrality_probe_executed"]) is True
    assert bool(claim_map["v79_whole_loan_full_universe_optimality"]) is False
    assert bool(claim_map["v79_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v79 support-restricted binary MILP integrality probe." in set(
        current_boundaries["claim"]
    )
    assert "v79 proves whole-loan full-universe optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v79: Support Integrality Probe" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v80_full_pool_milp_gap_probe_is_not_full_universe() -> None:
    status = _read_json("paper4_v80_status.json")

    assert status["phase"] == "v80_focused_full_pool_milp_gap_probe"
    assert status["summary_rows_v80"] == 2
    assert status["allocation_rows_v80"] == 18915
    assert status["source_summary_rows_v80"] == 102
    assert status["constraint_rows_v80"] == 181
    assert status["claim_blocker_rows_v80"] == 3
    assert status["pool_rows_v80"] == 18915
    assert status["lp_fractional_rows_v80"] == 3
    assert status["milp_solver_success_v80"] is True
    assert status["milp_incumbent_available_v80"] is True
    assert status["milp_status_v80"] == 0
    assert status["milp_gap_v80"] < 1e-5
    assert status["milp_node_count_v80"] > 0
    assert status["milp_selected_rows_v80"] == 171
    assert status["milp_delta_return_vs_lp_v80"] == pytest.approx(-28.726313016571112)
    assert status["milp_delta_cvar90_vs_lp_v80"] == pytest.approx(-35.06590165325906)
    assert status["milp_source_cap_violations_v80"] == 0
    assert status["whole_loan_full_universe_claim_allowed_v80"] is False
    assert status["paper1_promotion_allowed_v80"] is False
    assert status["paper4_working_champion_changed_v80"] is False
    assert status["paper4_final_promotion_created"] is False

    summary = _read_csv("paper4_v80_full_pool_milp_gap_summary.csv")
    assert {
        "portfolio_label_v80",
        "solver_success_v80",
        "pool_rows_v80",
        "selected_rows_v80",
        "fractional_rows_v80",
        "portfolio_exposure_v80",
        "objective_return_v80",
        "scenario_loss_cvar90_v80",
        "milp_gap_v80",
        "binary_variables_v80",
        "delta_return_vs_lp_v80",
        "delta_cvar90_vs_lp_v80",
        "claim_boundary_v80",
    }.issubset(summary.columns)
    assert set(summary["portfolio_label_v80"]) == {
        "continuous_lp_reference",
        "focused_full_pool_binary_milp",
    }
    lp = summary.loc[summary["portfolio_label_v80"].eq("continuous_lp_reference")].iloc[0]
    milp_row = summary.loc[summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")].iloc[
        0
    ]
    assert int(lp["fractional_rows_v80"]) == 3
    assert int(milp_row["fractional_rows_v80"]) == 0
    assert int(milp_row["pool_rows_v80"]) == status["pool_rows_v80"]
    assert int(milp_row["binary_variables_v80"]) == status["pool_rows_v80"]
    assert int(milp_row["selected_rows_v80"]) == status["milp_selected_rows_v80"]
    assert float(milp_row["portfolio_exposure_v80"]) == pytest.approx(842400.0)
    assert float(milp_row["milp_gap_v80"]) == pytest.approx(status["milp_gap_v80"])
    assert float(milp_row["delta_return_vs_lp_v80"]) == pytest.approx(
        status["milp_delta_return_vs_lp_v80"]
    )
    assert float(milp_row["delta_cvar90_vs_lp_v80"]) == pytest.approx(
        status["milp_delta_cvar90_vs_lp_v80"]
    )
    assert bool(milp_row["solver_success_v80"]) is True
    assert int(milp_row["source_cap_violations_v80"]) == 0
    assert summary["claim_boundary_v80"].str.contains("not full-universe").all()

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v80_full_pool_milp_gap_allocations.parquet")
    assert not allocations.empty
    assert {
        "policy_id",
        "regime_v80",
        "loan_id",
        "allocation_fraction_v76",
        "focused_full_pool_binary_selected_v80",
        "focused_full_pool_binary_exposure_v80",
        "claim_boundary_v80",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v80"]
    assert set(allocations["focused_full_pool_binary_selected_v80"].unique()).issubset({0.0, 1.0})
    assert (
        int(allocations["focused_full_pool_binary_selected_v80"].sum())
        == status["milp_selected_rows_v80"]
    )
    assert allocations["focused_full_pool_binary_exposure_v80"].sum() == pytest.approx(842400.0)
    selected_roles = set(
        allocations.loc[
            allocations["focused_full_pool_binary_selected_v80"].eq(1), "master_role_v69"
        ]
    )
    assert {
        "incumbent_v63_book",
        "v68_pricing_candidate",
        "v71_negative_reduced_cost_column",
        "v73_negative_reduced_cost_column",
    }.issubset(selected_roles)
    assert allocations["claim_boundary_v80"].str.contains("restricted-pool binary").all()

    source_summary = _read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    assert {
        "portfolio_label_v80",
        "source_family",
        "source_id",
        "cap_share_v80",
        "source_share_v80",
        "source_cap_violated_v80",
    }.issubset(source_summary.columns)
    assert set(source_summary["portfolio_label_v80"]) == {
        "continuous_lp_reference",
        "focused_full_pool_binary_milp",
    }
    assert not source_summary["source_cap_violated_v80"].astype(bool).any()

    constraints = _read_csv("paper4_v80_full_pool_milp_gap_constraints.csv")
    assert {"constraint_type_v80", "constraint_row_v80"}.issubset(constraints.columns)
    assert len(constraints) == status["constraint_rows_v80"]
    assert (constraints["constraint_type_v80"] == "source_share").sum() == 51
    assert (constraints["constraint_type_v80"] == "cvar_path_excess").sum() == 128
    assert (constraints["constraint_type_v80"] == "budget_range").sum() == 1

    blockers = _read_csv("paper4_v80_claim_blockers.csv")
    assert {"blocker_id_v80", "blocking_v80", "evidence_count_v80", "claim_boundary_v80"}.issubset(
        blockers.columns
    )
    blocker_map = dict(zip(blockers["blocker_id_v80"], blockers["blocking_v80"], strict=False))
    assert bool(blocker_map["focused_full_pool_milp_gap_recorded"]) is False
    assert bool(blocker_map["full_universe_integer_pricing_missing"]) is True
    assert bool(blocker_map["paper_estrella_or_final_promotion_not_allowed"]) is True

    claim_delta = _read_csv("paper4_v80_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v80_focused_full_pool_milp_gap_probe_executed"]) is True
    assert bool(claim_map["v80_whole_loan_full_universe_optimality"]) is False
    assert bool(claim_map["v80_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v80 focused full-pool binary MILP/gap probe." in set(
        current_boundaries["claim"]
    )
    assert "v80 proves whole-loan full-universe optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v80: Focused Full-Pool MILP Gap Probe" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v81_integer_single_add_screen_is_not_global_optimality() -> None:
    status = _read_json("paper4_v81_status.json")

    assert status["phase"] == "v81_integer_omitted_single_add_screen"
    assert status["screen_rows_v81"] == 257954
    assert status["summary_rows_v81"] == 1
    assert status["source_summary_rows_v81"] == 6
    assert status["top_candidate_rows_v81"] == 200
    assert status["claim_blocker_rows_v81"] == 3
    assert status["omitted_rows_screened_v81"] == 257954
    assert status["budget_feasible_rows_v81"] == 63528
    assert status["positive_return_rows_v81"] == 3
    assert status["source_feasible_rows_v81"] == 203707
    assert status["cvar_feasible_rows_v81"] == 49492
    assert status["single_add_feasible_rows_v81"] == 49492
    assert status["single_add_improving_rows_v81"] == 0
    assert status["best_single_add_return_delta_v81"] == pytest.approx(-92.29890441894531)
    assert status["best_any_omitted_return_delta_v81"] == pytest.approx(161.50875854492188)
    assert status["integer_single_add_screen_cleared_v81"] is True
    assert status["full_universe_integer_optimality_claim_allowed_v81"] is False
    assert status["paper1_promotion_allowed_v81"] is False
    assert status["paper4_working_champion_changed_v81"] is False
    assert status["paper4_final_promotion_created"] is False

    screen = pd.read_parquet(TABLE_DIR / "paper4_v81_integer_omitted_single_add_screen.parquet")
    assert not screen.empty
    assert {
        "policy_id",
        "regime_v78",
        "loan_id",
        "loan_amnt",
        "mean_return_if_added_v81",
        "budget_add_feasible_v81",
        "source_add_feasible_v81",
        "cvar_add_feasible_v81",
        "single_add_feasible_v81",
        "single_add_improves_return_v81",
        "claim_boundary_v81",
    }.issubset(screen.columns)
    assert len(screen) == status["screen_rows_v81"]
    assert int(screen["budget_add_feasible_v81"].sum()) == status["budget_feasible_rows_v81"]
    assert int(screen["source_add_feasible_v81"].sum()) == status["source_feasible_rows_v81"]
    assert int(screen["cvar_add_feasible_v81"].sum()) == status["cvar_feasible_rows_v81"]
    assert int(screen["single_add_feasible_v81"].sum()) == status["single_add_feasible_rows_v81"]
    assert int(screen["single_add_improves_return_v81"].sum()) == 0
    assert int(screen["mean_return_if_added_v81"].gt(0).sum()) == status["positive_return_rows_v81"]
    positive = screen.loc[screen["mean_return_if_added_v81"].gt(0)]
    assert not positive.empty
    assert not positive["single_add_feasible_v81"].astype(bool).any()
    assert screen["claim_boundary_v81"].str.contains("not multi-swap or global proof").all()

    summary = _read_csv("paper4_v81_integer_omitted_single_add_summary.csv")
    assert {
        "omitted_rows_screened_v81",
        "budget_feasible_rows_v81",
        "positive_return_rows_v81",
        "source_feasible_rows_v81",
        "cvar_feasible_rows_v81",
        "single_add_feasible_rows_v81",
        "single_add_improving_rows_v81",
        "integer_single_add_screen_cleared_v81",
        "full_universe_integer_optimality_claim_allowed_v81",
        "claim_boundary_v81",
    }.issubset(summary.columns)
    row = summary.iloc[0]
    assert int(row["omitted_rows_screened_v81"]) == status["omitted_rows_screened_v81"]
    assert int(row["single_add_improving_rows_v81"]) == 0
    assert bool(row["integer_single_add_screen_cleared_v81"]) is True
    assert bool(row["full_universe_integer_optimality_claim_allowed_v81"]) is False
    assert "multi-loan swaps" in str(row["claim_boundary_v81"])

    source_summary = _read_csv("paper4_v81_integer_omitted_single_add_source_summary.csv")
    assert {
        "source_family",
        "rows_checked_v81",
        "source_blocked_rows_v81",
        "source_feasible_rows_v81",
        "claim_boundary_v81",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v81"]
    assert source_summary["rows_checked_v81"].eq(status["screen_rows_v81"]).all()
    assert source_summary["source_blocked_rows_v81"].sum() > 0

    top_candidates = _read_csv("paper4_v81_integer_omitted_single_add_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v81"]
    assert not top_candidates["single_add_improves_return_v81"].astype(bool).any()
    assert top_candidates["single_add_feasible_v81"].astype(bool).any()

    blockers = _read_csv("paper4_v81_claim_blockers.csv")
    assert {"blocker_id_v81", "blocking_v81", "evidence_count_v81", "claim_boundary_v81"}.issubset(
        blockers.columns
    )
    blocker_map = dict(zip(blockers["blocker_id_v81"], blockers["blocking_v81"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v81"], blockers["evidence_count_v81"], strict=False)
    )
    assert bool(blocker_map["single_add_integer_improvement_found"]) is False
    assert int(evidence_map["single_add_integer_improvement_found"]) == 0
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v81_claim_matrix_delta.csv")
    assert {"claim_id", "allowed", "artifact", "boundary"}.issubset(claim_delta.columns)
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v81_single_add_integer_screen_executed"]) is True
    assert bool(claim_map["v81_single_add_integer_screen_cleared"]) is True
    assert bool(claim_map["v81_full_universe_integer_optimality"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v81 single-add integer screen over omitted v55 loans." in set(
        current_boundaries["claim"]
    )
    assert "v81 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v81: Integer Omitted Single-Add Screen" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v82_one_swap_probe_finds_repair_signals_not_optimality() -> None:
    status = _read_json("paper4_v82_status.json")

    assert status["phase"] == "v82_one_swap_integer_pricing_probe"
    assert status["summary_rows_v82"] == 1
    assert status["stage_summary_rows_v82"] == 6
    assert status["candidate_pair_rows_v82"] == 8126
    assert status["top_candidate_rows_v82"] == 200
    assert status["claim_blocker_rows_v82"] == 3
    assert status["selected_rows_v82"] == 171
    assert status["omitted_rows_v82"] == 257954
    assert status["total_pair_rows_screened_v82"] == 44110134
    assert status["return_improving_pair_rows_v82"] == 784725
    assert status["budget_return_feasible_pair_rows_v82"] == 596651
    assert status["source_prefilter_pair_rows_v82"] == 9636
    assert status["source_exact_pair_rows_v82"] == 8126
    assert status["cvar_feasible_pair_rows_v82"] == 8126
    assert status["one_swap_improving_rows_v82"] == 8126
    assert status["best_one_swap_return_delta_v82"] == pytest.approx(201.74349922313803)
    assert status["best_one_swap_cvar90_after_v82"] == pytest.approx(91625.73939095305)
    assert status["one_swap_local_optimality_cleared_v82"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v82"] is False
    assert status["paper1_promotion_allowed_v82"] is False
    assert status["paper4_working_champion_changed_v82"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v82_one_swap_integer_pricing_probe.csv")
    assert {
        "added_loan_id_v82",
        "dropped_loan_id_v82",
        "return_delta_v82",
        "objective_return_after_swap_v82",
        "exposure_after_swap_v82",
        "budget_swap_feasible_v82",
        "source_swap_feasible_v82",
        "source_cap_violations_after_swap_v82",
        "cvar90_after_swap_v82",
        "cvar_swap_feasible_v82",
        "one_swap_improves_return_v82",
        "claim_boundary_v82",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v82"]
    assert probe["return_delta_v82"].gt(0).all()
    assert probe["budget_swap_feasible_v82"].astype(bool).all()
    assert probe["source_swap_feasible_v82"].astype(bool).all()
    assert probe["cvar_swap_feasible_v82"].astype(bool).all()
    assert probe["one_swap_improves_return_v82"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v82"].sum()) == 0
    assert probe["return_delta_v82"].max() == pytest.approx(
        status["best_one_swap_return_delta_v82"]
    )
    assert probe["claim_boundary_v82"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v82_one_swap_integer_pricing_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v82"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v82"]) == "154722969"
    assert str(best["dropped_loan_id_v82"]) == "127135245"
    assert float(best["return_delta_v82"]) == pytest.approx(
        status["best_one_swap_return_delta_v82"]
    )
    assert bool(best["one_swap_improves_return_v82"]) is True

    summary = _read_csv("paper4_v82_one_swap_integer_pricing_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v82"]) == status["one_swap_improving_rows_v82"]
    assert bool(row["one_swap_local_optimality_cleared_v82"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v82"]) is False
    assert "feasible improving swaps" in str(row["claim_boundary_v82"])

    stage_summary = _read_csv("paper4_v82_one_swap_screen_stage_summary.csv")
    assert {"stage_v82", "pair_rows_v82", "claim_boundary_v82"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v82"], stage_summary["pair_rows_v82"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v82"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v82"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v82"]

    blockers = _read_csv("paper4_v82_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v82"], blockers["blocking_v82"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v82"], blockers["evidence_count_v82"], strict=False)
    )
    assert bool(blocker_map["one_swap_integer_improvement_found"]) is True
    assert int(evidence_map["one_swap_integer_improvement_found"]) == 8126
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v82_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v82_one_swap_integer_probe_executed"]) is True
    assert bool(claim_map["v82_feasible_improving_one_swaps_found"]) is True
    assert bool(claim_map["v82_v80_one_swap_local_optimality"]) is False
    assert bool(claim_map["v82_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v82_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v82 one-swap integer pricing probe over omitted v55 loans." in set(
        current_boundaries["claim"]
    )
    assert "v82 proves v80 is one-swap locally optimal." in set(current_boundaries["claim"])
    assert "v82 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v82: One-Swap Integer Pricing Probe" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v83_best_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v83_status.json")

    assert status["phase"] == "v83_best_one_swap_repair"
    assert status["allocation_rows_v83"] == 171
    assert status["summary_rows_v83"] == 1
    assert status["action_rows_v83"] == 1
    assert status["source_summary_rows_v83"] == 51
    assert status["claim_blocker_rows_v83"] == 4
    assert status["added_loan_id_v83"] == "154722969"
    assert status["dropped_loan_id_v83"] == "127135245"
    assert status["selected_rows_v83"] == 171
    assert status["portfolio_exposure_v83"] == pytest.approx(842400.0)
    assert status["objective_return_v83"] == pytest.approx(-3854.1520292167534)
    assert status["scenario_loss_cvar90_v83"] == pytest.approx(91625.73939095305)
    assert status["source_cap_violations_v83"] == 0
    assert status["delta_return_vs_v80_v83"] == pytest.approx(201.74350209599243)
    assert status["delta_cvar90_vs_v80_v83"] == pytest.approx(231.72304392619117)
    assert status["delta_exposure_vs_v80_v83"] == pytest.approx(0.0)
    assert status["budget_feasible_v83"] is True
    assert status["source_feasible_v83"] is True
    assert status["cvar_feasible_v83"] is True
    assert status["repair_candidate_feasible_v83"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v83"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v83"] is False
    assert status["paper1_promotion_allowed_v83"] is False
    assert status["paper4_working_champion_changed_v83"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v83_best_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v83",
        "selected_v83",
        "portfolio_label_v83",
        "repair_action_v83",
        "claim_boundary_v83",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v83"]
    assert int(allocations["selected_v83"].sum()) == status["selected_rows_v83"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v83"])
    assert "154722969" in set(allocations["loan_id"].astype(str))
    assert "127135245" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v83"]) == {
        "added_from_v82_best_swap",
        "kept_from_v80",
    }
    assert allocations["claim_boundary_v83"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v83_best_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v83"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v83"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v83"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v83"])

    action = _read_csv("paper4_v83_best_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v83"]) == status["added_loan_id_v83"]
    assert str(action_row["dropped_loan_id_v83"]) == status["dropped_loan_id_v83"]
    assert float(action_row["return_delta_v83"]) == pytest.approx(201.74349922313803)
    assert int(action_row["source_cap_violations_after_repair_v83"]) == 0

    source_summary = _read_csv("paper4_v83_best_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v83",
        "source_slack_v83",
        "source_cap_violated_v83",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v83"]
    assert not source_summary["source_cap_violated_v83"].astype(bool).any()

    blockers = _read_csv("paper4_v83_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v83"], blockers["blocking_v83"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v83"], blockers["evidence_count_v83"], strict=False)
    )
    assert bool(blocker_map["best_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["best_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v83_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v83_best_one_swap_repair_executed"]) is True
    assert bool(claim_map["v83_repair_candidate_feasible"]) is True
    assert bool(claim_map["v83_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v83_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v83_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v83 best one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v83 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v83 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v83: Best One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v84_post_repair_one_swap_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v84_status.json")

    assert status["phase"] == "v84_post_repair_one_swap_reprice"
    assert status["summary_rows_v84"] == 1
    assert status["stage_summary_rows_v84"] == 6
    assert status["candidate_pair_rows_v84"] == 8000
    assert status["top_candidate_rows_v84"] == 200
    assert status["claim_blocker_rows_v84"] == 3
    assert status["selected_rows_v84"] == 171
    assert status["candidate_add_rows_v84"] == 276698
    assert status["total_pair_rows_screened_v84"] == 47315358
    assert status["return_improving_pair_rows_v84"] == 2142058
    assert status["budget_return_feasible_pair_rows_v84"] == 1290982
    assert status["source_prefilter_pair_rows_v84"] == 9323
    assert status["source_exact_pair_rows_v84"] == 8000
    assert status["cvar_feasible_pair_rows_v84"] == 8000
    assert status["one_swap_improving_rows_v84"] == 8000
    assert status["best_one_swap_return_delta_v84"] == pytest.approx(198.2450405042807)
    assert status["best_one_swap_cvar90_after_v84"] == pytest.approx(91667.73069994824)
    assert status["post_repair_one_swap_local_optimality_cleared_v84"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v84"] is False
    assert status["paper1_promotion_allowed_v84"] is False
    assert status["paper4_working_champion_changed_v84"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v84_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v84",
        "dropped_loan_id_v84",
        "return_delta_v84",
        "objective_return_after_swap_v84",
        "budget_swap_feasible_v84",
        "source_swap_feasible_v84",
        "source_cap_violations_after_swap_v84",
        "cvar_swap_feasible_v84",
        "one_swap_improves_return_v84",
        "claim_boundary_v84",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v84"]
    assert probe["return_delta_v84"].gt(0).all()
    assert probe["budget_swap_feasible_v84"].astype(bool).all()
    assert probe["source_swap_feasible_v84"].astype(bool).all()
    assert probe["cvar_swap_feasible_v84"].astype(bool).all()
    assert probe["one_swap_improves_return_v84"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v84"].sum()) == 0
    assert probe["return_delta_v84"].max() == pytest.approx(
        status["best_one_swap_return_delta_v84"]
    )
    assert probe["claim_boundary_v84"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v84_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v84"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v84"]) == "143293919"
    assert str(best["dropped_loan_id_v84"]) == "127392383"
    assert float(best["return_delta_v84"]) == pytest.approx(
        status["best_one_swap_return_delta_v84"]
    )
    assert bool(best["one_swap_improves_return_v84"]) is True

    summary = _read_csv("paper4_v84_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v84"]) == status["one_swap_improving_rows_v84"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v84"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v84"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v84"])

    stage_summary = _read_csv("paper4_v84_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v84", "pair_rows_v84", "claim_boundary_v84"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v84"], stage_summary["pair_rows_v84"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v84"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v84"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v84"]

    blockers = _read_csv("paper4_v84_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v84"], blockers["blocking_v84"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v84"], blockers["evidence_count_v84"], strict=False)
    )
    assert bool(blocker_map["post_repair_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_repair_one_swap_improvement_found"]) == 8000
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v84_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v84_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v84_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v84_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v84_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v84 post-repair one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v84 proves the v83 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v84 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v84: Post-Repair One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v85_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v85_status.json")

    assert status["phase"] == "v85_next_one_swap_repair"
    assert status["allocation_rows_v85"] == 171
    assert status["summary_rows_v85"] == 1
    assert status["action_rows_v85"] == 1
    assert status["source_summary_rows_v85"] == 51
    assert status["claim_blocker_rows_v85"] == 4
    assert status["added_loan_id_v85"] == "143293919"
    assert status["dropped_loan_id_v85"] == "127392383"
    assert status["selected_rows_v85"] == 171
    assert status["portfolio_exposure_v85"] == pytest.approx(842400.0)
    assert status["objective_return_v85"] == pytest.approx(-3655.9069887124715)
    assert status["scenario_loss_cvar90_v85"] == pytest.approx(91667.73069994825)
    assert status["source_cap_violations_v85"] == 0
    assert status["delta_return_vs_v83_v85"] == pytest.approx(198.24504050428231)
    assert status["delta_cvar90_vs_v83_v85"] == pytest.approx(41.99130899521697)
    assert status["delta_exposure_vs_v83_v85"] == pytest.approx(0.0)
    assert status["budget_feasible_v85"] is True
    assert status["source_feasible_v85"] is True
    assert status["cvar_feasible_v85"] is True
    assert status["repair_candidate_feasible_v85"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v85"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v85"] is False
    assert status["paper1_promotion_allowed_v85"] is False
    assert status["paper4_working_champion_changed_v85"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v85_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v85",
        "selected_v85",
        "portfolio_label_v85",
        "repair_action_v85",
        "claim_boundary_v85",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v85"]
    assert int(allocations["selected_v85"].sum()) == status["selected_rows_v85"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v85"])
    assert "143293919" in set(allocations["loan_id"].astype(str))
    assert "127392383" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v85"]) == {
        "added_from_v84_best_swap",
        "kept_from_v83",
    }
    assert allocations["claim_boundary_v85"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v85_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v85"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v85"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v85"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v85"])

    action = _read_csv("paper4_v85_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v85"]) == status["added_loan_id_v85"]
    assert str(action_row["dropped_loan_id_v85"]) == status["dropped_loan_id_v85"]
    assert float(action_row["return_delta_v85"]) == pytest.approx(198.2450405042807)
    assert int(action_row["source_cap_violations_after_repair_v85"]) == 0

    source_summary = _read_csv("paper4_v85_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v85",
        "source_slack_v85",
        "source_cap_violated_v85",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v85"]
    assert not source_summary["source_cap_violated_v85"].astype(bool).any()

    blockers = _read_csv("paper4_v85_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v85"], blockers["blocking_v85"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v85"], blockers["evidence_count_v85"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v85_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v85_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v85_repair_candidate_feasible"]) is True
    assert bool(claim_map["v85_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v85_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v85_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v85 second one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v85 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v85 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v85: Second One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v86_post_v85_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v86_status.json")

    assert status["phase"] == "v86_post_v85_one_swap_reprice"
    assert status["summary_rows_v86"] == 1
    assert status["stage_summary_rows_v86"] == 6
    assert status["candidate_pair_rows_v86"] == 7847
    assert status["top_candidate_rows_v86"] == 200
    assert status["claim_blocker_rows_v86"] == 3
    assert status["selected_rows_v86"] == 171
    assert status["candidate_add_rows_v86"] == 276698
    assert status["total_pair_rows_screened_v86"] == 47315358
    assert status["return_improving_pair_rows_v86"] == 2127315
    assert status["budget_return_feasible_pair_rows_v86"] == 1283115
    assert status["source_prefilter_pair_rows_v86"] == 9142
    assert status["source_exact_pair_rows_v86"] == 7847
    assert status["cvar_feasible_pair_rows_v86"] == 7847
    assert status["one_swap_improving_rows_v86"] == 7847
    assert status["best_one_swap_return_delta_v86"] == pytest.approx(186.46858045165027)
    assert status["best_one_swap_cvar90_after_v86"] == pytest.approx(91778.02003085389)
    assert status["post_repair_one_swap_local_optimality_cleared_v86"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v86"] is False
    assert status["paper1_promotion_allowed_v86"] is False
    assert status["paper4_working_champion_changed_v86"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v86_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v86",
        "dropped_loan_id_v86",
        "return_delta_v86",
        "objective_return_after_swap_v86",
        "budget_swap_feasible_v86",
        "source_swap_feasible_v86",
        "source_cap_violations_after_swap_v86",
        "cvar_swap_feasible_v86",
        "one_swap_improves_return_v86",
        "claim_boundary_v86",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v86"]
    assert probe["return_delta_v86"].gt(0).all()
    assert probe["budget_swap_feasible_v86"].astype(bool).all()
    assert probe["source_swap_feasible_v86"].astype(bool).all()
    assert probe["cvar_swap_feasible_v86"].astype(bool).all()
    assert probe["one_swap_improves_return_v86"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v86"].sum()) == 0
    assert probe["return_delta_v86"].max() == pytest.approx(
        status["best_one_swap_return_delta_v86"]
    )
    assert probe["claim_boundary_v86"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v86_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v86"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v86"]) == "129036264"
    assert str(best["dropped_loan_id_v86"]) == "127726917"
    assert float(best["return_delta_v86"]) == pytest.approx(
        status["best_one_swap_return_delta_v86"]
    )
    assert bool(best["one_swap_improves_return_v86"]) is True

    summary = _read_csv("paper4_v86_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v86"]) == status["one_swap_improving_rows_v86"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v86"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v86"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v86"])

    stage_summary = _read_csv("paper4_v86_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v86", "pair_rows_v86", "claim_boundary_v86"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v86"], stage_summary["pair_rows_v86"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v86"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v86"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v86"]

    blockers = _read_csv("paper4_v86_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v86"], blockers["blocking_v86"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v86"], blockers["evidence_count_v86"], strict=False)
    )
    assert bool(blocker_map["post_v85_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v85_one_swap_improvement_found"]) == 7847
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v86_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v86_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v86_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v86_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v86_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v86 post-v85 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v86 proves the v85 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v86 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v86: Post-v85 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v87_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v87_status.json")

    assert status["phase"] == "v87_next_one_swap_repair"
    assert status["allocation_rows_v87"] == 171
    assert status["summary_rows_v87"] == 1
    assert status["action_rows_v87"] == 1
    assert status["source_summary_rows_v87"] == 51
    assert status["claim_blocker_rows_v87"] == 4
    assert status["added_loan_id_v87"] == "129036264"
    assert status["dropped_loan_id_v87"] == "127726917"
    assert status["selected_rows_v87"] == 171
    assert status["portfolio_exposure_v87"] == pytest.approx(842500.0)
    assert status["objective_return_v87"] == pytest.approx(-3469.4384082608176)
    assert status["scenario_loss_cvar90_v87"] == pytest.approx(91778.02003085389)
    assert status["source_cap_violations_v87"] == 0
    assert status["delta_return_vs_v85_v87"] == pytest.approx(186.46858045165436)
    assert status["delta_cvar90_vs_v85_v87"] == pytest.approx(110.2893309056526)
    assert status["delta_exposure_vs_v85_v87"] == pytest.approx(100.0)
    assert status["budget_feasible_v87"] is True
    assert status["source_feasible_v87"] is True
    assert status["cvar_feasible_v87"] is True
    assert status["repair_candidate_feasible_v87"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v87"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v87"] is False
    assert status["paper1_promotion_allowed_v87"] is False
    assert status["paper4_working_champion_changed_v87"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v87_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v87",
        "selected_v87",
        "portfolio_label_v87",
        "repair_action_v87",
        "claim_boundary_v87",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v87"]
    assert int(allocations["selected_v87"].sum()) == status["selected_rows_v87"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v87"])
    assert "129036264" in set(allocations["loan_id"].astype(str))
    assert "127726917" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v87"]) == {
        "added_from_v86_best_swap",
        "kept_from_v85",
    }
    assert allocations["claim_boundary_v87"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v87_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v87"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v87"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v87"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v87"])

    action = _read_csv("paper4_v87_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v87"]) == status["added_loan_id_v87"]
    assert str(action_row["dropped_loan_id_v87"]) == status["dropped_loan_id_v87"]
    assert float(action_row["return_delta_v87"]) == pytest.approx(186.46858045165027)
    assert int(action_row["source_cap_violations_after_repair_v87"]) == 0

    source_summary = _read_csv("paper4_v87_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v87",
        "source_slack_v87",
        "source_cap_violated_v87",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v87"]
    assert not source_summary["source_cap_violated_v87"].astype(bool).any()

    blockers = _read_csv("paper4_v87_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v87"], blockers["blocking_v87"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v87"], blockers["evidence_count_v87"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v87_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v87_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v87_repair_candidate_feasible"]) is True
    assert bool(claim_map["v87_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v87_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v87_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v87 third one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v87 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v87 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v87: Third One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v88_post_v87_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v88_status.json")

    assert status["phase"] == "v88_post_v87_one_swap_reprice"
    assert status["summary_rows_v88"] == 1
    assert status["stage_summary_rows_v88"] == 6
    assert status["candidate_pair_rows_v88"] == 9044
    assert status["top_candidate_rows_v88"] == 200
    assert status["claim_blocker_rows_v88"] == 3
    assert status["selected_rows_v88"] == 171
    assert status["candidate_add_rows_v88"] == 276698
    assert status["total_pair_rows_screened_v88"] == 47315358
    assert status["return_improving_pair_rows_v88"] == 2114323
    assert status["budget_return_feasible_pair_rows_v88"] == 1277960
    assert status["source_prefilter_pair_rows_v88"] == 11763
    assert status["source_exact_pair_rows_v88"] == 9044
    assert status["cvar_feasible_pair_rows_v88"] == 9044
    assert status["one_swap_improving_rows_v88"] == 9044
    assert status["best_one_swap_return_delta_v88"] == pytest.approx(193.93226645730118)
    assert status["best_one_swap_cvar90_after_v88"] == pytest.approx(91851.98990775931)
    assert status["post_repair_one_swap_local_optimality_cleared_v88"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v88"] is False
    assert status["paper1_promotion_allowed_v88"] is False
    assert status["paper4_working_champion_changed_v88"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v88_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v88",
        "dropped_loan_id_v88",
        "return_delta_v88",
        "objective_return_after_swap_v88",
        "budget_swap_feasible_v88",
        "source_swap_feasible_v88",
        "source_cap_violations_after_swap_v88",
        "cvar_swap_feasible_v88",
        "one_swap_improves_return_v88",
        "claim_boundary_v88",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v88"]
    assert probe["return_delta_v88"].gt(0).all()
    assert probe["budget_swap_feasible_v88"].astype(bool).all()
    assert probe["source_swap_feasible_v88"].astype(bool).all()
    assert probe["cvar_swap_feasible_v88"].astype(bool).all()
    assert probe["one_swap_improves_return_v88"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v88"].sum()) == 0
    assert probe["return_delta_v88"].max() == pytest.approx(
        status["best_one_swap_return_delta_v88"]
    )
    assert probe["claim_boundary_v88"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v88_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v88"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v88"]) == "165964183"
    assert str(best["dropped_loan_id_v88"]) == "126916681"
    assert float(best["return_delta_v88"]) == pytest.approx(
        status["best_one_swap_return_delta_v88"]
    )
    assert bool(best["one_swap_improves_return_v88"]) is True

    summary = _read_csv("paper4_v88_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v88"]) == status["one_swap_improving_rows_v88"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v88"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v88"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v88"])

    stage_summary = _read_csv("paper4_v88_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v88", "pair_rows_v88", "claim_boundary_v88"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v88"], stage_summary["pair_rows_v88"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v88"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v88"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v88"]

    blockers = _read_csv("paper4_v88_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v88"], blockers["blocking_v88"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v88"], blockers["evidence_count_v88"], strict=False)
    )
    assert bool(blocker_map["post_v87_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v87_one_swap_improvement_found"]) == 9044
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v88_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v88_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v88_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v88_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v88_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v88 post-v87 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v88 proves the v87 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v88 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v88: Post-v87 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v89_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v89_status.json")

    assert status["phase"] == "v89_next_one_swap_repair"
    assert status["allocation_rows_v89"] == 171
    assert status["summary_rows_v89"] == 1
    assert status["action_rows_v89"] == 1
    assert status["source_summary_rows_v89"] == 51
    assert status["claim_blocker_rows_v89"] == 4
    assert status["added_loan_id_v89"] == "165964183"
    assert status["dropped_loan_id_v89"] == "126916681"
    assert status["selected_rows_v89"] == 171
    assert status["portfolio_exposure_v89"] == pytest.approx(842450.0)
    assert status["objective_return_v89"] == pytest.approx(-3275.50614180352)
    assert status["scenario_loss_cvar90_v89"] == pytest.approx(91851.98990775933)
    assert status["source_cap_violations_v89"] == 0
    assert status["delta_return_vs_v87_v89"] == pytest.approx(193.93226645729737)
    assert status["delta_cvar90_vs_v87_v89"] == pytest.approx(73.96987690545211)
    assert status["delta_exposure_vs_v87_v89"] == pytest.approx(-50.0)
    assert status["budget_feasible_v89"] is True
    assert status["source_feasible_v89"] is True
    assert status["cvar_feasible_v89"] is True
    assert status["repair_candidate_feasible_v89"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v89"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v89"] is False
    assert status["paper1_promotion_allowed_v89"] is False
    assert status["paper4_working_champion_changed_v89"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v89_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v89",
        "selected_v89",
        "portfolio_label_v89",
        "repair_action_v89",
        "claim_boundary_v89",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v89"]
    assert int(allocations["selected_v89"].sum()) == status["selected_rows_v89"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v89"])
    assert "165964183" in set(allocations["loan_id"].astype(str))
    assert "126916681" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v89"]) == {
        "added_from_v88_best_swap",
        "kept_from_v87",
    }
    assert allocations["claim_boundary_v89"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v89_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v89"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v89"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v89"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v89"])

    action = _read_csv("paper4_v89_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v89"]) == status["added_loan_id_v89"]
    assert str(action_row["dropped_loan_id_v89"]) == status["dropped_loan_id_v89"]
    assert float(action_row["return_delta_v89"]) == pytest.approx(193.93226645730118)
    assert int(action_row["source_cap_violations_after_repair_v89"]) == 0

    source_summary = _read_csv("paper4_v89_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v89",
        "source_slack_v89",
        "source_cap_violated_v89",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v89"]
    assert not source_summary["source_cap_violated_v89"].astype(bool).any()

    blockers = _read_csv("paper4_v89_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v89"], blockers["blocking_v89"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v89"], blockers["evidence_count_v89"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v89_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v89_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v89_repair_candidate_feasible"]) is True
    assert bool(claim_map["v89_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v89_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v89_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v89 fourth one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v89 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v89 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v89: Fourth One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v90_post_v89_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v90_status.json")

    assert status["phase"] == "v90_post_v89_one_swap_reprice"
    assert status["summary_rows_v90"] == 1
    assert status["stage_summary_rows_v90"] == 6
    assert status["candidate_pair_rows_v90"] == 7594
    assert status["top_candidate_rows_v90"] == 200
    assert status["claim_blocker_rows_v90"] == 3
    assert status["selected_rows_v90"] == 171
    assert status["candidate_add_rows_v90"] == 276698
    assert status["total_pair_rows_screened_v90"] == 47315358
    assert status["return_improving_pair_rows_v90"] == 2099862
    assert status["budget_return_feasible_pair_rows_v90"] == 1263338
    assert status["source_prefilter_pair_rows_v90"] == 8738
    assert status["source_exact_pair_rows_v90"] == 7594
    assert status["cvar_feasible_pair_rows_v90"] == 7594
    assert status["one_swap_improving_rows_v90"] == 7594
    assert status["best_one_swap_return_delta_v90"] == pytest.approx(164.41628375317225)
    assert status["best_one_swap_cvar90_after_v90"] == pytest.approx(91931.29075279877)
    assert status["post_repair_one_swap_local_optimality_cleared_v90"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v90"] is False
    assert status["paper1_promotion_allowed_v90"] is False
    assert status["paper4_working_champion_changed_v90"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v90_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v90",
        "dropped_loan_id_v90",
        "return_delta_v90",
        "objective_return_after_swap_v90",
        "budget_swap_feasible_v90",
        "source_swap_feasible_v90",
        "source_cap_violations_after_swap_v90",
        "cvar_swap_feasible_v90",
        "one_swap_improves_return_v90",
        "claim_boundary_v90",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v90"]
    assert probe["return_delta_v90"].gt(0).all()
    assert probe["budget_swap_feasible_v90"].astype(bool).all()
    assert probe["source_swap_feasible_v90"].astype(bool).all()
    assert probe["cvar_swap_feasible_v90"].astype(bool).all()
    assert probe["one_swap_improves_return_v90"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v90"].sum()) == 0
    assert probe["return_delta_v90"].max() == pytest.approx(
        status["best_one_swap_return_delta_v90"]
    )
    assert probe["claim_boundary_v90"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v90_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v90"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v90"]) == "135930178"
    assert str(best["dropped_loan_id_v90"]) == "126739229"
    assert float(best["return_delta_v90"]) == pytest.approx(
        status["best_one_swap_return_delta_v90"]
    )
    assert bool(best["one_swap_improves_return_v90"]) is True

    summary = _read_csv("paper4_v90_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v90"]) == status["one_swap_improving_rows_v90"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v90"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v90"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v90"])

    stage_summary = _read_csv("paper4_v90_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v90", "pair_rows_v90", "claim_boundary_v90"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v90"], stage_summary["pair_rows_v90"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v90"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v90"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v90"]

    blockers = _read_csv("paper4_v90_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v90"], blockers["blocking_v90"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v90"], blockers["evidence_count_v90"], strict=False)
    )
    assert bool(blocker_map["post_v89_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v89_one_swap_improvement_found"]) == 7594
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v90_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v90_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v90_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v90_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v90_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v90 post-v89 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v90 proves the v89 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v90 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v90: Post-v89 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v91_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v91_status.json")

    assert status["phase"] == "v91_next_one_swap_repair"
    assert status["allocation_rows_v91"] == 171
    assert status["summary_rows_v91"] == 1
    assert status["action_rows_v91"] == 1
    assert status["source_summary_rows_v91"] == 51
    assert status["claim_blocker_rows_v91"] == 4
    assert status["added_loan_id_v91"] == "135930178"
    assert status["dropped_loan_id_v91"] == "126739229"
    assert status["selected_rows_v91"] == 171
    assert status["portfolio_exposure_v91"] == pytest.approx(842450.0)
    assert status["objective_return_v91"] == pytest.approx(-3111.089858050349)
    assert status["scenario_loss_cvar90_v91"] == pytest.approx(91931.29075279877)
    assert status["source_cap_violations_v91"] == 0
    assert status["delta_return_vs_v89_v91"] == pytest.approx(164.41628375317123)
    assert status["delta_cvar90_vs_v89_v91"] == pytest.approx(79.30084503945545)
    assert status["delta_exposure_vs_v89_v91"] == pytest.approx(0.0)
    assert status["budget_feasible_v91"] is True
    assert status["source_feasible_v91"] is True
    assert status["cvar_feasible_v91"] is True
    assert status["repair_candidate_feasible_v91"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v91"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v91"] is False
    assert status["paper1_promotion_allowed_v91"] is False
    assert status["paper4_working_champion_changed_v91"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v91_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v91",
        "selected_v91",
        "portfolio_label_v91",
        "repair_action_v91",
        "claim_boundary_v91",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v91"]
    assert int(allocations["selected_v91"].sum()) == status["selected_rows_v91"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v91"])
    assert "135930178" in set(allocations["loan_id"].astype(str))
    assert "126739229" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v91"]) == {
        "added_from_v90_best_swap",
        "kept_from_v89",
    }
    assert allocations["claim_boundary_v91"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v91_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v91"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v91"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v91"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v91"])

    action = _read_csv("paper4_v91_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v91"]) == status["added_loan_id_v91"]
    assert str(action_row["dropped_loan_id_v91"]) == status["dropped_loan_id_v91"]
    assert float(action_row["return_delta_v91"]) == pytest.approx(164.41628375317225)
    assert int(action_row["source_cap_violations_after_repair_v91"]) == 0

    source_summary = _read_csv("paper4_v91_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v91",
        "source_slack_v91",
        "source_cap_violated_v91",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v91"]
    assert not source_summary["source_cap_violated_v91"].astype(bool).any()

    blockers = _read_csv("paper4_v91_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v91"], blockers["blocking_v91"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v91"], blockers["evidence_count_v91"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v91_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v91_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v91_repair_candidate_feasible"]) is True
    assert bool(claim_map["v91_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v91_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v91_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v91 fifth one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v91 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v91 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v91: Fifth One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v92_post_v91_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v92_status.json")

    assert status["phase"] == "v92_post_v91_one_swap_reprice"
    assert status["summary_rows_v92"] == 1
    assert status["stage_summary_rows_v92"] == 6
    assert status["candidate_pair_rows_v92"] == 7563
    assert status["top_candidate_rows_v92"] == 200
    assert status["claim_blocker_rows_v92"] == 3
    assert status["selected_rows_v92"] == 171
    assert status["candidate_add_rows_v92"] == 276698
    assert status["total_pair_rows_screened_v92"] == 47315358
    assert status["return_improving_pair_rows_v92"] == 2088968
    assert status["budget_return_feasible_pair_rows_v92"] == 1257771
    assert status["source_prefilter_pair_rows_v92"] == 8693
    assert status["source_exact_pair_rows_v92"] == 7563
    assert status["cvar_feasible_pair_rows_v92"] == 7563
    assert status["one_swap_improving_rows_v92"] == 7563
    assert status["best_one_swap_return_delta_v92"] == pytest.approx(159.45366133732136)
    assert status["best_one_swap_cvar90_after_v92"] == pytest.approx(91819.71312940198)
    assert status["post_repair_one_swap_local_optimality_cleared_v92"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v92"] is False
    assert status["paper1_promotion_allowed_v92"] is False
    assert status["paper4_working_champion_changed_v92"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v92_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v92",
        "dropped_loan_id_v92",
        "return_delta_v92",
        "objective_return_after_swap_v92",
        "budget_swap_feasible_v92",
        "source_swap_feasible_v92",
        "source_cap_violations_after_swap_v92",
        "cvar_swap_feasible_v92",
        "one_swap_improves_return_v92",
        "claim_boundary_v92",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v92"]
    assert probe["return_delta_v92"].gt(0).all()
    assert probe["budget_swap_feasible_v92"].astype(bool).all()
    assert probe["source_swap_feasible_v92"].astype(bool).all()
    assert probe["cvar_swap_feasible_v92"].astype(bool).all()
    assert probe["one_swap_improves_return_v92"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v92"].sum()) == 0
    assert probe["return_delta_v92"].max() == pytest.approx(
        status["best_one_swap_return_delta_v92"]
    )
    assert probe["claim_boundary_v92"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v92_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v92"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v92"]) == "145248195"
    assert str(best["dropped_loan_id_v92"]) == "126782976"
    assert float(best["return_delta_v92"]) == pytest.approx(
        status["best_one_swap_return_delta_v92"]
    )
    assert bool(best["one_swap_improves_return_v92"]) is True

    summary = _read_csv("paper4_v92_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v92"]) == status["one_swap_improving_rows_v92"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v92"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v92"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v92"])

    stage_summary = _read_csv("paper4_v92_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v92", "pair_rows_v92", "claim_boundary_v92"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v92"], stage_summary["pair_rows_v92"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v92"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v92"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v92"]

    blockers = _read_csv("paper4_v92_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v92"], blockers["blocking_v92"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v92"], blockers["evidence_count_v92"], strict=False)
    )
    assert bool(blocker_map["post_v91_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v91_one_swap_improvement_found"]) == 7563
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v92_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v92_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v92_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v92_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v92_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v92 post-v91 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v92 proves the v91 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v92 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v92: Post-v91 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v93_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v93_status.json")

    assert status["phase"] == "v93_next_one_swap_repair"
    assert status["allocation_rows_v93"] == 171
    assert status["summary_rows_v93"] == 1
    assert status["action_rows_v93"] == 1
    assert status["source_summary_rows_v93"] == 51
    assert status["claim_blocker_rows_v93"] == 4
    assert status["added_loan_id_v93"] == "145248195"
    assert status["dropped_loan_id_v93"] == "126782976"
    assert status["selected_rows_v93"] == 171
    assert status["portfolio_exposure_v93"] == pytest.approx(842450.0)
    assert status["objective_return_v93"] == pytest.approx(-2951.636196713027)
    assert status["scenario_loss_cvar90_v93"] == pytest.approx(91819.71312940196)
    assert status["source_cap_violations_v93"] == 0
    assert status["delta_return_vs_v91_v93"] == pytest.approx(159.45366133732205)
    assert status["delta_cvar90_vs_v91_v93"] == pytest.approx(-111.57762339679175)
    assert status["delta_exposure_vs_v91_v93"] == pytest.approx(0.0)
    assert status["budget_feasible_v93"] is True
    assert status["source_feasible_v93"] is True
    assert status["cvar_feasible_v93"] is True
    assert status["repair_candidate_feasible_v93"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v93"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v93"] is False
    assert status["paper1_promotion_allowed_v93"] is False
    assert status["paper4_working_champion_changed_v93"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v93_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v93",
        "selected_v93",
        "portfolio_label_v93",
        "repair_action_v93",
        "claim_boundary_v93",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v93"]
    assert int(allocations["selected_v93"].sum()) == status["selected_rows_v93"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v93"])
    assert "145248195" in set(allocations["loan_id"].astype(str))
    assert "126782976" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v93"]) == {
        "added_from_v92_best_swap",
        "kept_from_v91",
    }
    assert allocations["claim_boundary_v93"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v93_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v93"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v93"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v93"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v93"])

    action = _read_csv("paper4_v93_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v93"]) == status["added_loan_id_v93"]
    assert str(action_row["dropped_loan_id_v93"]) == status["dropped_loan_id_v93"]
    assert float(action_row["return_delta_v93"]) == pytest.approx(159.45366133732136)
    assert int(action_row["source_cap_violations_after_repair_v93"]) == 0

    source_summary = _read_csv("paper4_v93_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v93",
        "source_slack_v93",
        "source_cap_violated_v93",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v93"]
    assert not source_summary["source_cap_violated_v93"].astype(bool).any()

    blockers = _read_csv("paper4_v93_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v93"], blockers["blocking_v93"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v93"], blockers["evidence_count_v93"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v93_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v93_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v93_repair_candidate_feasible"]) is True
    assert bool(claim_map["v93_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v93_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v93_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v93 sixth one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v93 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v93 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v93: Sixth One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v94_post_v93_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v94_status.json")

    assert status["phase"] == "v94_post_v93_one_swap_reprice"
    assert status["summary_rows_v94"] == 1
    assert status["stage_summary_rows_v94"] == 6
    assert status["candidate_pair_rows_v94"] == 7334
    assert status["top_candidate_rows_v94"] == 200
    assert status["claim_blocker_rows_v94"] == 3
    assert status["selected_rows_v94"] == 171
    assert status["candidate_add_rows_v94"] == 276698
    assert status["total_pair_rows_screened_v94"] == 47315358
    assert status["return_improving_pair_rows_v94"] == 2076873
    assert status["budget_return_feasible_pair_rows_v94"] == 1251997
    assert status["source_prefilter_pair_rows_v94"] == 8416
    assert status["source_exact_pair_rows_v94"] == 7334
    assert status["cvar_feasible_pair_rows_v94"] == 7334
    assert status["one_swap_improving_rows_v94"] == 7334
    assert status["best_one_swap_return_delta_v94"] == pytest.approx(157.0655454317594)
    assert status["best_one_swap_cvar90_after_v94"] == pytest.approx(92095.88635549336)
    assert status["post_repair_one_swap_local_optimality_cleared_v94"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v94"] is False
    assert status["paper1_promotion_allowed_v94"] is False
    assert status["paper4_working_champion_changed_v94"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v94_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v94",
        "dropped_loan_id_v94",
        "return_delta_v94",
        "objective_return_after_swap_v94",
        "budget_swap_feasible_v94",
        "source_swap_feasible_v94",
        "source_cap_violations_after_swap_v94",
        "cvar_swap_feasible_v94",
        "one_swap_improves_return_v94",
        "claim_boundary_v94",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v94"]
    assert probe["return_delta_v94"].gt(0).all()
    assert probe["budget_swap_feasible_v94"].astype(bool).all()
    assert probe["source_swap_feasible_v94"].astype(bool).all()
    assert probe["cvar_swap_feasible_v94"].astype(bool).all()
    assert probe["one_swap_improves_return_v94"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v94"].sum()) == 0
    assert probe["return_delta_v94"].max() == pytest.approx(
        status["best_one_swap_return_delta_v94"]
    )
    assert probe["claim_boundary_v94"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v94_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v94"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v94"]) == "152314563"
    assert str(best["dropped_loan_id_v94"]) == "127844421"
    assert float(best["return_delta_v94"]) == pytest.approx(
        status["best_one_swap_return_delta_v94"]
    )
    assert bool(best["one_swap_improves_return_v94"]) is True

    summary = _read_csv("paper4_v94_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v94"]) == status["one_swap_improving_rows_v94"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v94"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v94"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v94"])

    stage_summary = _read_csv("paper4_v94_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v94", "pair_rows_v94", "claim_boundary_v94"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v94"], stage_summary["pair_rows_v94"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v94"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v94"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v94"]

    blockers = _read_csv("paper4_v94_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v94"], blockers["blocking_v94"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v94"], blockers["evidence_count_v94"], strict=False)
    )
    assert bool(blocker_map["post_v93_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v93_one_swap_improvement_found"]) == 7334
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v94_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v94_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v94_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v94_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v94_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v94 post-v93 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v94 proves the v93 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v94 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v94: Post-v93 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v95_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v95_status.json")

    assert status["phase"] == "v95_next_one_swap_repair"
    assert status["allocation_rows_v95"] == 171
    assert status["summary_rows_v95"] == 1
    assert status["action_rows_v95"] == 1
    assert status["source_summary_rows_v95"] == 51
    assert status["claim_blocker_rows_v95"] == 4
    assert status["added_loan_id_v95"] == "152314563"
    assert status["dropped_loan_id_v95"] == "127844421"
    assert status["selected_rows_v95"] == 171
    assert status["portfolio_exposure_v95"] == pytest.approx(842450.0)
    assert status["objective_return_v95"] == pytest.approx(-2794.570651281263)
    assert status["scenario_loss_cvar90_v95"] == pytest.approx(92095.88635549332)
    assert status["source_cap_violations_v95"] == 0
    assert status["delta_return_vs_v93_v95"] == pytest.approx(157.0655454317639)
    assert status["delta_cvar90_vs_v93_v95"] == pytest.approx(276.1732260913559)
    assert status["delta_exposure_vs_v93_v95"] == pytest.approx(0.0)
    assert status["budget_feasible_v95"] is True
    assert status["source_feasible_v95"] is True
    assert status["cvar_feasible_v95"] is True
    assert status["repair_candidate_feasible_v95"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v95"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v95"] is False
    assert status["paper1_promotion_allowed_v95"] is False
    assert status["paper4_working_champion_changed_v95"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v95_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v95",
        "selected_v95",
        "portfolio_label_v95",
        "repair_action_v95",
        "claim_boundary_v95",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v95"]
    assert int(allocations["selected_v95"].sum()) == status["selected_rows_v95"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v95"])
    assert "152314563" in set(allocations["loan_id"].astype(str))
    assert "127844421" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v95"]) == {
        "added_from_v94_best_swap",
        "kept_from_v93",
    }
    assert allocations["claim_boundary_v95"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v95_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v95"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v95"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v95"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v95"])

    action = _read_csv("paper4_v95_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v95"]) == status["added_loan_id_v95"]
    assert str(action_row["dropped_loan_id_v95"]) == status["dropped_loan_id_v95"]
    assert float(action_row["return_delta_v95"]) == pytest.approx(157.0655454317594)
    assert int(action_row["source_cap_violations_after_repair_v95"]) == 0

    source_summary = _read_csv("paper4_v95_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v95",
        "source_slack_v95",
        "source_cap_violated_v95",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v95"]
    assert not source_summary["source_cap_violated_v95"].astype(bool).any()

    blockers = _read_csv("paper4_v95_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v95"], blockers["blocking_v95"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v95"], blockers["evidence_count_v95"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v95_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v95_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v95_repair_candidate_feasible"]) is True
    assert bool(claim_map["v95_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v95_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v95_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v95 seventh one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v95 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v95 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v95: Seventh One-Swap Repair Candidate" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v96_post_v95_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v96_status.json")

    assert status["phase"] == "v96_post_v95_one_swap_reprice"
    assert status["summary_rows_v96"] == 1
    assert status["stage_summary_rows_v96"] == 6
    assert status["candidate_pair_rows_v96"] == 7239
    assert status["top_candidate_rows_v96"] == 200
    assert status["claim_blocker_rows_v96"] == 3
    assert status["selected_rows_v96"] == 171
    assert status["candidate_add_rows_v96"] == 276698
    assert status["total_pair_rows_screened_v96"] == 47315358
    assert status["return_improving_pair_rows_v96"] == 2066025
    assert status["budget_return_feasible_pair_rows_v96"] == 1247288
    assert status["source_prefilter_pair_rows_v96"] == 8318
    assert status["source_exact_pair_rows_v96"] == 7239
    assert status["cvar_feasible_pair_rows_v96"] == 7239
    assert status["one_swap_improving_rows_v96"] == 7239
    assert status["best_one_swap_return_delta_v96"] == pytest.approx(152.74243610682214)
    assert status["best_one_swap_cvar90_after_v96"] == pytest.approx(92221.79683903579)
    assert status["post_repair_one_swap_local_optimality_cleared_v96"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v96"] is False
    assert status["paper1_promotion_allowed_v96"] is False
    assert status["paper4_working_champion_changed_v96"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v96_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v96",
        "dropped_loan_id_v96",
        "return_delta_v96",
        "objective_return_after_swap_v96",
        "budget_swap_feasible_v96",
        "source_swap_feasible_v96",
        "source_cap_violations_after_swap_v96",
        "cvar_swap_feasible_v96",
        "one_swap_improves_return_v96",
        "claim_boundary_v96",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v96"]
    assert probe["return_delta_v96"].gt(0).all()
    assert probe["budget_swap_feasible_v96"].astype(bool).all()
    assert probe["source_swap_feasible_v96"].astype(bool).all()
    assert probe["cvar_swap_feasible_v96"].astype(bool).all()
    assert probe["one_swap_improves_return_v96"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v96"].sum()) == 0
    assert probe["return_delta_v96"].max() == pytest.approx(
        status["best_one_swap_return_delta_v96"]
    )
    assert probe["claim_boundary_v96"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v96_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v96"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v96"]) == "142037168"
    assert str(best["dropped_loan_id_v96"]) == "126777878"
    assert float(best["return_delta_v96"]) == pytest.approx(
        status["best_one_swap_return_delta_v96"]
    )
    assert bool(best["one_swap_improves_return_v96"]) is True

    summary = _read_csv("paper4_v96_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v96"]) == status["one_swap_improving_rows_v96"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v96"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v96"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v96"])

    stage_summary = _read_csv("paper4_v96_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v96", "pair_rows_v96", "claim_boundary_v96"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v96"], stage_summary["pair_rows_v96"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v96"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v96"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v96"]

    blockers = _read_csv("paper4_v96_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v96"], blockers["blocking_v96"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v96"], blockers["evidence_count_v96"], strict=False)
    )
    assert bool(blocker_map["post_v95_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v95_one_swap_improvement_found"]) == 7239
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v96_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v96_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v96_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v96_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v96_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v96 post-v95 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v96 proves the v95 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v96 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v96: Post-v95 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v97_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v97_status.json")

    assert status["phase"] == "v97_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.97"
    assert status["allocation_rows_v97"] == 171
    assert status["summary_rows_v97"] == 1
    assert status["action_rows_v97"] == 1
    assert status["source_summary_rows_v97"] == 51
    assert status["claim_blocker_rows_v97"] == 4
    assert status["added_loan_id_v97"] == "142037168"
    assert status["dropped_loan_id_v97"] == "126777878"
    assert status["selected_rows_v97"] == 171
    assert status["portfolio_exposure_v97"] == pytest.approx(842450.0)
    assert status["objective_return_v97"] == pytest.approx(-2641.8282151744443)
    assert status["scenario_loss_cvar90_v97"] == pytest.approx(92221.79683903579)
    assert status["source_cap_violations_v97"] == 0
    assert status["delta_return_vs_v95_v97"] == pytest.approx(152.7424361068188)
    assert status["delta_cvar90_vs_v95_v97"] == pytest.approx(125.91048354246595)
    assert status["delta_exposure_vs_v95_v97"] == pytest.approx(0.0)
    assert status["budget_feasible_v97"] is True
    assert status["source_feasible_v97"] is True
    assert status["cvar_feasible_v97"] is True
    assert status["repair_candidate_feasible_v97"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v97"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v97"] is False
    assert status["paper1_promotion_allowed_v97"] is False
    assert status["paper4_working_champion_changed_v97"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v97_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v97",
        "selected_v97",
        "portfolio_label_v97",
        "repair_action_v97",
        "claim_boundary_v97",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v97"]
    assert int(allocations["selected_v97"].sum()) == status["selected_rows_v97"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v97"])
    assert "142037168" in set(allocations["loan_id"].astype(str))
    assert "126777878" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v97"]) == {
        "added_from_v96_best_swap",
        "kept_from_v95",
    }
    assert allocations["claim_boundary_v97"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v97_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v97"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v97"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v97"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v97"])

    action = _read_csv("paper4_v97_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v97"]) == status["added_loan_id_v97"]
    assert str(action_row["dropped_loan_id_v97"]) == status["dropped_loan_id_v97"]
    assert float(action_row["return_delta_v97"]) == pytest.approx(152.74243610682214)
    assert int(action_row["source_cap_violations_after_repair_v97"]) == 0

    source_summary = _read_csv("paper4_v97_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v97",
        "source_slack_v97",
        "source_cap_violated_v97",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v97"]
    assert not source_summary["source_cap_violated_v97"].astype(bool).any()

    blockers = _read_csv("paper4_v97_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v97"], blockers["blocking_v97"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v97"], blockers["evidence_count_v97"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v97_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v97_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v97_repair_candidate_feasible"]) is True
    assert bool(claim_map["v97_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v97_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v97_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v97 eighth one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v97 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v97 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v97: Eighth One-Swap Repair Candidate" in notebook
    assert "next required experiment is v98" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v98_post_v97_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v98_status.json")

    assert status["phase"] == "v98_post_v97_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.98"
    assert status["summary_rows_v98"] == 1
    assert status["stage_summary_rows_v98"] == 6
    assert status["candidate_pair_rows_v98"] == 6783
    assert status["top_candidate_rows_v98"] == 200
    assert status["claim_blocker_rows_v98"] == 3
    assert status["selected_rows_v98"] == 171
    assert status["candidate_add_rows_v98"] == 276698
    assert status["total_pair_rows_screened_v98"] == 47315358
    assert status["return_improving_pair_rows_v98"] == 2055524
    assert status["budget_return_feasible_pair_rows_v98"] == 1240100
    assert status["source_prefilter_pair_rows_v98"] == 7862
    assert status["source_exact_pair_rows_v98"] == 6783
    assert status["cvar_feasible_pair_rows_v98"] == 6783
    assert status["one_swap_improving_rows_v98"] == 6783
    assert status["best_one_swap_return_delta_v98"] == pytest.approx(146.68859776698605)
    assert status["best_one_swap_cvar90_after_v98"] == pytest.approx(92333.23436577116)
    assert status["post_repair_one_swap_local_optimality_cleared_v98"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v98"] is False
    assert status["paper1_promotion_allowed_v98"] is False
    assert status["paper4_working_champion_changed_v98"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v98_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v98",
        "dropped_loan_id_v98",
        "return_delta_v98",
        "objective_return_after_swap_v98",
        "budget_swap_feasible_v98",
        "source_swap_feasible_v98",
        "source_cap_violations_after_swap_v98",
        "cvar_swap_feasible_v98",
        "one_swap_improves_return_v98",
        "claim_boundary_v98",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v98"]
    assert probe["return_delta_v98"].gt(0).all()
    assert probe["budget_swap_feasible_v98"].astype(bool).all()
    assert probe["source_swap_feasible_v98"].astype(bool).all()
    assert probe["cvar_swap_feasible_v98"].astype(bool).all()
    assert probe["one_swap_improves_return_v98"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v98"].sum()) == 0
    assert probe["return_delta_v98"].max() == pytest.approx(
        status["best_one_swap_return_delta_v98"]
    )
    assert probe["claim_boundary_v98"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v98_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v98"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v98"]) == "128253688"
    assert str(best["dropped_loan_id_v98"]) == "127051964"
    assert float(best["return_delta_v98"]) == pytest.approx(
        status["best_one_swap_return_delta_v98"]
    )
    assert float(best["exposure_after_swap_v98"]) == pytest.approx(842650.0)
    assert bool(best["one_swap_improves_return_v98"]) is True

    summary = _read_csv("paper4_v98_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v98"]) == status["one_swap_improving_rows_v98"]
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v98"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v98"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v98"])

    stage_summary = _read_csv("paper4_v98_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v98", "pair_rows_v98", "claim_boundary_v98"}.issubset(stage_summary.columns)
    stage_map = dict(zip(stage_summary["stage_v98"], stage_summary["pair_rows_v98"], strict=False))
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v98"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v98"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v98"]

    blockers = _read_csv("paper4_v98_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v98"], blockers["blocking_v98"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v98"], blockers["evidence_count_v98"], strict=False)
    )
    assert bool(blocker_map["post_v97_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v97_one_swap_improvement_found"]) == 6783
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v98_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v98_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v98_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v98_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v98_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v98 post-v97 one-swap pricing screen." in set(current_boundaries["claim"])
    assert "v98 proves the v97 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v98 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v98: Post-v97 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v99_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v99_status.json")

    assert status["phase"] == "v99_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.99"
    assert status["allocation_rows_v99"] == 171
    assert status["summary_rows_v99"] == 1
    assert status["action_rows_v99"] == 1
    assert status["source_summary_rows_v99"] == 51
    assert status["claim_blocker_rows_v99"] == 4
    assert status["added_loan_id_v99"] == "128253688"
    assert status["dropped_loan_id_v99"] == "127051964"
    assert status["selected_rows_v99"] == 171
    assert status["portfolio_exposure_v99"] == pytest.approx(842650.0)
    assert status["objective_return_v99"] == pytest.approx(-2495.13961740746)
    assert status["scenario_loss_cvar90_v99"] == pytest.approx(92333.23436577116)
    assert status["source_cap_violations_v99"] == 0
    assert status["delta_return_vs_v97_v99"] == pytest.approx(146.68859776698446)
    assert status["delta_cvar90_vs_v97_v99"] == pytest.approx(111.43752673536073)
    assert status["delta_exposure_vs_v97_v99"] == pytest.approx(200.0)
    assert status["budget_feasible_v99"] is True
    assert status["source_feasible_v99"] is True
    assert status["cvar_feasible_v99"] is True
    assert status["repair_candidate_feasible_v99"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v99"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v99"] is False
    assert status["paper1_promotion_allowed_v99"] is False
    assert status["paper4_working_champion_changed_v99"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(TABLE_DIR / "paper4_v99_next_one_swap_repair_allocations.parquet")
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v99",
        "selected_v99",
        "portfolio_label_v99",
        "repair_action_v99",
        "claim_boundary_v99",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v99"]
    assert int(allocations["selected_v99"].sum()) == status["selected_rows_v99"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v99"])
    assert "128253688" in set(allocations["loan_id"].astype(str))
    assert "127051964" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v99"]) == {
        "added_from_v98_best_swap",
        "kept_from_v97",
    }
    assert allocations["claim_boundary_v99"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v99_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v99"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v99"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v99"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v99"])

    action = _read_csv("paper4_v99_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v99"]) == status["added_loan_id_v99"]
    assert str(action_row["dropped_loan_id_v99"]) == status["dropped_loan_id_v99"]
    assert float(action_row["return_delta_v99"]) == pytest.approx(146.68859776698605)
    assert int(action_row["source_cap_violations_after_repair_v99"]) == 0

    source_summary = _read_csv("paper4_v99_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v99",
        "source_slack_v99",
        "source_cap_violated_v99",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v99"]
    assert not source_summary["source_cap_violated_v99"].astype(bool).any()

    blockers = _read_csv("paper4_v99_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v99"], blockers["blocking_v99"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v99"], blockers["evidence_count_v99"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v99_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v99_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v99_repair_candidate_feasible"]) is True
    assert bool(claim_map["v99_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v99_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v99_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v99 ninth one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v99 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v99 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v99: Ninth One-Swap Repair Candidate" in notebook
    assert "next required experiment is v100" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v100_post_v99_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v100_status.json")

    assert status["phase"] == "v100_post_v99_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.100"
    assert status["summary_rows_v100"] == 1
    assert status["stage_summary_rows_v100"] == 6
    assert status["candidate_pair_rows_v100"] == 10946
    assert status["top_candidate_rows_v100"] == 200
    assert status["claim_blocker_rows_v100"] == 3
    assert status["selected_rows_v100"] == 171
    assert status["candidate_add_rows_v100"] == 276698
    assert status["total_pair_rows_screened_v100"] == 47315358
    assert status["return_improving_pair_rows_v100"] == 2046327
    assert status["budget_return_feasible_pair_rows_v100"] == 1234402
    assert status["source_prefilter_pair_rows_v100"] == 18046
    assert status["source_exact_pair_rows_v100"] == 10946
    assert status["cvar_feasible_pair_rows_v100"] == 10946
    assert status["one_swap_improving_rows_v100"] == 10946
    assert status["best_one_swap_return_delta_v100"] == pytest.approx(191.1881191206193)
    assert status["best_one_swap_cvar90_after_v100"] == pytest.approx(92311.43781816204)
    assert status["post_repair_one_swap_local_optimality_cleared_v100"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v100"] is False
    assert status["paper1_promotion_allowed_v100"] is False
    assert status["paper4_working_champion_changed_v100"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v100_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v100",
        "dropped_loan_id_v100",
        "return_delta_v100",
        "objective_return_after_swap_v100",
        "budget_swap_feasible_v100",
        "source_swap_feasible_v100",
        "source_cap_violations_after_swap_v100",
        "cvar_swap_feasible_v100",
        "one_swap_improves_return_v100",
        "claim_boundary_v100",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v100"]
    assert probe["return_delta_v100"].gt(0).all()
    assert probe["budget_swap_feasible_v100"].astype(bool).all()
    assert probe["source_swap_feasible_v100"].astype(bool).all()
    assert probe["cvar_swap_feasible_v100"].astype(bool).all()
    assert probe["one_swap_improves_return_v100"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v100"].sum()) == 0
    assert probe["return_delta_v100"].max() == pytest.approx(
        status["best_one_swap_return_delta_v100"]
    )
    assert probe["claim_boundary_v100"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v100_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v100"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v100"]) == "142000768"
    assert str(best["dropped_loan_id_v100"]) == "127061180"
    assert float(best["return_delta_v100"]) == pytest.approx(
        status["best_one_swap_return_delta_v100"]
    )
    assert float(best["exposure_after_swap_v100"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v100"]) is True

    summary = _read_csv("paper4_v100_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v100"]) == status["one_swap_improving_rows_v100"]
    assert float(row["current_exposure_v100"]) == pytest.approx(842650.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v100"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v100"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v100"])

    stage_summary = _read_csv("paper4_v100_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v100", "pair_rows_v100", "claim_boundary_v100"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v100"], stage_summary["pair_rows_v100"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v100"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v100"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v100"]

    blockers = _read_csv("paper4_v100_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v100"], blockers["blocking_v100"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v100"], blockers["evidence_count_v100"], strict=False)
    )
    assert bool(blocker_map["post_v99_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v99_one_swap_improvement_found"]) == 10946
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v100_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v100_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v100_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v100_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v100_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v100 post-v99 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v100 proves the v99 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v100 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v100: Post-v99 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v101_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v101_status.json")

    assert status["phase"] == "v101_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.101"
    assert status["allocation_rows_v101"] == 171
    assert status["summary_rows_v101"] == 1
    assert status["action_rows_v101"] == 1
    assert status["source_summary_rows_v101"] == 51
    assert status["claim_blocker_rows_v101"] == 4
    assert status["added_loan_id_v101"] == "142000768"
    assert status["dropped_loan_id_v101"] == "127061180"
    assert status["selected_rows_v101"] == 171
    assert status["portfolio_exposure_v101"] == pytest.approx(842450.0)
    assert status["objective_return_v101"] == pytest.approx(-2303.9514982868386)
    assert status["scenario_loss_cvar90_v101"] == pytest.approx(92311.43781816206)
    assert status["source_cap_violations_v101"] == 0
    assert status["delta_return_vs_v99_v101"] == pytest.approx(191.1881191206212)
    assert status["delta_cvar90_vs_v99_v101"] == pytest.approx(-21.79654760910489)
    assert status["delta_exposure_vs_v99_v101"] == pytest.approx(-200.0)
    assert status["budget_feasible_v101"] is True
    assert status["source_feasible_v101"] is True
    assert status["cvar_feasible_v101"] is True
    assert status["repair_candidate_feasible_v101"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v101"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v101"] is False
    assert status["paper1_promotion_allowed_v101"] is False
    assert status["paper4_working_champion_changed_v101"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v101_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v101",
        "selected_v101",
        "portfolio_label_v101",
        "repair_action_v101",
        "claim_boundary_v101",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v101"]
    assert int(allocations["selected_v101"].sum()) == status["selected_rows_v101"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v101"])
    assert "142000768" in set(allocations["loan_id"].astype(str))
    assert "127061180" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v101"]) == {
        "added_from_v100_best_swap",
        "kept_from_v99",
    }
    assert allocations["claim_boundary_v101"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v101_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v101"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v101"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v101"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v101"])

    action = _read_csv("paper4_v101_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v101"]) == status["added_loan_id_v101"]
    assert str(action_row["dropped_loan_id_v101"]) == status["dropped_loan_id_v101"]
    assert float(action_row["return_delta_v101"]) == pytest.approx(191.1881191206193)
    assert int(action_row["source_cap_violations_after_repair_v101"]) == 0

    source_summary = _read_csv("paper4_v101_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v101",
        "source_slack_v101",
        "source_cap_violated_v101",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v101"]
    assert not source_summary["source_cap_violated_v101"].astype(bool).any()

    blockers = _read_csv("paper4_v101_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v101"], blockers["blocking_v101"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v101"], blockers["evidence_count_v101"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v101_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v101_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v101_repair_candidate_feasible"]) is True
    assert bool(claim_map["v101_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v101_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v101_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v101 tenth one-swap repair candidate." in set(current_boundaries["claim"])
    assert "v101 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v101 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v101: Tenth One-Swap Repair Candidate" in notebook
    assert "v102 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v102_post_v101_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v102_status.json")

    assert status["phase"] == "v102_post_v101_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.102"
    assert status["summary_rows_v102"] == 1
    assert status["stage_summary_rows_v102"] == 6
    assert status["candidate_pair_rows_v102"] == 6481
    assert status["top_candidate_rows_v102"] == 200
    assert status["claim_blocker_rows_v102"] == 3
    assert status["selected_rows_v102"] == 171
    assert status["candidate_add_rows_v102"] == 276698
    assert status["total_pair_rows_screened_v102"] == 47315358
    assert status["return_improving_pair_rows_v102"] == 2032532
    assert status["budget_return_feasible_pair_rows_v102"] == 1229892
    assert status["source_prefilter_pair_rows_v102"] == 7612
    assert status["source_exact_pair_rows_v102"] == 6481
    assert status["cvar_feasible_pair_rows_v102"] == 6481
    assert status["one_swap_improving_rows_v102"] == 6481
    assert status["best_one_swap_return_delta_v102"] == pytest.approx(146.6809005878472)
    assert status["best_one_swap_cvar90_after_v102"] == pytest.approx(92438.43605767138)
    assert status["post_repair_one_swap_local_optimality_cleared_v102"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v102"] is False
    assert status["paper1_promotion_allowed_v102"] is False
    assert status["paper4_working_champion_changed_v102"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v102_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v102",
        "dropped_loan_id_v102",
        "return_delta_v102",
        "objective_return_after_swap_v102",
        "budget_swap_feasible_v102",
        "source_swap_feasible_v102",
        "source_cap_violations_after_swap_v102",
        "cvar_swap_feasible_v102",
        "one_swap_improves_return_v102",
        "claim_boundary_v102",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v102"]
    assert probe["return_delta_v102"].gt(0).all()
    assert probe["budget_swap_feasible_v102"].astype(bool).all()
    assert probe["source_swap_feasible_v102"].astype(bool).all()
    assert probe["cvar_swap_feasible_v102"].astype(bool).all()
    assert probe["one_swap_improves_return_v102"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v102"].sum()) == 0
    assert probe["return_delta_v102"].max() == pytest.approx(
        status["best_one_swap_return_delta_v102"]
    )
    assert probe["claim_boundary_v102"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v102_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v102"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v102"]) == "159720541"
    assert str(best["dropped_loan_id_v102"]) == "128000272"
    assert float(best["return_delta_v102"]) == pytest.approx(
        status["best_one_swap_return_delta_v102"]
    )
    assert float(best["exposure_after_swap_v102"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v102"]) is True

    summary = _read_csv("paper4_v102_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v102"]) == status["one_swap_improving_rows_v102"]
    assert float(row["current_exposure_v102"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v102"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v102"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v102"])

    stage_summary = _read_csv("paper4_v102_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v102", "pair_rows_v102", "claim_boundary_v102"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v102"], stage_summary["pair_rows_v102"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v102"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v102"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v102"]

    blockers = _read_csv("paper4_v102_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v102"], blockers["blocking_v102"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v102"], blockers["evidence_count_v102"], strict=False)
    )
    assert bool(blocker_map["post_v101_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v101_one_swap_improvement_found"]) == 6481
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v102_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v102_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v102_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v102_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v102_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v102 post-v101 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v102 proves the v101 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v102 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v102: Post-v101 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v103_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v103_status.json")

    assert status["phase"] == "v103_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.103"
    assert status["allocation_rows_v103"] == 171
    assert status["summary_rows_v103"] == 1
    assert status["action_rows_v103"] == 1
    assert status["source_summary_rows_v103"] == 51
    assert status["claim_blocker_rows_v103"] == 4
    assert status["added_loan_id_v103"] == "159720541"
    assert status["dropped_loan_id_v103"] == "128000272"
    assert status["selected_rows_v103"] == 171
    assert status["portfolio_exposure_v103"] == pytest.approx(842450.0)
    assert status["objective_return_v103"] == pytest.approx(-2157.2705976989946)
    assert status["scenario_loss_cvar90_v103"] == pytest.approx(92438.43605767141)
    assert status["source_cap_violations_v103"] == 0
    assert status["delta_return_vs_v101_v103"] == pytest.approx(146.680900587844)
    assert status["delta_cvar90_vs_v101_v103"] == pytest.approx(126.99823950935388)
    assert status["delta_exposure_vs_v101_v103"] == pytest.approx(0.0)
    assert status["budget_feasible_v103"] is True
    assert status["source_feasible_v103"] is True
    assert status["cvar_feasible_v103"] is True
    assert status["repair_candidate_feasible_v103"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v103"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v103"] is False
    assert status["paper1_promotion_allowed_v103"] is False
    assert status["paper4_working_champion_changed_v103"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v103_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v103",
        "selected_v103",
        "portfolio_label_v103",
        "repair_action_v103",
        "claim_boundary_v103",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v103"]
    assert int(allocations["selected_v103"].sum()) == status["selected_rows_v103"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v103"])
    assert "159720541" in set(allocations["loan_id"].astype(str))
    assert "128000272" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v103"]) == {
        "added_from_v102_best_swap",
        "kept_from_v101",
    }
    assert allocations["claim_boundary_v103"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v103_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v103"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v103"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v103"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v103"])

    action = _read_csv("paper4_v103_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v103"]) == status["added_loan_id_v103"]
    assert str(action_row["dropped_loan_id_v103"]) == status["dropped_loan_id_v103"]
    assert float(action_row["return_delta_v103"]) == pytest.approx(146.6809005878472)
    assert int(action_row["source_cap_violations_after_repair_v103"]) == 0

    source_summary = _read_csv("paper4_v103_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v103",
        "source_slack_v103",
        "source_cap_violated_v103",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v103"]
    assert not source_summary["source_cap_violated_v103"].astype(bool).any()

    blockers = _read_csv("paper4_v103_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v103"], blockers["blocking_v103"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v103"], blockers["evidence_count_v103"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v103_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v103_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v103_repair_candidate_feasible"]) is True
    assert bool(claim_map["v103_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v103_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v103_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v103 eleventh one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v103 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v103 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v103: Eleventh One-Swap Repair Candidate" in notebook
    assert "v104 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v104_post_v103_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v104_status.json")

    assert status["phase"] == "v104_post_v103_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.104"
    assert status["summary_rows_v104"] == 1
    assert status["stage_summary_rows_v104"] == 6
    assert status["candidate_pair_rows_v104"] == 6398
    assert status["top_candidate_rows_v104"] == 200
    assert status["claim_blocker_rows_v104"] == 3
    assert status["selected_rows_v104"] == 171
    assert status["candidate_add_rows_v104"] == 276698
    assert status["total_pair_rows_screened_v104"] == 47315358
    assert status["return_improving_pair_rows_v104"] == 2022487
    assert status["budget_return_feasible_pair_rows_v104"] == 1225519
    assert status["source_prefilter_pair_rows_v104"] == 7524
    assert status["source_exact_pair_rows_v104"] == 6398
    assert status["cvar_feasible_pair_rows_v104"] == 6398
    assert status["one_swap_improving_rows_v104"] == 6398
    assert status["best_one_swap_return_delta_v104"] == pytest.approx(145.24803479210811)
    assert status["best_one_swap_cvar90_after_v104"] == pytest.approx(92582.02702682154)
    assert status["post_repair_one_swap_local_optimality_cleared_v104"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v104"] is False
    assert status["paper1_promotion_allowed_v104"] is False
    assert status["paper4_working_champion_changed_v104"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v104_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v104",
        "dropped_loan_id_v104",
        "return_delta_v104",
        "objective_return_after_swap_v104",
        "budget_swap_feasible_v104",
        "source_swap_feasible_v104",
        "source_cap_violations_after_swap_v104",
        "cvar_swap_feasible_v104",
        "one_swap_improves_return_v104",
        "claim_boundary_v104",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v104"]
    assert probe["return_delta_v104"].gt(0).all()
    assert probe["budget_swap_feasible_v104"].astype(bool).all()
    assert probe["source_swap_feasible_v104"].astype(bool).all()
    assert probe["cvar_swap_feasible_v104"].astype(bool).all()
    assert probe["one_swap_improves_return_v104"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v104"].sum()) == 0
    assert probe["return_delta_v104"].max() == pytest.approx(
        status["best_one_swap_return_delta_v104"]
    )
    assert probe["claim_boundary_v104"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v104_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v104"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v104"]) == "156935428"
    assert str(best["dropped_loan_id_v104"]) == "126821354"
    assert float(best["return_delta_v104"]) == pytest.approx(
        status["best_one_swap_return_delta_v104"]
    )
    assert float(best["exposure_after_swap_v104"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v104"]) is True

    summary = _read_csv("paper4_v104_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v104"]) == status["one_swap_improving_rows_v104"]
    assert float(row["current_exposure_v104"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v104"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v104"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v104"])

    stage_summary = _read_csv("paper4_v104_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v104", "pair_rows_v104", "claim_boundary_v104"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v104"], stage_summary["pair_rows_v104"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v104"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v104"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v104"]

    blockers = _read_csv("paper4_v104_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v104"], blockers["blocking_v104"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v104"], blockers["evidence_count_v104"], strict=False)
    )
    assert bool(blocker_map["post_v103_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v103_one_swap_improvement_found"]) == 6398
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v104_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v104_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v104_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v104_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v104_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v104 post-v103 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v104 proves the v103 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v104 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v104: Post-v103 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v105_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v105_status.json")

    assert status["phase"] == "v105_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.105"
    assert status["allocation_rows_v105"] == 171
    assert status["summary_rows_v105"] == 1
    assert status["action_rows_v105"] == 1
    assert status["source_summary_rows_v105"] == 51
    assert status["claim_blocker_rows_v105"] == 4
    assert status["added_loan_id_v105"] == "156935428"
    assert status["dropped_loan_id_v105"] == "126821354"
    assert status["selected_rows_v105"] == 171
    assert status["portfolio_exposure_v105"] == pytest.approx(842450.0)
    assert status["objective_return_v105"] == pytest.approx(-2012.022562906881)
    assert status["scenario_loss_cvar90_v105"] == pytest.approx(92582.02702682154)
    assert status["source_cap_violations_v105"] == 0
    assert status["delta_return_vs_v103_v105"] == pytest.approx(145.24803479211369)
    assert status["delta_cvar90_vs_v103_v105"] == pytest.approx(143.5909691501438)
    assert status["delta_exposure_vs_v103_v105"] == pytest.approx(0.0)
    assert status["budget_feasible_v105"] is True
    assert status["source_feasible_v105"] is True
    assert status["cvar_feasible_v105"] is True
    assert status["repair_candidate_feasible_v105"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v105"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v105"] is False
    assert status["paper1_promotion_allowed_v105"] is False
    assert status["paper4_working_champion_changed_v105"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v105_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v105",
        "selected_v105",
        "portfolio_label_v105",
        "repair_action_v105",
        "claim_boundary_v105",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v105"]
    assert int(allocations["selected_v105"].sum()) == status["selected_rows_v105"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v105"])
    assert "156935428" in set(allocations["loan_id"].astype(str))
    assert "126821354" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v105"]) == {
        "added_from_v104_best_swap",
        "kept_from_v103",
    }
    assert allocations["claim_boundary_v105"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v105_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v105"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v105"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v105"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v105"])

    action = _read_csv("paper4_v105_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v105"]) == status["added_loan_id_v105"]
    assert str(action_row["dropped_loan_id_v105"]) == status["dropped_loan_id_v105"]
    assert float(action_row["return_delta_v105"]) == pytest.approx(145.24803479210811)
    assert int(action_row["source_cap_violations_after_repair_v105"]) == 0

    source_summary = _read_csv("paper4_v105_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v105",
        "source_slack_v105",
        "source_cap_violated_v105",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v105"]
    assert not source_summary["source_cap_violated_v105"].astype(bool).any()

    blockers = _read_csv("paper4_v105_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v105"], blockers["blocking_v105"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v105"], blockers["evidence_count_v105"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v105_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v105_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v105_repair_candidate_feasible"]) is True
    assert bool(claim_map["v105_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v105_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v105_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v105 twelfth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v105 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v105 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v105: Twelfth One-Swap Repair Candidate" in notebook
    assert "v106 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v106_post_v105_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v106_status.json")

    assert status["phase"] == "v106_post_v105_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.106"
    assert status["summary_rows_v106"] == 1
    assert status["stage_summary_rows_v106"] == 6
    assert status["candidate_pair_rows_v106"] == 6317
    assert status["top_candidate_rows_v106"] == 200
    assert status["claim_blocker_rows_v106"] == 3
    assert status["selected_rows_v106"] == 171
    assert status["candidate_add_rows_v106"] == 276698
    assert status["total_pair_rows_screened_v106"] == 47315358
    assert status["return_improving_pair_rows_v106"] == 2012547
    assert status["budget_return_feasible_pair_rows_v106"] == 1221190
    assert status["source_prefilter_pair_rows_v106"] == 7438
    assert status["source_exact_pair_rows_v106"] == 6317
    assert status["cvar_feasible_pair_rows_v106"] == 6317
    assert status["one_swap_improving_rows_v106"] == 6317
    assert status["best_one_swap_return_delta_v106"] == pytest.approx(142.09376124853478)
    assert status["best_one_swap_cvar90_after_v106"] == pytest.approx(92734.88809495818)
    assert status["post_repair_one_swap_local_optimality_cleared_v106"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v106"] is False
    assert status["paper1_promotion_allowed_v106"] is False
    assert status["paper4_working_champion_changed_v106"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v106_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v106",
        "dropped_loan_id_v106",
        "return_delta_v106",
        "objective_return_after_swap_v106",
        "budget_swap_feasible_v106",
        "source_swap_feasible_v106",
        "source_cap_violations_after_swap_v106",
        "cvar_swap_feasible_v106",
        "one_swap_improves_return_v106",
        "claim_boundary_v106",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v106"]
    assert probe["return_delta_v106"].gt(0).all()
    assert probe["budget_swap_feasible_v106"].astype(bool).all()
    assert probe["source_swap_feasible_v106"].astype(bool).all()
    assert probe["cvar_swap_feasible_v106"].astype(bool).all()
    assert probe["one_swap_improves_return_v106"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v106"].sum()) == 0
    assert probe["return_delta_v106"].max() == pytest.approx(
        status["best_one_swap_return_delta_v106"]
    )
    assert probe["claim_boundary_v106"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v106_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v106"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v106"]) == "155861848"
    assert str(best["dropped_loan_id_v106"]) == "126633227"
    assert float(best["return_delta_v106"]) == pytest.approx(
        status["best_one_swap_return_delta_v106"]
    )
    assert float(best["exposure_after_swap_v106"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v106"]) is True

    summary = _read_csv("paper4_v106_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v106"]) == status["one_swap_improving_rows_v106"]
    assert float(row["current_exposure_v106"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v106"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v106"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v106"])

    stage_summary = _read_csv("paper4_v106_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v106", "pair_rows_v106", "claim_boundary_v106"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v106"], stage_summary["pair_rows_v106"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v106"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v106"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v106"]

    blockers = _read_csv("paper4_v106_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v106"], blockers["blocking_v106"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v106"], blockers["evidence_count_v106"], strict=False)
    )
    assert bool(blocker_map["post_v105_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v105_one_swap_improvement_found"]) == 6317
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v106_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v106_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v106_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v106_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v106_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v106 post-v105 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v106 proves the v105 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v106 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v106: Post-v105 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v107_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v107_status.json")

    assert status["phase"] == "v107_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.107"
    assert status["allocation_rows_v107"] == 171
    assert status["summary_rows_v107"] == 1
    assert status["action_rows_v107"] == 1
    assert status["source_summary_rows_v107"] == 51
    assert status["claim_blocker_rows_v107"] == 4
    assert status["added_loan_id_v107"] == "155861848"
    assert status["dropped_loan_id_v107"] == "126633227"
    assert status["selected_rows_v107"] == 171
    assert status["portfolio_exposure_v107"] == pytest.approx(842450.0)
    assert status["objective_return_v107"] == pytest.approx(-1869.9288016583505)
    assert status["scenario_loss_cvar90_v107"] == pytest.approx(92734.88809495818)
    assert status["source_cap_violations_v107"] == 0
    assert status["delta_return_vs_v105_v107"] == pytest.approx(142.09376124853043)
    assert status["delta_cvar90_vs_v105_v107"] == pytest.approx(152.86106813664082)
    assert status["delta_exposure_vs_v105_v107"] == pytest.approx(0.0)
    assert status["budget_feasible_v107"] is True
    assert status["source_feasible_v107"] is True
    assert status["cvar_feasible_v107"] is True
    assert status["repair_candidate_feasible_v107"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v107"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v107"] is False
    assert status["paper1_promotion_allowed_v107"] is False
    assert status["paper4_working_champion_changed_v107"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v107_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v107",
        "selected_v107",
        "portfolio_label_v107",
        "repair_action_v107",
        "claim_boundary_v107",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v107"]
    assert int(allocations["selected_v107"].sum()) == status["selected_rows_v107"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v107"])
    assert "155861848" in set(allocations["loan_id"].astype(str))
    assert "126633227" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v107"]) == {
        "added_from_v106_best_swap",
        "kept_from_v105",
    }
    assert allocations["claim_boundary_v107"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v107_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v107"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v107"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v107"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v107"])

    action = _read_csv("paper4_v107_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v107"]) == status["added_loan_id_v107"]
    assert str(action_row["dropped_loan_id_v107"]) == status["dropped_loan_id_v107"]
    assert float(action_row["return_delta_v107"]) == pytest.approx(142.09376124853478)
    assert int(action_row["source_cap_violations_after_repair_v107"]) == 0

    source_summary = _read_csv("paper4_v107_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v107",
        "source_slack_v107",
        "source_cap_violated_v107",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v107"]
    assert not source_summary["source_cap_violated_v107"].astype(bool).any()

    blockers = _read_csv("paper4_v107_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v107"], blockers["blocking_v107"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v107"], blockers["evidence_count_v107"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v107_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v107_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v107_repair_candidate_feasible"]) is True
    assert bool(claim_map["v107_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v107_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v107_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v107 thirteenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v107 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v107 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v107: Thirteenth One-Swap Repair Candidate" in notebook
    assert "v108 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v108_post_v107_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v108_status.json")

    assert status["phase"] == "v108_post_v107_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.108"
    assert status["summary_rows_v108"] == 1
    assert status["stage_summary_rows_v108"] == 6
    assert status["candidate_pair_rows_v108"] == 6165
    assert status["top_candidate_rows_v108"] == 200
    assert status["claim_blocker_rows_v108"] == 3
    assert status["selected_rows_v108"] == 171
    assert status["candidate_add_rows_v108"] == 276698
    assert status["total_pair_rows_screened_v108"] == 47315358
    assert status["return_improving_pair_rows_v108"] == 2002105
    assert status["budget_return_feasible_pair_rows_v108"] == 1215503
    assert status["source_prefilter_pair_rows_v108"] == 7259
    assert status["source_exact_pair_rows_v108"] == 6165
    assert status["cvar_feasible_pair_rows_v108"] == 6165
    assert status["one_swap_improving_rows_v108"] == 6165
    assert status["best_one_swap_return_delta_v108"] == pytest.approx(136.78708648947654)
    assert status["best_one_swap_cvar90_after_v108"] == pytest.approx(92923.29605141311)
    assert status["post_repair_one_swap_local_optimality_cleared_v108"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v108"] is False
    assert status["paper1_promotion_allowed_v108"] is False
    assert status["paper4_working_champion_changed_v108"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v108_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v108",
        "dropped_loan_id_v108",
        "return_delta_v108",
        "objective_return_after_swap_v108",
        "budget_swap_feasible_v108",
        "source_swap_feasible_v108",
        "source_cap_violations_after_swap_v108",
        "cvar_swap_feasible_v108",
        "one_swap_improves_return_v108",
        "claim_boundary_v108",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v108"]
    assert probe["return_delta_v108"].gt(0).all()
    assert probe["budget_swap_feasible_v108"].astype(bool).all()
    assert probe["source_swap_feasible_v108"].astype(bool).all()
    assert probe["cvar_swap_feasible_v108"].astype(bool).all()
    assert probe["one_swap_improves_return_v108"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v108"].sum()) == 0
    assert probe["return_delta_v108"].max() == pytest.approx(
        status["best_one_swap_return_delta_v108"]
    )
    assert probe["claim_boundary_v108"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v108_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v108"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v108"]) == "133746505"
    assert str(best["dropped_loan_id_v108"]) == "126053700"
    assert float(best["return_delta_v108"]) == pytest.approx(
        status["best_one_swap_return_delta_v108"]
    )
    assert float(best["exposure_after_swap_v108"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v108"]) is True

    summary = _read_csv("paper4_v108_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v108"]) == status["one_swap_improving_rows_v108"]
    assert float(row["current_exposure_v108"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v108"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v108"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v108"])

    stage_summary = _read_csv("paper4_v108_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v108", "pair_rows_v108", "claim_boundary_v108"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v108"], stage_summary["pair_rows_v108"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v108"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v108"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v108"]

    blockers = _read_csv("paper4_v108_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v108"], blockers["blocking_v108"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v108"], blockers["evidence_count_v108"], strict=False)
    )
    assert bool(blocker_map["post_v107_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v107_one_swap_improvement_found"]) == 6165
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v108_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v108_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v108_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v108_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v108_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v108 post-v107 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v108 proves the v107 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v108 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v108: Post-v107 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v109_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v109_status.json")

    assert status["phase"] == "v109_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.109"
    assert status["allocation_rows_v109"] == 171
    assert status["summary_rows_v109"] == 1
    assert status["action_rows_v109"] == 1
    assert status["source_summary_rows_v109"] == 51
    assert status["claim_blocker_rows_v109"] == 4
    assert status["added_loan_id_v109"] == "133746505"
    assert status["dropped_loan_id_v109"] == "126053700"
    assert status["selected_rows_v109"] == 171
    assert status["portfolio_exposure_v109"] == pytest.approx(842450.0)
    assert status["objective_return_v109"] == pytest.approx(-1733.1417151688747)
    assert status["scenario_loss_cvar90_v109"] == pytest.approx(92923.29605141311)
    assert status["source_cap_violations_v109"] == 0
    assert status["delta_return_vs_v107_v109"] == pytest.approx(136.78708648947577)
    assert status["delta_cvar90_vs_v107_v109"] == pytest.approx(188.407956454932)
    assert status["delta_exposure_vs_v107_v109"] == pytest.approx(0.0)
    assert status["budget_feasible_v109"] is True
    assert status["source_feasible_v109"] is True
    assert status["cvar_feasible_v109"] is True
    assert status["repair_candidate_feasible_v109"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v109"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v109"] is False
    assert status["paper1_promotion_allowed_v109"] is False
    assert status["paper4_working_champion_changed_v109"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v109_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v109",
        "selected_v109",
        "portfolio_label_v109",
        "repair_action_v109",
        "claim_boundary_v109",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v109"]
    assert int(allocations["selected_v109"].sum()) == status["selected_rows_v109"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v109"])
    assert "133746505" in set(allocations["loan_id"].astype(str))
    assert "126053700" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v109"]) == {
        "added_from_v108_best_swap",
        "kept_from_v107",
    }
    assert allocations["claim_boundary_v109"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v109_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v109"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v109"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v109"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v109"])

    action = _read_csv("paper4_v109_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v109"]) == status["added_loan_id_v109"]
    assert str(action_row["dropped_loan_id_v109"]) == status["dropped_loan_id_v109"]
    assert float(action_row["return_delta_v109"]) == pytest.approx(136.78708648947654)
    assert int(action_row["source_cap_violations_after_repair_v109"]) == 0

    source_summary = _read_csv("paper4_v109_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v109",
        "source_slack_v109",
        "source_cap_violated_v109",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v109"]
    assert not source_summary["source_cap_violated_v109"].astype(bool).any()

    blockers = _read_csv("paper4_v109_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v109"], blockers["blocking_v109"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v109"], blockers["evidence_count_v109"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v109_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v109_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v109_repair_candidate_feasible"]) is True
    assert bool(claim_map["v109_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v109_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v109_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v109 fourteenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v109 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v109 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v109: Fourteenth One-Swap Repair Candidate" in notebook
    assert "v110 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v110_post_v109_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v110_status.json")

    assert status["phase"] == "v110_post_v109_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.110"
    assert status["summary_rows_v110"] == 1
    assert status["stage_summary_rows_v110"] == 6
    assert status["candidate_pair_rows_v110"] == 5991
    assert status["top_candidate_rows_v110"] == 200
    assert status["claim_blocker_rows_v110"] == 3
    assert status["selected_rows_v110"] == 171
    assert status["candidate_add_rows_v110"] == 276698
    assert status["total_pair_rows_screened_v110"] == 47315358
    assert status["return_improving_pair_rows_v110"] == 1991672
    assert status["budget_return_feasible_pair_rows_v110"] == 1210628
    assert status["source_prefilter_pair_rows_v110"] == 7030
    assert status["source_exact_pair_rows_v110"] == 5991
    assert status["cvar_feasible_pair_rows_v110"] == 5991
    assert status["one_swap_improving_rows_v110"] == 5991
    assert status["best_one_swap_return_delta_v110"] == pytest.approx(131.64172216618448)
    assert status["best_one_swap_cvar90_after_v110"] == pytest.approx(93308.72000762806)
    assert status["post_repair_one_swap_local_optimality_cleared_v110"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v110"] is False
    assert status["paper1_promotion_allowed_v110"] is False
    assert status["paper4_working_champion_changed_v110"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v110_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v110",
        "dropped_loan_id_v110",
        "return_delta_v110",
        "objective_return_after_swap_v110",
        "budget_swap_feasible_v110",
        "source_swap_feasible_v110",
        "source_cap_violations_after_swap_v110",
        "cvar_swap_feasible_v110",
        "one_swap_improves_return_v110",
        "claim_boundary_v110",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v110"]
    assert probe["return_delta_v110"].gt(0).all()
    assert probe["budget_swap_feasible_v110"].astype(bool).all()
    assert probe["source_swap_feasible_v110"].astype(bool).all()
    assert probe["cvar_swap_feasible_v110"].astype(bool).all()
    assert probe["one_swap_improves_return_v110"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v110"].sum()) == 0
    assert probe["return_delta_v110"].max() == pytest.approx(
        status["best_one_swap_return_delta_v110"]
    )
    assert probe["claim_boundary_v110"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v110_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v110"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v110"]) == "158940609"
    assert str(best["dropped_loan_id_v110"]) == "126736783"
    assert float(best["return_delta_v110"]) == pytest.approx(
        status["best_one_swap_return_delta_v110"]
    )
    assert float(best["exposure_after_swap_v110"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v110"]) is True

    summary = _read_csv("paper4_v110_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v110"]) == status["one_swap_improving_rows_v110"]
    assert float(row["current_exposure_v110"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v110"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v110"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v110"])

    stage_summary = _read_csv("paper4_v110_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v110", "pair_rows_v110", "claim_boundary_v110"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v110"], stage_summary["pair_rows_v110"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v110"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v110"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v110"]

    blockers = _read_csv("paper4_v110_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v110"], blockers["blocking_v110"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v110"], blockers["evidence_count_v110"], strict=False)
    )
    assert bool(blocker_map["post_v109_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v109_one_swap_improvement_found"]) == 5991
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v110_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v110_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v110_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v110_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v110_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v110 post-v109 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v110 proves the v109 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v110 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v110: Post-v109 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v111_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v111_status.json")

    assert status["phase"] == "v111_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.111"
    assert status["allocation_rows_v111"] == 171
    assert status["summary_rows_v111"] == 1
    assert status["action_rows_v111"] == 1
    assert status["source_summary_rows_v111"] == 51
    assert status["claim_blocker_rows_v111"] == 4
    assert status["added_loan_id_v111"] == "158940609"
    assert status["dropped_loan_id_v111"] == "126736783"
    assert status["selected_rows_v111"] == 171
    assert status["portfolio_exposure_v111"] == pytest.approx(842450.0)
    assert status["objective_return_v111"] == pytest.approx(-1601.4999930026916)
    assert status["scenario_loss_cvar90_v111"] == pytest.approx(93308.72000762806)
    assert status["source_cap_violations_v111"] == 0
    assert status["delta_return_vs_v109_v111"] == pytest.approx(131.64172216618317)
    assert status["delta_cvar90_vs_v109_v111"] == pytest.approx(385.4239562149305)
    assert status["delta_exposure_vs_v109_v111"] == pytest.approx(0.0)
    assert status["budget_feasible_v111"] is True
    assert status["source_feasible_v111"] is True
    assert status["cvar_feasible_v111"] is True
    assert status["repair_candidate_feasible_v111"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v111"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v111"] is False
    assert status["paper1_promotion_allowed_v111"] is False
    assert status["paper4_working_champion_changed_v111"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v111_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v111",
        "selected_v111",
        "portfolio_label_v111",
        "repair_action_v111",
        "claim_boundary_v111",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v111"]
    assert int(allocations["selected_v111"].sum()) == status["selected_rows_v111"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v111"])
    assert "158940609" in set(allocations["loan_id"].astype(str))
    assert "126736783" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v111"]) == {
        "added_from_v110_best_swap",
        "kept_from_v109",
    }
    assert allocations["claim_boundary_v111"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v111_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v111"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v111"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v111"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v111"])

    action = _read_csv("paper4_v111_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v111"]) == status["added_loan_id_v111"]
    assert str(action_row["dropped_loan_id_v111"]) == status["dropped_loan_id_v111"]
    assert float(action_row["return_delta_v111"]) == pytest.approx(131.64172216618448)
    assert int(action_row["source_cap_violations_after_repair_v111"]) == 0

    source_summary = _read_csv("paper4_v111_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v111",
        "source_slack_v111",
        "source_cap_violated_v111",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v111"]
    assert not source_summary["source_cap_violated_v111"].astype(bool).any()

    blockers = _read_csv("paper4_v111_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v111"], blockers["blocking_v111"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v111"], blockers["evidence_count_v111"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v111_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v111_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v111_repair_candidate_feasible"]) is True
    assert bool(claim_map["v111_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v111_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v111_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v111 fifteenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v111 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v111 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v111: Fifteenth One-Swap Repair Candidate" in notebook
    assert "v112 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v112_post_v111_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v112_status.json")

    assert status["phase"] == "v112_post_v111_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.112"
    assert status["summary_rows_v112"] == 1
    assert status["stage_summary_rows_v112"] == 6
    assert status["candidate_pair_rows_v112"] == 5915
    assert status["top_candidate_rows_v112"] == 200
    assert status["claim_blocker_rows_v112"] == 3
    assert status["selected_rows_v112"] == 171
    assert status["candidate_add_rows_v112"] == 276698
    assert status["total_pair_rows_screened_v112"] == 47315358
    assert status["return_improving_pair_rows_v112"] == 1982862
    assert status["budget_return_feasible_pair_rows_v112"] == 1206782
    assert status["source_prefilter_pair_rows_v112"] == 6949
    assert status["source_exact_pair_rows_v112"] == 5915
    assert status["cvar_feasible_pair_rows_v112"] == 5915
    assert status["one_swap_improving_rows_v112"] == 5915
    assert status["best_one_swap_return_delta_v112"] == pytest.approx(128.01854187366177)
    assert status["best_one_swap_cvar90_after_v112"] == pytest.approx(93291.5115872918)
    assert status["post_repair_one_swap_local_optimality_cleared_v112"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v112"] is False
    assert status["paper1_promotion_allowed_v112"] is False
    assert status["paper4_working_champion_changed_v112"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v112_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v112",
        "dropped_loan_id_v112",
        "return_delta_v112",
        "objective_return_after_swap_v112",
        "budget_swap_feasible_v112",
        "source_swap_feasible_v112",
        "source_cap_violations_after_swap_v112",
        "cvar_swap_feasible_v112",
        "one_swap_improves_return_v112",
        "claim_boundary_v112",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v112"]
    assert probe["return_delta_v112"].gt(0).all()
    assert probe["budget_swap_feasible_v112"].astype(bool).all()
    assert probe["source_swap_feasible_v112"].astype(bool).all()
    assert probe["cvar_swap_feasible_v112"].astype(bool).all()
    assert probe["one_swap_improves_return_v112"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v112"].sum()) == 0
    assert probe["return_delta_v112"].max() == pytest.approx(
        status["best_one_swap_return_delta_v112"]
    )
    assert probe["claim_boundary_v112"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v112_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v112"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v112"]) == "136918234"
    assert str(best["dropped_loan_id_v112"]) == "127277178"
    assert float(best["return_delta_v112"]) == pytest.approx(
        status["best_one_swap_return_delta_v112"]
    )
    assert float(best["exposure_after_swap_v112"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v112"]) is True

    summary = _read_csv("paper4_v112_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v112"]) == status["one_swap_improving_rows_v112"]
    assert float(row["current_exposure_v112"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v112"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v112"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v112"])

    stage_summary = _read_csv("paper4_v112_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v112", "pair_rows_v112", "claim_boundary_v112"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v112"], stage_summary["pair_rows_v112"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v112"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v112"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v112"]

    blockers = _read_csv("paper4_v112_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v112"], blockers["blocking_v112"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v112"], blockers["evidence_count_v112"], strict=False)
    )
    assert bool(blocker_map["post_v111_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v111_one_swap_improvement_found"]) == 5915
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v112_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v112_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v112_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v112_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v112_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v112 post-v111 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v112 proves the v111 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v112 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v112: Post-v111 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v113_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v113_status.json")

    assert status["phase"] == "v113_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.113"
    assert status["allocation_rows_v113"] == 171
    assert status["summary_rows_v113"] == 1
    assert status["action_rows_v113"] == 1
    assert status["source_summary_rows_v113"] == 51
    assert status["claim_blocker_rows_v113"] == 4
    assert status["added_loan_id_v113"] == "136918234"
    assert status["dropped_loan_id_v113"] == "127277178"
    assert status["selected_rows_v113"] == 171
    assert status["portfolio_exposure_v113"] == pytest.approx(842450.0)
    assert status["objective_return_v113"] == pytest.approx(-1473.4814511290242)
    assert status["scenario_loss_cvar90_v113"] == pytest.approx(93291.5115872918)
    assert status["source_cap_violations_v113"] == 0
    assert status["delta_return_vs_v111_v113"] == pytest.approx(128.01854187366735)
    assert status["delta_cvar90_vs_v111_v113"] == pytest.approx(-17.208420336261042)
    assert status["delta_exposure_vs_v111_v113"] == pytest.approx(0.0)
    assert status["budget_feasible_v113"] is True
    assert status["source_feasible_v113"] is True
    assert status["cvar_feasible_v113"] is True
    assert status["repair_candidate_feasible_v113"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v113"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v113"] is False
    assert status["paper1_promotion_allowed_v113"] is False
    assert status["paper4_working_champion_changed_v113"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v113_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v113",
        "selected_v113",
        "portfolio_label_v113",
        "repair_action_v113",
        "claim_boundary_v113",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v113"]
    assert int(allocations["selected_v113"].sum()) == status["selected_rows_v113"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v113"])
    assert "136918234" in set(allocations["loan_id"].astype(str))
    assert "127277178" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v113"]) == {
        "added_from_v112_best_swap",
        "kept_from_v111",
    }
    assert allocations["claim_boundary_v113"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v113_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v113"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v113"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v113"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v113"])

    action = _read_csv("paper4_v113_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v113"]) == status["added_loan_id_v113"]
    assert str(action_row["dropped_loan_id_v113"]) == status["dropped_loan_id_v113"]
    assert float(action_row["return_delta_v113"]) == pytest.approx(128.01854187366177)
    assert int(action_row["source_cap_violations_after_repair_v113"]) == 0

    source_summary = _read_csv("paper4_v113_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v113",
        "source_slack_v113",
        "source_cap_violated_v113",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v113"]
    assert not source_summary["source_cap_violated_v113"].astype(bool).any()

    blockers = _read_csv("paper4_v113_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v113"], blockers["blocking_v113"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v113"], blockers["evidence_count_v113"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v113_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v113_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v113_repair_candidate_feasible"]) is True
    assert bool(claim_map["v113_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v113_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v113_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v113 sixteenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v113 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v113 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v113: Sixteenth One-Swap Repair Candidate" in notebook
    assert "v114 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v114_post_v113_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v114_status.json")

    assert status["phase"] == "v114_post_v113_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.114"
    assert status["summary_rows_v114"] == 1
    assert status["stage_summary_rows_v114"] == 6
    assert status["candidate_pair_rows_v114"] == 5488
    assert status["top_candidate_rows_v114"] == 200
    assert status["claim_blocker_rows_v114"] == 3
    assert status["selected_rows_v114"] == 171
    assert status["candidate_add_rows_v114"] == 276698
    assert status["total_pair_rows_screened_v114"] == 47315358
    assert status["return_improving_pair_rows_v114"] == 1974003
    assert status["budget_return_feasible_pair_rows_v114"] == 1200847
    assert status["source_prefilter_pair_rows_v114"] == 6522
    assert status["source_exact_pair_rows_v114"] == 5488
    assert status["cvar_feasible_pair_rows_v114"] == 5488
    assert status["one_swap_improving_rows_v114"] == 5488
    assert status["best_one_swap_return_delta_v114"] == pytest.approx(122.87735193601367)
    assert status["best_one_swap_cvar90_after_v114"] == pytest.approx(93380.73432561096)
    assert status["post_repair_one_swap_local_optimality_cleared_v114"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v114"] is False
    assert status["paper1_promotion_allowed_v114"] is False
    assert status["paper4_working_champion_changed_v114"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v114_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v114",
        "dropped_loan_id_v114",
        "return_delta_v114",
        "objective_return_after_swap_v114",
        "budget_swap_feasible_v114",
        "source_swap_feasible_v114",
        "source_cap_violations_after_swap_v114",
        "cvar_swap_feasible_v114",
        "one_swap_improves_return_v114",
        "claim_boundary_v114",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v114"]
    assert probe["return_delta_v114"].gt(0).all()
    assert probe["budget_swap_feasible_v114"].astype(bool).all()
    assert probe["source_swap_feasible_v114"].astype(bool).all()
    assert probe["cvar_swap_feasible_v114"].astype(bool).all()
    assert probe["one_swap_improves_return_v114"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v114"].sum()) == 0
    assert probe["return_delta_v114"].max() == pytest.approx(
        status["best_one_swap_return_delta_v114"]
    )
    assert probe["claim_boundary_v114"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v114_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v114"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v114"]) == "157340979"
    assert str(best["dropped_loan_id_v114"]) == "126349193"
    assert float(best["return_delta_v114"]) == pytest.approx(
        status["best_one_swap_return_delta_v114"]
    )
    assert float(best["exposure_after_swap_v114"]) == pytest.approx(842550.0)
    assert bool(best["one_swap_improves_return_v114"]) is True

    summary = _read_csv("paper4_v114_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v114"]) == status["one_swap_improving_rows_v114"]
    assert float(row["current_exposure_v114"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v114"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v114"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v114"])

    stage_summary = _read_csv("paper4_v114_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v114", "pair_rows_v114", "claim_boundary_v114"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v114"], stage_summary["pair_rows_v114"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v114"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v114"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v114"]

    blockers = _read_csv("paper4_v114_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v114"], blockers["blocking_v114"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v114"], blockers["evidence_count_v114"], strict=False)
    )
    assert bool(blocker_map["post_v113_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v113_one_swap_improvement_found"]) == 5488
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v114_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v114_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v114_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v114_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v114_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v114 post-v113 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v114 proves the v113 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v114 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v114: Post-v113 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v115_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v115_status.json")

    assert status["phase"] == "v115_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.115"
    assert status["allocation_rows_v115"] == 171
    assert status["summary_rows_v115"] == 1
    assert status["action_rows_v115"] == 1
    assert status["source_summary_rows_v115"] == 51
    assert status["claim_blocker_rows_v115"] == 4
    assert status["added_loan_id_v115"] == "157340979"
    assert status["dropped_loan_id_v115"] == "126349193"
    assert status["selected_rows_v115"] == 171
    assert status["portfolio_exposure_v115"] == pytest.approx(842550.0)
    assert status["objective_return_v115"] == pytest.approx(-1350.6040991930186)
    assert status["scenario_loss_cvar90_v115"] == pytest.approx(93380.73432561096)
    assert status["source_cap_violations_v115"] == 0
    assert status["delta_return_vs_v113_v115"] == pytest.approx(122.87735193600565)
    assert status["delta_cvar90_vs_v113_v115"] == pytest.approx(89.22273831916391)
    assert status["delta_exposure_vs_v113_v115"] == pytest.approx(100.0)
    assert status["budget_feasible_v115"] is True
    assert status["source_feasible_v115"] is True
    assert status["cvar_feasible_v115"] is True
    assert status["repair_candidate_feasible_v115"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v115"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v115"] is False
    assert status["paper1_promotion_allowed_v115"] is False
    assert status["paper4_working_champion_changed_v115"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v115_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v115",
        "selected_v115",
        "portfolio_label_v115",
        "repair_action_v115",
        "claim_boundary_v115",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v115"]
    assert int(allocations["selected_v115"].sum()) == status["selected_rows_v115"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v115"])
    assert "157340979" in set(allocations["loan_id"].astype(str))
    assert "126349193" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v115"]) == {
        "added_from_v114_best_swap",
        "kept_from_v113",
    }
    assert allocations["claim_boundary_v115"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v115_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v115"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v115"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v115"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v115"])

    action = _read_csv("paper4_v115_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v115"]) == status["added_loan_id_v115"]
    assert str(action_row["dropped_loan_id_v115"]) == status["dropped_loan_id_v115"]
    assert float(action_row["return_delta_v115"]) == pytest.approx(122.87735193601367)
    assert int(action_row["source_cap_violations_after_repair_v115"]) == 0

    source_summary = _read_csv("paper4_v115_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v115",
        "source_slack_v115",
        "source_cap_violated_v115",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v115"]
    assert not source_summary["source_cap_violated_v115"].astype(bool).any()

    blockers = _read_csv("paper4_v115_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v115"], blockers["blocking_v115"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v115"], blockers["evidence_count_v115"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v115_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v115_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v115_repair_candidate_feasible"]) is True
    assert bool(claim_map["v115_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v115_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v115_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v115 seventeenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v115 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v115 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v115: Seventeenth One-Swap Repair Candidate" in notebook
    assert "v116 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v116_post_v115_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v116_status.json")

    assert status["phase"] == "v116_post_v115_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.116"
    assert status["summary_rows_v116"] == 1
    assert status["stage_summary_rows_v116"] == 6
    assert status["candidate_pair_rows_v116"] == 6511
    assert status["top_candidate_rows_v116"] == 200
    assert status["claim_blocker_rows_v116"] == 3
    assert status["selected_rows_v116"] == 171
    assert status["candidate_add_rows_v116"] == 276698
    assert status["total_pair_rows_screened_v116"] == 47315358
    assert status["return_improving_pair_rows_v116"] == 1964553
    assert status["budget_return_feasible_pair_rows_v116"] == 1185383
    assert status["source_prefilter_pair_rows_v116"] == 7752
    assert status["source_exact_pair_rows_v116"] == 6511
    assert status["cvar_feasible_pair_rows_v116"] == 6511
    assert status["one_swap_improving_rows_v116"] == 6511
    assert status["best_one_swap_return_delta_v116"] == pytest.approx(146.50898236741978)
    assert status["best_one_swap_cvar90_after_v116"] == pytest.approx(93254.73943831668)
    assert status["post_repair_one_swap_local_optimality_cleared_v116"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v116"] is False
    assert status["paper1_promotion_allowed_v116"] is False
    assert status["paper4_working_champion_changed_v116"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v116_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v116",
        "dropped_loan_id_v116",
        "return_delta_v116",
        "objective_return_after_swap_v116",
        "budget_swap_feasible_v116",
        "source_swap_feasible_v116",
        "source_cap_violations_after_swap_v116",
        "cvar_swap_feasible_v116",
        "one_swap_improves_return_v116",
        "claim_boundary_v116",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v116"]
    assert probe["return_delta_v116"].gt(0).all()
    assert probe["budget_swap_feasible_v116"].astype(bool).all()
    assert probe["source_swap_feasible_v116"].astype(bool).all()
    assert probe["cvar_swap_feasible_v116"].astype(bool).all()
    assert probe["one_swap_improves_return_v116"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v116"].sum()) == 0
    assert probe["return_delta_v116"].max() == pytest.approx(
        status["best_one_swap_return_delta_v116"]
    )
    assert probe["claim_boundary_v116"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v116_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v116"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v116"]) == "145935771"
    assert str(best["dropped_loan_id_v116"]) == "127794335"
    assert float(best["return_delta_v116"]) == pytest.approx(
        status["best_one_swap_return_delta_v116"]
    )
    assert float(best["exposure_after_swap_v116"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v116"]) is True

    summary = _read_csv("paper4_v116_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v116"]) == status["one_swap_improving_rows_v116"]
    assert float(row["current_exposure_v116"]) == pytest.approx(842550.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v116"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v116"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v116"])

    stage_summary = _read_csv("paper4_v116_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v116", "pair_rows_v116", "claim_boundary_v116"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v116"], stage_summary["pair_rows_v116"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v116"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v116"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v116"]

    blockers = _read_csv("paper4_v116_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v116"], blockers["blocking_v116"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v116"], blockers["evidence_count_v116"], strict=False)
    )
    assert bool(blocker_map["post_v115_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v115_one_swap_improvement_found"]) == 6511
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v116_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v116_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v116_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v116_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v116_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v116 post-v115 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v116 proves the v115 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v116 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v116: Post-v115 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v117_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v117_status.json")

    assert status["phase"] == "v117_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.117"
    assert status["allocation_rows_v117"] == 171
    assert status["summary_rows_v117"] == 1
    assert status["action_rows_v117"] == 1
    assert status["source_summary_rows_v117"] == 51
    assert status["claim_blocker_rows_v117"] == 4
    assert status["added_loan_id_v117"] == "145935771"
    assert status["dropped_loan_id_v117"] == "127794335"
    assert status["selected_rows_v117"] == 171
    assert status["portfolio_exposure_v117"] == pytest.approx(842450.0)
    assert status["objective_return_v117"] == pytest.approx(-1204.0951168255942)
    assert status["scenario_loss_cvar90_v117"] == pytest.approx(93254.73943831668)
    assert status["source_cap_violations_v117"] == 0
    assert status["delta_return_vs_v115_v117"] == pytest.approx(146.50898236742432)
    assert status["delta_cvar90_vs_v115_v117"] == pytest.approx(-125.99488729427685)
    assert status["delta_exposure_vs_v115_v117"] == pytest.approx(-100.0)
    assert status["budget_feasible_v117"] is True
    assert status["source_feasible_v117"] is True
    assert status["cvar_feasible_v117"] is True
    assert status["repair_candidate_feasible_v117"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v117"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v117"] is False
    assert status["paper1_promotion_allowed_v117"] is False
    assert status["paper4_working_champion_changed_v117"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v117_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v117",
        "selected_v117",
        "portfolio_label_v117",
        "repair_action_v117",
        "claim_boundary_v117",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v117"]
    assert int(allocations["selected_v117"].sum()) == status["selected_rows_v117"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v117"])
    assert "145935771" in set(allocations["loan_id"].astype(str))
    assert "127794335" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v117"]) == {
        "added_from_v116_best_swap",
        "kept_from_v115",
    }
    assert allocations["claim_boundary_v117"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v117_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v117"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v117"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v117"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v117"])

    action = _read_csv("paper4_v117_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v117"]) == status["added_loan_id_v117"]
    assert str(action_row["dropped_loan_id_v117"]) == status["dropped_loan_id_v117"]
    assert float(action_row["return_delta_v117"]) == pytest.approx(146.50898236741978)
    assert int(action_row["source_cap_violations_after_repair_v117"]) == 0

    source_summary = _read_csv("paper4_v117_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v117",
        "source_slack_v117",
        "source_cap_violated_v117",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v117"]
    assert not source_summary["source_cap_violated_v117"].astype(bool).any()

    blockers = _read_csv("paper4_v117_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v117"], blockers["blocking_v117"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v117"], blockers["evidence_count_v117"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v117_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v117_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v117_repair_candidate_feasible"]) is True
    assert bool(claim_map["v117_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v117_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v117_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v117 eighteenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v117 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v117 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v117: Eighteenth One-Swap Repair Candidate" in notebook
    assert "v118 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v118_post_v117_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v118_status.json")

    assert status["phase"] == "v118_post_v117_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.118"
    assert status["summary_rows_v118"] == 1
    assert status["stage_summary_rows_v118"] == 6
    assert status["candidate_pair_rows_v118"] == 5239
    assert status["top_candidate_rows_v118"] == 200
    assert status["claim_blocker_rows_v118"] == 3
    assert status["selected_rows_v118"] == 171
    assert status["candidate_add_rows_v118"] == 276698
    assert status["total_pair_rows_screened_v118"] == 47315358
    assert status["return_improving_pair_rows_v118"] == 1953470
    assert status["budget_return_feasible_pair_rows_v118"] == 1190731
    assert status["source_prefilter_pair_rows_v118"] == 6060
    assert status["source_exact_pair_rows_v118"] == 5239
    assert status["cvar_feasible_pair_rows_v118"] == 5239
    assert status["one_swap_improving_rows_v118"] == 5239
    assert status["best_one_swap_return_delta_v118"] == pytest.approx(121.23652794012801)
    assert status["best_one_swap_cvar90_after_v118"] == pytest.approx(93372.68065506514)
    assert status["post_repair_one_swap_local_optimality_cleared_v118"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v118"] is False
    assert status["paper1_promotion_allowed_v118"] is False
    assert status["paper4_working_champion_changed_v118"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v118_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v118",
        "dropped_loan_id_v118",
        "return_delta_v118",
        "objective_return_after_swap_v118",
        "budget_swap_feasible_v118",
        "source_swap_feasible_v118",
        "source_cap_violations_after_swap_v118",
        "cvar_swap_feasible_v118",
        "one_swap_improves_return_v118",
        "claim_boundary_v118",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v118"]
    assert probe["return_delta_v118"].gt(0).all()
    assert probe["budget_swap_feasible_v118"].astype(bool).all()
    assert probe["source_swap_feasible_v118"].astype(bool).all()
    assert probe["cvar_swap_feasible_v118"].astype(bool).all()
    assert probe["one_swap_improves_return_v118"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v118"].sum()) == 0
    assert probe["return_delta_v118"].max() == pytest.approx(
        status["best_one_swap_return_delta_v118"]
    )
    assert probe["claim_boundary_v118"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v118_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v118"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v118"]) == "163964785"
    assert str(best["dropped_loan_id_v118"]) == "127742550"
    assert float(best["return_delta_v118"]) == pytest.approx(
        status["best_one_swap_return_delta_v118"]
    )
    assert float(best["exposure_after_swap_v118"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v118"]) is True

    summary = _read_csv("paper4_v118_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v118"]) == status["one_swap_improving_rows_v118"]
    assert float(row["current_exposure_v118"]) == pytest.approx(842450.0)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v118"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v118"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v118"])

    stage_summary = _read_csv("paper4_v118_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v118", "pair_rows_v118", "claim_boundary_v118"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v118"], stage_summary["pair_rows_v118"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v118"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v118"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v118"]

    blockers = _read_csv("paper4_v118_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v118"], blockers["blocking_v118"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v118"], blockers["evidence_count_v118"], strict=False)
    )
    assert bool(blocker_map["post_v117_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v117_one_swap_improvement_found"]) == 5239
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v118_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v118_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v118_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v118_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v118_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v118 post-v117 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v118 proves the v117 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v118 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v118: Post-v117 One-Swap Repricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v119_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v119_status.json")

    assert status["phase"] == "v119_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.119"
    assert status["allocation_rows_v119"] == 171
    assert status["summary_rows_v119"] == 1
    assert status["action_rows_v119"] == 1
    assert status["source_summary_rows_v119"] == 51
    assert status["claim_blocker_rows_v119"] == 4
    assert status["added_loan_id_v119"] == "163964785"
    assert status["dropped_loan_id_v119"] == "127742550"
    assert status["selected_rows_v119"] == 171
    assert status["portfolio_exposure_v119"] == pytest.approx(842450.0)
    assert status["objective_return_v119"] == pytest.approx(-1082.8585888854686)
    assert status["scenario_loss_cvar90_v119"] == pytest.approx(93372.68065506514)
    assert status["source_cap_violations_v119"] == 0
    assert status["delta_return_vs_v117_v119"] == pytest.approx(121.2365279401256)
    assert status["delta_cvar90_vs_v117_v119"] == pytest.approx(117.94121674845519)
    assert status["delta_exposure_vs_v117_v119"] == pytest.approx(0.0)
    assert status["budget_feasible_v119"] is True
    assert status["source_feasible_v119"] is True
    assert status["cvar_feasible_v119"] is True
    assert status["repair_candidate_feasible_v119"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v119"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v119"] is False
    assert status["paper1_promotion_allowed_v119"] is False
    assert status["paper4_working_champion_changed_v119"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v119_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v119",
        "selected_v119",
        "portfolio_label_v119",
        "repair_action_v119",
        "claim_boundary_v119",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v119"]
    assert int(allocations["selected_v119"].sum()) == status["selected_rows_v119"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v119"])
    assert "163964785" in set(allocations["loan_id"].astype(str))
    assert "127742550" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v119"]) == {
        "added_from_v118_best_swap",
        "kept_from_v117",
    }
    assert allocations["claim_boundary_v119"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v119_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v119"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v119"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v119"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v119"])

    action = _read_csv("paper4_v119_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v119"]) == status["added_loan_id_v119"]
    assert str(action_row["dropped_loan_id_v119"]) == status["dropped_loan_id_v119"]
    assert float(action_row["return_delta_v119"]) == pytest.approx(121.23652794012801)
    assert int(action_row["source_cap_violations_after_repair_v119"]) == 0

    source_summary = _read_csv("paper4_v119_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v119",
        "source_slack_v119",
        "source_cap_violated_v119",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v119"]
    assert not source_summary["source_cap_violated_v119"].astype(bool).any()

    blockers = _read_csv("paper4_v119_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v119"], blockers["blocking_v119"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v119"], blockers["evidence_count_v119"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v119_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v119_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v119_repair_candidate_feasible"]) is True
    assert bool(claim_map["v119_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v119_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v119_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v119 nineteenth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v119 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v119 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v119: Nineteenth One-Swap Repair Candidate" in notebook
    assert "v120 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v120_post_v119_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v120_status.json")

    assert status["phase"] == "v120_post_v119_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.120"
    assert status["summary_rows_v120"] == 1
    assert status["stage_summary_rows_v120"] == 6
    assert status["candidate_pair_rows_v120"] == 4835
    assert status["top_candidate_rows_v120"] == 200
    assert status["claim_blocker_rows_v120"] == 3
    assert status["selected_rows_v120"] == 171
    assert status["candidate_add_rows_v120"] == 276698
    assert status["total_pair_rows_screened_v120"] == 47315358
    assert status["return_improving_pair_rows_v120"] == 1945037
    assert status["budget_return_feasible_pair_rows_v120"] == 1185131
    assert status["source_prefilter_pair_rows_v120"] == 5656
    assert status["source_exact_pair_rows_v120"] == 4835
    assert status["cvar_feasible_pair_rows_v120"] == 4835
    assert status["one_swap_improving_rows_v120"] == 4835
    assert status["best_one_swap_return_delta_v120"] == pytest.approx(118.8179334966386)
    assert status["best_one_swap_cvar90_after_v120"] == pytest.approx(93390.96271634115)
    assert status["post_repair_one_swap_local_optimality_cleared_v120"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v120"] is False
    assert status["paper1_promotion_allowed_v120"] is False
    assert status["paper4_working_champion_changed_v120"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v120_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v120",
        "dropped_loan_id_v120",
        "return_delta_v120",
        "objective_return_after_swap_v120",
        "budget_swap_feasible_v120",
        "source_swap_feasible_v120",
        "source_cap_violations_after_swap_v120",
        "cvar_swap_feasible_v120",
        "one_swap_improves_return_v120",
        "claim_boundary_v120",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v120"]
    assert probe["return_delta_v120"].gt(0).all()
    assert probe["budget_swap_feasible_v120"].astype(bool).all()
    assert probe["source_swap_feasible_v120"].astype(bool).all()
    assert probe["cvar_swap_feasible_v120"].astype(bool).all()
    assert probe["one_swap_improves_return_v120"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v120"].sum()) == 0
    assert probe["return_delta_v120"].max() == pytest.approx(
        status["best_one_swap_return_delta_v120"]
    )
    assert probe["claim_boundary_v120"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v120_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v120"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v120"]) == "160560314"
    assert str(best["dropped_loan_id_v120"]) == "126864348"
    assert float(best["return_delta_v120"]) == pytest.approx(
        status["best_one_swap_return_delta_v120"]
    )
    assert float(best["exposure_after_swap_v120"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v120"]) is True

    summary = _read_csv("paper4_v120_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v120"]) == status["one_swap_improving_rows_v120"]
    assert float(row["current_exposure_v120"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v120"]) == pytest.approx(-1082.8585888854686)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v120"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v120"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v120"])

    stage_summary = _read_csv("paper4_v120_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v120", "pair_rows_v120", "claim_boundary_v120"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v120"], stage_summary["pair_rows_v120"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v120"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v120"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v120"]

    blockers = _read_csv("paper4_v120_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v120"], blockers["blocking_v120"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v120"], blockers["evidence_count_v120"], strict=False)
    )
    assert bool(blocker_map["post_v119_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v119_one_swap_improvement_found"]) == 4835
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v120_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v120_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v120_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v120_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v120_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v120 post-v119 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v120 proves the v119 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v120 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v120: Post-v119 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4835`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v121_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v121_status.json")

    assert status["phase"] == "v121_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.121"
    assert status["allocation_rows_v121"] == 171
    assert status["summary_rows_v121"] == 1
    assert status["action_rows_v121"] == 1
    assert status["source_summary_rows_v121"] == 51
    assert status["claim_blocker_rows_v121"] == 4
    assert status["added_loan_id_v121"] == "160560314"
    assert status["dropped_loan_id_v121"] == "126864348"
    assert status["selected_rows_v121"] == 171
    assert status["portfolio_exposure_v121"] == pytest.approx(842450.0)
    assert status["objective_return_v121"] == pytest.approx(-964.0406553888279)
    assert status["scenario_loss_cvar90_v121"] == pytest.approx(93390.96271634116)
    assert status["source_cap_violations_v121"] == 0
    assert status["delta_return_vs_v119_v121"] == pytest.approx(118.81793349664076)
    assert status["delta_cvar90_vs_v119_v121"] == pytest.approx(18.28206127602607)
    assert status["delta_exposure_vs_v119_v121"] == pytest.approx(0.0)
    assert status["budget_feasible_v121"] is True
    assert status["source_feasible_v121"] is True
    assert status["cvar_feasible_v121"] is True
    assert status["repair_candidate_feasible_v121"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v121"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v121"] is False
    assert status["paper1_promotion_allowed_v121"] is False
    assert status["paper4_working_champion_changed_v121"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v121_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v121",
        "selected_v121",
        "portfolio_label_v121",
        "repair_action_v121",
        "claim_boundary_v121",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v121"]
    assert int(allocations["selected_v121"].sum()) == status["selected_rows_v121"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v121"])
    assert "160560314" in set(allocations["loan_id"].astype(str))
    assert "126864348" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v121"]) == {
        "added_from_v120_best_swap",
        "kept_from_v119",
    }
    assert allocations["claim_boundary_v121"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v121_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v121"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v121"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v121"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v121"])

    action = _read_csv("paper4_v121_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v121"]) == status["added_loan_id_v121"]
    assert str(action_row["dropped_loan_id_v121"]) == status["dropped_loan_id_v121"]
    assert float(action_row["return_delta_v121"]) == pytest.approx(118.8179334966386)
    assert int(action_row["source_cap_violations_after_repair_v121"]) == 0

    source_summary = _read_csv("paper4_v121_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v121",
        "source_slack_v121",
        "source_cap_violated_v121",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v121"]
    assert not source_summary["source_cap_violated_v121"].astype(bool).any()

    blockers = _read_csv("paper4_v121_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v121"], blockers["blocking_v121"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v121"], blockers["evidence_count_v121"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v121_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v121_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v121_repair_candidate_feasible"]) is True
    assert bool(claim_map["v121_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v121_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v121_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v121 twentieth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v121 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v121 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v121: Twentieth One-Swap Repair Candidate" in notebook
    assert "v122 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v122_post_v121_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v122_status.json")

    assert status["phase"] == "v122_post_v121_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.122"
    assert status["summary_rows_v122"] == 1
    assert status["stage_summary_rows_v122"] == 6
    assert status["candidate_pair_rows_v122"] == 4755
    assert status["top_candidate_rows_v122"] == 200
    assert status["claim_blocker_rows_v122"] == 3
    assert status["selected_rows_v122"] == 171
    assert status["candidate_add_rows_v122"] == 276698
    assert status["total_pair_rows_screened_v122"] == 47315358
    assert status["return_improving_pair_rows_v122"] == 1937078
    assert status["budget_return_feasible_pair_rows_v122"] == 1181596
    assert status["source_prefilter_pair_rows_v122"] == 5554
    assert status["source_exact_pair_rows_v122"] == 4755
    assert status["cvar_feasible_pair_rows_v122"] == 4755
    assert status["one_swap_improving_rows_v122"] == 4755
    assert status["best_one_swap_return_delta_v122"] == pytest.approx(108.53567836025547)
    assert status["best_one_swap_cvar90_after_v122"] == pytest.approx(93424.6751064591)
    assert status["post_repair_one_swap_local_optimality_cleared_v122"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v122"] is False
    assert status["paper1_promotion_allowed_v122"] is False
    assert status["paper4_working_champion_changed_v122"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v122_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v122",
        "dropped_loan_id_v122",
        "return_delta_v122",
        "objective_return_after_swap_v122",
        "budget_swap_feasible_v122",
        "source_swap_feasible_v122",
        "source_cap_violations_after_swap_v122",
        "cvar_swap_feasible_v122",
        "one_swap_improves_return_v122",
        "claim_boundary_v122",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v122"]
    assert probe["return_delta_v122"].gt(0).all()
    assert probe["budget_swap_feasible_v122"].astype(bool).all()
    assert probe["source_swap_feasible_v122"].astype(bool).all()
    assert probe["cvar_swap_feasible_v122"].astype(bool).all()
    assert probe["one_swap_improves_return_v122"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v122"].sum()) == 0
    assert probe["return_delta_v122"].max() == pytest.approx(
        status["best_one_swap_return_delta_v122"]
    )
    assert probe["claim_boundary_v122"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v122_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v122"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v122"]) == "147944552"
    assert str(best["dropped_loan_id_v122"]) == "127232386"
    assert float(best["return_delta_v122"]) == pytest.approx(
        status["best_one_swap_return_delta_v122"]
    )
    assert float(best["exposure_after_swap_v122"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v122"]) is True

    summary = _read_csv("paper4_v122_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v122"]) == status["one_swap_improving_rows_v122"]
    assert float(row["current_exposure_v122"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v122"]) == pytest.approx(-964.040655388828)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v122"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v122"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v122"])

    stage_summary = _read_csv("paper4_v122_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v122", "pair_rows_v122", "claim_boundary_v122"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v122"], stage_summary["pair_rows_v122"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v122"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v122"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v122"]

    blockers = _read_csv("paper4_v122_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v122"], blockers["blocking_v122"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v122"], blockers["evidence_count_v122"], strict=False)
    )
    assert bool(blocker_map["post_v121_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v121_one_swap_improvement_found"]) == 4755
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v122_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v122_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v122_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v122_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v122_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v122 post-v121 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v122 proves the v121 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v122 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v122: Post-v121 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4755`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v123_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v123_status.json")

    assert status["phase"] == "v123_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.123"
    assert status["allocation_rows_v123"] == 171
    assert status["summary_rows_v123"] == 1
    assert status["action_rows_v123"] == 1
    assert status["source_summary_rows_v123"] == 51
    assert status["claim_blocker_rows_v123"] == 4
    assert status["added_loan_id_v123"] == "147944552"
    assert status["dropped_loan_id_v123"] == "127232386"
    assert status["selected_rows_v123"] == 171
    assert status["portfolio_exposure_v123"] == pytest.approx(842450.0)
    assert status["objective_return_v123"] == pytest.approx(-855.5049770285732)
    assert status["scenario_loss_cvar90_v123"] == pytest.approx(93424.67510645912)
    assert status["source_cap_violations_v123"] == 0
    assert status["delta_return_vs_v121_v123"] == pytest.approx(108.53567836025479)
    assert status["delta_cvar90_vs_v121_v123"] == pytest.approx(33.71239011795842)
    assert status["delta_exposure_vs_v121_v123"] == pytest.approx(0.0)
    assert status["budget_feasible_v123"] is True
    assert status["source_feasible_v123"] is True
    assert status["cvar_feasible_v123"] is True
    assert status["repair_candidate_feasible_v123"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v123"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v123"] is False
    assert status["paper1_promotion_allowed_v123"] is False
    assert status["paper4_working_champion_changed_v123"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v123_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v123",
        "selected_v123",
        "portfolio_label_v123",
        "repair_action_v123",
        "claim_boundary_v123",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v123"]
    assert int(allocations["selected_v123"].sum()) == status["selected_rows_v123"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v123"])
    assert "147944552" in set(allocations["loan_id"].astype(str))
    assert "127232386" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v123"]) == {
        "added_from_v122_best_swap",
        "kept_from_v121",
    }
    assert allocations["claim_boundary_v123"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v123_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v123"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v123"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v123"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v123"])

    action = _read_csv("paper4_v123_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v123"]) == status["added_loan_id_v123"]
    assert str(action_row["dropped_loan_id_v123"]) == status["dropped_loan_id_v123"]
    assert float(action_row["return_delta_v123"]) == pytest.approx(108.53567836025547)
    assert int(action_row["source_cap_violations_after_repair_v123"]) == 0

    source_summary = _read_csv("paper4_v123_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v123",
        "source_slack_v123",
        "source_cap_violated_v123",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v123"]
    assert not source_summary["source_cap_violated_v123"].astype(bool).any()

    blockers = _read_csv("paper4_v123_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v123"], blockers["blocking_v123"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v123"], blockers["evidence_count_v123"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v123_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v123_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v123_repair_candidate_feasible"]) is True
    assert bool(claim_map["v123_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v123_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v123_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v123 twenty-first one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v123 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v123 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v123: Twenty-First One-Swap Repair Candidate" in notebook
    assert "v124 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v124_post_v123_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v124_status.json")

    assert status["phase"] == "v124_post_v123_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.124"
    assert status["summary_rows_v124"] == 1
    assert status["stage_summary_rows_v124"] == 6
    assert status["candidate_pair_rows_v124"] == 4377
    assert status["top_candidate_rows_v124"] == 200
    assert status["claim_blocker_rows_v124"] == 3
    assert status["selected_rows_v124"] == 171
    assert status["candidate_add_rows_v124"] == 276698
    assert status["total_pair_rows_screened_v124"] == 47315358
    assert status["return_improving_pair_rows_v124"] == 1929510
    assert status["budget_return_feasible_pair_rows_v124"] == 1176556
    assert status["source_prefilter_pair_rows_v124"] == 5176
    assert status["source_exact_pair_rows_v124"] == 4377
    assert status["cvar_feasible_pair_rows_v124"] == 4377
    assert status["one_swap_improving_rows_v124"] == 4377
    assert status["best_one_swap_return_delta_v124"] == pytest.approx(105.32830814768198)
    assert status["best_one_swap_cvar90_after_v124"] == pytest.approx(93519.99529357199)
    assert status["post_repair_one_swap_local_optimality_cleared_v124"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v124"] is False
    assert status["paper1_promotion_allowed_v124"] is False
    assert status["paper4_working_champion_changed_v124"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v124_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v124",
        "dropped_loan_id_v124",
        "return_delta_v124",
        "objective_return_after_swap_v124",
        "budget_swap_feasible_v124",
        "source_swap_feasible_v124",
        "source_cap_violations_after_swap_v124",
        "cvar_swap_feasible_v124",
        "one_swap_improves_return_v124",
        "claim_boundary_v124",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v124"]
    assert probe["return_delta_v124"].gt(0).all()
    assert probe["budget_swap_feasible_v124"].astype(bool).all()
    assert probe["source_swap_feasible_v124"].astype(bool).all()
    assert probe["cvar_swap_feasible_v124"].astype(bool).all()
    assert probe["one_swap_improves_return_v124"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v124"].sum()) == 0
    assert probe["return_delta_v124"].max() == pytest.approx(
        status["best_one_swap_return_delta_v124"]
    )
    assert probe["claim_boundary_v124"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v124_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v124"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v124"]) == "130095750"
    assert str(best["dropped_loan_id_v124"]) == "127867847"
    assert float(best["return_delta_v124"]) == pytest.approx(
        status["best_one_swap_return_delta_v124"]
    )
    assert float(best["exposure_after_swap_v124"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v124"]) is True

    summary = _read_csv("paper4_v124_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v124"]) == status["one_swap_improving_rows_v124"]
    assert float(row["current_exposure_v124"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v124"]) == pytest.approx(-855.5049770285732)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v124"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v124"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v124"])

    stage_summary = _read_csv("paper4_v124_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v124", "pair_rows_v124", "claim_boundary_v124"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v124"], stage_summary["pair_rows_v124"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v124"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v124"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v124"]

    blockers = _read_csv("paper4_v124_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v124"], blockers["blocking_v124"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v124"], blockers["evidence_count_v124"], strict=False)
    )
    assert bool(blocker_map["post_v123_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v123_one_swap_improvement_found"]) == 4377
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v124_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v124_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v124_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v124_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v124_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v124 post-v123 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v124 proves the v123 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v124 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v124: Post-v123 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4377`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v125_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v125_status.json")

    assert status["phase"] == "v125_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.125"
    assert status["allocation_rows_v125"] == 171
    assert status["summary_rows_v125"] == 1
    assert status["action_rows_v125"] == 1
    assert status["source_summary_rows_v125"] == 51
    assert status["claim_blocker_rows_v125"] == 4
    assert status["added_loan_id_v125"] == "130095750"
    assert status["dropped_loan_id_v125"] == "127867847"
    assert status["selected_rows_v125"] == 171
    assert status["portfolio_exposure_v125"] == pytest.approx(842450.0)
    assert status["objective_return_v125"] == pytest.approx(-750.1766688808912)
    assert status["scenario_loss_cvar90_v125"] == pytest.approx(93519.99529357199)
    assert status["source_cap_violations_v125"] == 0
    assert status["delta_return_vs_v123_v125"] == pytest.approx(105.32830814768204)
    assert status["delta_cvar90_vs_v123_v125"] == pytest.approx(95.32018711286946)
    assert status["delta_exposure_vs_v123_v125"] == pytest.approx(0.0)
    assert status["budget_feasible_v125"] is True
    assert status["source_feasible_v125"] is True
    assert status["cvar_feasible_v125"] is True
    assert status["repair_candidate_feasible_v125"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v125"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v125"] is False
    assert status["paper1_promotion_allowed_v125"] is False
    assert status["paper4_working_champion_changed_v125"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v125_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v125",
        "selected_v125",
        "portfolio_label_v125",
        "repair_action_v125",
        "claim_boundary_v125",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v125"]
    assert int(allocations["selected_v125"].sum()) == status["selected_rows_v125"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v125"])
    assert "130095750" in set(allocations["loan_id"].astype(str))
    assert "127867847" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v125"]) == {
        "added_from_v124_best_swap",
        "kept_from_v123",
    }
    assert allocations["claim_boundary_v125"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v125_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v125"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v125"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v125"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v125"])

    action = _read_csv("paper4_v125_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v125"]) == status["added_loan_id_v125"]
    assert str(action_row["dropped_loan_id_v125"]) == status["dropped_loan_id_v125"]
    assert float(action_row["return_delta_v125"]) == pytest.approx(105.32830814768198)
    assert int(action_row["source_cap_violations_after_repair_v125"]) == 0

    source_summary = _read_csv("paper4_v125_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v125",
        "source_slack_v125",
        "source_cap_violated_v125",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v125"]
    assert not source_summary["source_cap_violated_v125"].astype(bool).any()

    blockers = _read_csv("paper4_v125_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v125"], blockers["blocking_v125"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v125"], blockers["evidence_count_v125"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v125_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v125_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v125_repair_candidate_feasible"]) is True
    assert bool(claim_map["v125_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v125_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v125_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v125 twenty-second one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v125 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v125 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v125: Twenty-Second One-Swap Repair Candidate" in notebook
    assert "v126 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v126_post_v125_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v126_status.json")

    assert status["phase"] == "v126_post_v125_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.126"
    assert status["summary_rows_v126"] == 1
    assert status["stage_summary_rows_v126"] == 6
    assert status["candidate_pair_rows_v126"] == 4364
    assert status["top_candidate_rows_v126"] == 200
    assert status["claim_blocker_rows_v126"] == 3
    assert status["selected_rows_v126"] == 171
    assert status["candidate_add_rows_v126"] == 276698
    assert status["total_pair_rows_screened_v126"] == 47315358
    assert status["return_improving_pair_rows_v126"] == 1922871
    assert status["budget_return_feasible_pair_rows_v126"] == 1173209
    assert status["source_prefilter_pair_rows_v126"] == 5163
    assert status["source_exact_pair_rows_v126"] == 4364
    assert status["cvar_feasible_pair_rows_v126"] == 4364
    assert status["one_swap_improving_rows_v126"] == 4364
    assert status["best_one_swap_return_delta_v126"] == pytest.approx(101.1474560355366)
    assert status["best_one_swap_cvar90_after_v126"] == pytest.approx(93507.96054728286)
    assert status["post_repair_one_swap_local_optimality_cleared_v126"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v126"] is False
    assert status["paper1_promotion_allowed_v126"] is False
    assert status["paper4_working_champion_changed_v126"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v126_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v126",
        "dropped_loan_id_v126",
        "return_delta_v126",
        "objective_return_after_swap_v126",
        "budget_swap_feasible_v126",
        "source_swap_feasible_v126",
        "source_cap_violations_after_swap_v126",
        "cvar_swap_feasible_v126",
        "one_swap_improves_return_v126",
        "claim_boundary_v126",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v126"]
    assert probe["return_delta_v126"].gt(0).all()
    assert probe["budget_swap_feasible_v126"].astype(bool).all()
    assert probe["source_swap_feasible_v126"].astype(bool).all()
    assert probe["cvar_swap_feasible_v126"].astype(bool).all()
    assert probe["one_swap_improves_return_v126"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v126"].sum()) == 0
    assert probe["return_delta_v126"].max() == pytest.approx(
        status["best_one_swap_return_delta_v126"]
    )
    assert probe["claim_boundary_v126"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v126_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v126"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v126"]) == "129751891"
    assert str(best["dropped_loan_id_v126"]) == "126608904"
    assert float(best["return_delta_v126"]) == pytest.approx(
        status["best_one_swap_return_delta_v126"]
    )
    assert float(best["exposure_after_swap_v126"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v126"]) is True

    summary = _read_csv("paper4_v126_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v126"]) == status["one_swap_improving_rows_v126"]
    assert float(row["current_exposure_v126"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v126"]) == pytest.approx(-750.1766688808912)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v126"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v126"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v126"])

    stage_summary = _read_csv("paper4_v126_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v126", "pair_rows_v126", "claim_boundary_v126"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v126"], stage_summary["pair_rows_v126"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v126"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v126"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v126"]

    blockers = _read_csv("paper4_v126_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v126"], blockers["blocking_v126"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v126"], blockers["evidence_count_v126"], strict=False)
    )
    assert bool(blocker_map["post_v125_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v125_one_swap_improvement_found"]) == 4364
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v126_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v126_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v126_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v126_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v126_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v126 post-v125 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v126 proves the v125 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v126 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v126: Post-v125 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4364`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v127_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v127_status.json")

    assert status["phase"] == "v127_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.127"
    assert status["allocation_rows_v127"] == 171
    assert status["summary_rows_v127"] == 1
    assert status["action_rows_v127"] == 1
    assert status["source_summary_rows_v127"] == 51
    assert status["claim_blocker_rows_v127"] == 4
    assert status["added_loan_id_v127"] == "129751891"
    assert status["dropped_loan_id_v127"] == "126608904"
    assert status["selected_rows_v127"] == 171
    assert status["portfolio_exposure_v127"] == pytest.approx(842450.0)
    assert status["objective_return_v127"] == pytest.approx(-649.029212845353)
    assert status["scenario_loss_cvar90_v127"] == pytest.approx(93507.96054728286)
    assert status["source_cap_violations_v127"] == 0
    assert status["delta_return_vs_v125_v127"] == pytest.approx(101.1474560355382)
    assert status["delta_cvar90_vs_v125_v127"] == pytest.approx(-12.034746289151371)
    assert status["delta_exposure_vs_v125_v127"] == pytest.approx(0.0)
    assert status["budget_feasible_v127"] is True
    assert status["source_feasible_v127"] is True
    assert status["cvar_feasible_v127"] is True
    assert status["repair_candidate_feasible_v127"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v127"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v127"] is False
    assert status["paper1_promotion_allowed_v127"] is False
    assert status["paper4_working_champion_changed_v127"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v127_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v127",
        "selected_v127",
        "portfolio_label_v127",
        "repair_action_v127",
        "claim_boundary_v127",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v127"]
    assert int(allocations["selected_v127"].sum()) == status["selected_rows_v127"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v127"])
    assert "129751891" in set(allocations["loan_id"].astype(str))
    assert "126608904" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v127"]) == {
        "added_from_v126_best_swap",
        "kept_from_v125",
    }
    assert allocations["claim_boundary_v127"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v127_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v127"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v127"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v127"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v127"])

    action = _read_csv("paper4_v127_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v127"]) == status["added_loan_id_v127"]
    assert str(action_row["dropped_loan_id_v127"]) == status["dropped_loan_id_v127"]
    assert float(action_row["return_delta_v127"]) == pytest.approx(101.1474560355366)
    assert int(action_row["source_cap_violations_after_repair_v127"]) == 0

    source_summary = _read_csv("paper4_v127_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v127",
        "source_slack_v127",
        "source_cap_violated_v127",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v127"]
    assert not source_summary["source_cap_violated_v127"].astype(bool).any()

    blockers = _read_csv("paper4_v127_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v127"], blockers["blocking_v127"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v127"], blockers["evidence_count_v127"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v127_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v127_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v127_repair_candidate_feasible"]) is True
    assert bool(claim_map["v127_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v127_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v127_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v127 twenty-third one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v127 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v127 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v127: Twenty-Third One-Swap Repair Candidate" in notebook
    assert "v128 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v128_post_v127_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v128_status.json")

    assert status["phase"] == "v128_post_v127_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.128"
    assert status["summary_rows_v128"] == 1
    assert status["stage_summary_rows_v128"] == 6
    assert status["candidate_pair_rows_v128"] == 4326
    assert status["top_candidate_rows_v128"] == 200
    assert status["claim_blocker_rows_v128"] == 3
    assert status["selected_rows_v128"] == 171
    assert status["candidate_add_rows_v128"] == 276698
    assert status["total_pair_rows_screened_v128"] == 47315358
    assert status["return_improving_pair_rows_v128"] == 1915852
    assert status["budget_return_feasible_pair_rows_v128"] == 1168819
    assert status["source_prefilter_pair_rows_v128"] == 5117
    assert status["source_exact_pair_rows_v128"] == 4326
    assert status["cvar_feasible_pair_rows_v128"] == 4326
    assert status["one_swap_improving_rows_v128"] == 4326
    assert status["best_one_swap_return_delta_v128"] == pytest.approx(101.13466517860724)
    assert status["best_one_swap_cvar90_after_v128"] == pytest.approx(93600.69280130665)
    assert status["post_repair_one_swap_local_optimality_cleared_v128"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v128"] is False
    assert status["paper1_promotion_allowed_v128"] is False
    assert status["paper4_working_champion_changed_v128"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v128_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v128",
        "dropped_loan_id_v128",
        "return_delta_v128",
        "objective_return_after_swap_v128",
        "budget_swap_feasible_v128",
        "source_swap_feasible_v128",
        "source_cap_violations_after_swap_v128",
        "cvar_swap_feasible_v128",
        "one_swap_improves_return_v128",
        "claim_boundary_v128",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v128"]
    assert probe["return_delta_v128"].gt(0).all()
    assert probe["budget_swap_feasible_v128"].astype(bool).all()
    assert probe["source_swap_feasible_v128"].astype(bool).all()
    assert probe["cvar_swap_feasible_v128"].astype(bool).all()
    assert probe["one_swap_improves_return_v128"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v128"].sum()) == 0
    assert probe["return_delta_v128"].max() == pytest.approx(
        status["best_one_swap_return_delta_v128"]
    )
    assert probe["claim_boundary_v128"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v128_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v128"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v128"]) == "140960427"
    assert str(best["dropped_loan_id_v128"]) == "127108843"
    assert float(best["return_delta_v128"]) == pytest.approx(
        status["best_one_swap_return_delta_v128"]
    )
    assert float(best["exposure_after_swap_v128"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v128"]) is True

    summary = _read_csv("paper4_v128_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v128"]) == status["one_swap_improving_rows_v128"]
    assert float(row["current_exposure_v128"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v128"]) == pytest.approx(-649.029212845353)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v128"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v128"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v128"])

    stage_summary = _read_csv("paper4_v128_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v128", "pair_rows_v128", "claim_boundary_v128"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v128"], stage_summary["pair_rows_v128"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v128"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v128"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v128"]

    blockers = _read_csv("paper4_v128_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v128"], blockers["blocking_v128"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v128"], blockers["evidence_count_v128"], strict=False)
    )
    assert bool(blocker_map["post_v127_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v127_one_swap_improvement_found"]) == 4326
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v128_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v128_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v128_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v128_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v128_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v128 post-v127 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v128 proves the v127 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v128 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v128: Post-v127 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4326`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v129_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v129_status.json")

    assert status["phase"] == "v129_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.129"
    assert status["allocation_rows_v129"] == 171
    assert status["summary_rows_v129"] == 1
    assert status["action_rows_v129"] == 1
    assert status["source_summary_rows_v129"] == 51
    assert status["claim_blocker_rows_v129"] == 4
    assert status["added_loan_id_v129"] == "140960427"
    assert status["dropped_loan_id_v129"] == "127108843"
    assert status["selected_rows_v129"] == 171
    assert status["portfolio_exposure_v129"] == pytest.approx(842450.0)
    assert status["objective_return_v129"] == pytest.approx(-547.894547666745)
    assert status["scenario_loss_cvar90_v129"] == pytest.approx(93600.69280130665)
    assert status["source_cap_violations_v129"] == 0
    assert status["delta_return_vs_v127_v129"] == pytest.approx(101.13466517860797)
    assert status["delta_cvar90_vs_v127_v129"] == pytest.approx(92.73225402379467)
    assert status["delta_exposure_vs_v127_v129"] == pytest.approx(0.0)
    assert status["budget_feasible_v129"] is True
    assert status["source_feasible_v129"] is True
    assert status["cvar_feasible_v129"] is True
    assert status["repair_candidate_feasible_v129"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v129"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v129"] is False
    assert status["paper1_promotion_allowed_v129"] is False
    assert status["paper4_working_champion_changed_v129"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v129_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v129",
        "selected_v129",
        "portfolio_label_v129",
        "repair_action_v129",
        "claim_boundary_v129",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v129"]
    assert int(allocations["selected_v129"].sum()) == status["selected_rows_v129"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v129"])
    assert "140960427" in set(allocations["loan_id"].astype(str))
    assert "127108843" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v129"]) == {
        "added_from_v128_best_swap",
        "kept_from_v127",
    }
    assert allocations["claim_boundary_v129"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v129_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v129"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v129"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v129"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v129"])

    action = _read_csv("paper4_v129_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v129"]) == status["added_loan_id_v129"]
    assert str(action_row["dropped_loan_id_v129"]) == status["dropped_loan_id_v129"]
    assert float(action_row["return_delta_v129"]) == pytest.approx(101.13466517860724)
    assert int(action_row["source_cap_violations_after_repair_v129"]) == 0

    source_summary = _read_csv("paper4_v129_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v129",
        "source_slack_v129",
        "source_cap_violated_v129",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v129"]
    assert not source_summary["source_cap_violated_v129"].astype(bool).any()

    blockers = _read_csv("paper4_v129_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v129"], blockers["blocking_v129"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v129"], blockers["evidence_count_v129"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v129_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v129_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v129_repair_candidate_feasible"]) is True
    assert bool(claim_map["v129_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v129_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v129_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v129 twenty-fourth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v129 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v129 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v129: Twenty-Fourth One-Swap Repair Candidate" in notebook
    assert "v130 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v130_post_v129_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v130_status.json")

    assert status["phase"] == "v130_post_v129_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.130"
    assert status["summary_rows_v130"] == 1
    assert status["stage_summary_rows_v130"] == 6
    assert status["candidate_pair_rows_v130"] == 4266
    assert status["top_candidate_rows_v130"] == 200
    assert status["claim_blocker_rows_v130"] == 3
    assert status["selected_rows_v130"] == 171
    assert status["candidate_add_rows_v130"] == 276698
    assert status["total_pair_rows_screened_v130"] == 47315358
    assert status["return_improving_pair_rows_v130"] == 1907873
    assert status["budget_return_feasible_pair_rows_v130"] == 1165150
    assert status["source_prefilter_pair_rows_v130"] == 5041
    assert status["source_exact_pair_rows_v130"] == 4266
    assert status["cvar_feasible_pair_rows_v130"] == 4266
    assert status["one_swap_improving_rows_v130"] == 4266
    assert status["best_one_swap_return_delta_v130"] == pytest.approx(99.93508283527802)
    assert status["best_one_swap_cvar90_after_v130"] == pytest.approx(93640.91093453715)
    assert status["post_repair_one_swap_local_optimality_cleared_v130"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v130"] is False
    assert status["paper1_promotion_allowed_v130"] is False
    assert status["paper4_working_champion_changed_v130"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v130_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v130",
        "dropped_loan_id_v130",
        "return_delta_v130",
        "objective_return_after_swap_v130",
        "budget_swap_feasible_v130",
        "source_swap_feasible_v130",
        "source_cap_violations_after_swap_v130",
        "cvar_swap_feasible_v130",
        "one_swap_improves_return_v130",
        "claim_boundary_v130",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v130"]
    assert probe["return_delta_v130"].gt(0).all()
    assert probe["budget_swap_feasible_v130"].astype(bool).all()
    assert probe["source_swap_feasible_v130"].astype(bool).all()
    assert probe["cvar_swap_feasible_v130"].astype(bool).all()
    assert probe["one_swap_improves_return_v130"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v130"].sum()) == 0
    assert probe["return_delta_v130"].max() == pytest.approx(
        status["best_one_swap_return_delta_v130"]
    )
    assert probe["claim_boundary_v130"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v130_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v130"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v130"]) == "137925780"
    assert str(best["dropped_loan_id_v130"]) == "127561240"
    assert float(best["return_delta_v130"]) == pytest.approx(
        status["best_one_swap_return_delta_v130"]
    )
    assert float(best["exposure_after_swap_v130"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v130"]) is True

    summary = _read_csv("paper4_v130_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v130"]) == status["one_swap_improving_rows_v130"]
    assert float(row["current_exposure_v130"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v130"]) == pytest.approx(-547.894547666745)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v130"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v130"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v130"])

    stage_summary = _read_csv("paper4_v130_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v130", "pair_rows_v130", "claim_boundary_v130"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v130"], stage_summary["pair_rows_v130"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v130"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v130"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v130"]

    blockers = _read_csv("paper4_v130_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v130"], blockers["blocking_v130"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v130"], blockers["evidence_count_v130"], strict=False)
    )
    assert bool(blocker_map["post_v129_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v129_one_swap_improvement_found"]) == 4266
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v130_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v130_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v130_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v130_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v130_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v130 post-v129 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v130 proves the v129 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v130 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v130: Post-v129 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4266`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v131_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v131_status.json")

    assert status["phase"] == "v131_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.131"
    assert status["allocation_rows_v131"] == 171
    assert status["summary_rows_v131"] == 1
    assert status["action_rows_v131"] == 1
    assert status["source_summary_rows_v131"] == 51
    assert status["claim_blocker_rows_v131"] == 4
    assert status["added_loan_id_v131"] == "137925780"
    assert status["dropped_loan_id_v131"] == "127561240"
    assert status["selected_rows_v131"] == 171
    assert status["portfolio_exposure_v131"] == pytest.approx(842450.0)
    assert status["objective_return_v131"] == pytest.approx(-447.95946483147054)
    assert status["scenario_loss_cvar90_v131"] == pytest.approx(93640.91093453715)
    assert status["source_cap_violations_v131"] == 0
    assert status["delta_return_vs_v129_v131"] == pytest.approx(99.93508283527444)
    assert status["delta_cvar90_vs_v129_v131"] == pytest.approx(40.218133230519015)
    assert status["delta_exposure_vs_v129_v131"] == pytest.approx(0.0)
    assert status["budget_feasible_v131"] is True
    assert status["source_feasible_v131"] is True
    assert status["cvar_feasible_v131"] is True
    assert status["repair_candidate_feasible_v131"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v131"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v131"] is False
    assert status["paper1_promotion_allowed_v131"] is False
    assert status["paper4_working_champion_changed_v131"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v131_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v131",
        "selected_v131",
        "portfolio_label_v131",
        "repair_action_v131",
        "claim_boundary_v131",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v131"]
    assert int(allocations["selected_v131"].sum()) == status["selected_rows_v131"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v131"])
    assert "137925780" in set(allocations["loan_id"].astype(str))
    assert "127561240" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v131"]) == {
        "added_from_v130_best_swap",
        "kept_from_v129",
    }
    assert allocations["claim_boundary_v131"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v131_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v131"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v131"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v131"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v131"])

    action = _read_csv("paper4_v131_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v131"]) == status["added_loan_id_v131"]
    assert str(action_row["dropped_loan_id_v131"]) == status["dropped_loan_id_v131"]
    assert float(action_row["return_delta_v131"]) == pytest.approx(99.93508283527802)
    assert int(action_row["source_cap_violations_after_repair_v131"]) == 0

    source_summary = _read_csv("paper4_v131_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v131",
        "source_slack_v131",
        "source_cap_violated_v131",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v131"]
    assert not source_summary["source_cap_violated_v131"].astype(bool).any()

    blockers = _read_csv("paper4_v131_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v131"], blockers["blocking_v131"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v131"], blockers["evidence_count_v131"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v131_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v131_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v131_repair_candidate_feasible"]) is True
    assert bool(claim_map["v131_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v131_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v131_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v131 twenty-fifth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v131 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v131 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v131: Twenty-Fifth One-Swap Repair Candidate" in notebook
    assert "v132 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v132_post_v131_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v132_status.json")

    assert status["phase"] == "v132_post_v131_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.132"
    assert status["summary_rows_v132"] == 1
    assert status["stage_summary_rows_v132"] == 6
    assert status["candidate_pair_rows_v132"] == 4261
    assert status["top_candidate_rows_v132"] == 200
    assert status["claim_blocker_rows_v132"] == 3
    assert status["selected_rows_v132"] == 171
    assert status["candidate_add_rows_v132"] == 276698
    assert status["total_pair_rows_screened_v132"] == 47315358
    assert status["return_improving_pair_rows_v132"] == 1900428
    assert status["budget_return_feasible_pair_rows_v132"] == 1161895
    assert status["source_prefilter_pair_rows_v132"] == 5034
    assert status["source_exact_pair_rows_v132"] == 4261
    assert status["cvar_feasible_pair_rows_v132"] == 4261
    assert status["one_swap_improving_rows_v132"] == 4261
    assert status["best_one_swap_return_delta_v132"] == pytest.approx(93.11511711504707)
    assert status["best_one_swap_cvar90_after_v132"] == pytest.approx(93628.98132751856)
    assert status["post_repair_one_swap_local_optimality_cleared_v132"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v132"] is False
    assert status["paper1_promotion_allowed_v132"] is False
    assert status["paper4_working_champion_changed_v132"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v132_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v132",
        "dropped_loan_id_v132",
        "return_delta_v132",
        "objective_return_after_swap_v132",
        "budget_swap_feasible_v132",
        "source_swap_feasible_v132",
        "source_cap_violations_after_swap_v132",
        "cvar_swap_feasible_v132",
        "one_swap_improves_return_v132",
        "claim_boundary_v132",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v132"]
    assert probe["return_delta_v132"].gt(0).all()
    assert probe["budget_swap_feasible_v132"].astype(bool).all()
    assert probe["source_swap_feasible_v132"].astype(bool).all()
    assert probe["cvar_swap_feasible_v132"].astype(bool).all()
    assert probe["one_swap_improves_return_v132"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v132"].sum()) == 0
    assert probe["return_delta_v132"].max() == pytest.approx(
        status["best_one_swap_return_delta_v132"]
    )
    assert probe["claim_boundary_v132"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v132_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v132"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v132"]) == "164191748"
    assert str(best["dropped_loan_id_v132"]) == "127190641"
    assert float(best["return_delta_v132"]) == pytest.approx(
        status["best_one_swap_return_delta_v132"]
    )
    assert float(best["exposure_after_swap_v132"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v132"]) is True

    summary = _read_csv("paper4_v132_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v132"]) == status["one_swap_improving_rows_v132"]
    assert float(row["current_exposure_v132"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v132"]) == pytest.approx(-447.95946483147054)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v132"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v132"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v132"])

    stage_summary = _read_csv("paper4_v132_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v132", "pair_rows_v132", "claim_boundary_v132"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v132"], stage_summary["pair_rows_v132"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v132"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v132"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v132"]

    blockers = _read_csv("paper4_v132_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v132"], blockers["blocking_v132"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v132"], blockers["evidence_count_v132"], strict=False)
    )
    assert bool(blocker_map["post_v131_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v131_one_swap_improvement_found"]) == 4261
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v132_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v132_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v132_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v132_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v132_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v132 post-v131 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v132 proves the v131 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v132 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v132: Post-v131 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `4261`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v133_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v133_status.json")

    assert status["phase"] == "v133_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.133"
    assert status["allocation_rows_v133"] == 171
    assert status["summary_rows_v133"] == 1
    assert status["action_rows_v133"] == 1
    assert status["source_summary_rows_v133"] == 51
    assert status["claim_blocker_rows_v133"] == 4
    assert status["added_loan_id_v133"] == "164191748"
    assert status["dropped_loan_id_v133"] == "127190641"
    assert status["selected_rows_v133"] == 171
    assert status["portfolio_exposure_v133"] == pytest.approx(842450.0)
    assert status["objective_return_v133"] == pytest.approx(-354.8443477164192)
    assert status["scenario_loss_cvar90_v133"] == pytest.approx(93628.98132751856)
    assert status["source_cap_violations_v133"] == 0
    assert status["delta_return_vs_v131_v133"] == pytest.approx(93.11511711505136)
    assert status["delta_cvar90_vs_v131_v133"] == pytest.approx(-11.929607018595561)
    assert status["delta_exposure_vs_v131_v133"] == pytest.approx(0.0)
    assert status["budget_feasible_v133"] is True
    assert status["source_feasible_v133"] is True
    assert status["cvar_feasible_v133"] is True
    assert status["repair_candidate_feasible_v133"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v133"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v133"] is False
    assert status["paper1_promotion_allowed_v133"] is False
    assert status["paper4_working_champion_changed_v133"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v133_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v133",
        "selected_v133",
        "portfolio_label_v133",
        "repair_action_v133",
        "claim_boundary_v133",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v133"]
    assert int(allocations["selected_v133"].sum()) == status["selected_rows_v133"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v133"])
    assert "164191748" in set(allocations["loan_id"].astype(str))
    assert "127190641" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v133"]) == {
        "added_from_v132_best_swap",
        "kept_from_v131",
    }
    assert allocations["claim_boundary_v133"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v133_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v133"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v133"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v133"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v133"])

    action = _read_csv("paper4_v133_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v133"]) == status["added_loan_id_v133"]
    assert str(action_row["dropped_loan_id_v133"]) == status["dropped_loan_id_v133"]
    assert float(action_row["return_delta_v133"]) == pytest.approx(93.11511711504708)
    assert int(action_row["source_cap_violations_after_repair_v133"]) == 0

    source_summary = _read_csv("paper4_v133_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v133",
        "source_slack_v133",
        "source_cap_violated_v133",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v133"]
    assert not source_summary["source_cap_violated_v133"].astype(bool).any()

    blockers = _read_csv("paper4_v133_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v133"], blockers["blocking_v133"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v133"], blockers["evidence_count_v133"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v133_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v133_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v133_repair_candidate_feasible"]) is True
    assert bool(claim_map["v133_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v133_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v133_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v133 twenty-sixth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v133 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v133 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v133: Twenty-Sixth One-Swap Repair Candidate" in notebook
    assert "v134 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v134_post_v133_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v134_status.json")

    assert status["phase"] == "v134_post_v133_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.134"
    assert status["summary_rows_v134"] == 1
    assert status["stage_summary_rows_v134"] == 6
    assert status["candidate_pair_rows_v134"] == 3929
    assert status["top_candidate_rows_v134"] == 200
    assert status["claim_blocker_rows_v134"] == 3
    assert status["selected_rows_v134"] == 171
    assert status["candidate_add_rows_v134"] == 276698
    assert status["total_pair_rows_screened_v134"] == 47315358
    assert status["return_improving_pair_rows_v134"] == 1893951
    assert status["budget_return_feasible_pair_rows_v134"] == 1157656
    assert status["source_prefilter_pair_rows_v134"] == 4702
    assert status["source_exact_pair_rows_v134"] == 3929
    assert status["cvar_feasible_pair_rows_v134"] == 3929
    assert status["one_swap_improving_rows_v134"] == 3929
    assert status["best_one_swap_return_delta_v134"] == pytest.approx(92.97979012658101)
    assert status["best_one_swap_cvar90_after_v134"] == pytest.approx(93837.71761179007)
    assert status["post_repair_one_swap_local_optimality_cleared_v134"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v134"] is False
    assert status["paper1_promotion_allowed_v134"] is False
    assert status["paper4_working_champion_changed_v134"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v134_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v134",
        "dropped_loan_id_v134",
        "return_delta_v134",
        "objective_return_after_swap_v134",
        "budget_swap_feasible_v134",
        "source_swap_feasible_v134",
        "source_cap_violations_after_swap_v134",
        "cvar_swap_feasible_v134",
        "one_swap_improves_return_v134",
        "claim_boundary_v134",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v134"]
    assert probe["return_delta_v134"].gt(0).all()
    assert probe["budget_swap_feasible_v134"].astype(bool).all()
    assert probe["source_swap_feasible_v134"].astype(bool).all()
    assert probe["cvar_swap_feasible_v134"].astype(bool).all()
    assert probe["one_swap_improves_return_v134"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v134"].sum()) == 0
    assert probe["return_delta_v134"].max() == pytest.approx(
        status["best_one_swap_return_delta_v134"]
    )
    assert probe["claim_boundary_v134"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v134_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v134"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v134"]) == "152337011"
    assert str(best["dropped_loan_id_v134"]) == "127875030"
    assert float(best["return_delta_v134"]) == pytest.approx(
        status["best_one_swap_return_delta_v134"]
    )
    assert float(best["exposure_after_swap_v134"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v134"]) is True

    summary = _read_csv("paper4_v134_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v134"]) == status["one_swap_improving_rows_v134"]
    assert float(row["current_exposure_v134"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v134"]) == pytest.approx(-354.8443477164192)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v134"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v134"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v134"])

    stage_summary = _read_csv("paper4_v134_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v134", "pair_rows_v134", "claim_boundary_v134"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v134"], stage_summary["pair_rows_v134"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v134"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v134"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v134"]

    blockers = _read_csv("paper4_v134_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v134"], blockers["blocking_v134"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v134"], blockers["evidence_count_v134"], strict=False)
    )
    assert bool(blocker_map["post_v133_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v133_one_swap_improvement_found"]) == 3929
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v134_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v134_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v134_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v134_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v134_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v134 post-v133 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v134 proves the v133 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v134 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v134: Post-v133 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3929`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v135_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v135_status.json")

    assert status["phase"] == "v135_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.135"
    assert status["allocation_rows_v135"] == 171
    assert status["summary_rows_v135"] == 1
    assert status["action_rows_v135"] == 1
    assert status["source_summary_rows_v135"] == 51
    assert status["claim_blocker_rows_v135"] == 4
    assert status["added_loan_id_v135"] == "152337011"
    assert status["dropped_loan_id_v135"] == "127875030"
    assert status["selected_rows_v135"] == 171
    assert status["portfolio_exposure_v135"] == pytest.approx(842450.0)
    assert status["objective_return_v135"] == pytest.approx(-261.864557589839)
    assert status["scenario_loss_cvar90_v135"] == pytest.approx(93837.71761179007)
    assert status["source_cap_violations_v135"] == 0
    assert status["delta_return_vs_v133_v135"] == pytest.approx(92.97979012658016)
    assert status["delta_cvar90_vs_v133_v135"] == pytest.approx(208.73628427150834)
    assert status["delta_exposure_vs_v133_v135"] == pytest.approx(0.0)
    assert status["budget_feasible_v135"] is True
    assert status["source_feasible_v135"] is True
    assert status["cvar_feasible_v135"] is True
    assert status["repair_candidate_feasible_v135"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v135"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v135"] is False
    assert status["paper1_promotion_allowed_v135"] is False
    assert status["paper4_working_champion_changed_v135"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v135_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v135",
        "selected_v135",
        "portfolio_label_v135",
        "repair_action_v135",
        "claim_boundary_v135",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v135"]
    assert int(allocations["selected_v135"].sum()) == status["selected_rows_v135"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v135"])
    assert "152337011" in set(allocations["loan_id"].astype(str))
    assert "127875030" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v135"]) == {
        "added_from_v134_best_swap",
        "kept_from_v133",
    }
    assert allocations["claim_boundary_v135"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v135_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v135"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v135"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v135"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v135"])

    action = _read_csv("paper4_v135_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v135"]) == status["added_loan_id_v135"]
    assert str(action_row["dropped_loan_id_v135"]) == status["dropped_loan_id_v135"]
    assert float(action_row["return_delta_v135"]) == pytest.approx(92.979790126581)
    assert int(action_row["source_cap_violations_after_repair_v135"]) == 0

    source_summary = _read_csv("paper4_v135_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v135",
        "source_slack_v135",
        "source_cap_violated_v135",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v135"]
    assert not source_summary["source_cap_violated_v135"].astype(bool).any()

    blockers = _read_csv("paper4_v135_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v135"], blockers["blocking_v135"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v135"], blockers["evidence_count_v135"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v135_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v135_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v135_repair_candidate_feasible"]) is True
    assert bool(claim_map["v135_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v135_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v135_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v135 twenty-seventh one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v135 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v135 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v135: Twenty-Seventh One-Swap Repair Candidate" in notebook
    assert "v136 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v136_post_v135_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v136_status.json")

    assert status["phase"] == "v136_post_v135_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.136"
    assert status["summary_rows_v136"] == 1
    assert status["stage_summary_rows_v136"] == 6
    assert status["candidate_pair_rows_v136"] == 3807
    assert status["top_candidate_rows_v136"] == 200
    assert status["claim_blocker_rows_v136"] == 3
    assert status["selected_rows_v136"] == 171
    assert status["candidate_add_rows_v136"] == 276698
    assert status["total_pair_rows_screened_v136"] == 47315358
    assert status["return_improving_pair_rows_v136"] == 1886684
    assert status["budget_return_feasible_pair_rows_v136"] == 1154382
    assert status["source_prefilter_pair_rows_v136"] == 4542
    assert status["source_exact_pair_rows_v136"] == 3807
    assert status["cvar_feasible_pair_rows_v136"] == 3807
    assert status["one_swap_improving_rows_v136"] == 3807
    assert status["best_one_swap_return_delta_v136"] == pytest.approx(92.38723628164689)
    assert status["best_one_swap_cvar90_after_v136"] == pytest.approx(93943.3075895581)
    assert status["post_repair_one_swap_local_optimality_cleared_v136"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v136"] is False
    assert status["paper1_promotion_allowed_v136"] is False
    assert status["paper4_working_champion_changed_v136"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v136_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v136",
        "dropped_loan_id_v136",
        "return_delta_v136",
        "objective_return_after_swap_v136",
        "budget_swap_feasible_v136",
        "source_swap_feasible_v136",
        "source_cap_violations_after_swap_v136",
        "cvar_swap_feasible_v136",
        "one_swap_improves_return_v136",
        "claim_boundary_v136",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v136"]
    assert probe["return_delta_v136"].gt(0).all()
    assert probe["budget_swap_feasible_v136"].astype(bool).all()
    assert probe["source_swap_feasible_v136"].astype(bool).all()
    assert probe["cvar_swap_feasible_v136"].astype(bool).all()
    assert probe["one_swap_improves_return_v136"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v136"].sum()) == 0
    assert probe["return_delta_v136"].max() == pytest.approx(
        status["best_one_swap_return_delta_v136"]
    )
    assert probe["claim_boundary_v136"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v136_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v136"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v136"]) == "143784342"
    assert str(best["dropped_loan_id_v136"]) == "126425436"
    assert float(best["return_delta_v136"]) == pytest.approx(
        status["best_one_swap_return_delta_v136"]
    )
    assert float(best["exposure_after_swap_v136"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v136"]) is True

    summary = _read_csv("paper4_v136_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v136"]) == status["one_swap_improving_rows_v136"]
    assert float(row["current_exposure_v136"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v136"]) == pytest.approx(-261.864557589839)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v136"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v136"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v136"])

    stage_summary = _read_csv("paper4_v136_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v136", "pair_rows_v136", "claim_boundary_v136"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v136"], stage_summary["pair_rows_v136"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v136"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v136"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v136"]

    blockers = _read_csv("paper4_v136_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v136"], blockers["blocking_v136"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v136"], blockers["evidence_count_v136"], strict=False)
    )
    assert bool(blocker_map["post_v135_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v135_one_swap_improvement_found"]) == 3807
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v136_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v136_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v136_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v136_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v136_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v136 post-v135 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v136 proves the v135 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v136 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v136: Post-v135 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3807`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v137_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v137_status.json")

    assert status["phase"] == "v137_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.137"
    assert status["allocation_rows_v137"] == 171
    assert status["summary_rows_v137"] == 1
    assert status["action_rows_v137"] == 1
    assert status["source_summary_rows_v137"] == 51
    assert status["claim_blocker_rows_v137"] == 4
    assert status["added_loan_id_v137"] == "143784342"
    assert status["dropped_loan_id_v137"] == "126425436"
    assert status["selected_rows_v137"] == 171
    assert status["portfolio_exposure_v137"] == pytest.approx(842450.0)
    assert status["objective_return_v137"] == pytest.approx(-169.47732130819077)
    assert status["scenario_loss_cvar90_v137"] == pytest.approx(93943.3075895581)
    assert status["source_cap_violations_v137"] == 0
    assert status["delta_return_vs_v135_v137"] == pytest.approx(92.38723628164826)
    assert status["delta_cvar90_vs_v135_v137"] == pytest.approx(105.58997776801698)
    assert status["delta_exposure_vs_v135_v137"] == pytest.approx(0.0)
    assert status["budget_feasible_v137"] is True
    assert status["source_feasible_v137"] is True
    assert status["cvar_feasible_v137"] is True
    assert status["repair_candidate_feasible_v137"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v137"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v137"] is False
    assert status["paper1_promotion_allowed_v137"] is False
    assert status["paper4_working_champion_changed_v137"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v137_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v137",
        "selected_v137",
        "portfolio_label_v137",
        "repair_action_v137",
        "claim_boundary_v137",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v137"]
    assert int(allocations["selected_v137"].sum()) == status["selected_rows_v137"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v137"])
    assert "143784342" in set(allocations["loan_id"].astype(str))
    assert "126425436" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v137"]) == {
        "added_from_v136_best_swap",
        "kept_from_v135",
    }
    assert allocations["claim_boundary_v137"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v137_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v137"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v137"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v137"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v137"])

    action = _read_csv("paper4_v137_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v137"]) == status["added_loan_id_v137"]
    assert str(action_row["dropped_loan_id_v137"]) == status["dropped_loan_id_v137"]
    assert float(action_row["return_delta_v137"]) == pytest.approx(92.38723628164688)
    assert int(action_row["source_cap_violations_after_repair_v137"]) == 0

    source_summary = _read_csv("paper4_v137_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v137",
        "source_slack_v137",
        "source_cap_violated_v137",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v137"]
    assert not source_summary["source_cap_violated_v137"].astype(bool).any()

    blockers = _read_csv("paper4_v137_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v137"], blockers["blocking_v137"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v137"], blockers["evidence_count_v137"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v137_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v137_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v137_repair_candidate_feasible"]) is True
    assert bool(claim_map["v137_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v137_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v137_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v137 twenty-eighth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v137 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v137 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v137: Twenty-Eighth One-Swap Repair Candidate" in notebook
    assert "v138 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v138_post_v137_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v138_status.json")

    assert status["phase"] == "v138_post_v137_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.138"
    assert status["summary_rows_v138"] == 1
    assert status["stage_summary_rows_v138"] == 6
    assert status["candidate_pair_rows_v138"] == 3482
    assert status["top_candidate_rows_v138"] == 200
    assert status["claim_blocker_rows_v138"] == 3
    assert status["selected_rows_v138"] == 171
    assert status["candidate_add_rows_v138"] == 276698
    assert status["total_pair_rows_screened_v138"] == 47315358
    assert status["return_improving_pair_rows_v138"] == 1880271
    assert status["budget_return_feasible_pair_rows_v138"] == 1150188
    assert status["source_prefilter_pair_rows_v138"] == 4217
    assert status["source_exact_pair_rows_v138"] == 3482
    assert status["cvar_feasible_pair_rows_v138"] == 3482
    assert status["one_swap_improving_rows_v138"] == 3482
    assert status["best_one_swap_return_delta_v138"] == pytest.approx(89.71082424435784)
    assert status["best_one_swap_cvar90_after_v138"] == pytest.approx(94148.70501264239)
    assert status["post_repair_one_swap_local_optimality_cleared_v138"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v138"] is False
    assert status["paper1_promotion_allowed_v138"] is False
    assert status["paper4_working_champion_changed_v138"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v138_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v138",
        "dropped_loan_id_v138",
        "return_delta_v138",
        "objective_return_after_swap_v138",
        "budget_swap_feasible_v138",
        "source_swap_feasible_v138",
        "source_cap_violations_after_swap_v138",
        "cvar_swap_feasible_v138",
        "one_swap_improves_return_v138",
        "claim_boundary_v138",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v138"]
    assert probe["return_delta_v138"].gt(0).all()
    assert probe["budget_swap_feasible_v138"].astype(bool).all()
    assert probe["source_swap_feasible_v138"].astype(bool).all()
    assert probe["cvar_swap_feasible_v138"].astype(bool).all()
    assert probe["one_swap_improves_return_v138"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v138"].sum()) == 0
    assert probe["return_delta_v138"].max() == pytest.approx(
        status["best_one_swap_return_delta_v138"]
    )
    assert probe["claim_boundary_v138"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v138_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v138"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v138"]) == "143127376"
    assert str(best["dropped_loan_id_v138"]) == "127409242"
    assert float(best["return_delta_v138"]) == pytest.approx(
        status["best_one_swap_return_delta_v138"]
    )
    assert float(best["exposure_after_swap_v138"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v138"]) is True

    summary = _read_csv("paper4_v138_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v138"]) == status["one_swap_improving_rows_v138"]
    assert float(row["current_exposure_v138"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v138"]) == pytest.approx(-169.47732130819077)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v138"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v138"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v138"])

    stage_summary = _read_csv("paper4_v138_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v138", "pair_rows_v138", "claim_boundary_v138"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v138"], stage_summary["pair_rows_v138"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v138"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v138"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v138"]

    blockers = _read_csv("paper4_v138_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v138"], blockers["blocking_v138"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v138"], blockers["evidence_count_v138"], strict=False)
    )
    assert bool(blocker_map["post_v137_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v137_one_swap_improvement_found"]) == 3482
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v138_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v138_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v138_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v138_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v138_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v138 post-v137 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v138 proves the v137 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v138 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v138: Post-v137 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3482`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v139_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v139_status.json")

    assert status["phase"] == "v139_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.139"
    assert status["allocation_rows_v139"] == 171
    assert status["summary_rows_v139"] == 1
    assert status["action_rows_v139"] == 1
    assert status["source_summary_rows_v139"] == 51
    assert status["claim_blocker_rows_v139"] == 4
    assert status["added_loan_id_v139"] == "143127376"
    assert status["dropped_loan_id_v139"] == "127409242"
    assert status["selected_rows_v139"] == 171
    assert status["portfolio_exposure_v139"] == pytest.approx(842450.0)
    assert status["objective_return_v139"] == pytest.approx(-79.76649706383432)
    assert status["scenario_loss_cvar90_v139"] == pytest.approx(94148.70501264239)
    assert status["source_cap_violations_v139"] == 0
    assert status["delta_return_vs_v137_v139"] == pytest.approx(89.71082424435644)
    assert status["delta_cvar90_vs_v137_v139"] == pytest.approx(205.39742308428686)
    assert status["delta_exposure_vs_v137_v139"] == pytest.approx(0.0)
    assert status["budget_feasible_v139"] is True
    assert status["source_feasible_v139"] is True
    assert status["cvar_feasible_v139"] is True
    assert status["repair_candidate_feasible_v139"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v139"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v139"] is False
    assert status["paper1_promotion_allowed_v139"] is False
    assert status["paper4_working_champion_changed_v139"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v139_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v139",
        "selected_v139",
        "portfolio_label_v139",
        "repair_action_v139",
        "claim_boundary_v139",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v139"]
    assert int(allocations["selected_v139"].sum()) == status["selected_rows_v139"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v139"])
    assert "143127376" in set(allocations["loan_id"].astype(str))
    assert "127409242" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v139"]) == {
        "added_from_v138_best_swap",
        "kept_from_v137",
    }
    assert allocations["claim_boundary_v139"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v139_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v139"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v139"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v139"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v139"])

    action = _read_csv("paper4_v139_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v139"]) == status["added_loan_id_v139"]
    assert str(action_row["dropped_loan_id_v139"]) == status["dropped_loan_id_v139"]
    assert float(action_row["return_delta_v139"]) == pytest.approx(89.71082424435784)
    assert int(action_row["source_cap_violations_after_repair_v139"]) == 0

    source_summary = _read_csv("paper4_v139_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v139",
        "source_slack_v139",
        "source_cap_violated_v139",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v139"]
    assert not source_summary["source_cap_violated_v139"].astype(bool).any()

    blockers = _read_csv("paper4_v139_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v139"], blockers["blocking_v139"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v139"], blockers["evidence_count_v139"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v139_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v139_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v139_repair_candidate_feasible"]) is True
    assert bool(claim_map["v139_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v139_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v139_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v139 twenty-ninth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v139 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v139 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v139: Twenty-Ninth One-Swap Repair Candidate" in notebook
    assert "v140 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v140_post_v139_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v140_status.json")

    assert status["phase"] == "v140_post_v139_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.140"
    assert status["summary_rows_v140"] == 1
    assert status["stage_summary_rows_v140"] == 6
    assert status["candidate_pair_rows_v140"] == 3181
    assert status["top_candidate_rows_v140"] == 200
    assert status["claim_blocker_rows_v140"] == 3
    assert status["selected_rows_v140"] == 171
    assert status["candidate_add_rows_v140"] == 276698
    assert status["total_pair_rows_screened_v140"] == 47315358
    assert status["return_improving_pair_rows_v140"] == 1874065
    assert status["budget_return_feasible_pair_rows_v140"] == 1146139
    assert status["source_prefilter_pair_rows_v140"] == 3916
    assert status["source_exact_pair_rows_v140"] == 3181
    assert status["cvar_feasible_pair_rows_v140"] == 3181
    assert status["one_swap_improving_rows_v140"] == 3181
    assert status["best_one_swap_return_delta_v140"] == pytest.approx(87.47095956261964)
    assert status["best_one_swap_cvar90_after_v140"] == pytest.approx(94116.39079886669)
    assert status["post_repair_one_swap_local_optimality_cleared_v140"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v140"] is False
    assert status["paper1_promotion_allowed_v140"] is False
    assert status["paper4_working_champion_changed_v140"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v140_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v140",
        "dropped_loan_id_v140",
        "return_delta_v140",
        "objective_return_after_swap_v140",
        "budget_swap_feasible_v140",
        "source_swap_feasible_v140",
        "source_cap_violations_after_swap_v140",
        "cvar_swap_feasible_v140",
        "one_swap_improves_return_v140",
        "claim_boundary_v140",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v140"]
    assert probe["return_delta_v140"].gt(0).all()
    assert probe["budget_swap_feasible_v140"].astype(bool).all()
    assert probe["source_swap_feasible_v140"].astype(bool).all()
    assert probe["cvar_swap_feasible_v140"].astype(bool).all()
    assert probe["one_swap_improves_return_v140"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v140"].sum()) == 0
    assert probe["return_delta_v140"].max() == pytest.approx(
        status["best_one_swap_return_delta_v140"]
    )
    assert probe["claim_boundary_v140"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v140_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v140"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v140"]) == "139102238"
    assert str(best["dropped_loan_id_v140"]) == "126788147"
    assert float(best["return_delta_v140"]) == pytest.approx(
        status["best_one_swap_return_delta_v140"]
    )
    assert float(best["exposure_after_swap_v140"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v140"]) is True

    summary = _read_csv("paper4_v140_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v140"]) == status["one_swap_improving_rows_v140"]
    assert float(row["current_exposure_v140"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v140"]) == pytest.approx(-79.76649706383432)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v140"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v140"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v140"])

    stage_summary = _read_csv("paper4_v140_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v140", "pair_rows_v140", "claim_boundary_v140"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v140"], stage_summary["pair_rows_v140"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v140"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v140"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v140"]

    blockers = _read_csv("paper4_v140_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v140"], blockers["blocking_v140"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v140"], blockers["evidence_count_v140"], strict=False)
    )
    assert bool(blocker_map["post_v139_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v139_one_swap_improvement_found"]) == 3181
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v140_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v140_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v140_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v140_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v140_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v140 post-v139 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v140 proves the v139 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v140 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v140: Post-v139 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3181`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v141_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v141_status.json")

    assert status["phase"] == "v141_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.141"
    assert status["allocation_rows_v141"] == 171
    assert status["summary_rows_v141"] == 1
    assert status["action_rows_v141"] == 1
    assert status["source_summary_rows_v141"] == 51
    assert status["claim_blocker_rows_v141"] == 4
    assert status["added_loan_id_v141"] == "139102238"
    assert status["dropped_loan_id_v141"] == "126788147"
    assert status["selected_rows_v141"] == 171
    assert status["portfolio_exposure_v141"] == pytest.approx(842450.0)
    assert status["objective_return_v141"] == pytest.approx(7.704462498784778)
    assert status["scenario_loss_cvar90_v141"] == pytest.approx(94116.39079886672)
    assert status["source_cap_violations_v141"] == 0
    assert status["delta_return_vs_v139_v141"] == pytest.approx(87.4709595626191)
    assert status["delta_cvar90_vs_v139_v141"] == pytest.approx(-32.3142137756804)
    assert status["delta_exposure_vs_v139_v141"] == pytest.approx(0.0)
    assert status["budget_feasible_v141"] is True
    assert status["source_feasible_v141"] is True
    assert status["cvar_feasible_v141"] is True
    assert status["repair_candidate_feasible_v141"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v141"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v141"] is False
    assert status["paper1_promotion_allowed_v141"] is False
    assert status["paper4_working_champion_changed_v141"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v141_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v141",
        "selected_v141",
        "portfolio_label_v141",
        "repair_action_v141",
        "claim_boundary_v141",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v141"]
    assert int(allocations["selected_v141"].sum()) == status["selected_rows_v141"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v141"])
    assert "139102238" in set(allocations["loan_id"].astype(str))
    assert "126788147" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v141"]) == {
        "added_from_v140_best_swap",
        "kept_from_v139",
    }
    assert allocations["claim_boundary_v141"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v141_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v141"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v141"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v141"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v141"])

    action = _read_csv("paper4_v141_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v141"]) == status["added_loan_id_v141"]
    assert str(action_row["dropped_loan_id_v141"]) == status["dropped_loan_id_v141"]
    assert float(action_row["return_delta_v141"]) == pytest.approx(87.47095956261964)
    assert int(action_row["source_cap_violations_after_repair_v141"]) == 0

    source_summary = _read_csv("paper4_v141_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v141",
        "source_slack_v141",
        "source_cap_violated_v141",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v141"]
    assert not source_summary["source_cap_violated_v141"].astype(bool).any()

    blockers = _read_csv("paper4_v141_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v141"], blockers["blocking_v141"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v141"], blockers["evidence_count_v141"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v141_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v141_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v141_repair_candidate_feasible"]) is True
    assert bool(claim_map["v141_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v141_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v141_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v141 thirtieth one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v141 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v141 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v141: Thirtieth One-Swap Repair Candidate" in notebook
    assert "v142 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v142_post_v141_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v142_status.json")

    assert status["phase"] == "v142_post_v141_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.142"
    assert status["summary_rows_v142"] == 1
    assert status["stage_summary_rows_v142"] == 6
    assert status["candidate_pair_rows_v142"] == 3177
    assert status["top_candidate_rows_v142"] == 200
    assert status["claim_blocker_rows_v142"] == 3
    assert status["selected_rows_v142"] == 171
    assert status["candidate_add_rows_v142"] == 276698
    assert status["total_pair_rows_screened_v142"] == 47315358
    assert status["return_improving_pair_rows_v142"] == 1867449
    assert status["budget_return_feasible_pair_rows_v142"] == 1142750
    assert status["source_prefilter_pair_rows_v142"] == 3912
    assert status["source_exact_pair_rows_v142"] == 3177
    assert status["cvar_feasible_pair_rows_v142"] == 3177
    assert status["one_swap_improving_rows_v142"] == 3177
    assert status["best_one_swap_return_delta_v142"] == pytest.approx(84.91258805462799)
    assert status["best_one_swap_cvar90_after_v142"] == pytest.approx(94043.11004841489)
    assert status["post_repair_one_swap_local_optimality_cleared_v142"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v142"] is False
    assert status["paper1_promotion_allowed_v142"] is False
    assert status["paper4_working_champion_changed_v142"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v142_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v142",
        "dropped_loan_id_v142",
        "return_delta_v142",
        "objective_return_after_swap_v142",
        "budget_swap_feasible_v142",
        "source_swap_feasible_v142",
        "source_cap_violations_after_swap_v142",
        "cvar_swap_feasible_v142",
        "one_swap_improves_return_v142",
        "claim_boundary_v142",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v142"]
    assert probe["return_delta_v142"].gt(0).all()
    assert probe["budget_swap_feasible_v142"].astype(bool).all()
    assert probe["source_swap_feasible_v142"].astype(bool).all()
    assert probe["cvar_swap_feasible_v142"].astype(bool).all()
    assert probe["one_swap_improves_return_v142"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v142"].sum()) == 0
    assert probe["return_delta_v142"].max() == pytest.approx(
        status["best_one_swap_return_delta_v142"]
    )
    assert probe["claim_boundary_v142"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v142_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v142"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v142"]) == "141895475"
    assert str(best["dropped_loan_id_v142"]) == "126844634"
    assert float(best["return_delta_v142"]) == pytest.approx(
        status["best_one_swap_return_delta_v142"]
    )
    assert float(best["exposure_after_swap_v142"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v142"]) is True

    summary = _read_csv("paper4_v142_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v142"]) == status["one_swap_improving_rows_v142"]
    assert float(row["current_exposure_v142"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v142"]) == pytest.approx(7.704462498784778)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v142"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v142"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v142"])

    stage_summary = _read_csv("paper4_v142_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v142", "pair_rows_v142", "claim_boundary_v142"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v142"], stage_summary["pair_rows_v142"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v142"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v142"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v142"]

    blockers = _read_csv("paper4_v142_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v142"], blockers["blocking_v142"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v142"], blockers["evidence_count_v142"], strict=False)
    )
    assert bool(blocker_map["post_v141_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v141_one_swap_improvement_found"]) == 3177
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v142_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v142_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v142_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v142_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v142_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v142 post-v141 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v142 proves the v141 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v142 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v142: Post-v141 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3177`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v143_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v143_status.json")

    assert status["phase"] == "v143_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.143"
    assert status["allocation_rows_v143"] == 171
    assert status["summary_rows_v143"] == 1
    assert status["action_rows_v143"] == 1
    assert status["source_summary_rows_v143"] == 51
    assert status["claim_blocker_rows_v143"] == 4
    assert status["added_loan_id_v143"] == "141895475"
    assert status["dropped_loan_id_v143"] == "126844634"
    assert status["selected_rows_v143"] == 171
    assert status["portfolio_exposure_v143"] == pytest.approx(842450.0)
    assert status["objective_return_v143"] == pytest.approx(92.61705055341372)
    assert status["scenario_loss_cvar90_v143"] == pytest.approx(94043.11004841488)
    assert status["source_cap_violations_v143"] == 0
    assert status["delta_return_vs_v141_v143"] == pytest.approx(84.91258805462894)
    assert status["delta_cvar90_vs_v141_v143"] == pytest.approx(-73.28075045184232)
    assert status["delta_exposure_vs_v141_v143"] == pytest.approx(0.0)
    assert status["budget_feasible_v143"] is True
    assert status["source_feasible_v143"] is True
    assert status["cvar_feasible_v143"] is True
    assert status["repair_candidate_feasible_v143"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v143"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v143"] is False
    assert status["paper1_promotion_allowed_v143"] is False
    assert status["paper4_working_champion_changed_v143"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v143_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v143",
        "selected_v143",
        "portfolio_label_v143",
        "repair_action_v143",
        "claim_boundary_v143",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v143"]
    assert int(allocations["selected_v143"].sum()) == status["selected_rows_v143"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v143"])
    assert "141895475" in set(allocations["loan_id"].astype(str))
    assert "126844634" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v143"]) == {
        "added_from_v142_best_swap",
        "kept_from_v141",
    }
    assert allocations["claim_boundary_v143"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v143_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v143"]) is True
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v143"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v143"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v143"])

    action = _read_csv("paper4_v143_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v143"]) == status["added_loan_id_v143"]
    assert str(action_row["dropped_loan_id_v143"]) == status["dropped_loan_id_v143"]
    assert float(action_row["return_delta_v143"]) == pytest.approx(84.91258805462799)
    assert int(action_row["source_cap_violations_after_repair_v143"]) == 0

    source_summary = _read_csv("paper4_v143_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v143",
        "source_slack_v143",
        "source_cap_violated_v143",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v143"]
    assert not source_summary["source_cap_violated_v143"].astype(bool).any()

    blockers = _read_csv("paper4_v143_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v143"], blockers["blocking_v143"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v143"], blockers["evidence_count_v143"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v143_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v143_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v143_repair_candidate_feasible"]) is True
    assert bool(claim_map["v143_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v143_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v143_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v143 thirty-first one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v143 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v143 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v143: Thirty-First One-Swap Repair Candidate" in notebook
    assert "v144 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v144_post_v143_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v144_status.json")

    assert status["phase"] == "v144_post_v143_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.144"
    assert status["summary_rows_v144"] == 1
    assert status["stage_summary_rows_v144"] == 6
    assert status["candidate_pair_rows_v144"] == 3076
    assert status["top_candidate_rows_v144"] == 200
    assert status["claim_blocker_rows_v144"] == 3
    assert status["selected_rows_v144"] == 171
    assert status["candidate_add_rows_v144"] == 276698
    assert status["total_pair_rows_screened_v144"] == 47315358
    assert status["return_improving_pair_rows_v144"] == 1861729
    assert status["budget_return_feasible_pair_rows_v144"] == 1138759
    assert status["source_prefilter_pair_rows_v144"] == 3811
    assert status["source_exact_pair_rows_v144"] == 3076
    assert status["cvar_feasible_pair_rows_v144"] == 3076
    assert status["one_swap_improving_rows_v144"] == 3076
    assert status["best_one_swap_return_delta_v144"] == pytest.approx(82.33186597435457)
    assert status["best_one_swap_cvar90_after_v144"] == pytest.approx(94205.09531368258)
    assert status["post_repair_one_swap_local_optimality_cleared_v144"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v144"] is False
    assert status["paper1_promotion_allowed_v144"] is False
    assert status["paper4_working_champion_changed_v144"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v144_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v144",
        "dropped_loan_id_v144",
        "return_delta_v144",
        "objective_return_after_swap_v144",
        "budget_swap_feasible_v144",
        "source_swap_feasible_v144",
        "source_cap_violations_after_swap_v144",
        "cvar_swap_feasible_v144",
        "one_swap_improves_return_v144",
        "claim_boundary_v144",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v144"]
    assert probe["return_delta_v144"].gt(0).all()
    assert probe["budget_swap_feasible_v144"].astype(bool).all()
    assert probe["source_swap_feasible_v144"].astype(bool).all()
    assert probe["cvar_swap_feasible_v144"].astype(bool).all()
    assert probe["one_swap_improves_return_v144"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v144"].sum()) == 0
    assert probe["return_delta_v144"].max() == pytest.approx(
        status["best_one_swap_return_delta_v144"]
    )
    assert probe["claim_boundary_v144"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v144_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v144"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v144"]) == "162406873"
    assert str(best["dropped_loan_id_v144"]) == "127178217"
    assert float(best["return_delta_v144"]) == pytest.approx(
        status["best_one_swap_return_delta_v144"]
    )
    assert float(best["exposure_after_swap_v144"]) == pytest.approx(842550.0)
    assert bool(best["one_swap_improves_return_v144"]) is True

    summary = _read_csv("paper4_v144_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v144"]) == status["one_swap_improving_rows_v144"]
    assert float(row["current_exposure_v144"]) == pytest.approx(842450.0)
    assert float(row["current_objective_return_v144"]) == pytest.approx(92.61705055341372)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v144"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v144"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v144"])

    stage_summary = _read_csv("paper4_v144_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v144", "pair_rows_v144", "claim_boundary_v144"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v144"], stage_summary["pair_rows_v144"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v144"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v144"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v144"]

    blockers = _read_csv("paper4_v144_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v144"], blockers["blocking_v144"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v144"], blockers["evidence_count_v144"], strict=False)
    )
    assert bool(blocker_map["post_v143_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v143_one_swap_improvement_found"]) == 3076
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v144_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v144_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v144_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v144_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v144_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v144 post-v143 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v144 proves the v143 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v144 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v144: Post-v143 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3076`" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v145_next_one_swap_repair_requires_repricing() -> None:
    status = _read_json("paper4_v145_status.json")

    assert status["phase"] == "v145_next_one_swap_repair"
    assert status["schema_version"] == "2026-05-15.145"
    assert status["allocation_rows_v145"] == 171
    assert status["summary_rows_v145"] == 1
    assert status["action_rows_v145"] == 1
    assert status["source_summary_rows_v145"] == 51
    assert status["claim_blocker_rows_v145"] == 4
    assert status["added_loan_id_v145"] == "162406873"
    assert status["dropped_loan_id_v145"] == "127178217"
    assert status["selected_rows_v145"] == 171
    assert status["portfolio_exposure_v145"] == pytest.approx(842550.0)
    assert status["objective_return_v145"] == pytest.approx(174.94891652776732)
    assert status["scenario_loss_cvar90_v145"] == pytest.approx(94205.09531368258)
    assert status["source_cap_violations_v145"] == 0
    assert status["delta_return_vs_v143_v145"] == pytest.approx(82.3318659743536)
    assert status["delta_cvar90_vs_v143_v145"] == pytest.approx(161.9852652677073)
    assert status["delta_exposure_vs_v143_v145"] == pytest.approx(100.0)
    assert status["budget_feasible_v145"] is True
    assert status["source_feasible_v145"] is True
    assert status["cvar_feasible_v145"] is True
    assert status["repair_candidate_feasible_v145"] is True
    assert status["post_repair_one_swap_optimality_claim_allowed_v145"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v145"] is False
    assert status["paper1_promotion_allowed_v145"] is False
    assert status["paper4_working_champion_changed_v145"] is False
    assert status["paper4_final_promotion_created"] is False

    allocations = pd.read_parquet(
        TABLE_DIR / "paper4_v145_next_one_swap_repair_allocations.parquet"
    )
    assert {
        "loan_id",
        "loan_amnt",
        "mean_return_v145",
        "selected_v145",
        "portfolio_label_v145",
        "repair_action_v145",
        "claim_boundary_v145",
    }.issubset(allocations.columns)
    assert len(allocations) == status["allocation_rows_v145"]
    assert int(allocations["selected_v145"].sum()) == status["selected_rows_v145"]
    assert allocations["loan_amnt"].sum() == pytest.approx(status["portfolio_exposure_v145"])
    assert "162406873" in set(allocations["loan_id"].astype(str))
    assert "127178217" not in set(allocations["loan_id"].astype(str))
    assert set(allocations["repair_action_v145"]) == {
        "added_from_v144_best_swap",
        "kept_from_v143",
    }
    assert allocations["claim_boundary_v145"].str.contains("requires post-repair repricing").all()

    summary = _read_csv("paper4_v145_next_one_swap_repair_summary.csv")
    row = summary.iloc[0]
    assert bool(row["repair_candidate_feasible_v145"]) is True
    assert float(row["min_source_slack_v145"]) == pytest.approx(0.00011275295234702831)
    assert bool(row["post_repair_one_swap_optimality_claim_allowed_v145"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v145"]) is False
    assert "must rerun omitted-universe pricing" in str(row["claim_boundary_v145"])

    action = _read_csv("paper4_v145_next_one_swap_repair_action.csv")
    action_row = action.iloc[0]
    assert str(action_row["added_loan_id_v145"]) == status["added_loan_id_v145"]
    assert str(action_row["dropped_loan_id_v145"]) == status["dropped_loan_id_v145"]
    assert float(action_row["return_delta_v145"]) == pytest.approx(82.33186597435457)
    assert float(action_row["exposure_after_repair_v145"]) == pytest.approx(842550.0)
    assert int(action_row["source_cap_violations_after_repair_v145"]) == 0

    source_summary = _read_csv("paper4_v145_next_one_swap_repair_source_summary.csv")
    assert {
        "source_family",
        "source_id",
        "source_share_v145",
        "source_slack_v145",
        "source_cap_violated_v145",
    }.issubset(source_summary.columns)
    assert len(source_summary) == status["source_summary_rows_v145"]
    assert not source_summary["source_cap_violated_v145"].astype(bool).any()

    blockers = _read_csv("paper4_v145_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v145"], blockers["blocking_v145"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v145"], blockers["evidence_count_v145"], strict=False)
    )
    assert bool(blocker_map["next_one_swap_repair_candidate_created"]) is False
    assert int(evidence_map["next_one_swap_repair_candidate_created"]) == 1
    assert bool(blocker_map["post_repair_one_swap_repricing_missing"]) is True
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v145_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v145_next_one_swap_repair_executed"]) is True
    assert bool(claim_map["v145_repair_candidate_feasible"]) is True
    assert bool(claim_map["v145_post_repair_one_swap_optimality"]) is False
    assert bool(claim_map["v145_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v145_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v145 thirty-second one-swap repair candidate." in set(
        current_boundaries["claim"]
    )
    assert "v145 repaired portfolio is post-repair locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v145 replaces Paper Estrella or proves full-universe integer optimality." in set(
        current_boundaries["claim"]
    )

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v145: Thirty-Second One-Swap Repair Candidate" in notebook
    assert "v146 post-repair one-swap pricing" in notebook
    assert set(_registered_paper4_pages()) == CURATED_PAPER4_PAGES
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v146_post_v145_reprice_still_finds_improvements() -> None:
    status = _read_json("paper4_v146_status.json")

    assert status["phase"] == "v146_post_v145_one_swap_reprice"
    assert status["schema_version"] == "2026-05-15.146"
    assert status["summary_rows_v146"] == 1
    assert status["stage_summary_rows_v146"] == 6
    assert status["candidate_pair_rows_v146"] == 3529
    assert status["top_candidate_rows_v146"] == 200
    assert status["claim_blocker_rows_v146"] == 3
    assert status["selected_rows_v146"] == 171
    assert status["candidate_add_rows_v146"] == 276698
    assert status["total_pair_rows_screened_v146"] == 47315358
    assert status["return_improving_pair_rows_v146"] == 1855901
    assert status["budget_return_feasible_pair_rows_v146"] == 1124085
    assert status["source_prefilter_pair_rows_v146"] == 4157
    assert status["source_exact_pair_rows_v146"] == 3529
    assert status["cvar_feasible_pair_rows_v146"] == 3529
    assert status["one_swap_improving_rows_v146"] == 3529
    assert status["best_one_swap_return_delta_v146"] == pytest.approx(136.7713455210524)
    assert status["best_one_swap_cvar90_after_v146"] == pytest.approx(94370.5136893005)
    assert status["post_repair_one_swap_local_optimality_cleared_v146"] is False
    assert status["full_universe_integer_optimality_claim_allowed_v146"] is False
    assert status["paper1_promotion_allowed_v146"] is False
    assert status["paper4_working_champion_changed_v146"] is False
    assert status["paper4_final_promotion_created"] is False

    probe = _read_csv("paper4_v146_post_repair_one_swap_reprice.csv")
    assert {
        "added_loan_id_v146",
        "dropped_loan_id_v146",
        "return_delta_v146",
        "objective_return_after_swap_v146",
        "budget_swap_feasible_v146",
        "source_swap_feasible_v146",
        "source_cap_violations_after_swap_v146",
        "cvar_swap_feasible_v146",
        "one_swap_improves_return_v146",
        "claim_boundary_v146",
    }.issubset(probe.columns)
    assert len(probe) == status["candidate_pair_rows_v146"]
    assert probe["return_delta_v146"].gt(0).all()
    assert probe["budget_swap_feasible_v146"].astype(bool).all()
    assert probe["source_swap_feasible_v146"].astype(bool).all()
    assert probe["cvar_swap_feasible_v146"].astype(bool).all()
    assert probe["one_swap_improves_return_v146"].astype(bool).all()
    assert int(probe["source_cap_violations_after_swap_v146"].sum()) == 0
    assert probe["return_delta_v146"].max() == pytest.approx(
        status["best_one_swap_return_delta_v146"]
    )
    assert probe["claim_boundary_v146"].str.contains("not multi-swap or global proof").all()

    top_candidates = _read_csv("paper4_v146_post_repair_one_swap_top_candidates.csv")
    assert len(top_candidates) == status["top_candidate_rows_v146"]
    best = top_candidates.iloc[0]
    assert str(best["added_loan_id_v146"]) == "151432552"
    assert str(best["dropped_loan_id_v146"]) == "127146282"
    assert float(best["return_delta_v146"]) == pytest.approx(
        status["best_one_swap_return_delta_v146"]
    )
    assert float(best["exposure_after_swap_v146"]) == pytest.approx(842450.0)
    assert bool(best["one_swap_improves_return_v146"]) is True

    summary = _read_csv("paper4_v146_post_repair_one_swap_summary.csv")
    row = summary.iloc[0]
    assert int(row["one_swap_improving_rows_v146"]) == status["one_swap_improving_rows_v146"]
    assert float(row["current_exposure_v146"]) == pytest.approx(842550.0)
    assert float(row["current_objective_return_v146"]) == pytest.approx(174.94891652776732)
    assert bool(row["post_repair_one_swap_local_optimality_cleared_v146"]) is False
    assert bool(row["full_universe_integer_optimality_claim_allowed_v146"]) is False
    assert "repeat repair/repricing" in str(row["claim_boundary_v146"])

    stage_summary = _read_csv("paper4_v146_post_repair_one_swap_stage_summary.csv")
    assert {"stage_v146", "pair_rows_v146", "claim_boundary_v146"}.issubset(stage_summary.columns)
    stage_map = dict(
        zip(stage_summary["stage_v146"], stage_summary["pair_rows_v146"], strict=False)
    )
    assert int(stage_map["all_pairs"]) == status["total_pair_rows_screened_v146"]
    assert int(stage_map["return_improving"]) == status["return_improving_pair_rows_v146"]
    assert int(stage_map["cvar_feasible_improving"]) == status["one_swap_improving_rows_v146"]

    blockers = _read_csv("paper4_v146_claim_blockers.csv")
    blocker_map = dict(zip(blockers["blocker_id_v146"], blockers["blocking_v146"], strict=False))
    evidence_map = dict(
        zip(blockers["blocker_id_v146"], blockers["evidence_count_v146"], strict=False)
    )
    assert bool(blocker_map["post_v145_one_swap_improvement_found"]) is True
    assert int(evidence_map["post_v145_one_swap_improvement_found"]) == 3529
    assert bool(blocker_map["multi_swap_integer_pricing_missing"]) is True
    assert bool(blocker_map["global_integer_gap_certificate_missing"]) is True

    claim_delta = _read_csv("paper4_v146_claim_matrix_delta.csv")
    claim_map = dict(zip(claim_delta["claim_id"], claim_delta["allowed"], strict=False))
    assert bool(claim_map["v146_post_repair_one_swap_reprice_executed"]) is True
    assert bool(claim_map["v146_post_repair_one_swap_local_optimality"]) is False
    assert bool(claim_map["v146_full_universe_integer_optimality"]) is False
    assert bool(claim_map["v146_paper1_or_final_promotion"]) is False

    current_boundaries = _read_csv("paper4_current_claim_boundaries.csv")
    assert "Paper 4 has a v146 post-v145 one-swap pricing screen." in set(
        current_boundaries["claim"]
    )
    assert "v146 proves the v145 repaired portfolio is locally optimal." in set(
        current_boundaries["claim"]
    )
    assert "v146 proves full-universe integer optimality." in set(current_boundaries["claim"])

    notebook = (PAPER4_ROOT / "notes" / "paper4_living_lab_notebook.md").read_text(encoding="utf-8")
    assert "Wave v146: Post-v145 One-Swap Repricing" in notebook
    assert "CVaR-feasible improving one-swaps: `3529`" in notebook
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
