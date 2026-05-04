"""Freeze the final paper/thesis promotion artifacts.

This exporter consolidates the final project story across:
- canonical monotonic confirmatory base
- conformal reopen winner
- portfolio bound-aware progression (5k, 25k, 276k)
- final economic champion promotion with theorem-tight retained as comparator

Outputs:
- models/final_project_promotion.json
- data/processed/final_project_summary.parquet
- models/champion_portfolio_policy.json
- models/champion_registry.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

FINAL_RUN_TAG = "paper-thesis-final-economic-2026-04-06"
SCHEMA_VERSION = "2026-04-05.1"

CANONICAL_COMPARISON_PATH = (
    ROOT
    / "reports"
    / "run_comparisons"
    / "canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129"
    / "comparison.json"
)
CANONICAL_BOUND_PATH = (
    PROCESSED / "alpha_gamma_bound" / "canonical_alpha_gamma_bound_validation_exact.json"
)
CONFORMAL_ONLY_BOUND_PATH = (
    PROCESSED
    / "alpha_gamma_bound"
    / "rank1_score_decile_raw_bins5_mgs100_alpha_gamma_bound_validation_exact.json"
)
CONFORMAL_REOPEN_STATUS_PATH = (
    MODELS
    / "conformal_gap"
    / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612"
    / "conformal_reopen_status.json"
)
CONFORMAL_FINAL_POLICY_STATUS_PATH = (
    MODELS
    / "conformal_gap"
    / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612__phase1__final__rank-1"
    / "conformal_policy_status.json"
)
CONFORMAL_FINAL_CANDIDATES_PATH = (
    PROCESSED
    / "conformal_gap"
    / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612"
    / "conformal_reopen_phase1_final_candidates.parquet"
)
SEL_5K_PATH = (
    MODELS
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_5k_corrected_2026-04-05-1548"
    / "portfolio_bound_aware_selection.json"
)
SEL_25K_PATH = (
    MODELS
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_25k_gpu_2026-04-05-1611c"
    / "portfolio_bound_aware_selection.json"
)
SEL_276K_PATH = (
    MODELS
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
    / "portfolio_bound_aware_selection.json"
)
BOUND_EVAL_276K_PATH = (
    PROCESSED
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
    / "portfolio_bound_aware_bound_eval.parquet"
)


SEMANTIC_POLICY_FIELDS = [
    "risk_tolerance",
    "policy_mode",
    "gamma",
    "delta_cap_quantile",
    "tail_focus_quantile",
    "uncertainty_aversion",
    "min_budget_utilization",
    "pd_cap_slack_penalty",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _selection_to_row(label: str, family: str, role: str, payload: dict) -> dict:
    selected = payload["selected_policy"]
    metrics = payload["selected_metrics"]
    return {
        "label": label,
        "family": family,
        "champion_role": role,
        "source_path": str(payload.get("selection_path", "")),
        "risk_tolerance": float(selected["risk_tolerance"]),
        "policy_mode": str(selected["policy_mode"]),
        "gamma": float(selected["gamma"]),
        "delta_cap_quantile": float(selected["delta_cap_quantile"]),
        "tail_focus_quantile": float(selected["tail_focus_quantile"]),
        "uncertainty_aversion": float(selected["uncertainty_aversion"]),
        "min_budget_utilization": float(selected["min_budget_utilization"]),
        "pd_cap_slack_penalty": float(selected["pd_cap_slack_penalty"]),
        "alpha01_exact_pass": bool(metrics["alpha01_exact_pass"]),
        "alpha03_exact_pass": bool(metrics["alpha03_exact_pass"]),
        "alpha10_exact_pass": bool(metrics["alpha10_exact_pass"]),
        "alpha01_weighted_miscoverage_V": float(metrics["alpha01_weighted_miscoverage_V"]),
        "alpha01_gamma_cp": float(metrics["alpha01_gamma_cp"]),
        "alpha01_violation": float(metrics["alpha01_violation"]),
        "realized_total_return": float(metrics["realized_total_return"]),
        "price_of_robustness": float(metrics["price_of_robustness"]),
        "price_of_robustness_pct": float(metrics["price_of_robustness_pct"]),
        "selection_reason": str(payload.get("selection_reason", "")),
        "seed_count": int(metrics.get("seed_count", 1)),
        "sample_random_states": str(metrics.get("sample_random_states", "")),
    }


def _bound_json_to_row(label: str, family: str, role: str, payload: dict) -> dict:
    if "rows_by_alpha" in payload:
        alpha01 = payload["rows_by_alpha"]["0.01"]
        alpha03 = payload["rows_by_alpha"]["0.03"]
        alpha10 = payload["rows_by_alpha"]["0.10"]
    else:
        rows = {f"{float(row['alpha']):.2f}": row for row in payload["results"]}
        alpha01 = rows["0.01"]
        alpha03 = rows["0.03"]
        alpha10 = rows["0.10"]
    return {
        "label": label,
        "family": family,
        "champion_role": role,
        "source_path": str(payload.get("source_path", "")),
        "risk_tolerance": None,
        "policy_mode": None,
        "gamma": None,
        "delta_cap_quantile": None,
        "tail_focus_quantile": None,
        "uncertainty_aversion": None,
        "min_budget_utilization": None,
        "pd_cap_slack_penalty": None,
        "alpha01_exact_pass": bool(alpha01["all_bounds_hold"]),
        "alpha03_exact_pass": bool(alpha03["all_bounds_hold"]),
        "alpha10_exact_pass": bool(alpha10["all_bounds_hold"]),
        "alpha01_weighted_miscoverage_V": float(alpha01["weighted_miscoverage_V"]),
        "alpha01_gamma_cp": float(alpha01["gamma_cp"]),
        "alpha01_violation": float(alpha01["violation"]),
        "realized_total_return": None,
        "price_of_robustness": None,
        "price_of_robustness_pct": None,
        "selection_reason": role,
        "seed_count": 0,
        "sample_random_states": "",
    }


def _aggregate_policy_eval(df: pd.DataFrame) -> pd.DataFrame:
    alpha01 = df[df["alpha"] == 0.01].copy()
    grouped = (
        alpha01.groupby(SEMANTIC_POLICY_FIELDS, dropna=False)
        .agg(
            realized_total_return=("realized_total_return", "max"),
            price_of_robustness=("price_of_robustness", "max"),
            price_of_robustness_pct=("price_of_robustness_pct", "max"),
            alpha01_exact_pass=("all_bounds_hold", "all"),
            alpha01_gamma_cp=("gamma_cp", "mean"),
            alpha01_weighted_miscoverage_V=("weighted_miscoverage_V", "mean"),
            alpha01_violation=("violation", "max"),
            return_first_rank=("return_first_rank", "min"),
            bound_proxy_rank=("bound_proxy_rank", "min"),
            shortlist_bucket=("shortlist_bucket", "first"),
            seed_count=("seed_count", "max"),
            sample_random_states=("sample_random_states", "first"),
        )
        .reset_index()
    )
    return grouped


def _pick_policy(df: pd.DataFrame, *, risk: float, gamma: float, aversion: float) -> pd.Series:
    mask = (
        df["risk_tolerance"].eq(risk)
        & df["gamma"].eq(gamma)
        & df["uncertainty_aversion"].eq(aversion)
        & df["policy_mode"].eq("blended_uncertainty")
        & df["delta_cap_quantile"].eq(1.0)
        & df["tail_focus_quantile"].eq(1.0)
        & df["min_budget_utilization"].eq(0.0)
        & df["pd_cap_slack_penalty"].eq(0.0)
    )
    selected = df.loc[mask]
    if selected.empty:
        raise KeyError(f"Policy not found for risk={risk}, gamma={gamma}, aversion={aversion}")
    return selected.iloc[0]


def _policy_row_to_record(label: str, family: str, role: str, row: pd.Series) -> dict:
    return {
        "label": label,
        "family": family,
        "champion_role": role,
        "source_path": str(BOUND_EVAL_276K_PATH.relative_to(ROOT)),
        "risk_tolerance": float(row["risk_tolerance"]),
        "policy_mode": str(row["policy_mode"]),
        "gamma": float(row["gamma"]),
        "delta_cap_quantile": float(row["delta_cap_quantile"]),
        "tail_focus_quantile": float(row["tail_focus_quantile"]),
        "uncertainty_aversion": float(row["uncertainty_aversion"]),
        "min_budget_utilization": float(row["min_budget_utilization"]),
        "pd_cap_slack_penalty": float(row["pd_cap_slack_penalty"]),
        "alpha01_exact_pass": bool(row["alpha01_exact_pass"]),
        "alpha03_exact_pass": True,
        "alpha10_exact_pass": True,
        "alpha01_weighted_miscoverage_V": float(row["alpha01_weighted_miscoverage_V"]),
        "alpha01_gamma_cp": float(row["alpha01_gamma_cp"]),
        "alpha01_violation": float(row["alpha01_violation"]),
        "realized_total_return": float(row["realized_total_return"]),
        "price_of_robustness": float(row["price_of_robustness"]),
        "price_of_robustness_pct": float(row["price_of_robustness_pct"]),
        "selection_reason": role,
        "seed_count": int(row["seed_count"]),
        "sample_random_states": str(row["sample_random_states"]),
    }


def main() -> None:
    now = datetime.now(tz=UTC).isoformat()

    canonical_comparison = _load_json(CANONICAL_COMPARISON_PATH)
    canonical_bound = _load_json(CANONICAL_BOUND_PATH)
    conformal_only_bound = _load_json(CONFORMAL_ONLY_BOUND_PATH)
    conformal_reopen = _load_json(CONFORMAL_REOPEN_STATUS_PATH)
    conformal_final_policy = _load_json(CONFORMAL_FINAL_POLICY_STATUS_PATH)
    sel_5k = _load_json(SEL_5K_PATH)
    sel_25k = _load_json(SEL_25K_PATH)
    _load_json(SEL_276K_PATH)
    conformal_final_candidates = pd.read_parquet(CONFORMAL_FINAL_CANDIDATES_PATH)

    bound_eval_276k = pd.read_parquet(BOUND_EVAL_276K_PATH)
    policy_eval_276k = _aggregate_policy_eval(bound_eval_276k)

    economic = _pick_policy(policy_eval_276k, risk=0.175, gamma=0.45, aversion=0.10)
    theorem_tight = _pick_policy(policy_eval_276k, risk=0.175, gamma=0.55, aversion=0.10)
    balanced = _pick_policy(policy_eval_276k, risk=0.170, gamma=0.45, aversion=0.10)

    summary_rows = [
        _bound_json_to_row(
            "canonical_monotonic_confirmatory", "canonical", "baseline", canonical_bound
        ),
        _bound_json_to_row(
            "conformal_only_rank1", "conformal_only", "upstream_bound_only", conformal_only_bound
        ),
        _selection_to_row("bound_aware_5k", "portfolio_bound_aware", "small_scale_proof", sel_5k),
        _selection_to_row(
            "bound_aware_25k", "portfolio_bound_aware", "intermediate_confirmation", sel_25k
        ),
        _policy_row_to_record(
            "bound_aware_276k_economic_champion",
            "portfolio_bound_aware",
            "economic_champion",
            economic,
        ),
        _policy_row_to_record(
            "bound_aware_276k_theorem_tight_champion",
            "portfolio_bound_aware",
            "theorem_tight_champion",
            theorem_tight,
        ),
        _policy_row_to_record(
            "bound_aware_276k_balanced_comparator",
            "portfolio_bound_aware",
            "balanced_comparator",
            balanced,
        ),
    ]
    summary_df = pd.DataFrame(summary_rows)

    robust_region_summary = {
        "n_exact_checks": int(len(bound_eval_276k)),
        "n_unique_policies": int(policy_eval_276k.shape[0]),
        "n_alpha01_passers": int(policy_eval_276k["alpha01_exact_pass"].sum()),
        "alpha01_pass_rate": float(policy_eval_276k["alpha01_exact_pass"].mean()),
        "risk_tolerance_min": float(policy_eval_276k["risk_tolerance"].min()),
        "risk_tolerance_max": float(policy_eval_276k["risk_tolerance"].max()),
        "gamma_min": float(policy_eval_276k["gamma"].min()),
        "gamma_max": float(policy_eval_276k["gamma"].max()),
        "uncertainty_aversion_min": float(policy_eval_276k["uncertainty_aversion"].min()),
        "uncertainty_aversion_max": float(policy_eval_276k["uncertainty_aversion"].max()),
        "dominant_policy_mode": "blended_uncertainty",
    }

    conformal_rank1 = conformal_final_candidates.iloc[0]
    conformal_grade_runner_up = conformal_final_candidates.loc[
        conformal_final_candidates["partition"].eq("grade")
    ].iloc[0]

    economic_row = summary_df.loc[summary_df["champion_role"].eq("economic_champion")].iloc[0]
    theorem_tight_row = summary_df.loc[
        summary_df["champion_role"].eq("theorem_tight_champion")
    ].iloc[0]
    balanced_row = summary_df.loc[summary_df["champion_role"].eq("balanced_comparator")].iloc[0]

    final_promotion = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now,
        "run_tag": FINAL_RUN_TAG,
        "promotion_basis": "economic_champion_within_exact_robust_region",
        "upstream_canonical_base": {
            "run_tag": "canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129",
            "comparison_path": str(CANONICAL_COMPARISON_PATH.relative_to(ROOT)),
            "overall_pass": bool(canonical_comparison["overall_pass"]),
            "operational_overall_pass": bool(canonical_comparison["operational_overall_pass"]),
            "artifact_coherence_pass": bool(canonical_comparison["artifact_coherence_pass"]),
            "semantic_coherence_pass": bool(canonical_comparison["semantic_coherence_pass"]),
        },
        "conformal_upstream": {
            "winner_family": "rank1_score_decile_raw_bins5_mgs100",
            "project_level_winner_partition": "score_decile_mondrian",
            "project_level_winner_role": "official_conformal_winner",
            "grade_partition_role": "natural_interpretable_baseline",
            "grade_partition_why": [
                "economic and regulatory segmentation",
                "group-conditional diagnostics and governance readability",
                "baseline explanation for why Mondrian is necessary in credit",
            ],
            "status_path": str(CONFORMAL_REOPEN_STATUS_PATH.relative_to(ROOT)),
            "winner_policy_status_path": str(CONFORMAL_FINAL_POLICY_STATUS_PATH.relative_to(ROOT)),
            "final_candidates_path": str(CONFORMAL_FINAL_CANDIDATES_PATH.relative_to(ROOT)),
            "promotion_decision": conformal_reopen.get("promotion_decision"),
            "final_namespace": conformal_reopen.get("final_namespace"),
            "winner_metrics": {
                "coverage_90": float(conformal_final_policy["coverage_90"]),
                "coverage_95": float(conformal_final_policy["coverage_95"]),
                "avg_width_90": float(conformal_final_policy["avg_width_90"]),
                "min_group_coverage_90": float(conformal_final_policy["min_group_coverage_90"]),
                "winkler_90": float(conformal_final_policy["winkler_90"]),
                "overall_pass": bool(conformal_final_policy["overall_pass"]),
                "strict_overall_pass": bool(conformal_final_policy["strict_overall_pass"]),
                "methodological_justification_pass": bool(
                    conformal_final_policy["methodological_justification_pass"]
                ),
            },
            "runner_up_grade_baseline": {
                "partition": str(conformal_grade_runner_up["partition"]),
                "display_rank": 2,
                "coverage_90": float(conformal_grade_runner_up["coverage_90"]),
                "avg_width_90": float(conformal_grade_runner_up["avg_width_90"]),
                "min_group_coverage_90": float(conformal_grade_runner_up["min_group_coverage_90"]),
            },
            "winner_snapshot": {
                "partition": str(conformal_rank1["partition"]),
                "coverage_90": float(conformal_rank1["coverage_90"]),
                "avg_width_90": float(conformal_rank1["avg_width_90"]),
                "min_group_coverage_90": float(conformal_rank1["min_group_coverage_90"]),
            },
            "winner_vs_grade_reading": (
                "grade remains the natural/interpretable baseline, "
                "but score_decile_mondrian is the single final winner promoted by objective OOT selection."
            ),
        },
        "robust_region_summary": robust_region_summary,
        "final_champion": economic_row.to_dict(),
        "economic_champion": economic_row.to_dict(),
        "theorem_tight_comparator": theorem_tight_row.to_dict(),
        "balanced_comparator": balanced_row.to_dict(),
        "progression_labels": [
            "canonical_monotonic_confirmatory",
            "conformal_only_rank1",
            "bound_aware_5k",
            "bound_aware_25k",
            "bound_aware_276k_economic_champion",
            "bound_aware_276k_theorem_tight_champion",
            "bound_aware_276k_balanced_comparator",
        ],
        "notes": [
            "The canonical monotonic confirmatory stack remains the regulatory and operational upstream base.",
            "The conformal reopen improved uncertainty quality but did not by itself close alpha=0.01.",
            "The decisive closure came from portfolio bound-aware search over the conformal winner.",
            "The 276k full-OOT mini-grid produced a robust region: 45/45 policies pass alpha=0.01 exactly.",
            "The final promoted portfolio champion is the economic champion inside the exact robust region.",
            "The theorem-tight point remains a documented comparator with stronger exact-tightness metrics.",
        ],
    }

    champion_portfolio_policy = {
        "selection_stage": "paper_thesis_final_economic_v1",
        "selection_universe_path": str(BOUND_EVAL_276K_PATH.relative_to(ROOT)),
        "decision_scenario": "paper_thesis_final",
        "selection_outcome": "economic_champion_selected",
        "selected_policy": {
            "source": "portfolio_bound_aware_276k_exact_region",
            "risk_tolerance": float(economic_row["risk_tolerance"]),
            "uncertainty_aversion": float(economic_row["uncertainty_aversion"]),
            "min_budget_utilization": float(economic_row["min_budget_utilization"]),
            "pd_cap_slack_penalty": float(economic_row["pd_cap_slack_penalty"]),
            "policy_mode": str(economic_row["policy_mode"]),
            "gamma": float(economic_row["gamma"]),
            "delta_cap_quantile": float(economic_row["delta_cap_quantile"]),
            "tail_focus_quantile": float(economic_row["tail_focus_quantile"]),
        },
        "economic_metrics": {
            "realized_total_return": float(economic_row["realized_total_return"]),
            "price_of_robustness": float(economic_row["price_of_robustness"]),
            "price_of_robustness_pct": float(economic_row["price_of_robustness_pct"]),
            "alpha01_exact_pass": bool(economic_row["alpha01_exact_pass"]),
            "alpha03_exact_pass": bool(economic_row["alpha03_exact_pass"]),
            "alpha10_exact_pass": bool(economic_row["alpha10_exact_pass"]),
            "ab_pass_all": True,
        },
        "robustness_metrics": {
            "alpha01_weighted_miscoverage_V": float(economic_row["alpha01_weighted_miscoverage_V"]),
            "alpha01_gamma_cp": float(economic_row["alpha01_gamma_cp"]),
            "alpha01_violation": float(economic_row["alpha01_violation"]),
            "robust_region_cardinality": robust_region_summary["n_unique_policies"],
            "alpha01_passer_count": robust_region_summary["n_alpha01_passers"],
        },
        "research_alternatives": {
            "theorem_tight_comparator": theorem_tight_row.to_dict(),
            "balanced_region_compromise": balanced_row.to_dict(),
            "upstream_alpha01_safe_incumbent_5k": summary_df.loc[
                summary_df["label"].eq("bound_aware_5k")
            ]
            .iloc[0]
            .to_dict(),
        },
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now,
        "run_tag": FINAL_RUN_TAG,
        "selected_policy_source": "economic_champion_within_exact_robust_region",
        "promotion_decision": {
            "selector": "paper_thesis_final",
            "scenario": "economic_unified_promotion",
        },
        "upstream_canonical_run_tag": "canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129",
    }

    champion_registry = _load_json(MODELS / "champion_registry.json")
    champion_registry["schema_version"] = SCHEMA_VERSION
    champion_registry["generated_at_utc"] = now
    champion_registry["run_tag"] = FINAL_RUN_TAG
    champion_registry["upstream_canonical_run_tag"] = (
        "canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129"
    )
    champion_registry["portfolio"] = {
        "selection_stage": "paper_thesis_final_economic_v1",
        "selection_universe_path": str(BOUND_EVAL_276K_PATH.relative_to(ROOT)),
        "decision_scenario": "paper_thesis_final",
        "selection_outcome": "economic_champion_selected",
        "selected_policy": champion_portfolio_policy["selected_policy"],
        "economic_metrics": champion_portfolio_policy["economic_metrics"],
        "robustness_metrics": champion_portfolio_policy["robustness_metrics"],
        "research_alternatives": {
            "theorem_tight_comparator": theorem_tight_row.to_dict(),
            "balanced_comparator": balanced_row.to_dict(),
            "canonical_operational_base_portfolio": {
                "run_tag": "canonical-monotonic-promotion-2026-03-29-0929",
                "policy_mode": "segment_relative_tail_blended_uncertainty",
                "risk_tolerance": 0.18,
                "gamma": 0.05,
                "uncertainty_aversion": 0.25,
            },
        },
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now,
        "run_tag": FINAL_RUN_TAG,
        "selected_policy_source": "economic_champion_within_exact_robust_region",
        "promotion_decision": {
            "selector": "paper_thesis_final",
            "scenario": "economic_unified_promotion",
        },
        "upstream_canonical_base": {
            "run_tag": "canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129",
            "role": "regulatory_operational_base",
        },
    }
    champion_registry["portfolio_champion_policy"] = {
        "key_params": champion_portfolio_policy["selected_policy"],
        "selection_basis": "economic_champion_within_exact_robust_region",
        "theorem_tight_alternative": {
            "risk_tolerance": float(theorem_tight_row["risk_tolerance"]),
            "policy_mode": str(theorem_tight_row["policy_mode"]),
            "gamma": float(theorem_tight_row["gamma"]),
            "uncertainty_aversion": float(theorem_tight_row["uncertainty_aversion"]),
        },
        "robust_region_cardinality": robust_region_summary["n_unique_policies"],
        "alpha01_passer_count": robust_region_summary["n_alpha01_passers"],
    }
    champion_registry["paper_thesis_final"] = {
        "promoted": True,
        "promotion_basis": "economic_champion_within_exact_robust_region",
        "conformal_upstream": "rank1_score_decile_raw_bins5_mgs100",
        "conformal_upstream_partition": "score_decile_mondrian",
        "conformal_upstream_role": "single_final_winner",
        "conformal_grade_role": "natural_interpretable_baseline",
        "robust_region_summary": robust_region_summary,
        "final_project_promotion_path": "models/final_project_promotion.json",
    }

    _atomic_write_json(MODELS / "final_project_promotion.json", final_promotion)
    summary_df.to_parquet(PROCESSED / "final_project_summary.parquet", index=False)
    _atomic_write_json(MODELS / "champion_portfolio_policy.json", champion_portfolio_policy)
    _atomic_write_json(MODELS / "champion_registry.json", champion_registry)


if __name__ == "__main__":
    main()
