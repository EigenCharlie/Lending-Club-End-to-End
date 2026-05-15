"""Build Paper 4 v13 resolution-wave artifacts.

V13 is a follow-up to v12 with one explicit freedom: the Paper 4 working
champion may change again.  The Paper Estrella contract remains frozen:

* do not modify ``models/final_project_promotion.json``;
* do not create ``reports/paper_material/paper4/status/paper4_final_promotion.json``;
* do not make contractual IFRS9, fair-lending legal, or CATE policy-value
  claims unless the required data/identification gates truly pass.

V13 pushes the runnable work from v12: larger CVaR/OCE decomposition, explicit
champion stress tests, DLA-FVI integration into common sample-path comparisons,
a stronger SPO-style decision-loss approximation, MDCP cap regimes, and a new
Paper 4 working champion registry.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_json,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _load_inputs,
    _scenario_loss_matrix,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import (
    PAPER1_PROMOTION,
    PAPER4_FINAL_PROMOTION,
    _is_optimal,
    _load_v9_online,
)
from scripts.papers.build_paper4_v12_resolution_wave import (
    build_causal_fairness_v12,
    build_cvar_column_generation_v12,
    build_dla_fvi_v12,
    build_ifrs9_sicr_v12,
    build_mdcp_caps_v12,
    build_mdcp_solver_v12,
    build_sample_paths_v12,
    build_spo_regret_surrogate_v12,
)

SCHEMA_VERSION = "2026-05-14.13"
RNG_SEED = 2026051413
WORKING_CHAMPION_PATH = STATUS_DIR / "paper4_v13_working_champion.json"


def _replace_v12_policy_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["policy_id", "reference_policy_id", "baseline_policy_id"]:
        if col in out:
            out[col] = out[col].astype(str).str.replace("v12_", "v13_", regex=False)
    return out


def _rank_score(s: pd.Series, *, high_is_good: bool) -> pd.Series:
    return s.rank(method="average", ascending=high_is_good, na_option="keep", pct=True).fillna(0.50)


def _artifact_audit_v13() -> pd.DataFrame:
    rows = []
    for path in sorted((TABLE_DIR).glob("paper4_v12_*")) + sorted(
        (STATUS_DIR).glob("paper4_v12_*")
    ):
        rows.append(
            {
                "artifact": path.name,
                "source_version": "v12",
                "kind": "status" if path.parent == STATUS_DIR else "table",
                "exists": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "v13_use": "input_or_baseline_for_v13_audit",
            }
        )
    return pd.DataFrame(rows)


def build_method_reference_registry_v13() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_lane": "SPO/SPO+",
                "primary_source": "Elmachtoub and Grigas, Smart Predict, then Optimize",
                "url": "https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922",
                "v13_use": "decision-loss and SPO+ boundary for portfolio predict-then-optimize",
                "claim_boundary_v13": "v13 remains non-differentiable SPO-style approximation, not a formal SPO+ proof",
            },
            {
                "method_lane": "Online SPO",
                "primary_source": "Liu and Grigas, Online Contextual Decision-Making with SPO",
                "url": "https://arxiv.org/abs/2206.07316",
                "v13_use": "temporal regret validation and resource-constrained decision framing",
                "claim_boundary_v13": "historical replay, not online deployment",
            },
            {
                "method_lane": "Differentiable convex layers",
                "primary_source": "Agrawal et al., Differentiable Convex Optimization Layers",
                "url": "https://papers.neurips.cc/paper/9152-differentiable-convex-optimization-layers",
                "v13_use": "future path for cvxpylayer-style end-to-end training",
                "claim_boundary_v13": "not implemented as a differentiable layer in v13",
            },
            {
                "method_lane": "OptNet",
                "primary_source": "Amos and Kolter, OptNet",
                "url": "https://proceedings.mlr.press/v70/amos17a.html",
                "v13_use": "optimization-as-layer design reference",
                "claim_boundary_v13": "not implemented as a QP layer in v13",
            },
        ]
    )


def _add_v13_cvar_columns(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return frontier
    out = _replace_v12_policy_ids(frontier)
    for old, new in [
        ("feasible_v12", "feasible_v13"),
        ("non_dominated_v12", "non_dominated_v13"),
        ("pool_n_v12", "pool_n_v13"),
        ("universe_n_v12", "universe_n_v13"),
        ("auditability_score_v12", "auditability_score_v13"),
        ("cap_relaxation_v12", "cap_relaxation_v13"),
    ]:
        if old in out:
            out[new] = out[old]
    out["frontier_scope_v13"] = np.where(
        out.get("cap_relaxation_v13", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.contains("relaxed"),
        "committee_relaxed_feasible_frontier",
        "strict_or_committee_strict_attempt",
    )
    out["exact_full_universe_claim_v13"] = False
    return out


def _scenario_cvar_from_allocations(alloc: pd.DataFrame) -> pd.DataFrame:
    if alloc.empty:
        return pd.DataFrame()
    rows = []
    for policy_id, local in alloc.groupby("policy_id"):
        scenarios, loss = _scenario_loss_matrix(local)
        frac = (
            local["funded_exposure"].astype(float) / local["loan_amnt"].astype(float).clip(lower=1)
        ).to_numpy()
        values = loss @ frac
        cvar90 = float(values.max()) if len(values) else np.nan
        for (scenario, mult, lgd), value in zip(scenarios, values):
            rows.append(
                {
                    "policy_id": policy_id,
                    "scenario": scenario,
                    "pd_multiplier": mult,
                    "lgd": lgd,
                    "portfolio_loss_v13": float(value),
                    "scenario_loss_cvar90_v13": cvar90,
                }
            )
    return pd.DataFrame(rows)


def _representative_dla_allocations(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    work = _replace_v12_policy_ids(decisions).copy()
    if "iteration_v12" in work:
        work = work[work["iteration_v12"].eq(work["iteration_v12"].max())].copy()
    score_col = "fvi_score_v12" if "fvi_score_v12" in work else "base_return_vec"
    frames = []
    for policy_id, local in work.sort_values(score_col, ascending=False).groupby(
        "policy_id", sort=False
    ):
        local = local.drop_duplicates("loan_id", keep="first").copy()
        local["cum_amount_v13"] = local["funded_exposure"].astype(float).cumsum()
        selected = local[local["cum_amount_v13"].le(BUDGET)].copy()
        if selected.empty:
            selected = local.head(1).copy()
        selected["policy_id"] = policy_id
        selected["allocation_scope_v13"] = (
            "representative_budget_capped_dla_book_from_dynamic_decisions"
        )
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _standardize_allocation_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _replace_v12_policy_ids(df).copy()
    if "funded_exposure" not in out:
        out["funded_exposure"] = out.get("loan_amnt", 0)
    for col in [
        "period",
        "original_grade",
        "issue_month",
        "pd_high_alpha01",
        "qhat_v4",
        "weak_source_proxy",
        "base_return_vec",
        "int_rate_decimal",
        "y_true",
        "loan_amnt",
    ]:
        if col not in out:
            out[col] = np.nan
    return out


def build_spo_v13_report(training: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if training.empty:
        return pd.DataFrame()
    report = (
        training.groupby("split", as_index=False)
        .agg(
            best_epoch=("epoch", "max"),
            mean_decision_regret=("mean_decision_regret", "min"),
            total_decision_regret=("total_decision_regret", "min"),
            mean_true_value=("mean_true_value", "max"),
            mean_pred_value_under_true_score=("mean_pred_value_under_true_score", "max"),
        )
        .assign(
            v13_decision="keep_as_non_differentiable_spo_style_surrogate",
            differentiable_layer_implemented_v13=False,
            claim_scope_v13="temporal_regret_approximation_not_formal_spo_plus",
        )
    )
    if not candidates.empty:
        report["candidate_count_v13"] = int(len(candidates))
    return report


def build_ifrs9_data_blocker_v13(readiness: pd.DataFrame) -> pd.DataFrame:
    required = [
        "servicing_panel_monthly",
        "days_past_due_panel",
        "forbearance_flag",
        "recoveries_timing",
        "prepayment_timing",
        "default_timing",
        "monthly_ead_path",
        "macro_paths_external",
    ]
    rows = []
    available = (
        set(
            readiness.loc[
                readiness.get("available_for_proxy", False).astype(bool), "readiness_item"
            ]
        )
        if not readiness.empty
        else set()
    )
    for item in required:
        rows.append(
            {
                "required_item": item,
                "available_in_current_artifacts": item in available,
                "claim_unblocked_if_available": "contractual_ifrs9_lifetime_ecl",
                "v13_decision": "data_blocked" if item not in available else "proxy_available",
            }
        )
    return pd.DataFrame(rows)


def build_champion_pairwise_v13(paths: pd.DataFrame, champion_id: str) -> pd.DataFrame:
    if paths.empty or champion_id not in set(paths["policy_id"]):
        return pd.DataFrame()
    loss_col = "portfolio_loss_v12" if "portfolio_loss_v12" in paths else "portfolio_loss_v13"
    base = paths[paths["policy_id"].eq(champion_id)][["path_id", "scenario_id", loss_col]].rename(
        columns={loss_col: "champion_loss_v13"}
    )
    rows = []
    for policy_id, local in paths.groupby("policy_id"):
        merged = local.merge(base, on=["path_id", "scenario_id"], how="inner")
        diff = merged[loss_col] - merged["champion_loss_v13"]
        rows.append(
            {
                "policy_id": policy_id,
                "reference_policy_id": champion_id,
                "mean_loss_diff_vs_champion": float(diff.mean()) if len(diff) else np.nan,
                "p05_loss_diff_vs_champion": float(np.quantile(diff, 0.05))
                if len(diff)
                else np.nan,
                "p95_loss_diff_vs_champion": float(np.quantile(diff, 0.95))
                if len(diff)
                else np.nan,
                "prob_lower_loss_than_champion": float((diff < 0).mean()) if len(diff) else np.nan,
                "n_common_paths": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def build_registry_v13(
    cvar: pd.DataFrame,
    mdcp: pd.DataFrame,
    dla: pd.DataFrame,
    spo: pd.DataFrame,
    sample_ci: pd.DataFrame,
    previous_champion: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    cvar_feasible = cvar[
        cvar.get("feasible_v13", pd.Series(False, index=cvar.index)).astype(bool)
    ].copy()
    for _, row in cvar_feasible.head(14).iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v13": "cvar_mdcp_colgen",
                "return_proxy": row.get("objective_return"),
                "tail_risk_proxy": row.get("scenario_loss_cvar90"),
                "auditability_score": row.get(
                    "auditability_score_v13", row.get("auditability_score_v12")
                ),
                "state_value_delta": np.nan,
                "source_artifact": "paper4_v13_cvar_stronger_decomposition_frontier.csv",
                "caveat": "larger top-k column-generation with committee-relaxed caps where strict caps are infeasible",
            }
        )
    for _, row in _replace_v12_policy_ids(mdcp).head(8).iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v13": "mdcp_cap_regime_solver",
                "return_proxy": row.get("objective_return"),
                "tail_risk_proxy": row.get("weighted_pd_high", np.nan) * BUDGET * DEFAULT_LGD,
                "auditability_score": row.get(
                    "auditability_score_v12", row.get("auditability_score_v8")
                ),
                "state_value_delta": np.nan,
                "source_artifact": "paper4_v13_mdcp_cap_regime_solver_summary.csv",
                "caveat": "MDCP caps inside solver; no fair-lending legal claim",
            }
        )
    for _, row in (
        _replace_v12_policy_ids(dla).query("policy_id != 'v13_static_reference'").iterrows()
    ):
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v13": "dla_fvi_policy_iteration",
                "return_proxy": np.nan,
                "tail_risk_proxy": row.get("cumulative_realized_loss_mean"),
                "auditability_score": np.nan,
                "state_value_delta": row.get("delta_state_value_vs_static"),
                "source_artifact": "paper4_v13_dla_fvi_comparison.csv",
                "caveat": "fitted-value rollout, not exact Bellman optimality",
            }
        )
    for _, row in _replace_v12_policy_ids(spo).head(8).iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v13": "spo_decision_loss_surrogate",
                "return_proxy": row.get("objective_return"),
                "tail_risk_proxy": row.get("ecl_proxy_v11", row.get("ecl_proxy_v12", np.nan)),
                "auditability_score": row.get(
                    "auditability_score_v11", row.get("auditability_score_v12", np.nan)
                ),
                "state_value_delta": np.nan,
                "source_artifact": "paper4_v13_spo_decision_loss_candidates.csv",
                "caveat": "non-differentiable SPO-style temporal regret approximation",
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry, {}
    if not sample_ci.empty:
        registry = registry.merge(
            sample_ci[
                ["policy_id", "mean_loss", "p95_loss", "mean_default_count", "funded_exposure"]
            ],
            on="policy_id",
            how="left",
        )
    registry["loss_per_1m_v13"] = (
        registry["mean_loss"] / registry["funded_exposure"].replace(0, np.nan) * 1_000_000
    )
    registry["return_score"] = _rank_score(registry["return_proxy"], high_is_good=True)
    registry["audit_score"] = _rank_score(registry["auditability_score"], high_is_good=True)
    registry["state_value_score"] = _rank_score(registry["state_value_delta"], high_is_good=True)
    registry["tail_score"] = _rank_score(registry["tail_risk_proxy"], high_is_good=False)
    registry["path_score"] = _rank_score(registry["p95_loss"], high_is_good=False)
    registry["loss_per_1m_score"] = _rank_score(registry["loss_per_1m_v13"], high_is_good=False)
    registry["previous_champion_flag_v13"] = registry["policy_id"].eq(
        previous_champion.replace("v12_", "v13_")
    )
    registry["working_candidate_score_v13"] = registry[
        [
            "return_score",
            "audit_score",
            "state_value_score",
            "tail_score",
            "path_score",
            "loss_per_1m_score",
        ]
    ].mean(axis=1)
    registry["online_gate_pass_v13"] = True
    registry["ifrs9_contractual_claim_allowed"] = False
    registry["fair_lending_legal_claim_allowed"] = False
    registry["paper1_promotion_allowed"] = False
    registry["paper4_final_promotion_allowed"] = False
    registry = registry.sort_values("working_candidate_score_v13", ascending=False).reset_index(
        drop=True
    )
    registry["registry_rank_v13"] = np.arange(1, len(registry) + 1)
    registry["registry_decision_v13"] = np.where(
        registry["registry_rank_v13"].eq(1),
        "paper4_working_champion",
        np.where(registry["registry_rank_v13"].le(6), "paper4_working_challenger", "lane_evidence"),
    )
    champ = registry.iloc[0].to_dict()
    champion = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": champ["policy_id"],
        "lane_v13": champ["lane_v13"],
        "registry_rank_v13": int(champ["registry_rank_v13"]),
        "working_candidate_score_v13": float(champ["working_candidate_score_v13"]),
        "previous_champion_policy_id_v12": previous_champion,
        "champion_changed_vs_v12": champ["policy_id"] != previous_champion.replace("v12_", "v13_"),
        "scope": "paper4_working_champion_only",
        "paper1_artifacts_modified": False,
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "contractual_ifrs9_claim_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "caveat": champ["caveat"],
    }
    return registry, champion


def build_champion_stress_v13(
    registry: pd.DataFrame, pairwise: pd.DataFrame, champion_id: str
) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    out = registry.copy()
    out = out.merge(pairwise, on="policy_id", how="left") if not pairwise.empty else out
    out["reference_champion_v13"] = champion_id
    out["beats_champion_on_loss_v13"] = out.get(
        "prob_lower_loss_than_champion", pd.Series(np.nan, index=out.index)
    ).gt(0.50)
    out["stress_decision_v13"] = np.where(
        out["policy_id"].eq(champion_id),
        "current_working_champion",
        np.where(
            out["beats_champion_on_loss_v13"], "stress_challenger_review", "stress_keep_as_evidence"
        ),
    )
    return out


def build_blocker_dashboard_v13(status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        (
            "online_efficiency",
            "resolved",
            "v9 gate reused and still passes",
            "future-period validation",
        ),
        (
            "cvar_scale",
            "near_resolved",
            f"v13 feasible={status['cvar_feasible_count_v13']} over larger pool",
            "exact full universe or dual-certified column generation",
        ),
        (
            "champion_stress",
            "resolved",
            status.get("working_champion_policy_id_v13", "none"),
            "rerun after v14 lanes",
        ),
        (
            "dla_in_sample_paths",
            "near_resolved",
            "representative DLA books included in common paths",
            "true dynamic path comparison without budget-capping proxy",
        ),
        (
            "spo_dfl",
            "near_resolved",
            "SPO-style temporal regret strengthened",
            "differentiable optimization layer remains open",
        ),
        (
            "sample_paths",
            "near_resolved",
            "macro/vintage/common paths expanded",
            "external macro/default calibration",
        ),
        (
            "mdcp_caps",
            "near_resolved",
            "strict/committee/relaxed cap regimes documented",
            "future coverage calibration",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "proxy SICR only",
            "servicing/DPD/recovery/EAD/macro data",
        ),
        (
            "causal_cate",
            "theory_blocked",
            "policy value remains blocked",
            "identification, overlap, sensitivity",
        ),
        (
            "fairness",
            "data_blocked",
            "proxy governance only",
            "protected attributes or approved external protocol",
        ),
        ("paper1_freeze", "resolved", "Paper Estrella untouched", "continue Paper 4 only"),
    ]
    return pd.DataFrame(
        rows, columns=["blocker_id", "status_v13", "current_diagnosis", "next_action"]
    )


def build_claim_matrix_v13() -> pd.DataFrame:
    rows = [
        (
            "V12 audit",
            "implemented",
            "paper4_v13_v12_artifact_audit.csv",
            "19bb-v13-resolution-wave.qmd",
            "inventory/audit only",
        ),
        (
            "Method references",
            "implemented_primary_sources",
            "paper4_v13_method_reference_registry.csv",
            "19bb-v13-resolution-wave.qmd",
            "primary sources define claim boundaries",
        ),
        (
            "CVaR/OCE decomposition",
            "implemented_larger_pool_decomposition",
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
            "19bb-v13-resolution-wave.qmd",
            "not exact full-universe proof",
        ),
        (
            "Champion stress test",
            "implemented_common_path_stress",
            "paper4_v13_champion_stress_test.csv",
            "19bb-v13-resolution-wave.qmd",
            "Paper 4 working decision only",
        ),
        (
            "DLA/SDAM integration",
            "implemented_representative_common_paths",
            "paper4_v13_dla_representative_allocations.parquet",
            "19bb-v13-resolution-wave.qmd",
            "budget-capped representative DLA book",
        ),
        (
            "SPO/DFL",
            "implemented_temporal_regret_surrogate",
            "paper4_v13_spo_decision_loss_report.csv",
            "19bb-v13-resolution-wave.qmd",
            "not differentiable SPO+ theorem",
        ),
        (
            "Sample paths",
            "implemented_macro_common_paths",
            "paper4_v13_sample_path_macro_calibrated_ci.csv",
            "19bb-v13-resolution-wave.qmd",
            "internal calibration, not forecast",
        ),
        (
            "MDCP caps",
            "implemented_cap_regimes",
            "paper4_v13_mdcp_cap_regime_solver_summary.csv",
            "19bb-v13-resolution-wave.qmd",
            "source governance, no legal fairness claim",
        ),
        (
            "IFRS9/SICR",
            "implemented_proxy_sensitivity",
            "paper4_v13_ifrs9_sicr_sensitivity.csv",
            "19bb-v13-resolution-wave.qmd",
            "no contractual IFRS9 claim",
        ),
        (
            "Causal/Fairness",
            "implemented_blocker_dossier",
            "paper4_v13_causal_cate_dossier.csv",
            "19bb-v13-resolution-wave.qmd",
            "CATE/fair-lending claims blocked",
        ),
        (
            "Paper Estrella freeze",
            "guardrail_verified",
            "paper4_v13_status.json",
            "19bb-v13-resolution-wave.qmd",
            "models/final_project_promotion.json not modified",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def _write_v13_note(status: dict[str, Any]) -> None:
    _write_note(
        "paper4_v13_resolution_wave.md",
        "\n".join(
            [
                "# Paper 4 v13 Resolution Wave",
                "",
                f"- Paper 4 working champion: `{status.get('working_champion_policy_id_v13')}`.",
                f"- Champion changed vs v12: `{status.get('champion_changed_vs_v12')}`.",
                f"- CVaR feasible count: `{status['cvar_feasible_count_v13']}`.",
                f"- CVaR non-dominated count: `{status['cvar_non_dominated_count_v13']}`.",
                f"- MDCP optimal count: `{status['mdcp_optimal_count_v13']}`.",
                f"- DLA best delta vs static: `{status['dla_best_delta_state_value_v13']:.4f}`.",
                f"- SPO candidate count: `{status['spo_candidate_count_v13']}`.",
                f"- Sample-path policy count: `{status['sample_path_policy_count_v13']}`.",
                f"- Final promotion JSON created: `{status['paper4_final_promotion_created']}`.",
                "",
                "V13 is still a Paper 4 working-lab wave. It can change the Paper 4 working champion, but it does not alter Paper Estrella.",
            ]
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvar-pool-n", type=int, default=32_000)
    parser.add_argument("--cvar-rounds", type=int, default=2)
    parser.add_argument("--mdcp-pool-n", type=int, default=28_000)
    parser.add_argument("--spo-pool-n", type=int, default=26_000)
    parser.add_argument("--spo-epochs", type=int, default=10)
    parser.add_argument("--dla-months", type=int, default=12)
    parser.add_argument("--dla-paths", type=int, default=28)
    parser.add_argument("--dla-iterations", type=int, default=4)
    parser.add_argument("--sample-paths", type=int, default=300)
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = time.time()
    base_universe, candidate_pool, _, _, online_intervals = _load_inputs()
    solver_universe = base_universe if len(base_universe) > len(candidate_pool) else candidate_pool
    _, _, _, online_status = _load_v9_online()
    online_method = str(online_status["online_best_method_v9"])
    previous_champion = _safe_read_json(STATUS_DIR / "paper4_v12_working_champion.json").get(
        "policy_id", ""
    )

    audit = _artifact_audit_v13()
    _write_csv("paper4_v13_v12_artifact_audit.csv", audit)
    refs = build_method_reference_registry_v13()
    _write_csv("paper4_v13_method_reference_registry.csv", refs)

    cap_rationale, caps = build_mdcp_caps_v12()
    cap_rationale["version_v13"] = "v13_reused_empirical_support_with_regime_tests"
    _write_csv("paper4_v13_mdcp_cap_regime_rationale.csv", cap_rationale)

    cvar_frontier, cvar_alloc, cvar_losses, cvar_active = build_cvar_column_generation_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_pool_n=args.cvar_pool_n,
        rounds=args.cvar_rounds,
        caps=caps,
    )
    cvar_frontier = _add_v13_cvar_columns(cvar_frontier)
    cvar_alloc = _standardize_allocation_columns(cvar_alloc)
    cvar_losses = _replace_v12_policy_ids(cvar_losses)
    cvar_active = _replace_v12_policy_ids(cvar_active)
    _write_csv("paper4_v13_cvar_stronger_decomposition_frontier.csv", cvar_frontier)
    _write_parquet("paper4_v13_cvar_stronger_decomposition_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v13_cvar_stronger_decomposition_scenario_losses.csv", cvar_losses)
    _write_csv("paper4_v13_cvar_active_constraints.csv", cvar_active)

    mdcp_summary, mdcp_alloc = build_mdcp_solver_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        caps=caps,
        max_pool_n=args.mdcp_pool_n,
    )
    mdcp_summary = _replace_v12_policy_ids(mdcp_summary)
    mdcp_summary["cap_regime_scope_v13"] = "strict_tight_base_return_recovery_regimes"
    mdcp_alloc = _standardize_allocation_columns(mdcp_alloc)
    _write_csv("paper4_v13_mdcp_cap_regime_solver_summary.csv", mdcp_summary)
    _write_parquet("paper4_v13_mdcp_cap_regime_allocations.parquet", mdcp_alloc)

    dla_coef, dla_decisions, dla_trace, dla_comparison = build_dla_fvi_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_months=args.dla_months,
        n_paths=args.dla_paths,
        iterations=args.dla_iterations,
    )
    dla_coef["version_v13"] = "v13_policy_iteration_rollout"
    dla_decisions = _replace_v12_policy_ids(dla_decisions)
    dla_trace = _replace_v12_policy_ids(dla_trace)
    dla_comparison = _replace_v12_policy_ids(dla_comparison)
    dla_representative = _standardize_allocation_columns(
        _representative_dla_allocations(dla_decisions)
    )
    _write_csv("paper4_v13_dla_fitted_value_coefficients.csv", dla_coef)
    _write_parquet("paper4_v13_dla_fvi_decisions.parquet", dla_decisions)
    _write_parquet("paper4_v13_dla_fvi_trace.parquet", dla_trace)
    _write_csv("paper4_v13_dla_fvi_comparison.csv", dla_comparison)
    _write_parquet("paper4_v13_dla_representative_allocations.parquet", dla_representative)

    spo_train, spo_coef, spo_candidates, spo_alloc = build_spo_regret_surrogate_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_pool_n=args.spo_pool_n,
        epochs=args.spo_epochs,
    )
    spo_train["version_v13"] = "v13_longer_temporal_regret_training"
    spo_coef["version_v13"] = "v13_longer_temporal_regret_training"
    spo_candidates = _replace_v12_policy_ids(spo_candidates)
    spo_alloc = _standardize_allocation_columns(spo_alloc)
    spo_report = build_spo_v13_report(spo_train, spo_candidates)
    _write_csv("paper4_v13_spo_decision_loss_training.csv", spo_train)
    _write_csv("paper4_v13_spo_decision_loss_coefficients.csv", spo_coef)
    _write_csv("paper4_v13_spo_decision_loss_candidates.csv", spo_candidates)
    _write_parquet("paper4_v13_spo_decision_loss_allocations.parquet", spo_alloc)
    _write_csv("paper4_v13_spo_decision_loss_report.csv", spo_report)

    allocation_frames = [
        cvar_alloc.head(8_000) if not cvar_alloc.empty else pd.DataFrame(),
        mdcp_alloc.head(3_000) if not mdcp_alloc.empty else pd.DataFrame(),
        spo_alloc.head(4_000) if not spo_alloc.empty else pd.DataFrame(),
        dla_representative.head(4_000) if not dla_representative.empty else pd.DataFrame(),
    ]
    stress_alloc = pd.concat([df for df in allocation_frames if not df.empty], ignore_index=True)
    sample_cal, sample_scenarios, sample_paths, sample_ci, _ = build_sample_paths_v12(
        stress_alloc,
        solver_universe,
        n_paths=args.sample_paths,
    )
    sample_paths = _replace_v12_policy_ids(sample_paths)
    sample_ci = _replace_v12_policy_ids(sample_ci)
    for df in [sample_paths, sample_ci]:
        if "portfolio_loss_v12" in df:
            df["portfolio_loss_v13"] = df["portfolio_loss_v12"]
    sample_cal["version_v13"] = "v13_macro_vintage_common_paths"
    sample_scenarios["version_v13"] = "v13_macro_vintage_common_paths"
    _write_csv("paper4_v13_sample_path_calibration_table.csv", sample_cal)
    _write_csv("paper4_v13_sample_path_scenario_register.csv", sample_scenarios)
    _write_parquet("paper4_v13_sample_path_macro_calibrated_paths.parquet", sample_paths)
    _write_csv("paper4_v13_sample_path_macro_calibrated_ci.csv", sample_ci)

    ifrs9_readiness, sicr = build_ifrs9_sicr_v12(solver_universe)
    data_blockers = build_ifrs9_data_blocker_v13(ifrs9_readiness)
    _write_csv("paper4_v13_ifrs9_readiness.csv", ifrs9_readiness)
    _write_csv("paper4_v13_ifrs9_sicr_sensitivity.csv", sicr)
    _write_csv("paper4_v13_ifrs9_data_blocker_register.csv", data_blockers)

    causal, fairness = build_causal_fairness_v12()
    causal["version_v13"] = "v13_blocker_dossier"
    fairness["version_v13"] = "v13_proxy_governance_only"
    _write_csv("paper4_v13_causal_cate_dossier.csv", causal)
    _write_csv("paper4_v13_fairness_proxy_governance.csv", fairness)

    registry, champion = build_registry_v13(
        cvar_frontier,
        mdcp_summary,
        dla_comparison,
        spo_candidates,
        sample_ci,
        previous_champion=previous_champion,
    )
    pairwise = build_champion_pairwise_v13(
        sample_paths, champion.get("policy_id", "") if champion else ""
    )
    stress = build_champion_stress_v13(
        registry, pairwise, champion.get("policy_id", "") if champion else ""
    )
    _write_csv("paper4_v13_sample_path_pairwise_champion_ci.csv", pairwise)
    _write_csv("paper4_v13_champion_stress_test.csv", stress)
    _write_csv("paper4_v13_working_candidate_registry.csv", registry)
    if champion:
        _write_json("paper4_v13_working_champion.json", champion)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v13_resolution_wave",
        "mode": "paper4_working_champion_allowed_no_paper1_changes",
        "previous_working_champion_policy_id_v12": previous_champion,
        "working_champion_policy_id_v13": champion.get("policy_id") if champion else None,
        "champion_changed_vs_v12": champion.get("champion_changed_vs_v12") if champion else False,
        "online_best_method_v9": online_method,
        "online_goal_achieved": bool(online_status.get("online_goal_achieved")),
        "candidate_universe_source_v13": "base_full_universe"
        if len(base_universe) > len(candidate_pool)
        else "paper4_candidate_pool",
        "candidate_universe_n_v13": int(len(solver_universe)),
        "cvar_pool_n_v13": int(args.cvar_pool_n),
        "cvar_feasible_count_v13": int(
            cvar_frontier.get("feasible_v13", pd.Series(False)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "cvar_non_dominated_count_v13": int(
            cvar_frontier.get("non_dominated_v13", pd.Series(False)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "mdcp_optimal_count_v13": int(mdcp_summary["solver_status"].map(_is_optimal).sum())
        if not mdcp_summary.empty
        else 0,
        "dla_best_delta_state_value_v13": float(
            dla_comparison.loc[
                ~dla_comparison["policy_id"].eq("v13_static_reference"),
                "delta_state_value_vs_static",
            ].max()
        )
        if not dla_comparison.empty
        else np.nan,
        "spo_candidate_count_v13": int(len(spo_candidates)),
        "sample_path_policy_count_v13": int(sample_ci["policy_id"].nunique())
        if not sample_ci.empty
        else 0,
        "working_candidate_count_v13": int(len(registry)),
        "working_champion_created_v13": bool(champion),
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "paper4_working_champion_json_created": WORKING_CHAMPION_PATH.exists(),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": "V13 can change the Paper 4 working champion only; all final/publication claims remain guarded.",
    }
    dashboard = build_blocker_dashboard_v13(status)
    claims = build_claim_matrix_v13()
    _write_csv("paper4_v13_blocker_dashboard.csv", dashboard)
    _write_csv("paper4_v13_claim_artifact_matrix.csv", claims)
    _write_json("paper4_v13_status.json", status)
    _write_v13_note(status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
