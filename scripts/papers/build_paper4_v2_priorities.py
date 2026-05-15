"""Build Paper 4 v2 priority artifacts.

This generator implements the next ten Paper 4 priorities as a diagnostic v2
layer. It keeps the Paper Estrella champion protected and never writes
``paper4_final_promotion.json``.

The exact replay is intentionally named ``exact_limited``: HiGHS solves the LP
exactly inside a documented candidate subset for each top-k policy. It is not a
claim that every one of the 276,869 OOT loans was considered by the exact LP.
That full-universe run remains a later, heavier execution mode.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _load_base_loan_frame,
    _normalise,
    _policy_effective_pd,
    _safe_read_csv,
    _safe_read_parquet,
    _weighted_average,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD, load_policy_universe
from src.optimization.portfolio_model import optimize_portfolio_allocation

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
STATUS_DIR = OUT_ROOT / "status"
NOTE_DIR = OUT_ROOT / "notes"
SCHEMA_VERSION = "2026-05-12.3"
TOPK_N = 5
EXACT_LIMITED_POOL_N = 3_000


def _write_csv(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def _write_parquet(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_parquet(path, index=False)
    return path


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_note(name: str, text: str) -> Path:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTE_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def _topk_policy_ids() -> list[str]:
    selector = _safe_read_csv(TABLE_DIR / "paper4_table14_ifrs9_tail_satisficing_selector.csv")
    if selector.empty:
        return ["paper1_economic_champion"]
    top = selector.sort_values("diagnostic_selector_rank")["policy_id"].astype(str).head(TOPK_N)
    ids = ["paper1_economic_champion", *top.tolist()]
    return list(dict.fromkeys(ids))


def _canonical_champion_rows() -> pd.DataFrame:
    evidence = _safe_read_parquet(TABLE_DIR / "paper4_policy_loan_level_evidence.parquet")
    champion = evidence[evidence["policy_id"].eq("paper1_economic_champion")].copy()
    champion["exact_scope"] = "paper1_canonical_full_universe"
    champion["solver_status"] = "paper1_canonical_exact"
    champion["candidate_pool_n"] = np.nan
    champion["v2_source"] = "paper1_exact_champion"
    return champion


def _policy_score(base: pd.DataFrame, policy: pd.Series, effective_pd: np.ndarray) -> np.ndarray:
    rates = base["int_rate_decimal"].to_numpy(dtype=float)
    point = base["pd_point_alpha01"].to_numpy(dtype=float)
    high = base["pd_high_alpha01"].to_numpy(dtype=float)
    tau = float(policy["risk_tolerance"])
    return (
        rates
        - point * DEFAULT_LGD
        - float(policy.get("uncertainty_aversion", 0.0))
        * np.clip(high - point, 0.0, 1.0)
        * DEFAULT_LGD
        - 0.25 * np.maximum(effective_pd - tau, 0.0)
    )


def _solve_exact_limited_policy(
    base: pd.DataFrame,
    policy: pd.Series,
    proxy_evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy_id = str(policy["policy_id"])
    effective_pd = _policy_effective_pd(base, policy)
    score = _policy_score(base, policy, effective_pd)
    top_idx = np.argsort(-score)[:EXACT_LIMITED_POOL_N]

    proxy_ids = set(
        proxy_evidence.loc[proxy_evidence["policy_id"].eq(policy_id), "loan_id"].astype(str)
    )
    proxy_idx = base.index[base["id"].astype(str).isin(proxy_ids)].to_numpy(dtype=int)
    candidate_idx = np.array(sorted(set(top_idx.tolist()) | set(proxy_idx.tolist())), dtype=int)
    pool = base.iloc[candidate_idx].reset_index(drop=True)
    pool_effective_pd = effective_pd[candidate_idx]
    solution = optimize_portfolio_allocation(
        loans=pool,
        pd_point=pool["pd_point_alpha01"].to_numpy(dtype=float),
        pd_low=pool["pd_low_alpha01"].to_numpy(dtype=float),
        pd_high=pool["pd_high_alpha01"].to_numpy(dtype=float),
        lgd=np.full(len(pool), DEFAULT_LGD, dtype=float),
        int_rates=pool["int_rate_decimal"].to_numpy(dtype=float),
        total_budget=BUDGET,
        max_concentration=0.25,
        max_portfolio_pd=float(policy["risk_tolerance"]),
        robust=True,
        uncertainty_aversion=float(policy.get("uncertainty_aversion", 0.0)),
        min_budget_utilization=float(policy.get("min_budget_utilization", 0.0)),
        pd_cap_slack_penalty=float(policy.get("pd_cap_slack_penalty", 0.0)),
        pd_constraint_override=pool_effective_pd,
        time_limit=120,
        threads=4,
        solver_backend="highs",
    )
    allocation = np.array(
        [float(solution["allocation"].get(i, 0.0)) for i in range(len(pool))],
        dtype=float,
    )
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["policy_id"] = policy_id
    funded["loan_id"] = funded["id"].astype(str)
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["portfolio_weight"] = funded["funded_exposure"] / max(
        float(funded["funded_exposure"].sum()), 1e-12
    )
    funded["pd_point"] = funded["pd_point_alpha01"]
    funded["pd_high_alpha01"] = funded["pd_high_alpha01"]
    funded["effective_pd_alpha01"] = pool_effective_pd[mask]
    funded["miscovered_alpha01"] = funded["y_true"].gt(funded["pd_high_alpha01"])
    funded["funded_flag"] = True
    funded["is_champion_exact"] = False
    funded["exact_scope"] = "limited_candidate_subset"
    funded["solver_status"] = str(solution.get("solver_status", "unknown"))
    funded["candidate_pool_n"] = int(len(pool))
    funded["v2_source"] = "highs_exact_limited_topk"
    funded["reconstruction_method"] = "highs_exact_limited_topk_candidate_subset"
    funded["realized_return_proxy_lgd45"] = (
        funded["funded_exposure"] * funded["int_rate_decimal"] * (1.0 - funded["y_true"])
        - funded["funded_exposure"] * DEFAULT_LGD * funded["y_true"]
    )
    funded["ecl_baseline_lgd45"] = (
        funded["pd_high_alpha01"] * DEFAULT_LGD * funded["funded_exposure"]
    )
    keep = [
        "policy_id",
        "loan_id",
        "issue_d",
        "issue_month",
        "period",
        "original_grade",
        "sub_grade",
        "term",
        "loan_amnt",
        "int_rate",
        "int_rate_decimal",
        "y_true",
        "allocation_fraction",
        "funded_exposure",
        "portfolio_weight",
        "pd_point",
        "pd_high_alpha01",
        "effective_pd_alpha01",
        "miscovered_alpha01",
        "funded_flag",
        "reconstruction_method",
        "is_champion_exact",
        "exact_scope",
        "solver_status",
        "candidate_pool_n",
        "v2_source",
        "realized_return_proxy_lgd45",
        "ecl_baseline_lgd45",
    ]
    metrics = {
        "policy_id": policy_id,
        "candidate_pool_n": int(len(pool)),
        "solver_status": str(solution.get("solver_status", "unknown")),
        "objective_value": float(solution.get("objective_value", np.nan)),
        "n_funded_exact_limited": int(mask.sum()),
        "total_allocated_exact_limited": float(funded["funded_exposure"].sum()),
        "realized_return_exact_limited": float(funded["realized_return_proxy_lgd45"].sum()),
        "weighted_pd_high_exact_limited": _weighted_average(
            funded["pd_high_alpha01"], funded["funded_exposure"]
        ),
        "coverage_alpha01_exact_limited": float(1.0 - funded["miscovered_alpha01"].mean()),
    }
    return funded[keep], metrics


def build_exact_limited_topk(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    proxy = _safe_read_parquet(TABLE_DIR / "paper4_policy_loan_level_evidence.parquet")
    top_ids = _topk_policy_ids()
    rows = [_canonical_champion_rows()]
    metrics: list[dict[str, Any]] = []
    for policy_id in top_ids:
        if policy_id == "paper1_economic_champion":
            champion = rows[0]
            metrics.append(
                {
                    "policy_id": policy_id,
                    "candidate_pool_n": np.nan,
                    "solver_status": "paper1_canonical_exact",
                    "objective_value": np.nan,
                    "n_funded_exact_limited": int(champion["loan_id"].nunique()),
                    "total_allocated_exact_limited": float(champion["funded_exposure"].sum()),
                    "realized_return_exact_limited": float(
                        champion["realized_return_proxy_lgd45"].sum()
                    ),
                    "weighted_pd_high_exact_limited": _weighted_average(
                        champion["pd_high_alpha01"], champion["funded_exposure"]
                    ),
                    "coverage_alpha01_exact_limited": float(
                        1.0 - champion["miscovered_alpha01"].astype(float).mean()
                    ),
                }
            )
            continue
        policy = policy_universe[policy_universe["policy_id"].eq(policy_id)].iloc[0]
        funded, local_metrics = _solve_exact_limited_policy(base, policy, proxy)
        rows.append(funded)
        metrics.append(local_metrics)
    allocations = pd.concat(rows, ignore_index=True)
    eval_table = pd.DataFrame(metrics)

    proxy_eval = (
        proxy[proxy["policy_id"].isin(top_ids)]
        .groupby("policy_id", as_index=False)
        .agg(
            n_funded_proxy=("loan_id", "nunique"),
            total_allocated_proxy=("funded_exposure", "sum"),
            realized_return_proxy=("realized_return_proxy_lgd45", "sum"),
            ecl_proxy=("ecl_baseline_lgd45", "sum"),
        )
    )
    exact_eval = eval_table.merge(proxy_eval, on="policy_id", how="left")
    exact_eval["return_diff_exact_limited_vs_proxy"] = (
        exact_eval["realized_return_exact_limited"] - exact_eval["realized_return_proxy"]
    )
    exact_eval["allocation_diff_exact_limited_vs_proxy"] = (
        exact_eval["total_allocated_exact_limited"] - exact_eval["total_allocated_proxy"]
    )

    overlap_rows = []
    for policy_id in top_ids:
        exact_ids = set(
            allocations.loc[allocations["policy_id"].eq(policy_id), "loan_id"].astype(str)
        )
        proxy_ids = set(proxy.loc[proxy["policy_id"].eq(policy_id), "loan_id"].astype(str))
        union = exact_ids | proxy_ids
        overlap_rows.append(
            {
                "policy_id": policy_id,
                "exact_limited_n": len(exact_ids),
                "proxy_n": len(proxy_ids),
                "overlap_n": len(exact_ids & proxy_ids),
                "jaccard_exact_limited_vs_proxy": len(exact_ids & proxy_ids) / max(len(union), 1),
                "status": "exact_limited_candidate_subset_comparison",
            }
        )
    comparison = pd.DataFrame(overlap_rows).merge(exact_eval, on="policy_id", how="left")
    allocations["loan_id"] = allocations["loan_id"].astype(str)
    allocations["policy_id"] = allocations["policy_id"].astype(str)
    allocations["issue_d"] = pd.to_datetime(allocations["issue_d"], errors="coerce").astype(str)
    allocations["issue_month"] = pd.to_datetime(allocations["issue_month"], errors="coerce")
    for column in [
        "period",
        "original_grade",
        "sub_grade",
        "term",
        "reconstruction_method",
        "exact_scope",
        "solver_status",
        "v2_source",
    ]:
        allocations[column] = allocations[column].astype(str)
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": "exact_limited_topk_no_promotion",
        "topk_n": TOPK_N,
        "candidate_pool_n_target": EXACT_LIMITED_POOL_N,
        "policy_ids": top_ids,
        "paper1_champion_protected": True,
        "promotion_json_created": False,
        "caveat": (
            "HiGHS solves each top-k policy exactly inside a policy-specific candidate subset. "
            "This is not a full 276k-loan exact rerun."
        ),
    }
    return allocations, eval_table, comparison, status


def build_v2_policy_loan_evidence(exact_allocations: pd.DataFrame) -> pd.DataFrame:
    proxy = _safe_read_parquet(TABLE_DIR / "paper4_policy_loan_level_evidence.parquet")
    exact_ids = set(exact_allocations["policy_id"].astype(str))
    remaining = proxy[~proxy["policy_id"].astype(str).isin(exact_ids)].copy()
    remaining["exact_scope"] = "proxy_from_p0_greedy_reconstruction"
    remaining["solver_status"] = "not_solved_in_v2"
    remaining["candidate_pool_n"] = np.nan
    remaining["v2_source"] = "p0_proxy_fallback"
    evidence = pd.concat([exact_allocations, remaining], ignore_index=True, sort=False)
    evidence["loan_id"] = evidence["loan_id"].astype(str)
    evidence["policy_id"] = evidence["policy_id"].astype(str)
    return evidence


def build_ifrs9_v2(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [
        {"scenario": "baseline", "pd_multiplier": 1.00, "lgd": 0.45},
        {"scenario": "adverse", "pd_multiplier": 1.25, "lgd": 0.55},
        {"scenario": "severe", "pd_multiplier": 1.60, "lgd": 0.65},
    ]
    rules = [
        {"sicr_rule": "absolute_pd_30", "relative": np.inf, "absolute": 0.30},
        {"sicr_rule": "relative_pd_1p5", "relative": 1.50, "absolute": np.inf},
        {"sicr_rule": "hybrid_pd_grade", "relative": 1.50, "absolute": 0.30},
    ]
    rows = []
    for scenario in scenarios:
        for rule in rules:
            work = evidence.copy()
            work["scenario"] = scenario["scenario"]
            work["sicr_rule"] = rule["sicr_rule"]
            work["lgd_scenario"] = float(scenario["lgd"])
            work["scenario_pd"] = np.clip(
                pd.to_numeric(work["pd_high_alpha01"], errors="coerce").fillna(0.0)
                * float(scenario["pd_multiplier"]),
                0.0,
                1.0,
            )
            rel = work["scenario_pd"] / np.maximum(
                pd.to_numeric(work["pd_point"], errors="coerce").fillna(0.0), 1e-6
            )
            grade_stress = work["original_grade"].astype(str).isin(["E", "F", "G"]) & work[
                "scenario"
            ].isin(["adverse", "severe"])
            sicr = work["scenario_pd"].ge(float(rule["absolute"])) | rel.ge(float(rule["relative"]))
            if rule["sicr_rule"] == "hybrid_pd_grade":
                sicr = sicr | grade_stress
            work["ifrs9_stage_v2"] = np.select(
                [work["y_true"].eq(1.0), sicr],
                ["Stage 3 observed", "Stage 2 lifetime proxy"],
                default="Stage 1 12m proxy",
            )
            lifetime_factor = np.where(
                work["ifrs9_stage_v2"].eq("Stage 2 lifetime proxy"), 3.0, 1.0
            )
            work["ecl_v2"] = np.where(
                work["ifrs9_stage_v2"].eq("Stage 3 observed"),
                work["lgd_scenario"] * work["funded_exposure"],
                np.minimum(work["scenario_pd"] * lifetime_factor, 1.0)
                * work["lgd_scenario"]
                * work["funded_exposure"],
            )
            work["provision_v2"] = work["ecl_v2"]
            rows.append(work)
    grid = pd.concat(rows, ignore_index=True)
    summary = (
        grid.groupby(["policy_id", "scenario", "sicr_rule"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
            ecl_v2=("ecl_v2", "sum"),
            provision_v2=("provision_v2", "sum"),
            weighted_pd_scenario=(
                "scenario_pd",
                lambda x: _weighted_average(x, grid.loc[x.index, "funded_exposure"]),
            ),
            stage2_lifetime_share=(
                "ifrs9_stage_v2",
                lambda x: float(x.eq("Stage 2 lifetime proxy").mean()),
            ),
            stage3_observed_share=(
                "ifrs9_stage_v2",
                lambda x: float(x.eq("Stage 3 observed").mean()),
            ),
        )
        .sort_values(["sicr_rule", "scenario", "ecl_v2"])
    )
    summary["net_return_after_ecl_v2"] = (
        summary["realized_return_proxy_lgd45"] - summary["provision_v2"]
    )
    sicr_compare = (
        summary[summary["scenario"].eq("baseline")]
        .pivot_table(
            index="policy_id",
            columns="sicr_rule",
            values=["ecl_v2", "stage2_lifetime_share", "net_return_after_ecl_v2"],
            aggfunc="first",
        )
        .reset_index()
    )
    sicr_compare.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in sicr_compare.columns
    ]
    stress = summary.pivot_table(
        index=["policy_id", "sicr_rule"],
        columns="scenario",
        values=["ecl_v2", "net_return_after_ecl_v2"],
        aggfunc="first",
    ).reset_index()
    stress.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in stress.columns
    ]
    if "ecl_v2_severe" in stress and "ecl_v2_baseline" in stress:
        stress["severe_minus_baseline_ecl"] = stress["ecl_v2_severe"] - stress["ecl_v2_baseline"]
    if not sicr_compare.empty:
        stress = stress.merge(sicr_compare, on="policy_id", how="left")
    return grid, summary, stress.sort_values(["policy_id", "sicr_rule"]).reset_index(drop=True)


def build_selector_v2(
    policy_universe: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline = ifrs9_summary[
        ifrs9_summary["scenario"].eq("baseline") & ifrs9_summary["sicr_rule"].eq("hybrid_pd_grade")
    ].copy()
    selector = policy_universe.merge(baseline, on="policy_id", how="left")
    tail = build_tail_risk_metrics_v2(evidence)
    mdcp = _safe_read_csv(TABLE_DIR / "paper4_mdcp_worst_source_coverage.csv")
    fairness = _safe_read_csv(TABLE_DIR / "paper4_fairness_constraint_screen.csv")
    sat = _safe_read_csv(TABLE_DIR / "paper4_robust_satisficing_policy_eval.csv")
    if not mdcp.empty:
        mdcp_agg = mdcp.groupby("policy_id", as_index=False).agg(
            worst_source_coverage_alpha01=("worst_source_coverage_alpha01", "min")
        )
        selector = selector.merge(mdcp_agg, on="policy_id", how="left")
    if not fairness.empty:
        selector = selector.merge(
            fairness[["policy_id", "max_abs_representation_gap", "fairness_proxy_pass"]],
            on="policy_id",
            how="left",
        )
    if not sat.empty:
        selector = selector.merge(
            sat[["policy_id", "robust_satisficing_decision"]],
            on="policy_id",
            how="left",
        )
    selector = selector.merge(tail, on="policy_id", how="left")
    selector["score_net_ecl_v2"] = _normalise(
        selector["net_return_after_ecl_v2"], higher_is_better=True
    )
    selector["score_ecl_v2"] = _normalise(selector["ecl_v2"], higher_is_better=False)
    selector["score_tail_v2"] = _normalise(selector["cvar_95_loss_rate_v2"], higher_is_better=False)
    selector["score_mdcp"] = selector["worst_source_coverage_alpha01"].fillna(0.0).clip(0, 1)
    selector["score_fairness"] = _normalise(
        selector["max_abs_representation_gap"].fillna(1.0), higher_is_better=False
    )
    selector["score_auditability"] = 0.5 * _normalise(
        selector["gamma_cp"], higher_is_better=False
    ) + 0.5 * _normalise(selector["weighted_miscoverage_V"], higher_is_better=False)
    weights = {
        "score_net_ecl_v2": 0.35,
        "score_ecl_v2": 0.15,
        "score_tail_v2": 0.15,
        "score_mdcp": 0.15,
        "score_fairness": 0.10,
        "score_auditability": 0.10,
    }
    selector["selector_v2_score"] = sum(selector[k] * v for k, v in weights.items())
    selector["selector_v2_gate_pass"] = (
        selector["all_bounds_hold"].astype(bool)
        & selector["ab_pass_all"].astype(bool)
        & selector["worst_source_coverage_alpha01"].fillna(0.0).ge(0.80)
        & selector["max_abs_representation_gap"].fillna(1.0).le(0.20)
    )
    selector["selector_v2_rank"] = (
        selector["selector_v2_score"].rank(ascending=False, method="first").astype(int)
    )
    selector["selector_v2_decision"] = np.select(
        [
            selector["policy_id"].eq("paper1_economic_champion"),
            selector["selector_v2_gate_pass"] & selector["selector_v2_rank"].le(5),
            selector["selector_v2_gate_pass"],
        ],
        ["protected_paper1_champion", "exact_rerun_candidate", "keep"],
        default="park",
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selector_id": "paper4_selector_v2_ifrs9_tail_mdcp_fairness",
        "mode": "diagnostic_no_promotion",
        "weights": weights,
        "gates": [
            "all_bounds_hold",
            "ab_pass_all",
            "worst_source_coverage_alpha01 >= 0.80",
            "max_abs_representation_gap <= 0.20",
        ],
        "promotion_json_created": False,
        "paper1_champion_protected": True,
    }
    return config, selector.sort_values("selector_v2_rank").reset_index(drop=True)


def build_tail_risk_metrics_v2(evidence: pd.DataFrame) -> pd.DataFrame:
    work = evidence.copy()
    work["loss_rate"] = DEFAULT_LGD * work["y_true"]
    rows = []
    for policy_id, group in work.groupby("policy_id"):
        weights = group["funded_exposure"]
        loss = pd.to_numeric(group["loss_rate"], errors="coerce").fillna(0.0)
        q95 = float(loss.quantile(0.95))
        tail = loss[loss >= q95]
        theta = 5.0
        rows.append(
            {
                "policy_id": policy_id,
                "mean_loss_rate_v2": _weighted_average(loss, weights),
                "cvar_95_loss_rate_v2": float(tail.mean()) if len(tail) else 0.0,
                "entropic_oce_theta5_v2": float(np.log(np.mean(np.exp(theta * loss))) / theta),
            }
        )
    return pd.DataFrame(rows)


def build_online_conformal_real(
    base: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = base[["id", "issue_month", "period", "original_grade", "y_true", "y_pred"]].copy()
    work["loan_id"] = work["id"].astype(str)
    work["score_abs"] = (work["y_true"] - work["y_pred"]).abs()
    months = sorted(work["issue_month"].dropna().unique())
    frames = []
    for month in months:
        current = work[work["issue_month"].eq(month)].copy()
        prior = work[work["issue_month"].lt(month)]
        if prior.empty:
            current["qhat_90"] = np.nan
            current["qhat_95"] = np.nan
            current["online_source"] = "first_month_offline_fallback"
            current["pd_low_online_90"] = np.clip(current["y_pred"] - 0.50, 0.0, 1.0)
            current["pd_high_online_90"] = np.clip(current["y_pred"] + 0.50, 0.0, 1.0)
            current["pd_low_online_95"] = np.clip(current["y_pred"] - 0.60, 0.0, 1.0)
            current["pd_high_online_95"] = np.clip(current["y_pred"] + 0.60, 0.0, 1.0)
            frames.append(current)
            continue
        global_q90 = float(prior["score_abs"].quantile(0.90))
        global_q95 = float(prior["score_abs"].quantile(0.95))
        grade_stats = (
            prior.assign(original_grade=lambda df: df["original_grade"].astype(str))
            .groupby("original_grade", as_index=False)
            .agg(
                grade_n=("score_abs", "size"),
                grade_q90=("score_abs", lambda x: float(x.quantile(0.90))),
                grade_q95=("score_abs", lambda x: float(x.quantile(0.95))),
            )
        )
        current["original_grade"] = current["original_grade"].astype(str)
        current = current.merge(grade_stats, on="original_grade", how="left")
        use_grade = current["grade_n"].fillna(0).ge(100)
        current["qhat_90"] = np.where(use_grade, current["grade_q90"], global_q90)
        current["qhat_95"] = np.where(use_grade, current["grade_q95"], global_q95)
        current["online_source"] = np.where(use_grade, "rolling_grade", "rolling_global")
        current["pd_low_online_90"] = np.clip(current["y_pred"] - current["qhat_90"], 0.0, 1.0)
        current["pd_high_online_90"] = np.clip(current["y_pred"] + current["qhat_90"], 0.0, 1.0)
        current["pd_low_online_95"] = np.clip(current["y_pred"] - current["qhat_95"], 0.0, 1.0)
        current["pd_high_online_95"] = np.clip(current["y_pred"] + current["qhat_95"], 0.0, 1.0)
        frames.append(current)
    intervals = pd.concat(frames, ignore_index=True)
    intervals["covered_online_90"] = intervals["y_true"].between(
        intervals["pd_low_online_90"], intervals["pd_high_online_90"]
    )
    intervals["covered_online_95"] = intervals["y_true"].between(
        intervals["pd_low_online_95"], intervals["pd_high_online_95"]
    )
    replay = evidence.merge(
        intervals[
            [
                "loan_id",
                "pd_low_online_90",
                "pd_high_online_90",
                "pd_low_online_95",
                "pd_high_online_95",
                "covered_online_90",
                "covered_online_95",
                "online_source",
            ]
        ],
        on="loan_id",
        how="left",
    )
    policy_replay = (
        replay.groupby(["policy_id", "issue_month", "period"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            coverage_online_90=("covered_online_90", "mean"),
            coverage_online_95=("covered_online_95", "mean"),
            funded_exposure=("funded_exposure", "sum"),
            online_sources=("online_source", lambda x: ",".join(sorted(set(map(str, x.dropna()))))),
        )
        .rename(columns={"issue_month": "month"})
    )
    policy_replay["coverage_regret_90"] = (0.90 - policy_replay["coverage_online_90"]).clip(lower=0)
    policy_replay["coverage_regret_95"] = (0.95 - policy_replay["coverage_online_95"]).clip(lower=0)
    regret = policy_replay.sort_values(["policy_id", "month"]).assign(
        coverage_regret_90_cum=lambda df: df.groupby("policy_id")["coverage_regret_90"].cumsum(),
        coverage_regret_95_cum=lambda df: df.groupby("policy_id")["coverage_regret_95"].cumsum(),
    )
    return intervals, policy_replay, regret


def build_regret_auditability_v2(
    selector_v2: pd.DataFrame,
    online_replay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = _safe_read_parquet(TABLE_DIR / "paper4_monthly_policy_replay.parquet")
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work = monthly.merge(
        online_replay[["policy_id", "month", "coverage_online_90"]],
        on=["policy_id", "month"],
        how="left",
    )
    oracle = work.groupby("month", as_index=False).agg(
        oracle_net_return=("net_return_after_ecl", "max")
    )
    work = work.merge(oracle, on="month", how="left")
    work["decision_regret_net_ecl"] = work["oracle_net_return"] - work["net_return_after_ecl"]
    work = work.merge(
        selector_v2[["policy_id", "gamma_cp", "weighted_miscoverage_V", "selector_v2_score"]],
        on="policy_id",
        how="left",
    )
    work["auditability_score_v2"] = (
        0.40 * (1.0 - work["weighted_miscoverage_V"].fillna(0.10).clip(0, 0.10) / 0.10)
        + 0.30 * (1.0 - work["gamma_cp"].fillna(0.30).clip(0, 0.30) / 0.30)
        + 0.20 * work["coverage_online_90"].fillna(0.0).clip(0, 1)
        + 0.10
    ).clip(0, 1)
    replay = work[
        [
            "policy_id",
            "month",
            "period",
            "net_return_after_ecl",
            "oracle_net_return",
            "decision_regret_net_ecl",
            "coverage_online_90",
            "auditability_score_v2",
            "selector_v2_score",
        ]
    ].copy()
    pareto = (
        replay.groupby("policy_id", as_index=False)
        .agg(
            mean_regret_net_ecl=("decision_regret_net_ecl", "mean"),
            total_regret_net_ecl=("decision_regret_net_ecl", "sum"),
            mean_coverage_online_90=("coverage_online_90", "mean"),
            auditability_score_v2=("auditability_score_v2", "mean"),
        )
        .sort_values(["mean_regret_net_ecl", "auditability_score_v2"], ascending=[True, False])
    )
    hybrid = pareto.head(10).copy()
    hybrid["hybrid_id"] = "spo_plus_crpto_conformal_hybrid_candidate"
    hybrid["hybrid_score"] = 0.5 * _normalise(
        hybrid["mean_regret_net_ecl"], higher_is_better=False
    ) + 0.5 * _normalise(hybrid["auditability_score_v2"], higher_is_better=True)
    hybrid["status"] = "diagnostic_hybrid_candidate_not_solver"
    return replay, pareto, hybrid.sort_values("hybrid_score", ascending=False)


def build_tail_solver_prototype(
    evidence: pd.DataFrame, selector_v2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    tail = build_tail_risk_metrics_v2(evidence)
    grid_rows = []
    for cvar_cap in [0.20, 0.30, 0.45]:
        for penalty in [0.0, 0.25, 0.50]:
            work = selector_v2.merge(tail, on="policy_id", how="left", suffixes=("", "_tail"))
            work["tail_gate_pass"] = work["cvar_95_loss_rate_v2"].le(cvar_cap)
            work["tail_solver_score"] = _normalise(
                work["net_return_after_ecl_v2"], higher_is_better=True
            ) - penalty * _normalise(work["cvar_95_loss_rate_v2"], higher_is_better=True)
            selected = (
                work[work["tail_gate_pass"]]
                .sort_values("tail_solver_score", ascending=False)
                .head(1)
            )
            if selected.empty:
                continue
            row = selected.iloc[0]
            grid_rows.append(
                {
                    "cvar_cap": cvar_cap,
                    "oce_penalty": penalty,
                    "selected_policy_id": row["policy_id"],
                    "selected_tail_solver_score": float(row["tail_solver_score"]),
                    "selected_net_return_after_ecl_v2": float(row["net_return_after_ecl_v2"]),
                    "selected_cvar_95_loss_rate_v2": float(row["cvar_95_loss_rate_v2"]),
                    "status": "tail_constraint_selector_prototype",
                }
            )
    grid = pd.DataFrame(grid_rows)
    selector = selector_v2.merge(tail, on="policy_id", how="left", suffixes=("", "_tail"))
    selector["tail_selector_rank"] = (
        (
            _normalise(selector["net_return_after_ecl_v2"], higher_is_better=True)
            + _normalise(selector["cvar_95_loss_rate_v2"], higher_is_better=False)
        )
        .rank(ascending=False, method="first")
        .astype(int)
    )

    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(selector["cvar_95_loss_rate_v2"], selector["net_return_after_ecl_v2"], s=70)
        ax.set_xlabel("CVaR95 loss rate")
        ax.set_ylabel("Net return after ECL v2")
        ax.set_title("Paper 4 tail-risk selector prototype")
        ax.grid(True, alpha=0.25)
        path = FIGURE_DIR / "paper4_fig7_tail_policy_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    except Exception:
        pass
    return selector.sort_values("tail_selector_rank"), grid, paths


def build_multi_period_solver_v2(
    selector_v2: pd.DataFrame,
    online_replay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = _safe_read_parquet(TABLE_DIR / "paper4_monthly_policy_replay.parquet")
    candidates = (
        selector_v2[selector_v2["selector_v2_rank"].le(8)]["policy_id"].astype(str).tolist()
    )
    work = monthly[monthly["policy_id"].astype(str).isin(candidates)].merge(
        online_replay[["policy_id", "month", "coverage_online_90"]],
        on=["policy_id", "month"],
        how="left",
    )
    rows = []
    budget = BUDGET
    for month, group in work.sort_values("month").groupby("month"):
        feasible = group[group["coverage_online_90"].fillna(0.0).ge(0.80)].copy()
        if feasible.empty:
            feasible = group.copy()
        chosen = feasible.sort_values("net_return_after_ecl", ascending=False).iloc[0]
        budget_end = max(
            0.0,
            budget + float(chosen["realized_return_proxy_lgd45"]) - float(chosen["provision"]),
        )
        rows.append(
            {
                "month": month,
                "period": chosen["period"],
                "chosen_policy_id": chosen["policy_id"],
                "budget_start": budget,
                "funded_exposure": float(chosen["funded_exposure"]),
                "realized_return_proxy_lgd45": float(chosen["realized_return_proxy_lgd45"]),
                "provision": float(chosen["provision"]),
                "net_return_after_ecl": float(chosen["net_return_after_ecl"]),
                "coverage_online_90": float(chosen.get("coverage_online_90", np.nan)),
                "budget_end": budget_end,
                "status": "rolling_monthly_policy_choice_toy_solver",
            }
        )
        budget = budget_end
    path = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "strategy": "dynamic_monthly_top8",
                "months": int(path["month"].nunique()),
                "total_net_return_after_ecl": float(path["net_return_after_ecl"].sum()),
                "ending_budget": float(path["budget_end"].iloc[-1]) if not path.empty else np.nan,
                "mean_coverage_online_90": float(path["coverage_online_90"].mean()),
            }
        ]
    )
    champion = monthly[monthly["policy_id"].eq("paper1_economic_champion")]
    if not champion.empty:
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "strategy": "static_paper1_champion",
                            "months": int(champion["month"].nunique()),
                            "total_net_return_after_ecl": float(
                                champion["net_return_after_ecl"].sum()
                            ),
                            "ending_budget": np.nan,
                            "mean_coverage_online_90": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    comparison = summary.copy()
    if len(comparison) >= 2:
        baseline = float(
            comparison.loc[
                comparison["strategy"].eq("static_paper1_champion"),
                "total_net_return_after_ecl",
            ].iloc[0]
        )
        comparison["delta_vs_static_champion"] = comparison["total_net_return_after_ecl"] - baseline
    return path, summary, comparison


def build_mdcp_formal_v2(
    evidence: pd.DataFrame, online_intervals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched = evidence.merge(
        online_intervals[["loan_id", "covered_online_90", "y_pred"]],
        on="loan_id",
        how="left",
    )
    enriched["score_decile"] = pd.qcut(
        enriched["y_pred"].rank(method="first"), 10, labels=False, duplicates="drop"
    )
    definitions = pd.DataFrame(
        [
            {"source_id": "grade", "columns": "original_grade", "status": "implemented"},
            {"source_id": "period", "columns": "period", "status": "implemented"},
            {
                "source_id": "grade_period",
                "columns": "original_grade|period",
                "status": "implemented",
            },
            {"source_id": "score_decile", "columns": "score_decile", "status": "implemented"},
        ]
    )
    rows = []
    for source_id, cols in {
        "grade": ["original_grade"],
        "period": ["period"],
        "grade_period": ["original_grade", "period"],
        "score_decile": ["score_decile"],
    }.items():
        local = (
            enriched.groupby(["policy_id", *cols], as_index=False)
            .agg(
                n=("loan_id", "nunique"),
                funded_exposure=("funded_exposure", "sum"),
                coverage_online_90=("covered_online_90", "mean"),
            )
            .copy()
        )
        local["source_id"] = source_id
        local["source_value"] = local[cols].astype(str).agg("|".join, axis=1)
        rows.append(
            local[
                [
                    "policy_id",
                    "source_id",
                    "source_value",
                    "n",
                    "funded_exposure",
                    "coverage_online_90",
                ]
            ]
        )
    eval_table = pd.concat(rows, ignore_index=True)
    frontier = eval_table.groupby(["policy_id", "source_id"], as_index=False).agg(
        worst_source_coverage_online_90=("coverage_online_90", "min"),
        sources=("source_value", "nunique"),
    )
    frontier["mdcp_v2_pass_80"] = frontier["worst_source_coverage_online_90"].ge(0.80)
    return definitions, eval_table, frontier


def build_cate_v2(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cate = _safe_read_parquet(ROOT / "data" / "processed" / "cate_estimates_oot.parquet")
    if cate.empty:
        return pd.DataFrame(), pd.DataFrame()
    top_ids = set(_topk_policy_ids())
    cate = cate.copy()
    cate["id"] = cate["id"].astype(str)
    work = evidence[evidence["policy_id"].isin(top_ids)].merge(
        cate[["id", "cate", "cate_lb", "cate_ub", "annual_inc", "dti"]].rename(
            columns={"id": "loan_id"}
        ),
        on="loan_id",
        how="left",
    )
    overlap = work.groupby("policy_id", as_index=False).agg(
        n_funded=("loan_id", "nunique"),
        mean_annual_inc=("annual_inc", "mean"),
        p05_annual_inc=("annual_inc", lambda x: float(np.nanquantile(x, 0.05))),
        p95_annual_inc=("annual_inc", lambda x: float(np.nanquantile(x, 0.95))),
        mean_dti=("dti", "mean"),
        p05_dti=("dti", lambda x: float(np.nanquantile(x, 0.05))),
        p95_dti=("dti", lambda x: float(np.nanquantile(x, 0.95))),
        cate_ci_crosses_zero_share=(
            "cate_lb",
            lambda x: float(((x <= 0) & (work.loc[x.index, "cate_ub"] >= 0)).mean()),
        ),
    )
    work["cate_loss_reduction_value"] = (
        -work["cate"].clip(upper=0.0) * DEFAULT_LGD * work["funded_exposure"]
    )
    value = (
        work.groupby("policy_id", as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            mean_cate=("cate", "mean"),
            share_negative_cate=("cate", lambda x: float((x < 0).mean())),
            cate_loss_reduction_value=("cate_loss_reduction_value", "sum"),
        )
        .sort_values("cate_loss_reduction_value", ascending=False)
    )
    return overlap, value


def build_artifact_catalog(generated: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_files = sorted(OUT_ROOT.glob("**/*"))
    rows = []
    for path in all_files:
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "artifact": rel,
                "suffix": path.suffix,
                "bytes": path.stat().st_size,
                "paper4_layer": "v2_priorities" if path in generated else "foundation_or_prior",
                "exists": True,
            }
        )
    catalog = pd.DataFrame(rows)
    claims = pd.DataFrame(
        [
            {
                "priority": "P1 exact replay top-k",
                "claim_status": "exact_limited_diagnostic",
                "artifact": "paper4_exact_limited_topk_allocations.parquet",
                "quarto_page": "19p-exact-replay-topk.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Exact inside candidate subset, not full 276k universe.",
            },
            {
                "priority": "P2 IFRS9 v2",
                "claim_status": "diagnostic_lifetime_proxy",
                "artifact": "paper4_ifrs9_lifetime_ecl_grid.parquet",
                "quarto_page": "19q-ifrs9-v2-lifetime-ecl.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Lifetime factor and SICR rules are proxies.",
            },
            {
                "priority": "P3 selector v2",
                "claim_status": "diagnostic_no_promotion",
                "artifact": "paper4_selector_v2_results.csv",
                "quarto_page": "19q-ifrs9-v2-lifetime-ecl.qmd",
                "test_guardrail": "test_paper4_v2_no_promotion",
                "caveat": "Ranks candidates only.",
            },
            {
                "priority": "P4 online conformal real",
                "claim_status": "forward_only_interval_prototype",
                "artifact": "paper4_online_conformal_intervals.parquet",
                "quarto_page": "19r-online-conformal-real.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Rolling score intervals, not final online CP method.",
            },
            {
                "priority": "P5 regret-auditability replay",
                "claim_status": "diagnostic_replay",
                "artifact": "paper4_regret_auditability_replay_v2.parquet",
                "quarto_page": "19s-regret-auditability-replay.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Uses policy replay artifacts and diagnostic auditability score.",
            },
            {
                "priority": "P6 tail-risk solver prototype",
                "claim_status": "selector_prototype",
                "artifact": "paper4_tail_risk_selector_results.csv",
                "quarto_page": "19v-tail-mdcp-causal-v2.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Constraint grid selector, not a full CVaR LP.",
            },
            {
                "priority": "P7 multi-period solver",
                "claim_status": "toy_dynamic_policy",
                "artifact": "paper4_multi_period_solver_results.parquet",
                "quarto_page": "19t-multi-period-solver.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Monthly policy switching toy.",
            },
            {
                "priority": "P8 MDCP formal",
                "claim_status": "source_registry_and_worst_source_eval",
                "artifact": "paper4_mdcp_policy_eval_v2.csv",
                "quarto_page": "19v-tail-mdcp-causal-v2.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Formal source definitions, still proxy coverage.",
            },
            {
                "priority": "P9 CATE gated",
                "claim_status": "gated_toy_value",
                "artifact": "paper4_cate_exact_topk_value.csv",
                "quarto_page": "19v-tail-mdcp-causal-v2.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "No causal promotion.",
            },
            {
                "priority": "P10 artifact browser/catalog",
                "claim_status": "implemented",
                "artifact": "paper4_artifact_catalog.csv",
                "quarto_page": "19u-artifact-catalog-and-claims.qmd",
                "test_guardrail": "test_paper4_v2_priority_artifacts_exist",
                "caveat": "Catalog only.",
            },
        ]
    )
    return catalog, claims


def write_figures(
    selector_v2: pd.DataFrame,
    regret_pareto: pd.DataFrame,
    mdcp_frontier: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(selector_v2["ecl_v2"], selector_v2["net_return_after_ecl_v2"], s=70)
        ax.set_xlabel("ECL v2 baseline")
        ax.set_ylabel("Net return after ECL v2")
        ax.set_title("Paper 4 selector v2 frontier")
        ax.grid(True, alpha=0.25)
        path = FIGURE_DIR / "paper4_fig8_selector_v2_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        if not regret_pareto.empty:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            ax.scatter(
                regret_pareto["mean_regret_net_ecl"],
                regret_pareto["auditability_score_v2"],
                s=70,
            )
            ax.set_xlabel("Mean regret net of ECL")
            ax.set_ylabel("Auditability score v2")
            ax.set_title("Paper 4 regret-auditability replay v2")
            ax.grid(True, alpha=0.25)
            path = FIGURE_DIR / "paper4_fig9_regret_auditability_v2.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)

        if not mdcp_frontier.empty:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            agg = mdcp_frontier.groupby("policy_id", as_index=False).agg(
                worst=("worst_source_coverage_online_90", "min")
            )
            ax.hist(agg["worst"].dropna(), bins=15)
            ax.set_xlabel("Worst-source online coverage 90")
            ax.set_ylabel("Policies")
            ax.set_title("Paper 4 MDCP v2 worst-source distribution")
            path = FIGURE_DIR / "paper4_fig10_mdcp_worst_source_v2.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    except Exception:
        pass
    return paths


def main() -> None:
    generated: list[Path] = []
    base = _load_base_loan_frame()
    policy_universe = load_policy_universe()

    exact_alloc, exact_eval, exact_comparison, exact_status = build_exact_limited_topk(
        base, policy_universe
    )
    generated.append(_write_parquet("paper4_exact_limited_topk_allocations.parquet", exact_alloc))
    generated.append(_write_csv("paper4_exact_limited_topk_policy_eval.csv", exact_eval))
    generated.append(_write_csv("paper4_exact_vs_proxy_comparison.csv", exact_comparison))
    generated.append(_write_json("paper4_exact_replay_topk_status.json", exact_status))

    v2_evidence = build_v2_policy_loan_evidence(exact_alloc)
    generated.append(_write_parquet("paper4_v2_policy_loan_evidence.parquet", v2_evidence))

    ifrs9_grid, ifrs9_summary, sicr_compare = build_ifrs9_v2(v2_evidence)
    generated.append(_write_parquet("paper4_ifrs9_lifetime_ecl_grid.parquet", ifrs9_grid))
    generated.append(_write_csv("paper4_sicr_rule_comparison.csv", sicr_compare))
    generated.append(_write_csv("paper4_ifrs9_stage_transition_stress.csv", sicr_compare))
    generated.append(_write_csv("paper4_ifrs9_v2_policy_summary.csv", ifrs9_summary))

    selector_config, selector_v2 = build_selector_v2(policy_universe, ifrs9_summary, v2_evidence)
    generated.append(_write_json("paper4_selector_v2_config.json", selector_config))
    generated.append(_write_csv("paper4_selector_v2_results.csv", selector_v2))
    selector_memo = """# Paper 4 Selector v2 Decision Memo

Selector v2 is diagnostic. It combines IFRS9 lifetime proxy, tail risk, MDCP,
fairness proxy and auditability. It does not create a champion and does not
write `paper4_final_promotion.json`.

Decision: use the top candidates as an exact-full-universe rerun queue, not as a
replacement for the Paper Estrella champion.
"""
    generated.append(_write_note("paper4_selector_v2_decision_memo.md", selector_memo))

    online_intervals, online_policy_replay, online_regret = build_online_conformal_real(
        base, v2_evidence
    )
    generated.append(_write_parquet("paper4_online_conformal_intervals.parquet", online_intervals))
    generated.append(
        _write_parquet("paper4_online_conformal_policy_replay.parquet", online_policy_replay)
    )
    generated.append(_write_csv("paper4_online_conformal_coverage_regret_v2.csv", online_regret))

    regret_replay, regret_pareto, hybrid = build_regret_auditability_v2(
        selector_v2, online_policy_replay
    )
    generated.append(_write_parquet("paper4_regret_auditability_replay_v2.parquet", regret_replay))
    generated.append(_write_csv("paper4_regret_auditability_pareto_v2.csv", regret_pareto))
    generated.append(_write_csv("paper4_spo_crpto_hybrid_results.csv", hybrid))

    tail_selector, tail_grid, tail_figs = build_tail_solver_prototype(v2_evidence, selector_v2)
    generated.append(_write_csv("paper4_tail_risk_selector_results.csv", tail_selector))
    generated.append(_write_csv("paper4_oce_cvar_constraint_grid.csv", tail_grid))
    generated.extend(tail_figs)

    mp_path, mp_summary, mp_comparison = build_multi_period_solver_v2(
        selector_v2, online_policy_replay
    )
    generated.append(_write_parquet("paper4_multi_period_solver_results.parquet", mp_path))
    generated.append(_write_csv("paper4_multi_period_policy_path.csv", mp_path))
    generated.append(_write_csv("paper4_dla_vs_static_policy_comparison.csv", mp_comparison))

    mdcp_defs, mdcp_eval, mdcp_frontier = build_mdcp_formal_v2(v2_evidence, online_intervals)
    generated.append(_write_csv("paper4_mdcp_source_definitions.csv", mdcp_defs))
    generated.append(_write_csv("paper4_mdcp_policy_eval_v2.csv", mdcp_eval))
    generated.append(_write_csv("paper4_mdcp_worst_source_frontier.csv", mdcp_frontier))

    cate_overlap, cate_value = build_cate_v2(v2_evidence)
    generated.append(_write_csv("paper4_overlap_policy_funded_sets.csv", cate_overlap))
    generated.append(_write_csv("paper4_cate_exact_topk_value.csv", cate_value))
    causal_note = """# Paper 4 CATE v2 Gate

CATE v2 is evaluated only for the exact-limited top-k funded sets. It remains a
gated toy value. No causal policy is promoted until treatment, outcome, overlap
and sensitivity are accepted.
"""
    generated.append(_write_note("paper4_causal_v2_gate.md", causal_note))

    generated.extend(write_figures(selector_v2, regret_pareto, mdcp_frontier))

    catalog, claims = build_artifact_catalog(generated)
    generated.append(_write_csv("paper4_artifact_catalog.csv", catalog))
    generated.append(_write_csv("paper4_claim_artifact_test_matrix.csv", claims))

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v2_priorities_1_to_10",
        "mode": "diagnostic_no_promotion",
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "priorities_completed": 10,
        "exact_limited_topk": exact_status,
        "generated_artifacts": [p.relative_to(ROOT).as_posix() for p in generated],
    }
    generated.append(_write_json("paper4_v2_priorities_status.json", status))
    print(json.dumps({"generated": [p.relative_to(ROOT).as_posix() for p in generated]}, indent=2))


if __name__ == "__main__":
    main()
