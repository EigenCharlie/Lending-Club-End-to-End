"""Build Paper 4 next-wave experiment artifacts.

This layer advances the living lab beyond v2 diagnostics. It runs the most
important searches that can be executed with the current repository artifacts:

1. full-universe exact top-k solves;
2. exact-funded IFRS9/tail/MDCP refresh;
3. selector gate sensitivity;
4. online conformal method search;
5. tail-risk LP prototype;
6. multi-period sample-path search;
7. causal treatment identification grid;
8. fairness proxy governance grid;
9. non-promoted lane readiness dashboard;
10. next-wave claim matrix.

It still does not write ``paper4_final_promotion.json``. Every output remains
diagnostic unless its own artifact says otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _load_base_loan_frame,
    _policy_effective_pd,
    _safe_read_csv,
    _safe_read_json,
    _safe_read_parquet,
    _weighted_average,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD, load_policy_universe
from scripts.papers.build_paper4_v2_priorities import (
    _canonical_champion_rows,
    _policy_score,
    _topk_policy_ids,
)
from src.optimization.portfolio_model import optimize_portfolio_allocation

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
STATUS_DIR = OUT_ROOT / "status"
FIGURE_DIR = OUT_ROOT / "figures"
NOTE_DIR = OUT_ROOT / "notes"

SCHEMA_VERSION = "2026-05-12.4"
FULL_SOLVE_TIME_LIMIT_SECONDS = 300
TAIL_POOL_N = 20_000


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


def _as_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _prepare_base(base: pd.DataFrame) -> pd.DataFrame:
    work = base.copy()
    work["loan_id"] = work["id"].astype(str)
    work["issue_month"] = _as_month(work["issue_month"])
    work["original_grade"] = work["original_grade"].astype(str)
    work["term"] = pd.to_numeric(work["term"], errors="coerce").fillna(0).astype(int)
    work["score_decile"] = pd.qcut(
        work["y_pred"].rank(method="first"), 10, labels=False, duplicates="drop"
    ).astype(int)
    work["income_band"] = pd.qcut(
        pd.to_numeric(work["annual_inc"], errors="coerce").rank(method="first"),
        5,
        labels=["inc_q1", "inc_q2", "inc_q3", "inc_q4", "inc_q5"],
        duplicates="drop",
    ).astype(str)
    work["dti_band"] = pd.qcut(
        pd.to_numeric(work["dti"], errors="coerce").rank(method="first"),
        5,
        labels=["dti_q1", "dti_q2", "dti_q3", "dti_q4", "dti_q5"],
        duplicates="drop",
    ).astype(str)
    state_counts = work["addr_state"].fillna("unknown").astype(str).value_counts()
    top_states = set(state_counts.head(20).index)
    work["state_top20"] = np.where(
        work["addr_state"].fillna("unknown").astype(str).isin(top_states),
        work["addr_state"].fillna("unknown").astype(str),
        "other_state",
    )
    return work


def _funded_metrics(funded: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    if funded.empty:
        return {
            f"{prefix}n_funded": 0,
            f"{prefix}funded_exposure": 0.0,
            f"{prefix}realized_return": 0.0,
            f"{prefix}weighted_pd_high": np.nan,
            f"{prefix}coverage_alpha01": np.nan,
        }
    return {
        f"{prefix}n_funded": int(funded["loan_id"].nunique()),
        f"{prefix}funded_exposure": float(funded["funded_exposure"].sum()),
        f"{prefix}realized_return": float(funded["realized_return_proxy_lgd45"].sum()),
        f"{prefix}weighted_pd_high": _weighted_average(
            funded["pd_high_alpha01"], funded["funded_exposure"]
        ),
        f"{prefix}coverage_alpha01": float(1.0 - funded["miscovered_alpha01"].mean()),
    }


def _solve_full_policy(
    base: pd.DataFrame, policy: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy_id = str(policy["policy_id"])
    effective_pd = _policy_effective_pd(base, policy)
    solution = optimize_portfolio_allocation(
        loans=base,
        pd_point=base["pd_point_alpha01"].to_numpy(dtype=float),
        pd_low=base["pd_low_alpha01"].to_numpy(dtype=float),
        pd_high=base["pd_high_alpha01"].to_numpy(dtype=float),
        lgd=np.full(len(base), DEFAULT_LGD, dtype=float),
        int_rates=base["int_rate_decimal"].to_numpy(dtype=float),
        total_budget=BUDGET,
        max_concentration=0.25,
        max_portfolio_pd=float(policy["risk_tolerance"]),
        robust=True,
        uncertainty_aversion=float(policy.get("uncertainty_aversion", 0.0)),
        min_budget_utilization=float(policy.get("min_budget_utilization", 0.0)),
        pd_cap_slack_penalty=float(policy.get("pd_cap_slack_penalty", 0.0)),
        pd_constraint_override=effective_pd,
        time_limit=FULL_SOLVE_TIME_LIMIT_SECONDS,
        threads=4,
        solver_backend="highs",
    )
    allocation = np.array(
        [float(solution["allocation"].get(i, 0.0)) for i in range(len(base))],
        dtype=float,
    )
    mask = allocation > 1e-8
    funded = base.loc[mask].copy()
    funded["policy_id"] = policy_id
    funded["loan_id"] = funded["id"].astype(str)
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["portfolio_weight"] = funded["funded_exposure"] / max(
        float(funded["funded_exposure"].sum()), 1e-12
    )
    funded["pd_point"] = funded["pd_point_alpha01"]
    funded["effective_pd_alpha01"] = effective_pd[mask]
    funded["miscovered_alpha01"] = funded["y_true"].gt(funded["pd_high_alpha01"])
    funded["funded_flag"] = True
    funded["is_champion_exact"] = False
    funded["exact_scope"] = "full_universe_276869"
    funded["solver_status"] = str(solution.get("solver_status", "unknown"))
    funded["candidate_pool_n"] = int(len(base))
    funded["next_wave_source"] = "highs_full_universe_topk"
    funded["reconstruction_method"] = "highs_full_universe_exact"
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
        "purpose",
        "home_ownership",
        "addr_state",
        "state_top20",
        "income_band",
        "dti_band",
        "score_decile",
        "loan_amnt",
        "int_rate",
        "int_rate_decimal",
        "annual_inc",
        "dti",
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
        "next_wave_source",
        "realized_return_proxy_lgd45",
        "ecl_baseline_lgd45",
    ]
    metrics = {
        "policy_id": policy_id,
        "candidate_pool_n": int(len(base)),
        "solver_status": str(solution.get("solver_status", "unknown")),
        "objective_value": float(solution.get("objective_value", np.nan)),
        **_funded_metrics(funded, prefix="full_"),
    }
    return funded[keep], metrics


def build_full_universe_topk(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
    *,
    max_policies: int,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    alloc_path = TABLE_DIR / "paper4_full_universe_topk_allocations.parquet"
    eval_path = TABLE_DIR / "paper4_full_universe_topk_policy_eval.csv"
    comp_path = TABLE_DIR / "paper4_full_universe_vs_exact_limited_comparison.csv"
    status_path = STATUS_DIR / "paper4_full_universe_exact_topk_status.json"
    if (
        not force
        and alloc_path.exists()
        and eval_path.exists()
        and comp_path.exists()
        and status_path.exists()
    ):
        return (
            pd.read_parquet(alloc_path),
            pd.read_csv(eval_path),
            pd.read_csv(comp_path),
            json.loads(status_path.read_text(encoding="utf-8")),
        )

    selected_ids = _topk_policy_ids()[:max_policies]
    rows = []
    metrics: list[dict[str, Any]] = []
    champion = _canonical_champion_rows().copy()
    if not champion.empty:
        champion["exact_scope"] = "paper1_canonical_full_universe"
        champion["next_wave_source"] = "paper1_exact_champion"
        rows.append(champion)
        metrics.append(
            {
                "policy_id": "paper1_economic_champion",
                "candidate_pool_n": int(len(base)),
                "solver_status": "paper1_canonical_exact",
                "objective_value": np.nan,
                **_funded_metrics(champion, prefix="full_"),
            }
        )
    for policy_id in selected_ids:
        if policy_id == "paper1_economic_champion":
            continue
        policy = policy_universe[policy_universe["policy_id"].eq(policy_id)].iloc[0]
        funded, local_metrics = _solve_full_policy(base, policy)
        rows.append(funded)
        metrics.append(local_metrics)

    allocations = pd.concat(rows, ignore_index=True, sort=False)
    for column in [
        "loan_id",
        "policy_id",
        "period",
        "original_grade",
        "sub_grade",
        "term",
        "purpose",
        "home_ownership",
        "addr_state",
        "state_top20",
        "income_band",
        "dti_band",
        "exact_scope",
        "solver_status",
        "next_wave_source",
        "reconstruction_method",
    ]:
        if column in allocations:
            allocations[column] = allocations[column].astype(str)
    if "issue_d" in allocations:
        allocations["issue_d"] = allocations["issue_d"].astype(str)
    allocations["issue_month"] = _as_month(allocations["issue_month"])
    eval_table = pd.DataFrame(metrics)

    limited = _safe_read_parquet(TABLE_DIR / "paper4_exact_limited_topk_allocations.parquet")
    comparison_rows = []
    for policy_id, group in allocations.groupby("policy_id"):
        full_ids = set(group["loan_id"].astype(str))
        limited_ids = set(
            limited.loc[limited["policy_id"].astype(str).eq(str(policy_id)), "loan_id"].astype(str)
        )
        union = full_ids | limited_ids
        comparison_rows.append(
            {
                "policy_id": policy_id,
                "full_universe_n": len(full_ids),
                "exact_limited_n": len(limited_ids),
                "overlap_n": len(full_ids & limited_ids),
                "jaccard_full_vs_exact_limited": len(full_ids & limited_ids) / max(len(union), 1),
                "full_universe_return": float(group["realized_return_proxy_lgd45"].sum()),
                "exact_limited_return": float(
                    limited.loc[
                        limited["policy_id"].astype(str).eq(str(policy_id)),
                        "realized_return_proxy_lgd45",
                    ].sum()
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison["return_delta_full_minus_limited"] = (
        comparison["full_universe_return"] - comparison["exact_limited_return"]
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": "full_universe_exact_topk_no_promotion",
        "candidate_pool_n": int(len(base)),
        "policy_ids": selected_ids,
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "promotion_json_created": False,
        "time_limit_seconds_per_solve": FULL_SOLVE_TIME_LIMIT_SECONDS,
    }
    return allocations, eval_table, comparison, status


def build_exact_ifrs9_tail_grid(allocations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = [
        ("baseline", 1.00, 0.45),
        ("adverse", 1.25, 0.55),
        ("severe", 1.60, 0.65),
    ]
    rules = [
        ("absolute_pd_30", math.inf, 0.30),
        ("relative_pd_1p5", 1.50, math.inf),
        ("hybrid_pd_grade", 1.50, 0.30),
    ]
    frames = []
    base_pd = pd.to_numeric(allocations["pd_point"], errors="coerce").fillna(0.0)
    for scenario, pd_multiplier, lgd in scenarios:
        for sicr_rule, rel_threshold, abs_threshold in rules:
            work = allocations.copy()
            work["scenario"] = scenario
            work["sicr_rule"] = sicr_rule
            work["lgd_scenario"] = lgd
            work["scenario_pd"] = np.clip(
                pd.to_numeric(work["pd_high_alpha01"], errors="coerce").fillna(0.0) * pd_multiplier,
                0.0,
                1.0,
            )
            rel = work["scenario_pd"] / np.maximum(base_pd.reindex(work.index).to_numpy(), 1e-6)
            grade_stress = work["original_grade"].astype(str).isin(["E", "F", "G"]) & work[
                "scenario"
            ].isin(["adverse", "severe"])
            sicr = work["scenario_pd"].ge(abs_threshold) | (rel >= rel_threshold)
            if sicr_rule == "hybrid_pd_grade":
                sicr = sicr | grade_stress
            work["ifrs9_stage_next_wave"] = np.select(
                [work["y_true"].eq(1.0), sicr],
                ["Stage 3 observed", "Stage 2 lifetime proxy"],
                default="Stage 1 12m proxy",
            )
            lifetime_factor = np.where(
                work["ifrs9_stage_next_wave"].eq("Stage 2 lifetime proxy"), 3.0, 1.0
            )
            work["ecl_next_wave"] = np.where(
                work["ifrs9_stage_next_wave"].eq("Stage 3 observed"),
                work["lgd_scenario"] * work["funded_exposure"],
                np.minimum(work["scenario_pd"] * lifetime_factor, 1.0)
                * work["lgd_scenario"]
                * work["funded_exposure"],
            )
            work["provision_next_wave"] = work["ecl_next_wave"]
            frames.append(work)
    grid = pd.concat(frames, ignore_index=True)
    summary = grid.groupby(["policy_id", "scenario", "sicr_rule"], as_index=False).agg(
        n_funded=("loan_id", "nunique"),
        funded_exposure=("funded_exposure", "sum"),
        realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
        ecl_next_wave=("ecl_next_wave", "sum"),
        provision_next_wave=("provision_next_wave", "sum"),
        stage2_lifetime_share=(
            "ifrs9_stage_next_wave",
            lambda x: float(x.eq("Stage 2 lifetime proxy").mean()),
        ),
        stage3_observed_share=(
            "ifrs9_stage_next_wave",
            lambda x: float(x.eq("Stage 3 observed").mean()),
        ),
        mean_loss_rate=("y_true", lambda x: float(DEFAULT_LGD * x.mean())),
        cvar_95_pd_lgd=("scenario_pd", lambda x: float((x * DEFAULT_LGD).quantile(0.95))),
    )
    summary["net_return_after_ecl_next_wave"] = (
        summary["realized_return_proxy_lgd45"] - summary["provision_next_wave"]
    )
    return grid, summary


def _rolling_interval_frame(base: pd.DataFrame, method: str) -> pd.DataFrame:
    work = base[
        [
            "loan_id",
            "issue_month",
            "period",
            "original_grade",
            "term",
            "score_decile",
            "y_true",
            "y_pred",
        ]
    ].copy()
    work["score_abs"] = (work["y_true"] - work["y_pred"]).abs()
    months = sorted(work["issue_month"].dropna().unique())
    frames = []
    alpha = 0.10
    eta = 0.05
    for month in months:
        current = work[work["issue_month"].eq(month)].copy()
        prior = work[work["issue_month"].lt(month)]
        if prior.empty:
            qhat = pd.Series(np.full(len(current), 0.50), index=current.index)
            source = "first_month_fallback"
        else:
            if method == "rolling_global":
                q = float(prior["score_abs"].quantile(0.90))
                qhat = pd.Series(np.full(len(current), q), index=current.index)
                source = "global"
            elif method == "rolling_grade":
                stats = prior.groupby("original_grade")["score_abs"].quantile(0.90)
                q_global = float(prior["score_abs"].quantile(0.90))
                qhat = current["original_grade"].map(stats).fillna(q_global)
                source = "grade"
            elif method == "rolling_grade_term":
                stats = prior.groupby(["original_grade", "term"])["score_abs"].quantile(0.90)
                q_global = float(prior["score_abs"].quantile(0.90))
                qhat = current.set_index(["original_grade", "term"]).index.map(stats).astype(float)
                qhat = pd.Series(qhat, index=current.index).fillna(q_global)
                source = "grade_term"
            elif method == "rolling_score_decile":
                stats = prior.groupby("score_decile")["score_abs"].quantile(0.90)
                q_global = float(prior["score_abs"].quantile(0.90))
                qhat = current["score_decile"].map(stats).fillna(q_global)
                source = "score_decile"
            elif method == "aci_global":
                target_q = 1.0 - float(np.clip(alpha, 0.02, 0.30))
                q = float(prior["score_abs"].quantile(target_q))
                qhat = pd.Series(np.full(len(current), q), index=current.index)
                source = "aci_global"
            else:
                raise ValueError(f"Unsupported online method: {method}")
        current["online_method"] = method
        current["online_source"] = source
        current["qhat"] = np.asarray(qhat, dtype=float)
        current["pd_low_online"] = np.clip(current["y_pred"] - current["qhat"], 0.0, 1.0)
        current["pd_high_online"] = np.clip(current["y_pred"] + current["qhat"], 0.0, 1.0)
        current["covered_online"] = current["y_true"].between(
            current["pd_low_online"], current["pd_high_online"]
        )
        current["interval_width_online"] = current["pd_high_online"] - current["pd_low_online"]
        if method == "aci_global" and not current.empty:
            err = 1.0 - float(current["covered_online"].mean())
            alpha = float(np.clip(alpha + eta * (0.10 - err), 0.02, 0.30))
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def build_online_method_search(
    base: pd.DataFrame,
    allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [
        "rolling_global",
        "rolling_grade",
        "rolling_grade_term",
        "rolling_score_decile",
        "aci_global",
    ]
    summaries = []
    policy_months = []
    for method in methods:
        intervals = _rolling_interval_frame(base, method)
        merged = allocations[["policy_id", "loan_id", "issue_month", "funded_exposure"]].merge(
            intervals[
                [
                    "loan_id",
                    "online_method",
                    "online_source",
                    "covered_online",
                    "interval_width_online",
                ]
            ],
            on="loan_id",
            how="left",
        )
        policy_month = (
            merged.groupby(["online_method", "policy_id", "issue_month"], as_index=False)
            .agg(
                n_funded=("loan_id", "nunique"),
                funded_exposure=("funded_exposure", "sum"),
                coverage_online=("covered_online", "mean"),
                avg_width_online=("interval_width_online", "mean"),
            )
            .rename(columns={"issue_month": "month"})
        )
        policy_month["coverage_regret_90"] = (0.90 - policy_month["coverage_online"]).clip(lower=0)
        policy_months.append(policy_month)
        summaries.append(
            {
                "online_method": method,
                "coverage_online_mean": float(policy_month["coverage_online"].mean()),
                "coverage_online_min_month_policy": float(policy_month["coverage_online"].min()),
                "avg_width_online": float(merged["interval_width_online"].mean()),
                "total_coverage_regret_90": float(policy_month["coverage_regret_90"].sum()),
                "policy_month_rows": int(len(policy_month)),
                "promotion_candidate": bool(
                    policy_month["coverage_online"].mean() >= 0.90
                    and policy_month["coverage_online"].min() >= 0.80
                ),
            }
        )
    return pd.DataFrame(summaries), pd.concat(policy_months, ignore_index=True)


def build_selector_gate_sensitivity(selector: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    mdcp_thresholds = [0.50, 0.60, 0.70, 0.80, 0.85]
    fairness_thresholds = [0.20, 0.25, 0.30, 0.35]
    min_score_quantiles = [0.00, 0.50, 0.75]
    for mdcp in mdcp_thresholds:
        for fairness in fairness_thresholds:
            for score_q in min_score_quantiles:
                min_score = float(selector["selector_v2_score"].quantile(score_q))
                work = selector.copy()
                work["gate_pass_search"] = (
                    work["all_bounds_hold"].astype(bool)
                    & work["ab_pass_all"].astype(bool)
                    & work["worst_source_coverage_alpha01"].fillna(0.0).ge(mdcp)
                    & work["max_abs_representation_gap"].fillna(1.0).le(fairness)
                    & work["selector_v2_score"].ge(min_score)
                )
                passed = work[work["gate_pass_search"]].sort_values(
                    "selector_v2_score", ascending=False
                )
                rows.append(
                    {
                        "mdcp_threshold": mdcp,
                        "fairness_gap_threshold": fairness,
                        "min_score_quantile": score_q,
                        "min_score": min_score,
                        "n_gate_pass": int(len(passed)),
                        "top_policy_id": "" if passed.empty else str(passed.iloc[0]["policy_id"]),
                        "top_selector_v2_score": np.nan
                        if passed.empty
                        else float(passed.iloc[0]["selector_v2_score"]),
                        "status": "candidate_gate_search",
                    }
                )
    grid = pd.DataFrame(rows)
    summary = (
        grid.groupby(["mdcp_threshold", "fairness_gap_threshold"], as_index=False)
        .agg(
            max_gate_pass=("n_gate_pass", "max"),
            mean_gate_pass=("n_gate_pass", "mean"),
            distinct_top_policies=(
                "top_policy_id",
                lambda x: int(pd.Series(x).replace("", np.nan).nunique()),
            ),
        )
        .sort_values(["max_gate_pass", "mean_gate_pass"], ascending=False)
    )
    return grid, summary


def _tail_metrics_for_allocations(allocations: pd.DataFrame) -> dict[str, float]:
    if allocations.empty:
        return {
            "mean_stress_loss_rate": np.nan,
            "cvar_95_stress_loss_rate": np.nan,
            "oce_theta5": np.nan,
        }
    stress_loss = (
        pd.to_numeric(allocations["pd_high_alpha01"], errors="coerce").fillna(0.0) * DEFAULT_LGD
    )
    weights = pd.to_numeric(allocations["funded_exposure"], errors="coerce").fillna(0.0)
    q95 = float(stress_loss.quantile(0.95))
    tail = stress_loss[stress_loss >= q95]
    theta = 5.0
    return {
        "mean_stress_loss_rate": _weighted_average(stress_loss, weights),
        "cvar_95_stress_loss_rate": float(tail.mean()) if len(tail) else np.nan,
        "oce_theta5": float(np.log(np.mean(np.exp(theta * stress_loss))) / theta),
    }


def build_tail_penalty_lp_search(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    champion_policy = (
        policy_universe[policy_universe["policy_id"].ne("paper1_economic_champion")]
        .sort_values("risk_tolerance")
        .iloc[len(policy_universe) // 2]
    )
    score = _policy_score(base, champion_policy, _policy_effective_pd(base, champion_policy))
    pool = base.iloc[np.argsort(-score)[:TAIL_POOL_N]].reset_index(drop=True).copy()
    rows = []
    allocation_rows = []
    for risk_tolerance in [0.155, 0.165, 0.175]:
        for penalty in [0.00, 0.25, 0.50, 1.00]:
            adjusted_rates = pool["int_rate_decimal"].to_numpy(dtype=float) - (
                penalty * pool["pd_high_alpha01"].to_numpy(dtype=float) * DEFAULT_LGD
            )
            sol = optimize_portfolio_allocation(
                loans=pool,
                pd_point=pool["pd_point_alpha01"].to_numpy(dtype=float),
                pd_low=pool["pd_low_alpha01"].to_numpy(dtype=float),
                pd_high=pool["pd_high_alpha01"].to_numpy(dtype=float),
                lgd=np.full(len(pool), DEFAULT_LGD, dtype=float),
                int_rates=adjusted_rates,
                total_budget=BUDGET,
                max_concentration=0.25,
                max_portfolio_pd=risk_tolerance,
                robust=True,
                uncertainty_aversion=0.10,
                pd_constraint_override=pool["pd_high_alpha01"].to_numpy(dtype=float),
                time_limit=120,
                threads=4,
                solver_backend="highs",
            )
            allocation = np.array(
                [float(sol["allocation"].get(i, 0.0)) for i in range(len(pool))],
                dtype=float,
            )
            mask = allocation > 1e-8
            funded = pool.loc[mask].copy()
            funded["funded_exposure"] = allocation[mask] * funded["loan_amnt"]
            funded["loan_id"] = funded["id"].astype(str)
            funded["tail_lp_policy_id"] = f"tail_lp_rt{risk_tolerance:.3f}_pen{penalty:.2f}"
            funded["allocation_fraction"] = allocation[mask]
            funded["realized_return_proxy_lgd45"] = (
                funded["funded_exposure"] * funded["int_rate_decimal"] * (1.0 - funded["y_true"])
                - funded["funded_exposure"] * DEFAULT_LGD * funded["y_true"]
            )
            metrics = _tail_metrics_for_allocations(funded)
            rows.append(
                {
                    "tail_lp_policy_id": funded["tail_lp_policy_id"].iloc[0]
                    if not funded.empty
                    else "",
                    "risk_tolerance": risk_tolerance,
                    "tail_penalty": penalty,
                    "candidate_pool_n": len(pool),
                    "solver_status": str(sol.get("solver_status", "unknown")),
                    "objective_value": float(sol.get("objective_value", np.nan)),
                    "n_funded": int(mask.sum()),
                    "funded_exposure": float(funded["funded_exposure"].sum()),
                    "realized_return_proxy_lgd45": float(
                        funded["realized_return_proxy_lgd45"].sum()
                    ),
                    **metrics,
                    "status": "tail_penalty_lp_candidate_pool_search",
                }
            )
            allocation_rows.append(
                funded[
                    [
                        "tail_lp_policy_id",
                        "loan_id",
                        "issue_month",
                        "original_grade",
                        "loan_amnt",
                        "funded_exposure",
                        "allocation_fraction",
                        "pd_high_alpha01",
                        "y_true",
                        "realized_return_proxy_lgd45",
                    ]
                ]
            )
    return pd.DataFrame(rows), pd.concat(allocation_rows, ignore_index=True)


def build_mdcp_and_fairness_search(
    base: pd.DataFrame,
    allocations: pd.DataFrame,
    online_method_summary: pd.DataFrame,
    online_policy_month: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_method = online_method_summary.sort_values(
        ["promotion_candidate", "total_coverage_regret_90", "avg_width_online"],
        ascending=[False, True, True],
    ).iloc[0]["online_method"]
    # Use v2 online intervals for loan-level source coverage if method-specific intervals
    # are not materialized. The search ranking still comes from method-level results.
    intervals = _safe_read_parquet(TABLE_DIR / "paper4_online_conformal_intervals.parquet")
    covered = intervals[["loan_id", "covered_online_90"]].rename(
        columns={"covered_online_90": "covered_source_search"}
    )
    enriched = allocations.merge(covered, on="loan_id", how="left")
    source_specs = {
        "grade": ["original_grade"],
        "period": ["period"],
        "grade_period": ["original_grade", "period"],
        "term": ["term"],
        "purpose": ["purpose"],
        "home_ownership": ["home_ownership"],
        "state_top20": ["state_top20"],
        "income_band": ["income_band"],
        "dti_band": ["dti_band"],
        "score_decile": ["score_decile"],
    }
    mdcp_rows = []
    fairness_rows = []
    universe_exposure = base.assign(universe_exposure=base["loan_amnt"].astype(float))
    for source_id, cols in source_specs.items():
        local = enriched.copy()
        local["source_value"] = local[cols].astype(str).agg("|".join, axis=1)
        source_eval = local.groupby(["policy_id", "source_value"], as_index=False).agg(
            n=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            coverage=("covered_source_search", "mean"),
        )
        source_eval = source_eval[source_eval["n"].ge(5)]
        if source_eval.empty:
            continue
        frontier = source_eval.groupby("policy_id", as_index=False).agg(
            worst_source_coverage=("coverage", "min"),
            mean_source_coverage=("coverage", "mean"),
            n_sources=("source_value", "nunique"),
        )
        frontier["source_id"] = source_id
        frontier["online_method_rank_reference"] = best_method
        frontier["mdcp_search_pass_80"] = frontier["worst_source_coverage"].ge(0.80)
        mdcp_rows.append(frontier)

        base_source = universe_exposure.copy()
        base_source["source_value"] = base_source[cols].astype(str).agg("|".join, axis=1)
        base_dist = base_source.groupby("source_value", as_index=False)["universe_exposure"].sum()
        base_total = max(float(base_dist["universe_exposure"].sum()), 1e-12)
        base_dist["universe_share"] = base_dist["universe_exposure"] / base_total
        funded_dist = local.groupby(["policy_id", "source_value"], as_index=False)[
            "funded_exposure"
        ].sum()
        funded_totals = funded_dist.groupby("policy_id")["funded_exposure"].transform("sum")
        funded_dist["funded_share"] = funded_dist["funded_exposure"] / funded_totals.clip(
            lower=1e-12
        )
        gap = funded_dist.merge(
            base_dist[["source_value", "universe_share"]], on="source_value", how="outer"
        )
        gap["funded_share"] = gap["funded_share"].fillna(0.0)
        gap["universe_share"] = gap["universe_share"].fillna(0.0)
        gap["policy_id"] = gap["policy_id"].fillna("missing_policy")
        gap["abs_gap"] = (gap["funded_share"] - gap["universe_share"]).abs()
        fairness = (
            gap[gap["policy_id"].ne("missing_policy")]
            .groupby("policy_id", as_index=False)
            .agg(max_abs_gap=("abs_gap", "max"), mean_abs_gap=("abs_gap", "mean"))
        )
        fairness["source_id"] = source_id
        fairness["fairness_proxy_pass_25"] = fairness["max_abs_gap"].le(0.25)
        fairness_rows.append(fairness)
    return (
        pd.concat(mdcp_rows, ignore_index=True),
        pd.concat(fairness_rows, ignore_index=True),
        online_policy_month[online_policy_month["online_method"].eq(best_method)].copy(),
    )


def build_causal_identification_search(
    base: pd.DataFrame,
    allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = base.copy()
    selected_ids = set(allocations["loan_id"].astype(str))
    work["treat_term_60m"] = work["term"].astype(int).eq(60)
    grade_median_rate = work.groupby("original_grade")["int_rate_decimal"].transform("median")
    work["treat_high_rate_within_grade"] = work["int_rate_decimal"].gt(grade_median_rate)
    income_median_loan = work.groupby("income_band")["loan_amnt"].transform("median")
    work["treat_large_loan_within_income"] = work["loan_amnt"].gt(income_median_loan)
    work["treat_selected_by_full_topk"] = work["loan_id"].astype(str).isin(selected_ids)
    treatments = [
        ("term_60m", "treat_term_60m", "term length intervention proxy"),
        ("high_rate_within_grade", "treat_high_rate_within_grade", "pricing/tightness proxy"),
        ("large_loan_within_income", "treat_large_loan_within_income", "exposure sizing proxy"),
        (
            "selected_by_full_topk",
            "treat_selected_by_full_topk",
            "policy selection proxy, not causal",
        ),
    ]
    confounders = ["annual_inc", "dti", "loan_amnt", "fico_score", "pd_point_alpha01"]
    rows = []
    detail_rows = []
    for treatment_id, col, interpretation in treatments:
        t = work[col].astype(bool)
        y = work["y_true"].astype(float)
        n_treat = int(t.sum())
        n_control = int((~t).sum())
        prevalence = float(t.mean())
        naive_default_delta = float(y[t].mean() - y[~t].mean()) if n_treat and n_control else np.nan
        smds = []
        for conf in confounders:
            treated = pd.to_numeric(work.loc[t, conf], errors="coerce")
            control = pd.to_numeric(work.loc[~t, conf], errors="coerce")
            pooled = (
                math.sqrt((float(treated.var()) + float(control.var())) / 2.0)
                if len(treated) and len(control)
                else np.nan
            )
            smd = (
                abs(float(treated.mean() - control.mean()) / pooled)
                if pooled and not math.isnan(pooled)
                else np.nan
            )
            smds.append(smd)
            detail_rows.append(
                {
                    "treatment_id": treatment_id,
                    "diagnostic": f"smd_{conf}",
                    "value": smd,
                }
            )
        strata = work.groupby(["original_grade", "period"], as_index=False).agg(
            n=("loan_id", "nunique"), treat_share=(col, "mean")
        )
        eligible = strata[strata["n"].ge(50)]
        support_ok_share = (
            float(eligible["treat_share"].between(0.05, 0.95).mean())
            if not eligible.empty
            else np.nan
        )
        rows.append(
            {
                "treatment_id": treatment_id,
                "interpretation": interpretation,
                "n_treat": n_treat,
                "n_control": n_control,
                "prevalence": prevalence,
                "naive_default_delta": naive_default_delta,
                "max_abs_smd": float(np.nanmax(smds)),
                "support_ok_share_grade_period": support_ok_share,
                "identification_gate": "pass_for_dossier"
                if support_ok_share >= 0.80
                and float(np.nanmax(smds)) <= 0.35
                and treatment_id != "selected_by_full_topk"
                else "gated_research_only",
                "promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def build_multi_period_sample_path_search(
    allocations: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
) -> pd.DataFrame:
    policy_total_exposure = (
        allocations.groupby("policy_id")["funded_exposure"].sum().clip(lower=1e-12).to_dict()
    )
    base_monthly = (
        allocations.groupby(["policy_id", "issue_month", "period"], as_index=False)
        .agg(
            funded_exposure=("funded_exposure", "sum"),
            realized_return=("realized_return_proxy_lgd45", "sum"),
            coverage_alpha01=("miscovered_alpha01", lambda x: float(1.0 - x.mean())),
        )
        .rename(columns={"issue_month": "month"})
    )
    baseline = ifrs9_summary[
        ifrs9_summary["scenario"].eq("baseline") & ifrs9_summary["sicr_rule"].eq("hybrid_pd_grade")
    ][["policy_id", "ecl_next_wave"]].copy()
    ecl_scale = baseline.set_index("policy_id")["ecl_next_wave"].to_dict()
    rows = []
    months = sorted(base_monthly["month"].dropna().unique())
    strategy_specs = ["static_champion", "greedy_net_return", "coverage_gate", "ecl_penalty"]
    for horizon in [3, 6, 12, len(months)]:
        path_months = months[:horizon]
        for scenario, provision_mult in [("baseline", 1.0), ("adverse", 1.25), ("severe", 1.60)]:
            for strategy in strategy_specs:
                budget = BUDGET
                total_net = 0.0
                chosen = []
                for month in path_months:
                    group = base_monthly[base_monthly["month"].eq(month)].copy()
                    if group.empty:
                        continue
                    group["provision_proxy"] = group["policy_id"].map(ecl_scale).fillna(0.0)
                    group["monthly_provision"] = (
                        provision_mult
                        * group["provision_proxy"]
                        * group["funded_exposure"]
                        / group["policy_id"].map(policy_total_exposure).fillna(1e-12)
                    )
                    group["net_return_after_provision"] = (
                        group["realized_return"] - group["monthly_provision"]
                    )
                    if strategy == "static_champion":
                        candidates = group[group["policy_id"].eq("paper1_economic_champion")]
                    elif strategy == "coverage_gate":
                        candidates = group[group["coverage_alpha01"].ge(0.90)]
                    elif strategy == "ecl_penalty":
                        candidates = group.assign(
                            score=lambda df: df["realized_return"] - 1.5 * df["monthly_provision"]
                        )
                    else:
                        candidates = group
                    if candidates.empty:
                        candidates = group
                    sort_col = (
                        "score" if strategy == "ecl_penalty" else "net_return_after_provision"
                    )
                    row = candidates.sort_values(sort_col, ascending=False).iloc[0]
                    budget = max(0.0, budget + float(row["net_return_after_provision"]))
                    total_net += float(row["net_return_after_provision"])
                    chosen.append(str(row["policy_id"]))
                rows.append(
                    {
                        "strategy": strategy,
                        "scenario": scenario,
                        "horizon_months": horizon,
                        "ending_budget": budget,
                        "total_net_return_after_provision": total_net,
                        "unique_policies_used": len(set(chosen)),
                        "policy_path": " > ".join(chosen[:8]) + (" ..." if len(chosen) > 8 else ""),
                        "status": "sample_path_search_not_deployment_policy",
                    }
                )
    return pd.DataFrame(rows)


def build_nonpromoted_lane_dashboard() -> pd.DataFrame:
    cqr = _safe_read_json(ROOT / "models" / "cqr_mondrian_status.json")
    ts = _safe_read_json(ROOT / "models" / "time_series_research_status.json")
    causal_overlap = _safe_read_json(ROOT / "models" / "causal_overlap_status.json")
    causal_sens = _safe_read_json(ROOT / "models" / "causal_sensitivity_status.json")
    rows = [
        {
            "lane": "CATE / causal CRPTO",
            "current_status": "gated_research_only",
            "best_current_signal": f"overlap_pass={causal_overlap.get('overlap_pass')}; sensitivity_pass={causal_sens.get('sensitivity_pass')}",
            "blocker": "treatment/outcome identification and sensitivity",
            "next_search": "causal_treatment_identification_grid + sensitivity rerun",
            "promotion_condition": "identification gate, overlap, sensitivity, policy value all pass",
            "artifact": "paper4_causal_treatment_identification_grid.csv",
        },
        {
            "lane": "Online conformal",
            "current_status": "prototype",
            "best_current_signal": "forward-only rolling search implemented",
            "blocker": "method not final; interval efficiency and worst-month coverage",
            "next_search": "rolling vs Mondrian vs ACI vs UP-OCP",
            "promotion_condition": "mean coverage >= .90 and worst policy-month >= .80 with usable width",
            "artifact": "paper4_online_conformal_method_search.csv",
        },
        {
            "lane": "Temporal TS / ECL",
            "current_status": "research_only",
            "best_current_signal": f"interval_promotable={ts.get('summary', {}).get('interval_promotable')}",
            "blocker": "90% interval coverage failed official gate",
            "next_search": "TS interval redesign + downstream ECL value",
            "promotion_condition": "coverage by horizon and forward coherence pass",
            "artifact": "paper4_nonpromoted_lane_dashboard.csv",
        },
        {
            "lane": "CQR decision-aware",
            "current_status": "comparator_only",
            "best_current_signal": f"best_min_group={cqr.get('summary', {}).get('best_min_group_cov')}",
            "blocker": "zero eligible loans and wide intervals",
            "next_search": "decision-aware CQR / CROMS objective",
            "promotion_condition": "coverage, group coverage, width and decision eligibility all improve",
            "artifact": "paper4_nonpromoted_lane_dashboard.csv",
        },
        {
            "lane": "CVaR/OCE solver",
            "current_status": "prototype",
            "best_current_signal": "tail penalty LP candidate-pool search",
            "blocker": "not yet full scenario CVaR LP",
            "next_search": "scenario CVaR LP with auxiliary variables",
            "promotion_condition": "tail loss drops without unacceptable net return/ECL loss",
            "artifact": "paper4_tail_penalty_lp_search.csv",
        },
        {
            "lane": "MDCP",
            "current_status": "diagnostic",
            "best_current_signal": "worst-source coverage by source family",
            "blocker": "source definitions still proxy",
            "next_search": "source family search and formal MDCP protocol",
            "promotion_condition": "worst-source coverage passes over defended sources",
            "artifact": "paper4_mdcp_source_family_search.csv",
        },
        {
            "lane": "Fairness formal",
            "current_status": "proxy_governance",
            "best_current_signal": "composition-gap grid available",
            "blocker": "no direct protected attributes",
            "next_search": "proxy governance and sensitivity by available segments",
            "promotion_condition": "claim stays proxy unless protected attributes are legally available",
            "artifact": "paper4_fairness_proxy_governance_grid.csv",
        },
        {
            "lane": "SPO+ / DFL hybrid",
            "current_status": "diagnostic_replay",
            "best_current_signal": "regret-auditability replay exists",
            "blocker": "no integrated decision-loss training",
            "next_search": "hybrid objective search preserving CRPTO constraints",
            "promotion_condition": "lower regret while retaining coverage/auditability",
            "artifact": "paper4_regret_auditability_pareto_v2.csv",
        },
        {
            "lane": "Multi-dataset replication",
            "current_status": "future",
            "best_current_signal": "none",
            "blocker": "no second credit dataset in repo",
            "next_search": "external dataset acquisition and leakage audit",
            "promotion_condition": "protocol repeats outside Lending Club",
            "artifact": "paper4_nonpromoted_lane_dashboard.csv",
        },
        {
            "lane": "Open-source CRPTO package",
            "current_status": "parked",
            "best_current_signal": "internal APIs still moving",
            "blocker": "research API not stable",
            "next_search": "API stabilization after full exact + selector gates",
            "promotion_condition": "stable allocation/evaluation API and docs",
            "artifact": "paper4_nonpromoted_lane_dashboard.csv",
        },
    ]
    return pd.DataFrame(rows)


def build_next_wave_claims() -> pd.DataFrame:
    rows = [
        (
            "Full-universe exact top-k",
            "exact_full_universe_diagnostic",
            "paper4_full_universe_topk_allocations.parquet",
            "19w-next-wave-exact-topk.qmd",
            "no promotion JSON",
        ),
        (
            "Exact-funded IFRS9/tail",
            "diagnostic_exact_funded",
            "paper4_full_universe_ifrs9_tail_eval.csv",
            "19w-next-wave-exact-topk.qmd",
            "lifetime ECL still proxy",
        ),
        (
            "Online CP method search",
            "method_search",
            "paper4_online_conformal_method_search.csv",
            "19x-next-wave-method-searches.qmd",
            "not final UP-OCP",
        ),
        (
            "Selector gate sensitivity",
            "gate_search",
            "paper4_selector_gate_sensitivity_grid.csv",
            "19x-next-wave-method-searches.qmd",
            "thresholds exploratory",
        ),
        (
            "Tail penalty LP",
            "candidate_pool_lp_search",
            "paper4_tail_penalty_lp_search.csv",
            "19x-next-wave-method-searches.qmd",
            "20k candidate pool",
        ),
        (
            "MDCP source search",
            "source_family_search",
            "paper4_mdcp_source_family_search.csv",
            "19y-nonpromoted-lanes-and-causal-gates.qmd",
            "sources proxy",
        ),
        (
            "Fairness governance",
            "proxy_governance",
            "paper4_fairness_proxy_governance_grid.csv",
            "19y-nonpromoted-lanes-and-causal-gates.qmd",
            "no protected attributes",
        ),
        (
            "Causal identification",
            "gated_identification_search",
            "paper4_causal_treatment_identification_grid.csv",
            "19y-nonpromoted-lanes-and-causal-gates.qmd",
            "no causal promotion",
        ),
        (
            "Multi-period sample paths",
            "sample_path_search",
            "paper4_multi_period_sample_path_search.csv",
            "19z-next-wave-promotion-dashboard.qmd",
            "toy state transition",
        ),
        (
            "Non-promoted lanes",
            "readiness_dashboard",
            "paper4_nonpromoted_lane_dashboard.csv",
            "19z-next-wave-promotion-dashboard.qmd",
            "dashboard only",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"],
    )


def write_figures(
    full_comparison: pd.DataFrame,
    online_summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    dashboard: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_df = full_comparison[full_comparison["policy_id"].ne("paper1_economic_champion")]
        ax.barh(plot_df["policy_id"], plot_df["jaccard_full_vs_exact_limited"])
        ax.set_xlabel("Jaccard full universe vs exact-limited")
        ax.set_title("Paper 4 next-wave full exact top-k overlap")
        ax.grid(True, axis="x", alpha=0.25)
        path = FIGURE_DIR / "paper4_fig11_full_exact_overlap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(
            online_summary["avg_width_online"],
            online_summary["coverage_online_mean"],
            s=90,
        )
        for _, row in online_summary.iterrows():
            ax.annotate(
                row["online_method"],
                (row["avg_width_online"], row["coverage_online_mean"]),
                fontsize=8,
            )
        ax.axhline(0.90, color="firebrick", linestyle="--", linewidth=1)
        ax.set_xlabel("Average interval width")
        ax.set_ylabel("Mean policy-month coverage")
        ax.set_title("Online conformal method search")
        ax.grid(True, alpha=0.25)
        path = FIGURE_DIR / "paper4_fig12_online_method_search.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(
            gate_summary["mean_gate_pass"],
            gate_summary["max_gate_pass"],
            c=gate_summary["distinct_top_policies"],
            cmap="viridis",
            s=80,
        )
        ax.set_xlabel("Mean pass count across score quantiles")
        ax.set_ylabel("Max pass count")
        ax.set_title("Selector gate sensitivity")
        ax.grid(True, alpha=0.25)
        path = FIGURE_DIR / "paper4_fig13_selector_gate_sensitivity.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        status_counts = dashboard["current_status"].value_counts()
        ax.bar(status_counts.index, status_counts.values)
        ax.set_ylabel("Lanes")
        ax.set_title("Non-promoted lane status dashboard")
        ax.tick_params(axis="x", rotation=30)
        path = FIGURE_DIR / "paper4_fig14_nonpromoted_lane_status.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    except Exception:
        return paths
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-full-solve", action="store_true")
    parser.add_argument("--max-full-policies", type=int, default=6)
    args = parser.parse_args(argv)

    base = _prepare_base(_load_base_loan_frame())
    policy_universe = load_policy_universe()
    generated: list[Path] = []

    allocations, full_eval, full_comp, full_status = build_full_universe_topk(
        base,
        policy_universe,
        max_policies=int(args.max_full_policies),
        force=bool(args.force_full_solve),
    )
    generated.append(_write_parquet("paper4_full_universe_topk_allocations.parquet", allocations))
    generated.append(_write_csv("paper4_full_universe_topk_policy_eval.csv", full_eval))
    generated.append(_write_csv("paper4_full_universe_vs_exact_limited_comparison.csv", full_comp))
    generated.append(_write_json("paper4_full_universe_exact_topk_status.json", full_status))

    ifrs9_grid, ifrs9_summary = build_exact_ifrs9_tail_grid(allocations)
    generated.append(_write_parquet("paper4_full_universe_ifrs9_tail_grid.parquet", ifrs9_grid))
    generated.append(_write_csv("paper4_full_universe_ifrs9_tail_eval.csv", ifrs9_summary))

    online_summary, online_policy_month = build_online_method_search(base, allocations)
    generated.append(_write_csv("paper4_online_conformal_method_search.csv", online_summary))
    generated.append(
        _write_parquet("paper4_online_conformal_method_policy_month.parquet", online_policy_month)
    )

    selector = _safe_read_csv(TABLE_DIR / "paper4_selector_v2_results.csv")
    gate_grid, gate_summary = build_selector_gate_sensitivity(selector)
    generated.append(_write_csv("paper4_selector_gate_sensitivity_grid.csv", gate_grid))
    generated.append(_write_csv("paper4_selector_gate_sensitivity_summary.csv", gate_summary))

    tail_search, tail_allocations = build_tail_penalty_lp_search(base, policy_universe)
    generated.append(_write_csv("paper4_tail_penalty_lp_search.csv", tail_search))
    generated.append(_write_parquet("paper4_tail_penalty_lp_allocations.parquet", tail_allocations))

    mdcp, fairness, best_online_policy_month = build_mdcp_and_fairness_search(
        base, allocations, online_summary, online_policy_month
    )
    generated.append(_write_csv("paper4_mdcp_source_family_search.csv", mdcp))
    generated.append(_write_csv("paper4_fairness_proxy_governance_grid.csv", fairness))
    generated.append(
        _write_parquet("paper4_best_online_policy_month_replay.parquet", best_online_policy_month)
    )

    causal_grid, causal_detail = build_causal_identification_search(base, allocations)
    generated.append(_write_csv("paper4_causal_treatment_identification_grid.csv", causal_grid))
    generated.append(_write_csv("paper4_causal_balance_diagnostics.csv", causal_detail))

    sample_paths = build_multi_period_sample_path_search(allocations, ifrs9_summary)
    generated.append(_write_csv("paper4_multi_period_sample_path_search.csv", sample_paths))

    dashboard = build_nonpromoted_lane_dashboard()
    generated.append(_write_csv("paper4_nonpromoted_lane_dashboard.csv", dashboard))

    claims = build_next_wave_claims()
    generated.append(_write_csv("paper4_next_wave_claim_artifact_matrix.csv", claims))

    figures = write_figures(full_comp, online_summary, gate_summary, dashboard)
    generated.extend(figures)

    note = """---
title: "Paper 4 Next-Wave Promotion Gate Memo"
format: html
---

# Paper 4 Next-Wave Promotion Gate Memo

El next wave ejecuta búsquedas reales sobre los carriles más valiosos, pero no
promueve un champion. La regla se mantiene:

- full-universe exact top-k reduce el caveat proxy;
- online conformal, CVaR/OCE, MDCP, fairness y causalidad siguen sujetos a gates;
- CATE no entra a `C_t` ni a `X^pi` hasta que identificación y sensibilidad pasen;
- no se escribe `paper4_final_promotion.json`.

La decisión científica posterior debe mirar la matriz:
`paper4_next_wave_claim_artifact_matrix.csv`.
"""
    generated.append(_write_note("paper4_next_wave_promotion_gate_memo.qmd", note))

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "next_wave_top10_plus_nonpromoted_lanes",
        "mode": "searches_and_gates_no_promotion",
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "priorities_completed": 10,
        "nonpromoted_lanes_dashboard_rows": int(len(dashboard)),
        "generated_artifacts": [str(path.relative_to(ROOT)) for path in generated],
    }
    generated.append(_write_json("paper4_next_wave_status.json", status))
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
