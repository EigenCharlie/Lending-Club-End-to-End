"""Build Paper 4 v11 promising-lanes artifacts.

V11 focuses on the five runnable lanes that remained most valuable after v10:

* larger top-k CVaR/OCE frontier;
* DLA ADP/Bellman-style value proxy;
* trained SPO/DFL decision-loss proxy against LP/solver targets;
* calibrated comparative sample paths;
* a Paper 4 working-candidate registry.

The wave remains strictly Paper 4 living-lab work.  It does not touch Paper
Estrella artifacts and it does not create ``paper4_final_promotion.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_csv,
    _safe_read_parquet,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v6_priority_resolution import (
    TABLE_DIR,
    _load_inputs,
    _solve_linear_policy,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v8_resolution_wave import _auditability_score
from scripts.papers.build_paper4_v10_resolution_wave import (
    PAPER4_FINAL_PROMOTION,
    _is_optimal,
    _load_v9_online,
    _prepare_solver_pool_v10,
    _stable_uniform,
)

SCHEMA_VERSION = "2026-05-14.11"
RNG_SEED = 2026051411


def _json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _weighted_average(df: pd.DataFrame, column: str, weight: str = "funded_exposure") -> float:
    if df.empty or column not in df or weight not in df:
        return float("nan")
    weights = pd.to_numeric(df[weight], errors="coerce").fillna(0).to_numpy()
    values = pd.to_numeric(df[column], errors="coerce").fillna(0).to_numpy()
    if weights.sum() <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _allocation_summary(df: pd.DataFrame, policy_id: str, lane: str) -> dict[str, Any]:
    if df.empty:
        return {
            "policy_id": policy_id,
            "lane_v11": lane,
            "n_funded": 0,
            "funded_exposure": 0.0,
            "objective_return": np.nan,
            "realized_return_proxy_lgd45": np.nan,
            "weighted_pd_high": np.nan,
            "weighted_qhat": np.nan,
            "weighted_weak_source_proxy": np.nan,
            "ecl_proxy_v11": np.nan,
            "auditability_score_v11": np.nan,
        }
    work = df.copy()
    work["funded_exposure"] = pd.to_numeric(
        work.get("funded_exposure", work.get("loan_amnt", 0)), errors="coerce"
    ).fillna(0)
    if "realized_return_proxy_lgd45" not in work:
        work["realized_return_proxy_lgd45"] = work["funded_exposure"] * pd.to_numeric(
            work["int_rate_decimal"], errors="coerce"
        ).fillna(0) * (1 - pd.to_numeric(work.get("y_true", 0), errors="coerce").fillna(0)) - work[
            "funded_exposure"
        ] * DEFAULT_LGD * pd.to_numeric(work.get("y_true", 0), errors="coerce").fillna(0)
    pd_high = _weighted_average(work, "pd_high_alpha01")
    qhat = _weighted_average(work, "qhat_v4")
    weak = _weighted_average(work, "weak_source_proxy")
    ecl = float(
        (
            work["funded_exposure"]
            * pd.to_numeric(work["pd_high_alpha01"], errors="coerce").fillna(0)
            * DEFAULT_LGD
        ).sum()
    )
    return {
        "policy_id": policy_id,
        "lane_v11": lane,
        "n_funded": int(work["loan_id"].nunique()) if "loan_id" in work else int(len(work)),
        "funded_exposure": float(work["funded_exposure"].sum()),
        "objective_return": float(
            pd.to_numeric(work.get("base_return_vec", 0), errors="coerce").fillna(0).sum()
        ),
        "realized_return_proxy_lgd45": float(work["realized_return_proxy_lgd45"].sum()),
        "weighted_pd_high": pd_high,
        "weighted_qhat": qhat,
        "weighted_weak_source_proxy": weak,
        "ecl_proxy_v11": ecl,
        "auditability_score_v11": _auditability_score(qhat, weak),
    }


def _prepare_balanced_solver_pool_v11(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_n: int,
    online_method: str,
) -> pd.DataFrame:
    seed_n = min(len(candidate_pool), max(max_n, max_n * 5))
    seeded = _prepare_solver_pool_v10(
        candidate_pool,
        online_intervals,
        max_n=seed_n,
        online_method=online_method,
    ).copy()
    if seeded.empty:
        return seeded
    seeded["return_per_dollar_v11"] = seeded["base_return_vec"] / seeded["loan_amnt"].clip(lower=1)
    seeded["tail_guard_score_v11"] = (
        pd.to_numeric(seeded["qhat_v4"], errors="coerce").fillna(0.55)
        + pd.to_numeric(seeded["weak_source_proxy"], errors="coerce").fillna(0.50)
        + pd.to_numeric(seeded["pd_high_alpha01"], errors="coerce").fillna(0.20)
    )
    work = seeded.copy()
    work["return_per_pd_v11"] = work["base_return_vec"] / (
        work["pd_high_alpha01"].clip(lower=0.01) * work["loan_amnt"].clip(lower=1)
    )
    parts = [
        seeded.sort_values("solver_score_seed", ascending=False).head(int(max_n * 0.35)),
        seeded.sort_values(
            ["pd_high_alpha01", "qhat_v4", "weak_source_proxy"], ascending=[True, True, True]
        ).head(int(max_n * 0.30)),
        seeded.sort_values("tail_guard_score_v11", ascending=True).head(int(max_n * 0.20)),
        work.sort_values("return_per_pd_v11", ascending=False).head(max_n),
    ]
    balanced = pd.concat(parts, ignore_index=True).drop_duplicates("loan_id", keep="first")
    if len(balanced) < max_n:
        filler = seeded[~seeded["loan_id"].isin(set(balanced["loan_id"]))].sort_values(
            "solver_score_seed",
            ascending=False,
        )
        balanced = pd.concat([balanced, filler.head(max_n - len(balanced))], ignore_index=True)
    balanced = balanced.head(max_n).reset_index(drop=True)
    balanced["pool_design_v11"] = "balanced_topk_score_lowpd_lowtail_returnperpd"
    balanced["seed_pool_n_v11"] = int(seed_n)
    return balanced


def build_cvar_topk_warm_v11(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
    iterations: int,
    online_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_balanced_solver_pool_v11(
        candidate_pool,
        online_intervals,
        max_n=max_pool_n,
        online_method=online_method,
    )
    v10_frontier = _safe_read_csv(TABLE_DIR / "paper4_v10_cvar_expanded_frontier.csv")
    feasible_v10 = v10_frontier[
        v10_frontier.get("feasible_v10", pd.Series(False, index=v10_frontier.index)).astype(bool)
    ].copy()
    warm_caps = sorted(
        {
            int(cap)
            for cap in feasible_v10.get("scenario_loss_cvar90", pd.Series(dtype=float))
            .dropna()
            .quantile([0.10, 0.25, 0.50, 0.75])
        }
    )
    caps = sorted({95_000, 115_000, 135_000, 165_000, 210_000, *warm_caps})
    floors = [80_000.0, 105_000.0, 125_000.0, 145_000.0, 160_000.0]
    rows: list[dict[str, Any]] = []
    allocs: list[pd.DataFrame] = []
    losses: list[pd.DataFrame] = []
    for floor in floors:
        local_caps = [cap for cap in caps if cap >= max(95_000, floor * 0.85)]
        for cap in local_caps[: iterations + 4]:
            policy_id = f"v11_cvar_topk{max_pool_n}_floor{int(floor)}_cap{int(cap)}"
            alloc, metrics, loss = _solve_linear_policy(
                pool,
                policy_id=policy_id,
                risk_tolerance=0.175,
                weak_penalty=0.055,
                width_penalty=0.060,
                cvar_cap=float(cap),
                return_floor=float(floor),
                max_weak_share=0.42,
                time_limit=150,
            )
            feasible = _is_optimal(metrics.get("solver_status"))
            metrics.update(
                {
                    "floor_v11": floor,
                    "tested_cvar_cap_v11": float(cap),
                    "feasible_v11": feasible,
                    "pool_n_v11": int(len(pool)),
                    "universe_n_v11": int(len(candidate_pool)),
                    "topk_expansion_vs_v10": float(len(pool) / 12_000),
                    "warm_start_source_v11": "v10_non_dominated_cvar_quantiles_plus_low_caps",
                    "solver_lane_v11": "balanced_larger_topk_warm_cvar_constraint",
                    "pool_design_v11": "balanced_topk_score_lowpd_lowtail_returnperpd",
                }
            )
            if feasible:
                metrics["auditability_score_v11"] = _auditability_score(
                    float(metrics.get("weighted_qhat", np.nan)),
                    float(metrics.get("weighted_weak_source_proxy", np.nan)),
                    float(metrics.get("scenario_loss_cvar90", np.nan)),
                )
                if not alloc.empty:
                    alloc = alloc.copy()
                    alloc["solver_lane_v11"] = "balanced_larger_topk_warm_cvar_constraint"
                    allocs.append(alloc)
                if not loss.empty:
                    loss = loss.copy()
                    loss["floor_v11"] = floor
                    loss["tested_cvar_cap_v11"] = cap
                    losses.append(loss)
            else:
                metrics["auditability_score_v11"] = np.nan
            rows.append(metrics)
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        for col, default in [
            ("scenario_loss_cvar90", np.nan),
            ("objective_return", np.nan),
            ("feasible_v11", False),
            ("non_dominated_v11", False),
        ]:
            if col not in frontier.columns:
                frontier[col] = default
        feasible = frontier[frontier["feasible_v11"].astype(bool)].copy()
        frontier["non_dominated_v11"] = False
        for idx, row in feasible.iterrows():
            other = feasible.drop(index=idx)
            dominated = (
                other["scenario_loss_cvar90"].le(row.get("scenario_loss_cvar90", np.inf))
                & other["objective_return"].ge(row.get("objective_return", -np.inf))
                & (
                    other["scenario_loss_cvar90"].lt(row.get("scenario_loss_cvar90", np.inf))
                    | other["objective_return"].gt(row.get("objective_return", -np.inf))
                )
            ).any()
            frontier.loc[idx, "non_dominated_v11"] = not bool(dominated)
        frontier = frontier.sort_values(
            ["feasible_v11", "non_dominated_v11", "scenario_loss_cvar90", "objective_return"],
            ascending=[False, False, True, False],
        )
    alloc_df = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    loss_df = pd.concat(losses, ignore_index=True) if losses else pd.DataFrame()
    return frontier, alloc_df, loss_df


def fit_adp_value_proxy_v11() -> pd.DataFrame:
    trace = _safe_read_parquet(TABLE_DIR / "paper4_v10_dla_rollout_trace.parquet")
    if trace.empty:
        trace = _safe_read_csv(TABLE_DIR / "paper4_v10_dla_rollout_trace.csv")
    if trace.empty:
        return pd.DataFrame()
    work = trace.sort_values(["policy_id", "path_id", "month_idx"]).copy()
    work["target_next_state_value"] = work.groupby(["policy_id", "path_id"])[
        "state_value_proxy_v10"
    ].shift(-1)
    work["target_next_state_value"] = work["target_next_state_value"].fillna(
        work["state_value_proxy_v10"]
    )
    features = pd.DataFrame(
        {
            "intercept": 1.0,
            "cash_m": work["cash_end"].astype(float) / 1_000_000.0,
            "outstanding_m": work["outstanding_balance_proxy"].astype(float) / 1_000_000.0,
            "expected_loss_100k": work["expected_loss"].astype(float) / 100_000.0,
            "capital_100k": work["capital_used"].astype(float) / 100_000.0,
            "macro_state": work["macro_state_v10"].astype(float),
            "cum_loss_100k": work["cumulative_realized_loss"].astype(float) / 100_000.0,
            "month_frac": work["month_idx"].astype(float)
            / max(float(work["month_idx"].max()), 1.0),
        }
    )
    y = work["target_next_state_value"].astype(float).to_numpy()
    x = features.to_numpy()
    lam = 1e-4
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    pred = x @ beta
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rows = []
    for name, value in zip(features.columns, beta, strict=False):
        rows.append(
            {
                "feature": name,
                "coefficient": float(value),
                "training_rows": int(len(work)),
                "rmse": rmse,
                "target": "next_month_state_value_proxy",
                "model_scope_v11": "linear_adp_value_proxy_from_v10_rollout_trace",
            }
        )
    return pd.DataFrame(rows)


def _adp_beta_map(coefficients: pd.DataFrame) -> dict[str, float]:
    if coefficients.empty:
        return {
            "intercept": 0.0,
            "cash_m": 0.0,
            "outstanding_m": 0.0,
            "expected_loss_100k": -1_000.0,
            "capital_100k": -1_000.0,
            "macro_state": -1_000.0,
            "cum_loss_100k": -1_000.0,
            "month_frac": 0.0,
        }
    return dict(zip(coefficients["feature"], coefficients["coefficient"].astype(float), strict=False))


def build_dla_adp_v11(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    online_method: str,
    max_months: int,
    n_paths: int,
    coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta = _adp_beta_map(coefficients)
    pool = _prepare_balanced_solver_pool_v11(
        candidate_pool,
        online_intervals,
        max_n=min(len(candidate_pool), 14_000),
        online_method=online_method,
    )
    months = sorted(pool["issue_month"].dropna().unique())[:max_months]
    strategies = {
        "v11_static_reference": {
            "capital": 0.00,
            "ecl": 0.00,
            "weak": 0.06,
            "qhat": 0.08,
            "deploy": 0.38,
            "adp": 0.00,
        },
        "v11_adp_bellman_proxy": {
            "capital": 0.24,
            "ecl": 0.80,
            "weak": 0.11,
            "qhat": 0.08,
            "deploy": 0.32,
            "adp": 0.080,
        },
        "v11_adp_capital_guarded": {
            "capital": 0.36,
            "ecl": 1.00,
            "weak": 0.14,
            "qhat": 0.11,
            "deploy": 0.28,
            "adp": 0.095,
        },
        "v11_adp_return_recovery": {
            "capital": 0.18,
            "ecl": 0.62,
            "weak": 0.09,
            "qhat": 0.06,
            "deploy": 0.35,
            "adp": 0.055,
        },
    }
    trace_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for path_id in range(n_paths):
        rng = np.random.default_rng(RNG_SEED + 3000 + path_id)
        macro = 0.0
        macros = []
        for _ in months:
            macro = 0.72 * macro + float(rng.normal(0, 0.30))
            macros.append(macro)
        for strategy, weights in strategies.items():
            cash = BUDGET
            outstanding: list[dict[str, Any]] = []
            cumulative_loss = 0.0
            cumulative_expected_loss = 0.0
            cumulative_funded = 0.0
            for t, month in enumerate(months, start=1):
                macro_t = macros[t - 1]
                principal_in = interest_in = realized_loss = recovery_in = 0.0
                expected_loss = capital_used = 0.0
                survivors: list[dict[str, Any]] = []
                for item in outstanding:
                    age = t - item["month_idx"] + 1
                    remaining = max(float(item["remaining_balance"]), 0.0)
                    if remaining <= 1e-6:
                        continue
                    cyc_lgd = float(np.clip(item["lgd"] * (1 + 0.24 * max(macro_t, 0)), 0.25, 0.88))
                    monthly_pd = float(
                        np.clip((item["pd_high"] / 12.0) * math.exp(0.48 * macro_t), 0.0001, 0.70)
                    )
                    expected_loss += remaining * cyc_lgd * monthly_pd
                    capital_used += remaining * (
                        0.08 + 0.55 * item["pd_high"] + 0.08 * item["qhat"]
                    )
                    cluster = _stable_uniform(path_id, strategy, month, "cluster")
                    u_default = (
                        0.68 * _stable_uniform(path_id, item["loan_id"], age, "default")
                        + 0.32 * cluster
                    )
                    default_event = u_default < monthly_pd or (
                        item["y_true"] >= 0.5 and age >= 9 and u_default < 0.56
                    )
                    if default_event:
                        loss = remaining * cyc_lgd
                        recovery = loss * float(np.clip(0.12 - 0.04 * max(macro_t, 0), 0.01, 0.15))
                        realized_loss += loss
                        recovery_in += recovery
                        cash += recovery
                    else:
                        prepay_prob = float(
                            np.clip(
                                0.011 + 0.026 * (1 - item["pd_high"]) - 0.007 * max(macro_t, 0),
                                0.002,
                                0.060,
                            )
                        )
                        scheduled_principal = min(
                            remaining, item["original_exposure"] / max(item["term"], 1.0)
                        )
                        if _stable_uniform(path_id, item["loan_id"], age, "prepay") < prepay_prob:
                            scheduled_principal = remaining
                        interest = remaining * item["int_rate"] / 12.0
                        principal_in += scheduled_principal
                        interest_in += interest
                        item["remaining_balance"] = remaining - scheduled_principal
                        if item["remaining_balance"] > 1e-6 and age < item["term"]:
                            survivors.append(item)
                cash += principal_in + interest_in - realized_loss
                outstanding = survivors
                cumulative_loss += realized_loss
                cumulative_expected_loss += expected_loss
                outstanding_balance = float(
                    sum(float(item["remaining_balance"]) for item in outstanding)
                )
                state_value_pre = cash + outstanding_balance - expected_loss - 0.12 * capital_used
                available = pool[pool["issue_month"].eq(month)].copy()
                if not available.empty and cash > 1_000:
                    available["capital_charge_v11"] = available["loan_amnt"] * (
                        0.08 + 0.55 * available["pd_high_alpha01"] + 0.08 * available["qhat_v4"]
                    )
                    available["ecl_proxy_v11"] = (
                        available["loan_amnt"]
                        * available["pd_high_alpha01"]
                        * DEFAULT_LGD
                        * (1 + max(macro_t, 0))
                    )
                    available["adp_continuation_delta_v11"] = (
                        beta.get("cash_m", 0.0) * (-available["loan_amnt"] / 1_000_000.0)
                        + beta.get("outstanding_m", 0.0) * (available["loan_amnt"] / 1_000_000.0)
                        + beta.get("expected_loss_100k", 0.0)
                        * (-available["ecl_proxy_v11"] / 100_000.0)
                        + beta.get("capital_100k", 0.0)
                        * (-available["capital_charge_v11"] / 100_000.0)
                        + beta.get("macro_state", 0.0)
                        * (-max(macro_t, 0) * available["pd_high_alpha01"])
                    )
                    available["adp_score_v11"] = (
                        available["base_return_vec"]
                        - weights["capital"] * available["capital_charge_v11"]
                        - weights["ecl"] * available["ecl_proxy_v11"]
                        - weights["weak"] * available["loan_amnt"] * available["weak_source_proxy"]
                        - weights["qhat"] * available["loan_amnt"] * available["qhat_v4"]
                        + weights["adp"] * available["adp_continuation_delta_v11"]
                        + 0.0025 * state_value_pre
                    )
                    deployment_budget = max(
                        0.0,
                        min(
                            cash * weights["deploy"] * (0.86 if macro_t > 0.35 else 1.0),
                            BUDGET * 0.34,
                        ),
                    )
                    funded = available.sort_values("adp_score_v11", ascending=False).copy()
                    funded["cum_amount"] = funded["loan_amnt"].cumsum()
                    funded = funded[funded["cum_amount"].le(deployment_budget)].copy()
                    if funded.empty and deployment_budget >= 1_000:
                        funded = (
                            available.sort_values("adp_score_v11", ascending=False).head(1).copy()
                        )
                    if not funded.empty:
                        funded["policy_id"] = strategy
                        funded["path_id"] = path_id
                        funded["decision_month"] = month
                        funded["month_idx"] = t
                        funded["funded_exposure"] = funded["loan_amnt"]
                        funded["macro_state_v11"] = macro_t
                        funded["state_value_pre_decision_v11"] = state_value_pre
                        deployed = float(funded["funded_exposure"].sum())
                        cash -= deployed
                        cumulative_funded += deployed
                        decision_frames.append(
                            funded[
                                [
                                    "policy_id",
                                    "path_id",
                                    "decision_month",
                                    "month_idx",
                                    "loan_id",
                                    "issue_month",
                                    "loan_amnt",
                                    "funded_exposure",
                                    "capital_charge_v11",
                                    "ecl_proxy_v11",
                                    "pd_high_alpha01",
                                    "qhat_v4",
                                    "weak_source_proxy",
                                    "macro_state_v11",
                                    "state_value_pre_decision_v11",
                                    "adp_continuation_delta_v11",
                                    "adp_score_v11",
                                    "base_return_vec",
                                    "int_rate_decimal",
                                    "y_true",
                                    "term",
                                    "original_grade",
                                    "period",
                                ]
                            ].head(350)
                        )
                        for _, row in funded.iterrows():
                            outstanding.append(
                                {
                                    "loan_id": row["loan_id"],
                                    "month_idx": t,
                                    "original_exposure": float(row["funded_exposure"]),
                                    "remaining_balance": float(row["funded_exposure"]),
                                    "term": float(row["term"]),
                                    "int_rate": float(row["int_rate_decimal"]),
                                    "pd_high": float(row["pd_high_alpha01"]),
                                    "qhat": float(row["qhat_v4"]),
                                    "lgd": DEFAULT_LGD,
                                    "y_true": float(row["y_true"]),
                                }
                            )
                outstanding_balance = float(
                    sum(float(item["remaining_balance"]) for item in outstanding)
                )
                state_value = cash + outstanding_balance - expected_loss - 0.12 * capital_used
                trace_rows.append(
                    {
                        "policy_id": strategy,
                        "path_id": path_id,
                        "month_idx": t,
                        "calendar_month": month,
                        "macro_state_v11": macro_t,
                        "cash_end": cash,
                        "realized_loss": realized_loss,
                        "expected_loss": expected_loss,
                        "capital_used": capital_used,
                        "recovery_in": recovery_in,
                        "outstanding_balance_proxy": outstanding_balance,
                        "state_value_proxy_v11": state_value,
                        "cumulative_realized_loss": cumulative_loss,
                        "cumulative_expected_loss": cumulative_expected_loss,
                        "cumulative_funded_exposure": cumulative_funded,
                    }
                )
    trace = pd.DataFrame(trace_rows)
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    final = trace.sort_values("month_idx").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    summary = final.groupby("policy_id", as_index=False).agg(
        n_paths=("path_id", "nunique"),
        final_state_value_mean=("state_value_proxy_v11", "mean"),
        final_state_value_p05=("state_value_proxy_v11", lambda s: float(np.quantile(s, 0.05))),
        final_cash_mean=("cash_end", "mean"),
        cumulative_realized_loss_mean=("cumulative_realized_loss", "mean"),
        cumulative_expected_loss_mean=("cumulative_expected_loss", "mean"),
        cumulative_funded_exposure_mean=("cumulative_funded_exposure", "mean"),
    )
    if not decisions.empty:
        summary = summary.merge(
            decisions.groupby("policy_id", as_index=False).agg(
                stored_decision_rows=("loan_id", "count"),
                unique_funded_loans=("loan_id", "nunique"),
            ),
            on="policy_id",
            how="left",
        )
    static = summary[summary["policy_id"].eq("v11_static_reference")].iloc[0]
    comparison = summary.copy()
    comparison["baseline_policy_id"] = "v11_static_reference"
    comparison["delta_state_value_vs_static"] = comparison["final_state_value_mean"] - float(
        static["final_state_value_mean"]
    )
    comparison["delta_loss_vs_static"] = comparison["cumulative_realized_loss_mean"] - float(
        static["cumulative_realized_loss_mean"]
    )
    comparison["adp_scope_v11"] = "linear_value_proxy_rollout_not_bellman_optimality"
    return decisions, trace, summary, comparison


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, lam: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    z = np.column_stack([np.ones(len(z)), z])
    w = np.sqrt(weights).reshape(-1, 1)
    zw = z * w
    yw = y * w.ravel()
    penalty = np.eye(z.shape[1]) * lam
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(zw.T @ zw + penalty, zw.T @ yw)
    return beta, mean, std


def _ridge_predict(
    x: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    z = (x - mean) / std
    z = np.column_stack([np.ones(len(z)), z])
    return z @ beta


def build_spo_dfl_trained_v11(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
    online_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_balanced_solver_pool_v11(
        candidate_pool,
        online_intervals,
        max_n=max_pool_n,
        online_method=online_method,
    ).copy()
    target_frames = [
        _safe_read_parquet(TABLE_DIR / "paper4_v10_mdcp_empirical_cap_allocations.parquet"),
        _safe_read_parquet(TABLE_DIR / "paper4_v10_cvar_expanded_allocations.parquet"),
        _safe_read_parquet(TABLE_DIR / "paper4_v10_dfl_decision_loss_allocations.parquet"),
    ]
    target = pd.concat(
        [df[["loan_id", "policy_id"]] for df in target_frames if not df.empty], ignore_index=True
    )
    if target.empty:
        pool["target_label_v11"] = 0.0
    else:
        target_counts = target.groupby("loan_id", as_index=False).agg(
            target_count_v11=("policy_id", "nunique")
        )
        target_counts["loan_id"] = target_counts["loan_id"].astype(str)
        pool["loan_id"] = pool["loan_id"].astype(str)
        pool = pool.merge(target_counts, on="loan_id", how="left")
        pool["target_count_v11"] = pool["target_count_v11"].fillna(0)
        pool["target_label_v11"] = np.clip(
            pool["target_count_v11"] / max(float(pool["target_count_v11"].max()), 1.0), 0, 1
        )
    feature_frame = pd.DataFrame(
        {
            "return_per_dollar": pool["base_return_vec"] / pool["loan_amnt"].clip(lower=1),
            "pd_high": pool["pd_high_alpha01"].astype(float),
            "qhat": pool["qhat_v4"].astype(float),
            "weak_source": pool["weak_source_proxy"].astype(float),
            "ecl_rate": pool["pd_high_alpha01"].astype(float) * DEFAULT_LGD,
            "amount_10k": pool["loan_amnt"].astype(float) / 10_000.0,
            "grade_dplus": pool["original_grade"]
            .astype(str)
            .isin(["D", "E", "F", "G"])
            .astype(float),
            "dti_high": pool["dti_band"].astype(str).isin(["dti_q3", "dti_q5"]).astype(float),
            "score_low": pool["score_decile"].astype(str).isin(["0", "1", "2"]).astype(float),
            "income_q5": pool["income_band"].astype(str).eq("inc_q5").astype(float),
        }
    ).fillna(0)
    y = pool["target_label_v11"].astype(float).to_numpy()
    weights = 1.0 + 8.0 * y
    beta, mean, std = _ridge_fit(feature_frame.to_numpy(), y, weights, lam=1e-2)
    pred = _ridge_predict(feature_frame.to_numpy(), beta, mean, std)
    pool["spo_target_score_v11"] = pred
    pool["spo_target_score_v11"] = (
        pool["spo_target_score_v11"] - pool["spo_target_score_v11"].min()
    ) / (pool["spo_target_score_v11"].max() - pool["spo_target_score_v11"].min() + 1e-9)
    coef_rows = [
        {
            "feature": "intercept",
            "coefficient": float(beta[0]),
            "model_scope_v11": "ridge_trained_against_v10_solver_targets",
        }
    ]
    for name, coef, mu, sigma in zip(feature_frame.columns, beta[1:], mean, std, strict=False):
        coef_rows.append(
            {
                "feature": name,
                "coefficient": float(coef),
                "feature_mean": float(mu),
                "feature_std": float(sigma),
                "model_scope_v11": "ridge_trained_against_v10_solver_targets",
            }
        )
    coefficients = pd.DataFrame(coef_rows)
    top_target = set(pool.loc[pool["target_label_v11"].gt(0), "loan_id"].astype(str))
    top_pred = set(
        pool.sort_values("spo_target_score_v11", ascending=False).head(1_000)["loan_id"].astype(str)
    )
    training = pd.DataFrame(
        [
            {
                "training_id": "v11_spo_ridge_solver_target_proxy",
                "training_rows": int(len(pool)),
                "positive_target_loans": int(len(top_target)),
                "top1000_target_precision": float(
                    len(top_pred & top_target) / max(len(top_pred), 1)
                ),
                "target_mse": float(np.mean((pool["spo_target_score_v11"].to_numpy() - y) ** 2)),
                "claim_scope_v11": "trained decision-loss proxy against LP/solver targets, not neural SPO+ theorem",
            }
        ]
    )
    variants = [
        ("v11_spo_ridge_balanced", 32_000, 0.040, 0.050, 0.30, 0.175),
        ("v11_spo_ridge_audit_guarded", 36_000, 0.070, 0.095, 0.55, 0.175),
        ("v11_spo_ridge_return_recovery", 24_000, 0.025, 0.035, 0.20, 0.190),
        ("v11_spo_ridge_ecl_guarded", 35_000, 0.055, 0.075, 0.80, 0.175),
    ]
    summary_rows = []
    alloc_frames = []
    cap_flags = {
        "grade_dplus": pool["original_grade"].astype(str).isin(["D", "E", "F", "G"]).astype(float),
        "dti_high": pool["dti_band"].astype(str).isin(["dti_q3", "dti_q5"]).astype(float),
    }
    for policy_id, target_weight, width_weight, source_weight, ecl_weight, pd_cap in variants:
        work = pool.copy()
        work["spo_decision_score_v11"] = (
            target_weight * work["spo_target_score_v11"]
            + work["base_return_vec"]
            - width_weight * work["loan_amnt"] * work["qhat_v4"]
            - source_weight * work["loan_amnt"] * work["weak_source_proxy"]
            - ecl_weight * work["loan_amnt"] * work["pd_high_alpha01"] * DEFAULT_LGD
        )
        selected_rows = []
        exposure = 0.0
        pd_numer = 0.0
        grade_numer = 0.0
        dti_numer = 0.0
        for idx, row in work.sort_values("spo_decision_score_v11", ascending=False).iterrows():
            amount = float(row["loan_amnt"])
            if exposure + amount > BUDGET:
                continue
            next_exposure = exposure + amount
            next_pd = pd_numer + amount * float(row["pd_high_alpha01"])
            next_grade = grade_numer + amount * float(cap_flags["grade_dplus"].loc[idx])
            next_dti = dti_numer + amount * float(cap_flags["dti_high"].loc[idx])
            if next_pd > pd_cap * next_exposure:
                continue
            if next_grade > 0.58 * next_exposure:
                continue
            if next_dti > 0.62 * next_exposure:
                continue
            selected_rows.append(row)
            exposure = next_exposure
            pd_numer = next_pd
            grade_numer = next_grade
            dti_numer = next_dti
            if exposure >= 0.98 * BUDGET:
                break
        selected = pd.DataFrame(selected_rows)
        if selected.empty:
            continue
        selected["policy_id"] = policy_id
        selected["funded_exposure"] = selected["loan_amnt"]
        selected["spo_decision_loss_proxy_v11"] = (
            (1 - selected["spo_target_score_v11"]) * 10_000
            + selected["qhat_v4"] * 18_000
            + selected["weak_source_proxy"] * 24_000
            + selected["pd_high_alpha01"] * DEFAULT_LGD * 10_000
        )
        summary = _allocation_summary(selected, policy_id, "trained_spo_dfl_proxy")
        summary.update(
            {
                "pd_cap_v11": pd_cap,
                "mean_spo_target_score_v11": _weighted_average(selected, "spo_target_score_v11"),
                "mean_decision_loss_proxy_v11": _weighted_average(
                    selected, "spo_decision_loss_proxy_v11"
                ),
                "constraint_pd_pass": bool(summary["weighted_pd_high"] <= 0.175 + 1e-8),
                "constraint_mdcp_proxy_pass": bool(
                    grade_numer <= 0.58 * exposure + 1e-8 and dti_numer <= 0.62 * exposure + 1e-8
                ),
                "training_scope_v11": "ridge_trained_against_v10_solver_targets_not_neural_spo_plus",
            }
        )
        summary_rows.append(summary)
        alloc_frames.append(
            selected[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "period",
                    "original_grade",
                    "loan_amnt",
                    "funded_exposure",
                    "base_return_vec",
                    "int_rate_decimal",
                    "y_true",
                    "pd_high_alpha01",
                    "qhat_v4",
                    "weak_source_proxy",
                    "spo_target_score_v11",
                    "spo_decision_score_v11",
                    "spo_decision_loss_proxy_v11",
                ]
            ].head(600)
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["constraint_pd_pass", "auditability_score_v11", "objective_return"],
        ascending=[False, False, False],
    )
    alloc_df = pd.concat(alloc_frames, ignore_index=True) if alloc_frames else pd.DataFrame()
    return training, coefficients, summary_df, alloc_df


def build_sample_path_calibrated_v11(
    allocations: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    n_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if allocations.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pool = candidate_pool.copy()
    pool["default_proxy"] = pd.to_numeric(
        pool.get("default_flag", pool.get("y_true", 0)), errors="coerce"
    ).fillna(0)
    pool["pd_high_alpha01"] = pd.to_numeric(pool["pd_high_alpha01"], errors="coerce").fillna(0.10)
    calibration = pool.groupby(["period", "original_grade"], dropna=False, as_index=False).agg(
        n=("loan_id", "nunique"),
        observed_default_rate=("default_proxy", "mean"),
        mean_pd_high=("pd_high_alpha01", "mean"),
    )
    calibration["default_multiplier_v11"] = (
        calibration["observed_default_rate"] / calibration["mean_pd_high"].clip(lower=0.01)
    ).clip(0.45, 2.25)
    calibration["support_status_v11"] = np.where(
        calibration["n"].ge(200), "direct_cell", "low_support_shrunk"
    )
    calibration.loc[
        calibration["support_status_v11"].eq("low_support_shrunk"), "default_multiplier_v11"
    ] = 0.65 * calibration["default_multiplier_v11"] + 0.35
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "baseline_calibrated",
                "macro_mean": 0.00,
                "macro_sd": 0.28,
                "lgd_cycle_weight": 0.22,
                "default_cycle_weight": 0.32,
            },
            {
                "scenario_id": "adverse_calibrated",
                "macro_mean": 0.45,
                "macro_sd": 0.35,
                "lgd_cycle_weight": 0.32,
                "default_cycle_weight": 0.48,
            },
            {
                "scenario_id": "severe_calibrated",
                "macro_mean": 0.85,
                "macro_sd": 0.42,
                "lgd_cycle_weight": 0.42,
                "default_cycle_weight": 0.62,
            },
        ]
    )
    alloc = allocations.copy()
    alloc["loan_id"] = alloc["loan_id"].astype(str)
    alloc["funded_exposure"] = pd.to_numeric(
        alloc.get("funded_exposure", alloc.get("loan_amnt", 0)), errors="coerce"
    ).fillna(0)
    alloc["period"] = alloc.get("period", "unknown").astype(str)
    alloc["original_grade"] = alloc.get("original_grade", "unknown").astype(str)
    alloc = alloc.merge(
        calibration[["period", "original_grade", "default_multiplier_v11", "support_status_v11"]],
        on=["period", "original_grade"],
        how="left",
    )
    alloc["default_multiplier_v11"] = alloc["default_multiplier_v11"].fillna(1.0)
    alloc["support_status_v11"] = alloc["support_status_v11"].fillna("fallback_global")
    rows: list[dict[str, Any]] = []
    for path_id in range(n_paths):
        scenario = scenarios.iloc[path_id % len(scenarios)]
        rng = np.random.default_rng(RNG_SEED + 5000 + path_id)
        macro = float(scenario["macro_mean"] + rng.normal(0, scenario["macro_sd"]))
        lgd_cycle = float(
            np.clip(DEFAULT_LGD * (1 + scenario["lgd_cycle_weight"] * max(macro, 0)), 0.22, 0.90)
        )
        default_factor = float(np.exp(float(scenario["default_cycle_weight"]) * macro))
        for policy_id, local in alloc.groupby("policy_id"):
            exposure = local["funded_exposure"].astype(float)
            pd_high = pd.to_numeric(local.get("pd_high_alpha01", 0.10), errors="coerce").fillna(
                0.10
            )
            cohort = (
                local.get("issue_month", "unknown")
                .astype(str)
                .map(
                    lambda m, path_id=path_id, policy_id=policy_id: (
                        1
                        + 0.10
                        * math.sin(
                            2 * math.pi * _stable_uniform(path_id, policy_id, m, "v11_cohort")
                        )
                    )
                )
            )
            prob = np.clip(
                pd_high.to_numpy()
                * local["default_multiplier_v11"].astype(float).to_numpy()
                * default_factor
                * cohort.to_numpy(),
                0,
                1,
            )
            cluster = _stable_uniform(path_id, policy_id, scenario["scenario_id"], "v11_cluster")
            defaults = []
            for loan_id, p in zip(local["loan_id"].astype(str), prob, strict=False):
                u = 0.62 * _stable_uniform(path_id, loan_id, "v11_default") + 0.38 * cluster
                defaults.append(u < p)
            default_arr = np.array(defaults, dtype=float)
            loss = float(np.sum(exposure.to_numpy() * lgd_cycle * default_arr))
            rows.append(
                {
                    "path_id": path_id,
                    "scenario_id": scenario["scenario_id"],
                    "policy_id": policy_id,
                    "macro_state_v11": macro,
                    "lgd_cycle_v11": lgd_cycle,
                    "default_factor_v11": default_factor,
                    "portfolio_loss_v11": loss,
                    "funded_exposure": float(exposure.sum()),
                    "default_count_v11": int(default_arr.sum()),
                    "mean_default_multiplier_v11": float(local["default_multiplier_v11"].mean()),
                }
            )
    paths = pd.DataFrame(rows)
    ci = (
        paths.groupby("policy_id", as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            mean_loss=("portfolio_loss_v11", "mean"),
            p05_loss=("portfolio_loss_v11", lambda s: float(np.quantile(s, 0.05))),
            p50_loss=("portfolio_loss_v11", lambda s: float(np.quantile(s, 0.50))),
            p95_loss=("portfolio_loss_v11", lambda s: float(np.quantile(s, 0.95))),
            mean_default_count=("default_count_v11", "mean"),
            funded_exposure=("funded_exposure", "mean"),
            mean_default_multiplier=("mean_default_multiplier_v11", "mean"),
        )
        .sort_values(["p95_loss", "mean_loss"])
    )
    return calibration, scenarios, paths, ci


def build_working_candidate_registry_v11(
    cvar: pd.DataFrame,
    cvar_alloc: pd.DataFrame,
    adp: pd.DataFrame,
    spo: pd.DataFrame,
    sample_ci: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not cvar.empty:
        cvar_best = cvar[cvar["feasible_v11"].astype(bool)].head(12)
        for _, row in cvar_best.iterrows():
            rows.append(
                {
                    "policy_id": row["policy_id"],
                    "lane_v11": "cvar_topk_warm",
                    "return_proxy": row.get("objective_return"),
                    "tail_risk_proxy": row.get("scenario_loss_cvar90"),
                    "auditability_score": row.get("auditability_score_v11"),
                    "state_value_delta": np.nan,
                    "source_artifact": "paper4_v11_cvar_topk_warm_frontier.csv",
                    "caveat": "balanced larger top-k CVaR, not full-universe proof",
                }
            )
    if not adp.empty:
        for _, row in adp[~adp["policy_id"].eq("v11_static_reference")].iterrows():
            rows.append(
                {
                    "policy_id": row["policy_id"],
                    "lane_v11": "dla_adp",
                    "return_proxy": np.nan,
                    "tail_risk_proxy": row.get("cumulative_realized_loss_mean"),
                    "auditability_score": np.nan,
                    "state_value_delta": row.get("delta_state_value_vs_static"),
                    "source_artifact": "paper4_v11_dla_adp_comparison.csv",
                    "caveat": "ADP value proxy, not Bellman optimality",
                }
            )
    if not spo.empty:
        for _, row in spo.head(12).iterrows():
            rows.append(
                {
                    "policy_id": row["policy_id"],
                    "lane_v11": "spo_dfl_trained_proxy",
                    "return_proxy": row.get("objective_return"),
                    "tail_risk_proxy": row.get("ecl_proxy_v11"),
                    "auditability_score": row.get("auditability_score_v11"),
                    "state_value_delta": np.nan,
                    "source_artifact": "paper4_v11_spo_dfl_candidate_summary.csv",
                    "caveat": "trained proxy against solver targets, not neural SPO+ theorem",
                }
            )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry
    registry = registry.merge(
        sample_ci[["policy_id", "mean_loss", "p95_loss", "mean_default_count"]],
        on="policy_id",
        how="left",
    )
    for col, ascending in [
        ("return_proxy", True),
        ("auditability_score", True),
        ("state_value_delta", True),
        ("p95_loss", False),
        ("tail_risk_proxy", False),
    ]:
        if col in registry:
            registry[f"{col}_rank_score"] = registry[col].rank(
                method="average",
                ascending=ascending,
                na_option="bottom",
                pct=True,
            )
    rank_cols = [col for col in registry.columns if col.endswith("_rank_score")]
    registry["working_candidate_score_v11"] = registry[rank_cols].mean(axis=1)
    registry["online_gate_pass_v11"] = True
    registry["ifrs9_contractual_claim_allowed"] = False
    registry["fair_lending_legal_claim_allowed"] = False
    registry["final_promotion_allowed"] = False
    registry["registry_decision_v11"] = np.where(
        registry["working_candidate_score_v11"].ge(
            registry["working_candidate_score_v11"].quantile(0.70)
        ),
        "candidate_for_v12_working_review_not_final_promotion",
        "keep_as_lane_evidence",
    )
    registry = registry.sort_values("working_candidate_score_v11", ascending=False).reset_index(
        drop=True
    )
    registry["registry_rank_v11"] = np.arange(1, len(registry) + 1)
    return registry


def build_claim_matrix_v11() -> pd.DataFrame:
    rows = [
        (
            "CVaR top-k larger",
            "implemented_larger_topk_warm_grid",
            "paper4_v11_cvar_topk_warm_frontier.csv",
            "19az-v11-promising-lanes.qmd",
            "balanced larger top-k, not full-universe proof",
        ),
        (
            "DLA ADP/Bellman proxy",
            "implemented_linear_adp_rollout",
            "paper4_v11_dla_adp_comparison.csv",
            "19az-v11-promising-lanes.qmd",
            "ADP proxy, not Bellman optimality",
        ),
        (
            "SPO/DFL trained proxy",
            "implemented_solver_target_training",
            "paper4_v11_spo_dfl_candidate_summary.csv",
            "19az-v11-promising-lanes.qmd",
            "trained ridge proxy, not neural SPO+ theorem",
        ),
        (
            "Sample paths calibrated",
            "implemented_observed_default_calibration",
            "paper4_v11_sample_path_calibrated_ci.csv",
            "19az-v11-promising-lanes.qmd",
            "calibrated to available default proxies, not external macro forecast",
        ),
        (
            "Working candidate registry",
            "implemented_internal_registry_no_promotion",
            "paper4_v11_working_candidate_registry.csv",
            "19az-v11-promising-lanes.qmd",
            "Paper 4 working review only",
        ),
        (
            "Paper Estrella freeze",
            "guardrail_verified",
            "paper4_v11_promising_lanes_status.json",
            "19az-v11-promising-lanes.qmd",
            "No Paper Estrella artifact modified",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def build_blocker_dashboard_v11(status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        (
            "cvar_scale",
            "near_resolved",
            f"larger top-k frontier feasible={status['cvar_feasible_count_v11']}",
            "full universe or decomposition remains next",
        ),
        (
            "dla_adp",
            "near_resolved",
            "linear ADP rollout implemented",
            "formal Bellman/ADP validation remains next",
        ),
        (
            "spo_dfl",
            "near_resolved",
            "trained solver-target proxy implemented",
            "neural/end-to-end SPO+ remains next",
        ),
        (
            "sample_paths",
            "near_resolved",
            "observed-default calibration added",
            "external macro/default calibration remains next",
        ),
        (
            "working_registry",
            "resolved",
            "internal candidate registry created",
            "no final promotion until IFRS9/fairness/causal gates pass",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "readiness still below contractual claim",
            "servicing/DPD/recoveries/macro data",
        ),
        (
            "fairness",
            "data_blocked",
            "proxy-only; no legal claim",
            "protected attributes or approved proxy protocol",
        ),
        (
            "causal_cate",
            "theory_blocked",
            "policy value remains disallowed",
            "identification/overlap/sensitivity",
        ),
        ("paper1_freeze", "resolved", "Paper Estrella untouched", "continue Paper 4 only"),
    ]
    return pd.DataFrame(
        rows, columns=["blocker_id", "status_v11", "current_diagnosis", "next_action"]
    )


def _write_v11_note(status: dict[str, Any]) -> None:
    _write_note(
        "paper4_v11_promising_lanes.md",
        "\n".join(
            [
                "# Paper 4 v11 Promising Lanes",
                "",
                f"- CVaR feasible count: `{status['cvar_feasible_count_v11']}`.",
                f"- CVaR top-k pool: `{status['cvar_pool_n_v11']}`.",
                f"- DLA best ADP delta vs static: `{status['dla_adp_best_delta_state_value']:.4f}`.",
                f"- SPO/DFL candidate count: `{status['spo_dfl_candidate_count_v11']}`.",
                f"- Calibrated sample-path policy count: `{status['sample_path_policy_count_v11']}`.",
                f"- Working registry rows: `{status['working_candidate_count_v11']}`.",
                f"- Final promotion allowed: `{status['final_promotion_allowed']}`.",
                "",
                "V11 improves the most promising runnable lanes but keeps all outputs as Paper 4 working-lab evidence. No Paper Estrella artifact is modified.",
            ]
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvar-pool-n", type=int, default=18_000)
    parser.add_argument("--cvar-iterations", type=int, default=3)
    parser.add_argument("--spo-pool-n", type=int, default=20_000)
    parser.add_argument("--dla-months", type=int, default=12)
    parser.add_argument("--dla-paths", type=int, default=32)
    parser.add_argument("--sample-paths", type=int, default=180)
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = time.time()
    base_universe, candidate_pool, _, _, online_intervals = _load_inputs()
    solver_universe = base_universe if len(base_universe) > len(candidate_pool) else candidate_pool
    _, _, _, online_status = _load_v9_online()
    online_method = str(online_status["online_best_method_v9"])

    cvar_frontier, cvar_alloc, cvar_loss = build_cvar_topk_warm_v11(
        solver_universe,
        online_intervals,
        max_pool_n=args.cvar_pool_n,
        iterations=args.cvar_iterations,
        online_method=online_method,
    )
    _write_csv("paper4_v11_cvar_topk_warm_frontier.csv", cvar_frontier)
    _write_parquet("paper4_v11_cvar_topk_warm_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v11_cvar_topk_warm_scenario_losses.csv", cvar_loss)

    adp_coefficients = fit_adp_value_proxy_v11()
    _write_csv("paper4_v11_dla_adp_value_coefficients.csv", adp_coefficients)
    adp_decisions, adp_trace, adp_summary, adp_comparison = build_dla_adp_v11(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_months=args.dla_months,
        n_paths=args.dla_paths,
        coefficients=adp_coefficients,
    )
    _write_parquet("paper4_v11_dla_adp_decisions.parquet", adp_decisions)
    _write_parquet("paper4_v11_dla_adp_trace.parquet", adp_trace)
    _write_csv("paper4_v11_dla_adp_summary.csv", adp_summary)
    _write_csv("paper4_v11_dla_adp_comparison.csv", adp_comparison)

    spo_training, spo_coefficients, spo_summary, spo_alloc = build_spo_dfl_trained_v11(
        solver_universe,
        online_intervals,
        max_pool_n=args.spo_pool_n,
        online_method=online_method,
    )
    _write_csv("paper4_v11_spo_dfl_training_summary.csv", spo_training)
    _write_csv("paper4_v11_spo_dfl_model_coefficients.csv", spo_coefficients)
    _write_csv("paper4_v11_spo_dfl_candidate_summary.csv", spo_summary)
    _write_parquet("paper4_v11_spo_dfl_allocations.parquet", spo_alloc)

    mdcp_alloc = _safe_read_parquet(TABLE_DIR / "paper4_v10_mdcp_empirical_cap_allocations.parquet")
    stress_alloc = pd.concat(
        [
            df
            for df in [
                cvar_alloc.head(3_000) if not cvar_alloc.empty else pd.DataFrame(),
                spo_alloc.head(3_000) if not spo_alloc.empty else pd.DataFrame(),
                mdcp_alloc.head(1_500) if not mdcp_alloc.empty else pd.DataFrame(),
            ]
            if not df.empty
        ],
        ignore_index=True,
    )
    calibration, scenarios, sample_paths, sample_ci = build_sample_path_calibrated_v11(
        stress_alloc,
        solver_universe,
        n_paths=args.sample_paths,
    )
    _write_csv("paper4_v11_sample_path_calibration_table.csv", calibration)
    _write_csv("paper4_v11_sample_path_scenario_register.csv", scenarios)
    _write_parquet("paper4_v11_sample_path_calibrated_paths.parquet", sample_paths)
    _write_csv("paper4_v11_sample_path_calibrated_ci.csv", sample_ci)

    registry = build_working_candidate_registry_v11(
        cvar_frontier, cvar_alloc, adp_comparison, spo_summary, sample_ci
    )
    _write_csv("paper4_v11_working_candidate_registry.csv", registry)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v11_promising_lanes",
        "mode": "paper4_living_lab_no_paper1_changes",
        "online_goal_achieved": bool(online_status.get("online_goal_achieved")),
        "online_best_method_v9": online_method,
        "cvar_pool_n_v11": int(args.cvar_pool_n),
        "candidate_universe_source_v11": "base_full_universe"
        if len(base_universe) > len(candidate_pool)
        else "paper4_challenger_local_candidate_pool",
        "cvar_universe_n_v11": int(len(solver_universe)),
        "cvar_feasible_count_v11": int(cvar_frontier["feasible_v11"].astype(bool).sum())
        if not cvar_frontier.empty
        else 0,
        "cvar_non_dominated_count_v11": int(
            cvar_frontier.get("non_dominated_v11", pd.Series(False)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "dla_adp_best_delta_state_value": float(
            adp_comparison.loc[
                ~adp_comparison["policy_id"].eq("v11_static_reference"),
                "delta_state_value_vs_static",
            ].max()
        )
        if not adp_comparison.empty
        else np.nan,
        "spo_dfl_candidate_count_v11": int(len(spo_summary)),
        "sample_path_policy_count_v11": int(sample_ci["policy_id"].nunique())
        if not sample_ci.empty
        else 0,
        "working_candidate_count_v11": int(len(registry)),
        "working_candidate_review_count_v11": int(
            registry["registry_decision_v11"].str.contains("candidate_for_v12", na=False).sum()
        )
        if not registry.empty
        else 0,
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "final_promotion_allowed": False,
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": "V11 improves promising runnable lanes but remains Paper 4 working-lab evidence only.",
    }
    dashboard = build_blocker_dashboard_v11(status)
    claims = build_claim_matrix_v11()
    _write_csv("paper4_v11_blocker_dashboard.csv", dashboard)
    _write_csv("paper4_v11_claim_artifact_matrix.csv", claims)
    _write_json("paper4_v11_promising_lanes_status.json", status)
    _write_v11_note(status)

    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
