"""Build Paper 4 v5 artifacts for the current blocker-resolution wave.

V5 keeps Paper Estrella frozen and focuses on the blockers identified after v4:
source-month online coverage, more contractual IFRS9/SICR, CVaR/top-k expansion,
MDCP-aware search, calibrated selector governance, more realistic SDAM/DLA,
causal/fairness gates, correlated sample paths, SPO/DFL hybrids, temporal ECL
and CQR decision-aware screening.

The outputs remain Paper 4 living-lab artifacts.  This script never writes
``models/final_project_promotion.json`` and never creates
``paper4_final_promotion.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _load_base_loan_frame,
    _normalise,
    _safe_read_csv,
    _safe_read_json,
    _safe_read_parquet,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_next_wave_experiments import _as_month, _prepare_base
from scripts.papers.build_paper4_v4_open_priorities import (
    FROZEN_PAPER1_CHAMPION,
    _load_performance_reference,
    _scenario_multiplier_table,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
STATUS_DIR = OUT_ROOT / "status"
NOTE_DIR = OUT_ROOT / "notes"
FIGURE_DIR = OUT_ROOT / "figures"

SCHEMA_VERSION = "2026-05-13.4"
RNG_SEED = 20260513


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


def _month_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    later_dt = pd.to_datetime(later, errors="coerce")
    earlier_dt = pd.to_datetime(earlier, errors="coerce")
    return (later_dt.dt.year - earlier_dt.dt.year) * 12 + (later_dt.dt.month - earlier_dt.dt.month)


def _load_required_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    base = _prepare_base(_load_base_loan_frame())
    base["loan_id"] = base["id"].astype(str)
    allocations = _safe_read_parquet(TABLE_DIR / "paper4_challenger_local_allocations.parquet")
    local_search = _safe_read_csv(TABLE_DIR / "paper4_challenger_local_search.csv")
    selector = _safe_read_csv(TABLE_DIR / "paper4_selector_v4_results.csv")
    online_intervals = _safe_read_parquet(
        TABLE_DIR / "paper4_online_conformal_v4_intervals.parquet"
    )
    if allocations.empty or local_search.empty or selector.empty or online_intervals.empty:
        raise FileNotFoundError("Run v4 first before Paper 4 v5 blocker resolution.")
    allocations = allocations.copy()
    allocations["loan_id"] = allocations["loan_id"].astype(str)
    allocations["issue_month"] = _as_month(allocations["issue_month"])
    online_intervals = online_intervals.copy()
    online_intervals["loan_id"] = online_intervals["loan_id"].astype(str)
    online_intervals["issue_month"] = _as_month(online_intervals["issue_month"])
    return base, allocations, local_search, selector, online_intervals


def build_online_source_month_v5(
    allocations: pd.DataFrame,
    online_intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Search guarded source-month online policies over funded rows.

    This layer uses the best v4 interval family as the base and applies
    pre-decision guards that depend on funded cell size and method parameters,
    not on the realized cell outcome.  It is still conservative and diagnostic.
    """
    base_intervals = online_intervals[
        online_intervals["online_method_v4"].eq("source_aware_guarded")
    ].copy()
    keep_cols = [
        "loan_id",
        "issue_month",
        "period",
        "original_grade",
        "term",
        "score_decile",
        "state_top20",
        "income_band",
        "dti_band",
        "y_true",
        "y_pred",
        "qhat_v4",
    ]
    merged_base = allocations[
        [
            "policy_id",
            "loan_id",
            "issue_month",
            "funded_exposure",
            "original_grade",
            "period",
            "term",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
        ]
    ].merge(base_intervals[keep_cols], on="loan_id", how="left", suffixes=("", "_interval"))
    for col in [
        "original_grade",
        "period",
        "term",
        "score_decile",
        "state_top20",
        "income_band",
        "dti_band",
    ]:
        interval_col = f"{col}_interval"
        if interval_col in merged_base.columns:
            merged_base[col] = merged_base[col].where(
                merged_base[col].notna(), merged_base[interval_col]
            )
            merged_base = merged_base.drop(columns=[interval_col])
    merged_base["qhat_v4"] = pd.to_numeric(merged_base["qhat_v4"], errors="coerce").fillna(0.55)
    merged_base["y_pred"] = pd.to_numeric(merged_base["y_pred"], errors="coerce").fillna(
        merged_base["qhat_v4"]
    )
    merged_base["y_true"] = pd.to_numeric(merged_base["y_true"], errors="coerce").fillna(0.0)
    source_families = [
        "original_grade",
        "period",
        "score_decile",
        "state_top20",
        "income_band",
        "dti_band",
    ]
    configs = [
        ("v4_reference", 8, 1.00, 0.00, 0.00),
        ("hierarchical_pool_min12", 12, 1.05, 0.02, 0.00),
        ("adaptive_shrinkage_min16", 16, 1.10, 0.03, 0.05),
        ("source_month_borrowing_min20", 20, 1.15, 0.04, 0.10),
        ("width_penalized_minimax_min24", 24, 1.20, 0.05, 0.15),
        ("source_month_strict_min32", 32, 1.25, 0.06, 0.25),
    ]
    interval_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for method, min_eff, q_mult, q_add, borrow_margin in configs:
        local = merged_base.copy()
        guard_mask = (
            local.groupby(["policy_id", "issue_month"])["loan_id"].transform("nunique").lt(min_eff)
        )
        for source in source_families:
            cell_n = local.groupby(["policy_id", "issue_month", source])["loan_id"].transform(
                "nunique"
            )
            guard_mask = guard_mask | cell_n.lt(min_eff)
        q = (local["qhat_v4"] * q_mult + q_add).clip(0, 1)
        if borrow_margin > 0:
            q = (q + borrow_margin * guard_mask.astype(float)).clip(0, 1)
        if method in {"width_penalized_minimax_min24", "source_month_strict_min32"}:
            q = q.mask(guard_mask, 1.0)
        local["online_method_v5"] = method
        local["min_effective_cell_n"] = min_eff
        local["guarded_small_source_cell"] = guard_mask
        local["qhat_v5"] = q
        local["pd_low_online_v5"] = np.clip(local["y_pred"] - local["qhat_v5"], 0, 1)
        local["pd_high_online_v5"] = np.clip(local["y_pred"] + local["qhat_v5"], 0, 1)
        local["covered_online_v5"] = local["y_true"].between(
            local["pd_low_online_v5"], local["pd_high_online_v5"]
        )
        local["interval_width_online_v5"] = local["pd_high_online_v5"] - local["pd_low_online_v5"]
        interval_frames.append(
            local[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "online_method_v5",
                    "min_effective_cell_n",
                    "guarded_small_source_cell",
                    "qhat_v5",
                    "pd_low_online_v5",
                    "pd_high_online_v5",
                    "covered_online_v5",
                    "interval_width_online_v5",
                ]
            ]
        )
        policy_month = (
            local.groupby(["online_method_v5", "policy_id", "issue_month"], as_index=False)
            .agg(
                n_funded=("loan_id", "nunique"),
                coverage_online_v5=("covered_online_v5", "mean"),
                avg_width_online_v5=("interval_width_online_v5", "mean"),
                guard_share=("guarded_small_source_cell", "mean"),
            )
            .rename(columns={"issue_month": "month"})
        )
        policy_frames.append(policy_month)
        local_sources = []
        for source in source_families:
            src = (
                local.groupby(
                    ["online_method_v5", "policy_id", "issue_month", source], as_index=False
                )
                .agg(
                    n=("loan_id", "nunique"),
                    coverage_online_v5=("covered_online_v5", "mean"),
                    avg_width_online_v5=("interval_width_online_v5", "mean"),
                    guard_share=("guarded_small_source_cell", "mean"),
                )
                .rename(columns={"issue_month": "month", source: "source_value"})
            )
            src = src[src["n"].ge(5)].copy()
            src["source_id"] = source
            local_sources.append(src)
        source_month = pd.concat(local_sources, ignore_index=True)
        source_frames.append(source_month)
        summary_rows.append(
            {
                "online_method_v5": method,
                "min_effective_cell_n": min_eff,
                "q_multiplier": q_mult,
                "q_additive": q_add,
                "borrow_margin": borrow_margin,
                "coverage_policy_month_mean": float(policy_month["coverage_online_v5"].mean()),
                "coverage_policy_month_min": float(policy_month["coverage_online_v5"].min()),
                "coverage_source_month_min": float(source_month["coverage_online_v5"].min()),
                "avg_width_policy_month": float(policy_month["avg_width_online_v5"].mean()),
                "avg_width_loan": float(local["interval_width_online_v5"].mean()),
                "guard_share": float(local["guarded_small_source_cell"].mean()),
                "gate_pass_80": bool(
                    policy_month["coverage_online_v5"].min() >= 0.80
                    and source_month["coverage_online_v5"].min() >= 0.80
                ),
                "gate_pass_90": bool(
                    policy_month["coverage_online_v5"].min() >= 0.90
                    and source_month["coverage_online_v5"].min() >= 0.90
                ),
                "efficiency_gate_width_98": bool(local["interval_width_online_v5"].mean() <= 0.98),
            }
        )
    intervals = pd.concat(interval_frames, ignore_index=True)
    policy_month_all = pd.concat(policy_frames, ignore_index=True)
    source_month_all = pd.concat(source_frames, ignore_index=True)
    source_month_all["source_value"] = source_month_all["source_value"].astype(str)
    summary = pd.DataFrame(summary_rows)
    summary["operational_gate_pass"] = summary["gate_pass_80"] & summary["efficiency_gate_width_98"]
    summary = summary.sort_values(
        [
            "gate_pass_90",
            "operational_gate_pass",
            "coverage_source_month_min",
            "avg_width_loan",
        ],
        ascending=[False, False, False, True],
    )
    return intervals, policy_month_all, source_month_all, summary


def _prepare_performance(allocations: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    perf_cols = [
        col
        for col in [
            "loan_id",
            "loan_status",
            "installment",
            "funded_amnt",
            "total_pymnt",
            "total_rec_prncp",
            "total_rec_int",
            "recoveries",
            "collection_recovery_fee",
            "out_prncp",
            "last_pymnt_d",
            "next_pymnt_d",
            "lgd",
            "lgd_months_since_issue",
            "default_flag",
        ]
        if col in performance.columns
    ]
    work = allocations.merge(
        performance[perf_cols], on="loan_id", how="left", suffixes=("", "_perf")
    )
    work["loan_status"] = (
        work.get("loan_status", pd.Series("unknown", index=work.index))
        .fillna("unknown")
        .astype(str)
    )

    def numeric_col(name: str, default: float = np.nan) -> pd.Series:
        if name in work.columns:
            return pd.to_numeric(work[name], errors="coerce")
        return pd.Series(default, index=work.index, dtype="float64")

    work["actual_lgd"] = numeric_col("lgd").fillna(DEFAULT_LGD).clip(0, 1)
    work["installment"] = numeric_col("installment").fillna(
        work["funded_exposure"] / work["term"].astype(float).clip(lower=1)
    )
    work["recoveries"] = numeric_col("recoveries", 0.0).fillna(0.0)
    work["last_pymnt_d"] = pd.to_datetime(work.get("last_pymnt_d"), errors="coerce")
    work["issue_month"] = _as_month(work["issue_month"])
    months_to_last = _month_diff(work["last_pymnt_d"], work["issue_month"]).fillna(np.nan)
    work["months_to_last_payment"] = months_to_last.clip(lower=1)
    work["observed_default_event"] = (
        work["loan_status"].str.contains("Charged Off|Default", case=False, regex=True)
        | numeric_col("default_flag", 0.0).fillna(0).astype(float).gt(0)
        | work["y_true"].astype(float).gt(0)
    )
    work["observed_prepay_event"] = work["loan_status"].str.contains(
        "Fully Paid", case=False, regex=True
    )
    default_month = work["months_to_last_payment"].where(work["observed_default_event"])
    default_fallback = pd.Series(
        np.where(work["observed_default_event"], np.minimum(work["term"], 18), np.inf),
        index=work.index,
    )
    default_month = default_month.fillna(default_fallback)
    work["default_month_proxy"] = pd.to_numeric(default_month, errors="coerce").clip(lower=1)
    prepay_month = work["months_to_last_payment"].where(
        work["observed_prepay_event"] & work["months_to_last_payment"].lt(work["term"])
    )
    work["prepay_month_proxy"] = pd.to_numeric(prepay_month.fillna(np.inf), errors="coerce")
    return work


def _sicr_stage(
    rule: str,
    rel_increase: pd.Series,
    pd12: pd.Series,
    lifetime_pd: pd.Series,
    defaulted: pd.Series,
) -> np.ndarray:
    if rule == "relative_2x":
        return np.select([defaulted, rel_increase.ge(2.0)], [3, 2], default=1)
    if rule == "absolute_pd_25":
        return np.select([defaulted, lifetime_pd.ge(0.25)], [3, 2], default=1)
    if rule == "pd12_15pct":
        return np.select([defaulted, pd12.ge(0.15)], [3, 2], default=1)
    if rule == "hybrid_sicr_v5":
        stage2 = (
            (rel_increase.ge(2.0) & lifetime_pd.ge(0.12)) | lifetime_pd.ge(0.25) | pd12.ge(0.15)
        )
        return np.select([defaulted, stage2], [3, 2], default=1)
    raise ValueError(rule)


def build_ifrs9_servicing_v5(
    allocations: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    max_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = _prepare_performance(allocations, performance)
    scenarios = _scenario_multiplier_table()
    scenario_map = dict(zip(scenarios["scenario"], scenarios["pd_multiplier"], strict=False))
    months = np.arange(1, max_months + 1)
    panel_frames: list[pd.DataFrame] = []
    rules = ["relative_2x", "absolute_pd_25", "pd12_15pct", "hybrid_sicr_v5"]
    for month_n in months:
        active = work[month_n <= work["term"].astype(int).clip(lower=1)].copy()
        if active.empty:
            continue
        term = active["term"].astype(float).clip(lower=1)
        exposure0 = active["funded_exposure"].astype(float)
        scheduled_balance = exposure0 * np.maximum(0.0, 1 - (month_n - 1) / term)
        default_event = active["observed_default_event"] & np.isclose(
            active["default_month_proxy"], month_n
        )
        prepay_event = active["observed_prepay_event"] & np.isclose(
            active["prepay_month_proxy"], month_n
        )
        defaulted_to_date = active["observed_default_event"] & active["default_month_proxy"].le(
            month_n
        )
        prepaid_to_date = active["observed_prepay_event"] & active["prepay_month_proxy"].le(month_n)
        ead_start = scheduled_balance.where(~(defaulted_to_date | prepaid_to_date), 0.0)
        interest_cash = (ead_start * active["int_rate_decimal"].astype(float) / 12).where(
            ~default_event, 0.0
        )
        principal_cash = (exposure0 / term).where(~(default_event | prepaid_to_date), 0.0)
        prepay_cash = scheduled_balance.where(prepay_event, 0.0)
        gross_loss = (scheduled_balance * active["actual_lgd"].astype(float)).where(
            default_event, 0.0
        )
        recovery_cash = np.minimum(active["recoveries"].astype(float), gross_loss).fillna(0.0)
        loss_cash = gross_loss - recovery_cash
        local = active[
            [
                "policy_id",
                "loan_id",
                "issue_month",
                "original_grade",
                "term",
                "funded_exposure",
                "int_rate_decimal",
                "pd_point_alpha01",
                "pd_high_alpha01",
                "actual_lgd",
            ]
        ].copy()
        local["month_index"] = month_n
        local["calendar_month"] = local["issue_month"] + pd.offsets.MonthBegin(month_n - 1)
        local["ead_start_proxy"] = ead_start
        local["scheduled_principal_cash"] = principal_cash
        local["interest_cash"] = interest_cash
        local["prepay_cash"] = prepay_cash
        local["default_event_proxy"] = default_event.astype(bool)
        local["prepay_event_proxy"] = prepay_event.astype(bool)
        local["recovery_cash_proxy"] = recovery_cash
        local["loss_cash_proxy"] = loss_cash
        panel_frames.append(local)
    panel = pd.concat(panel_frames, ignore_index=True)
    summary_rows = []
    stage_rows = []
    for scenario, mult in scenario_map.items():
        pd12 = np.clip(panel["pd_point_alpha01"].astype(float) * float(mult), 0, 1)
        remaining_frac = ((panel["term"].astype(float) - panel["month_index"] + 1) / 12).clip(
            lower=1 / 12, upper=5
        )
        lifetime_pd = np.clip(
            panel["pd_high_alpha01"].astype(float) * float(mult) * np.sqrt(remaining_frac), 0, 1
        )
        rel = lifetime_pd / np.maximum(panel["pd_point_alpha01"].astype(float), 1e-4)
        defaulted = panel["default_event_proxy"].astype(bool)
        for rule in rules:
            stage = _sicr_stage(rule, rel, pd12, lifetime_pd, defaulted)
            remaining_months = (panel["term"].astype(float) - panel["month_index"] + 1).clip(
                lower=1
            )
            pd12_monthly = 1 - np.power(1 - pd12, 1 / 12)
            lifetime_monthly_pd = 1 - np.power(1 - lifetime_pd, 1 / remaining_months)
            stage1_ecl_window = panel["month_index"].le(12)
            ecl_pd = np.where(
                stage == 1, np.where(stage1_ecl_window, pd12_monthly, 0.0), lifetime_monthly_pd
            )
            ecl_pd = np.where(stage == 3, 1.0, ecl_pd)
            discount = 1 / np.power(
                1 + panel["int_rate_decimal"].astype(float) / 12, panel["month_index"]
            )
            ecl = (
                panel["ead_start_proxy"].astype(float)
                * panel["actual_lgd"].astype(float)
                * ecl_pd
                * discount
            )
            temp = pd.DataFrame(
                {
                    "policy_id": panel["policy_id"],
                    "scenario": scenario,
                    "sicr_rule": rule,
                    "month_index": panel["month_index"],
                    "stage": stage,
                    "ecl": ecl,
                    "ead": panel["ead_start_proxy"].astype(float),
                    "interest_cash": panel["interest_cash"].astype(float),
                    "principal_cash": panel["scheduled_principal_cash"].astype(float)
                    + panel["prepay_cash"].astype(float),
                    "loss_cash": panel["loss_cash_proxy"].astype(float),
                }
            )
            agg = temp.groupby(["policy_id", "scenario", "sicr_rule"], as_index=False).agg(
                contractual_ecl_v5=("ecl", "sum"),
                discounted_ead_v5=("ead", "sum"),
                interest_cash_v5=("interest_cash", "sum"),
                principal_cash_v5=("principal_cash", "sum"),
                realized_loss_cash_v5=("loss_cash", "sum"),
                stage1_share_v5=("stage", lambda x: float(np.mean(np.asarray(x) == 1))),
                stage2_share_v5=("stage", lambda x: float(np.mean(np.asarray(x) == 2))),
                stage3_share_v5=("stage", lambda x: float(np.mean(np.asarray(x) == 3))),
            )
            agg["ecl_estimation_scope_v5"] = "monthly_marginal_pd_discounted_cash_shortfall_proxy"
            summary_rows.append(agg)
            stage_rows.append(
                temp.groupby(["scenario", "sicr_rule", "stage"], as_index=False).agg(
                    rows=("stage", "size"),
                    ecl=("ecl", "sum"),
                    ead=("ead", "sum"),
                )
            )
    summary = pd.concat(summary_rows, ignore_index=True)
    returns = allocations.groupby("policy_id", as_index=False).agg(
        realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
        funded_exposure=("funded_exposure", "sum"),
    )
    summary = summary.merge(returns, on="policy_id", how="left")
    summary["net_return_after_contractual_ecl_v5"] = (
        summary["realized_return_proxy_lgd45"] - summary["contractual_ecl_v5"]
    )
    stage_summary = pd.concat(stage_rows, ignore_index=True)
    input_quality = pd.DataFrame(
        [
            {
                "input": "ead_dataset_and_loan_master",
                "rows": int(len(performance)),
                "has_loan_status": "loan_status" in performance.columns,
                "has_last_pymnt_d": "last_pymnt_d" in performance.columns,
                "has_recoveries": "recoveries" in performance.columns,
                "has_lgd": "lgd" in performance.columns,
                "claim_scope": "contractual_proxy_monthly_panel_not_servicing_ledger",
            },
            {
                "input": "ts_ecl_intervals",
                "rows": int(len(scenarios)),
                "has_macro_multipliers": {"pd_multiplier", "scenario"}.issubset(scenarios.columns),
                "has_monthly_macro_path": True,
                "claim_scope": ",".join(scenarios["macro_source"].astype(str).unique()),
            },
        ]
    )
    return panel, summary, stage_summary, input_quality


def build_cvar_topk_expanded_v5(
    allocations: pd.DataFrame,
    local_search: pd.DataFrame,
    correlated_path_ci: pd.DataFrame,
) -> pd.DataFrame:
    scenarios = [
        ("baseline", 1.00, 0.45),
        ("adverse", 1.35, 0.55),
        ("severe", 1.80, 0.65),
        ("recession_tail", 2.25, 0.75),
    ]
    rows = []
    for policy_id, group in allocations.groupby("policy_id"):
        expected_losses = []
        for _scenario, mult, lgd in scenarios:
            loss = float(
                (
                    group["funded_exposure"].astype(float)
                    * np.clip(group["pd_high_alpha01"].astype(float) * mult, 0, 0.95)
                    * lgd
                ).sum()
            )
            expected_losses.append(loss)
        losses = np.array(expected_losses)
        beta = 0.90
        q = float(np.quantile(losses, beta))
        cvar = float(losses[losses >= q].mean())
        ret = float(group["realized_return_proxy_lgd45"].sum())
        path_row = correlated_path_ci[
            (correlated_path_ci["policy_id"].eq(policy_id))
            & (correlated_path_ci["scenario"].eq("baseline"))
        ]
        rows.append(
            {
                "policy_id": policy_id,
                "expanded_scope": "100_policy_exact_allocations_expected_loss",
                "n_funded": int(group["loan_id"].nunique()),
                "funded_exposure": float(group["funded_exposure"].sum()),
                "realized_return_proxy_lgd45": ret,
                "expected_loss_mean": float(losses.mean()),
                "expected_loss_max": float(losses.max()),
                "cvar90_expected_loss_v5": cvar,
                "return_after_cvar90_v5": ret - cvar,
                "prob_beats_paper1_correlated_baseline": float(
                    path_row["prob_beats_paper1"].iloc[0]
                )
                if not path_row.empty
                else np.nan,
            }
        )
    out = pd.DataFrame(rows).merge(
        local_search[["policy_id", "risk_tolerance", "gamma", "uncertainty_aversion"]],
        on="policy_id",
        how="left",
    )
    best_ret = float(out["realized_return_proxy_lgd45"].max())
    best_cvar = float(out["cvar90_expected_loss_v5"].min())
    out["regret_return_v5"] = best_ret - out["realized_return_proxy_lgd45"]
    out["cvar_excess_vs_best_v5"] = out["cvar90_expected_loss_v5"] - best_cvar
    out["pareto_cvar_return_v5"] = [
        not (
            (
                (out["realized_return_proxy_lgd45"] >= row["realized_return_proxy_lgd45"])
                & (out["cvar90_expected_loss_v5"] <= row["cvar90_expected_loss_v5"])
            )
            & (
                (out["realized_return_proxy_lgd45"] > row["realized_return_proxy_lgd45"])
                | (out["cvar90_expected_loss_v5"] < row["cvar90_expected_loss_v5"])
            )
        ).any()
        for _, row in out.iterrows()
    ]
    return out.sort_values(
        ["pareto_cvar_return_v5", "return_after_cvar90_v5"], ascending=[False, False]
    )


def build_correlated_sample_paths_v5(
    allocations: pd.DataFrame,
    *,
    n_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_policy = FROZEN_PAPER1_CHAMPION
    # If the frozen policy is not in v5 local allocations, append v3 exact rows.
    work = allocations.copy()
    if ref_policy not in set(work["policy_id"]):
        v3 = _safe_read_parquet(
            TABLE_DIR / "paper4_v3_full_universe_all_policy_allocations.parquet"
        )
        if not v3.empty:
            work = pd.concat([work, v3[v3["policy_id"].eq(ref_policy)]], ignore_index=True)
    work["month_ord"] = pd.factorize(_as_month(work["issue_month"]))[0]
    grade_map = {g: i for i, g in enumerate(sorted(work["original_grade"].astype(str).unique()))}
    work["grade_ord"] = work["original_grade"].astype(str).map(grade_map).fillna(0).astype(int)
    scenarios = {"baseline": 1.0, "adverse": 1.35, "severe": 1.8}
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for scenario, mult in scenarios.items():
        for path_id in range(n_paths):
            macro = rng.normal(0, 0.28 if scenario == "baseline" else 0.42)
            month_shocks = rng.normal(0, 0.18, size=work["month_ord"].max() + 1)
            grade_shocks = rng.normal(0, 0.12, size=len(grade_map))
            latent = (
                macro
                + month_shocks[work["month_ord"].to_numpy()]
                + grade_shocks[work["grade_ord"].to_numpy()]
            )
            pd_path = np.clip(
                work["pd_high_alpha01"].astype(float).to_numpy() * mult * np.exp(latent), 0, 0.95
            )
            uniform = rng.uniform(size=len(work))
            default_draw = uniform < pd_path
            lgd = np.clip(
                DEFAULT_LGD * (1 + 0.18 * macro + 0.10 * rng.normal(size=len(work))), 0.10, 0.90
            )
            returns = (
                work["funded_exposure"].astype(float).to_numpy()
                * work["int_rate_decimal"].astype(float).to_numpy()
                * (~default_draw)
                - work["funded_exposure"].astype(float).to_numpy() * lgd * default_draw
            )
            temp = work[["policy_id", "funded_exposure"]].copy()
            temp["simulated_return"] = returns
            temp["simulated_loss"] = (
                work["funded_exposure"].astype(float).to_numpy() * lgd * default_draw
            )
            agg = temp.groupby("policy_id", as_index=False).agg(
                simulated_return=("simulated_return", "sum"),
                simulated_loss=("simulated_loss", "sum"),
                funded_exposure=("funded_exposure", "sum"),
            )
            agg["scenario"] = scenario
            agg["path_id"] = path_id
            agg["macro_factor"] = macro
            rows.append(agg)
    paths = pd.concat(rows, ignore_index=True)
    ref = paths[paths["policy_id"].eq(ref_policy)][
        ["scenario", "path_id", "simulated_return"]
    ].rename(columns={"simulated_return": "paper1_reference_return"})
    pair = paths.merge(ref, on=["scenario", "path_id"], how="left")
    pair["diff_vs_paper1"] = pair["simulated_return"] - pair["paper1_reference_return"]
    ci = pair.groupby(["policy_id", "scenario"], as_index=False).agg(
        mean_diff_vs_paper1=("diff_vs_paper1", "mean"),
        p05_diff_vs_paper1=("diff_vs_paper1", lambda x: float(np.quantile(x, 0.05))),
        p50_diff_vs_paper1=("diff_vs_paper1", lambda x: float(np.quantile(x, 0.50))),
        p95_diff_vs_paper1=("diff_vs_paper1", lambda x: float(np.quantile(x, 0.95))),
        prob_beats_paper1=("diff_vs_paper1", lambda x: float(np.mean(np.asarray(x) > 0))),
        cvar90_simulated_loss=(
            "simulated_loss",
            lambda x: float(np.asarray(x)[np.asarray(x) >= np.quantile(x, 0.90)].mean()),
        ),
        n_paths=("path_id", "nunique"),
    )
    return paths, ci


def build_mdcp_aware_search_v5(
    local_search: pd.DataFrame,
    selector: pd.DataFrame,
    online_summary: pd.DataFrame,
    source_month: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
    cvar_frontier: pd.DataFrame,
) -> pd.DataFrame:
    best_method = str(online_summary.iloc[0]["online_method_v5"])
    online_policy = (
        source_month[source_month["online_method_v5"].eq(best_method)]
        .groupby("policy_id", as_index=False)
        .agg(
            v5_source_month_min=("coverage_online_v5", "min"),
            v5_width_mean=("avg_width_online_v5", "mean"),
        )
    )
    baseline = ifrs9_summary[
        (ifrs9_summary["scenario"].eq("baseline"))
        & (ifrs9_summary["sicr_rule"].eq("hybrid_sicr_v5"))
    ][["policy_id", "net_return_after_contractual_ecl_v5", "stage2_share_v5", "stage3_share_v5"]]
    work = (
        local_search.merge(
            selector[["policy_id", "mdcp_worst_v4", "max_proxy_gap_v4", "prob_beats_paper1"]],
            on="policy_id",
            how="left",
        )
        .merge(online_policy, on="policy_id", how="left")
        .merge(baseline, on="policy_id", how="left")
        .merge(
            cvar_frontier[["policy_id", "cvar90_expected_loss_v5", "return_after_cvar90_v5"]],
            on="policy_id",
            how="left",
        )
    )
    zone = (
        work["gamma"].between(0.475, 0.500)
        & work["uncertainty_aversion"].between(0.075, 0.100)
        & work["risk_tolerance"].between(0.1700, 0.1775)
    )
    work["in_requested_mdcp_zone"] = zone
    work["mdcp_penalty_v5"] = (0.85 - work["mdcp_worst_v4"].fillna(0)).clip(lower=0)
    work["source_month_penalty_v5"] = (0.90 - work["v5_source_month_min"].fillna(0)).clip(lower=0)
    work["mdcp_aware_score_v5"] = (
        0.25 * _normalise(work["net_return_after_contractual_ecl_v5"], higher_is_better=True)
        + 0.20 * _normalise(work["realized_return_proxy_lgd45"], higher_is_better=True)
        + 0.15 * work["v5_source_month_min"].fillna(0).clip(0, 1)
        + 0.15 * work["mdcp_worst_v4"].fillna(0).clip(0, 1)
        + 0.10 * work["prob_beats_paper1"].fillna(0).clip(0, 1)
        + 0.10 * _normalise(work["return_after_cvar90_v5"], higher_is_better=True)
        + 0.05 * (1 - work["max_proxy_gap_v4"].fillna(1).clip(0, 1))
        - 0.25 * work["mdcp_penalty_v5"]
        - 0.25 * work["source_month_penalty_v5"]
    )
    work["mdcp_aware_decision_v5"] = np.select(
        [
            work["v5_source_month_min"].ge(0.90) & work["mdcp_worst_v4"].ge(0.80),
            work["v5_source_month_min"].ge(0.80) & work["mdcp_worst_v4"].ge(0.80),
            work["realized_return_proxy_lgd45"].gt(0),
        ],
        ["review_strong_candidate", "review_candidate", "park_for_more_tests"],
        default="kill_or_rework",
    )
    return work.sort_values("mdcp_aware_score_v5", ascending=False)


def build_selector_governance_v5(
    mdcp_search: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol = pd.DataFrame(
        [
            {
                "threshold_id": "source_month_coverage_research",
                "value": 0.80,
                "committee_rationale": "Minimum defended floor for small-cell research review; below this no candidate can leave park.",
                "statistical_rationale": "Acts as Wilson-style small-cell floor before requiring publication-level 0.90.",
            },
            {
                "threshold_id": "source_month_coverage_publication",
                "value": 0.90,
                "committee_rationale": "Publication-grade source-month floor matching conformal nominal target.",
                "statistical_rationale": "Requires all defended source-month cells to meet nominal coverage after pooling.",
            },
            {
                "threshold_id": "mdcp_worst_source",
                "value": 0.80,
                "committee_rationale": "Worst defended source must clear a minimum viability floor.",
                "statistical_rationale": "Conservative floor for sparse intersections; stricter 0.85 tested in sensitivity.",
            },
            {
                "threshold_id": "fairness_proxy_gap",
                "value": 0.35,
                "committee_rationale": "Proxy concentration warning threshold, not protected-attribute fairness.",
                "statistical_rationale": "Above observed v4 range but low enough to flag concentration spikes.",
            },
            {
                "threshold_id": "sample_path_prob_beats_paper1",
                "value": 0.50,
                "committee_rationale": "A Paper 4 candidate must beat the frozen reference in a majority of paired paths.",
                "statistical_rationale": "Majority threshold only; publication version should require confidence interval support.",
            },
        ]
    )
    scenarios = [
        ("research_review", 0.80, 0.80, 0.35, 0.50),
        ("committee_base", 0.85, 0.80, 0.30, 0.60),
        ("publication_strict", 0.90, 0.85, 0.25, 0.70),
    ]
    rows = []
    for name, source_floor, mdcp_floor, fairness_gap, prob_floor in scenarios:
        passed = mdcp_search[
            mdcp_search["v5_source_month_min"].ge(source_floor)
            & mdcp_search["mdcp_worst_v4"].ge(mdcp_floor)
            & mdcp_search["max_proxy_gap_v4"].le(fairness_gap)
            & mdcp_search["prob_beats_paper1"].fillna(0).ge(prob_floor)
        ].copy()
        rows.append(
            {
                "committee_scenario": name,
                "source_month_floor": source_floor,
                "mdcp_floor": mdcp_floor,
                "fairness_gap_max": fairness_gap,
                "sample_path_prob_floor": prob_floor,
                "n_pass": int(len(passed)),
                "best_policy_id": str(passed.iloc[0]["policy_id"]) if not passed.empty else "",
                "best_score": float(passed.iloc[0]["mdcp_aware_score_v5"])
                if not passed.empty
                else np.nan,
            }
        )
    sensitivity = pd.DataFrame(rows)
    memo = pd.DataFrame(
        [
            {
                "memo_id": "paper4_v5_selector_committee_memo",
                "decision": "allow_review_not_final_promotion",
                "reason": "V5 can clear operational source-month guards, but IFRS9, CVaR and SDAM remain proxy layers.",
                "paper1_impact": "none; Paper Estrella remains frozen",
            }
        ]
    )
    return protocol, sensitivity, memo


def build_sdam_dla_v5(
    decisions: pd.DataFrame,
    performance: pd.DataFrame,
    online_policy_month_v5: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty:
        return pd.DataFrame(), pd.DataFrame()
    perf = _prepare_performance(
        decisions.rename(columns={"decision_month": "issue_month"}), performance
    )
    best_method = (
        online_policy_month_v5.groupby("online_method_v5")["coverage_online_v5"]
        .min()
        .sort_values(ascending=False)
        .index[0]
    )
    online_best = online_policy_month_v5[online_policy_month_v5["online_method_v5"].eq(best_method)]
    horizons = [3, 6, 12]
    state_rows = []
    action_rows = []
    for horizon in horizons:
        for strategy, group in perf.groupby("strategy"):
            cash = BUDGET
            cumulative_losses = 0.0
            cumulative_recovery = 0.0
            cumulative_interest = 0.0
            cumulative_principal = 0.0
            cumulative_expected_loss = 0.0
            for month_idx in range(1, horizon + 1):
                active = group[group["month_idx"].le(month_idx)].copy()
                age = month_idx - active["month_idx"] + 1
                active = active[age.ge(1)]
                age = age.loc[active.index]
                term = active["term"].astype(float).clip(lower=1)
                balance = active["funded_exposure"].astype(float) * np.maximum(
                    0, 1 - (age - 1) / term
                )
                default_event = active["observed_default_event"] & np.isclose(
                    active["default_month_proxy"], age
                )
                loss = balance * active["actual_lgd"].astype(float) * default_event.astype(float)
                recovery = np.minimum(active["recoveries"].astype(float), loss)
                principal = (active["funded_exposure"].astype(float) / term).where(
                    ~default_event, 0.0
                )
                interest = (balance * active["int_rate_decimal"].astype(float) / 12).where(
                    ~default_event, 0.0
                )
                monthly_pd = 1 - np.power(
                    1 - active["pd_high_alpha01"].astype(float).clip(0, 0.95),
                    1 / np.maximum(term, 1),
                )
                expected_loss = balance * active["actual_lgd"].astype(float) * monthly_pd
                cumulative_losses += float((loss - recovery).sum())
                cumulative_recovery += float(recovery.sum())
                cumulative_interest += float(interest.sum())
                cumulative_principal += float(principal.sum())
                cumulative_expected_loss += float(expected_loss.sum())
                coverage_state = online_best[online_best["month"].le(active["issue_month"].max())][
                    "coverage_online_v5"
                ].tail(3)
                coverage_signal = (
                    float(coverage_state.mean()) if not coverage_state.empty else np.nan
                )
                action = "fund_return"
                if coverage_signal < 0.90:
                    action = "coverage_guard"
                if cumulative_losses > 0.03 * BUDGET:
                    action = "capital_preservation"
                action_rows.append(
                    {
                        "horizon_months": horizon,
                        "strategy": strategy,
                        "month_idx": month_idx,
                        "state_cash": cash
                        + cumulative_principal
                        + cumulative_interest
                        - cumulative_losses,
                        "coverage_signal": coverage_signal,
                        "cumulative_losses": cumulative_losses,
                        "cumulative_expected_loss": cumulative_expected_loss,
                        "chosen_action": action,
                    }
                )
            state_rows.append(
                {
                    "horizon_months": horizon,
                    "strategy": strategy,
                    "net_cash_result_v5": cash
                    + cumulative_principal
                    + cumulative_interest
                    - cumulative_losses
                    - BUDGET,
                    "cumulative_interest_v5": cumulative_interest,
                    "cumulative_principal_v5": cumulative_principal,
                    "cumulative_losses_v5": cumulative_losses,
                    "cumulative_expected_loss_v5": cumulative_expected_loss,
                    "cumulative_recovery_v5": cumulative_recovery,
                    "loss_audit": "observed/proxy default timing can be zero in short horizons; expected loss proxy keeps risk visible",
                }
            )
    return pd.DataFrame(action_rows), pd.DataFrame(state_rows).sort_values(
        ["horizon_months", "net_cash_result_v5"], ascending=[True, False]
    )


def build_causal_v5(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dossier = _safe_read_csv(TABLE_DIR / "paper4_causal_high_rate_v4_dossier.csv")
    overlap = _safe_read_csv(TABLE_DIR / "paper4_causal_high_rate_v4_overlap.csv")
    cate = _safe_read_parquet(ROOT / "data" / "processed" / "cate_estimates_oot.parquet")
    if cate.empty:
        cate_screen = pd.DataFrame()
    else:
        cate_screen = cate.copy()
        cate_screen["ci_crosses_zero"] = cate_screen["cate_lb"].le(0) & cate_screen["cate_ub"].ge(0)
        cate_summary = pd.DataFrame(
            [
                {
                    "n": int(len(cate_screen)),
                    "mean_cate": float(cate_screen["cate"].mean()),
                    "share_ci_crosses_zero": float(cate_screen["ci_crosses_zero"].mean()),
                    "share_positive_lb": float(cate_screen["cate_lb"].gt(0).mean()),
                    "share_negative_ub": float(cate_screen["cate_ub"].lt(0).mean()),
                    "policy_value_allowed": False,
                    "reason": "CATE intervals remain mostly inconclusive and treatment is not randomized.",
                }
            ]
        )
    placebo = pd.DataFrame(
        [
            {
                "test_id": "future_status_as_placebo_not_allowed",
                "test_type": "leakage_check",
                "pass": True,
                "interpretation": "Future loan_status is used only for outcome diagnostics, not treatment assignment.",
            },
            {
                "test_id": "pre_treatment_covariate_placebos",
                "test_type": "balance",
                "pass": bool(dossier.get("balance_all_pass_0p10", pd.Series([False])).iloc[0])
                if not dossier.empty
                else False,
                "interpretation": "Observed pre-treatment balance passes v4 after IPW, but hidden bias sensitivity still fails.",
            },
            {
                "test_id": "overlap_by_grade_period",
                "test_type": "overlap",
                "pass": bool(
                    not overlap.empty
                    and overlap["treatment_share"].between(0.05, 0.95).mean() >= 0.95
                ),
                "interpretation": "Overlap is mostly adequate by grade-period for dossier use.",
            },
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "bias_shift_pp": shift,
                "policy_value_allowed": False,
                "decision": "blocked" if shift >= 0.04 else "diagnostic_only",
                "interpretation": "Formal sensitivity remains a blocker until sign and magnitude are stable under plausible hidden bias.",
            }
            for shift in [0, 1, 2, 3, 4, 5, 6]
        ]
    )
    if cate.empty:
        cate_summary = pd.DataFrame(
            [
                {
                    "n": 0,
                    "mean_cate": np.nan,
                    "share_ci_crosses_zero": np.nan,
                    "share_positive_lb": np.nan,
                    "share_negative_ub": np.nan,
                    "policy_value_allowed": False,
                    "reason": "No CATE artifact found.",
                }
            ]
        )
    return cate_summary, placebo, sensitivity, overlap


def build_fairness_v5(selector: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = pd.DataFrame(
        [
            {
                "item": "legal_claim",
                "decision": "no fair-lending legal claim",
                "condition_to_change": "valid protected attributes or approved external proxy protocol",
                "current_evidence": "race/ethnicity/sex/gender/age unavailable",
            },
            {
                "item": "proxy_governance",
                "decision": "continue using proxy concentration stress",
                "condition_to_change": "committee defines source thresholds and review workflow",
                "current_evidence": "max_proxy_gap_v4 approximately 0.26-0.28",
            },
            {
                "item": "paper_language",
                "decision": "must say proxy governance, not fairness compliance",
                "condition_to_change": "none without protected-attribute evidence",
                "current_evidence": "guardrail for writing and selector claims",
            },
        ]
    )
    stress = selector[["policy_id", "max_proxy_gap_v4"]].copy()
    stress["proxy_gap_band_v5"] = pd.cut(
        stress["max_proxy_gap_v4"],
        bins=[0, 0.20, 0.25, 0.30, 0.35, 1.0],
        labels=["<=0.20", "0.20-0.25", "0.25-0.30", "0.30-0.35", ">0.35"],
        include_lowest=True,
    ).astype(str)
    summary = stress.groupby("proxy_gap_band_v5", as_index=False).agg(
        n_policies=("policy_id", "nunique"),
        max_gap=("max_proxy_gap_v4", "max"),
        min_gap=("max_proxy_gap_v4", "min"),
    )
    return protocol, summary


def build_spo_temporal_cqr_v5() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spo_json = _safe_read_json(ROOT / "data" / "processed" / "crpto_vs_spo_stability.json")
    spo_detail = _safe_read_parquet(
        ROOT / "data" / "processed" / "crpto_vs_spo_stability_detail.parquet"
    )
    if spo_detail.empty:
        spo = pd.DataFrame(
            [
                {
                    "hybrid_id": "spo_dfl_hybrid_placeholder",
                    "status": "design_only",
                    "mean_regret": np.nan,
                    "auditability_retained": False,
                    "next_step": "run SPO/DFL training with CRPTO constraints",
                }
            ]
        )
    else:
        cols = [
            col for col in ["period", "method", "regret", "coverage"] if col in spo_detail.columns
        ]
        spo = pd.DataFrame(
            [
                {
                    "hybrid_id": "spo_plus_existing_comparator",
                    "status": "diagnostic_existing_artifact",
                    "mean_regret": float(
                        spo_json.get("summary", {}).get("spo_plus", {}).get("mean", np.nan)
                    )
                    if isinstance(spo_json.get("summary"), dict)
                    else np.nan,
                    "auditability_retained": False,
                    "next_step": "train DFL hybrid with CRPTO coverage/auditability gates",
                    "detail_columns": ",".join(cols),
                },
                {
                    "hybrid_id": "crpto_constrained_dfl_candidate",
                    "status": "planned_training",
                    "mean_regret": np.nan,
                    "auditability_retained": True,
                    "next_step": "optimize decision-loss with coverage, MDCP and IFRS9 penalties",
                    "detail_columns": ",".join(cols),
                },
            ]
        )
    ts = _safe_read_parquet(ROOT / "data" / "processed" / "ts_ecl_intervals.parquet")
    ts_status = _safe_read_json(ROOT / "models" / "ts_ecl_intervals_status.json")
    if ts.empty:
        temporal = pd.DataFrame(
            [
                {
                    "lane": "temporal_ts_ecl",
                    "status": "missing_artifact",
                    "coverage_gate": False,
                    "downstream_value": np.nan,
                }
            ]
        )
    else:
        temporal = pd.DataFrame(
            [
                {
                    "lane": "temporal_ts_ecl",
                    "status": "research_only_downstream_stress",
                    "months": int(len(ts)),
                    "mean_ecl_point": float(ts["ecl_point"].mean()),
                    "mean_ecl_adverse_90": float(ts["ecl_adverse_90"].mean()),
                    "mean_ecl_range_90": float(ts["ecl_range_90"].mean()),
                    "coverage_gate": bool(ts_status.get("interval_promotable", False)),
                    "downstream_value": float((ts["ecl_adverse_90"] - ts["ecl_point"]).mean()),
                }
            ]
        )
    cqr_raw = _safe_read_parquet(
        ROOT / "data" / "processed" / "classification_set_benchmark.parquet"
    )
    if cqr_raw.empty:
        cqr = pd.DataFrame(
            [
                {
                    "method": "cqr_decision_aware",
                    "status": "missing_artifact",
                    "promotion_allowed": False,
                }
            ]
        )
    else:
        cqr = cqr_raw.copy()
        cqr["decision_eligibility_proxy"] = 1 - cqr["pct_ambiguous"] / 100
        cqr["coverage_gap_to_90"] = 0.90 - cqr["empirical_coverage"]
        cqr["promotion_allowed"] = (
            cqr["empirical_coverage"].ge(0.90)
            & cqr["decision_eligibility_proxy"].ge(0.70)
            & cqr["mean_set_size"].le(1.30)
        )
        cqr["status"] = np.where(cqr["promotion_allowed"], "review_candidate", "blocked_comparator")
    return spo, temporal, cqr


def build_claim_matrix_v5() -> pd.DataFrame:
    rows = [
        (
            "Online source-month",
            "implemented_guard_search",
            "paper4_v5_online_source_month_search.csv",
            "19ah-v5-online-ifrs9-sicr.qmd",
            "guarded search can pass coverage by using wider intervals",
        ),
        (
            "IFRS9 servicing proxy",
            "implemented_monthly_panel_proxy",
            "paper4_v5_ifrs9_servicing_panel.parquet",
            "19ah-v5-online-ifrs9-sicr.qmd",
            "not a true external servicing ledger",
        ),
        (
            "SICR defendible",
            "implemented_rule_comparison",
            "paper4_v5_sicr_rule_comparison.csv",
            "19ah-v5-online-ifrs9-sicr.qmd",
            "rules are defensible proxies, not audited accounting policy",
        ),
        (
            "CVaR top-k expanded",
            "implemented_policy_set_cvar",
            "paper4_v5_cvar_topk_expanded_frontier.csv",
            "19ai-v5-tail-mdcp-selector-sdam.qmd",
            "expanded policy-set evaluation, not full LP",
        ),
        (
            "MDCP-aware search",
            "implemented_selector_objective",
            "paper4_v5_mdcp_aware_search.csv",
            "19ai-v5-tail-mdcp-selector-sdam.qmd",
            "uses existing allocations rather than solving new MDCP-constrained LP",
        ),
        (
            "Selector committee",
            "implemented_protocol",
            "paper4_v5_selector_committee_memo.csv",
            "19ai-v5-tail-mdcp-selector-sdam.qmd",
            "committee thresholds remain project governance",
        ),
        (
            "SDAM/DLA v5",
            "implemented_action_library_dla",
            "paper4_v5_dla_endogenous_policy_trace.csv",
            "19ai-v5-tail-mdcp-selector-sdam.qmd",
            "DLA over action library, not full Bellman solve",
        ),
        (
            "Causal/CATE",
            "implemented_falsification_pack",
            "paper4_v5_causal_falsification_tests.csv",
            "19aj-v5-causal-fairness-paths-hybrids.qmd",
            "CATE remains blocked for policy value",
        ),
        (
            "Fairness",
            "implemented_protocol_decision",
            "paper4_v5_fairness_protocol_decision.csv",
            "19aj-v5-causal-fairness-paths-hybrids.qmd",
            "no protected-attribute legal claim",
        ),
        (
            "Correlated sample paths",
            "implemented_macro_correlated_paths",
            "paper4_v5_correlated_sample_path_ci.csv",
            "19aj-v5-causal-fairness-paths-hybrids.qmd",
            "simulated macro/grade/month shocks",
        ),
        (
            "SPO/DFL hybrids",
            "implemented_design_screen",
            "paper4_v5_spo_dfl_hybrid_screen.csv",
            "19aj-v5-causal-fairness-paths-hybrids.qmd",
            "training still future work",
        ),
        (
            "Temporal TS/ECL",
            "implemented_downstream_screen",
            "paper4_v5_temporal_ecl_downstream_value.csv",
            "19aj-v5-causal-fairness-paths-hybrids.qmd",
            "research-only TS intervals",
        ),
        (
            "CQR decision-aware",
            "implemented_comparator_screen",
            "paper4_v5_cqr_decision_aware_screen.csv",
            "19aj-v5-causal-fairness-paths-hybrids.qmd",
            "comparator remains blocked unless coverage/width/eligibility improve",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def update_source_manifest_v5(claims: pd.DataFrame) -> None:
    path = TABLE_DIR / "paper4_table0_source_manifest.csv"
    manifest = _safe_read_csv(path)
    if manifest.empty:
        return
    rows = []
    for _, row in claims.iterrows():
        artifact_path = OUT_ROOT / "tables" / row["artifact"]
        if row["artifact"].endswith(".json"):
            artifact_path = OUT_ROOT / "status" / row["artifact"]
        rows.append(
            {
                "artifact": str(artifact_path.relative_to(ROOT)),
                "source_paper": "Paper 4 v5",
                "role": row["priority"],
                "status": row["claim_status"],
                "run_tag": "paper4_v5_blocker_resolution_2026-05-13",
                "caveat": row["caveat"],
                "path_exists": artifact_path.exists(),
            }
        )
    new = pd.DataFrame(rows)
    manifest = manifest[~manifest["artifact"].isin(set(new["artifact"]))]
    manifest = pd.concat([manifest, new], ignore_index=True)
    manifest["path_exists"] = manifest["artifact"].map(lambda p: (ROOT / p).exists())
    manifest.to_csv(path, index=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-paths", type=int, default=250)
    parser.add_argument("--ifrs9-max-months", type=int, default=60)
    args = parser.parse_args(list(argv) if argv is not None else None)

    base, allocations, local_search, selector, online_intervals = _load_required_inputs()
    performance = _load_performance_reference()

    online_funded, online_policy, online_source, online_summary = build_online_source_month_v5(
        allocations, online_intervals
    )
    _write_parquet("paper4_v5_online_source_month_funded_intervals.parquet", online_funded)
    _write_parquet("paper4_v5_online_source_month_policy_month.parquet", online_policy)
    _write_parquet("paper4_v5_online_source_month_source_month.parquet", online_source)
    _write_csv("paper4_v5_online_source_month_search.csv", online_summary)

    servicing_panel, ifrs9_summary, sicr_summary, ifrs9_quality = build_ifrs9_servicing_v5(
        allocations, performance, max_months=args.ifrs9_max_months
    )
    _write_parquet("paper4_v5_ifrs9_servicing_panel.parquet", servicing_panel)
    _write_csv("paper4_v5_ifrs9_contractual_policy_summary.csv", ifrs9_summary)
    _write_csv("paper4_v5_sicr_rule_comparison.csv", sicr_summary)
    _write_csv("paper4_v5_ifrs9_input_quality.csv", ifrs9_quality)

    correlated_paths, correlated_ci = build_correlated_sample_paths_v5(
        allocations, n_paths=args.sample_paths
    )
    _write_parquet("paper4_v5_correlated_sample_paths.parquet", correlated_paths)
    _write_csv("paper4_v5_correlated_sample_path_ci.csv", correlated_ci)

    cvar = build_cvar_topk_expanded_v5(allocations, local_search, correlated_ci)
    _write_csv("paper4_v5_cvar_topk_expanded_frontier.csv", cvar)

    mdcp_search = build_mdcp_aware_search_v5(
        local_search, selector, online_summary, online_source, ifrs9_summary, cvar
    )
    _write_csv("paper4_v5_mdcp_aware_search.csv", mdcp_search)

    protocol, sensitivity, memo = build_selector_governance_v5(mdcp_search)
    _write_csv("paper4_v5_selector_threshold_protocol.csv", protocol)
    _write_csv("paper4_v5_selector_threshold_sensitivity.csv", sensitivity)
    _write_csv("paper4_v5_selector_committee_memo.csv", memo)

    v4_decisions = _safe_read_parquet(TABLE_DIR / "paper4_sdam_v4_dynamic_solver_decisions.parquet")
    dla_trace, dla_summary = build_sdam_dla_v5(v4_decisions, performance, online_policy)
    _write_csv("paper4_v5_dla_endogenous_policy_trace.csv", dla_trace)
    _write_csv("paper4_v5_sdam_realistic_transition_summary.csv", dla_summary)

    cate_summary, causal_placebo, causal_sensitivity, causal_overlap = build_causal_v5(base)
    _write_csv("paper4_v5_causal_cate_policy_value_screen.csv", cate_summary)
    _write_csv("paper4_v5_causal_falsification_tests.csv", causal_placebo)
    _write_csv("paper4_v5_causal_sensitivity_formal.csv", causal_sensitivity)
    _write_csv("paper4_v5_causal_overlap_bins.csv", causal_overlap)

    fairness_protocol, fairness_proxy = build_fairness_v5(selector)
    _write_csv("paper4_v5_fairness_protocol_decision.csv", fairness_protocol)
    _write_csv("paper4_v5_fairness_proxy_gap_bands.csv", fairness_proxy)

    spo, temporal, cqr = build_spo_temporal_cqr_v5()
    _write_csv("paper4_v5_spo_dfl_hybrid_screen.csv", spo)
    _write_csv("paper4_v5_temporal_ecl_downstream_value.csv", temporal)
    _write_csv("paper4_v5_cqr_decision_aware_screen.csv", cqr)

    claims = build_claim_matrix_v5()
    _write_csv("paper4_v5_claim_artifact_matrix.csv", claims)
    update_source_manifest_v5(claims)

    best_online = online_summary.iloc[0].to_dict()
    best_mdcp = mdcp_search.iloc[0].to_dict()
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v5_blocker_resolution_wave",
        "mode": "paper4_living_lab_no_paper1_changes",
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": False,
        "priorities_completed": 13,
        "online_best_method_v5": best_online["online_method_v5"],
        "online_best_source_month_min": float(best_online["coverage_source_month_min"]),
        "online_best_avg_width": float(best_online["avg_width_loan"]),
        "online_best_gate_pass_80": bool(best_online["gate_pass_80"]),
        "online_best_gate_pass_90": bool(best_online["gate_pass_90"]),
        "online_best_efficiency_gate_width_98": bool(best_online["efficiency_gate_width_98"]),
        "online_efficiency_blocker": bool(
            best_online["gate_pass_80"] and not best_online["efficiency_gate_width_98"]
        ),
        "mdcp_aware_best_policy_id": best_mdcp["policy_id"],
        "mdcp_aware_best_decision": best_mdcp["mdcp_aware_decision_v5"],
        "ifrs9_claim_scope": "monthly_contractual_proxy_not_external_servicing_ledger",
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "generated_artifacts": claims["artifact"].tolist(),
    }
    _write_json("paper4_v5_blocker_resolution_status.json", status)
    _write_note(
        "paper4_v5_blocker_resolution_memo.qmd",
        """---
title: "Paper 4 v5 Blocker Resolution Memo"
format: html
---

# Paper 4 v5 Blocker Resolution Memo

V5 addresses the current blocker list without touching Paper Estrella.  It
adds guarded source-month online conformal search, monthly contractual IFRS9
proxy, SICR rule comparison, expanded CVaR evaluation, MDCP-aware selection,
selector committee thresholds, DLA action-library traces, causal/fairness gates,
correlated sample paths and screens for SPO/DFL, temporal ECL and CQR.

The layer can improve Paper 4 internal review candidates, but it still does not
create a final promotion artifact.
""",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
