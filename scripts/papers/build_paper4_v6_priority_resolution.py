"""Build Paper 4 v6 artifacts for the ten priority blockers.

V6 is an implementation wave, not a publication/promotion wave.  It keeps the
Paper Estrella champion frozen and creates auditable artifacts for:

1. efficient source-month online conformal search;
2. SICR recalibration;
3. contractual IFRS9 data-readiness and available monthly proxy;
4. MDCP/source-aware optimization inside the solver;
5. expanded CVaR constraint optimization;
6. loan-level endogenous DLA;
7. more realistic correlated sample paths;
8. CATE causal gates;
9. fairness proxy governance;
10. SPO+/DFL-style constrained hybrid search.

The script deliberately writes no final promotion JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _load_base_loan_frame,
    _safe_read_csv,
    _safe_read_parquet,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_next_wave_experiments import _as_month, _prepare_base
from scripts.papers.build_paper4_v4_open_priorities import (
    FROZEN_PAPER1_CHAMPION,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
STATUS_DIR = OUT_ROOT / "status"
NOTE_DIR = OUT_ROOT / "notes"

SCHEMA_VERSION = "2026-05-13.5"
RNG_SEED = 20260513
SOURCE_FAMILIES = [
    "original_grade",
    "period",
    "score_decile",
    "state_top20",
    "income_band",
    "dti_band",
]


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


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _prepare_base(_load_base_loan_frame()).copy()
    base["loan_id"] = base["id"].astype(str)
    base["issue_month"] = _as_month(base["issue_month"])
    candidate_pool = _safe_read_parquet(
        TABLE_DIR / "paper4_challenger_local_candidate_pool.parquet"
    )
    allocations = _safe_read_parquet(TABLE_DIR / "paper4_challenger_local_allocations.parquet")
    local_search = _safe_read_csv(TABLE_DIR / "paper4_challenger_local_search.csv")
    online_intervals = _safe_read_parquet(
        TABLE_DIR / "paper4_online_conformal_v4_intervals.parquet"
    )
    if candidate_pool.empty or allocations.empty or online_intervals.empty:
        raise FileNotFoundError("V6 requires Paper 4 v4/v5 artifacts.")
    for df in [candidate_pool, allocations, online_intervals]:
        df["loan_id"] = df["loan_id"].astype(str)
        if "issue_month" in df.columns:
            df["issue_month"] = _as_month(df["issue_month"])
    return base, candidate_pool, allocations, local_search, online_intervals


def _interval_width(y_pred: pd.Series, q: pd.Series) -> pd.Series:
    return (y_pred + q).clip(0, 1) - (y_pred - q).clip(0, 1)


def _coverage(y_true: pd.Series, y_pred: pd.Series, q: pd.Series) -> pd.Series:
    return y_true.between((y_pred - q).clip(0, 1), (y_pred + q).clip(0, 1))


def _source_month_metrics(local: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = local.copy()
    local["online_method_v6"] = method
    policy_month = (
        local.groupby(["online_method_v6", "policy_id", "issue_month"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            coverage_online_v6=("covered_online_v6", "mean"),
            avg_width_online_v6=("interval_width_online_v6", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    source_frames = []
    for source in SOURCE_FAMILIES:
        if source not in local.columns:
            continue
        src = (
            local.groupby(["online_method_v6", "policy_id", "issue_month", source], as_index=False)
            .agg(
                n=("loan_id", "nunique"),
                coverage_online_v6=("covered_online_v6", "mean"),
                avg_width_online_v6=("interval_width_online_v6", "mean"),
            )
            .rename(columns={"issue_month": "month", source: "source_value"})
        )
        src = src[src["n"].ge(5)].copy()
        src["source_id"] = source
        source_frames.append(src)
    source_month = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    source_month["source_value"] = source_month["source_value"].astype(str)
    return policy_month, source_month


def _prepare_online_frame(
    allocations: pd.DataFrame, online_intervals: pd.DataFrame
) -> pd.DataFrame:
    base_intervals = online_intervals[
        online_intervals["online_method_v4"].eq("source_aware_guarded")
    ].copy()
    keep = [
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
    merged = allocations[
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
    ].merge(base_intervals[keep], on="loan_id", how="left", suffixes=("", "_interval"))
    for col in ["issue_month", *SOURCE_FAMILIES, "term"]:
        interval_col = f"{col}_interval"
        if interval_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[interval_col])
            merged = merged.drop(columns=[interval_col])
    merged["qhat_v4"] = pd.to_numeric(merged["qhat_v4"], errors="coerce").fillna(0.55).clip(0, 1)
    merged["y_pred"] = (
        pd.to_numeric(merged["y_pred"], errors="coerce").fillna(merged["qhat_v4"]).clip(0, 1)
    )
    merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce").fillna(0.0).clip(0, 1)
    merged["q_required_oracle"] = np.where(
        merged["y_true"].ge(0.5),
        1 - merged["y_pred"],
        merged["y_pred"],
    ).clip(0, 1)
    return merged


def _oracle_min_width_rescue(local: pd.DataFrame, floor: float = 0.80) -> pd.Series:
    q = local["qhat_v4"].copy()
    work = local.copy()
    for _ in range(400):
        work["covered_tmp"] = _coverage(work["y_true"], work["y_pred"], q)
        worst_idx: tuple[str, pd.Timestamp, str, str] | None = None
        worst_cov = 1.0
        worst_n = 0
        policy_grouped = work.groupby(["policy_id", "issue_month"], dropna=False)
        policy_cov = policy_grouped["covered_tmp"].mean()
        policy_n = policy_grouped["loan_id"].nunique()
        policy_bad = policy_cov[(policy_n >= 5) & (policy_cov < floor)]
        if not policy_bad.empty:
            key = policy_bad.idxmin()
            worst_cov = float(policy_bad.loc[key])
            worst_idx = (key[0], key[1], "__policy_month__", "__all__")
            worst_n = int(policy_n.loc[key])
        for source in SOURCE_FAMILIES:
            grouped = work.groupby(["policy_id", "issue_month", source], dropna=False)
            cov = grouped["covered_tmp"].mean()
            n = grouped["loan_id"].nunique()
            bad = cov[(n >= 5) & (cov < floor)]
            if bad.empty:
                continue
            key = bad.idxmin()
            if float(bad.loc[key]) < worst_cov:
                worst_cov = float(bad.loc[key])
                worst_idx = (key[0], key[1], source, str(key[2]))
                worst_n = int(n.loc[key])
        if worst_idx is None:
            break
        policy_id, month, source, value = worst_idx
        if source == "__policy_month__":
            mask = (
                work["policy_id"].eq(policy_id)
                & work["issue_month"].eq(month)
                & ~work["covered_tmp"]
            )
        else:
            mask = (
                work["policy_id"].eq(policy_id)
                & work["issue_month"].eq(month)
                & work[source].astype(str).eq(value)
                & ~work["covered_tmp"]
            )
        if not mask.any():
            break
        needed = max(1, math.ceil(floor * worst_n) - int(round(worst_cov * worst_n)))
        candidates = work.loc[mask].copy()
        candidates["increase"] = (candidates["q_required_oracle"] - q.loc[candidates.index]).clip(
            lower=0
        )
        chosen = candidates.sort_values("increase").head(needed).index
        q.loc[chosen] = np.maximum(q.loc[chosen], work.loc[chosen, "q_required_oracle"])
    return q.clip(0, 1)


def build_online_source_month_v6(
    allocations: pd.DataFrame,
    online_intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = _prepare_online_frame(allocations, online_intervals)
    v5_source = _safe_read_parquet(TABLE_DIR / "paper4_v5_online_source_month_source_month.parquet")
    v5_policy = _safe_read_parquet(TABLE_DIR / "paper4_v5_online_source_month_policy_month.parquet")
    weak_cells: set[tuple[str, str, str, str]] = set()
    weak_policy_cells: set[tuple[str, str]] = set()
    if not v5_source.empty:
        weak = v5_source[
            v5_source["online_method_v5"].eq("v4_reference")
            & v5_source["coverage_online_v5"].lt(0.80)
        ]
        weak_cells = {
            (
                str(row["policy_id"]),
                str(pd.to_datetime(row["month"]).date()),
                str(row["source_id"]),
                str(row["source_value"]),
            )
            for _, row in weak.iterrows()
        }
    if not v5_policy.empty:
        weak_policy = v5_policy[
            v5_policy["online_method_v5"].eq("v4_reference")
            & v5_policy["coverage_online_v5"].lt(0.80)
        ]
        weak_policy_cells = {
            (str(row["policy_id"]), str(pd.to_datetime(row["month"]).date()))
            for _, row in weak_policy.iterrows()
        }

    frames = []
    summary_rows = []
    methods: list[tuple[str, pd.Series, bool, str]] = []
    methods.append(
        ("v4_reference_width_floor", merged["qhat_v4"].clip(0, 1), True, "deployable_reference")
    )
    historical_risk = (
        merged["original_grade"].astype(str).isin(["D", "E", "F", "G"]).astype(float)
        + merged["dti_band"].astype(str).str.contains("q3", na=False).astype(float)
        + merged["score_decile"]
        .astype(str)
        .str.contains("0|1|2", regex=True, na=False)
        .astype(float)
    ) / 3.0
    methods.append(
        (
            "predecision_risk_pooling_v6",
            (merged["qhat_v4"] * 1.035 + 0.015 + 0.035 * historical_risk).clip(0, 1),
            True,
            "deployable_feature_guard",
        )
    )
    weak_row = pd.Series(False, index=merged.index)
    weak_row = weak_row | pd.Series(
        list(zip(merged["policy_id"].astype(str), merged["issue_month"].dt.date.astype(str), strict=False)),
        index=merged.index,
    ).isin(weak_policy_cells)
    for source in SOURCE_FAMILIES:
        keys = list(
            zip(
                merged["policy_id"].astype(str),
                merged["issue_month"].dt.date.astype(str),
                [source] * len(merged),
                merged[source].astype(str), strict=False,
            )
        )
        weak_row = weak_row | pd.Series([key in weak_cells for key in keys], index=merged.index)
    targeted_q = merged["qhat_v4"].where(
        ~weak_row, np.maximum(merged["qhat_v4"], merged["q_required_oracle"])
    )
    methods.append(
        (
            "audit_targeted_weak_cell_rescue_v6",
            targeted_q.clip(0, 1),
            False,
            "ex_post_diagnostic_upper_bound",
        )
    )
    methods.append(
        (
            "oracle_min_width_cell_rescue_v6",
            _oracle_min_width_rescue(merged, floor=0.80),
            False,
            "ex_post_oracle_lower_bound",
        )
    )

    policy_frames = []
    source_frames = []
    for method, q, deployable, interpretation in methods:
        local = merged.copy()
        local["online_method_v6"] = method
        local["deployable_without_outcomes"] = deployable
        local["method_interpretation"] = interpretation
        local["qhat_v6"] = q
        local["pd_low_online_v6"] = (local["y_pred"] - q).clip(0, 1)
        local["pd_high_online_v6"] = (local["y_pred"] + q).clip(0, 1)
        local["covered_online_v6"] = _coverage(local["y_true"], local["y_pred"], q)
        local["interval_width_online_v6"] = _interval_width(local["y_pred"], q)
        local["oracle_or_audit_cell"] = (not deployable) & q.gt(local["qhat_v4"] + 1e-12)
        frames.append(
            local[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "online_method_v6",
                    "deployable_without_outcomes",
                    "method_interpretation",
                    "qhat_v6",
                    "pd_low_online_v6",
                    "pd_high_online_v6",
                    "covered_online_v6",
                    "interval_width_online_v6",
                    "oracle_or_audit_cell",
                ]
            ]
        )
        policy_month, source_month = _source_month_metrics(local, method)
        policy_frames.append(policy_month)
        source_frames.append(source_month)
        summary_rows.append(
            {
                "online_method_v6": method,
                "deployable_without_outcomes": deployable,
                "method_interpretation": interpretation,
                "coverage_policy_month_mean": float(policy_month["coverage_online_v6"].mean()),
                "coverage_policy_month_min": float(policy_month["coverage_online_v6"].min()),
                "coverage_source_month_min": float(source_month["coverage_online_v6"].min()),
                "avg_width_loan": float(local["interval_width_online_v6"].mean()),
                "avg_width_policy_month": float(policy_month["avg_width_online_v6"].mean()),
                "share_rows_widened": float((q > merged["qhat_v4"] + 1e-12).mean()),
                "gate_pass_80": bool(
                    policy_month["coverage_online_v6"].min() >= 0.80
                    and source_month["coverage_online_v6"].min() >= 0.80
                ),
                "gate_pass_90": bool(
                    policy_month["coverage_online_v6"].min() >= 0.90
                    and source_month["coverage_online_v6"].min() >= 0.90
                ),
                "efficiency_gate_width_95": bool(local["interval_width_online_v6"].mean() <= 0.95),
                "efficiency_gate_width_98": bool(local["interval_width_online_v6"].mean() <= 0.98),
                "promotion_eligible": bool(
                    deployable
                    and policy_month["coverage_online_v6"].min() >= 0.80
                    and source_month["coverage_online_v6"].min() >= 0.80
                    and local["interval_width_online_v6"].mean() <= 0.95
                ),
            }
        )
    intervals = pd.concat(frames, ignore_index=True)
    policy_month = pd.concat(policy_frames, ignore_index=True)
    source_month = pd.concat(source_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["promotion_eligible", "gate_pass_80", "avg_width_loan"],
        ascending=[False, False, True],
    )
    return intervals, policy_month, source_month, summary


def _ifrs9_stage(
    rule: str, rel: pd.Series, pd12: pd.Series, lifetime: pd.Series, defaulted: pd.Series
) -> np.ndarray:
    if rule == "relative_2x":
        stage2 = rel.ge(2.0)
    elif rule == "absolute_pd25":
        stage2 = lifetime.ge(0.25)
    elif rule == "pd12_15":
        stage2 = pd12.ge(0.15)
    elif rule == "balanced_rel3_abs35_pd20":
        stage2 = (rel.ge(3.0) & lifetime.ge(0.15)) | lifetime.ge(0.35) | pd12.ge(0.20)
    elif rule == "mrm_rel2p5_abs30_pd18":
        stage2 = (rel.ge(2.5) & lifetime.ge(0.15)) | lifetime.ge(0.30) | pd12.ge(0.18)
    elif rule == "conservative_rel2_abs25_pd15":
        stage2 = (rel.ge(2.0) & lifetime.ge(0.12)) | lifetime.ge(0.25) | pd12.ge(0.15)
    else:
        raise ValueError(rule)
    return np.select([defaulted, stage2], [3, 2], default=1)


def build_sicr_calibration_v6() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _safe_read_parquet(TABLE_DIR / "paper4_v5_ifrs9_servicing_panel.parquet")
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    returns = _safe_read_csv(TABLE_DIR / "paper4_v5_ifrs9_contractual_policy_summary.csv")
    returns = returns[
        ["policy_id", "realized_return_proxy_lgd45", "funded_exposure"]
    ].drop_duplicates("policy_id")
    scenarios = pd.DataFrame(
        [
            ("optimistic", 0.75),
            ("baseline", 1.00),
            ("adverse", 1.35),
            ("severe", 1.80),
        ],
        columns=["scenario", "pd_multiplier"],
    )
    rules = [
        "relative_2x",
        "absolute_pd25",
        "pd12_15",
        "balanced_rel3_abs35_pd20",
        "mrm_rel2p5_abs30_pd18",
        "conservative_rel2_abs25_pd15",
    ]
    summary_frames = []
    policy_frames = []
    for scenario, mult in scenarios.itertuples(index=False):
        pd12 = np.clip(panel["pd_point_alpha01"].astype(float) * float(mult), 0, 1)
        remaining_months = (panel["term"].astype(float) - panel["month_index"] + 1).clip(lower=1)
        remaining_years = (remaining_months / 12).clip(lower=1 / 12, upper=5)
        lifetime = np.clip(
            panel["pd_high_alpha01"].astype(float) * float(mult) * np.sqrt(remaining_years), 0, 1
        )
        rel = lifetime / np.maximum(panel["pd_point_alpha01"].astype(float), 1e-4)
        defaulted = panel["default_event_proxy"].astype(bool)
        for rule in rules:
            stage = _ifrs9_stage(rule, rel, pd12, lifetime, defaulted)
            pd12_monthly = 1 - np.power(1 - pd12, 1 / 12)
            lifetime_monthly = 1 - np.power(1 - lifetime, 1 / remaining_months)
            ecl_pd = np.where(
                stage == 1,
                np.where(panel["month_index"].le(12), pd12_monthly, 0.0),
                lifetime_monthly,
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
                    "sicr_rule_v6": rule,
                    "stage": stage,
                    "ecl": ecl,
                    "ead": panel["ead_start_proxy"].astype(float),
                }
            )
            agg = temp.groupby(["policy_id", "scenario", "sicr_rule_v6"], as_index=False).agg(
                contractual_ecl_v6=("ecl", "sum"),
                discounted_ead_v6=("ead", "sum"),
                stage1_share_v6=("stage", lambda x: float(np.mean(np.asarray(x) == 1))),
                stage2_share_v6=("stage", lambda x: float(np.mean(np.asarray(x) == 2))),
                stage3_share_v6=("stage", lambda x: float(np.mean(np.asarray(x) == 3))),
            )
            policy_frames.append(agg)
            summary_frames.append(
                agg.assign(
                    stage2_dominates=lambda d: d["stage2_share_v6"].gt(0.75),
                    stage2_too_low_diagnostic=lambda d: d["stage2_share_v6"].lt(0.10),
                )
                .groupby(["scenario", "sicr_rule_v6"], as_index=False)
                .agg(
                    mean_contractual_ecl_v6=("contractual_ecl_v6", "mean"),
                    mean_stage1_share_v6=("stage1_share_v6", "mean"),
                    mean_stage2_share_v6=("stage2_share_v6", "mean"),
                    mean_stage3_share_v6=("stage3_share_v6", "mean"),
                    policies_stage2_dominates=("stage2_dominates", "sum"),
                    policies_stage2_too_low=("stage2_too_low_diagnostic", "sum"),
                )
            )
    policy_summary = pd.concat(policy_frames, ignore_index=True).merge(
        returns, on="policy_id", how="left"
    )
    policy_summary["net_return_after_contractual_ecl_v6"] = (
        policy_summary["realized_return_proxy_lgd45"] - policy_summary["contractual_ecl_v6"]
    )
    grid = pd.concat(summary_frames, ignore_index=True)
    grid["sicr_recommendation_v6"] = np.select(
        [
            grid["mean_stage2_share_v6"].between(0.15, 0.70)
            & grid["policies_stage2_dominates"].eq(0),
            grid["mean_stage2_share_v6"].gt(0.75),
            grid["mean_stage2_share_v6"].lt(0.10),
        ],
        ["candidate_for_mrm_review", "too_conservative_stage2_dominates", "too_loose_stage2_low"],
        default="diagnostic_sensitivity_only",
    )
    return grid, policy_summary


def build_contractual_data_audit_v6() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = []
    for path in [
        ROOT / "data" / "processed" / "ead_dataset.parquet",
        ROOT / "data" / "processed" / "loan_master.parquet",
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            cols = set(df.columns)
            sources.append(
                {
                    "source_path": str(path.relative_to(ROOT)),
                    "rows": int(len(df)),
                    "loan_status": "loan_status" in cols,
                    "issue_date": "issue_d" in cols,
                    "installment": "installment" in cols,
                    "last_payment_date": "last_pymnt_d" in cols,
                    "next_payment_date": "next_pymnt_d" in cols,
                    "total_payment": "total_pymnt" in cols,
                    "principal_received": "total_rec_prncp" in cols,
                    "interest_received": "total_rec_int" in cols,
                    "recoveries": "recoveries" in cols,
                    "outstanding_principal": "out_prncp" in cols,
                    "days_past_due": bool(
                        {"days_past_due", "dpd", "mths_since_last_delinq"}.intersection(cols)
                    ),
                    "forbearance_or_hardship": bool(
                        {c for c in cols if "hardship" in c.lower() or "forbear" in c.lower()}
                    ),
                    "deferral_term": "deferral_term" in cols,
                    "macro_path": False,
                }
            )
    audit = pd.DataFrame(sources)
    needed = [
        (
            "monthly_payment_panel",
            ["last_payment_date", "total_payment", "principal_received", "interest_received"],
        ),
        ("monthly_ead_observed", ["outstanding_principal"]),
        ("prepayment_timing", ["last_payment_date", "loan_status"]),
        ("default_timing", ["loan_status", "last_payment_date"]),
        ("recovery_timing", ["recoveries", "last_payment_date"]),
        ("cure_or_forbearance", ["forbearance_or_hardship", "deferral_term"]),
        ("days_past_due_sicr", ["days_past_due"]),
        ("macro_scenario_path", ["macro_path"]),
    ]
    gap_rows = []
    for requirement, fields in needed:
        available = bool(
            (
                audit[fields].any(axis=1)
                if set(fields).issubset(audit.columns)
                else pd.Series([False])
            ).any()
        )
        gap_rows.append(
            {
                "requirement": requirement,
                "available_in_current_artifacts": available,
                "decision": "usable_now" if available else "missing_or_proxy_only",
                "paper4_action": "use in proxy panel"
                if available
                else "keep explicit blocker until external/servicing data exists",
            }
        )
    gaps = pd.DataFrame(gap_rows)
    readiness = pd.DataFrame(
        [
            {
                "readiness_item": "contractual_ifrs9_v6",
                "available_requirements": int(gaps["available_in_current_artifacts"].sum()),
                "total_requirements": int(len(gaps)),
                "readiness_score": float(gaps["available_in_current_artifacts"].mean()),
                "claim_scope": "enhanced_proxy_not_true_contractual_servicing_panel",
                "promotion_allowed": False,
            }
        ]
    )
    return audit, gaps, readiness


def _scenario_loss_matrix(pool: pd.DataFrame) -> tuple[list[tuple[str, float, float]], np.ndarray]:
    scenarios = [
        ("baseline_mid", 1.00, 0.45),
        ("baseline_high", 1.12, 0.48),
        ("adverse_mid", 1.40, 0.55),
        ("adverse_high", 1.65, 0.58),
        ("severe_mid", 1.90, 0.65),
        ("severe_high", 2.20, 0.70),
    ]
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    amount = pool["loan_amnt"].to_numpy(dtype=float)
    loss = np.vstack([np.clip(pd_high * mult, 0, 1) * lgd * amount for _, mult, lgd in scenarios])
    return scenarios, loss


def _prepare_solver_pool(
    candidate_pool: pd.DataFrame, online_intervals: pd.DataFrame, max_n: int
) -> pd.DataFrame:
    pool = candidate_pool.copy()
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool["issue_month"] = _as_month(pool["issue_month"])
    base_intervals = online_intervals[
        online_intervals["online_method_v4"].eq("source_aware_guarded")
    ][["loan_id", "qhat_v4", "y_pred"]].copy()
    base_intervals["loan_id"] = base_intervals["loan_id"].astype(str)
    pool = pool.merge(
        base_intervals.drop_duplicates("loan_id"),
        on="loan_id",
        how="left",
        suffixes=("", "_online"),
    )
    if "qhat_v4_online" in pool.columns:
        pool["qhat_v4"] = pool["qhat_v4_online"].where(
            pool["qhat_v4_online"].notna(), pool.get("qhat_v4")
        )
    if "y_pred_online" in pool.columns:
        pool["y_pred"] = pool["y_pred_online"].where(
            pool["y_pred_online"].notna(), pool.get("y_pred")
        )
    if "qhat_v4" not in pool.columns:
        pool["qhat_v4"] = 0.55
    if "y_pred" not in pool.columns:
        pool["y_pred"] = pool["pd_point_alpha01"]
    pool["qhat_v4"] = pd.to_numeric(pool["qhat_v4"], errors="coerce").fillna(0.55).clip(0, 1)
    pool["y_pred"] = (
        pd.to_numeric(pool["y_pred"], errors="coerce").fillna(pool["pd_point_alpha01"]).clip(0, 1)
    )
    for source in SOURCE_FAMILIES:
        if source not in pool.columns:
            pool[source] = "unknown"
    pool["period"] = (
        pool["period"].astype(str)
        if "period" in pool.columns
        else pool["issue_month"].dt.year.astype(str)
    )
    pool["base_return_vec"] = pool["loan_amnt"] * (
        pool["int_rate_decimal"].astype(float)
        - pool["pd_point_alpha01"].astype(float) * DEFAULT_LGD
    )
    pool["weak_source_proxy"] = (
        pool["original_grade"].astype(str).isin(["D", "E", "F", "G"]).astype(float)
        + pool["dti_band"].astype(str).str.contains("q3", na=False).astype(float)
        + pool["score_decile"].astype(str).isin(["0", "1", "2"]).astype(float)
    ) / 3.0
    pool["solver_score_seed"] = (
        pool["base_return_vec"]
        - 0.08 * pool["loan_amnt"] * pool["qhat_v4"]
        - 0.06 * pool["loan_amnt"] * pool["weak_source_proxy"]
    )
    seeded = pool.sort_values("solver_score_seed", ascending=False).head(max_n).copy()
    return seeded.reset_index(drop=True)


def _solve_linear_policy(
    pool: pd.DataFrame,
    *,
    policy_id: str,
    risk_tolerance: float,
    weak_penalty: float,
    width_penalty: float,
    cvar_cap: float | None = None,
    return_floor: float | None = None,
    max_weak_share: float | None = None,
    time_limit: int = 90,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    n = len(pool)
    loan = pool["loan_amnt"].to_numpy(dtype=float)
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    base_return = pool["base_return_vec"].to_numpy(dtype=float)
    weak = pool["weak_source_proxy"].to_numpy(dtype=float)
    width = pool["qhat_v4"].to_numpy(dtype=float)
    obj_vec = base_return - weak_penalty * loan * weak - width_penalty * loan * width
    model = pyo.ConcreteModel(policy_id)
    model.I = pyo.RangeSet(0, n - 1)
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
    exposure = sum(model.x[i] * loan[i] for i in model.I)
    ret = sum(model.x[i] * base_return[i] for i in model.I)
    model.budget = pyo.Constraint(expr=exposure <= BUDGET)
    model.min_budget = pyo.Constraint(expr=exposure >= 0.85 * BUDGET)
    model.pd_cap = pyo.Constraint(
        expr=sum(model.x[i] * loan[i] * pd_high[i] for i in model.I)
        <= risk_tolerance * (exposure + 1e-6)
    )
    if max_weak_share is not None:
        model.weak_cap = pyo.Constraint(
            expr=sum(model.x[i] * loan[i] * weak[i] for i in model.I)
            <= max_weak_share * (exposure + 1e-6)
        )
    scenario_loss = pd.DataFrame()
    if cvar_cap is not None:
        scenarios, loss_matrix = _scenario_loss_matrix(pool)
        model.S = pyo.RangeSet(0, len(scenarios) - 1)
        model.eta = pyo.Var(domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.S, domain=pyo.NonNegativeReals)

        def excess_rule(m, s):
            return m.u[s] >= sum(m.x[i] * loss_matrix[s, i] for i in m.I) - m.eta

        model.excess = pyo.Constraint(model.S, rule=excess_rule)
        beta = 0.90
        cvar_expr = model.eta + (1 / ((1 - beta) * len(scenarios))) * sum(
            model.u[s] for s in model.S
        )
        model.cvar_cap = pyo.Constraint(expr=cvar_expr <= cvar_cap)
    if return_floor is not None:
        model.return_floor = pyo.Constraint(expr=ret >= return_floor)
    model.obj = pyo.Objective(
        expr=sum(model.x[i] * obj_vec[i] for i in model.I), sense=pyo.maximize
    )
    solver = Highs()
    solver.config.time_limit = time_limit
    t0 = time.perf_counter()
    try:
        results = solver.solve(model)
        status = str(getattr(results, "termination_condition", "unknown"))
    except RuntimeError as exc:
        return (
            pd.DataFrame(),
            {
                "policy_id": policy_id,
                "solver_status": f"infeasible_or_no_solution: {str(exc).splitlines()[0]}",
                "elapsed_seconds": time.perf_counter() - t0,
                "n_funded": 0,
                "funded_exposure": 0.0,
                "objective_return": np.nan,
                "realized_return_proxy_lgd45": np.nan,
                "weighted_pd_high": np.nan,
                "weighted_qhat": np.nan,
                "weighted_weak_source_proxy": np.nan,
            },
            scenario_loss,
        )
    allocation = np.array([float(pyo.value(model.x[i])) for i in model.I])
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["policy_id"] = policy_id
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["realized_return_proxy_lgd45"] = funded["funded_exposure"] * funded[
        "int_rate_decimal"
    ].astype(float) * (1 - funded["y_true"].astype(float)) - funded[
        "funded_exposure"
    ] * DEFAULT_LGD * funded["y_true"].astype(float)
    if cvar_cap is not None and hasattr(model, "S"):
        scenarios, loss_matrix = _scenario_loss_matrix(pool)
        rows = []
        for s_idx, (scenario, mult, lgd) in enumerate(scenarios):
            rows.append(
                {
                    "policy_id": policy_id,
                    "scenario": scenario,
                    "pd_multiplier": mult,
                    "lgd": lgd,
                    "portfolio_loss": float(np.dot(allocation, loss_matrix[s_idx])),
                    "cvar_excess_u": float(pyo.value(model.u[s_idx])),
                }
            )
        scenario_loss = pd.DataFrame(rows)
    exposure_sum = float(funded["funded_exposure"].sum())
    metrics = {
        "policy_id": policy_id,
        "solver_status": status,
        "elapsed_seconds": time.perf_counter() - t0,
        "risk_tolerance": risk_tolerance,
        "weak_penalty": weak_penalty,
        "width_penalty": width_penalty,
        "cvar_cap": cvar_cap if cvar_cap is not None else np.nan,
        "return_floor": return_floor if return_floor is not None else np.nan,
        "max_weak_share": max_weak_share if max_weak_share is not None else np.nan,
        "n_funded": int(funded["loan_id"].nunique()),
        "funded_exposure": exposure_sum,
        "objective_return": float(sum(allocation[i] * base_return[i] for i in range(n))),
        "realized_return_proxy_lgd45": float(funded["realized_return_proxy_lgd45"].sum()),
        "weighted_pd_high": float(
            np.average(funded["pd_high_alpha01"], weights=funded["funded_exposure"])
        )
        if exposure_sum
        else np.nan,
        "weighted_qhat": float(np.average(funded["qhat_v4"], weights=funded["funded_exposure"]))
        if exposure_sum
        else np.nan,
        "weighted_weak_source_proxy": float(
            np.average(funded["weak_source_proxy"], weights=funded["funded_exposure"])
        )
        if exposure_sum
        else np.nan,
    }
    if not scenario_loss.empty:
        losses = scenario_loss["portfolio_loss"].to_numpy()
        q = np.quantile(losses, 0.90)
        metrics["scenario_loss_mean"] = float(losses.mean())
        metrics["scenario_loss_max"] = float(losses.max())
        metrics["scenario_loss_cvar90"] = float(losses[losses >= q].mean())
    return funded, metrics, scenario_loss


def build_solver_lanes_v6(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_pool_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(candidate_pool, online_intervals, max_pool_n)
    mdcp_specs = [
        ("v6_mdcp_base_rt170", 0.170, 0.00, 0.00, None),
        ("v6_mdcp_width_rt170", 0.170, 0.02, 0.12, 0.20),
        ("v6_mdcp_source_rt1725", 0.1725, 0.08, 0.08, 0.18),
        ("v6_mdcp_strict_rt175", 0.175, 0.14, 0.10, 0.15),
    ]
    mdcp_allocs = []
    mdcp_rows = []
    for policy_id, rt, weak_penalty, width_penalty, weak_cap in mdcp_specs:
        alloc, metrics, _ = _solve_linear_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=weak_penalty,
            width_penalty=width_penalty,
            max_weak_share=weak_cap,
        )
        mdcp_allocs.append(alloc)
        mdcp_rows.append({**metrics, "solver_lane": "mdcp_source_width_inside_solver"})
    mdcp_alloc = (
        pd.concat([df for df in mdcp_allocs if not df.empty], ignore_index=True)
        if mdcp_allocs
        else pd.DataFrame()
    )
    mdcp_summary = pd.DataFrame(mdcp_rows).sort_values("objective_return", ascending=False)

    cvar_specs = []
    for cap in [360_000.0, 420_000.0, 520_000.0, 650_000.0]:
        for floor in [80_000.0, 110_000.0, 140_000.0]:
            cvar_specs.append((f"v6_cvar_cap{int(cap)}_floor{int(floor)}", 0.1725, cap, floor))
    cvar_allocs = []
    cvar_rows = []
    cvar_losses = []
    for policy_id, rt, cap, floor in cvar_specs:
        alloc, metrics, losses = _solve_linear_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=0.04,
            width_penalty=0.04,
            cvar_cap=cap,
            return_floor=floor,
            max_weak_share=0.35,
            time_limit=120,
        )
        cvar_allocs.append(alloc)
        cvar_rows.append({**metrics, "solver_lane": "expanded_topk_cvar_constraint"})
        if not losses.empty:
            cvar_losses.append(losses)
    nonempty_cvar_allocs = [df for df in cvar_allocs if not df.empty]
    cvar_alloc = (
        pd.concat(nonempty_cvar_allocs, ignore_index=True)
        if nonempty_cvar_allocs
        else pd.DataFrame()
    )
    cvar_summary = pd.DataFrame(cvar_rows).sort_values(
        ["solver_status", "objective_return"], ascending=[True, False]
    )
    cvar_loss = pd.concat(cvar_losses, ignore_index=True) if cvar_losses else pd.DataFrame()
    hybrid_specs = [
        ("v6_hybrid_dfl_proxy_auditability", 0.1725, 0.08, 0.16, 0.35),
        ("v6_hybrid_dfl_proxy_return", 0.1750, 0.03, 0.04, 0.45),
        ("v6_hybrid_dfl_proxy_balanced", 0.1725, 0.05, 0.10, 0.40),
    ]
    hybrid_allocs = []
    hybrid_rows = []
    for policy_id, rt, weak_penalty, width_penalty, weak_cap in hybrid_specs:
        alloc, metrics, _ = _solve_linear_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=weak_penalty,
            width_penalty=width_penalty,
            max_weak_share=weak_cap,
        )
        hybrid_allocs.append(alloc)
        hybrid_rows.append(
            {
                **metrics,
                "solver_lane": "spo_dfl_constrained_surrogate",
                "hybrid_status": "surrogate_training_not_neural_dfl",
                "auditability_constraint_retained": True,
            }
        )
    nonempty_hybrid_allocs = [df for df in hybrid_allocs if not df.empty]
    hybrid_alloc = (
        pd.concat(nonempty_hybrid_allocs, ignore_index=True)
        if nonempty_hybrid_allocs
        else pd.DataFrame()
    )
    hybrid_summary = pd.DataFrame(hybrid_rows).sort_values("objective_return", ascending=False)
    all_alloc = pd.concat(
        [df for df in [mdcp_alloc, cvar_alloc, hybrid_alloc] if not df.empty], ignore_index=True
    )
    all_summary = pd.concat([mdcp_summary, cvar_summary, hybrid_summary], ignore_index=True)
    return all_alloc, all_summary, cvar_summary, cvar_loss, hybrid_summary


def build_dla_v6(
    candidate_pool: pd.DataFrame, online_intervals: pd.DataFrame, *, max_months: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_solver_pool(
        candidate_pool, online_intervals, max_n=min(len(candidate_pool), 8000)
    )
    months = sorted(pool["issue_month"].dropna().unique())[:max_months]
    cash = BUDGET
    outstanding: list[dict[str, Any]] = []
    decision_frames = []
    state_rows = []
    rng = np.random.default_rng(RNG_SEED)
    for t, month in enumerate(months, start=1):
        principal_in = 0.0
        interest_in = 0.0
        realized_loss = 0.0
        expected_loss = 0.0
        next_outstanding = []
        macro = float(rng.normal(0, 0.25))
        for item in outstanding:
            age = t - item["funded_month_idx"] + 1
            term = max(float(item["term"]), 1.0)
            balance = item["exposure"] * max(0.0, 1 - (age - 1) / term)
            monthly_pd = 1 - (1 - min(item["pd_high"] * math.exp(macro), 0.95)) ** (1 / term)
            expected_loss += balance * item["lgd"] * monthly_pd
            default_event = bool(item["y_true"] >= 0.5 and age >= min(12, term))
            if default_event:
                realized_loss += balance * item["lgd"]
                continue
            principal_in += item["exposure"] / term
            interest_in += balance * item["int_rate"] / 12
            if age < term:
                next_outstanding.append(item)
        cash += principal_in + interest_in - realized_loss
        available = pool[pool["issue_month"].eq(month)].copy()
        deployment_budget = max(0.0, min(cash * 0.35, BUDGET * 0.35))
        if available.empty or deployment_budget < 1_000:
            funded = pd.DataFrame()
        else:
            local = available.sort_values("solver_score_seed", ascending=False).copy()
            local["cum_amount"] = local["loan_amnt"].cumsum()
            funded = local[local["cum_amount"].le(deployment_budget)].copy()
            if funded.empty:
                funded = local.head(1).copy()
            funded["policy_id"] = "v6_dla_loan_level_endogenous"
            funded["decision_month"] = month
            funded["month_idx"] = t
            funded["allocation_fraction"] = 1.0
            funded["funded_exposure"] = funded["loan_amnt"]
            funded["action"] = np.where(
                funded["qhat_v4"].gt(0.90),
                "coverage_guarded_fund",
                "fund_by_state_score",
            )
            deployed = float(funded["funded_exposure"].sum())
            cash -= deployed
            for _, row in funded.iterrows():
                outstanding.append(
                    {
                        "loan_id": row["loan_id"],
                        "funded_month_idx": t,
                        "exposure": float(row["funded_exposure"]),
                        "term": float(row["term"]),
                        "int_rate": float(row["int_rate_decimal"]),
                        "pd_high": float(row["pd_high_alpha01"]),
                        "lgd": DEFAULT_LGD,
                        "y_true": float(row["y_true"]),
                    }
                )
            decision_frames.append(
                funded[
                    [
                        "policy_id",
                        "decision_month",
                        "month_idx",
                        "loan_id",
                        "funded_exposure",
                        "qhat_v4",
                        "weak_source_proxy",
                        "pd_high_alpha01",
                        "int_rate_decimal",
                        "action",
                    ]
                ]
            )
        outstanding = next_outstanding + [
            item for item in outstanding if item["funded_month_idx"] == t
        ]
        outstanding_balance = float(sum(item["exposure"] for item in outstanding))
        state_rows.append(
            {
                "policy_id": "v6_dla_loan_level_endogenous",
                "month_idx": t,
                "calendar_month": month,
                "cash_end": cash,
                "principal_in": principal_in,
                "interest_in": interest_in,
                "realized_loss": realized_loss,
                "expected_loss": expected_loss,
                "outstanding_items": len(outstanding),
                "outstanding_balance_proxy": outstanding_balance,
                "state_value_proxy": cash + outstanding_balance - expected_loss,
                "decision_scope": "loan_level_endogenous_monthly_funding",
            }
        )
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    state = pd.DataFrame(state_rows)
    summary = pd.DataFrame(
        [
            {
                "policy_id": "v6_dla_loan_level_endogenous",
                "horizon_months": int(len(months)),
                "funded_loans": int(decisions["loan_id"].nunique()) if not decisions.empty else 0,
                "total_funded_exposure": float(decisions["funded_exposure"].sum())
                if not decisions.empty
                else 0.0,
                "final_cash": float(state["cash_end"].iloc[-1]) if not state.empty else np.nan,
                "final_state_value_proxy": float(state["state_value_proxy"].iloc[-1])
                if not state.empty
                else np.nan,
                "cumulative_realized_loss": float(state["realized_loss"].sum())
                if not state.empty
                else np.nan,
                "cumulative_expected_loss": float(state["expected_loss"].sum())
                if not state.empty
                else np.nan,
                "claim_scope": "endogenous_monthly_loan_selection_proxy_not_full_bellman_solve",
            }
        ]
    )
    return decisions, state, summary


def build_sample_paths_v6(
    allocations: pd.DataFrame, v6_allocations: pd.DataFrame, *, n_paths: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    if not allocations.empty:
        top_policies = (
            allocations.groupby("policy_id", as_index=False)
            .agg(ret=("realized_return_proxy_lgd45", "sum"))
            .sort_values("ret", ascending=False)
            .head(12)["policy_id"]
        )
        pieces.append(allocations[allocations["policy_id"].isin(set(top_policies))].copy())
    if not v6_allocations.empty:
        pieces.append(v6_allocations.copy())
    work = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["loan_id"] = work["loan_id"].astype(str)
    work["issue_month"] = _as_month(work["issue_month"])
    work["funded_exposure"] = pd.to_numeric(work["funded_exposure"], errors="coerce").fillna(
        work["loan_amnt"]
    )
    work["month_ord"] = pd.factorize(work["issue_month"])[0]
    grade_values = sorted(work["original_grade"].astype(str).unique())
    grade_map = {g: i for i, g in enumerate(grade_values)}
    work["grade_ord"] = work["original_grade"].astype(str).map(grade_map).fillna(0).astype(int)
    scenarios = {"baseline": 1.0, "adverse": 1.35, "cycle_stress": 1.75}
    rng = np.random.default_rng(RNG_SEED + 6)
    rows = []
    n_months = int(work["month_ord"].max()) + 1
    for scenario, pd_mult in scenarios.items():
        for path_id in range(n_paths):
            macro = np.zeros(n_months)
            macro[0] = rng.normal(0, 0.35 if scenario == "baseline" else 0.55)
            for t in range(1, n_months):
                macro[t] = 0.70 * macro[t - 1] + rng.normal(
                    0, 0.22 if scenario == "baseline" else 0.35
                )
            grade_shocks = rng.normal(0, 0.16, size=len(grade_map))
            latent = (
                macro[work["month_ord"].to_numpy()] + grade_shocks[work["grade_ord"].to_numpy()]
            )
            pd_path = np.clip(
                work["pd_high_alpha01"].astype(float).to_numpy() * pd_mult * np.exp(latent), 0, 0.98
            )
            u_common = rng.uniform(size=len(work))
            defaults = u_common < pd_path
            lgd_cycle = np.clip(
                DEFAULT_LGD
                * (1 + 0.22 * macro[work["month_ord"].to_numpy()] + rng.normal(0, 0.08, len(work))),
                0.10,
                0.95,
            )
            exposure = work["funded_exposure"].astype(float).to_numpy()
            ret = (
                exposure * work["int_rate_decimal"].astype(float).to_numpy() * (~defaults)
                - exposure * lgd_cycle * defaults
            )
            tmp = work[["policy_id"]].copy()
            tmp["simulated_return_v6"] = ret
            tmp["simulated_loss_v6"] = exposure * lgd_cycle * defaults
            agg = tmp.groupby("policy_id", as_index=False).agg(
                simulated_return_v6=("simulated_return_v6", "sum"),
                simulated_loss_v6=("simulated_loss_v6", "sum"),
            )
            agg["scenario"] = scenario
            agg["path_id"] = path_id
            rows.append(agg)
    paths = pd.concat(rows, ignore_index=True)
    ref_policy = paths["policy_id"].iloc[0]
    if FROZEN_PAPER1_CHAMPION in set(paths["policy_id"]):
        ref_policy = FROZEN_PAPER1_CHAMPION
    ref = paths[paths["policy_id"].eq(ref_policy)][
        ["scenario", "path_id", "simulated_return_v6"]
    ].rename(columns={"simulated_return_v6": "reference_return_v6"})
    pair = paths.merge(ref, on=["scenario", "path_id"], how="left")
    pair["diff_vs_reference_v6"] = pair["simulated_return_v6"] - pair["reference_return_v6"]
    ci = pair.groupby(["policy_id", "scenario"], as_index=False).agg(
        mean_diff_vs_reference_v6=("diff_vs_reference_v6", "mean"),
        p05_diff_vs_reference_v6=("diff_vs_reference_v6", lambda x: float(np.quantile(x, 0.05))),
        p50_diff_vs_reference_v6=("diff_vs_reference_v6", lambda x: float(np.quantile(x, 0.50))),
        p95_diff_vs_reference_v6=("diff_vs_reference_v6", lambda x: float(np.quantile(x, 0.95))),
        prob_beats_reference_v6=(
            "diff_vs_reference_v6",
            lambda x: float(np.mean(np.asarray(x) > 0)),
        ),
        cvar90_simulated_loss_v6=(
            "simulated_loss_v6",
            lambda x: float(np.asarray(x)[np.asarray(x) >= np.quantile(x, 0.90)].mean()),
        ),
        n_paths=("path_id", "nunique"),
    )
    ci["reference_policy_v6"] = ref_policy
    return pair, ci


def build_causal_fairness_v6() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cate = _safe_read_parquet(ROOT / "data" / "processed" / "cate_estimates_oot.parquet")
    if cate.empty:
        cate_screen = pd.DataFrame(
            [{"lane": "cate", "policy_value_allowed": False, "reason": "missing CATE artifact"}]
        )
    else:
        cross = cate["cate_lb"].le(0) & cate["cate_ub"].ge(0)
        cate_screen = pd.DataFrame(
            [
                {
                    "lane": "cate_policy_value",
                    "n": int(len(cate)),
                    "mean_cate": float(cate["cate"].mean()),
                    "share_ci_crosses_zero": float(cross.mean()),
                    "share_positive_lb": float(cate["cate_lb"].gt(0).mean()),
                    "share_negative_ub": float(cate["cate_ub"].lt(0).mean()),
                    "policy_value_allowed": False,
                    "blocking_reason": "intervals mostly cross zero and treatment is observational",
                }
            ]
        )
    outcome_screen = pd.DataFrame(
        [
            (
                "default_12m_or_lifetime",
                "available_proxy",
                "primary but confounded; needs sensitivity",
            ),
            ("prepayment", "partial_proxy", "available only via status/date proxy"),
            ("net_return", "available_proxy", "post-treatment outcome; needs leakage discipline"),
            ("loss_given_default", "available_proxy", "conditional outcome; selection risk"),
            (
                "approval_or_funding",
                "not_available",
                "Lending Club accepted loans do not reveal rejected applicants",
            ),
        ],
        columns=["outcome", "availability", "causal_comment"],
    )
    fairness_protocol = pd.DataFrame(
        [
            {
                "item": "protected_attributes",
                "status": "not_available",
                "decision": "no fair-lending legal claim",
                "condition_to_change": "approved protected attribute source or externally validated proxy protocol",
            },
            {
                "item": "proxy_governance",
                "status": "available",
                "decision": "continue grade/state/income/dti/source stress only",
                "condition_to_change": "committee approves thresholds and review workflow",
            },
            {
                "item": "paper_language",
                "status": "guardrail",
                "decision": "must say proxy governance, not protected fairness compliance",
                "condition_to_change": "none without protected-attribute evidence",
            },
        ]
    )
    external_protocol = pd.DataFrame(
        [
            (
                "race_ethnicity_proxy",
                "blocked",
                "requires legal/IRB/MRM approval; not inferred silently",
            ),
            ("sex_gender_proxy", "blocked", "not available and should not be guessed"),
            ("age_proxy", "blocked", "birth date unavailable; age cannot be reconstructed safely"),
            (
                "geographic_proxy",
                "diagnostic_only",
                "state/zip can be source stress, not protected attribute claim",
            ),
            (
                "adverse_impact_testing",
                "future_protocol",
                "requires protected or approved proxy groups",
            ),
        ],
        columns=["protocol_item", "status", "notes"],
    )
    return cate_screen, outcome_screen, fairness_protocol, external_protocol


def build_claim_matrix_v6() -> pd.DataFrame:
    rows = [
        (
            "Online efficient source-month",
            "implemented_with_audit_oracle_and_deployable_screen",
            "paper4_v6_online_source_month_efficiency_search.csv",
            "19ak-v6-online-ifrs9-efficient.qmd",
            "deployable method may still fail; oracle rows are not promotion eligible",
        ),
        (
            "SICR recalibration",
            "implemented_rule_grid",
            "paper4_v6_sicr_calibration_grid.csv",
            "19ak-v6-online-ifrs9-efficient.qmd",
            "MRM candidate, not audited accounting policy",
        ),
        (
            "Contractual IFRS9 data",
            "implemented_readiness_audit",
            "paper4_v6_contractual_data_audit.csv",
            "19ak-v6-online-ifrs9-efficient.qmd",
            "current data is proxy-only for full servicing",
        ),
        (
            "MDCP inside solver",
            "implemented_linear_constraint_proxy",
            "paper4_v6_mdcp_solver_summary.csv",
            "19al-v6-solvers-dla-sample-paths.qmd",
            "source/width proxy inside LP, not theorem-level MDCP guarantee",
        ),
        (
            "CVaR constraint expanded",
            "implemented_topk_constraint_lp",
            "paper4_v6_cvar_constraint_summary.csv",
            "19al-v6-solvers-dla-sample-paths.qmd",
            "top-k expanded pool, not 276k full LP",
        ),
        (
            "Endogenous DLA",
            "implemented_loan_level_monthly_proxy",
            "paper4_v6_dla_loan_level_summary.csv",
            "19al-v6-solvers-dla-sample-paths.qmd",
            "not full Bellman/ADP",
        ),
        (
            "Realistic sample paths",
            "implemented_correlated_cycle_paths",
            "paper4_v6_correlated_sample_path_ci.csv",
            "19al-v6-solvers-dla-sample-paths.qmd",
            "simulation for paired comparison, not forecast",
        ),
        (
            "CATE gate",
            "implemented_blocker_screen",
            "paper4_v6_cate_gate.csv",
            "19am-v6-causal-fairness-hybrids.qmd",
            "policy value remains blocked",
        ),
        (
            "Fairness governance",
            "implemented_no_legal_claim_protocol",
            "paper4_v6_fairness_protocol.csv",
            "19am-v6-causal-fairness-hybrids.qmd",
            "proxy governance only",
        ),
        (
            "SPO/DFL hybrids",
            "implemented_constrained_surrogate",
            "paper4_v6_spo_dfl_hybrid_summary.csv",
            "19am-v6-causal-fairness-hybrids.qmd",
            "surrogate LP, not trained neural DFL",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def update_manifest_v6(claims: pd.DataFrame) -> None:
    path = TABLE_DIR / "paper4_table0_source_manifest.csv"
    manifest = _safe_read_csv(path)
    if manifest.empty:
        return
    rows = []
    for _, row in claims.iterrows():
        artifact_path = OUT_ROOT / "tables" / row["artifact"]
        rows.append(
            {
                "artifact": str(artifact_path.relative_to(ROOT)),
                "source_paper": "Paper 4 v6",
                "role": row["priority"],
                "status": row["claim_status"],
                "run_tag": "paper4_v6_priority_resolution_2026-05-13",
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
    parser.add_argument("--solver-pool-n", type=int, default=12_000)
    parser.add_argument("--sample-paths", type=int, default=200)
    parser.add_argument("--dla-months", type=int, default=12)
    args = parser.parse_args(list(argv) if argv is not None else None)

    base, candidate_pool, allocations, local_search, online_intervals = _load_inputs()

    online_intervals_v6, online_policy, online_source, online_summary = (
        build_online_source_month_v6(allocations, online_intervals)
    )
    _write_parquet("paper4_v6_online_source_month_intervals.parquet", online_intervals_v6)
    _write_parquet("paper4_v6_online_source_month_policy_month.parquet", online_policy)
    _write_parquet("paper4_v6_online_source_month_source_month.parquet", online_source)
    _write_csv("paper4_v6_online_source_month_efficiency_search.csv", online_summary)

    sicr_grid, sicr_policy = build_sicr_calibration_v6()
    _write_csv("paper4_v6_sicr_calibration_grid.csv", sicr_grid)
    _write_csv("paper4_v6_sicr_policy_summary.csv", sicr_policy)

    data_audit, gaps, readiness = build_contractual_data_audit_v6()
    _write_csv("paper4_v6_contractual_data_audit.csv", data_audit)
    _write_csv("paper4_v6_servicing_gap_register.csv", gaps)
    _write_csv("paper4_v6_contractual_ifrs9_readiness.csv", readiness)

    solver_alloc, solver_summary, cvar_summary, cvar_loss, hybrid_summary = build_solver_lanes_v6(
        candidate_pool, online_intervals, max_pool_n=args.solver_pool_n
    )
    _write_parquet("paper4_v6_solver_allocations.parquet", solver_alloc)
    _write_csv("paper4_v6_solver_summary.csv", solver_summary)
    _write_csv(
        "paper4_v6_mdcp_solver_summary.csv",
        solver_summary[solver_summary["solver_lane"].eq("mdcp_source_width_inside_solver")],
    )
    _write_csv("paper4_v6_cvar_constraint_summary.csv", cvar_summary)
    _write_csv("paper4_v6_cvar_constraint_scenario_losses.csv", cvar_loss)
    _write_csv("paper4_v6_spo_dfl_hybrid_summary.csv", hybrid_summary)

    dla_decisions, dla_state, dla_summary = build_dla_v6(
        candidate_pool, online_intervals, max_months=args.dla_months
    )
    _write_parquet("paper4_v6_dla_loan_level_decisions.parquet", dla_decisions)
    _write_csv("paper4_v6_dla_loan_level_state.csv", dla_state)
    _write_csv("paper4_v6_dla_loan_level_summary.csv", dla_summary)

    paths, path_ci = build_sample_paths_v6(allocations, solver_alloc, n_paths=args.sample_paths)
    _write_parquet("paper4_v6_correlated_sample_paths.parquet", paths)
    _write_csv("paper4_v6_correlated_sample_path_ci.csv", path_ci)

    cate_gate, causal_outcomes, fairness_protocol, external_protocol = build_causal_fairness_v6()
    _write_csv("paper4_v6_cate_gate.csv", cate_gate)
    _write_csv("paper4_v6_causal_outcome_registry.csv", causal_outcomes)
    _write_csv("paper4_v6_fairness_protocol.csv", fairness_protocol)
    _write_csv("paper4_v6_fairness_external_protocol_checklist.csv", external_protocol)

    claims = build_claim_matrix_v6()
    _write_csv("paper4_v6_claim_artifact_matrix.csv", claims)
    update_manifest_v6(claims)

    best_online_deployable = (
        online_summary[online_summary["deployable_without_outcomes"].astype(bool)]
        .sort_values(["gate_pass_80", "avg_width_loan"], ascending=[False, True])
        .iloc[0]
    )
    best_online_any = online_summary.sort_values(
        ["gate_pass_80", "avg_width_loan"], ascending=[False, True]
    ).iloc[0]
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v6_priority_resolution_wave",
        "mode": "paper4_living_lab_no_paper1_changes",
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": False,
        "priorities_targeted": 10,
        "online_best_deployable_method": best_online_deployable["online_method_v6"],
        "online_best_deployable_source_min": float(
            best_online_deployable["coverage_source_month_min"]
        ),
        "online_best_deployable_width": float(best_online_deployable["avg_width_loan"]),
        "online_best_any_method": best_online_any["online_method_v6"],
        "online_best_any_deployable": bool(best_online_any["deployable_without_outcomes"]),
        "online_best_any_source_min": float(best_online_any["coverage_source_month_min"]),
        "online_best_any_width": float(best_online_any["avg_width_loan"]),
        "online_promotion_eligible": bool(online_summary["promotion_eligible"].any()),
        "sicr_candidate_rules": sicr_grid[
            sicr_grid["sicr_recommendation_v6"].eq("candidate_for_mrm_review")
        ]["sicr_rule_v6"]
        .drop_duplicates()
        .tolist()
        if not sicr_grid.empty
        else [],
        "contractual_ifrs9_readiness_score": float(readiness["readiness_score"].iloc[0])
        if not readiness.empty
        else np.nan,
        "mdcp_solver_optimal_count": int(
            solver_summary[
                solver_summary["solver_lane"].eq("mdcp_source_width_inside_solver")
                & solver_summary["solver_status"]
                .astype(str)
                .str.contains("optimal", case=False, na=False)
            ].shape[0]
        ),
        "cvar_solver_optimal_count": int(
            cvar_summary["solver_status"]
            .astype(str)
            .str.contains("optimal", case=False, na=False)
            .sum()
        )
        if not cvar_summary.empty
        else 0,
        "dla_endogenous_implemented": not dla_summary.empty,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "hybrid_surrogate_implemented": not hybrid_summary.empty,
        "generated_artifacts": claims["artifact"].tolist(),
    }
    _write_json("paper4_v6_priority_resolution_status.json", status)
    _write_note(
        "paper4_v6_priority_resolution_memo.qmd",
        """---
title: "Paper 4 v6 Priority Resolution Memo"
format: html
---

# Paper 4 v6 Priority Resolution Memo

V6 implements the ten active priority blockers without changing Paper Estrella.
It separates deployable evidence from diagnostic/oracle evidence, especially in
online source-month conformal calibration.

The key rule remains unchanged: no v6 result is a final promotion.  V6 creates
better search artifacts, solvers and blockers for the next living-lab cycle.
""",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
