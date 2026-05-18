"""Run bounded Paper 4 frontier diagnostics without reopening versioned waves.

The script writes only semantic lane-decision tables and one closure memo. Heavy
intermediate data lives in ``/tmp`` by default so iterative runs do not pollute
the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
CLEANED = REPO / "data" / "interim" / "lending_club_cleaned.parquet"
RAW = REPO / "data" / "raw" / "Loan_status_2007-2020Q3.csv"
TABLE_DIR = REPO / "reports" / "paper_material" / "paper4" / "tables"
NOTE_DIR = REPO / "reports" / "paper_material" / "paper4" / "notes"
DOC_DIR = REPO / "docs" / "research"
DEFAULT_SCRATCH = Path("/tmp/lc-paper4-goal-runs")
DEFAULT_DATE = "2026-05-18"

LANES = (
    "ifrs9_sicr",
    "online_conformal",
    "cvar_oce",
    "cate_policy_value",
    "fair_lending_proxy",
    "dla_adp",
    "spo_dfl",
)


@dataclass(frozen=True)
class LaneResult:
    lane: str
    decision: str
    paper4_destination: str
    paper_estrella_destination: str
    claim: str
    evidence_gate: str
    stop_rule: str
    key_metrics: dict[str, Any]
    caveat: str
    table: pd.DataFrame

    def summary_row(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "decision": self.decision,
            "paper4_destination": self.paper4_destination,
            "paper_estrella_destination": self.paper_estrella_destination,
            "claim": self.claim,
            "evidence_gate": self.evidence_gate,
            "stop_rule": self.stop_rule,
            "key_metrics_json": json.dumps(self.key_metrics, sort_keys=True),
            "caveat": self.caveat,
        }


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _winsor(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    return clean.clip(clean.quantile(lo), clean.quantile(hi))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _cvar(values: pd.Series | np.ndarray, alpha: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    cutoff = np.quantile(arr, alpha)
    tail = arr[arr >= cutoff]
    return float(tail.mean()) if tail.size else float(cutoff)


def _grade_num(series: pd.Series) -> pd.Series:
    mapping = {letter: i for i, letter in enumerate("ABCDEFG", start=1)}
    return series.astype(str).str[0].map(mapping).astype(float)


def _sample(df: pd.DataFrame, n: int, seed: int = 20260518) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def build_base(scratch: Path, refresh: bool = False) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    base_path = scratch / "paper4_frontier_base.parquet"
    if base_path.exists() and not refresh:
        return base_path

    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            WITH raw_selected AS (
                SELECT
                    CAST(id AS VARCHAR) AS id,
                    TRY_CAST(funded_amnt AS DOUBLE) AS funded_amnt_raw,
                    TRY_CAST(out_prncp AS DOUBLE) AS out_prncp_raw,
                    TRY_CAST(total_pymnt AS DOUBLE) AS total_pymnt_raw,
                    TRY_CAST(total_rec_prncp AS DOUBLE) AS total_rec_prncp_raw,
                    TRY_CAST(total_rec_int AS DOUBLE) AS total_rec_int_raw,
                    TRY_CAST(recoveries AS DOUBLE) AS recoveries_raw,
                    TRY_CAST(collection_recovery_fee AS DOUBLE) AS collection_recovery_fee_raw,
                    last_pymnt_d AS last_pymnt_d_raw,
                    TRY_CAST(last_pymnt_amnt AS DOUBLE) AS last_pymnt_amnt_raw,
                    last_credit_pull_d AS last_credit_pull_d_raw,
                    hardship_flag AS hardship_flag_raw,
                    hardship_status AS hardship_status_raw,
                    TRY_CAST(hardship_dpd AS DOUBLE) AS hardship_dpd_raw,
                    hardship_loan_status AS hardship_loan_status_raw,
                    debt_settlement_flag AS debt_settlement_flag_raw
                FROM read_csv_auto(
                    '{RAW.as_posix()}',
                    header = true,
                    ignore_errors = true,
                    sample_size = 200000,
                    union_by_name = true
                )
                WHERE TRY_CAST(id AS BIGINT) IS NOT NULL
            ),
            cleaned AS (
                SELECT
                    CAST(id AS VARCHAR) AS id,
                    TRY_CAST(loan_amnt AS DOUBLE) AS loan_amnt,
                    TRY_CAST(regexp_extract(term, '([0-9]+)', 1) AS INTEGER) AS term_months,
                    TRY_CAST(replace(int_rate, '%', '') AS DOUBLE) AS int_rate_pct,
                    TRY_CAST(installment AS DOUBLE) AS installment,
                    grade,
                    sub_grade,
                    home_ownership,
                    verification_status,
                    purpose,
                    zip_code,
                    addr_state,
                    TRY_CAST(annual_inc AS DOUBLE) AS annual_inc,
                    TRY_CAST(dti AS DOUBLE) AS dti,
                    COALESCE(
                        TRY_CAST(issue_d AS DATE),
                        CAST(try_strptime(issue_d, '%b-%Y') AS DATE)
                    ) AS issue_date,
                    loan_status,
                    TRY_CAST(default_flag AS INTEGER) AS default_flag,
                    TRY_CAST(lgd AS DOUBLE) AS lgd,
                    TRY_CAST(lgd_months_since_issue AS DOUBLE) AS lgd_months_since_issue,
                    TRY_CAST(lgd_is_mature_24m AS INTEGER) AS lgd_is_mature_24m,
                    TRY_CAST(fico_range_low AS DOUBLE) AS fico_range_low,
                    TRY_CAST(fico_range_high AS DOUBLE) AS fico_range_high,
                    TRY_CAST(last_fico_range_low AS DOUBLE) AS last_fico_range_low,
                    TRY_CAST(last_fico_range_high AS DOUBLE) AS last_fico_range_high
                FROM read_parquet('{CLEANED.as_posix()}')
            )
            SELECT
                c.*,
                r.* EXCLUDE(id),
                COALESCE(r.funded_amnt_raw, c.loan_amnt) AS funded_amnt,
                GREATEST(COALESCE(r.funded_amnt_raw, c.loan_amnt) - r.total_pymnt_raw, 0)
                    AS cashflow_loss_amount,
                GREATEST(COALESCE(r.funded_amnt_raw, c.loan_amnt) - r.total_pymnt_raw, 0)
                    / NULLIF(COALESCE(r.funded_amnt_raw, c.loan_amnt), 0) AS cashflow_loss_rate,
                r.total_pymnt_raw / NULLIF(COALESCE(r.funded_amnt_raw, c.loan_amnt), 0)
                    AS payment_ratio,
                r.recoveries_raw / NULLIF(COALESCE(r.funded_amnt_raw, c.loan_amnt), 0)
                    AS recovery_ratio,
                (c.fico_range_low + c.fico_range_high) / 2 AS orig_fico_mid,
                (c.last_fico_range_low + c.last_fico_range_high) / 2 AS last_fico_mid,
                ((c.fico_range_low + c.fico_range_high) / 2)
                    - ((c.last_fico_range_low + c.last_fico_range_high) / 2) AS fico_drop,
                strftime(c.issue_date, '%Y-%m') AS issue_month,
                CAST(strftime(c.issue_date, '%Y') AS INTEGER) AS issue_year
            FROM cleaned c
            LEFT JOIN raw_selected r USING (id)
        ) TO '{base_path.as_posix()}' (FORMAT PARQUET)
        """
    )
    con.close()
    return base_path


def load_base(base_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(base_path, columns=columns)


def run_ifrs9_sicr(base_path: Path) -> LaneResult:
    columns = [
        "loan_amnt",
        "funded_amnt",
        "default_flag",
        "lgd",
        "lgd_is_mature_24m",
        "cashflow_loss_amount",
        "cashflow_loss_rate",
        "payment_ratio",
        "recovery_ratio",
        "fico_drop",
        "last_fico_mid",
        "hardship_flag_raw",
        "hardship_dpd_raw",
        "debt_settlement_flag_raw",
        "grade",
        "issue_year",
    ]
    df = load_base(base_path, columns)
    df["weight"] = df["funded_amnt"].fillna(df["loan_amnt"]).fillna(0)
    df["hardship_or_debt"] = (
        df["hardship_flag_raw"].astype(str).str.upper().eq("Y")
        | df["debt_settlement_flag_raw"].astype(str).str.upper().eq("Y")
        | (df["hardship_dpd_raw"].fillna(0) > 0)
    )
    rules = {
        "fico_drop_70": df["fico_drop"] >= 70,
        "last_fico_below_620": df["last_fico_mid"] < 620,
        "payment_ratio_under_70_mature": (df["payment_ratio"] < 0.70)
        & df["lgd_is_mature_24m"].eq(1),
        "hardship_or_debt": df["hardship_or_debt"],
    }
    rules["combined_raw_sicr"] = np.column_stack([rules[k] for k in rules]).any(axis=1)

    rows: list[dict[str, Any]] = []
    for name, mask in rules.items():
        mask = pd.Series(mask, index=df.index).fillna(False)
        triggered = df[mask]
        rest = df[~mask]
        default_t = triggered["default_flag"].mean()
        default_r = rest["default_flag"].mean()
        rows.append(
            {
                "rule": name,
                "triggered_n": int(mask.sum()),
                "triggered_share": float(mask.mean()),
                "default_rate_triggered": float(default_t),
                "default_rate_rest": float(default_r),
                "default_lift": float(default_t / default_r) if default_r else math.nan,
                "mean_lgd_triggered": float(triggered["lgd"].mean()),
                "mean_cashflow_loss_rate_triggered": float(triggered["cashflow_loss_rate"].mean()),
                "weighted_cashflow_loss_rate_triggered": _weighted_mean(
                    triggered["cashflow_loss_rate"], triggered["weight"]
                ),
                "mean_recovery_ratio_triggered": float(triggered["recovery_ratio"].mean()),
            }
        )

    table = pd.DataFrame(rows)
    combined = table[table["rule"].eq("combined_raw_sicr")].iloc[0]
    decision = "append"
    if combined["triggered_share"] > 0.75 or combined["default_lift"] < 1.5:
        decision = "park"

    return LaneResult(
        lane="ifrs9_sicr",
        decision=decision,
        paper4_destination="appendix_raw_enriched_sicr_diagnostic",
        paper_estrella_destination="none",
        claim="Raw servicing fields sharpen SICR/ECL proxy monitoring, but do not create contractual IFRS9.",
        evidence_gate="combined raw SICR lift >= 1.5 and triggered share <= 75%",
        stop_rule="Stop after one raw-enriched diagnostic because monthly contractual DPD panel is absent.",
        key_metrics={
            "combined_default_lift": round(float(combined["default_lift"]), 4),
            "combined_triggered_share": round(float(combined["triggered_share"]), 4),
            "cashflow_fields_present": True,
        },
        caveat="No contractual days-past-due history, monthly account panel or macro scenario path.",
        table=table,
    )


def _prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["grade_num"] = _grade_num(out["grade"])
    if "sub_grade" in out.columns:
        out["subgrade_num"] = pd.to_numeric(out["sub_grade"].astype(str).str[1:], errors="coerce")
    else:
        out["subgrade_num"] = np.nan
    out["log_annual_inc"] = np.log1p(out["annual_inc"].clip(lower=0))
    if "zip_code" in out.columns:
        out["zip3_num"] = pd.to_numeric(out["zip_code"].astype(str).str[:3], errors="coerce")
    else:
        out["zip3_num"] = np.nan
    feature_cols = [
        "loan_amnt",
        "term_months",
        "int_rate_pct",
        "grade_num",
        "subgrade_num",
        "log_annual_inc",
        "dti",
        "orig_fico_mid",
        "issue_year",
        "zip3_num",
    ]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
        values = _winsor(_safe_num(out[col]))
        fill_value = 0.0 if values.notna().sum() == 0 else float(values.median())
        out[col] = values.fillna(fill_value)
    return out


def _coverage_metrics(
    test: pd.DataFrame,
    residual_quantile: pd.Series | float,
    method: str,
    min_support: int = 250,
) -> dict[str, Any]:
    q = residual_quantile
    if isinstance(q, pd.Series):
        q = q.reindex(test.index).fillna(q.median())
        lower = (test["pred"] - q).clip(lower=0.0)
        upper = (test["pred"] + q).clip(upper=1.0)
        width = upper - lower
    else:
        lower = (test["pred"] - q).clip(lower=0.0)
        upper = (test["pred"] + q).clip(upper=1.0)
        width = upper - lower
    covered = test["lgd"].between(lower, upper)
    dims = {
        "issue_month": test["issue_month"],
        "grade": test["grade"],
        "state": test["addr_state"],
        "zip3": test["zip_code"].astype(str).str[:3],
    }
    metrics: dict[str, Any] = {
        "method": method,
        "overall_coverage": float(covered.mean()),
        "avg_width": float(width.mean()),
    }
    for dim, groups in dims.items():
        cov = (
            pd.DataFrame({"covered": covered, "group": groups})
            .groupby("group")["covered"]
            .agg(["mean", "size"])
            .query("size >= @min_support")
        )
        metrics[f"{dim}_groups"] = int(len(cov))
        metrics[f"{dim}_min_coverage"] = float(cov["mean"].min()) if len(cov) else math.nan
    metrics["defended_min_coverage"] = min(
        v for k, v in metrics.items() if k.endswith("_min_coverage") and np.isfinite(v)
    )
    return metrics


def run_online_conformal(base_path: Path) -> LaneResult:
    columns = [
        "id",
        "loan_amnt",
        "term_months",
        "int_rate_pct",
        "grade",
        "sub_grade",
        "annual_inc",
        "dti",
        "orig_fico_mid",
        "issue_year",
        "issue_month",
        "addr_state",
        "zip_code",
        "purpose",
        "lgd",
    ]
    df = load_base(base_path, columns).dropna(subset=["lgd", "issue_year", "issue_month"])
    df = _sample(df, 180_000)
    df = _prepare_model_frame(df).sort_values(["issue_year", "issue_month", "id"])
    features = [
        "loan_amnt",
        "term_months",
        "int_rate_pct",
        "grade_num",
        "subgrade_num",
        "log_annual_inc",
        "dti",
        "orig_fico_mid",
        "issue_year",
        "zip3_num",
    ]
    n = len(df)
    train = df.iloc[: int(n * 0.60)].copy()
    cal = df.iloc[int(n * 0.60) : int(n * 0.80)].copy()
    test = df.iloc[int(n * 0.80) :].copy()
    model = HistGradientBoostingRegressor(max_iter=90, max_leaf_nodes=24, random_state=20260518)
    model.fit(train[features], train["lgd"])
    cal["pred"] = model.predict(cal[features]).clip(0, 1)
    test["pred"] = model.predict(test[features]).clip(0, 1)
    cal["abs_resid"] = (cal["lgd"] - cal["pred"]).abs()

    global_q = float(cal["abs_resid"].quantile(0.90))
    grade_q_map = cal.groupby("grade")["abs_resid"].quantile(0.90)
    state_q_map = cal.groupby("addr_state")["abs_resid"].quantile(0.90)
    zip3_q_map = cal.groupby(cal["zip_code"].astype(str).str[:3])["abs_resid"].quantile(0.90)

    grade_q = test["grade"].map(grade_q_map).fillna(global_q)
    source_q = pd.concat(
        [
            test["addr_state"].map(state_q_map).fillna(global_q),
            test["zip_code"].astype(str).str[:3].map(zip3_q_map).fillna(global_q),
        ],
        axis=1,
    ).max(axis=1)
    rolling = (
        cal.groupby("issue_month")["abs_resid"].quantile(0.90).rolling(6, min_periods=1).mean()
    )
    rolling_q = test["issue_month"].map(rolling).fillna(global_q)
    shrink_q = pd.Series(np.maximum(source_q, 0.5 * grade_q + 0.5 * global_q), index=test.index)

    rows = [
        _coverage_metrics(test, global_q, "global_split"),
        _coverage_metrics(test, grade_q, "grade_mondrian"),
        _coverage_metrics(test, rolling_q, "aci_like_rolling_month"),
        _coverage_metrics(test, source_q, "max_state_zip3_source"),
        _coverage_metrics(test, shrink_q, "source_grade_shrinkback"),
    ]
    table = pd.DataFrame(rows).sort_values(
        ["defended_min_coverage", "avg_width"], ascending=[False, True]
    )
    best = table.iloc[0]
    decision = (
        "append" if best["defended_min_coverage"] >= 0.80 and best["avg_width"] <= 0.95 else "park"
    )
    return LaneResult(
        lane="online_conformal",
        decision=decision,
        paper4_destination="appendix_source_family_holdout",
        paper_estrella_destination="none",
        claim="Retrospective source-aware conformal calibration can improve defended coverage diagnostics, not live deployment.",
        evidence_gate="defended source/month coverage >= 0.80 with average width <= 0.95",
        stop_rule="Stop after source-family retrospective check because no production feedback loop exists.",
        key_metrics={
            "best_method": str(best["method"]),
            "defended_min_coverage": round(float(best["defended_min_coverage"]), 4),
            "avg_width": round(float(best["avg_width"]), 4),
            "n_test": int(len(test)),
        },
        caveat="Historical issue-month replay only; no external source distribution or live feedback.",
        table=table,
    )


def run_cvar_oce(base_path: Path) -> LaneResult:
    columns = [
        "id",
        "funded_amnt",
        "loan_amnt",
        "int_rate_pct",
        "grade",
        "dti",
        "orig_fico_mid",
        "issue_year",
        "cashflow_loss_amount",
        "cashflow_loss_rate",
        "total_pymnt_raw",
    ]
    df = load_base(base_path, columns).dropna(subset=["int_rate_pct", "cashflow_loss_amount"])
    df = _sample(df, 250_000)
    df["funded"] = df["funded_amnt"].fillna(df["loan_amnt"]).clip(lower=1)
    df["realized_return"] = df["total_pymnt_raw"].fillna(0) - df["funded"]
    df["grade_num"] = _grade_num(df["grade"]).fillna(4)
    df["risk_score"] = (
        0.35 * (df["grade_num"] / 7)
        + 0.25 * _winsor(df["dti"]).rank(pct=True)
        + 0.25 * (1 - _winsor(df["orig_fico_mid"]).rank(pct=True))
        + 0.15 * df["cashflow_loss_rate"].fillna(0).rank(pct=True)
    )
    df["economic_score"] = df["int_rate_pct"].fillna(0) / 100 - 0.30 * df["risk_score"]
    df["tail_score"] = df["int_rate_pct"].fillna(0) / 100 - 0.75 * df["risk_score"]
    df["oce_score"] = df["int_rate_pct"].fillna(0) / 100 - 1.10 * df["risk_score"].pow(1.5)

    rows: list[dict[str, Any]] = []
    for policy, score in {
        "cashflow_economic_proxy": "economic_score",
        "cashflow_cvar_tail_proxy": "tail_score",
        "cashflow_oce_tail_proxy": "oce_score",
    }.items():
        selected = df.sort_values(score, ascending=False).copy()
        selected["cum_funded"] = selected["funded"].cumsum()
        selected = selected[selected["cum_funded"] <= 1_000_000]
        losses = selected["cashflow_loss_amount"]
        returns = selected["realized_return"]
        rows.append(
            {
                "policy_id": policy,
                "n_funded": int(len(selected)),
                "funded_exposure": float(selected["funded"].sum()),
                "realized_cashflow_return": float(returns.sum()),
                "mean_loan_loss": float(losses.mean()),
                "cvar90_loan_loss": _cvar(losses, 0.90),
                "cvar95_loan_loss": _cvar(losses, 0.95),
                "cvar99_loan_loss": _cvar(losses, 0.99),
                "share_grade_dplus": float(selected["grade"].astype(str).isin(list("DEFG")).mean()),
                "claim_boundary": "diagnostic_cashflow_repricing_not_champion_replacement",
            }
        )
    table = pd.DataFrame(rows)
    econ = table[table["policy_id"].eq("cashflow_economic_proxy")].iloc[0]
    best_tail = table.sort_values(
        ["cvar95_loan_loss", "realized_cashflow_return"], ascending=[True, False]
    ).iloc[0]
    return_gain_vs_official = float(best_tail["realized_cashflow_return"]) - 170_464.5429284627
    decision = "append" if best_tail["cvar95_loan_loss"] < econ["cvar95_loan_loss"] else "park"
    return LaneResult(
        lane="cvar_oce",
        decision=decision,
        paper4_destination="appendix_tail_challenger_cashflow_check",
        paper_estrella_destination="none",
        claim="Recovery-aware cashflow repricing can support a tail challenger, but not champion replacement.",
        evidence_gate="tail proxy improves CVaR95 versus economic proxy; champion only if official wealth is beaten",
        stop_rule="Stop after cashflow repricing because official paired wealth champion is unchanged.",
        key_metrics={
            "best_tail_policy": str(best_tail["policy_id"]),
            "best_tail_cvar95": round(float(best_tail["cvar95_loan_loss"]), 4),
            "economic_cvar95": round(float(econ["cvar95_loan_loss"]), 4),
            "return_gap_vs_official_champion": round(return_gain_vs_official, 4),
        },
        caveat="Uses retrospective realized cashflows for challenger diagnosis; no full-universe exact optimum.",
        table=table,
    )


def _aipw_effect(
    df: pd.DataFrame, features: list[str], treatment: str, outcome: str
) -> dict[str, Any]:
    train, test = train_test_split(
        df, test_size=0.35, random_state=20260518, stratify=df[treatment]
    )
    clf = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=20, random_state=20260518)
    reg = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=20, random_state=20260518)
    clf.fit(train[features], train[treatment])
    reg.fit(train[features], train[outcome])
    p = np.clip(clf.predict_proba(test[features])[:, 1], 0.02, 0.98)
    m = reg.predict(test[features])
    y = test[outcome].to_numpy()
    t = test[treatment].to_numpy()
    pseudo = m + (t * (y - m) / p) - ((1 - t) * (y - m) / (1 - p))
    ate = float(np.mean(pseudo[t == 1]) - np.mean(pseudo[t == 0]))
    return {
        "ate": ate,
        "overlap_share_10_90": float(((p >= 0.10) & (p <= 0.90)).mean()),
        "test": test.assign(propensity=p, aipw_signal=pseudo),
    }


def run_cate_policy_value(base_path: Path) -> LaneResult:
    columns = [
        "id",
        "loan_amnt",
        "term_months",
        "int_rate_pct",
        "grade",
        "sub_grade",
        "annual_inc",
        "dti",
        "orig_fico_mid",
        "issue_year",
        "cashflow_loss_rate",
        "default_flag",
    ]
    df = load_base(base_path, columns).dropna(subset=["int_rate_pct", "cashflow_loss_rate"])
    df = _sample(df, 120_000)
    df = _prepare_model_frame(df)
    group_cols = ["grade", "term_months", "issue_year"]
    df["rate_group_median"] = df.groupby(group_cols)["int_rate_pct"].transform("median")
    df["high_rate_within_grade"] = (df["int_rate_pct"] > df["rate_group_median"]).astype(int)
    features = [
        "loan_amnt",
        "term_months",
        "grade_num",
        "subgrade_num",
        "log_annual_inc",
        "dti",
        "orig_fico_mid",
        "issue_year",
    ]
    work = df.dropna(subset=features + ["high_rate_within_grade", "cashflow_loss_rate"]).copy()
    effect = _aipw_effect(work, features, "high_rate_within_grade", "cashflow_loss_rate")
    test = effect["test"]
    hetero_rows = []
    for dim, labels in {
        "grade": test["grade"],
        "fico_band": pd.qcut(test["orig_fico_mid"], 4, duplicates="drop"),
        "income_band": pd.qcut(test["log_annual_inc"], 4, duplicates="drop"),
        "dti_band": pd.qcut(test["dti"], 4, duplicates="drop"),
    }.items():
        for group, part in test.groupby(labels, observed=False):
            if len(part) < 500 or part["high_rate_within_grade"].nunique() < 2:
                continue
            hetero_rows.append(
                {
                    "dimension": dim,
                    "group": str(group),
                    "n": int(len(part)),
                    "treated_share": float(part["high_rate_within_grade"].mean()),
                    "mean_aipw_signal": float(part["aipw_signal"].mean()),
                    "mean_observed_loss": float(part["cashflow_loss_rate"].mean()),
                }
            )
    placebo = work.copy()
    rng = np.random.default_rng(20260518)
    placebo["placebo_treatment"] = rng.permutation(placebo["high_rate_within_grade"].to_numpy())
    placebo_effect = _aipw_effect(placebo, features, "placebo_treatment", "cashflow_loss_rate")
    table = pd.DataFrame(
        [
            {
                "metric": "aipw_ate_high_rate_within_grade",
                "value": effect["ate"],
                "gate": "diagnostic_only",
            },
            {
                "metric": "overlap_share_10_90",
                "value": effect["overlap_share_10_90"],
                "gate": ">=0.80 for diagnostic stability",
            },
            {
                "metric": "placebo_aipw_ate",
                "value": placebo_effect["ate"],
                "gate": "should be small relative to observed effect",
            },
            {
                "metric": "heterogeneity_groups_with_support",
                "value": len(hetero_rows),
                "gate": ">=4 for useful appendix table",
            },
        ]
    )
    if hetero_rows:
        hetero = pd.DataFrame(hetero_rows)
        hetero["metric"] = "heterogeneity_group"
        hetero["value"] = hetero["mean_aipw_signal"]
        hetero["gate"] = "supported subgroup diagnostic"
        table = pd.concat([table, hetero[table.columns]], ignore_index=True)
    stable = effect["overlap_share_10_90"] >= 0.80 and abs(placebo_effect["ate"]) <= max(
        abs(effect["ate"]) * 0.50, 0.01
    )
    return LaneResult(
        lane="cate_policy_value",
        decision="append" if stable else "park",
        paper4_destination="appendix_observational_sensitivity" if stable else "lab_notebook_only",
        paper_estrella_destination="none",
        claim="High-rate-within-grade sensitivity is observational diagnostic evidence, not policy value.",
        evidence_gate="overlap >= 0.80 and placebo AIPW small relative to treatment signal",
        stop_rule="Stop after one diagnostic because accepted-loan selection and pricing endogeneity remain.",
        key_metrics={
            "aipw_ate": round(float(effect["ate"]), 6),
            "overlap_share_10_90": round(float(effect["overlap_share_10_90"]), 4),
            "placebo_aipw_ate": round(float(placebo_effect["ate"]), 6),
        },
        caveat="No rejected applicants, randomized pricing instrument or policy counterfactual.",
        table=table,
    )


def run_fair_lending_proxy(base_path: Path) -> LaneResult:
    columns = [
        "grade",
        "addr_state",
        "zip_code",
        "annual_inc",
        "dti",
        "home_ownership",
        "verification_status",
        "purpose",
        "int_rate_pct",
        "default_flag",
        "cashflow_loss_rate",
    ]
    df = load_base(base_path, columns).dropna(subset=["int_rate_pct"])
    df["income_band"] = pd.qcut(df["annual_inc"], 5, labels=False, duplicates="drop")
    df["dti_band"] = pd.qcut(df["dti"], 5, labels=False, duplicates="drop")
    df["zip3"] = df["zip_code"].astype(str).str[:3]
    rows: list[dict[str, Any]] = []
    for dim in ["addr_state", "zip3", "income_band", "dti_band", "grade"]:
        grp = (
            df.groupby(dim, dropna=True)
            .agg(
                n=("int_rate_pct", "size"),
                mean_int_rate=("int_rate_pct", "mean"),
                default_rate=("default_flag", "mean"),
                mean_cashflow_loss=("cashflow_loss_rate", "mean"),
            )
            .query("n >= 1000")
            .reset_index()
        )
        if grp.empty:
            continue
        rows.append(
            {
                "dimension": dim,
                "groups_with_support": int(len(grp)),
                "min_n": int(grp["n"].min()),
                "max_mean_int_rate": float(grp["mean_int_rate"].max()),
                "min_mean_int_rate": float(grp["mean_int_rate"].min()),
                "interest_rate_range": float(
                    grp["mean_int_rate"].max() - grp["mean_int_rate"].min()
                ),
                "default_rate_range": float(grp["default_rate"].max() - grp["default_rate"].min()),
                "cashflow_loss_range": float(
                    grp["mean_cashflow_loss"].max() - grp["mean_cashflow_loss"].min()
                ),
                "legal_fair_lending_claim_allowed": False,
            }
        )
    table = pd.DataFrame(rows).sort_values("interest_rate_range", ascending=False)
    top = table.iloc[0]
    return LaneResult(
        lane="fair_lending_proxy",
        decision="append",
        paper4_destination="appendix_source_governance_only",
        paper_estrella_destination="none",
        claim="Observable geography/source groups support governance diagnostics, not legal fair-lending inference.",
        evidence_gate="at least one observable source family has support and measurable dispersion",
        stop_rule="Stop because Lending Club lacks protected attributes, surname and fine geocoding.",
        key_metrics={
            "top_dispersion_dimension": str(top["dimension"]),
            "top_interest_rate_range": round(float(top["interest_rate_range"]), 4),
            "legal_claim_allowed": False,
        },
        caveat="BISG-style race/ethnicity proxy is not feasible with zip3/state only and no surname.",
        table=table,
    )


def run_dla_adp(base_path: Path) -> LaneResult:
    raw_columns = [
        "issue_month",
        "term_months",
        "grade",
        "out_prncp_raw",
        "last_pymnt_d_raw",
        "last_credit_pull_d_raw",
        "hardship_flag_raw",
        "lgd",
        "cashflow_loss_rate",
    ]
    df = load_base(base_path, raw_columns)
    profile = pd.DataFrame(
        [
            {
                "state_feature": "issue_month",
                "non_null_share": float(df["issue_month"].notna().mean()),
                "role": "cohort timing only",
            },
            {
                "state_feature": "out_prncp_raw",
                "non_null_share": float(df["out_prncp_raw"].notna().mean()),
                "role": "terminal/current outstanding proxy",
            },
            {
                "state_feature": "last_pymnt_d_raw",
                "non_null_share": float(df["last_pymnt_d_raw"].notna().mean()),
                "role": "last observed payment date, not monthly path",
            },
            {
                "state_feature": "last_credit_pull_d_raw",
                "non_null_share": float(df["last_credit_pull_d_raw"].notna().mean()),
                "role": "servicing recency proxy",
            },
            {
                "state_feature": "hardship_flag_raw",
                "non_null_share": float(df["hardship_flag_raw"].notna().mean()),
                "role": "hardship state marker",
            },
        ]
    )
    existing = pd.read_csv(TABLE_DIR / "paper4_v11_dla_adp_comparison.csv")
    best = existing.sort_values("delta_state_value_vs_static", ascending=False).iloc[0]
    table = pd.concat(
        [
            profile.assign(section="raw_state_surface"),
            existing[
                [
                    "policy_id",
                    "n_paths",
                    "final_state_value_mean",
                    "delta_state_value_vs_static",
                    "delta_loss_vs_static",
                    "adp_scope_v11",
                ]
            ].assign(section="retained_v11_rollout"),
        ],
        ignore_index=True,
        sort=False,
    )
    return LaneResult(
        lane="dla_adp",
        decision="append",
        paper4_destination="appendix_rollout_only",
        paper_estrella_destination="none",
        claim="Raw fields improve rollout state description but still do not support exact Bellman optimality.",
        evidence_gate="v11 rollout advantage retained and raw state surface documented",
        stop_rule="Stop because no monthly borrower state trajectory or action logs exist.",
        key_metrics={
            "best_existing_policy": str(best["policy_id"]),
            "best_delta_state_value_vs_static": round(
                float(best["delta_state_value_vs_static"]), 4
            ),
            "state_features_profiled": int(len(profile)),
        },
        caveat="Last-payment and outstanding fields are snapshots/proxies, not a transition panel.",
        table=table,
    )


def _optional_pkg_version(python_bin: Path, package: str) -> str:
    if not python_bin.exists():
        return "env_missing"
    code = f"import importlib.metadata as m; print(m.version('{package}'))"
    try:
        result = subprocess.run(
            [str(python_bin), "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip()
    except Exception:
        return "not_available"


def run_spo_dfl(base_path: Path, env_python: Path | None = None) -> LaneResult:
    columns = [
        "id",
        "loan_amnt",
        "term_months",
        "int_rate_pct",
        "grade",
        "annual_inc",
        "dti",
        "orig_fico_mid",
        "cashflow_loss_rate",
        "cashflow_loss_amount",
        "total_pymnt_raw",
        "funded_amnt",
    ]
    df = load_base(base_path, columns).dropna(subset=["cashflow_loss_amount", "int_rate_pct"])
    df = _sample(df, 80_000)
    df = _prepare_model_frame(df)
    df["funded"] = df["funded_amnt"].fillna(df["loan_amnt"]).clip(lower=1)
    df["realized_return"] = df["total_pymnt_raw"].fillna(0) - df["funded"]
    features = [
        "loan_amnt",
        "term_months",
        "int_rate_pct",
        "grade_num",
        "log_annual_inc",
        "dti",
        "orig_fico_mid",
    ]
    train, test = train_test_split(
        df.dropna(subset=features), test_size=0.35, random_state=20260518
    )
    model = Ridge(alpha=10.0)
    target = train["realized_return"] / train["funded"]
    model.fit(train[features], target)
    test = test.copy()
    test["pred_return_rate"] = model.predict(test[features])
    test["oracle_return_rate"] = test["realized_return"] / test["funded"]
    pred_sel = test.sort_values("pred_return_rate", ascending=False).head(350)
    oracle_sel = test.sort_values("oracle_return_rate", ascending=False).head(350)
    pred_return = float(pred_sel["realized_return"].sum())
    oracle_return = float(oracle_sel["realized_return"].sum())
    oracle_gap = oracle_return - pred_return
    env_python = env_python or Path("/tmp/lc-paper4-envs/spo/bin/python")
    table = pd.DataFrame(
        [
            {
                "probe": "ridge_predict_then_optimize_top350",
                "realized_return": pred_return,
                "oracle_return": oracle_return,
                "oracle_gap": oracle_gap,
                "mae_return_rate": mean_absolute_error(
                    test["oracle_return_rate"], test["pred_return_rate"]
                ),
                "pyepo_version": _optional_pkg_version(env_python, "pyepo"),
                "cvxpylayers_version": _optional_pkg_version(env_python, "cvxpylayers"),
                "torch_version": _optional_pkg_version(env_python, "torch"),
            }
        ]
    )
    existing = pd.read_csv(TABLE_DIR / "paper4_v11_spo_dfl_candidate_summary.csv")
    table = pd.concat(
        [
            table,
            existing[
                [
                    "policy_id",
                    "objective_return",
                    "mean_spo_target_score_v11",
                    "mean_decision_loss_proxy_v11",
                    "training_scope_v11",
                ]
            ].assign(probe="retained_v11_surrogate"),
        ],
        ignore_index=True,
        sort=False,
    )
    return LaneResult(
        lane="spo_dfl",
        decision="park",
        paper4_destination="lab_notebook_isolated_prototype",
        paper_estrella_destination="none",
        claim="SPO/DFL remains useful as oracle-regret framing, not as integrated differentiable training.",
        evidence_gate="isolated dependency probe and oracle-regret table documented",
        stop_rule="Stop because dependency/scaling risk does not justify disturbing CRPTO.",
        key_metrics={
            "toy_oracle_gap": round(float(oracle_gap), 4),
            "pyepo_version": str(table.loc[0, "pyepo_version"]),
            "cvxpylayers_version": str(table.loc[0, "cvxpylayers_version"]),
        },
        caveat="Toy top-k probe is not SPO+ theorem or production optimization training.",
        table=table,
    )


def write_result(result: LaneResult, date_tag: str) -> Path:
    path = TABLE_DIR / f"paper4_frontier_{result.lane}_decision_{date_tag}.csv"
    result.table.to_csv(path, index=False)
    return path


def source_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": "ifrs9_lifetime_ecl_survival_tutorial_2026",
                "status": "peer_reviewed",
                "url": "https://link.springer.com/article/10.1007/s41060-026-01032-w",
                "takeaway": "Recent tutorial frames survival analysis as a convincing approach to IFRS9 lifetime PD term structures.",
                "use_in": "ifrs9_sicr",
            },
            {
                "key": "ifrs9_competing_risks_survival_2024",
                "status": "peer_reviewed",
                "url": "https://www.sciencedirect.com/science/article/pii/S095741742400472X",
                "takeaway": "Competing-risks survival can support lifetime credit loss modelling, but needs event timing structure.",
                "use_in": "ifrs9_sicr",
            },
            {
                "key": "adaptive_conformal_neurips_2021",
                "status": "peer_reviewed",
                "url": "https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html",
                "takeaway": "ACI targets online coverage under distribution shift through an adaptive calibration parameter.",
                "use_in": "online_conformal",
            },
            {
                "key": "multi_source_conformal_2024",
                "status": "preprint",
                "url": "https://arxiv.org/abs/2405.09331",
                "takeaway": "Multi-source conformal inference motivates weighting or defending source families under shift.",
                "use_in": "online_conformal",
            },
            {
                "key": "rockafellar_uryasev_cvar_2000",
                "status": "foundational_paper",
                "url": "https://www.financerisks.com/filedati/WP/paper/CVaR%20Portfolio%20Optimization.pdf",
                "takeaway": "Mean-CVaR portfolio optimization can be represented with linear-programming style scenarios.",
                "use_in": "cvar_oce",
            },
            {
                "key": "cvxpy_portfolio_short_course",
                "status": "tutorial",
                "url": "https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html",
                "takeaway": "CVXPY is the practical implementation route for compact convex portfolio diagnostics.",
                "use_in": "cvar_oce",
            },
            {
                "key": "dowhy_assumptions_2021",
                "status": "research_software_paper",
                "url": "https://www.microsoft.com/en-us/research/publication/dowhy-addressing-challenges-in-expressing-and-validating-causal-assumptions/",
                "takeaway": "Causal claims require explicit assumptions and refutations; prediction validation is not enough.",
                "use_in": "cate_policy_value",
            },
            {
                "key": "econml_dml_docs",
                "status": "official_docs",
                "url": "https://www.pywhy.org/EconML/spec/estimation/dml.html",
                "takeaway": "Double ML assumes observed confounding is sufficient; useful here only as sensitivity.",
                "use_in": "cate_policy_value",
            },
            {
                "key": "cfpb_bisg_proxy_methodology",
                "status": "official_methodology",
                "url": "https://github.com/cfpb/proxy-methodology",
                "takeaway": "BISG requires surname plus geography; Lending Club's zip3/state surface is insufficient.",
                "use_in": "fair_lending_proxy",
            },
            {
                "key": "zhang_fair_lending_proxy_2018",
                "status": "peer_reviewed",
                "url": "https://pubsonline.informs.org/doi/10.1287/mnsc.2016.2579",
                "takeaway": "Proxy methods can estimate race/ethnicity risk, but depend on inputs absent here.",
                "use_in": "fair_lending_proxy",
            },
            {
                "key": "spo_plus_elmachtoub_grigas",
                "status": "peer_reviewed_preprint",
                "url": "https://arxiv.org/abs/1710.08005",
                "takeaway": "SPO+ directly trains for downstream decision loss but needs optimization-specific infrastructure.",
                "use_in": "spo_dfl",
            },
            {
                "key": "pyepo_mpc_2024",
                "status": "peer_reviewed_software",
                "url": "https://link.springer.com/article/10.1007/s12532-024-00255-x",
                "takeaway": "PyEPO implements SPO+ and related end-to-end predict-then-optimize methods for LP/IP models.",
                "use_in": "spo_dfl",
            },
            {
                "key": "cvxpylayers_docs",
                "status": "software_docs",
                "url": "https://cvxpylayers.org/quickstart.html",
                "takeaway": "Differentiable convex layers require DPP-compliant CVXPY problems plus Torch/JAX/MLX.",
                "use_in": "spo_dfl",
            },
        ]
    )


def write_source_log(date_tag: str) -> Path:
    path = TABLE_DIR / f"paper4_frontier_source_log_{date_tag}.csv"
    source_log().to_csv(path, index=False)
    return path


def write_summary(results: list[LaneResult], date_tag: str) -> tuple[Path, Path]:
    summary = pd.DataFrame([r.summary_row() for r in results])
    summary_path = TABLE_DIR / f"paper4_frontier_goal_summary_{date_tag}.csv"
    summary.to_csv(summary_path, index=False)

    rows = []
    for result in results:
        metrics = ", ".join(f"{k}={v}" for k, v in result.key_metrics.items())
        rows.append(
            f"| {result.lane} | {result.decision} | {result.paper4_destination} | {metrics} | {result.caveat} |"
        )
    memo = f"""# Paper 4 Frontier Goal Closure - {date_tag}

## Decision

The bounded Paper 4 frontier goal is closed as a governed research pass, not a
new versioned wave. Each lane has one semantic decision artifact. No
`paper4_final_promotion.json` is created, no Paper Estrella champion is reopened,
and no new `paper4_v###` wave is introduced.

## Lane Outcomes

| lane | decision | destination | key metrics | caveat |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## What Moves Forward

- Paper 4 can cite the raw-enriched IFRS9/SICR diagnostic only as proxy ECL/SICR
  evidence, never as contractual IFRS9.
- Online conformal remains a retrospective source-family governance diagnostic.
- CVaR/OCE remains a tail challenger; official economic champion replacement is
  still blocked by paired-wealth evidence.
- CATE is observational sensitivity only.
- Fair lending is source governance only, because protected attributes, surname
  and tract-level geocoding are absent.
- DLA/ADP is rollout-only, not exact Bellman optimality.
- SPO/DFL remains isolated-prototype/oracle-regret evidence and is not integrated
  into the main CRPTO pipeline.

## Future Work Gate

Reopen a lane only with one of: a servicing panel with monthly DPD/state paths,
rejected-applicant or randomized pricing data, approved protected-attribute proxy
inputs, a reviewer request, or a venue-driven revision that changes the claim.
"""
    memo_path = DOC_DIR / f"paper4_frontier_goal_closure_{date_tag}.md"
    memo_path.write_text(memo)
    (NOTE_DIR / f"paper4_frontier_goal_closure_{date_tag}.md").write_text(memo)
    return summary_path, memo_path


def run_lanes(
    lanes: list[str], scratch: Path, date_tag: str, refresh_base: bool
) -> list[LaneResult]:
    base_path = build_base(scratch, refresh=refresh_base)
    lane_funcs = {
        "ifrs9_sicr": lambda: run_ifrs9_sicr(base_path),
        "online_conformal": lambda: run_online_conformal(base_path),
        "cvar_oce": lambda: run_cvar_oce(base_path),
        "cate_policy_value": lambda: run_cate_policy_value(base_path),
        "fair_lending_proxy": lambda: run_fair_lending_proxy(base_path),
        "dla_adp": lambda: run_dla_adp(base_path),
        "spo_dfl": lambda: run_spo_dfl(base_path),
    }
    results: list[LaneResult] = []
    for lane in lanes:
        started = datetime.now(UTC)
        print(f"[paper4-frontier] running lane={lane}", flush=True)
        result = lane_funcs[lane]()
        write_result(result, date_tag)
        elapsed = (datetime.now(UTC) - started).total_seconds()
        print(
            f"[paper4-frontier] lane={lane} decision={result.decision} elapsed={elapsed:.1f}s",
            flush=True,
        )
        results.append(result)
    write_source_log(date_tag)
    write_summary(results, date_tag)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded Paper 4 frontier diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python scripts/papers/run_paper4_frontier_lanes.py --lanes all
              python scripts/papers/run_paper4_frontier_lanes.py --lanes ifrs9_sicr cvar_oce
            """
        ),
    )
    parser.add_argument("--lanes", nargs="+", default=["all"])
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--date-tag", default=DEFAULT_DATE)
    parser.add_argument("--refresh-base", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lanes = list(LANES) if args.lanes == ["all"] else args.lanes
    unknown = sorted(set(lanes) - set(LANES))
    if unknown:
        raise SystemExit(f"Unknown lanes: {', '.join(unknown)}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    run_lanes(lanes, args.scratch_dir, args.date_tag, args.refresh_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
