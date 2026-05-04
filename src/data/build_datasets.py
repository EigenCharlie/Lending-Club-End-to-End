"""Build analytical datasets used across the credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_GRADE_LEVELS = tuple("ABCDEFG")
_SUBGRADE_LEVELS = tuple(f"{grade}{bucket}" for grade in _GRADE_LEVELS for bucket in range(1, 6))
_PURPOSE_LEVELS = (
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "medical",
)
_VERIFICATION_LEVELS = ("Not Verified", "Source Verified", "Verified")
_HOME_OWNERSHIP_LEVELS = ("RENT", "MORTGAGE", "OWN")
_APPLICATION_LEVELS = ("Individual", "Joint App")


def clean_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean raw Lending Club column formats."""
    df = df.copy()

    def _is_text_dtype(col: pd.Series) -> bool:
        return pd.api.types.is_object_dtype(col.dtype) or pd.api.types.is_string_dtype(col.dtype)

    if "int_rate" in df.columns and _is_text_dtype(df["int_rate"]):
        df["int_rate"] = df["int_rate"].astype(str).str.strip().str.rstrip("%").astype(float)
    if "term" in df.columns and _is_text_dtype(df["term"]):
        df["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if "revol_util" in df.columns and _is_text_dtype(df["revol_util"]):
        df["revol_util"] = df["revol_util"].astype(str).str.strip().str.rstrip("%")
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    if "revol_util" in df.columns and "rev_utilization" not in df.columns:
        df["rev_utilization"] = df["revol_util"] / 100.0
    if "delinq_2yrs" in df.columns and "num_delinq_2yrs" not in df.columns:
        df["num_delinq_2yrs"] = df["delinq_2yrs"]
    if "mths_since_last_delinq" in df.columns and "days_since_last_delinq" not in df.columns:
        df["days_since_last_delinq"] = df["mths_since_last_delinq"] * 30
    if "dti" in df.columns:
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    if "annual_inc" in df.columns:
        df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")
    if "loan_amnt" in df.columns:
        df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    if "default_flag" in df.columns:
        df["default_flag"] = (
            pd.to_numeric(df["default_flag"], errors="coerce").fillna(0).astype(int)
        )
    return df


def build_loan_master(df: pd.DataFrame) -> pd.DataFrame:
    """Build loan-level dataset for PD, LGD, and survival models."""
    feature_cols = [
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "loan_to_income",
        "installment_burden",
        "dti",
        "rev_utilization",
        "revol_bal_to_income",
        "open_acc_ratio",
        "fico_score",
        "credit_age_years",
        "emp_length_num",
        "grade_woe",
        "sub_grade_woe",
        "home_ownership_woe",
        "verification_status_woe",
        "term_woe",
        "int_rate_woe",
        "dti_woe",
        "annual_inc_woe",
        "loan_amnt_woe",
        "fico_score_woe",
        "installment_burden_woe",
        "inq_last_6mths_woe",
        "int_rate_bucket",
        "dti_bucket",
        "fico_bucket",
        "int_rate_bucket__grade",
        "term",
        "grade",
        "sub_grade",
        "home_ownership",
        "purpose",
        "emp_length",
        "verification_status",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "credit_history_months",
        "delinq_severity",
        "delinq_recency",
        "il_ratio",
        "high_util_pct",
        "log_annual_inc",
        "log_revol_bal",
        "has_delinq_2yrs",
        "has_pub_rec",
        "has_bankruptcy",
        "has_recent_inq",
        "has_mortgage",
        "many_recent_opens",
        "recent_chargeoff",
        "bc_util",
        "bc_open_to_buy",
        "percent_bc_gt_75",
        "acc_open_past_24mths",
        "tot_cur_bal",
        "tot_hi_cred_lim",
        "total_bal_ex_mort",
        "total_bc_limit",
        "total_il_high_credit_limit",
        "avg_cur_bal",
        "pct_tl_nvr_dlq",
        "mths_since_recent_bc",
        "num_accts_ever_120_pd",
        "num_actv_bc_tl",
        "num_actv_rev_tl",
        "num_bc_sats",
        "num_bc_tl",
        "num_il_tl",
        "num_op_rev_tl",
        "num_rev_accts",
        "num_rev_tl_bal_gt_0",
        "num_sats",
        "num_tl_30dpd",
        "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m",
        "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl",
        "loan_to_income_sq",
        "fico_x_dti",
    ]
    target_cols = [
        "default_flag",
        "lgd",
        "lgd_months_since_issue",
        "lgd_is_mature_24m",
        "issue_d",
        "loan_status",
    ]
    id_cols = ["id"] if "id" in df.columns else []
    available = [c for c in feature_cols + target_cols + id_cols if c in df.columns]
    loan_master = df[available].copy()
    if "annual_inc" in loan_master.columns:
        annual_inc = pd.to_numeric(loan_master["annual_inc"], errors="coerce")
        zero_income_count = int(annual_inc.eq(0).fillna(False).sum())
        if zero_income_count:
            logger.warning(
                "loan_master contains {} rows with annual_inc == 0; schema accepts non-negative "
                "income, and downstream ratios are handled with safe division.",
                zero_income_count,
            )
    try:
        from src.features.schemas import validate_loan_master

        validate_loan_master(loan_master)
        logger.info("Pandera: loan_master schema validated ✓")
    except Exception as exc:  # noqa: BLE001 — Pandera SchemaError + import/type errors; warn-only gate
        logger.warning("Pandera: loan_master validation failed — {}", exc)
    logger.info("Built loan_master: {}", loan_master.shape)
    return loan_master


def _aggregate_time_series(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frame = df.copy()
    if "default_flag" not in frame.columns:
        frame["default_flag"] = 0
    frame = frame.dropna(subset=["issue_d", "loan_amnt"])
    frame["issue_month"] = pd.to_datetime(frame["issue_d"]).dt.to_period("M").dt.to_timestamp()
    agg_spec: dict[str, tuple[str, str]] = {
        "loan_count": ("loan_amnt", "count"),
        "default_count": ("default_flag", "sum"),
        "total_amt_funded": ("loan_amnt", "sum"),
        "avg_loan_amnt": ("loan_amnt", "mean"),
    }
    if "int_rate" in frame.columns:
        agg_spec["avg_int_rate"] = ("int_rate", "mean")
    if "dti" in frame.columns:
        agg_spec["avg_dti"] = ("dti", "mean")
    grouped = (
        frame.groupby(["issue_month", *group_cols], dropna=False, observed=True)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"issue_month": "ds"})
    )
    if "avg_int_rate" not in grouped.columns:
        grouped["avg_int_rate"] = np.nan
    if "avg_dti" not in grouped.columns:
        grouped["avg_dti"] = np.nan
    grouped["default_rate"] = (
        grouped["default_count"].astype(float) / grouped["loan_count"].replace(0, np.nan)
    ).fillna(0.0)
    return grouped.sort_values(["ds", *group_cols]).reset_index(drop=True)


def _safe_float_series(values: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(default).astype(float)


def _safe_string_series(values: pd.Series | object, default: str = "UNKNOWN") -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(str).replace({"nan": default, "None": default}).fillna(default)
    return pd.Series([default], dtype="object")


def _prepare_time_series_vnext_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = clean_raw_columns(df)
    if "default_flag" not in frame.columns:
        frame["default_flag"] = 0
    frame = frame.dropna(subset=["issue_d", "loan_amnt"]).copy()
    frame["issue_month"] = (
        pd.to_datetime(frame["issue_d"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    frame = frame.dropna(subset=["issue_month"]).copy()

    frame["grade"] = _safe_string_series(
        frame.get("grade", pd.Series(["UNKNOWN"] * len(frame)))
    ).str.upper()
    frame["sub_grade"] = _safe_string_series(
        frame.get("sub_grade", pd.Series(["UNKNOWN"] * len(frame)))
    ).str.upper()
    frame["purpose"] = _safe_string_series(
        frame.get("purpose", pd.Series(["other"] * len(frame))), default="other"
    ).str.lower()
    frame["verification_status"] = _safe_string_series(
        frame.get("verification_status", pd.Series(["UNKNOWN"] * len(frame)))
    )
    frame["home_ownership"] = _safe_string_series(
        frame.get("home_ownership", pd.Series(["UNKNOWN"] * len(frame)))
    ).str.upper()
    frame["application_type"] = _safe_string_series(
        frame.get("application_type", pd.Series(["Individual"] * len(frame))),
        default="Individual",
    )
    frame["term_months"] = pd.to_numeric(frame.get("term"), errors="coerce")
    if "term" in frame.columns and frame["term_months"].isna().all():
        frame["term_months"] = (
            frame["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        )
    frame["term_months"] = frame["term_months"].fillna(-1).astype(int)
    frame["annual_inc"] = _safe_float_series(
        frame.get("annual_inc", pd.Series([np.nan] * len(frame)))
    )
    frame["installment"] = _safe_float_series(
        frame.get("installment", pd.Series([np.nan] * len(frame)))
    )
    frame["dti"] = _safe_float_series(frame.get("dti", pd.Series([np.nan] * len(frame))))
    frame["int_rate"] = _safe_float_series(frame.get("int_rate", pd.Series([np.nan] * len(frame))))
    frame["loan_amnt"] = _safe_float_series(
        frame.get("loan_amnt", pd.Series([np.nan] * len(frame)))
    )
    frame["revol_util"] = _safe_float_series(
        frame.get("revol_util", pd.Series([np.nan] * len(frame)))
    )
    frame["mort_acc"] = _safe_float_series(frame.get("mort_acc", pd.Series([np.nan] * len(frame))))
    frame["inq_last_6mths"] = _safe_float_series(
        frame.get("inq_last_6mths", pd.Series([np.nan] * len(frame)))
    )
    frame["delinq_2yrs"] = _safe_float_series(
        frame.get("delinq_2yrs", pd.Series([np.nan] * len(frame)))
    )
    frame["acc_now_delinq"] = _safe_float_series(
        frame.get("acc_now_delinq", pd.Series([np.nan] * len(frame)))
    )
    frame["num_tl_30dpd"] = _safe_float_series(
        frame.get("num_tl_30dpd", pd.Series([np.nan] * len(frame)))
    )
    frame["num_tl_90g_dpd_24m"] = _safe_float_series(
        frame.get("num_tl_90g_dpd_24m", pd.Series([np.nan] * len(frame)))
    )
    frame["fico_score"] = _safe_float_series(
        frame.get("fico_range_low", pd.Series([np.nan] * len(frame)))
    )
    fico_high = _safe_float_series(frame.get("fico_range_high", pd.Series([np.nan] * len(frame))))
    has_high = fico_high.notna() & (fico_high > 0)
    frame.loc[has_high, "fico_score"] = (
        frame.loc[has_high, "fico_score"].to_numpy(dtype=float)
        + fico_high.loc[has_high].to_numpy(dtype=float)
    ) / 2.0

    frame["share_term_36"] = frame["term_months"].eq(36).astype(float)
    frame["share_term_60"] = frame["term_months"].eq(60).astype(float)
    frame["share_verified"] = frame["verification_status"].eq("Verified").astype(float)
    frame["share_source_verified"] = (
        frame["verification_status"].eq("Source Verified").astype(float)
    )
    frame["share_not_verified"] = frame["verification_status"].eq("Not Verified").astype(float)
    frame["share_home_mortgage"] = frame["home_ownership"].eq("MORTGAGE").astype(float)
    frame["share_home_own"] = frame["home_ownership"].eq("OWN").astype(float)
    frame["share_home_rent"] = frame["home_ownership"].eq("RENT").astype(float)
    frame["share_joint_app"] = frame["application_type"].eq("Joint App").astype(float)

    for grade in _GRADE_LEVELS:
        frame[f"share_grade_{grade}"] = frame["grade"].eq(grade).astype(float)
    for sub_grade in _SUBGRADE_LEVELS:
        frame[f"share_subgrade_{sub_grade}"] = frame["sub_grade"].eq(sub_grade).astype(float)
    for purpose in _PURPOSE_LEVELS:
        frame[f"share_purpose_{purpose}"] = frame["purpose"].eq(purpose).astype(float)
    frame["share_purpose_other"] = (~frame["purpose"].isin(_PURPOSE_LEVELS)).astype(float)
    return frame


def _aggregate_time_series_vnext(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frame = _prepare_time_series_vnext_frame(df)
    agg_spec: dict[str, tuple[str, str | callable]] = {
        "loan_count": ("loan_amnt", "count"),
        "default_count": ("default_flag", "sum"),
        "total_amt_funded": ("loan_amnt", "sum"),
        "avg_loan_amnt": ("loan_amnt", "mean"),
        "std_loan_amnt": ("loan_amnt", "std"),
        "avg_installment": ("installment", "mean"),
        "avg_int_rate": ("int_rate", "mean"),
        "std_int_rate": ("int_rate", "std"),
        "avg_dti": ("dti", "mean"),
        "std_dti": ("dti", "std"),
        "avg_annual_inc": ("annual_inc", "mean"),
        "std_annual_inc": ("annual_inc", "std"),
        "avg_fico_score": ("fico_score", "mean"),
        "std_fico_score": ("fico_score", "std"),
        "avg_revol_util": ("revol_util", "mean"),
        "avg_mort_acc": ("mort_acc", "mean"),
        "avg_inq_last_6mths": ("inq_last_6mths", "mean"),
        "avg_delinq_2yrs": ("delinq_2yrs", "mean"),
        "avg_acc_now_delinq": ("acc_now_delinq", "mean"),
        "avg_num_tl_30dpd": ("num_tl_30dpd", "mean"),
        "avg_num_tl_90g_dpd_24m": ("num_tl_90g_dpd_24m", "mean"),
    }
    share_cols = [
        "share_term_36",
        "share_term_60",
        "share_verified",
        "share_source_verified",
        "share_not_verified",
        "share_home_mortgage",
        "share_home_own",
        "share_home_rent",
        "share_joint_app",
        *[f"share_grade_{grade}" for grade in _GRADE_LEVELS],
        *[f"share_subgrade_{sub_grade}" for sub_grade in _SUBGRADE_LEVELS],
        *[f"share_purpose_{purpose}" for purpose in _PURPOSE_LEVELS],
        "share_purpose_other",
    ]
    for col in share_cols:
        agg_spec[col] = (col, "mean")

    grouped = (
        frame.groupby(["issue_month", *group_cols], dropna=False, observed=True)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"issue_month": "ds"})
    )
    for col in grouped.columns:
        if col.startswith("std_"):
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
    grouped["loan_count"] = pd.to_numeric(grouped["loan_count"], errors="coerce").fillna(0.0)
    grouped["default_count"] = pd.to_numeric(grouped["default_count"], errors="coerce").fillna(0.0)
    grouped["default_rate"] = (
        grouped["default_count"].astype(float) / grouped["loan_count"].replace(0, np.nan)
    ).fillna(0.0)
    grouped["smoothed_default_rate"] = (
        (grouped["default_count"].astype(float) + 0.5) / (grouped["loan_count"].astype(float) + 1.0)
    ).clip(1e-6, 1.0 - 1e-6)
    grouped["default_rate_logit"] = np.log(
        grouped["smoothed_default_rate"] / (1.0 - grouped["smoothed_default_rate"])
    )
    grouped["exposure_loan_count"] = grouped["loan_count"].astype(float)
    grouped["unique_id"] = "portfolio"
    grouped["y"] = grouped["default_rate"].astype(float)
    grouped["y_logit"] = grouped["default_rate_logit"].astype(float)
    return grouped.sort_values(["ds", *group_cols]).reset_index(drop=True)


def _complete_monthly_grid(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str = "ds",
) -> pd.DataFrame:
    """Fill missing monthly timestamps within each series lifecycle.

    For issuance-count panels, missing months inside a series should be represented
    explicitly as zero-volume months rather than irregular timestamps.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).reset_index(drop=True)
    if work.empty:
        return work

    key_cols = list(group_cols)
    numeric_fill_zero = [
        col for col in ["loan_count", "default_count", "total_amt_funded"] if col in work.columns
    ]

    frames: list[pd.DataFrame] = []
    grouped = work.groupby(key_cols, dropna=False, observed=True) if key_cols else [((), work)]
    for keys, part in grouped:
        group_df = part.sort_values(date_col).reset_index(drop=True)
        start = group_df[date_col].min()
        end = group_df[date_col].max()
        if pd.isna(start) or pd.isna(end):
            continue

        full_dates = pd.date_range(start=start, end=end, freq="MS")
        base = pd.DataFrame({date_col: full_dates})
        if key_cols:
            key_values = keys if isinstance(keys, tuple) else (keys,)
            for col, value in zip(key_cols, key_values, strict=False):
                base[col] = value

        merged = base.merge(group_df, on=[date_col, *key_cols], how="left", sort=True)
        for col in numeric_fill_zero:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        if "default_rate" in merged.columns:
            merged["default_rate"] = (
                pd.to_numeric(merged.get("default_count"), errors="coerce").fillna(0.0)
                / pd.to_numeric(merged.get("loan_count"), errors="coerce").replace(0, np.nan)
            ).fillna(0.0)
        frames.append(merged)

    if not frames:
        return work.sort_values([*key_cols, date_col]).reset_index(drop=True)
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values([*key_cols, date_col])
        .reset_index(drop=True)
    )


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical monthly portfolio series for forecasting."""
    ts = _aggregate_time_series(df, [])
    ts = _complete_monthly_grid(ts, group_cols=[])
    ts["unique_id"] = "portfolio"
    ts["y"] = ts["default_rate"].astype(float)
    logger.info("Built time_series: {} ({} -> {})", ts.shape, ts["ds"].min(), ts["ds"].max())
    return ts


def build_time_series_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build coherent panel series for portfolio -> grade -> grade x term."""
    frame = df.copy()
    frame["grade"] = (
        frame.get("grade", pd.Series(["UNKNOWN"] * len(frame))).astype(str).fillna("UNKNOWN")
    )
    frame["term_months"] = pd.to_numeric(frame.get("term"), errors="coerce")
    if "term" in frame.columns and frame["term_months"].isna().all():
        frame["term_months"] = (
            frame["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        )
    frame["term_months"] = frame["term_months"].fillna(-1).astype(int)

    grade_term = _complete_monthly_grid(
        _aggregate_time_series(frame, ["grade", "term_months"]),
        group_cols=["grade", "term_months"],
    )
    grade_term["series_level"] = "grade_term"
    grade_term["unique_id"] = grade_term.apply(
        lambda row: f"grade_term::{row['grade']}__{int(row['term_months'])}", axis=1
    )

    grade = _complete_monthly_grid(_aggregate_time_series(frame, ["grade"]), group_cols=["grade"])
    grade["series_level"] = "grade"
    grade["term_months"] = np.nan
    grade["unique_id"] = grade["grade"].map(lambda grade_value: f"grade::{grade_value}")

    portfolio = build_time_series(frame)
    portfolio["series_level"] = "portfolio"
    portfolio["grade"] = "ALL"
    portfolio["term_months"] = np.nan

    panel = pd.concat([portfolio, grade, grade_term], ignore_index=True, sort=False)
    panel["grade"] = panel.get("grade", pd.Series(["ALL"] * len(panel))).fillna("ALL")
    panel["term_months"] = pd.to_numeric(panel.get("term_months"), errors="coerce")
    panel = panel.sort_values(["series_level", "unique_id", "ds"]).reset_index(drop=True)
    logger.info("Built time_series_panel: {}", panel.shape)
    return panel


def build_time_series_vnext(df: pd.DataFrame) -> pd.DataFrame:
    """Build enriched monthly portfolio series with exposure-aware targets."""
    ts = _aggregate_time_series_vnext(df, [])
    ts = _complete_monthly_grid(ts, group_cols=[])
    numeric_cols = [col for col in ts.columns if col not in {"ds", "unique_id"}]
    ts[numeric_cols] = ts[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ts["unique_id"] = "portfolio"
    ts["default_rate"] = (
        ts["default_count"].astype(float) / ts["loan_count"].replace(0, np.nan)
    ).fillna(0.0)
    ts["smoothed_default_rate"] = (
        (ts["default_count"].astype(float) + 0.5) / (ts["loan_count"].astype(float) + 1.0)
    ).clip(1e-6, 1.0 - 1e-6)
    ts["y"] = ts["default_rate"].astype(float)
    ts["y_logit"] = np.log(
        ts["smoothed_default_rate"].clip(1e-6, 1.0 - 1e-6)
        / (1.0 - ts["smoothed_default_rate"].clip(1e-6, 1.0 - 1e-6))
    )
    logger.info("Built time_series_vnext: {} ({} -> {})", ts.shape, ts["ds"].min(), ts["ds"].max())
    return ts


def build_time_series_panel_vnext(df: pd.DataFrame) -> pd.DataFrame:
    """Build enriched panel series with exposure-aware targets for vNext research."""
    frame = _prepare_time_series_vnext_frame(df)

    grade_term = _complete_monthly_grid(
        _aggregate_time_series_vnext(frame, ["grade", "term_months"]),
        group_cols=["grade", "term_months"],
    )
    grade_term["series_level"] = "grade_term"
    grade_term["unique_id"] = grade_term.apply(
        lambda row: f"grade_term::{row['grade']}__{int(row['term_months'])}", axis=1
    )

    grade = _complete_monthly_grid(
        _aggregate_time_series_vnext(frame, ["grade"]),
        group_cols=["grade"],
    )
    grade["series_level"] = "grade"
    grade["term_months"] = np.nan
    grade["unique_id"] = grade["grade"].map(lambda grade_value: f"grade::{grade_value}")

    portfolio = build_time_series_vnext(frame)
    portfolio["series_level"] = "portfolio"
    portfolio["grade"] = "ALL"
    portfolio["term_months"] = np.nan

    panel = pd.concat([portfolio, grade, grade_term], ignore_index=True, sort=False)
    panel["grade"] = panel.get("grade", pd.Series(["ALL"] * len(panel))).fillna("ALL")
    panel["term_months"] = pd.to_numeric(panel.get("term_months"), errors="coerce")
    numeric_cols = [
        col for col in panel.columns if col not in {"ds", "unique_id", "series_level", "grade"}
    ]
    panel[numeric_cols] = panel[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    panel["default_rate"] = (
        panel["default_count"].astype(float) / panel["loan_count"].replace(0, np.nan)
    ).fillna(0.0)
    panel["smoothed_default_rate"] = (
        (panel["default_count"].astype(float) + 0.5) / (panel["loan_count"].astype(float) + 1.0)
    ).clip(1e-6, 1.0 - 1e-6)
    panel["y"] = panel["default_rate"].astype(float)
    panel["y_logit"] = np.log(
        panel["smoothed_default_rate"].clip(1e-6, 1.0 - 1e-6)
        / (1.0 - panel["smoothed_default_rate"].clip(1e-6, 1.0 - 1e-6))
    )
    panel = panel.sort_values(["series_level", "unique_id", "ds"]).reset_index(drop=True)
    logger.info("Built time_series_panel_vnext: {}", panel.shape)
    return panel


def build_ead_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ead = df[df["default_flag"] == 1].copy()
    try:
        from src.features.schemas import ead_schema

        ead_schema.validate(ead)
        logger.info("Pandera: ead_dataset schema validated ✓")
    except Exception as exc:  # noqa: BLE001 — Pandera SchemaError + import/type errors; warn-only gate
        logger.warning("Pandera: ead_dataset validation failed — {}", exc)
    logger.info("Built ead_dataset: {}", ead.shape)
    return ead


def load_historical_time_series_source(
    input_path: str | Path = "data/processed/train.parquet",
) -> pd.DataFrame:
    """Load full-history splits when available; fallback to the provided input frame."""
    input_frame = pd.read_parquet(input_path)
    split_paths = [
        Path("data/processed/train.parquet"),
        Path("data/processed/calibration.parquet"),
        Path("data/processed/test.parquet"),
    ]
    if all(path.exists() for path in split_paths):
        parts = [pd.read_parquet(path) for path in split_paths]
        full_history = pd.concat(parts, ignore_index=True, sort=False)
        logger.info("Loaded full historical splits for time series: {}", full_history.shape)
        return full_history
    logger.info("Falling back to input-only time series source: {}", input_frame.shape)
    return input_frame


def save_datasets(
    loan_master: pd.DataFrame,
    time_series: pd.DataFrame,
    ead_dataset: pd.DataFrame,
    output_dir: str | Path = "data/processed/",
    *,
    time_series_full: pd.DataFrame | None = None,
    time_series_panel: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loan_master.to_parquet(output_dir / "loan_master.parquet", index=False)
    ead_dataset.to_parquet(output_dir / "ead_dataset.parquet", index=False)
    time_series.to_parquet(output_dir / "time_series.parquet", index=False)
    (time_series_full if time_series_full is not None else time_series).to_parquet(
        output_dir / "time_series_full.parquet", index=False
    )
    if time_series_panel is not None:
        time_series_panel.to_parquet(output_dir / "time_series_panel.parquet", index=False)
    logger.info("Saved analytical datasets to {}", output_dir)


def main(input_path: str = "data/processed/train_fe.parquet", output_dir: str = "data/processed/"):
    from src.utils.io_utils import read_split_with_fe_fallback

    train_df = read_split_with_fe_fallback(input_path)
    logger.info("Loaded training split: {} rows from {}", len(train_df), input_path)

    train_clean = clean_raw_columns(train_df)
    loan_master = build_loan_master(train_clean)
    ead_dataset = build_ead_dataset(train_clean)

    history_df = clean_raw_columns(load_historical_time_series_source(input_path))
    time_series_full = build_time_series(history_df)
    time_series_panel = build_time_series_panel(history_df)

    save_datasets(
        loan_master,
        time_series_full,
        ead_dataset,
        output_dir,
        time_series_full=time_series_full,
        time_series_panel=time_series_panel,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train_fe.parquet")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    main(args.input, args.output)
