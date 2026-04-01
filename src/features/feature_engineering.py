"""Canonical feature engineering and feature-artifact helpers for PD reruns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.pipeline_runtime import (
    atomic_write_parquet,
    atomic_write_pickle,
    atomic_write_text,
)

TARGET = "default_flag"

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "loan_to_income",
    "installment_burden",
    "rev_utilization",
    "revol_bal_to_income",
    "open_acc_ratio",
    "fico_score",
    "credit_age_years",
    "emp_length_num",
    "open_acc",
    "total_acc",
    "revol_bal",
    "pub_rec",
    "inq_last_6mths",
    "mort_acc",
    "delinq_severity",
    "delinq_recency",
    "il_ratio",
    "high_util_pct",
    "log_annual_inc",
    "log_revol_bal",
    "loan_to_income_sq",
    "fico_x_dti",
]
FLAG_FEATURES = [
    "has_delinq_2yrs",
    "has_pub_rec",
    "has_bankruptcy",
    "has_recent_inq",
    "has_mortgage",
    "many_recent_opens",
    "recent_chargeoff",
]
CATEGORICAL_FEATURES = [
    "grade",
    "sub_grade",
    "home_ownership",
    "purpose",
    "verification_status",
    "term",
    "int_rate_bucket",
    "dti_bucket",
    "fico_bucket",
]
WOE_SOURCE_FEATURES = [
    "grade",
    "sub_grade",
    "purpose",
    "home_ownership",
    "verification_status",
    "term",
    "int_rate",
    "dti",
    "annual_inc",
    "loan_amnt",
    "fico_score",
    "installment_burden",
    "inq_last_6mths",
]
WOE_FEATURES = [f"{feature}_woe" for feature in WOE_SOURCE_FEATURES]
INTERACTION_FEATURES = ["int_rate_bucket__grade"]
CATBOOST_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES + INTERACTION_FEATURES
LOGREG_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + WOE_FEATURES

HIGH_COVERAGE_BUREAU_FEATURES = [
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
]
MEDIUM_COVERAGE_CHALLENGER_FEATURES = [
    "inq_fi",
    "inq_last_12m",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "open_rv_12m",
    "open_rv_24m",
    "mths_since_recent_inq",
    "mths_since_last_delinq",
]
EXCLUDED_CORE_PATTERNS = ("sec_app_",)
EXCLUDED_CORE_FEATURES = {"verification_status_joint"}
ID_COLUMNS = ["id"]
META_COLUMNS = ["issue_d", "loan_status"]
MISSINGNESS_INDICATOR_SUFFIX = "__is_missing"
SURVIVAL_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + HIGH_COVERAGE_BUREAU_FEATURES


@dataclass
class WOEEncoderArtifact:
    """Serializable WOE encoder metadata for one feature."""

    feature: str
    kind: str
    default_woe: float
    iv: float
    mapping: dict[str, float] | None = None
    bin_edges: list[float] | None = None


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Lending Club columns before derived features."""
    out = df.copy()
    if "int_rate" in out.columns and not pd.api.types.is_numeric_dtype(out["int_rate"]):
        out["int_rate"] = (
            out["int_rate"]
            .astype(str)
            .str.strip()
            .str.rstrip("%")
            .pipe(pd.to_numeric, errors="coerce")
        )
    if "term" in out.columns and not pd.api.types.is_numeric_dtype(out["term"]):
        out["term"] = (
            out["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        )
    if "revol_util" in out.columns and not pd.api.types.is_numeric_dtype(out["revol_util"]):
        out["revol_util"] = (
            out["revol_util"]
            .astype(str)
            .str.strip()
            .str.rstrip("%")
            .pipe(pd.to_numeric, errors="coerce")
        )
    numeric_candidates = [
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "revol_bal",
        "revol_util",
        "open_acc",
        "total_acc",
        "pub_rec",
        "inq_last_6mths",
        "mort_acc",
        "delinq_2yrs",
        "mths_since_last_delinq",
        "mths_since_recent_inq",
        "collections_12_mths_ex_med",
        "chargeoff_within_12_mths",
        "pub_rec_bankruptcies",
        "tot_cur_bal",
        "tot_hi_cred_lim",
        "total_bal_ex_mort",
        "total_bc_limit",
        "total_il_high_credit_limit",
        "avg_cur_bal",
        "acc_open_past_24mths",
        "bc_util",
        "bc_open_to_buy",
        "percent_bc_gt_75",
        "pct_tl_nvr_dlq",
        "mths_since_recent_bc",
        "inq_fi",
        "inq_last_12m",
        "open_acc_6m",
        "open_act_il",
        "open_il_12m",
        "open_il_24m",
        "open_rv_12m",
        "open_rv_24m",
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
        "fico_range_low",
        "fico_range_high",
        "all_util",
        "total_bal_il",
    ]
    out = _coerce_numeric(out, numeric_candidates)
    if "issue_d" in out.columns:
        out["issue_d"] = pd.to_datetime(out["issue_d"], errors="coerce")
    if "earliest_cr_line" in out.columns:
        out["earliest_cr_line"] = pd.to_datetime(out["earliest_cr_line"], errors="coerce")
    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    return out


def _safe_divide(
    num: pd.Series | None,
    den: pd.Series | None,
    *,
    default: float = np.nan,
) -> pd.Series:
    if num is None and den is None:
        return pd.Series(dtype=float)
    index = (
        num.index
        if isinstance(num, pd.Series)
        else den.index
        if isinstance(den, pd.Series)
        else None
    )
    if index is None:
        return pd.Series(dtype=float)
    den_clean = (
        pd.to_numeric(den, errors="coerce") if den is not None else pd.Series(np.nan, index=index)
    )
    num_clean = (
        pd.to_numeric(num, errors="coerce") if num is not None else pd.Series(np.nan, index=index)
    )
    result = np.where(den_clean > 0, num_clean / den_clean, default)
    return pd.Series(result, index=index, dtype=float)


def _parse_emp_length(series: pd.Series) -> pd.Series:
    def _parse(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        raw = str(value).strip().lower()
        if raw in {"n/a", "nan", "none", ""}:
            return np.nan
        if raw.startswith("<"):
            return 0.0
        if "10+" in raw:
            return 10.0
        match = re.search(r"(\d+)", raw)
        return float(match.group(1)) if match else np.nan

    return series.map(_parse).astype(float)


def _ensure_fico_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fico_score" in out.columns:
        out["fico_score"] = pd.to_numeric(out["fico_score"], errors="coerce")
        return out
    low = out.get("fico_range_low")
    high = out.get("fico_range_high")
    if isinstance(low, pd.Series) and isinstance(high, pd.Series):
        low = pd.to_numeric(low, errors="coerce")
        high = pd.to_numeric(high, errors="coerce")
        out["fico_score"] = ((low.fillna(high) + high.fillna(low)) / 2.0).astype(float)
    return out


def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial ratio features."""
    out = normalize_raw_columns(df)
    out["loan_to_income"] = _safe_divide(out.get("loan_amnt"), out.get("annual_inc"))

    if "rev_utilization" not in out.columns:
        if "revol_util" in out.columns:
            out["rev_utilization"] = out["revol_util"] / 100.0
        elif "revol_bal" in out.columns and "total_rev_hi_lim" in out.columns:
            out["rev_utilization"] = _safe_divide(out["revol_bal"], out["total_rev_hi_lim"])

    if "installment" in out.columns and "annual_inc" in out.columns:
        out["installment_burden"] = _safe_divide(out["installment"] * 12.0, out["annual_inc"])
    if "revol_bal" in out.columns and "annual_inc" in out.columns:
        out["revol_bal_to_income"] = _safe_divide(out["revol_bal"], out["annual_inc"])
    if "open_acc" in out.columns and "total_acc" in out.columns:
        out["open_acc_ratio"] = _safe_divide(out["open_acc"], out["total_acc"])

    if "num_il_tl" in out.columns and "total_acc" in out.columns:
        out["il_ratio"] = _safe_divide(out["num_il_tl"], out["total_acc"], default=0.5)
    elif "total_bal_il" in out.columns and "total_bal_ex_mort" in out.columns:
        out["il_ratio"] = _safe_divide(out["total_bal_il"], out["total_bal_ex_mort"], default=0.0)

    if "percent_bc_gt_75" in out.columns:
        out["high_util_pct"] = pd.to_numeric(out["percent_bc_gt_75"], errors="coerce") / 100.0
    elif "all_util" in out.columns:
        out["high_util_pct"] = pd.to_numeric(out["all_util"], errors="coerce") / 100.0

    if "annual_inc" in out.columns:
        out["log_annual_inc"] = np.log1p(
            pd.to_numeric(out["annual_inc"], errors="coerce").clip(lower=0)
        )
    if "revol_bal" in out.columns:
        out["log_revol_bal"] = np.log1p(
            pd.to_numeric(out["revol_bal"], errors="coerce").clip(lower=0)
        )

    logger.info("Created ratio features")
    return out


def create_credit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create credit history, delinquency and account-behavior features."""
    out = _ensure_fico_score(normalize_raw_columns(df))
    if "emp_length" in out.columns:
        out["emp_length_num"] = _parse_emp_length(out["emp_length"])
    if "earliest_cr_line" in out.columns and "issue_d" in out.columns:
        out["credit_history_months"] = (
            (out["issue_d"] - out["earliest_cr_line"]).dt.days / 30.44
        ).clip(lower=0)
        out["credit_age_years"] = out["credit_history_months"] / 12.0
    recency_source = (
        "mths_since_last_delinq"
        if "mths_since_last_delinq" in out.columns
        else "mths_since_delinq"
        if "mths_since_delinq" in out.columns
        else None
    )
    if recency_source is not None:
        out["delinq_recency"] = pd.to_numeric(out[recency_source], errors="coerce").fillna(999.0)

    delinquency_components = [
        col
        for col in [
            "delinq_2yrs",
            "collections_12_mths_ex_med",
            "chargeoff_within_12_mths",
            "tax_liens",
        ]
        if col in out.columns
    ]
    if delinquency_components:
        out["delinq_severity"] = (
            out[delinquency_components]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        )
    logger.info("Created credit-history features")
    return out


def create_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary governance-friendly flags."""
    out = normalize_raw_columns(df)
    if "delinq_2yrs" in out.columns:
        out["has_delinq_2yrs"] = (out["delinq_2yrs"] > 0).astype(int)
    if "pub_rec" in out.columns:
        out["has_pub_rec"] = (out["pub_rec"] > 0).astype(int)
    if "pub_rec_bankruptcies" in out.columns:
        out["has_bankruptcy"] = (out["pub_rec_bankruptcies"] > 0).astype(int)
    elif "pub_rec" in out.columns:
        out["has_bankruptcy"] = 0
    if "inq_last_6mths" in out.columns:
        out["has_recent_inq"] = (out["inq_last_6mths"] > 0).astype(int)
    if "mort_acc" in out.columns:
        out["has_mortgage"] = (out["mort_acc"] > 0).astype(int)
    if "num_tl_op_past_12m" in out.columns:
        out["many_recent_opens"] = (out["num_tl_op_past_12m"] >= 3).astype(int)
    elif "acc_open_past_24mths" in out.columns:
        out["many_recent_opens"] = (out["acc_open_past_24mths"] >= 4).astype(int)
    if "chargeoff_within_12_mths" in out.columns:
        out["recent_chargeoff"] = (out["chargeoff_within_12_mths"] > 0).astype(int)
    elif "collections_12_mths_ex_med" in out.columns:
        out["recent_chargeoff"] = (out["collections_12_mths_ex_med"] > 0).astype(int)
    logger.info("Created flag features")
    return out


def create_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Create bucketed features for interpretability."""
    out = _ensure_fico_score(normalize_raw_columns(df))
    if "int_rate" in out.columns:
        out["int_rate_bucket"] = pd.cut(
            out["int_rate"],
            bins=[0, 8, 12, 16, 20, 100],
            labels=["very_low", "low", "medium", "high", "very_high"],
            include_lowest=True,
        )
    if "dti" in out.columns:
        out["dti_bucket"] = pd.cut(
            out["dti"],
            bins=[-1, 10, 20, 30, 40, 100],
            labels=["low", "moderate", "high", "very_high", "extreme"],
            include_lowest=True,
        )
    if "fico_score" in out.columns:
        out["fico_bucket"] = pd.cut(
            out["fico_score"],
            bins=[300, 580, 670, 740, 800, 900],
            labels=["poor", "fair", "good", "very_good", "excellent"],
            include_lowest=True,
        )
    logger.info("Created bucket features")
    return out


def create_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    out = df.copy()
    if "int_rate_bucket" in out.columns and "grade" in out.columns:
        out["int_rate_bucket__grade"] = (
            out["int_rate_bucket"].astype("string").fillna("missing")
            + "__"
            + out["grade"].astype("string").fillna("missing")
        )
    if "loan_to_income" in out.columns:
        out["loan_to_income_sq"] = pd.to_numeric(out["loan_to_income"], errors="coerce") ** 2
    if "fico_score" in out.columns and "dti" in out.columns:
        out["fico_x_dti"] = pd.to_numeric(out["fico_score"], errors="coerce") * pd.to_numeric(
            out["dti"], errors="coerce"
        )
    logger.info("Created interaction features")
    return out


def add_missingness_indicators(
    df: pd.DataFrame,
    features: list[str] = MEDIUM_COVERAGE_CHALLENGER_FEATURES,
) -> pd.DataFrame:
    """Create challenger-only missingness indicators."""
    out = df.copy()
    for feature in features:
        if feature in out.columns:
            out[f"{feature}{MISSINGNESS_INDICATOR_SUFFIX}"] = out[feature].isna().astype(int)
    return out


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    out = normalize_raw_columns(df)
    out = create_ratios(out)
    out = create_credit_history_features(out)
    out = create_flags(out)
    out = create_buckets(out)
    out = create_interactions(out)
    out = add_missingness_indicators(out)
    logger.info("Feature pipeline complete: {}", out.shape)
    return out


def _woe_counts(
    y_true: pd.Series,
    groups: pd.Series,
) -> tuple[dict[str, float], float]:
    tmp = pd.DataFrame({"group": groups.astype(str), TARGET: y_true.astype(int)})
    agg = tmp.groupby("group", dropna=False, observed=True)[TARGET].agg(["count", "sum"])
    agg["bad"] = agg["sum"].astype(float)
    agg["good"] = agg["count"].astype(float) - agg["bad"]
    good_total = float(agg["good"].sum())
    bad_total = float(agg["bad"].sum())
    mapping: dict[str, float] = {}
    iv = 0.0
    for key, row in agg.iterrows():
        good_dist = (float(row["good"]) + 0.5) / max(good_total + 0.5 * len(agg), 1e-9)
        bad_dist = (float(row["bad"]) + 0.5) / max(bad_total + 0.5 * len(agg), 1e-9)
        woe = float(math.log(good_dist / bad_dist))
        mapping[str(key)] = woe
        iv += float((good_dist - bad_dist) * woe)
    return mapping, float(iv)


def fit_woe_encoder(
    df: pd.DataFrame,
    feature: str,
    *,
    target: str = TARGET,
    max_bins: int = 6,
) -> WOEEncoderArtifact:
    """Fit a deterministic train-only WOE encoder."""
    x = df[feature]
    y = df[target].astype(int)

    if pd.api.types.is_numeric_dtype(x):
        series = pd.to_numeric(x, errors="coerce")
        non_null = series.dropna()
        if non_null.nunique(dropna=True) >= 4:
            q = max(2, min(max_bins, int(non_null.nunique(dropna=True))))
            try:
                _, edges = pd.qcut(non_null, q=q, duplicates="drop", retbins=True)
                edges = np.unique(edges.astype(float))
            except ValueError:
                edges = np.array([], dtype=float)
        else:
            edges = np.array([], dtype=float)

        if len(edges) >= 2:
            grouped = pd.cut(series, bins=edges, include_lowest=True).astype("string")
        else:
            grouped = series.round(4).astype("string")
        grouped = grouped.fillna("__MISSING__")
        mapping, iv = _woe_counts(y, grouped)
        return WOEEncoderArtifact(
            feature=feature,
            kind="numerical",
            default_woe=float(mapping.get("__MISSING__", 0.0)),
            iv=iv,
            mapping=mapping,
            bin_edges=edges.astype(float).tolist() if len(edges) >= 2 else None,
        )

    grouped = x.astype("string").fillna("__MISSING__")
    mapping, iv = _woe_counts(y, grouped)
    return WOEEncoderArtifact(
        feature=feature,
        kind="categorical",
        default_woe=float(mapping.get("__MISSING__", 0.0)),
        iv=iv,
        mapping=mapping,
        bin_edges=None,
    )


def apply_woe_encoder(df: pd.DataFrame, encoder: WOEEncoderArtifact) -> pd.Series:
    """Apply a persisted WOE encoder to a split."""
    if encoder.feature not in df.columns:
        return pd.Series(
            np.full(len(df), encoder.default_woe, dtype=float),
            index=df.index,
            name=f"{encoder.feature}_woe",
        )

    x = df[encoder.feature]
    if encoder.kind == "numerical":
        series = pd.to_numeric(x, errors="coerce")
        if encoder.bin_edges and len(encoder.bin_edges) >= 2:
            grouped = pd.cut(series, bins=encoder.bin_edges, include_lowest=True).astype("string")
        else:
            grouped = series.round(4).astype("string")
    else:
        grouped = x.astype("string")
    grouped = grouped.fillna("__MISSING__")
    mapping = encoder.mapping or {}
    values = grouped.map(lambda value: mapping.get(str(value), encoder.default_woe)).astype(float)
    values.name = f"{encoder.feature}_woe"
    return values


def fit_woe_encoders(
    train_df: pd.DataFrame,
    *,
    features: list[str] = WOE_SOURCE_FEATURES,
    target: str = TARGET,
) -> tuple[dict[str, WOEEncoderArtifact], dict[str, float]]:
    """Fit train-only WOE encoders and collect IV scores."""
    encoders: dict[str, WOEEncoderArtifact] = {}
    iv_scores: dict[str, float] = {}
    for feature in features:
        if feature not in train_df.columns:
            continue
        encoder = fit_woe_encoder(train_df, feature, target=target)
        encoders[feature] = encoder
        iv_scores[feature] = float(encoder.iv)
    return encoders, iv_scores


def apply_woe_encoders(
    df: pd.DataFrame,
    encoders: dict[str, WOEEncoderArtifact],
) -> pd.DataFrame:
    """Apply a fitted WOE encoder dict to a dataframe."""
    out = df.copy()
    for feature, encoder in encoders.items():
        out[f"{feature}_woe"] = apply_woe_encoder(out, encoder)
    return out


def available_feature_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in df.columns]


def build_feature_config(
    df: pd.DataFrame,
    *,
    iv_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the canonical feature config artifact for downstream consumers."""
    medium_missing_indicators = [
        f"{feature}{MISSINGNESS_INDICATOR_SUFFIX}"
        for feature in MEDIUM_COVERAGE_CHALLENGER_FEATURES
        if f"{feature}{MISSINGNESS_INDICATOR_SUFFIX}" in df.columns
    ]
    high_coverage = available_feature_columns(df, HIGH_COVERAGE_BUREAU_FEATURES)
    medium_coverage = available_feature_columns(df, MEDIUM_COVERAGE_CHALLENGER_FEATURES)
    excluded_present = [
        col
        for col in df.columns
        if col in EXCLUDED_CORE_FEATURES
        or any(col.startswith(prefix) for prefix in EXCLUDED_CORE_PATTERNS)
    ]
    return {
        "NUMERIC_FEATURES": available_feature_columns(df, NUMERIC_FEATURES),
        "FLAG_FEATURES": available_feature_columns(df, FLAG_FEATURES),
        "CATEGORICAL_FEATURES": available_feature_columns(df, CATEGORICAL_FEATURES),
        "WOE_FEATURES": available_feature_columns(df, WOE_FEATURES),
        "INTERACTION_FEATURES": available_feature_columns(df, INTERACTION_FEATURES),
        "CATBOOST_FEATURES": available_feature_columns(df, CATBOOST_FEATURES),
        "LOGREG_FEATURES": available_feature_columns(df, LOGREG_FEATURES),
        "HIGH_COVERAGE_BUREAU_FEATURES": high_coverage,
        "MEDIUM_COVERAGE_CHALLENGER_FEATURES": medium_coverage,
        "MISSINGNESS_INDICATORS": medium_missing_indicators,
        "CHALLENGER_ONLY_FEATURES": high_coverage + medium_coverage + medium_missing_indicators,
        "CORE_FEATURE_SET_V2": available_feature_columns(df, CATBOOST_FEATURES),
        "CHALLENGER_FEATURE_POOL_V2": available_feature_columns(
            df,
            CATBOOST_FEATURES + high_coverage + medium_coverage + medium_missing_indicators,
        ),
        "SURVIVAL_FEATURES": available_feature_columns(
            df,
            SURVIVAL_FEATURES + medium_missing_indicators,
        ),
        "ID_COLUMNS": available_feature_columns(df, ID_COLUMNS),
        "TARGET_COLUMNS": available_feature_columns(df, [TARGET]),
        "META_COLUMNS": available_feature_columns(df, META_COLUMNS),
        "EXCLUDED_CORE_FEATURES_PRESENT": excluded_present,
        "iv_scores": dict(sorted((iv_scores or {}).items(), key=lambda item: item[0])),
        "schema_version": "2026-03-26.1",
    }


def build_feature_manifest(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    """Build an auditable manifest with feature provenance and split coverage."""
    categories = set(feature_config.get("CATEGORICAL_FEATURES", []))
    core_catboost = set(feature_config.get("CATBOOST_FEATURES", []))
    core_logreg = set(feature_config.get("LOGREG_FEATURES", []))
    challenger_only = set(feature_config.get("CHALLENGER_ONLY_FEATURES", []))
    missing_indicators = set(feature_config.get("MISSINGNESS_INDICATORS", []))
    excluded = set(feature_config.get("EXCLUDED_CORE_FEATURES_PRESENT", []))

    all_features = sorted(
        {
            *train_df.columns,
            *calibration_df.columns,
            *test_df.columns,
        }
    )
    rows: list[dict[str, Any]] = []
    for feature in all_features:
        if feature in feature_config.get("ID_COLUMNS", []) or feature in feature_config.get(
            "META_COLUMNS", []
        ):
            role = "metadata"
        elif feature in feature_config.get("TARGET_COLUMNS", []):
            role = "target"
        elif feature in excluded:
            role = "excluded_sparse"
        elif feature in missing_indicators:
            role = "challenger_missing_indicator"
        elif feature in challenger_only and feature not in core_catboost:
            role = "challenger_only"
        elif feature in core_catboost and feature in core_logreg:
            role = "core_shared"
        elif feature in core_catboost:
            role = "core_catboost_only"
        elif feature in core_logreg:
            role = "core_logreg_only"
        else:
            role = "passthrough"

        rows.append(
            {
                "feature": feature,
                "dtype_train": str(train_df[feature].dtype)
                if feature in train_df.columns
                else "missing",
                "coverage_train": float(train_df[feature].notna().mean())
                if feature in train_df.columns
                else 0.0,
                "coverage_calibration": float(calibration_df[feature].notna().mean())
                if feature in calibration_df.columns
                else 0.0,
                "coverage_test": float(test_df[feature].notna().mean())
                if feature in test_df.columns
                else 0.0,
                "is_categorical_core": bool(feature in categories),
                "is_core_catboost": bool(feature in core_catboost),
                "is_core_logreg": bool(feature in core_logreg),
                "is_challenger_only": bool(
                    feature in challenger_only and feature not in core_catboost
                ),
                "family": _feature_family(feature),
                "role": role,
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "feature"]).reset_index(drop=True)


def _feature_family(feature: str) -> str:
    if feature in NUMERIC_FEATURES:
        return "core_numeric"
    if feature in FLAG_FEATURES:
        return "core_flag"
    if feature in CATEGORICAL_FEATURES:
        return "core_categorical"
    if feature in WOE_FEATURES:
        return "woe"
    if feature in HIGH_COVERAGE_BUREAU_FEATURES:
        return "bureau_high_coverage"
    if feature in MEDIUM_COVERAGE_CHALLENGER_FEATURES:
        return "bureau_medium_coverage"
    if feature.endswith(MISSINGNESS_INDICATOR_SUFFIX):
        return "missingness_indicator"
    if feature in INTERACTION_FEATURES:
        return "interaction"
    if feature in META_COLUMNS:
        return "metadata"
    if feature in ID_COLUMNS:
        return "id"
    if feature == TARGET:
        return "target"
    return "other"


def save_feature_artifacts(
    *,
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_config: dict[str, Any],
    woe_encoders: dict[str, WOEEncoderArtifact],
    output_dir: str | Path = "data/processed",
) -> None:
    """Persist canonical feature artifacts for the rerun pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_parquet(train_df, out_dir / "train_fe.parquet", index=False)
    atomic_write_parquet(calibration_df, out_dir / "calibration_fe.parquet", index=False)
    atomic_write_parquet(test_df, out_dir / "test_fe.parquet", index=False)

    atomic_write_pickle(out_dir / "feature_config.pkl", feature_config)
    atomic_write_pickle(out_dir / "woe_encoders.pkl", woe_encoders)

    manifest = build_feature_manifest(
        train_df,
        calibration_df,
        test_df,
        feature_config=feature_config,
    )
    atomic_write_parquet(manifest, out_dir / "feature_manifest_v2.parquet", index=False)
    atomic_write_text(
        out_dir / "feature_manifest_v2.json", manifest.to_json(orient="records", indent=2)
    )
    logger.info("Saved canonical feature artifacts to {}", out_dir)
