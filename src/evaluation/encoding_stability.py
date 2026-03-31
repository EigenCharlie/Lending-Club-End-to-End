"""ADSFCR-inspired encoding and binning stability diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.evaluation.backtesting import population_stability_index


def categorical_psi(train: pd.Series, test: pd.Series, *, eps: float = 1e-6) -> float:
    train_share = train.astype("string").fillna("MISSING").value_counts(normalize=True)
    test_share = test.astype("string").fillna("MISSING").value_counts(normalize=True)
    categories = train_share.index.union(test_share.index)
    tr = train_share.reindex(categories, fill_value=0.0).astype(float) + eps
    te = test_share.reindex(categories, fill_value=0.0).astype(float) + eps
    return float(np.sum((te - tr) * np.log(te / tr)))


def _safe_spearman(feature: pd.Series, target: pd.Series) -> float:
    merged = (
        pd.DataFrame({"feature": feature, "target": target})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(merged) < 25:
        return float("nan")
    return float(merged["feature"].corr(merged["target"], method="spearman"))


def woe_stability_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = "default_flag",
    psi_threshold: float = 0.25,
    correlation_delta_threshold: float = 0.15,
) -> pd.DataFrame:
    rows: list[dict[str, float | bool | str]] = []
    woe_features = sorted(
        col for col in train_df.columns if str(col).endswith("_woe") and col in test_df.columns
    )

    for feature in woe_features:
        tr = pd.to_numeric(train_df[feature], errors="coerce")
        te = pd.to_numeric(test_df[feature], errors="coerce")
        psi = population_stability_index(tr, te, n_bins=10)
        train_corr = _safe_spearman(tr, train_df[target_col])
        test_corr = _safe_spearman(te, test_df[target_col])
        sign_consistent = bool(
            np.isnan(train_corr)
            or np.isnan(test_corr)
            or np.sign(train_corr) == np.sign(test_corr)
            or np.isclose(train_corr, 0.0)
            or np.isclose(test_corr, 0.0)
        )
        corr_delta = (
            float(abs(train_corr - test_corr))
            if np.isfinite(train_corr) and np.isfinite(test_corr)
            else float("nan")
        )
        unstable = bool(
            psi > psi_threshold
            or (np.isfinite(corr_delta) and corr_delta > correlation_delta_threshold)
            or not sign_consistent
        )
        rows.append(
            {
                "feature": str(feature),
                "base_feature": str(feature).removesuffix("_woe"),
                "psi": float(psi),
                "train_spearman_default": train_corr,
                "test_spearman_default": test_corr,
                "abs_spearman_delta": corr_delta,
                "sign_consistent": sign_consistent,
                "overall_pass": not unstable,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "base_feature",
                "psi",
                "train_spearman_default",
                "test_spearman_default",
                "abs_spearman_delta",
                "sign_consistent",
                "overall_pass",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["overall_pass", "psi", "abs_spearman_delta"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )


def bucket_stability_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = "default_flag",
    feature_cols: Iterable[str] | None = None,
    share_shift_threshold: float = 0.15,
    rank_corr_threshold: float = 0.50,
) -> pd.DataFrame:
    bucket_features = (
        list(feature_cols)
        if feature_cols is not None
        else sorted(
            col
            for col in train_df.columns
            if str(col).endswith("_bucket") and col in test_df.columns
        )
    )
    rows: list[dict[str, float | bool | str]] = []
    for feature in bucket_features:
        tr = train_df[feature].astype("string").fillna("MISSING")
        te = test_df[feature].astype("string").fillna("MISSING")
        psi = categorical_psi(tr, te)
        tr_share = tr.value_counts(normalize=True)
        te_share = te.value_counts(normalize=True)
        categories = tr_share.index.union(te_share.index)
        max_share_shift = float(
            (
                tr_share.reindex(categories, fill_value=0.0)
                - te_share.reindex(categories, fill_value=0.0)
            )
            .abs()
            .max()
        )
        tr_rate = train_df.assign(_cat=tr).groupby("_cat", observed=True)[target_col].mean()
        te_rate = test_df.assign(_cat=te).groupby("_cat", observed=True)[target_col].mean()
        common = tr_rate.index.intersection(te_rate.index)
        if len(common) >= 3:
            rank_corr = float(tr_rate.loc[common].corr(te_rate.loc[common], method="spearman"))
        else:
            rank_corr = float("nan")
        unstable = bool(
            psi > 0.25
            or max_share_shift > share_shift_threshold
            or (np.isfinite(rank_corr) and rank_corr < rank_corr_threshold)
        )
        rows.append(
            {
                "feature": str(feature),
                "category_psi": float(psi),
                "max_share_shift": max_share_shift,
                "rank_corr_train_vs_test": rank_corr,
                "overall_pass": not unstable,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "category_psi",
                "max_share_shift",
                "rank_corr_train_vs_test",
                "overall_pass",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["overall_pass", "category_psi", "max_share_shift"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )
