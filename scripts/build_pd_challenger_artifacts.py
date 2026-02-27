"""Build feature-selection and monotonic challenger artifacts for PD modeling.

Usage:
    uv run python scripts/build_pd_challenger_artifacts.py --config configs/pd_model.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.models.pd_model import TARGET, resolve_feature_sets
from src.utils.io_utils import read_split_with_fe_fallback


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_rows(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    if n_rows <= 0 or len(df) <= n_rows:
        return df.reset_index(drop=True)
    return df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)


def _prepare_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in features:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    fill_values = X.median(numeric_only=True).fillna(0.0)
    return X.fillna(fill_values).fillna(0.0)


def _boruta_proxy_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
) -> tuple[pd.Series, pd.Series]:
    """Boruta-style proxy: compare real feature importances vs shuffled shadows."""
    rng = np.random.RandomState(random_state)
    shadow = X.apply(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    shadow.columns = [f"shadow__{c}" for c in shadow.columns]

    X_aug = pd.concat([X, shadow], axis=1)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_aug, y)

    importances = pd.Series(model.feature_importances_, index=X_aug.columns, dtype=float)
    real_imp = importances[[c for c in X.columns if c in importances.index]]
    shadow_imp = importances[[c for c in shadow.columns if c in importances.index]]
    threshold = float(shadow_imp.max()) if len(shadow_imp) else float(real_imp.quantile(0.75))
    support = (real_imp > threshold).astype(bool)
    return real_imp, support


def _permutation_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    max_rows: int,
) -> pd.Series:
    if len(X) > max_rows > 0:
        idx = np.random.RandomState(random_state).choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=150,
        learning_rate=0.05,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    perm = permutation_importance(
        clf,
        X_val,
        y_val,
        n_repeats=3,
        random_state=random_state,
        scoring="roc_auc",
    )
    return pd.Series(perm.importances_mean, index=X.columns, dtype=float)


def main(config_path: str = "configs/pd_model.yaml") -> None:
    cfg = _load_config(config_path)
    challenger_cfg = cfg.get("challenger_pipeline", {})
    data_cfg = cfg.get("data", {})

    random_state = int(challenger_cfg.get("random_state", 42))
    sample_rows = int(challenger_cfg.get("sample_rows", 150_000))
    top_k = int(challenger_cfg.get("top_k", 30))
    perm_max_rows = int(challenger_cfg.get("permutation_max_rows", 80_000))
    boruta_n_estimators = int(challenger_cfg.get("boruta_n_estimators", 120))

    train_df = read_split_with_fe_fallback(
        data_cfg.get("train_path", "data/processed/train_fe.parquet")
    )
    train_df = _sample_rows(train_df, sample_rows, random_state)
    if TARGET not in train_df.columns:
        raise KeyError(f"Missing target column '{TARGET}' in sampled training data")

    feature_src_cfg = cfg.get("feature_source", {})
    feature_sets = resolve_feature_sets(
        train_df,
        feature_source=feature_src_cfg.get("mode", "auto"),
        feature_config_path=feature_src_cfg.get(
            "feature_config_path", "data/processed/feature_config.pkl"
        ),
    )

    catboost_features = feature_sets["catboost_features"]
    categorical_features = set(feature_sets["categorical_features"])
    numeric_features = [
        f for f in catboost_features if f not in categorical_features and f in train_df.columns
    ]
    if not numeric_features:
        raise ValueError("No numeric features available for challenger feature selection")

    X = _prepare_numeric_frame(train_df, numeric_features)
    y = train_df[TARGET].astype(int).to_numpy()

    mi = mutual_info_classif(X, y, random_state=random_state)
    mi_series = pd.Series(mi, index=X.columns, dtype=float)

    boruta_importance, boruta_support = _boruta_proxy_importance(
        X,
        y,
        random_state=random_state,
        n_estimators=boruta_n_estimators,
    )

    perm_scores = _permutation_scores(
        X,
        y,
        random_state=random_state,
        max_rows=perm_max_rows,
    )

    out = pd.DataFrame(
        {
            "feature": X.columns,
            "mutual_info": [float(mi_series.get(c, 0.0)) for c in X.columns],
            "boruta_importance": [float(boruta_importance.get(c, 0.0)) for c in X.columns],
            "boruta_support": [bool(boruta_support.get(c, False)) for c in X.columns],
            "permutation_importance_auc": [float(perm_scores.get(c, 0.0)) for c in X.columns],
        }
    )
    out["rank_mi"] = out["mutual_info"].rank(ascending=False, method="dense")
    out["rank_boruta"] = out["boruta_importance"].rank(ascending=False, method="dense")
    out["rank_permutation"] = out["permutation_importance_auc"].rank(
        ascending=False, method="dense"
    )
    out["rank_aggregate"] = out[["rank_mi", "rank_boruta", "rank_permutation"]].mean(axis=1)
    out["method_votes"] = (
        (out["mutual_info"] > out["mutual_info"].median()).astype(int)
        + out["boruta_support"].astype(int)
        + (out["permutation_importance_auc"] > out["permutation_importance_auc"].median()).astype(
            int
        )
    )
    out = out.sort_values(["rank_aggregate", "method_votes"], ascending=[True, False]).reset_index(
        drop=True
    )
    out["selected_topk"] = False
    out.loc[: max(top_k, 1) - 1, "selected_topk"] = True

    constraints_cfg = challenger_cfg.get("monotonic_constraints", {})
    monotonic_vector = [int(constraints_cfg.get(feature, 0)) for feature in catboost_features]

    no_smote_policy = bool(challenger_cfg.get("no_smote", True))
    spec = {
        "schema_version": "2026-02-27.1",
        "sample_rows_used": int(len(train_df)),
        "top_k": top_k,
        "target": TARGET,
        "feature_source": feature_sets.get("feature_source", "unknown"),
        "catboost_feature_count": int(len(catboost_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "selected_features": out.loc[out["selected_topk"], "feature"].astype(str).tolist(),
        "selection_methods": [
            "mrmr_proxy_mutual_info",
            "boruta_proxy_shadow_importance",
            "permutation_importance_auc",
        ],
        "monotonic_constraints": {
            "by_feature": {k: int(v) for k, v in constraints_cfg.items()},
            "vector_in_catboost_order": monotonic_vector,
        },
        "modeling_policies": {
            "no_smote": no_smote_policy,
            "challenger_only": True,
        },
        "notes": [
            "Feature-selection and monotonic constraints are challenger-only until full gate pass.",
            "No synthetic oversampling (SMOTE) is allowed by policy in challenger profiles.",
        ],
    }

    out_path = Path(
        challenger_cfg.get(
            "feature_selection_output", "data/processed/challenger_feature_selection.parquet"
        )
    )
    spec_path = Path(challenger_cfg.get("spec_output", "models/pd_challenger_spec.json"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(out_path, index=False)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=str)

    logger.info(
        "Challenger artifacts saved: {} ({} rows), {}",
        out_path,
        len(out),
        spec_path,
    )
    logger.info(
        "Selected top-k features: {}",
        ", ".join(spec["selected_features"][:10]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PD challenger feature-selection artifacts")
    parser.add_argument("--config", default="configs/pd_model.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
