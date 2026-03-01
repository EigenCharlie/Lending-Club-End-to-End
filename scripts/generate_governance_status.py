"""Generate governance drift status for MRM gating.

Builds per-feature drift diagnostics (PSI, KS, CvM) and multivariate C2ST,
then emits:
- data/processed/drift_monitoring.parquet
- models/governance_status.json

Usage:
    uv run python scripts/generate_governance_status.py --config configs/mrm_policy.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from src.evaluation.backtesting import classifier_two_sample_test, drift_monitoring_report
from src.utils.io_utils import read_split_with_fe_fallback


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_feature_contract() -> list[str]:
    contract_path = Path("models/pd_model_contract.json")
    if not contract_path.exists():
        return []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    features = contract.get("feature_names", [])
    if not isinstance(features, list):
        return []
    return [str(f) for f in features]


def _resolve_numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    contract_features = _load_feature_contract()
    common = [f for f in contract_features if f in train_df.columns and f in test_df.columns]

    numeric = []
    for f in common:
        if pd.api.types.is_numeric_dtype(train_df[f]) or pd.api.types.is_numeric_dtype(test_df[f]):
            numeric.append(f)

    if numeric:
        return numeric

    # Fallback: numeric intersection from both frames.
    train_num = set(train_df.select_dtypes(include=["number"]).columns)
    test_num = set(test_df.select_dtypes(include=["number"]).columns)
    fallback = sorted(train_num.intersection(test_num))
    return [c for c in fallback if c != "default_flag"][:80]


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def main(config_path: str = "configs/mrm_policy.yaml") -> None:
    cfg = _load_cfg(config_path)

    triggers = cfg.get("retraining_triggers", {})
    checks = cfg.get("governance_checks", {})
    outputs = cfg.get("governance_output", {})

    psi_threshold = float(triggers.get("psi_threshold", 0.25))
    ks_pvalue_min = float(checks.get("ks_pvalue_min", 0.01))
    cvm_pvalue_min = float(checks.get("cvm_pvalue_min", 0.01))
    c2st_auc_max = float(checks.get("c2st_auc_max", 0.60))
    max_feature_breach_ratio = float(checks.get("max_feature_breach_ratio", 0.15))
    c2st_max_rows = int(checks.get("c2st_max_rows_per_split", 50_000))

    drift_path = Path(
        outputs.get("drift_monitoring_path", "data/processed/drift_monitoring.parquet")
    )
    drift_v2_path = Path(
        outputs.get("drift_monitoring_v2_path", "data/processed/drift_monitoring_v2.parquet")
    )
    status_path = Path(outputs.get("governance_status_path", "models/governance_status.json"))
    status_v2_path = Path(
        outputs.get("governance_status_v2_path", "models/governance_status_v2.json")
    )

    train_df = read_split_with_fe_fallback("data/processed/train_fe.parquet")
    test_df = read_split_with_fe_fallback("data/processed/test_fe.parquet")

    features = _resolve_numeric_features(train_df, test_df)
    if not features:
        raise ValueError("No numeric features available for governance drift checks.")

    logger.info("Governance drift checks on {} numeric features", len(features))

    drift_df = drift_monitoring_report(
        train_df=train_df,
        test_df=test_df,
        features=features,
        psi_threshold=psi_threshold,
        ks_pvalue_threshold=ks_pvalue_min,
        cvm_pvalue_threshold=cvm_pvalue_min,
        n_bins=int(checks.get("psi_bins", 10)),
    )

    c2st = classifier_two_sample_test(
        train_df=train_df,
        test_df=test_df,
        features=features,
        max_rows_per_split=c2st_max_rows,
        random_state=int(checks.get("random_state", 42)),
    )

    n_features = int(len(drift_df))
    psi_breaches = (
        int((~drift_df.get("pass_psi", pd.Series(dtype=bool))).sum()) if n_features else 0
    )
    ks_breaches = int((~drift_df.get("pass_ks", pd.Series(dtype=bool))).sum()) if n_features else 0
    cvm_breaches = (
        int((~drift_df.get("pass_cvm", pd.Series(dtype=bool))).sum()) if n_features else 0
    )
    feature_breach_ratio = float(
        (psi_breaches + ks_breaches + cvm_breaches) / max(n_features * 3, 1)
    )

    max_psi = float(drift_df["psi"].max()) if n_features else 0.0
    mean_psi = _safe_mean(drift_df["psi"]) if n_features else 0.0
    min_ks_pvalue = float(drift_df["ks_pvalue"].min()) if n_features else 1.0
    min_cvm_pvalue = float(drift_df["cvm_pvalue"].min()) if n_features else 1.0
    c2st_auc = float(c2st["c2st_auc"])

    pass_psi = bool(max_psi <= psi_threshold)
    pass_c2st = bool(c2st_auc <= c2st_auc_max)
    pass_breach_ratio = bool(feature_breach_ratio <= max_feature_breach_ratio)

    overall_pass = bool(pass_psi and pass_c2st and pass_breach_ratio)

    drift_path.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_parquet(drift_path, index=False)
    drift_v2_path.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_parquet(drift_v2_path, index=False)

    top_breaches = drift_df.head(10).to_dict(orient="records") if n_features else []
    status = {
        "overall_pass": overall_pass,
        "checks": {
            "pass_psi": pass_psi,
            "pass_c2st": pass_c2st,
            "pass_breach_ratio": pass_breach_ratio,
        },
        "thresholds": {
            "psi_threshold": psi_threshold,
            "ks_pvalue_min": ks_pvalue_min,
            "cvm_pvalue_min": cvm_pvalue_min,
            "c2st_auc_max": c2st_auc_max,
            "max_feature_breach_ratio": max_feature_breach_ratio,
        },
        "summary": {
            "n_features": n_features,
            "max_psi": max_psi,
            "mean_psi": mean_psi,
            "min_ks_pvalue": min_ks_pvalue,
            "min_cvm_pvalue": min_cvm_pvalue,
            "c2st_auc": c2st_auc,
            "psi_breaches": psi_breaches,
            "ks_breaches": ks_breaches,
            "cvm_breaches": cvm_breaches,
            "feature_breach_ratio": feature_breach_ratio,
            "c2st_rows_used": int(c2st.get("n_rows", 0)),
        },
        "artifacts": {
            "drift_monitoring_path": str(drift_path),
            "drift_monitoring_v2_path": str(drift_v2_path),
        },
        "top_drift_features": top_breaches,
        "policy_config": config_path,
    }

    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    status_v2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_v2_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    logger.info("Saved drift monitoring: {}", drift_path)
    logger.info("Saved governance status: {}", status_path)
    logger.info(
        "Governance checks pass={} (max_psi={:.4f}, c2st_auc={:.4f}, breach_ratio={:.4f})",
        overall_pass,
        max_psi,
        c2st_auc,
        feature_breach_ratio,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate governance drift status")
    parser.add_argument("--config", default="configs/mrm_policy.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
