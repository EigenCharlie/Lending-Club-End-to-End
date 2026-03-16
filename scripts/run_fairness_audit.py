"""Run fairness audit across multiple protected attributes.

Computes demographic parity, equalized odds, and disparate impact
for each attribute defined in the fairness policy config.
Supports policy `outcome_mode`:
- `default`: fairness over predicted default events.
- `approval`: fairness over favorable credit decision (approved loans).

Usage:
    uv run python scripts/run_fairness_audit.py
    uv run python scripts/run_fairness_audit.py --config configs/fairness_policy.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from loguru import logger
from sklearn.metrics import accuracy_score

from src.evaluation.fairness import (
    build_intersectional_groups,
    fairness_report_from_binary,
    fairness_threshold_frontier,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.threshold_semantics import write_threshold_semantics

SCHEMA_VERSION = "2026-03-06.1"


def _load_config(config_path: str) -> dict:
    """Load fairness policy YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_groups_dict(
    df: pd.DataFrame,
    attributes: list[dict],
) -> dict[str, np.ndarray]:
    """Build groups dict from config attribute definitions."""
    groups_dict: dict[str, np.ndarray] = {}

    for attr in attributes:
        name = attr["name"]
        col = attr["column"]

        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in data, skipping attribute '{name}'")
            continue

        if attr.get("binning") == "quartile":
            groups_dict[name] = (
                pd.qcut(df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                .astype(str)
                .values
            )
        else:
            groups_dict[name] = df[col].astype(str).values

    return groups_dict


def _resolve_prediction_threshold(cfg: dict, policy: dict) -> tuple[float, str]:
    """Resolve prediction threshold from artifact when configured."""
    fallback_threshold = float(policy.get("prediction_threshold", 0.5))
    threshold_cfg = cfg.get("threshold_policy", {}) or {}
    if not bool(threshold_cfg.get("use_artifact", True)):
        return fallback_threshold, "policy_default"

    artifact_path = Path(threshold_cfg.get("artifact_path", "models/decision_threshold.json"))
    if not artifact_path.exists():
        return fallback_threshold, "policy_default_missing_artifact"

    key = str(threshold_cfg.get("selected_threshold_key", "selected_threshold"))
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        resolved = float(payload.get(key, fallback_threshold))
        return resolved, "artifact"
    except Exception:
        return fallback_threshold, "policy_default_artifact_error"


def _resolve_outcome_mode(policy: dict) -> str:
    raw = str(policy.get("outcome_mode", "default")).strip().lower()
    if raw in {"approval", "approve", "good", "non_default"}:
        return "approval"
    return "default"


def _resolve_frontier_thresholds(primary_threshold: float, cfg: dict) -> list[float]:
    frontier_cfg = cfg.get("threshold_frontier", {}) or {}
    if not bool(frontier_cfg.get("enabled", True)):
        return [float(primary_threshold)]

    explicit = frontier_cfg.get("thresholds", []) or []
    if explicit:
        values = [float(x) for x in explicit]
    else:
        radius = float(frontier_cfg.get("window_radius", 0.10))
        step = float(frontier_cfg.get("step", 0.05))
        low = max(0.01, float(primary_threshold) - radius)
        high = min(0.99, float(primary_threshold) + radius)
        values = np.arange(low, high + step * 0.5, step).tolist()
        values.append(float(primary_threshold))
    clipped = [min(max(float(x), 0.01), 0.99) for x in values]
    return sorted({round(x, 4) for x in clipped})


def _decision_policy_cfg(cfg: dict) -> dict:
    return cfg.get("decision_policy", {}) or {}


def _select_threshold_from_frontier(
    frontier: pd.DataFrame,
    *,
    y_pred_proba_eval: np.ndarray,
) -> dict[str, float | int]:
    if frontier.empty:
        return {
            "selected_threshold": 0.5,
            "n_passed": 0,
            "worst_eo_gap": 1.0,
            "approval_rate": 0.0,
        }

    rows: list[dict[str, float | int]] = []
    for threshold, grp in frontier.groupby("threshold", observed=True):
        rows.append(
            {
                "threshold": float(threshold),
                "n_passed": int(grp["passed_all"].sum()),
                "worst_eo_gap": float(grp["eo_gap"].max()),
                "approval_rate": float(
                    (np.asarray(y_pred_proba_eval, dtype=float) >= float(threshold)).mean()
                ),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["n_passed", "worst_eo_gap", "approval_rate"],
        ascending=[False, True, False],
    )
    selected = ranking.iloc[0].to_dict()
    selected["selected_threshold"] = float(selected["threshold"])
    return selected


def _load_decision_policy_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _apply_decision_policy(
    *,
    y_pred_proba: np.ndarray,
    groups_all: dict[str, np.ndarray],
    decision_policy: dict,
    default_threshold: float,
) -> np.ndarray:
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    thresholds = np.full(len(y_pred_proba), float(default_threshold), dtype=float)
    overrides = decision_policy.get("overrides", []) if isinstance(decision_policy, dict) else []
    if not isinstance(overrides, list):
        overrides = []

    for override in overrides:
        if not isinstance(override, dict):
            continue
        attribute = str(override.get("attribute", "")).strip()
        group = str(override.get("group", "")).strip()
        threshold = float(override.get("threshold", default_threshold))
        labels = groups_all.get(attribute)
        if labels is None:
            continue
        mask = pd.Series(labels).astype(str).eq(group).to_numpy()
        thresholds[mask] = threshold

    return (y_pred_proba >= thresholds).astype(float)


def main(config_path: str = "configs/fairness_policy.yaml", run_tag: str | None = None) -> None:
    """Run the fairness audit pipeline."""
    cfg = _load_config(config_path)
    policy = cfg["policy"]
    artifacts = cfg["artifacts"]
    output = cfg["output"]

    # Load test predictions and test data
    pred_path = Path(artifacts["test_predictions_path"])
    data_path = Path(artifacts["test_data_path"])

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {pred_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test data: {data_path}")

    preds = pd.read_parquet(pred_path)
    data = pd.read_parquet(data_path)

    # Extract y_true and y_pred_proba
    y_true_col = "default_flag"
    y_proba_col = "y_pred_proba" if "y_pred_proba" in preds.columns else "pd_calibrated"

    if y_true_col not in data.columns:
        raise KeyError(f"Missing target column '{y_true_col}' in test data")
    if y_proba_col not in preds.columns:
        raise KeyError(
            f"Missing probability column in predictions. Available: {list(preds.columns)}"
        )

    # Align lengths (both should be OOT test set)
    n = min(len(preds), len(data))
    y_true = data[y_true_col].values[:n]
    y_proba = preds[y_proba_col].values[:n]

    logger.info(f"Loaded {n} observations for fairness audit")

    # Build groups from attributes config
    groups_dict = _build_groups_dict(data.iloc[:n], cfg["attributes"])

    if not groups_dict:
        logger.error("No valid attributes found for fairness audit")
        return

    threshold, threshold_source = _resolve_prediction_threshold(cfg, policy)
    outcome_mode = _resolve_outcome_mode(policy)
    if outcome_mode == "approval":
        # Fairness in credit decisions should be audited on favorable outcome (approval).
        y_true_eval = 1.0 - y_true
        y_proba_eval = 1.0 - y_proba
    else:
        y_true_eval = y_true
        y_proba_eval = y_proba
    resolved_run_tag = resolve_run_tag(run_tag, require_explicit=True)

    intersectional_cfg = cfg.get("intersectional", {}) or {}
    intersectional_groups = (
        build_intersectional_groups(
            groups_dict,
            max_order=int(intersectional_cfg.get("max_order", 2)),
            min_group_size=int(intersectional_cfg.get("min_group_size", 300)),
        )
        if bool(intersectional_cfg.get("enabled", True))
        else {}
    )
    groups_all = dict(groups_dict)
    groups_all.update(intersectional_groups)

    frontier_thresholds = _resolve_frontier_thresholds(float(threshold), cfg)
    frontier = fairness_threshold_frontier(
        y_true=y_true_eval,
        y_pred_proba=y_proba_eval,
        groups_dict=groups_all,
        thresholds=frontier_thresholds,
        primary_threshold=float(threshold),
        dpd_threshold=policy["dpd_threshold"],
        eo_gap_threshold=policy["eo_gap_threshold"],
        dir_threshold=policy["dir_threshold"],
    )
    if not frontier.empty:
        frontier["attribute_type"] = np.where(
            frontier["attribute"].astype(str).str.contains("__x__"),
            "intersectional",
            "base",
        )
    frontier_path = Path(
        output.get("frontier_parquet", "data/processed/fairness_threshold_frontier.parquet")
    )
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    frontier.to_parquet(frontier_path, index=False)
    logger.info(f"Saved fairness threshold frontier: {frontier_path}")

    decision_policy_cfg = _decision_policy_cfg(cfg)
    decision_policy_path = Path(
        decision_policy_cfg.get("artifact_path", "models/fairness_decision_policy.json")
    )
    auto_select = bool(decision_policy_cfg.get("auto_select", False))
    decision_policy = _load_decision_policy_artifact(decision_policy_path)
    selected_threshold_info = _select_threshold_from_frontier(
        frontier, y_pred_proba_eval=y_proba_eval
    )
    primary_threshold = float(
        selected_threshold_info.get("selected_threshold", float(threshold))
        if auto_select
        else float(threshold)
    )

    if auto_select:
        decision_policy = {
            "global_threshold": primary_threshold,
            "overrides": decision_policy.get("overrides", [])
            if isinstance(decision_policy, dict)
            else [],
            "selection": {
                "source": "fairness_frontier_auto_select",
                "n_passed": int(selected_threshold_info.get("n_passed", 0)),
                "worst_eo_gap": float(selected_threshold_info.get("worst_eo_gap", 0.0)),
                "approval_rate": float(selected_threshold_info.get("approval_rate", 0.0)),
            },
            **build_artifact_metadata(
                schema_version=SCHEMA_VERSION,
                run_tag=resolved_run_tag,
                require_explicit=True,
            ),
        }
        decision_policy_path.parent.mkdir(parents=True, exist_ok=True)
        decision_policy_path.write_text(
            json.dumps(decision_policy, indent=2, default=str),
            encoding="utf-8",
        )
        threshold_source = "decision_policy_artifact_auto_selected"
    elif decision_policy:
        primary_threshold = float(decision_policy.get("global_threshold", threshold))
        threshold_source = "decision_policy_artifact"

    y_pred_binary = _apply_decision_policy(
        y_pred_proba=y_proba_eval,
        groups_all=groups_all,
        decision_policy=decision_policy,
        default_threshold=primary_threshold,
    )
    report = fairness_report_from_binary(
        y_true=y_true_eval,
        y_pred_binary=y_pred_binary,
        groups_dict=groups_all,
        dpd_threshold=policy["dpd_threshold"],
        eo_gap_threshold=policy["eo_gap_threshold"],
        dir_threshold=policy["dir_threshold"],
    )
    if not report.empty:
        report["attribute_type"] = np.where(
            report["attribute"].astype(str).str.contains("__x__"),
            "intersectional",
            "base",
        )

    audit_path = Path(output["audit_parquet"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(audit_path, index=False)
    logger.info(f"Saved fairness audit: {audit_path}")

    # Build and save status JSON
    overall_pass = bool(report["passed_all"].all())
    primary_frontier = (
        frontier.loc[np.isclose(frontier["threshold"].astype(float), primary_threshold)]
        if not frontier.empty
        else pd.DataFrame()
    )
    worst_primary_attribute = ""
    if not primary_frontier.empty:
        worst_primary_attribute = str(
            primary_frontier.sort_values(
                by=["passed_all", "eo_gap", "dpd", "dir"],
                ascending=[True, False, False, True],
            ).iloc[0]["attribute"]
        )
    status = {
        "overall_pass": overall_pass,
        "n_attributes": len(report),
        "n_base_attributes": int(
            (report.get("attribute_type", pd.Series(dtype=str)) == "base").sum()
        ),
        "n_intersectional_attributes": int(
            (report.get("attribute_type", pd.Series(dtype=str)) == "intersectional").sum()
        ),
        "n_passed": int(report["passed_all"].sum()),
        "attributes": report.to_dict(orient="records"),
        "prediction_threshold": float(primary_threshold),
        "primary_threshold": float(primary_threshold),
        "prediction_threshold_source": threshold_source,
        "outcome_mode": outcome_mode,
        "thresholds": {
            "dpd": policy["dpd_threshold"],
            "eo_gap": policy["eo_gap_threshold"],
            "dir": policy["dir_threshold"],
        },
        "threshold_frontier": {
            "path": str(frontier_path),
            "thresholds": frontier_thresholds,
            "worst_primary_attribute": worst_primary_attribute,
            "selected_threshold": float(primary_threshold),
            "all_primary_pass": bool(
                primary_frontier.get("passed_all", pd.Series(dtype=bool)).all()
            )
            if not primary_frontier.empty
            else True,
        },
        "decision_policy": {
            "path": str(decision_policy_path),
            "global_threshold": float(primary_threshold),
            "n_overrides": int(len(decision_policy.get("overrides", [])))
            if isinstance(decision_policy, dict)
            else 0,
        },
        "policy_config": str(config_path),
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }

    status_path = Path(output["status_json"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Saved fairness status: {status_path}")

    sidecar_cfg = cfg.get("fairlearn_sidecar", {}) or {}
    if bool(sidecar_cfg.get("enabled", True)):
        group_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        rng = np.random.default_rng(int(sidecar_cfg.get("bootstrap_random_state", 42)))
        n_boot = int(sidecar_cfg.get("bootstrap_samples", 200))
        bootstrap_max_rows = int(sidecar_cfg.get("bootstrap_max_rows", 50_000))
        y_true_arr = np.asarray(y_true_eval, dtype=float)
        y_pred_arr = np.asarray(y_pred_binary, dtype=float)
        if bootstrap_max_rows > 0 and len(y_true_arr) > bootstrap_max_rows:
            bootstrap_idx = np.sort(
                rng.choice(len(y_true_arr), size=bootstrap_max_rows, replace=False)
            )
        else:
            bootstrap_idx = np.arange(len(y_true_arr))

        for attribute, labels in groups_all.items():
            sensitive = pd.Series(labels).astype(str).reset_index(drop=True)
            mf = MetricFrame(
                metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
                y_true=y_true_arr,
                y_pred=y_pred_arr,
                sensitive_features=sensitive,
            )
            by_group = mf.by_group.reset_index()
            by_group.columns = ["group", *[str(col) for col in by_group.columns[1:]]]
            for row in by_group.to_dict(orient="records"):
                row["attribute"] = attribute
                group_rows.append(row)

            dpd = float(
                demographic_parity_difference(
                    y_true=y_true_arr,
                    y_pred=y_pred_arr,
                    sensitive_features=sensitive,
                )
            )
            eo = float(
                equalized_odds_difference(
                    y_true=y_true_arr,
                    y_pred=y_pred_arr,
                    sensitive_features=sensitive,
                )
            )
            boot_sensitive_base = sensitive.iloc[bootstrap_idx].reset_index(drop=True)
            boot_true_base = y_true_arr[bootstrap_idx]
            boot_pred_base = y_pred_arr[bootstrap_idx]
            dpd_boot: list[float] = []
            eo_boot: list[float] = []
            for _ in range(max(n_boot, 0)):
                idx = rng.integers(0, len(boot_true_base), len(boot_true_base))
                boot_sensitive = boot_sensitive_base.iloc[idx]
                boot_true = boot_true_base[idx]
                boot_pred = boot_pred_base[idx]
                dpd_boot.append(
                    float(
                        demographic_parity_difference(
                            y_true=boot_true,
                            y_pred=boot_pred,
                            sensitive_features=boot_sensitive,
                        )
                    )
                )
                eo_boot.append(
                    float(
                        equalized_odds_difference(
                            y_true=boot_true,
                            y_pred=boot_pred,
                            sensitive_features=boot_sensitive,
                        )
                    )
                )
            summary_rows.append(
                {
                    "attribute": attribute,
                    "demographic_parity_difference": dpd,
                    "equalized_odds_difference": eo,
                    "dpd_ci_low": float(np.quantile(dpd_boot, 0.025)) if dpd_boot else None,
                    "dpd_ci_high": float(np.quantile(dpd_boot, 0.975)) if dpd_boot else None,
                    "eo_ci_low": float(np.quantile(eo_boot, 0.025)) if eo_boot else None,
                    "eo_ci_high": float(np.quantile(eo_boot, 0.975)) if eo_boot else None,
                }
            )

        sidecar_path = Path(sidecar_cfg.get("status_json", "models/fairlearn_fairness_status.json"))
        group_metrics_path = Path(
            sidecar_cfg.get(
                "group_metrics_parquet", "data/processed/fairlearn_group_metrics.parquet"
            )
        )
        group_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(group_rows).to_parquet(group_metrics_path, index=False)
        sidecar_payload = {
            "primary_status_path": str(status_path),
            "group_metrics_path": str(group_metrics_path),
            "n_attributes": len(summary_rows),
            "attributes": summary_rows,
            "bootstrap_samples": n_boot,
            "bootstrap_rows_used": int(len(bootstrap_idx)),
            "bootstrap_max_rows": bootstrap_max_rows,
            "prediction_threshold": float(primary_threshold),
            "outcome_mode": outcome_mode,
            **build_artifact_metadata(
                schema_version=f"{SCHEMA_VERSION}-fairlearn",
                run_tag=resolved_run_tag,
                require_explicit=True,
            ),
        }
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, default=str), encoding="utf-8"
        )
        logger.info(f"Saved fairlearn sidecar status: {sidecar_path}")
    write_threshold_semantics(
        fairness_primary_threshold=float(primary_threshold),
        decision_policy_global_threshold=float(
            decision_policy.get("global_threshold", primary_threshold)
        )
        if isinstance(decision_policy, dict)
        else float(primary_threshold),
        source_artifacts={
            "fairness_status": str(status_path),
            "fairness_decision_policy": str(decision_policy_path),
            "fairness_frontier": str(frontier_path),
        },
        run_tag=resolved_run_tag,
        extra={
            "fairness_threshold_source": threshold_source,
            "outcome_mode": outcome_mode,
        },
    )

    pass_label = "PASS" if overall_pass else "FAIL"
    logger.info(
        f"Fairness audit: {pass_label} ({status['n_passed']}/{status['n_attributes']} attributes)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fairness audit")
    parser.add_argument("--config", default="configs/fairness_policy.yaml")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(config_path=args.config, run_tag=args.run_tag)
