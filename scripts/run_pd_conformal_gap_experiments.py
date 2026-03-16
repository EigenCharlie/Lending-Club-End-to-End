"""Targeted conformal gap search focused on closing the Winkler_90 gap."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from scripts import validate_conformal_policy as validate_policy_mod
from scripts.generate_conformal_intervals import (
    _build_feature_matrix,
    _load_calibrator,
    _load_model,
    _resolve_artifact_paths,
    _resolve_features,
    _stage_metrics,
)
from scripts.generate_conformal_intervals import (
    main as generate_conformal_main,
)
from src.models.conformal import build_mondrian_partition_labels, create_pd_intervals_mondrian
from src.models.conformal_tuning import (
    enforce_group_coverage_floor,
    split_calibration_for_tuning,
)
from src.utils.io_utils import read_with_fallback

TARGET_COL = "default_flag"
GROUP_COL = "grade"


def _candidate_sort(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["global_ok"] = df["coverage_90"] >= 0.90
    df["group_ok"] = df["min_group_coverage_90"] >= 0.88
    df["coverage_gap"] = (df["coverage_90"] - 0.90).abs()
    df["group_gap"] = (0.88 - df["min_group_coverage_90"]).clip(lower=0.0)
    return df.sort_values(
        [
            "global_ok",
            "group_ok",
            "group_gap",
            "winkler_90",
            "avg_width_90",
            "stability_over_time",
            "coverage_gap",
        ],
        ascending=[False, False, True, True, True, True, True],
    ).reset_index(drop=True)


def _variant_key(row: dict[str, Any]) -> str:
    return (
        f"{row['partition']}|scaled={int(bool(row['scaled_scores']))}"
        f"|mgs={int(row['min_group_size'])}|alpha={float(row['alpha_used_90']):.3f}"
    )


def _write_validation_config(
    *,
    cfg_template_path: str,
    results_path: Path,
    group_metrics_path: Path,
    intervals_path: Path,
    backtest_monthly_path: Path,
    backtest_alerts_path: Path,
    status_path: Path,
    checks_path: Path,
    out_path: Path,
) -> None:
    cfg = yaml.safe_load(Path(cfg_template_path).read_text(encoding="utf-8")) or {}
    cfg["artifacts"] = {
        "conformal_results_path": str(results_path),
        "group_metrics_path": str(group_metrics_path),
        "backtest_monthly_path": str(backtest_monthly_path),
        "backtest_alerts_path": str(backtest_alerts_path),
        "intervals_path": str(intervals_path),
    }
    cfg["output"] = {
        "policy_status_json": str(status_path),
        "policy_checks_parquet": str(checks_path),
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    model, _ = _load_model()
    calibrator = _load_calibrator()
    cal_df = read_with_fallback(
        "data/processed/calibration_fe.parquet", "data/processed/calibration.parquet"
    )
    test_df = read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")
    features, categorical = _resolve_features(model, cal_df, test_df)
    X_cal = _build_feature_matrix(cal_df, features, categorical)
    y_cal = cal_df[TARGET_COL].astype(float)
    group_cal_base = cal_df[GROUP_COL].fillna("UNKNOWN").astype(str)
    issue_cal = pd.to_datetime(cal_df.get("issue_d"), errors="coerce")

    idx_fit, idx_tune = split_calibration_for_tuning(
        y_cal=y_cal,
        group_cal=group_cal_base,
        issue_dates=issue_cal,
        holdout_ratio=0.20,
        random_state=42,
    )
    X_cal_fit = X_cal.iloc[idx_fit].reset_index(drop=True)
    y_cal_fit = y_cal.iloc[idx_fit].reset_index(drop=True)
    X_tune = X_cal.iloc[idx_tune].reset_index(drop=True)
    y_tune = y_cal.iloc[idx_tune].reset_index(drop=True)
    group_cal_fit_base = group_cal_base.iloc[idx_fit].reset_index(drop=True)
    group_tune_base = group_cal_base.iloc[idx_tune].reset_index(drop=True)
    issue_tune = issue_cal.iloc[idx_tune].reset_index(drop=True)
    y_prob_fit_raw = model.predict_proba(X_cal_fit)[:, 1]
    y_prob_tune_raw = model.predict_proba(X_tune)[:, 1]

    experiments: list[dict[str, Any]] = []

    stage_a_rows: list[dict[str, Any]] = []
    base_variants = []
    for alpha_used in (0.09, 0.095, 0.10):
        for min_group_size in (200, 500):
            for scaled_scores in (True, False):
                base_variants.append(
                    {
                        "partition": "grade",
                        "scaled_scores": scaled_scores,
                        "min_group_size": min_group_size,
                        "alpha_used_90": alpha_used,
                    }
                )
            base_variants.append(
                {
                    "partition": "score_decile_mondrian",
                    "scaled_scores": True,
                    "min_group_size": min_group_size,
                    "alpha_used_90": alpha_used,
                }
            )

    for variant in base_variants:
        group_cal_part, group_tune_part, _meta = build_mondrian_partition_labels(
            y_prob_cal=y_prob_fit_raw,
            y_prob_eval=y_prob_tune_raw,
            partition=str(variant["partition"]),
            base_groups_cal=group_cal_fit_base,
            base_groups_eval=group_tune_base,
            n_score_bins=10,
            min_group_size=int(variant["min_group_size"]),
        )
        y_pred_tune, y_int_tune, _ = create_pd_intervals_mondrian(
            classifier=model,
            X_cal=X_cal_fit,
            y_cal=y_cal_fit,
            X_test=X_tune,
            group_cal=group_cal_part,
            group_test=group_tune_part,
            alpha=float(variant["alpha_used_90"]),
            min_group_size=int(variant["min_group_size"]),
            calibrator=calibrator,
            scaled_scores=bool(variant["scaled_scores"]),
        )
        row = _stage_metrics(
            dataset_scope="tune_holdout",
            stage="stage_a_base",
            y_true=y_tune.to_numpy(dtype=float),
            y_pred=y_pred_tune,
            y_intervals=y_int_tune,
            groups=group_tune_part,
            issue_dates=issue_tune,
            alpha=0.10,
            target_coverage=0.90,
        )
        row.update(
            {
                "experiment_stage": "A",
                "partition": str(variant["partition"]),
                "scaled_scores": bool(variant["scaled_scores"]),
                "min_group_size": int(variant["min_group_size"]),
                "alpha_used_90": float(variant["alpha_used_90"]),
                "variant_key": _variant_key(variant),
            }
        )
        stage_a_rows.append(row)

    stage_a = _candidate_sort(pd.DataFrame(stage_a_rows))
    stage_a["selection_rank"] = np.arange(1, len(stage_a) + 1, dtype=int)
    experiments.extend(stage_a.to_dict(orient="records"))
    top4 = stage_a.head(4).to_dict(orient="records")

    stage_b_rows: list[dict[str, Any]] = []
    for parent in top4:
        group_cal_part, group_tune_part, _meta = build_mondrian_partition_labels(
            y_prob_cal=y_prob_fit_raw,
            y_prob_eval=y_prob_tune_raw,
            partition=str(parent["partition"]),
            base_groups_cal=group_cal_fit_base,
            base_groups_eval=group_tune_base,
            n_score_bins=10,
            min_group_size=int(parent["min_group_size"]),
        )
        y_pred_tune, y_int_tune, _ = create_pd_intervals_mondrian(
            classifier=model,
            X_cal=X_cal_fit,
            y_cal=y_cal_fit,
            X_test=X_tune,
            group_cal=group_cal_part,
            group_test=group_tune_part,
            alpha=float(parent["alpha_used_90"]),
            min_group_size=int(parent["min_group_size"]),
            calibrator=calibrator,
            scaled_scores=bool(parent["scaled_scores"]),
        )
        for floor_target in (0.90, 0.91, 0.92):
            for coverage_guardband in (0.00, 0.01, 0.015):
                for max_width_budget in (0.76, 0.78, 0.80):
                    for grid_name, grid in {
                        "coarse": (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20),
                        "medium": (1.0, 1.01, 1.03, 1.05, 1.08, 1.12, 1.16, 1.20),
                    }.items():
                        y_adj, group_factors, _report = enforce_group_coverage_floor(
                            y_true=y_tune.to_numpy(dtype=float),
                            y_pred=y_pred_tune,
                            y_intervals=y_int_tune,
                            groups=group_tune_part,
                            target_coverage=float(floor_target),
                            multiplier_grid=grid,
                        )
                        row = _stage_metrics(
                            dataset_scope="tune_holdout",
                            stage="stage_b_group_floor",
                            y_true=y_tune.to_numpy(dtype=float),
                            y_pred=y_pred_tune,
                            y_intervals=y_adj,
                            groups=group_tune_part,
                            issue_dates=issue_tune,
                            alpha=0.10,
                            target_coverage=0.90,
                        )
                        row.update(
                            {
                                "experiment_stage": "B",
                                "parent_variant_key": str(parent["variant_key"]),
                                "partition": str(parent["partition"]),
                                "scaled_scores": bool(parent["scaled_scores"]),
                                "min_group_size": int(parent["min_group_size"]),
                                "alpha_used_90": float(parent["alpha_used_90"]),
                                "group_coverage_floor_target_90": float(floor_target),
                                "coverage_guardband_90": float(coverage_guardband),
                                "max_width_budget_90": float(max_width_budget),
                                "multiplier_grid_name": grid_name,
                                "group_factors_n": int(len(group_factors)),
                            }
                        )
                        stage_b_rows.append(row)

    stage_b = _candidate_sort(pd.DataFrame(stage_b_rows))
    stage_b["selection_rank"] = np.arange(1, len(stage_b) + 1, dtype=int)
    experiments.extend(stage_b.to_dict(orient="records"))
    top3 = stage_b.head(3).to_dict(orient="records")

    stage_c_rows: list[dict[str, Any]] = []
    generated_namespaces: list[str] = []
    with TemporaryDirectory(prefix="pd_conformal_gap_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, parent in enumerate(top3, start=1):
            for temporal_floor_enabled in (True, False):
                for temporal_segment_min_size in (250, 500):
                    namespace = (
                        f"gap_stage_c_{idx}_{parent['partition']}"
                        f"_scaled{int(bool(parent['scaled_scores']))}"
                        f"_mgs{int(parent['min_group_size'])}"
                        f"_alpha{str(parent['alpha_used_90']).replace('.', '')}"
                        f"_floor{str(parent['group_coverage_floor_target_90']).replace('.', '')}"
                        f"_t{int(temporal_floor_enabled)}_ts{int(temporal_segment_min_size)}"
                    )
                    generated_namespaces.append(namespace)
                    generate_conformal_main(
                        alpha_target_90=0.10,
                        alpha_95=0.05,
                        alpha_candidates_90=(float(parent["alpha_used_90"]),),
                        min_group_sizes=(int(parent["min_group_size"]),),
                        min_group_coverage_target=0.88,
                        group_coverage_floor_target_90=float(
                            parent["group_coverage_floor_target_90"]
                        ),
                        max_width_budget_90=float(parent["max_width_budget_90"]),
                        coverage_guardband_90=float(parent["coverage_guardband_90"]),
                        min_group_guardband_90=0.0,
                        tuning_holdout_ratio=0.20,
                        tuning_random_state=42,
                        temporal_segment_floor_enabled=bool(temporal_floor_enabled),
                        temporal_segment_freq="Q",
                        temporal_segment_min_size=int(temporal_segment_min_size),
                        group_coverage_floor_enabled=True,
                        shrinkback_enabled=True,
                        group_multiplier_grid=(
                            (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20)
                            if parent["multiplier_grid_name"] == "coarse"
                            else (1.0, 1.01, 1.03, 1.05, 1.08, 1.12, 1.16, 1.20)
                        ),
                        temporal_multiplier_grid=(
                            (1.0, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20)
                            if parent["multiplier_grid_name"] == "coarse"
                            else (1.0, 1.01, 1.03, 1.05, 1.08, 1.12, 1.16, 1.20)
                        ),
                        partition=str(parent["partition"]),
                        artifact_namespace=namespace,
                        scaled_scores_options=(bool(parent["scaled_scores"]),),
                    )
                    paths = _resolve_artifact_paths(namespace)
                    base_cfg_path = tmp_root / f"{namespace}_policy.yaml"
                    sens_cfg_path = tmp_root / f"{namespace}_policy_sensitivity.yaml"
                    _write_validation_config(
                        cfg_template_path="configs/conformal_policy.yaml",
                        results_path=paths["results"],
                        group_metrics_path=paths["group_metrics"],
                        intervals_path=paths["intervals"],
                        backtest_monthly_path=Path(
                            "data/processed/conformal_backtest_monthly.parquet"
                        ),
                        backtest_alerts_path=Path(
                            "data/processed/conformal_backtest_alerts.parquet"
                        ),
                        status_path=paths["models_dir"] / "conformal_policy_status.json",
                        checks_path=paths["data_dir"] / "conformal_policy_checks.parquet",
                        out_path=base_cfg_path,
                    )
                    _write_validation_config(
                        cfg_template_path="configs/conformal_policy_sensitivity.yaml",
                        results_path=paths["results"],
                        group_metrics_path=paths["group_metrics"],
                        intervals_path=paths["intervals"],
                        backtest_monthly_path=Path(
                            "data/processed/conformal_backtest_monthly.parquet"
                        ),
                        backtest_alerts_path=Path(
                            "data/processed/conformal_backtest_alerts.parquet"
                        ),
                        status_path=paths["models_dir"]
                        / "conformal_policy_sensitivity_status.json",
                        checks_path=paths["data_dir"]
                        / "conformal_policy_sensitivity_checks.parquet",
                        out_path=sens_cfg_path,
                    )
                    validate_policy_mod.main(str(base_cfg_path))
                    validate_policy_mod.main(str(sens_cfg_path))
                    policy_status = json.loads(
                        (paths["models_dir"] / "conformal_policy_status.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    sensitivity_status = json.loads(
                        (
                            paths["models_dir"] / "conformal_policy_sensitivity_status.json"
                        ).read_text(encoding="utf-8")
                    )
                    width_attr = pd.read_parquet(paths["width_attr"])
                    final_test = (
                        width_attr.loc[
                            (width_attr["dataset_scope"] == "test")
                            & (width_attr["stage"] == "after_shrinkback")
                        ]
                        .iloc[0]
                        .to_dict()
                    )
                    stage_c_rows.append(
                        {
                            "experiment_stage": "C",
                            "artifact_namespace": namespace,
                            "parent_variant_key": str(parent["parent_variant_key"]),
                            "partition": str(parent["partition"]),
                            "scaled_scores": bool(parent["scaled_scores"]),
                            "min_group_size": int(parent["min_group_size"]),
                            "alpha_used_90": float(parent["alpha_used_90"]),
                            "group_coverage_floor_target_90": float(
                                parent["group_coverage_floor_target_90"]
                            ),
                            "coverage_guardband_90": float(parent["coverage_guardband_90"]),
                            "max_width_budget_90": float(parent["max_width_budget_90"]),
                            "multiplier_grid_name": str(parent["multiplier_grid_name"]),
                            "temporal_segment_floor_enabled": bool(temporal_floor_enabled),
                            "temporal_segment_min_size": int(temporal_segment_min_size),
                            "coverage_90": float(final_test["coverage_90"]),
                            "min_group_coverage_90": float(final_test["min_group_coverage_90"]),
                            "avg_width_90": float(final_test["avg_width_90"]),
                            "winkler_90": float(final_test["winkler_90"]),
                            "max_monthly_gap": float(final_test["max_monthly_gap"]),
                            "stability_over_time": float(final_test["stability_over_time"]),
                            "strict_overall_pass": bool(policy_status["strict_overall_pass"]),
                            "methodological_justification_pass": bool(
                                policy_status["methodological_justification_pass"]
                            ),
                            "failing_non_statistical_checks": policy_status[
                                "failing_non_statistical_checks"
                            ],
                            "checks_passed": int(policy_status["checks_passed"]),
                            "sensitivity_results": sensitivity_status.get("policy_sensitivity", {}),
                        }
                    )

    stage_c = pd.DataFrame(stage_c_rows)
    stage_c = stage_c.sort_values(
        [
            "strict_overall_pass",
            "methodological_justification_pass",
            "checks_passed",
            "winkler_90",
            "avg_width_90",
            "stability_over_time",
        ],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    if not stage_c.empty:
        stage_c["selection_rank"] = np.arange(1, len(stage_c) + 1, dtype=int)
        experiments.extend(stage_c.to_dict(orient="records"))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_df = pd.DataFrame(experiments)
    experiments_path = out_dir / "pd_conformal_gap_experiments.parquet"
    experiments_df.to_parquet(experiments_path, index=False)
    top_candidates_path = out_dir / "pd_conformal_gap_top_candidates.parquet"
    stage_c.head(10).to_parquet(top_candidates_path, index=False)

    best_final = stage_c.iloc[0].to_dict() if not stage_c.empty else {}
    status = {
        "schema_version": "2026-03-13.1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "status": "completed" if not stage_c.empty else "empty",
        "stage_a_evaluated": int(len(stage_a)),
        "stage_b_evaluated": int(len(stage_b)),
        "stage_c_evaluated": int(len(stage_c)),
        "generated_namespaces": generated_namespaces,
        "best_final_candidate": best_final,
        "experiments_path": str(experiments_path),
        "top_candidates_path": str(top_candidates_path),
        "success_gate": {
            "target_winkler_90": 1.20,
            "reviewable_winkler_90": 1.22,
            "target_min_group_coverage_90": 0.88,
            "target_coverage_90": 0.90,
        },
    }
    status_path = models_dir / "pd_conformal_gap_experiment_status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Saved PD conformal gap experiments: {}", experiments_path)
    logger.info("Saved PD conformal gap top candidates: {}", top_candidates_path)
    logger.info("Saved PD conformal gap status: {}", status_path)


if __name__ == "__main__":
    _ = argparse.ArgumentParser().parse_args()
    main()
