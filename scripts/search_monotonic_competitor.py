"""Search serious monotonic CatBoost competitors against the frozen clean baseline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_pd_model import _prepare_catboost_frame
from src.evaluation.fairness import (
    build_intersectional_groups,
    fairness_report,
    fairness_threshold_frontier,
)
from src.models.calibration import expected_calibration_error
from src.models.pd_model import TARGET, temporal_train_val_split
from src.models.venn_abers import VennAbersScoreCalibrator
from src.utils.io_utils import read_split_with_fe_fallback
from src.utils.replay_manifest import load_replay_manifest, manifest_section
from src.utils.threshold_semantics import load_threshold_semantics, resolve_operational_threshold


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _fit_calibrator(method: str, y_true: np.ndarray, scores: np.ndarray) -> Any:
    method_norm = str(method).strip().lower()
    if method_norm == "isotonic":
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        model.fit(scores, y_true)
        return model
    if method_norm == "venn_abers":
        model = VennAbersScoreCalibrator()
        model.fit(scores, y_true)
        return model
    if method_norm == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), y_true)
        return model
    raise ValueError(f"Unsupported calibration method for monotonic competitor search: {method}")


def _apply_calibrator(calibrator: Any, scores: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(scores), 0.0, 1.0)
    if isinstance(calibrator, VennAbersScoreCalibrator):
        return np.clip(calibrator.predict(scores), 0.0, 1.0)
    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    raise TypeError("Unsupported calibrator type")


def _constraints_vector(
    feature_names: list[str],
    *,
    profile_cfg: dict[str, Any],
    declared_constraints: dict[str, int],
) -> list[int]:
    if bool(profile_cfg.get("use_all_declared_constraints", False)):
        use_map = declared_constraints
    else:
        use_map = {str(k): int(v) for k, v in dict(profile_cfg.get("features") or {}).items()}
    return [int(use_map.get(feature, 0)) for feature in feature_names]


def _constraints_spec(
    feature_names: list[str],
    *,
    profile_cfg: dict[str, Any],
    declared_constraints: dict[str, int],
) -> tuple[list[int], str | None]:
    vector = _constraints_vector(
        feature_names,
        profile_cfg=profile_cfg,
        declared_constraints=declared_constraints,
    )
    constrained_pairs = [
        f"{feature}:{int(direction)}"
        for feature, direction in zip(feature_names, vector, strict=False)
        if int(direction) != 0
    ]
    if not constrained_pairs:
        return vector, None
    return vector, ",".join(constrained_pairs)


def _load_fairness_groups(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], float]:
    policy_path = Path("configs/fairness_policy.yaml")
    if not policy_path.exists():
        return {}, 0.5
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    attributes = payload.get("attributes", []) or []
    policy = payload.get("policy", {}) or {}
    groups: dict[str, np.ndarray] = {}
    for attr in attributes:
        name = str(attr.get("name", "")).strip()
        column = str(attr.get("column", "")).strip()
        if not name or not column or column not in df.columns:
            continue
        if str(attr.get("binning", "")).strip().lower() == "quartile":
            groups[name] = (
                pd.qcut(
                    pd.to_numeric(df[column], errors="coerce"),
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop",
                )
                .astype(str)
                .to_numpy()
            )
        else:
            groups[name] = df[column].astype(str).to_numpy()
    semantics = load_threshold_semantics()
    threshold = resolve_operational_threshold(
        semantics, default=float(policy.get("prediction_threshold", 0.5))
    )
    return groups, float(threshold)


def _resolve_outcome_mode(policy_payload: dict[str, Any]) -> str:
    raw = str(policy_payload.get("outcome_mode", "default")).strip().lower()
    if raw in {"approval", "approve", "good", "non_default"}:
        return "approval"
    return "default"


def _prepare_official_fairness_context(
    df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, Any]:
    policy_path = Path("configs/fairness_policy.yaml")
    if not policy_path.exists():
        return {
            "outcome_mode": "default",
            "base_groups": {},
            "all_groups": {},
            "y_true_eval": np.asarray(y_true, dtype=float),
            "y_pred_eval": np.asarray(y_pred_proba, dtype=float),
            "threshold": 0.5,
            "policy": {},
        }

    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    policy = dict(payload.get("policy") or {})
    groups_base, threshold = _load_fairness_groups(df)
    intersectional_cfg = dict(payload.get("intersectional") or {})
    groups_all = dict(groups_base)
    if bool(intersectional_cfg.get("enabled", True)):
        groups_all.update(
            build_intersectional_groups(
                groups_base,
                max_order=int(intersectional_cfg.get("max_order", 2)),
                min_group_size=int(intersectional_cfg.get("min_group_size", 300)),
            )
        )

    outcome_mode = _resolve_outcome_mode(policy)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred_proba, dtype=float)
    if outcome_mode == "approval":
        y_true_eval = 1.0 - y_true_arr
        y_pred_eval = 1.0 - y_pred_arr
    else:
        y_true_eval = y_true_arr
        y_pred_eval = y_pred_arr

    return {
        "outcome_mode": outcome_mode,
        "base_groups": groups_base,
        "all_groups": groups_all,
        "y_true_eval": y_true_eval,
        "y_pred_eval": y_pred_eval,
        "threshold": float(threshold),
        "policy": policy,
    }


def _sample_frame(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= int(max_rows):
        return df
    return df.sample(n=int(max_rows), random_state=int(random_state)).sort_index().copy()


def _fairness_ok(y_true: np.ndarray, y_prob: np.ndarray, df: pd.DataFrame) -> bool:
    groups, threshold = _load_fairness_groups(df)
    if not groups:
        return True
    report = fairness_report(
        y_true=y_true,
        y_pred_proba=y_prob,
        groups_dict=groups,
        threshold=threshold,
        dpd_threshold=0.10,
        eo_gap_threshold=0.11,
        dir_threshold=0.80,
    )
    return bool(report["passed_all"].all())


def _report_summary(report: pd.DataFrame) -> dict[str, Any]:
    if report.empty:
        return {
            "n_attributes": 0,
            "n_passed": 0,
            "passed_all": True,
            "worst_eo_gap": 0.0,
            "worst_dpd": 0.0,
            "min_dir": 1.0,
            "failed_attributes": [],
            "rows": [],
        }

    failed = report.loc[~report["passed_all"].astype(bool), "attribute"].astype(str).tolist()
    return {
        "n_attributes": int(len(report)),
        "n_passed": int(report["passed_all"].astype(bool).sum()),
        "passed_all": bool(report["passed_all"].astype(bool).all()),
        "worst_eo_gap": float(report["eo_gap"].astype(float).max()),
        "worst_dpd": float(report["dpd"].astype(float).max()),
        "min_dir": float(report["dir"].astype(float).min()),
        "failed_attributes": failed,
        "rows": report.to_dict(orient="records"),
    }


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


def _variant_params(
    params_base: dict[str, Any], variant: dict[str, Any], seed: int
) -> dict[str, Any]:
    params = dict(params_base)
    params["learning_rate"] = float(params_base.get("learning_rate", 0.03)) * float(
        variant.get("learning_rate_multiplier", 1.0)
    )
    params["l2_leaf_reg"] = float(params_base.get("l2_leaf_reg", 10.0)) * float(
        variant.get("l2_leaf_reg_multiplier", 1.0)
    )
    params["depth"] = max(3, int(params_base.get("depth", 6)) + int(variant.get("depth_offset", 0)))
    params["iterations"] = max(
        300,
        int(
            round(
                float(params_base.get("iterations", 1000))
                * float(variant.get("iterations_multiplier", 1.0))
            )
        ),
    )
    params["random_strength"] = float(params_base.get("random_strength", 1.0)) * float(
        variant.get("random_strength_multiplier", 1.0)
    )
    if "border_count_override" in variant:
        params["border_count"] = int(variant["border_count_override"])
    if "grow_policy_override" in variant:
        params["grow_policy"] = str(variant["grow_policy_override"])
    if "rsm_override" in variant:
        params["rsm"] = float(variant["rsm_override"])
    if "subsample_override" in variant:
        params["subsample"] = float(variant["subsample_override"])
        params["bootstrap_type"] = str(variant.get("bootstrap_type_override", "Bernoulli"))
    elif "bootstrap_type_override" in variant:
        params["bootstrap_type"] = str(variant["bootstrap_type_override"])
    if "bagging_temperature_override" in variant:
        params["bagging_temperature"] = float(variant["bagging_temperature_override"])
    if "min_data_in_leaf_override" in variant:
        params["min_data_in_leaf"] = int(variant["min_data_in_leaf_override"])
    params["random_seed"] = int(seed)
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search serious monotonic CatBoost competitors.")
    parser.add_argument("--config", default="configs/monotonic_competitor.yaml")
    parser.add_argument("--run-tag", default="untracked")
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    defaults = dict(cfg.get("defaults") or {})
    policy = dict(cfg.get("promotion_policy") or {})
    replay_manifest = load_replay_manifest(defaults.get("replay_manifest"))
    pd_manifest = manifest_section(replay_manifest, "pd")
    if not pd_manifest:
        raise ValueError("Frozen PD replay manifest is required for monotonic competitor search.")

    pd_cfg = _load_yaml(defaults.get("config_path", "configs/pd_model.champion.yaml"))
    data_cfg = dict(pd_cfg.get("data") or {})
    declared_constraints = dict(
        (pd_cfg.get("challenger_pipeline") or {}).get("monotonic_constraints") or {}
    )

    train = read_split_with_fe_fallback(
        data_cfg.get("train_path", "data/processed/train_fe.parquet")
    )
    cal = read_split_with_fe_fallback(
        data_cfg.get("calibration_path", "data/processed/calibration_fe.parquet")
    )
    test = read_split_with_fe_fallback(data_cfg.get("test_path", "data/processed/test_fe.parquet"))

    feature_names = [str(x) for x in pd_manifest.get("feature_names", [])]
    categorical = [str(x) for x in pd_manifest.get("categorical_features", [])]
    params_base = dict(pd_manifest.get("selected_params") or {})
    calibration_method = str(pd_manifest.get("selected_calibration_method", "venn_abers"))
    expected = dict(pd_manifest.get("expectations") or {})
    calibration_methods = [
        str(x).strip()
        for x in list(cfg.get("calibration_methods") or [calibration_method])
        if str(x).strip()
    ]
    if not calibration_methods:
        calibration_methods = [calibration_method]
    seed_offsets = [int(x) for x in list(cfg.get("seed_offsets") or [0])]
    base_seed = int(params_base.get("random_seed", 42))
    threshold_candidates = [float(x) for x in list(cfg.get("decision_threshold_candidates") or [])]
    sampling_cfg = dict(cfg.get("sampling") or {})
    sampling_seed = int(sampling_cfg.get("random_state", 42))

    train = _sample_frame(
        train,
        int(sampling_cfg["train_max_rows"]) if "train_max_rows" in sampling_cfg else None,
        sampling_seed,
    )
    cal = _sample_frame(
        cal,
        int(sampling_cfg["calibration_max_rows"])
        if "calibration_max_rows" in sampling_cfg
        else None,
        sampling_seed + 1,
    )
    test = _sample_frame(
        test,
        int(sampling_cfg["test_max_rows"]) if "test_max_rows" in sampling_cfg else None,
        sampling_seed + 2,
    )

    train_fit, train_val = temporal_train_val_split(train, val_fraction=0.15, date_col="issue_d")
    X_fit = _prepare_catboost_frame(train_fit, feature_names, categorical)
    X_val = _prepare_catboost_frame(train_val, feature_names, categorical)
    X_cal = _prepare_catboost_frame(cal, feature_names, categorical)
    X_test = _prepare_catboost_frame(test, feature_names, categorical)
    y_fit = train_fit[TARGET].astype(int).to_numpy()
    y_val = train_val[TARGET].astype(int).to_numpy()
    y_cal = cal[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    search_rows: list[dict[str, Any]] = []
    for profile_name, profile_cfg in dict(cfg.get("constraint_profiles") or {}).items():
        vector, constraints_spec = _constraints_spec(
            feature_names,
            profile_cfg=dict(profile_cfg or {}),
            declared_constraints=declared_constraints,
        )
        for variant in list(cfg.get("param_variants") or []):
            variant = dict(variant or {})
            for calibrator_method in calibration_methods:
                for seed_offset in seed_offsets:
                    seed = base_seed + int(seed_offset)
                    params = _variant_params(params_base, variant, seed)
                    if constraints_spec:
                        params["monotone_constraints"] = constraints_spec
                    else:
                        params.pop("monotone_constraints", None)

                    model = CatBoostClassifier(**params)
                    model.fit(
                        X_fit,
                        y_fit,
                        cat_features=categorical,
                        eval_set=(X_val, y_val),
                        use_best_model=True,
                        verbose=False,
                    )
                    cal_scores = model.predict_proba(X_cal)[:, 1]
                    calibrator = _fit_calibrator(calibrator_method, y_cal, cal_scores)
                    test_scores = model.predict_proba(X_test)[:, 1]
                    test_prob = _apply_calibrator(calibrator, test_scores)

                    auc = float(roc_auc_score(y_test, test_prob))
                    brier = float(brier_score_loss(y_test, test_prob))
                    ece = float(expected_calibration_error(y_test, test_prob))
                    auc_drop = float(float(expected.get("auc_roc", auc)) - auc)
                    brier_increase = float(
                        brier / max(float(expected.get("brier_score", brier)), 1e-9) - 1.0
                    )
                    ece_delta = float(ece - float(expected.get("ece", ece)))
                    fairness_ctx = _prepare_official_fairness_context(
                        test,
                        y_true=y_test,
                        y_pred_proba=test_prob,
                    )
                    groups_base = dict(fairness_ctx["base_groups"])
                    groups_all = dict(fairness_ctx["all_groups"])
                    default_threshold = float(fairness_ctx["threshold"])
                    fairness_policy = dict(fairness_ctx["policy"])
                    dpd_threshold = float(fairness_policy.get("dpd_threshold", 0.10))
                    eo_gap_threshold = float(fairness_policy.get("eo_gap_threshold", 0.11))
                    dir_threshold = float(fairness_policy.get("dir_threshold", 0.80))
                    n_attributes = int(len(groups_all))
                    if groups_all:
                        thresholds = threshold_candidates or [float(default_threshold)]
                        frontier = fairness_threshold_frontier(
                            y_true=np.asarray(fairness_ctx["y_true_eval"], dtype=float),
                            y_pred_proba=np.asarray(fairness_ctx["y_pred_eval"], dtype=float),
                            groups_dict=groups_all,
                            thresholds=thresholds,
                            primary_threshold=float(default_threshold),
                            dpd_threshold=dpd_threshold,
                            eo_gap_threshold=eo_gap_threshold,
                            dir_threshold=dir_threshold,
                        )
                        selected_threshold_info = _select_threshold_from_frontier(
                            frontier,
                            y_pred_proba_eval=np.asarray(fairness_ctx["y_pred_eval"], dtype=float),
                        )
                        selected_threshold = float(
                            selected_threshold_info.get("selected_threshold", default_threshold)
                        )
                        report_all = fairness_report(
                            y_true=np.asarray(fairness_ctx["y_true_eval"], dtype=float),
                            y_pred_proba=np.asarray(fairness_ctx["y_pred_eval"], dtype=float),
                            groups_dict=groups_all,
                            threshold=selected_threshold,
                            dpd_threshold=dpd_threshold,
                            eo_gap_threshold=eo_gap_threshold,
                            dir_threshold=dir_threshold,
                        )
                        report_base = fairness_report(
                            y_true=np.asarray(fairness_ctx["y_true_eval"], dtype=float),
                            y_pred_proba=np.asarray(fairness_ctx["y_pred_eval"], dtype=float),
                            groups_dict=groups_base,
                            threshold=selected_threshold,
                            dpd_threshold=dpd_threshold,
                            eo_gap_threshold=eo_gap_threshold,
                            dir_threshold=dir_threshold,
                        )
                        summary_all = _report_summary(report_all)
                        summary_base = _report_summary(report_base)
                        fairness_ok = bool(summary_all["passed_all"])
                        fairness_n_passed = int(summary_all["n_passed"])
                        fairness_worst_eo_gap = float(summary_all["worst_eo_gap"])
                        fairness_approval_rate = float(
                            selected_threshold_info.get("approval_rate", 0.0)
                        )
                    else:
                        fairness_ok = True
                        selected_threshold = float(default_threshold)
                        fairness_n_passed = 0
                        fairness_worst_eo_gap = 0.0
                        summary_all = _report_summary(pd.DataFrame())
                        summary_base = _report_summary(pd.DataFrame())
                        fairness_approval_rate = float(
                            (
                                np.asarray(fairness_ctx["y_pred_eval"], dtype=float)
                                >= float(default_threshold)
                            ).mean()
                        )

                    promotable = bool(
                        auc_drop <= float(policy.get("max_auc_drop", 0.003))
                        and brier_increase <= float(policy.get("max_brier_increase_pct", 0.015))
                        and ece_delta <= float(policy.get("max_ece_delta", 0.01))
                        and (fairness_ok or not bool(policy.get("require_fairness_ok", True)))
                    )
                    search_rows.append(
                        {
                            "profile": profile_name,
                            "variant_id": str(variant.get("id", "base")),
                            "calibration_method": calibrator_method,
                            "seed": seed,
                            "seed_offset": int(seed_offset),
                            "auc": auc,
                            "brier": brier,
                            "ece": ece,
                            "auc_drop": auc_drop,
                            "brier_increase_pct": brier_increase,
                            "ece_delta": ece_delta,
                            "fairness_ok": fairness_ok,
                            "fairness_n_passed": fairness_n_passed,
                            "fairness_n_attributes": int(n_attributes),
                            "selected_threshold": selected_threshold,
                            "fairness_worst_eo_gap": fairness_worst_eo_gap,
                            "fairness_approval_rate": fairness_approval_rate,
                            "outcome_mode": str(fairness_ctx["outcome_mode"]),
                            "official_fairness_all_attributes": summary_all,
                            "official_fairness_base_only": summary_base,
                            "promotable": promotable,
                            "params": params,
                            "n_constrained": int(sum(1 for value in vector if value != 0)),
                        }
                    )

    if not search_rows:
        raise ValueError("No monotonic competitor variants were evaluated.")

    detail_rows = sorted(
        search_rows,
        key=lambda row: (
            not bool(row["promotable"]),
            float(row["auc_drop"]),
            float(row["brier_increase_pct"]),
            float(row["ece_delta"]),
        ),
    )

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_keys = ["profile", "variant_id", "calibration_method"]
    grouped = pd.DataFrame(search_rows).groupby(aggregate_keys, dropna=False, as_index=False)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        profile_name, variant_id, calibrator_method = keys
        auc_values = frame["auc"].astype(float).to_numpy()
        brier_values = frame["brier"].astype(float).to_numpy()
        ece_values = frame["ece"].astype(float).to_numpy()
        auc_drop_values = frame["auc_drop"].astype(float).to_numpy()
        brier_inc_values = frame["brier_increase_pct"].astype(float).to_numpy()
        ece_delta_values = frame["ece_delta"].astype(float).to_numpy()
        fairness_values = frame["fairness_ok"].astype(bool).to_numpy()
        promotable_values = frame["promotable"].astype(bool).to_numpy()
        aggregate_rows.append(
            {
                "profile": str(profile_name),
                "variant_id": str(variant_id),
                "calibration_method": str(calibrator_method),
                "n_runs": int(len(frame)),
                "n_seeds": int(len(frame)),
                "n_constrained": int(frame["n_constrained"].max()),
                "auc_mean": float(np.mean(auc_values)),
                "auc_std": float(np.std(auc_values)),
                "brier_mean": float(np.mean(brier_values)),
                "brier_std": float(np.std(brier_values)),
                "ece_mean": float(np.mean(ece_values)),
                "ece_std": float(np.std(ece_values)),
                "auc_drop_mean": float(np.mean(auc_drop_values)),
                "auc_drop_max": float(np.max(auc_drop_values)),
                "brier_increase_pct_mean": float(np.mean(brier_inc_values)),
                "brier_increase_pct_max": float(np.max(brier_inc_values)),
                "ece_delta_mean": float(np.mean(ece_delta_values)),
                "ece_delta_max": float(np.max(ece_delta_values)),
                "fairness_ok_all": bool(np.all(fairness_values)),
                "fairness_ok_rate": float(np.mean(fairness_values.astype(float))),
                "fairness_n_passed_min": int(frame["fairness_n_passed"].min()),
                "fairness_n_passed_mean": float(frame["fairness_n_passed"].mean()),
                "fairness_n_attributes": int(frame["fairness_n_attributes"].max()),
                "outcome_mode": str(frame.iloc[0]["outcome_mode"]),
                "official_fairness_all_attributes_reference": dict(
                    frame.iloc[0]["official_fairness_all_attributes"]
                ),
                "official_fairness_base_only_reference": dict(
                    frame.iloc[0]["official_fairness_base_only"]
                ),
                "selected_threshold_mean": float(frame["selected_threshold"].mean()),
                "selected_threshold_values": sorted(
                    {float(x) for x in frame["selected_threshold"].astype(float).tolist()}
                ),
                "worst_eo_gap_mean": float(frame["fairness_worst_eo_gap"].mean()),
                "approval_rate_mean": float(frame["fairness_approval_rate"].mean()),
                "promotable_all_seeds": bool(np.all(promotable_values)),
                "promotable_any_seed": bool(np.any(promotable_values)),
                "params_reference": dict(frame.iloc[0]["params"]),
            }
        )

    aggregate_rows = sorted(
        aggregate_rows,
        key=lambda row: (
            not bool(row["promotable_all_seeds"]),
            float(row["auc_drop_mean"]),
            float(row["brier_increase_pct_mean"]),
            float(row["ece_delta_mean"]),
            float(row["auc_std"]),
        ),
    )
    best = dict(aggregate_rows[0])
    resolved_run_tag = str(args.run_tag or "untracked").strip() or "untracked"
    payload = {
        "schema_version": "2026-03-29.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "baseline_manifest": str(defaults.get("replay_manifest", "")),
        "baseline_expectations": expected,
        "promotion_policy": policy,
        "calibration_methods": calibration_methods,
        "seed_offsets": seed_offsets,
        "decision_threshold_candidates": threshold_candidates,
        "detail_results": detail_rows,
        "aggregate_results": aggregate_rows,
        "best_variant": best,
        "best_variant_promotable": bool(best["promotable_all_seeds"]),
    }

    output_path = Path(
        str(defaults.get("output_path", "models/monotonic_competitor_search.json")).format(
            run_tag=resolved_run_tag
        )
    )
    best_output_path = Path(
        str(defaults.get("best_output_path", "models/monotonic_competitor_best.json")).format(
            run_tag=resolved_run_tag
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    best_output_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    print(
        "[monotonic] best_variant={}::{}::{} promotable_all_seeds={} auc_drop_mean={:.6f} brier_increase_pct_mean={:.6f} ece_delta_mean={:.6f}".format(
            best["profile"],
            best["variant_id"],
            best["calibration_method"],
            bool(best["promotable_all_seeds"]),
            float(best["auc_drop_mean"]),
            float(best["brier_increase_pct_mean"]),
            float(best["ece_delta_mean"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
