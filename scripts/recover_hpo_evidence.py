"""Recover HPO traceability artifacts from an existing Optuna study.

This script does not execute new Optuna trials. It replays top trials across
multiple seeds (seed replay) and can patch `models/pd_training_record.pkl`
with recovered HPO metadata for thesis/audit reporting.

Usage examples:
    # Full-data replay + patch training record (no new HPO trials)
    uv run python scripts/recover_hpo_evidence.py \
      --config configs/pd_model.yaml \
      --cutoff-trial-number 925 \
      --update-training-record

    # Quick smoke on a sample
    uv run python scripts/recover_hpo_evidence.py \
      --config configs/pd_model.yaml \
      --sample-size 250000
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from scripts.train_pd_model import (
    _normalize_percent_columns,
    _prepare_catboost_frame,
    _replay_top_optuna_trials,
    load_config,
)
from src.models.pd_model import TARGET, resolve_feature_sets, temporal_train_val_split
from src.utils.io_utils import read_split_with_fe_fallback


def _to_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) <= 0:
        return None
    return int(value)


def _load_study_summary(
    *,
    storage: str,
    study_name: str,
    base_params: dict[str, Any],
    cutoff_trial_number: int | None,
) -> dict[str, Any]:
    import optuna

    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [t for t in study.trials if t.state.name != "RUNNING"]
    if cutoff_trial_number is not None:
        trials = [t for t in trials if int(t.number) <= int(cutoff_trial_number)]

    if not trials:
        raise ValueError("No trials available after applying filters (RUNNING/cutoff).")

    states = Counter([t.state.name for t in trials])
    complete = [t for t in trials if t.state.name == "COMPLETE" and t.value is not None]
    if not complete:
        raise ValueError("No COMPLETE Optuna trials available after applying filters.")

    best = max(complete, key=lambda t: float(t.value))
    best_trial_params = dict(best.params)
    best_params_merged = {**base_params, **best_trial_params}

    return {
        "study_name": study_name,
        "study_storage": storage,
        "cutoff_trial_number": cutoff_trial_number,
        "n_trials_total_in_study": int(len(study.trials)),
        "n_trials_selected": int(len(trials)),
        "state_counts_selected": dict(states),
        "n_complete_selected": int(len(complete)),
        "best_trial_number_selected": int(best.number),
        "best_validation_auc_selected": float(best.value),
        "best_trial_params_selected": best_trial_params,
        "best_params_selected_merged_with_base": best_params_merged,
    }


def _load_pd_splits(config: dict[str, Any], sample_size: int | None) -> tuple[pd.DataFrame, ...]:
    train = _normalize_percent_columns(read_split_with_fe_fallback(config["data"]["train_path"]))
    test = _normalize_percent_columns(read_split_with_fe_fallback(config["data"]["test_path"]))
    cal = _normalize_percent_columns(
        read_split_with_fe_fallback(config["data"]["calibration_path"])
    )

    if sample_size is not None:
        if sample_size < len(train):
            train = train.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(test):
            test = test.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(cal):
            cal = cal.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return train, cal, test


def _resolve_pd_features(
    config: dict[str, Any],
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    feature_src_cfg = config.get("feature_source", {})
    feature_mode = feature_src_cfg.get("mode", "auto")
    feature_config_path = feature_src_cfg.get(
        "feature_config_path", "data/processed/feature_config.pkl"
    )

    feature_sets = resolve_feature_sets(
        train,
        feature_source=feature_mode,
        feature_config_path=feature_config_path,
    )
    catboost_features = feature_sets["catboost_features"]
    categorical_features = feature_sets["categorical_features"]

    catboost_features = [
        c
        for c in catboost_features
        if c in train.columns and c in cal.columns and c in test.columns
    ]
    categorical_features = [c for c in categorical_features if c in catboost_features]

    if not catboost_features:
        raise ValueError("No CatBoost features resolved across train/cal/test splits.")
    return catboost_features, categorical_features


def _patch_training_record(
    *,
    record_path: Path,
    study_summary: dict[str, Any],
    seed_replay_report: dict[str, Any],
) -> None:
    if not record_path.exists():
        raise FileNotFoundError(f"Training record not found: {record_path}")

    with open(record_path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict):
        raise TypeError("Training record payload is not a dict.")

    best_params = (
        seed_replay_report.get("selected_params")
        or study_summary.get("best_params_selected_merged_with_base")
        or {}
    )
    record["hpo_trials_executed"] = int(study_summary["n_trials_selected"])
    record["hpo_best_validation_auc"] = float(study_summary["best_validation_auc_selected"])
    record["optuna_best_params"] = dict(best_params)
    record["seed_replay_report"] = dict(seed_replay_report)
    record["hpo_recovery_metadata"] = {
        "recovered_at_utc": datetime.now(tz=UTC).isoformat(),
        "mode": "posthoc_from_existing_study",
        "study_name": study_summary["study_name"],
        "study_storage": study_summary["study_storage"],
        "cutoff_trial_number": study_summary["cutoff_trial_number"],
        "selected_trial_for_best_auc": study_summary["best_trial_number_selected"],
    }

    tuned = record.get("catboost_tuned_metrics", {})
    if isinstance(tuned, dict):
        tuned["hpo_trials_executed"] = int(study_summary["n_trials_selected"])
        tuned["hpo_best_validation_auc"] = float(study_summary["best_validation_auc_selected"])
        tuned["best_params"] = dict(best_params)
        if seed_replay_report.get("enabled"):
            tuned["seed_replay_enabled"] = True
            tuned["seed_replay_selected_trial"] = seed_replay_report.get("selected_trial")
        record["catboost_tuned_metrics"] = tuned

    with open(record_path, "wb") as f:
        pickle.dump(record, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pd_model.yaml")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--cutoff-trial-number", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default="reports/hpo_recovery_seed_replay.json",
        help="Path to write recovery report JSON.",
    )
    parser.add_argument(
        "--update-training-record",
        action="store_true",
        help="Patch models/pd_training_record.pkl with recovered HPO fields.",
    )
    args = parser.parse_args()

    sample_size = _to_int_or_none(args.sample_size)
    cutoff_trial_number = _to_int_or_none(args.cutoff_trial_number)

    config = load_config(args.config)
    hpo_cfg = dict(config.get("hpo", {}))
    hpo_cfg["enabled"] = True

    storage = str(hpo_cfg.get("study_storage", "") or "").strip()
    study_name = str(hpo_cfg.get("study_name", "") or "").strip()
    if not storage or not study_name:
        raise ValueError(
            "Missing hpo.study_storage or hpo.study_name in config. "
            "These are required for post-hoc recovery."
        )

    base_params = dict(config.get("model", {}).get("params", {}) or {})
    study_summary = _load_study_summary(
        storage=storage,
        study_name=study_name,
        base_params=base_params,
        cutoff_trial_number=cutoff_trial_number,
    )

    train, cal, test = _load_pd_splits(config, sample_size=sample_size)
    catboost_features, categorical_features = _resolve_pd_features(config, train, cal, test)

    val_cfg = config.get("validation", {})
    seed_replay_cfg = val_cfg.get("seed_replay", {})
    val_fraction = float(val_cfg.get("val_from_tail_fraction_of_train", 0.15))
    train_fit, train_val = temporal_train_val_split(
        train, val_fraction=val_fraction, date_col="issue_d"
    )
    y_train_fit = train_fit[TARGET].astype(int)
    y_val = train_val[TARGET].astype(int)

    X_train_fit_cb = _prepare_catboost_frame(train_fit, catboost_features, categorical_features)
    X_val_cb = _prepare_catboost_frame(train_val, catboost_features, categorical_features)

    seed_replay_report = _replay_top_optuna_trials(
        hpo_cfg=hpo_cfg,
        base_params=base_params,
        X_train_fit_cb=X_train_fit_cb,
        y_train_fit=y_train_fit,
        X_val_cb=X_val_cb,
        y_val=y_val,
        cat_features=categorical_features,
        seeds=[int(s) for s in seed_replay_cfg.get("seeds", [42, 52, 62])],
        top_k_trials=int(seed_replay_cfg.get("top_k_trials", 3)),
        prioritize_gate_pass=bool(seed_replay_cfg.get("prioritize_gate_pass", True)),
    )

    payload = {
        "schema_version": "2026-02-28.1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "config_path": args.config,
        "sample_size": sample_size,
        "study_summary": study_summary,
        "seed_replay_report": seed_replay_report,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved HPO recovery report: {}", output_path)

    if args.update_training_record:
        record_path = Path("models/pd_training_record.pkl")
        _patch_training_record(
            record_path=record_path,
            study_summary=study_summary,
            seed_replay_report=seed_replay_report,
        )
        logger.info("Updated training record with recovered HPO evidence: {}", record_path)

    logger.info(
        "Recovery summary | selected_trials={} | best_trial={} | best_val_auc={:.6f} | "
        "seed_replay_enabled={} | seed_replay_selected_trial={}",
        study_summary["n_trials_selected"],
        study_summary["best_trial_number_selected"],
        study_summary["best_validation_auc_selected"],
        bool(seed_replay_report.get("enabled")),
        seed_replay_report.get("selected_trial"),
    )


if __name__ == "__main__":
    main()
