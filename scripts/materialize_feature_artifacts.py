"""Materialize canonical feature-engineered splits and feature artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.feature_engineering import (
    TARGET,
    apply_woe_encoders,
    build_feature_config,
    fit_woe_encoders,
    run_feature_pipeline,
    save_feature_artifacts,
)
from src.utils.pipeline_runtime import (
    write_last_valid_artifact,
    write_runtime_checkpoint,
    write_runtime_status,
)


def _load_split(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    logger.info("Loaded split {} with shape {}", path, frame.shape)
    return frame


def _featureize_split(frame: pd.DataFrame) -> pd.DataFrame:
    out = run_feature_pipeline(frame)
    if TARGET in frame.columns and TARGET not in out.columns:
        out[TARGET] = frame[TARGET]
    return out


def main(output_dir: str = "data/processed") -> None:
    stage_name = "feature_artifacts"
    write_runtime_status(stage_name, phase="loading_splits", state="running")
    output_path = Path(output_dir)
    train = _load_split(output_path / "train.parquet")
    calibration = _load_split(output_path / "calibration.parquet")
    test = _load_split(output_path / "test.parquet")

    train_fe = _featureize_split(train)
    calibration_fe = _featureize_split(calibration)
    test_fe = _featureize_split(test)

    encoders, iv_scores = fit_woe_encoders(train_fe)
    train_fe = apply_woe_encoders(train_fe, encoders)
    calibration_fe = apply_woe_encoders(calibration_fe, encoders)
    test_fe = apply_woe_encoders(test_fe, encoders)
    write_runtime_checkpoint(
        stage_name,
        "feature_engineering_complete",
        {
            "train_rows": int(len(train_fe)),
            "calibration_rows": int(len(calibration_fe)),
            "test_rows": int(len(test_fe)),
            "woe_feature_count": int(len(encoders)),
        },
    )
    write_runtime_status(stage_name, phase="persisting_artifacts", state="running")

    feature_config = build_feature_config(train_fe, iv_scores=iv_scores)
    save_feature_artifacts(
        train_df=train_fe,
        calibration_df=calibration_fe,
        test_df=test_fe,
        feature_config=feature_config,
        woe_encoders=encoders,
        output_dir=output_dir,
    )
    write_last_valid_artifact(
        stage_name,
        artifact_key="feature_manifest_v2",
        artifact_path=output_path / "feature_manifest_v2.parquet",
        extra={
            "core_feature_count": int(len(feature_config.get("CORE_FEATURE_SET_V2", []))),
            "challenger_feature_count": int(
                len(feature_config.get("CHALLENGER_FEATURE_POOL_V2", []))
            ),
        },
    )
    write_runtime_status(
        stage_name,
        phase="completed",
        state="completed",
        extra={
            "feature_manifest_path": str(output_path / "feature_manifest_v2.parquet"),
            "feature_config_path": str(output_path / "feature_config.pkl"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
