"""Benchmark PD across fit-only, HPO, and full-stage CPU vs GPU paths."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_pd_model import _prepare_catboost_frame  # noqa: E402
from src.models.calibration import expected_calibration_error  # noqa: E402
from src.models.optuna_tuning import train_catboost_tuned_optuna  # noqa: E402
from src.models.pd_model import (  # noqa: E402
    TARGET,
    _catboost_base_params,
    resolve_feature_sets,
    temporal_train_val_split,
    train_catboost_default,
)


def _artifact_root(run_tag: str) -> Path:
    return ROOT / "reports" / "gpu_replay" / run_tag / "artifacts"


def _sample_train_val(
    train_df: pd.DataFrame,
    *,
    fit_sample_size: int,
    val_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fit, train_val = temporal_train_val_split(train_df, val_fraction=0.15, date_col="issue_d")
    if fit_sample_size > 0 and len(train_fit) > fit_sample_size:
        train_fit = train_fit.tail(fit_sample_size).copy()
    if val_sample_size > 0 and len(train_val) > val_sample_size:
        train_val = train_val.tail(val_sample_size).copy()
    return train_fit, train_val


def _score_binary(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_true_arr = y_true.astype(int).to_numpy()
    return {
        "auc": float(roc_auc_score(y_true_arr, y_prob)),
        "brier": float(brier_score_loss(y_true_arr, y_prob)),
        "ece": float(expected_calibration_error(y_true_arr, y_prob)),
    }


def _run_fit_only(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit_sample_size: int,
    val_sample_size: int,
) -> list[dict[str, Any]]:
    feature_sets = resolve_feature_sets(train_df, feature_source="auto")
    feature_cols = feature_sets["catboost_features"]
    cat_features = feature_sets["categorical_features"]
    train_fit, train_val = _sample_train_val(
        train_df,
        fit_sample_size=fit_sample_size,
        val_sample_size=val_sample_size,
    )
    X_fit = _prepare_catboost_frame(train_fit, feature_cols, cat_features)
    X_val = _prepare_catboost_frame(train_val, feature_cols, cat_features)
    X_test = _prepare_catboost_frame(test_df, feature_cols, cat_features)

    rows: list[dict[str, Any]] = []
    for backend in ("cpu", "gpu"):
        params = {
            **_catboost_base_params(
                {
                    "iterations": 3000,
                    "verbose": 0,
                    "task_type": "GPU" if backend == "gpu" else "CPU",
                }
            )
        }
        if backend == "gpu":
            params["devices"] = "0"

        started = time.perf_counter()
        model, metrics = train_catboost_default(
            X_fit,
            train_fit[TARGET].astype(int),
            X_val,
            train_val[TARGET].astype(int),
            X_test=X_test,
            y_test=test_df[TARGET].astype(int),
            cat_features=cat_features,
            params=params,
        )
        fit_seconds = time.perf_counter() - started
        pred_started = time.perf_counter()
        y_prob_test = model.predict_proba(X_test)[:, 1]
        predict_seconds = time.perf_counter() - pred_started
        scores = _score_binary(test_df[TARGET], y_prob_test)
        rows.append(
            {
                "stage": "fit_only",
                "backend": backend,
                "rows_fit": int(len(train_fit)),
                "rows_val": int(len(train_val)),
                "rows_test": int(len(test_df)),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "total_seconds": float(fit_seconds + predict_seconds),
                "best_iteration": int(metrics.get("best_iteration", 0)),
                **scores,
            }
        )
    return rows


def _run_hpo_backend(
    *,
    backend: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit_sample_size: int,
    val_sample_size: int,
    n_trials: int,
) -> dict[str, Any]:
    feature_sets = resolve_feature_sets(train_df, feature_source="auto")
    feature_cols = feature_sets["catboost_features"]
    cat_features = feature_sets["categorical_features"]
    train_fit, train_val = _sample_train_val(
        train_df,
        fit_sample_size=fit_sample_size,
        val_sample_size=val_sample_size,
    )
    X_fit = _prepare_catboost_frame(train_fit, feature_cols, cat_features)
    X_val = _prepare_catboost_frame(train_val, feature_cols, cat_features)
    X_test = _prepare_catboost_frame(test_df, feature_cols, cat_features)

    base_params = {
        "iterations": 1200,
        "verbose": 0,
        "task_type": "GPU" if backend == "gpu" else "CPU",
    }
    if backend == "gpu":
        base_params["devices"] = "0"
    started = time.perf_counter()
    model, metrics = train_catboost_tuned_optuna(
        X_fit,
        train_fit[TARGET].astype(int),
        X_val,
        train_val[TARGET].astype(int),
        X_test=X_test,
        y_test=test_df[TARGET].astype(int),
        cat_features=cat_features,
        base_params=base_params,
        n_trials=n_trials,
        timeout_minutes=0,
        n_startup_trials=min(4, n_trials),
        pruner_n_startup_trials=min(3, n_trials),
        pruner_n_warmup_steps=25,
        use_pruning_callback=False,
    )
    total_seconds = time.perf_counter() - started
    y_prob_test = model.predict_proba(X_test)[:, 1]
    scores = _score_binary(test_df[TARGET], y_prob_test)
    return {
        "stage": "hpo",
        "backend": backend,
        "rows_fit": int(len(train_fit)),
        "rows_val": int(len(train_val)),
        "rows_test": int(len(test_df)),
        "n_trials": int(n_trials),
        "status": "completed",
        "total_seconds": float(total_seconds),
        "best_iteration": int(metrics.get("best_iteration", 0)),
        "best_validation_auc": float(metrics.get("validation_auc", np.nan)),
        "study_name": str(metrics.get("study_name", "")),
        **scores,
    }


def _run_hpo(
    *,
    root: Path,
    fit_sample_size: int,
    val_sample_size: int,
    n_trials: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in ("cpu", "gpu"):
        worker_json = root / f"hpo_{backend}_worker.json"
        worker_log = root / f"hpo_{backend}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-mode",
            "hpo",
            "--worker-output",
            str(worker_json),
            "--worker-backend",
            backend,
            "--hpo-sample-size",
            str(fit_sample_size),
            "--hpo-val-size",
            str(val_sample_size),
            "--hpo-trials",
            str(n_trials),
        ]
        with worker_log.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, check=False
            )
        if proc.returncode == 0 and worker_json.exists():
            rows.append(json.loads(worker_json.read_text(encoding="utf-8")))
            continue
        rows.append(
            {
                "stage": "hpo",
                "backend": backend,
                "rows_fit": int(fit_sample_size),
                "rows_val": int(val_sample_size),
                "rows_test": int(
                    len(pd.read_parquet(ROOT / "data" / "processed" / "test_fe.parquet"))
                ),
                "n_trials": int(n_trials),
                "status": "failed",
                "return_code": int(proc.returncode),
                "log_path": str(worker_log),
            }
        )
    return rows


def _parse_cpu_pd_seconds(run_tag: str) -> float | None:
    path = ROOT / "reports" / "run_logs" / run_tag / "master.log"
    if not path.exists():
        return None
    pattern = re.compile(
        r"^\[(?P<ts>.+?)\] STEP_SUBPHASE_(?P<kind>START|END) name=(?P<name>\S+) idx=(?P<idx>\d+/\d+)(?: cmd=(?P<cmd>.*)| ec=(?P<ec>\d+))?$"
    )
    starts: dict[tuple[str, str], tuple[datetime, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        ts = datetime.fromisoformat(match.group("ts"))
        key = (match.group("name"), match.group("idx"))
        if match.group("kind") == "START":
            starts[key] = (ts, match.group("cmd") or "")
            continue
        started = starts.get(key)
        if not started:
            continue
        start_ts, cmd = started
        if "scripts/train_pd_model.py" in cmd:
            return float((ts - start_ts).total_seconds())
    return None


def _parse_gpu_pd_metrics(run_tag: str) -> dict[str, Any]:
    summary_path = ROOT / "reports" / "gpu_replay" / run_tag / "run_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in payload.get("stage_results", []):
        if row.get("stage") == "pd":
            return row
    return {}


def _parse_final_metrics_from_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    pattern = re.compile(
        r"Final metrics \| AUC=(?P<auc>[0-9.]+)\s+Gini=(?P<gini>[0-9.]+)\s+KS=(?P<ks>[0-9.]+)\s+Brier=(?P<brier>[0-9.]+)\s+ECE=(?P<ece>[0-9.]+)"
    )
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        match = pattern.search(line)
        if match:
            return {k: float(v) for k, v in match.groupdict().items()}
    return {}


def _run_full_stage(
    *,
    cpu_run_tag: str,
    gpu_run_tag: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cpu_metrics_path = ROOT / "data" / "processed" / "model_comparison.json"
    cpu_metrics = (
        json.loads(cpu_metrics_path.read_text(encoding="utf-8"))
        if cpu_metrics_path.exists()
        else {}
    )
    cpu_row = (
        cpu_metrics.get("current_model", {}) if cpu_metrics.get("run_tag") == cpu_run_tag else {}
    )
    cpu_seconds = _parse_cpu_pd_seconds(cpu_run_tag)
    rows.append(
        {
            "stage": "full_stage",
            "backend": "cpu",
            "total_seconds": float(cpu_seconds or 0.0),
            "auc": float(cpu_row.get("auc", np.nan)),
            "brier": float(cpu_row.get("brier", np.nan)),
            "ece": float(cpu_row.get("ece", np.nan)),
        }
    )

    gpu_stage = _parse_gpu_pd_metrics(gpu_run_tag)
    gpu_log_path = ROOT / str(gpu_stage.get("log_path", ""))
    gpu_metrics = _parse_final_metrics_from_log(gpu_log_path)
    rows.append(
        {
            "stage": "full_stage",
            "backend": "gpu",
            "total_seconds": float(gpu_stage.get("duration_seconds", 0.0)),
            "auc": float(gpu_metrics.get("auc", np.nan)),
            "brier": float(gpu_metrics.get("brier", np.nan)),
            "ece": float(gpu_metrics.get("ece", np.nan)),
            "peak_gpu_util": float(gpu_stage.get("gpu_metrics", {}).get("peak_gpu_util", np.nan)),
            "peak_memory_used_mb": float(
                gpu_stage.get("gpu_metrics", {}).get("peak_memory_used_mb", np.nan)
            ),
        }
    )
    return rows


def main(
    *,
    run_tag: str,
    fit_sample_size: int,
    fit_val_size: int,
    hpo_sample_size: int,
    hpo_val_size: int,
    hpo_trials: int,
    cpu_run_tag: str,
    gpu_run_tag: str,
) -> None:
    root = _artifact_root(run_tag)
    data_dir = root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(ROOT / "data" / "processed" / "train_fe.parquet")
    test_df = pd.read_parquet(ROOT / "data" / "processed" / "test_fe.parquet")

    rows: list[dict[str, Any]] = []
    rows.extend(
        _run_fit_only(
            train_df=train_df,
            test_df=test_df,
            fit_sample_size=fit_sample_size,
            val_sample_size=fit_val_size,
        )
    )
    rows.extend(
        _run_hpo(
            root=root,
            fit_sample_size=hpo_sample_size,
            val_sample_size=hpo_val_size,
            n_trials=hpo_trials,
        )
    )
    rows.extend(_run_full_stage(cpu_run_tag=cpu_run_tag, gpu_run_tag=gpu_run_tag))

    summary_df = pd.DataFrame(rows)
    summary_df.to_parquet(data_dir / "pd_rapids_benchmark_summary.parquet", index=False)

    grouped: list[dict[str, Any]] = []
    for stage, stage_df in summary_df.groupby("stage"):
        stage_cpu = stage_df.loc[stage_df["backend"] == "cpu"].iloc[0].to_dict()
        stage_gpu = stage_df.loc[stage_df["backend"] == "gpu"].iloc[0].to_dict()
        cpu_seconds = float(stage_cpu.get("total_seconds") or stage_cpu.get("fit_seconds") or 0.0)
        gpu_seconds = float(stage_gpu.get("total_seconds") or stage_gpu.get("fit_seconds") or 0.0)
        grouped.append(
            {
                "stage": stage,
                "cpu_seconds": cpu_seconds,
                "gpu_seconds": gpu_seconds,
                "speedup_gpu_vs_cpu": float(cpu_seconds / max(gpu_seconds, 1e-9)),
                "cpu_auc": stage_cpu.get("auc"),
                "gpu_auc": stage_gpu.get("auc"),
                "cpu_brier": stage_cpu.get("brier"),
                "gpu_brier": stage_gpu.get("brier"),
                "cpu_ece": stage_cpu.get("ece"),
                "gpu_ece": stage_gpu.get("ece"),
            }
        )

    payload = {
        "schema_version": "2026-03-10.1",
        "run_tag": run_tag,
        "cpu_run_tag": cpu_run_tag,
        "gpu_run_tag": gpu_run_tag,
        "fit_sample_size": fit_sample_size,
        "fit_val_size": fit_val_size,
        "hpo_sample_size": hpo_sample_size,
        "hpo_val_size": hpo_val_size,
        "hpo_trials": hpo_trials,
        "stages": grouped,
        "rows": rows,
    }
    (root / "run_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved PD RAPIDS benchmark summary to {}", root / "run_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="2026-03-10-pd-rapids-benchmark")
    parser.add_argument("--fit-sample-size", type=int, default=250_000)
    parser.add_argument("--fit-val-size", type=int, default=50_000)
    parser.add_argument("--hpo-sample-size", type=int, default=120_000)
    parser.add_argument("--hpo-val-size", type=int, default=30_000)
    parser.add_argument("--hpo-trials", type=int, default=6)
    parser.add_argument("--cpu-run-tag", default="2026-03-09-C-official-champion64safe")
    parser.add_argument("--gpu-run-tag", default="2026-03-09-official-gpu-replay-rapids-final")
    parser.add_argument("--worker-mode", default="")
    parser.add_argument("--worker-output", default="")
    parser.add_argument("--worker-backend", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    if args.worker_mode == "hpo":
        train_df = pd.read_parquet(ROOT / "data" / "processed" / "train_fe.parquet")
        test_df = pd.read_parquet(ROOT / "data" / "processed" / "test_fe.parquet")
        row = _run_hpo_backend(
            backend=args.worker_backend,
            train_df=train_df,
            test_df=test_df,
            fit_sample_size=args.hpo_sample_size,
            val_sample_size=args.hpo_val_size,
            n_trials=args.hpo_trials,
        )
        Path(args.worker_output).write_text(json.dumps(row, indent=2), encoding="utf-8")
        raise SystemExit(0)
    main(
        run_tag=args.run_tag,
        fit_sample_size=args.fit_sample_size,
        fit_val_size=args.fit_val_size,
        hpo_sample_size=args.hpo_sample_size,
        hpo_val_size=args.hpo_val_size,
        hpo_trials=args.hpo_trials,
        cpu_run_tag=args.cpu_run_tag,
        gpu_run_tag=args.gpu_run_tag,
    )
