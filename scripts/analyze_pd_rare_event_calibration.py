"""Analyze PD calibration under class imbalance and subgroup slices."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss

_CLI_RUN_TAG: str | None = None


def _resolve_run_tag() -> str:
    """Resolve run_tag: CLI arg > pipeline_summary > fallback."""
    if _CLI_RUN_TAG:
        return _CLI_RUN_TAG
    pipeline_path = Path("data/processed/pipeline_summary.json")
    if pipeline_path.exists():
        try:
            data = json.loads(pipeline_path.read_text(encoding="utf-8"))
            tag = data.get("run_tag")
            if tag:
                return str(tag)
        except Exception:
            pass
    return "untracked"


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(y_true_arr), 1)
    ece = 0.0
    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_arr >= lower) & (y_prob_arr <= upper)
        else:
            mask = (y_prob_arr >= lower) & (y_prob_arr < upper)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true_arr[mask]))
        conf = float(np.mean(y_prob_arr[mask]))
        weight = float(np.sum(mask)) / total
        ece += abs(acc - conf) * weight
    return float(ece)


def _score_deciles_report(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["y_true", "y_prob_final"]].copy()
    work["score_decile"] = pd.qcut(
        work["y_prob_final"].rank(method="first"),
        q=min(10, len(work)),
        labels=False,
        duplicates="drop",
    )
    report = (
        work.groupby("score_decile", observed=True)
        .agg(
            n=("y_true", "size"),
            prevalence=("y_true", "mean"),
            mean_score=("y_prob_final", "mean"),
            brier=(
                "y_true",
                lambda s: float(np.mean((s - work.loc[s.index, "y_prob_final"]) ** 2)),
            ),
        )
        .reset_index()
        .sort_values("score_decile")
    )
    report["ece_component"] = (report["prevalence"] - report["mean_score"]).abs() * (
        report["n"] / max(report["n"].sum(), 1)
    )
    report["report_type"] = "score_decile"
    return report


def _slice_report(df: pd.DataFrame, column: str, report_type: str) -> pd.DataFrame:
    work = df.loc[df[column].notna(), [column, "y_true", "y_prob_final"]].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "report_type",
                "slice_name",
                "slice_value",
                "n",
                "prevalence",
                "mean_score",
                "brier",
                "ece",
                "pr_auc",
            ]
        )
    out = (
        work.groupby(column, observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n": int(len(g)),
                    "prevalence": float(g["y_true"].mean()),
                    "mean_score": float(g["y_prob_final"].mean()),
                    "brier": float(brier_score_loss(g["y_true"], g["y_prob_final"])),
                    "ece": float(_ece(g["y_true"], g["y_prob_final"], n_bins=min(10, len(g)))),
                    "pr_auc": float(average_precision_score(g["y_true"], g["y_prob_final"]))
                    if g["y_true"].nunique() > 1
                    else float("nan"),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={column: "slice_value"})
    )
    out.insert(0, "slice_name", column)
    out.insert(0, "report_type", report_type)
    return out


def _prevalence_bin_report(group_reports: list[pd.DataFrame]) -> pd.DataFrame:
    if not group_reports:
        return pd.DataFrame(
            columns=["report_type", "prevalence_bin", "n_groups", "avg_brier", "avg_ece"]
        )
    merged = pd.concat(group_reports, ignore_index=True)
    merged = merged.loc[merged["n"] > 0].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=["report_type", "prevalence_bin", "n_groups", "avg_brier", "avg_ece"]
        )
    bins = [-np.inf, 0.02, 0.05, 0.10, np.inf]
    labels = ["<=2%", "2-5%", "5-10%", ">10%"]
    merged["prevalence_bin"] = pd.cut(merged["prevalence"], bins=bins, labels=labels)
    out = (
        merged.groupby("prevalence_bin", observed=True)
        .agg(
            n_groups=("slice_value", "size"),
            avg_brier=("brier", "mean"),
            avg_ece=("ece", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "report_type", "prevalence_bin")
    return out


def _load_fairness_columns(path: str = "configs/fairness_policy.yaml") -> list[str]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return []
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cols: list[str] = []
    for item in payload.get("attributes", []) or []:
        col = str((item or {}).get("column", "")).strip()
        if col:
            cols.append(col)
    return cols


def main() -> None:
    preds_path = Path("data/processed/test_predictions.parquet")
    meta_path = Path("data/processed/test_fe.parquet")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions artifact: {preds_path}")
    if not meta_path.exists():
        meta_path = Path("data/processed/test.parquet")
    if not meta_path.exists():
        raise FileNotFoundError("No test metadata artifact found for rare-event calibration audit.")

    preds = pd.read_parquet(preds_path)
    meta = pd.read_parquet(meta_path)
    n = min(len(preds), len(meta))
    preds = preds.iloc[:n].reset_index(drop=True)
    meta = meta.iloc[:n].reset_index(drop=True)

    score_col = "y_prob_final" if "y_prob_final" in preds.columns else "pd_calibrated"
    if score_col not in preds.columns:
        raise KeyError("Expected `y_prob_final` or `pd_calibrated` in test_predictions.parquet.")

    df = pd.DataFrame(
        {
            "y_true": pd.to_numeric(preds["y_true"], errors="coerce").fillna(0.0).astype(int),
            "y_prob_final": pd.to_numeric(preds[score_col], errors="coerce").fillna(0.0),
        }
    )
    for col in ("grade", "home_ownership", "term", "issue_d"):
        if col in meta.columns:
            df[col] = meta[col]
    if "issue_d" in df.columns:
        df["issue_quarter"] = (
            pd.to_datetime(df["issue_d"], errors="coerce").dt.to_period("Q").astype(str)
        )

    fairness_cols = [col for col in _load_fairness_columns() if col in meta.columns]
    reports: list[pd.DataFrame] = [_score_deciles_report(df)]
    group_reports: list[pd.DataFrame] = []

    seen_cols: set[str] = set()
    for col in ["grade", "home_ownership", "term", "issue_quarter", *fairness_cols]:
        if col in seen_cols:
            continue
        seen_cols.add(col)
        if col not in df.columns:
            continue
        report_type = "protected_group" if col in fairness_cols else col
        report = _slice_report(df, col, report_type)
        reports.append(report)
        if report_type in {"grade", "protected_group", "term"}:
            group_reports.append(report)

    reports.append(_prevalence_bin_report(group_reports))
    report_df = pd.concat(reports, ignore_index=True, sort=False)
    if "slice_value" in report_df.columns:
        report_df["slice_value"] = report_df["slice_value"].astype("string")
    if "prevalence_bin" in report_df.columns:
        report_df["prevalence_bin"] = report_df["prevalence_bin"].astype("string")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pd_rare_event_calibration_report.parquet"
    report_df.to_parquet(report_path, index=False)

    protected = report_df.loc[report_df["report_type"] == "protected_group"].copy()
    grades = report_df.loc[report_df["report_type"] == "grade"].copy()
    deciles = report_df.loc[report_df["report_type"] == "score_decile"].copy()
    worst_protected_group = None
    if not protected.empty:
        row = protected.sort_values(["ece", "brier", "n"], ascending=[False, False, False]).iloc[0]
        worst_protected_group = {
            "slice_name": str(row["slice_name"]),
            "slice_value": str(row["slice_value"]),
            "ece": float(row["ece"]),
            "brier": float(row["brier"]),
            "pr_auc": float(row["pr_auc"]) if pd.notna(row["pr_auc"]) else None,
            "n": int(row["n"]),
        }
    worst_grade_group = None
    if not grades.empty:
        row = grades.sort_values(["brier", "ece", "n"], ascending=[False, False, False]).iloc[0]
        worst_grade_group = {
            "slice_value": str(row["slice_value"]),
            "ece": float(row["ece"]),
            "brier": float(row["brier"]),
            "pr_auc": float(row["pr_auc"]) if pd.notna(row["pr_auc"]) else None,
            "n": int(row["n"]),
        }
    status: dict[str, Any] = {
        "schema_version": "2026-03-13.1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": _resolve_run_tag(),
        "summary": {
            "n_obs": int(len(df)),
            "prevalence": float(df["y_true"].mean()),
            "pr_auc": float(average_precision_score(df["y_true"], df["y_prob_final"])),
            "brier": float(brier_score_loss(df["y_true"], df["y_prob_final"])),
            "ece": float(_ece(df["y_true"], df["y_prob_final"], n_bins=10)),
        },
        "global": {
            "n_obs": int(len(df)),
            "prevalence": float(df["y_true"].mean()),
            "pr_auc": float(average_precision_score(df["y_true"], df["y_prob_final"])),
            "brier": float(brier_score_loss(df["y_true"], df["y_prob_final"])),
            "ece": float(_ece(df["y_true"], df["y_prob_final"], n_bins=10)),
        },
        "worst_protected_group_ece": float(protected["ece"].max()) if not protected.empty else None,
        "worst_grade_brier": float(grades["brier"].max()) if not grades.empty else None,
        "max_decile_gap": float((deciles["prevalence"] - deciles["mean_score"]).abs().max())
        if not deciles.empty
        else None,
        "worst_protected_group": worst_protected_group,
        "worst_grade_group": worst_grade_group,
        "report_path": str(report_path),
        "notes": [
            "Rare-event calibration audit is diagnostic; it should inform PD selection but not replace the core champion metric stack.",
            "Coverage-by-group should be interpreted jointly with fairness/governance, without collapsing validity into fairness claims.",
        ],
    }
    status_path = Path("models/pd_rare_event_calibration_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Saved PD rare-event calibration report: {}", report_path)
    logger.info("Saved PD rare-event calibration status: {}", status_path)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--run-tag", default=None, help="Run tag to stamp on status artifact")
    _args = _parser.parse_args()
    if _args.run_tag:
        _CLI_RUN_TAG = _args.run_tag
    main()
