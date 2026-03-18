"""Competing risks analysis: default vs prepayment CIF.

Estimates Cumulative Incidence Functions (CIF) using the Aalen-Johansen
estimator treating default (Charged Off) and prepayment (Fully Paid) as
competing risks, with active/late loans as right-censored.

Key insight for credit risk: standard Kaplan-Meier overestimates default
probability because it ignores the competing prepayment risk. CIF provides
cause-specific incidence accounting for both exit events.

Usage:
    uv run python scripts/run_competing_risks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.survival import compute_cif
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

SCHEMA_VERSION = "2026-03-16.1"

DEFAULT_STATUSES = ("Charged Off", "Default")
PREPAY_STATUSES = ("Fully Paid",)


def _load_data() -> pd.DataFrame:
    """Load and prepare data for competing risks analysis."""
    # Use the full cleaned dataset for richer survival data
    interim_path = Path("data/interim/lending_club_cleaned.parquet")
    if not interim_path.exists():
        raise FileNotFoundError(f"Interim data not found: {interim_path}")

    df = pd.read_parquet(
        interim_path,
        columns=[
            "loan_status",
            "lgd_months_since_issue",
            "grade",
            "term",
            "loan_amnt",
            "int_rate",
            "default_flag",
        ],
    )
    logger.info("Loaded {:,} rows from interim data", len(df))

    # Map loan_status to event encoding
    # Keep only resolved loans + a sample of active for censoring
    resolved_mask = df["loan_status"].isin(DEFAULT_STATUSES + PREPAY_STATUSES)
    active_mask = ~resolved_mask

    n_resolved = int(resolved_mask.sum())
    n_active = int(active_mask.sum())
    logger.info("Resolved: {:,} | Active (censored): {:,}", n_resolved, n_active)

    # For active loans, censor at their current elapsed time
    # lgd_months_since_issue is already computed in make_dataset
    df_resolved = df[resolved_mask].copy()
    df_active = df[active_mask].copy()

    # Combine: resolved loans get their actual event time
    # active loans are right-censored
    df_all = pd.concat([df_resolved, df_active], ignore_index=True)

    # Drop rows with missing time
    df_all = df_all.dropna(subset=["lgd_months_since_issue"])
    df_all = df_all[df_all["lgd_months_since_issue"] > 0]
    logger.info("After cleaning: {:,} rows", len(df_all))

    return df_all


def _km_default_rate(df: pd.DataFrame, time_col: str, status_col: str) -> dict[str, float]:
    """Naive KM-based default rate (ignores competing risk)."""
    from lifelines import KaplanMeierFitter

    event = df[status_col].isin(DEFAULT_STATUSES).astype(int)
    time = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event, label="default_km")

    # Extract 12m, 36m, 60m estimates
    times_of_interest = [12.0, 24.0, 36.0, 60.0]
    result = {}
    for t in times_of_interest:
        km_surv = kmf.survival_function_at_times([t]).values[0]
        result[f"km_default_prob_t{int(t)}m"] = float(1.0 - km_surv)
    return result


def _summarize_cif(cif_result: dict, label: str) -> dict[str, float]:
    """Extract key statistics from a CIF result dict."""
    overall = cif_result.get("overall", {})
    if not overall or not overall.get("cif_default"):
        return {}

    times = np.array(overall["times"])
    cif_d = np.array(overall["cif_default"])
    cif_p = np.array(overall["cif_prepay"])

    summary: dict[str, float] = {
        f"{label}_n": int(overall.get("n", 0)),
        f"{label}_n_default": int(overall.get("n_default", 0)),
        f"{label}_n_prepay": int(overall.get("n_prepay", 0)),
    }

    for t_m in [12, 24, 36, 60]:
        idx = np.searchsorted(times, t_m, side="right") - 1
        if 0 <= idx < len(times):
            summary[f"{label}_cif_default_t{t_m}m"] = float(cif_d[idx])
            summary[f"{label}_cif_prepay_t{t_m}m"] = float(cif_p[idx])
        else:
            summary[f"{label}_cif_default_t{t_m}m"] = float("nan")
            summary[f"{label}_cif_prepay_t{t_m}m"] = float("nan")

    return summary


def main() -> int:
    """Run competing risks CIF analysis."""
    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info("Competing risks analysis starting | run_tag={}", run_tag)

    df = _load_data()

    # ── Overall CIF ───────────────────────────────────────────────────────────
    logger.info("=== Computing overall Aalen-Johansen CIF ===")
    cif_overall = compute_cif(
        df,
        time_col="lgd_months_since_issue",
        status_col="loan_status",
        group_col="grade",
        conf_level=0.95,
    )

    # ── KM comparison (baseline, ignores competing) ───────────────────────────
    logger.info("=== Computing KM baseline (ignores competing risk) ===")
    km_stats = _km_default_rate(df, "lgd_months_since_issue", "loan_status")
    logger.info("KM default prob: {}", km_stats)

    # ── Build summary table for parquet ───────────────────────────────────────
    records: list[dict] = []

    # Overall row
    overall_row: dict[str, object] = {"grade": "ALL"}
    overall_row.update(_summarize_cif(cif_overall, "overall"))
    overall_row.update(km_stats)
    records.append(overall_row)

    # Per-grade rows
    by_group = cif_overall.get("by_group", {})
    for grade, grp_result in by_group.items():
        row: dict[str, object] = {"grade": grade}
        fake_cif = {"overall": grp_result, "by_group": {}}
        row.update(_summarize_cif(fake_cif, grade))
        records.append(row)

    summary_df = pd.DataFrame(records)
    summary_path = Path("data/processed/competing_risks_cif.parquet")
    summary_df.to_parquet(summary_path, index=False)
    logger.info("Saved CIF summary: {} ({} rows)", summary_path, len(summary_df))

    # ── Save full CIF time series (overall only for size) ─────────────────────
    if cif_overall.get("overall"):
        ts_df = pd.DataFrame(
            {
                "time_months": cif_overall["overall"]["times"],
                "cif_default": cif_overall["overall"]["cif_default"],
                "cif_prepay": cif_overall["overall"]["cif_prepay"],
            }
        )
        if cif_overall["overall"].get("cif_default_lo"):
            ts_df["cif_default_lo"] = cif_overall["overall"]["cif_default_lo"]
            ts_df["cif_default_hi"] = cif_overall["overall"]["cif_default_hi"]
        ts_path = Path("data/processed/competing_risks_cif_timeseries.parquet")
        ts_df.to_parquet(ts_path, index=False)
        logger.info("Saved CIF time series: {} ({} time points)", ts_path, len(ts_df))

    # ── Compute KM bias (over-estimation due to competing risk) ───────────────
    km_bias_note: dict[str, float] = {}
    if cif_overall.get("overall"):
        overall_data = cif_overall["overall"]
        cif_d_arr = np.array(overall_data["cif_default"])
        times_arr = np.array(overall_data["times"])
        for t_m in [12, 24, 36, 60]:
            idx = np.searchsorted(times_arr, t_m, side="right") - 1
            if 0 <= idx < len(times_arr):
                cif_val = float(cif_d_arr[idx])
                km_val = km_stats.get(f"km_default_prob_t{t_m}m", float("nan"))
                km_bias_note[f"km_overestimate_t{t_m}m"] = float(km_val - cif_val)

    logger.info("KM over-estimation of default (CIF correction): {}", km_bias_note)

    # ── Status JSON ───────────────────────────────────────────────────────────
    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={
            "n_total": int(len(df)),
            "n_default": int(df["loan_status"].isin(DEFAULT_STATUSES).sum()),
            "n_prepay": int(df["loan_status"].isin(PREPAY_STATUSES).sum()),
        },
    )

    cif_12m = float("nan")
    cif_36m = float("nan")
    if cif_overall.get("overall"):
        overall_data = cif_overall["overall"]
        times_arr = np.array(overall_data["times"])
        cif_d_arr = np.array(overall_data["cif_default"])
        idx12 = np.searchsorted(times_arr, 12, side="right") - 1
        idx36 = np.searchsorted(times_arr, 36, side="right") - 1
        if 0 <= idx12 < len(cif_d_arr):
            cif_12m = float(cif_d_arr[idx12])
        if 0 <= idx36 < len(cif_d_arr):
            cif_36m = float(cif_d_arr[idx36])

    status: dict[str, object] = {
        **metadata,
        "cif_summary_path": str(summary_path),
        "n_grades_computed": len(by_group),
        "km_default_rates": km_stats,
        "km_bias_vs_cif": km_bias_note,
        "overall": {
            "cif_default_12m": cif_12m,
            "cif_default_36m": cif_36m,
            "cif_default_final": float(cif_overall["overall"]["cif_default"][-1])
            if cif_overall.get("overall") and cif_overall["overall"].get("cif_default")
            else float("nan"),
            "cif_prepay_final": float(cif_overall["overall"]["cif_prepay"][-1])
            if cif_overall.get("overall") and cif_overall["overall"].get("cif_prepay")
            else float("nan"),
        },
        "note": (
            "KM overestimates PD at all horizons because it ignores prepayment as "
            "a competing exit. CIF (Aalen-Johansen) corrects this by accounting for "
            "the sub-distribution hazard of each competing event."
        ),
    }

    status_path = Path("models/competing_risks_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info("Saved status: {}", status_path)

    # Log summary
    logger.info("=== Competing Risks Summary ===")
    logger.info("CIF default @12m: {:.4f}", cif_12m)
    logger.info("CIF default @36m: {:.4f}", cif_36m)
    for k, v in km_bias_note.items():
        logger.info("  {}: {:.4f} (KM over-estimates by this much)", k, v)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
