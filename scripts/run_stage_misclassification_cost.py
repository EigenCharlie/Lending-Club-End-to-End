"""Stage 1 vs Stage 2 Misclassification Cost Analysis (Paper 2 — IFRS9/JBF).

Quantifies the asymmetric costs of SICR misclassification errors:

  False Negative (FN): Loan that should be Stage 2 (SICR) is classified as Stage 1.
    → Under-provisioning: missed provision = (PD_lifetime - PD_12m) × LGD × EAD
    → Regulatory consequence: under-stated ECL, capital shortfall risk.

  False Positive (FP): Loan that should stay in Stage 1 is moved to Stage 2.
    → Over-provisioning: excess provision = (PD_lifetime - PD_12m) × LGD × EAD
    → Economic consequence: unnecessary capital tie-up, profitability impact.

The asymmetric cost function:
    Total_cost(t) = w_fn × Σ FN_cost_i + w_fp × Σ FP_cost_i

Under IFRS9 prudential regulation, FN errors are penalized more severely (w_fn >> w_fp).
Default weights: w_fn=5.0, w_fp=1.0 (reflecting typical regulatory asymmetry).

Sweeps:
  - Part A: Pure PD threshold SICR (width trigger disabled) as baseline.
  - Part B: Combined PD + width trigger — finds cost-optimal (pd_thr, width_thr) pair.
  - Part C: Cost reduction from adding width trigger vs pure PD, by grade.

Key contribution for Paper 2:
  Shows that the conformal width trigger is not only better by F1 metrics but also
  reduces total expected IFRS9 misclassification cost by ~X% vs pure PD threshold.

Outputs:
  data/processed/stage_misclassification_cost.parquet  — full (pd × width) cost grid
  data/processed/stage_cost_by_grade.parquet           — per-grade cost breakdown
  models/stage_misclassification_status.json           — optimal thresholds + cost reduction

Usage:
    uv run python scripts/run_stage_misclassification_cost.py
    uv run python scripts/run_stage_misclassification_cost.py --w-fn 5.0 --w-fp 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata

SCHEMA_VERSION = "2026-03-17.1"
ROOT = Path(__file__).resolve().parents[1]

LGD_DEFAULT = 0.40
PD_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
WIDTH_THRESHOLDS = np.round(np.arange(0.05, 1.01, 0.05), 2).tolist()
GRADE_LIFETIME_PD_FALLBACK = {
    "A": 0.178,
    "B": 0.266,
    "C": 0.352,
    "D": 0.456,
    "E": 0.634,
    "F": 0.757,
    "G": 0.821,
    "ALL": 0.380,
}


# ── 1. Data loading ──────────────────────────────────────────────────────────


def _load_data() -> tuple[pd.DataFrame, dict[str, float]]:
    """Load conformal intervals and per-grade lifetime PD."""
    ci_path = ROOT / "data/processed/conformal_intervals_mondrian.parquet"
    if not ci_path.exists():
        raise FileNotFoundError(f"Missing: {ci_path}")

    cols = ["y_true", "y_pred", "width_90", "grade", "loan_amnt"]
    df = pd.read_parquet(ci_path, columns=cols)
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["width_90"] = pd.to_numeric(df["width_90"], errors="coerce")
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(10_000.0)
    df["grade"] = df["grade"].astype(str).fillna("C")

    # Per-grade lifetime PD
    ecl_path = ROOT / "data/processed/ifrs9_ecl_comparison.parquet"
    if ecl_path.exists():
        ecl_df = pd.read_parquet(ecl_path)
        grade_pd = {str(r["Grade"]): float(r["PD_lifetime"]) for _, r in ecl_df.iterrows()}
        grade_pd.setdefault("ALL", float(np.mean(list(grade_pd.values()))))
        logger.info("Loaded lifetime PD for {} grades from ECL table", len(grade_pd))
    else:
        grade_pd = GRADE_LIFETIME_PD_FALLBACK
        logger.warning("ECL table not found — using fallback lifetime PD estimates")

    logger.info("Conformal data: {:,} loans, default_rate={:.2%}", len(df), df["y_true"].mean())
    return df, grade_pd


# ── 2. Cost computation ──────────────────────────────────────────────────────


def _provision_increment(
    df: pd.DataFrame,
    mask: pd.Series,
    grade_pd: dict[str, float],
    lgd: float,
) -> pd.Series:
    """Compute (PD_lifetime - PD_12m) * LGD * loan_amnt for each loan in mask.

    This is the Stage 1 → Stage 2 provisioning increment per loan.
    Always non-negative since PD_lifetime >= PD_12m.
    """
    sub = df[mask].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    pd_lifetime = sub["grade"].map(grade_pd).fillna(grade_pd.get("ALL", 0.35))
    increment = (pd_lifetime - sub["y_pred"]).clip(lower=0) * lgd * sub["loan_amnt"]
    return increment


def _compute_cost_row(
    df: pd.DataFrame,
    pd_thr: float,
    w_thr: float | None,
    grade_pd: dict[str, float],
    lgd: float,
    w_fn: float,
    w_fp: float,
) -> dict:
    """Compute FN/FP costs for a single (pd_thr, w_thr) threshold combination.

    Stage 2 trigger:
      - Pure PD: y_pred > pd_thr
      - Combined: (y_pred > pd_thr) OR (width_90 > w_thr)
    Ground truth (retrospective):
      - True Stage 2 need: y_true = 1 (loan defaulted during observation)
      - True Stage 1 ok:   y_true = 0

    FN: y_true=1 AND predicted Stage 1 → under-provisioned
    FP: y_true=0 AND predicted Stage 2 → over-provisioned
    """
    n_total = len(df)

    if w_thr is None:
        stage2_pred = df["y_pred"] > pd_thr
    else:
        stage2_pred = (df["y_pred"] > pd_thr) | (df["width_90"] > w_thr)

    # FN: defaulted but not staged → missed provision
    fn_mask = (df["y_true"] == 1) & ~stage2_pred
    # FP: non-default but staged → excess provision
    fp_mask = (df["y_true"] == 0) & stage2_pred

    fn_cost = float(_provision_increment(df, fn_mask, grade_pd, lgd).sum())
    fp_cost = float(_provision_increment(df, fp_mask, grade_pd, lgd).sum())
    total_cost = w_fn * fn_cost + w_fp * fp_cost

    return {
        "pd_threshold": pd_thr,
        "width_threshold": w_thr if w_thr is not None else np.nan,
        "use_width_trigger": w_thr is not None,
        "n_stage2_pred": int(stage2_pred.sum()),
        "pct_stage2": int(stage2_pred.sum()) / n_total,
        "n_fn": int(fn_mask.sum()),
        "n_fp": int(fp_mask.sum()),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_cost": total_cost,
        "fn_rate": float(fn_mask.sum()) / max(int((df["y_true"] == 1).sum()), 1),
        "fp_rate": float(fp_mask.sum()) / max(int((df["y_true"] == 0).sum()), 1),
    }


# ── 3. Grade-level analysis ──────────────────────────────────────────────────


def _cost_by_grade(
    df: pd.DataFrame,
    pd_thr: float,
    w_thr: float | None,
    grade_pd: dict[str, float],
    lgd: float,
    w_fn: float,
    w_fp: float,
) -> pd.DataFrame:
    """Compute misclassification costs per grade for optimal threshold pair."""
    rows = []
    for grade in sorted(df["grade"].unique()):
        sub = df[df["grade"] == grade]
        row = _compute_cost_row(sub, pd_thr, w_thr, grade_pd, lgd, w_fn, w_fp)
        row["grade"] = grade
        row["n_loans"] = len(sub)
        row["default_rate"] = float(sub["y_true"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("grade").reset_index(drop=True)


# ── 4. Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="IFRS9 Stage 1/2 misclassification cost analysis.")
    parser.add_argument("--run-tag", default="untracked")
    parser.add_argument("--lgd", type=float, default=LGD_DEFAULT)
    parser.add_argument("--w-fn", type=float, default=5.0, help="FN cost weight")
    parser.add_argument("--w-fp", type=float, default=1.0, help="FP cost weight")
    args = parser.parse_args()

    run_tag = str(args.run_tag).strip()
    logger.info(
        "Stage misclassification cost | lgd={} w_fn={} w_fp={} run_tag={}",
        args.lgd,
        args.w_fn,
        args.w_fp,
        run_tag,
    )

    df, grade_pd = _load_data()
    lgd = args.lgd
    w_fn = args.w_fn
    w_fp = args.w_fp

    # Part A: Pure PD threshold baseline
    logger.info("Part A: pure PD threshold cost sweep...")
    pure_pd_rows = []
    for pd_thr in PD_THRESHOLDS:
        row = _compute_cost_row(df, pd_thr, None, grade_pd, lgd, w_fn, w_fp)
        pure_pd_rows.append(row)
    pure_pd_df = pd.DataFrame(pure_pd_rows)
    best_pure_pd = pure_pd_df.loc[pure_pd_df["total_cost"].idxmin()]
    logger.info(
        "Best pure PD threshold: pd_thr={} | total_cost=${:.1f}M | FN=${:.1f}M | FP=${:.1f}M",
        best_pure_pd["pd_threshold"],
        best_pure_pd["total_cost"] / 1e6,
        best_pure_pd["fn_cost"] / 1e6,
        best_pure_pd["fp_cost"] / 1e6,
    )

    # Part B: Combined PD + width trigger
    logger.info(
        "Part B: combined PD + width cost grid ({} × {})...",
        len(PD_THRESHOLDS),
        len(WIDTH_THRESHOLDS),
    )
    combined_rows = []
    for pd_thr in PD_THRESHOLDS:
        for w_thr in WIDTH_THRESHOLDS:
            row = _compute_cost_row(df, pd_thr, w_thr, grade_pd, lgd, w_fn, w_fp)
            combined_rows.append(row)
    combined_df = pd.DataFrame(combined_rows)

    best_combined = combined_df.loc[combined_df["total_cost"].idxmin()]
    logger.info(
        "Best combined: pd_thr={} | w_thr={} | total_cost=${:.1f}M | FN=${:.1f}M | FP=${:.1f}M",
        best_combined["pd_threshold"],
        best_combined["width_threshold"],
        best_combined["total_cost"] / 1e6,
        best_combined["fn_cost"] / 1e6,
        best_combined["fp_cost"] / 1e6,
    )

    cost_reduction_pct = float(
        (best_pure_pd["total_cost"] - best_combined["total_cost"])
        / max(best_pure_pd["total_cost"], 1)
        * 100
    )
    logger.info(
        "Cost reduction from width trigger: {:.1f}%",
        cost_reduction_pct,
    )

    # Combine all rows for output
    all_rows = pd.concat([pure_pd_df, combined_df], ignore_index=True)
    out_path = ROOT / "data/processed/stage_misclassification_cost.parquet"
    all_rows.to_parquet(out_path, index=False)
    logger.info("Saved: {} ({} rows)", out_path.relative_to(ROOT), len(all_rows))

    # Part C: per-grade breakdown at optimal combined threshold
    logger.info("Part C: per-grade cost at optimal combined threshold...")
    grade_df = _cost_by_grade(
        df,
        float(best_combined["pd_threshold"]),
        float(best_combined["width_threshold"]),
        grade_pd,
        lgd,
        w_fn,
        w_fp,
    )
    grade_path = ROOT / "data/processed/stage_cost_by_grade.parquet"
    grade_df.to_parquet(grade_path, index=False)
    logger.info("Saved: {} ({} rows)", grade_path.relative_to(ROOT), len(grade_df))

    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={"script": "run_stage_misclassification_cost.py"},
    )
    status = {
        **metadata,
        "lgd": lgd,
        "w_fn": w_fn,
        "w_fp": w_fp,
        "n_loans": int(len(df)),
        "default_rate": round(float(df["y_true"].mean()), 4),
        "part_a_pure_pd": {
            "best_pd_threshold": float(best_pure_pd["pd_threshold"]),
            "total_cost_M": round(float(best_pure_pd["total_cost"]) / 1e6, 2),
            "fn_cost_M": round(float(best_pure_pd["fn_cost"]) / 1e6, 2),
            "fp_cost_M": round(float(best_pure_pd["fp_cost"]) / 1e6, 2),
            "fn_rate": round(float(best_pure_pd["fn_rate"]), 4),
            "fp_rate": round(float(best_pure_pd["fp_rate"]), 4),
        },
        "part_b_combined": {
            "best_pd_threshold": float(best_combined["pd_threshold"]),
            "best_width_threshold": float(best_combined["width_threshold"]),
            "total_cost_M": round(float(best_combined["total_cost"]) / 1e6, 2),
            "fn_cost_M": round(float(best_combined["fn_cost"]) / 1e6, 2),
            "fp_cost_M": round(float(best_combined["fp_cost"]) / 1e6, 2),
            "fn_rate": round(float(best_combined["fn_rate"]), 4),
            "fp_rate": round(float(best_combined["fp_rate"]), 4),
            "cost_reduction_vs_pure_pd_pct": round(cost_reduction_pct, 2),
        },
        "key_finding": (
            f"Adding the conformal width trigger reduces total IFRS9 misclassification cost "
            f"by {cost_reduction_pct:.1f}% vs pure PD threshold. "
            f"The asymmetric cost function (w_fn={w_fn}, w_fp={w_fp}) reflects that "
            f"under-provisioning (FN) is regulatorily costlier than over-provisioning (FP). "
            f"The combined trigger reduces FN errors by capturing high-uncertainty loans "
            f"that the PD threshold alone misses."
        ),
    }
    status_path = ROOT / "models/stage_misclassification_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved status: {}", status_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
