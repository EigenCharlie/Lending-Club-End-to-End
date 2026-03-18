"""SICR Conformal Trigger + ECL Sensitivity Analysis for Paper 2 (IFRS9).

Part A — SICR Threshold Optimization:
  For each conformal width threshold t, measures whether (width_90 > t) adds
  predictive value as an additional SICR trigger beyond the traditional PD threshold.
  Outputs precision/recall/F1 of the width trigger across a grid of (pd_threshold, t).

Part B — ECL Sensitivity to Alpha Conformal:
  Fixes the optimal SICR width threshold and sweeps the conformal alpha level.
  Uses the Mondrian alpha-sweep results to scale per-loan widths proportionally.
  Quantifies how total ECL changes as a function of the confidence level choice.

Key insight for Paper 2:
  The width of a conformal interval encodes epistemic uncertainty.
  A wide interval signals the model is uncertain about that loan — a
  legitimate additional SICR signal beyond point-estimate PD.
  The ECL sensitivity to alpha quantifies the *regulatory cost* of choosing
  a more conservative (lower alpha) confidence level.

Outputs:
  data/processed/sicr_conformal_grid.parquet      — Part A: full (pd_threshold x width_threshold) grid
  data/processed/ecl_alpha_sensitivity.parquet    — Part B: alpha -> ECL
  models/sicr_conformal_status.json               — summary + optimal thresholds

Usage:
    uv run python scripts/run_sicr_conformal.py
    uv run python scripts/run_sicr_conformal.py --pd-threshold 0.15 --lgd 0.40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

SCHEMA_VERSION = "2026-03-17.1"
LGD_DEFAULT = 0.40

# Width threshold grid for Part A
WIDTH_THRESHOLDS = np.round(np.arange(0.05, 1.01, 0.05), 2).tolist()

# PD thresholds to benchmark as traditional SICR triggers
PD_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


# ── 1. Data loading ──────────────────────────────────────────────────────────


def _load_conformal_data() -> pd.DataFrame:
    """Load Mondrian conformal intervals (test set, aligned row-by-row)."""
    path = Path("data/processed/conformal_intervals_mondrian.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}. Run scripts/generate_conformal_intervals.py first."
        )
    cols = ["y_true", "y_pred", "width_90", "grade", "loan_amnt"]
    df = pd.read_parquet(path, columns=cols)
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["width_90"] = pd.to_numeric(df["width_90"], errors="coerce")
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(10_000.0)
    df["grade"] = df["grade"].astype(str).fillna("C")
    logger.info("Conformal data: {:,} loans, default_rate={:.2%}", len(df), df["y_true"].mean())
    return df


def _load_grade_lifetime_pd() -> dict[str, float]:
    """Load per-grade lifetime PD from IFRS9 ECL table."""
    path = Path("data/processed/ifrs9_ecl_comparison.parquet")
    if not path.exists():
        logger.warning("IFRS9 ECL parquet not found — using fallback lifetime PD estimates")
        return {
            "A": 0.178,
            "B": 0.266,
            "C": 0.352,
            "D": 0.456,
            "E": 0.634,
            "F": 0.757,
            "G": 0.821,
            "ALL": 0.380,
        }
    ecl_df = pd.read_parquet(path)
    result = {}
    for _, row in ecl_df.iterrows():
        grade = str(row["Grade"])
        result[grade] = float(row["PD_lifetime"])
    result.setdefault("ALL", float(np.mean(list(result.values()))))
    logger.info("Loaded lifetime PD for {} grades", len(result))
    return result


def _load_alpha_sweep() -> pd.DataFrame:
    """Load Mondrian alpha sweep pareto for Part B scaling."""
    path = Path("data/processed/alpha_sweep_pareto_both.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run scripts/run_alpha_sweep.py --both first.")
    df = pd.read_parquet(path)
    mondrian = df[df["method"] == "mondrian"].copy()
    if mondrian.empty:
        raise ValueError("No Mondrian rows in alpha_sweep_pareto_both.parquet")
    return mondrian.sort_values("alpha").reset_index(drop=True)


# ── 2. ECL helpers ───────────────────────────────────────────────────────────


def _ecl_additional_vectorized(
    df: pd.DataFrame,
    mask: pd.Series,
    grade_lifetime_pd: dict[str, float],
    lgd: float,
) -> float:
    """Compute incremental ECL from reclassifying `mask` loans Stage 1→Stage 2.

    Additional ECL per loan = (PD_lifetime - PD_12m) * LGD * loan_amnt
    (the difference between lifetime and 12-month provision).
    """
    sub = df[mask]
    if sub.empty:
        return 0.0
    pd_lifetime = sub["grade"].map(grade_lifetime_pd).fillna(grade_lifetime_pd.get("ALL", 0.35))
    extra = (pd_lifetime - sub["y_pred"]) * lgd * sub["loan_amnt"]
    # Clip at 0: only count positive increments (lifetime > 12m, which is almost always true)
    return float(extra.clip(lower=0).sum())


# ── 3. Part A — SICR threshold optimization ──────────────────────────────────


def _run_sicr_grid(
    df: pd.DataFrame,
    grade_lifetime_pd: dict[str, float],
    lgd: float,
) -> pd.DataFrame:
    """Sweep (pd_threshold × width_threshold) grid and compute SICR trigger metrics."""
    n_total = len(df)
    n_defaults = int(df["y_true"].sum())
    rows = []

    for pd_thr in PD_THRESHOLDS:
        pd_triggered = df["y_pred"] > pd_thr
        n_pd_triggered = int(pd_triggered.sum())
        n_pd_defaults_caught = int((pd_triggered & (df["y_true"] == 1)).sum())
        n_pd_recall = n_pd_defaults_caught / max(n_defaults, 1)

        for w_thr in WIDTH_THRESHOLDS:
            width_triggered = df["width_90"] > w_thr
            # Only count ADDITIONAL loans not already PD-triggered
            additional = width_triggered & ~pd_triggered
            n_add = int(additional.sum())

            if n_add == 0:
                precision = np.nan
                recall_add = 0.0
                f1 = np.nan
                ecl_add = 0.0
            else:
                # Precision: of newly staged loans, what fraction defaulted?
                precision = float((additional & (df["y_true"] == 1)).sum()) / n_add

                # Recall (additional): of defaults NOT caught by PD trigger, what fraction
                # does the width trigger catch?
                missed_defaults_mask = (df["y_true"] == 1) & ~pd_triggered
                n_missed = int(missed_defaults_mask.sum())
                if n_missed > 0:
                    recall_add = float((additional & (df["y_true"] == 1)).sum()) / n_missed
                else:
                    recall_add = 1.0

                denom = precision + recall_add
                f1 = 2 * precision * recall_add / denom if denom > 0 else 0.0

                ecl_add = _ecl_additional_vectorized(df, additional, grade_lifetime_pd, lgd)

            rows.append(
                {
                    "pd_threshold": pd_thr,
                    "width_threshold": w_thr,
                    "n_total": n_total,
                    "n_pd_triggered": n_pd_triggered,
                    "n_pd_defaults_caught": n_pd_defaults_caught,
                    "n_pd_recall": n_pd_recall,
                    "n_additional_sicr": n_add,
                    "pct_additional": n_add / n_total,
                    "precision_width": precision,
                    "recall_width_of_missed": recall_add,
                    "f1_width": f1,
                    "ecl_additional": ecl_add,
                }
            )

    return pd.DataFrame(rows)


# ── 4. Part B — ECL sensitivity to alpha ─────────────────────────────────────


def _run_alpha_sensitivity(
    df: pd.DataFrame,
    mondrian_sweep: pd.DataFrame,
    pd_threshold: float,
    optimal_width_threshold: float,
    grade_lifetime_pd: dict[str, float],
    lgd: float,
) -> pd.DataFrame:
    """Sweep alpha levels, scale per-loan widths, compute ECL change."""
    # Reference avg_width at alpha=0.10 for scaling
    ref_row = mondrian_sweep[np.isclose(mondrian_sweep["alpha"], 0.10, atol=0.005)]
    if ref_row.empty:
        logger.warning("No Mondrian row at alpha≈0.10; using first available as reference")
        ref_row = mondrian_sweep.iloc[[0]]
    ref_avg_width = float(ref_row.iloc[0]["avg_width"])
    logger.info("Reference avg_width at alpha=0.10: {:.4f}", ref_avg_width)

    pd_triggered = df["y_pred"] > pd_threshold
    n_total = len(df)
    rows = []

    for _, sweep_row in mondrian_sweep.iterrows():
        alpha = float(sweep_row["alpha"])
        avg_width = float(sweep_row["avg_width"])
        scale = (
            avg_width / ref_avg_width
        )  # >1 for lower alpha (wider intervals), <1 for higher alpha

        # Scale per-loan widths proportionally (approximation: same scaling ratio for all loans)
        scaled_width = df["width_90"] * scale
        scaled_width = scaled_width.clip(upper=1.0)  # PD ∈ [0,1], width can't exceed 1

        width_triggered_scaled = scaled_width > optimal_width_threshold
        additional = width_triggered_scaled & ~pd_triggered

        n_add = int(additional.sum())
        ecl_add = _ecl_additional_vectorized(df, additional, grade_lifetime_pd, lgd)

        missed_defaults_mask = (df["y_true"] == 1) & ~pd_triggered
        n_missed = int(missed_defaults_mask.sum())
        recall_add = float((additional & (df["y_true"] == 1)).sum()) / max(n_missed, 1)
        precision_add = (
            float((additional & (df["y_true"] == 1)).sum()) / max(n_add, 1) if n_add > 0 else np.nan
        )

        rows.append(
            {
                "alpha": alpha,
                "confidence_level": 1.0 - alpha,
                "avg_width_mondrian": avg_width,
                "scale_factor": scale,
                "optimal_width_threshold_scaled": optimal_width_threshold,
                "n_additional_sicr": n_add,
                "pct_loans_reclassified": n_add / n_total,
                "ecl_additional": ecl_add,
                "precision_width": precision_add,
                "recall_width_of_missed": recall_add,
                "empirical_coverage": float(sweep_row.get("empirical_coverage", np.nan)),
                "min_group_coverage": float(sweep_row.get("min_group_coverage", np.nan)),
            }
        )

    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


# ── 5. Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="SICR conformal trigger + ECL alpha sensitivity")
    parser.add_argument(
        "--pd-threshold",
        type=float,
        default=0.20,
        help="PD_12m threshold for traditional SICR (default 0.20)",
    )
    parser.add_argument(
        "--lgd",
        type=float,
        default=LGD_DEFAULT,
        help="Loss Given Default assumption (default 0.40)",
    )
    args = parser.parse_args()

    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info(
        "SICR conformal analysis | pd_threshold={} lgd={} run_tag={}",
        args.pd_threshold,
        args.lgd,
        run_tag,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    conf_df = _load_conformal_data()
    grade_lifetime_pd = _load_grade_lifetime_pd()
    mondrian_sweep = _load_alpha_sweep()

    # ── Part A: SICR threshold grid ───────────────────────────────────────────
    logger.info(
        "Part A: SICR threshold grid ({} pd × {} width levels)...",
        len(PD_THRESHOLDS),
        len(WIDTH_THRESHOLDS),
    )
    grid_df = _run_sicr_grid(conf_df, grade_lifetime_pd, args.lgd)

    # Find optimal width threshold for the primary PD threshold (F1-maximizing)
    primary_grid = grid_df[np.isclose(grid_df["pd_threshold"], args.pd_threshold)]
    valid = primary_grid.dropna(subset=["f1_width"])
    if not valid.empty:
        best_row = valid.loc[valid["f1_width"].idxmax()]
        optimal_t = float(best_row["width_threshold"])
        optimal_f1 = float(best_row["f1_width"])
        optimal_precision = float(best_row["precision_width"])
        optimal_recall = float(best_row["recall_width_of_missed"])
        optimal_ecl_add = float(best_row["ecl_additional"])
    else:
        optimal_t = 0.50
        optimal_f1 = optimal_precision = optimal_recall = optimal_ecl_add = float("nan")

    logger.info(
        "Optimal width threshold (pd_thr={:.2f}): t*={:.2f} | F1={:.4f} | "
        "precision={:.4f} | recall_of_missed={:.4f} | ECL_add=${:,.0f}",
        args.pd_threshold,
        optimal_t,
        optimal_f1,
        optimal_precision,
        optimal_recall,
        optimal_ecl_add,
    )

    grid_path = Path("data/processed/sicr_conformal_grid.parquet")
    grid_df.to_parquet(grid_path, index=False)
    logger.info("Saved: {} ({} rows)", grid_path, len(grid_df))

    # Print summary table for primary pd_threshold
    summary_cols = [
        "width_threshold",
        "n_additional_sicr",
        "pct_additional",
        "precision_width",
        "recall_width_of_missed",
        "f1_width",
        "ecl_additional",
    ]
    logger.info(
        "\nPart A summary (pd_thr={:.2f}):\n{}",
        args.pd_threshold,
        primary_grid[summary_cols].round(4).to_string(index=False),
    )

    # ── Part B: ECL sensitivity to alpha ──────────────────────────────────────
    logger.info("Part B: ECL alpha sensitivity (optimal t*={:.2f})...", optimal_t)
    alpha_df = _run_alpha_sensitivity(
        conf_df,
        mondrian_sweep,
        args.pd_threshold,
        optimal_t,
        grade_lifetime_pd,
        args.lgd,
    )

    alpha_path = Path("data/processed/ecl_alpha_sensitivity.parquet")
    alpha_df.to_parquet(alpha_path, index=False)
    logger.info("Saved: {} ({} rows)", alpha_path, len(alpha_df))

    logger.info(
        "\nPart B — ECL sensitivity to alpha (t*={:.2f}, pd_thr={:.2f}):",
        optimal_t,
        args.pd_threshold,
    )
    disp_cols = [
        "alpha",
        "avg_width_mondrian",
        "scale_factor",
        "n_additional_sicr",
        "pct_loans_reclassified",
        "ecl_additional",
        "empirical_coverage",
    ]
    logger.info("\n{}", alpha_df[disp_cols].round(4).to_string(index=False))

    # ── Save status ───────────────────────────────────────────────────────────
    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={
            "n_loans": len(conf_df),
            "default_rate": round(float(conf_df["y_true"].mean()), 4),
            "pd_threshold": args.pd_threshold,
            "lgd": args.lgd,
            "n_pd_triggered": int((conf_df["y_pred"] > args.pd_threshold).sum()),
        },
    )

    status = {
        **metadata,
        "part_a": {
            "optimal_width_threshold": optimal_t,
            "optimal_f1": round(optimal_f1, 4) if np.isfinite(optimal_f1) else None,
            "optimal_precision": round(optimal_precision, 4)
            if np.isfinite(optimal_precision)
            else None,
            "optimal_recall_of_missed": round(optimal_recall, 4)
            if np.isfinite(optimal_recall)
            else None,
            "ecl_additional_at_optimal": round(optimal_ecl_add, 2)
            if np.isfinite(optimal_ecl_add)
            else None,
            "n_pd_thresholds_tested": len(PD_THRESHOLDS),
            "n_width_thresholds_tested": len(WIDTH_THRESHOLDS),
        },
        "part_b": {
            "alpha_levels_tested": sorted(mondrian_sweep["alpha"].tolist()),
            "ecl_at_alpha_010": round(
                float(
                    alpha_df[np.isclose(alpha_df["alpha"], 0.10, atol=0.005)].iloc[0][
                        "ecl_additional"
                    ]
                ),
                2,
            )
            if not alpha_df[np.isclose(alpha_df["alpha"], 0.10, atol=0.005)].empty
            else None,
            "ecl_at_alpha_005": round(
                float(
                    alpha_df[np.isclose(alpha_df["alpha"], 0.05, atol=0.005)].iloc[0][
                        "ecl_additional"
                    ]
                ),
                2,
            )
            if not alpha_df[np.isclose(alpha_df["alpha"], 0.05, atol=0.005)].empty
            else None,
            "ecl_at_alpha_020": round(
                float(
                    alpha_df[np.isclose(alpha_df["alpha"], 0.20, atol=0.005)].iloc[0][
                        "ecl_additional"
                    ]
                ),
                2,
            )
            if not alpha_df[np.isclose(alpha_df["alpha"], 0.20, atol=0.005)].empty
            else None,
        },
        "note": (
            "SICR trigger: a loan is reclassified from Stage 1 to Stage 2 if "
            "pd_calibrated > pd_threshold OR conformal_width_90 > t. "
            "Part A optimizes t via F1 of the width trigger for loans missed by PD threshold. "
            "Part B shows how additional ECL changes as a function of conformal alpha "
            "(smaller alpha = wider intervals = more loans reclassified = higher ECL reserve)."
        ),
    }

    status_path = Path("models/sicr_conformal_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info("Saved status: {}", status_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
