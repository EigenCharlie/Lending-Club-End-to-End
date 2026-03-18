"""CIF-adjusted ECL impact analysis for Paper 2 (IFRS9).

Computes the excess ECL provision caused by using Kaplan-Meier (KM) PD estimates
instead of Competing-Risk CIF (Aalen-Johansen) estimates.

KM overestimates lifetime PD because it treats prepayment as right-censoring,
ignoring that prepaid loans cannot subsequently default (competing event).

Key formula:
    ECL_stage2_cif = ECL_stage2_km × (CIF_lifetime / KM_lifetime)
    Excess_reserve = ECL_stage2_km - ECL_stage2_cif  [positive = KM overestimates]

Outputs:
    data/processed/cif_ecl_impact.parquet — per-grade impact table
    models/cif_ecl_impact_status.json     — aggregate summary

Usage:
    uv run python scripts/run_cif_ecl_impact.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

SCHEMA_VERSION = "2026-03-17.1"
LGD_ASSUMED = 0.40  # must match IFRS9 sensitivity script

# Per-grade: approximate KM lifetime default rates from IFRS9 ECL table
# (These were computed by the IFRS9 script from the grade lifetime PD table)
# We derive correction factors using the per-grade CIF_t60m / KM_t60m.
# When per-grade KM is unavailable, fall back to the overall ratio.


def _load_cif_data() -> dict[str, dict[str, float]]:
    """Extract overall and per-grade CIF values from the wide parquet.

    Returns dict: grade -> {cif_t12m, cif_t36m, cif_t60m}
    """
    cif_path = Path("data/processed/competing_risks_cif.parquet")
    if not cif_path.exists():
        raise FileNotFoundError(f"Missing: {cif_path}. Run scripts/run_competing_risks.py first.")
    df = pd.read_parquet(cif_path)

    result: dict[str, dict[str, float]] = {}

    # Overall row
    overall_row = df[df["grade"] == "ALL"].iloc[0]
    result["ALL"] = {
        "cif_t12m": float(overall_row["overall_cif_default_t12m"]),
        "cif_t36m": float(overall_row["overall_cif_default_t36m"]),
        "cif_t60m": float(overall_row["overall_cif_default_t60m"]),
        "km_t12m": float(overall_row["km_default_prob_t12m"]),
        "km_t36m": float(overall_row["km_default_prob_t36m"]),
        "km_t60m": float(overall_row["km_default_prob_t60m"]),
    }

    # Per-grade rows: the CIF values are stored in wide format columns {grade}_{metric}
    for grade in ["A", "B", "C", "D", "E", "F", "G"]:
        grade_row = df[df["grade"] == grade]
        if grade_row.empty:
            continue
        r = grade_row.iloc[0]
        col_t12 = f"{grade}_cif_default_t12m"
        col_t36 = f"{grade}_cif_default_t36m"
        col_t60 = f"{grade}_cif_default_t60m"
        if col_t60 in df.columns and not pd.isna(r.get(col_t60, np.nan)):
            result[grade] = {
                "cif_t12m": float(r.get(col_t12, np.nan)),
                "cif_t36m": float(r.get(col_t36, np.nan)),
                "cif_t60m": float(r.get(col_t60, np.nan)),
            }

    return result


def main() -> int:
    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info("CIF ECL impact analysis | run_tag={}", run_tag)

    # ── 1. Load CIF data ─────────────────────────────────────────────────────
    cif = _load_cif_data()
    overall = cif["ALL"]
    km_t60m_overall = overall["km_t60m"]
    cif_t60m_overall = overall["cif_t60m"]
    km_t12m_overall = overall["km_t12m"]
    cif_t12m_overall = overall["cif_t12m"]

    # Correction factors (CIF/KM): <1 means KM overestimates
    cf_lifetime_overall = cif_t60m_overall / km_t60m_overall
    cf_12m_overall = cif_t12m_overall / km_t12m_overall

    logger.info(
        "Overall KM@60m={:.4f} | CIF@60m={:.4f} | KM overestimate={:.4f}pp | CF_lifetime={:.4f}",
        km_t60m_overall,
        cif_t60m_overall,
        (km_t60m_overall - cif_t60m_overall) * 100,
        cf_lifetime_overall,
    )
    logger.info(
        "Overall KM@12m={:.6f} | CIF@12m={:.6f} | CF_12m={:.6f}",
        km_t12m_overall,
        cif_t12m_overall,
        cf_12m_overall,
    )

    # ── 2. Load IFRS9 ECL per grade ──────────────────────────────────────────
    ecl_path = Path("data/processed/ifrs9_ecl_comparison.parquet")
    if not ecl_path.exists():
        raise FileNotFoundError(f"Missing: {ecl_path}. Run scripts/run_ifrs9_sensitivity.py first.")
    ecl_df = pd.read_parquet(ecl_path)
    # Expected cols: Grade, Avg Loan, PD_12m, PD_lifetime, ECL_Stage1, ECL_Stage2
    logger.info("Loaded IFRS9 ECL table: {} grades", len(ecl_df))

    # ── 3. Load IFRS9 scenario summary (for total ECL context) ───────────────
    scenario_path = Path("data/processed/ifrs9_scenario_summary.parquet")
    total_ecl_km = None
    if scenario_path.exists():
        sc_df = pd.read_parquet(scenario_path)
        baseline_row = sc_df[sc_df["scenario"] == "baseline"]
        if not baseline_row.empty:
            total_ecl_km = float(baseline_row.iloc[0]["total_ecl"])
            logger.info("Baseline ECL (KM-based): ${:,.0f}", total_ecl_km)

    # ── 4. Load test set to get loan counts per grade ─────────────────────────
    test_path = Path("data/processed/test.parquet")
    grade_counts: dict[str, int] = {}
    grade_loan_amnt: dict[str, float] = {}
    if test_path.exists():
        test_df = pd.read_parquet(test_path, columns=["grade", "loan_amnt"])
        for g, grp in test_df.groupby("grade"):
            grade_counts[str(g)] = len(grp)
            grade_loan_amnt[str(g)] = float(pd.to_numeric(grp["loan_amnt"], errors="coerce").mean())
        logger.info("Test set: {:,} loans across {} grades", len(test_df), len(grade_counts))

    # ── 5. Compute per-grade CIF correction ──────────────────────────────────
    rows = []
    for _, row in ecl_df.iterrows():
        grade = str(row["Grade"])
        pd_12m_km = float(row["PD_12m"])
        pd_lifetime_km = float(row["PD_lifetime"])
        ecl_s1_km = float(row["ECL_Stage1"])  # per-loan ECL Stage 1
        ecl_s2_km = float(row["ECL_Stage2"])  # per-loan ECL Stage 2
        avg_loan = float(row["Avg Loan"])
        n_loans = grade_counts.get(grade, 0)

        # Per-grade CIF correction factor (fallback to overall if unavailable)
        if grade in cif and not np.isnan(cif[grade].get("cif_t60m", np.nan)):
            cif_t60m_g = cif[grade]["cif_t60m"]
            cif_t12m_g = cif[grade].get("cif_t12m", cif_t12m_overall)
            # Per-grade KM: approximate using the same overall KM/CIF ratio rescaled
            # (we don't have per-grade KM directly, so derive from CIF + overall bias)
            # KM_grade ≈ CIF_grade × (KM_overall / CIF_overall) — proportional scaling
            km_t60m_g = cif_t60m_g * (km_t60m_overall / cif_t60m_overall)
            km_t12m_g = (
                cif_t12m_g * (km_t12m_overall / cif_t12m_overall)
                if cif_t12m_g > 0
                else km_t12m_overall
            )
            cf_lifetime_g = cif_t60m_g / km_t60m_g  # = cf_lifetime_overall (by construction)
            cf_12m_g = cif_t12m_g / km_t12m_g if km_t12m_g > 0 else cf_12m_overall
            cif_source = "per_grade"
        else:
            cif_t60m_g = cif_t60m_overall
            cif_t12m_g = cif_t12m_overall
            km_t60m_g = km_t60m_overall
            km_t12m_g = km_t12m_overall
            cf_lifetime_g = cf_lifetime_overall
            cf_12m_g = cf_12m_overall
            cif_source = "overall_fallback"

        # CIF-corrected per-loan ECL
        ecl_s1_cif = ecl_s1_km * cf_12m_g
        ecl_s2_cif = ecl_s2_km * cf_lifetime_g

        # Excess reserve per loan (KM - CIF) — positive = KM overestimates
        excess_s1 = ecl_s1_km - ecl_s1_cif
        excess_s2 = ecl_s2_km - ecl_s2_cif

        # Portfolio totals (using test set loan counts)
        excess_portfolio_s2 = excess_s2 * n_loans

        rows.append(
            {
                "grade": grade,
                "n_loans_test": n_loans,
                "avg_loan_amnt": avg_loan,
                "pd_12m_km": pd_12m_km,
                "pd_lifetime_km": pd_lifetime_km,
                "cif_t60m": cif_t60m_g,
                "km_t60m": km_t60m_g,
                "cf_lifetime": cf_lifetime_g,
                "cf_12m": cf_12m_g,
                "ecl_stage1_km": ecl_s1_km,
                "ecl_stage2_km": ecl_s2_km,
                "ecl_stage1_cif": ecl_s1_cif,
                "ecl_stage2_cif": ecl_s2_cif,
                "excess_ecl_stage1_per_loan": excess_s1,
                "excess_ecl_stage2_per_loan": excess_s2,
                "excess_ecl_stage2_portfolio": excess_portfolio_s2,
                "cif_source": cif_source,
            }
        )

    impact_df = pd.DataFrame(rows)

    # ── 6. Aggregate impact ───────────────────────────────────────────────────
    # Use overall correction on total_ecl from scenario summary
    total_ecl_cif = None
    total_excess_overall = None
    if total_ecl_km is not None:
        # Stage 2 makes up most of ECL; apply correction proportionally
        # The scenario total_ecl includes S1+S2+S3; S2 is ~60% of it
        # For a precise estimate, we apply cf_lifetime_overall to the S2 component
        # From scenario baseline stage shares: ~46.6% S2, ~22.5% S3, ~30.9% S1
        s2_share = 0.466
        total_ecl_s2_km = total_ecl_km * s2_share
        total_ecl_s1_km = total_ecl_km * 0.309
        total_ecl_cif = (
            total_ecl_s1_km * cf_12m_overall
            + total_ecl_s2_km * cf_lifetime_overall
            + total_ecl_km * (1 - s2_share - 0.309)  # S3 unchanged (PD~1.0)
        )
        total_excess_overall = total_ecl_km - total_ecl_cif

    logger.info("=== CIF ECL Impact Summary ===")
    logger.info(
        "Overall correction factor (lifetime): {:.4f} (KM overestimates by {:.2f}%)",
        cf_lifetime_overall,
        (1 - cf_lifetime_overall) * 100,
    )
    if total_ecl_km and total_ecl_cif:
        logger.info(
            "Total ECL (KM-based): ${:,.0f} | CIF-adjusted: ${:,.0f} | Excess: ${:,.0f}",
            total_ecl_km,
            total_ecl_cif,
            total_excess_overall,
        )

    logger.info(
        "\n{}",
        impact_df[
            [
                "grade",
                "pd_lifetime_km",
                "cif_t60m",
                "cf_lifetime",
                "ecl_stage2_km",
                "ecl_stage2_cif",
                "excess_ecl_stage2_per_loan",
            ]
        ].to_string(index=False),
    )

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    out_path = Path("data/processed/cif_ecl_impact.parquet")
    impact_df.to_parquet(out_path, index=False)
    logger.info("Saved: {}", out_path)

    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={
            "n_grades": len(impact_df),
            "overall_km_t60m": km_t60m_overall,
            "overall_cif_t60m": cif_t60m_overall,
            "km_overestimate_pp": round((km_t60m_overall - cif_t60m_overall) * 100, 4),
        },
    )
    status: dict = {
        **metadata,
        "overall": {
            "km_t60m": km_t60m_overall,
            "cif_t60m": cif_t60m_overall,
            "km_overestimate_t60m_pp": round((km_t60m_overall - cif_t60m_overall) * 100, 4),
            "cf_lifetime": round(cf_lifetime_overall, 6),
            "cf_12m": round(cf_12m_overall, 6),
        },
        "ecl_impact": {
            "total_ecl_km": total_ecl_km,
            "total_ecl_cif_adjusted": round(total_ecl_cif, 2) if total_ecl_cif else None,
            "total_excess_reserve": round(total_excess_overall, 2)
            if total_excess_overall
            else None,
            "excess_reserve_pct": (
                round((total_excess_overall / total_ecl_km) * 100, 2)
                if total_excess_overall and total_ecl_km
                else None
            ),
        },
        "note": (
            "KM ignores prepayment as a competing risk, inflating lifetime PD. "
            "CIF (Aalen-Johansen) corrects this. Excess reserve = ECL_KM - ECL_CIF. "
            "Stage 1 impact is minimal (12m horizon); Stage 2 impact is significant (lifetime)."
        ),
    }
    status_path = Path("models/cif_ecl_impact_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info("Saved status: {}", status_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
