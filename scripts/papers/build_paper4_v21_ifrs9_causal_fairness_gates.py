"""Build Paper 4 v21 IFRS9, causal/CATE and fairness governance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
from scripts.papers.build_paper4_v6_priority_resolution import (
    SOURCE_FAMILIES,
    TABLE_DIR,
    _load_inputs,
    _prepare_solver_pool,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.21"
DATA_DIR = Path("data")


def _stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _available_columns(path: Path, *, nrows: int = 500) -> set[str]:
    try:
        if path.suffix == ".parquet":
            return set(pd.read_parquet(path).columns)
        if path.suffix == ".csv":
            return set(pd.read_csv(path, nrows=nrows).columns)
    except Exception:
        return set()
    return set()


def _ifrs9_data_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = [
        Path("data/processed/loan_master.parquet"),
        Path("data/processed/ead_dataset.parquet"),
        Path("data/processed/ifrs9_ecl_comparison.parquet"),
        Path("data/processed/ifrs9_scenario_summary.parquet"),
        Path("data/processed/competing_risks_cif.parquet"),
        Path("data/raw/Loan_status_2007-2020Q3.csv"),
        TABLE_DIR / "paper4_v5_ifrs9_servicing_panel.parquet",
        TABLE_DIR / "paper4_v17_ifrs9_proxy_monthly_panel.parquet",
    ]
    requirements = {
        "servicing_panel_monthly": ["month", "loan_id", "out_prncp", "total_pymnt"],
        "monthly_days_past_due": ["days_past_due", "dpd", "mths_since_last_delinq"],
        "forbearance_hardship": ["hardship_flag", "hardship_status", "deferral_term"],
        "default_timing": ["loan_status", "default_flag", "last_pymnt_d"],
        "cure_timing": ["cure", "recoveries", "collection_recovery_fee"],
        "prepayment_timing": ["last_pymnt_d", "total_rec_prncp", "out_prncp"],
        "monthly_ead_path": ["ead", "ead_start", "outstanding", "out_prncp"],
        "coherent_macro_scenarios": ["scenario", "macro_regime", "unemployment_rate"],
    }
    audit_rows = []
    for path in files:
        cols = _available_columns(path)
        for req, aliases in requirements.items():
            hits = sorted(
                {col for col in cols for alias in aliases if alias.lower() in col.lower()}
            )
            audit_rows.append(
                {
                    "artifact": str(path),
                    "exists": path.exists(),
                    "requirement": req,
                    "matching_columns": ";".join(hits),
                    "available_for_contractual_ifrs9": bool(req in {"default_timing"} and hits),
                    "available_for_proxy_ecl": bool(hits),
                    "decision_v21": "usable_proxy_component" if hits else "data_blocked",
                }
            )
    audit = pd.DataFrame(audit_rows)
    blocker = (
        audit.groupby("requirement", as_index=False)
        .agg(
            any_matching_column=("available_for_proxy_ecl", "max"),
            contractual_ready=("available_for_contractual_ifrs9", "max"),
            evidence_artifacts=("artifact", lambda s: "; ".join(sorted(set(s))[:4])),
        )
        .assign(
            contractual_claim_allowed=False,
            blocker_status=lambda d: np.where(
                d["contractual_ready"].astype(bool), "partial_proxy_only", "data_blocked"
            ),
            claim_boundary="IFRS9-inspired ECL proxy only; no contractual IFRS9 claim",
        )
    )
    return audit, blocker


def _ifrs9_proxy_panel_and_sicr(books: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = (
        books.groupby(["policy_id", "issue_month", "original_grade"], as_index=False)
        .agg(
            loans=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            default_rate_proxy=("y_true", "mean"),
            avg_pd_high=("pd_high_alpha01", "mean"),
            avg_pd_point=("pd_point_alpha01", "mean"),
            avg_lgd_proxy=("lgd", "mean"),
            avg_width=("qhat_v4", "mean"),
            avg_weak_source=("weak_source_proxy", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    scenarios = pd.DataFrame(
        [
            ("baseline", 1.00, 1.00),
            ("adverse", 1.25, 1.10),
            ("severe", 1.55, 1.22),
        ],
        columns=["scenario", "pd_multiplier", "lgd_multiplier"],
    )
    panel = panel.merge(scenarios, how="cross")
    panel["ecl_proxy_v21"] = (
        panel["funded_exposure"]
        * np.clip(panel["avg_pd_high"] * panel["pd_multiplier"], 0, 1)
        * np.clip(
            panel["avg_lgd_proxy"].fillna(v15.DEFAULT_LGD) * panel["lgd_multiplier"], 0.10, 0.95
        )
    )
    panel["sicr_pd_absolute"] = panel["avg_pd_high"].ge(0.20)
    panel["sicr_pd_relative_proxy"] = panel["avg_pd_high"].ge(
        1.5 * panel["avg_pd_point"].clip(lower=0.02)
    )
    panel["sicr_width"] = panel["avg_width"].ge(0.80)
    panel["sicr_weak_source"] = panel["avg_weak_source"].ge(0.66)
    panel["stage2_composite_v21"] = (
        panel["sicr_pd_absolute"]
        | panel["sicr_pd_relative_proxy"]
        | panel["sicr_width"]
        | panel["sicr_weak_source"]
    ).astype(int)
    panel["claim_scope_v21"] = "IFRS9-inspired proxy panel, not contractual lifetime ECL"

    sicr = (
        panel.groupby(["policy_id", "scenario"], as_index=False)
        .agg(
            ecl_proxy_total_v21=("ecl_proxy_v21", "sum"),
            stage2_share_absolute_pd=("sicr_pd_absolute", "mean"),
            stage2_share_relative_pd=("sicr_pd_relative_proxy", "mean"),
            stage2_share_width=("sicr_width", "mean"),
            stage2_share_composite=("stage2_composite_v21", "mean"),
            funded_exposure=("funded_exposure", "sum"),
        )
        .assign(
            contractual_ifrs9_claim_allowed=False,
            sensitivity_interpretation="SICR proxy sensitivity; stage shares should not be read as IFRS9 production staging",
        )
    )
    return panel, sicr


def _standardized_mean_difference(df: pd.DataFrame, treatment: pd.Series, col: str) -> float:
    x = pd.to_numeric(df[col], errors="coerce")
    treated = x[treatment].dropna()
    control = x[~treatment].dropna()
    if treated.empty or control.empty:
        return float("nan")
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled)


def _causal_dossier(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = base.copy()
    df["issue_month"] = (
        pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    df["int_rate_decimal"] = pd.to_numeric(df["int_rate_decimal"], errors="coerce")
    grade_median = df.groupby("original_grade")["int_rate_decimal"].transform("median")
    treatment = df["int_rate_decimal"].gt(grade_median)
    df["high_rate_within_grade_v21"] = treatment.astype(int)
    covariates = [
        col
        for col in [
            "pd_point_alpha01",
            "pd_high_alpha01",
            "qhat_v4",
            "loan_amnt",
            "annual_inc",
            "dti",
            "fico_score",
            "installment_burden",
        ]
        if col in df
    ]
    balance_rows = [
        {
            "treatment_id": "high_rate_within_grade_v21",
            "covariate": col,
            "smd_unweighted": _standardized_mean_difference(df, treatment, col),
            "balance_gate_pass": abs(_standardized_mean_difference(df, treatment, col)) <= 0.10
            if np.isfinite(_standardized_mean_difference(df, treatment, col))
            else False,
        }
        for col in covariates
    ]
    overlap = (
        df.assign(treatment=treatment.astype(int))
        .groupby(["original_grade", "score_decile"], as_index=False)
        .agg(
            n=("loan_id", "nunique"),
            treatment_rate=("treatment", "mean"),
            default_rate=("y_true", "mean"),
            avg_pd=("pd_high_alpha01", "mean"),
        )
        .assign(
            overlap_gate_pass=lambda d: d["n"].ge(30) & d["treatment_rate"].between(0.05, 0.95),
            accepted_loan_only_caveat="no reject-inference counterfactuals",
        )
    )
    default_diff = float(df.loc[treatment, "y_true"].mean() - df.loc[~treatment, "y_true"].mean())
    hidden_bias_pp_to_flip = abs(default_diff) * 100.0
    placebo_rows = []
    for placebo in ["hash_even_loan_id", "issue_month_even", "pseudo_random_outcome"]:
        if placebo == "hash_even_loan_id":
            outcome = (
                df["loan_id"].astype(str).map(lambda x: int(_stable_uniform("placebo", x) > 0.5))
            )
        elif placebo == "issue_month_even":
            outcome = df["issue_month"].dt.month.mod(2).fillna(0).astype(int)
        else:
            outcome = (
                df["loan_id"].astype(str).map(lambda x: int(_stable_uniform("pseudo", x) > 0.5))
            )
        placebo_rows.append(
            {
                "treatment_id": "high_rate_within_grade_v21",
                "placebo_outcome": placebo,
                "treated_mean": float(outcome[treatment].mean()),
                "control_mean": float(outcome[~treatment].mean()),
                "absolute_difference": float(
                    abs(outcome[treatment].mean() - outcome[~treatment].mean())
                ),
                "placebo_gate_pass": bool(
                    abs(outcome[treatment].mean() - outcome[~treatment].mean()) <= 0.03
                ),
            }
        )
    dossier = pd.DataFrame(
        [
            {
                "gate": "clean_outcome",
                "status_v21": "partial_proxy",
                "evidence": "default/loss exists for accepted loans; reject counterfactuals absent",
                "policy_value_allowed": False,
            },
            {
                "gate": "identification",
                "status_v21": "theory_blocked",
                "evidence": "accepted-loan selection and pricing endogeneity remain unresolved",
                "policy_value_allowed": False,
            },
            {
                "gate": "overlap",
                "status_v21": "review",
                "evidence": f"overlap cells passing support/rate gate: {int(overlap['overlap_gate_pass'].sum())}/{len(overlap)}",
                "policy_value_allowed": False,
            },
            {
                "gate": "balance",
                "status_v21": "review",
                "evidence": f"max |SMD| = {float(np.nanmax(np.abs([r['smd_unweighted'] for r in balance_rows]))):.3f}",
                "policy_value_allowed": False,
            },
            {
                "gate": "hidden_bias_sensitivity",
                "status_v21": "blocked",
                "evidence": f"hidden-bias proxy to flip naive default contrast = {hidden_bias_pp_to_flip:.2f}pp",
                "policy_value_allowed": False,
            },
            {
                "gate": "falsification/placebo",
                "status_v21": "review",
                "evidence": "placebo outcomes are reported but do not solve identification",
                "policy_value_allowed": False,
            },
        ]
    )
    gate = dossier.copy()
    gate["cate_policy_value_allowed"] = False
    gate["claim_boundary_v21"] = "causal diagnostics only; no CATE policy-value claim"
    return pd.DataFrame(balance_rows), overlap, pd.DataFrame(placebo_rows), gate


def _fairness_protocol_and_source_governance(
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    references = pd.DataFrame(
        [
            {
                "reference_id": "cfpb_proxy_methodology",
                "url": "https://github.com/cfpb/proxy-methodology",
                "use_v21": "methodological boundary reference only",
                "claim_boundary": "do not infer protected classes without data/protocol approval",
            },
            {
                "reference_id": "cfpb_fair_lending",
                "url": "https://www.consumerfinance.gov/compliance/compliance-resources/other-applicable-requirements/fair-lending/",
                "use_v21": "legal/compliance boundary reference",
                "claim_boundary": "Paper 4 makes no fair-lending legal claim",
            },
        ]
    )
    rows = []
    for family in [col for col in SOURCE_FAMILIES if col in books.columns]:
        local = (
            books.groupby(["policy_id", family], as_index=False)
            .agg(
                loans=("loan_id", "nunique"),
                exposure=("funded_exposure", "sum"),
                avg_pd=("pd_high_alpha01", "mean"),
                avg_width=("qhat_v4", "mean"),
                avg_weak_source=("weak_source_proxy", "mean"),
            )
            .rename(columns={family: "source_id"})
        )
        local["source_family"] = family
        rows.append(local)
    source = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not source.empty:
        source["support_gate_pass_v21"] = source["loans"].ge(25)
        source["coverage_claim_scope_v21"] = (
            "source governance/composition proxy, not protected-class fairness"
        )
    flags = pd.DataFrame(
        [
            (
                "protected_attributes",
                False,
                "data_blocked",
                "no protected attributes in Lending Club artifacts",
            ),
            (
                "external_proxy_protocol",
                False,
                "protocol_blocked",
                "BISG/proxy methods referenced but not implemented as legal claim",
            ),
            (
                "fair_lending_legal_claim",
                False,
                "prohibited_claim",
                "no legal claim without protected attributes or approved proxy protocol",
            ),
            (
                "source_governance",
                True,
                "usable_proxy",
                "grade/month/state/income/DTI/score groups remain usable diagnostics",
            ),
        ],
        columns=["claim_or_requirement", "allowed_v21", "status_v21", "evidence"],
    )
    return references, source, flags


def build_v21(solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    base, candidate, _, _, intervals = _load_inputs()
    source = base if len(base) > len(candidate) else candidate
    pool = _prepare_solver_pool(source, intervals, max_n=min(solver_pool_n, len(source)))
    books, _ = v15._load_policy_books(pool)
    if books.empty:
        raise RuntimeError("No policy books available for v21.")

    audit, blocker = _ifrs9_data_audit()
    proxy_panel, sicr = _ifrs9_proxy_panel_and_sicr(books)
    _write_csv("paper4_v21_ifrs9_data_field_audit.csv", audit)
    _write_csv("paper4_v21_ifrs9_readiness_matrix.csv", blocker)
    _write_parquet("paper4_v21_ifrs9_proxy_monthly_panel.parquet", proxy_panel)
    _write_csv("paper4_v21_ifrs9_sicr_sensitivity.csv", sicr)
    _write_csv("paper4_v21_ifrs9_data_blocker_register.csv", blocker)

    balance, overlap, placebo, cate_gate = _causal_dossier(pool)
    _write_csv("paper4_v21_causal_identification_dossier.csv", balance)
    _write_csv("paper4_v21_causal_overlap_diagnostics.csv", overlap)
    _write_csv("paper4_v21_causal_falsification_outcomes.csv", placebo)
    _write_csv("paper4_v21_cate_gate_report.csv", cate_gate)

    references, source_diag, no_claim = _fairness_protocol_and_source_governance(books)
    _write_csv("paper4_v21_fairness_reference_registry.csv", references)
    _write_csv("paper4_v21_source_governance_diagnostics.csv", source_diag)
    _write_csv("paper4_v21_no_legal_claim_flags.csv", no_claim)
    _write_csv("paper4_v21_fairness_proxy_only_protocol.csv", no_claim)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v21_ifrs9_causal_fairness_gates",
        "ifrs9_contractual_claim_allowed": False,
        "ifrs9_readiness_rows_v21": int(len(blocker)),
        "ifrs9_contractual_ready_requirements_v21": int(
            blocker["contractual_ready"].astype(bool).sum()
        ),
        "causal_policy_value_allowed": False,
        "cate_gate_rows_v21": int(len(cate_gate)),
        "fair_lending_legal_claim": False,
        "source_governance_rows_v21": int(len(source_diag)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "IFRS9 proxy-only, CATE diagnostics-only, fairness proxy-governance only",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v21_status.json", status)
    _write_note(
        "paper4_v21_ifrs9_causal_fairness_gates.md",
        "\n".join(
            [
                "# Paper 4 v21 IFRS9/Causal/Fairness Gates",
                "",
                f"- IFRS9 contractual claim allowed: `{status['ifrs9_contractual_claim_allowed']}`.",
                f"- CATE policy value allowed: `{status['causal_policy_value_allowed']}`.",
                f"- Fair-lending legal claim: `{status['fair_lending_legal_claim']}`.",
                f"- Source governance rows: `{status['source_governance_rows_v21']}`.",
                "",
                "All three lanes are intentionally claim-bounded.",
            ]
        ),
    )
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    status = build_v21(args.solver_pool_n)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
