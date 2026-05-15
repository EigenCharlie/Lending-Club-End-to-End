"""Build Paper 4 v25 IFRS9 proxy, causal/CATE and fairness/source upgrades."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
import scripts.papers.build_paper4_v21_ifrs9_causal_fairness_gates as v21
from scripts.papers.build_paper4_v6_priority_resolution import (
    SOURCE_FAMILIES,
    _load_inputs,
    _prepare_solver_pool,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.25"


def _stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _load_pool_books(max_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, candidate, _, _, intervals = _load_inputs()
    source = base if len(base) > len(candidate) else candidate
    pool = _prepare_solver_pool(source, intervals, max_n=min(max_n, len(source)))
    books, _ = v15._load_policy_books(pool)
    return pool, books


def _merge_book_raw(books: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [
        col
        for col in [
            "loan_id",
            "loan_status",
            "last_pymnt_d",
            "total_pymnt",
            "total_rec_prncp",
            "recoveries",
            "collection_recovery_fee",
            "out_prncp",
            "out_prncp_inv",
            "term",
        ]
        if col in pool.columns
    ]
    raw = pool[raw_cols].drop_duplicates("loan_id") if raw_cols else pd.DataFrame({"loan_id": []})
    out = books.merge(raw, on="loan_id", how="left", suffixes=("", "_raw"))
    for col in [
        "total_pymnt",
        "total_rec_prncp",
        "recoveries",
        "collection_recovery_fee",
        "out_prncp",
        "out_prncp_inv",
    ]:
        if col not in out:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["loan_status"] = out.get("loan_status", pd.Series("", index=out.index)).astype(str)
    return out


def _ifrs9_cashflow_proxy(
    books: pd.DataFrame, pool: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = _merge_book_raw(books, pool)
    rows = []
    max_horizon = 36
    for _, row in merged.iterrows():
        exposure = float(row.get("funded_exposure", row.get("loan_amnt", 0.0)))
        if exposure <= 0:
            continue
        term = int(row.get("term_months", 36) or 36)
        horizon = min(max_horizon, max(1, term))
        default_flag = bool(float(row.get("y_true", 0.0)) >= 0.5)
        fully_paid = "Fully Paid" in str(row.get("loan_status", ""))
        default_month = (
            1 + int(_stable_uniform("ifrs9-default-month", row["loan_id"]) * max(1, horizon))
            if default_flag
            else None
        )
        prepay_month = (
            1 + int(_stable_uniform("ifrs9-prepay-month", row["loan_id"]) * max(1, horizon))
            if fully_paid and not default_flag
            else None
        )
        recovery_lag = 3
        lgd = float(
            np.clip(
                max(float(row.get("lgd", v15.DEFAULT_LGD) or v15.DEFAULT_LGD), v15.DEFAULT_LGD),
                0.10,
                0.95,
            )
        )
        pd_high = float(np.clip(row.get("pd_high_alpha01", 0.18), 0.0, 1.0))
        for m in range(1, horizon + 1):
            ead_start = exposure * max(0.0, 1.0 - (m - 1) / term)
            ead_end = exposure * max(0.0, 1.0 - m / term)
            scheduled_principal = max(0.0, ead_start - ead_end)
            interest = ead_start * float(row.get("int_rate_decimal", 0.12)) / 12.0
            default_event = default_flag and m == default_month
            prepay_event = prepay_month is not None and m == prepay_month
            recovery_cash = 0.0
            loss_cash = 0.0
            if default_event:
                loss_cash = ead_start * lgd
                ead_end = 0.0
            if (
                default_flag
                and default_month is not None
                and m == min(horizon, default_month + recovery_lag)
            ):
                recovery_cash = ead_start * (1.0 - lgd)
            if prepay_event:
                scheduled_principal = ead_start
                ead_end = 0.0
            dpd_proxy = int(default_flag and default_month is not None and m >= default_month)
            sicr_abs = pd_high >= 0.20
            sicr_rel = pd_high >= 1.5 * max(float(row.get("pd_point_alpha01", pd_high)), 0.02)
            sicr_width = float(row.get("qhat_v4", 0.55)) >= 0.80
            sicr_dpd = dpd_proxy == 1
            stage = (
                3
                if default_event
                else (2 if (sicr_abs or sicr_rel or sicr_width or sicr_dpd) else 1)
            )
            rows.append(
                {
                    "policy_id": row["policy_id"],
                    "loan_id": row["loan_id"],
                    "month_index": m,
                    "scenario": "baseline",
                    "ead_start_proxy_v25": ead_start,
                    "ead_end_proxy_v25": ead_end,
                    "scheduled_principal_proxy": scheduled_principal,
                    "interest_cash_proxy": interest,
                    "default_event_proxy": default_event,
                    "prepayment_event_proxy": prepay_event,
                    "recovery_cash_proxy": recovery_cash,
                    "loss_cash_proxy": loss_cash,
                    "dpd_proxy": dpd_proxy,
                    "stage_proxy_v25": stage,
                    "sicr_absolute_pd": sicr_abs,
                    "sicr_relative_pd": sicr_rel,
                    "sicr_conformal_width": sicr_width,
                    "sicr_dpd_proxy": sicr_dpd,
                    "ecl_proxy_v25": ead_start * pd_high * lgd,
                    "claim_scope_v25": "IFRS9-inspired monthly proxy; not contractual IFRS9",
                }
            )
    panel = pd.DataFrame(rows)
    summary = (
        panel.groupby(["policy_id", "scenario"], as_index=False)
        .agg(
            ecl_proxy_total_v25=("ecl_proxy_v25", "sum"),
            loss_cash_proxy_total=("loss_cash_proxy", "sum"),
            recovery_cash_proxy_total=("recovery_cash_proxy", "sum"),
            stage2_share_proxy_v25=("stage_proxy_v25", lambda s: float((s == 2).mean())),
            stage3_share_proxy_v25=("stage_proxy_v25", lambda s: float((s == 3).mean())),
            ead_start_total=("ead_start_proxy_v25", "sum"),
        )
        .assign(contractual_ifrs9_claim_allowed=False)
    )
    sicr = (
        panel.groupby(["policy_id"], as_index=False)
        .agg(
            stage2_abs_pd=("sicr_absolute_pd", "mean"),
            stage2_rel_pd=("sicr_relative_pd", "mean"),
            stage2_width=("sicr_conformal_width", "mean"),
            stage2_dpd_proxy=("sicr_dpd_proxy", "mean"),
            stage2_composite=("stage_proxy_v25", lambda s: float((s == 2).mean())),
        )
        .assign(sicr_claim_boundary="proxy SICR sensitivity, not production IFRS9 staging")
    )
    return panel, summary, sicr


def _causal_upgrade(
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pool.copy()
    df["issue_month"] = (
        pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    df["int_rate_decimal"] = pd.to_numeric(df["int_rate_decimal"], errors="coerce")
    treatment = (
        df["int_rate_decimal"]
        .gt(df.groupby("original_grade")["int_rate_decimal"].transform("median"))
        .fillna(False)
    )
    df["treatment"] = treatment.astype(int)
    covars = [
        c
        for c in [
            "pd_high_alpha01",
            "qhat_v4",
            "loan_amnt",
            "annual_inc",
            "dti",
            "fico_score",
            "installment_burden",
        ]
        if c in df
    ]
    x = (
        df[covars]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(df[covars].median(numeric_only=True))
    )
    prop = np.repeat(float(df["treatment"].mean()), len(df))
    model_name = "mean_propensity_fallback"
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs"))
        model.fit(x, df["treatment"])
        prop = model.predict_proba(x)[:, 1]
        model_name = "logistic_propensity"
    except Exception:
        pass
    df["propensity_v25"] = np.clip(prop, 0.02, 0.98)
    df["trimmed_overlap_v25"] = df["propensity_v25"].between(0.05, 0.95)
    df["ipw_weight_v25"] = np.where(
        df["treatment"].eq(1), 1 / df["propensity_v25"], 1 / (1 - df["propensity_v25"])
    )
    rows = []
    for covar in covars:
        raw = v21._standardized_mean_difference(df, treatment, covar)
        trimmed_mask = df["trimmed_overlap_v25"]
        trimmed = v21._standardized_mean_difference(
            df[trimmed_mask], treatment[trimmed_mask], covar
        )
        rows.append(
            {
                "treatment_id": "high_rate_within_grade_v25",
                "covariate": covar,
                "smd_raw": raw,
                "smd_trimmed": trimmed,
                "balance_gate_trimmed_pass": bool(np.isfinite(trimmed) and abs(trimmed) <= 0.10),
            }
        )
    balance = pd.DataFrame(rows)
    outcomes = []
    for outcome in ["y_true", "base_return_vec"]:
        if outcome not in df:
            continue
        y = pd.to_numeric(df[outcome], errors="coerce").fillna(0.0)
        treated = df["treatment"].eq(1)
        naive = float(y[treated].mean() - y[~treated].mean())
        ipw_t = (
            float(np.average(y[treated], weights=df.loc[treated, "ipw_weight_v25"]))
            if treated.any()
            else np.nan
        )
        ipw_c = (
            float(np.average(y[~treated], weights=df.loc[~treated, "ipw_weight_v25"]))
            if (~treated).any()
            else np.nan
        )
        outcomes.append(
            {
                "outcome": outcome,
                "naive_difference": naive,
                "ipw_difference_proxy": ipw_t - ipw_c,
                "model": model_name,
                "causal_claim_allowed": False,
                "claim_boundary": "accepted-loan causal diagnostic only",
            }
        )
    sensitivity = pd.DataFrame(
        [
            {
                "hidden_bias_shift_pp": shift,
                "default_effect_after_shift": float(
                    outcomes[0]["ipw_difference_proxy"] if outcomes else 0.0
                )
                - shift / 100.0,
                "sensitivity_pass": abs(
                    float(outcomes[0]["ipw_difference_proxy"] if outcomes else 0.0)
                )
                > shift / 100.0,
                "claim_boundary": "diagnostic hidden-bias grid; not identification proof",
            }
            for shift in [1, 2, 3, 5, 7, 10]
        ]
    )
    gate = pd.DataFrame(
        [
            (
                "identification",
                "theory_blocked",
                "accepted-loan/reject-inference and pricing endogeneity remain",
                False,
            ),
            (
                "overlap",
                "review",
                f"trimmed overlap share={df['trimmed_overlap_v25'].mean():.3f}",
                False,
            ),
            (
                "balance",
                "review",
                f"max |trimmed SMD|={balance['smd_trimmed'].abs().max():.3f}",
                False,
            ),
            ("sensitivity", "blocked", "hidden-bias grid remains diagnostic only", False),
            ("falsification", "review", "placebos required for appendix, not promotion", False),
        ],
        columns=["gate", "status_v25", "evidence", "cate_policy_value_allowed"],
    )
    return balance, pd.DataFrame(outcomes), sensitivity, gate


def _fairness_source_governance(
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    families = [c for c in SOURCE_FAMILIES if c in books.columns]
    for fam in families:
        local = (
            books.groupby(["policy_id", fam], as_index=False)
            .agg(
                loans=("loan_id", "nunique"),
                exposure=("funded_exposure", "sum"),
                avg_pd=("pd_high_alpha01", "mean"),
                avg_width=("qhat_v4", "mean"),
            )
            .rename(columns={fam: "source_id"})
        )
        local["source_family"] = fam
        rows.append(local)
    for f1, f2 in [
        ("original_grade", "state_top20"),
        ("income_band", "dti_band"),
        ("score_decile", "original_grade"),
    ]:
        if f1 in books and f2 in books:
            local = books.groupby(["policy_id", f1, f2], as_index=False).agg(
                loans=("loan_id", "nunique"),
                exposure=("funded_exposure", "sum"),
                avg_pd=("pd_high_alpha01", "mean"),
                avg_width=("qhat_v4", "mean"),
            )
            local["source_id"] = local[f1].astype(str) + "|" + local[f2].astype(str)
            local["source_family"] = f"{f1}__{f2}"
            rows.append(
                local[
                    [
                        "policy_id",
                        "source_id",
                        "loans",
                        "exposure",
                        "avg_pd",
                        "avg_width",
                        "source_family",
                    ]
                ]
            )
    diag = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not diag.empty:
        diag["support_gate_pass_v25"] = diag["loans"].ge(25)
        diag["pooling_decision_v25"] = np.where(
            diag["support_gate_pass_v25"], "standalone_cell", "pool_to_parent_family"
        )
        diag["source_cap_proxy_v25"] = np.where(
            diag["source_family"].str.contains("__", regex=False), 0.15, 0.35
        )
        diag["no_fair_lending_legal_claim"] = True
    flags = pd.DataFrame(
        [
            ("protected_attributes", False, "data_blocked", "not available"),
            (
                "approved_external_proxy_protocol",
                False,
                "protocol_blocked",
                "not approved or implemented",
            ),
            ("fair_lending_legal_claim", False, "prohibited_claim", "explicitly not made"),
            (
                "source_governance_diagnostic",
                True,
                "usable_proxy",
                "observable source/cell monitoring only",
            ),
        ],
        columns=["claim_or_requirement", "allowed_v25", "status_v25", "evidence"],
    )
    return diag, flags, flags.copy()


def build_v25(solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    pool, books = _load_pool_books(solver_pool_n)
    panel, ifrs9_summary, sicr = _ifrs9_cashflow_proxy(books, pool)
    balance, outcomes, sensitivity, cate_gate = _causal_upgrade(pool)
    source_diag, fairness_protocol, no_claim = _fairness_source_governance(books)
    _write_parquet("paper4_v25_ifrs9_proxy_cashflow_panel.parquet", panel)
    _write_csv("paper4_v25_ifrs9_proxy_policy_summary.csv", ifrs9_summary)
    _write_csv("paper4_v25_ifrs9_sicr_sensitivity.csv", sicr)
    _write_csv("paper4_v25_causal_balance_trim_ipw.csv", balance)
    _write_csv("paper4_v25_causal_outcome_sensitivity.csv", outcomes)
    _write_csv("paper4_v25_causal_hidden_bias_grid.csv", sensitivity)
    _write_csv("paper4_v25_cate_gate_report.csv", cate_gate)
    _write_csv("paper4_v25_source_governance_diagnostics.csv", source_diag)
    _write_csv("paper4_v25_fairness_proxy_only_protocol.csv", fairness_protocol)
    _write_csv("paper4_v25_no_legal_claim_flags.csv", no_claim)
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v25_ifrs9_causal_fairness_upgrade",
        "ifrs9_contractual_claim_allowed": False,
        "ifrs9_cashflow_panel_rows_v25": int(len(panel)),
        "causal_policy_value_allowed": False,
        "causal_balance_rows_v25": int(len(balance)),
        "fair_lending_legal_claim": False,
        "source_governance_rows_v25": int(len(source_diag)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "IFRS9 proxy, causal diagnostics and source governance only",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v25_status.json", status)
    _write_note(
        "paper4_v25_ifrs9_causal_fairness_upgrade.md",
        "\n".join(
            [
                "# Paper 4 v25 IFRS9/Causal/Fairness Upgrade",
                "",
                f"- IFRS9 panel rows: `{status['ifrs9_cashflow_panel_rows_v25']}`.",
                f"- CATE policy value allowed: `{status['causal_policy_value_allowed']}`.",
                f"- Fair-lending legal claim: `{status['fair_lending_legal_claim']}`.",
                "",
                "All three lanes remain explicitly claim-bounded.",
            ]
        ),
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_v25(args.solver_pool_n)


if __name__ == "__main__":
    main()
