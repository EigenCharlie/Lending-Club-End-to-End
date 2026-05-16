#!/usr/bin/env python3
"""Build Paper 4 v299 v295-specific cashflow/online holdout rerun artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 299
SOURCE_CANDIDATE_VERSION = 295
GATE_AUDIT_VERSION = 298
NEXT_VERSION = 300
FAMILIES = ["grade", "score_decile", "income_band", "dti_band", "period", "state_top20"]
FEATURE_COLS = [
    "loan_id",
    "loan_amnt",
    "grade",
    "score_decile",
    "income_band",
    "dti_band",
    "period",
    "state_top20",
    "term_months",
    "y_true",
    "pd_low_90",
    "pd_high_90",
    "width_90",
]
CASH_COLS = [
    "ead_start_proxy_v25",
    "ead_end_proxy_v25",
    "scheduled_principal_proxy",
    "interest_cash_proxy",
    "recovery_cash_proxy",
    "loss_cash_proxy",
    "ecl_proxy_v29",
    "net_cash_proxy_v47",
    "ead_path_proxy_v47",
]
RATE_COLS = [
    "default_event_proxy",
    "prepayment_event_proxy",
    "dpd_proxy",
    "sicr_absolute_pd",
    "sicr_relative_pd",
    "sicr_conformal_width",
    "sicr_dpd_proxy",
]


def _panel_base(universe: pd.DataFrame) -> pd.DataFrame:
    panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet").copy()
    panel["loan_id"] = panel["loan_id"].astype(str)
    keep_cols = [
        "loan_id",
        "month_index",
        "scenario",
        *[col for col in CASH_COLS + RATE_COLS if col in panel.columns],
    ]
    panel = panel[keep_cols].drop_duplicates(["loan_id", "month_index", "scenario"])
    features = universe[[col for col in FEATURE_COLS if col in universe.columns]].copy()
    features["loan_id"] = features["loan_id"].astype(str)
    return panel.merge(features, on="loan_id", how="left")


def _template_lookup(
    *,
    templates: pd.DataFrame,
    month_index: int,
    scenario: str,
    loan: pd.Series,
) -> tuple[pd.Series, str]:
    candidates = templates.loc[
        templates["month_index"].eq(month_index) & templates["scenario"].astype(str).eq(scenario)
    ]
    levels = [
        ("grade_score_period", ["grade", "score_decile", "period"]),
        ("grade_score", ["grade", "score_decile"]),
        ("grade", ["grade"]),
        ("global", []),
    ]
    for level_name, cols in levels:
        group = candidates
        for col in cols:
            group = group.loc[group[col].astype(str).eq(str(loan[col]))]
        if not group.empty:
            return group.iloc[0], level_name
    raise RuntimeError("No v299 IFRS9 proxy template available.")


def _build_cashflow_proxy_panel(
    *,
    universe: pd.DataFrame,
    v295: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _panel_base(universe)
    v295 = v295.copy()
    v295["loan_id"] = v295["loan_id"].astype(str)
    v295_ids = set(v295["loan_id"])
    observed = panel.loc[panel["loan_id"].isin(v295_ids)].copy()
    observed_ids = set(observed["loan_id"])
    missing = v295.loc[~v295["loan_id"].isin(observed_ids)].copy()

    value_cols = [col for col in CASH_COLS + RATE_COLS if col in panel.columns]
    group_cols = ["month_index", "scenario", "grade", "score_decile", "period"]
    templates = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(group_cols, dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v299=("loan_amnt", "mean"),
            donor_rows_v299=("loan_id", "nunique"),
        )
        .reset_index()
    )
    fallback_cols = ["month_index", "scenario", "grade", "score_decile"]
    fallback_grade_score = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(fallback_cols, dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v299=("loan_amnt", "mean"),
            donor_rows_v299=("loan_id", "nunique"),
        )
        .reset_index()
    )
    fallback_grade_score["period"] = "__any__"
    fallback_grade = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(["month_index", "scenario", "grade"], dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v299=("loan_amnt", "mean"),
            donor_rows_v299=("loan_id", "nunique"),
        )
        .reset_index()
    )
    fallback_grade["score_decile"] = "__any__"
    fallback_grade["period"] = "__any__"
    fallback_global = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(["month_index", "scenario"], dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v299=("loan_amnt", "mean"),
            donor_rows_v299=("loan_id", "nunique"),
        )
        .reset_index()
    )
    fallback_global["grade"] = "__any__"
    fallback_global["score_decile"] = "__any__"
    fallback_global["period"] = "__any__"
    all_templates = pd.concat(
        [templates, fallback_grade_score, fallback_grade, fallback_global],
        ignore_index=True,
    )

    scenarios = sorted(panel["scenario"].astype(str).dropna().unique())
    months = sorted(panel["month_index"].dropna().astype(int).unique())
    imputed_rows: list[dict[str, Any]] = []
    for _, loan in missing.iterrows():
        for scenario in scenarios:
            for month_index in months:
                template, level = _template_lookup(
                    templates=all_templates,
                    month_index=int(month_index),
                    scenario=str(scenario),
                    loan=loan,
                )
                scale = float(loan["loan_amnt"]) / max(
                    float(template["template_loan_amnt_v299"]), 1.0
                )
                row: dict[str, Any] = {
                    "loan_id": str(loan["loan_id"]),
                    "month_index": int(month_index),
                    "scenario": str(scenario),
                    "loan_amnt": float(loan["loan_amnt"]),
                    "grade": str(loan["grade"]),
                    "score_decile": str(loan["score_decile"]),
                    "income_band": str(loan["income_band"]),
                    "dti_band": str(loan["dti_band"]),
                    "period": str(loan["period"]),
                    "state_top20": str(loan["state_top20"]),
                    "term_months": int(loan.get("term_months", 36)),
                    f"proxy_source_v{VERSION}": f"imputed_{level}",
                    f"donor_rows_v{VERSION}": int(template["donor_rows_v299"]),
                    f"imputation_scale_v{VERSION}": scale,
                    f"claim_boundary_v{VERSION}": (
                        "v295-specific IFRS9-inspired proxy imputation; not contractual IFRS9"
                    ),
                }
                for col in value_cols:
                    value = float(template[col])
                    row[col] = value * scale if col in CASH_COLS else value
                imputed_rows.append(row)
    imputed = pd.DataFrame(imputed_rows)

    observed = observed.loc[observed["loan_id"].isin(v295_ids)].copy()
    observed[f"proxy_source_v{VERSION}"] = "observed_v47_proxy"
    observed[f"donor_rows_v{VERSION}"] = 1
    observed[f"imputation_scale_v{VERSION}"] = 1.0
    observed[f"claim_boundary_v{VERSION}"] = (
        "observed historical IFRS9-inspired proxy row; not contractual IFRS9"
    )
    full_panel = pd.concat([observed, imputed], ignore_index=True, sort=False)
    full_panel["loan_id"] = full_panel["loan_id"].astype(str)
    source_summary = (
        full_panel.groupby(f"proxy_source_v{VERSION}", dropna=False)
        .agg(
            loan_rows_v299=("loan_id", "nunique"),
            panel_rows_v299=("loan_id", "size"),
            ecl_proxy_total_v299=("ecl_proxy_v29", "sum"),
            net_cash_proxy_total_v299=("net_cash_proxy_v47", "sum"),
        )
        .reset_index()
    )
    source_summary[f"claim_boundary_v{VERSION}"] = (
        "v299 proxy source mix only; imputed rows are not contractual IFRS9 evidence"
    )
    summary = pd.DataFrame(
        [
            {
                f"candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"selected_rows_v{VERSION}": int(len(v295_ids)),
                f"observed_proxy_loan_rows_v{VERSION}": int(len(observed_ids)),
                f"imputed_proxy_loan_rows_v{VERSION}": int(len(missing)),
                f"cashflow_proxy_panel_rows_v{VERSION}": int(len(full_panel)),
                f"cashflow_proxy_months_v{VERSION}": int(full_panel["month_index"].nunique()),
                f"cashflow_proxy_scenarios_v{VERSION}": int(full_panel["scenario"].nunique()),
                f"post_imputation_coverage_share_v{VERSION}": float(
                    full_panel["loan_id"].nunique() / max(len(v295_ids), 1)
                ),
                f"observed_coverage_share_v{VERSION}": float(
                    len(observed_ids) / max(len(v295_ids), 1)
                ),
                f"ecl_proxy_total_v{VERSION}": float(full_panel["ecl_proxy_v29"].sum()),
                f"net_cash_proxy_total_v{VERSION}": float(full_panel["net_cash_proxy_v47"].sum()),
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "v295-specific proxy panel with explicit imputation; no contractual IFRS9 claim"
                ),
            }
        ]
    )
    return full_panel, summary, source_summary


def _online_temporal_rerun(
    universe: pd.DataFrame, v295: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = universe[FEATURE_COLS].copy()
    features["loan_id"] = features["loan_id"].astype(str)
    selected = v295[["loan_id"]].copy()
    selected["loan_id"] = selected["loan_id"].astype(str)
    panel = selected.merge(features, on="loan_id", how="left")
    winners = read_csv("paper4_v65_online_margin_repair_winners.csv")
    rows: list[dict[str, Any]] = []
    for _, winner in winners.iterrows():
        family = str(winner["source_family"])
        source_col = "period" if family == "period" else "term_months"
        global_delta = float(winner["global_delta_v65"])
        work = panel.copy()
        work[f"interval_low_v{VERSION}"] = (work["pd_low_90"] - global_delta).clip(lower=0.0)
        work[f"interval_high_v{VERSION}"] = (work["pd_high_90"] + global_delta).clip(upper=1.0)
        work[f"interval_width_v{VERSION}"] = (
            work[f"interval_high_v{VERSION}"] - work[f"interval_low_v{VERSION}"]
        )
        work[f"covered_online_v{VERSION}"] = work["y_true"].ge(
            work[f"interval_low_v{VERSION}"]
        ) & work["y_true"].le(work[f"interval_high_v{VERSION}"])
        for holdout_period, holdout in work.groupby("period", dropna=False):
            policy_coverage = float(holdout[f"covered_online_v{VERSION}"].mean())
            policy_width = float(holdout[f"interval_width_v{VERSION}"].mean())
            source_rows = []
            for source_id, source in holdout.groupby(source_col, dropna=False):
                source_rows.append(
                    {
                        f"online_method_v{VERSION}": "v65_margin_repair_replayed_on_v295",
                        "source_family": family,
                        "source_id": str(source_id),
                        f"holdout_period_v{VERSION}": str(holdout_period),
                        f"loan_rows_v{VERSION}": int(len(source)),
                        f"coverage_v{VERSION}": float(source[f"covered_online_v{VERSION}"].mean()),
                        f"avg_interval_width_v{VERSION}": float(
                            source[f"interval_width_v{VERSION}"].mean()
                        ),
                        f"policy_period_coverage_v{VERSION}": policy_coverage,
                        f"policy_period_avg_width_v{VERSION}": policy_width,
                        f"source_gate80_v{VERSION}": bool(
                            source[f"covered_online_v{VERSION}"].mean() >= 0.80
                        ),
                        f"policy_gate90_v{VERSION}": bool(policy_coverage >= 0.90),
                        f"width_gate95_v{VERSION}": bool(policy_width <= 0.95),
                        f"external_holdout_available_v{VERSION}": False,
                        f"strict_live_claim_allowed_v{VERSION}": False,
                        f"claim_boundary_v{VERSION}": (
                            "v295 selected-book temporal replay only; no external online claim"
                        ),
                    }
                )
            rows.extend(source_rows)
    cells = pd.DataFrame(rows)
    summary = (
        cells.groupby(["source_family", f"online_method_v{VERSION}"], dropna=False)
        .agg(
            holdout_periods_v299=(f"holdout_period_v{VERSION}", "nunique"),
            source_cell_rows_v299=("source_id", "size"),
            worst_source_coverage_v299=(f"coverage_v{VERSION}", "min"),
            worst_policy_period_coverage_v299=(f"policy_period_coverage_v{VERSION}", "min"),
            max_policy_period_width_v299=(f"policy_period_avg_width_v{VERSION}", "max"),
            source_gate_pass_rows_v299=(f"source_gate80_v{VERSION}", "sum"),
            policy_gate_pass_rows_v299=(f"policy_gate90_v{VERSION}", "sum"),
            width_gate_pass_rows_v299=(f"width_gate95_v{VERSION}", "sum"),
        )
        .reset_index()
    )
    summary[f"all_internal_gates_pass_v{VERSION}"] = (
        summary["source_gate_pass_rows_v299"].eq(summary["source_cell_rows_v299"])
        & summary["policy_gate_pass_rows_v299"].eq(summary["source_cell_rows_v299"])
        & summary["width_gate_pass_rows_v299"].eq(summary["source_cell_rows_v299"])
    )
    summary[f"external_holdout_available_v{VERSION}"] = False
    summary[f"strict_live_claim_allowed_v{VERSION}"] = False
    summary[f"claim_boundary_v{VERSION}"] = (
        "v295 temporal selected-book replay; external holdout required for live claim"
    )
    return cells, summary


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v299 v295-specific cashflow proxy panel.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v299_v295_cashflow_proxy_summary.csv"
                ),
                "boundary": "Observed plus explicitly imputed IFRS9-inspired proxy rows.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "Paper 4 has a v299 v295 temporal online rerun.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v299_v295_online_temporal_summary.csv"
                ),
                "boundary": "Selected-book temporal replay only; no external holdout.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v299 implements contractual IFRS9 for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v299_claim_blockers.csv"
                ),
                "boundary": "Panel includes imputed proxy rows, not contractual servicing data.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v299 authorizes live online deployability for v295.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v299_claim_blockers.csv"
                ),
                "boundary": "No external holdout is available.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v299 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v299_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Online/IFRS9/SPO-DLA",
                "executable_item": (
                    "v299 builds a v295-specific cashflow proxy panel and temporal online rerun."
                ),
                "status": "v295_cashflow_proxy_and_temporal_online_rerun_executed",
                "next_artifact": f"paper4_v{NEXT_VERSION}_v295_source_governance_or_branch_price_protocol.csv",
                "success_condition": (
                    "reduce imputation/external holdout blockers or move to source-governance/"
                    "branch-price protocol without promoting"
                ),
                "last_wave": "v299",
                "execution_result": "cashflow_proxy_complete_with_imputation_online_temporal_not_live",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v299")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V299_V295_CASHFLOW_ONLINE_RERUN_START -->"
    end = "<!-- V299_V295_CASHFLOW_ONLINE_RERUN_END -->"
    block = f"""
{start}

## Wave v299: v295 Cashflow Proxy / Online Temporal Rerun

Generated: {status["generated_at_utc"]}

### Objective

v298 showed that v295 needed actual candidate-specific cashflow/online reruns.
v299 builds a complete v295 IFRS9-inspired proxy cashflow panel and replays the
v65 online margin logic on v295 selected loans by temporal/source cells.

### Results

- Cashflow proxy panel rows: `{status["cashflow_proxy_panel_rows_v299"]}`.
- Observed proxy loan rows: `{status["observed_proxy_loan_rows_v299"]}`.
- Imputed proxy loan rows: `{status["imputed_proxy_loan_rows_v299"]}`.
- Post-imputation coverage share:
  `{status["post_imputation_coverage_share_v299"]}`.
- Online temporal cell rows: `{status["online_temporal_cell_rows_v299"]}`.
- Online internal all-gate family rows:
  `{status["online_internal_all_gate_family_rows_v299"]}`.
- Strict live deployability claim allowed:
  `{status["strict_live_deployability_claim_allowed_v299"]}`.
- Contractual IFRS9 claim allowed:
  `{status["contractual_ifrs9_claim_allowed_v299"]}`.

### Interpretation

v299 improves coverage materially by creating a full v295 proxy panel, but 76
loans require imputation. The online replay is candidate-specific and temporal,
yet it is still internal selected-book evidence, not external live validation.

### Claim Impact

- Allowed: v295-specific cashflow proxy panel and temporal online replay.
- Still prohibited: contractual IFRS9, live online deployability, Paper Estrella
  replacement, final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v299 in the living notebook. Promotion remains blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    v295 = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(drop=True)
    v298_status = json.loads((STATUS_DIR / "paper4_v298_status.json").read_text(encoding="utf-8"))
    if universe.empty or v295.empty:
        raise RuntimeError("Missing v55 or v295 inputs for v299.")
    if int(v298_status["ifrs9_proxy_uncovered_loan_rows_v298"]) <= 0:
        raise RuntimeError("v299 expects v298 to identify v295 IFRS9 proxy coverage gaps.")

    cashflow_panel, cashflow_summary, proxy_source_summary = _build_cashflow_proxy_panel(
        universe=universe,
        v295=v295,
    )
    online_cells, online_summary = _online_temporal_rerun(universe=universe, v295=v295)
    cashflow_row = cashflow_summary.iloc[0]
    online_all_gate_rows = int(online_summary[f"all_internal_gates_pass_v{VERSION}"].sum())
    strict_live_allowed = False
    contractual_ifrs9_allowed = False
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "cashflow_proxy_imputation_required",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    cashflow_row[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": "future_observed_v295_cashflow_panel",
                f"claim_boundary_v{VERSION}": "imputed rows block contractual IFRS9",
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_holdout_or_temporal_rerun",
                f"claim_boundary_v{VERSION}": "v299 online replay is internal selected-book only",
            },
            {
                f"blocker_id_v{VERSION}": "strict_live_claim_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_live_deployment_validation",
                f"claim_boundary_v{VERSION}": "no live deployment validation is created",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_working_champion_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_promotion_gate",
                f"claim_boundary_v{VERSION}": "working champion replacement remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v299_v295_cashflow_proxy_panel_executed",
                "allowed": True,
                "artifact": "paper4_v299_v295_cashflow_proxy_summary.csv",
                "boundary": "proxy panel with imputation",
            },
            {
                "claim_id": "v299_v295_online_temporal_rerun_executed",
                "allowed": True,
                "artifact": "paper4_v299_v295_online_temporal_summary.csv",
                "boundary": "internal selected-book replay",
            },
            {
                "claim_id": "v299_contractual_ifrs9",
                "allowed": False,
                "artifact": "paper4_v299_claim_blockers.csv",
                "boundary": "imputed proxy rows and missing servicing panel",
            },
            {
                "claim_id": "v299_strict_live_online_deployability",
                "allowed": False,
                "artifact": "paper4_v299_claim_blockers.csv",
                "boundary": "external holdout missing",
            },
            {
                "claim_id": "v299_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v299_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    cashflow_panel.to_parquet(
        TABLE_DIR / "paper4_v299_v295_cashflow_proxy_panel.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v299_v295_cashflow_proxy_summary.csv", cashflow_summary)
    write_csv(
        TABLE_DIR / "paper4_v299_v295_cashflow_proxy_source_summary.csv", proxy_source_summary
    )
    write_csv(TABLE_DIR / "paper4_v299_v295_online_temporal_cells.csv", online_cells)
    write_csv(TABLE_DIR / "paper4_v299_v295_online_temporal_summary.csv", online_summary)
    write_csv(TABLE_DIR / "paper4_v299_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v299_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v299_v295_cashflow_or_online_holdout_rerun",
        "schema_version": "2026-05-15.299",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v299": SOURCE_CANDIDATE_VERSION,
        "gate_audit_version_v299": GATE_AUDIT_VERSION,
        "selected_rows_v299": int(cashflow_row[f"selected_rows_v{VERSION}"]),
        "cashflow_proxy_panel_rows_v299": int(
            cashflow_row[f"cashflow_proxy_panel_rows_v{VERSION}"]
        ),
        "observed_proxy_loan_rows_v299": int(cashflow_row[f"observed_proxy_loan_rows_v{VERSION}"]),
        "imputed_proxy_loan_rows_v299": int(cashflow_row[f"imputed_proxy_loan_rows_v{VERSION}"]),
        "post_imputation_coverage_share_v299": float(
            cashflow_row[f"post_imputation_coverage_share_v{VERSION}"]
        ),
        "observed_coverage_share_v299": float(cashflow_row[f"observed_coverage_share_v{VERSION}"]),
        "cashflow_proxy_months_v299": int(cashflow_row[f"cashflow_proxy_months_v{VERSION}"]),
        "cashflow_proxy_scenarios_v299": int(cashflow_row[f"cashflow_proxy_scenarios_v{VERSION}"]),
        "cashflow_proxy_source_rows_v299": int(len(proxy_source_summary)),
        "online_temporal_cell_rows_v299": int(len(online_cells)),
        "online_temporal_summary_rows_v299": int(len(online_summary)),
        "online_internal_all_gate_family_rows_v299": online_all_gate_rows,
        "strict_live_deployability_claim_allowed_v299": strict_live_allowed,
        "contractual_ifrs9_claim_allowed_v299": contractual_ifrs9_allowed,
        "working_champion_claim_allowed_v299": False,
        "paper1_promotion_allowed_v299": False,
        "paper4_working_champion_changed_v299": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v299": int(len(blockers)),
        "claim_matrix_rows_v299": int(len(claim_matrix)),
        "next_artifact_v299": (
            f"paper4_v{NEXT_VERSION}_v295_source_governance_or_branch_price_protocol.csv"
        ),
        "claim_boundary": (
            "v299 builds v295-specific proxy cashflow and online temporal rerun artifacts; "
            "contractual IFRS9, strict live deployability, working champion, and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v299_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v299": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
