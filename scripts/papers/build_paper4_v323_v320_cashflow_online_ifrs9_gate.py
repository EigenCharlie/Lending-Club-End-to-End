#!/usr/bin/env python3
"""Build Paper 4 v323 v320-specific cashflow, online, and IFRS9 gate artifacts."""

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

VERSION = 323
SOURCE_CANDIDATE_VERSION = 320
GATE_AUDIT_VERSION = 322
NEXT_VERSION = 324
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v320_proxy_gap_repair_or_branch_price_protocol.csv"
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
    raise RuntimeError("No v323 IFRS9 proxy template available.")


def _build_cashflow_proxy_panel(
    *,
    universe: pd.DataFrame,
    v320: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _panel_base(universe)
    v320 = v320.copy()
    v320["loan_id"] = v320["loan_id"].astype(str)
    v320_ids = set(v320["loan_id"])
    observed = panel.loc[panel["loan_id"].isin(v320_ids)].copy()
    observed_ids = set(observed["loan_id"])
    missing = v320.loc[~v320["loan_id"].isin(observed_ids)].copy()

    value_cols = [col for col in CASH_COLS + RATE_COLS if col in panel.columns]
    group_cols = ["month_index", "scenario", "grade", "score_decile", "period"]
    templates = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(group_cols, dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v323=("loan_amnt", "mean"),
            donor_rows_v323=("loan_id", "nunique"),
        )
        .reset_index()
    )
    fallback_grade_score = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(["month_index", "scenario", "grade", "score_decile"], dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v323=("loan_amnt", "mean"),
            donor_rows_v323=("loan_id", "nunique"),
        )
        .reset_index()
    )
    fallback_grade_score["period"] = "__any__"
    fallback_grade = (
        panel.dropna(subset=["loan_amnt"])
        .groupby(["month_index", "scenario", "grade"], dropna=False)
        .agg(
            **{col: (col, "mean") for col in value_cols},
            template_loan_amnt_v323=("loan_amnt", "mean"),
            donor_rows_v323=("loan_id", "nunique"),
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
            template_loan_amnt_v323=("loan_amnt", "mean"),
            donor_rows_v323=("loan_id", "nunique"),
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
                    float(template["template_loan_amnt_v323"]), 1.0
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
                    f"donor_rows_v{VERSION}": int(template["donor_rows_v323"]),
                    f"imputation_scale_v{VERSION}": scale,
                    f"claim_boundary_v{VERSION}": (
                        "v320-specific IFRS9-inspired proxy imputation; not contractual IFRS9"
                    ),
                }
                for col in value_cols:
                    value = float(template[col])
                    row[col] = value * scale if col in CASH_COLS else value
                imputed_rows.append(row)
    imputed = pd.DataFrame(imputed_rows)

    observed = observed.loc[observed["loan_id"].isin(v320_ids)].copy()
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
            loan_rows_v323=("loan_id", "nunique"),
            panel_rows_v323=("loan_id", "size"),
            ecl_proxy_total_v323=("ecl_proxy_v29", "sum"),
            net_cash_proxy_total_v323=("net_cash_proxy_v47", "sum"),
        )
        .reset_index()
    )
    source_summary[f"claim_boundary_v{VERSION}"] = (
        "v323 proxy source mix only; imputed rows are not contractual IFRS9 evidence"
    )
    summary = pd.DataFrame(
        [
            {
                f"candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"selected_rows_v{VERSION}": int(len(v320_ids)),
                f"observed_proxy_loan_rows_v{VERSION}": int(len(observed_ids)),
                f"imputed_proxy_loan_rows_v{VERSION}": int(len(missing)),
                f"cashflow_proxy_panel_rows_v{VERSION}": int(len(full_panel)),
                f"cashflow_proxy_months_v{VERSION}": int(full_panel["month_index"].nunique()),
                f"cashflow_proxy_scenarios_v{VERSION}": int(full_panel["scenario"].nunique()),
                f"post_imputation_coverage_share_v{VERSION}": float(
                    full_panel["loan_id"].nunique() / max(len(v320_ids), 1)
                ),
                f"observed_coverage_share_v{VERSION}": float(
                    len(observed_ids) / max(len(v320_ids), 1)
                ),
                f"ecl_proxy_total_v{VERSION}": float(full_panel["ecl_proxy_v29"].sum()),
                f"net_cash_proxy_total_v{VERSION}": float(full_panel["net_cash_proxy_v47"].sum()),
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "v320-specific proxy panel with explicit imputation; no contractual IFRS9 claim"
                ),
            }
        ]
    )
    return full_panel, summary, source_summary


def _online_temporal_rerun(
    universe: pd.DataFrame, v320: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = universe[FEATURE_COLS].copy()
    features["loan_id"] = features["loan_id"].astype(str)
    selected = v320[["loan_id"]].copy()
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
            for source_id, source in holdout.groupby(source_col, dropna=False):
                rows.append(
                    {
                        f"online_method_v{VERSION}": "v65_margin_repair_replayed_on_v320",
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
                            "v320 selected-book temporal replay only; no external online claim"
                        ),
                    }
                )
    cells = pd.DataFrame(rows)
    summary = (
        cells.groupby(["source_family", f"online_method_v{VERSION}"], dropna=False)
        .agg(
            **{
                f"holdout_periods_v{VERSION}": (f"holdout_period_v{VERSION}", "nunique"),
                f"source_cell_rows_v{VERSION}": ("source_id", "size"),
                f"worst_source_coverage_v{VERSION}": (f"coverage_v{VERSION}", "min"),
                f"worst_policy_period_coverage_v{VERSION}": (
                    f"policy_period_coverage_v{VERSION}",
                    "min",
                ),
                f"max_policy_period_width_v{VERSION}": (
                    f"policy_period_avg_width_v{VERSION}",
                    "max",
                ),
                f"source_gate_pass_rows_v{VERSION}": (f"source_gate80_v{VERSION}", "sum"),
                f"policy_gate_pass_rows_v{VERSION}": (f"policy_gate90_v{VERSION}", "sum"),
                f"width_gate_pass_rows_v{VERSION}": (f"width_gate95_v{VERSION}", "sum"),
            }
        )
        .reset_index()
    )
    summary[f"all_internal_gates_pass_v{VERSION}"] = (
        summary[f"source_gate_pass_rows_v{VERSION}"].eq(summary[f"source_cell_rows_v{VERSION}"])
        & summary[f"policy_gate_pass_rows_v{VERSION}"].eq(summary[f"source_cell_rows_v{VERSION}"])
        & summary[f"width_gate_pass_rows_v{VERSION}"].eq(summary[f"source_cell_rows_v{VERSION}"])
    )
    summary[f"external_holdout_available_v{VERSION}"] = False
    summary[f"strict_live_claim_allowed_v{VERSION}"] = False
    summary[f"claim_boundary_v{VERSION}"] = (
        "v320 temporal selected-book replay; external holdout required for live claim"
    )
    return cells, summary


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v323 v320-specific cashflow proxy panel.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v323_v320_cashflow_proxy_summary.csv"
                ),
                "boundary": "Observed plus explicitly imputed IFRS9-inspired proxy rows.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "Paper 4 has a v323 v320 temporal online rerun.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v323_v320_online_temporal_summary.csv"
                ),
                "boundary": "Selected-book temporal replay only; no external holdout.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v323 documents v320 proxy coverage regression versus v316.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v323_v320_cashflow_online_ifrs9_gate.csv"
                ),
                "boundary": "Diagnostic proxy-coverage comparison only; not a performance claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v323 implements contractual IFRS9 for v320.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v323_claim_blockers.csv"
                ),
                "boundary": "Panel includes imputed proxy rows, not contractual servicing data.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v323 authorizes live online deployability for v320.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v323_claim_blockers.csv"
                ),
                "boundary": "No external holdout is available.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v323 resolves v320 global or matched-period dynamic blockers.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v323_claim_blockers.csv"
                ),
                "boundary": "v323 is an online/cashflow/proxy gate, not a global certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v323 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v323_claim_blockers.csv"
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
                    "v323 builds a v320-specific cashflow proxy panel and temporal online rerun."
                ),
                "status": "v320_cashflow_proxy_and_temporal_online_rerun_executed",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "reduce the v320 proxy/imputation gap or build a branch-price/global "
                    "certificate without promoting"
                ),
                "last_wave": "v323",
                "execution_result": (
                    "cashflow_proxy_complete_with_79_imputed_rows_online_temporal_not_live"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v323")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V323_V320_CASHFLOW_ONLINE_IFRS9_GATE_START -->"
    end = "<!-- V323_V320_CASHFLOW_ONLINE_IFRS9_GATE_END -->"
    block = f"""
{start}

## Wave v323: v320 Cashflow / Online / IFRS9 Gate

Generated: {status["generated_at_utc"]}

### Objective

v322 showed that v320 beats both v295 and v316 in the final-period static-book
proxy but still lacks v320-specific cashflow/online/IFRS9 evidence. v323 builds
that candidate-specific proxy panel and temporal online replay.

### Results

- Cashflow proxy panel rows: `{status["cashflow_proxy_panel_rows_v323"]}`.
- Observed proxy loan rows: `{status["observed_proxy_loan_rows_v323"]}`.
- Imputed proxy loan rows: `{status["imputed_proxy_loan_rows_v323"]}`.
- Observed proxy delta vs v316:
  `{status["observed_proxy_delta_vs_v316_v323"]}`.
- Missing proxy delta vs v316:
  `{status["missing_proxy_delta_vs_v316_v323"]}`.
- Post-imputation coverage share:
  `{status["post_imputation_coverage_share_v323"]}`.
- Online temporal cell rows: `{status["online_temporal_cell_rows_v323"]}`.
- Online internal all-gate family rows:
  `{status["online_internal_all_gate_family_rows_v323"]}`.
- Strict live deployability claim allowed:
  `{status["strict_live_deployability_claim_allowed_v323"]}`.
- Contractual IFRS9 claim allowed:
  `{status["contractual_ifrs9_claim_allowed_v323"]}`.
- Working champion claim allowed:
  `{status["working_champion_claim_allowed_v323"]}`.

### Interpretation

v323 fills a real evidence gap by making the v320 cashflow/online evidence
candidate-specific. It also documents the price of the v320 matched-period
repair: 79 loans require imputation, 6 more than v316. Missing external holdout
data, imputed proxy rows and the global certificate gap continue to block live,
contractual and champion claims.

### Claim Impact

- Allowed: v320-specific cashflow proxy panel and temporal online replay.
- Still prohibited: contractual IFRS9, live online deployability, matched-period
  dynamic superiority, full-universe global optimality, Paper Estrella
  replacement, final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v323 in the living notebook. Promotion remains blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    v320 = read_parquet("paper4_v320_matched_period_allocations.parquet").reset_index(drop=True)
    v322_status = json.loads((STATUS_DIR / "paper4_v322_status.json").read_text(encoding="utf-8"))
    if universe.empty or v320.empty:
        raise RuntimeError("Missing v55 or v320 inputs for v323.")
    if int(v322_status["v320_missing_v47_proxy_rows_v322"]) <= 0:
        raise RuntimeError("v323 expects v322 to identify v320 IFRS9 proxy coverage gaps.")
    observed_proxy_delta_vs_v295 = int(v322_status["v320_observed_proxy_delta_vs_v295_v322"])
    missing_proxy_delta_vs_v295 = int(v322_status["v320_missing_proxy_delta_vs_v295_v322"])
    observed_proxy_delta_vs_v316 = int(v322_status["v320_observed_proxy_delta_vs_v316_v322"])
    missing_proxy_delta_vs_v316 = int(v322_status["v320_missing_proxy_delta_vs_v316_v322"])

    cashflow_panel, cashflow_summary, proxy_source_summary = _build_cashflow_proxy_panel(
        universe=universe,
        v320=v320,
    )
    online_cells, online_summary = _online_temporal_rerun(universe=universe, v320=v320)
    cashflow_row = cashflow_summary.iloc[0]
    online_all_gate_rows = int(online_summary[f"all_internal_gates_pass_v{VERSION}"].sum())
    strict_live_allowed = False
    contractual_ifrs9_allowed = False
    working_champion_allowed = False

    gate_summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v323_v320_cashflow_online_ifrs9_gate",
                f"candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"gate_audit_version_v{VERSION}": GATE_AUDIT_VERSION,
                f"selected_rows_v{VERSION}": int(cashflow_row[f"selected_rows_v{VERSION}"]),
                f"cashflow_proxy_panel_rows_v{VERSION}": int(
                    cashflow_row[f"cashflow_proxy_panel_rows_v{VERSION}"]
                ),
                f"observed_proxy_loan_rows_v{VERSION}": int(
                    cashflow_row[f"observed_proxy_loan_rows_v{VERSION}"]
                ),
                f"imputed_proxy_loan_rows_v{VERSION}": int(
                    cashflow_row[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"post_imputation_coverage_share_v{VERSION}": float(
                    cashflow_row[f"post_imputation_coverage_share_v{VERSION}"]
                ),
                f"observed_coverage_share_v{VERSION}": float(
                    cashflow_row[f"observed_coverage_share_v{VERSION}"]
                ),
                f"cashflow_proxy_months_v{VERSION}": int(
                    cashflow_row[f"cashflow_proxy_months_v{VERSION}"]
                ),
                f"cashflow_proxy_scenarios_v{VERSION}": int(
                    cashflow_row[f"cashflow_proxy_scenarios_v{VERSION}"]
                ),
                f"cashflow_proxy_source_rows_v{VERSION}": int(len(proxy_source_summary)),
                f"observed_proxy_delta_vs_v295_v{VERSION}": observed_proxy_delta_vs_v295,
                f"missing_proxy_delta_vs_v295_v{VERSION}": missing_proxy_delta_vs_v295,
                f"observed_proxy_delta_vs_v316_v{VERSION}": observed_proxy_delta_vs_v316,
                f"missing_proxy_delta_vs_v316_v{VERSION}": missing_proxy_delta_vs_v316,
                f"online_temporal_cell_rows_v{VERSION}": int(len(online_cells)),
                f"online_temporal_summary_rows_v{VERSION}": int(len(online_summary)),
                f"online_internal_all_gate_family_rows_v{VERSION}": online_all_gate_rows,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": contractual_ifrs9_allowed,
                f"strict_live_deployability_claim_allowed_v{VERSION}": strict_live_allowed,
                f"working_champion_claim_allowed_v{VERSION}": working_champion_allowed,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "v320 candidate-specific cashflow/online/IFRS9 proxy gate; not live or contractual"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "cashflow_proxy_imputation_required",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    cashflow_row[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "imputed rows block contractual IFRS9",
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_regression_vs_v316",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": abs(observed_proxy_delta_vs_v316),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "v320 has fewer observed v47 proxy rows than the v316 base"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_holdout_or_temporal_rerun",
                f"claim_boundary_v{VERSION}": "v323 online replay is internal selected-book only",
            },
            {
                f"blocker_id_v{VERSION}": "dynamic_live_replay_missing_after_v322",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    not bool(v322_status["matched_period_dynamic_claim_allowed_v322"])
                ),
                f"required_next_artifact_v{VERSION}": "future_matched_period_dynamic_replay",
                f"claim_boundary_v{VERSION}": (
                    "v322 matches v295 period counts but still lacks live dynamic replay"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "global_branch_price_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no global full-universe certificate is created",
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
                "claim_id": "v323_v320_cashflow_proxy_panel_executed",
                "allowed": True,
                "artifact": "paper4_v323_v320_cashflow_proxy_summary.csv",
                "boundary": "proxy panel with imputation",
            },
            {
                "claim_id": "v323_v320_online_temporal_rerun_executed",
                "allowed": True,
                "artifact": "paper4_v323_v320_online_temporal_summary.csv",
                "boundary": "internal selected-book replay",
            },
            {
                "claim_id": "v323_v320_internal_online_all_gates",
                "allowed": online_all_gate_rows == len(online_summary),
                "artifact": "paper4_v323_v320_online_temporal_summary.csv",
                "boundary": "internal historical replay only, not live deployability",
            },
            {
                "claim_id": "v323_v320_proxy_coverage_regression_vs_v316_documented",
                "allowed": True,
                "artifact": "paper4_v323_v320_cashflow_online_ifrs9_gate.csv",
                "boundary": "diagnostic coverage comparison only",
            },
            {
                "claim_id": "v323_contractual_ifrs9",
                "allowed": False,
                "artifact": "paper4_v323_claim_blockers.csv",
                "boundary": "imputed proxy rows and missing servicing panel",
            },
            {
                "claim_id": "v323_strict_live_online_deployability",
                "allowed": False,
                "artifact": "paper4_v323_claim_blockers.csv",
                "boundary": "external holdout missing",
            },
            {
                "claim_id": "v323_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v323_claim_blockers.csv",
                "boundary": "working champion and final promotion remain blocked",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v323_v320_cashflow_online_ifrs9_gate.csv", gate_summary)
    cashflow_panel.to_parquet(
        TABLE_DIR / "paper4_v323_v320_cashflow_proxy_panel.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v323_v320_cashflow_proxy_summary.csv", cashflow_summary)
    write_csv(
        TABLE_DIR / "paper4_v323_v320_cashflow_proxy_source_summary.csv", proxy_source_summary
    )
    write_csv(TABLE_DIR / "paper4_v323_v320_online_temporal_cells.csv", online_cells)
    write_csv(TABLE_DIR / "paper4_v323_v320_online_temporal_summary.csv", online_summary)
    write_csv(TABLE_DIR / "paper4_v323_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v323_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    gate_row = gate_summary.iloc[0]
    status = {
        "phase": "v323_v320_cashflow_online_ifrs9_gate",
        "schema_version": "2026-05-16.323",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v323": SOURCE_CANDIDATE_VERSION,
        "gate_audit_version_v323": GATE_AUDIT_VERSION,
        "selected_rows_v323": int(gate_row[f"selected_rows_v{VERSION}"]),
        "cashflow_proxy_panel_rows_v323": int(gate_row[f"cashflow_proxy_panel_rows_v{VERSION}"]),
        "observed_proxy_loan_rows_v323": int(gate_row[f"observed_proxy_loan_rows_v{VERSION}"]),
        "imputed_proxy_loan_rows_v323": int(gate_row[f"imputed_proxy_loan_rows_v{VERSION}"]),
        "post_imputation_coverage_share_v323": float(
            gate_row[f"post_imputation_coverage_share_v{VERSION}"]
        ),
        "observed_coverage_share_v323": float(gate_row[f"observed_coverage_share_v{VERSION}"]),
        "cashflow_proxy_months_v323": int(gate_row[f"cashflow_proxy_months_v{VERSION}"]),
        "cashflow_proxy_scenarios_v323": int(gate_row[f"cashflow_proxy_scenarios_v{VERSION}"]),
        "cashflow_proxy_source_rows_v323": int(gate_row[f"cashflow_proxy_source_rows_v{VERSION}"]),
        "observed_proxy_delta_vs_v295_v323": int(
            gate_row[f"observed_proxy_delta_vs_v295_v{VERSION}"]
        ),
        "missing_proxy_delta_vs_v295_v323": int(
            gate_row[f"missing_proxy_delta_vs_v295_v{VERSION}"]
        ),
        "observed_proxy_delta_vs_v316_v323": int(
            gate_row[f"observed_proxy_delta_vs_v316_v{VERSION}"]
        ),
        "missing_proxy_delta_vs_v316_v323": int(
            gate_row[f"missing_proxy_delta_vs_v316_v{VERSION}"]
        ),
        "online_temporal_cell_rows_v323": int(gate_row[f"online_temporal_cell_rows_v{VERSION}"]),
        "online_temporal_summary_rows_v323": int(
            gate_row[f"online_temporal_summary_rows_v{VERSION}"]
        ),
        "online_internal_all_gate_family_rows_v323": online_all_gate_rows,
        "strict_live_deployability_claim_allowed_v323": strict_live_allowed,
        "contractual_ifrs9_claim_allowed_v323": contractual_ifrs9_allowed,
        "working_champion_claim_allowed_v323": working_champion_allowed,
        "full_universe_integer_optimality_claim_allowed_v323": False,
        "paper1_promotion_allowed_v323": False,
        "paper4_working_champion_changed_v323": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v323": int(len(blockers)),
        "claim_matrix_rows_v323": int(len(claim_matrix)),
        "next_artifact_v323": gate_row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v323 builds v320-specific proxy cashflow and online temporal rerun artifacts; "
            "contractual IFRS9, strict live deployability, working champion, and promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v323_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v323": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
