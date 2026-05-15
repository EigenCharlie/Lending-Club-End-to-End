"""Build Paper 4 v29 macro/sample-path, IFRS9, CATE and source-governance artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v19_dynamic_engine_v2 as v19
import scripts.papers.build_paper4_v25_ifrs9_causal_fairness_upgrade as v25
from scripts.papers.build_paper4_v6_priority_resolution import (
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.29"


def _macro_cache_v29() -> tuple[pd.DataFrame, pd.DataFrame]:
    source, macro = v19._external_macro_context()
    source = source.copy()
    source["version_v29"] = "external_macro_cache_retry"
    source["cache_artifact_v29"] = (
        "paper4_v29_external_macro_context.csv" if not macro.empty else ""
    )
    source["claim_boundary_v29"] = "official macro context only; internal paths are not forecasts"
    if not macro.empty:
        macro = macro.copy()
        macro["version_v29"] = "fred_context_cache"
        macro["macro_stress_label_v29"] = np.select(
            [
                macro.get("USREC", pd.Series(0, index=macro.index)).fillna(0).gt(0),
                macro.get("UNRATE", pd.Series(np.nan, index=macro.index))
                .fillna(0)
                .ge(macro.get("UNRATE", pd.Series(np.nan, index=macro.index)).quantile(0.80)),
            ],
            ["recession_context", "high_unemployment_context"],
            default="normal_context",
        )
    return source, macro


def _sample_paths_v4(macro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = TABLE_DIR / "paper4_v27_sample_paths.parquet"
    if path.exists():
        paths = pd.read_parquet(path).copy()
    else:
        paths = pd.read_parquet(TABLE_DIR / "paper4_v23_sample_paths.parquet").copy()
    paths["month"] = (
        pd.to_datetime(paths["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    if not macro.empty:
        macro_keep = macro[
            [
                "month",
                *[
                    c
                    for c in ["UNRATE", "FEDFUNDS", "DROCLACBS", "USREC", "macro_stress_label_v29"]
                    if c in macro
                ],
            ]
        ].copy()
        paths = paths.merge(macro_keep, on="month", how="left", suffixes=("", "_fred_v29"))
        for col in ["UNRATE", "FEDFUNDS", "DROCLACBS", "USREC"]:
            if col in paths:
                paths[col] = paths.groupby("path_id")[col].ffill().bfill()
        paths["external_macro_context_used_v29"] = True
    else:
        paths["macro_stress_label_v29"] = "internal_only"
        paths["external_macro_context_used_v29"] = False
    paths["vintage_factor_v29"] = 1.0 + 0.10 * np.sin(
        (pd.to_datetime(paths["month"]).dt.month.fillna(1).astype(float) / 12.0) * 2 * np.pi
    )
    paths["dependent_default_factor_v29"] = pd.to_numeric(
        paths.get("default_factor_v15", 1.0), errors="coerce"
    ).fillna(1.0) * (
        1.0
        + 0.08
        * pd.to_numeric(paths.get("systemic_factor_v15", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(-2, 2)
    )
    paths["cyclic_lgd_factor_v29"] = pd.to_numeric(
        paths.get("lgd_factor_v15", 1.0), errors="coerce"
    ).fillna(1.0) * (
        1.0
        + 0.05
        * pd.to_numeric(paths.get("systemic_factor_v15", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(-2, 2)
    )
    paths["prepayment_timing_factor_v29"] = pd.to_numeric(
        paths.get("prepay_factor_v15", 1.0), errors="coerce"
    ).fillna(1.0) * (
        1.0
        - 0.03
        * pd.to_numeric(paths.get("systemic_factor_v15", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(-2, 2)
    )
    paths["recovery_timing_lag_months_v29"] = np.where(
        paths["cyclic_lgd_factor_v29"].ge(paths["cyclic_lgd_factor_v29"].median()), 4, 3
    )
    paths["sample_path_version_v29"] = "v4_internal_calibration_with_optional_macro_context"
    paths["claim_boundary_v29"] = (
        "internal calibration and official context labels only; not forecast validation"
    )
    diag = (
        paths.groupby(["macro_regime_v15", "macro_stress_label_v29"], as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            n_months=("month", "nunique"),
            dependent_default_mean=("dependent_default_factor_v29", "mean"),
            dependent_default_p95=(
                "dependent_default_factor_v29",
                lambda s: float(np.quantile(s, 0.95)),
            ),
            cyclic_lgd_mean=("cyclic_lgd_factor_v29", "mean"),
            prepayment_timing_mean=("prepayment_timing_factor_v29", "mean"),
            recovery_lag_mean=("recovery_timing_lag_months_v29", "mean"),
        )
        .assign(calibration_scope_v29="internal path calibration v4")
    )
    family = (
        paths.groupby(["macro_regime_v15"], as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            default_factor_mean=("dependent_default_factor_v29", "mean"),
            lgd_factor_mean=("cyclic_lgd_factor_v29", "mean"),
            prepay_factor_mean=("prepayment_timing_factor_v29", "mean"),
            external_macro_used=("external_macro_context_used_v29", "max"),
        )
        .assign(path_family_claim_scope="stress design diagnostic, not forecast")
    )
    return paths, diag, family


def _ifrs9_v2(
    panel: pd.DataFrame, summary: pd.DataFrame, sicr: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    multipliers = pd.DataFrame(
        [
            ("baseline", 1.00, 1.00, 1.00),
            ("mild_stress", 1.15, 1.05, 0.95),
            ("macro_stress", 1.35, 1.15, 0.90),
        ],
        columns=["scenario_v29", "pd_multiplier", "lgd_multiplier", "prepay_multiplier"],
    )
    scenarios = []
    for _, sc in multipliers.iterrows():
        local = panel.copy()
        local["scenario_v29"] = sc["scenario_v29"]
        local["ecl_proxy_v29"] = (
            local["ecl_proxy_v25"] * float(sc["pd_multiplier"]) * float(sc["lgd_multiplier"])
        )
        local["prepayment_event_proxy_v29"] = local["prepayment_event_proxy"].astype(float) * float(
            sc["prepay_multiplier"]
        )
        local["contractual_ifrs9_claim_allowed"] = False
        scenarios.append(local)
    panel_v2 = pd.concat(scenarios, ignore_index=True)
    summary_v2 = (
        panel_v2.groupby(["policy_id", "scenario_v29"], as_index=False)
        .agg(
            ecl_proxy_total_v29=("ecl_proxy_v29", "sum"),
            loss_cash_proxy_total=("loss_cash_proxy", "sum"),
            recovery_cash_proxy_total=("recovery_cash_proxy", "sum"),
            prepayment_event_proxy_mean=("prepayment_event_proxy_v29", "mean"),
            stage2_share_proxy_v29=("stage_proxy_v25", lambda s: float((s == 2).mean())),
            stage3_share_proxy_v29=("stage_proxy_v25", lambda s: float((s == 3).mean())),
        )
        .assign(
            contractual_ifrs9_claim_allowed=False,
            claim_boundary_v29="IFRS9-inspired proxy, not contractual IFRS9",
        )
    )
    sicr = sicr.copy()
    for threshold in [0.15, 0.18, 0.20, 0.25]:
        sicr[f"stage2_abs_pd_threshold_{threshold:.2f}_sensitivity"] = np.clip(
            sicr["stage2_abs_pd"] * (0.20 / threshold), 0, 1
        )
    sicr["sicr_sensitivity_decision_v29"] = "proxy_stage2_review_not_production_ifrs9"
    sicr["contractual_ifrs9_claim_allowed"] = False
    return panel_v2, summary_v2, sicr


def _source_governance_v2(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = diag.copy()
    if out.empty:
        return out, pd.DataFrame()
    out["support_threshold_v29"] = np.select(
        [
            out["source_family"].astype(str).str.contains("__", regex=False),
            out["source_family"].astype(str).str.contains("month|period", case=False, regex=True),
        ],
        [50, 20],
        default=25,
    )
    out["support_gate_pass_v29"] = out["loans"].ge(out["support_threshold_v29"])
    out["pooling_decision_v29"] = np.where(
        out["support_gate_pass_v29"], "standalone_cell", "hierarchical_pool_to_parent"
    )
    out["source_cap_empirical_v29"] = np.where(
        out["support_gate_pass_v29"],
        np.minimum(
            0.40,
            np.maximum(
                0.10, out["exposure"] / out.groupby("policy_id")["exposure"].transform("sum")
            ),
        ),
        np.nan,
    )
    out["source_cap_use_v29"] = np.where(
        out["support_gate_pass_v29"], "solver_cap_candidate", "diagnostic_only_small_cell"
    )
    out["no_fair_lending_legal_claim"] = True
    cap_sensitivity = (
        out.groupby(["policy_id", "source_family"], as_index=False)
        .agg(
            worst_avg_pd=("avg_pd", "max"),
            worst_avg_width=("avg_width", "max"),
            support_pass_share=("support_gate_pass_v29", "mean"),
            diagnostic_cells=("source_id", "nunique"),
        )
        .assign(
            claim_boundary_v29="observable source governance only; no protected-attribute inference"
        )
    )
    return out, cap_sensitivity


def _cate_v2(
    balance: pd.DataFrame, outcomes: pd.DataFrame, sensitivity: pd.DataFrame, gate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balance = balance.copy()
    outcomes = outcomes.copy()
    sensitivity = sensitivity.copy()
    gate = gate.copy()
    if not balance.empty:
        balance["balance_gate_pass_v29"] = (
            pd.to_numeric(balance["smd_trimmed"], errors="coerce").abs().le(0.10)
        )
    if not outcomes.empty:
        outcomes["primary_outcome_flag_v29"] = outcomes["outcome"].eq("y_true")
        outcomes["secondary_outcome_flag_v29"] = outcomes["outcome"].isin(["base_return_vec"])
        outcomes["causal_policy_value_allowed"] = False
    if not sensitivity.empty:
        sensitivity["hidden_bias_grid_version_v29"] = "diagnostic_only"
        sensitivity["sensitivity_gate_pass_v29"] = False
    gate["status_v29"] = gate["status_v25"]
    gate["cate_policy_value_allowed"] = False
    gate["claim_boundary_v29"] = "accepted-loan causal diagnostic only; reject inference unresolved"
    return balance, outcomes, sensitivity, gate


def build_v29(solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    pool, books = v25._load_pool_books(solver_pool_n)
    macro_source, macro = _macro_cache_v29()
    paths, path_diag, path_family = _sample_paths_v4(macro)
    panel, ifrs9_summary, sicr = v25._ifrs9_cashflow_proxy(books, pool)
    panel_v2, ifrs9_summary_v2, sicr_v2 = _ifrs9_v2(panel, ifrs9_summary, sicr)
    balance, outcomes, sensitivity, cate_gate = v25._causal_upgrade(pool)
    balance_v2, outcomes_v2, sensitivity_v2, cate_gate_v2 = _cate_v2(
        balance, outcomes, sensitivity, cate_gate
    )
    source_diag, fairness_protocol, no_claim = v25._fairness_source_governance(books)
    source_v2, cap_sensitivity = _source_governance_v2(source_diag)
    fairness_protocol = fairness_protocol.copy()
    fairness_protocol["version_v29"] = "proxy_governance_only_v2"
    fairness_protocol["no_fair_lending_legal_claim"] = True
    no_claim = no_claim.copy()
    no_claim["allowed_v29"] = no_claim["allowed_v25"]
    no_claim.loc[no_claim["claim_or_requirement"].eq("fair_lending_legal_claim"), "allowed_v29"] = (
        False
    )

    _write_csv("paper4_v29_external_macro_source_registry.csv", macro_source)
    if not macro.empty:
        _write_csv("paper4_v29_external_macro_context.csv", macro)
    _write_parquet("paper4_v29_sample_paths.parquet", paths)
    _write_csv("paper4_v29_path_calibration_diagnostics.csv", path_diag)
    _write_csv("paper4_v29_path_family_stress_summary.csv", path_family)
    _write_parquet("paper4_v29_ifrs9_proxy_cashflow_panel.parquet", panel_v2)
    _write_csv("paper4_v29_ifrs9_proxy_policy_summary.csv", ifrs9_summary_v2)
    _write_csv("paper4_v29_ifrs9_sicr_sensitivity.csv", sicr_v2)
    _write_csv("paper4_v29_causal_balance_trim_ipw.csv", balance_v2)
    _write_csv("paper4_v29_causal_outcome_sensitivity.csv", outcomes_v2)
    _write_csv("paper4_v29_causal_hidden_bias_grid.csv", sensitivity_v2)
    _write_csv("paper4_v29_cate_gate_report.csv", cate_gate_v2)
    _write_csv("paper4_v29_source_governance_diagnostics.csv", source_v2)
    _write_csv("paper4_v29_source_cap_sensitivity.csv", cap_sensitivity)
    _write_csv("paper4_v29_fairness_proxy_only_protocol.csv", fairness_protocol)
    _write_csv("paper4_v29_no_legal_claim_flags.csv", no_claim)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v29_ifrs9_causal_fairness_source_upgrade",
        "external_macro_context_status_v29": str(
            macro_source.iloc[0].get("fetch_status", "not_attempted")
        ),
        "sample_path_rows_v29": int(len(paths)),
        "ifrs9_contractual_claim_allowed": False,
        "ifrs9_cashflow_panel_rows_v29": int(len(panel_v2)),
        "causal_policy_value_allowed": False,
        "causal_balance_rows_v29": int(len(balance_v2)),
        "fair_lending_legal_claim": False,
        "source_governance_rows_v29": int(len(source_v2)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "macro context/sample paths, IFRS9 proxy, causal diagnostics and source governance only",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v29_status.json", status)
    _write_note(
        "paper4_v29_ifrs9_causal_fairness_source_upgrade.md",
        "\n".join(
            [
                "# Paper 4 v29 IFRS9/CATE/Fairness/Source Upgrade",
                "",
                f"- Macro fetch status: `{status['external_macro_context_status_v29']}`.",
                f"- IFRS9 panel rows: `{status['ifrs9_cashflow_panel_rows_v29']}`.",
                "- Contractual IFRS9, CATE policy value and fair-lending legal claims remain blocked.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args()
    build_v29(args.solver_pool_n)


if __name__ == "__main__":
    main()
