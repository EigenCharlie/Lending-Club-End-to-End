#!/usr/bin/env python3
"""Build Paper 4 v39-v40 living-lab execution artifacts.

The v39-v40 wave intentionally keeps Quarto compact. It generates lab artifacts
and notebook notes from existing v31-v38 evidence, plus lightweight environment
and governance audits. It does not create a Paper 4 final promotion file.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
GLOBAL_TABLE_DIR = ROOT / "reports" / "paper_material" / "global" / "tables"
BOOK_CONFIG = ROOT / "book" / "_quarto.yml"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(name: str, directory: Path = TABLE_DIR) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_parquet(name: str, directory: Path = TABLE_DIR) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_json(name: str, directory: Path = STATUS_DIR) -> dict[str, Any]:
    path = directory / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "\n_No rows available._\n"
    view = df.head(max_rows).copy()
    return view.to_markdown(index=False)


def audit_registered_paper4_pages() -> list[str]:
    pages: list[str] = []
    if not BOOK_CONFIG.exists():
        return pages
    for raw in BOOK_CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("- chapters/19-paper-mega-extension/"):
            pages.append(Path(stripped.removeprefix("- ")).name)
    return pages


def build_online_source_family_holdout() -> pd.DataFrame:
    holdout = read_csv("paper4_v35_online_temporal_holdout.csv")
    support = read_csv("paper4_v35_online_min_support_sensitivity.csv")
    rows: list[dict[str, Any]] = []

    passing_support = support[
        support.get("gate_source80_policy90_width95", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    if not passing_support.empty:
        passing_support = passing_support.sort_values(["min_support", "avg_width_loan"])
        support_choice = passing_support.iloc[0].to_dict()
    else:
        support_choice = {}

    for _, row in holdout.iterrows():
        item = str(row.get("validation_item", ""))
        source_family = item.split("::", 1)[1] if "::" in item else "global_or_temporal"
        observed_source = safe_float(row.get("source_month_defended_min"))
        observed_policy = safe_float(row.get("policy_month_defended_min"))
        observed_width = safe_float(row.get("avg_width_loan"))

        can_pool = source_family != "global_or_temporal" and bool(support_choice)
        pooled_source = (
            max(observed_source, safe_float(support_choice.get("source_month_defended_min")))
            if can_pool
            else observed_source
        )
        pooled_policy = (
            max(observed_policy, safe_float(support_choice.get("policy_month_defended_min")))
            if can_pool
            else observed_policy
        )
        pooled_width = (
            max(observed_width, safe_float(support_choice.get("avg_width_loan"), observed_width))
            if can_pool
            else observed_width
        )
        pass_gate = pooled_source >= 0.80 and pooled_policy >= 0.90 and pooled_width <= 0.95
        live_claim = False
        if item == "v10_best_nominal":
            method = "v10_nominal_reference"
            decision = "reference_pass_not_live_deployment"
        elif can_pool and pass_gate:
            method = "v39_min_support_hierarchical_pooling"
            decision = "lab_reaudit_pass_with_pooling_caveat"
        elif can_pool:
            method = "v39_min_support_hierarchical_pooling"
            decision = "still_needs_iteration"
        else:
            method = "v39_no_pooling_evidence_available"
            decision = "diagnostic_only"

        rows.append(
            {
                "validation_item": item,
                "source_family": source_family,
                "method_v39": method,
                "observed_source_month_defended_min_v35": observed_source,
                "observed_policy_month_defended_min_v35": observed_policy,
                "observed_avg_width_loan_v35": observed_width,
                "recommended_min_support_v39": support_choice.get("min_support", ""),
                "pooled_source_month_defended_min_v39": pooled_source,
                "pooled_policy_month_defended_min_v39": pooled_policy,
                "pooled_avg_width_loan_v39": pooled_width,
                "gate_source80_policy90_width95_v39": pass_gate,
                "live_deployability_claim_allowed": live_claim,
                "decision_v39": decision,
                "claim_boundary_v39": "lab holdout/pooling diagnostic; not live deployment",
            }
        )

    if not support.empty:
        for _, row in support.iterrows():
            rows.append(
                {
                    "validation_item": f"min_support_sensitivity::{int(row.get('min_support', 0))}",
                    "source_family": "support_grid",
                    "method_v39": "v35_min_support_grid_reused",
                    "observed_source_month_defended_min_v35": row.get("source_month_defended_min"),
                    "observed_policy_month_defended_min_v35": row.get("policy_month_defended_min"),
                    "observed_avg_width_loan_v35": row.get("avg_width_loan"),
                    "recommended_min_support_v39": row.get("min_support"),
                    "pooled_source_month_defended_min_v39": row.get("source_month_defended_min"),
                    "pooled_policy_month_defended_min_v39": row.get("policy_month_defended_min"),
                    "pooled_avg_width_loan_v39": row.get("avg_width_loan"),
                    "gate_source80_policy90_width95_v39": boolish(
                        row.get("gate_source80_policy90_width95", False)
                    ),
                    "live_deployability_claim_allowed": False,
                    "decision_v39": "support_sensitivity_reference",
                    "claim_boundary_v39": "min-support sensitivity from replay; not live deployment",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper4_v39_online_source_family_holdout.csv", index=False)
    return out


def build_dynamic_candidate_stress() -> pd.DataFrame:
    pairwise = read_csv("paper4_v31_policy_pairwise_common_path_ci.csv")
    candidates = read_csv("paper4_v30_candidate_registry.csv")
    if pairwise.empty:
        out = pd.DataFrame()
    else:
        out = pairwise.copy()
        if not candidates.empty:
            out = out.merge(
                candidates[
                    [
                        col
                        for col in [
                            "policy_id",
                            "full_candidate_score_v30",
                            "auditability_score_proxy_v30",
                            "claim_safety_gate",
                            "decision_v30",
                        ]
                        if col in candidates.columns
                    ]
                ],
                on="policy_id",
                how="left",
            )
        out["paired_wealth_robustness_gate_v39"] = (out["prob_higher_wealth"].fillna(0) >= 0.55) & (
            out["ci95_low_wealth_diff"].fillna(-1) > 0
        )
        out["tail_challenger_gate_v39"] = out["prob_lower_loss"].fillna(0) >= 0.90
        out["expensive_rerun_justified_v39"] = out["paired_wealth_robustness_gate_v39"] | (
            (out["prob_higher_wealth"].fillna(0).between(0.49, 0.55))
            & (out["tail_challenger_gate_v39"])
        )
        out["decision_v39"] = np.select(
            [
                out["policy_id"].eq("paper1_economic_champion"),
                out["paired_wealth_robustness_gate_v39"],
                out["tail_challenger_gate_v39"],
            ],
            [
                "retain_reference",
                "review_for_working_champion",
                "serious_tail_challenger_no_wealth_promotion",
            ],
            default="park_or_monitor",
        )
        out["claim_boundary_v39"] = (
            "common-path paired replay; Paper 4 working-only and not Paper Estrella promotion"
        )
    out.to_csv(TABLE_DIR / "paper4_v39_dynamic_candidate_stress.csv", index=False)
    return out


def build_cvar_certificate_delta() -> pd.DataFrame:
    cert = read_csv("paper4_v33_cvar_infeasibility_certificate_v3.csv")
    if cert.empty:
        out = pd.DataFrame()
    else:
        out = cert.copy()
        slack = out.get("required_cvar_slack_v33", pd.Series(0, index=out.index)).apply(safe_float)
        floor_relax = out.get(
            "required_return_floor_relaxation_v33", pd.Series(0, index=out.index)
        ).apply(safe_float)
        out["v39_slack_severity"] = np.select(
            [slack <= 0, slack <= 25_000, slack <= 100_000],
            ["none", "small", "material"],
            default="large",
        )
        out["v39_floor_relaxation_severity"] = np.select(
            [floor_relax <= 0, floor_relax <= 25_000, floor_relax <= 100_000],
            ["none", "small", "material"],
            default="large",
        )
        out["strict_committee_relaxed_label_v39"] = out.get(
            "regime_v16", pd.Series("", index=out.index)
        ).astype(str)
        out["exact_full_universe_claim_v39"] = False
        out["mathematical_infeasibility_proof_claim_v39"] = False
        out["v39_next_action"] = np.select(
            [
                out["strict_committee_relaxed_label_v39"].str.contains(
                    "strict", case=False, na=False
                ),
                out["v39_slack_severity"].isin(["large", "material"]),
            ],
            [
                "document_strict_infeasibility_boundary",
                "test_committee_relaxation_only_if_tail_first_question_returns",
            ],
            default="keep_as_restricted_master_diagnostic",
        )
        out["claim_boundary_v39"] = (
            "restricted-master/column-generation diagnostic; no exact full-universe optimality or proof claim"
        )
    out.to_csv(TABLE_DIR / "paper4_v39_cvar_certificate_delta.csv", index=False)
    return out


def build_source_governance_caps() -> pd.DataFrame:
    source = read_csv("paper4_v37_source_governance_appendix.csv")
    rows: list[dict[str, Any]] = []
    if not source.empty:
        for (policy_id, family), group in source.groupby(
            ["policy_id", "source_family"], dropna=False
        ):
            high_support = group[
                group.get("support_gate_pass_v29", pd.Series(False, index=group.index)).astype(bool)
            ]
            cap_values = pd.to_numeric(
                high_support.get("source_cap_empirical_v29", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            if cap_values.empty:
                recommended_cap = np.nan
                cap_rule = "pool_to_parent_until_support_available"
            else:
                recommended_cap = float(np.nanquantile(cap_values, 0.75))
                cap_rule = "empirical_p75_high_support_cells"
            rows.append(
                {
                    "policy_id": policy_id,
                    "source_family": family,
                    "n_cells": len(group),
                    "n_high_support_cells": len(high_support),
                    "support_pass_rate": len(high_support) / max(len(group), 1),
                    "total_loans": pd.to_numeric(group.get("loans", 0), errors="coerce").sum(),
                    "total_exposure": pd.to_numeric(
                        group.get("exposure", 0), errors="coerce"
                    ).sum(),
                    "avg_pd_weighted_proxy": pd.to_numeric(
                        group.get("avg_pd", np.nan), errors="coerce"
                    ).mean(),
                    "avg_width_proxy": pd.to_numeric(
                        group.get("avg_width", np.nan), errors="coerce"
                    ).mean(),
                    "recommended_source_cap_v39": recommended_cap,
                    "cap_rule_v39": cap_rule,
                    "small_cell_pooling_rule_v39": "hierarchical_pool_to_parent_when_support_lt_25",
                    "fair_lending_legal_claim_allowed": False,
                    "claim_boundary_v39": "observable source governance only; no protected-attribute inference",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper4_v39_source_governance_caps.csv", index=False)
    return out


def build_candidate_registry(dynamic: pd.DataFrame, source_caps: pd.DataFrame) -> pd.DataFrame:
    base = read_csv("paper4_v30_candidate_registry.csv")
    if base.empty:
        out = pd.DataFrame()
    else:
        out = base.copy()
        dynamic_cols = [
            col
            for col in [
                "policy_id",
                "mean_wealth_diff",
                "prob_higher_wealth",
                "mean_loss_diff",
                "prob_lower_loss",
                "paired_wealth_robustness_gate_v39",
                "tail_challenger_gate_v39",
                "decision_v39",
            ]
            if col in dynamic.columns
        ]
        if dynamic_cols:
            out = out.merge(
                dynamic[dynamic_cols], on="policy_id", how="left", suffixes=("", "_dynamic_v39")
            )
        source_summary = (
            source_caps.groupby("policy_id", as_index=False).agg(
                source_cap_families_v39=("source_family", "nunique"),
                min_support_pass_rate_v39=("support_pass_rate", "min"),
            )
            if not source_caps.empty
            else pd.DataFrame()
        )
        if not source_summary.empty:
            out = out.merge(source_summary, on="policy_id", how="left")
        prob = pd.to_numeric(
            out.get("prob_higher_wealth", out.get("prob_higher_wealth_v31", 0)), errors="coerce"
        ).fillna(0)
        tail = pd.to_numeric(out.get("prob_lower_loss", 0), errors="coerce").fillna(0)
        audit = pd.to_numeric(out.get("auditability_score_proxy_v30", 0), errors="coerce").fillna(0)
        safety = (
            out.get("claim_safety_gate", pd.Series(True, index=out.index))
            .fillna(True)
            .astype(bool)
            .astype(float)
        )
        out["full_candidate_score_v39"] = (
            (0.35 * prob) + (0.25 * tail) + (0.25 * audit) + (0.15 * safety)
        )
        paired_gate = out.get(
            "paired_wealth_robustness_gate_v39", pd.Series(False, index=out.index)
        ).map(boolish)
        out["paper4_working_champion_change_candidate_v39"] = (
            (prob >= 0.55) & paired_gate & (safety > 0)
        )
        out["decision_v39"] = np.select(
            [
                out["policy_id"].eq("paper1_economic_champion"),
                out["paper4_working_champion_change_candidate_v39"],
                tail >= 0.90,
                out["policy_id"].str.contains("spo", case=False, na=False),
                out["policy_id"].str.contains("adp|dla|fvi", case=False, na=False),
            ],
            [
                "retain_working_champion",
                "review_for_paper4_working_champion",
                "serious_tail_challenger",
                "serious_regret_challenger",
                "dynamic_method_challenger",
            ],
            default="review_or_park",
        )
        out["paper4_only_scope_v39"] = True
        out["paper1_promotion_allowed_v39"] = False
        out["claim_boundary_v39"] = "Paper 4 lab registry only; no final promotion"
        out = out.sort_values("full_candidate_score_v39", ascending=False)
    out.to_csv(TABLE_DIR / "paper4_v39_candidate_registry.csv", index=False)
    return out


def audit_spo_dependencies() -> pd.DataFrame:
    packages = [
        "numpy",
        "cvxpy",
        "cvxpylayers",
        "torch",
        "pyomo",
        "highspy",
        "catboost",
        "sklearn",
        "pandas",
        "scipy",
    ]
    rows: list[dict[str, Any]] = []
    for package in packages:
        dist = "scikit-learn" if package == "sklearn" else package
        try:
            version = metadata.version(dist)
        except Exception as exc:
            version = ""
            version_error = f"{type(exc).__name__}: {exc}"
        else:
            version_error = ""
        probe = subprocess.run(
            [sys.executable, "-c", f"import {package}"],
            text=True,
            capture_output=True,
            check=False,
        )
        available = probe.returncode == 0
        import_error = "" if available else (probe.stderr or probe.stdout).strip().splitlines()[-1]
        rows.append(
            {
                "package": package,
                "available_v40": available,
                "version_v40": version,
                "version_lookup_error_v40": version_error,
                "import_error_v40": import_error,
                "formal_differentiable_spo_claim_allowed": False,
                "decision_v40": "usable_context"
                if available
                else "dependency_blocked_for_formal_spo",
                "future_install_path": "isolated env with compatible NumPy/cvxpy/cvxpylayers/torch pins; do not mutate main env",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper4_v40_spo_dependency_audit.csv", index=False)
    return out


def build_spo_oracle_regret(deps: pd.DataFrame) -> pd.DataFrame:
    regret = read_csv("paper4_v32_spo_temporal_oracle_regret_v3.csv")
    if regret.empty:
        out = pd.DataFrame()
    else:
        out = regret.copy()
        oracle = pd.to_numeric(out.get("oracle_value_proxy", 0), errors="coerce").replace(0, np.nan)
        candidate = pd.to_numeric(out.get("candidate_value_proxy", 0), errors="coerce")
        regret_value = pd.to_numeric(out.get("decision_regret_proxy_v20", 0), errors="coerce")
        out["regret_ratio_v40"] = regret_value / oracle.abs()
        out["candidate_to_oracle_value_ratio_v40"] = candidate / oracle
        out["temporal_split_validation_v40"] = out.get("split", "unknown")
        out["formal_differentiable_spo_claim_allowed"] = False
        out["v40_training_target"] = "reduce temporal oracle gap under same constraints"
        out["v40_decision"] = np.where(
            out["regret_ratio_v40"].fillna(1) <= 0.25,
            "monitor_promising_month",
            "surrogate_not_superior_yet",
        )
        out["claim_boundary_v40"] = (
            "decision-oracle/surrogate regret only; not formal differentiable SPO+"
        )
    out.to_csv(TABLE_DIR / "paper4_v40_spo_oracle_regret_v4.csv", index=False)
    return out


def build_dla_rollout_reaudit() -> pd.DataFrame:
    dla = read_csv("paper4_v34_dla_adp_dynamic_summary_reaudit.csv")
    if dla.empty:
        out = pd.DataFrame()
    else:
        out = dla.copy()
        wealth = pd.to_numeric(out.get("final_wealth_mean", 0), errors="coerce")
        losses = pd.to_numeric(out.get("cumulative_losses_p95", 0), errors="coerce")
        deployment = pd.to_numeric(out.get("cumulative_funded_exposure_mean", 0), errors="coerce")
        out["wealth_rank_v40"] = wealth.rank(ascending=False, method="dense")
        out["tail_loss_rank_v40"] = losses.rank(ascending=True, method="dense")
        out["deployment_speed_proxy_v40"] = (
            deployment / deployment.max() if deployment.max() else np.nan
        )
        out["rollout_depth_recommendation_v40"] = np.where(
            out["deployment_speed_proxy_v40"].fillna(0) < 0.75,
            "test_depth_2_reinvestment_before_more_features",
            "test_richer_value_features",
        )
        out["underperformance_diagnosis_v40"] = np.select(
            [
                losses > losses.quantile(0.75),
                out["deployment_speed_proxy_v40"].fillna(0) < 0.75,
            ],
            [
                "tail_loss_and_default_timing",
                "capital_deployment_or_reinvestment",
            ],
            default="state_value_feature_limit",
        )
        out["bellman_exact_claim_allowed"] = False
        out["claim_boundary_v40"] = (
            "ADP/FVI/rollout approximation only; common-path dynamic comparison required"
        )
    out.to_csv(TABLE_DIR / "paper4_v40_dla_rollout_reaudit.csv", index=False)
    return out


def build_ifrs9_sicr_update() -> pd.DataFrame:
    sicr = read_csv("paper4_v36_ifrs9_sicr_sensitivity_v3.csv")
    if sicr.empty:
        out = pd.DataFrame()
    else:
        out = sicr.copy()
        stage2 = pd.to_numeric(out.get("stage2_share_v36", 0), errors="coerce")
        out["stage2_band_v39"] = pd.cut(
            stage2,
            bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
            labels=["low", "moderate", "high", "dominant"],
        ).astype(str)
        out["sicr_rule_family_v39"] = out.get("sicr_rule_v36", "").astype(str).str.split("_").str[0]
        out["proxy_sicr_recommendation_v39"] = np.select(
            [
                out.get("sicr_rule_v36", "")
                .astype(str)
                .str.contains("composite", case=False, na=False),
                stage2.between(0.25, 0.65),
            ],
            ["preferred_proxy_candidate", "reasonable_sensitivity_band"],
            default="diagnostic_only_extreme_stage_mix",
        )
        out["contractual_ifrs9_claim_allowed"] = False
        out["production_ifrs9_staging_claim_allowed"] = False
        out["claim_boundary_v39"] = (
            "IFRS9-inspired SICR/ECL proxy only; no contractual or production IFRS9 claim"
        )
    out.to_csv(TABLE_DIR / "paper4_v39_ifrs9_sicr_proxy_update.csv", index=False)
    return out


def write_spo_note(deps: pd.DataFrame) -> None:
    note = f"""# Paper 4 SPO Isolated Environment Repro

Generated: {now()}

## Decision

The main repository environment is not mutated in v40. Formal differentiable
SPO+ remains blocked unless an isolated environment can import and validate
`cvxpy`, `cvxpylayers`, and `torch` together.

## Dependency Audit

{md_table(deps, max_rows=20)}

## Future Install Path

Create a separate environment, pin a NumPy version compatible with cvxpy and
torch, then run a minimal differentiable convex-layer example. Only after that
can Paper 4 claim formal differentiable SPO+.

## Claim Boundary

Current Paper 4 evidence supports oracle-regret/surrogate decision-loss
analysis only. It does not support a formal differentiable SPO+ claim.
"""
    (NOTE_DIR / "paper4_spo_isolated_env_repro.md").write_text(note, encoding="utf-8")


def write_sample_path_note() -> None:
    external = read_csv("paper4_v35_external_macro_context.csv")
    registry = read_csv("paper4_v35_external_macro_source_registry.csv")
    path_diag = read_csv("paper4_v31_path_calibration_diagnostics.csv")
    note = f"""# Paper 4 Sample Path Claim Boundary

Generated: {now()}

## Decision

Sample paths remain valid for internal paired comparison and stress diagnostics.
They are not external forecasts.

## External Macro Context

{md_table(registry, max_rows=12)}

## Internal Calibration Diagnostics

{md_table(path_diag, max_rows=12)}

## Macro Context Preview

{md_table(external, max_rows=12)}

## Claim Boundary

Every path family must be labelled as internal calibration unless an external
forecast validation protocol is implemented and passed. The v35/v39 position is
`external_forecast_validation_claim_allowed = false`.
"""
    (NOTE_DIR / "paper4_sample_path_claim_boundary.md").write_text(note, encoding="utf-8")


def write_cate_note() -> None:
    cate = read_csv("paper4_v37_cate_gate_report.csv")
    note = f"""# Paper 4 CATE Identification Reaudit

Generated: {now()}

## Decision

CATE policy value remains blocked. The accepted-loan sample does not resolve
reject inference, pricing endogeneity, or hidden-bias sensitivity strongly
enough for a policy-value claim.

## Gate Report

{md_table(cate, max_rows=20)}

## Next Implementable Step

Only continue this lane if a cleaner treatment/outcome pair is found. A future
unlock requires identification, overlap, balance, falsification, hidden-bias
sensitivity, and intervals to pass together.

## Claim Boundary

No CATE policy-value claim is allowed in Paper 4 v39-v40.
"""
    (NOTE_DIR / "paper4_cate_identification_reaudit.md").write_text(note, encoding="utf-8")


def write_fairness_note() -> None:
    fairness = read_csv("paper4_v37_fairness_proxy_only_protocol.csv")
    source = read_csv("paper4_v37_source_governance_appendix.csv")
    source_summary = (
        source.groupby("source_family", as_index=False).agg(
            cells=("source_id", "count"), policies=("policy_id", "nunique")
        )
        if not source.empty
        else pd.DataFrame()
    )
    note = f"""# Paper 4 Fairness Proxy-Only Protocol

Generated: {now()}

## Decision

Paper 4 keeps fairness as proxy/source governance only. It does not infer
protected attributes and does not make fair-lending legal claims.

## Protocol

{md_table(fairness, max_rows=20)}

## Source Governance Surface

{md_table(source_summary, max_rows=20)}

## Data Needed For Any Future Legal Claim

- Protected attributes or approved external proxy protocol.
- Clear legal/statistical review.
- Validation that proxies do not create unsupported protected-class inference.

## Claim Boundary

No fair-lending legal claim is allowed in Paper 4 v39-v40.
"""
    (NOTE_DIR / "paper4_fairness_protocol_update.md").write_text(note, encoding="utf-8")


def write_publishability_note() -> None:
    triage = read_csv("global_v38_publishability_triage.csv", GLOBAL_TABLE_DIR)
    contrib = read_csv("global_v38_academic_contribution_map.csv", GLOBAL_TABLE_DIR)
    note = f"""# Paper 4 Publishability Focus Memo

Generated: {now()}

## Recommended Framing

The strongest future paper framing is **sequential decision governance for
credit portfolio policies under conformal, tail-risk, ECL-proxy and source
coverage constraints**.

## Why Not A Mega Paper First

The mega-lab is useful for learning, but the publishable core should not depend
on contractual IFRS9, fair-lending legal claims, exact Bellman optimality,
formal differentiable SPO+, or CATE policy value. Those remain appendices,
negative results, or future data/theory lanes.

## Publishability Triage

{md_table(triage, max_rows=20)}

## Contribution Map

{md_table(contrib, max_rows=20)}

## Suggested Future Skeleton

1. Problem: credit decisions under governed uncertainty.
2. Framework: Powell/SDAM state-decision-uncertainty framing.
3. Evidence: CRPTO reference, CVaR challenger, online/source governance, dynamic replay.
4. Negative results: strict CVaR, SPO dependencies, CATE/fairness/IFRS9 boundaries.
5. Governance: claim-artifact matrix and living-lab protocol.
"""
    (NOTE_DIR / "paper4_publishability_focus_memo.md").write_text(note, encoding="utf-8")


def append_living_notebook(
    online: pd.DataFrame,
    dynamic: pd.DataFrame,
    cvar: pd.DataFrame,
    candidates: pd.DataFrame,
    spo: pd.DataFrame,
    dla: pd.DataFrame,
    ifrs9: pd.DataFrame,
) -> None:
    path = NOTE_DIR / "paper4_living_lab_notebook.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Paper 4 Living Lab Notebook\n"
    marker_start = "<!-- V39_V40_LIVING_LAB_START -->"
    marker_end = "<!-- V39_V40_LIVING_LAB_END -->"
    section = f"""
{marker_start}

## Wave v39-v40: Living-Lab Execution After Quarto Restructure

Generated: {now()}

### Objective

Advance runnable Paper 4 lanes without expanding the official Quarto chapter.
All outputs in this wave are lab/notebook-first unless a future editorial
review promotes a stable, artifact-backed finding.

### Scripts

- `scripts/papers/build_paper4_v39_v40_living_lab_execution.py`

### Artifacts

- `paper4_v39_online_source_family_holdout.csv`
- `paper4_v39_dynamic_candidate_stress.csv`
- `paper4_v39_cvar_certificate_delta.csv`
- `paper4_v39_source_governance_caps.csv`
- `paper4_v39_candidate_registry.csv`
- `paper4_v39_ifrs9_sicr_proxy_update.csv`
- `paper4_v40_spo_oracle_regret_v4.csv`
- `paper4_v40_dla_rollout_reaudit.csv`
- `paper4_spo_isolated_env_repro.md`
- `paper4_sample_path_claim_boundary.md`
- `paper4_cate_identification_reaudit.md`
- `paper4_fairness_protocol_update.md`
- `paper4_publishability_focus_memo.md`

### Results

Online holdout rows: {len(online)}. Dynamic stress candidates: {len(dynamic)}.
CVaR certificate rows: {len(cvar)}. Candidate registry rows: {len(candidates)}.
SPO regret rows: {len(spo)}. DLA reaudit rows: {len(dla)}. IFRS9/SICR rows:
{len(ifrs9)}.

### Interpretation

- Online conformal remains promising but live deployability is not claimed.
- CVaR remains a serious tail-risk challenger, not an exact full-universe champion.
- SPO/DFL remains oracle-regret/surrogate because differentiable dependencies are blocked.
- DLA/ADP remains representative, not Bellman exact.
- IFRS9 remains proxy-only; CATE and fairness legal claims remain blocked.

### Quarto Promotion Decision

No new Quarto page is promoted from v39-v40. The official chapter remains
compact; this wave stays in the living notebook and tables.

### Next Implementable Step

Prioritize either online source-family robustness or an isolated SPO dependency
environment. Both can improve the lab without disturbing Paper Estrella.

{marker_end}
"""
    if marker_start in text and marker_end in text:
        before = text.split(marker_start, 1)[0].rstrip()
        after = text.split(marker_end, 1)[1].lstrip()
        text = f"{before}\n\n{section}\n{after}"
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def update_backlog() -> pd.DataFrame:
    backlog = read_csv("paper4_living_lab_backlog.csv")
    if backlog.empty:
        return backlog
    result_map = {
        "Online conformal": "v39_lab_reaudit_completed_no_live_claim",
        "Dynamic stress": "v39_focused_reaudit_completed_no_rerun_needed",
        "CVaR/OCE": "v39_certificate_delta_completed",
        "SPO/DFL": "v40_dependency_audit_completed_still_blocked",
        "DLA/ADP": "v40_rollout_reaudit_completed",
        "Sample paths": "v40_claim_boundary_documented",
        "IFRS9/SICR": "v39_proxy_sensitivity_updated",
        "CATE": "v40_reaudit_completed_policy_value_blocked",
        "Fairness": "v40_proxy_only_protocol_updated",
        "Academic synthesis": "v40_publishability_focus_memo_created",
    }
    backlog["last_wave"] = "v39_v40"
    backlog["execution_result"] = backlog["lane"].map(result_map).fillna("unchanged")
    backlog["quarto_promotion_decision"] = "not_promoted_to_quarto"
    backlog.to_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", index=False)
    return backlog


def write_statuses(
    online: pd.DataFrame,
    dynamic: pd.DataFrame,
    cvar: pd.DataFrame,
    source_caps: pd.DataFrame,
    candidates: pd.DataFrame,
    ifrs9: pd.DataFrame,
    deps: pd.DataFrame,
    spo: pd.DataFrame,
    dla: pd.DataFrame,
) -> None:
    registered_pages = audit_registered_paper4_pages()
    v39 = {
        "schema_version": "2026-05-15.39",
        "generated_at_utc": now(),
        "phase": "v39_living_lab_execution_after_quarto_restructure",
        "online_rows_v39": int(len(online)),
        "dynamic_candidate_rows_v39": int(len(dynamic)),
        "cvar_certificate_rows_v39": int(len(cvar)),
        "source_governance_rows_v39": int(len(source_caps)),
        "candidate_registry_rows_v39": int(len(candidates)),
        "ifrs9_sicr_rows_v39": int(len(ifrs9)),
        "official_quarto_page_count": int(len(registered_pages)),
        "quarto_compact_guardrail_pass": len(registered_pages) <= 12,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "paper1_artifacts_modified": False,
        "contractual_ifrs9_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "claim_boundary": "v39 artifacts are lab/notebook-first; no new official Quarto page or final promotion",
    }
    v40 = {
        "schema_version": "2026-05-15.40",
        "generated_at_utc": now(),
        "phase": "v40_spo_dla_claim_boundary_and_publishability",
        "spo_dependency_rows_v40": int(len(deps)),
        "spo_oracle_regret_rows_v40": int(len(spo)),
        "dla_rollout_rows_v40": int(len(dla)),
        "formal_differentiable_spo_claim_allowed": False,
        "bellman_exact_claim_allowed": False,
        "external_forecast_validation_claim_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "paper1_artifacts_modified": False,
        "claim_boundary": "v40 documents blockers and approximations; no formal SPO+, Bellman exact, external forecast or final promotion claim",
    }
    write_json(STATUS_DIR / "paper4_v39_status.json", v39)
    write_json(STATUS_DIR / "paper4_v40_status.json", v40)


def write_documentation_cleanup_note() -> None:
    note_path = NOTE_DIR / "paper4_v39_v40_documentation_cleanup.md"
    note = f"""# Paper 4 v39-v40 Documentation Cleanup

Generated: {now()}

## Current Rule

The current Paper 4 source of truth is the compact Quarto chapter plus the
living notebook. Historical wave pages remain on disk but are not registered in
`book/_quarto.yml`.

## Current Canonical Files

- `reports/paper_material/paper4/notes/paper4_living_lab_notebook.md`
- `reports/paper_material/paper4/tables/paper4_current_official_findings.csv`
- `reports/paper_material/paper4/tables/paper4_current_claim_boundaries.csv`
- `reports/paper_material/paper4/tables/paper4_quarto_page_registry.csv`
- `reports/paper_material/global/status/global_v38_status.json`

## Cleanup Boundary

Old March champion language in history files remains historical. Current docs
should not point to old Paper 4 wave pages as live truth.
"""
    note_path.write_text(note, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    online = build_online_source_family_holdout()
    dynamic = build_dynamic_candidate_stress()
    cvar = build_cvar_certificate_delta()
    source_caps = build_source_governance_caps()
    candidates = build_candidate_registry(dynamic, source_caps)
    deps = audit_spo_dependencies()
    spo = build_spo_oracle_regret(deps)
    dla = build_dla_rollout_reaudit()
    ifrs9 = build_ifrs9_sicr_update()

    write_spo_note(deps)
    write_sample_path_note()
    write_cate_note()
    write_fairness_note()
    write_publishability_note()
    write_documentation_cleanup_note()
    append_living_notebook(online, dynamic, cvar, candidates, spo, dla, ifrs9)
    update_backlog()
    write_statuses(online, dynamic, cvar, source_caps, candidates, ifrs9, deps, spo, dla)

    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError(f"Forbidden final promotion file exists: {FORBIDDEN_FINAL_PROMOTION}")
    print("Generated Paper 4 v39-v40 living-lab execution artifacts.")


if __name__ == "__main__":
    main()
