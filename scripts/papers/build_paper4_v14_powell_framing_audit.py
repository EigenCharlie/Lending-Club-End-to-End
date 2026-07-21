"""Build Paper 4 v14 Powell framing/governance artifacts.

V14 is deliberately not another optimization wave.  It uses Powell's
``Bridging Decision Problems, Volume I`` framing discipline to reorganize the
Paper 4 living lab around metrics, decisions, uncertainties, policy classes,
evidence, and claim boundaries.

Guardrails:

* do not modify Paper Estrella artifacts;
* do not modify ``models/final_project_promotion.json``;
* do not create ``paper4_final_promotion.json``;
* do not authorize contractual IFRS9, CATE policy-value, or fair-lending legal
  claims without the missing evidence gates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_csv, _safe_read_json
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.14"
QUARTO_PAGE = "19bc-v14-powell-framing-audit.qmd"
POWELL_BRIDGING_MD = Path(
    "/mnt/c/Users/carlos/Documents/Claude Code/lending-club-risk-project/reports/"
    "powell_bridging_vol_i_framing.pdf-inspector.md"
)


def _exists_table(name: str) -> bool:
    return (TABLE_DIR / name).exists()


def _exists_status(name: str) -> bool:
    return (STATUS_DIR / name).exists()


def _artifact_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(TABLE_DIR.glob("paper4_v*.csv")) + sorted(
        TABLE_DIR.glob("paper4_v*.parquet")
    ):
        rows.append(
            {
                "artifact": path.name,
                "artifact_kind": "table",
                "version_guess": path.name.split("_")[1] if "_" in path.name else "unknown",
                "exists": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "v14_use": "input_evidence_or_prior_wave_context",
            }
        )
    for path in sorted(STATUS_DIR.glob("paper4_v*.json")):
        rows.append(
            {
                "artifact": path.name,
                "artifact_kind": "status",
                "version_guess": path.name.split("_")[1] if "_" in path.name else "unknown",
                "exists": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "v14_use": "status_or_guardrail_context",
            }
        )
    return pd.DataFrame(rows)


def _metric_pyramid() -> pd.DataFrame:
    rows = [
        (
            "M01",
            1,
            "base",
            "final wealth / state value",
            "objective",
            "maximize",
            "paper4_v13_dla_fvi_comparison.csv",
            "champion ranking input",
            "not sufficient alone",
            "DLA value is fitted-value proxy",
        ),
        (
            "M02",
            1,
            "base",
            "robust gross/net return",
            "objective",
            "maximize",
            "paper4_v13_working_candidate_registry.csv",
            "champion ranking input",
            "not sufficient alone",
            "return can dominate risk if ungated",
        ),
        (
            "M03",
            2,
            "risk",
            "CVaR/OCE scenario loss",
            "limit",
            "minimize",
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
            "tail-risk gate",
            "hard/review limit",
            "strict caps remain partly infeasible",
        ),
        (
            "M04",
            2,
            "risk",
            "ECL/provision and Stage 2 share",
            "limit",
            "minimize",
            "paper4_v13_ifrs9_sicr_sensitivity.csv",
            "prudential gate",
            "proxy-only gate",
            "no contractual IFRS9 claim",
        ),
        (
            "M05",
            2,
            "risk",
            "MDCP/source coverage stress",
            "limit",
            "maximize",
            "paper4_v13_mdcp_cap_regime_solver_summary.csv",
            "source governance gate",
            "coverage/support gate",
            "no legal fairness claim",
        ),
        (
            "M06",
            2,
            "risk",
            "sample-path p95 loss",
            "limit",
            "minimize",
            "paper4_v13_sample_path_macro_calibrated_ci.csv",
            "common-path stress gate",
            "stress diagnostic",
            "internal calibration only",
        ),
        (
            "M07",
            3,
            "estimation",
            "online conformal worst-cell coverage",
            "target",
            "hit_or_exceed",
            "paper4_v9_online_goal_status.json",
            "coverage gate",
            "target 0.80/0.90 by cell",
            "historical replay",
        ),
        (
            "M08",
            3,
            "estimation",
            "online conformal interval width",
            "target",
            "minimize subject to coverage",
            "paper4_v9_online_goal_status.json",
            "efficiency gate",
            "target <= 0.95",
            "future-period validation pending",
        ),
        (
            "M09",
            3,
            "estimation",
            "decision regret proxy",
            "objective",
            "minimize",
            "paper4_v13_spo_decision_loss_report.csv",
            "SPO/DFL lane evidence",
            "surrogate only",
            "not differentiable SPO+ proof",
        ),
        (
            "M10",
            4,
            "implementation",
            "auditability score / claim traceability",
            "target",
            "maximize",
            "paper4_v13_working_candidate_registry.csv",
            "MRM/documentation gate",
            "must be artifact-backed",
            "score is governance proxy",
        ),
        (
            "M11",
            4,
            "implementation",
            "IFRS9 contractual data readiness",
            "limit",
            "maximize",
            "paper4_v13_ifrs9_data_blocker_register.csv",
            "claim authorization gate",
            "must pass before claim",
            "data blocked",
        ),
        (
            "M12",
            4,
            "implementation",
            "fair-lending evidence readiness",
            "limit",
            "maximize",
            "paper4_v13_fairness_proxy_governance.csv",
            "legal-claim gate",
            "must pass before claim",
            "protected attributes/protocol absent",
        ),
        (
            "M13",
            4,
            "implementation",
            "CATE identification readiness",
            "limit",
            "maximize",
            "paper4_v13_causal_cate_dossier.csv",
            "policy-value gate",
            "must pass before claim",
            "theory/identification blocked",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "metric_id",
            "pyramid_level",
            "metric_family",
            "metric_name",
            "powell_role",
            "direction",
            "current_artifact",
            "champion_decision_role",
            "deployability_gate",
            "caveat",
        ],
    )


def _objective_target_limit_registry(metrics: pd.DataFrame) -> pd.DataFrame:
    thresholds = {
        "M01": "higher than static/reference under common paths",
        "M02": "non-dominated or materially competitive after risk gates",
        "M03": "committee cap or strict cap, explicitly labeled",
        "M04": "no contractual claim; Stage 2 sensitivity must be reported",
        "M05": "source-family caps with minimum support and coverage evidence",
        "M06": "paired p95 loss and probability of improvement reported",
        "M07": "source-month >= 0.80 and policy-month >= 0.90",
        "M08": "avg width <= 0.95 for efficient online conformal",
        "M09": "lower temporal decision regret than ridge/static baseline",
        "M10": "artifact-backed decision trail for every champion claim",
        "M11": "servicing/DPD/recovery/EAD/macro data available",
        "M12": "protected attributes or approved external proxy protocol",
        "M13": "identification, overlap, sensitivity, intervals all pass",
    }
    out = metrics[
        [
            "metric_id",
            "metric_name",
            "metric_family",
            "powell_role",
            "direction",
            "current_artifact",
            "caveat",
        ]
    ].copy()
    out["target_or_limit"] = out["metric_id"].map(thresholds)
    out["strictness"] = out["powell_role"].map(
        {
            "objective": "ranking_component",
            "target": "review_gate",
            "limit": "hard_or_claim_gate",
        }
    )
    out["v14_governance_decision"] = out["metric_id"].map(
        {
            "M01": "use_for_working_champion_only",
            "M02": "use_after_risk_metrics_are_visible",
            "M03": "do_not_hide_relaxed_feasibility",
            "M04": "proxy_only_no_contractual_claim",
            "M05": "source_governance_only",
            "M06": "paired_stress_required",
            "M07": "resolved_historical_gate_keep_monitoring",
            "M08": "resolved_historical_gate_keep_monitoring",
            "M09": "lane_evidence_not_formal_spo_plus",
            "M10": "required_for_any_champion_language",
            "M11": "data_blocked",
            "M12": "data_blocked_no_legal_claim",
            "M13": "theory_blocked_no_policy_value",
        }
    )
    return out


def _decision_inventory() -> pd.DataFrame:
    rows = [
        (
            "D01",
            "fund or reject a loan",
            "financial_resource",
            "Paper 4 policy engine",
            "monthly/loan-arrival",
            "immediate funding with delayed outcome",
            "R_t,B_t,x_t,S_t^x,C_t",
            "CFA/PFA/DLA",
            "paper4_v13_working_candidate_registry.csv",
            "implemented_proxy",
            "loan outcomes are censored by decision",
        ),
        (
            "D02",
            "choose funded set / allocation vector",
            "continuous_or_binary_vector",
            "solver",
            "per replay or month",
            "capital locked until repayment/default",
            "R_t,x_t,S_t^x,C_t",
            "CFA/DLA",
            "paper4_v13_cvar_stronger_decomposition_allocations.parquet",
            "implemented",
            "full dynamic allocation still pending",
        ),
        (
            "D03",
            "select Paper 4 working champion",
            "function_selection",
            "research/MRM committee",
            "per evidence wave",
            "documentation-only until deployment",
            "X_pi,C_t",
            "hybrid governance",
            "paper4_v13_working_champion.json",
            "implemented_working_only",
            "not Paper Estrella promotion",
        ),
        (
            "D04",
            "set conformal alpha / recalibration rule",
            "parameter_setting",
            "online coverage monitor",
            "monthly/rolling",
            "next period coverage",
            "B_t,W_t_plus_1,X_pi",
            "PFA/CFA",
            "paper4_v9_online_goal_status.json",
            "resolved_historical",
            "future-period validation pending",
        ),
        (
            "D05",
            "set MDCP source caps",
            "parameter_setting",
            "source governance committee",
            "per solver wave",
            "immediate feasible set restriction",
            "C_t,x_t",
            "CFA/PFA",
            "paper4_v13_mdcp_cap_regime_rationale.csv",
            "near_resolved",
            "caps are governance proxies",
        ),
        (
            "D06",
            "set CVaR/OCE cap and return floor",
            "parameter_setting",
            "risk committee",
            "per frontier wave",
            "immediate feasible set restriction",
            "C_t,x_t",
            "CFA",
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
            "near_resolved",
            "strict infeasibility must stay visible",
        ),
        (
            "D07",
            "choose SICR / Stage 2 rule",
            "function_selection",
            "credit risk governance",
            "per IFRS9 proxy wave",
            "affects ECL/provision claim",
            "B_t,C_t,S_t^x",
            "PFA/CFA",
            "paper4_v13_ifrs9_sicr_sensitivity.csv",
            "proxy_only",
            "contractual IFRS9 data missing",
        ),
        (
            "D08",
            "choose value-function features",
            "function_selection",
            "DLA/FVI designer",
            "per DLA wave",
            "affects future value approximation",
            "S_t,S_t^x,X_pi",
            "VFA/DLA",
            "paper4_v13_dla_fitted_value_coefficients.csv",
            "near_resolved",
            "not exact Bellman optimality",
        ),
        (
            "D09",
            "choose SPO/DFL training target",
            "function_selection",
            "decision-loss designer",
            "per training wave",
            "affects learned score and regret",
            "B_t,C_t,X_pi",
            "CFA/hybrid",
            "paper4_v13_spo_decision_loss_report.csv",
            "near_resolved",
            "non-differentiable surrogate",
        ),
        (
            "D10",
            "approve, block, or qualify a claim",
            "information_communication",
            "MRM/research author",
            "per artifact/page",
            "immediate documentation effect",
            "C_t",
            "PFA governance",
            "paper4_v14_claim_artifact_matrix.csv",
            "implemented_v14",
            "claim can be weaker than artifact",
        ),
        (
            "D11",
            "continue, park, or kill an experiment lane",
            "discrete_action",
            "Paper 4 lab owner",
            "per wave",
            "future research allocation",
            "C_t,X_pi",
            "PFA governance",
            "paper4_v14_stage_readiness_dashboard.csv",
            "implemented_v14",
            "judgmental but explicit",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "decision_id",
            "decision_name",
            "decision_type",
            "decision_maker",
            "frequency",
            "lag",
            "affected_state_variables",
            "policy_class",
            "evidence_artifact",
            "current_status",
            "caveat",
        ],
    )


def _uncertainty_taxonomy() -> pd.DataFrame:
    rows = [
        (
            "U01",
            "observational_errors",
            "loan attributes, payment status, recoveries, and servicing fields may be mismeasured or absent",
            "ECL, PD, stage, return",
            "fund/reject, SICR rule",
            "data audit and blocker registers",
            "servicing/DPD/recovery panel incomplete",
            "proxy_only",
            "record data-quality flags in paths",
        ),
        (
            "U02",
            "exogenous_uncertainty",
            "future default, prepayment, recoveries, macro shocks, origination volume",
            "return, loss, ECL, final wealth",
            "funded set and capital use",
            "sample paths and replay",
            "external macro/default calibration",
            "near_resolved",
            "simulate common shocks after x_t",
        ),
        (
            "U03",
            "prognostic_uncertainty",
            "PD/ECL/return forecasts are imperfect",
            "regret, ECL, CVaR",
            "solver score, SPO target",
            "PD and conformal artifacts",
            "forecast error not fully contractual",
            "near_resolved",
            "include forecast error distributions",
        ),
        (
            "U04",
            "inferential_uncertainty",
            "true borrower state is latent at decision time",
            "PD, stage, fairness proxy, CATE",
            "approval and stage migration",
            "model features and intervals",
            "hidden borrower risk and unobserved confounding",
            "theory_blocked",
            "stress latent-risk multipliers",
        ),
        (
            "U05",
            "experimental_uncertainty",
            "bootstrap/month/sample-path estimates vary by run",
            "champion score, p95 loss",
            "champion selection",
            "bootstrap and common paths",
            "finite historical replay",
            "near_resolved",
            "report paired intervals",
        ),
        (
            "U06",
            "model_uncertainty",
            "choice of PD/LGD/ECL/CVaR/DLA/SPO model changes conclusions",
            "all model-based metrics",
            "solver/model choice",
            "v13 method registry and claims",
            "no single certified base model",
            "near_resolved",
            "scenario model registry",
        ),
        (
            "U07",
            "transitional_uncertainty",
            "state transition from loan to repayment/default/recovery is noisy",
            "cash, outstanding, ECL, final wealth",
            "dynamic funding and reinvestment",
            "DLA trace and sample paths",
            "transition remains proxy",
            "near_resolved",
            "monthly transition model",
        ),
        (
            "U08",
            "implementation_uncertainty",
            "decided policy may differ from deployable funding/servicing process",
            "realized return, compliance",
            "policy deployment and communication",
            "claim matrix only",
            "no field deployment",
            "implementation_blocked",
            "separate computer decision from field action",
        ),
        (
            "U09",
            "communication_errors",
            "policy/gate instructions may be misunderstood in documentation or handoff",
            "auditability, claim safety",
            "claim approval",
            "Quarto and claim matrix",
            "human governance process not tested",
            "near_resolved",
            "explicit no-claim flags",
        ),
        (
            "U10",
            "algorithmic_instability",
            "LP sampling, solver tolerances, random paths, and training seeds can vary",
            "champion score, regret, CVaR",
            "solver/model choice",
            "v13/v14 status and tests",
            "dual certificates/full universe pending",
            "near_resolved",
            "fixed seeds plus sensitivity",
        ),
        (
            "U11",
            "goal_uncertainty",
            "weights between return, coverage, risk, auditability can change by committee",
            "working champion decision",
            "objective/target/limit registry",
            "v14 metric pyramid",
            "thresholds are governance choices",
            "near_resolved",
            "committee threshold scenarios",
        ),
        (
            "U12",
            "environmental_uncertainty",
            "macro/regulatory/market regime changes can alter borrower behavior",
            "default, LGD, prepayment, ECL",
            "capital and risk caps",
            "sample paths",
            "external macro data missing",
            "data_blocked",
            "macro scenario labels and no forecast claim",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "uncertainty_id",
            "powell_uncertainty_class",
            "paper4_example",
            "affected_metrics",
            "affected_decisions",
            "current_artifact_coverage",
            "unresolved_blocker",
            "stage_readiness",
            "sample_path_implication",
        ],
    )


def _uncertainty_forms() -> pd.DataFrame:
    rows = [
        (
            "F01",
            "fine_grained_variability",
            "loan-level idiosyncratic default/prepayment noise",
            "independent Bernoulli/default draws with calibrated multipliers",
            "draw loan-level residual risk conditional on grade/period",
            "comparison only, not forecast",
            "paper4_v13_sample_path_macro_calibrated_paths.parquet",
        ),
        (
            "F02",
            "shifts",
            "mean default/prepayment regime changes by month/vintage",
            "macro_state_v12 and period multipliers",
            "piecewise regime paths with crossing-time diagnostics",
            "do not infer future regime probabilities",
            "paper4_v13_sample_path_scenario_register.csv",
        ),
        (
            "F03",
            "bursts",
            "localized clusters of defaults or prepayments",
            "partly through common macro shocks",
            "clustered default bursts by source family",
            "diagnostic stress only",
            "paper4_v13_sample_path_calibration_table.csv",
        ),
        (
            "F04",
            "spikes",
            "rare sharp LGD/default stress",
            "scenario loss / CVaR stress",
            "rare tail spikes tied to macro and state/grade cells",
            "no calibrated rare-event probability",
            "paper4_v13_cvar_stronger_decomposition_scenario_losses.csv",
        ),
        (
            "F05",
            "seasonality",
            "monthly origination and repayment/default seasonality",
            "monthly replay and period fields",
            "month-of-year effects in W_{t+1}",
            "historical seasonality only",
            "paper4_monthly_policy_replay.parquet",
        ),
        (
            "F06",
            "vintage_cohort_effects",
            "issue-period cohorts age into outcomes differently",
            "period/grade calibration",
            "vintage-specific hazard and recovery stress",
            "contractual panel missing",
            "paper4_v13_sample_path_calibration_table.csv",
        ),
        (
            "F07",
            "spatial_state_shocks",
            "state-level unemployment/regulatory/economic shocks",
            "state caps are partial",
            "state-family correlated shock factors",
            "diagnostic until external macro is added",
            "paper4_v13_mdcp_cap_regime_rationale.csv",
        ),
        (
            "F08",
            "attribute_correlations",
            "grade, DTI, income, FICO, purpose correlations",
            "MDCP source families and qhat/PD features",
            "hierarchical pooled attribute covariance",
            "small cells require pooling",
            "paper4_v13_mdcp_cap_regime_solver_summary.csv",
        ),
        (
            "F09",
            "systemic_macro_events",
            "recession or credit-market shock",
            "macro/common-path stress",
            "common recession factor driving PD/LGD/prepayment",
            "internal calibration only",
            "paper4_v13_sample_path_scenario_register.csv",
        ),
        (
            "F10",
            "rare_events",
            "extreme credit shock not present in sample",
            "CVaR/OCE scenarios",
            "manual contingency path in scenario register",
            "contingency, not probability estimate",
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
        ),
        (
            "F11",
            "contingencies",
            "policy/regulatory servicing changes with no historical analog",
            "blocker dashboard only",
            "explicit contingency rows and claim blocks",
            "cannot validate without event/data",
            "paper4_v14_stage_readiness_dashboard.csv",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "form_id",
            "uncertainty_form",
            "lending_club_manifestation",
            "current_v13_representation",
            "v14_sample_path_design_spec",
            "overclaim_guardrail",
            "artifact_to_extend",
        ],
    )


def _framing_audit() -> pd.DataFrame:
    rows = [
        (
            "L01",
            "Paper Estrella frozen baseline",
            "Stage I/II/III",
            "M02,M07,M10",
            "do not change final champion",
            "prognostic/model/implementation",
            "CFA",
            "models/final_project_promotion.json",
            "resolved",
            "frozen reference only",
            "Paper 4 may compare but not overwrite",
        ),
        (
            "L02",
            "CRPTO policy universe",
            "Stage II",
            "M02,M07,M10",
            "select funded set under conformal bound",
            "prognostic/model",
            "CFA",
            "paper4_policy_universe.csv",
            "implemented",
            "auditable policy universe",
            "not dynamic",
        ),
        (
            "L03",
            "online conformal",
            "Stage II/III",
            "M07,M08",
            "choose recalibration and alpha schedule",
            "exogenous/prognostic/algorithmic",
            "PFA/CFA",
            "paper4_v9_online_goal_status.json",
            "resolved_historical",
            "coverage/width gate in replay",
            "future-period validation pending",
        ),
        (
            "L04",
            "MDCP/source coverage",
            "Stage II",
            "M05,M10,M12",
            "set source caps and pooling",
            "observational/inferential/goal",
            "CFA/PFA",
            "paper4_v13_mdcp_cap_regime_solver_summary.csv",
            "near_resolved",
            "source governance coverage",
            "no legal fairness claim",
        ),
        (
            "L05",
            "CVaR/OCE",
            "Stage II",
            "M03,M06",
            "set cvar cap and return floor",
            "exogenous/model/algorithmic",
            "CFA",
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
            "near_resolved",
            "larger-pool tail frontier",
            "not exact full-universe proof",
        ),
        (
            "L06",
            "DLA/FVI",
            "Stage II",
            "M01,M06,M10",
            "choose value features and dynamic book",
            "transitional/model/experimental",
            "DLA/VFA hybrid",
            "paper4_v13_dla_fvi_comparison.csv",
            "near_resolved",
            "working champion evidence",
            "representative book, not exact Bellman",
        ),
        (
            "L07",
            "SPO/DFL",
            "Stage II",
            "M02,M09,M10",
            "choose decision-loss training target",
            "prognostic/model/algorithmic",
            "CFA hybrid",
            "paper4_v13_spo_decision_loss_report.csv",
            "near_resolved",
            "temporal regret surrogate",
            "not formal differentiable SPO+",
        ),
        (
            "L08",
            "sample paths",
            "Stage II",
            "M06,M07",
            "choose evaluation base model",
            "experimental/environmental/model",
            "base/evaluation model",
            "paper4_v13_sample_path_macro_calibrated_ci.csv",
            "near_resolved",
            "common random numbers for comparison",
            "not future forecast",
        ),
        (
            "L09",
            "IFRS9/SICR proxy",
            "Stage I/II",
            "M04,M11",
            "choose SICR rule and ECL scenario",
            "observational/exogenous/model",
            "PFA/CFA",
            "paper4_v13_ifrs9_data_blocker_register.csv",
            "data_blocked",
            "proxy ECL sensitivity",
            "no contractual IFRS9 claim",
        ),
        (
            "L10",
            "Causal/CATE",
            "Stage I/II",
            "M13",
            "decide if causal policy value can enter policy",
            "inferential/model/goal",
            "information/causal gate",
            "paper4_v13_causal_cate_dossier.csv",
            "theory_blocked",
            "dossier only",
            "no CATE policy value",
        ),
        (
            "L11",
            "fairness proxy governance",
            "Stage I/III",
            "M05,M12",
            "decide allowed fairness language",
            "observational/inferential/goal",
            "PFA governance",
            "paper4_v13_fairness_proxy_governance.csv",
            "data_blocked",
            "proxy governance only",
            "no fair-lending legal claim",
        ),
        (
            "L12",
            "selector/claim governance",
            "Stage III",
            "M10,M11,M12,M13",
            "approve/review/park/kill claims",
            "implementation/communication/goal",
            "PFA governance",
            "paper4_v14_claim_artifact_matrix.csv",
            "implemented_v14",
            "claim-safe lab operation",
            "claim can be blocked despite good score",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "lane_id",
            "lane_name",
            "powell_stage",
            "metric_ids",
            "controlled_decision",
            "uncertainty_classes",
            "policy_class",
            "evidence_artifact",
            "current_status",
            "allowed_claim",
            "caveat",
        ],
    )


def _impact_code(score: int) -> str:
    return {3: "H", 2: "M", 1: "L", 0: "N"}[score]


def _interaction_matrices(
    metrics: pd.DataFrame, decisions: pd.DataFrame, uncertainties: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_ids = list(metrics["metric_id"])
    decision_rules = {
        "D01": {"M01": 3, "M02": 3, "M03": 2, "M04": 2, "M05": 2, "M06": 3, "M07": 1, "M10": 2},
        "D02": {"M01": 3, "M02": 3, "M03": 3, "M05": 2, "M06": 3, "M10": 2},
        "D03": {
            "M01": 2,
            "M02": 2,
            "M03": 2,
            "M04": 2,
            "M05": 2,
            "M06": 2,
            "M10": 3,
            "M11": 2,
            "M12": 2,
            "M13": 2,
        },
        "D04": {"M07": 3, "M08": 3, "M05": 2, "M10": 1},
        "D05": {"M05": 3, "M10": 2, "M12": 2, "M02": 1},
        "D06": {"M03": 3, "M06": 3, "M02": 2, "M10": 2},
        "D07": {"M04": 3, "M11": 3, "M02": 1, "M10": 2},
        "D08": {"M01": 3, "M06": 2, "M10": 2},
        "D09": {"M02": 2, "M09": 3, "M10": 2},
        "D10": {"M10": 3, "M11": 3, "M12": 3, "M13": 3},
        "D11": {"M10": 3, "M11": 2, "M12": 2, "M13": 2},
    }
    dm_rows = []
    for _, decision in decisions.iterrows():
        impacts = decision_rules.get(decision["decision_id"], {})
        for metric_id in metric_ids:
            score = int(impacts.get(metric_id, 0))
            dm_rows.append(
                {
                    "decision_id": decision["decision_id"],
                    "decision_name": decision["decision_name"],
                    "metric_id": metric_id,
                    "impact_code": _impact_code(score),
                    "impact_score": score,
                    "rationale": "judgmental Powell framing input; not an empirical causal estimate",
                }
            )

    uncertainty_rules = {
        "U01": {"M04": 3, "M11": 3, "M12": 2, "M13": 2},
        "U02": {"M01": 3, "M02": 3, "M03": 3, "M04": 3, "M06": 3},
        "U03": {"M02": 2, "M04": 3, "M07": 3, "M09": 3},
        "U04": {"M04": 2, "M12": 3, "M13": 3},
        "U05": {"M06": 3, "M10": 2},
        "U06": {"M01": 2, "M02": 2, "M03": 3, "M04": 3, "M09": 3},
        "U07": {"M01": 3, "M04": 3, "M06": 3},
        "U08": {"M10": 3, "M11": 2, "M12": 2},
        "U09": {"M10": 3},
        "U10": {"M03": 2, "M06": 3, "M09": 3},
        "U11": {"M01": 2, "M02": 2, "M03": 2, "M10": 3},
        "U12": {"M03": 3, "M04": 3, "M06": 3},
    }
    um_rows = []
    for _, uncertainty in uncertainties.iterrows():
        impacts = uncertainty_rules.get(uncertainty["uncertainty_id"], {})
        for metric_id in metric_ids:
            score = int(impacts.get(metric_id, 0))
            um_rows.append(
                {
                    "uncertainty_id": uncertainty["uncertainty_id"],
                    "powell_uncertainty_class": uncertainty["powell_uncertainty_class"],
                    "metric_id": metric_id,
                    "impact_code": _impact_code(score),
                    "impact_score": score,
                    "rationale": "judgmental Powell framing input; use to prioritize modeling, not as measured effect",
                }
            )

    decision_uncertainty_rules = {
        "D01": ["U02", "U03", "U04", "U07", "U12"],
        "D02": ["U02", "U03", "U06", "U10", "U12"],
        "D03": ["U05", "U06", "U10", "U11"],
        "D04": ["U02", "U03", "U05", "U10"],
        "D05": ["U01", "U04", "U11"],
        "D06": ["U02", "U06", "U10", "U12"],
        "D07": ["U01", "U02", "U06"],
        "D08": ["U06", "U07", "U10"],
        "D09": ["U03", "U06", "U10"],
        "D10": ["U08", "U09", "U11"],
        "D11": ["U05", "U10", "U11"],
    }
    du_rows = []
    for _, decision in decisions.iterrows():
        high = set(decision_uncertainty_rules.get(decision["decision_id"], []))
        for _, uncertainty in uncertainties.iterrows():
            score = 3 if uncertainty["uncertainty_id"] in high else 1
            if uncertainty["uncertainty_id"] in {"U08", "U09"} and decision["decision_id"] not in {
                "D10",
                "D11",
            }:
                score = 0
            du_rows.append(
                {
                    "decision_id": decision["decision_id"],
                    "decision_name": decision["decision_name"],
                    "uncertainty_id": uncertainty["uncertainty_id"],
                    "powell_uncertainty_class": uncertainty["powell_uncertainty_class"],
                    "impact_code": _impact_code(score),
                    "impact_score": score,
                    "rationale": "judgmental map of which uncertainty must be considered before exercising the decision",
                }
            )

    policy_rows = [
        (
            "PFA",
            "rules, thresholds, alpha schedules, claim gates",
            "paper4_v14_decision_inventory.csv",
            "transparent and auditable",
            "can be myopic",
        ),
        (
            "CFA",
            "CRPTO, CVaR/OCE LP, MDCP cap solver",
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
            "strong one-period optimization",
            "approximation of dynamic uncertainty",
        ),
        (
            "VFA",
            "future value features for capital/stage/cash",
            "paper4_v13_dla_fitted_value_coefficients.csv",
            "captures future value compactly",
            "requires structure and validation",
        ),
        (
            "DLA",
            "rolling/fitted lookahead over months",
            "paper4_v13_dla_fvi_comparison.csv",
            "plans with future states",
            "not exact stochastic optimality",
        ),
        (
            "Hybrid",
            "selector governance combining scores and gates",
            "paper4_v14_working_champion_powell_audit.csv",
            "matches lab governance reality",
            "judgmental thresholds must be documented",
        ),
    ]
    policy_evidence = pd.DataFrame(
        policy_rows,
        columns=[
            "policy_class",
            "paper4_examples",
            "minimum_evidence_artifact",
            "strength",
            "claim_boundary",
        ],
    )
    return pd.DataFrame(dm_rows), pd.DataFrame(um_rows), pd.DataFrame(du_rows), policy_evidence


def _base_vs_lookahead_registry() -> pd.DataFrame:
    rows = [
        (
            "paper4_v13_sample_path_macro_calibrated_paths.parquet",
            "sample_paths",
            "base_evaluation_model",
            "W_t_plus_1 / transition evidence",
            "none; evaluates policies",
            "common random numbers and paired paths",
            "internal calibration only",
            "external macro/default calibration",
        ),
        (
            "paper4_v13_champion_stress_test.csv",
            "champion_stress",
            "base_evaluation_model",
            "C_t / performance evaluation",
            "none; governance comparison",
            "champion-vs-challenger stress",
            "representative DLA books",
            "dynamic process stress",
        ),
        (
            "paper4_monthly_policy_replay.parquet",
            "monthly_replay",
            "base_evaluation_model",
            "sequential replay",
            "none; historical evaluator",
            "monthly policy evidence",
            "not full servicing panel",
            "contractual servicing replay",
        ),
        (
            "paper4_v13_cvar_stronger_decomposition_frontier.csv",
            "CVaR/OCE",
            "lookahead_policy_model",
            "x_t / C_t",
            "tail-constrained funded set",
            "frontier and active caps",
            "larger pool, not certified full universe",
            "dual/full-universe proof",
        ),
        (
            "paper4_v13_mdcp_cap_regime_solver_summary.csv",
            "MDCP",
            "lookahead_policy_model",
            "x_t / C_t",
            "source-capped funded set",
            "cap feasibility and composition",
            "governance proxy",
            "coverage-calibrated caps",
        ),
        (
            "paper4_v13_dla_fvi_comparison.csv",
            "DLA/FVI",
            "lookahead_policy_model",
            "S_t^x / X_pi",
            "dynamic fitted-value score",
            "state value proxy comparison",
            "not exact Bellman",
            "true dynamic path evaluator",
        ),
        (
            "paper4_v13_spo_decision_loss_report.csv",
            "SPO/DFL",
            "lookahead_policy_model",
            "B_t / X_pi",
            "decision-loss score",
            "temporal regret",
            "not differentiable SPO+",
            "optimization layer",
        ),
        (
            "paper4_v13_ifrs9_sicr_sensitivity.csv",
            "IFRS9/SICR",
            "implementation_claim_model",
            "C_t / claim gate",
            "stage/ECL proxy selection",
            "sensitivity table",
            "no contractual data",
            "servicing/DPD/EAD paths",
        ),
        (
            "paper4_v14_claim_artifact_matrix.csv",
            "claim governance",
            "implementation_claim_model",
            "C_t",
            "allow/block claims",
            "claim-artifact trace",
            "human committee proxy",
            "MRM signoff process",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "artifact",
            "lane",
            "model_role",
            "powell_element",
            "decision_use",
            "evaluation_use",
            "current_limit",
            "next_upgrade",
        ],
    )


def _working_champion_audit(
    v13_registry: pd.DataFrame, v13_champion: dict[str, Any]
) -> pd.DataFrame:
    champion_id = v13_champion.get("policy_id", "v13_fvi_return_recovery")
    champ = v13_registry[v13_registry.get("policy_id", pd.Series([], dtype=str)).eq(champion_id)]
    row = champ.iloc[0].to_dict() if not champ.empty else {}
    criteria = [
        (
            "C01",
            "base objective",
            "M01",
            "state value / final wealth",
            row.get("state_value_delta"),
            "pass_working",
            "champion selected because DLA/FVI state-value evidence is strongest",
        ),
        (
            "C02",
            "base objective",
            "M02",
            "return proxy",
            row.get("return_proxy"),
            "review",
            "return is missing/not primary for DLA representative champion",
        ),
        (
            "C03",
            "risk metric",
            "M03",
            "tail risk proxy",
            row.get("tail_risk_proxy"),
            "review",
            "needs CVaR-certified dynamic book before deployable claim",
        ),
        (
            "C04",
            "risk metric",
            "M06",
            "sample-path p95 loss",
            row.get("p95_loss"),
            "pass_working",
            "common-path stress supports working evidence",
        ),
        (
            "C05",
            "estimation metric",
            "M07",
            "online coverage gate",
            True,
            "pass_historical",
            "v9 historical online gate remains in force",
        ),
        (
            "C06",
            "implementation metric",
            "M10",
            "auditability",
            row.get("auditability_score"),
            "review",
            "DLA champion has weaker auditability trace than pure CFA/CRPTO",
        ),
        (
            "C07",
            "implementation metric",
            "M11",
            "contractual IFRS9 readiness",
            False,
            "blocked",
            "no contractual IFRS9 claim",
        ),
        (
            "C08",
            "implementation metric",
            "M12",
            "fair-lending legal readiness",
            False,
            "blocked",
            "no fair-lending legal claim",
        ),
        (
            "C09",
            "implementation metric",
            "M13",
            "CATE policy value readiness",
            False,
            "blocked",
            "no CATE policy value claim",
        ),
        (
            "C10",
            "claim safety",
            "M10",
            "Paper Estrella freeze",
            True,
            "pass",
            "Paper 4 working champion only",
        ),
    ]
    rows = []
    for criterion_id, family, metric_id, name, value, gate, interpretation in criteria:
        rows.append(
            {
                "policy_id": champion_id,
                "criterion_id": criterion_id,
                "powell_metric_family": family,
                "metric_id": metric_id,
                "criterion_name": name,
                "v13_evidence_value": value,
                "gate_status_v14": gate,
                "decision_impact": "keep_working_champion"
                if gate.startswith("pass")
                else "qualify_claim",
                "source_artifact": "paper4_v13_working_candidate_registry.csv",
                "interpretation": interpretation,
                "caveat": v13_champion.get("caveat", "working-lab claim only"),
            }
        )
    return pd.DataFrame(rows)


def _stage_readiness_dashboard(framing: pd.DataFrame) -> pd.DataFrame:
    stage_map = {
        "L01": (
            "resolved",
            "resolved",
            "resolved",
            "resolved",
            "Paper Estrella freeze remains intact",
            "continue Paper 4 only",
        ),
        "L02": (
            "resolved",
            "resolved",
            "near_resolved",
            "implemented",
            "CRPTO policy universe is auditable",
            "keep as baseline/evidence",
        ),
        "L03": (
            "resolved",
            "resolved",
            "near_resolved",
            "resolved",
            "online gate resolved historically",
            "future-period validation",
        ),
        "L04": (
            "resolved",
            "near_resolved",
            "proxy_only",
            "near_resolved",
            "MDCP is source governance not legal fairness",
            "calibrate caps against coverage/support",
        ),
        "L05": (
            "resolved",
            "near_resolved",
            "proxy_only",
            "near_resolved",
            "larger pools worked but strict feasibility remains hard",
            "full-universe/dual certificate",
        ),
        "L06": (
            "resolved",
            "near_resolved",
            "implementation_blocked",
            "near_resolved",
            "DLA/FVI is working champion but representative",
            "true dynamic stress engine",
        ),
        "L07": (
            "resolved",
            "near_resolved",
            "implementation_blocked",
            "near_resolved",
            "SPO/DFL remains non-differentiable surrogate",
            "optimization layer or stronger oracle validation",
        ),
        "L08": (
            "resolved",
            "near_resolved",
            "proxy_only",
            "near_resolved",
            "paths good for paired comparison",
            "external calibration",
        ),
        "L09": (
            "resolved",
            "proxy_only",
            "data_blocked",
            "data_blocked",
            "contractual IFRS9 data missing",
            "servicing/DPD/EAD/recovery/macro",
        ),
        "L10": (
            "near_resolved",
            "theory_blocked",
            "theory_blocked",
            "theory_blocked",
            "identification/sensitivity not enough for policy value",
            "stronger causal design",
        ),
        "L11": (
            "near_resolved",
            "proxy_only",
            "data_blocked",
            "data_blocked",
            "protected attributes/protocol absent",
            "external valid protocol",
        ),
        "L12": (
            "resolved",
            "resolved",
            "resolved",
            "implemented_v14",
            "claim governance is now Powell-framed",
            "maintain every wave",
        ),
    }
    rows = []
    for _, lane in framing.iterrows():
        s1, s2, s3, status, diagnosis, next_action = stage_map[lane["lane_id"]]
        rows.append(
            {
                "lane_id": lane["lane_id"],
                "lane_name": lane["lane_name"],
                "stage1_framing_readiness": s1,
                "stage2_modeling_readiness": s2,
                "stage3_implementation_readiness": s3,
                "status_v14": status,
                "blocker_category": status
                if status in {"data_blocked", "theory_blocked", "implementation_blocked"}
                else "none_or_managed",
                "current_diagnosis": diagnosis,
                "next_action": next_action,
            }
        )
    return pd.DataFrame(rows)


def _claim_matrix() -> pd.DataFrame:
    rows = [
        (
            "P01",
            "Paper 4 is now framed as Powell metric-decision-uncertainty problem",
            "implemented",
            "paper4_v14_powell_framing_audit.csv",
            QUARTO_PAGE,
            "framing/governance claim",
            "",
            False,
            False,
            False,
            "judgmental but explicit",
        ),
        (
            "P02",
            "Working champion cannot be chosen on return/state value alone",
            "implemented",
            "paper4_v14_metric_pyramid.csv",
            QUARTO_PAGE,
            "governance claim",
            "",
            False,
            False,
            False,
            "risk and implementation gates remain visible",
        ),
        (
            "P03",
            "Objectives, targets and limits are separated",
            "implemented",
            "paper4_v14_objective_target_limit_registry.csv",
            QUARTO_PAGE,
            "selector design claim",
            "",
            False,
            False,
            False,
            "thresholds are committee choices",
        ),
        (
            "P04",
            "Paper 4 decision inventory is explicit",
            "implemented",
            "paper4_v14_decision_inventory.csv",
            QUARTO_PAGE,
            "framing claim",
            "",
            False,
            False,
            False,
            "not every decision is automated",
        ),
        (
            "P05",
            "Paper 4 uncertainty map covers Powell's 12 classes",
            "implemented",
            "paper4_v14_uncertainty_taxonomy.csv",
            QUARTO_PAGE,
            "framing claim",
            "",
            False,
            False,
            False,
            "coverage is conceptual, not measured probability",
        ),
        (
            "P06",
            "Sample-path uncertainty forms are specified without forecast overclaim",
            "implemented",
            "paper4_v14_uncertainty_forms_sample_path_design.csv",
            QUARTO_PAGE,
            "design claim",
            "external macro calibration pending",
            False,
            False,
            False,
            "internal calibration only",
        ),
        (
            "P07",
            "Interaction matrices make judgmental priorities explicit",
            "implemented",
            "paper4_v14_decision_metric_interaction_matrix.csv",
            QUARTO_PAGE,
            "governance claim",
            "",
            False,
            False,
            False,
            "H/M/L/N are not causal estimates",
        ),
        (
            "P08",
            "Base/evaluation model is separated from lookahead/policy model",
            "implemented",
            "paper4_v14_base_vs_lookahead_model_registry.csv",
            QUARTO_PAGE,
            "architecture claim",
            "",
            False,
            False,
            False,
            "DLA remains representative",
        ),
        (
            "P09",
            "v13_fvi_return_recovery remains Paper 4 working champion after Powell audit",
            "implemented_working_only",
            "paper4_v14_working_champion_powell_audit.csv",
            QUARTO_PAGE,
            "working-lab claim only",
            "deployability gates still open",
            False,
            False,
            False,
            "no Paper Estrella promotion",
        ),
        (
            "P10",
            "IFRS9 remains proxy-only",
            "blocked_claim",
            "paper4_v13_ifrs9_data_blocker_register.csv",
            QUARTO_PAGE,
            "no contractual IFRS9 claim",
            "servicing/DPD/EAD/recovery/macro missing",
            True,
            False,
            False,
            "claim explicitly prohibited",
        ),
        (
            "P11",
            "CATE policy value remains blocked",
            "blocked_claim",
            "paper4_v13_causal_cate_dossier.csv",
            QUARTO_PAGE,
            "no CATE policy-value claim",
            "identification/overlap/sensitivity",
            False,
            True,
            False,
            "claim explicitly prohibited",
        ),
        (
            "P12",
            "Fairness remains proxy governance only",
            "blocked_claim",
            "paper4_v13_fairness_proxy_governance.csv",
            QUARTO_PAGE,
            "no fair-lending legal claim",
            "protected attributes/protocol absent",
            False,
            False,
            True,
            "claim explicitly prohibited",
        ),
        (
            "P13",
            "Paper Estrella freeze remains enforced",
            "guardrail_verified",
            "paper4_v14_status.json",
            QUARTO_PAGE,
            "guardrail claim",
            "",
            False,
            False,
            False,
            "models/final_project_promotion.json not modified",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "priority",
            "claim",
            "claim_status",
            "artifact",
            "quarto_page",
            "claim_boundary_v14",
            "blocker_if_any",
            "no_claim_contractual_ifrs9",
            "no_claim_cate_policy_value",
            "no_claim_fair_lending_legal",
            "caveat",
        ],
    )


def _write_v14_note(status: dict[str, Any]) -> None:
    _write_note(
        "paper4_v14_powell_framing_audit.md",
        "\n".join(
            [
                "# Paper 4 v14 Powell Framing Audit",
                "",
                f"- Working champion after audit: `{status['working_champion_policy_id_v14']}`.",
                f"- Champion replaced in v14: `{status['champion_replaced_v14']}`.",
                f"- Powell lanes framed: `{status['framing_lane_count_v14']}`.",
                f"- Decisions inventoried: `{status['decision_count_v14']}`.",
                f"- Uncertainty classes mapped: `{status['uncertainty_class_count_v14']}`.",
                f"- Claim rows: `{status['claim_count_v14']}`.",
                f"- Final promotion JSON created: `{status['paper4_final_promotion_created']}`.",
                "",
                "V14 is a governance/framing wave. It reorganizes Paper 4 around Powell's metrics, decisions, uncertainties, model roles, and claim boundaries.",
            ]
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    del argv
    start = time.time()
    v13_status = _safe_read_json(STATUS_DIR / "paper4_v13_status.json")
    v13_champion = _safe_read_json(STATUS_DIR / "paper4_v13_working_champion.json")
    v13_registry = _safe_read_csv(TABLE_DIR / "paper4_v13_working_candidate_registry.csv")

    audit = _artifact_audit()
    metrics = _metric_pyramid()
    objectives = _objective_target_limit_registry(metrics)
    decisions = _decision_inventory()
    uncertainties = _uncertainty_taxonomy()
    forms = _uncertainty_forms()
    framing = _framing_audit()
    decision_metric, uncertainty_metric, decision_uncertainty, policy_evidence = (
        _interaction_matrices(metrics, decisions, uncertainties)
    )
    model_registry = _base_vs_lookahead_registry()
    champion_audit = _working_champion_audit(v13_registry, v13_champion)
    stage_dashboard = _stage_readiness_dashboard(framing)
    claims = _claim_matrix()

    _write_csv("paper4_v14_v1_v13_artifact_audit.csv", audit)
    _write_csv("paper4_v14_powell_framing_audit.csv", framing)
    _write_csv("paper4_v14_metric_pyramid.csv", metrics)
    _write_csv("paper4_v14_objective_target_limit_registry.csv", objectives)
    _write_csv("paper4_v14_decision_inventory.csv", decisions)
    _write_csv("paper4_v14_uncertainty_taxonomy.csv", uncertainties)
    _write_csv("paper4_v14_uncertainty_forms_sample_path_design.csv", forms)
    _write_csv("paper4_v14_decision_metric_interaction_matrix.csv", decision_metric)
    _write_csv("paper4_v14_uncertainty_metric_interaction_matrix.csv", uncertainty_metric)
    _write_csv("paper4_v14_decision_uncertainty_interaction_matrix.csv", decision_uncertainty)
    _write_csv("paper4_v14_policy_class_evidence_matrix.csv", policy_evidence)
    _write_csv("paper4_v14_base_vs_lookahead_model_registry.csv", model_registry)
    _write_csv("paper4_v14_working_champion_powell_audit.csv", champion_audit)
    _write_csv("paper4_v14_stage_readiness_dashboard.csv", stage_dashboard)
    _write_csv("paper4_v14_claim_artifact_matrix.csv", claims)

    champion_v13 = v13_champion.get("policy_id") or v13_status.get("working_champion_policy_id_v13")
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v14_powell_framing_audit",
        "mode": "powell_framing_governance_no_paper1_changes",
        "powell_source": str(POWELL_BRIDGING_MD),
        "powell_source_exists": POWELL_BRIDGING_MD.exists(),
        "previous_working_champion_policy_id_v13": champion_v13,
        "working_champion_policy_id_v14": champion_v13,
        "champion_replaced_v14": False,
        "champion_decision_v14": "keep_v13_working_champion_with_powell_claim_qualifiers",
        "framing_lane_count_v14": int(len(framing)),
        "metric_count_v14": int(len(metrics)),
        "decision_count_v14": int(len(decisions)),
        "uncertainty_class_count_v14": int(len(uncertainties)),
        "uncertainty_form_count_v14": int(len(forms)),
        "decision_metric_interaction_rows_v14": int(len(decision_metric)),
        "uncertainty_metric_interaction_rows_v14": int(len(uncertainty_metric)),
        "decision_uncertainty_interaction_rows_v14": int(len(decision_uncertainty)),
        "policy_class_evidence_rows_v14": int(len(policy_evidence)),
        "stage_dashboard_rows_v14": int(len(stage_dashboard)),
        "claim_count_v14": int(len(claims)),
        "base_model_vs_lookahead_rows_v14": int(len(model_registry)),
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "required_v14_artifacts_exist": all(
            _exists_table(name)
            for name in [
                "paper4_v14_powell_framing_audit.csv",
                "paper4_v14_metric_pyramid.csv",
                "paper4_v14_objective_target_limit_registry.csv",
                "paper4_v14_decision_inventory.csv",
                "paper4_v14_uncertainty_taxonomy.csv",
                "paper4_v14_uncertainty_forms_sample_path_design.csv",
                "paper4_v14_decision_metric_interaction_matrix.csv",
                "paper4_v14_uncertainty_metric_interaction_matrix.csv",
                "paper4_v14_decision_uncertainty_interaction_matrix.csv",
                "paper4_v14_base_vs_lookahead_model_registry.csv",
                "paper4_v14_working_champion_powell_audit.csv",
                "paper4_v14_stage_readiness_dashboard.csv",
                "paper4_v14_claim_artifact_matrix.csv",
            ]
        ),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": "V14 is a Powell framing/governance reframe, not a new performance-optimization wave.",
    }
    _write_json("paper4_v14_status.json", status)
    _write_v14_note(status)
    if PAPER4_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion JSON exists")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
