"""Run bounded Lab 4 lanes on top of retained Paper 4 artifacts.

This script is intentionally semantic rather than versioned. It consumes the
current Paper 4 living-lab artifact surface and writes compact lane outputs for
the literature-driven lanes that are worth testing before any Paper 1 export.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "reports" / "paper_material" / "paper4" / "tables"
DOC_DIR = REPO / "docs" / "research"
DEFAULT_DATE = "2026-05-18"
OFFICIAL_CHAMPION = "paper1_economic_champion"
OFFICIAL_CHAMPION_RETURN = 170464.5429284627


@dataclass(frozen=True)
class OutputBundle:
    lane1: pd.DataFrame
    lane2: pd.DataFrame
    lane5: pd.DataFrame
    summary: pd.DataFrame
    memo: str


def _read_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_parquet(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).fillna(False)


def _minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        return pd.Series(1.0, index=series.index)
    score = (values - lo) / (hi - lo)
    if not higher_is_better:
        score = 1.0 - score
    return score.clip(0.0, 1.0).fillna(0.0)


def _format_float(value: float | int | str | None, digits: int = 4) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _mdcp_by_policy() -> pd.DataFrame:
    mdcp = _read_csv("paper4_mdcp_worst_source_frontier.csv")
    grouped = (
        mdcp.groupby("policy_id", as_index=False)
        .agg(
            mdcp_worst_source_coverage=("worst_source_coverage_online_90", "min"),
            mdcp_mean_source_coverage=("worst_source_coverage_online_90", "mean"),
            mdcp_source_families=("source_id", "nunique"),
            mdcp_all_sources_pass_80=("mdcp_v2_pass_80", "all"),
        )
        .sort_values("policy_id")
    )
    return grouped


def build_lane1_crc_ltt_gates() -> pd.DataFrame:
    """Predeclare CRC/LTT-inspired gates over the retained policy universe."""
    selector = _read_csv("paper4_diagnostic_selector_results.csv")
    mdcp = _mdcp_by_policy()
    work = selector.merge(mdcp, on="policy_id", how="left")

    sqrt_alpha01 = math.sqrt(0.01)
    return_floor = 0.95 * OFFICIAL_CHAMPION_RETURN
    source_floor = 0.80

    work["gate_bounds_hold"] = _truthy(work["all_bounds_hold"]) & _truthy(work["ab_pass_all"])
    work["gate_no_bound_violation"] = (
        pd.to_numeric(work["violation"], errors="coerce").fillna(1.0).le(0)
    )
    work["gate_crc_miscoverage"] = (
        pd.to_numeric(work["weighted_miscoverage_V"], errors="coerce").fillna(1.0).le(sqrt_alpha01)
    )
    work["gate_gamma_cp"] = pd.to_numeric(work["gamma_cp"], errors="coerce").fillna(1.0).le(0.20)
    work["gate_return_floor_95pct_champion"] = (
        pd.to_numeric(work["realized_total_return"], errors="coerce")
        .fillna(-np.inf)
        .ge(return_floor)
    )
    work["gate_satisficing_clean"] = pd.to_numeric(
        work["satisficing_pass_rate"], errors="coerce"
    ).fillna(0.0).ge(1.0) & pd.to_numeric(work["min_satisficing_margin"], errors="coerce").fillna(
        -np.inf
    ).ge(-1e-6)
    work["gate_source_defended_80"] = (
        pd.to_numeric(work["mdcp_worst_source_coverage"], errors="coerce")
        .fillna(0.0)
        .ge(source_floor)
    )

    risk_gates = [
        "gate_bounds_hold",
        "gate_no_bound_violation",
        "gate_crc_miscoverage",
        "gate_gamma_cp",
    ]
    operational_gates = risk_gates + [
        "gate_return_floor_95pct_champion",
        "gate_satisficing_clean",
    ]
    all_gates = operational_gates + ["gate_source_defended_80"]
    work["crc_ltt_risk_control_pass"] = work[risk_gates].all(axis=1)
    work["crc_ltt_operational_pass"] = work[operational_gates].all(axis=1)
    work["crc_ltt_source_hardened_pass"] = work[all_gates].all(axis=1)

    work["failed_gate_count"] = (~work[all_gates]).sum(axis=1)
    work["failed_gates"] = work[all_gates].apply(
        lambda row: ";".join(row.index[~row].tolist()) if (~row).any() else "none",
        axis=1,
    )
    work["lane1_decision"] = np.select(
        [
            work["crc_ltt_source_hardened_pass"],
            work["crc_ltt_operational_pass"],
            work["crc_ltt_risk_control_pass"],
        ],
        [
            "append_source_hardened_gate_pass",
            "append_operational_pass_source_fragile",
            "append_risk_pass_operational_gap",
        ],
        default="park_gate_fail",
    )
    work.loc[
        work["policy_id"].eq(OFFICIAL_CHAMPION),
        "lane1_decision",
    ] = "protect_official_champion_source_caveat"

    keep_cols = [
        "policy_id",
        "is_paper1_champion",
        "realized_total_return",
        "weighted_miscoverage_V",
        "gamma_cp",
        "satisficing_pass_rate",
        "min_satisficing_margin",
        "mdcp_worst_source_coverage",
        "mdcp_mean_source_coverage",
        "mdcp_source_families",
        "crc_ltt_risk_control_pass",
        "crc_ltt_operational_pass",
        "crc_ltt_source_hardened_pass",
        "failed_gate_count",
        "failed_gates",
        "diagnostic_selector_rank",
        "lane1_decision",
    ]
    return work[keep_cols].sort_values(
        ["crc_ltt_source_hardened_pass", "realized_total_return"],
        ascending=[False, False],
    )


def _pick_policy(
    frame: pd.DataFrame, mask: pd.Series, sort_cols: list[str], ascending: list[bool]
) -> pd.Series:
    candidates = frame[mask].copy()
    if candidates.empty:
        candidates = frame.copy()
    return candidates.sort_values(sort_cols, ascending=ascending).iloc[0]


def build_lane2_croms_lite_selection(lane1: pd.DataFrame) -> pd.DataFrame:
    selector = _read_csv("paper4_diagnostic_selector_results.csv")
    full = _read_csv("paper4_full_universe_topk_policy_eval.csv")
    exact = _read_csv("paper4_exact_limited_topk_policy_eval.csv")
    mdcp = _mdcp_by_policy()

    base = selector.merge(mdcp, on="policy_id", how="left").merge(
        lane1[
            [
                "policy_id",
                "crc_ltt_risk_control_pass",
                "crc_ltt_operational_pass",
                "crc_ltt_source_hardened_pass",
                "lane1_decision",
            ]
        ],
        on="policy_id",
        how="left",
    )
    base = base.merge(
        full[
            [
                "policy_id",
                "full_realized_return",
                "full_weighted_pd_high",
                "full_coverage_alpha01",
            ]
        ],
        on="policy_id",
        how="left",
    ).merge(
        exact[
            [
                "policy_id",
                "realized_return_exact_limited",
                "coverage_alpha01_exact_limited",
            ]
        ],
        on="policy_id",
        how="left",
    )

    base["score_croms_lite_balanced"] = (
        0.30 * _minmax_score(base["realized_total_return"], higher_is_better=True)
        + 0.25 * _minmax_score(base["mdcp_worst_source_coverage"], higher_is_better=True)
        + 0.20 * _minmax_score(base["gamma_cp"], higher_is_better=False)
        + 0.15 * _minmax_score(base["weighted_miscoverage_V"], higher_is_better=False)
        + 0.10
        * pd.to_numeric(base["satisficing_pass_rate"], errors="coerce").fillna(0.0).clip(0, 1)
    )
    base["score_tail_return_proxy"] = (
        0.45 * _minmax_score(base["realized_total_return"], higher_is_better=True)
        + 0.30 * _minmax_score(base["mean_loss_rate"], higher_is_better=False)
        + 0.25 * _minmax_score(base["mdcp_worst_source_coverage"], higher_is_better=True)
    )

    always = pd.Series(True, index=base.index)
    source_hardened = base["crc_ltt_source_hardened_pass"].fillna(False)
    operational = base["crc_ltt_operational_pass"].fillna(False)
    return_floor = pd.to_numeric(base["realized_total_return"], errors="coerce").ge(
        0.95 * OFFICIAL_CHAMPION_RETURN
    )

    strategies: list[tuple[str, str, pd.Series, list[str], list[bool]]] = [
        (
            "official_paper1_champion",
            "fixed official champion; no Lab 4 selector can promote over this without a promotion protocol",
            base["policy_id"].eq(OFFICIAL_CHAMPION),
            ["policy_id"],
            [True],
        ),
        (
            "diagnostic_selector_rank",
            "retained Paper 4 diagnostic selector rank",
            always,
            ["diagnostic_selector_rank"],
            [True],
        ),
        (
            "return_subject_to_crc_ltt",
            "maximize realized return among operational CRC/LTT passes",
            operational,
            ["realized_total_return"],
            [False],
        ),
        (
            "source_defended_return",
            "maximize realized return among source-hardened CRC/LTT passes",
            source_hardened,
            ["realized_total_return"],
            [False],
        ),
        (
            "coverage_only_source",
            "maximize worst source-family coverage regardless of return",
            always,
            ["mdcp_worst_source_coverage", "realized_total_return"],
            [False, False],
        ),
        (
            "min_gamma_cp_near_champion",
            "minimize Gamma_CP among policies within 95% of champion return",
            return_floor,
            ["gamma_cp", "realized_total_return"],
            [True, False],
        ),
        (
            "croms_lite_balanced_score",
            "decision-risk selection balancing return, source coverage, Gamma_CP, V and satisficing",
            always,
            ["score_croms_lite_balanced", "realized_total_return"],
            [False, False],
        ),
        (
            "tail_return_source_proxy",
            "tail-aware proxy using return, mean loss and source coverage",
            always,
            ["score_tail_return_proxy", "realized_total_return"],
            [False, False],
        ),
    ]

    rows: list[dict[str, object]] = []
    champion = base[base["policy_id"].eq(OFFICIAL_CHAMPION)].iloc[0]
    for strategy, description, mask, sort_cols, ascending in strategies:
        row = _pick_policy(base, mask, sort_cols, ascending)
        rows.append(
            {
                "selection_strategy": strategy,
                "description": description,
                "selected_policy_id": row["policy_id"],
                "is_official_champion": bool(row["policy_id"] == OFFICIAL_CHAMPION),
                "lane1_decision": row.get("lane1_decision", ""),
                "realized_total_return": float(row["realized_total_return"]),
                "delta_return_vs_official": float(
                    row["realized_total_return"] - champion["realized_total_return"]
                ),
                "weighted_miscoverage_V": float(row["weighted_miscoverage_V"]),
                "gamma_cp": float(row["gamma_cp"]),
                "mdcp_worst_source_coverage": float(row["mdcp_worst_source_coverage"]),
                "delta_worst_source_coverage_vs_official": float(
                    row["mdcp_worst_source_coverage"] - champion["mdcp_worst_source_coverage"]
                ),
                "satisficing_pass_rate": float(row["satisficing_pass_rate"]),
                "diagnostic_selector_rank": int(row["diagnostic_selector_rank"]),
                "full_realized_return": row.get("full_realized_return"),
                "full_coverage_alpha01": row.get("full_coverage_alpha01"),
                "score_croms_lite_balanced": float(row["score_croms_lite_balanced"]),
                "selector_decision": (
                    "paper1_protected_reference"
                    if row["policy_id"] == OFFICIAL_CHAMPION
                    else "append_tradeoff_evidence_not_promotion"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_lane5_cvar_oce_challenger() -> pd.DataFrame:
    frontier_cashflow = _read_csv("paper4_frontier_cvar_oce_decision_2026-05-18.csv")
    stress = _read_csv("paper4_v31_champion_vs_cvar_stress_memo.csv")
    v33 = _read_csv("paper4_v33_cvar_frontier_v3.csv")
    v467 = _read_csv("paper4_v467_cvar_frontier_probe.csv")
    paper1_tail = pd.read_csv(
        REPO
        / "reports"
        / "paper_material"
        / "paper1"
        / "tables"
        / "paper1_tableA12_tail_risk_oce_cvar.csv"
    )

    rows: list[dict[str, object]] = []

    official_lgd45 = paper1_tail.iloc[(paper1_tail["lgd"] - 0.45).abs().argsort()[:1]].iloc[0]
    rows.append(
        {
            "evidence_block": "paper1_official_tail_profile_lgd45",
            "source_artifact": "reports/paper_material/paper1/tables/paper1_tableA12_tail_risk_oce_cvar.csv",
            "candidate_or_policy": OFFICIAL_CHAMPION,
            "primary_metric": "funded_set_repriced_return",
            "primary_value": float(official_lgd45["funded_set_repriced_return"]),
            "secondary_metric": "cvar_95_loss_rate",
            "secondary_value": float(official_lgd45["cvar_95_loss_rate"]),
            "decision": "paper1_reference_only",
            "claim_boundary": "Official champion tail profile; not a CVaR optimizer.",
        }
    )

    s = stress.iloc[0]
    rows.append(
        {
            "evidence_block": "paired_common_path_champion_vs_cvar",
            "source_artifact": "reports/paper_material/paper4/tables/paper4_v31_champion_vs_cvar_stress_memo.csv",
            "candidate_or_policy": s["challenger_policy_id"],
            "primary_metric": "prob_challenger_beats_reference",
            "primary_value": float(s["prob_challenger_beats_reference"]),
            "secondary_metric": "loss_p95_reduction",
            "secondary_value": float(s["reference_loss_p95"] - s["challenger_loss_p95"]),
            "decision": "append_tail_challenger_retain_champion",
            "claim_boundary": "Tail loss improves but paired wealth does not robustly beat the reference.",
        }
    )

    cashflow_econ = frontier_cashflow[
        frontier_cashflow["policy_id"].eq("cashflow_economic_proxy")
    ].iloc[0]
    cashflow_best = frontier_cashflow.sort_values(
        ["cvar95_loan_loss", "realized_cashflow_return"], ascending=[True, False]
    ).iloc[0]
    rows.append(
        {
            "evidence_block": "raw_cashflow_tail_proxy",
            "source_artifact": "reports/paper_material/paper4/tables/paper4_frontier_cvar_oce_decision_2026-05-18.csv",
            "candidate_or_policy": cashflow_best["policy_id"],
            "primary_metric": "cvar95_loan_loss_reduction_vs_cashflow_economic",
            "primary_value": float(
                cashflow_econ["cvar95_loan_loss"] - cashflow_best["cvar95_loan_loss"]
            ),
            "secondary_metric": "return_gap_vs_official_champion",
            "secondary_value": float(
                cashflow_best["realized_cashflow_return"] - OFFICIAL_CHAMPION_RETURN
            ),
            "decision": "append_cashflow_challenger_not_promotion",
            "claim_boundary": "Retrospective realized cashflow diagnostic, not an ex ante deployable optimizer.",
        }
    )

    feasible = v33[
        v33["solver_status"].astype(str).str.contains("optimal", case=False, na=False)
    ].copy()
    if not feasible.empty:
        best = feasible.sort_values(
            ["tail_champion_score_v33", "realized_return_proxy_lgd45"], ascending=[False, False]
        ).iloc[0]
        rows.append(
            {
                "evidence_block": "restricted_master_cvar_frontier",
                "source_artifact": "reports/paper_material/paper4/tables/paper4_v33_cvar_frontier_v3.csv",
                "candidate_or_policy": best["policy_id"],
                "primary_metric": "tail_champion_score_v33",
                "primary_value": float(best["tail_champion_score_v33"]),
                "secondary_metric": "exact_full_universe_claim_v33",
                "secondary_value": bool(best["exact_full_universe_claim_v33"]),
                "decision": "append_restricted_master_only",
                "claim_boundary": "Restricted-master non-dominated diagnostic, not exact full-universe CVaR optimality.",
            }
        )

    best_v467 = v467.sort_values(
        ["local_frontier_candidate_v467", "objective_return_v467"], ascending=[False, False]
    ).iloc[0]
    rows.append(
        {
            "evidence_block": "curated_local_frontier_probe",
            "source_artifact": "reports/paper_material/paper4/tables/paper4_v467_cvar_frontier_probe.csv",
            "candidate_or_policy": best_v467["candidate_label_v467"],
            "primary_metric": "strict_return_cvar_improvement_vs_predecessor_v467",
            "primary_value": bool(best_v467["strict_return_cvar_improvement_vs_predecessor_v467"]),
            "secondary_metric": "one_swap_local_optimality_cleared_v467",
            "secondary_value": bool(best_v467["one_swap_local_optimality_cleared_v467"]),
            "decision": "append_local_probe_boundary",
            "claim_boundary": str(best_v467["claim_boundary_v467"]),
        }
    )

    return pd.DataFrame(rows)


def _dependency_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def build_lane3_e2e_conformal_readiness() -> pd.DataFrame:
    """Audit whether an E2E conformal calibration prototype is ready to promote."""
    method_summary = _read_csv("paper4_online_conformal_v4_method_summary.csv")
    selector = _read_csv("paper4_lab4_lane2_croms_lite_selection_2026-05-18.csv")
    dep_rows = []
    for package in ["torch", "cvxpy", "cvxpylayers", "pyepo"]:
        dep_rows.append((package, _dependency_available(package)))

    best_method = method_summary.sort_values(
        ["coverage_source_month_min", "avg_width_loan"], ascending=[False, True]
    ).iloc[0]
    best_selector = selector[selector["selection_strategy"].eq("croms_lite_balanced_score")].iloc[0]
    e2e_deps_ready = all(available for _, available in dep_rows)

    rows = [
        {
            "evidence_block": "dependency_surface",
            "metric": "all_e2e_dependencies_available_in_current_env",
            "value": e2e_deps_ready,
            "decision": "park_e2e_training",
            "claim_boundary": "Current main environment is not the isolated differentiable optimization environment.",
        },
        {
            "evidence_block": "dependency_surface",
            "metric": "available_packages",
            "value": ";".join(f"{pkg}={available}" for pkg, available in dep_rows),
            "decision": "park_e2e_training",
            "claim_boundary": "Do not mutate the main environment for cvxpylayers/PyEPO until a prototype proves value.",
        },
        {
            "evidence_block": "posthoc_online_conformal",
            "metric": "best_source_month_min_coverage",
            "value": float(best_method["coverage_source_month_min"]),
            "decision": "proxy_only",
            "claim_boundary": "This is post-hoc online/source conformal evidence, not end-to-end learned calibration.",
        },
        {
            "evidence_block": "croms_lite_proxy",
            "metric": "balanced_selector_delta_return_vs_official",
            "value": float(best_selector["delta_return_vs_official"]),
            "decision": "proxy_only",
            "claim_boundary": "CROMS-lite selector audit is decision-aware selection, not differentiable E2E conformal training.",
        },
        {
            "evidence_block": "lane_decision",
            "metric": "promotable_e2e_conformal_calibration",
            "value": False,
            "decision": "park",
            "claim_boundary": "Reopen only with an isolated reproducible prototype that preserves OOT coverage and improves decision risk.",
        },
    ]
    return pd.DataFrame(rows)


def build_lane4_online_multisource_conformal() -> pd.DataFrame:
    frontier = _read_csv("paper4_frontier_online_conformal_decision_2026-05-18.csv")
    method_summary = _read_csv("paper4_online_conformal_v4_method_summary.csv")
    strict = _read_csv("paper4_v35_online_temporal_holdout.csv")
    monitoring = _read_csv("paper4_v470_online_monitoring_proxy_summary.csv")
    tight_sources = _read_csv("paper4_v468_tight_source_rankings.csv")
    mdcp = _mdcp_by_policy()

    best_frontier = frontier.sort_values(
        ["defended_min_coverage", "avg_width"], ascending=[False, True]
    ).iloc[0]
    best_method = method_summary.sort_values(
        ["coverage_source_month_min", "avg_width_loan"], ascending=[False, True]
    ).iloc[0]
    m = monitoring.iloc[0]
    tight = tight_sources.sort_values("source_slack_rank_v468").iloc[0]

    mdcp_pass = mdcp[
        pd.to_numeric(mdcp["mdcp_worst_source_coverage"], errors="coerce").ge(0.80)
        & _truthy(mdcp["mdcp_all_sources_pass_80"])
    ]
    rows = [
        {
            "evidence_block": "frontier_source_holdout",
            "metric": "best_defended_min_coverage",
            "value": float(best_frontier["defended_min_coverage"]),
            "method_or_source": best_frontier["method"],
            "decision": "park_nominal_shortfall",
            "claim_boundary": "Best bounded frontier misses the 0.80 defended-source gate.",
        },
        {
            "evidence_block": "v4_online_method_summary",
            "metric": "best_source_month_min_coverage",
            "value": float(best_method["coverage_source_month_min"]),
            "method_or_source": best_method["online_method_v4"],
            "decision": "append_monitoring_proxy",
            "claim_boundary": "Strongest method is still not a live online deployment claim.",
        },
        {
            "evidence_block": "v35_strict_holdouts",
            "metric": "strict_holdout_pass_rows",
            "value": int(_truthy(strict["pass_gate"]).sum()),
            "method_or_source": f"total={len(strict)}",
            "decision": "park_strict_universal_claim",
            "claim_boundary": "Strict temporal/source holdouts do not survive universally.",
        },
        {
            "evidence_block": "v470_monitoring_proxy",
            "metric": "strict_live_claim_allowed",
            "value": bool(m["v341_strict_live_claim_allowed_v470"]),
            "method_or_source": m["v9_best_method_v470"],
            "decision": "append_monitoring_proxy_only",
            "claim_boundary": str(m["claim_boundary_v470"]),
        },
        {
            "evidence_block": "mdcp_policy_frontier",
            "metric": "policies_passing_all_source_family_80",
            "value": int(len(mdcp_pass)),
            "method_or_source": "source families over retained policy universe",
            "decision": "append_source_governance",
            "claim_boundary": "Policy-level MDCP evidence is a source-governance diagnostic.",
        },
        {
            "evidence_block": "tight_source_blocker",
            "metric": "tightest_source_family",
            "value": str(tight["source_family_v468"]),
            "method_or_source": str(tight["source_id_v468"]),
            "decision": "append_limitation",
            "claim_boundary": str(tight["claim_boundary_v468"]),
        },
    ]
    return pd.DataFrame(rows)


def build_lane6_spo_dfl_comparator() -> pd.DataFrame:
    frontier = _read_csv("paper4_frontier_spo_dfl_decision_2026-05-18.csv")
    candidates = _read_csv("paper4_v32_spo_candidate_comparison_v3.csv")
    blockers = _read_csv("paper4_v32_spo_dependency_blockers.csv")
    temporal = _read_csv("paper4_v32_spo_temporal_oracle_regret_v3.csv")

    toy = frontier[frontier["probe"].eq("ridge_predict_then_optimize_top350")].iloc[0]
    retained = frontier[frontier["probe"].eq("retained_v11_surrogate")].copy()
    best_retained = retained.sort_values("objective_return", ascending=False).iloc[0]
    serious = candidates[candidates["spo_candidate_status_v32"].eq("serious_comparator")]
    dep_blocked = blockers[
        blockers["decision_v32"].astype(str).str.contains("dependency_blocked", na=False)
    ]
    temporal_regret = pd.to_numeric(temporal["decision_regret_proxy_v20"], errors="coerce").dropna()

    rows = [
        {
            "evidence_block": "frontier_toy_probe",
            "metric": "toy_oracle_gap",
            "value": float(toy["oracle_gap"]),
            "decision": "park_toy_only",
            "claim_boundary": "Toy top-k probe is not formal SPO+ or production DFL training.",
        },
        {
            "evidence_block": "retained_surrogate",
            "metric": "best_retained_objective_return",
            "value": float(best_retained["objective_return"]),
            "decision": "append_comparator_only",
            "claim_boundary": str(best_retained["training_scope_v11"]),
        },
        {
            "evidence_block": "v32_candidate_registry",
            "metric": "serious_spo_comparators",
            "value": int(len(serious)),
            "decision": "append_oracle_regret_comparator",
            "claim_boundary": "Compare against oracle regret only; no formal differentiable SPO+ claim.",
        },
        {
            "evidence_block": "v32_dependency_blockers",
            "metric": "blocked_differentiable_dependencies",
            "value": int(len(dep_blocked)),
            "decision": "park_integrated_dfl",
            "claim_boundary": "Main environment does not support formal differentiable SPO+ integration.",
        },
        {
            "evidence_block": "current_env_dependency_probe",
            "metric": "torch_cvxpylayers_pyepo_available",
            "value": ";".join(
                f"{pkg}={_dependency_available(pkg)}" for pkg in ["torch", "cvxpylayers", "pyepo"]
            ),
            "decision": "park_integrated_dfl",
            "claim_boundary": "Use isolated environments only for PyEPO/cvxpylayers prototypes.",
        },
        {
            "evidence_block": "temporal_oracle_regret",
            "metric": "median_monthly_regret_proxy",
            "value": float(temporal_regret.median()),
            "decision": "append_regret_diagnostic",
            "claim_boundary": "Decision-oracle/surrogate regret only; not formal differentiable SPO+.",
        },
    ]
    return pd.DataFrame(rows)


def build_lane7_ifrs9_proxy() -> pd.DataFrame:
    frontier = _read_csv("paper4_frontier_ifrs9_sicr_decision_2026-05-18.csv")
    readiness = _read_csv("paper4_v36_ifrs9_readiness_matrix.csv")
    sensitivity = _read_csv("paper4_v36_ifrs9_sicr_sensitivity_v3.csv")
    req = _read_csv("paper4_v472_ifrs9_requirement_audit.csv")
    quality = _read_csv("paper4_ifrs9_v4_input_quality.csv")
    policy = _read_csv("paper4_ifrs9_v4_contractual_policy_summary.csv")

    combined = frontier[frontier["rule"].eq("combined_raw_sicr")].iloc[0]
    missing_req = req[req["availability_v472"].eq("missing")]
    production_allowed = bool(_truthy(sensitivity["production_ifrs9_staging_claim_allowed"]).any())
    severe = policy[policy["scenario"].eq("severe")]
    rows = [
        {
            "evidence_block": "raw_enriched_sicr",
            "metric": "combined_default_lift",
            "value": float(combined["default_lift"]),
            "decision": "append_proxy_diagnostic",
            "claim_boundary": "Raw servicing fields sharpen SICR proxy but do not create contractual IFRS9.",
        },
        {
            "evidence_block": "raw_enriched_sicr",
            "metric": "combined_triggered_share",
            "value": float(combined["triggered_share"]),
            "decision": "append_proxy_diagnostic",
            "claim_boundary": "Proxy trigger is broad enough for monitoring, not accounting staging.",
        },
        {
            "evidence_block": "v36_readiness",
            "metric": "missing_contractual_requirements",
            "value": int(
                readiness.loc[readiness["availability_v36"].eq("missing"), "requirements"].sum()
            ),
            "decision": "block_contractual_ifrs9",
            "claim_boundary": "Contractual IFRS9 claim is not allowed.",
        },
        {
            "evidence_block": "v472_requirement_audit",
            "metric": "missing_named_requirements",
            "value": ";".join(missing_req["requirement_v472"].astype(str).tolist()),
            "decision": "block_contractual_ifrs9",
            "claim_boundary": "Monthly servicing panel and coherent macro scenario process remain required.",
        },
        {
            "evidence_block": "v4_input_quality",
            "metric": "performance_reference_rows",
            "value": int(quality.loc[quality["input"].eq("performance_reference"), "rows"].iloc[0]),
            "decision": "append_proxy_only",
            "claim_boundary": str(
                quality.loc[quality["input"].eq("performance_reference"), "claim_scope"].iloc[0]
            ),
        },
        {
            "evidence_block": "v4_policy_summary",
            "metric": "median_severe_net_return_after_proxy_ecl",
            "value": float(
                pd.to_numeric(
                    severe["net_return_after_contractual_ecl_v4"], errors="coerce"
                ).median()
            ),
            "decision": "append_proxy_only",
            "claim_boundary": "Scenario ECL proxy can stress policies but is not a contractual allowance.",
        },
        {
            "evidence_block": "v36_sensitivity",
            "metric": "production_ifrs9_staging_claim_allowed",
            "value": production_allowed,
            "decision": "block_contractual_ifrs9",
            "claim_boundary": "SICR sensitivity only; no production IFRS9 staging claim.",
        },
    ]
    return pd.DataFrame(rows)


def build_lane8_governance_fairness_proxy() -> pd.DataFrame:
    frontier = _read_csv("paper4_frontier_fair_lending_proxy_decision_2026-05-18.csv")
    protocol = _read_csv("paper4_v37_fairness_proxy_only_protocol.csv")
    source = _read_csv("paper4_v37_source_governance_appendix.csv")
    constrained = _read_parquet("fairness_constrained_policy_eval.parquet")

    top_dispersion = frontier.sort_values("interest_rate_range", ascending=False).iloc[0]
    allowed_protocol = protocol[_truthy(protocol["allowed_claim"])]
    legal_claim_allowed = bool(_truthy(frontier["legal_fair_lending_claim_allowed"]).any())
    source_high_support = source[source["support_band_v37"].eq("high_support")]
    constrained_pass_all = bool(_truthy(constrained["passed_all"]).all())

    rows = [
        {
            "evidence_block": "observable_dispersion",
            "metric": "top_interest_rate_dispersion_dimension",
            "value": str(top_dispersion["dimension"]),
            "decision": "append_governance_only",
            "claim_boundary": "Observable source dispersion is not protected-attribute disparity.",
        },
        {
            "evidence_block": "observable_dispersion",
            "metric": "top_interest_rate_range",
            "value": float(top_dispersion["interest_rate_range"]),
            "decision": "append_governance_only",
            "claim_boundary": "Use as source governance, not legal fair-lending evidence.",
        },
        {
            "evidence_block": "v37_protocol",
            "metric": "allowed_protocol_items",
            "value": ";".join(allowed_protocol["protocol_item"].astype(str).tolist()),
            "decision": "append_governance_only",
            "claim_boundary": "Only observable source caps are allowed as governance diagnostics.",
        },
        {
            "evidence_block": "v37_protocol",
            "metric": "legal_fair_lending_claim_allowed",
            "value": legal_claim_allowed,
            "decision": "block_legal_claim",
            "claim_boundary": "No legal claim without protected attributes and approved proxy protocol.",
        },
        {
            "evidence_block": "source_governance_appendix",
            "metric": "high_support_source_cells",
            "value": int(len(source_high_support)),
            "decision": "append_governance_only",
            "claim_boundary": "Support-aware source governance; protected attributes are not inferred.",
        },
        {
            "evidence_block": "proxy_fairness_policy_eval",
            "metric": "proxy_attributes_pass_all",
            "value": constrained_pass_all,
            "decision": "append_proxy_diagnostic",
            "claim_boundary": "Proxy fairness diagnostic is not protected-attribute proof.",
        },
    ]
    return pd.DataFrame(rows)


def build_lane9_causal_cate_boundary() -> pd.DataFrame:
    frontier = _read_csv("paper4_frontier_cate_policy_value_decision_2026-05-18.csv")
    gate = _read_csv("paper4_v37_cate_gate_report.csv")
    protocol = _read_csv("paper4_v37_causal_identification_protocol.csv")

    metrics = {
        row["metric"]: row["value"]
        for _, row in frontier.drop_duplicates("metric", keep="first").iterrows()
    }
    blocked_gates = gate[~_truthy(gate["cate_policy_value_allowed"])]
    protocol_blocked = protocol[~_truthy(protocol["cate_policy_value_allowed"])]
    ate = float(metrics.get("aipw_ate_high_rate_within_grade", np.nan))
    placebo = float(metrics.get("placebo_aipw_ate", np.nan))
    placebo_ratio = abs(placebo) / abs(ate) if ate else np.nan

    rows = [
        {
            "evidence_block": "frontier_aipw_screen",
            "metric": "aipw_ate_high_rate_within_grade",
            "value": ate,
            "decision": "park_policy_value",
            "claim_boundary": "Accepted-loan observational sensitivity, not causal policy value.",
        },
        {
            "evidence_block": "frontier_overlap",
            "metric": "overlap_share_10_90",
            "value": float(metrics.get("overlap_share_10_90", np.nan)),
            "decision": "park_policy_value",
            "claim_boundary": "Overlap remains far below the 0.80 diagnostic stability gate.",
        },
        {
            "evidence_block": "frontier_placebo",
            "metric": "placebo_to_effect_abs_ratio",
            "value": float(placebo_ratio),
            "decision": "park_policy_value",
            "claim_boundary": "Placebo is nonzero and supports caution.",
        },
        {
            "evidence_block": "v37_gate_report",
            "metric": "blocked_or_unpromotable_gates",
            "value": int(len(blocked_gates)),
            "decision": "park_policy_value",
            "claim_boundary": "CATE policy value blocked unless identification, overlap, sensitivity, falsification and intervals pass.",
        },
        {
            "evidence_block": "v37_identification_protocol",
            "metric": "protocol_items_not_allowing_policy_value",
            "value": int(len(protocol_blocked)),
            "decision": "park_policy_value",
            "claim_boundary": "Reject inference and application selection remain unresolved.",
        },
    ]
    return pd.DataFrame(rows)


def build_remaining_summary(
    lane3: pd.DataFrame,
    lane4: pd.DataFrame,
    lane6: pd.DataFrame,
    lane7: pd.DataFrame,
    lane8: pd.DataFrame,
    lane9: pd.DataFrame,
) -> pd.DataFrame:
    lane4_best = lane4[lane4["evidence_block"].eq("frontier_source_holdout")].iloc[0]
    lane6_gap = lane6[lane6["evidence_block"].eq("frontier_toy_probe")].iloc[0]
    lane7_lift = lane7[
        (lane7["evidence_block"].eq("raw_enriched_sicr"))
        & (lane7["metric"].eq("combined_default_lift"))
    ].iloc[0]
    lane8_legal = lane8[lane8["metric"].eq("legal_fair_lending_claim_allowed")].iloc[0]
    lane9_overlap = lane9[lane9["metric"].eq("overlap_share_10_90")].iloc[0]

    rows = [
        {
            "lane": "lane3_e2e_conformal_calibration",
            "decision": "park",
            "paper4_destination": "lab4_future_prototype_note",
            "paper1_destination": "none",
            "key_result": "No promotable E2E conformal training artifact exists in the current Lab 4 surface.",
            "evidence_gate": "requires isolated reproducible prototype with coverage preserved and decision risk improved",
            "stop_rule": "Do not mutate main environment or imply full E2E implementation.",
            "claim_boundary": "Proxy audits are not end-to-end conformal calibration.",
        },
        {
            "lane": "lane4_online_multisource_conformal",
            "decision": "park_with_appendix_limitation",
            "paper4_destination": "lab4_source_holdout_appendix",
            "paper1_destination": "possible_limitation_only",
            "key_result": f"Best bounded defended min coverage is {_format_float(lane4_best['value'])}.",
            "evidence_gate": ">=0.80 defended source coverage plus strict holdout survival",
            "stop_rule": "Do not claim live online validity without production feedback/external holdout.",
            "claim_boundary": "Retrospective source governance only.",
        },
        {
            "lane": "lane6_spo_dfl_comparator",
            "decision": "park_integrated_dfl_append_oracle_regret",
            "paper4_destination": "lab4_spo_oracle_regret_appendix",
            "paper1_destination": "related_work_only",
            "key_result": f"Toy oracle gap remains {_format_float(lane6_gap['value'], 2)}.",
            "evidence_gate": "requires formal differentiable SPO+ or robust OOT comparator",
            "stop_rule": "Keep PyEPO/cvxpylayers isolated until a small prototype changes regret evidence.",
            "claim_boundary": "Oracle-regret/surrogate evidence only; not integrated DFL.",
        },
        {
            "lane": "lane7_ifrs9_proxy",
            "decision": "append",
            "paper4_destination": "lab4_ifrs9_proxy_appendix",
            "paper1_destination": "none_now",
            "key_result": f"Combined raw SICR lift is {_format_float(lane7_lift['value'])}.",
            "evidence_gate": "proxy ECL/SICR evidence plus explicit missing-contractual-requirement table",
            "stop_rule": "Stop before contractual IFRS9 claims.",
            "claim_boundary": "IFRS9-inspired proxy only.",
        },
        {
            "lane": "lane8_governance_fairness_proxy",
            "decision": "append",
            "paper4_destination": "lab4_governance_appendix",
            "paper1_destination": "possible_model_risk_context_only",
            "key_result": f"Legal fair-lending claim allowed = {lane8_legal['value']}.",
            "evidence_gate": "observable-source governance only; no protected attribute inference",
            "stop_rule": "Stop before legal fairness/compliance claims.",
            "claim_boundary": "Proxy/source governance, not fair-lending proof.",
        },
        {
            "lane": "lane9_causal_cate_boundary",
            "decision": "park",
            "paper4_destination": "lab4_causal_boundary_note",
            "paper1_destination": "none",
            "key_result": f"Overlap share 10-90 is {_format_float(lane9_overlap['value'])}.",
            "evidence_gate": "identification, overlap, sensitivity, falsification and intervals must pass",
            "stop_rule": "Do not continue without rejected applicants, instrument, or reviewer request.",
            "claim_boundary": "Observational sensitivity only, not policy value.",
        },
    ]
    return pd.DataFrame(rows)


def build_remaining_memo(
    lane3: pd.DataFrame,
    lane4: pd.DataFrame,
    lane6: pd.DataFrame,
    lane7: pd.DataFrame,
    lane8: pd.DataFrame,
    lane9: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    def row_value(frame: pd.DataFrame, block: str, metric: str) -> object:
        rows = frame[(frame["evidence_block"].eq(block)) & (frame["metric"].eq(metric))]
        return rows["value"].iloc[0] if not rows.empty else "NA"

    lines = [
        "# Paper 4 Lab 4 Lanes 3, 4, 6, 7, 8 and 9 Execution Memo - 2026-05-18",
        "",
        "## Scope",
        "",
        "This memo executes the remaining literature-driven Lab 4 lanes over the retained",
        "Paper 4 living-lab artifact surface. The outputs classify evidence as append,",
        "park, or future-prototype material. No Paper Estrella champion search is reopened.",
        "",
        "## Lane Results",
        "",
        "| lane | decision | key result | boundary |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['lane']} | {row['decision']} | {row['key_result']} | {row['claim_boundary']} |"
        )

    lines += [
        "",
        "## Details",
        "",
        "### Lane 3 - E2E Conformal Calibration",
        "",
        "- Decision: `park`.",
        f"- Dependency surface: `{row_value(lane3, 'dependency_surface', 'available_packages')}`.",
        "- The existing evidence is post-hoc/source-conformal plus CROMS-lite selector audit,",
        "  not an end-to-end learned uncertainty set.",
        "",
        "### Lane 4 - Online / Multi-Source Conformal",
        "",
        "- Decision: `park_with_appendix_limitation`.",
        f"- Best bounded defended min coverage: `{_format_float(row_value(lane4, 'frontier_source_holdout', 'best_defended_min_coverage'))}`.",
        f"- Strict holdout pass rows: `{row_value(lane4, 'v35_strict_holdouts', 'strict_holdout_pass_rows')}`.",
        "- The lane is useful as source governance, but not as live online validity.",
        "",
        "### Lane 6 - SPO / DFL",
        "",
        "- Decision: `park_integrated_dfl_append_oracle_regret`.",
        f"- Toy oracle gap: `{_format_float(row_value(lane6, 'frontier_toy_probe', 'toy_oracle_gap'), 2)}`.",
        f"- Current dependency probe: `{row_value(lane6, 'current_env_dependency_probe', 'torch_cvxpylayers_pyepo_available')}`.",
        "- Keep oracle-regret/surrogate evidence; do not claim formal differentiable SPO+.",
        "",
        "### Lane 7 - IFRS9 Proxy",
        "",
        "- Decision: `append`.",
        f"- Combined raw SICR lift: `{_format_float(row_value(lane7, 'raw_enriched_sicr', 'combined_default_lift'))}`.",
        f"- Missing contractual requirements: `{row_value(lane7, 'v472_requirement_audit', 'missing_named_requirements')}`.",
        "- Keep IFRS9-inspired ECL/SICR proxy; contractual IFRS9 remains blocked.",
        "",
        "### Lane 8 - Governance / Fairness Proxy",
        "",
        "- Decision: `append`.",
        f"- Legal fair-lending claim allowed: `{row_value(lane8, 'v37_protocol', 'legal_fair_lending_claim_allowed')}`.",
        f"- Top dispersion dimension: `{row_value(lane8, 'observable_dispersion', 'top_interest_rate_dispersion_dimension')}`.",
        "- Keep source/proxy governance only; do not infer protected attributes.",
        "",
        "### Lane 9 - Causal / CATE Boundary",
        "",
        "- Decision: `park`.",
        f"- Overlap share 10-90: `{_format_float(row_value(lane9, 'frontier_overlap', 'overlap_share_10_90'))}`.",
        f"- Protocol items not allowing policy value: `{row_value(lane9, 'v37_identification_protocol', 'protocol_items_not_allowing_policy_value')}`.",
        "- Keep as causal boundary note; no policy-value claim.",
        "",
        "## Stop Rules",
        "",
        "- Do not mutate the main environment for Lane 3 or Lane 6 prototypes.",
        "- Do not claim live online/source validity from historical replay.",
        "- Do not claim contractual IFRS9, legal fair-lending compliance, or causal policy value.",
        "- Export to Paper Estrella only as limitation, model-risk context, or reviewer defense.",
    ]
    return "\n".join(lines) + "\n"


def build_summary(lane1: pd.DataFrame, lane2: pd.DataFrame, lane5: pd.DataFrame) -> pd.DataFrame:
    source_hardened = int(lane1["crc_ltt_source_hardened_pass"].sum())
    operational = int(lane1["crc_ltt_operational_pass"].sum())
    champion_lane1 = lane1[lane1["policy_id"].eq(OFFICIAL_CHAMPION)].iloc[0]
    best_source = lane2[lane2["selection_strategy"].eq("source_defended_return")].iloc[0]
    cvar_stress = lane5[lane5["evidence_block"].eq("paired_common_path_champion_vs_cvar")].iloc[0]

    rows = [
        {
            "lane": "lane1_crc_ltt_decision_loss_gates",
            "decision": "append",
            "paper4_destination": "lab4_governance_appendix",
            "paper1_destination": "possible_reviewer_defense_only",
            "key_result": f"{operational} policies pass operational CRC/LTT gates; {source_hardened} also pass source-hardening.",
            "evidence_gate": "bounds, V<=sqrt(alpha01), Gamma_CP<=0.20, return floor, satisficing and source gate",
            "stop_rule": "Do not convert gate pass into Paper 1 promotion without a promotion protocol.",
            "claim_boundary": f"Official champion decision: {champion_lane1['lane1_decision']}.",
        },
        {
            "lane": "lane2_croms_lite_selection",
            "decision": "append",
            "paper4_destination": "lab4_selector_tradeoff_appendix",
            "paper1_destination": "none_now",
            "key_result": (
                "Source-defended selector chooses "
                f"{best_source['selected_policy_id']} with return delta "
                f"{_format_float(best_source['delta_return_vs_official'], 2)} vs official."
            ),
            "evidence_gate": "selector must improve a declared risk dimension without reopening champion",
            "stop_rule": "Stop because selector choices are tradeoff evidence, not official promotion.",
            "claim_boundary": "CROMS-lite is a selection audit over retained artifacts, not a full CROMS implementation.",
        },
        {
            "lane": "lane5_cvar_oce_tail_challenger",
            "decision": "append",
            "paper4_destination": "lab4_tail_challenger_appendix",
            "paper1_destination": "possible_appendix_caveat_only",
            "key_result": (
                "CVaR challenger reduces loss p95 by "
                f"{_format_float(cvar_stress['secondary_value'], 2)} but prob beats reference is "
                f"{_format_float(cvar_stress['primary_value'], 4)}."
            ),
            "evidence_gate": "tail improvement plus paired wealth dominance required for promotion",
            "stop_rule": "Retain official champion because paired wealth dominance is absent.",
            "claim_boundary": "CVaR/OCE is a serious tail challenger, not the economic champion.",
        },
    ]
    return pd.DataFrame(rows)


def build_memo(
    lane1: pd.DataFrame, lane2: pd.DataFrame, lane5: pd.DataFrame, summary: pd.DataFrame
) -> str:
    top_l1 = lane1.head(5)
    top_l2 = lane2
    l5 = lane5

    lines = [
        "# Paper 4 Lab 4 Lanes 1, 2 and 5 Execution Memo - 2026-05-18",
        "",
        "## Scope",
        "",
        "This execution treats Lab 4 as the full living-lab surface of Paper 4: all retained",
        "Paper 4 artifacts, Paper 1 champion references and curated frontier diagnostics can",
        "be used as inputs. The outputs remain Lab 4 evidence until a later promote/append/",
        "park/delete decision is made.",
        "",
        "No `paper4_v###` artifacts, promotion JSONs or per-iteration status packets were",
        "created. The official Paper Estrella champion remains protected.",
        "",
        "## Lane 1 - CRC/LTT Decision-Loss Gates",
        "",
        f"- Operational CRC/LTT gate passes: `{int(lane1['crc_ltt_operational_pass'].sum())}` policies.",
        f"- Source-hardened passes: `{int(lane1['crc_ltt_source_hardened_pass'].sum())}` policies.",
        f"- Official champion lane decision: `{lane1[lane1['policy_id'].eq(OFFICIAL_CHAMPION)].iloc[0]['lane1_decision']}`.",
        "",
        "Top source-hardened or near-source-hardened policies:",
        "",
        "| policy | return | V | Gamma_CP | worst source | decision |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in top_l1.iterrows():
        lines.append(
            "| "
            f"{row['policy_id']} | "
            f"{_format_float(row['realized_total_return'], 2)} | "
            f"{_format_float(row['weighted_miscoverage_V'])} | "
            f"{_format_float(row['gamma_cp'])} | "
            f"{_format_float(row['mdcp_worst_source_coverage'])} | "
            f"{row['lane1_decision']} |"
        )

    lines += [
        "",
        "Interpretation: the CRC/LTT framing is useful as governance. It creates explicit",
        "pass/fail gates for risk control, operational return, satisficing and source",
        "hardening. It does not by itself promote any challenger to Paper Estrella.",
        "",
        "## Lane 2 - CROMS-Lite Selection Audit",
        "",
        "| selector | selected policy | delta return vs official | delta worst source vs official | decision |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for _, row in top_l2.iterrows():
        lines.append(
            "| "
            f"{row['selection_strategy']} | "
            f"{row['selected_policy_id']} | "
            f"{_format_float(row['delta_return_vs_official'], 2)} | "
            f"{_format_float(row['delta_worst_source_coverage_vs_official'])} | "
            f"{row['selector_decision']} |"
        )

    lines += [
        "",
        "Interpretation: CROMS-lite is valuable as a selector audit. Different objectives",
        "select different policies, especially when source-family robustness is made hard.",
        "That is Paper 4 evidence, not a full CROMS implementation and not a Paper 1",
        "promotion protocol.",
        "",
        "## Lane 5 - CVaR/OCE Tail Challenger",
        "",
        "| evidence block | candidate | primary metric | primary value | decision |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for _, row in l5.iterrows():
        lines.append(
            "| "
            f"{row['evidence_block']} | "
            f"{row['candidate_or_policy']} | "
            f"{row['primary_metric']} | "
            f"{_format_float(row['primary_value'])} | "
            f"{row['decision']} |"
        )

    lines += [
        "",
        "Interpretation: CVaR/OCE remains the strongest quantitative challenger lane.",
        "The tail evidence is real, but paired wealth dominance is absent, so it belongs",
        "in Paper 4 as a tail challenger appendix and, at most, in Paper Estrella as a",
        "robustness caveat.",
        "",
        "## Consolidated Decisions",
        "",
        "| lane | decision | Paper 4 destination | Paper 1 destination |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['lane']} | {row['decision']} | {row['paper4_destination']} | {row['paper1_destination']} |"
        )

    lines += [
        "",
        "## Stop Rules",
        "",
        "- Do not reopen the official champion from these lanes.",
        "- Do not treat source-holdout replay as live online validity.",
        "- Do not call CROMS-lite a full implementation of CROMS.",
        "- Do not call CVaR/OCE an economic champion unless paired wealth dominance is shown.",
    ]
    return "\n".join(lines) + "\n"


def run(date: str) -> OutputBundle:
    lane1 = build_lane1_crc_ltt_gates()
    lane2 = build_lane2_croms_lite_selection(lane1)
    lane5 = build_lane5_cvar_oce_challenger()
    summary = build_summary(lane1, lane2, lane5)
    memo = build_memo(lane1, lane2, lane5, summary)

    _write_csv(lane1, f"paper4_lab4_lane1_crc_ltt_decision_gates_{date}.csv")
    _write_csv(lane2, f"paper4_lab4_lane2_croms_lite_selection_{date}.csv")
    _write_csv(lane5, f"paper4_lab4_lane5_cvar_oce_challenger_{date}.csv")
    _write_csv(summary, f"paper4_lab4_lane_summary_{date}.csv")
    _write_text(memo, DOC_DIR / f"paper4_lab4_lane1_2_5_execution_memo_{date}.md")
    return OutputBundle(lane1=lane1, lane2=lane2, lane5=lane5, summary=summary, memo=memo)


def run_remaining(date: str) -> dict[str, pd.DataFrame | str]:
    lane3 = build_lane3_e2e_conformal_readiness()
    lane4 = build_lane4_online_multisource_conformal()
    lane6 = build_lane6_spo_dfl_comparator()
    lane7 = build_lane7_ifrs9_proxy()
    lane8 = build_lane8_governance_fairness_proxy()
    lane9 = build_lane9_causal_cate_boundary()
    summary = build_remaining_summary(lane3, lane4, lane6, lane7, lane8, lane9)
    memo = build_remaining_memo(lane3, lane4, lane6, lane7, lane8, lane9, summary)

    _write_csv(lane3, f"paper4_lab4_lane3_e2e_conformal_readiness_{date}.csv")
    _write_csv(lane4, f"paper4_lab4_lane4_online_multisource_conformal_{date}.csv")
    _write_csv(lane6, f"paper4_lab4_lane6_spo_dfl_comparator_{date}.csv")
    _write_csv(lane7, f"paper4_lab4_lane7_ifrs9_proxy_{date}.csv")
    _write_csv(lane8, f"paper4_lab4_lane8_governance_fairness_proxy_{date}.csv")
    _write_csv(lane9, f"paper4_lab4_lane9_causal_cate_boundary_{date}.csv")
    _write_csv(summary, f"paper4_lab4_remaining_lane_summary_{date}.csv")
    _write_text(memo, DOC_DIR / f"paper4_lab4_lane3_4_6_7_8_9_execution_memo_{date}.md")
    return {
        "lane3": lane3,
        "lane4": lane4,
        "lane6": lane6,
        "lane7": lane7,
        "lane8": lane8,
        "lane9": lane9,
        "summary": summary,
        "memo": memo,
    }


def build_selector_final_table(date: str) -> pd.DataFrame:
    lane2 = _read_csv(f"paper4_lab4_lane2_croms_lite_selection_{date}.csv")
    wanted = [
        "official_paper1_champion",
        "source_defended_return",
        "croms_lite_balanced_score",
        "coverage_only_source",
        "diagnostic_selector_rank",
    ]
    work = lane2[lane2["selection_strategy"].isin(wanted)].copy()
    work["paper4_use"] = np.select(
        [
            work["selection_strategy"].eq("official_paper1_champion"),
            work["selection_strategy"].eq("source_defended_return"),
            work["selection_strategy"].eq("croms_lite_balanced_score"),
            work["selection_strategy"].eq("coverage_only_source"),
        ],
        [
            "reference champion",
            "source-governance tradeoff",
            "CROMS-lite tradeoff evidence",
            "coverage-only negative control",
        ],
        default="diagnostic selector anchor",
    )
    work["paper1_use"] = np.where(
        work["selection_strategy"].eq("official_paper1_champion"),
        "official reference",
        "none now",
    )
    work["claim_boundary"] = np.where(
        work["selection_strategy"].eq("official_paper1_champion"),
        "Protected Paper Estrella champion.",
        "Paper 4 selector audit only; not a promotion protocol.",
    )
    cols = [
        "selection_strategy",
        "selected_policy_id",
        "realized_total_return",
        "delta_return_vs_official",
        "mdcp_worst_source_coverage",
        "delta_worst_source_coverage_vs_official",
        "weighted_miscoverage_V",
        "gamma_cp",
        "satisficing_pass_rate",
        "paper4_use",
        "paper1_use",
        "claim_boundary",
    ]
    order = {name: i for i, name in enumerate(wanted)}
    work["order"] = work["selection_strategy"].map(order)
    return work.sort_values("order")[cols]


def build_all_lane_summary(date: str) -> pd.DataFrame:
    first = _read_csv(f"paper4_lab4_lane_summary_{date}.csv")
    remaining = _read_csv(f"paper4_lab4_remaining_lane_summary_{date}.csv")
    all_lanes = pd.concat([first, remaining], ignore_index=True)
    all_lanes["final_sink"] = np.select(
        [
            all_lanes["decision"].str.contains("append", case=False, na=False),
            all_lanes["decision"].str.contains("park", case=False, na=False),
        ],
        ["paper4_appendix_or_governance", "parked_with_boundary"],
        default="paper4_lab_only",
    )
    all_lanes["paper1_export_decision"] = np.select(
        [
            all_lanes["paper1_destination"].str.contains(
                "reviewer|caveat|context", case=False, na=False
            ),
            all_lanes["paper1_destination"].str.contains("none", case=False, na=False),
        ],
        ["possible_limited_context_only", "no_export_now"],
        default="no_export_now",
    )
    return all_lanes


def build_all_lanes_memo(date: str, all_lanes: pd.DataFrame, selector: pd.DataFrame) -> str:
    append_lanes = all_lanes[all_lanes["final_sink"].eq("paper4_appendix_or_governance")]
    parked_lanes = all_lanes[all_lanes["final_sink"].eq("parked_with_boundary")]
    lines = [
        "# Paper 4 Lab 4 All-Lane Synthesis - 2026-05-18",
        "",
        "## Decision",
        "",
        "The literature-driven Lab 4 pass is closed as a Paper 4 living-lab synthesis.",
        "The work uses retained Paper 4 artifacts and source-review evidence, but it does",
        "not reopen the Paper Estrella champion and does not create a Paper 4 final",
        "promotion artifact.",
        "",
        f"- Append/governance lanes: `{len(append_lanes)}`.",
        f"- Parked lanes: `{len(parked_lanes)}`.",
        "- Paper Estrella export: limited to possible reviewer-defense, caveat or",
        "  model-risk context; no new champion claim.",
        "",
        "## Lane Decisions",
        "",
        "| lane | decision | final sink | Paper 1 export | key result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in all_lanes.iterrows():
        lines.append(
            "| "
            f"{row['lane']} | {row['decision']} | {row['final_sink']} | "
            f"{row['paper1_export_decision']} | {row['key_result']} |"
        )

    lines += [
        "",
        "## Selector Table",
        "",
        "The compact selector table is useful because it makes the CROMS-lite result",
        "readable: source robustness, balanced decision risk and official economic",
        "champion objectives choose different policies. This is exactly Paper 4",
        "material: tradeoff evidence, not a promotion protocol.",
        "",
        "| selector | selected policy | delta return | delta worst source | Paper 4 use |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for _, row in selector.iterrows():
        lines.append(
            "| "
            f"{row['selection_strategy']} | {row['selected_policy_id']} | "
            f"{_format_float(row['delta_return_vs_official'], 2)} | "
            f"{_format_float(row['delta_worst_source_coverage_vs_official'])} | "
            f"{row['paper4_use']} |"
        )

    lines += [
        "",
        "## What Enters Paper 4",
        "",
        "- Lane 1 enters only as governance: CRC/LTT-style gates document what passes",
        "  risk, return, satisficing and source-hardening checks.",
        "- Lane 2 enters as a selector tradeoff table, because it shows why objective",
        "  choice matters and why no selector automatically promotes a new champion.",
        "- Lane 5 enters as a strong tail-risk appendix: CVaR/OCE improves tail views",
        "  but does not beat paired wealth robustly.",
        "- Lane 7 enters as an IFRS9-inspired SICR/ECL proxy appendix, explicitly not",
        "  contractual IFRS9.",
        "- Lane 8 enters as source/proxy governance, explicitly not legal fair-lending",
        "  evidence.",
        "",
        "## What Stays Parked",
        "",
        "- Lane 3 is parked because there is no end-to-end learned conformal calibration",
        "  artifact and the main environment is not the isolated differentiable stack.",
        "- Lane 4 is parked as a live/online claim because defended coverage is close",
        "  but below gate and strict holdouts do not survive universally.",
        "- Lane 6 is parked for integrated DFL/SPO+ because current evidence is",
        "  oracle-regret/surrogate only.",
        "- Lane 9 is parked because CATE/policy value remains blocked by accepted-loan",
        "  selection, weak overlap and unresolved identification.",
        "",
        "## Stop Rules",
        "",
        "- Do not create `paper4_final_promotion.json`.",
        "- Do not create `paper4_v###` follow-up waves from this synthesis.",
        "- Reopen a parked lane only with new data, an isolated working prototype, a",
        "  formal proof, or a reviewer request that changes a manuscript claim.",
    ]
    return "\n".join(lines) + "\n"


def run_synthesis(date: str) -> dict[str, pd.DataFrame | str]:
    selector = build_selector_final_table(date)
    all_lanes = build_all_lane_summary(date)
    memo = build_all_lanes_memo(date, all_lanes, selector)
    _write_csv(selector, f"paper4_lab4_selector_final_table_{date}.csv")
    _write_csv(all_lanes, f"paper4_lab4_all_lane_summary_{date}.csv")
    _write_text(memo, DOC_DIR / f"paper4_lab4_all_lanes_synthesis_{date}.md")
    return {"selector": selector, "all_lanes": all_lanes, "memo": memo}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run semantic Lab 4 lanes over retained Paper 4 artifacts."
    )
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--group",
        choices=["first", "remaining", "all"],
        default="first",
        help="first runs lanes 1/2/5; remaining runs lanes 3/4/6/7/8/9; all runs both.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.group in {"first", "all"}:
        bundle = run(args.date)
        print(
            textwrap.dedent(
                f"""
                Wrote Lab 4 lane outputs for {args.date}
                lane1 rows: {len(bundle.lane1)}
                lane2 rows: {len(bundle.lane2)}
                lane5 rows: {len(bundle.lane5)}
                summary rows: {len(bundle.summary)}
                """
            ).strip()
        )
    if args.group in {"remaining", "all"}:
        remaining = run_remaining(args.date)
        print(
            textwrap.dedent(
                f"""
                Wrote remaining Lab 4 lane outputs for {args.date}
                lane3 rows: {len(remaining["lane3"])}
                lane4 rows: {len(remaining["lane4"])}
                lane6 rows: {len(remaining["lane6"])}
                lane7 rows: {len(remaining["lane7"])}
                lane8 rows: {len(remaining["lane8"])}
                lane9 rows: {len(remaining["lane9"])}
                summary rows: {len(remaining["summary"])}
                """
            ).strip()
        )
    if args.group == "all":
        synthesis = run_synthesis(args.date)
        print(
            textwrap.dedent(
                f"""
                Wrote all-lane Lab 4 synthesis for {args.date}
                selector rows: {len(synthesis["selector"])}
                all-lane rows: {len(synthesis["all_lanes"])}
                """
            ).strip()
        )


if __name__ == "__main__":
    main()
