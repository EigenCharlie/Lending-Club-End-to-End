"""Build Paper 4 v22 academic synthesis and working champion registry."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
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

SCHEMA_VERSION = "2026-05-14.22"


def _best_working_champion() -> dict[str, Any]:
    summary = _safe_read_csv(TABLE_DIR / "paper4_v20_dynamic_policy_summary.csv")
    prior = _safe_read_json(STATUS_DIR / "paper4_v18_working_champion.json")
    memo = _safe_read_csv(TABLE_DIR / "paper4_v20_champion_decision_memo.csv")
    if summary.empty:
        return {
            "policy_id": prior.get("policy_id", "paper1_economic_champion"),
            "selection_status": "retained_v18_due_missing_v20_summary",
            "score": 0.0,
        }
    df = summary.copy()
    score_col = (
        "paper4_champion_score_v15"
        if "paper4_champion_score_v15" in df
        else "dynamic_value_score_v15"
    )
    df["claim_safe_v22"] = (
        df["no_temporal_leakage_rate"].ge(1.0)
        & df.get("online_gate_pass_v15", True).astype(bool)
        & df.get("paper4_working_only", True).astype(bool)
    )
    df["full_governance_score_v22"] = np.where(
        df["claim_safe_v22"],
        pd.to_numeric(df[score_col], errors="coerce").fillna(-1.0),
        -1.0,
    )
    top = df.sort_values("full_governance_score_v22", ascending=False).iloc[0]
    memo_recommends_change = (
        bool(memo["working_champion_change_recommended_v20"].iloc[0]) if not memo.empty else False
    )
    evidence_split = not memo_recommends_change and str(top["policy_id"]) != str(
        prior.get("policy_id", "")
    )
    selected = top
    selection_status = "selected_by_v22_full_governance_score"
    if evidence_split:
        prior_rows = df[df["policy_id"].eq(str(prior.get("policy_id", "")))]
        if not prior_rows.empty:
            selected = prior_rows.iloc[0]
            selection_status = "retained_v18_due_evidence_split_in_v20_memo"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": str(selected["policy_id"]),
        "previous_working_champion_policy_id_v18": str(prior.get("policy_id", "")),
        "champion_changed_vs_v18": str(selected["policy_id"]) != str(prior.get("policy_id", "")),
        "scope": "paper4_working_champion_only",
        "selection_status": selection_status,
        "full_governance_score_v22": float(selected["full_governance_score_v22"]),
        "highest_score_challenger_policy_id": str(top["policy_id"]),
        "highest_score_challenger_score_v22": float(top["full_governance_score_v22"]),
        "v20_memo_recommends_change": memo_recommends_change,
        "evidence_split_vs_highest_score_challenger": evidence_split,
        "final_wealth_mean": float(selected["final_wealth_mean"]),
        "final_wealth_p05": float(selected["final_wealth_p05"]),
        "cumulative_losses_p95": float(selected["cumulative_losses_p95"]),
        "no_temporal_leakage_rate": float(selected["no_temporal_leakage_rate"]),
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "contractual_ifrs9_claim_allowed": False,
        "cate_policy_value_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "caveat": "Paper 4 lab champion only; if score and paired robustness disagree, v22 retains the prior working champion and records the challenger",
    }


def _blocker_dashboard() -> pd.DataFrame:
    rows = [
        (
            "dynamic_engine_v2",
            "resolved",
            "v19 dynamic trace, horizons and path-family sensitivity exist",
            "expand to 128/256/512 only when manuscript needs tighter intervals",
        ),
        (
            "sample_paths_v2",
            "near_resolved_with_plateau",
            "v19 internal paths include vintage/cohort/default-dependence/LGD/prepayment and optional FRED context",
            "external forecast validation remains outside static dataset",
        ),
        (
            "dla_endogenous_solver",
            "near_resolved_with_plateau",
            "v20 endogenous monthly greedy DLA approximation implemented",
            "exact Bellman/ADP proof remains future method work",
        ),
        (
            "cvar_full_universe",
            "near_resolved_with_plateau",
            "v20 preserves strict/committee/relaxed restricted-master evidence and certificates",
            "no full-universe exact optimality claim",
        ),
        (
            "spo_dfl",
            "dependency_blocked",
            "v20 CatBoost/sklearn regret surrogate and oracle diagnostics implemented",
            "formal differentiable SPO+ blocked by cvxpy/cvxpylayers/torch environment",
        ),
        (
            "champion_decomposition",
            "resolved",
            "v20 overlap, selected-vs-avoided and decision memo exist",
            "case studies are audit narratives, not causal explanations",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "v21 field audit and SICR sensitivity exist",
            "contractual IFRS9 blocked by missing servicing/DPD/cure/recovery/prepayment/EAD/macros",
        ),
        (
            "cate_policy_value",
            "theory_blocked",
            "v21 balance/overlap/falsification/sensitivity diagnostics exist",
            "accepted-loan selection/reject-inference remain unresolved",
        ),
        (
            "fair_lending",
            "prohibited_claim",
            "v21 proxy governance and no-claim flags exist",
            "no protected attributes or approved external protocol",
        ),
        (
            "paper1_freeze",
            "resolved",
            "promotion JSON remains protected",
            "continue Paper 4 working-only champions",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["blocker_id", "status_v22", "current_diagnosis", "next_action"]
    )


def _claim_matrix() -> pd.DataFrame:
    rows = [
        (
            "Paper 4 can evaluate policies as monthly dynamic processes",
            True,
            "paper4_v19_dynamic_policy_summary.csv",
            "19bh-v19-dynamic-engine-v2.qmd",
            "internal replay, not deployment",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has calibrated internal common sample paths",
            True,
            "paper4_v19_sample_paths.parquet",
            "19bh-v19-dynamic-engine-v2.qmd",
            "internal calibration, not forecast",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has endogenous DLA approximations",
            True,
            "paper4_v20_endogenous_dla_policy_summary.csv",
            "19bi-v20-dla-cvar-spo-resolution.qmd",
            "approximate monthly greedy DLA, not Bellman proof",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has exact full-universe CVaR optimality",
            False,
            "paper4_v20_cvar_oce_frontier_v2.csv",
            "19bi-v20-dla-cvar-spo-resolution.qmd",
            "restricted-master/diagnostic evidence only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has formal differentiable SPO+",
            False,
            "paper4_v20_spo_dependency_blockers.csv",
            "19bi-v20-dla-cvar-spo-resolution.qmd",
            "decision-regret surrogate only unless dependencies pass",
            False,
            False,
            False,
        ),
        (
            "Paper 4 can select a working lab champion",
            True,
            "paper4_v22_working_champion.json",
            "19bk-v22-academic-synthesis.qmd",
            "Paper 4 only; no Paper Estrella promotion",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has contractual IFRS9 lifetime ECL",
            False,
            "paper4_v21_ifrs9_readiness_matrix.csv",
            "19bj-v21-ifrs9-causal-fairness-gates.qmd",
            "servicing data blockers remain",
            True,
            False,
            False,
        ),
        (
            "Paper 4 has CATE policy value",
            False,
            "paper4_v21_cate_gate_report.csv",
            "19bj-v21-ifrs9-causal-fairness-gates.qmd",
            "identification/reject-inference blocked",
            False,
            True,
            False,
        ),
        (
            "Paper 4 makes fair-lending legal claims",
            False,
            "paper4_v21_no_legal_claim_flags.csv",
            "19bj-v21-ifrs9-causal-fairness-gates.qmd",
            "protected attributes/protocol absent",
            False,
            False,
            True,
        ),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "claim",
            "allowed",
            "artifact",
            "quarto_page",
            "claim_boundary_v22",
            "no_claim_contractual_ifrs9",
            "no_claim_cate_policy_value",
            "no_claim_fair_lending_legal",
        ],
    )
    df["artifact_exists"] = df["artifact"].map(
        lambda name: (TABLE_DIR / name).exists() or (STATUS_DIR / name).exists()
    )
    return df


def _contribution_map() -> pd.DataFrame:
    rows = [
        (
            "Dynamic sequential governance",
            "paper4_v19_dynamic_policy_trace.parquet",
            "core_defensible",
            "turns funded books into comparable monthly processes",
        ),
        (
            "Internal path calibration and paired CIs",
            "paper4_v19_policy_pairwise_common_path_ci.csv",
            "core_defensible",
            "supports champion-vs-challenger uncertainty statements",
        ),
        (
            "Endogenous DLA approximations",
            "paper4_v20_endogenous_dla_decisions.parquet",
            "promising_method",
            "moves from adapters to monthly loan selection, still approximate",
        ),
        (
            "CVaR strict infeasibility/committee frontier",
            "paper4_v20_cvar_strict_infeasibility_v2.csv",
            "negative_result_appendix",
            "documents why strict governance can be too tight",
        ),
        (
            "SPO oracle-regret surrogate",
            "paper4_v20_spo_oracle_regret.csv",
            "promising_but_not_formal",
            "decision-loss evidence without differentiable optimizer",
        ),
        (
            "Champion economic decomposition",
            "paper4_v20_champion_decomposition_summary.csv",
            "interpretability_contribution",
            "explains selected/avoided risk pockets",
        ),
        (
            "IFRS9 readiness boundary",
            "paper4_v21_ifrs9_readiness_matrix.csv",
            "data_blocked_boundary",
            "prevents overclaiming contractual IFRS9",
        ),
        (
            "CATE blocker with diagnostics",
            "paper4_v21_cate_gate_report.csv",
            "theory_blocked_boundary",
            "keeps causal claims honest",
        ),
        (
            "Fairness proxy-only protocol",
            "paper4_v21_fairness_proxy_only_protocol.csv",
            "governance_boundary",
            "separates source governance from legal fairness",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "finding",
            "primary_artifact",
            "publishability_class",
            "contribution_interpretation",
        ],
    )


def _triage() -> pd.DataFrame:
    rows = [
        (
            "journal_core",
            "sequential decision governance with CRPTO/CVaR/DLA under common paths",
            "viable after tightening intervals and simplifying scope",
            "main future paper candidate",
        ),
        (
            "journal_appendix",
            "strict CVaR infeasibility and committee-relaxed frontier",
            "publishable as negative/governance result",
            "appendix or robustness section",
        ),
        (
            "method_extension",
            "SPO/DFL decision-regret surrogate",
            "needs formal optimizer or clear non-differentiable framing",
            "do not claim formal SPO+ yet",
        ),
        (
            "lab_notebook",
            "IFRS9 contractual proxy panel",
            "useful but not contractual",
            "keep as IFRS9-inspired ECL only",
        ),
        (
            "blocked",
            "CATE policy value",
            "not publishable as causal policy without new identification design",
            "blocked by theory/data",
        ),
        ("prohibited_claim", "fair-lending legal claim", "not allowed", "proxy governance only"),
    ]
    return pd.DataFrame(
        rows, columns=["triage_bucket", "lane", "current_publishability", "decision"]
    )


def build_v22() -> dict[str, Any]:
    start = time.time()
    champion = _best_working_champion()
    blockers = _blocker_dashboard()
    claims = _claim_matrix()
    contributions = _contribution_map()
    triage = _triage()
    _write_json("paper4_v22_working_champion.json", champion)
    _write_csv("paper4_v22_blocker_dashboard.csv", blockers)
    _write_csv("paper4_v22_claim_artifact_matrix.csv", claims)
    _write_csv("paper4_v22_academic_contribution_map.csv", contributions)
    _write_csv("paper4_v22_publishability_triage.csv", triage)
    _write_csv("paper4_v22_claim_boundaries_final_for_lab.csv", claims)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v22_academic_synthesis",
        "working_champion_policy_id_v22": champion["policy_id"],
        "champion_changed_vs_v18": champion.get("champion_changed_vs_v18", False),
        "blocker_rows_v22": int(len(blockers)),
        "claim_rows_v22": int(len(claims)),
        "all_claim_artifacts_exist_v22": bool(claims["artifact_exists"].all()),
        "contribution_rows_v22": int(len(contributions)),
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "Paper 4 lab synthesis only; working champion is not a final promotion",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v22_status.json", status)
    _write_note(
        "paper4_v22_academic_synthesis.md",
        "\n".join(
            [
                "# Paper 4 v22 Academic Synthesis",
                "",
                f"- Working champion: `{status['working_champion_policy_id_v22']}`.",
                f"- Champion changed vs v18: `{status['champion_changed_vs_v18']}`.",
                f"- Claims with artifacts: `{status['all_claim_artifacts_exist_v22']}`.",
                f"- Remaining blocker rows: `{status['blocker_rows_v22']}`.",
                "",
                "This is the v19-v22 lab synthesis. It does not promote anything to Paper Estrella.",
            ]
        ),
    )
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args(list(argv) if argv is not None else None)
    status = build_v22()
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
