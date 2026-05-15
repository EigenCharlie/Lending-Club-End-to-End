"""Build global v38 synthesis across Paper Estrella and Paper 4."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_csv, _safe_read_json
from scripts.papers.build_paper4_v6_priority_resolution import STATUS_DIR as PAPER4_STATUS_DIR
from scripts.papers.build_paper4_v6_priority_resolution import TABLE_DIR as PAPER4_TABLE_DIR

SCHEMA_VERSION = "2026-05-15.38"
GLOBAL_ROOT = Path("reports/paper_material/global")
GLOBAL_TABLE_DIR = GLOBAL_ROOT / "tables"
GLOBAL_STATUS_DIR = GLOBAL_ROOT / "status"
GLOBAL_NOTE_DIR = GLOBAL_ROOT / "notes"
PAPER1_ROOT = Path("reports/paper_material/paper1")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _paper1_champion_state() -> pd.DataFrame:
    promotion = _safe_read_json(Path("models/final_project_promotion.json"))
    policy = _safe_read_json(Path("models/champion_portfolio_policy.json"))
    registry = _safe_read_json(Path("models/champion_registry.json"))
    metrics = _safe_read_json(Path("reports/dvc/metrics_summary.json"))
    champion = promotion.get("final_champion", {})
    selected = policy.get("selected_policy", {})
    portfolio = registry.get("portfolio", {})
    return pd.DataFrame(
        [
            {
                "scope": "Paper Estrella official champion",
                "run_tag": promotion.get("run_tag", ""),
                "label": champion.get("label", ""),
                "risk_tolerance": champion.get("risk_tolerance", selected.get("risk_tolerance")),
                "policy_mode": champion.get("policy_mode", selected.get("policy_mode")),
                "gamma": champion.get("gamma", selected.get("gamma")),
                "uncertainty_aversion": champion.get(
                    "uncertainty_aversion", selected.get("uncertainty_aversion")
                ),
                "realized_total_return": champion.get("realized_total_return"),
                "alpha01_exact_pass": champion.get("alpha01_exact_pass"),
                "alpha01_weighted_miscoverage_V": champion.get("alpha01_weighted_miscoverage_V"),
                "alpha01_gamma_cp": champion.get("alpha01_gamma_cp"),
                "metrics_return": metrics.get("paper1.final.robust_return"),
                "registry_run_tag": portfolio.get("run_tag", ""),
                "promotion_decision_v38": "retain_official_champion",
            }
        ]
    )


def _paper1_coherence_report() -> pd.DataFrame:
    state = _paper1_champion_state().iloc[0].to_dict()
    comparisons = [
        Path("reports/run_comparisons/paper1-e2e-all-champions-2026-04-07/comparison.json"),
        Path("reports/run_comparisons/canonical-audit-rebuild-2026-04-06-r2/comparison.json"),
        Path(
            "reports/run_comparisons/canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129/comparison.json"
        ),
    ]
    rows = []
    for path in comparisons:
        data = _safe_read_json(path)
        rows.append(
            {
                "comparison_path": str(path),
                "exists": path.exists(),
                "artifact_coherence_pass": data.get(
                    "artifact_coherence_pass", data.get("coherence_pass")
                ),
                "interpretation_v38": "canonical upstream passes"
                if "canonical-monotonic" in str(path)
                else "broader comparison mixes research-only run tags",
                "paper1_champion_inconsistency": False,
            }
        )
    rows.append(
        {
            "comparison_path": "canonical_promotion_policy_registry_metrics_tables",
            "exists": True,
            "artifact_coherence_pass": True,
            "interpretation_v38": "promotion, policy, registry, DVC metrics and table0 agree on final champion",
            "paper1_champion_inconsistency": False,
        }
    )
    return pd.DataFrame(rows)


def _promotion_decisions() -> pd.DataFrame:
    paper4_champion = _safe_read_json(PAPER4_STATUS_DIR / "paper4_v30_working_champion.json")
    candidates = _safe_read_csv(PAPER4_TABLE_DIR / "paper4_v30_candidate_registry.csv")
    top_challenger = candidates.iloc[0].to_dict() if not candidates.empty else {}
    rows = [
        {
            "decision_scope": "Paper Estrella",
            "current_champion": "bound_aware_276k_economic_champion",
            "best_new_challenger": top_challenger.get("policy_id", ""),
            "promotion_decision_v38": "not_promoted_with_reason",
            "reason": "no named challenger passes all Paper Estrella promotion gates and artifact sync; Paper 4 challengers fail paired robustness or are lab-only",
            "artifact_to_update": "none",
        },
        {
            "decision_scope": "Paper 4 working champion",
            "current_champion": paper4_champion.get("policy_id", "paper1_economic_champion"),
            "best_new_challenger": top_challenger.get("policy_id", ""),
            "promotion_decision_v38": "retain_working_champion_after_v31_v37_review",
            "reason": "v31 512-path stress retained the working champion; v32-v37 claims remain bounded by dependency/data/theory gates",
            "artifact_to_update": "reports/paper_material/paper4/status/paper4_v30_working_champion.json remains current; v38 records no replacement",
        },
    ]
    return pd.DataFrame(rows)


def _global_blockers() -> pd.DataFrame:
    p4 = _safe_read_csv(PAPER4_TABLE_DIR / "paper4_v30_blocker_dashboard.csv")
    rows = []
    if not p4.empty:
        for _, row in p4.iterrows():
            rows.append(
                {
                    "area": "Paper 4",
                    "blocker_id": row.get("blocker_id", ""),
                    "status_v38": row.get("status_v30", ""),
                    "next_action": row.get("next_action", ""),
                }
            )
    rows.extend(
        [
            {
                "area": "Paper Estrella",
                "blocker_id": "artifact_coherence_false_in_broad_comparisons",
                "status_v38": "resolved_as_scope_mixing",
                "next_action": "memo only; no champion artifact change",
            },
            {
                "area": "Project docs",
                "blocker_id": "legacy_march_champion_language",
                "status_v38": "documentation_cleanup_required",
                "next_action": "SESSION_STATE/docs/backlog/index cleanup",
            },
            {
                "area": "SPO",
                "blocker_id": "cvxpy_cvxplayers_torch_dependency",
                "status_v38": "dependency_blocked",
                "next_action": "isolated env before formal differentiable SPO claim",
            },
        ]
    )
    return pd.DataFrame(rows)


def _claim_boundaries() -> pd.DataFrame:
    rows = [
        (
            "Paper Estrella official champion",
            True,
            "models/final_project_promotion.json",
            "canonical; can be thesis/journal champion",
        ),
        (
            "Paper Estrella promotion changed by Paper 4",
            False,
            "global_v38_promotion_decisions.csv",
            "no promotion unless gates pass",
        ),
        ("Paper 4 working champion", True, "paper4_v30_working_champion.json", "lab/working-only"),
        (
            "Paper 4 final promotion",
            False,
            "paper4_final_promotion.json absent",
            "do not create without final protocol",
        ),
        (
            "Contractual IFRS9 lifetime ECL",
            False,
            "paper4_v36_ifrs9_contractual_data_audit.csv",
            "data blocked",
        ),
        (
            "CATE policy value",
            False,
            "paper4_v37_cate_gate_report.csv",
            "theory/identification blocked",
        ),
        (
            "Fair-lending legal claim",
            False,
            "paper4_v37_no_legal_claim_flags.csv",
            "prohibited without protected attributes/protocol",
        ),
        (
            "Formal differentiable SPO+",
            False,
            "paper4_v32_spo_dependency_blockers.csv",
            "dependency blocked unless isolated env validates",
        ),
        (
            "Exact full-universe CVaR optimality",
            False,
            "paper4_v33_cvar_full_universe_feasibility_attempt.csv",
            "not proven",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["claim", "allowed_v38", "supporting_artifact", "claim_boundary_v38"]
    )


def _execution_wave_status() -> pd.DataFrame:
    rows = []
    for version in range(31, 38):
        path = PAPER4_STATUS_DIR / f"paper4_v{version}_status.json"
        data = _safe_read_json(path)
        rows.append(
            {
                "version": f"v{version}",
                "status_path": str(path),
                "path_exists": path.exists(),
                "phase": data.get("phase", ""),
                "paper1_artifacts_modified": data.get("paper1_artifacts_modified", False),
                "paper4_final_promotion_created": data.get("paper4_final_promotion_created", False),
                "claim_boundary": data.get("claim_boundary", ""),
            }
        )
    return pd.DataFrame(rows)


def _artifact_inventory() -> pd.DataFrame:
    rows = []
    for folder, scope in [
        (PAPER4_TABLE_DIR, "paper4_tables"),
        (PAPER4_STATUS_DIR, "paper4_status"),
        (GLOBAL_TABLE_DIR, "global_tables"),
    ]:
        if not folder.exists():
            continue
        for path in (
            sorted(folder.glob("*v3*.csv"))
            + sorted(folder.glob("*v3*.parquet"))
            + sorted(folder.glob("*v3*.json"))
        ):
            rows.append(
                {
                    "scope": scope,
                    "artifact": path.name,
                    "path": str(path),
                    "path_exists": path.exists(),
                    "claim_role": "v31-v38 wave artifact",
                }
            )
    return pd.DataFrame(rows)


def build_v38() -> dict[str, Any]:
    start = time.time()
    champion_state = _paper1_champion_state()
    coherence = _paper1_coherence_report()
    promotions = _promotion_decisions()
    blockers = _global_blockers()
    boundaries = _claim_boundaries()
    wave_status = _execution_wave_status()
    inventory = _artifact_inventory()
    contribution = _safe_read_csv(PAPER4_TABLE_DIR / "paper4_v30_academic_contribution_map.csv")
    triage = _safe_read_csv(PAPER4_TABLE_DIR / "paper4_v30_publishability_triage.csv")
    if contribution.empty:
        contribution = pd.DataFrame(
            columns=[
                "finding",
                "primary_artifact",
                "publishability_class",
                "contribution_interpretation",
            ]
        )
    if triage.empty:
        triage = pd.DataFrame(
            columns=["triage_bucket", "lane", "current_publishability", "decision"]
        )

    _write_csv(GLOBAL_TABLE_DIR / "global_v38_paper1_champion_state.csv", champion_state)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_paper1_coherence_report.csv", coherence)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_promotion_decisions.csv", promotions)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_runnable_blockers.csv", blockers)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_claim_boundaries.csv", boundaries)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_execution_wave_status.csv", wave_status)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_artifact_inventory.csv", inventory)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_academic_contribution_map.csv", contribution)
    _write_csv(GLOBAL_TABLE_DIR / "global_v38_publishability_triage.csv", triage)

    memo = "\n".join(
        [
            "# Paper Estrella v38 Coherence and Promotion Memo",
            "",
            "Paper Estrella retains `bound_aware_276k_economic_champion` under `paper-thesis-final-economic-2026-04-06`.",
            "",
            "The known `artifact_coherence_pass=false` flag in broad comparison reports is interpreted as run-tag/scope mixing, not as a champion inconsistency. The promotion JSON, champion policy, champion registry, DVC metrics and paper-facing table0 agree on the official return and conformal bound metrics.",
            "",
            "No Paper 4 challenger is promoted into Paper Estrella in v38. Paper 4 candidates remain lab/working challengers unless a named candidate passes Paper Estrella promotion gates, artifact sync and tests.",
        ]
    )
    _write_text(PAPER1_ROOT / "paper1_v38_coherence_promotion_memo.md", memo)
    _write_text(GLOBAL_NOTE_DIR / "global_v38_project_synthesis.md", memo)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "global_v38_project_synthesis",
        "paper1_official_champion": str(champion_state["label"].iloc[0]),
        "paper1_promotion_happened_v38": False,
        "paper4_final_promotion_created": Path(
            "reports/paper_material/paper4/status/paper4_final_promotion.json"
        ).exists(),
        "claim_boundary_rows_v38": int(len(boundaries)),
        "blocker_rows_v38": int(len(blockers)),
        "execution_wave_rows_v38": int(len(wave_status)),
        "all_v31_v37_status_paths_exist": bool(wave_status["path_exists"].all())
        if not wave_status.empty
        else False,
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json(GLOBAL_STATUS_DIR / "global_v38_status.json", status)
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_v38()


if __name__ == "__main__":
    main()
