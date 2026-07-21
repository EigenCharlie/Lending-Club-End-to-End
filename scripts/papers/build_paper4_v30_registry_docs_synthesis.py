"""Build Paper 4 v30 registry, blocker dashboard and synthesis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
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

SCHEMA_VERSION = "2026-05-14.30"
BOOK_DIR = Path("book/chapters/19-paper-mega-extension")


def _file_hash(path: Path, max_bytes: int = 4_000_000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def _shape(path: Path) -> tuple[int | None, int | None]:
    try:
        if path.suffix == ".csv":
            head = pd.read_csv(path, nrows=5)
            rows = max(sum(1 for _ in path.open("rb")) - 1, 0)
            return rows, len(head.columns)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return len(df), len(df.columns)
        if path.suffix == ".json":
            return 1, None
    except Exception:
        return None, None
    return None, None


def _score01(series: pd.Series, *, high_is_good: bool) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    pct = x.rank(method="average", pct=True, ascending=True)
    return pct.fillna(0.5) if high_is_good else (1.0 - pct + 1.0 / len(x)).fillna(0.5)


def _artifact_registry() -> pd.DataFrame:
    page_map = {
        "v23": "19bl-v23-dynamic-scale-paths.qmd",
        "v24": "19bm-v24-dla-cvar-spo-upgrade.qmd",
        "v25": "19bn-v25-ifrs9-causal-fairness-upgrade.qmd",
        "v26": "19bo-v26-registry-and-synthesis.qmd",
        "v27": "19bp-v27-dynamic-scale-and-champion-stress.qmd",
        "v28": "19bq-v28-cvar-dla-spo-upgrade.qmd",
        "v29": "19br-v29-ifrs9-causal-fairness-source.qmd",
        "v30": "19bs-v30-registry-and-synthesis.qmd",
    }
    rows = []
    for folder in [TABLE_DIR, STATUS_DIR]:
        for path in (
            sorted(folder.glob("paper4_v*.csv"))
            + sorted(folder.glob("paper4_v*.parquet"))
            + sorted(folder.glob("paper4_v*.json"))
        ):
            name = path.name
            version = next(
                (v for v in page_map if f"_{v}_" in name or name.startswith(f"paper4_{v}_")),
                "unknown",
            )
            n_rows, n_cols = _shape(path)
            rows.append(
                {
                    "artifact": name,
                    "version": version,
                    "path": str(path),
                    "rows": n_rows,
                    "columns": n_cols,
                    "sha16": _file_hash(path),
                    "source_script": f"scripts/papers/build_paper4_{version}_*.py"
                    if version != "unknown"
                    else "unknown",
                    "linked_claim": "see paper4_v30_claim_artifact_matrix.csv",
                    "quarto_page": page_map.get(version, ""),
                    "status": "implemented" if path.exists() else "missing",
                    "caveat": "Paper 4 lab artifact; use per-claim boundaries",
                    "path_exists": path.exists(),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("artifact")


def _candidate_registry() -> pd.DataFrame:
    summary = _safe_read_csv(TABLE_DIR / "paper4_v28_dynamic_combined_summary.csv")
    if summary.empty:
        summary = _safe_read_csv(TABLE_DIR / "paper4_v27_dynamic_policy_summary.csv")
    if summary.empty:
        return pd.DataFrame()
    pairwise = _safe_read_csv(TABLE_DIR / "paper4_v27_policy_pairwise_common_path_ci.csv")
    d = summary.copy()
    if not pairwise.empty:
        d = d.merge(
            pairwise[
                [
                    "policy_id",
                    "mean_wealth_diff",
                    "prob_higher_wealth",
                    "mean_loss_diff",
                    "prob_lower_loss",
                ]
            ],
            on="policy_id",
            how="left",
        )
    for col in [
        "final_wealth_mean",
        "final_wealth_p05",
        "cumulative_losses_p95",
        "ECL_final_mean",
        "source_exposure_weak_share_final_mean",
        "cumulative_defaults_mean",
        "no_temporal_leakage_rate",
        "mean_wealth_diff",
        "prob_higher_wealth",
    ]:
        if col not in d:
            d[col] = np.nan
    d["claim_safety_gate"] = (
        pd.to_numeric(d["no_temporal_leakage_rate"], errors="coerce").fillna(0.0).ge(1.0)
    )
    d["paired_robustness_gate_v30"] = pd.to_numeric(
        d["prob_higher_wealth"], errors="coerce"
    ).fillna(0.0).ge(0.60) & pd.to_numeric(d["mean_wealth_diff"], errors="coerce").fillna(-1.0).gt(
        0.0
    )
    d["auditability_score_proxy_v30"] = (
        0.35 * _score01(d["cumulative_losses_p95"], high_is_good=False)
        + 0.25 * _score01(d["source_exposure_weak_share_final_mean"], high_is_good=False)
        + 0.20 * _score01(d["ECL_final_mean"], high_is_good=False)
        + 0.20 * d["claim_safety_gate"].astype(float)
    )
    d["full_candidate_score_v30"] = (
        0.30 * _score01(d["final_wealth_mean"], high_is_good=True)
        + 0.18 * _score01(d["final_wealth_p05"], high_is_good=True)
        + 0.18 * _score01(d["cumulative_losses_p95"], high_is_good=False)
        + 0.16 * d["auditability_score_proxy_v30"]
        + 0.10 * _score01(d["prob_higher_wealth"].fillna(0.0), high_is_good=True)
        + 0.08 * _score01(d["cumulative_defaults_mean"], high_is_good=False)
    )
    current = str(
        _safe_read_json(STATUS_DIR / "paper4_v26_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    d["decision_v30"] = "review"
    d.loc[
        d["full_candidate_score_v30"].lt(d["full_candidate_score_v30"].quantile(0.25)),
        "decision_v30",
    ] = "park"
    d.loc[~d["claim_safety_gate"].astype(bool), "decision_v30"] = "kill"
    d.loc[d["paired_robustness_gate_v30"] & d["claim_safety_gate"], "decision_v30"] = (
        "serious_challenger"
    )
    d.loc[d["policy_id"].eq("v13_spo_regret_audit_guarded"), "decision_v30"] = "serious_challenger"
    d.loc[d["policy_id"].eq(current), "decision_v30"] = "retain_working_champion"
    keep = [
        "policy_id",
        "final_wealth_mean",
        "final_wealth_p05",
        "cumulative_losses_p95",
        "ECL_final_mean",
        "source_exposure_weak_share_final_mean",
        "mean_wealth_diff",
        "prob_higher_wealth",
        "auditability_score_proxy_v30",
        "full_candidate_score_v30",
        "claim_safety_gate",
        "paired_robustness_gate_v30",
        "decision_v30",
    ]
    return (
        d[[c for c in keep if c in d.columns]]
        .drop_duplicates("policy_id")
        .sort_values("full_candidate_score_v30", ascending=False)
    )


def _working_champion(registry: pd.DataFrame) -> dict[str, Any]:
    prior = _safe_read_json(STATUS_DIR / "paper4_v26_working_champion.json")
    current = str(prior.get("policy_id", "paper1_economic_champion"))
    selected = current
    challenger = ""
    changed = False
    if not registry.empty:
        top = registry.iloc[0]
        challenger = str(top["policy_id"])
        if (
            challenger != current
            and bool(top.get("paired_robustness_gate_v30", False))
            and bool(top.get("claim_safety_gate", False))
        ):
            selected = challenger
            changed = True
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": selected,
        "previous_working_champion_policy_id_v26": current,
        "highest_score_challenger_policy_id_v30": challenger,
        "champion_changed_vs_v26": changed,
        "scope": "paper4_working_champion_only",
        "selection_status": "paper4_working_champion_changed"
        if changed
        else "retain_v26_pending_paired_challenger_robustness",
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "contractual_ifrs9_claim_allowed": False,
        "cate_policy_value_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "caveat": "Paper 4 lab champion only; no Paper Estrella promotion",
    }


def _blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "dynamic_engine_scale",
                "resolved",
                "v27 larger-path replay and champion stress implemented",
                "512 paths only if manuscript requires narrower intervals",
            ),
            (
                "champion_stress",
                "resolved",
                "direct CVaR challenger stress memo implemented",
                "use paired robustness gate for champion changes",
            ),
            (
                "sample_paths_v4",
                "near_resolved_with_plateau",
                "v29 internal path calibration plus macro cache/failure registry",
                "external forecast validation remains out of scope",
            ),
            (
                "dla_adp_rollout",
                "near_resolved_with_plateau",
                "v28 ADP rollout v2 and underperformance diagnosis implemented",
                "exact Bellman proof remains future work",
            ),
            (
                "cvar_column_generation",
                "near_resolved_with_plateau",
                "v28 column-generation v2 and certificate implemented",
                "no exact full-universe optimality claim",
            ),
            (
                "spo_dfl",
                "dependency_blocked",
                "v28 oracle-regret v2 implemented",
                "differentiable SPO stack remains blocked unless dependencies are isolated",
            ),
            (
                "ifrs9_contractual",
                "data_blocked",
                "v29 proxy panel/SICR scenarios improved",
                "contractual IFRS9 blocked by servicing/DPD/cure/timing/macros",
            ),
            (
                "cate_policy_value",
                "theory_blocked",
                "v29 diagnostics improved",
                "reject-inference/identification unresolved",
            ),
            (
                "fair_lending",
                "prohibited_claim",
                "v29 proxy source governance implemented",
                "no protected attributes/protocol",
            ),
            (
                "artifact_registry",
                "resolved",
                "v30 artifact registry implemented",
                "keep updated on future waves",
            ),
            (
                "candidate_registry",
                "resolved",
                "v30 candidate registry implemented",
                "working champion is Paper 4 only",
            ),
        ],
        columns=["blocker_id", "status_v30", "current_diagnosis", "next_action"],
    )


def _claims() -> pd.DataFrame:
    rows = [
        (
            "Paper 4 has 256-path dynamic scale-up",
            True,
            "paper4_v27_dynamic_policy_summary.csv",
            "19bp-v27-dynamic-scale-and-champion-stress.qmd",
            "internal replay only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has direct champion-vs-CVaR stress",
            True,
            "paper4_v27_champion_vs_cvar_stress_memo.csv",
            "19bp-v27-dynamic-scale-and-champion-stress.qmd",
            "Paper 4 working-only decision support",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has CVaR column-generation v2 diagnostics",
            True,
            "paper4_v28_cvar_frontier_non_dominated.csv",
            "19bq-v28-cvar-dla-spo-upgrade.qmd",
            "restricted-master diagnostic only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has exact full-universe CVaR optimality",
            False,
            "paper4_v28_cvar_infeasibility_certificate_formalized.csv",
            "19bq-v28-cvar-dla-spo-upgrade.qmd",
            "no exact full-universe proof",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has formal differentiable SPO+",
            False,
            "paper4_v28_spo_dependency_blockers.csv",
            "19bq-v28-cvar-dla-spo-upgrade.qmd",
            "dependency blocked; oracle-regret only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has ADP rollout v2",
            True,
            "paper4_v28_dla_adp_dynamic_summary.csv",
            "19bq-v28-cvar-dla-spo-upgrade.qmd",
            "not exact Bellman optimality",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has IFRS9-inspired proxy ECL/SICR v2",
            True,
            "paper4_v29_ifrs9_proxy_policy_summary.csv",
            "19br-v29-ifrs9-causal-fairness-source.qmd",
            "proxy only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has contractual IFRS9 lifetime ECL",
            False,
            "paper4_v29_ifrs9_proxy_policy_summary.csv",
            "19br-v29-ifrs9-causal-fairness-source.qmd",
            "data blocked",
            True,
            False,
            False,
        ),
        (
            "Paper 4 has CATE policy value",
            False,
            "paper4_v29_cate_gate_report.csv",
            "19br-v29-ifrs9-causal-fairness-source.qmd",
            "identification blocked",
            False,
            True,
            False,
        ),
        (
            "Paper 4 makes fair-lending legal claims",
            False,
            "paper4_v29_no_legal_claim_flags.csv",
            "19br-v29-ifrs9-causal-fairness-source.qmd",
            "prohibited claim",
            False,
            False,
            True,
        ),
        (
            "Paper 4 has v30 registry and synthesis",
            True,
            "paper4_v30_artifact_registry.csv",
            "19bs-v30-registry-and-synthesis.qmd",
            "lab registry only",
            False,
            False,
            False,
        ),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "claim",
            "allowed",
            "artifact",
            "quarto_page",
            "claim_boundary_v30",
            "no_claim_contractual_ifrs9",
            "no_claim_cate_policy_value",
            "no_claim_fair_lending_legal",
        ],
    )
    df["artifact_exists"] = df["artifact"].map(
        lambda name: (
            (TABLE_DIR / name).exists()
            or (STATUS_DIR / name).exists()
            or name == "paper4_v30_artifact_registry.csv"
        )
    )
    return df


def _powell_readiness() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "framing",
                "resolved",
                "metrics/decisions/uncertainties carried from v14 and updated through v30",
            ),
            (
                "base_model",
                "near_resolved_with_plateau",
                "dynamic replay and internal sample paths support evaluation but not forecast validation",
            ),
            (
                "lookahead_model",
                "near_resolved_with_plateau",
                "CVaR, ADP, SPO oracle-regret exist but exact claims remain bounded",
            ),
            (
                "implementation",
                "lab_ready",
                "reproducible scripts, artifacts, Quarto and guardrails",
            ),
            (
                "claims",
                "governed",
                "contractual/legal/causal/exact-optimality claims blocked unless evidence exists",
            ),
        ],
        columns=["powell_stage", "readiness_v30", "evidence"],
    )


def _contribution_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "dynamic champion stress",
                "paper4_v27_champion_vs_cvar_stress_memo.csv",
                "core_defensible",
                "working champion decisions now require paired robustness",
            ),
            (
                "CVaR strict infeasibility",
                "paper4_v28_cvar_infeasibility_certificate_formalized.csv",
                "negative_governance_result",
                "strict constraints can be academically useful even when infeasible",
            ),
            (
                "ADP rollout v2",
                "paper4_v28_dla_fvi_underperformance_diagnosis.csv",
                "promising_method",
                "DLA lane is stronger but still not Bellman exact",
            ),
            (
                "SPO oracle-regret v2",
                "paper4_v28_spo_temporal_oracle_regret.csv",
                "promising_but_dependency_blocked",
                "decision-loss route survives without differentiable layers",
            ),
            (
                "IFRS9 proxy/SICR v2",
                "paper4_v29_ifrs9_sicr_sensitivity.csv",
                "proxy_appendix",
                "ECL/SICR sensitivity improved without contractual claim",
            ),
            (
                "CATE/fairness boundaries",
                "paper4_v29_cate_gate_report.csv",
                "blocked_with_evidence",
                "blocked claims are cleanly documented",
            ),
            (
                "registry synthesis",
                "paper4_v30_candidate_registry.csv",
                "lab_infrastructure",
                "future waves can keep memory and claim governance synchronized",
            ),
        ],
        columns=[
            "finding",
            "primary_artifact",
            "publishability_class",
            "contribution_interpretation",
        ],
    )


def _triage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "journal_core",
                "dynamic sequential decision governance",
                "viable if scoped",
                "center CRPTO/CVaR/DLA/SPO comparison under governed claims",
            ),
            (
                "main_result_candidate",
                "champion-vs-CVaR paired robustness",
                "needs final narrative",
                "use v27/v30 memo; avoid Paper Estrella promotion",
            ),
            (
                "method_appendix",
                "ADP/SPO",
                "promising but caveated",
                "keep exact Bellman/formal SPO+ out of headline",
            ),
            (
                "negative_result",
                "strict CVaR infeasibility",
                "useful",
                "document as governance feasibility boundary",
            ),
            (
                "proxy_appendix",
                "IFRS9-inspired ECL/SICR",
                "useful but bounded",
                "do not call contractual IFRS9",
            ),
            (
                "blocked",
                "CATE policy value",
                "not publishable as causal policy",
                "requires identification/reject-inference",
            ),
            (
                "prohibited_claim",
                "fair-lending legal claim",
                "not allowed",
                "source governance only",
            ),
        ],
        columns=["triage_bucket", "lane", "current_publishability", "decision"],
    )


def build_v30() -> dict[str, Any]:
    start = time.time()
    artifacts = _artifact_registry()
    candidates = _candidate_registry()
    blockers = _blockers()
    claims = _claims()
    readiness = _powell_readiness()
    contributions = _contribution_map()
    triage = _triage()
    champion = _working_champion(candidates)

    _write_csv("paper4_v30_artifact_registry.csv", artifacts)
    _write_csv("paper4_v30_candidate_registry.csv", candidates)
    _write_csv("paper4_v30_blocker_dashboard.csv", blockers)
    _write_csv("paper4_v30_claim_artifact_matrix.csv", claims)
    _write_csv("paper4_v30_powell_readiness_dashboard.csv", readiness)
    _write_csv("paper4_v30_academic_contribution_map.csv", contributions)
    _write_csv("paper4_v30_publishability_triage.csv", triage)
    _write_json("paper4_v30_working_champion.json", champion)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v30_registry_docs_synthesis",
        "artifact_registry_rows_v30": int(len(artifacts)),
        "candidate_registry_rows_v30": int(len(candidates)),
        "claim_rows_v30": int(len(claims)),
        "all_claim_artifacts_exist_v30": bool(claims["artifact_exists"].all()),
        "working_champion_policy_id_v30": champion["policy_id"],
        "champion_changed_vs_v26": bool(champion["champion_changed_vs_v26"]),
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v30_status.json", status)
    _write_note(
        "paper4_v30_registry_docs_synthesis.md",
        "\n".join(
            [
                "# Paper 4 v30 Registry and Synthesis",
                "",
                f"- Artifact rows: `{status['artifact_registry_rows_v30']}`.",
                f"- Candidate rows: `{status['candidate_registry_rows_v30']}`.",
                f"- Working champion: `{status['working_champion_policy_id_v30']}`.",
                f"- Champion changed vs v26: `{status['champion_changed_vs_v26']}`.",
                "- Paper Estrella remains frozen.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_v30()


if __name__ == "__main__":
    main()
