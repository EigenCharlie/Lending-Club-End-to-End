"""Build Paper 4 v26 artifact/candidate registry and lab synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_json
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.26"
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
            df = pd.read_csv(path, nrows=5)
            rows = sum(1 for _ in path.open("rb")) - 1
            return max(rows, 0), len(df.columns)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return len(df), len(df.columns)
        if path.suffix == ".json":
            return 1, None
    except Exception:
        return None, None
    return None, None


def _artifact_registry() -> pd.DataFrame:
    rows = []
    page_map = {
        "v19": "19bh-v19-dynamic-engine-v2.qmd",
        "v20": "19bi-v20-dla-cvar-spo-resolution.qmd",
        "v21": "19bj-v21-ifrs9-causal-fairness-gates.qmd",
        "v22": "19bk-v22-academic-synthesis.qmd",
        "v23": "19bl-v23-dynamic-scale-paths.qmd",
        "v24": "19bm-v24-dla-cvar-spo-upgrade.qmd",
        "v25": "19bn-v25-ifrs9-causal-fairness-upgrade.qmd",
        "v26": "19bo-v26-registry-and-synthesis.qmd",
    }
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
                    "linked_claim": "see paper4_v26_claim_artifact_matrix.csv",
                    "quarto_page": page_map.get(version, ""),
                    "status": "implemented" if path.exists() else "missing",
                    "caveat": "Paper 4 lab artifact; respect per-claim boundaries",
                    "path_exists": path.exists(),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("artifact")


def _candidate_registry() -> pd.DataFrame:
    frames = []
    for name, version in [
        ("paper4_v23_dynamic_policy_summary.csv", "v23_dynamic"),
        ("paper4_v24_dynamic_combined_summary.csv", "v24_combined"),
        ("paper4_v20_dynamic_policy_summary.csv", "v20_combined"),
    ]:
        p = TABLE_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            df["registry_source"] = version
            frames.append(df)
            break
    if not frames:
        return pd.DataFrame()
    df = frames[0].copy()
    for col in [
        "final_wealth_mean",
        "final_wealth_p05",
        "cumulative_losses_p95",
        "cumulative_defaults_mean",
        "ECL_final_mean",
        "source_exposure_weak_share_final_mean",
        "no_temporal_leakage_rate",
    ]:
        if col not in df:
            df[col] = np.nan
    df["wealth_rank"] = df["final_wealth_mean"].rank(ascending=False, method="min")
    df["p05_wealth_rank"] = df["final_wealth_p05"].rank(ascending=False, method="min")
    df["p95_loss_rank"] = df["cumulative_losses_p95"].rank(ascending=True, method="min")
    df["ecl_rank"] = df["ECL_final_mean"].rank(ascending=True, method="min")
    df["source_rank"] = df["source_exposure_weak_share_final_mean"].rank(
        ascending=True, method="min"
    )
    df["claim_safety_gate"] = df["no_temporal_leakage_rate"].ge(1.0)

    def score01(series: pd.Series, *, high_is_good: bool) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() <= 1:
            return pd.Series(0.5, index=series.index)
        pct = numeric.rank(method="average", pct=True, ascending=True)
        return pct.fillna(0.5) if high_is_good else (1.0 - pct + (1.0 / len(series))).fillna(0.5)

    df["auditability_score_proxy_v26"] = (
        0.35 * score01(df["cumulative_losses_p95"], high_is_good=False)
        + 0.25 * score01(df["source_exposure_weak_share_final_mean"], high_is_good=False)
        + 0.20 * score01(df["ECL_final_mean"], high_is_good=False)
        + 0.20 * df["claim_safety_gate"].astype(float)
    )
    df["full_candidate_score_v26"] = (
        0.35 * score01(df["final_wealth_mean"], high_is_good=True)
        + 0.20 * score01(df["final_wealth_p05"], high_is_good=True)
        + 0.20 * score01(df["cumulative_losses_p95"], high_is_good=False)
        + 0.15 * df["auditability_score_proxy_v26"]
        + 0.10 * score01(df["cumulative_defaults_mean"], high_is_good=False)
    )
    current = str(
        _safe_read_json(STATUS_DIR / "paper4_v22_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    df["decision_v26"] = "review"
    df.loc[
        df["full_candidate_score_v26"].lt(df["full_candidate_score_v26"].quantile(0.25)),
        "decision_v26",
    ] = "park"
    df.loc[~df["claim_safety_gate"].astype(bool), "decision_v26"] = "kill"
    df.loc[df["policy_id"].eq(current), "decision_v26"] = "retain_working_champion"
    df.loc[df["policy_id"].eq("v13_spo_regret_audit_guarded"), "decision_v26"] = (
        "serious_challenger"
    )
    keep = [
        "policy_id",
        "registry_source",
        "final_wealth_mean",
        "final_wealth_p05",
        "cumulative_losses_p95",
        "ECL_final_mean",
        "source_exposure_weak_share_final_mean",
        "wealth_rank",
        "p05_wealth_rank",
        "p95_loss_rank",
        "ecl_rank",
        "source_rank",
        "auditability_score_proxy_v26",
        "full_candidate_score_v26",
        "claim_safety_gate",
        "decision_v26",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values(
        "full_candidate_score_v26", ascending=False
    )


def _blockers() -> pd.DataFrame:
    rows = [
        (
            "dynamic_engine_scale",
            "resolved",
            "v23 128-path replay and convergence artifacts implemented",
            "scale to 256/512 only if manuscript needs tighter intervals",
        ),
        (
            "sample_paths_v3",
            "near_resolved_with_plateau",
            "internal path v3 plus macro event/context registry implemented",
            "external forecast validation remains out of scope",
        ),
        (
            "dla_adp_rollout",
            "near_resolved_with_plateau",
            "v24 ADP rollout approximations implemented",
            "exact Bellman proof remains future work",
        ),
        (
            "cvar_column_generation",
            "near_resolved_with_plateau",
            "v24 pricing log/frontier/certificate diagnostics implemented",
            "no exact full-universe optimality claim",
        ),
        (
            "spo_dfl",
            "dependency_blocked",
            "v24 oracle-regret surrogate implemented",
            "cvxpy/cvxpylayers/torch path blocked",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "v25 cashflow proxy/SICR improved",
            "contractual IFRS9 still blocked by servicing/DPD/cure/timing/macros",
        ),
        (
            "cate_policy_value",
            "theory_blocked",
            "v25 trimming/IPW/sensitivity diagnostics implemented",
            "reject-inference/identification unresolved",
        ),
        (
            "fair_lending",
            "prohibited_claim",
            "v25 source governance and no-claim flags implemented",
            "no protected attributes/protocol",
        ),
        (
            "artifact_registry",
            "resolved",
            "v26 artifact manifest implemented",
            "keep updated on future waves",
        ),
        (
            "candidate_registry",
            "resolved",
            "v26 candidate registry implemented",
            "review serious challengers only under full evidence",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["blocker_id", "status_v26", "current_diagnosis", "next_action"]
    )


def _claims() -> pd.DataFrame:
    rows = [
        (
            "Paper 4 has 128-path dynamic scale-up",
            True,
            "paper4_v23_dynamic_policy_summary.csv",
            "19bl-v23-dynamic-scale-paths.qmd",
            "internal replay only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has sample paths v3 with calibration diagnostics",
            True,
            "paper4_v23_path_calibration_diagnostics.csv",
            "19bl-v23-dynamic-scale-paths.qmd",
            "not forecast validation",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has ADP rollout approximations",
            True,
            "paper4_v24_dla_adp_dynamic_summary.csv",
            "19bm-v24-dla-cvar-spo-upgrade.qmd",
            "not exact Bellman optimality",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has exact full-universe CVaR optimality",
            False,
            "paper4_v24_cvar_frontier_non_dominated.csv",
            "19bm-v24-dla-cvar-spo-upgrade.qmd",
            "restricted-master diagnostic only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has formal differentiable SPO+",
            False,
            "paper4_v24_spo_dependency_blockers.csv",
            "19bm-v24-dla-cvar-spo-upgrade.qmd",
            "dependency blocked; oracle-regret only",
            False,
            False,
            False,
        ),
        (
            "Paper 4 has contractual IFRS9 lifetime ECL",
            False,
            "paper4_v25_ifrs9_proxy_policy_summary.csv",
            "19bn-v25-ifrs9-causal-fairness-upgrade.qmd",
            "proxy only",
            True,
            False,
            False,
        ),
        (
            "Paper 4 has CATE policy value",
            False,
            "paper4_v25_cate_gate_report.csv",
            "19bn-v25-ifrs9-causal-fairness-upgrade.qmd",
            "identification blocked",
            False,
            True,
            False,
        ),
        (
            "Paper 4 makes fair-lending legal claims",
            False,
            "paper4_v25_no_legal_claim_flags.csv",
            "19bn-v25-ifrs9-causal-fairness-upgrade.qmd",
            "no protected attributes/protocol",
            False,
            False,
            True,
        ),
        (
            "Paper 4 has artifact and candidate registries",
            True,
            "paper4_v26_artifact_registry.csv",
            "19bo-v26-registry-and-synthesis.qmd",
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
            "claim_boundary_v26",
            "no_claim_contractual_ifrs9",
            "no_claim_cate_policy_value",
            "no_claim_fair_lending_legal",
        ],
    )
    df["artifact_exists"] = df["artifact"].map(
        lambda name: (TABLE_DIR / name).exists() or (STATUS_DIR / name).exists()
    )
    return df


def _contributions() -> pd.DataFrame:
    rows = [
        (
            "128-path dynamic scale",
            "paper4_v23_scale_convergence.csv",
            "core_defensible",
            "ranking stability and interval convergence under larger paths",
        ),
        (
            "Sample paths v3",
            "paper4_v23_path_calibration_diagnostics.csv",
            "core_defensible_with_caveat",
            "better internal calibration without forecast claim",
        ),
        (
            "ADP rollout approximations",
            "paper4_v24_dla_adp_dynamic_summary.csv",
            "promising_method",
            "moves DLA further toward endogenous monthly decision",
        ),
        (
            "CVaR column-generation diagnostics",
            "paper4_v24_cvar_column_generation_log.csv",
            "negative_or_governance_result",
            "strict infeasibility and relaxations are explainable",
        ),
        (
            "SPO oracle-regret upgrade",
            "paper4_v24_spo_temporal_oracle_regret.csv",
            "promising_but_not_formal_spo_plus",
            "decision-loss path survives dependency blockers",
        ),
        (
            "IFRS9 proxy cashflow panel",
            "paper4_v25_ifrs9_proxy_cashflow_panel.parquet",
            "proxy_appendix",
            "better ECL/SICR evidence without contractual claim",
        ),
        (
            "Causal diagnostics upgrade",
            "paper4_v25_causal_balance_trim_ipw.csv",
            "blocked_with_evidence",
            "trimming/IPW improves diagnostics but not identification",
        ),
        (
            "Fairness source governance",
            "paper4_v25_source_governance_diagnostics.csv",
            "governance_boundary",
            "observable-source monitoring only",
        ),
        (
            "Artifact/candidate registry",
            "paper4_v26_candidate_registry.csv",
            "lab_infrastructure",
            "keeps serious challengers visible",
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
            "dynamic sequential governance with 128-path evidence",
            "increasingly viable",
            "tighten scope around CRPTO/CVaR/DLA/SPO governance",
        ),
        (
            "method_appendix",
            "ADP rollout and SPO oracle-regret",
            "promising but caveated",
            "needs theory/dependency stabilization for main claim",
        ),
        (
            "governance_appendix",
            "CVaR strict infeasibility certificates",
            "useful negative result",
            "keep relaxed/committee labels explicit",
        ),
        (
            "proxy_appendix",
            "IFRS9-inspired cashflow/ECL proxy",
            "useful but not contractual",
            "do not headline as IFRS9 paper",
        ),
        (
            "blocked",
            "CATE policy value",
            "not publishable as causal policy",
            "requires identification/reject-inference design",
        ),
        ("prohibited_claim", "fair-lending legal claim", "not allowed", "proxy governance only"),
    ]
    return pd.DataFrame(
        rows, columns=["triage_bucket", "lane", "current_publishability", "decision"]
    )


def _working_champion(candidate_registry: pd.DataFrame) -> dict[str, Any]:
    prior = _safe_read_json(STATUS_DIR / "paper4_v22_working_champion.json")
    current = str(prior.get("policy_id", "paper1_economic_champion"))
    if candidate_registry.empty:
        selected = current
        challenger = ""
        changed = False
        score = 0.0
    else:
        top = candidate_registry.iloc[0]
        challenger = str(top["policy_id"])
        selected = current
        changed = False
        score = (
            float(
                candidate_registry.loc[
                    candidate_registry["policy_id"].eq(current), "full_candidate_score_v26"
                ].iloc[0]
            )
            if current in set(candidate_registry["policy_id"])
            else 0.0
        )
        if challenger != current and str(top.get("decision_v26")) == "serious_challenger":
            selected = current
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": selected,
        "previous_working_champion_policy_id_v22": current,
        "highest_score_challenger_policy_id_v26": challenger,
        "champion_changed_vs_v22": changed,
        "scope": "paper4_working_champion_only",
        "selection_status": "retain_v22_pending_paired_challenger_robustness",
        "full_candidate_score_v26": score,
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "contractual_ifrs9_claim_allowed": False,
        "cate_policy_value_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "caveat": "Paper 4 lab champion only; challengers require paired robustness and claim safety",
    }


def build_v26() -> dict[str, Any]:
    start = time.time()
    artifacts = _artifact_registry()
    candidates = _candidate_registry()
    blockers = _blockers()
    claims = _claims()
    contributions = _contributions()
    triage = _triage()
    champion = _working_champion(candidates)
    _write_csv("paper4_v26_artifact_registry.csv", artifacts)
    _write_csv("paper4_v26_candidate_registry.csv", candidates)
    _write_csv("paper4_v26_blocker_dashboard.csv", blockers)
    _write_csv("paper4_v26_claim_artifact_matrix.csv", claims)
    _write_csv("paper4_v26_academic_contribution_map.csv", contributions)
    _write_csv("paper4_v26_publishability_triage.csv", triage)
    _write_json("paper4_v26_working_champion.json", champion)
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v26_registry_docs_synthesis",
        "artifact_registry_rows_v26": int(len(artifacts)),
        "candidate_registry_rows_v26": int(len(candidates)),
        "claim_rows_v26": int(len(claims)),
        "all_claim_artifacts_exist_v26": bool(claims["artifact_exists"].all()),
        "working_champion_policy_id_v26": champion["policy_id"],
        "champion_changed_vs_v22": champion["champion_changed_vs_v22"],
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v26_status.json", status)
    _write_note(
        "paper4_v26_registry_docs_synthesis.md",
        "\n".join(
            [
                "# Paper 4 v26 Registry and Synthesis",
                "",
                f"- Working champion: `{status['working_champion_policy_id_v26']}`.",
                f"- Artifact rows: `{status['artifact_registry_rows_v26']}`.",
                f"- Candidate rows: `{status['candidate_registry_rows_v26']}`.",
                f"- All claim artifacts exist: `{status['all_claim_artifacts_exist_v26']}`.",
                "",
                "This is a Paper 4 lab synthesis only.",
            ]
        ),
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args(list(argv) if argv is not None else None)
    build_v26()


if __name__ == "__main__":
    main()
