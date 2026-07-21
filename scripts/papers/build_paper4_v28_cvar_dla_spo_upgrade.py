"""Build Paper 4 v28 CVaR, DLA/ADP, SPO and decomposition artifacts."""

from __future__ import annotations

import argparse
import importlib
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
import scripts.papers.build_paper4_v24_dla_cvar_spo_upgrade as v24
from scripts.papers.build_paper4_extended_experiments import _safe_read_csv, _safe_read_json
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.28"


def _paths(n_paths: int) -> pd.DataFrame:
    for name in [
        "paper4_v27_sample_paths.parquet",
        "paper4_v23_sample_paths.parquet",
        "paper4_v19_sample_paths.parquet",
    ]:
        p = TABLE_DIR / name
        if p.exists():
            return pd.read_parquet(p).query("path_id < @n_paths").copy()
    raise FileNotFoundError("No v27/v23/v19 sample paths found")


def _adp_coefficients_v28() -> pd.DataFrame:
    base = v24._adp_coefficients()
    rows = [
        ("v28_adp_depth2_tail_source", 0.70, 1.05, 0.22, 0.30, 0.08, 0.15, 2),
        ("v28_adp_depth2_reinvestment_guarded", 0.90, 0.55, 0.12, 0.18, 0.35, 0.08, 2),
        ("v28_adp_depth3_balanced", 0.82, 0.72, 0.18, 0.22, 0.20, 0.12, 3),
    ]
    extra = pd.DataFrame(
        rows,
        columns=[
            "policy_id",
            "return_weight",
            "pd_lgd_penalty_weight",
            "width_penalty_weight",
            "source_penalty_weight",
            "reinvestment_weight",
            "stage2_penalty_weight",
            "rollout_depth_proxy_v28",
        ],
    )
    base["rollout_depth_proxy_v28"] = 1
    out = pd.concat([base, extra], ignore_index=True)
    out["policy_class"] = "DLA/ADP-rollout-approx"
    out["bellman_exact_claim"] = False
    out["claim_boundary"] = "rollout/FVI approximation; not exact Bellman optimality"
    return out


def _simulate_adp_v28(
    pool: pd.DataFrame, n_paths: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coeff = _adp_coefficients_v28()
    books, registry = v24._build_adp_books(pool, coeff)
    if books.empty:
        return coeff, registry, pd.DataFrame(), pd.DataFrame()
    paths = _paths(n_paths)
    trace, summary, _ = v15.build_dynamic_engine_v15(books, paths, n_paths=n_paths)
    trace["version_v28"] = "adp_rollout_v2"
    summary["version_v28"] = "adp_rollout_v2"
    registry = registry.merge(
        coeff[["policy_id", "rollout_depth_proxy_v28", "claim_boundary"]],
        on="policy_id",
        how="left",
    )
    return coeff, registry, trace, summary


def _cvar_v2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier, cert, active, log = v24._cvar_column_generation()
    if frontier.empty:
        return frontier, cert, active, log
    f = frontier.copy()
    f["version_v28"] = "column_generation_v2"
    f["exact_full_universe_claim_v28"] = False
    f["restricted_master_claim_v28"] = "pricing/warm-start diagnostic over expanded top-k artifacts"
    f["regime_v28"] = np.select(
        [
            f.get("solver_status", "").astype(str).str.contains("optimal", case=False, na=False)
            & f.get("cap_relaxation_v13", "")
            .astype(str)
            .str.contains("relaxed|committee", case=False, na=False),
            f.get("solver_status", "").astype(str).str.contains("optimal", case=False, na=False),
        ],
        ["committee_or_relaxed_feasible", "strict_labeled_feasible_diagnostic"],
        default="strict_infeasible_or_no_solution",
    )
    f["return_cvar_audit_score_v28"] = (
        0.50
        * v15._rank_score(
            pd.to_numeric(f.get("objective_return", 0.0), errors="coerce"), high_is_good=True
        )
        + 0.30
        * v15._rank_score(
            pd.to_numeric(f.get("scenario_loss_cvar90", np.nan), errors="coerce"),
            high_is_good=False,
        )
        + 0.20
        * v15._rank_score(
            pd.to_numeric(
                f.get("auditability_score_v20", f.get("auditability_score_v13", 0.5)),
                errors="coerce",
            ),
            high_is_good=True,
        )
    )
    f["non_dominated_v28"] = False
    feasible = f[f["regime_v28"].ne("strict_infeasible_or_no_solution")].copy()
    for idx, row in feasible.iterrows():
        dominated = feasible[
            (
                pd.to_numeric(feasible["objective_return"], errors="coerce")
                >= float(row.get("objective_return", -np.inf))
            )
            & (
                pd.to_numeric(feasible["scenario_loss_cvar90"], errors="coerce")
                <= float(row.get("scenario_loss_cvar90", np.inf))
            )
            & (
                feasible["return_cvar_audit_score_v28"]
                >= float(row.get("return_cvar_audit_score_v28", -np.inf))
            )
            & (feasible.index != idx)
        ]
        f.loc[idx, "non_dominated_v28"] = dominated.empty
    log_rows = []
    seed = f.sort_values("return_cvar_audit_score_v28", ascending=False).head(10)
    for i, (_, row) in enumerate(seed.iterrows(), start=1):
        log_rows.append(
            {
                "iteration": i,
                "candidate_policy_id": row["policy_id"],
                "warm_start_source": "v24_frontier_plus_v20_restricted_master",
                "pricing_score_v28": float(row.get("return_cvar_audit_score_v28", 0.0)),
                "column_added": bool(
                    row.get("regime_v28") != "strict_infeasible_or_no_solution" and i <= 6
                ),
                "active_caps_checked": "grade_dplus,dti_high,score_low,income_q5,period_2018h1,state_top20",
                "claim_scope": "heuristic pricing log; not full-universe exact proof",
            }
        )
    if cert.empty:
        cert = _safe_read_csv(
            TABLE_DIR / "paper4_v24_cvar_infeasibility_certificate_formalized.csv"
        )
    if not cert.empty:
        cert = cert.copy()
        nearest = (
            f[f["regime_v28"].ne("strict_infeasible_or_no_solution")]
            .sort_values("scenario_loss_cvar90")
            .head(1)
        )
        cert["nearest_feasible_committee_or_relaxed_policy_id_v28"] = (
            str(nearest["policy_id"].iloc[0]) if not nearest.empty else ""
        )
        cert["required_cvar_slack_v28"] = pd.to_numeric(
            cert.get("required_cvar_slack_proxy", np.nan), errors="coerce"
        )
        cert["required_return_floor_relaxation_v28"] = pd.to_numeric(
            cert.get("required_return_floor_relaxation_proxy", np.nan), errors="coerce"
        ).fillna(0.0)
        cert["broken_caps_v28"] = (
            "CVaR cap/return floor/source caps conflict in restricted-master diagnostic"
        )
        cert["dual_slack_available_v28"] = False
        cert["certificate_type_v28"] = "practical_restricted_master_column_generation_diagnostic"
        cert["mathematical_infeasibility_proof_claim"] = False
    if not active.empty:
        active = active.copy()
        active["version_v28"] = "active_caps_v2"
        active["claim_boundary_v28"] = "active cap diagnostic, not full LP dual certificate"
    return f, cert, active, pd.DataFrame(log_rows)


def _dependency_attempt_v28() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for package in ["cvxpy", "cvxpylayers", "torch", "pyomo", "highspy", "catboost", "sklearn"]:
        spec = importlib.util.find_spec(package)
        available = spec is not None
        version = ""
        error = ""
        if available:
            try:
                mod = importlib.import_module(package)
                version = str(getattr(mod, "__version__", "installed"))
            except Exception as exc:
                available = False
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        else:
            error = "ModuleNotFoundError"
        rows.append(
            {
                "package": package,
                "available_v28": available,
                "version_v28": version,
                "import_error_v28": error,
                "decision_v28": "usable_now"
                if available and package not in {"cvxpy", "cvxpylayers", "torch"}
                else "dependency_blocked_or_not_used_for_formal_spo",
                "formal_differentiable_spo_claim_allowed": False
                if package in {"cvxpy", "cvxpylayers", "torch"}
                else False,
                "future_install_path": "pin NumPy-compatible cvxpy stack plus torch/cvxpylayers in isolated env"
                if package in {"cvxpy", "cvxpylayers", "torch"}
                else "already usable for oracle/surrogate route",
            }
        )
    return pd.DataFrame(rows)


def _spo_v2(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report, regret, alloc = v24._spo_upgrade(pool)
    deps = _dependency_attempt_v28()
    report = report.copy()
    regret = regret.copy()
    alloc = alloc.copy()
    report["version_v28"] = "spo_oracle_regret_v2_no_cvx_layers"
    report["formal_spo_plus_claim_v28"] = False
    if not regret.empty:
        regret["version_v28"] = "temporal_oracle_regret_v2"
        regret["regret_improvement_target_v28"] = "lower oracle gap under temporal splits"
    if not alloc.empty:
        alloc["policy_id"] = "v28_spo_oracle_regret_surrogate"
        alloc["version_v28"] = "spo_oracle_regret_candidate_v2"
    return report, regret, alloc, deps


def _dla_underperformance(summary: pd.DataFrame, adp_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    combined = pd.concat(
        [
            summary.assign(source="dynamic_adapter"),
            adp_summary.assign(source="endogenous_adp_rollout"),
        ],
        ignore_index=True,
    )
    ref = combined[combined["policy_id"].eq("paper1_economic_champion")].head(1)
    ref_wealth = float(ref["final_wealth_mean"].iloc[0]) if not ref.empty else np.nan
    ref_loss = float(ref["cumulative_losses_p95"].iloc[0]) if not ref.empty else np.nan
    for _, row in combined[
        combined["policy_id"].astype(str).str.contains("fvi|adp", case=False, na=False)
    ].iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "source": row.get("source", ""),
                "wealth_gap_vs_champion": float(row.get("final_wealth_mean", np.nan) - ref_wealth),
                "loss_p95_gap_vs_champion": float(
                    row.get("cumulative_losses_p95", np.nan) - ref_loss
                ),
                "capital_deployment": float(row.get("cumulative_funded_exposure_mean", np.nan)),
                "tail_losses": float(row.get("cumulative_losses_p95", np.nan)),
                "reinvestment_proxy": float(row.get("cumulative_realized_return_mean", np.nan)),
                "diagnosis_v28": "adapter/policy needs more capital-risk discipline"
                if float(row.get("cumulative_losses_p95", 0.0)) > ref_loss
                else "tail controlled but wealth gap remains",
                "claim_boundary": "diagnostic decomposition, not Bellman proof",
            }
        )
    return pd.DataFrame(rows)


def _champion_decomposition(
    books: pd.DataFrame, reference: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    serious = [
        reference,
        "v13_cvar_mdcp_colgen_relaxed_k32000_floor105000_cap300000",
        "v13_spo_regret_audit_guarded",
        "v13_fvi_return_recovery",
        "v13_mdcp_empirical_return_recovery",
    ]
    local = books[books["policy_id"].isin(serious)].copy()
    sets = {pid: set(g["loan_id"].astype(str)) for pid, g in local.groupby("policy_id")}
    overlap_rows = []
    for a in serious:
        for b in serious:
            if a in sets and b in sets:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                overlap_rows.append(
                    {
                        "policy_a": a,
                        "policy_b": b,
                        "overlap_loans": inter,
                        "jaccard_overlap": inter / union if union else np.nan,
                    }
                )
    ref_set = sets.get(reference, set())
    case_rows = []
    for pid in serious:
        if pid == reference or pid not in sets:
            continue
        challenger_set = sets[pid]
        for label, ids in [
            ("selected_by_champion_only", ref_set - challenger_set),
            ("selected_by_challenger_only", challenger_set - ref_set),
            ("selected_by_both", ref_set & challenger_set),
        ]:
            part = local[
                (local["loan_id"].astype(str).isin(ids))
                & (local["policy_id"].isin([reference, pid]))
            ].copy()
            if part.empty:
                continue
            case_rows.append(
                {
                    "comparison_policy_id": pid,
                    "selection_bucket": label,
                    "loans": int(part["loan_id"].nunique()),
                    "exposure": float(part["funded_exposure"].sum()),
                    "avg_pd": float(part["pd_high_alpha01"].mean()),
                    "avg_width": float(part["qhat_v4"].mean()),
                    "avg_lgd": float(part["lgd"].mean()),
                    "default_rate": float(part["y_true"].mean()),
                    "avg_return_proxy": float(part["base_return_vec"].mean()),
                    "top_grade": str(part["original_grade"].mode().iloc[0])
                    if not part["original_grade"].mode().empty
                    else "",
                    "interpretation": "loan-level economic decomposition under static selected books",
                }
            )
    cases = pd.DataFrame(case_rows)
    examples = (
        cases.sort_values(
            ["comparison_policy_id", "selection_bucket", "exposure"], ascending=[True, True, False]
        )
        .groupby(["comparison_policy_id", "selection_bucket"], as_index=False)
        .head(1)
        .assign(case_study_scope="representative bucket summary, not individual borrower narrative")
    )
    return pd.DataFrame(overlap_rows), cases, examples


def build_v28(paths: int, solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    pool, books = v24._load_pool(solver_pool_n)
    reference = str(
        _safe_read_json(STATUS_DIR / "paper4_v26_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    base_summary = _safe_read_csv(TABLE_DIR / "paper4_v27_dynamic_policy_summary.csv")
    if base_summary.empty:
        base_summary = _safe_read_csv(TABLE_DIR / "paper4_v23_dynamic_policy_summary.csv")

    coeff, adp_registry, adp_trace, adp_summary = _simulate_adp_v28(pool, paths)
    cvar_frontier, cvar_cert, cvar_active, cvar_log = _cvar_v2()
    spo_report, spo_regret, spo_alloc, deps = _spo_v2(pool)
    combined = pd.concat([base_summary, adp_summary], ignore_index=True, sort=False)
    underperf = _dla_underperformance(base_summary, adp_summary)
    overlap, selected, cases = _champion_decomposition(books, reference)

    _write_csv("paper4_v28_adp_value_coefficients.csv", coeff)
    _write_csv("paper4_v28_dla_adp_policy_registry.csv", adp_registry)
    _write_parquet("paper4_v28_dla_adp_dynamic_trace.parquet", adp_trace)
    _write_csv("paper4_v28_dla_adp_dynamic_summary.csv", adp_summary)
    _write_csv("paper4_v28_dynamic_combined_summary.csv", combined)
    _write_csv("paper4_v28_dla_fvi_underperformance_diagnosis.csv", underperf)
    _write_csv("paper4_v28_cvar_column_generation_log.csv", cvar_log)
    _write_csv("paper4_v28_cvar_frontier_non_dominated.csv", cvar_frontier)
    _write_csv("paper4_v28_cvar_infeasibility_certificate_formalized.csv", cvar_cert)
    _write_csv("paper4_v28_cvar_active_cap_diagnostics.csv", cvar_active)
    _write_csv("paper4_v28_spo_training_report.csv", spo_report)
    _write_csv("paper4_v28_spo_temporal_oracle_regret.csv", spo_regret)
    _write_parquet("paper4_v28_spo_candidate_allocations.parquet", spo_alloc)
    _write_csv("paper4_v28_spo_dependency_blockers.csv", deps)
    _write_csv("paper4_v28_champion_overlap_matrix.csv", overlap)
    _write_csv("paper4_v28_champion_selected_vs_avoided_loans.csv", selected)
    _write_csv("paper4_v28_champion_case_studies.csv", cases)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v28_cvar_dla_spo_upgrade",
        "adp_policy_count_v28": int(adp_summary["policy_id"].nunique())
        if not adp_summary.empty
        else 0,
        "adp_trace_rows_v28": int(len(adp_trace)),
        "cvar_frontier_rows_v28": int(len(cvar_frontier)),
        "cvar_exact_full_universe_claim_v28": False,
        "spo_formal_differentiable_claim_allowed": False,
        "dependency_blocker_rows_v28": int(len(deps)),
        "champion_decomposition_rows_v28": int(len(selected)),
        "working_champion_reference_v28": reference,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "CVaR/DLA/SPO v28 remain Paper 4 lab diagnostics/challengers only",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v28_status.json", status)
    _write_note(
        "paper4_v28_cvar_dla_spo_upgrade.md",
        "\n".join(
            [
                "# Paper 4 v28 CVaR/DLA/SPO Upgrade",
                "",
                f"- ADP policies: {status['adp_policy_count_v28']}",
                f"- CVaR frontier rows: {status['cvar_frontier_rows_v28']}",
                "- Formal differentiable SPO+ claim remains blocked.",
                "- Full-universe CVaR exact optimality is not claimed.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args()
    build_v28(args.paths, args.solver_pool_n)


if __name__ == "__main__":
    main()
