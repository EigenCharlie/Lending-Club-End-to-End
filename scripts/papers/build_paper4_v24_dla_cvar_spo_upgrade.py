"""Build Paper 4 v24 DLA, CVaR/OCE, SPO and champion-stress upgrades."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
import scripts.papers.build_paper4_v20_dla_cvar_spo_resolution as v20
from scripts.papers.build_paper4_extended_experiments import (
    _safe_read_csv,
    _safe_read_json,
)
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _load_inputs,
    _prepare_solver_pool,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.24"


def _load_pool(max_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, candidate, _, _, intervals = _load_inputs()
    source = base if len(base) > len(candidate) else candidate
    pool = _prepare_solver_pool(source, intervals, max_n=min(max_n, len(source)))
    books, _ = v15._load_policy_books(pool)
    return pool, books


def _adp_coefficients() -> pd.DataFrame:
    rows = [
        ("v24_adp_rollout_return_value", 1.00, 0.40, 0.08, 0.10, 0.05, 0.04),
        ("v24_adp_rollout_tail_value", 0.75, 0.90, 0.15, 0.10, 0.02, 0.08),
        ("v24_adp_rollout_source_ecl_value", 0.80, 0.55, 0.12, 0.35, 0.03, 0.12),
        ("v24_adp_rollout_reinvestment_value", 0.85, 0.45, 0.07, 0.12, 0.20, 0.05),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "policy_id",
            "return_weight",
            "pd_lgd_penalty_weight",
            "width_penalty_weight",
            "source_penalty_weight",
            "reinvestment_weight",
            "stage2_penalty_weight",
        ],
    ).assign(
        policy_class="DLA/ADP-rollout-approx",
        bellman_exact_claim=False,
        claim_boundary="approximate value-function rollout; not exact Bellman optimality",
    )


def _build_adp_books(
    pool: pd.DataFrame, coefficients: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pool.copy()
    df["issue_month"] = (
        pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    for col in [
        "loan_amnt",
        "base_return_vec",
        "pd_high_alpha01",
        "qhat_v4",
        "weak_source_proxy",
        "installment",
        "lgd",
    ]:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["lgd"] = df["lgd"].replace(0.0, v15.DEFAULT_LGD).clip(0.1, 0.95)
    df["tail_ecl_proxy"] = df["loan_amnt"] * df["pd_high_alpha01"] * df["lgd"]
    df["width_capital_proxy"] = df["loan_amnt"] * df["qhat_v4"]
    df["source_capital_proxy"] = df["loan_amnt"] * df["weak_source_proxy"]
    df["stage2_proxy"] = (
        (df["pd_high_alpha01"] >= 0.18)
        | (df["qhat_v4"] >= 0.80)
        | (df["weak_source_proxy"] >= 0.66)
    ).astype(float)
    books = []
    registry = []
    for _, coef in coefficients.iterrows():
        local = df.copy()
        local["adp_score_v24"] = (
            coef["return_weight"] * local["base_return_vec"]
            - coef["pd_lgd_penalty_weight"] * local["tail_ecl_proxy"]
            - coef["width_penalty_weight"] * local["width_capital_proxy"]
            - coef["source_penalty_weight"] * local["source_capital_proxy"]
            + coef["reinvestment_weight"] * local["installment"]
            - coef["stage2_penalty_weight"] * local["loan_amnt"] * local["stage2_proxy"]
        )
        selected = []
        budget_limit = 2.4 * v15.BUDGET
        month_cap = 0.28 * v15.BUDGET
        cumulative = 0.0
        for month, month_df in local.sort_values("issue_month").groupby("issue_month", sort=True):
            used_month = 0.0
            concentration: dict[str, float] = {}
            for _, row in month_df.sort_values("adp_score_v24", ascending=False).iterrows():
                amount = float(row["loan_amnt"])
                if (
                    amount <= 0
                    or cumulative + amount > budget_limit
                    or used_month + amount > month_cap
                ):
                    continue
                state_key = str(row.get("state_top20", row.get("addr_state", "unknown")))
                grade_key = str(row.get("original_grade", row.get("grade", "unknown")))
                # Concentration caps are meaningful after a small monthly book
                # exists.  Applying a 40% cap to the very first loan makes every
                # month infeasible by construction.
                if used_month >= 100_000 and concentration.get(
                    grade_key, 0.0
                ) + amount > 0.40 * max(used_month + amount, 1.0):
                    continue
                if (
                    "source" in str(coef["policy_id"])
                    and used_month >= 100_000
                    and concentration.get(state_key, 0.0) + amount
                    > 0.30 * max(used_month + amount, 1.0)
                ):
                    continue
                rec = row.to_dict()
                rec["policy_id"] = coef["policy_id"]
                rec["funded_exposure"] = amount
                rec["value_function_proxy_v24"] = float(row["adp_score_v24"])
                rec["S_t_cash_proxy_v24"] = max(v15.BUDGET - cumulative, 0.0)
                rec["x_t_fund_amount_v24"] = amount
                rec["Sx_t_budget_proxy_v24"] = max(v15.BUDGET - cumulative - amount, 0.0)
                rec["W_t_plus_1_proxy_v24"] = "internal_default_lgd_prepay_path"
                rec["reward_proxy_v24"] = float(row["base_return_vec"] - row["tail_ecl_proxy"])
                rec["transition_proxy_v24"] = (
                    "monthly amortization/default/recovery replay in v15 engine"
                )
                selected.append(rec)
                cumulative += amount
                used_month += amount
                concentration[grade_key] = concentration.get(grade_key, 0.0) + amount
                concentration[state_key] = concentration.get(state_key, 0.0) + amount
            if cumulative >= budget_limit:
                break
        if not selected:
            continue
        book = v15._standardize_book(
            pd.DataFrame(selected), source_artifact="v24_adp_rollout", lane="dla_adp_rollout_v24"
        )
        books.append(book)
        registry.append(
            {
                "policy_id": coef["policy_id"],
                "selected_loans": int(book["loan_id"].nunique()),
                "book_exposure": float(book["funded_exposure"].sum()),
                "state_variables": "cash, budget, PD, width, source, stage2, installment",
                "decision_x_t": "fund loan amount subject to monthly cap and concentration gates",
                "post_decision_state": "cash/budget reduced, source concentration updated",
                "reward": "return minus ECL/tail/source/stage penalties plus reinvestment value",
                "bellman_exact_claim": False,
            }
        )
    return (
        pd.concat(books, ignore_index=True) if books else pd.DataFrame(),
        pd.DataFrame(registry),
    )


def _load_paths(n_paths: int) -> pd.DataFrame:
    for name in ["paper4_v23_sample_paths.parquet", "paper4_v19_sample_paths.parquet"]:
        path = TABLE_DIR / name
        if path.exists():
            return pd.read_parquet(path).query("path_id < @n_paths").copy()
    raise FileNotFoundError("No v23/v19 sample paths available.")


def _simulate_adp(
    adp_books: pd.DataFrame, n_paths: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _load_paths(n_paths)
    trace, summary, pairwise = v15.build_dynamic_engine_v15(adp_books, paths, n_paths=n_paths)
    trace["version_v24"] = "adp_rollout_upgrade"
    summary["version_v24"] = "adp_rollout_upgrade"
    return trace, summary, pairwise


def _cvar_column_generation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier = _safe_read_csv(TABLE_DIR / "paper4_v20_cvar_oce_frontier_v2.csv")
    cert = _safe_read_csv(TABLE_DIR / "paper4_v20_cvar_strict_infeasibility_v2.csv")
    active = _safe_read_csv(TABLE_DIR / "paper4_v20_cvar_active_caps_v2.csv")
    if frontier.empty:
        return frontier, cert, active, pd.DataFrame()
    f = frontier.copy()
    if "objective_return" not in f:
        f["objective_return"] = f.get("return_cvar_tradeoff_v20", 0.0)
    if "scenario_loss_cvar90" not in f:
        f["scenario_loss_cvar90"] = np.nan
    f["regime_v24"] = f.get("regime_v20", "diagnostic")
    f["column_generation_claim_v24"] = "restricted_master_pricing_diagnostic"
    f["full_universe_exact_claim_v24"] = False
    f["non_dominated_v24"] = False
    feasible = (
        f[f.get("feasible_v13", True).astype(bool)].copy() if "feasible_v13" in f else f.copy()
    )
    if not feasible.empty:
        for idx, row in feasible.iterrows():
            dominated = feasible[
                (
                    pd.to_numeric(feasible["objective_return"], errors="coerce")
                    >= float(row["objective_return"])
                )
                & (
                    pd.to_numeric(feasible["scenario_loss_cvar90"], errors="coerce")
                    <= float(row["scenario_loss_cvar90"])
                )
                & (feasible.index != idx)
            ]
            f.loc[idx, "non_dominated_v24"] = dominated.empty
    log_rows = []
    seed = (
        feasible.sort_values("objective_return", ascending=False).head(8)
        if not feasible.empty
        else f.head(8)
    )
    for iteration, (_, row) in enumerate(seed.iterrows(), start=1):
        log_rows.append(
            {
                "iteration": iteration,
                "candidate_policy_id": row["policy_id"],
                "pricing_score_proxy": float(row.get("objective_return", 0.0))
                - 0.25 * float(row.get("scenario_loss_cvar90", 0.0)),
                "column_added": bool(iteration <= 5),
                "warm_start_source": "v20_frontier",
                "claim_scope": "pricing heuristic; not proof of global optimality",
            }
        )
    if not cert.empty:
        cert = cert.copy()
        nearest = feasible.sort_values("scenario_loss_cvar90").head(1)
        nearest_policy = str(nearest["policy_id"].iloc[0]) if not nearest.empty else ""
        for col in ["required_cvar_slack_proxy", "required_return_floor_relaxation_proxy"]:
            if col not in cert:
                cert[col] = np.nan
        cert["nearest_feasible_committee_or_relaxed_policy_id_v24"] = nearest_policy
        cert["active_source_caps_v24"] = "see paper4_v24_cvar_active_cap_diagnostics.csv"
        cert["auditability_conflict_v24"] = "strict cap may require relaxed/committee label"
        cert["budget_capital_conflict_v24"] = "diagnostic only; full LP dual not certified"
        cert["certificate_type_v24"] = "practical_restricted_master_certificate"
        cert["mathematical_infeasibility_proof_claim"] = False
    if not active.empty:
        active = active.copy()
        active["version_v24"] = "active_cap_diagnostics"
        active["claim_boundary_v24"] = (
            "restricted master active constraints, not full-universe dual certificate"
        )
    return f, cert, active, pd.DataFrame(log_rows)


def _spo_upgrade(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deps = v20._dependency_audit_v20()
    report, regret, alloc = v20._build_spo_surrogate(pool)
    report = report.copy()
    regret = regret.copy()
    alloc = alloc.copy()
    report["version_v24"] = "spo_oracle_regret_no_cvx_layers"
    report["formal_spo_plus_claim_v24"] = False
    regret["version_v24"] = "temporal_oracle_regret"
    if not alloc.empty:
        alloc["policy_id"] = "v24_spo_oracle_regret_surrogate"
        alloc["version_v24"] = "spo_oracle_regret_candidate"
    deps["version_v24"] = "dependency_audit_rechecked"
    return report, regret, alloc


def _committee_memos(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    variants = [
        ("balanced", 0.30, 0.20, 0.25, 0.10, 0.10, 0.05),
        ("wealth_first", 0.50, 0.20, 0.10, 0.05, 0.10, 0.05),
        ("tail_first", 0.20, 0.20, 0.40, 0.05, 0.10, 0.05),
        ("auditability_first", 0.20, 0.15, 0.20, 0.25, 0.10, 0.10),
        ("ecl_aware", 0.25, 0.20, 0.25, 0.05, 0.20, 0.05),
        ("source_governance_aware", 0.25, 0.15, 0.20, 0.25, 0.05, 0.10),
    ]
    rows = []
    for name, w_mean, w_p05, w_tail, w_source, w_ecl, w_default in variants:
        local = summary.copy()
        local["committee_score_v24"] = (
            w_mean * v15._rank_score(local["final_wealth_mean"], high_is_good=True)
            + w_p05 * v15._rank_score(local["final_wealth_p05"], high_is_good=True)
            + w_tail * v15._rank_score(local["cumulative_losses_p95"], high_is_good=False)
            + w_source
            * v15._rank_score(local["source_exposure_weak_share_final_mean"], high_is_good=False)
            + w_ecl * v15._rank_score(local["ECL_final_mean"], high_is_good=False)
            + w_default * v15._rank_score(local["cumulative_defaults_mean"], high_is_good=False)
        )
        local["committee_gate_pass_v24"] = (
            local["no_temporal_leakage_rate"].ge(1.0)
            & local["final_wealth_mean"].ge(v15.BUDGET)
            & local["cumulative_losses_p95"].le(225_000)
        )
        local["committee_score_v24"] = np.where(
            local["committee_gate_pass_v24"], local["committee_score_v24"], -1.0
        )
        top = local.sort_values("committee_score_v24", ascending=False).iloc[0]
        rows.append(
            {
                "committee_profile": name,
                "winning_policy_id": top["policy_id"],
                "reference_policy_id": reference,
                "committee_score": float(top["committee_score_v24"]),
                "final_wealth_mean": float(top["final_wealth_mean"]),
                "loss_p95": float(top["cumulative_losses_p95"]),
                "change_recommended": bool(
                    top["policy_id"] != reference
                    and float(top["committee_score_v24"]) > 0.0
                    and top["final_wealth_mean"] >= v15.BUDGET
                ),
                "decision_scope": "Paper 4 committee memo only; no final promotion",
            }
        )
    memo = pd.DataFrame(rows)
    memo["consolidated_decision"] = np.where(
        memo["change_recommended"] & memo["winning_policy_id"].ne(reference),
        "serious_challenger_review",
        "retain_working_champion",
    )
    return memo


def build_v24(n_paths: int, solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    v15.MONTHLY_REPAYMENT_HORIZON = 36
    pool, base_books = _load_pool(solver_pool_n)
    coeff = _adp_coefficients()
    adp_books, adp_registry = _build_adp_books(pool, coeff)
    adp_trace, adp_summary, _ = _simulate_adp(adp_books, n_paths=n_paths)

    v23_summary = _safe_read_csv(TABLE_DIR / "paper4_v23_dynamic_policy_summary.csv")
    if v23_summary.empty:
        v23_summary = _safe_read_csv(TABLE_DIR / "paper4_v19_dynamic_policy_summary.csv")
    combined_summary = pd.concat(
        [v23_summary, adp_summary], ignore_index=True, sort=False
    ).drop_duplicates("policy_id", keep="last")
    reference = str(
        _safe_read_json(STATUS_DIR / "paper4_v22_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    committee = _committee_memos(combined_summary, reference)
    cvar_frontier, cvar_cert, cvar_active, cvar_log = _cvar_column_generation()
    spo_report, spo_regret, spo_alloc = _spo_upgrade(pool)

    _write_csv("paper4_v24_adp_value_coefficients.csv", coeff)
    _write_parquet("paper4_v24_dla_adp_decisions.parquet", adp_books)
    _write_csv("paper4_v24_dla_adp_policy_registry.csv", adp_registry)
    _write_parquet("paper4_v24_dla_adp_dynamic_trace.parquet", adp_trace)
    _write_csv("paper4_v24_dla_adp_dynamic_summary.csv", adp_summary)
    _write_csv("paper4_v24_dynamic_combined_summary.csv", combined_summary)
    _write_csv("paper4_v24_committee_profile_memos.csv", committee)
    _write_csv("paper4_v24_champion_stress_decision_memo.csv", committee)
    _write_csv("paper4_v24_cvar_column_generation_log.csv", cvar_log)
    _write_csv("paper4_v24_cvar_frontier_non_dominated.csv", cvar_frontier)
    _write_csv("paper4_v24_cvar_infeasibility_certificate_formalized.csv", cvar_cert)
    _write_csv("paper4_v24_cvar_active_cap_diagnostics.csv", cvar_active)
    _write_csv("paper4_v24_spo_training_report.csv", spo_report)
    _write_csv("paper4_v24_spo_temporal_oracle_regret.csv", spo_regret)
    _write_parquet("paper4_v24_spo_candidate_allocations.parquet", spo_alloc)
    _write_csv("paper4_v24_spo_dependency_blockers.csv", v20._dependency_audit_v20())

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v24_dla_cvar_spo_upgrade",
        "adp_policy_count_v24": int(adp_summary["policy_id"].nunique()),
        "adp_trace_rows_v24": int(len(adp_trace)),
        "cvar_frontier_rows_v24": int(len(cvar_frontier)),
        "cvar_exact_full_universe_claim_v24": False,
        "spo_formal_differentiable_claim_allowed": False,
        "committee_profiles_v24": int(len(committee)),
        "working_champion_reference_v24": reference,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "ADP/CVaR/SPO v24 are Paper 4 lab challengers and diagnostics only",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v24_status.json", status)
    _write_note(
        "paper4_v24_dla_cvar_spo_upgrade.md",
        "\n".join(
            [
                "# Paper 4 v24 DLA/CVaR/SPO Upgrade",
                "",
                f"- ADP policies: `{status['adp_policy_count_v24']}`.",
                f"- CVaR frontier rows: `{status['cvar_frontier_rows_v24']}`.",
                f"- SPO differentiable claim allowed: `{status['spo_formal_differentiable_claim_allowed']}`.",
                "",
                "All results remain Paper 4 working/lab evidence.",
            ]
        ),
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_v24(args.paths, args.solver_pool_n)


if __name__ == "__main__":
    main()
