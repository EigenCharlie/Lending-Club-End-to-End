"""Build Paper 4 v20 DLA/CVaR/SPO resolution artifacts.

v20 turns the dynamic evidence from v19 into a stronger comparison layer:

* endogenous monthly DLA approximations built from the available candidate pool;
* CVaR/OCE strict-vs-committee diagnostics with restricted-master caveats;
* SPO/DFL dependency audit plus a decision-oracle regret surrogate;
* champion stress/decomposition under common paths.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
from scripts.papers.build_paper4_extended_experiments import (
    _safe_read_csv,
    _safe_read_json,
    _safe_read_parquet,
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

SCHEMA_VERSION = "2026-05-14.20"
FEATURES = [
    "loan_amnt",
    "int_rate_decimal",
    "installment",
    "pd_high_alpha01",
    "qhat_v4",
    "weak_source_proxy",
    "dti",
    "annual_inc",
    "fico_score",
    "score_decile",
]


def _load_solver_pool(max_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_universe, candidate_pool, _, _, online_intervals = _load_inputs()
    solver_source = base_universe if len(base_universe) > len(candidate_pool) else candidate_pool
    solver_pool = _prepare_solver_pool(
        solver_source, online_intervals, max_n=min(max_n, len(solver_source))
    )
    books, registry = v15._load_policy_books(solver_pool)
    if books.empty:
        raise RuntimeError("No policy books available for v20.")
    return solver_pool, books, registry


def _month_greedy_book(
    pool: pd.DataFrame,
    *,
    policy_id: str,
    strategy: str,
    total_book_budget: float,
) -> pd.DataFrame:
    df = pool.copy()
    df["issue_month"] = (
        pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    df = df.dropna(subset=["issue_month"]).copy()
    df["base"] = pd.to_numeric(df["base_return_vec"], errors="coerce").fillna(0.0)
    df["tail_loss_proxy"] = (
        pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["pd_high_alpha01"], errors="coerce").fillna(0.18)
        * np.maximum(
            pd.to_numeric(df.get("lgd", v15.DEFAULT_LGD), errors="coerce").fillna(v15.DEFAULT_LGD),
            v15.DEFAULT_LGD,
        )
    )
    df["width_penalty"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(
        0.0
    ) * pd.to_numeric(df["qhat_v4"], errors="coerce").fillna(0.55)
    df["weak_penalty"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(
        0.0
    ) * pd.to_numeric(df["weak_source_proxy"], errors="coerce").fillna(0.33)
    if strategy == "return":
        df["dla_score_v20"] = df["base"]
    elif strategy == "tail_guarded":
        df["dla_score_v20"] = df["base"] - 0.90 * df["tail_loss_proxy"] - 0.08 * df["width_penalty"]
    elif strategy == "source_balanced":
        df["dla_score_v20"] = df["base"] - 0.55 * df["weak_penalty"] - 0.12 * df["width_penalty"]
    elif strategy == "reinvestment":
        df["dla_score_v20"] = (
            df["base"]
            - 0.35 * df["tail_loss_proxy"]
            + 0.015 * pd.to_numeric(df["installment"], errors="coerce").fillna(0.0)
        )
    else:
        raise ValueError(f"Unknown v20 DLA strategy: {strategy}")

    selected = []
    cumulative = 0.0
    source_exposure: dict[tuple[str, str], float] = {}
    monthly_cap = v15.BUDGET * 0.30
    for _month, local in df.sort_values("issue_month").groupby("issue_month", sort=True):
        month_total = 0.0
        ranked = local.sort_values("dla_score_v20", ascending=False)
        for _, row in ranked.iterrows():
            amount = float(row.get("loan_amnt", row.get("funded_exposure", 0.0)))
            if (
                amount <= 0
                or cumulative + amount > total_book_budget
                or month_total + amount > monthly_cap
            ):
                continue
            if strategy == "source_balanced":
                grade_key = (
                    "original_grade",
                    str(row.get("original_grade", row.get("grade", "unknown"))),
                )
                state_key = (
                    "state_top20",
                    str(row.get("state_top20", row.get("addr_state", "unknown"))),
                )
                if source_exposure.get(grade_key, 0.0) + amount > 0.38 * max(
                    cumulative + amount, 1.0
                ):
                    continue
                if source_exposure.get(state_key, 0.0) + amount > 0.28 * max(
                    cumulative + amount, 1.0
                ):
                    continue
            rec = row.to_dict()
            rec["policy_id"] = policy_id
            rec["funded_exposure"] = amount
            rec["decision_rule_v20"] = f"monthly_endogenous_greedy_{strategy}"
            rec["state_cash_before_proxy_v20"] = max(v15.BUDGET - cumulative, 0.0)
            rec["post_decision_budget_proxy_v20"] = max(v15.BUDGET - cumulative - amount, 0.0)
            rec["source_pressure_proxy_v20"] = float(row.get("weak_source_proxy", 0.33))
            selected.append(rec)
            cumulative += amount
            month_total += amount
            if strategy == "source_balanced":
                for key in [grade_key, state_key]:
                    source_exposure[key] = source_exposure.get(key, 0.0) + amount
        if cumulative >= total_book_budget:
            break
    if not selected:
        return pd.DataFrame()
    return v15._standardize_book(
        pd.DataFrame(selected),
        source_artifact="v20_endogenous_dla_greedy",
        lane="dla_endogenous_v20",
    )


def _build_endogenous_dla_books(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = {
        "v20_dla_endogenous_return": "return",
        "v20_dla_endogenous_tail_guarded": "tail_guarded",
        "v20_dla_endogenous_source_balanced": "source_balanced",
        "v20_dla_endogenous_reinvestment": "reinvestment",
    }
    books = []
    rows = []
    for policy_id, strategy in strategies.items():
        book = _month_greedy_book(
            pool, policy_id=policy_id, strategy=strategy, total_book_budget=2.3 * v15.BUDGET
        )
        if book.empty:
            continue
        books.append(book)
        rows.append(
            {
                "policy_id": policy_id,
                "policy_class": "DLA/ADP-approx",
                "strategy": strategy,
                "adapter_type_v20": "endogenous_monthly_greedy_approximation",
                "selected_loans": int(book["loan_id"].nunique()),
                "book_exposure": float(book["funded_exposure"].sum()),
                "state_variables_used": "cash_proxy, issue_month, PD, conformal width, source pressure, expected return",
                "bellman_exact_claim": False,
                "claim_boundary": "monthly endogenous approximation, not exact Bellman optimality",
            }
        )
    return (pd.concat(books, ignore_index=True) if books else pd.DataFrame(), pd.DataFrame(rows))


def _load_v19_paths(default_paths: int, horizon: int, books: pd.DataFrame) -> pd.DataFrame:
    path = TABLE_DIR / "paper4_v19_sample_paths.parquet"
    if path.exists():
        out = pd.read_parquet(path)
        return out[out["path_id"].lt(default_paths)].copy()
    v15.MONTHLY_REPAYMENT_HORIZON = horizon
    _, _, out = v15.build_sample_paths_v15(books, n_paths=default_paths)
    return out


def _dependency_audit_v20() -> pd.DataFrame:
    rows = []
    for package in ["cvxpy", "cvxpylayers", "torch", "pyomo", "highspy", "catboost", "sklearn"]:
        spec = importlib.util.find_spec(package)
        available = False
        version = ""
        error = ""
        if spec is not None:
            try:
                mod = importlib.import_module(package)
                version = str(getattr(mod, "__version__", "installed"))
                available = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        else:
            error = "ModuleNotFoundError"
        if package in {"cvxpy", "cvxpylayers", "torch"} and not available:
            decision = "formal_differentiable_spo_blocked"
        elif package in {"pyomo", "highspy"} and available:
            decision = "usable_for_oracle_lp_route"
        elif package in {"catboost", "sklearn"} and available:
            decision = "usable_for_regret_surrogate"
        else:
            decision = "not_used"
        rows.append(
            {
                "package": package,
                "available_v20": available,
                "version": version,
                "decision_v20": decision,
                "blocker_detail": error,
                "formal_differentiable_spo_claim_allowed": bool(
                    package in {"cvxpy", "cvxpylayers", "torch"} and available
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_spo_surrogate(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pool.copy()
    df["issue_month"] = (
        pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    df = df.dropna(subset=["issue_month"]).copy()
    for col in FEATURES:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            df[col].median() if pd.to_numeric(df[col], errors="coerce").notna().any() else 0.0
        )
    y = (
        pd.to_numeric(df["base_return_vec"], errors="coerce").fillna(0.0)
        - 0.65
        * pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["pd_high_alpha01"], errors="coerce").fillna(0.18)
        * v15.DEFAULT_LGD
        - 0.05
        * pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["qhat_v4"], errors="coerce").fillna(0.55)
    )
    months = sorted(df["issue_month"].dropna().unique())
    cut1 = months[int(0.60 * len(months))] if months else df["issue_month"].min()
    cut2 = months[int(0.80 * len(months))] if months else df["issue_month"].max()
    split = np.where(
        df["issue_month"].lt(cut1),
        "train",
        np.where(df["issue_month"].lt(cut2), "validation", "test"),
    )
    preds = np.repeat(float(y.mean()), len(df))
    model_name = "mean_score_fallback"
    try:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=120,
            depth=5,
            learning_rate=0.08,
            loss_function="RMSE",
            verbose=False,
            random_seed=2026051420,
        )
        train_mask = split == "train"
        model.fit(df.loc[train_mask, FEATURES], y.loc[train_mask])
        preds = model.predict(df[FEATURES])
        model_name = "CatBoostRegressor_regret_surrogate"
    except Exception:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            train_mask = split == "train"
            model = HistGradientBoostingRegressor(
                max_iter=160, learning_rate=0.06, random_state=2026051420
            )
            model.fit(df.loc[train_mask, FEATURES], y.loc[train_mask])
            preds = model.predict(df[FEATURES])
            model_name = "HistGradientBoostingRegressor_regret_surrogate"
        except Exception:
            pass
    df["spo_regret_surrogate_score_v20"] = preds
    df["split_v20"] = split

    reports = []
    targets = []
    for split_name, local in df.groupby("split_v20"):
        if local.empty:
            continue
        error = (
            pd.to_numeric(local["spo_regret_surrogate_score_v20"], errors="coerce")
            - y.loc[local.index]
        )
        reports.append(
            {
                "model_id": "v20_spo_decision_oracle_surrogate",
                "model_name": model_name,
                "split": split_name,
                "rmse_target_proxy": float(np.sqrt(np.mean(error**2))),
                "mean_target_proxy": float(y.loc[local.index].mean()),
                "formal_differentiable_spo_plus": False,
                "claim_scope_v20": "decision-oracle regret surrogate, not formal differentiable SPO+",
            }
        )
        for month, month_df in local.groupby("issue_month"):
            oracle = month_df.sort_values("base_return_vec", ascending=False).head(40)
            candidate = month_df.sort_values(
                "spo_regret_surrogate_score_v20", ascending=False
            ).head(40)
            targets.append(
                {
                    "month": month,
                    "split": split_name,
                    "oracle_value_proxy": float(
                        pd.to_numeric(oracle["base_return_vec"], errors="coerce").sum()
                    ),
                    "candidate_value_proxy": float(
                        pd.to_numeric(candidate["base_return_vec"], errors="coerce").sum()
                    ),
                    "decision_regret_proxy_v20": float(
                        pd.to_numeric(oracle["base_return_vec"], errors="coerce").sum()
                        - pd.to_numeric(candidate["base_return_vec"], errors="coerce").sum()
                    ),
                    "oracle_route": "monthly_top40_return_proxy",
                    "candidate_route": model_name,
                }
            )
    selected = df.sort_values("spo_regret_surrogate_score_v20", ascending=False).copy()
    selected["cum_exposure"] = (
        pd.to_numeric(selected["loan_amnt"], errors="coerce").fillna(0.0).cumsum()
    )
    selected = selected[selected["cum_exposure"].le(2.0 * v15.BUDGET)].copy()
    selected["policy_id"] = "v20_spo_decision_oracle_surrogate"
    selected["funded_exposure"] = selected["loan_amnt"]
    selected["allocation_fraction"] = 1.0
    allocation = v15._standardize_book(
        selected,
        source_artifact="v20_spo_regret_surrogate",
        lane="spo_decision_oracle_surrogate_v20",
    )
    return pd.DataFrame(reports), pd.DataFrame(targets), allocation


def _cvar_v20() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier = _safe_read_csv(TABLE_DIR / "paper4_v16_cvar_full_or_colgen_frontier.csv")
    cert = _safe_read_csv(TABLE_DIR / "paper4_v16_cvar_strict_infeasibility_certificate.csv")
    active = _safe_read_csv(TABLE_DIR / "paper4_v16_cvar_active_constraints.csv")
    alloc = _safe_read_parquet(TABLE_DIR / "paper4_v16_cvar_allocations.parquet")
    if frontier.empty:
        return frontier, cert, active, alloc
    out = frontier.copy()
    out["regime_v20"] = np.select(
        [
            out.get("cap_relaxation_v13", "")
            .astype(str)
            .str.contains("strict", case=False, na=False),
            out.get("cap_relaxation_v13", "")
            .astype(str)
            .str.contains("committee", case=False, na=False),
            out.get("cap_relaxation_v13", "")
            .astype(str)
            .str.contains("relaxed", case=False, na=False),
        ],
        ["strict", "committee", "relaxed"],
        default="diagnostic",
    )
    out["full_universe_exact_claim_v20"] = False
    out["restricted_master_claim_boundary_v20"] = (
        "column-generation/restricted-master diagnostic; not full-universe exact optimality"
    )
    out["auditability_score_v20"] = (
        1.0
        - 0.20 * out["regime_v20"].eq("relaxed").astype(float)
        - 0.10 * out["regime_v20"].eq("committee").astype(float)
    )
    if "scenario_loss_cvar90" in out and "objective_return" in out:
        out["return_cvar_tradeoff_v20"] = pd.to_numeric(
            out["objective_return"], errors="coerce"
        ) - 0.35 * pd.to_numeric(out["scenario_loss_cvar90"], errors="coerce")
    else:
        out["return_cvar_tradeoff_v20"] = np.nan
    active = active.copy()
    if not active.empty:
        active["diagnostic_v20"] = (
            "active caps in restricted-master evidence; use for infeasibility narrative"
        )
    if not cert.empty:
        cert = cert.copy()
        cert["strict_infeasibility_label_v20"] = "practical_restricted_master_certificate"
    if not alloc.empty:
        alloc = alloc.copy()
        alloc["version_v20"] = "cvar_allocations_reused_for_dynamic_challenger_pool"
    return out, cert, active, alloc


def _champion_decomposition(
    books: pd.DataFrame, summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return v15.build_champion_decomposition_v16(books, summary)


def _decision_memo(summary: pd.DataFrame, pairwise: pd.DataFrame, current: str) -> pd.DataFrame:
    score_col = (
        "paper4_champion_score_v15"
        if "paper4_champion_score_v15" in summary
        else "dynamic_value_score_v15"
    )
    top = summary.sort_values(score_col, ascending=False).iloc[0]
    row_pair = pairwise[pairwise["policy_id"].eq(top["policy_id"])]
    prob = (
        float(row_pair["prob_higher_wealth"].iloc[0])
        if not row_pair.empty and "prob_higher_wealth" in row_pair
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "candidate_policy_id": top["policy_id"],
                "current_reference_policy_id": current,
                "working_champion_change_recommended_v20": bool(
                    top["policy_id"] != current
                    and top[score_col] > 0
                    and (pd.isna(prob) or prob >= 0.50)
                ),
                "governance_score_v20": float(top[score_col]),
                "prob_higher_wealth_vs_current": prob,
                "mean_final_wealth": float(top["final_wealth_mean"]),
                "loss_p95": float(top["cumulative_losses_p95"]),
                "decision_scope": "Paper 4 working champion only; no Paper Estrella promotion",
            }
        ]
    )


def build_v20(paths: int, horizon_months: int, solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    v15.MONTHLY_REPAYMENT_HORIZON = horizon_months
    pool, base_books, base_registry = _load_solver_pool(solver_pool_n)
    endogenous_books, dla_registry = _build_endogenous_dla_books(pool)
    _write_parquet("paper4_v20_endogenous_dla_decisions.parquet", endogenous_books)
    _write_csv("paper4_v20_endogenous_dla_state_policy_registry.csv", dla_registry)

    spo_report, spo_regret, spo_book = _build_spo_surrogate(pool)
    deps = _dependency_audit_v20()
    _write_csv("paper4_v20_spo_dependency_blockers.csv", deps)
    _write_csv("paper4_v20_spo_training_report.csv", spo_report)
    _write_csv("paper4_v20_spo_oracle_regret.csv", spo_regret)
    _write_parquet("paper4_v20_spo_candidate_allocations.parquet", spo_book)

    combined_books = pd.concat(
        [df for df in [base_books, endogenous_books, spo_book] if df is not None and not df.empty],
        ignore_index=True,
    )
    paths_df = _load_v19_paths(paths, horizon_months, combined_books)
    trace, summary, _ = v15.build_dynamic_engine_v15(combined_books, paths_df, n_paths=paths)
    summary = summary.copy()
    summary["version_v20"] = "dla_cvar_spo_dynamic_comparison"
    trace["version_v20"] = "dla_cvar_spo_dynamic_comparison"
    _write_parquet("paper4_v20_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v20_dynamic_policy_summary.csv", summary)
    _write_csv(
        "paper4_v20_endogenous_dla_policy_summary.csv",
        summary[summary["policy_id"].str.contains("v20_dla", regex=False)].copy(),
    )

    cvar_frontier, cvar_cert, cvar_active, cvar_alloc = _cvar_v20()
    _write_csv("paper4_v20_cvar_oce_frontier_v2.csv", cvar_frontier)
    _write_csv("paper4_v20_cvar_strict_infeasibility_v2.csv", cvar_cert)
    _write_csv("paper4_v20_cvar_active_caps_v2.csv", cvar_active)
    _write_parquet("paper4_v20_cvar_allocations.parquet", cvar_alloc)

    current = str(
        _safe_read_json(STATUS_DIR / "paper4_v18_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    pairwise = _pairwise_from_trace(trace, current)
    _write_csv("paper4_v20_champion_vs_challenger_dynamic_ci.csv", pairwise)
    decomp_summary, overlap, detail, cases = _champion_decomposition(combined_books, summary)
    memo = _decision_memo(summary, pairwise, current)
    _write_csv("paper4_v20_champion_decomposition_summary.csv", decomp_summary)
    _write_csv("paper4_v20_champion_overlap_matrix.csv", overlap)
    _write_parquet("paper4_v20_champion_selected_vs_avoided_loans.parquet", detail)
    _write_csv("paper4_v20_champion_case_studies.csv", cases)
    _write_csv("paper4_v20_champion_decision_memo.csv", memo)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v20_dla_cvar_spo_resolution",
        "dynamic_policy_count_v20": int(summary["policy_id"].nunique()),
        "endogenous_dla_policy_count_v20": int(dla_registry["policy_id"].nunique())
        if not dla_registry.empty
        else 0,
        "spo_formal_differentiable_claim_allowed": bool(
            deps["formal_differentiable_spo_claim_allowed"].all()
            and deps["package"].isin(["cvxpy", "cvxpylayers", "torch"]).any()
        ),
        "spo_dependency_blocker_count_v20": int((~deps["available_v20"].astype(bool)).sum()),
        "cvar_frontier_rows_v20": int(len(cvar_frontier)),
        "cvar_full_universe_exact_claim_v20": False,
        "working_champion_change_recommended_v20": bool(
            memo["working_champion_change_recommended_v20"].iloc[0]
        )
        if not memo.empty
        else False,
        "working_champion_candidate_v20": str(memo["candidate_policy_id"].iloc[0])
        if not memo.empty
        else "",
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "DLA/SPO/CVaR are Paper 4 lab challengers; no final promotion and no formal SPO+ claim unless dependency path passes",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v20_status.json", status)
    _write_note(
        "paper4_v20_dla_cvar_spo_resolution.md",
        "\n".join(
            [
                "# Paper 4 v20 DLA/CVaR/SPO Resolution",
                "",
                f"- Dynamic policies: `{status['dynamic_policy_count_v20']}`.",
                f"- Endogenous DLA policies: `{status['endogenous_dla_policy_count_v20']}`.",
                f"- SPO differentiable claim allowed: `{status['spo_formal_differentiable_claim_allowed']}`.",
                f"- CVaR full-universe exact claim: `{status['cvar_full_universe_exact_claim_v20']}`.",
                f"- Working champion candidate: `{status['working_champion_candidate_v20']}`.",
                "",
                "All decisions remain Paper 4 working/lab only.",
            ]
        ),
    )
    return status


def _pairwise_from_trace(trace: pd.DataFrame, reference: str) -> pd.DataFrame:
    final = trace.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    base = final[final["policy_id"].eq(reference)][
        ["path_id", "wealth", "cumulative_losses"]
    ].rename(columns={"wealth": "reference_wealth", "cumulative_losses": "reference_loss"})
    if base.empty:
        reference = str(
            final.groupby("policy_id")["wealth"].mean().sort_values(ascending=False).index[0]
        )
        base = final[final["policy_id"].eq(reference)][
            ["path_id", "wealth", "cumulative_losses"]
        ].rename(columns={"wealth": "reference_wealth", "cumulative_losses": "reference_loss"})
    rows = []
    for policy_id, local in final.groupby("policy_id"):
        merged = local.merge(base, on="path_id", how="inner")
        if merged.empty:
            continue
        diff = merged["wealth"] - merged["reference_wealth"]
        loss_diff = merged["cumulative_losses"] - merged["reference_loss"]
        se = float(diff.std(ddof=1) / np.sqrt(max(len(diff), 1))) if len(diff) > 1 else 0.0
        rows.append(
            {
                "policy_id": policy_id,
                "reference_policy_id": reference,
                "n_common_paths": int(len(diff)),
                "mean_wealth_diff": float(diff.mean()),
                "ci95_low_wealth_diff": float(diff.mean() - 1.96 * se),
                "ci95_high_wealth_diff": float(diff.mean() + 1.96 * se),
                "prob_higher_wealth": float((diff > 0).mean()),
                "mean_loss_diff": float(loss_diff.mean()),
                "prob_lower_loss": float((loss_diff < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    status = build_v20(args.paths, args.horizon_months, args.solver_pool_n)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
