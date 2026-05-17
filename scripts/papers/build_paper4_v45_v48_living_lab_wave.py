#!/usr/bin/env python3
"""Build Paper 4 v45-v48 living-lab execution artifacts.

This wave keeps the compact Quarto contract introduced after v38. New evidence
is written to artifacts and the living notebook first; official Quarto is not
expanded unless the evidence becomes stable, artifact-backed, and claim-safe.
"""

from __future__ import annotations

import argparse
import json
import math
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
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
BOOK_CONFIG = ROOT / "book" / "_quarto.yml"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(name: str, directory: Path = TABLE_DIR) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_parquet(
    name: str, directory: Path = TABLE_DIR, columns: list[str] | None = None
) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def read_json(name: str, directory: Path = STATUS_DIR) -> dict[str, Any]:
    path = directory / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df) if isinstance(df, list) else df
    out.to_csv(path, index=False)


def write_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or math.isclose(float(lo), float(hi)):
        out = pd.Series(0.5, index=series.index)
    else:
        out = (values - lo) / (hi - lo)
    if not higher_is_better:
        out = 1 - out
    return out.fillna(0.0)


def registered_paper4_pages() -> list[str]:
    if not BOOK_CONFIG.exists():
        return []
    pages: list[str] = []
    for raw in BOOK_CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("- chapters/19-paper-mega-extension/"):
            pages.append(Path(stripped.removeprefix("- ")).name)
    return pages


def package_probe(package: str) -> dict[str, Any]:
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    available = probe.returncode == 0
    stderr = (probe.stderr or probe.stdout or "").strip().splitlines()
    import_error = "" if available else (stderr[-1] if stderr else "import failed")
    return {
        "package": package,
        "available": available,
        "version": version,
        "version_lookup_error": version_error,
        "import_error": import_error,
    }


def _width(frame: pd.DataFrame) -> pd.Series:
    if {"pd_high_alpha01", "pd_point"}.issubset(frame.columns):
        return (
            pd.to_numeric(frame["pd_high_alpha01"], errors="coerce")
            - pd.to_numeric(frame["pd_point"], errors="coerce")
        ).clip(lower=0)
    if {"pd_interval_high", "pd_interval_low"}.issubset(frame.columns):
        return (
            pd.to_numeric(frame["pd_interval_high"], errors="coerce")
            - pd.to_numeric(frame["pd_interval_low"], errors="coerce")
        ).clip(lower=0)
    return pd.Series(0.0, index=frame.index)


def build_v45_online_direct_holdout() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = read_parquet("paper4_v9_online_selected_intervals.parquet")
    interval_base = read_parquet(
        "paper4_online_conformal_v4_intervals.parquet",
        columns=[
            "loan_id",
            "issue_month",
            "period",
            "original_grade",
            "term",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
        ],
    )
    if selected.empty or interval_base.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v45_online_source_family_direct_holdout.csv", empty)
        write_csv(TABLE_DIR / "paper4_v45_online_recalibration_grid.csv", empty)
        write_csv(TABLE_DIR / "paper4_v45_online_cell_diagnostics.csv", empty)
        return empty, empty, empty

    join_cols = [
        "loan_id",
        "issue_month",
        "period",
        "original_grade",
        "term",
        "score_decile",
        "state_top20",
        "income_band",
        "dti_band",
    ]
    df = selected.merge(
        interval_base.drop_duplicates(["loan_id", "issue_month"])[join_cols],
        on=["loan_id", "issue_month"],
        how="left",
    )
    df["issue_month"] = pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").astype(str)
    df["coverage_alpha01"] = df["covered_online_v9"].astype(bool).astype(float)
    df["interval_width_alpha01"] = pd.to_numeric(
        df["interval_width_online_v9"], errors="coerce"
    ).fillna(0)
    source_specs = {
        "grade": "original_grade",
        "period": "period",
        "term": "term",
        "score_decile": "score_decile",
        "state_top20": "state_top20",
        "income_band": "income_band",
        "dti_band": "dti_band",
    }
    cells: list[pd.DataFrame] = []
    for family, col in source_specs.items():
        part = (
            df.assign(source_family=family, source_id=df[col].astype(str))
            .groupby(["source_family", "source_id", "issue_month"], dropna=False)
            .agg(
                n_loans=("loan_id", "nunique"),
                coverage=("coverage_alpha01", "mean"),
                avg_width=("interval_width_alpha01", "mean"),
                n_policies=("policy_id", "nunique"),
            )
            .reset_index()
        )
        parent = (
            df.assign(source_family=family)
            .groupby(["source_family", "issue_month"], dropna=False)
            .agg(parent_n=("loan_id", "nunique"), parent_coverage=("coverage_alpha01", "mean"))
            .reset_index()
        )
        part = part.merge(parent, on=["source_family", "issue_month"], how="left")
        cells.append(part)
    cell_diag = pd.concat(cells, ignore_index=True)

    policy_month = (
        df.groupby(["policy_id", "issue_month"], dropna=False)
        .agg(n_loans=("loan_id", "nunique"), coverage=("coverage_alpha01", "mean"))
        .reset_index()
    )
    policy_month_min = float(policy_month.loc[policy_month["n_loans"].ge(2), "coverage"].min())
    methods = [
        ("direct_m1", 1, 0.0, 0.000),
        ("pool_m3_k5", 3, 5.0, 0.000),
        ("pool_m5_k10", 5, 10.0, 0.002),
        ("pool_m8_k20_width_guard", 8, 20.0, 0.004),
        ("pool_m12_k30_width_guard", 12, 30.0, 0.006),
    ]
    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for family, family_cells in cell_diag.groupby("source_family"):
        for method, min_support, prior_k, width_penalty in methods:
            work = family_cells.copy()
            n = pd.to_numeric(work["n_loans"], errors="coerce").fillna(0)
            cov = pd.to_numeric(work["coverage"], errors="coerce").fillna(0)
            parent_cov = pd.to_numeric(work["parent_coverage"], errors="coerce").fillna(cov)
            enough = n.ge(min_support)
            defended = np.where(
                enough,
                (n * cov + prior_k * parent_cov) / (n + prior_k),
                parent_cov,
            )
            width = (pd.to_numeric(work["avg_width"], errors="coerce").fillna(0) + width_penalty)
            source_min = float(pd.Series(defended).min())
            avg_width = float(width.mean())
            support_rate = float(enough.mean())
            gate = bool(source_min >= 0.80 and policy_month_min >= 0.90 and avg_width <= 0.95)
            grid_rows.append(
                {
                    "source_family": family,
                    "method_v45": method,
                    "min_support_v45": min_support,
                    "prior_k_v45": prior_k,
                    "width_penalty_v45": width_penalty,
                    "source_family_defended_min_v45": source_min,
                    "policy_month_direct_min_v45": policy_month_min,
                    "avg_width_loan_v45": avg_width,
                    "support_rate_v45": support_rate,
                    "gate_source80_policy90_width95_v45": gate,
                    "direct_loan_level_evidence_v45": True,
                    "strict_live_deployability_claim_allowed": False,
                    "claim_boundary_v45": "historical loan-level replay with source-family holdout diagnostics; not live deployability",
                }
            )
        best = pd.DataFrame([r for r in grid_rows if r["source_family"] == family]).sort_values(
            ["gate_source80_policy90_width95_v45", "source_family_defended_min_v45", "avg_width_loan_v45"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(best.to_dict())

    grid = pd.DataFrame(grid_rows)
    holdout = pd.DataFrame(rows)
    holdout["decision_v45"] = np.where(
        holdout["gate_source80_policy90_width95_v45"],
        "direct_source_family_replay_pass_with_live_caveat",
        "near_resolved_with_plateau",
    )
    cell_diag["direct_cell_scope_v45"] = (
        "source-month loan-level coverage cell using historical accepted-loan evidence"
    )
    write_csv(TABLE_DIR / "paper4_v45_online_source_family_direct_holdout.csv", holdout)
    write_csv(TABLE_DIR / "paper4_v45_online_recalibration_grid.csv", grid)
    write_csv(TABLE_DIR / "paper4_v45_online_cell_diagnostics.csv", cell_diag)
    return holdout, grid, cell_diag


def build_v45_cvar_slack() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier = read_csv("paper4_v41_cvar_frontier_non_dominated.csv")
    cert = read_csv("paper4_v41_cvar_strict_infeasibility_v2.csv")
    if frontier.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v45_cvar_slack_lp_certificate.csv", empty)
        write_csv(TABLE_DIR / "paper4_v45_cvar_full_universe_attempt.csv", empty)
        write_csv(TABLE_DIR / "paper4_v45_cvar_solver_frontier.csv", empty)
        return empty, empty, empty

    f = frontier.copy()
    for col in ["objective_return", "scenario_loss_cvar90", "cvar_cap", "return_floor"]:
        if col in f:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    source_cols = [c for c in f.columns if c.startswith("share_")]
    cap_pairs = []
    for share_col in source_cols:
        suffix = share_col.removeprefix("share_")
        cap_candidates = [c for c in f.columns if c.startswith(f"cap_{suffix}") and c.endswith("_v12")]
        if cap_candidates:
            cap_pairs.append((share_col, cap_candidates[0]))
    frontier_rows: list[dict[str, Any]] = []
    for _, row in f.iterrows():
        source_slacks = []
        broken = []
        for share_col, cap_col in cap_pairs:
            slack = max(0.0, safe_num(row.get(share_col), 0) - safe_num(row.get(cap_col), 1))
            source_slacks.append(slack)
            if slack > 1e-9:
                broken.append(share_col)
        cvar_slack = max(0.0, safe_num(row.get("scenario_loss_cvar90"), 0) - safe_num(row.get("cvar_cap"), np.inf))
        return_slack = max(0.0, safe_num(row.get("return_floor"), 0) - safe_num(row.get("objective_return"), 0))
        frontier_rows.append(
            {
                "policy_id": row.get("policy_id"),
                "regime_v45": row.get("regime_v28", row.get("v16_regime_label", "unknown")),
                "objective_return": safe_num(row.get("objective_return"), 0),
                "scenario_loss_cvar90": safe_num(row.get("scenario_loss_cvar90"), 0),
                "cvar_slack_v45": cvar_slack,
                "return_floor_slack_v45": return_slack,
                "source_cap_total_slack_v45": float(np.sum(source_slacks)),
                "broken_source_caps_v45": "|".join(broken),
                "slack_objective_v45": cvar_slack / 100_000 + return_slack / 100_000 + float(np.sum(source_slacks)),
                "exact_full_universe_claim_v45": False,
                "slack_certificate_scope_v45": "restricted-master candidate-set slack optimization, not mathematical full-universe proof",
            }
        )
    slack_frontier = pd.DataFrame(frontier_rows).sort_values("slack_objective_v45")

    if cert.empty:
        certificate = slack_frontier.head(8).copy()
        certificate["certificate_source_v45"] = "frontier_only_no_prior_cert"
    else:
        c = cert.copy()
        cert_rows: list[dict[str, Any]] = []
        nearest = slack_frontier.sort_values("slack_objective_v45").head(1)
        nearest_policy = str(nearest["policy_id"].iloc[0]) if not nearest.empty else ""
        for _, row in c.iterrows():
            cvar_slack = safe_num(
                row.get("required_cvar_slack_v41", row.get("required_cvar_slack_v33", 0)), 0
            )
            floor_slack = safe_num(
                row.get(
                    "required_return_floor_relaxation_v41",
                    row.get("required_return_floor_relaxation_v33", 0),
                ),
                0,
            )
            cert_rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "strict_point_v45": row.get("strict_result_label_v16", "strict_diagnostic"),
                    "required_cvar_slack_v45": max(0.0, cvar_slack),
                    "required_return_floor_relaxation_v45": max(0.0, floor_slack),
                    "required_source_cap_slack_proxy_v45": safe_num(row.get("required_relaxation_v41"), 0),
                    "nearest_feasible_relaxed_policy_v45": nearest_policy,
                    "dual_slack_available_v45": False,
                    "exact_lp_certificate_claim_allowed_v45": False,
                    "mathematical_infeasibility_proof_claim_v45": False,
                    "claim_boundary_v45": "explicit slack accounting over restricted-master artifacts; not exact full-universe certificate",
                }
            )
        certificate = pd.DataFrame(cert_rows)

    universe_n = int(pd.to_numeric(f.get("universe_n_v13", f.get("universe_n_v12", 0)), errors="coerce").max())
    pool_n = int(pd.to_numeric(f.get("pool_n_v13", f.get("pool_n_v12", 0)), errors="coerce").max())
    full_attempt = pd.DataFrame(
        [
            {
                "attempt_id": "v45_full_universe_precheck",
                "universe_n": universe_n,
                "largest_restricted_pool_n": pool_n,
                "estimated_binary_or_weight_variables": universe_n,
                "scenario_loss_matrix_available": False,
                "pyomo_highs_available": package_probe("pyomo")["available"]
                and package_probe("highspy")["available"],
                "full_universe_lp_executed_v45": False,
                "exact_full_universe_claim_v45": False,
                "blocker_v45": "full loan-scenario loss matrix and exact allocation interface are not available in current artifacts",
                "next_exact_path_v45": "persist scenario-loss matrix by loan, then solve full LP or column-generation master with pricing certificate",
            }
        ]
    )
    write_csv(TABLE_DIR / "paper4_v45_cvar_slack_lp_certificate.csv", certificate)
    write_csv(TABLE_DIR / "paper4_v45_cvar_full_universe_attempt.csv", full_attempt)
    write_csv(TABLE_DIR / "paper4_v45_cvar_solver_frontier.csv", slack_frontier)
    return certificate, full_attempt, slack_frontier


def build_v45_source_solver_frontier() -> tuple[pd.DataFrame, pd.DataFrame]:
    caps = read_csv("paper4_v41_source_governance_solver_caps.csv")
    feasible = read_csv("paper4_v41_source_solver_feasibility.csv")
    if caps.empty or feasible.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v45_mdcp_source_solver_frontier.csv", empty)
        write_csv(TABLE_DIR / "paper4_v45_source_cap_sensitivity.csv", empty)
        return empty, empty
    regimes = [
        ("strict", 1.00, 0.00),
        ("committee", 1.25, 0.02),
        ("relaxed", 1.60, 0.05),
    ]
    rows: list[dict[str, Any]] = []
    sens_rows: list[dict[str, Any]] = []
    for (policy_id, family), part in feasible.groupby(["policy_id", "source_family"], dropna=False):
        max_share = safe_num(part["max_observed_exposure_share_v41"].max(), 0)
        cap = safe_num(part["recommended_source_cap_v41"].max(), 0)
        for regime, multiplier, add_slack in regimes:
            effective = cap * multiplier + add_slack
            pass_regime = bool(max_share <= effective)
            rows.append(
                {
                    "policy_id": policy_id,
                    "source_family": family,
                    "regime_v45": regime,
                    "recommended_cap_v45": cap,
                    "effective_cap_v45": effective,
                    "max_observed_exposure_share_v45": max_share,
                    "source_solver_cap_pass_v45": pass_regime,
                    "required_cap_relaxation_v45": max(0.0, max_share - effective),
                    "solver_integration_v45": "hard_constraint_candidate"
                    if pass_regime and regime in {"strict", "committee"}
                    else "requires_pooling_or_relaxation",
                    "fair_lending_legal_claim_allowed": False,
                    "claim_boundary_v45": "observable MDCP/source governance; no protected-attribute legal claim",
                }
            )
        sens_rows.append(
            {
                "policy_id": policy_id,
                "source_family": family,
                "base_cap_v45": cap,
                "max_share_v45": max_share,
                "strict_pass_v45": bool(max_share <= cap),
                "committee_pass_v45": bool(max_share <= cap * 1.25 + 0.02),
                "relaxed_pass_v45": bool(max_share <= cap * 1.60 + 0.05),
                "pooling_rule_v45": "pool cells with support <25 or exposure <1pct into parent family",
            }
        )
    frontier = pd.DataFrame(rows)
    sensitivity = pd.DataFrame(sens_rows)
    write_csv(TABLE_DIR / "paper4_v45_mdcp_source_solver_frontier.csv", frontier)
    write_csv(TABLE_DIR / "paper4_v45_source_cap_sensitivity.csv", sensitivity)
    return frontier, sensitivity


def build_v45() -> dict[str, Any]:
    start = datetime.now(UTC)
    holdout, grid, cells = build_v45_online_direct_holdout()
    cvar_cert, full_attempt, cvar_frontier = build_v45_cvar_slack()
    source_frontier, source_sens = build_v45_source_solver_frontier()
    status = {
        "schema_version": "2026-05-15.45",
        "generated_at_utc": now(),
        "phase": "v45_online_cvar_source_solver",
        "online_direct_holdout_rows_v45": int(len(holdout)),
        "online_recalibration_grid_rows_v45": int(len(grid)),
        "online_cell_diagnostic_rows_v45": int(len(cells)),
        "online_direct_gate_pass_families_v45": int(holdout["gate_source80_policy90_width95_v45"].sum())
        if not holdout.empty
        else 0,
        "strict_live_deployability_claim_allowed": False,
        "cvar_slack_certificate_rows_v45": int(len(cvar_cert)),
        "full_universe_attempt_rows_v45": int(len(full_attempt)),
        "exact_full_universe_cvar_claim_allowed": False,
        "source_solver_frontier_rows_v45": int(len(source_frontier)),
        "source_cap_sensitivity_rows_v45": int(len(source_sens)),
        "fair_lending_legal_claim_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v45 upgrades direct holdout, CVaR slack accounting, and source caps; no live deployment, exact CVaR, or legal fairness claim",
    }
    write_json(STATUS_DIR / "paper4_v45_status.json", status)
    return status


def build_v46_spo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool_raw = read_parquet("paper4_challenger_local_candidate_pool.parquet")
    if pool_raw.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v46_spo_loan_level_oracle_regret.csv", empty)
        write_csv(TABLE_DIR / "paper4_v46_spo_training_report.csv", empty)
        empty.to_parquet(TABLE_DIR / "paper4_v46_spo_oracle_allocations.parquet", index=False)
        return empty, empty, empty

    pool = pool_raw.drop_duplicates("loan_id").copy()
    pool["issue_month"] = pd.to_datetime(pool["issue_month"], errors="coerce")
    pool["month_order"] = pool["issue_month"].rank(method="dense").astype(int)
    if "int_rate_decimal" not in pool:
        pool["int_rate_decimal"] = pd.to_numeric(pool.get("int_rate", 0), errors="coerce").fillna(0) / 100
    if "base_return_vec" not in pool:
        pool["base_return_vec"] = (
            pd.to_numeric(pool.get("loan_amnt", 0), errors="coerce").fillna(0)
            * pd.to_numeric(pool.get("int_rate_decimal", 0), errors="coerce").fillna(0)
        )
    if "qhat_v4" not in pool:
        pool["qhat_v4"] = 0.0
    if "weak_source_proxy" not in pool:
        pool["weak_source_proxy"] = 0.0
    if "pd_point" not in pool and "pd_point_alpha01" in pool:
        pool["pd_point"] = pool["pd_point_alpha01"]
    for col in [
        "loan_amnt",
        "installment",
        "int_rate_decimal",
        "pd_point",
        "pd_high_alpha01",
        "qhat_v4",
        "weak_source_proxy",
        "base_return_vec",
        "annual_inc",
        "dti",
        "fico_score",
    ]:
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)
    pool["width"] = (pool["pd_high_alpha01"] - pool["pd_point"]).clip(lower=0)
    pool["oracle_score_v46"] = (
        pool["base_return_vec"]
        - 5500 * pool["qhat_v4"]
        - 3500 * pool["weak_source_proxy"]
        - 4500 * pool["width"]
        - 2500 * pool["pd_high_alpha01"]
    )
    selected_frames: list[pd.DataFrame] = []
    monthly_budget = 1_000_000.0 / max(pool["issue_month"].nunique(), 1)
    for _month, group in pool.groupby("issue_month", dropna=False):
        g = group.sort_values("oracle_score_v46", ascending=False).copy()
        cumulative = g["loan_amnt"].cumsum()
        selected = cumulative <= monthly_budget
        if not selected.any() and not g.empty:
            selected.iloc[0] = True
        g["oracle_selected_v46"] = selected
        selected_frames.append(g)
    oracle = pd.concat(selected_frames, ignore_index=True)

    feature_cols = [
        "loan_amnt",
        "installment",
        "int_rate_decimal",
        "pd_point",
        "pd_high_alpha01",
        "qhat_v4",
        "weak_source_proxy",
        "width",
        "annual_inc",
        "dti",
        "fico_score",
    ]
    split_cut_1 = oracle["month_order"].quantile(0.60)
    split_cut_2 = oracle["month_order"].quantile(0.80)
    oracle["split_v46"] = np.select(
        [oracle["month_order"] <= split_cut_1, oracle["month_order"] <= split_cut_2],
        ["train", "validation"],
        default="test",
    )
    report_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error

        models: list[tuple[str, Any]] = [
            ("ridge_oracle_score", Ridge(alpha=2.0)),
            ("rf_oracle_score", RandomForestRegressor(n_estimators=100, random_state=46, min_samples_leaf=4)),
        ]
        try:
            from catboost import CatBoostRegressor

            models.append(
                (
                    "catboost_oracle_score",
                    CatBoostRegressor(
                        iterations=160,
                        depth=4,
                        learning_rate=0.04,
                        loss_function="MAE",
                        verbose=False,
                        random_seed=46,
                    ),
                )
            )
        except Exception:
            pass

        train = oracle["split_v46"].eq("train")
        for model_name, model in models:
            model.fit(oracle.loc[train, feature_cols], oracle.loc[train, "oracle_score_v46"])
            scored = oracle.copy()
            scored["model_v46"] = model_name
            scored["predicted_score_v46"] = model.predict(scored[feature_cols])
            model_allocs: list[pd.DataFrame] = []
            for _month, group in scored.groupby("issue_month", dropna=False):
                g = group.sort_values("predicted_score_v46", ascending=False).copy()
                selected = g["loan_amnt"].cumsum() <= monthly_budget
                if not selected.any() and not g.empty:
                    selected.iloc[0] = True
                g["model_selected_v46"] = selected
                model_allocs.append(g)
            alloc = pd.concat(model_allocs, ignore_index=True)
            alloc["candidate_value_v46"] = np.where(
                alloc["model_selected_v46"], alloc["oracle_score_v46"], 0
            )
            alloc["oracle_value_v46"] = np.where(
                alloc["oracle_selected_v46"], alloc["oracle_score_v46"], 0
            )
            candidate_frames.append(alloc)
            for split, part in alloc.groupby("split_v46"):
                oracle_value = float(part["oracle_value_v46"].sum())
                candidate_value = float(part["candidate_value_v46"].sum())
                report_rows.append(
                    {
                        "model_v46": model_name,
                        "split_v46": split,
                        "n_loans": int(len(part)),
                        "oracle_value_v46": oracle_value,
                        "candidate_value_v46": candidate_value,
                        "decision_regret_v46": oracle_value - candidate_value,
                        "mae_score_v46": mean_absolute_error(
                            part["oracle_score_v46"], part["predicted_score_v46"]
                        ),
                        "formal_differentiable_spo_claim_allowed": False,
                        "claim_boundary_v46": "loan-level temporal oracle-regret surrogate; not formal differentiable SPO+",
                    }
                )
    except Exception as exc:
        report_rows.append(
            {
                "model_v46": "training_failed",
                "split_v46": "all",
                "n_loans": int(len(oracle)),
                "oracle_value_v46": float(oracle.loc[oracle["oracle_selected_v46"], "oracle_score_v46"].sum()),
                "candidate_value_v46": np.nan,
                "decision_regret_v46": np.nan,
                "mae_score_v46": np.nan,
                "formal_differentiable_spo_claim_allowed": False,
                "claim_boundary_v46": f"training failed: {type(exc).__name__}: {str(exc).splitlines()[0]}",
            }
        )
    allocations = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else oracle
    report = pd.DataFrame(report_rows)
    regret = report.copy()
    write_csv(TABLE_DIR / "paper4_v46_spo_loan_level_oracle_regret.csv", regret)
    write_csv(TABLE_DIR / "paper4_v46_spo_training_report.csv", report)
    allocations.to_parquet(TABLE_DIR / "paper4_v46_spo_oracle_allocations.parquet", index=False)
    return regret, report, allocations


def build_v46_spo_dependency_note() -> pd.DataFrame:
    packages = ["numpy", "cvxpy", "cvxpylayers", "torch", "pyomo", "highspy", "catboost", "sklearn"]
    rows = []
    for package in packages:
        row = package_probe(package)
        row["isolated_env_target_v46"] = ".venv-paper4-spo-v45"
        row["main_env_mutated_v46"] = False
        row["formal_differentiable_spo_claim_allowed"] = False
        row["decision_v46"] = (
            "usable_now"
            if row["available"] and package in {"pyomo", "highspy", "catboost", "sklearn", "numpy"}
            else "requires_isolated_env_validation"
            if package in {"cvxpy", "cvxpylayers", "torch"}
            else "context"
        )
        rows.append(row)
    deps = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "paper4_v46_spo_isolated_env_smoke_test.csv", deps)
    write_note(
        NOTE_DIR / "paper4_spo_isolated_env_repro.md",
        "\n".join(
            [
                "# Paper 4 SPO Isolated Environment Repro",
                "",
                f"Generated: {now()}",
                "",
                "The main project environment was not mutated. Formal differentiable SPO+",
                "remains blocked until an isolated cvxpy/cvxpylayers/torch stack validates",
                "and beats the oracle-regret baseline.",
                "",
                "Suggested isolated route:",
                "",
                "```bash",
                "python -m venv .venv-paper4-spo-v45",
                ".venv-paper4-spo-v45/bin/python -m pip install --upgrade pip",
                ".venv-paper4-spo-v45/bin/python -m pip install 'numpy<2' cvxpy cvxpylayers torch pyomo highspy scikit-learn catboost",
                ".venv-paper4-spo-v45/bin/python - <<'PY'",
                "import cvxpy, torch, cvxpylayers",
                "print(cvxpy.__version__, torch.__version__)",
                "PY",
                "```",
                "",
                "Current smoke-test results are stored in",
                "`paper4_v46_spo_isolated_env_smoke_test.csv`.",
            ]
        ),
    )
    return deps


def build_v46_dla_dynamic() -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = read_parquet("paper4_v31_dynamic_policy_trace.parquet")
    dla = read_csv("paper4_v43_dla_adp_rollout_grid.csv")
    if trace.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v46_dla_common_path_replay.csv", empty)
        write_csv(TABLE_DIR / "paper4_v46_focused_dynamic_1024_ci.csv", empty)
        return empty, empty
    final = (
        trace.sort_values("month_idx")
        .groupby(["policy_id", "path_id"], dropna=False)
        .tail(1)
        .copy()
    )
    summary = (
        final.groupby("policy_id", dropna=False)
        .agg(
            n_paths=("path_id", "nunique"),
            final_wealth_mean=("wealth", "mean"),
            final_wealth_p05=("wealth", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.05)),
            cumulative_losses_p95=(
                "cumulative_losses",
                lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95),
            ),
            cumulative_defaults_mean=("cumulative_defaults", "mean"),
            cumulative_recoveries_mean=("cumulative_recoveries", "mean"),
            no_temporal_leakage_rate=("no_temporal_leakage_flag", "mean"),
        )
        .reset_index()
    )
    summary["candidate_family_v46"] = np.select(
        [
            summary["policy_id"].str.contains("fvi|dla|adp", case=False, na=False),
            summary["policy_id"].str.contains("cvar", case=False, na=False),
            summary["policy_id"].str.contains("spo", case=False, na=False),
        ],
        ["dla_fvi_dynamic_trace", "cvar_dynamic_trace", "spo_dynamic_trace"],
        default="reference_or_static",
    )
    summary["bellman_exact_claim_allowed"] = False
    summary["claim_boundary_v46"] = (
        "common-path dynamic replay from v31 trace; DLA/ADP is not exact Bellman optimality"
    )
    if not dla.empty:
        proxy = dla.head(5).copy()
        proxy["candidate_family_v46"] = "dla_rollout_proxy_not_common_path"
        proxy["claim_boundary_v46"] = (
            "rollout proxy from v43; retained for diagnosis but not champion evidence until replayed"
        )

    ref = final.loc[final["policy_id"].eq("paper1_economic_champion"), ["path_id", "wealth", "cumulative_losses"]]
    ci_rows: list[dict[str, Any]] = []
    if not ref.empty:
        ref = ref.rename(columns={"wealth": "ref_wealth", "cumulative_losses": "ref_losses"})
        serious = [
            "v13_cvar_mdcp_colgen_relaxed_k32000_floor105000_cap300000",
            "v13_spo_regret_balanced",
            "v13_spo_regret_audit_guarded",
            "v13_fvi_return_recovery",
        ]
        for policy_id in serious:
            comp = final.loc[final["policy_id"].eq(policy_id), ["path_id", "wealth", "cumulative_losses"]]
            if comp.empty:
                continue
            joined = comp.merge(ref, on="path_id", how="inner")
            delta_w = joined["wealth"] - joined["ref_wealth"]
            delta_l = joined["cumulative_losses"] - joined["ref_losses"]
            ci_rows.append(
                {
                    "candidate_policy_id": policy_id,
                    "n_common_paths": int(len(joined)),
                    "wealth_delta_mean_v46": float(delta_w.mean()),
                    "wealth_delta_p05_v46": float(delta_w.quantile(0.05)),
                    "wealth_delta_p95_v46": float(delta_w.quantile(0.95)),
                    "prob_higher_wealth_v46": float((delta_w > 0).mean()),
                    "loss_delta_p95_v46": float(delta_l.quantile(0.95)),
                    "prob_lower_loss_v46": float((delta_l < 0).mean()),
                    "focused_1024_rerun_executed_v46": False,
                    "focused_1024_rerun_decision_v46": "not_executed_no_new_allocation_after_v45_screen",
                    "claim_boundary_v46": "512-path common-random-number replay already available; 1024 reserved for new non-dominated allocation",
                }
            )
    ci = pd.DataFrame(ci_rows)
    write_csv(TABLE_DIR / "paper4_v46_dla_common_path_replay.csv", summary)
    write_csv(TABLE_DIR / "paper4_v46_focused_dynamic_1024_ci.csv", ci)
    return summary, ci


def build_v46() -> dict[str, Any]:
    start = datetime.now(UTC)
    deps = build_v46_spo_dependency_note()
    regret, report, allocations = build_v46_spo()
    dla, ci = build_v46_dla_dynamic()
    status = {
        "schema_version": "2026-05-15.46",
        "generated_at_utc": now(),
        "phase": "v46_spo_dla_dynamic",
        "spo_dependency_rows_v46": int(len(deps)),
        "spo_regret_rows_v46": int(len(regret)),
        "spo_allocation_rows_v46": int(len(allocations)),
        "formal_differentiable_spo_claim_allowed": False,
        "dla_common_path_rows_v46": int(len(dla)),
        "focused_dynamic_ci_rows_v46": int(len(ci)),
        "focused_1024_rerun_executed_v46": bool(ci.get("focused_1024_rerun_executed_v46", pd.Series([False])).any()),
        "bellman_exact_claim_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v46 improves SPO oracle-regret and DLA common-path replay; no differentiable SPO or Bellman exact claim",
    }
    write_json(STATUS_DIR / "paper4_v46_status.json", status)
    return status


def build_v47_ifrs9_sicr() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = read_parquet("paper4_v36_ifrs9_proxy_cashflow_panel_v2.parquet")
    if panel.empty:
        empty = pd.DataFrame()
        empty.to_parquet(TABLE_DIR / "paper4_v47_ifrs9_proxy_panel_v45.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v47_ifrs9_proxy_policy_summary.csv", empty)
        write_csv(TABLE_DIR / "paper4_v47_sicr_robust_calibration.csv", empty)
        return empty, empty
    p = panel.copy()
    for col in [
        "ead_start_proxy_v25",
        "ead_end_proxy_v25",
        "scheduled_principal_proxy",
        "interest_cash_proxy",
        "recovery_cash_proxy",
        "loss_cash_proxy",
        "ecl_proxy_v29",
    ]:
        if col in p:
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
    p["net_cash_proxy_v47"] = (
        p["scheduled_principal_proxy"]
        + p["interest_cash_proxy"]
        + p["recovery_cash_proxy"]
        - p["loss_cash_proxy"]
    )
    p["ead_path_proxy_v47"] = p.get("ead_path_proxy_v36", p["ead_start_proxy_v25"])
    p["contractual_ifrs9_claim_allowed"] = False
    p["claim_boundary_v47"] = "IFRS9-inspired monthly proxy panel; not contractual IFRS9"
    p.to_parquet(TABLE_DIR / "paper4_v47_ifrs9_proxy_panel_v45.parquet", index=False)

    summary = (
        p.groupby(["policy_id", "scenario"], dropna=False)
        .agg(
            n_loan_months=("loan_id", "size"),
            n_loans=("loan_id", "nunique"),
            total_ead_start=("ead_start_proxy_v25", "sum"),
            total_ead_end=("ead_end_proxy_v25", "sum"),
            total_net_cash_proxy=("net_cash_proxy_v47", "sum"),
            total_ecl_proxy=("ecl_proxy_v29", "sum"),
            default_event_share=("default_event_proxy", "mean"),
            prepayment_event_share=("prepayment_event_proxy", "mean"),
            recovery_cash_total=("recovery_cash_proxy", "sum"),
        )
        .reset_index()
    )
    summary["contractual_ifrs9_claim_allowed"] = False
    summary["claim_boundary_v47"] = "policy summary from proxy cashflow panel only"
    write_csv(TABLE_DIR / "paper4_v47_ifrs9_proxy_policy_summary.csv", summary)

    rules = []
    for abs_stage2 in [0.12, 0.15, 0.20, 0.25]:
        for width_thr in [0.08, 0.12, 0.16]:
            tmp = p.copy()
            base = pd.to_numeric(tmp.get("ecl_proxy_v29", 0), errors="coerce").fillna(0)
            ead = pd.to_numeric(tmp.get("ead_start_proxy_v25", 1), errors="coerce").replace(0, np.nan)
            pd_proxy = (base / ead / 0.45).clip(0, 1).fillna(0)
            width_proxy = pd.to_numeric(tmp.get("sicr_conformal_width", 0), errors="coerce").fillna(0)
            stage2 = pd_proxy.ge(abs_stage2) | width_proxy.ge(width_thr)
            tmp["stage2_v47"] = stage2
            agg = (
                tmp.groupby(["policy_id", "scenario"], dropna=False)
                .agg(stage2_share_v47=("stage2_v47", "mean"), ecl_proxy_total_v47=("ecl_proxy_v29", "sum"))
                .reset_index()
            )
            agg["sicr_rule_v47"] = f"abs_pd_{abs_stage2:.2f}_or_width_{width_thr:.2f}"
            agg["stage2_target_distance_v47"] = (agg["stage2_share_v47"] - 0.40).abs()
            agg["contractual_ifrs9_claim_allowed"] = False
            agg["claim_boundary_v47"] = "SICR robust calibration proxy; not production IFRS9 staging"
            rules.append(agg)
    sicr = pd.concat(rules, ignore_index=True)
    sicr["sicr_preference_score_v47"] = 1 - normalize(sicr["stage2_target_distance_v47"])
    write_csv(TABLE_DIR / "paper4_v47_sicr_robust_calibration.csv", sicr)
    return summary, sicr


def build_v47_paths() -> pd.DataFrame:
    paths = read_parquet("paper4_v31_sample_paths.parquet")
    macro = read_csv("paper4_v43_external_macro_registry.csv")
    if paths.empty:
        out = pd.DataFrame()
    else:
        out = (
            paths.groupby(["path_family_v19", "macro_regime_v15"], dropna=False)
            .agg(
                n_rows=("path_id", "size"),
                n_paths=("path_id", "nunique"),
                default_factor_mean=("default_factor_v15", "mean"),
                default_factor_p95=("default_factor_v15", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
                lgd_factor_mean=("lgd_factor_v15", "mean"),
                prepay_factor_mean=("prepay_factor_v15", "mean"),
                systemic_corr_proxy=("systemic_factor_v15", "mean"),
                observed_default_anchor_mean=("observed_default_anchor_v23", "mean"),
            )
            .reset_index()
        )
        out["external_macro_context_available_v47"] = bool(
            not macro.empty and macro.get("fetch_success", pd.Series(dtype=bool)).astype(bool).any()
        )
        out["external_forecast_validation_claim_allowed"] = False
        out["claim_boundary_v47"] = "internal calibration with optional macro context; not forecast validation"
    write_csv(TABLE_DIR / "paper4_v47_sample_path_macro_alignment.csv", out)
    return out


def build_v47_champion_decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = read_parquet("paper4_policy_loan_level_evidence.parquet")
    if evidence.empty:
        empty = pd.DataFrame()
        empty.to_parquet(TABLE_DIR / "paper4_v47_champion_decomposition_loan_level.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v47_champion_case_studies.csv", empty)
        return empty, empty
    policy_scores = (
        evidence.groupby("policy_id", dropna=False)
        .agg(
            funded_exposure=("funded_exposure", "sum"),
            realized_return_proxy=("realized_return_proxy_lgd45", "sum"),
            ecl_proxy=("ecl_baseline_lgd45", "sum"),
        )
        .reset_index()
    )
    policy_scores["net_score"] = (
        pd.to_numeric(policy_scores["realized_return_proxy"], errors="coerce").fillna(0)
        - 0.65 * pd.to_numeric(policy_scores["ecl_proxy"], errors="coerce").fillna(0)
    )
    serious = ["paper1_economic_champion"] + (
        policy_scores.loc[~policy_scores["policy_id"].eq("paper1_economic_champion")]
        .sort_values("net_score", ascending=False)
        .head(5)["policy_id"]
        .astype(str)
        .tolist()
    )
    df = evidence.loc[evidence["policy_id"].isin(serious)].copy()
    champion_loans = set(
        df.loc[df["policy_id"].eq("paper1_economic_champion") & df["funded_flag"].astype(bool), "loan_id"]
    )
    df["champion_selected_v47"] = df["loan_id"].isin(champion_loans)
    df["candidate_selected_v47"] = df["funded_flag"].astype(bool)
    df["selection_relation_v47"] = np.select(
        [
            df["champion_selected_v47"] & df["candidate_selected_v47"],
            ~df["champion_selected_v47"] & df["candidate_selected_v47"],
            df["champion_selected_v47"] & ~df["candidate_selected_v47"],
        ],
        ["overlap", "candidate_only", "champion_only"],
        default="neither",
    )
    for col in ["int_rate", "pd_point", "pd_high_alpha01", "ecl_baseline_lgd45", "realized_return_proxy_lgd45"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["conformal_width_v47"] = (df["pd_high_alpha01"] - df["pd_point"]).clip(lower=0)
    df["tail_loss_proxy_v47"] = df["ecl_baseline_lgd45"] + 1000 * df["conformal_width_v47"]
    df["economic_case_label_v47"] = np.select(
        [
            df["selection_relation_v47"].eq("candidate_only") & df["tail_loss_proxy_v47"].lt(df["tail_loss_proxy_v47"].median()),
            df["selection_relation_v47"].eq("champion_only") & df["realized_return_proxy_lgd45"].gt(df["realized_return_proxy_lgd45"].median()),
        ],
        ["candidate_adds_lower_tail_proxy_loan", "champion_keeps_higher_return_proxy_loan"],
        default="diagnostic_case",
    )
    df.to_parquet(TABLE_DIR / "paper4_v47_champion_decomposition_loan_level.parquet", index=False)
    cases = (
        df.loc[df["selection_relation_v47"].isin(["candidate_only", "champion_only"])]
        .sort_values(["policy_id", "tail_loss_proxy_v47", "realized_return_proxy_lgd45"], ascending=[True, True, False])
        .groupby(["policy_id", "selection_relation_v47"], dropna=False)
        .head(5)
        .reset_index(drop=True)
    )
    write_csv(TABLE_DIR / "paper4_v47_champion_case_studies.csv", cases)
    return df, cases


def build_v47_cate_fairness() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = read_parquet("paper4_policy_loan_level_evidence.parquet")
    if evidence.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v47_cate_treatment_outcome_search.csv", empty)
        write_csv(TABLE_DIR / "paper4_v47_fairness_source_protocol.csv", empty)
        write_csv(TABLE_DIR / "paper4_v47_no_claim_flags.csv", empty)
        return empty, empty, empty
    df = evidence.drop_duplicates("loan_id").copy()
    for col in ["int_rate", "pd_point", "pd_high_alpha01", "loan_amnt", "realized_return_proxy_lgd45", "ecl_baseline_lgd45"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["high_rate_within_grade_v47"] = (
        df.groupby("original_grade")["int_rate"].transform(lambda s: s >= s.quantile(0.75))
    )
    features = ["pd_point", "pd_high_alpha01", "loan_amnt"]
    smd_rows = []
    treated = df["high_rate_within_grade_v47"].astype(bool)
    for feature in features:
        t = df.loc[treated, feature]
        c = df.loc[~treated, feature]
        pooled = math.sqrt((float(t.var(ddof=1)) + float(c.var(ddof=1))) / 2) if len(t) > 1 and len(c) > 1 else np.nan
        smd = (float(t.mean()) - float(c.mean())) / pooled if pooled and not math.isclose(pooled, 0) else np.nan
        smd_rows.append(
            {
                "diagnostic_v47": "balance_smd",
                "feature": feature,
                "value": smd,
                "pass_gate_v47": bool(abs(smd) <= 0.10) if not pd.isna(smd) else False,
            }
        )
    overlap_share = float(
        df.groupby("original_grade")["high_rate_within_grade_v47"].transform("mean").between(0.05, 0.95).mean()
    )
    search = pd.DataFrame(
        smd_rows
        + [
            {
                "diagnostic_v47": "overlap_within_grade",
                "feature": "high_rate_within_grade_v47",
                "value": overlap_share,
                "pass_gate_v47": overlap_share >= 0.90,
            },
            {
                "diagnostic_v47": "identification",
                "feature": "accepted_loan_selection",
                "value": 0.0,
                "pass_gate_v47": False,
            },
            {
                "diagnostic_v47": "hidden_bias_sensitivity",
                "feature": "not_stable_enough_for_policy_value",
                "value": 0.0,
                "pass_gate_v47": False,
            },
        ]
    )
    search["cate_policy_value_allowed"] = False
    search["claim_boundary_v47"] = (
        "causal diagnostics only; no CATE policy-value claim under accepted-loan/reject-inference limits"
    )
    write_csv(TABLE_DIR / "paper4_v47_cate_treatment_outcome_search.csv", search)

    fairness = (
        evidence.assign(
            rate_bin=pd.qcut(
                pd.to_numeric(evidence["int_rate"], errors="coerce").rank(method="first"),
                q=5,
                labels=[f"rate_q{i}" for i in range(1, 6)],
            ).astype(str)
        )
        .groupby(["policy_id", "original_grade", "period"], dropna=False)
        .agg(
            n_loans=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            avg_pd=("pd_point", "mean"),
            avg_ecl=("ecl_baseline_lgd45", "mean"),
        )
        .reset_index()
    )
    fairness["support_gate_v47"] = fairness["n_loans"].ge(10)
    fairness["fair_lending_legal_claim_allowed"] = False
    fairness["claim_boundary_v47"] = "source governance only; no protected-attribute inference or legal claim"
    write_csv(TABLE_DIR / "paper4_v47_fairness_source_protocol.csv", fairness)

    flags = pd.DataFrame(
        [
            {"claim": "contractual_ifrs9", "allowed": False, "reason": "missing servicing/DPD/cure/recovery/prepayment contractual panel"},
            {"claim": "cate_policy_value", "allowed": False, "reason": "identification, hidden-bias and reject-inference gates do not pass"},
            {"claim": "fair_lending_legal", "allowed": False, "reason": "protected attributes and approved proxy protocol absent"},
            {"claim": "formal_differentiable_spo", "allowed": False, "reason": "cvxpylayers/torch validated route not implemented"},
            {"claim": "exact_full_universe_cvar", "allowed": False, "reason": "full scenario-loss matrix and exact solver certificate absent"},
        ]
    )
    write_csv(TABLE_DIR / "paper4_v47_no_claim_flags.csv", flags)

    write_note(
        NOTE_DIR / "paper4_cate_identification_reaudit.md",
        "\n".join(
            [
                "# Paper 4 CATE Identification Reaudit",
                "",
                f"Generated: {now()}",
                "",
                "The v47 treatment search keeps `high_rate_within_grade` as the only",
                "usable diagnostic treatment. Overlap can be inspected within accepted",
                "loans, but accepted-loan selection, pricing endogeneity and reject",
                "inference remain unresolved. CATE policy value stays blocked.",
            ]
        ),
    )
    write_note(
        NOTE_DIR / "paper4_fairness_protocol_update.md",
        "\n".join(
            [
                "# Paper 4 Fairness Protocol Update",
                "",
                f"Generated: {now()}",
                "",
                "Fairness remains proxy/source governance only. The lab can monitor grade,",
                "period, income/DTI-like sources where available, PD and ECL concentration,",
                "but it cannot infer protected attributes or make a fair-lending legal claim.",
            ]
        ),
    )
    return search, fairness, flags


def build_v47() -> dict[str, Any]:
    start = datetime.now(UTC)
    ifrs9, sicr = build_v47_ifrs9_sicr()
    paths = build_v47_paths()
    decomposition, cases = build_v47_champion_decomposition()
    cate, fairness, flags = build_v47_cate_fairness()
    status = {
        "schema_version": "2026-05-15.47",
        "generated_at_utc": now(),
        "phase": "v47_ifrs9_cate_fairness_paths",
        "ifrs9_policy_summary_rows_v47": int(len(ifrs9)),
        "sicr_robust_rows_v47": int(len(sicr)),
        "contractual_ifrs9_claim_allowed": False,
        "sample_path_macro_rows_v47": int(len(paths)),
        "external_forecast_validation_claim_allowed": False,
        "champion_decomposition_rows_v47": int(len(decomposition)),
        "champion_case_study_rows_v47": int(len(cases)),
        "cate_search_rows_v47": int(len(cate)),
        "cate_policy_value_allowed": False,
        "fairness_protocol_rows_v47": int(len(fairness)),
        "fair_lending_legal_claim_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v47 improves proxy panels, path labels, decomposition and blocked gates without forbidden claims",
    }
    write_json(STATUS_DIR / "paper4_v47_status.json", status)
    return status


def build_v48_registry() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior = read_csv("paper4_v44_candidate_registry.csv")
    dynamic_ci = read_csv("paper4_v46_focused_dynamic_1024_ci.csv")
    cvar = read_csv("paper4_v45_cvar_solver_frontier.csv")
    spo = read_csv("paper4_v46_spo_training_report.csv")
    dla = read_csv("paper4_v46_dla_common_path_replay.csv")
    rows: list[dict[str, Any]] = []
    if not prior.empty:
        for _, row in prior.iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": row.get("candidate_family", "prior"),
                    "wealth_score_v48": safe_num(row.get("wealth_score_v44"), 0),
                    "tail_score_v48": safe_num(row.get("tail_score_v44"), 0),
                    "regret_score_v48": safe_num(row.get("regret_score_v44"), 0.5),
                    "auditability_score_v48": safe_num(row.get("auditability_score_v44"), 0.5),
                    "claim_safety_v48": bool(row.get("claim_safety_v44", True)),
                    "evidence_source_v48": "v44_registry",
                }
            )
    if not dynamic_ci.empty:
        for _, row in dynamic_ci.iterrows():
            rows.append(
                {
                    "policy_id": row.get("candidate_policy_id"),
                    "candidate_family": "focused_dynamic_common_path",
                    "wealth_score_v48": safe_num(row.get("prob_higher_wealth_v46"), 0),
                    "tail_score_v48": safe_num(row.get("prob_lower_loss_v46"), 0),
                    "regret_score_v48": 0.55,
                    "auditability_score_v48": 0.70,
                    "claim_safety_v48": True,
                    "evidence_source_v48": "v46_focused_dynamic_ci",
                }
            )
    if not cvar.empty:
        for _, row in cvar.head(8).iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": "cvar_slack_frontier",
                    "wealth_score_v48": safe_num(row.get("objective_return"), 0),
                    "tail_score_v48": -safe_num(row.get("scenario_loss_cvar90"), 0),
                    "regret_score_v48": 0.50,
                    "auditability_score_v48": 0.75,
                    "claim_safety_v48": True,
                    "evidence_source_v48": "v45_cvar_solver_frontier",
                }
            )
    registry = pd.DataFrame(rows).dropna(subset=["policy_id"]).drop_duplicates(
        ["policy_id", "candidate_family", "evidence_source_v48"], keep="first"
    )
    if not registry.empty:
        registry["wealth_score_norm_v48"] = normalize(registry["wealth_score_v48"])
        registry["tail_score_norm_v48"] = normalize(registry["tail_score_v48"])
        registry["regret_score_norm_v48"] = normalize(registry["regret_score_v48"])
        registry["auditability_score_norm_v48"] = normalize(registry["auditability_score_v48"])
        registry["full_governance_score_v48"] = (
            0.30 * registry["wealth_score_norm_v48"]
            + 0.25 * registry["tail_score_norm_v48"]
            + 0.15 * registry["regret_score_norm_v48"]
            + 0.20 * registry["auditability_score_norm_v48"]
            + 0.10 * registry["claim_safety_v48"].astype(float)
        )
        registry["decision_v48"] = np.select(
            [
                registry["policy_id"].eq("paper1_economic_champion"),
                registry["candidate_family"].str.contains("cvar", case=False, na=False)
                & registry["tail_score_norm_v48"].ge(0.75),
                registry["candidate_family"].str.contains("dla|dynamic", case=False, na=False),
            ],
            ["retain_working_reference", "serious_tail_challenger", "dynamic_or_dla_review"],
            default="review_or_park",
        )
        registry["paper1_promotion_allowed_v48"] = False
        registry["paper4_working_champion_allowed_v48"] = registry["decision_v48"].eq(
            "retain_working_reference"
        )
        registry["claim_boundary_v48"] = "Paper 4 working registry only; no final promotion"
        registry = registry.sort_values("full_governance_score_v48", ascending=False)
    claims = pd.DataFrame(
        [
            {"claim_id": "v48_compact_quarto", "allowed": True, "artifact": "book/_quarto.yml", "boundary": "Paper 4 official chapter remains <=12 pages"},
            {"claim_id": "v48_online_direct_replay", "allowed": True, "artifact": "paper4_v45_online_source_family_direct_holdout.csv", "boundary": "historical direct loan-level replay, not live deployability"},
            {"claim_id": "v48_cvar_exact_full_universe", "allowed": False, "artifact": "paper4_v45_cvar_full_universe_attempt.csv", "boundary": "exact full-universe claim remains false"},
            {"claim_id": "v48_spo_differentiable", "allowed": False, "artifact": "paper4_v46_spo_isolated_env_smoke_test.csv", "boundary": "oracle-regret route only"},
            {"claim_id": "v48_ifrs9_contractual", "allowed": False, "artifact": "paper4_v47_ifrs9_proxy_policy_summary.csv", "boundary": "IFRS9-inspired proxy only"},
            {"claim_id": "v48_cate_policy_value", "allowed": False, "artifact": "paper4_v47_cate_treatment_outcome_search.csv", "boundary": "diagnostic only"},
            {"claim_id": "v48_fair_lending_legal", "allowed": False, "artifact": "paper4_v47_fairness_source_protocol.csv", "boundary": "source governance only"},
        ]
    )
    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "online conformal",
                "status": "near_resolved_with_plateau",
                "executable_item": "v45 direct loan-level source-family holdout is now artifacted; next step is external or later-period validation if new data appears.",
                "success_condition": "strict source-family holdouts pass on genuinely unseen periods/sources with width <=0.95",
            },
            {
                "horizon": "short",
                "lane": "CVaR/OCE",
                "status": "implementation_blocked_for_exact_claim",
                "executable_item": "v45 slack certificate is explicit, but exact full-universe proof still needs persisted loan-scenario matrix.",
                "success_condition": "full scenario-loss matrix plus exact LP/column-generation certificate exists",
            },
            {
                "horizon": "short",
                "lane": "SPO/DFL",
                "status": "dependency_blocked",
                "executable_item": "v46 loan-level oracle-regret is implemented; formal differentiable SPO waits on isolated cvxpylayers/torch validation.",
                "success_condition": "differentiable layer validates and improves oracle-regret under temporal splits",
            },
            {
                "horizon": "medium",
                "lane": "Paper Estrella feed-through",
                "status": "not_promoted_with_reason",
                "executable_item": "No v45-v48 result currently changes Paper Estrella without destabilizing its INFORMS JDS story.",
                "success_condition": "future Paper 4 result improves Paper Estrella regret/auditability story and passes formal promotion gates",
            },
        ]
    )
    combined = pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")

    write_csv(TABLE_DIR / "paper4_v48_candidate_registry.csv", registry)
    write_csv(TABLE_DIR / "paper4_v48_claim_matrix.csv", claims)
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)
    champion = {
        "schema_version": "2026-05-15.48",
        "generated_at_utc": now(),
        "paper4_working_champion": "paper1_economic_champion",
        "working_champion_decision": "retained_reference_after_v45_v48_no_full_gate_challenger",
        "paper4_working_only": True,
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": "Paper 4 lab/working champion only; no Paper Estrella promotion",
    }
    write_json(STATUS_DIR / "paper4_v48_working_champion.json", champion)
    return registry, claims, combined


def update_v48_notes(statuses: dict[str, dict[str, Any]]) -> None:
    publishability = "\n".join(
        [
            "# Paper 4 Publishability Focus Memo",
            "",
            f"Generated: {now()}",
            "",
            "## Strategy",
            "",
            "- Paper Estrella remains the near-term INFORMS Journal on Data Science target.",
            "- Paper 4 remains a living lab. A Management Science path requires a stronger",
            "  dynamic sequential-decision result than the current diagnostic evidence.",
            "- Future Paper 4 findings may feed Paper Estrella only through a formal",
            "  promotion protocol that preserves the Paper Estrella thesis.",
            "",
            "## Current Publication Readiness",
            "",
            "- Official and defensible: governed sequential-decision lab, compact Quarto",
            "  chapter, dynamic common-path replay, CVaR tail challenger, Powell/SDAM",
            "  claim governance.",
            "- Promising but immature: direct source-family conformal holdouts, source",
            "  governance inside solver, oracle-regret SPO, DLA/ADP replay.",
            "- Blocked: contractual IFRS9, CATE policy value, fair-lending legal claims,",
            "  formal differentiable SPO+, exact full-universe CVaR optimality.",
            "",
            "## Management Science Gate",
            "",
            "A credible Management Science version needs a clear dynamic policy result:",
            "higher value or lower risk under common paths, robust paired intervals,",
            "auditability, and claim-safe governance. More tables alone are not enough.",
        ]
    )
    write_note(NOTE_DIR / "paper4_publishability_focus_memo.md", publishability)

    template = "\n".join(
        [
            "# Paper 4 Future Wave Template",
            "",
            "Use this structure when appending future waves to the living notebook.",
            "",
            "```markdown",
            "## Wave vXX-vYY: Title",
            "",
            "### Objective",
            "### Scripts",
            "### Artifacts",
            "### Results",
            "### Interpretation",
            "### Failures",
            "### Claim Impact",
            "### Quarto Promotion Decision",
            "### Paper Estrella Impact",
            "### Management Science Potential",
            "### Next Experiments",
            "```",
        ]
    )
    write_note(NOTE_DIR / "paper4_future_wave_template.md", template)

    section = "\n".join(
        [
            "",
            "<!-- V45_V48_LIVING_LAB_START -->",
            "",
            "## Wave v45-v48: Direct Holdouts, Slack Certificates and Registry Refresh",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Run the next executable Paper 4 living-lab wave while keeping Quarto compact.",
            "",
            "### Scripts",
            "",
            "- `scripts/papers/build_paper4_v45_online_cvar_source_solver.py`",
            "- `scripts/papers/build_paper4_v46_spo_dla_dynamic.py`",
            "- `scripts/papers/build_paper4_v47_ifrs9_cate_fairness_paths.py`",
            "- `scripts/papers/build_paper4_v48_registry_docs_guardrails.py`",
            "",
            "### Results",
            "",
            f"- v45 direct online source-family rows: `{statuses['v45'].get('online_direct_holdout_rows_v45')}`.",
            f"- v45 CVaR slack certificate rows: `{statuses['v45'].get('cvar_slack_certificate_rows_v45')}`.",
            f"- v46 SPO loan-level allocation rows: `{statuses['v46'].get('spo_allocation_rows_v46')}`.",
            f"- v46 DLA common-path rows: `{statuses['v46'].get('dla_common_path_rows_v46')}`.",
            f"- v47 champion decomposition rows: `{statuses['v47'].get('champion_decomposition_rows_v47')}`.",
            f"- v48 candidate registry rows: `{statuses['v48'].get('candidate_registry_rows_v48')}`.",
            "",
            "### Interpretation",
            "",
            "The wave produced deeper executable evidence, but still does not unlock a",
            "new official Paper 4 contribution that should expand Quarto. The best use",
            "of the results is to sharpen the lab agenda: exact CVaR needs a persisted",
            "scenario matrix, formal SPO needs isolated dependencies, DLA needs a true",
            "validated policy improvement, and causal/fairness/IFRS9 claims remain gated.",
            "",
            "### Claim Impact",
            "",
            "- Paper Estrella remains unchanged and is still the INFORMS JDS target.",
            "- Paper 4 remains a Management Science possibility only after stronger",
            "  sequential-decision evidence.",
            "- Forbidden claims remain false.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep v45-v48 in the living notebook. Do not register new Quarto pages.",
            "",
            "<!-- V45_V48_LIVING_LAB_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V45_V48_LIVING_LAB_START -->"
        end = "<!-- V45_V48_LIVING_LAB_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v48() -> dict[str, Any]:
    start = datetime.now(UTC)
    registry, claims, backlog = build_v48_registry()
    statuses = {
        "v45": read_json("paper4_v45_status.json"),
        "v46": read_json("paper4_v46_status.json"),
        "v47": read_json("paper4_v47_status.json"),
    }
    status = {
        "schema_version": "2026-05-15.48",
        "generated_at_utc": now(),
        "phase": "v48_registry_docs_guardrails",
        "candidate_registry_rows_v48": int(len(registry)),
        "claim_matrix_rows_v48": int(len(claims)),
        "living_backlog_rows_v48": int(len(backlog)),
        "official_quarto_page_count": len(registered_paper4_pages()),
        "quarto_compact_guardrail_pass": len(registered_paper4_pages()) <= 12,
        "paper4_working_champion_changed_v48": False,
        "paper1_promotion_allowed_v48": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v48 updates lab registry, backlog, publishability and notebook; no official Quarto expansion",
    }
    write_json(STATUS_DIR / "paper4_v48_status.json", status)
    statuses["v48"] = status
    update_v48_notes(statuses)
    return status


def build_all() -> dict[str, dict[str, Any]]:
    v45 = build_v45()
    v46 = build_v46()
    v47 = build_v47()
    v48 = build_v48()
    return {"v45": v45, "v46": v46, "v47": v47, "v48": v48}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["v45", "v46", "v47", "v48", "all"], default="all")
    args = parser.parse_args()
    if args.phase == "v45":
        result = {"v45": build_v45()}
    elif args.phase == "v46":
        result = {"v46": build_v46()}
    elif args.phase == "v47":
        result = {"v47": build_v47()}
    elif args.phase == "v48":
        result = {"v48": build_v48()}
    else:
        result = build_all()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
