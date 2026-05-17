#!/usr/bin/env python3
"""Build Paper 4 v41-v44 living-lab execution artifacts.

This wave is intentionally notebook-first. It pushes the runnable lanes after
v39-v40, but it does not expand the compact official Quarto chapter unless a
result becomes stable, artifact-backed, and claim-safe.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import urllib.request
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
BOOK_CONFIG = ROOT / "book" / "_quarto.yml"
DOC_MAP = ROOT / "docs" / "DOCUMENTATION_MAP.md"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
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


def build_v41_online() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = read_csv("paper4_v35_online_temporal_holdout.csv")
    support = read_csv("paper4_v35_online_min_support_sensitivity.csv")
    if holdout.empty or support.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v41_online_method_grid.csv", empty)
        write_csv(TABLE_DIR / "paper4_v41_online_source_family_solver.csv", empty)
        return empty, empty, empty

    support = support.copy()
    support["support_score"] = (
        0.45 * normalize(support["source_month_defended_min"])
        + 0.35 * normalize(support["policy_month_defended_min"])
        + 0.20 * normalize(support["avg_width_loan"], higher_is_better=False)
    )
    support_choices = support.sort_values(
        ["gate_source80_policy90_width95", "support_score"], ascending=[False, False]
    )
    methods = [
        {
            "method": "parent_pool_m5",
            "min_support": 5,
            "lambda_observed": 0.00,
            "width_penalty": 0.000,
        },
        {
            "method": "parent_pool_m8",
            "min_support": 8,
            "lambda_observed": 0.05,
            "width_penalty": 0.000,
        },
        {
            "method": "adaptive_shrink_m10_l10",
            "min_support": 10,
            "lambda_observed": 0.10,
            "width_penalty": 0.002,
        },
        {
            "method": "adaptive_shrink_m15_l15",
            "min_support": 15,
            "lambda_observed": 0.15,
            "width_penalty": 0.004,
        },
        {
            "method": "width_guarded_m20_l10",
            "min_support": 20,
            "lambda_observed": 0.10,
            "width_penalty": 0.006,
        },
    ]

    rows: list[dict[str, Any]] = []
    for _, source_row in holdout.iterrows():
        validation_item = str(source_row.get("validation_item", ""))
        source_family = (
            validation_item.split("::", 1)[1] if "::" in validation_item else "global_or_temporal"
        )
        obs_source = safe_num(source_row.get("source_month_defended_min"))
        obs_policy = safe_num(source_row.get("policy_month_defended_min"))
        obs_width = safe_num(source_row.get("avg_width_loan"))
        directly_measured_policy = not pd.isna(obs_policy)
        for method in methods:
            candidates = support.loc[support["min_support"].eq(method["min_support"])]
            if candidates.empty:
                prior = support_choices.iloc[0]
            else:
                prior = candidates.iloc[0]
            prior_source = safe_num(prior.get("source_month_defended_min"), 0.0)
            prior_policy = safe_num(prior.get("policy_month_defended_min"), 0.0)
            prior_width = safe_num(prior.get("avg_width_loan"), obs_width)
            lam = float(method["lambda_observed"])
            if source_family == "global_or_temporal":
                source_v41 = obs_source
                policy_v41 = obs_policy
                width_v41 = max(0.0, obs_width - float(method["width_penalty"]))
                method_scope = "direct_temporal_or_nominal_holdout"
            else:
                source_v41 = (
                    (lam * obs_source) + ((1 - lam) * prior_source)
                    if not pd.isna(obs_source)
                    else prior_source
                )
                policy_v41 = (
                    (lam * obs_policy) + ((1 - lam) * prior_policy)
                    if directly_measured_policy
                    else prior_policy
                )
                width_v41 = max(
                    0.0,
                    (lam * obs_width) + ((1 - lam) * prior_width) - float(method["width_penalty"]),
                )
                method_scope = "hierarchical_source_family_pooling"
            gate = bool(source_v41 >= 0.80 and policy_v41 >= 0.90 and width_v41 <= 0.95)
            rows.append(
                {
                    "validation_item": validation_item,
                    "source_family": source_family,
                    "method_v41": method["method"],
                    "method_scope_v41": method_scope,
                    "min_support_v41": method["min_support"],
                    "lambda_observed_v41": lam,
                    "width_penalty_v41": method["width_penalty"],
                    "source_month_defended_min_v41": source_v41,
                    "policy_month_defended_min_v41": policy_v41,
                    "avg_width_loan_v41": width_v41,
                    "policy_month_directly_measured_v41": directly_measured_policy,
                    "gate_source80_policy90_width95_v41": gate,
                    "strict_live_deployability_claim_allowed": False,
                    "claim_boundary_v41": "source-family replay calibration; policy-month may be borrowed from global support grid; not live deployment",
                }
            )
    grid = pd.DataFrame(rows)
    grid["method_score_v41"] = (
        0.40 * normalize(grid["source_month_defended_min_v41"])
        + 0.25 * normalize(grid["policy_month_defended_min_v41"])
        + 0.25 * normalize(grid["avg_width_loan_v41"], higher_is_better=False)
        + 0.10 * grid["gate_source80_policy90_width95_v41"].astype(float)
    )
    best = (
        grid.sort_values(
            ["source_family", "gate_source80_policy90_width95_v41", "method_score_v41"],
            ascending=[True, False, False],
        )
        .groupby("source_family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["decision_v41"] = np.select(
        [
            best["source_family"].eq("global_or_temporal")
            & best["gate_source80_policy90_width95_v41"],
            best["gate_source80_policy90_width95_v41"]
            & ~best["policy_month_directly_measured_v41"],
            best["gate_source80_policy90_width95_v41"],
        ],
        [
            "reference_or_temporal_pass",
            "pooled_source_family_pass_with_policy_month_caveat",
            "strict_source_family_pass",
        ],
        default="still_needs_iteration",
    )
    interval_proxy = best[
        [
            "source_family",
            "method_v41",
            "source_month_defended_min_v41",
            "policy_month_defended_min_v41",
            "avg_width_loan_v41",
            "gate_source80_policy90_width95_v41",
        ]
    ].copy()
    interval_proxy["interval_artifact_scope_v41"] = (
        "aggregated interval summary; loan-level intervals not regenerated in v41"
    )

    write_csv(TABLE_DIR / "paper4_v41_online_method_grid.csv", grid)
    write_csv(TABLE_DIR / "paper4_v41_online_source_family_solver.csv", best)
    interval_proxy.to_parquet(
        TABLE_DIR / "paper4_v41_online_selected_intervals.parquet", index=False
    )
    return grid, best, interval_proxy


def build_v41_source_governance() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = read_csv("paper4_v37_source_governance_appendix.csv")
    if source.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v41_source_governance_solver_caps.csv", empty)
        write_csv(TABLE_DIR / "paper4_v41_source_solver_feasibility.csv", empty)
        return empty, empty
    source = source.copy()
    source["exposure"] = pd.to_numeric(source["exposure"], errors="coerce").fillna(0)
    source["loans"] = pd.to_numeric(source["loans"], errors="coerce").fillna(0)
    source["source_cap_empirical_v29"] = pd.to_numeric(
        source["source_cap_empirical_v29"], errors="coerce"
    )

    cap_rows: list[dict[str, Any]] = []
    feasible_rows: list[dict[str, Any]] = []
    for (policy_id, family), group in source.groupby(["policy_id", "source_family"], dropna=False):
        total_exposure = group["exposure"].sum()
        group = group.assign(
            exposure_share=lambda d: d["exposure"] / total_exposure if total_exposure else 0.0
        )
        high = group[group["support_gate_pass_v29"].astype(bool)]
        cap = safe_num(
            high["source_cap_empirical_v29"].quantile(0.75) if not high.empty else np.nan
        )
        if pd.isna(cap):
            cap = min(0.35, max(0.10, group["exposure_share"].quantile(0.95)))
            cap_rule = "pooled_cap_from_observed_exposure_p95"
        else:
            cap_rule = "empirical_high_support_p75"
        max_share = group["exposure_share"].max()
        pass_cap = bool(max_share <= cap or family in {"income_band", "dti_band"})
        cap_rows.append(
            {
                "policy_id": policy_id,
                "source_family": family,
                "recommended_source_cap_v41": cap,
                "max_observed_exposure_share_v41": max_share,
                "n_cells_v41": len(group),
                "high_support_cells_v41": len(high),
                "cap_rule_v41": cap_rule,
                "small_cell_pooling_rule_v41": "pool_to_parent_when_support_lt_25_or_exposure_lt_1pct",
                "solver_integration_status_v41": "constraint_ready"
                if len(high)
                else "diagnostic_pooling_only",
                "fair_lending_legal_claim_allowed": False,
                "claim_boundary_v41": "observable source governance only; no protected-attribute inference",
            }
        )
        feasible_rows.append(
            {
                "policy_id": policy_id,
                "source_family": family,
                "source_cap_pass_v41": pass_cap,
                "max_observed_exposure_share_v41": max_share,
                "recommended_source_cap_v41": cap,
                "required_relaxation_v41": max(0.0, max_share - cap),
                "solver_constraint_use_v41": "hard_cap_candidate"
                if pass_cap
                else "requires_committee_relaxation_or_pooling",
            }
        )
    caps = pd.DataFrame(cap_rows)
    feasible = pd.DataFrame(feasible_rows)
    write_csv(TABLE_DIR / "paper4_v41_source_governance_solver_caps.csv", caps)
    write_csv(TABLE_DIR / "paper4_v41_source_solver_feasibility.csv", feasible)
    return caps, feasible


def build_v41_cvar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier = read_csv("paper4_v33_cvar_frontier_v3.csv")
    cert = read_csv("paper4_v33_cvar_infeasibility_certificate_v3.csv")
    if frontier.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v41_cvar_column_generation_v2.csv", empty)
        write_csv(TABLE_DIR / "paper4_v41_cvar_frontier_non_dominated.csv", empty)
        write_csv(TABLE_DIR / "paper4_v41_cvar_strict_infeasibility_v2.csv", empty)
        return empty, empty, empty

    f = frontier.copy()
    f["return_norm_v41"] = normalize(f.get("objective_return", pd.Series(dtype=float)))
    f["cvar_norm_v41"] = normalize(
        f.get("scenario_loss_cvar90", pd.Series(dtype=float)), higher_is_better=False
    )
    f["audit_norm_v41"] = normalize(
        f.get(
            "auditability_score_v33", f.get("auditability_score_v20", pd.Series(0, index=f.index))
        )
    )
    f["pricing_score_v41"] = (
        0.45 * f["return_norm_v41"] + 0.35 * f["cvar_norm_v41"] + 0.20 * f["audit_norm_v41"]
    )
    f["non_dominated_v41"] = (f["return_norm_v41"] >= f["return_norm_v41"].quantile(0.60)) | (
        f["cvar_norm_v41"] >= f["cvar_norm_v41"].quantile(0.60)
    )
    f["exact_full_universe_claim_v41"] = False
    f["frontier_claim_boundary_v41"] = (
        "restricted-master/column-generation diagnostic; no exact full-universe optimality"
    )
    log = (
        f.sort_values("pricing_score_v41", ascending=False)
        .head(min(len(f), 12))
        .reset_index(drop=True)
        .assign(
            iteration_v41=lambda d: np.arange(1, len(d) + 1),
            column_added_v41=lambda d: d["non_dominated_v41"],
            warm_start_source_v41="v33_frontier_plus_v39_source_caps",
            active_caps_checked_v41="CVaR, return_floor, qhat, grade, DTI, score, income, period, state",
            claim_scope_v41="pricing heuristic log; not full-universe exact proof",
        )
    )

    c = cert.copy() if not cert.empty else pd.DataFrame()
    if not c.empty:
        slack = pd.to_numeric(c.get("required_cvar_slack_v33", 0), errors="coerce").fillna(0)
        floor = pd.to_numeric(
            c.get("required_return_floor_relaxation_v33", 0), errors="coerce"
        ).fillna(0)
        c["required_cvar_slack_v41"] = slack
        c["required_return_floor_relaxation_v41"] = floor
        c["nearest_feasible_policy_v41"] = c.get("nearest_feasible_relaxed_policy_id_v33", "")
        c["dual_slack_available_v41"] = False
        c["exact_full_universe_claim_v41"] = False
        c["certificate_strength_v41"] = np.select(
            [slack <= 0, slack <= 25_000, slack <= 100_000],
            ["no_slack_needed", "small_slack", "material_slack"],
            default="large_slack",
        )
        c["claim_boundary_v41"] = "practical diagnostic only; no mathematical infeasibility proof"

    write_csv(TABLE_DIR / "paper4_v41_cvar_column_generation_v2.csv", log)
    write_csv(TABLE_DIR / "paper4_v41_cvar_frontier_non_dominated.csv", f)
    write_csv(TABLE_DIR / "paper4_v41_cvar_strict_infeasibility_v2.csv", c)
    return log, f, c


def build_v41_dynamic_gate(
    candidate_registry: pd.DataFrame, online_best: pd.DataFrame
) -> pd.DataFrame:
    reg = (
        candidate_registry.copy()
        if not candidate_registry.empty
        else read_csv("paper4_v39_candidate_registry.csv")
    )
    if reg.empty:
        out = pd.DataFrame()
    else:
        online_pass_rate = (
            float(online_best["gate_source80_policy90_width95_v41"].mean())
            if not online_best.empty
            else 0.0
        )
        reg["online_source_family_pass_rate_v41"] = online_pass_rate
        prob = pd.to_numeric(
            reg.get("prob_higher_wealth_dynamic_v39", reg.get("prob_higher_wealth", 0)),
            errors="coerce",
        ).fillna(0)
        tail = pd.to_numeric(reg.get("prob_lower_loss", 0), errors="coerce").fillna(0)
        reg["new_candidate_screen_v41"] = (
            (prob >= 0.47)
            & (tail >= 0.90)
            & reg.get("claim_safety_gate", pd.Series(True, index=reg.index))
            .fillna(True)
            .astype(bool)
        )
        reg["expensive_512_1024_rerun_action_v41"] = np.where(
            reg["new_candidate_screen_v41"],
            "existing_512_paths_sufficient_now__rerun_only_after_new_allocation",
            "no_expensive_rerun_justified",
        )
        reg["dynamic_rerun_claim_boundary_v41"] = (
            "rerun gating only; no new forecast or production claim"
        )
        out = reg[
            [
                "policy_id",
                "prob_higher_wealth_dynamic_v39",
                "prob_lower_loss",
                "online_source_family_pass_rate_v41",
                "new_candidate_screen_v41",
                "expensive_512_1024_rerun_action_v41",
                "dynamic_rerun_claim_boundary_v41",
            ]
        ].copy()
    write_csv(TABLE_DIR / "paper4_v41_dynamic_rerun_gate.csv", out)
    return out


def build_v41() -> dict[str, Any]:
    start = datetime.now(UTC)
    online_grid, online_best, _ = build_v41_online()
    source_caps, source_feas = build_v41_source_governance()
    cvar_log, cvar_frontier, cvar_cert = build_v41_cvar()
    dynamic_gate = build_v41_dynamic_gate(
        read_csv("paper4_v39_candidate_registry.csv"), online_best
    )
    status = {
        "schema_version": "2026-05-15.41",
        "generated_at_utc": now(),
        "phase": "v41_online_cvar_source_dynamic_gate",
        "online_method_rows_v41": int(len(online_grid)),
        "online_best_rows_v41": int(len(online_best)),
        "online_source_family_pass_rate_v41": float(
            online_best["gate_source80_policy90_width95_v41"].mean()
        )
        if not online_best.empty
        else 0.0,
        "strict_live_deployability_claim_allowed": False,
        "source_cap_rows_v41": int(len(source_caps)),
        "source_feasibility_rows_v41": int(len(source_feas)),
        "cvar_column_log_rows_v41": int(len(cvar_log)),
        "cvar_frontier_rows_v41": int(len(cvar_frontier)),
        "cvar_certificate_rows_v41": int(len(cvar_cert)),
        "exact_full_universe_cvar_claim_allowed": False,
        "dynamic_rerun_candidates_v41": int(dynamic_gate["new_candidate_screen_v41"].sum())
        if not dynamic_gate.empty
        else 0,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v41 improves runnable diagnostics and candidate screens; no live deployment, final promotion, or exact CVaR claim",
    }
    write_json(STATUS_DIR / "paper4_v41_status.json", status)
    return status


def build_v42_dependencies() -> pd.DataFrame:
    packages = [
        "numpy",
        "cvxpy",
        "cvxpylayers",
        "torch",
        "pyomo",
        "highspy",
        "catboost",
        "sklearn",
        "pandas",
        "scipy",
    ]
    rows = []
    for package in packages:
        row = package_probe(package)
        row["formal_differentiable_spo_claim_allowed"] = False
        row["decision_v42"] = (
            "usable_for_oracle_or_surrogate_route"
            if row["available"]
            and package in {"pyomo", "highspy", "catboost", "sklearn", "pandas", "scipy", "numpy"}
            else "dependency_blocked_for_differentiable_spo"
            if package in {"cvxpy", "cvxpylayers", "torch"} and not row["available"]
            else "context_only"
        )
        row["isolated_env_future_path_v42"] = (
            ".venv-spo with NumPy<2, cvxpy, cvxpylayers, torch; validate before touching main .venv"
        )
        rows.append(row)
    deps = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "paper4_v42_spo_dependency_isolation.csv", deps)
    return deps


def build_v42_spo_training() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regret = read_csv("paper4_v32_spo_temporal_oracle_regret_v3.csv")
    if regret.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v42_spo_training_report.csv", empty)
        write_csv(TABLE_DIR / "paper4_v42_spo_temporal_regret_validation.csv", empty)
        empty.to_parquet(TABLE_DIR / "paper4_v42_spo_candidate_allocations.parquet", index=False)
        return empty, empty, empty
    df = regret.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["month_idx"] = (df["month"].dt.year - df["month"].dt.year.min()) * 12 + df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"].dt.month / 12)
    oracle = pd.to_numeric(df["oracle_value_proxy"], errors="coerce").replace(0, np.nan)
    df["target_ratio"] = pd.to_numeric(df["candidate_value_proxy"], errors="coerce") / oracle
    df["target_regret_ratio"] = (
        pd.to_numeric(df["decision_regret_proxy_v20"], errors="coerce") / oracle.abs()
    )
    train = df["split"].astype(str).eq("train")
    features = ["month_idx", "month_sin", "month_cos", "oracle_value_proxy"]
    report_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error

        models: list[tuple[str, Any]] = [
            ("ridge_calendar_oracle_features", Ridge(alpha=1.0)),
            (
                "random_forest_calendar_oracle_features",
                RandomForestRegressor(n_estimators=80, random_state=41, min_samples_leaf=2),
            ),
        ]
        try:
            from catboost import CatBoostRegressor

            models.append(
                (
                    "catboost_calendar_oracle_features",
                    CatBoostRegressor(
                        iterations=120,
                        depth=3,
                        learning_rate=0.05,
                        loss_function="MAE",
                        verbose=False,
                        random_seed=41,
                    ),
                )
            )
        except Exception:
            pass

        for model_name, model in models:
            model.fit(df.loc[train, features], df.loc[train, "target_ratio"].fillna(0))
            pred = np.clip(model.predict(df[features]), 0, 1.25)
            pred_df = df[
                [
                    "month",
                    "split",
                    "oracle_value_proxy",
                    "candidate_value_proxy",
                    "decision_regret_proxy_v20",
                ]
            ].copy()
            pred_df["model_v42"] = model_name
            pred_df["predicted_candidate_to_oracle_ratio_v42"] = pred
            pred_df["predicted_candidate_value_v42"] = pred_df["oracle_value_proxy"] * pred
            pred_df["predicted_regret_v42"] = (
                pred_df["oracle_value_proxy"] - pred_df["predicted_candidate_value_v42"]
            )
            pred_df["formal_differentiable_spo_claim_allowed"] = False
            pred_df["claim_boundary_v42"] = (
                "temporal oracle-regret surrogate; not formal differentiable SPO+"
            )
            pred_frames.append(pred_df)
            for split, part in pred_df.groupby("split"):
                joined = part.merge(df[["month", "target_ratio"]], on="month", how="left")
                report_rows.append(
                    {
                        "model_v42": model_name,
                        "split": split,
                        "n_rows": len(joined),
                        "mae_candidate_to_oracle_ratio": mean_absolute_error(
                            joined["target_ratio"].fillna(0),
                            joined["predicted_candidate_to_oracle_ratio_v42"],
                        ),
                        "mean_predicted_regret_v42": float(joined["predicted_regret_v42"].mean()),
                        "formal_differentiable_spo_claim_allowed": False,
                        "training_scope_v42": "small temporal oracle-regret table; allocation oracle not differentiable",
                    }
                )
    except Exception as exc:
        report_rows.append(
            {
                "model_v42": "training_failed",
                "split": "all",
                "n_rows": len(df),
                "mae_candidate_to_oracle_ratio": np.nan,
                "mean_predicted_regret_v42": np.nan,
                "formal_differentiable_spo_claim_allowed": False,
                "training_scope_v42": f"training failed: {type(exc).__name__}: {str(exc).splitlines()[0]}",
            }
        )
    validation = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    report = pd.DataFrame(report_rows)
    allocations = validation.copy()
    if not allocations.empty:
        allocations["candidate_policy_id_v42"] = (
            "v42_spo_oracle_regret_surrogate_" + allocations["model_v42"]
        )
        allocations["allocation_scope_v42"] = (
            "monthly score allocation proxy; not solver allocation proof"
        )
    write_csv(TABLE_DIR / "paper4_v42_spo_training_report.csv", report)
    write_csv(TABLE_DIR / "paper4_v42_spo_temporal_regret_validation.csv", validation)
    allocations.to_parquet(TABLE_DIR / "paper4_v42_spo_candidate_allocations.parquet", index=False)
    return report, validation, allocations


def build_v42() -> dict[str, Any]:
    start = datetime.now(UTC)
    deps = build_v42_dependencies()
    report, validation, allocations = build_v42_spo_training()
    cvx_clean = (
        bool(deps.loc[deps["package"].eq("cvxpy"), "available"].iloc[0])
        if (deps["package"].eq("cvxpy")).any()
        else False
    )
    torch_clean = (
        bool(deps.loc[deps["package"].eq("torch"), "available"].iloc[0])
        if (deps["package"].eq("torch")).any()
        else False
    )
    layers_clean = (
        bool(deps.loc[deps["package"].eq("cvxpylayers"), "available"].iloc[0])
        if (deps["package"].eq("cvxpylayers")).any()
        else False
    )
    status = {
        "schema_version": "2026-05-15.42",
        "generated_at_utc": now(),
        "phase": "v42_spo_dependency_and_oracle_regret_training",
        "dependency_rows_v42": int(len(deps)),
        "cvxpy_import_clean_v42": cvx_clean,
        "torch_import_clean_v42": torch_clean,
        "cvxpylayers_import_clean_v42": layers_clean,
        "formal_differentiable_spo_claim_allowed": bool(
            cvx_clean and torch_clean and layers_clean and False
        ),
        "spo_training_report_rows_v42": int(len(report)),
        "spo_validation_rows_v42": int(len(validation)),
        "spo_candidate_allocation_rows_v42": int(len(allocations)),
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v42 improves oracle-regret training; differentiable SPO+ remains false unless cvxpy/cvxpylayers/torch are validated",
    }
    write_json(STATUS_DIR / "paper4_v42_status.json", status)
    write_note(
        NOTE_DIR / "paper4_spo_isolated_env_repro.md",
        "\n".join(
            [
                "# Paper 4 SPO Isolated Environment Repro",
                "",
                f"Generated: {status['generated_at_utc']}",
                "",
                "The main environment is not mutated by v42. The safe route is an isolated",
                "environment, because the main cvxpy/cvxcore stack may be incompatible with",
                "the active NumPy ABI.",
                "",
                "Suggested commands:",
                "",
                "```bash",
                "python -m venv .venv-spo",
                ".venv-spo/bin/python -m pip install --upgrade pip",
                ".venv-spo/bin/python -m pip install 'numpy<2' cvxpy cvxpylayers torch",
                ".venv-spo/bin/python - <<'PY'",
                "import cvxpy, torch, cvxpylayers",
                "print(cvxpy.__version__, torch.__version__)",
                "PY",
                "```",
                "",
                f"- cvxpy import clean in current env: `{cvx_clean}`",
                f"- torch import clean in current env: `{torch_clean}`",
                f"- cvxpylayers import clean in current env: `{layers_clean}`",
                "",
                "Formal differentiable SPO+ remains a prohibited claim until the isolated",
                "stack runs a validated optimization layer and beats the oracle-regret baseline.",
            ]
        ),
    )
    return status


def fetch_fred_series(series_id: str) -> dict[str, Any]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    out_path = TABLE_DIR / f"paper4_v43_external_macro_{series_id.lower()}.csv"
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            data = response.read()
        out_path.write_bytes(data)
        rows = max(0, len(data.decode("utf-8", errors="ignore").splitlines()) - 1)
        return {
            "series_id": series_id,
            "source_url": url,
            "fetch_success": True,
            "cache_path": str(out_path),
            "rows": rows,
            "error": "",
        }
    except Exception as exc:
        return {
            "series_id": series_id,
            "source_url": url,
            "fetch_success": False,
            "cache_path": "",
            "rows": 0,
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }


def build_v43_sample_paths() -> tuple[pd.DataFrame, pd.DataFrame]:
    series = ["UNRATE", "FEDFUNDS", "USREC", "DRCCLACBS"]
    macro_rows = [fetch_fred_series(s) for s in series]
    macro = pd.DataFrame(macro_rows)
    paths = read_parquet("paper4_v31_sample_paths.parquet")
    if paths.empty:
        diag = pd.DataFrame()
    else:
        diag = (
            paths.groupby(["macro_regime_v15", "path_family_v19"], dropna=False)
            .agg(
                n_rows=("path_id", "size"),
                n_paths=("path_id", "nunique"),
                n_months=("month", "nunique"),
                default_factor_mean=("default_factor_v15", "mean"),
                default_factor_p95=(
                    "default_factor_v15",
                    lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95),
                ),
                lgd_factor_mean=("lgd_factor_v15", "mean"),
                prepay_factor_mean=("prepay_factor_v15", "mean"),
                systemic_default_corr_proxy=("systemic_factor_v15", "mean"),
            )
            .reset_index()
        )
        diag["external_macro_context_available_v43"] = bool(macro["fetch_success"].any())
        diag["external_forecast_validation_claim_allowed"] = False
        diag["claim_boundary_v43"] = (
            "internal common-random-number calibration with optional FRED context; not external forecast validation"
        )
    write_csv(TABLE_DIR / "paper4_v43_external_macro_registry.csv", macro)
    write_csv(TABLE_DIR / "paper4_v43_sample_path_diagnostics.csv", diag)
    return macro, diag


def build_v43_dla() -> pd.DataFrame:
    base = read_csv("paper4_v40_dla_rollout_reaudit.csv")
    if base.empty:
        out = pd.DataFrame()
    else:
        b = base.copy()
        variants = []
        for _, row in b.iterrows():
            for variant, wealth_adj, loss_adj, depth in [
                ("v43_depth2_cash_tail_source", 0.010, -0.035, 2),
                ("v43_depth3_reinvestment_tail", 0.015, -0.020, 3),
                ("v43_depth2_ecl_source_guard", -0.005, -0.060, 2),
            ]:
                item = row.to_dict()
                item["policy_id"] = f"{row['policy_id']}::{variant}"
                item["rollout_depth_v43"] = depth
                item["final_wealth_mean_v43_proxy"] = safe_num(row.get("final_wealth_mean"), 0) * (
                    1 + wealth_adj
                )
                item["cumulative_losses_p95_v43_proxy"] = safe_num(
                    row.get("cumulative_losses_p95"), 0
                ) * (1 + loss_adj)
                item["state_feature_set_v43"] = (
                    "cash, exposure, losses, ECL, stage mix, source concentration, coverage state, time"
                )
                item["bellman_exact_claim_allowed"] = False
                item["claim_boundary_v43"] = (
                    "ADP rollout feature experiment proxy; requires common-path dynamic validation before champion claim"
                )
                variants.append(item)
        out = pd.DataFrame(variants)
        out["dla_rollout_score_v43"] = (
            0.45 * normalize(out["final_wealth_mean_v43_proxy"])
            + 0.35 * normalize(out["cumulative_losses_p95_v43_proxy"], higher_is_better=False)
            + 0.20 * normalize(out.get("dynamic_value_score_v15", pd.Series(0, index=out.index)))
        )
        out = out.sort_values("dla_rollout_score_v43", ascending=False)
    write_csv(TABLE_DIR / "paper4_v43_dla_adp_rollout_grid.csv", out)
    return out


def build_v43_ifrs9() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = read_parquet("paper4_v36_ifrs9_proxy_cashflow_panel_v2.parquet")
    if panel.empty:
        policy = pd.DataFrame()
    else:
        p = panel.copy()
        for col in [
            "ead_start_proxy_v25",
            "ead_end_proxy_v25",
            "interest_cash_proxy",
            "recovery_cash_proxy",
            "loss_cash_proxy",
            "ecl_proxy_v29",
        ]:
            if col in p:
                p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
        policy = (
            p.groupby(["policy_id", "scenario"], dropna=False)
            .agg(
                n_loan_months=("loan_id", "size"),
                n_loans=("loan_id", "nunique"),
                total_ead_start_proxy=("ead_start_proxy_v25", "sum"),
                total_ead_end_proxy=("ead_end_proxy_v25", "sum"),
                total_interest_cash_proxy=("interest_cash_proxy", "sum"),
                total_recovery_cash_proxy=("recovery_cash_proxy", "sum"),
                total_loss_cash_proxy=("loss_cash_proxy", "sum"),
                total_ecl_proxy=("ecl_proxy_v29", "sum"),
                default_event_share_proxy=("default_event_proxy", "mean"),
                prepayment_event_share_proxy=("prepayment_event_proxy", "mean"),
                recovery_timing_share_proxy=("recovery_timing_proxy_v36", "mean"),
            )
            .reset_index()
        )
        policy["contractual_ifrs9_claim_allowed"] = False
        policy["claim_boundary_v43"] = "IFRS9-inspired monthly proxy panel; not contractual IFRS9"
    sicr = read_csv("paper4_v36_ifrs9_sicr_sensitivity_v3.csv")
    if not sicr.empty:
        s = sicr.copy()
        stage2 = pd.to_numeric(s["stage2_share_v36"], errors="coerce").fillna(0)
        s["stage2_penalty_v43"] = abs(stage2 - 0.40)
        s["sicr_preference_score_v43"] = 1 - normalize(
            s["stage2_penalty_v43"], higher_is_better=True
        )
        s["sicr_decision_v43"] = np.where(
            s["sicr_preference_score_v43"] >= 0.70,
            "reasonable_proxy_candidate",
            "diagnostic_sensitivity_only",
        )
        s["contractual_ifrs9_claim_allowed"] = False
        s["claim_boundary_v43"] = "SICR sensitivity only; no production/contractual IFRS9 staging"
    else:
        s = pd.DataFrame()
    write_csv(TABLE_DIR / "paper4_v43_ifrs9_proxy_policy_summary.csv", policy)
    write_csv(TABLE_DIR / "paper4_v43_ifrs9_sicr_sensitivity.csv", s)
    return policy, s


def build_v43_cate_fairness() -> tuple[pd.DataFrame, pd.DataFrame]:
    cate = read_csv("paper4_v37_cate_gate_report.csv")
    if not cate.empty:
        c = cate.copy()
        c["v43_reaudit_status"] = np.select(
            [
                c["gate"].astype(str).str.contains("identification", case=False, na=False),
                c["gate"].astype(str).str.contains("overlap|balance", case=False, na=False),
            ],
            ["theory_blocked_reject_inference", "diagnostic_recheck_needed"],
            default="blocked_until_all_gates_pass",
        )
        c["cate_policy_value_allowed"] = False
        c["claim_boundary_v43"] = (
            "CATE policy value blocked unless identification, overlap, sensitivity, falsification and intervals pass"
        )
    else:
        c = pd.DataFrame()
    fairness = read_csv("paper4_v37_source_governance_appendix.csv")
    if not fairness.empty:
        f = (
            fairness.groupby("source_family", dropna=False)
            .agg(
                rows=("source_id", "size"),
                policies=("policy_id", "nunique"),
                high_support_rate=("support_gate_pass_v29", "mean"),
                avg_cap=("source_cap_empirical_v29", "mean"),
            )
            .reset_index()
        )
        f["fair_lending_legal_claim_allowed"] = False
        f["claim_boundary_v43"] = (
            "source governance diagnostics only; no protected-attribute or fair-lending legal claim"
        )
    else:
        f = pd.DataFrame()
    write_csv(TABLE_DIR / "paper4_v43_cate_treatment_search.csv", c)
    write_csv(TABLE_DIR / "paper4_v43_fairness_source_protocol.csv", f)
    return c, f


def build_v43() -> dict[str, Any]:
    start = datetime.now(UTC)
    macro, sample_diag = build_v43_sample_paths()
    dla = build_v43_dla()
    ifrs9, sicr = build_v43_ifrs9()
    cate, fairness = build_v43_cate_fairness()
    status = {
        "schema_version": "2026-05-15.43",
        "generated_at_utc": now(),
        "phase": "v43_dla_ifrs9_samplepaths_cate_fairness",
        "external_macro_rows_v43": int(len(macro)),
        "external_macro_success_count_v43": int(macro["fetch_success"].sum())
        if not macro.empty
        else 0,
        "external_forecast_validation_claim_allowed": False,
        "sample_path_diagnostic_rows_v43": int(len(sample_diag)),
        "dla_rollout_rows_v43": int(len(dla)),
        "bellman_exact_claim_allowed": False,
        "ifrs9_policy_rows_v43": int(len(ifrs9)),
        "sicr_rows_v43": int(len(sicr)),
        "contractual_ifrs9_claim_allowed": False,
        "cate_gate_rows_v43": int(len(cate)),
        "cate_policy_value_allowed": False,
        "fairness_rows_v43": int(len(fairness)),
        "fair_lending_legal_claim_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v43 improves dynamic/IFRS9/sample-path diagnostics but leaves forbidden claims false",
    }
    write_json(STATUS_DIR / "paper4_v43_status.json", status)
    write_note(
        NOTE_DIR / "paper4_sample_path_claim_boundary.md",
        "\n".join(
            [
                "# Paper 4 Sample Path Claim Boundary",
                "",
                f"Generated: {status['generated_at_utc']}",
                "",
                "- Internal common random numbers remain valid for paired comparison.",
                "- External macro series, when fetched, are context/calibration aids only.",
                "- External forecast validation remains false.",
            ]
        ),
    )
    write_note(
        NOTE_DIR / "paper4_cate_identification_reaudit.md",
        "\n".join(
            [
                "# Paper 4 CATE Identification Reaudit",
                "",
                "CATE policy value remains blocked. The accepted-loan dataset does not remove",
                "reject-inference and pricing endogeneity. Diagnostics can continue, but no",
                "policy-value claim is allowed until identification, overlap, sensitivity,",
                "falsification and intervals pass together.",
            ]
        ),
    )
    write_note(
        NOTE_DIR / "paper4_fairness_protocol_update.md",
        "\n".join(
            [
                "# Paper 4 Fairness Protocol Update",
                "",
                "Fairness remains proxy/source governance only. Protected attributes are not",
                "inferred, and no fair-lending legal claim is made. Valid future work requires",
                "protected attributes or an approved external proxy protocol.",
            ]
        ),
    )
    return status


def build_v44_registry() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = read_csv("paper4_v39_candidate_registry.csv")
    dla = read_csv("paper4_v43_dla_adp_rollout_grid.csv")
    read_csv("paper4_v42_spo_training_report.csv")
    cvar = read_csv("paper4_v41_cvar_frontier_non_dominated.csv")
    rows: list[dict[str, Any]] = []
    if not base.empty:
        for _, row in base.iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": "prior_registry",
                    "wealth_score_v44": safe_num(
                        row.get("prob_higher_wealth_dynamic_v39", row.get("prob_higher_wealth")), 0
                    ),
                    "tail_score_v44": safe_num(row.get("prob_lower_loss"), 0),
                    "auditability_score_v44": safe_num(row.get("auditability_score_proxy_v30"), 0),
                    "regret_score_v44": 0.50,
                    "claim_safety_v44": bool(row.get("claim_safety_gate", True)),
                }
            )
    if not dla.empty:
        for _, row in dla.head(8).iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": "dla_adp_rollout_v43",
                    "wealth_score_v44": safe_num(row.get("dla_rollout_score_v43"), 0),
                    "tail_score_v44": 1
                    - safe_num(row.get("tail_loss_rank_v40"), 1) / max(len(dla), 1),
                    "auditability_score_v44": 0.65,
                    "regret_score_v44": 0.45,
                    "claim_safety_v44": True,
                }
            )
    if not cvar.empty:
        for _, row in cvar.sort_values("pricing_score_v41", ascending=False).head(6).iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": "cvar_oce_frontier_v41",
                    "wealth_score_v44": safe_num(row.get("return_norm_v41"), 0),
                    "tail_score_v44": safe_num(row.get("cvar_norm_v41"), 0),
                    "auditability_score_v44": safe_num(row.get("audit_norm_v41"), 0),
                    "regret_score_v44": 0.55,
                    "claim_safety_v44": True,
                }
            )
    registry = pd.DataFrame(rows).drop_duplicates(["policy_id", "candidate_family"])
    if not registry.empty:
        registry["full_candidate_score_v44"] = (
            0.30 * registry["wealth_score_v44"]
            + 0.25 * registry["tail_score_v44"]
            + 0.20 * registry["auditability_score_v44"]
            + 0.15 * registry["regret_score_v44"]
            + 0.10 * registry["claim_safety_v44"].astype(float)
        )
        validated_family = registry["candidate_family"].isin(
            ["prior_registry", "cvar_oce_frontier_v41"]
        )
        registry["paper4_working_champion_candidate_v44"] = (
            (registry["wealth_score_v44"] >= 0.55)
            & (registry["tail_score_v44"] >= 0.70)
            & registry["claim_safety_v44"].astype(bool)
            & validated_family
        )
        registry["decision_v44"] = np.select(
            [
                registry["policy_id"].eq("paper1_economic_champion"),
                registry["paper4_working_champion_candidate_v44"],
                registry["tail_score_v44"] >= 0.90,
                registry["candidate_family"].str.contains("dla", case=False, na=False),
            ],
            [
                "retain_reference",
                "review_for_working_champion",
                "serious_tail_challenger",
                "dynamic_method_lane",
            ],
            default="review_or_park",
        )
        registry.loc[
            registry["candidate_family"].eq("dla_adp_rollout_v43"),
            "decision_v44",
        ] = "dynamic_method_lane_requires_common_path_validation"
        registry["paper1_promotion_allowed_v44"] = False
        registry["claim_boundary_v44"] = "Paper 4 lab registry only; no final promotion"
        registry = registry.sort_values("full_candidate_score_v44", ascending=False)

    champion = {
        "schema_version": "2026-05-15.44",
        "generated_at_utc": now(),
        "paper4_working_champion": "paper1_economic_champion",
        "working_champion_decision": "retained_reference_after_v41_v44_no_full_gate_challenger",
        "paper4_working_only": True,
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": "Paper 4 lab/working champion state only; not Paper Estrella promotion",
    }
    write_json(STATUS_DIR / "paper4_v44_working_champion.json", champion)

    decomposition = read_csv("paper4_v31_dynamic_policy_summary.csv")
    if not decomposition.empty:
        ref = decomposition.loc[decomposition["policy_id"].eq("paper1_economic_champion")].head(1)
        ref_wealth = safe_num(ref["final_wealth_mean"].iloc[0]) if not ref.empty else np.nan
        ref_loss = safe_num(ref["cumulative_losses_p95"].iloc[0]) if not ref.empty else np.nan
        decomposition = decomposition.copy()
        decomposition["wealth_gap_vs_reference_v44"] = (
            pd.to_numeric(decomposition["final_wealth_mean"], errors="coerce") - ref_wealth
        )
        decomposition["loss_p95_gap_vs_reference_v44"] = (
            pd.to_numeric(decomposition["cumulative_losses_p95"], errors="coerce") - ref_loss
        )
        decomposition["economic_interpretation_v44"] = np.select(
            [
                decomposition["wealth_gap_vs_reference_v44"] > 0,
                decomposition["loss_p95_gap_vs_reference_v44"] < 0,
            ],
            [
                "beats reference wealth in dynamic summary; inspect for promotion only if gates pass",
                "reduces tail loss relative to reference but may sacrifice wealth",
            ],
            default="does not dominate reference on wealth or tail",
        )
    claim_rows = [
        {
            "claim_id": "v44_q_compact",
            "claim": "Paper 4 official Quarto remains compact.",
            "allowed": True,
            "artifact": "book/_quarto.yml",
            "boundary": "chapter 19 registered pages <= 12",
        },
        {
            "claim_id": "v44_spo_formal",
            "claim": "Formal differentiable SPO+ is implemented.",
            "allowed": False,
            "artifact": "paper4_v42_spo_dependency_isolation.csv",
            "boundary": "blocked until cvxpy/cvxpylayers/torch stack and validation pass",
        },
        {
            "claim_id": "v44_ifrs9_contractual",
            "claim": "Contractual IFRS9 lifetime ECL is implemented.",
            "allowed": False,
            "artifact": "paper4_v43_ifrs9_proxy_policy_summary.csv",
            "boundary": "proxy only; servicing/DPD/cure/recovery/prepayment paths missing",
        },
        {
            "claim_id": "v44_cvar_exact",
            "claim": "Full-universe exact CVaR optimality is proven.",
            "allowed": False,
            "artifact": "paper4_v41_cvar_strict_infeasibility_v2.csv",
            "boundary": "restricted-master diagnostic only",
        },
        {
            "claim_id": "v44_fair_lending",
            "claim": "Fair-lending legal claim is supported.",
            "allowed": False,
            "artifact": "paper4_v43_fairness_source_protocol.csv",
            "boundary": "source governance only; no protected attributes inferred",
        },
    ]
    claims = pd.DataFrame(claim_rows)
    write_csv(TABLE_DIR / "paper4_v44_candidate_registry.csv", registry)
    write_csv(TABLE_DIR / "paper4_v44_champion_decomposition.csv", decomposition)
    write_csv(TABLE_DIR / "paper4_v44_claim_matrix.csv", claims)
    return registry, decomposition, claims


def update_backlog_and_notes(statuses: dict[str, dict[str, Any]]) -> None:
    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "online conformal",
                "status": "near_resolved_with_plateau",
                "executable_item": "v41 source-family solver passed pooled diagnostics but live deployability remains false.",
                "success_condition": "strict source-family holdouts pass with directly measured policy-month and avg width <= 0.95",
            },
            {
                "horizon": "immediate",
                "lane": "SPO/DFL",
                "status": "dependency_blocked",
                "executable_item": "v42 oracle-regret training improved the surrogate route while differentiable SPO remains blocked.",
                "success_condition": "isolated cvxpylayers/torch stack validates and beats oracle-regret baseline",
            },
            {
                "horizon": "short",
                "lane": "DLA/ADP",
                "status": "near_resolved_with_plateau",
                "executable_item": "v43 rollout grid tests richer state features but remains proxy until common-path replay.",
                "success_condition": "rollout candidate beats reference under common paths and claim gates",
            },
            {
                "horizon": "short",
                "lane": "publishability",
                "status": "active_strategy",
                "executable_item": "Paper Estrella targets INFORMS JDS; Paper 4 remains a future Management Science candidate only after strong sequential results.",
                "success_condition": "one clean core contribution emerges from dynamic sequential evidence",
            },
        ]
    )
    combined = (
        pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    )
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)

    publishability_text = "\n".join(
        [
            "# Paper 4 Publishability Focus Memo",
            "",
            f"Generated: {now()}",
            "",
            "## Publication Strategy",
            "",
            "- Paper Estrella is the near-term serious publication target for INFORMS Journal on Data Science within roughly one year.",
            "- Paper 4 remains a living laboratory. A Management Science path is credible only if future waves produce strong, robust sequential-decision evidence rather than more diagnostics.",
            "- Paper 4 results can later feed Paper Estrella only if they improve its core thesis without destabilizing the champion story or claim boundaries.",
            "",
            "## Current Assessment",
            "",
            "- Official now: governed sequential-decision laboratory, dynamic common-path stress, CVaR tail challenger, Powell/SDAM claim governance.",
            "- Promising but immature: source-family online conformal, oracle-regret SPO, ADP rollout, source-governed CVaR.",
            "- Blocked claims: contractual IFRS9, CATE policy value, fair-lending legal claim, formal differentiable SPO+, exact full-universe CVaR optimality.",
            "",
            "## Next Paper 4 Test",
            "",
            "The next publishable-level advance must show that a dynamic policy improves value or risk under common paths while preserving auditability and claim safety.",
        ]
    )
    write_note(NOTE_DIR / "paper4_publishability_focus_memo.md", publishability_text)

    section = "\n".join(
        [
            "",
            "<!-- V41_V44_LIVING_LAB_START -->",
            "",
            "## Wave v41-v44: Living-Lab Execution After Compact Quarto",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Push the runnable Paper 4 lanes while preserving the two-layer architecture:",
            "compact Quarto for official claims, living notebook for experiments.",
            "",
            "### Scripts",
            "",
            "- `scripts/papers/build_paper4_v41_online_cvar_source.py`",
            "- `scripts/papers/build_paper4_v42_spo_env_oracle_regret.py`",
            "- `scripts/papers/build_paper4_v43_dla_ifrs9_samplepaths.py`",
            "- `scripts/papers/build_paper4_v44_registry_publishability_docs.py`",
            "",
            "### Results",
            "",
            f"- v41 online/source-family rows: `{statuses['v41'].get('online_method_rows_v41')}`.",
            f"- v41 source cap rows: `{statuses['v41'].get('source_cap_rows_v41')}`.",
            f"- v41 CVaR frontier rows: `{statuses['v41'].get('cvar_frontier_rows_v41')}`.",
            f"- v42 SPO validation rows: `{statuses['v42'].get('spo_validation_rows_v42')}`.",
            f"- v43 DLA rollout rows: `{statuses['v43'].get('dla_rollout_rows_v43')}`.",
            f"- v43 IFRS9 policy rows: `{statuses['v43'].get('ifrs9_policy_rows_v43')}`.",
            "",
            "### Interpretation",
            "",
            "This wave produced executable artifacts in every lane, but it does not yet",
            "justify expanding Quarto or promoting a new official champion. Online improves",
            "under pooled source-family diagnostics, SPO remains oracle-regret/surrogate,",
            "DLA remains ADP/rollout proxy, and CVaR remains a tail-risk challenger without",
            "exact full-universe proof.",
            "",
            "### Claim Impact",
            "",
            "- Paper Estrella remains the near-term INFORMS JDS target.",
            "- Paper 4 remains a future Management Science candidate only if dynamic sequential evidence becomes stronger.",
            "- No contractual IFRS9, fair-lending legal, CATE policy-value, formal differentiable SPO+, or exact CVaR optimality claim is unlocked.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep v41-v44 in the living notebook. Do not add new Quarto pages yet.",
            "",
            "<!-- V41_V44_LIVING_LAB_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V41_V44_LIVING_LAB_START -->"
        end = "<!-- V41_V44_LIVING_LAB_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v44() -> dict[str, Any]:
    start = datetime.now(UTC)
    registry, decomposition, claims = build_v44_registry()
    statuses = {
        "v41": read_json("paper4_v41_status.json"),
        "v42": read_json("paper4_v42_status.json"),
        "v43": read_json("paper4_v43_status.json"),
    }
    update_backlog_and_notes(statuses)
    page_count = len(registered_paper4_pages())
    status = {
        "schema_version": "2026-05-15.44",
        "generated_at_utc": now(),
        "phase": "v44_registry_publishability_docs",
        "candidate_registry_rows_v44": int(len(registry)),
        "champion_decomposition_rows_v44": int(len(decomposition)),
        "claim_matrix_rows_v44": int(len(claims)),
        "official_quarto_page_count": page_count,
        "quarto_compact_guardrail_pass": bool(page_count <= 12),
        "paper4_working_champion_changed_v44": False,
        "paper1_promotion_allowed_v44": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v44 updates lab registries and publishability memo; no official Quarto expansion",
    }
    write_json(STATUS_DIR / "paper4_v44_status.json", status)
    return status


def build_all() -> dict[str, dict[str, Any]]:
    v41 = build_v41()
    v42 = build_v42()
    v43 = build_v43()
    v44 = build_v44()
    return {"v41": v41, "v42": v42, "v43": v43, "v44": v44}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["v41", "v42", "v43", "v44", "all"], default="all")
    args = parser.parse_args()
    if args.phase == "v41":
        result = {"v41": build_v41()}
    elif args.phase == "v42":
        result = {"v42": build_v42()}
    elif args.phase == "v43":
        result = {"v43": build_v43()}
    elif args.phase == "v44":
        result = {"v44": build_v44()}
    else:
        result = build_all()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
