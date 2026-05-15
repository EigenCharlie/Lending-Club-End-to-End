#!/usr/bin/env python3
"""Build Paper 4 v49-v53 self-directed living-lab artifacts.

This script is the first checkpoint in the post-v48 loop. It deliberately
targets the blockers that the living lab kept circling around:

* a persisted loan-scenario loss matrix,
* a real restricted-pool CVaR/source-cap LP using that matrix,
* direct qhat recalibration diagnostics for weak online source cells,
* replayable SPO/DLA-style books over the same scenario matrix,
* an updated registry/backlog/claim matrix that says what remains runnable.
* a v53 repair when binary scenario losses create artificial zero-CVaR books.

The outputs remain lab artifacts. They do not expand Quarto and do not unlock
contractual IFRS9, fair-lending, CATE policy-value, differentiable SPO+, exact
full-universe CVaR, Bellman exactness, or final promotion claims.
"""

from __future__ import annotations

import argparse
import hashlib
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
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
BOOK_CONFIG = ROOT / "book" / "_quarto.yml"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
BUDGET = 1_000_000.0


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
    pages: list[str] = []
    if not BOOK_CONFIG.exists():
        return pages
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
        capture_output=True,
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


def stable_unit(*parts: Any) -> float:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _prepare_candidate_pool() -> pd.DataFrame:
    pool = read_parquet("paper4_challenger_local_candidate_pool.parquet")
    if pool.empty:
        return pool
    pool = pool.drop_duplicates("loan_id").copy().reset_index(drop=True)
    pool["loan_index_v49"] = np.arange(len(pool), dtype=np.int32)
    if "pd_point" not in pool and "pd_point_alpha01" in pool:
        pool["pd_point"] = pool["pd_point_alpha01"]
    if "int_rate_decimal" not in pool:
        pool["int_rate_decimal"] = pd.to_numeric(pool.get("int_rate", 0), errors="coerce") / 100
    if "base_return_vec" not in pool:
        pool["base_return_vec"] = pd.to_numeric(
            pool.get("loan_amnt", 0), errors="coerce"
        ) * pd.to_numeric(pool.get("int_rate_decimal", 0), errors="coerce")
    if "lgd" not in pool:
        pool["lgd"] = 0.45
    for col in [
        "loan_amnt",
        "pd_point",
        "pd_high_alpha01",
        "int_rate_decimal",
        "base_return_vec",
        "qhat_v4",
        "weak_source_proxy",
        "lgd",
        "annual_inc",
        "dti",
        "fico_score",
    ]:
        if col not in pool:
            pool[col] = 0.0
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0.0)
    pool["issue_month"] = (
        pd.to_datetime(pool["issue_month"], errors="coerce").dt.to_period("M").astype(str)
    )
    for col in ["grade", "period", "score_decile", "state_top20", "income_band", "dti_band"]:
        if col not in pool:
            pool[col] = "unknown"
        pool[col] = pool[col].astype(str).fillna("unknown")
    return pool


def build_v49_loss_matrix(n_paths: int = 128) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_candidate_pool()
    paths = read_parquet("paper4_v31_sample_paths.parquet")
    if pool.empty or paths.empty:
        empty = pd.DataFrame()
        empty.to_parquet(TABLE_DIR / "paper4_v49_loan_scenario_loss_matrix.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v49_loan_scenario_summary.csv", empty)
        write_csv(TABLE_DIR / "paper4_v49_solver_candidate_pool.csv", empty)
        return empty, empty, empty

    path_ids = sorted(paths["path_id"].drop_duplicates().astype(int).tolist())[:n_paths]
    p = paths.loc[paths["path_id"].isin(path_ids)].copy()
    p["issue_month"] = pd.to_datetime(p["month"], errors="coerce").dt.to_period("M").astype(str)
    path_cols = [
        "path_id",
        "issue_month",
        "macro_regime_v15",
        "default_factor_v15",
        "lgd_factor_v15",
        "prepay_factor_v15",
        "path_family_v19",
    ]
    p = p[path_cols].drop_duplicates(["path_id", "issue_month"])
    fallback = (
        p.sort_values("issue_month")
        .groupby("path_id", dropna=False)
        .head(1)
        .assign(issue_month="__fallback__")
    )
    p = pd.concat([p, fallback], ignore_index=True)

    rows: list[pd.DataFrame] = []
    pool_small = pool[
        [
            "loan_index_v49",
            "loan_id",
            "issue_month",
            "loan_amnt",
            "pd_point",
            "pd_high_alpha01",
            "base_return_vec",
            "lgd",
            "grade",
            "period",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
            "qhat_v4",
            "weak_source_proxy",
        ]
    ].copy()
    for path_id in path_ids:
        factors = p.loc[p["path_id"].eq(path_id)].drop(columns=["path_id"])
        merged = pool_small.merge(factors, on="issue_month", how="left")
        missing = merged["default_factor_v15"].isna()
        if missing.any():
            fallback_row = factors.loc[factors["issue_month"].eq("__fallback__")].head(1)
            for col in [
                "macro_regime_v15",
                "default_factor_v15",
                "lgd_factor_v15",
                "prepay_factor_v15",
                "path_family_v19",
            ]:
                merged.loc[missing, col] = (
                    fallback_row[col].iloc[0] if not fallback_row.empty else 1.0
                )
        default_prob = (
            merged["pd_high_alpha01"].clip(0, 1)
            * pd.to_numeric(merged["default_factor_v15"], errors="coerce").fillna(1.0)
        ).clip(0, 0.95)
        uniforms = np.fromiter(
            (stable_unit(path_id, loan_id, "v49_default") for loan_id in merged["loan_id"]),
            dtype=float,
            count=len(merged),
        )
        default_event = uniforms < default_prob.to_numpy()
        lgd_factor = (
            pd.to_numeric(merged["lgd_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        gross_loss = merged["loan_amnt"] * merged["lgd"].clip(0, 1) * lgd_factor
        loss_amount = np.where(default_event, gross_loss, 0.0)
        prepay_drag = (
            merged["loan_amnt"]
            * 0.015
            * pd.to_numeric(merged["prepay_factor_v15"], errors="coerce")
            .fillna(1.0)
            .clip(0.25, 2.5)
            * (~default_event)
        )
        return_amount = merged["base_return_vec"] - loss_amount - prepay_drag
        out = pd.DataFrame(
            {
                "scenario_id": path_id,
                "loan_index_v49": merged["loan_index_v49"].astype(np.int32),
                "loan_id": merged["loan_id"].astype(str),
                "issue_month": merged["issue_month"].astype(str),
                "macro_regime_v15": merged["macro_regime_v15"].astype(str),
                "path_family_v19": merged["path_family_v19"].astype(str),
                "default_prob_v49": default_prob.astype(float),
                "default_event_v49": default_event.astype(bool),
                "loss_amount_v49": loss_amount.astype(float),
                "return_amount_v49": return_amount.astype(float),
                "loan_amnt": merged["loan_amnt"].astype(float),
            }
        )
        rows.append(out)
    matrix = pd.concat(rows, ignore_index=True)
    matrix["loss_matrix_scope_v49"] = (
        "restricted 12k candidate pool x 128 common paths; internal calibration, not forecast"
    )
    matrix.to_parquet(TABLE_DIR / "paper4_v49_loan_scenario_loss_matrix.parquet", index=False)

    summary = (
        matrix.groupby(["scenario_id", "macro_regime_v15", "path_family_v19"], dropna=False)
        .agg(
            n_loans=("loan_id", "nunique"),
            default_rate=("default_event_v49", "mean"),
            mean_loss=("loss_amount_v49", "mean"),
            p95_loss=(
                "loss_amount_v49",
                lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95),
            ),
            mean_return=("return_amount_v49", "mean"),
        )
        .reset_index()
    )
    summary["external_forecast_validation_claim_allowed"] = False
    summary["claim_boundary_v49"] = (
        "scenario matrix for internal paired optimization only; no external forecast validation"
    )
    write_csv(TABLE_DIR / "paper4_v49_loan_scenario_summary.csv", summary)

    pool_out = pool[
        [
            "loan_index_v49",
            "loan_id",
            "issue_month",
            "loan_amnt",
            "base_return_vec",
            "pd_point",
            "pd_high_alpha01",
            "qhat_v4",
            "weak_source_proxy",
            "grade",
            "period",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
        ]
    ].copy()
    write_csv(TABLE_DIR / "paper4_v49_solver_candidate_pool.csv", pool_out)
    return matrix, summary, pool_out


def build_v49_online_repair() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = read_parquet("paper4_v9_online_selected_intervals.parquet")
    base = read_parquet(
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
            "y_true",
            "y_pred",
        ],
    )
    if selected.empty or base.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v49_online_qhat_recalibration_search.csv", empty)
        write_csv(TABLE_DIR / "paper4_v49_online_weak_cell_diagnostics.csv", empty)
        return empty, empty
    df = selected.merge(base, on=["loan_id", "issue_month"], how="left")
    df["issue_month"] = (
        pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").astype(str)
    )
    df["residual"] = (
        pd.to_numeric(df["y_true"], errors="coerce") - pd.to_numeric(df["y_pred"], errors="coerce")
    ).abs()
    df["qhat_v9"] = pd.to_numeric(df["qhat_v9"], errors="coerce").fillna(0)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0)
    source_specs = {
        "grade": "original_grade",
        "period": "period",
        "term": "term",
        "score_decile": "score_decile",
        "state_top20": "state_top20",
        "income_band": "income_band",
        "dti_band": "dti_band",
    }
    weak_rows: list[pd.DataFrame] = []
    search_rows: list[dict[str, Any]] = []
    deltas = np.round(np.arange(0.0, 0.161, 0.02), 3)
    for family, col in source_specs.items():
        work = df.assign(source_family=family, source_id=df[col].astype(str))
        base_cells = (
            work.groupby(["source_family", "source_id", "issue_month"], dropna=False)
            .agg(
                n=("loan_id", "size"),
                coverage=("covered_online_v9", "mean"),
                avg_width=("interval_width_online_v9", "mean"),
            )
            .reset_index()
        )
        weak_rows.append(
            base_cells.sort_values(["coverage", "n"], ascending=[True, False]).head(12)
        )
        for delta in deltas:
            qhat = (work["qhat_v9"] + delta).clip(0, 1)
            low = (work["y_pred"] - qhat).clip(0, 1)
            high = (work["y_pred"] + qhat).clip(0, 1)
            coverage = ((work["y_true"] >= low) & (work["y_true"] <= high)).astype(float)
            width = high - low
            tmp = work[["source_family", "source_id", "issue_month", "policy_id"]].copy()
            tmp["coverage"] = coverage
            tmp["width"] = width
            source_cells = (
                tmp.groupby(["source_id", "issue_month"], dropna=False)
                .agg(n=("coverage", "size"), coverage=("coverage", "mean"), width=("width", "mean"))
                .reset_index()
            )
            policy_cells = (
                tmp.groupby(["policy_id", "issue_month"], dropna=False)
                .agg(n=("coverage", "size"), coverage=("coverage", "mean"))
                .reset_index()
            )
            defended_source = source_cells.loc[source_cells["n"].ge(3), "coverage"]
            defended_policy = policy_cells.loc[policy_cells["n"].ge(3), "coverage"]
            source_min = float(defended_source.min()) if len(defended_source) else np.nan
            policy_min = float(defended_policy.min()) if len(defended_policy) else np.nan
            avg_width = float(tmp["width"].mean())
            gate = bool(source_min >= 0.80 and policy_min >= 0.90 and avg_width <= 0.95)
            search_rows.append(
                {
                    "source_family": family,
                    "delta_qhat_v49": delta,
                    "source_month_defended_min_v49": source_min,
                    "policy_month_defended_min_v49": policy_min,
                    "avg_width_loan_v49": avg_width,
                    "n_defended_source_cells_v49": int(len(defended_source)),
                    "n_defended_policy_cells_v49": int(len(defended_policy)),
                    "gate_source80_policy90_width95_v49": gate,
                    "strict_live_deployability_claim_allowed": False,
                    "claim_boundary_v49": "direct historical qhat recalibration search; no live deployability claim",
                }
            )
    weak = pd.concat(weak_rows, ignore_index=True)
    weak["weak_cell_scope_v49"] = "lowest direct source-month coverage cells before qhat repair"
    search = pd.DataFrame(search_rows)
    search["repair_decision_v49"] = np.where(
        search["gate_source80_policy90_width95_v49"],
        "candidate_repair_passes_historical_direct_gate",
        "still_below_direct_gate",
    )
    write_csv(TABLE_DIR / "paper4_v49_online_qhat_recalibration_search.csv", search)
    write_csv(TABLE_DIR / "paper4_v49_online_weak_cell_diagnostics.csv", weak)
    return search, weak


def build_v49() -> dict[str, Any]:
    start = datetime.now(UTC)
    matrix, summary, pool = build_v49_loss_matrix()
    online_search, weak = build_v49_online_repair()
    status = {
        "schema_version": "2026-05-15.49",
        "generated_at_utc": now(),
        "phase": "v49_loss_matrix_online_repair",
        "loss_matrix_rows_v49": int(len(matrix)),
        "loss_matrix_loans_v49": int(matrix["loan_id"].nunique()) if not matrix.empty else 0,
        "loss_matrix_scenarios_v49": int(matrix["scenario_id"].nunique())
        if not matrix.empty
        else 0,
        "loss_matrix_scope_v49": "restricted_candidate_pool_internal_scenarios",
        "external_forecast_validation_claim_allowed": False,
        "solver_candidate_pool_rows_v49": int(len(pool)),
        "online_repair_rows_v49": int(len(online_search)),
        "online_repair_gate_pass_rows_v49": int(
            online_search["gate_source80_policy90_width95_v49"].sum()
        )
        if not online_search.empty
        else 0,
        "strict_live_deployability_claim_allowed": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v49 creates restricted loss matrix and direct qhat search; no external forecast or live deployment claim",
    }
    write_json(STATUS_DIR / "paper4_v49_status.json", status)
    return status


def _matrix_for_solver(
    matrix: pd.DataFrame, pool: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, sparse.csr_matrix, list[int]]:
    exp = (
        matrix.groupby("loan_index_v49", dropna=False)
        .agg(
            expected_loss=("loss_amount_v49", "mean"), expected_return=("return_amount_v49", "mean")
        )
        .reset_index()
    )
    p = pool.merge(exp, on="loan_index_v49", how="inner")
    p["solver_score_v50"] = (
        pd.to_numeric(p["expected_return"], errors="coerce").fillna(0)
        - 0.60 * pd.to_numeric(p["expected_loss"], errors="coerce").fillna(0)
        - 1800 * pd.to_numeric(p["qhat_v4"], errors="coerce").fillna(0)
        - 1500 * pd.to_numeric(p["weak_source_proxy"], errors="coerce").fillna(0)
    )
    p = p.sort_values("solver_score_v50", ascending=False).head(top_n).reset_index(drop=True)
    old_to_new = {int(idx): i for i, idx in enumerate(p["loan_index_v49"].astype(int))}
    scen_ids = sorted(matrix["scenario_id"].drop_duplicates().astype(int).tolist())
    scen_to_row = {sid: i for i, sid in enumerate(scen_ids)}
    m = matrix.loc[matrix["loan_index_v49"].astype(int).isin(old_to_new)].copy()
    rows = m["scenario_id"].astype(int).map(scen_to_row).to_numpy()
    cols = m["loan_index_v49"].astype(int).map(old_to_new).to_numpy()
    losses = m["loss_amount_v49"].astype(float).to_numpy()
    loss_mat = sparse.coo_matrix((losses, (rows, cols)), shape=(len(scen_ids), len(p))).tocsr()
    returns = p["expected_return"].astype(float).to_numpy()
    amounts = p["loan_amnt"].astype(float).to_numpy()
    return p, returns, amounts, loss_mat, scen_ids


def _matrix_for_solver_columns(
    matrix: pd.DataFrame,
    pool: pd.DataFrame,
    top_n: int,
    loss_col: str,
    return_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, sparse.csr_matrix, list[int]]:
    exp = (
        matrix.groupby("loan_index_v49", dropna=False)
        .agg(expected_loss=(loss_col, "mean"), expected_return=(return_col, "mean"))
        .reset_index()
    )
    p = pool.merge(exp, on="loan_index_v49", how="inner")
    p[score_col] = (
        pd.to_numeric(p["expected_return"], errors="coerce").fillna(0)
        - 0.80 * pd.to_numeric(p["expected_loss"], errors="coerce").fillna(0)
        - 2000 * pd.to_numeric(p["qhat_v4"], errors="coerce").fillna(0)
        - 2200 * pd.to_numeric(p["weak_source_proxy"], errors="coerce").fillna(0)
    )
    p = p.sort_values(score_col, ascending=False).head(top_n).reset_index(drop=True)
    old_to_new = {int(idx): i for i, idx in enumerate(p["loan_index_v49"].astype(int))}
    scen_ids = sorted(matrix["scenario_id"].drop_duplicates().astype(int).tolist())
    scen_to_row = {sid: i for i, sid in enumerate(scen_ids)}
    m = matrix.loc[matrix["loan_index_v49"].astype(int).isin(old_to_new)].copy()
    rows = m["scenario_id"].astype(int).map(scen_to_row).to_numpy()
    cols = m["loan_index_v49"].astype(int).map(old_to_new).to_numpy()
    losses = m[loss_col].astype(float).to_numpy()
    loss_mat = sparse.coo_matrix((losses, (rows, cols)), shape=(len(scen_ids), len(p))).tocsr()
    returns = p["expected_return"].astype(float).to_numpy()
    amounts = p["loan_amnt"].astype(float).to_numpy()
    return p, returns, amounts, loss_mat, scen_ids


def _solve_cvar_lp(
    p: pd.DataFrame,
    returns: np.ndarray,
    amounts: np.ndarray,
    loss_mat: sparse.csr_matrix,
    regime: str,
    cvar_cap: float,
    return_floor: float,
    cap_multiplier: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    n = len(p)
    s = loss_mat.shape[0]
    eta_idx = n
    u_start = n + 1
    nvars = n + 1 + s
    alpha = 0.90
    coeff = 1.0 / ((1 - alpha) * s)

    c = np.zeros(nvars)
    c[:n] = -returns
    c[eta_idx] = 1e-4
    c[u_start:] = 1e-4 * coeff

    rows = []
    rhs = []
    senses = []
    # budget upper and deployment lower
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = amounts
    rows.append(row.tocsr())
    rhs.append(BUDGET)
    senses.append("budget_upper")
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -amounts
    rows.append(row.tocsr())
    rhs.append(-0.98 * BUDGET)
    senses.append("budget_lower")
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -returns
    rows.append(row.tocsr())
    rhs.append(-return_floor)
    senses.append("return_floor")
    row = sparse.lil_matrix((1, nvars))
    row[0, eta_idx] = 1.0
    row[0, u_start:] = coeff
    rows.append(row.tocsr())
    rhs.append(cvar_cap)
    senses.append("cvar_cap")

    # scenario constraints: L_s x - eta - u_s <= 0
    scen = sparse.lil_matrix((s, nvars))
    scen[:, :n] = loss_mat
    scen[:, eta_idx] = -1.0
    for i in range(s):
        scen[i, u_start + i] = -1.0
    rows.append(scen.tocsr())
    rhs.extend([0.0] * s)
    senses.extend(["scenario_excess"] * s)

    source_constraints = []
    for family in ["grade", "period", "score_decile", "income_band", "dti_band", "state_top20"]:
        if family not in p:
            continue
        shares = p.groupby(family)["loan_amnt"].sum() / p["loan_amnt"].sum()
        base_cap = float(min(0.55, max(0.12, shares.quantile(0.90))))
        for source_id in shares.index:
            mask = p[family].astype(str).eq(str(source_id)).to_numpy()
            if not mask.any():
                continue
            cap = min(0.90, base_cap * cap_multiplier + (0.02 if regime != "strict" else 0.0))
            row = sparse.lil_matrix((1, nvars))
            row[0, np.where(mask)[0]] = amounts[mask]
            rows.append(row.tocsr())
            rhs.append(cap * BUDGET)
            senses.append(f"source_cap::{family}::{source_id}")
            source_constraints.append((family, source_id, cap))

    A_ub = sparse.vstack(rows).tocsr()
    b_ub = np.array(rhs, dtype=float)
    bounds = [(0.0, 1.0)] * n + [(0.0, BUDGET)] + [(0.0, BUDGET)] * s
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    status = {
        "regime_v50": regime,
        "solver_success_v50": bool(result.success),
        "solver_status_v50": int(result.status),
        "solver_message_v50": str(result.message),
        "restricted_pool_n_v50": n,
        "scenario_count_v50": s,
        "cvar_cap_v50": cvar_cap,
        "return_floor_v50": return_floor,
        "source_cap_count_v50": len(source_constraints),
        "exact_full_universe_claim_v50": False,
        "claim_boundary_v50": "exact LP over restricted candidate pool and persisted scenario matrix; not full-universe optimality",
    }
    if not result.success:
        cert = pd.DataFrame(
            [
                {
                    **status,
                    "required_cvar_slack_proxy_v50": np.nan,
                    "required_return_floor_relaxation_proxy_v50": np.nan,
                    "certificate_scope_v50": "solver infeasible or failed before allocation; diagnostic only",
                }
            ]
        )
        return status, pd.DataFrame(), cert

    x = result.x[:n]
    eta = result.x[eta_idx]
    u = result.x[u_start:]
    scen_losses = np.asarray(loss_mat @ x).reshape(-1)
    cvar = float(eta + coeff * u.sum())
    expected_return = float(returns @ x)
    exposure = float(amounts @ x)
    allocations = p.loc[x > 1e-6].copy()
    allocations["allocation_fraction_v50"] = x[x > 1e-6]
    allocations["allocated_exposure_v50"] = (
        allocations["loan_amnt"] * allocations["allocation_fraction_v50"]
    )
    allocations["regime_v50"] = regime
    allocations["policy_id_v50"] = f"v50_cvar_source_lp_{regime}"
    allocations["claim_boundary_v50"] = status["claim_boundary_v50"]

    active_rows = []
    for family, source_id, cap in source_constraints:
        mask = p[family].astype(str).eq(str(source_id)).to_numpy()
        used = float(amounts[mask] @ x[mask])
        slack = cap * BUDGET - used
        if slack <= 0.01 * BUDGET:
            active_rows.append(
                {
                    "regime_v50": regime,
                    "constraint_type": "source_cap",
                    "source_family": family,
                    "source_id": source_id,
                    "cap_v50": cap,
                    "used_exposure_v50": used,
                    "slack_v50": slack,
                    "active_v50": slack <= 1e-5,
                }
            )
    scenario = pd.DataFrame(
        {
            "regime_v50": regime,
            "scenario_row": np.arange(s),
            "scenario_loss_v50": scen_losses,
            "eta_v50": eta,
            "excess_loss_v50": np.maximum(scen_losses - eta, 0),
        }
    )
    scenario.to_csv(TABLE_DIR / f"paper4_v50_cvar_scenario_losses_{regime}.csv", index=False)
    if active_rows:
        pd.DataFrame(active_rows).to_csv(
            TABLE_DIR / f"paper4_v50_cvar_active_constraints_{regime}.csv", index=False
        )
    status.update(
        {
            "objective_return_v50": expected_return,
            "allocated_exposure_v50": exposure,
            "n_allocated_loans_v50": int((x > 1e-6).sum()),
            "scenario_loss_mean_v50": float(scen_losses.mean()),
            "scenario_loss_p95_v50": float(np.quantile(scen_losses, 0.95)),
            "scenario_loss_cvar90_v50": cvar,
            "budget_slack_v50": BUDGET - exposure,
        }
    )
    cert = pd.DataFrame(
        [
            {
                **status,
                "required_cvar_slack_proxy_v50": max(0.0, cvar - cvar_cap),
                "required_return_floor_relaxation_proxy_v50": max(
                    0.0, return_floor - expected_return
                ),
                "certificate_scope_v50": "post-solve slack over restricted LP allocation",
            }
        ]
    )
    return status, allocations, cert


def _solve_cvar_lp_tagged(
    p: pd.DataFrame,
    returns: np.ndarray,
    amounts: np.ndarray,
    loss_mat: sparse.csr_matrix,
    regime: str,
    cvar_cap: float,
    return_floor: float,
    cap_multiplier: float,
    tag: str,
    output_stem: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    n = len(p)
    s = loss_mat.shape[0]
    eta_idx = n
    u_start = n + 1
    nvars = n + 1 + s
    alpha = 0.90
    coeff = 1.0 / ((1 - alpha) * s)

    c = np.zeros(nvars)
    c[:n] = -returns
    c[eta_idx] = 1e-4
    c[u_start:] = 1e-4 * coeff

    rows = []
    rhs = []
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = amounts
    rows.append(row.tocsr())
    rhs.append(BUDGET)
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -amounts
    rows.append(row.tocsr())
    rhs.append(-0.98 * BUDGET)
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -returns
    rows.append(row.tocsr())
    rhs.append(-return_floor)
    row = sparse.lil_matrix((1, nvars))
    row[0, eta_idx] = 1.0
    row[0, u_start:] = coeff
    rows.append(row.tocsr())
    rhs.append(cvar_cap)

    scen = sparse.lil_matrix((s, nvars))
    scen[:, :n] = loss_mat
    scen[:, eta_idx] = -1.0
    for i in range(s):
        scen[i, u_start + i] = -1.0
    rows.append(scen.tocsr())
    rhs.extend([0.0] * s)

    source_constraints = []
    for family in ["grade", "period", "score_decile", "income_band", "dti_band", "state_top20"]:
        if family not in p:
            continue
        shares = p.groupby(family)["loan_amnt"].sum() / p["loan_amnt"].sum()
        base_cap = float(min(0.55, max(0.12, shares.quantile(0.90))))
        for source_id in shares.index:
            mask = p[family].astype(str).eq(str(source_id)).to_numpy()
            if not mask.any():
                continue
            cap = min(0.90, base_cap * cap_multiplier + (0.02 if regime != "strict" else 0.0))
            row = sparse.lil_matrix((1, nvars))
            row[0, np.where(mask)[0]] = amounts[mask]
            rows.append(row.tocsr())
            rhs.append(cap * BUDGET)
            source_constraints.append((family, source_id, cap))

    result = linprog(
        c,
        A_ub=sparse.vstack(rows).tocsr(),
        b_ub=np.array(rhs, dtype=float),
        bounds=[(0.0, 1.0)] * n + [(0.0, BUDGET)] + [(0.0, BUDGET)] * s,
        method="highs",
    )
    status = {
        f"regime_{tag}": regime,
        f"solver_success_{tag}": bool(result.success),
        f"solver_status_{tag}": int(result.status),
        f"solver_message_{tag}": str(result.message),
        f"restricted_pool_n_{tag}": n,
        f"scenario_count_{tag}": s,
        f"cvar_cap_{tag}": cvar_cap,
        f"return_floor_{tag}": return_floor,
        f"source_cap_count_{tag}": len(source_constraints),
        f"exact_full_universe_claim_{tag}": False,
        f"claim_boundary_{tag}": (
            "exact LP over restricted candidate pool with expected/hybrid scenario "
            "losses; not full-universe optimality"
        ),
    }
    if not result.success:
        cert = pd.DataFrame(
            [
                {
                    **status,
                    f"required_cvar_slack_proxy_{tag}": np.nan,
                    f"required_return_floor_relaxation_proxy_{tag}": np.nan,
                    f"certificate_scope_{tag}": "solver infeasible or failed before allocation; diagnostic only",
                }
            ]
        )
        return status, pd.DataFrame(), cert

    x = result.x[:n]
    eta = result.x[eta_idx]
    u = result.x[u_start:]
    scen_losses = np.asarray(loss_mat @ x).reshape(-1)
    cvar = float(eta + coeff * u.sum())
    expected_return = float(returns @ x)
    exposure = float(amounts @ x)

    allocations = p.loc[x > 1e-6].copy()
    allocations[f"allocation_fraction_{tag}"] = x[x > 1e-6]
    allocations[f"allocated_exposure_{tag}"] = (
        allocations["loan_amnt"] * allocations[f"allocation_fraction_{tag}"]
    )
    allocations[f"regime_{tag}"] = regime
    allocations[f"policy_id_{tag}"] = f"{tag}_cvar_expected_loss_{regime}"
    allocations[f"claim_boundary_{tag}"] = status[f"claim_boundary_{tag}"]

    active_rows = []
    for family, source_id, cap in source_constraints:
        mask = p[family].astype(str).eq(str(source_id)).to_numpy()
        used = float(amounts[mask] @ x[mask])
        slack = cap * BUDGET - used
        if slack <= 0.01 * BUDGET:
            active_rows.append(
                {
                    f"regime_{tag}": regime,
                    "constraint_type": "source_cap",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_{tag}": cap,
                    f"used_exposure_{tag}": used,
                    f"slack_{tag}": slack,
                    f"active_{tag}": slack <= 1e-5,
                }
            )

    scenario = pd.DataFrame(
        {
            f"regime_{tag}": regime,
            "scenario_row": np.arange(s),
            f"scenario_loss_{tag}": scen_losses,
            f"eta_{tag}": eta,
            f"excess_loss_{tag}": np.maximum(scen_losses - eta, 0),
        }
    )
    scenario.to_csv(TABLE_DIR / f"{output_stem}_scenario_losses_{regime}.csv", index=False)
    if active_rows:
        pd.DataFrame(active_rows).to_csv(
            TABLE_DIR / f"{output_stem}_active_constraints_{regime}.csv", index=False
        )

    status.update(
        {
            f"objective_return_{tag}": expected_return,
            f"allocated_exposure_{tag}": exposure,
            f"n_allocated_loans_{tag}": int((x > 1e-6).sum()),
            f"scenario_loss_mean_{tag}": float(scen_losses.mean()),
            f"scenario_loss_p95_{tag}": float(np.quantile(scen_losses, 0.95)),
            f"scenario_loss_cvar90_{tag}": cvar,
            f"budget_slack_{tag}": BUDGET - exposure,
        }
    )
    cert = pd.DataFrame(
        [
            {
                **status,
                f"required_cvar_slack_proxy_{tag}": max(0.0, cvar - cvar_cap),
                f"required_return_floor_relaxation_proxy_{tag}": max(
                    0.0, return_floor - expected_return
                ),
                f"certificate_scope_{tag}": "post-solve slack over restricted LP allocation",
            }
        ]
    )
    return status, allocations, cert


def build_v50_cvar_solver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = read_parquet("paper4_v49_loan_scenario_loss_matrix.parquet")
    pool = read_csv("paper4_v49_solver_candidate_pool.csv")
    if matrix.empty or pool.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v50_cvar_source_lp_frontier.csv", empty)
        write_csv(TABLE_DIR / "paper4_v50_cvar_infeasibility_certificate.csv", empty)
        empty.to_parquet(TABLE_DIR / "paper4_v50_cvar_source_lp_allocations.parquet", index=False)
        return empty, empty, empty
    p, returns, amounts, loss_mat, _ = _matrix_for_solver(matrix, pool, top_n=3000)
    regimes = [
        ("strict", 90_000.0, 110_000.0, 1.00),
        ("committee", 140_000.0, 120_000.0, 1.25),
        ("relaxed", 210_000.0, 125_000.0, 1.60),
        ("tail_first", 110_000.0, 80_000.0, 1.60),
    ]
    frontier_rows = []
    allocs = []
    certs = []
    for regime, cvar_cap, return_floor, cap_mult in regimes:
        status, allocation, cert = _solve_cvar_lp(
            p, returns, amounts, loss_mat, regime, cvar_cap, return_floor, cap_mult
        )
        frontier_rows.append(status)
        if not allocation.empty:
            allocs.append(allocation)
        if not cert.empty:
            certs.append(cert)
    frontier = pd.DataFrame(frontier_rows)
    allocations = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    certificate = pd.concat(certs, ignore_index=True) if certs else pd.DataFrame()
    if not frontier.empty:
        frontier["return_norm_v50"] = normalize(
            frontier.get("objective_return_v50", pd.Series(dtype=float))
        )
        frontier["tail_norm_v50"] = normalize(
            frontier.get("scenario_loss_cvar90_v50", pd.Series(dtype=float)), higher_is_better=False
        )
        frontier["frontier_score_v50"] = (
            0.55 * frontier["return_norm_v50"] + 0.45 * frontier["tail_norm_v50"]
        )
        frontier["non_dominated_restricted_v50"] = frontier["solver_success_v50"].astype(bool) & (
            frontier["frontier_score_v50"] >= frontier["frontier_score_v50"].median()
        )
    write_csv(TABLE_DIR / "paper4_v50_cvar_source_lp_frontier.csv", frontier)
    write_csv(TABLE_DIR / "paper4_v50_cvar_infeasibility_certificate.csv", certificate)
    allocations.to_parquet(TABLE_DIR / "paper4_v50_cvar_source_lp_allocations.parquet", index=False)
    return frontier, allocations, certificate


def build_v50() -> dict[str, Any]:
    start = datetime.now(UTC)
    frontier, allocations, cert = build_v50_cvar_solver()
    status = {
        "schema_version": "2026-05-15.50",
        "generated_at_utc": now(),
        "phase": "v50_cvar_source_lp_solver",
        "cvar_frontier_rows_v50": int(len(frontier)),
        "cvar_success_rows_v50": int(frontier["solver_success_v50"].sum())
        if not frontier.empty
        else 0,
        "cvar_allocation_rows_v50": int(len(allocations)),
        "cvar_certificate_rows_v50": int(len(cert)),
        "exact_full_universe_cvar_claim_allowed": False,
        "restricted_pool_exact_lp_claim_allowed": bool(
            not frontier.empty and frontier["solver_success_v50"].astype(bool).any()
        ),
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v50 solves exact LPs over restricted candidate pool; full-universe optimality remains false",
    }
    write_json(STATUS_DIR / "paper4_v50_status.json", status)
    return status


def build_v51_spo_dla_cases() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_candidate_pool()
    matrix = read_parquet("paper4_v49_loan_scenario_loss_matrix.parquet")
    if pool.empty or matrix.empty:
        empty = pd.DataFrame()
        empty.to_parquet(TABLE_DIR / "paper4_v51_policy_books.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v51_policy_scenario_replay.csv", empty)
        write_csv(TABLE_DIR / "paper4_v51_spo_solver_oracle_bridge.csv", empty)
        write_csv(TABLE_DIR / "paper4_v51_champion_economic_case_studies.csv", empty)
        return empty, empty, empty, empty
    exp = (
        matrix.groupby("loan_index_v49", dropna=False)
        .agg(
            expected_loss=("loss_amount_v49", "mean"), expected_return=("return_amount_v49", "mean")
        )
        .reset_index()
    )
    p = pool.merge(exp, on="loan_index_v49", how="left")
    p["return_score"] = p["expected_return"] - 0.30 * p["expected_loss"]
    p["tail_guard_score"] = (
        p["expected_return"]
        - 0.90 * p["expected_loss"]
        - 2500 * p["qhat_v4"]
        - 2000 * p["weak_source_proxy"]
    )
    p["source_balanced_score"] = (
        p["expected_return"]
        - 0.65 * p["expected_loss"]
        - 3500 * p["weak_source_proxy"]
        - 1500 * p["qhat_v4"]
    )
    score_specs = {
        "v51_spo_solver_oracle_return": "return_score",
        "v51_dla_tail_guard_book": "tail_guard_score",
        "v51_dla_source_balanced_book": "source_balanced_score",
    }
    book_rows = []
    for policy_id, score_col in score_specs.items():
        for _month, group in p.groupby("issue_month", dropna=False):
            monthly_budget = BUDGET / max(p["issue_month"].nunique(), 1)
            g = group.sort_values(score_col, ascending=False).copy()
            selected = g["loan_amnt"].cumsum() <= monthly_budget
            if not selected.any() and not g.empty:
                selected.iloc[0] = True
            chosen = g.loc[selected].copy()
            chosen["policy_id"] = policy_id
            chosen["selection_score_v51"] = chosen[score_col]
            chosen["book_claim_boundary_v51"] = (
                "materialized loan book from scenario-matrix score; not Bellman exact and not differentiable SPO"
            )
            book_rows.append(chosen)
    books = pd.concat(book_rows, ignore_index=True) if book_rows else pd.DataFrame()
    books.to_parquet(TABLE_DIR / "paper4_v51_policy_books.parquet", index=False)

    replay_rows = []
    for policy_id, book in books.groupby("policy_id", dropna=False):
        selected_idx = set(book["loan_index_v49"].astype(int))
        m = matrix.loc[matrix["loan_index_v49"].astype(int).isin(selected_idx)].copy()
        replay = (
            m.groupby("scenario_id", dropna=False)
            .agg(
                scenario_loss=("loss_amount_v49", "sum"),
                scenario_return=("return_amount_v49", "sum"),
                funded_exposure=("loan_amnt", "sum"),
                defaults=("default_event_v49", "sum"),
            )
            .reset_index()
        )
        replay_rows.append(
            {
                "policy_id": policy_id,
                "n_scenarios": int(len(replay)),
                "n_loans": int(book["loan_id"].nunique()),
                "funded_exposure": float(book["loan_amnt"].sum()),
                "mean_scenario_return_v51": float(replay["scenario_return"].mean()),
                "p05_scenario_return_v51": float(replay["scenario_return"].quantile(0.05)),
                "p95_scenario_loss_v51": float(replay["scenario_loss"].quantile(0.95)),
                "mean_defaults_v51": float(replay["defaults"].mean()),
                "claim_boundary_v51": "scenario-matrix replay over materialized books; no production forecast claim",
            }
        )
    scenario_replay = pd.DataFrame(replay_rows)
    write_csv(TABLE_DIR / "paper4_v51_policy_scenario_replay.csv", scenario_replay)

    v46 = read_csv("paper4_v46_spo_training_report.csv")
    if not v46.empty and not scenario_replay.empty:
        best_oracle = scenario_replay.sort_values("mean_scenario_return_v51", ascending=False).head(
            1
        )
        bridge = v46.copy()
        bridge["solver_oracle_policy_v51"] = best_oracle["policy_id"].iloc[0]
        bridge["solver_oracle_mean_return_v51"] = best_oracle["mean_scenario_return_v51"].iloc[0]
        bridge["oracle_route_v51"] = (
            "score-greedy SPO is now benchmarked against scenario-matrix materialized books"
        )
        bridge["formal_differentiable_spo_claim_allowed"] = False
    else:
        bridge = pd.DataFrame()
    write_csv(TABLE_DIR / "paper4_v51_spo_solver_oracle_bridge.csv", bridge)

    champion = read_parquet("paper4_v47_champion_decomposition_loan_level.parquet")
    if not champion.empty:
        cases = (
            champion.sort_values(
                ["selection_relation_v47", "tail_loss_proxy_v47", "realized_return_proxy_lgd45"],
                ascending=[True, True, False],
            )
            .groupby(["policy_id", "selection_relation_v47"], dropna=False)
            .head(4)
            .reset_index(drop=True)
        )
        cases["case_study_scope_v51"] = (
            "economic loan-level case study: selected/avoided/replaced relative to champion"
        )
    else:
        cases = pd.DataFrame()
    write_csv(TABLE_DIR / "paper4_v51_champion_economic_case_studies.csv", cases)
    return books, scenario_replay, bridge, cases


def build_v51_sicr() -> pd.DataFrame:
    panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if panel.empty:
        out = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v51_sicr_targeted_recalibration.csv", out)
        return out
    p = panel.copy()
    ead = pd.to_numeric(p.get("ead_start_proxy_v25", 1), errors="coerce").replace(0, np.nan)
    ecl = pd.to_numeric(p.get("ecl_proxy_v29", 0), errors="coerce").fillna(0)
    p["pd_lifetime_proxy_v51"] = (ecl / ead / 0.45).clip(0, 1).fillna(0)
    rules = []
    for abs_thr in np.round(np.arange(0.18, 0.41, 0.03), 3):
        for top_q in [0.50, 0.60, 0.70, 0.80]:
            tmp = p.copy()
            q = tmp.groupby(["policy_id", "scenario"], dropna=False)[
                "pd_lifetime_proxy_v51"
            ].transform(lambda s, q=top_q: s.quantile(q))
            stage2 = tmp["pd_lifetime_proxy_v51"].ge(abs_thr) | tmp["pd_lifetime_proxy_v51"].ge(q)
            tmp["stage2_v51"] = stage2
            agg = (
                tmp.groupby(["policy_id", "scenario"], dropna=False)
                .agg(
                    stage2_share_v51=("stage2_v51", "mean"),
                    ecl_proxy_total_v51=("ecl_proxy_v29", "sum"),
                )
                .reset_index()
            )
            agg["sicr_rule_v51"] = f"abs_{abs_thr:.3f}_or_within_policy_topq_{top_q:.2f}"
            agg["stage2_target_distance_v51"] = (agg["stage2_share_v51"] - 0.35).abs()
            agg["contractual_ifrs9_claim_allowed"] = False
            agg["claim_boundary_v51"] = (
                "targeted SICR proxy calibration; not contractual IFRS9 staging"
            )
            rules.append(agg)
    out = pd.concat(rules, ignore_index=True)
    out["sicr_score_v51"] = 1 - normalize(out["stage2_target_distance_v51"])
    write_csv(TABLE_DIR / "paper4_v51_sicr_targeted_recalibration.csv", out)
    return out


def build_v51() -> dict[str, Any]:
    start = datetime.now(UTC)
    books, replay, bridge, cases = build_v51_spo_dla_cases()
    sicr = build_v51_sicr()
    status = {
        "schema_version": "2026-05-15.51",
        "generated_at_utc": now(),
        "phase": "v51_spo_dla_books_sicr_cases",
        "policy_book_rows_v51": int(len(books)),
        "policy_books_v51": int(books["policy_id"].nunique()) if not books.empty else 0,
        "policy_scenario_replay_rows_v51": int(len(replay)),
        "spo_solver_bridge_rows_v51": int(len(bridge)),
        "formal_differentiable_spo_claim_allowed": False,
        "bellman_exact_claim_allowed": False,
        "sicr_targeted_rows_v51": int(len(sicr)),
        "contractual_ifrs9_claim_allowed": False,
        "case_study_rows_v51": int(len(cases)),
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v51 materializes scenario-matrix books and SICR proxy search; no differentiable SPO, Bellman, or IFRS9 contractual claim",
    }
    write_json(STATUS_DIR / "paper4_v51_status.json", status)
    return status


def build_v52_registry_docs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v48 = read_csv("paper4_v48_candidate_registry.csv")
    v50 = read_csv("paper4_v50_cvar_source_lp_frontier.csv")
    v51 = read_csv("paper4_v51_policy_scenario_replay.csv")
    rows: list[dict[str, Any]] = []
    if not v48.empty:
        for _, row in v48.head(20).iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": row.get("candidate_family"),
                    "evidence_source_v52": "v48_registry",
                    "wealth_score_v52": safe_num(row.get("wealth_score_norm_v48"), 0),
                    "tail_score_v52": safe_num(row.get("tail_score_norm_v48"), 0),
                    "auditability_score_v52": safe_num(row.get("auditability_score_norm_v48"), 0.5),
                    "claim_safety_v52": bool(row.get("claim_safety_v48", True)),
                }
            )
    if not v50.empty:
        for _, row in v50.loc[v50["solver_success_v50"].astype(bool)].iterrows():
            rows.append(
                {
                    "policy_id": f"v50_cvar_source_lp_{row.get('regime_v50')}",
                    "candidate_family": "restricted_cvar_source_lp",
                    "evidence_source_v52": "v50_cvar_lp",
                    "wealth_score_v52": safe_num(row.get("return_norm_v50"), 0),
                    "tail_score_v52": safe_num(row.get("tail_norm_v50"), 0),
                    "auditability_score_v52": 0.78,
                    "claim_safety_v52": True,
                }
            )
    if not v51.empty:
        for _, row in v51.iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": "scenario_matrix_materialized_book",
                    "evidence_source_v52": "v51_policy_scenario_replay",
                    "wealth_score_v52": safe_num(row.get("mean_scenario_return_v51"), 0),
                    "tail_score_v52": -safe_num(row.get("p95_scenario_loss_v51"), 0),
                    "auditability_score_v52": 0.72,
                    "claim_safety_v52": True,
                }
            )
    registry = pd.DataFrame(rows).dropna(subset=["policy_id"])
    if not registry.empty:
        registry["wealth_norm_v52"] = normalize(registry["wealth_score_v52"])
        registry["tail_norm_v52"] = normalize(registry["tail_score_v52"])
        registry["audit_norm_v52"] = normalize(registry["auditability_score_v52"])
        registry["full_governance_score_v52"] = (
            0.35 * registry["wealth_norm_v52"]
            + 0.30 * registry["tail_norm_v52"]
            + 0.20 * registry["audit_norm_v52"]
            + 0.15 * registry["claim_safety_v52"].astype(float)
        )
        registry["decision_v52"] = np.select(
            [
                registry["policy_id"].eq("paper1_economic_champion"),
                registry["candidate_family"].eq("restricted_cvar_source_lp"),
                registry["candidate_family"].eq("scenario_matrix_materialized_book"),
            ],
            ["retain_reference", "serious_solver_challenger", "serious_lab_book_challenger"],
            default="review_or_park",
        )
        registry["paper1_promotion_allowed_v52"] = False
        registry["paper4_working_champion_allowed_v52"] = False
        registry["claim_boundary_v52"] = (
            "Paper 4 lab registry; no final promotion or Paper Estrella change"
        )
        registry = registry.sort_values("full_governance_score_v52", ascending=False)
    write_csv(TABLE_DIR / "paper4_v52_candidate_registry.csv", registry)

    claims = pd.DataFrame(
        [
            {
                "claim_id": "v52_loss_matrix_exists",
                "allowed": True,
                "artifact": "paper4_v49_loan_scenario_loss_matrix.parquet",
                "boundary": "restricted candidate pool and internal scenarios only",
            },
            {
                "claim_id": "v52_restricted_cvar_lp",
                "allowed": True,
                "artifact": "paper4_v50_cvar_source_lp_frontier.csv",
                "boundary": "exact LP over restricted pool; not full universe",
            },
            {
                "claim_id": "v52_exact_full_universe_cvar",
                "allowed": False,
                "artifact": "paper4_v50_cvar_source_lp_frontier.csv",
                "boundary": "full universe remains hard-blocked until all-loan scenario matrix exists",
            },
            {
                "claim_id": "v52_online_live_deployability",
                "allowed": False,
                "artifact": "paper4_v49_online_qhat_recalibration_search.csv",
                "boundary": "historical direct recalibration only",
            },
            {
                "claim_id": "v52_formal_differentiable_spo",
                "allowed": False,
                "artifact": "paper4_v51_spo_solver_oracle_bridge.csv",
                "boundary": "solver-oracle bridge only",
            },
            {
                "claim_id": "v52_bellman_exact_dla",
                "allowed": False,
                "artifact": "paper4_v51_policy_books.parquet",
                "boundary": "materialized books, not Bellman exact policy",
            },
            {
                "claim_id": "v52_contractual_ifrs9",
                "allowed": False,
                "artifact": "paper4_v51_sicr_targeted_recalibration.csv",
                "boundary": "SICR proxy only",
            },
            {
                "claim_id": "v52_fair_lending_legal",
                "allowed": False,
                "artifact": "paper4_v47_fairness_source_protocol.csv",
                "boundary": "source governance only",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v52_claim_matrix.csv", claims)

    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": "v50 restricted-pool CVaR/source LP is implemented; next exact-full-universe step needs all-loan scenario matrix beyond 12k candidate pool.",
                "status": "hard_blocked_until_new_data",
                "next_artifact": "paper4_full_universe_loan_scenario_loss_matrix.parquet",
                "success_condition": "all eligible loans have scenario loss rows and exact solver certificate",
                "last_wave": "v49_v52",
                "execution_result": "restricted_pool_lp_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "immediate",
                "lane": "Online conformal",
                "executable_item": "v49 direct qhat repair search is implemented; live deployability remains blocked by historical-only validation and weak cells.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "external_or_future_period_online_holdout.csv",
                "success_condition": "unseen source-family holdout passes without width >0.95",
                "last_wave": "v49_v52",
                "execution_result": "direct_recalibration_search_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "short",
                "lane": "SPO/DLA",
                "executable_item": "v51 materialized lab books over the scenario matrix; formal differentiable SPO and Bellman exact DLA remain dependency/theory blocked.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_v53_dynamic_engine_books_replay.csv",
                "success_condition": "materialized book beats reference under dynamic engine, not only scenario matrix replay",
                "last_wave": "v49_v52",
                "execution_result": "scenario_books_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "short",
                "lane": "SICR/IFRS9",
                "executable_item": "v51 targeted SICR proxy search reduces arbitrary Stage 2 behavior but remains non-contractual.",
                "status": "data_blocked",
                "next_artifact": "servicing_panel_monthly_dpd_recoveries.parquet",
                "success_condition": "monthly servicing panel supports contractual IFRS9 inputs",
                "last_wave": "v49_v52",
                "execution_result": "targeted_proxy_search_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
        ]
    )
    combined = (
        pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    )
    prior_v54_mask = (
        combined.get("next_artifact", pd.Series("", index=combined.index))
        .astype(str)
        .eq("paper4_v54_dynamic_budget_capped_book_replay.csv")
    )
    combined.loc[prior_v54_mask, "status"] = "resolved"
    combined.loc[prior_v54_mask, "execution_result"] = "dynamic_budget_book_replay_completed"
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)
    return registry, claims, combined


def update_v52_notebook(statuses: dict[str, dict[str, Any]]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V49_V52_SELF_DIRECTED_LOOP_START -->",
            "",
            "## Wave v49-v52: Self-Directed Loop Checkpoint",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Move from repeated diagnostics into harder artifacts: a loan-scenario loss",
            "matrix, restricted-pool CVaR/source LPs, direct online qhat repair,",
            "materialized SPO/DLA-style books, targeted SICR proxy search, and an",
            "updated self-directed backlog.",
            "",
            "### Scripts",
            "",
            "- `scripts/papers/build_paper4_v49_self_directed_loop.py`",
            "",
            "### Results",
            "",
            f"- v49 loss matrix rows: `{statuses['v49'].get('loss_matrix_rows_v49')}`.",
            f"- v49 scenarios: `{statuses['v49'].get('loss_matrix_scenarios_v49')}`.",
            f"- v49 online repair rows: `{statuses['v49'].get('online_repair_rows_v49')}`.",
            f"- v50 CVaR LP frontier rows: `{statuses['v50'].get('cvar_frontier_rows_v50')}`.",
            f"- v50 successful restricted LPs: `{statuses['v50'].get('cvar_success_rows_v50')}`.",
            f"- v51 materialized book rows: `{statuses['v51'].get('policy_book_rows_v51')}`.",
            f"- v51 SICR targeted rows: `{statuses['v51'].get('sicr_targeted_rows_v51')}`.",
            f"- v52 registry rows: `{statuses['v52'].get('candidate_registry_rows_v52')}`.",
            "",
            "### Interpretation",
            "",
            "This checkpoint materially advances the lab because CVaR now has a persisted",
            "scenario matrix and exact restricted-pool LP artifacts. It also clarifies",
            "the remaining hard blockers: full-universe optimality needs an all-loan",
            "scenario matrix, live online deployability needs genuinely unseen source",
            "holdouts, and formal SPO/DLA claims need either validated dependencies or",
            "strong dynamic replay of materialized books.",
            "",
            "### Claim Impact",
            "",
            "- New allowed claim: restricted candidate-pool scenario matrix exists.",
            "- New allowed claim: restricted-pool CVaR/source LP is implemented.",
            "- Still prohibited: full-universe CVaR, live online deployability, formal",
            "  differentiable SPO+, Bellman exact DLA, contractual IFRS9, CATE policy",
            "  value, fair-lending legal claims, and Paper Estrella promotion.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep v49-v52 in the living notebook for now. Promote later only if the",
            "restricted LP or materialized books survive dynamic validation strongly",
            "enough to become an official Paper 4 result.",
            "",
            "<!-- V49_V52_SELF_DIRECTED_LOOP_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V49_V52_SELF_DIRECTED_LOOP_START -->"
        end = "<!-- V49_V52_SELF_DIRECTED_LOOP_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v52() -> dict[str, Any]:
    start = datetime.now(UTC)
    registry, claims, backlog = build_v52_registry_docs()
    statuses = {
        "v49": read_json("paper4_v49_status.json"),
        "v50": read_json("paper4_v50_status.json"),
        "v51": read_json("paper4_v51_status.json"),
    }
    status = {
        "schema_version": "2026-05-15.52",
        "generated_at_utc": now(),
        "phase": "v52_registry_backlog_claims_self_directed",
        "candidate_registry_rows_v52": int(len(registry)),
        "claim_matrix_rows_v52": int(len(claims)),
        "living_backlog_rows_v52": int(len(backlog)),
        "official_quarto_page_count": len(registered_paper4_pages()),
        "quarto_compact_guardrail_pass": len(registered_paper4_pages()) <= 12,
        "paper1_promotion_allowed_v52": False,
        "paper4_working_champion_changed_v52": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v52 updates self-directed registry/backlog; no official Quarto expansion or promotion",
    }
    write_json(STATUS_DIR / "paper4_v52_status.json", status)
    write_json(
        STATUS_DIR / "paper4_v52_working_champion.json",
        {
            "schema_version": "2026-05-15.52",
            "generated_at_utc": now(),
            "paper4_working_champion": "paper1_economic_champion",
            "working_champion_decision": "retained_reference_after_v49_v52_no_dynamic_validated_replacement",
            "paper4_working_only": True,
            "paper1_promotion_allowed": False,
            "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
            "claim_boundary": "Paper 4 lab/working champion only; no Paper Estrella promotion",
        },
    )
    statuses["v52"] = status
    update_v52_notebook(statuses)
    return status


def build_v53_expected_loss_repair() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    matrix = read_parquet("paper4_v49_loan_scenario_loss_matrix.parquet")
    pool = _prepare_candidate_pool()
    if matrix.empty or pool.empty:
        empty = pd.DataFrame()
        empty.to_parquet(TABLE_DIR / "paper4_v53_expected_loss_matrix.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v53_expected_loss_frontier.csv", empty)
        write_csv(TABLE_DIR / "paper4_v53_binary_vs_expected_diagnostic.csv", empty)
        write_csv(TABLE_DIR / "paper4_v53_expected_loss_certificate.csv", empty)
        empty.to_parquet(TABLE_DIR / "paper4_v53_expected_loss_allocations.parquet", index=False)
        return empty, empty, empty, empty

    p_small = pool[
        [
            "loan_index_v49",
            "loan_id",
            "loan_amnt",
            "base_return_vec",
            "lgd",
            "qhat_v4",
            "weak_source_proxy",
            "grade",
            "period",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
        ]
    ].copy()
    repaired = matrix.merge(p_small, on=["loan_index_v49", "loan_id", "loan_amnt"], how="left")
    repaired["lgd"] = pd.to_numeric(repaired["lgd"], errors="coerce").fillna(0.45).clip(0.05, 1.0)
    repaired["default_prob_v49"] = (
        pd.to_numeric(repaired["default_prob_v49"], errors="coerce").fillna(0).clip(0, 0.95)
    )
    repaired["base_return_vec"] = pd.to_numeric(
        repaired["base_return_vec"], errors="coerce"
    ).fillna(0)
    repaired["expected_loss_amount_v53"] = (
        repaired["loan_amnt"].astype(float) * repaired["lgd"] * repaired["default_prob_v49"]
    )
    repaired["binary_loss_amount_v53"] = pd.to_numeric(
        repaired["loss_amount_v49"], errors="coerce"
    ).fillna(0)
    repaired["hybrid_loss_amount_v53"] = (
        0.35 * repaired["binary_loss_amount_v53"] + 0.65 * repaired["expected_loss_amount_v53"]
    )
    repaired["expected_prepay_drag_v53"] = (
        repaired["loan_amnt"].astype(float) * 0.015 * (1 - repaired["default_prob_v49"])
    )
    repaired["expected_return_amount_v53"] = (
        repaired["base_return_vec"]
        - repaired["expected_loss_amount_v53"]
        - repaired["expected_prepay_drag_v53"]
    )
    repaired["hybrid_return_amount_v53"] = (
        repaired["base_return_vec"]
        - repaired["hybrid_loss_amount_v53"]
        - repaired["expected_prepay_drag_v53"]
    )
    repaired["loss_model_v53"] = "hybrid_binary_expected_internal_calibration"
    repaired["claim_boundary_v53"] = (
        "expected/hybrid scenario loss repair for optimization robustness; "
        "internal calibration only, not external forecast"
    )
    keep_cols = [
        "scenario_id",
        "loan_index_v49",
        "loan_id",
        "issue_month",
        "macro_regime_v15",
        "path_family_v19",
        "default_prob_v49",
        "default_event_v49",
        "loan_amnt",
        "expected_loss_amount_v53",
        "binary_loss_amount_v53",
        "hybrid_loss_amount_v53",
        "expected_return_amount_v53",
        "hybrid_return_amount_v53",
        "loss_model_v53",
        "claim_boundary_v53",
    ]
    repaired[keep_cols].to_parquet(
        TABLE_DIR / "paper4_v53_expected_loss_matrix.parquet", index=False
    )

    summary = (
        repaired.groupby(["scenario_id", "macro_regime_v15", "path_family_v19"], dropna=False)
        .agg(
            n_loans=("loan_id", "nunique"),
            mean_binary_loss=("binary_loss_amount_v53", "mean"),
            mean_expected_loss=("expected_loss_amount_v53", "mean"),
            mean_hybrid_loss=("hybrid_loss_amount_v53", "mean"),
            p95_hybrid_loss=(
                "hybrid_loss_amount_v53",
                lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95),
            ),
            mean_hybrid_return=("hybrid_return_amount_v53", "mean"),
        )
        .reset_index()
    )
    summary["claim_boundary_v53"] = repaired["claim_boundary_v53"].iloc[0]
    write_csv(TABLE_DIR / "paper4_v53_expected_loss_summary.csv", summary)

    p, returns, amounts, loss_mat, _ = _matrix_for_solver_columns(
        repaired,
        p_small,
        top_n=3000,
        loss_col="hybrid_loss_amount_v53",
        return_col="hybrid_return_amount_v53",
        score_col="solver_score_v53",
    )
    regimes = [
        ("strict", 55_000.0, 85_000.0, 1.00),
        ("committee", 75_000.0, 95_000.0, 1.25),
        ("relaxed", 105_000.0, 105_000.0, 1.60),
        ("tail_first", 60_000.0, 65_000.0, 1.60),
    ]
    frontier_rows = []
    allocs = []
    certs = []
    for regime, cvar_cap, return_floor, cap_mult in regimes:
        status, allocation, cert = _solve_cvar_lp_tagged(
            p,
            returns,
            amounts,
            loss_mat,
            regime,
            cvar_cap,
            return_floor,
            cap_mult,
            tag="v53",
            output_stem="paper4_v53_cvar_expected_loss",
        )
        frontier_rows.append(status)
        if not allocation.empty:
            allocs.append(allocation)
        if not cert.empty:
            certs.append(cert)
    frontier = pd.DataFrame(frontier_rows)
    allocations = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    certificate = pd.concat(certs, ignore_index=True) if certs else pd.DataFrame()
    if not frontier.empty:
        frontier["return_norm_v53"] = normalize(
            frontier.get("objective_return_v53", pd.Series(dtype=float))
        )
        frontier["tail_norm_v53"] = normalize(
            frontier.get("scenario_loss_cvar90_v53", pd.Series(dtype=float)), higher_is_better=False
        )
        frontier["frontier_score_v53"] = (
            0.55 * frontier["return_norm_v53"] + 0.45 * frontier["tail_norm_v53"]
        )
        frontier["non_dominated_restricted_v53"] = frontier["solver_success_v53"].astype(bool) & (
            frontier["frontier_score_v53"] >= frontier["frontier_score_v53"].median()
        )
    write_csv(TABLE_DIR / "paper4_v53_expected_loss_frontier.csv", frontier)
    write_csv(TABLE_DIR / "paper4_v53_expected_loss_certificate.csv", certificate)
    allocations.to_parquet(TABLE_DIR / "paper4_v53_expected_loss_allocations.parquet", index=False)

    v50 = read_csv("paper4_v50_cvar_source_lp_frontier.csv")
    diagnostics = []
    if not v50.empty:
        diagnostics.append(
            {
                "diagnostic_id": "v50_binary_loss_zero_tail",
                "source_artifact": "paper4_v50_cvar_source_lp_frontier.csv",
                "rows": int(len(v50)),
                "min_cvar": float(
                    pd.to_numeric(v50["scenario_loss_cvar90_v50"], errors="coerce").min()
                ),
                "max_cvar": float(
                    pd.to_numeric(v50["scenario_loss_cvar90_v50"], errors="coerce").max()
                ),
                "interpretation": (
                    "v50 LP can select loans with no realized binary defaults in 128 paths; "
                    "therefore zero CVaR is a simulation artifact, not a robust tail result"
                ),
                "claim_impact": "v50 kept as negative diagnostic; v53 expected/hybrid repair is the current runnable CVaR evidence",
            }
        )
    if not frontier.empty:
        diagnostics.append(
            {
                "diagnostic_id": "v53_expected_loss_positive_tail",
                "source_artifact": "paper4_v53_expected_loss_frontier.csv",
                "rows": int(len(frontier)),
                "min_cvar": float(
                    pd.to_numeric(frontier["scenario_loss_cvar90_v53"], errors="coerce").min()
                ),
                "max_cvar": float(
                    pd.to_numeric(frontier["scenario_loss_cvar90_v53"], errors="coerce").max()
                ),
                "interpretation": "v53 uses expected/hybrid loss so tail metrics cannot be zeroed by finite binary default sparsity",
                "claim_impact": "restricted-pool CVaR evidence is more defensible but still not full-universe exact optimality",
            }
        )
    diagnostic_df = pd.DataFrame(diagnostics)
    write_csv(TABLE_DIR / "paper4_v53_binary_vs_expected_diagnostic.csv", diagnostic_df)
    return repaired[keep_cols], frontier, allocations, diagnostic_df


def build_v53_budget_capped_books(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_candidate_pool()
    if pool.empty or matrix.empty:
        empty = pd.DataFrame()
        empty.to_parquet(TABLE_DIR / "paper4_v53_budget_capped_policy_books.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v53_budget_capped_policy_replay.csv", empty)
        write_csv(TABLE_DIR / "paper4_v53_champion_decomposition_cases.csv", empty)
        return empty, empty, empty
    exp = (
        matrix.groupby("loan_index_v49", dropna=False)
        .agg(
            expected_loss_v53=("hybrid_loss_amount_v53", "mean"),
            expected_return_v53=("hybrid_return_amount_v53", "mean"),
        )
        .reset_index()
    )
    p = pool.merge(exp, on="loan_index_v49", how="left")
    p["v53_spo_return_score"] = p["expected_return_v53"] - 0.25 * p["expected_loss_v53"]
    p["v53_tail_guard_score"] = (
        p["expected_return_v53"]
        - 1.10 * p["expected_loss_v53"]
        - 2500 * p["qhat_v4"]
        - 2500 * p["weak_source_proxy"]
    )
    p["v53_source_guard_score"] = (
        p["expected_return_v53"]
        - 0.85 * p["expected_loss_v53"]
        - 5000 * p["weak_source_proxy"]
        - 1600 * p["qhat_v4"]
    )
    score_specs = {
        "v53_spo_expected_return_book": "v53_spo_return_score",
        "v53_dla_expected_tail_guard_book": "v53_tail_guard_score",
        "v53_dla_source_guard_book": "v53_source_guard_score",
    }
    book_rows = []
    for policy_id, score_col in score_specs.items():
        g = p.sort_values(score_col, ascending=False).copy()
        chosen = g.loc[g["loan_amnt"].cumsum().le(BUDGET)].copy()
        if chosen.empty and not g.empty:
            chosen = g.head(1).copy()
        chosen["policy_id"] = policy_id
        chosen["selection_score_v53"] = chosen[score_col]
        chosen["budget_capped_v53"] = chosen["loan_amnt"].sum() <= BUDGET + 1e-6
        chosen["book_claim_boundary_v53"] = (
            "budget-capped materialized book from expected/hybrid scenario matrix; "
            "not Bellman exact and not differentiable SPO"
        )
        book_rows.append(chosen)
    books = pd.concat(book_rows, ignore_index=True) if book_rows else pd.DataFrame()
    books.to_parquet(TABLE_DIR / "paper4_v53_budget_capped_policy_books.parquet", index=False)

    replay_rows = []
    for policy_id, book in books.groupby("policy_id", dropna=False):
        selected = set(book["loan_index_v49"].astype(int))
        m = matrix.loc[matrix["loan_index_v49"].astype(int).isin(selected)].copy()
        replay = (
            m.groupby("scenario_id", dropna=False)
            .agg(
                scenario_loss=("hybrid_loss_amount_v53", "sum"),
                scenario_return=("hybrid_return_amount_v53", "sum"),
                funded_exposure=("loan_amnt", "sum"),
                binary_defaults=("default_event_v49", "sum"),
            )
            .reset_index()
        )
        replay_rows.append(
            {
                "policy_id": policy_id,
                "n_scenarios": int(len(replay)),
                "n_loans": int(book["loan_id"].nunique()),
                "funded_exposure_v53": float(book["loan_amnt"].sum()),
                "budget_capped_v53": bool(book["loan_amnt"].sum() <= BUDGET + 1e-6),
                "mean_scenario_return_v53": float(replay["scenario_return"].mean()),
                "p05_scenario_return_v53": float(replay["scenario_return"].quantile(0.05)),
                "p95_scenario_loss_v53": float(replay["scenario_loss"].quantile(0.95)),
                "mean_binary_defaults_v53": float(replay["binary_defaults"].mean()),
                "claim_boundary_v53": "common-path expected/hybrid replay; no production forecast claim",
            }
        )
    replay_df = pd.DataFrame(replay_rows)
    write_csv(TABLE_DIR / "paper4_v53_budget_capped_policy_replay.csv", replay_df)

    top_cases = books.sort_values("expected_return_v53", ascending=False).head(60).copy()
    top_cases["case_study_scope_v53"] = (
        "budget-capped expected/hybrid books; use for economic interpretation, not official promotion"
    )
    write_csv(TABLE_DIR / "paper4_v53_champion_decomposition_cases.csv", top_cases)
    return books, replay_df, top_cases


def update_v53_registry_and_claims(
    frontier: pd.DataFrame, replay: pd.DataFrame, diagnostic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    v52 = read_csv("paper4_v52_candidate_registry.csv")
    if not v52.empty:
        for _, row in v52.head(12).iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": row.get("candidate_family"),
                    "evidence_source_v53": row.get("evidence_source_v52", "v52_registry"),
                    "wealth_score_v53": safe_num(row.get("wealth_norm_v52"), 0),
                    "tail_score_v53": safe_num(row.get("tail_norm_v52"), 0),
                    "auditability_score_v53": safe_num(row.get("audit_norm_v52"), 0.5),
                    "claim_safety_v53": bool(row.get("claim_safety_v52", True)),
                }
            )
    if not frontier.empty:
        for _, row in frontier.loc[frontier["solver_success_v53"].astype(bool)].iterrows():
            rows.append(
                {
                    "policy_id": f"v53_cvar_expected_loss_{row.get('regime_v53')}",
                    "candidate_family": "restricted_cvar_expected_loss_lp",
                    "evidence_source_v53": "v53_expected_loss_frontier",
                    "wealth_score_v53": safe_num(row.get("return_norm_v53"), 0),
                    "tail_score_v53": safe_num(row.get("tail_norm_v53"), 0),
                    "auditability_score_v53": 0.80,
                    "claim_safety_v53": True,
                }
            )
    if not replay.empty:
        for _, row in replay.iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "candidate_family": "budget_capped_expected_loss_book",
                    "evidence_source_v53": "v53_budget_capped_replay",
                    "wealth_score_v53": safe_num(row.get("mean_scenario_return_v53"), 0),
                    "tail_score_v53": -safe_num(row.get("p95_scenario_loss_v53"), 0),
                    "auditability_score_v53": 0.73,
                    "claim_safety_v53": bool(row.get("budget_capped_v53", False)),
                }
            )
    registry = pd.DataFrame(rows).dropna(subset=["policy_id"])
    if not registry.empty:
        registry["wealth_norm_v53"] = normalize(registry["wealth_score_v53"])
        registry["tail_norm_v53"] = normalize(registry["tail_score_v53"])
        registry["audit_norm_v53"] = normalize(registry["auditability_score_v53"])
        registry["full_governance_score_v53"] = (
            0.35 * registry["wealth_norm_v53"]
            + 0.30 * registry["tail_norm_v53"]
            + 0.20 * registry["audit_norm_v53"]
            + 0.15 * registry["claim_safety_v53"].astype(float)
        )
        registry["decision_v53"] = np.select(
            [
                registry["policy_id"].eq("paper1_economic_champion"),
                registry["candidate_family"].eq("restricted_cvar_expected_loss_lp"),
                registry["candidate_family"].eq("budget_capped_expected_loss_book"),
            ],
            ["retain_reference", "serious_solver_challenger", "serious_lab_book_challenger"],
            default="review_or_park",
        )
        registry["paper1_promotion_allowed_v53"] = False
        registry["paper4_working_champion_allowed_v53"] = False
        registry["claim_boundary_v53"] = (
            "Paper 4 lab registry repair; no final promotion or Paper Estrella change"
        )
        registry = registry.sort_values("full_governance_score_v53", ascending=False)
    write_csv(TABLE_DIR / "paper4_v53_candidate_registry_repair.csv", registry)

    claims = pd.DataFrame(
        [
            {
                "claim_id": "v53_expected_hybrid_loss_matrix",
                "allowed": True,
                "artifact": "paper4_v53_expected_loss_matrix.parquet",
                "boundary": "restricted candidate pool and internal scenarios only",
            },
            {
                "claim_id": "v53_v50_zero_cvar_artifact_identified",
                "allowed": True,
                "artifact": "paper4_v53_binary_vs_expected_diagnostic.csv",
                "boundary": "negative diagnostic; v50 binary-zero tail is not a robust CVaR result",
            },
            {
                "claim_id": "v53_restricted_expected_loss_cvar_lp",
                "allowed": True,
                "artifact": "paper4_v53_expected_loss_frontier.csv",
                "boundary": "exact LP over restricted expected/hybrid loss pool; not full universe",
            },
            {
                "claim_id": "v53_exact_full_universe_cvar",
                "allowed": False,
                "artifact": "paper4_v53_expected_loss_frontier.csv",
                "boundary": "full universe remains blocked by lack of all-loan scenario matrix with optimization features",
            },
            {
                "claim_id": "v53_bellman_exact_dla",
                "allowed": False,
                "artifact": "paper4_v53_budget_capped_policy_books.parquet",
                "boundary": "budget-capped books only, not Bellman exact DLA",
            },
            {
                "claim_id": "v53_contractual_ifrs9",
                "allowed": False,
                "artifact": "paper4_v51_sicr_targeted_recalibration.csv",
                "boundary": "SICR proxy only",
            },
        ]
    )
    if not diagnostic.empty:
        claims["diagnostic_rows_v53"] = int(len(diagnostic))
    write_csv(TABLE_DIR / "paper4_v53_claim_matrix_delta.csv", claims)
    return registry, claims


def update_v53_backlog_and_notebook(status: dict[str, Any]) -> None:
    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": "v53 repairs v50 binary-zero CVaR with expected/hybrid losses; next step is all-loan/full-universe feature matrix if source features can be joined safely.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_full_universe_expected_loss_matrix.parquet",
                "success_condition": "all eligible loans have scenario loss rows and exact or column-generation certificate",
                "last_wave": "v53",
                "execution_result": "expected_loss_cvar_repair_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "short",
                "lane": "SPO/DLA",
                "executable_item": "v53 budget-capped books repair v51 over-budget materialization; next step is dynamic engine replay against current working champion.",
                "status": "runnable",
                "next_artifact": "paper4_v54_dynamic_budget_capped_book_replay.csv",
                "success_condition": "budget-capped book beats reference under common dynamic paths",
                "last_wave": "v53",
                "execution_result": "budget_capped_books_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
        ]
    )
    combined = (
        pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    )
    prior_v54_mask = (
        combined.get("next_artifact", pd.Series("", index=combined.index))
        .astype(str)
        .eq("paper4_v54_dynamic_budget_capped_book_replay.csv")
    )
    combined.loc[prior_v54_mask, "status"] = "resolved"
    combined.loc[prior_v54_mask, "execution_result"] = "dynamic_budget_book_replay_completed"
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)

    section = "\n".join(
        [
            "",
            "<!-- V53_EXPECTED_LOSS_REPAIR_START -->",
            "",
            "## Wave v53: Expected-Loss CVaR Repair",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Repair the v50 CVaR artifact where binary default sparsity over 128",
            "internal paths allowed the solver to report zero portfolio tail loss.",
            "",
            "### Results",
            "",
            f"- v53 expected/hybrid matrix rows: `{status.get('expected_loss_matrix_rows_v53')}`.",
            f"- v53 CVaR frontier rows: `{status.get('expected_loss_frontier_rows_v53')}`.",
            f"- v53 successful LP rows: `{status.get('expected_loss_frontier_success_rows_v53')}`.",
            f"- v53 budget-capped policy books: `{status.get('budget_capped_policy_count_v53')}`.",
            f"- v53 minimum CVaR90: `{status.get('min_cvar90_v53')}`.",
            "",
            "### Interpretation",
            "",
            "The v50 zero-loss tail was not a credible CVaR finding. It was a finite",
            "binary-simulation artifact. v53 converts the same internal paths into an",
            "expected/hybrid loss matrix, reruns the restricted-pool LP, and creates",
            "budget-capped lab books so the next dynamic replay is not contaminated by",
            "over-budget monthly materialization.",
            "",
            "### Claim Impact",
            "",
            "- Allowed: restricted-pool expected/hybrid CVaR LP evidence.",
            "- Allowed: v50 zero-CVaR is documented as a negative diagnostic.",
            "- Still prohibited: exact full-universe CVaR, Bellman exact DLA, formal",
            "  differentiable SPO+, contractual IFRS9, fair-lending legal claims, CATE",
            "  policy value, final Paper 4 promotion, and Paper Estrella promotion.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep v53 in the living notebook. It should influence the official chapter",
            "only after the repaired CVaR books survive dynamic replay.",
            "",
            "<!-- V53_EXPECTED_LOSS_REPAIR_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V53_EXPECTED_LOSS_REPAIR_START -->"
        end = "<!-- V53_EXPECTED_LOSS_REPAIR_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v53() -> dict[str, Any]:
    start = datetime.now(UTC)
    matrix, frontier, allocations, diagnostic = build_v53_expected_loss_repair()
    books, replay, cases = build_v53_budget_capped_books(matrix)
    registry, claims = update_v53_registry_and_claims(frontier, replay, diagnostic)
    min_cvar = (
        float(pd.to_numeric(frontier.get("scenario_loss_cvar90_v53"), errors="coerce").min())
        if not frontier.empty
        else np.nan
    )
    status = {
        "schema_version": "2026-05-15.53",
        "generated_at_utc": now(),
        "phase": "v53_expected_loss_cvar_repair_and_budget_books",
        "expected_loss_matrix_rows_v53": int(len(matrix)),
        "expected_loss_matrix_loans_v53": int(matrix["loan_id"].nunique())
        if not matrix.empty
        else 0,
        "expected_loss_frontier_rows_v53": int(len(frontier)),
        "expected_loss_frontier_success_rows_v53": int(frontier["solver_success_v53"].sum())
        if not frontier.empty
        else 0,
        "expected_loss_allocation_rows_v53": int(len(allocations)),
        "diagnostic_rows_v53": int(len(diagnostic)),
        "budget_capped_policy_count_v53": int(replay["policy_id"].nunique())
        if not replay.empty
        else 0,
        "budget_capped_books_all_within_budget_v53": bool(
            books.groupby("policy_id")["loan_amnt"].sum().le(BUDGET + 1e-6).all()
        )
        if not books.empty
        else False,
        "candidate_registry_rows_v53": int(len(registry)),
        "claim_matrix_rows_v53": int(len(claims)),
        "min_cvar90_v53": min_cvar,
        "v50_zero_cvar_artifact_repaired_v53": bool(min_cvar > 0)
        if not pd.isna(min_cvar)
        else False,
        "exact_full_universe_cvar_claim_allowed": False,
        "formal_differentiable_spo_claim_allowed": False,
        "bellman_exact_claim_allowed": False,
        "contractual_ifrs9_claim_allowed": False,
        "paper1_promotion_allowed_v53": False,
        "paper4_working_champion_changed_v53": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v53 repairs restricted-pool CVaR loss model and budget-capped books; no official promotion",
    }
    write_json(STATUS_DIR / "paper4_v53_status.json", status)
    write_json(
        STATUS_DIR / "paper4_v53_working_champion.json",
        {
            "schema_version": "2026-05-15.53",
            "generated_at_utc": now(),
            "paper4_working_champion": "paper1_economic_champion",
            "working_champion_decision": "retained_reference_after_v53_repair_pending_dynamic_replay",
            "paper4_working_only": True,
            "paper1_promotion_allowed": False,
            "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
            "claim_boundary": "Paper 4 lab/working champion only; no Paper Estrella promotion",
        },
    )
    update_v53_backlog_and_notebook(status)
    return status


def _reference_proxy_book() -> pd.DataFrame:
    pool = _prepare_candidate_pool()
    if pool.empty:
        return pool
    score = (
        pd.to_numeric(pool["base_return_vec"], errors="coerce").fillna(0)
        - 0.75
        * pd.to_numeric(pool["loan_amnt"], errors="coerce").fillna(0)
        * pd.to_numeric(pool["pd_high_alpha01"], errors="coerce").fillna(0)
        * pd.to_numeric(pool["lgd"], errors="coerce").fillna(0.45)
        - 1800 * pd.to_numeric(pool["qhat_v4"], errors="coerce").fillna(0)
        - 1800 * pd.to_numeric(pool["weak_source_proxy"], errors="coerce").fillna(0)
    )
    p = pool.assign(selection_score_v54=score).sort_values("selection_score_v54", ascending=False)
    chosen = p.loc[p["loan_amnt"].cumsum().le(BUDGET)].copy()
    chosen["policy_id"] = "v54_paper1_reference_proxy_book"
    chosen["loan_amnt_dynamic_v54"] = chosen["loan_amnt"]
    chosen["allocation_fraction_v54"] = 1.0
    chosen["book_claim_boundary_v54"] = (
        "Paper Estrella-style restricted-pool proxy book; not the official full-universe Paper Estrella allocation"
    )
    return chosen


def _v53_cvar_allocation_books() -> pd.DataFrame:
    alloc = read_parquet("paper4_v53_expected_loss_allocations.parquet")
    if alloc.empty:
        return alloc
    frac = pd.to_numeric(alloc.get("allocation_fraction_v53"), errors="coerce").fillna(0).clip(0, 1)
    out = alloc.loc[frac > 1e-6].copy()
    out["allocation_fraction_v54"] = pd.to_numeric(
        out["allocation_fraction_v53"], errors="coerce"
    ).fillna(0)
    pool = _prepare_candidate_pool()
    join_cols = [
        "loan_index_v49",
        "issue_month",
        "base_return_vec",
        "qhat_v4",
        "weak_source_proxy",
        "grade",
        "period",
        "score_decile",
        "state_top20",
        "income_band",
        "dti_band",
    ]
    available = [c for c in join_cols if c in pool.columns and c not in out.columns]
    if "loan_index_v49" in pool.columns and available:
        out = out.merge(pool[["loan_index_v49", *available]], on="loan_index_v49", how="left")
    out["policy_id"] = out["policy_id_v53"].astype(str)
    out["loan_amnt_dynamic_v54"] = out["loan_amnt"] * out["allocation_fraction_v54"]
    out["selection_score_v54"] = pd.to_numeric(out.get("expected_return"), errors="coerce").fillna(
        0
    )
    out["book_claim_boundary_v54"] = (
        "fractional restricted-pool CVaR expected-loss LP allocation replay; not full-universe optimality"
    )
    return out


def _v53_budget_books_for_dynamic() -> pd.DataFrame:
    books = read_parquet("paper4_v53_budget_capped_policy_books.parquet")
    if books.empty:
        return books
    out = books.copy()
    out["loan_amnt_dynamic_v54"] = out["loan_amnt"]
    out["allocation_fraction_v54"] = 1.0
    out["book_claim_boundary_v54"] = out.get(
        "book_claim_boundary_v53",
        "budget-capped expected/hybrid scenario book; not Bellman exact",
    )
    return out


def build_v54_dynamic_budget_capped_replay() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = read_parquet("paper4_v53_expected_loss_matrix.parquet")
    if matrix.empty:
        empty = pd.DataFrame()
        empty.to_parquet(
            TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_trace.parquet", index=False
        )
        write_csv(TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_summary.csv", empty)
        write_csv(TABLE_DIR / "paper4_v54_dynamic_champion_decision_memo.csv", empty)
        return empty, empty, empty

    source = pd.concat(
        [
            _reference_proxy_book(),
            _v53_cvar_allocation_books(),
            _v53_budget_books_for_dynamic(),
        ],
        ignore_index=True,
        sort=False,
    )
    if source.empty:
        empty = pd.DataFrame()
        empty.to_parquet(
            TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_trace.parquet", index=False
        )
        write_csv(TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_summary.csv", empty)
        write_csv(TABLE_DIR / "paper4_v54_dynamic_champion_decision_memo.csv", empty)
        return empty, empty, empty

    raw_pool = read_parquet("paper4_challenger_local_candidate_pool.parquet")
    extra_cols = [
        "id",
        "installment",
        "term",
        "loan_status",
        "issue_d",
        "recoveries",
        "total_pymnt",
        "total_rec_prncp",
        "last_pymnt_d",
    ]
    extra = (
        raw_pool[[c for c in extra_cols if c in raw_pool.columns]].copy()
        if not raw_pool.empty
        else pd.DataFrame()
    )
    if not extra.empty:
        extra["loan_id"] = extra["id"].astype(str)
        source["loan_id"] = source["loan_id"].astype(str)
        source = source.merge(extra.drop(columns=["id"], errors="ignore"), on="loan_id", how="left")
    source["issue_month"] = (
        pd.to_datetime(source["issue_month"], errors="coerce").dt.to_period("M").astype(str)
    )
    installment_col = next(
        (c for c in ["installment", "installment_x", "installment_y"] if c in source), None
    )
    if installment_col:
        installment_values = pd.to_numeric(source[installment_col], errors="coerce")
    else:
        installment_values = pd.Series(np.nan, index=source.index)
    source["installment"] = installment_values.fillna(
        pd.to_numeric(source["loan_amnt"], errors="coerce").fillna(0) / 36
    )
    term_col = next((c for c in ["term", "term_x", "term_y"] if c in source), None)
    term_source = source[term_col] if term_col else pd.Series("", index=source.index)
    term_numeric = term_source.astype(str).str.extract(r"(\d+)")[0]
    source["term_months_v54"] = pd.to_numeric(term_numeric, errors="coerce").fillna(36).clip(12, 84)
    source["loan_amnt_dynamic_v54"] = pd.to_numeric(
        source["loan_amnt_dynamic_v54"], errors="coerce"
    ).fillna(0)
    source["installment_dynamic_v54"] = source["installment"] * (
        source["loan_amnt_dynamic_v54"]
        / pd.to_numeric(source["loan_amnt"], errors="coerce").replace(0, np.nan)
    ).fillna(1.0)

    months = sorted(matrix["issue_month"].dropna().astype(str).unique().tolist())[:24]
    month_to_idx = {month: i for i, month in enumerate(months)}
    scenario_ids = sorted(matrix["scenario_id"].drop_duplicates().astype(int).tolist())
    matrix_small = matrix[
        [
            "scenario_id",
            "loan_id",
            "hybrid_loss_amount_v53",
            "hybrid_return_amount_v53",
            "default_event_v49",
        ]
    ].copy()
    matrix_small["loan_id"] = matrix_small["loan_id"].astype(str)
    source["loan_id"] = source["loan_id"].astype(str)
    source["funding_month_idx_v54"] = (
        source["issue_month"].map(month_to_idx).fillna(0).astype(int).clip(0, len(months) - 1)
    )

    trace_rows = []
    for policy_id, book in source.groupby("policy_id", dropna=False):
        book_cols = [
            "loan_id",
            "loan_amnt_dynamic_v54",
            "installment_dynamic_v54",
            "term_months_v54",
            "funding_month_idx_v54",
            "allocation_fraction_v54",
        ]
        book_base = book[book_cols].drop_duplicates("loan_id").copy()
        book_base["allocation_fraction_v54"] = pd.to_numeric(
            book_base["allocation_fraction_v54"], errors="coerce"
        ).fillna(1.0)
        for scenario_id in scenario_ids:
            scenario = matrix_small.loc[matrix_small["scenario_id"].eq(scenario_id)].drop(
                columns=["scenario_id"]
            )
            events = book_base.merge(scenario, on="loan_id", how="left")
            events["hybrid_loss_amount_v53"] = pd.to_numeric(
                events["hybrid_loss_amount_v53"], errors="coerce"
            ).fillna(0)
            events["default_event_v49"] = events["default_event_v49"].fillna(False).astype(bool)
            events["funding_month_idx_v54"] = events["funding_month_idx_v54"].clip(
                0, len(months) - 1
            )
            timing = np.array(
                [
                    int(
                        stable_unit(scenario_id, loan_id, "v54_default_timing")
                        * max(1, min(int(term), len(months)))
                    )
                    for loan_id, term in zip(
                        events["loan_id"], events["term_months_v54"], strict=False
                    )
                ],
                dtype=int,
            )
            events["default_month_idx_v54"] = (
                events["funding_month_idx_v54"].to_numpy() + timing
            ).clip(0, len(months) - 1)
            events.loc[~events["default_event_v49"], "default_month_idx_v54"] = len(months) + 99
            events["loss_dynamic_v54"] = (
                events["hybrid_loss_amount_v53"] * events["allocation_fraction_v54"]
            )
            events["recovery_dynamic_v54"] = [
                0.12 * loss * (0.75 + 0.5 * stable_unit(scenario_id, loan_id, "v54_recovery"))
                for loss, loan_id in zip(
                    events["loss_dynamic_v54"], events["loan_id"], strict=False
                )
            ]

            funded_by_month = np.zeros(len(months))
            repayment_by_month = np.zeros(len(months))
            loss_by_month = np.zeros(len(months))
            recovery_by_month = np.zeros(len(months))
            default_by_month = np.zeros(len(months))
            for _, loan in events.iterrows():
                fund_idx = int(loan["funding_month_idx_v54"])
                amount = float(loan["loan_amnt_dynamic_v54"])
                funded_by_month[fund_idx] += amount
                end_idx = min(len(months), fund_idx + int(loan["term_months_v54"]))
                default_idx = int(loan["default_month_idx_v54"])
                repay_end = min(end_idx, default_idx if default_idx < len(months) else end_idx)
                if repay_end > fund_idx:
                    repayment_by_month[fund_idx:repay_end] += float(loan["installment_dynamic_v54"])
                if default_idx < len(months):
                    loss_by_month[default_idx] += float(loan["loss_dynamic_v54"])
                    recovery_by_month[default_idx] += float(loan["recovery_dynamic_v54"])
                    default_by_month[default_idx] += 1

            cumulative_funded = np.cumsum(funded_by_month)
            cumulative_repayments = np.cumsum(repayment_by_month)
            cumulative_losses = np.cumsum(loss_by_month)
            cumulative_recoveries = np.cumsum(recovery_by_month)
            outstanding_series = np.maximum(
                0.0, cumulative_funded - cumulative_repayments - cumulative_losses
            )
            cash_series = BUDGET - cumulative_funded + cumulative_repayments + cumulative_recoveries
            wealth_series = (
                cash_series + outstanding_series - cumulative_losses + cumulative_recoveries
            )
            cumulative_defaults = np.cumsum(default_by_month)
            cumulative_funded_count = [
                int((events["funding_month_idx_v54"] <= idx).sum()) for idx in range(len(months))
            ]
            for month_idx, month in enumerate(months):
                outstanding = float(outstanding_series[month_idx])
                trace_rows.append(
                    {
                        "policy_id": policy_id,
                        "scenario_id": scenario_id,
                        "month_index": month_idx,
                        "period": month,
                        "cash_v54": float(cash_series[month_idx]),
                        "outstanding_principal_v54": outstanding,
                        "funded_exposure_month_v54": float(funded_by_month[month_idx]),
                        "repayments_month_v54": float(repayment_by_month[month_idx]),
                        "defaults_month_v54": int(default_by_month[month_idx]),
                        "recoveries_month_v54": float(recovery_by_month[month_idx]),
                        "losses_month_v54": float(loss_by_month[month_idx]),
                        "cumulative_losses_v54": float(cumulative_losses[month_idx]),
                        "cumulative_recoveries_v54": float(cumulative_recoveries[month_idx]),
                        "cumulative_repayments_v54": float(cumulative_repayments[month_idx]),
                        "ecl_proxy_v54": outstanding * 0.45 * 0.12,
                        "wealth_v54": float(wealth_series[month_idx]),
                        "funded_count_v54": cumulative_funded_count[month_idx],
                        "defaulted_count_v54": int(cumulative_defaults[month_idx]),
                        "selection_uses_scenario_matrix_v54": True,
                        "live_no_leakage_claim_allowed_v54": False,
                        "claim_boundary_v54": (
                            "dynamic replay over pre-materialized lab books and internal paths; "
                            "not production no-leakage, not Bellman exact, not external forecast"
                        ),
                    }
                )
    trace = pd.DataFrame(trace_rows)
    trace.to_parquet(TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_trace.parquet", index=False)

    last = (
        trace.sort_values("month_index").groupby(["policy_id", "scenario_id"], dropna=False).tail(1)
    )
    summary = (
        last.groupby("policy_id", dropna=False)
        .agg(
            scenarios=("scenario_id", "nunique"),
            final_wealth_mean_v54=("wealth_v54", "mean"),
            final_wealth_p05_v54=(
                "wealth_v54",
                lambda s: pd.to_numeric(s, errors="coerce").quantile(0.05),
            ),
            final_loss_mean_v54=("cumulative_losses_v54", "mean"),
            final_loss_p95_v54=(
                "cumulative_losses_v54",
                lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95),
            ),
            defaults_mean_v54=("defaulted_count_v54", "mean"),
            ecl_proxy_mean_v54=("ecl_proxy_v54", "mean"),
            funded_count_mean_v54=("funded_count_v54", "mean"),
            live_no_leakage_claim_allowed_v54=("live_no_leakage_claim_allowed_v54", "max"),
        )
        .reset_index()
    )
    summary["wealth_norm_v54"] = normalize(summary["final_wealth_mean_v54"])
    summary["tail_norm_v54"] = normalize(summary["final_loss_p95_v54"], higher_is_better=False)
    summary["dynamic_score_v54"] = (
        0.55 * summary["wealth_norm_v54"] + 0.45 * summary["tail_norm_v54"]
    )
    summary["claim_boundary_v54"] = (
        "internal dynamic replay only; no Paper Estrella or final Paper 4 promotion"
    )
    write_csv(TABLE_DIR / "paper4_v54_dynamic_budget_capped_book_summary.csv", summary)

    best = summary.sort_values("dynamic_score_v54", ascending=False).head(1)
    memo_rows = []
    if not best.empty:
        winner = best.iloc[0]
        memo_rows.append(
            {
                "memo_id": "v54_dynamic_budget_book_decision",
                "best_policy_id": winner["policy_id"],
                "best_dynamic_score_v54": float(winner["dynamic_score_v54"]),
                "best_final_wealth_mean_v54": float(winner["final_wealth_mean_v54"]),
                "best_final_loss_p95_v54": float(winner["final_loss_p95_v54"]),
                "working_champion_change_allowed": False,
                "decision": "no_working_champion_change_until_compared_against_official_full_universe_reference",
                "interpretation": (
                    "v54 is useful for dynamic learning, but every book is restricted-pool "
                    "and pre-materialized from internal scenario evidence"
                ),
                "claim_boundary": "Paper 4 lab memo only",
            }
        )
    memo = pd.DataFrame(memo_rows)
    write_csv(TABLE_DIR / "paper4_v54_dynamic_champion_decision_memo.csv", memo)
    return trace, summary, memo


def update_v54_notebook_and_backlog(status: dict[str, Any]) -> None:
    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "short",
                "lane": "SPO/DLA",
                "executable_item": "v54 dynamic replay evaluates v53 budget-capped and CVaR books over monthly cash/outstanding/loss state.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_v55_dynamic_replay_against_official_reference_if_book_mapping_exists.csv",
                "success_condition": "official full-universe champion loan mapping is available for direct dynamic comparison",
                "last_wave": "v54",
                "execution_result": "dynamic_budget_book_replay_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            }
        ]
    )
    combined = (
        pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    )
    prior_v54_mask = (
        combined.get("next_artifact", pd.Series("", index=combined.index))
        .astype(str)
        .eq("paper4_v54_dynamic_budget_capped_book_replay.csv")
    )
    combined.loc[prior_v54_mask, "status"] = "resolved"
    combined.loc[prior_v54_mask, "execution_result"] = "dynamic_budget_book_replay_completed"
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)

    section = "\n".join(
        [
            "",
            "<!-- V54_DYNAMIC_REPLAY_START -->",
            "",
            "## Wave v54: Dynamic Replay Of Repaired Books",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Replay the repaired v53 budget-capped and CVaR books through monthly state",
            "variables instead of stopping at static book scores.",
            "",
            "### Results",
            "",
            f"- Dynamic trace rows: `{status.get('dynamic_trace_rows_v54')}`.",
            f"- Dynamic policies: `{status.get('dynamic_policy_count_v54')}`.",
            f"- Best replay policy: `{status.get('best_policy_id_v54')}`.",
            f"- Best mean final wealth: `{status.get('best_final_wealth_mean_v54')}`.",
            "",
            "### Interpretation",
            "",
            "v54 moves the repaired books into a sequential replay with cash, outstanding",
            "principal, repayments, defaults, recoveries, losses, ECL proxy and wealth.",
            "It remains a lab evaluation because the books are restricted-pool and selected",
            "from internal scenario evidence, but it closes the immediate v53 runnable step.",
            "",
            "### Claim Impact",
            "",
            "- Allowed: internal dynamic replay exists for repaired budget-capped books.",
            "- Still prohibited: Bellman exact DLA, production no-leakage deployment,",
            "  official Paper Estrella replacement, exact full-universe CVaR.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep v54 in the notebook. Promote only if later mapped against the official",
            "full-universe champion or repeated in an exact comparable universe.",
            "",
            "<!-- V54_DYNAMIC_REPLAY_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V54_DYNAMIC_REPLAY_START -->"
        end = "<!-- V54_DYNAMIC_REPLAY_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v54() -> dict[str, Any]:
    start = datetime.now(UTC)
    trace, summary, memo = build_v54_dynamic_budget_capped_replay()
    best = (
        summary.sort_values("dynamic_score_v54", ascending=False).head(1)
        if not summary.empty
        else pd.DataFrame()
    )
    status = {
        "schema_version": "2026-05-15.54",
        "generated_at_utc": now(),
        "phase": "v54_dynamic_budget_capped_book_replay",
        "dynamic_trace_rows_v54": int(len(trace)),
        "dynamic_summary_rows_v54": int(len(summary)),
        "dynamic_policy_count_v54": int(summary["policy_id"].nunique()) if not summary.empty else 0,
        "best_policy_id_v54": best["policy_id"].iloc[0] if not best.empty else "",
        "best_final_wealth_mean_v54": float(best["final_wealth_mean_v54"].iloc[0])
        if not best.empty
        else np.nan,
        "best_final_loss_p95_v54": float(best["final_loss_p95_v54"].iloc[0])
        if not best.empty
        else np.nan,
        "decision_memo_rows_v54": int(len(memo)),
        "live_no_leakage_claim_allowed": False,
        "bellman_exact_claim_allowed": False,
        "exact_full_universe_cvar_claim_allowed": False,
        "paper1_promotion_allowed_v54": False,
        "paper4_working_champion_changed_v54": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v54 dynamic replay over internal restricted-pool books; no official promotion",
    }
    write_json(STATUS_DIR / "paper4_v54_status.json", status)
    update_v54_notebook_and_backlog(status)
    return status


def build_all() -> dict[str, dict[str, Any]]:
    v49 = build_v49()
    v50 = build_v50()
    v51 = build_v51()
    v52 = build_v52()
    v53 = build_v53()
    v54 = build_v54()
    return {"v49": v49, "v50": v50, "v51": v51, "v52": v52, "v53": v53, "v54": v54}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["v49", "v50", "v51", "v52", "v53", "v54", "all"], default="all"
    )
    parser.add_argument("--paths", type=int, default=128)
    args = parser.parse_args()
    if args.phase == "v49":
        matrix, summary, pool = build_v49_loss_matrix(n_paths=args.paths)
        online, weak = build_v49_online_repair()
        result = {
            "v49": {
                "loss_matrix_rows_v49": int(len(matrix)),
                "loan_scenario_summary_rows_v49": int(len(summary)),
                "solver_candidate_pool_rows_v49": int(len(pool)),
                "online_repair_rows_v49": int(len(online)),
                "online_weak_cell_rows_v49": int(len(weak)),
            }
        }
        write_json(
            STATUS_DIR / "paper4_v49_status.json",
            {
                "schema_version": "2026-05-15.49",
                "generated_at_utc": now(),
                "phase": "v49_loss_matrix_online_repair",
                **result["v49"],
                "external_forecast_validation_claim_allowed": False,
                "strict_live_deployability_claim_allowed": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
            },
        )
    elif args.phase == "v50":
        result = {"v50": build_v50()}
    elif args.phase == "v51":
        result = {"v51": build_v51()}
    elif args.phase == "v52":
        result = {"v52": build_v52()}
    elif args.phase == "v53":
        result = {"v53": build_v53()}
    elif args.phase == "v54":
        result = {"v54": build_v54()}
    else:
        result = build_all()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
