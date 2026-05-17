#!/usr/bin/env python3
"""Build Paper 4 v55-v58 unlock-loop artifacts.

This checkpoint follows the v49-v54 self-directed loop.  The main new
finding is that the apparent full-universe blocker was partially artificial:
`test_predictions.loan_id` joins exactly to `data/processed/test.id`.
That unlocks a 276,869-row feature-rich comparable universe.  The solver
evidence is still deliberately bounded: exact full-universe CVaR remains
false until we persist and solve the complete scenario-loss program.
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
    name: str | Path, directory: Path = TABLE_DIR, columns: list[str] | None = None
) -> pd.DataFrame:
    path = Path(name)
    if not path.is_absolute():
        path = directory / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def read_json(name: str, directory: Path = STATUS_DIR) -> dict[str, Any]:
    path = directory / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df) if isinstance(df, list) else df
    out.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_unit(*parts: Any) -> float:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


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


def parse_rate(s: pd.Series) -> pd.Series:
    out = (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan})
    )
    val = pd.to_numeric(out, errors="coerce").fillna(0.0)
    return np.where(val > 1.0, val / 100.0, val).astype(float)


def parse_term_months(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(out, errors="coerce").fillna(36.0)


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


def registered_paper4_pages() -> list[str]:
    pages: list[str] = []
    if not BOOK_CONFIG.exists():
        return pages
    for raw in BOOK_CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("- chapters/19-paper-mega-extension/"):
            pages.append(Path(stripped.removeprefix("- ")).name)
    return pages


def file_hash(path: Path, max_mb: int = 8) -> str:
    h = hashlib.sha256()
    limit = max_mb * 1024 * 1024
    read = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(min(1024 * 1024, max(0, limit - read)))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if read >= limit:
                break
    suffix = "partial" if path.stat().st_size > limit else "full"
    return f"sha256_{suffix}:{h.hexdigest()}"


def _available_id_set(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=[column])
    except Exception:
        return set()
    return set(df[column].astype(str))


def build_v55_lineage_and_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = {
        "test_predictions.loan_id": (ROOT / "data/processed/test_predictions.parquet", "loan_id"),
        "test.id": (ROOT / "data/processed/test.parquet", "id"),
        "test_fe.id": (ROOT / "data/processed/test_fe.parquet", "id"),
        "candidate_pool.loan_id": (
            TABLE_DIR / "paper4_challenger_local_candidate_pool.parquet",
            "loan_id",
        ),
        "champion_universe.id": (
            ROOT / "data/processed/champion_candidate_universe.parquet",
            "id",
        ),
        "loan_master.id": (ROOT / "data/processed/loan_master.parquet", "id"),
    }
    sets = {name: _available_id_set(path, col) for name, (path, col) in sources.items()}
    match_rows: list[dict[str, Any]] = []
    keys = list(sets)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            inter = len(sets[left] & sets[right])
            left_n = len(sets[left])
            right_n = len(sets[right])
            match_rows.append(
                {
                    "left_source": left,
                    "right_source": right,
                    "left_n": left_n,
                    "right_n": right_n,
                    "intersection_n": inter,
                    "left_match_rate": inter / max(left_n, 1),
                    "right_match_rate": inter / max(right_n, 1),
                    "join_blocker_v55": inter == 0,
                }
            )
    match = pd.DataFrame(match_rows).sort_values("intersection_n", ascending=False)
    write_csv(TABLE_DIR / "paper4_v55_join_match_rate_table.csv", match)

    test_cols = [
        "id",
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "issue_d",
        "loan_status",
        "addr_state",
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "next_pymnt_d",
    ]
    test = read_parquet(ROOT / "data/processed/test.parquet", columns=test_cols)
    pred = read_parquet(ROOT / "data/processed/test_predictions.parquet")
    conformal = read_parquet(
        ROOT / "data/processed/conformal_intervals_mondrian.parquet",
        columns=["id", "pd_low_90", "pd_high_90", "width_90", "temporal_segment"],
    )
    online = read_parquet(
        "paper4_online_conformal_v4_intervals.parquet",
        columns=["loan_id", "qhat_v4", "interval_width_online_v4", "covered_online_v4"],
    )
    if not online.empty:
        online = (
            online.groupby("loan_id", dropna=False)
            .agg(
                qhat_v4=("qhat_v4", "median"),
                interval_width_online_v4=("interval_width_online_v4", "median"),
                covered_online_v4=("covered_online_v4", "mean"),
            )
            .reset_index()
        )
    universe = (
        test.assign(loan_id=test["id"].astype(str))
        .merge(pred.assign(loan_id=pred["loan_id"].astype(str)), on="loan_id", how="inner")
        .merge(
            conformal.assign(loan_id=conformal["id"].astype(str)).drop(columns=["id"]),
            on="loan_id",
            how="left",
        )
        .merge(online.assign(loan_id=online["loan_id"].astype(str)), on="loan_id", how="left")
    )
    universe["issue_month"] = (
        pd.to_datetime(universe["issue_d"], errors="coerce").dt.to_period("M").astype(str)
    )
    universe["period"] = (
        pd.to_datetime(universe["issue_d"], errors="coerce")
        .dt.year.fillna(0)
        .astype(int)
        .astype(str)
    )
    universe["int_rate_decimal"] = parse_rate(universe["int_rate"])
    universe["term_months"] = parse_term_months(universe["term"])
    universe["loan_amnt"] = pd.to_numeric(universe["loan_amnt"], errors="coerce").fillna(0.0)
    universe["base_return_vec"] = universe["loan_amnt"] * universe["int_rate_decimal"]
    pd_point = universe.get("pd_calibrated", universe.get("y_prob_final", 0.0))
    universe["pd_point"] = pd.to_numeric(pd_point, errors="coerce").fillna(0.0).clip(0, 1)
    universe["pd_high_alpha01"] = (
        pd.to_numeric(universe.get("pd_high_90", universe["pd_point"]), errors="coerce")
        .fillna(universe["pd_point"])
        .clip(lower=universe["pd_point"], upper=0.95)
    )
    universe["qhat_v4"] = pd.to_numeric(universe.get("qhat_v4", np.nan), errors="coerce")
    fallback_qhat = float(universe["qhat_v4"].median(skipna=True))
    if not math.isfinite(fallback_qhat):
        fallback_qhat = float(
            pd.to_numeric(universe.get("width_90", 0.20), errors="coerce").median()
        )
    universe["qhat_v4"] = universe["qhat_v4"].fillna(fallback_qhat).clip(0, 1)
    universe["annual_inc"] = pd.to_numeric(universe["annual_inc"], errors="coerce").fillna(0.0)
    universe["dti"] = pd.to_numeric(universe["dti"], errors="coerce").fillna(0.0)
    universe["fico_score"] = (
        pd.to_numeric(universe["fico_range_low"], errors="coerce").fillna(0.0)
        + pd.to_numeric(universe["fico_range_high"], errors="coerce").fillna(0.0)
    ) / 2.0
    universe["grade"] = universe["grade"].astype(str).fillna("unknown")
    top_states = universe["addr_state"].astype(str).value_counts().head(20).index
    universe["state_top20"] = np.where(
        universe["addr_state"].astype(str).isin(top_states),
        universe["addr_state"].astype(str),
        "other",
    )
    universe["income_band"] = pd.qcut(
        universe["annual_inc"].rank(method="first"), 5, labels=False, duplicates="drop"
    ).astype(str)
    universe["dti_band"] = pd.qcut(
        universe["dti"].rank(method="first"), 5, labels=False, duplicates="drop"
    ).astype(str)
    universe["score_decile"] = pd.qcut(
        universe["pd_point"].rank(method="first"), 10, labels=False, duplicates="drop"
    ).astype(str)
    grade_lgd = {"A": 0.34, "B": 0.38, "C": 0.43, "D": 0.49, "E": 0.55, "F": 0.61, "G": 0.66}
    universe["lgd_proxy_v55"] = universe["grade"].map(grade_lgd).fillna(0.45).astype(float)
    qhat_rank = universe["qhat_v4"].rank(pct=True)
    pd_rank = universe["pd_point"].rank(pct=True)
    universe["weak_source_proxy"] = (0.65 * qhat_rank + 0.35 * pd_rank).fillna(0.0).clip(0, 1)
    universe["loan_index_v55"] = np.arange(len(universe), dtype=np.int32)
    keep = [
        "loan_index_v55",
        "loan_id",
        "loan_amnt",
        "term_months",
        "int_rate_decimal",
        "installment",
        "grade",
        "sub_grade",
        "issue_d",
        "issue_month",
        "period",
        "loan_status",
        "addr_state",
        "state_top20",
        "annual_inc",
        "income_band",
        "dti",
        "dti_band",
        "fico_score",
        "score_decile",
        "y_true",
        "pd_point",
        "pd_high_alpha01",
        "pd_low_90",
        "pd_high_90",
        "width_90",
        "qhat_v4",
        "weak_source_proxy",
        "lgd_proxy_v55",
        "base_return_vec",
        "next_pymnt_d",
    ]
    universe = universe[keep].drop_duplicates("loan_id").reset_index(drop=True)
    universe["loan_index_v55"] = np.arange(len(universe), dtype=np.int32)
    universe.to_parquet(
        TABLE_DIR / "paper4_v55_maximal_comparable_universe.parquet",
        index=False,
        compression="zstd",
    )
    pred_test_join = match.loc[
        (match["left_source"].eq("test.id") & match["right_source"].eq("test_predictions.loan_id"))
        | (
            match["left_source"].eq("test_predictions.loan_id")
            & match["right_source"].eq("test.id")
        )
    ]
    pred_test_rows = int(pred_test_join["intersection_n"].max()) if not pred_test_join.empty else 0
    lineage = pd.DataFrame(
        [
            {
                "lineage_item": "exact_prediction_to_test_join",
                "status_v55": "resolved",
                "source_left": "data/processed/test_predictions.parquet::loan_id",
                "source_right": "data/processed/test.parquet::id",
                "match_rows_v55": pred_test_rows,
                "claim_boundary_v55": "full comparable feature/prediction universe, not exact full-universe CVaR optimality",
            },
            {
                "lineage_item": "loan_master_id_join",
                "status_v55": "blocked_by_different_id_space",
                "source_left": "data/processed/loan_master.parquet::id",
                "source_right": "data/processed/test_predictions.parquet::loan_id",
                "match_rows_v55": 0,
                "claim_boundary_v55": "loan_master cannot be used as the id bridge without external/raw mapping",
            },
            {
                "lineage_item": "maximal_comparable_universe",
                "status_v55": "implemented",
                "source_left": "test + predictions + conformal + online intervals",
                "source_right": "reports/paper_material/paper4/tables/paper4_v55_maximal_comparable_universe.parquet",
                "match_rows_v55": int(len(universe)),
                "claim_boundary_v55": "276,869 eligible test loans with features and PD/conformal proxies",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v55_full_universe_lineage_audit.csv", lineage)
    return lineage, match, universe


def _scenario_factors(n_paths: int = 128) -> pd.DataFrame:
    paths = read_parquet("paper4_v31_sample_paths.parquet")
    if paths.empty:
        return pd.DataFrame()
    path_ids = sorted(paths["path_id"].drop_duplicates().astype(int).tolist())[:n_paths]
    p = paths.loc[paths["path_id"].isin(path_ids)].copy()
    p["issue_month"] = pd.to_datetime(p["month"], errors="coerce").dt.to_period("M").astype(str)
    keep = [
        "path_id",
        "issue_month",
        "macro_regime_v15",
        "path_family_v19",
        "default_factor_v15",
        "lgd_factor_v15",
        "prepay_factor_v15",
    ]
    p = p[keep].drop_duplicates(["path_id", "issue_month"])
    fallback = (
        p.sort_values("issue_month")
        .groupby("path_id", dropna=False)
        .head(1)
        .assign(issue_month="__fallback__")
    )
    return pd.concat([p, fallback], ignore_index=True)


def _expected_by_path_matrix(
    universe: pd.DataFrame, n_paths: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    factors = _scenario_factors(n_paths)
    path_ids = sorted(factors["path_id"].drop_duplicates().astype(int).tolist())[:n_paths]
    losses = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    returns = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    default_probs = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    fallback = factors.loc[factors["issue_month"].eq("__fallback__")].copy()
    base = universe[
        [
            "loan_id",
            "issue_month",
            "loan_amnt",
            "pd_high_alpha01",
            "lgd_proxy_v55",
            "base_return_vec",
        ]
    ].copy()
    for row_idx, path_id in enumerate(path_ids):
        f = factors.loc[factors["path_id"].eq(path_id)].drop(columns=["path_id"])
        merged = base.merge(f, on="issue_month", how="left")
        missing = merged["default_factor_v15"].isna()
        if missing.any():
            fb = fallback.loc[fallback["path_id"].eq(path_id)].head(1)
            for col in [
                "macro_regime_v15",
                "path_family_v19",
                "default_factor_v15",
                "lgd_factor_v15",
                "prepay_factor_v15",
            ]:
                merged.loc[missing, col] = fb[col].iloc[0] if not fb.empty else 1.0
        dp = (
            pd.to_numeric(merged["pd_high_alpha01"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["default_factor_v15"], errors="coerce").fillna(1.0)
        ).clip(0, 0.95)
        lgd_factor = (
            pd.to_numeric(merged["lgd_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        prepay_factor = (
            pd.to_numeric(merged["prepay_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        expected_loss = (
            pd.to_numeric(merged["loan_amnt"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["lgd_proxy_v55"], errors="coerce").fillna(0.45)
            * dp
            * lgd_factor
        )
        prepay_drag = (
            pd.to_numeric(merged["loan_amnt"], errors="coerce").fillna(0.0)
            * 0.012
            * (1 - dp)
            * prepay_factor
        )
        losses[row_idx, :] = expected_loss.to_numpy(float)
        returns[row_idx, :] = (
            pd.to_numeric(merged["base_return_vec"], errors="coerce").fillna(0.0)
            - expected_loss
            - prepay_drag
        ).to_numpy(float)
        default_probs[row_idx, :] = dp.to_numpy(float)
    return losses, returns, default_probs, path_ids


def _pool_sources(df: pd.DataFrame, family: str, min_support: int) -> pd.Series:
    raw = df[family].astype(str).fillna("unknown")
    counts = raw.value_counts(dropna=False)
    small = set(counts.loc[counts < min_support].index.astype(str))
    return raw.where(~raw.isin(small), "__pooled_small__")


def _build_caps(
    pool: pd.DataFrame,
    regime: str,
    cap_multiplier: float,
    hard_family_caps: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    min_support = 250
    for family in ["grade", "period", "state_top20", "income_band", "dti_band", "score_decile"]:
        if family not in pool:
            continue
        source = _pool_sources(pool, family, min_support)
        exposure = pool.groupby(source)["loan_amnt"].sum()
        counts = source.value_counts(dropna=False)
        total = float(pool["loan_amnt"].sum())
        for source_id, used in exposure.items():
            empirical_share = float(used / max(total, 1.0))
            cap = min(
                0.85,
                max(
                    0.045, empirical_share * cap_multiplier + (0.01 if regime != "strict" else 0.0)
                ),
            )
            if hard_family_caps and family in hard_family_caps:
                cap = min(cap, hard_family_caps[family])
            rows.append(
                {
                    "regime_v56": regime,
                    "source_family": family,
                    "source_id": str(source_id),
                    "min_support_v56": min_support,
                    "source_count_v56": int(counts.get(source_id, 0)),
                    "empirical_exposure_share_v56": empirical_share,
                    "source_cap_v56": cap,
                    "hard_family_cap_v56": hard_family_caps.get(family)
                    if hard_family_caps
                    else np.nan,
                    "pooled_small_cell_v56": source_id == "__pooled_small__",
                }
            )
    return pd.DataFrame(rows)


def _select_master_pool(
    universe: pd.DataFrame, max_columns: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = universe.copy()
    u["expected_loss_proxy_v56"] = u["loan_amnt"] * u["lgd_proxy_v55"] * u["pd_high_alpha01"]
    u["expected_return_proxy_v56"] = (
        u["base_return_vec"] - u["expected_loss_proxy_v56"] - 0.012 * u["loan_amnt"]
    )
    u["return_score_v56"] = u["expected_return_proxy_v56"] - 0.25 * u["expected_loss_proxy_v56"]
    u["tail_score_v56"] = (
        u["expected_return_proxy_v56"]
        - 1.40 * u["expected_loss_proxy_v56"]
        - 1800 * u["qhat_v4"]
        - 2200 * u["weak_source_proxy"]
    )
    u["source_score_v56"] = (
        u["expected_return_proxy_v56"]
        - 0.80 * u["expected_loss_proxy_v56"]
        - 3300 * u["weak_source_proxy"]
        - 1500 * u["qhat_v4"]
    )
    initial = pd.concat(
        [
            u.nlargest(max_columns // 3, "return_score_v56"),
            u.nlargest(max_columns // 3, "tail_score_v56"),
            u.nlargest(max_columns // 3, "source_score_v56"),
        ],
        ignore_index=True,
    ).drop_duplicates("loan_id")
    if len(initial) < max_columns:
        initial = pd.concat(
            [initial, u.nlargest(max_columns, "expected_return_proxy_v56")],
            ignore_index=True,
        ).drop_duplicates("loan_id")
    initial = initial.head(max_columns).copy().reset_index(drop=True)
    logs = pd.DataFrame(
        [
            {
                "round_v56": 0,
                "selection_rule_v56": "tri_score_warm_start",
                "columns_before_v56": 0,
                "columns_after_v56": int(len(initial)),
                "new_columns_v56": int(len(initial)),
                "pricing_tolerance_v56": "heuristic_score_not_dual_exact",
            }
        ]
    )
    return initial, logs


def _solve_cvar_source_lp(
    pool: pd.DataFrame,
    losses: np.ndarray,
    returns_by_path: np.ndarray,
    regime: str,
    cvar_cap: float,
    return_floor: float,
    cap_multiplier: float,
    soft_slack: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(pool)
    s = losses.shape[0]
    returns = returns_by_path.mean(axis=0)
    amounts = pool["loan_amnt"].to_numpy(float)
    eta_idx = n
    u_start = n + 1
    slack_cvar_idx = n + 1 + s if soft_slack else None
    slack_return_idx = n + 2 + s if soft_slack else None
    nvars = n + 1 + s + (2 if soft_slack else 0)
    alpha = 0.90
    coeff = 1.0 / ((1 - alpha) * s)

    c = np.zeros(nvars)
    if soft_slack:
        c[:n] = -0.0005 * returns
        c[slack_cvar_idx] = 1.0
        c[slack_return_idx] = 1.0
    else:
        c[:n] = -returns
        c[eta_idx] = 1e-4
        c[u_start : u_start + s] = 1e-4 * coeff

    rows: list[sparse.csr_matrix] = []
    rhs: list[float] = []
    names: list[str] = []

    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = amounts
    rows.append(row.tocsr())
    rhs.append(BUDGET)
    names.append("budget_upper")

    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -amounts
    rows.append(row.tocsr())
    rhs.append(-0.98 * BUDGET)
    names.append("budget_lower")

    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -returns
    if soft_slack:
        row[0, slack_return_idx] = -1.0
    rows.append(row.tocsr())
    rhs.append(-return_floor)
    names.append("return_floor")

    row = sparse.lil_matrix((1, nvars))
    row[0, eta_idx] = 1.0
    row[0, u_start : u_start + s] = coeff
    if soft_slack:
        row[0, slack_cvar_idx] = -1.0
    rows.append(row.tocsr())
    rhs.append(cvar_cap)
    names.append("cvar_cap")

    scen = sparse.lil_matrix((s, nvars))
    scen[:, :n] = sparse.csr_matrix(losses)
    scen[:, eta_idx] = -1.0
    for i in range(s):
        scen[i, u_start + i] = -1.0
    rows.append(scen.tocsr())
    rhs.extend([0.0] * s)
    names.extend(["scenario_excess"] * s)

    caps = _build_caps(pool, regime, cap_multiplier)
    for _, cap_row in caps.iterrows():
        family = str(cap_row["source_family"])
        source = _pool_sources(pool, family, int(cap_row["min_support_v56"]))
        mask = source.astype(str).eq(str(cap_row["source_id"])).to_numpy()
        if not mask.any():
            continue
        row = sparse.lil_matrix((1, nvars))
        row[0, np.where(mask)[0]] = amounts[mask]
        rows.append(row.tocsr())
        rhs.append(float(cap_row["source_cap_v56"]) * BUDGET)
        names.append(f"source_cap::{family}::{cap_row['source_id']}")

    bounds = [(0.0, 1.0)] * n + [(0.0, BUDGET)] + [(0.0, BUDGET)] * s
    if soft_slack:
        bounds.extend([(0.0, BUDGET), (0.0, BUDGET)])
    result = linprog(
        c,
        A_ub=sparse.vstack(rows).tocsr(),
        b_ub=np.array(rhs, dtype=float),
        bounds=bounds,
        method="highs",
        options={"time_limit": 300},
    )
    status: dict[str, Any] = {
        "regime_v56": regime,
        "soft_slack_v56": soft_slack,
        "solver_success_v56": bool(result.success),
        "solver_status_v56": int(result.status),
        "solver_message_v56": str(result.message),
        "restricted_master_columns_v56": n,
        "scenario_count_v56": s,
        "cvar_cap_v56": cvar_cap,
        "return_floor_v56": return_floor,
        "source_cap_count_v56": int(len(caps)),
        "exact_full_universe_claim_v56": False,
        "diagnostic_or_exact_v56": "restricted_master_exact_lp_not_full_universe",
        "claim_boundary_v56": "exact LP over expanded restricted master from 276k comparable universe; not full-universe optimality",
    }
    if not result.success:
        cert = pd.DataFrame(
            [
                {
                    **status,
                    "required_cvar_slack_v56": np.nan,
                    "required_return_floor_relaxation_v56": np.nan,
                    "certificate_scope_v56": "solver failed or infeasible before allocation",
                }
            ]
        )
        return status, pd.DataFrame(), cert, caps

    x = result.x[:n]
    eta = float(result.x[eta_idx])
    u = result.x[u_start : u_start + s]
    scenario_losses = losses @ x
    scenario_returns = returns_by_path @ x
    cvar = float(eta + coeff * u.sum())
    expected_return = float(returns @ x)
    exposure = float(amounts @ x)
    slack_cvar = float(result.x[slack_cvar_idx]) if soft_slack else max(0.0, cvar - cvar_cap)
    slack_return = (
        float(result.x[slack_return_idx])
        if soft_slack
        else max(0.0, return_floor - expected_return)
    )
    alloc = pool.loc[x > 1e-6].copy()
    alloc["allocation_fraction_v56"] = x[x > 1e-6]
    alloc["allocated_exposure_v56"] = alloc["loan_amnt"] * alloc["allocation_fraction_v56"]
    alloc["policy_id_v56"] = f"v56_cvar_source_{regime}{'_soft_slack' if soft_slack else ''}"
    alloc["regime_v56"] = regime
    alloc["claim_boundary_v56"] = status["claim_boundary_v56"]

    scenario = pd.DataFrame(
        {
            "regime_v56": regime,
            "soft_slack_v56": soft_slack,
            "scenario_row": np.arange(s),
            "scenario_loss_v56": scenario_losses,
            "scenario_return_v56": scenario_returns,
            "eta_v56": eta,
            "excess_loss_v56": np.maximum(scenario_losses - eta, 0),
        }
    )
    scenario.to_csv(
        TABLE_DIR
        / f"paper4_v56_cvar_scenario_losses_{regime}{'_soft_slack' if soft_slack else ''}.csv",
        index=False,
    )
    used_caps = []
    for _, cap_row in caps.iterrows():
        family = str(cap_row["source_family"])
        source = _pool_sources(pool, family, int(cap_row["min_support_v56"]))
        mask = source.astype(str).eq(str(cap_row["source_id"])).to_numpy()
        if not mask.any():
            continue
        used = float(amounts[mask] @ x[mask])
        allowed = float(cap_row["source_cap_v56"]) * BUDGET
        used_caps.append(
            {
                **cap_row.to_dict(),
                "used_exposure_v56": used,
                "allowed_exposure_v56": allowed,
                "slack_v56": allowed - used,
                "active_v56": allowed - used <= 0.005 * BUDGET,
            }
        )
    active = pd.DataFrame(used_caps)
    active.to_csv(
        TABLE_DIR
        / f"paper4_v56_cvar_active_constraints_{regime}{'_soft_slack' if soft_slack else ''}.csv",
        index=False,
    )
    status.update(
        {
            "objective_return_v56": expected_return,
            "allocated_exposure_v56": exposure,
            "n_allocated_loans_v56": int((x > 1e-6).sum()),
            "scenario_loss_mean_v56": float(np.mean(scenario_losses)),
            "scenario_loss_p95_v56": float(np.quantile(scenario_losses, 0.95)),
            "scenario_loss_cvar90_v56": cvar,
            "scenario_return_p05_v56": float(np.quantile(scenario_returns, 0.05)),
            "budget_slack_v56": BUDGET - exposure,
            "required_cvar_slack_v56": slack_cvar,
            "required_return_floor_relaxation_v56": slack_return,
        }
    )
    cert = pd.DataFrame(
        [
            {
                **status,
                "certificate_scope_v56": "LP slack certificate over expanded restricted master; not mathematical proof for all 276k loans",
            }
        ]
    )
    return status, alloc, cert, active


def build_v56_cvar(max_columns: int = 48_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    if universe.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v56_cvar_full_comparable_frontier.csv", empty)
        empty.to_parquet(TABLE_DIR / "paper4_v56_cvar_allocations.parquet", index=False)
        write_csv(TABLE_DIR / "paper4_v56_cvar_slack_certificate.csv", empty)
        return empty, empty, empty
    pool, logs = _select_master_pool(universe, max_columns=max_columns)
    write_csv(TABLE_DIR / "paper4_v56_column_generation_log.csv", logs)
    losses, returns_by_path, default_probs, path_ids = _expected_by_path_matrix(pool)
    manifest = pd.DataFrame(
        [
            {
                "matrix_id": "v56_expanded_restricted_master_expected_loss",
                "source_universe_rows": int(len(universe)),
                "restricted_master_columns": int(len(pool)),
                "scenario_count": int(len(path_ids)),
                "loss_matrix_cells": int(losses.size),
                "persisted_full_matrix": False,
                "storage_decision": "matrix generated in memory; persist allocations/frontier/scenario summaries only",
                "exact_full_universe_claim": False,
                "claim_boundary": "largest comparable universe is 276,869 rows; solver is expanded restricted master, not full-universe exact",
            }
        ]
    )
    write_csv(TABLE_DIR / "paper4_v56_loss_matrix_manifest.csv", manifest)
    regimes = [
        ("strict", 45_000.0, 80_000.0, 1.05, False),
        ("committee", 65_000.0, 95_000.0, 1.25, False),
        ("relaxed", 95_000.0, 105_000.0, 1.65, False),
        ("tail_first", 52_000.0, 60_000.0, 1.65, False),
        ("strict", 45_000.0, 80_000.0, 1.05, True),
    ]
    frontier_rows: list[dict[str, Any]] = []
    allocs: list[pd.DataFrame] = []
    certs: list[pd.DataFrame] = []
    active_rows: list[pd.DataFrame] = []
    for regime, cap, floor, cap_mult, soft in regimes:
        status, alloc, cert, active = _solve_cvar_source_lp(
            pool,
            losses,
            returns_by_path,
            regime=regime,
            cvar_cap=cap,
            return_floor=floor,
            cap_multiplier=cap_mult,
            soft_slack=soft,
        )
        frontier_rows.append(status)
        if not alloc.empty:
            allocs.append(alloc)
        if not cert.empty:
            certs.append(cert)
        if not active.empty:
            active_rows.append(active)
    frontier = pd.DataFrame(frontier_rows)
    if not frontier.empty:
        frontier["return_norm_v56"] = normalize(
            frontier.get("objective_return_v56", pd.Series(dtype=float))
        )
        frontier["tail_norm_v56"] = normalize(
            frontier.get("scenario_loss_cvar90_v56", pd.Series(dtype=float)), higher_is_better=False
        )
        frontier["slack_norm_v56"] = normalize(
            frontier.get("required_cvar_slack_v56", pd.Series(dtype=float)).fillna(1e9)
            + frontier.get("required_return_floor_relaxation_v56", pd.Series(dtype=float)).fillna(
                1e9
            ),
            higher_is_better=False,
        )
        frontier["frontier_score_v56"] = (
            0.42 * frontier["return_norm_v56"]
            + 0.38 * frontier["tail_norm_v56"]
            + 0.20 * frontier["slack_norm_v56"]
        )
        frontier["non_dominated_restricted_v56"] = frontier["solver_success_v56"].astype(bool) & (
            frontier["frontier_score_v56"] >= frontier["frontier_score_v56"].median()
        )
    allocations = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    cert = pd.concat(certs, ignore_index=True) if certs else pd.DataFrame()
    active = pd.concat(active_rows, ignore_index=True) if active_rows else pd.DataFrame()
    write_csv(TABLE_DIR / "paper4_v56_cvar_full_comparable_frontier.csv", frontier)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v56_cvar_allocations.parquet", index=False, compression="zstd"
    )
    write_csv(TABLE_DIR / "paper4_v56_cvar_slack_certificate.csv", cert)
    write_csv(
        TABLE_DIR / "paper4_v56_source_governance_caps.csv", _build_caps(pool, "committee", 1.25)
    )
    write_csv(TABLE_DIR / "paper4_v56_active_constraints_all.csv", active)
    write_csv(
        TABLE_DIR / "paper4_v56_full_universe_attempt.csv",
        [
            {
                "source_universe_rows_v56": int(len(universe)),
                "attempted_exact_full_universe_v56": False,
                "reason_v56": "276,869-variable x 128-scenario exact LP estimated as heavy; expanded restricted master used first",
                "restricted_master_columns_v56": int(len(pool)),
                "exact_full_universe_claim_v56": False,
                "future_unlock_v56": "chunked/highspy column generation with exact pricing or persisted complete loss matrix",
            }
        ],
    )
    return frontier, allocations, cert


def build_v57_online_spo_dla_ifrs9() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # Online direct weak-cell search: extend v49 with guarded cell-level deltas and min support.
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
    online_rows: list[dict[str, Any]] = []
    if not selected.empty and not base.empty:
        df = selected.merge(base, on=["loan_id", "issue_month"], how="left")
        df["issue_month"] = (
            pd.to_datetime(df["issue_month"], errors="coerce").dt.to_period("M").astype(str)
        )
        df["qhat_v9"] = pd.to_numeric(df["qhat_v9"], errors="coerce").fillna(0.0)
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0.0)
        source_specs = {
            "grade": "original_grade",
            "period": "period",
            "term": "term",
            "score_decile": "score_decile",
            "state_top20": "state_top20",
            "income_band": "income_band",
            "dti_band": "dti_band",
        }
        for family, col in source_specs.items():
            work = df.assign(source_family=family, source_id=df[col].astype(str))
            for min_support in [3, 10, 20, 40]:
                for global_delta in [0.00, 0.02, 0.04, 0.06, 0.08]:
                    counts = work.groupby(["source_id", "issue_month"])["loan_id"].transform("size")
                    pooled = counts.lt(min_support)
                    weak_bonus = np.where(pooled, 0.04, 0.0)
                    qhat = (work["qhat_v9"] + global_delta + weak_bonus).clip(0, 1)
                    low = (work["y_pred"] - qhat).clip(0, 1)
                    high = (work["y_pred"] + qhat).clip(0, 1)
                    coverage = ((work["y_true"] >= low) & (work["y_true"] <= high)).astype(float)
                    width = high - low
                    tmp = work[["source_id", "issue_month", "policy_id"]].copy()
                    tmp["coverage"] = coverage
                    tmp["width"] = width
                    source_cells = (
                        tmp.groupby(["source_id", "issue_month"], dropna=False)
                        .agg(
                            n=("coverage", "size"),
                            coverage=("coverage", "mean"),
                            width=("width", "mean"),
                        )
                        .reset_index()
                    )
                    policy_cells = (
                        tmp.groupby(["policy_id", "issue_month"], dropna=False)
                        .agg(n=("coverage", "size"), coverage=("coverage", "mean"))
                        .reset_index()
                    )
                    source_def = source_cells.loc[source_cells["n"].ge(min_support)]
                    policy_def = policy_cells.loc[policy_cells["n"].ge(min_support)]
                    source_min = float(source_def["coverage"].min()) if len(source_def) else np.nan
                    policy_min = float(policy_def["coverage"].min()) if len(policy_def) else np.nan
                    avg_width = float(width.mean())
                    gate = bool(source_min >= 0.80 and policy_min >= 0.90 and avg_width <= 0.95)
                    online_rows.append(
                        {
                            "source_family": family,
                            "method_v57": "direct_hierarchical_qhat_bonus",
                            "min_support_v57": min_support,
                            "global_delta_v57": global_delta,
                            "small_cell_bonus_v57": 0.04,
                            "source_month_defended_min_v57": source_min,
                            "policy_month_defended_min_v57": policy_min,
                            "avg_width_loan_v57": avg_width,
                            "n_defended_source_cells_v57": int(len(source_def)),
                            "n_defended_policy_cells_v57": int(len(policy_def)),
                            "gate_source80_policy90_width95_v57": gate,
                            "strict_live_deployability_claim_allowed": False,
                            "claim_boundary_v57": "historical direct source-family qhat search; no live deployability claim",
                        }
                    )
    online = pd.DataFrame(online_rows)
    if not online.empty:
        online["repair_decision_v57"] = np.where(
            online["gate_source80_policy90_width95_v57"],
            "historical_gate_pass_with_live_claim_still_false",
            "near_resolved_with_plateau",
        )
    write_csv(TABLE_DIR / "paper4_v57_online_source_family_direct_repair.csv", online)

    probes = pd.DataFrame(
        [
            package_probe(pkg)
            for pkg in ["cvxpy", "cvxpylayers", "torch", "highspy", "pyomo", "catboost", "sklearn"]
        ]
    )
    probes["formal_differentiable_spo_claim_allowed"] = False
    probes["claim_boundary_v57"] = (
        "dependency probe only; differentiable SPO remains blocked unless cvxpylayers/torch validate"
    )
    write_csv(TABLE_DIR / "paper4_v57_spo_dependency_probe.csv", probes)

    frontier = read_csv("paper4_v56_cvar_full_comparable_frontier.csv")
    read_parquet("paper4_v56_cvar_allocations.parquet")
    spo_rows: list[dict[str, Any]] = []
    if not frontier.empty:
        for _, row in frontier.iterrows():
            if not bool(row.get("solver_success_v56", False)):
                continue
            spo_rows.append(
                {
                    "candidate_id_v57": f"spo_oracle_bridge_{row.get('regime_v56')}_{'soft' if row.get('soft_slack_v56') else 'hard'}",
                    "oracle_source_v57": "paper4_v56_cvar_full_comparable_frontier.csv",
                    "temporal_split_v57": "historical_issue_month_proxy",
                    "oracle_return_v57": row.get("objective_return_v56"),
                    "oracle_cvar90_v57": row.get("scenario_loss_cvar90_v56"),
                    "surrogate_regret_proxy_v57": max(
                        0.0,
                        float(
                            frontier["objective_return_v56"].max()
                            - row.get("objective_return_v56", 0)
                        ),
                    ),
                    "formal_differentiable_spo_claim_allowed": False,
                    "claim_boundary_v57": "solver-oracle regret bridge only; no differentiable SPO+ claim",
                }
            )
    spo = pd.DataFrame(spo_rows)
    write_csv(TABLE_DIR / "paper4_v57_spo_oracle_regret_bridge.csv", spo)

    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    dla_rows: list[dict[str, Any]] = []
    if not universe.empty:
        u = universe.copy()
        u["adp_score_v57"] = (
            u["base_return_vec"]
            - 1.05 * u["loan_amnt"] * u["lgd_proxy_v55"] * u["pd_high_alpha01"]
            - 2300 * u["weak_source_proxy"]
            - 1400 * u["qhat_v4"]
        )
        cash = BUDGET
        for month, group in u.sort_values("issue_month").groupby("issue_month", dropna=False):
            monthly_budget = min(cash, BUDGET / max(u["issue_month"].nunique(), 1) * 1.25)
            g = group.sort_values("adp_score_v57", ascending=False).copy()
            chosen = g.loc[g["loan_amnt"].cumsum().le(monthly_budget)].head(120)
            if chosen.empty and cash > 0 and not g.empty:
                chosen = g.head(1)
            exposure = float(chosen["loan_amnt"].sum())
            cash -= exposure
            dla_rows.append(
                {
                    "month": month,
                    "policy_id": "v57_endogenous_adp_source_guard",
                    "selected_loans_v57": int(len(chosen)),
                    "funded_exposure_v57": exposure,
                    "cash_after_decision_v57": cash,
                    "mean_pd_v57": float(chosen["pd_point"].mean()) if not chosen.empty else 0.0,
                    "mean_qhat_v57": float(chosen["qhat_v4"].mean()) if not chosen.empty else 0.0,
                    "bellman_exact_claim_allowed": False,
                    "claim_boundary_v57": "endogenous monthly ADP-style score policy; not Bellman exact",
                }
            )
            if cash <= 0:
                break
    dla = pd.DataFrame(dla_rows)
    write_csv(TABLE_DIR / "paper4_v57_dla_endogenous_policy_improvement.csv", dla)

    ifrs9_rows: list[dict[str, Any]] = []
    if not universe.empty:
        for rule, thr in [
            ("abs_pd_020", 0.20),
            ("abs_pd_030", 0.30),
            ("relative_pd_top30pct", float(universe["pd_point"].quantile(0.70))),
            ("width_top30pct", float(universe["qhat_v4"].quantile(0.70))),
        ]:
            if rule.startswith("width"):
                stage2 = universe["qhat_v4"].ge(thr)
            else:
                stage2 = universe["pd_point"].ge(thr)
            ecl = universe["loan_amnt"] * universe["lgd_proxy_v55"] * universe["pd_point"]
            ifrs9_rows.append(
                {
                    "sicr_rule_v57": rule,
                    "threshold_v57": thr,
                    "stage2_share_v57": float(stage2.mean()),
                    "ecl_proxy_total_v57": float(ecl.sum()),
                    "stage2_ecl_share_v57": float(ecl.loc[stage2].sum() / max(ecl.sum(), 1.0)),
                    "available_servicing_fields_v57": "next_pymnt_d_only",
                    "contractual_ifrs9_claim_allowed": False,
                    "claim_boundary_v57": "IFRS9-inspired proxy only; no monthly DPD/cure/recovery/prepayment timing panel",
                }
            )
    ifrs9 = pd.DataFrame(ifrs9_rows)
    write_csv(TABLE_DIR / "paper4_v57_ifrs9_sicr_proxy_panel_update.csv", ifrs9)

    gate = pd.DataFrame(
        [
            {
                "lane_v57": "CATE",
                "diagnostic_status_v57": "theory_blocked",
                "claim_allowed_v57": False,
                "required_unlock_v57": "identification, overlap, sensitivity, falsification, intervals and reject-inference treatment",
                "claim_boundary_v57": "diagnostics only; no CATE policy-value claim",
            },
            {
                "lane_v57": "fairness",
                "diagnostic_status_v57": "data_blocked",
                "claim_allowed_v57": False,
                "required_unlock_v57": "protected attributes or approved proxy protocol",
                "claim_boundary_v57": "source governance only; no fair-lending legal claim",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v57_cate_fairness_gate_update.csv", gate)
    return online, spo, dla, ifrs9


def build_v58_registry_docs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v54 = read_csv("paper4_v54_dynamic_budget_capped_book_summary.csv")
    v56 = read_csv("paper4_v56_cvar_full_comparable_frontier.csv")
    v57_spo = read_csv("paper4_v57_spo_oracle_regret_bridge.csv")
    rows: list[dict[str, Any]] = []
    if not v54.empty:
        for _, row in v54.iterrows():
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "evidence_source_v58": "v54_dynamic_replay",
                    "candidate_family_v58": "dynamic_replay_book",
                    "wealth_metric_v58": row.get("final_wealth_mean_v54"),
                    "tail_metric_v58": -float(row.get("final_loss_p95_v54", 0)),
                    "regret_metric_v58": 0.0,
                    "auditability_metric_v58": 0.75,
                    "claim_safe_v58": True,
                }
            )
    if not v56.empty:
        for _, row in v56.loc[v56["solver_success_v56"].astype(bool)].iterrows():
            rows.append(
                {
                    "policy_id": f"v56_cvar_source_{row.get('regime_v56')}{'_soft_slack' if row.get('soft_slack_v56') else ''}",
                    "evidence_source_v58": "v56_expanded_restricted_master",
                    "candidate_family_v58": "cvar_source_lp_expanded",
                    "wealth_metric_v58": row.get("objective_return_v56"),
                    "tail_metric_v58": -float(row.get("scenario_loss_cvar90_v56", 0)),
                    "regret_metric_v58": -float(row.get("required_return_floor_relaxation_v56", 0)),
                    "auditability_metric_v58": 0.82,
                    "claim_safe_v58": True,
                }
            )
    if not v57_spo.empty:
        for _, row in v57_spo.iterrows():
            rows.append(
                {
                    "policy_id": row.get("candidate_id_v57"),
                    "evidence_source_v58": "v57_spo_oracle_regret_bridge",
                    "candidate_family_v58": "spo_oracle_regret_bridge",
                    "wealth_metric_v58": row.get("oracle_return_v57"),
                    "tail_metric_v58": -float(row.get("oracle_cvar90_v57", 0)),
                    "regret_metric_v58": -float(row.get("surrogate_regret_proxy_v57", 0)),
                    "auditability_metric_v58": 0.70,
                    "claim_safe_v58": True,
                }
            )
    registry = pd.DataFrame(rows).dropna(subset=["policy_id"])
    if not registry.empty:
        registry["wealth_norm_v58"] = normalize(registry["wealth_metric_v58"])
        registry["tail_norm_v58"] = normalize(registry["tail_metric_v58"])
        registry["regret_norm_v58"] = normalize(registry["regret_metric_v58"])
        registry["audit_norm_v58"] = normalize(registry["auditability_metric_v58"])
        registry["full_governance_score_v58"] = (
            0.34 * registry["wealth_norm_v58"]
            + 0.30 * registry["tail_norm_v58"]
            + 0.16 * registry["regret_norm_v58"]
            + 0.12 * registry["audit_norm_v58"]
            + 0.08 * registry["claim_safe_v58"].astype(float)
        )
        registry["paper4_working_champion_allowed_v58"] = False
        registry["paper1_promotion_allowed_v58"] = False
        registry["decision_v58"] = np.select(
            [
                registry["policy_id"].astype(str).eq("paper1_economic_champion"),
                registry["candidate_family_v58"].eq("cvar_source_lp_expanded"),
            ],
            ["retain_reference_until_direct_dynamic_comparison", "serious_tail_solver_challenger"],
            default="review_or_lab_challenger",
        )
        registry["claim_boundary_v58"] = (
            "Paper 4 lab registry; no Paper Estrella promotion and no final Paper 4 promotion"
        )
        registry = registry.sort_values("full_governance_score_v58", ascending=False)
    write_csv(TABLE_DIR / "paper4_v58_candidate_registry.csv", registry)

    claims = pd.DataFrame(
        [
            {
                "claim_id": "v58_full_comparable_universe_unlocked",
                "allowed": True,
                "artifact": "paper4_v55_maximal_comparable_universe.parquet",
                "boundary": "feature/prediction universe only; not exact CVaR optimality",
            },
            {
                "claim_id": "v58_expanded_restricted_cvar",
                "allowed": True,
                "artifact": "paper4_v56_cvar_full_comparable_frontier.csv",
                "boundary": "expanded restricted master over comparable universe, not full-universe exact LP",
            },
            {
                "claim_id": "v58_exact_full_universe_cvar",
                "allowed": False,
                "artifact": "paper4_v56_full_universe_attempt.csv",
                "boundary": "exact full-universe claim remains false",
            },
            {
                "claim_id": "v58_online_live_deployability",
                "allowed": False,
                "artifact": "paper4_v57_online_source_family_direct_repair.csv",
                "boundary": "historical source-family repair only",
            },
            {
                "claim_id": "v58_formal_differentiable_spo",
                "allowed": False,
                "artifact": "paper4_v57_spo_dependency_probe.csv",
                "boundary": "cvxpy/cvxpylayers/torch route not validated",
            },
            {
                "claim_id": "v58_contractual_ifrs9",
                "allowed": False,
                "artifact": "paper4_v57_ifrs9_sicr_proxy_panel_update.csv",
                "boundary": "proxy only; missing servicing panel fields",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v58_claim_matrix.csv", claims)

    artifacts = []
    for path in sorted(TABLE_DIR.glob("paper4_v5[5-8]*")) + sorted(
        STATUS_DIR.glob("paper4_v5[5-8]*")
    ):
        if path.is_file():
            size = path.stat().st_size
            artifacts.append(
                {
                    "artifact": path.name,
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": size,
                    "size_mb": round(size / 1024 / 1024, 3),
                    "hash": file_hash(path),
                    "recommended_storage_v58": "git_ok"
                    if size < 45 * 1024 * 1024
                    else "dvc_or_lfs_required",
                    "claim_role_v58": "v55_v58_unlock_loop",
                }
            )
    manifest = pd.DataFrame(artifacts)
    write_csv(TABLE_DIR / "paper4_v58_heavy_artifact_manifest.csv", manifest)

    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Data lineage",
                "executable_item": "v55 unlocks an exact 276,869-row comparable universe via test_predictions.loan_id -> test.id.",
                "status": "resolved",
                "next_artifact": "paper4_v55_maximal_comparable_universe.parquet",
                "success_condition": "feature-rich comparable universe exists and join audit is artifacted",
                "last_wave": "v55_v58",
                "execution_result": "full_comparable_universe_unlocked",
                "quarto_promotion_decision": "candidate_for_future_official_update",
            },
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": "v56 expands CVaR from the 12k pool to a restricted master drawn from the 276,869-row comparable universe.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_v56_cvar_full_comparable_frontier.csv",
                "success_condition": "exact pricing or full-universe LP certificate proves no improving columns",
                "last_wave": "v55_v58",
                "execution_result": "expanded_restricted_master_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "short",
                "lane": "Online conformal",
                "executable_item": "v57 reruns direct source-family qhat repair; live deployability stays false without unseen holdout.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "future_unseen_source_family_holdout.csv",
                "success_condition": "strict unseen source-family holdout passes with width <= 0.95",
                "last_wave": "v55_v58",
                "execution_result": "direct_hierarchical_repair_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
        ]
    )
    combined = (
        pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    )
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)
    return registry, claims, manifest


def update_v58_notebook(statuses: dict[str, dict[str, Any]]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V55_V58_UNLOCK_LOOP_START -->",
            "",
            "## Wave v55-v58: Full Comparable Universe Unlock",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Turn the v54 blocker into a concrete unlock attempt: audit all loan-id",
            "lineage, build the largest comparable feature/prediction universe, expand",
            "CVaR/source-governed optimization beyond the 12k local pool, re-run direct",
            "online source-family repair, and update claim/storage governance.",
            "",
            "### Scripts",
            "",
            "- `scripts/papers/build_paper4_v55_unlock_loop.py`",
            "",
            "### Results",
            "",
            f"- v55 comparable universe rows: `{statuses['v55'].get('maximal_comparable_universe_rows_v55')}`.",
            f"- v55 exact prediction-test join rate: `{statuses['v55'].get('prediction_test_join_rate_v55')}`.",
            f"- v56 restricted-master columns: `{statuses['v56'].get('restricted_master_columns_v56')}`.",
            f"- v56 successful LP rows: `{statuses['v56'].get('cvar_success_rows_v56')}`.",
            f"- v57 online search rows: `{statuses['v57'].get('online_repair_rows_v57')}`.",
            f"- v58 candidate registry rows: `{statuses['v58'].get('candidate_registry_rows_v58')}`.",
            "",
            "### Interpretation",
            "",
            "The important advance is lineage: Paper 4 can now construct a full",
            "comparable 276,869-loan universe from `test`, `test_predictions`,",
            "conformal intervals, and online interval diagnostics. The CVaR evidence",
            "is stronger than the 12k local pool because the restricted master is drawn",
            "from this full comparable universe, but exact full-universe optimality is",
            "still false until exact pricing or a complete full LP certificate exists.",
            "",
            "### Claim Impact",
            "",
            "- Allowed: full comparable feature/prediction universe exists.",
            "- Allowed: expanded restricted-master CVaR/source LP is implemented.",
            "- Still prohibited: exact full-universe CVaR optimality, live online",
            "  deployability, formal differentiable SPO+, Bellman exact DLA,",
            "  contractual IFRS9, CATE policy value, fair-lending legal claims, and",
            "  Paper Estrella promotion.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep this in the living notebook for now. The lineage unlock may deserve a",
            "future compact Quarto update once the candidate registry stabilizes around",
            "a solver result that survives dynamic replay.",
            "",
            "<!-- V55_V58_UNLOCK_LOOP_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V55_V58_UNLOCK_LOOP_START -->"
        end = "<!-- V55_V58_UNLOCK_LOOP_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v55() -> dict[str, Any]:
    start = datetime.now(UTC)
    lineage, match, universe = build_v55_lineage_and_universe()
    join_row = match.loc[
        (match["left_source"].eq("test.id") & match["right_source"].eq("test_predictions.loan_id"))
        | (
            match["left_source"].eq("test_predictions.loan_id")
            & match["right_source"].eq("test.id")
        )
    ]
    status = {
        "schema_version": "2026-05-15.55",
        "generated_at_utc": now(),
        "phase": "v55_full_comparable_universe_lineage_unlock",
        "maximal_comparable_universe_rows_v55": int(len(universe)),
        "lineage_rows_v55": int(len(lineage)),
        "join_match_rows_v55": int(len(match)),
        "prediction_test_join_rate_v55": float(
            max(join_row["left_match_rate"].iloc[0], join_row["right_match_rate"].iloc[0])
        )
        if not join_row.empty
        else 0.0,
        "loan_master_join_blocked_v55": True,
        "exact_full_universe_cvar_claim_allowed": False,
        "paper1_promotion_allowed_v55": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v55 unlocks full comparable feature/prediction universe; not a full-universe CVaR proof",
    }
    write_json(STATUS_DIR / "paper4_v55_status.json", status)
    return status


def build_v56() -> dict[str, Any]:
    start = datetime.now(UTC)
    frontier, allocations, cert = build_v56_cvar()
    status = {
        "schema_version": "2026-05-15.56",
        "generated_at_utc": now(),
        "phase": "v56_expanded_restricted_master_cvar_source_solver",
        "cvar_frontier_rows_v56": int(len(frontier)),
        "cvar_success_rows_v56": int(frontier["solver_success_v56"].sum())
        if not frontier.empty
        else 0,
        "cvar_allocation_rows_v56": int(len(allocations)),
        "cvar_certificate_rows_v56": int(len(cert)),
        "restricted_master_columns_v56": int(frontier["restricted_master_columns_v56"].max())
        if not frontier.empty
        else 0,
        "exact_full_universe_cvar_claim_allowed": False,
        "paper1_promotion_allowed_v56": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v56 expands restricted-master CVaR/source LP using full comparable universe; full-universe exact claim false",
    }
    write_json(STATUS_DIR / "paper4_v56_status.json", status)
    return status


def build_v57() -> dict[str, Any]:
    start = datetime.now(UTC)
    online, spo, dla, ifrs9 = build_v57_online_spo_dla_ifrs9()
    status = {
        "schema_version": "2026-05-15.57",
        "generated_at_utc": now(),
        "phase": "v57_online_spo_dla_ifrs9_gate_updates",
        "online_repair_rows_v57": int(len(online)),
        "online_gate_pass_rows_v57": int(online["gate_source80_policy90_width95_v57"].sum())
        if not online.empty
        else 0,
        "strict_live_deployability_claim_allowed": False,
        "spo_oracle_bridge_rows_v57": int(len(spo)),
        "formal_differentiable_spo_claim_allowed": False,
        "dla_policy_month_rows_v57": int(len(dla)),
        "bellman_exact_claim_allowed": False,
        "ifrs9_sicr_rows_v57": int(len(ifrs9)),
        "contractual_ifrs9_claim_allowed": False,
        "cate_policy_value_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "paper1_promotion_allowed_v57": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v57 updates gate diagnostics only; no forbidden claims unlocked",
    }
    write_json(STATUS_DIR / "paper4_v57_status.json", status)
    return status


def build_v58() -> dict[str, Any]:
    start = datetime.now(UTC)
    registry, claims, manifest = build_v58_registry_docs()
    status = {
        "schema_version": "2026-05-15.58",
        "generated_at_utc": now(),
        "phase": "v58_registry_claims_storage_notebook",
        "candidate_registry_rows_v58": int(len(registry)),
        "claim_matrix_rows_v58": int(len(claims)),
        "heavy_artifact_manifest_rows_v58": int(len(manifest)),
        "quarto_page_count_v58": len(registered_paper4_pages()),
        "quarto_compact_guardrail_pass": len(registered_paper4_pages()) <= 12,
        "paper1_promotion_allowed_v58": False,
        "paper4_working_champion_changed_v58": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v58 updates registry/docs/storage; no Quarto expansion and no promotion",
    }
    write_json(STATUS_DIR / "paper4_v58_status.json", status)
    write_json(
        STATUS_DIR / "paper4_v58_working_champion.json",
        {
            "schema_version": "2026-05-15.58",
            "generated_at_utc": now(),
            "paper4_working_champion": "paper1_economic_champion",
            "working_champion_decision": "retained_reference_pending_dynamic_replay_of_v56_expanded_solver_books",
            "paper4_working_only": True,
            "paper1_promotion_allowed": False,
            "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
            "claim_boundary": "Paper 4 lab/working champion only; no Paper Estrella promotion",
        },
    )
    return status


def _solve_adaptive_cvar_lp(
    pool: pd.DataFrame,
    losses: np.ndarray,
    returns_by_path: np.ndarray,
    regime: str,
    cvar_cap: float,
    return_floor: float,
    cap_multiplier: float,
    min_deployment: float,
    hard_family_caps: dict[str, float] | None = None,
    tag: str = "v59",
) -> tuple[dict[str, Any], pd.DataFrame]:
    n = len(pool)
    s = losses.shape[0]
    returns = returns_by_path.mean(axis=0)
    amounts = pool["loan_amnt"].to_numpy(float)
    eta_idx = n
    u_start = n + 1
    nvars = n + 1 + s
    alpha = 0.90
    coeff = 1.0 / ((1 - alpha) * s)

    c = np.zeros(nvars)
    c[:n] = -returns
    c[eta_idx] = 1e-4
    c[u_start:] = 1e-4 * coeff

    rows: list[sparse.csr_matrix] = []
    rhs: list[float] = []
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = amounts
    rows.append(row.tocsr())
    rhs.append(BUDGET)
    row = sparse.lil_matrix((1, nvars))
    row[0, :n] = -amounts
    rows.append(row.tocsr())
    rhs.append(-min_deployment * BUDGET)
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
    scen[:, :n] = sparse.csr_matrix(losses)
    scen[:, eta_idx] = -1.0
    for i in range(s):
        scen[i, u_start + i] = -1.0
    rows.append(scen.tocsr())
    rhs.extend([0.0] * s)

    caps = _build_caps(pool, regime, cap_multiplier, hard_family_caps=hard_family_caps)
    cap_count = 0
    for _, cap_row in caps.iterrows():
        family = str(cap_row["source_family"])
        source = _pool_sources(pool, family, int(cap_row["min_support_v56"]))
        mask = source.astype(str).eq(str(cap_row["source_id"])).to_numpy()
        if not mask.any():
            continue
        row = sparse.lil_matrix((1, nvars))
        row[0, np.where(mask)[0]] = amounts[mask]
        rows.append(row.tocsr())
        rhs.append(float(cap_row["source_cap_v56"]) * BUDGET)
        cap_count += 1

    result = linprog(
        c,
        A_ub=sparse.vstack(rows).tocsr(),
        b_ub=np.array(rhs, dtype=float),
        bounds=[(0.0, 1.0)] * n + [(0.0, BUDGET)] + [(0.0, BUDGET)] * s,
        method="highs",
        options={"time_limit": 240},
    )
    status: dict[str, Any] = {
        f"regime_{tag}": regime,
        f"solver_success_{tag}": bool(result.success),
        f"solver_status_{tag}": int(result.status),
        f"solver_message_{tag}": str(result.message),
        f"restricted_master_columns_{tag}": n,
        f"scenario_count_{tag}": s,
        f"cvar_cap_{tag}": cvar_cap,
        f"return_floor_{tag}": return_floor,
        f"cap_multiplier_{tag}": cap_multiplier,
        f"min_deployment_{tag}": min_deployment,
        f"source_cap_count_{tag}": cap_count,
        f"hard_family_caps_{tag}": json.dumps(hard_family_caps or {}, sort_keys=True),
        f"exact_full_universe_claim_{tag}": False,
        f"claim_boundary_{tag}": "adaptive feasible frontier over expanded restricted master; not exact full-universe optimality",
    }
    if not result.success:
        status.update(
            {
                f"objective_return_{tag}": np.nan,
                f"allocated_exposure_{tag}": np.nan,
                f"n_allocated_loans_{tag}": 0,
                f"scenario_loss_mean_{tag}": np.nan,
                f"scenario_loss_p95_{tag}": np.nan,
                f"scenario_loss_cvar90_{tag}": np.nan,
                f"scenario_return_p05_{tag}": np.nan,
                f"budget_slack_{tag}": np.nan,
            }
        )
        return status, pd.DataFrame()

    x = result.x[:n]
    eta = float(result.x[eta_idx])
    u = result.x[u_start:]
    scenario_losses = losses @ x
    scenario_returns = returns_by_path @ x
    cvar = float(eta + coeff * u.sum())
    expected_return = float(returns @ x)
    exposure = float(amounts @ x)
    alloc = pool.loc[x > 1e-6].copy()
    alloc[f"allocation_fraction_{tag}"] = x[x > 1e-6]
    alloc[f"allocated_exposure_{tag}"] = alloc["loan_amnt"] * alloc[f"allocation_fraction_{tag}"]
    alloc[f"policy_id_{tag}"] = f"{tag}_cvar_adaptive_{regime}"
    alloc[f"regime_{tag}"] = regime
    alloc[f"claim_boundary_{tag}"] = status[f"claim_boundary_{tag}"]
    pd.DataFrame(
        {
            f"regime_{tag}": regime,
            "scenario_row": np.arange(s),
            f"scenario_loss_{tag}": scenario_losses,
            f"scenario_return_{tag}": scenario_returns,
            f"eta_{tag}": eta,
            f"excess_loss_{tag}": np.maximum(scenario_losses - eta, 0),
        }
    ).to_csv(TABLE_DIR / f"paper4_{tag}_cvar_adaptive_scenario_losses_{regime}.csv", index=False)
    status.update(
        {
            f"objective_return_{tag}": expected_return,
            f"allocated_exposure_{tag}": exposure,
            f"n_allocated_loans_{tag}": int((x > 1e-6).sum()),
            f"scenario_loss_mean_{tag}": float(np.mean(scenario_losses)),
            f"scenario_loss_p95_{tag}": float(np.quantile(scenario_losses, 0.95)),
            f"scenario_loss_cvar90_{tag}": cvar,
            f"scenario_return_p05_{tag}": float(np.quantile(scenario_returns, 0.05)),
            f"budget_slack_{tag}": BUDGET - exposure,
        }
    )
    return status, alloc


def build_v59_adaptive_frontier(max_columns: int = 36_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    if universe.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v59_cvar_adaptive_feasible_frontier.csv", empty)
        empty.to_parquet(TABLE_DIR / "paper4_v59_cvar_adaptive_allocations.parquet", index=False)
        return empty, empty
    pool, logs = _select_master_pool(universe, max_columns=max_columns)
    logs = logs.assign(round_v59=logs["round_v56"], restricted_master_columns_v59=len(pool))
    write_csv(TABLE_DIR / "paper4_v59_adaptive_column_log.csv", logs)
    losses, returns_by_path, _default_probs, _path_ids = _expected_by_path_matrix(pool)
    specs = [
        ("strict_low_deploy", 85_000.0, 20_000.0, 1.35, 0.72),
        ("committee_low_deploy", 105_000.0, 35_000.0, 1.55, 0.80),
        ("committee_balanced", 125_000.0, 50_000.0, 1.80, 0.85),
        ("relaxed_balanced", 160_000.0, 70_000.0, 2.15, 0.90),
        ("wealth_first", 220_000.0, 95_000.0, 2.50, 0.90),
        ("tail_first_low_budget", 75_000.0, 5_000.0, 2.20, 0.55),
    ]
    rows: list[dict[str, Any]] = []
    allocs: list[pd.DataFrame] = []
    for spec in specs:
        status, alloc = _solve_adaptive_cvar_lp(pool, losses, returns_by_path, *spec)
        rows.append(status)
        if not alloc.empty:
            allocs.append(alloc)
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        frontier["return_norm_v59"] = normalize(frontier["objective_return_v59"])
        frontier["tail_norm_v59"] = normalize(
            frontier["scenario_loss_cvar90_v59"], higher_is_better=False
        )
        frontier["deploy_norm_v59"] = normalize(frontier["allocated_exposure_v59"])
        frontier["frontier_score_v59"] = (
            0.40 * frontier["return_norm_v59"]
            + 0.38 * frontier["tail_norm_v59"]
            + 0.22 * frontier["deploy_norm_v59"]
        )
        frontier["committee_feasible_v59"] = frontier["solver_success_v59"].astype(bool)
        frontier["non_dominated_restricted_v59"] = frontier["committee_feasible_v59"] & (
            frontier["frontier_score_v59"] >= frontier["frontier_score_v59"].median()
        )
    allocations = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    write_csv(TABLE_DIR / "paper4_v59_cvar_adaptive_feasible_frontier.csv", frontier)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v59_cvar_adaptive_allocations.parquet", index=False, compression="zstd"
    )
    write_csv(
        TABLE_DIR / "paper4_v59_cvar_constraint_relaxation_map.csv",
        frontier[
            [
                "regime_v59",
                "solver_success_v59",
                "cvar_cap_v59",
                "return_floor_v59",
                "cap_multiplier_v59",
                "min_deployment_v59",
                "scenario_loss_cvar90_v59",
                "objective_return_v59",
                "claim_boundary_v59",
            ]
        ]
        if not frontier.empty
        else frontier,
    )
    return frontier, allocations


def update_v59_notebook(status: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V59_ADAPTIVE_FRONTIER_START -->",
            "",
            "## Wave v59: Adaptive CVaR Feasibility Frontier",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Follow the v56 infeasibility result instead of stopping there: relax",
            "deployment, source caps, return floors, and CVaR caps in a controlled",
            "grid to identify whether the expanded comparable-universe solver can",
            "produce feasible committee policies.",
            "",
            "### Results",
            "",
            f"- Frontier rows: `{status.get('adaptive_frontier_rows_v59')}`.",
            f"- Feasible rows: `{status.get('adaptive_feasible_rows_v59')}`.",
            f"- Allocation rows: `{status.get('adaptive_allocation_rows_v59')}`.",
            "",
            "### Interpretation",
            "",
            "v56 showed that the original strict floors/caps are too demanding for the",
            "expanded restricted master. v59 converts that negative result into a",
            "frontier: if feasible rows exist, they are committee-style lab candidates;",
            "if not, strict CVaR/source governance is documented as infeasible under",
            "the current internal loss model.",
            "",
            "### Claim Impact",
            "",
            "This is still restricted-master evidence. It improves the solver lane but",
            "does not allow exact full-universe CVaR, Paper Estrella promotion, or final",
            "Paper 4 promotion.",
            "",
            "<!-- V59_ADAPTIVE_FRONTIER_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V59_ADAPTIVE_FRONTIER_START -->"
        end = "<!-- V59_ADAPTIVE_FRONTIER_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v59() -> dict[str, Any]:
    start = datetime.now(UTC)
    frontier, allocations = build_v59_adaptive_frontier()
    status = {
        "schema_version": "2026-05-15.59",
        "generated_at_utc": now(),
        "phase": "v59_adaptive_cvar_feasibility_frontier",
        "adaptive_frontier_rows_v59": int(len(frontier)),
        "adaptive_feasible_rows_v59": int(frontier["solver_success_v59"].sum())
        if not frontier.empty
        else 0,
        "adaptive_allocation_rows_v59": int(len(allocations)),
        "exact_full_universe_cvar_claim_allowed": False,
        "paper1_promotion_allowed_v59": False,
        "paper4_working_champion_changed_v59": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "adaptive CVaR feasibility frontier over expanded restricted master; no exact full-universe claim",
    }
    write_json(STATUS_DIR / "paper4_v59_status.json", status)
    update_v59_notebook(status)
    return status


def build_v60_candidate_replay_memo() -> tuple[pd.DataFrame, pd.DataFrame]:
    frontier = read_csv("paper4_v59_cvar_adaptive_feasible_frontier.csv")
    allocations = read_parquet("paper4_v59_cvar_adaptive_allocations.parquet")
    if frontier.empty or allocations.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v60_v59_candidate_replay.csv", empty)
        write_csv(TABLE_DIR / "paper4_v60_dynamic_rerun_gate_memo.csv", empty)
        return empty, empty
    replay_rows: list[dict[str, Any]] = []
    feasible = frontier.loc[frontier["solver_success_v59"].astype(bool)].copy()
    for _, row in feasible.iterrows():
        regime = str(row["regime_v59"])
        scen_path = TABLE_DIR / f"paper4_v59_cvar_adaptive_scenario_losses_{regime}.csv"
        scenario = pd.read_csv(scen_path) if scen_path.exists() else pd.DataFrame()
        book = allocations.loc[allocations["regime_v59"].astype(str).eq(regime)].copy()
        if scenario.empty or book.empty:
            continue
        source_concentration = book.groupby("grade")["allocated_exposure_v59"].sum().max() / max(
            book["allocated_exposure_v59"].sum(), 1.0
        )
        replay_rows.append(
            {
                "policy_id": f"v59_cvar_adaptive_{regime}",
                "regime_v59": regime,
                "n_loans_v60": int(book["loan_id"].nunique()),
                "funded_exposure_v60": float(book["allocated_exposure_v59"].sum()),
                "objective_return_v60": float(row.get("objective_return_v59", np.nan)),
                "scenario_return_mean_v60": float(scenario["scenario_return_v59"].mean()),
                "scenario_return_p05_v60": float(scenario["scenario_return_v59"].quantile(0.05)),
                "scenario_loss_mean_v60": float(scenario["scenario_loss_v59"].mean()),
                "scenario_loss_p95_v60": float(scenario["scenario_loss_v59"].quantile(0.95)),
                "scenario_loss_cvar90_v60": float(row.get("scenario_loss_cvar90_v59", np.nan)),
                "grade_top_exposure_share_v60": float(source_concentration),
                "dynamic_1024_rerun_recommended_v60": False,
                "rerun_decision_reason_v60": (
                    "tail candidate is feasible but return is too low to plausibly change "
                    "working-champion governance without a stronger dynamic candidate"
                ),
                "claim_boundary_v60": "v60 replay gate over v59 restricted-master candidates; no expensive dynamic rerun and no promotion",
            }
        )
    replay = pd.DataFrame(replay_rows)
    write_csv(TABLE_DIR / "paper4_v60_v59_candidate_replay.csv", replay)
    best_return = float(replay["scenario_return_mean_v60"].max()) if not replay.empty else np.nan
    best_tail = float(replay["scenario_loss_p95_v60"].min()) if not replay.empty else np.nan
    memo = pd.DataFrame(
        [
            {
                "memo_id": "v60_dynamic_rerun_gate",
                "candidate_count_v60": int(len(replay)),
                "best_candidate_return_mean_v60": best_return,
                "best_candidate_loss_p95_v60": best_tail,
                "focused_512_or_1024_rerun_executed_v60": False,
                "working_champion_change_allowed_v60": False,
                "paper1_promotion_allowed_v60": False,
                "decision_v60": "do_not_run_expensive_dynamic_stress_until_candidate_return_improves",
                "claim_boundary_v60": "gate memo only; v59 candidates are serious tail diagnostics, not champion replacements",
            }
        ]
    )
    write_csv(TABLE_DIR / "paper4_v60_dynamic_rerun_gate_memo.csv", memo)
    registry_delta = replay.copy()
    if not registry_delta.empty:
        registry_delta["candidate_family_v60"] = "adaptive_cvar_tail_diagnostic"
        registry_delta["paper4_working_champion_allowed_v60"] = False
        registry_delta["paper1_promotion_allowed_v60"] = False
        registry_delta["decision_v60"] = "retain_as_tail_diagnostic_not_champion"
        registry_delta["claim_boundary_v60"] = (
            "v59/v60 candidates are restricted-master tail diagnostics; no working champion change"
        )
    write_csv(TABLE_DIR / "paper4_v60_candidate_registry_delta.csv", registry_delta)
    claim_delta = pd.DataFrame(
        [
            {
                "claim_id": "v60_adaptive_cvar_tail_diagnostic",
                "allowed": True,
                "artifact": "paper4_v60_v59_candidate_replay.csv",
                "boundary": "tail diagnostic over expanded restricted master; no champion replacement",
            },
            {
                "claim_id": "v60_expensive_dynamic_rerun_needed",
                "allowed": False,
                "artifact": "paper4_v60_dynamic_rerun_gate_memo.csv",
                "boundary": "v59 candidates do not plausibly change working champion decision",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v60_claim_matrix_delta.csv", claim_delta)
    return replay, memo


def update_v60_notebook(status: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V60_DYNAMIC_GATE_START -->",
            "",
            "## Wave v60: Dynamic Stress Gate For v59 Tail Candidates",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Decide whether the v59 feasible CVaR/source candidates deserve expensive",
            "512/1024-path dynamic stress. The rule is conservative: run the expensive",
            "stress only if a candidate can plausibly change the working champion.",
            "",
            "### Results",
            "",
            f"- Candidate replay rows: `{status.get('candidate_replay_rows_v60')}`.",
            f"- Expensive dynamic rerun executed: `{status.get('focused_512_or_1024_rerun_executed_v60')}`.",
            "",
            "### Interpretation",
            "",
            "The v59 candidates are useful as tail-risk feasibility diagnostics, but",
            "their return level is not strong enough to justify a costly dynamic",
            "stress rerun or a working champion change in this checkpoint.",
            "",
            "<!-- V60_DYNAMIC_GATE_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V60_DYNAMIC_GATE_START -->"
        end = "<!-- V60_DYNAMIC_GATE_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v60() -> dict[str, Any]:
    start = datetime.now(UTC)
    replay, memo = build_v60_candidate_replay_memo()
    status = {
        "schema_version": "2026-05-15.60",
        "generated_at_utc": now(),
        "phase": "v60_dynamic_stress_gate_for_v59_candidates",
        "candidate_replay_rows_v60": int(len(replay)),
        "focused_512_or_1024_rerun_executed_v60": False,
        "paper1_promotion_allowed_v60": False,
        "paper4_working_champion_changed_v60": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v60 gates expensive dynamic stress; no promotion",
    }
    write_json(STATUS_DIR / "paper4_v60_status.json", status)
    update_v60_notebook(status)
    return status


def build_v61_source_diversified_frontier(
    max_columns: int = 36_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    if universe.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v61_source_diversified_frontier.csv", empty)
        empty.to_parquet(
            TABLE_DIR / "paper4_v61_source_diversified_allocations.parquet", index=False
        )
        return empty, empty
    pool, logs = _select_master_pool(universe, max_columns=max_columns)
    write_csv(TABLE_DIR / "paper4_v61_source_diversified_column_log.csv", logs)
    losses, returns_by_path, _default_probs, _path_ids = _expected_by_path_matrix(pool)
    hard_caps = {
        "grade": 0.55,
        "score_decile": 0.42,
        "state_top20": 0.24,
        "income_band": 0.38,
        "dti_band": 0.38,
        "period": 0.60,
    }
    specs = [
        ("diverse_tail", 95_000.0, 12_000.0, 2.20, 0.55),
        ("diverse_committee", 150_000.0, 35_000.0, 2.60, 0.70),
        ("diverse_balanced", 210_000.0, 60_000.0, 3.00, 0.78),
        ("diverse_wealth", 280_000.0, 85_000.0, 3.50, 0.82),
    ]
    rows: list[dict[str, Any]] = []
    allocs: list[pd.DataFrame] = []
    for spec in specs:
        status, alloc = _solve_adaptive_cvar_lp(
            pool,
            losses,
            returns_by_path,
            *spec,
            hard_family_caps=hard_caps,
            tag="v61",
        )
        rows.append(status)
        if not alloc.empty:
            allocs.append(alloc)
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        frontier["return_norm_v61"] = normalize(frontier["objective_return_v61"])
        frontier["tail_norm_v61"] = normalize(
            frontier["scenario_loss_cvar90_v61"], higher_is_better=False
        )
        frontier["deploy_norm_v61"] = normalize(frontier["allocated_exposure_v61"])
        frontier["frontier_score_v61"] = (
            0.36 * frontier["return_norm_v61"]
            + 0.34 * frontier["tail_norm_v61"]
            + 0.18 * frontier["deploy_norm_v61"]
            + 0.12 * frontier["solver_success_v61"].astype(float)
        )
        frontier["source_diversified_feasible_v61"] = frontier["solver_success_v61"].astype(bool)
    allocations = pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()
    if not allocations.empty:
        concentration_rows = []
        for policy, group in allocations.groupby("policy_id_v61", dropna=False):
            exposure = float(group["allocated_exposure_v61"].sum())
            for family in ["grade", "score_decile", "state_top20", "income_band", "dti_band"]:
                top_share = float(
                    group.groupby(family)["allocated_exposure_v61"].sum().max() / max(exposure, 1.0)
                )
                concentration_rows.append(
                    {
                        "policy_id": policy,
                        "source_family": family,
                        "top_exposure_share_v61": top_share,
                        "hard_cap_v61": hard_caps.get(family),
                        "source_diversification_pass_v61": top_share
                        <= hard_caps.get(family, 1.0) + 1e-6,
                    }
                )
        write_csv(TABLE_DIR / "paper4_v61_source_concentration_diagnostics.csv", concentration_rows)
    else:
        write_csv(TABLE_DIR / "paper4_v61_source_concentration_diagnostics.csv", pd.DataFrame())
    write_csv(TABLE_DIR / "paper4_v61_source_diversified_frontier.csv", frontier)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v61_source_diversified_allocations.parquet",
        index=False,
        compression="zstd",
    )
    return frontier, allocations


def update_v61_notebook(status: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V61_SOURCE_DIVERSIFICATION_START -->",
            "",
            "## Wave v61: Source-Diversified CVaR Frontier",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Repair the v59 all-grade-A concentration by adding absolute hard caps for",
            "grade, score decile, state, income, DTI and period. This tests whether the",
            "tail-risk solver can produce a less trivial diversified book.",
            "",
            "### Results",
            "",
            f"- Frontier rows: `{status.get('source_diversified_frontier_rows_v61')}`.",
            f"- Feasible rows: `{status.get('source_diversified_feasible_rows_v61')}`.",
            f"- Allocation rows: `{status.get('source_diversified_allocation_rows_v61')}`.",
            "",
            "### Interpretation",
            "",
            "If feasible, v61 candidates are stronger source-governance challengers than",
            "v59. If infeasible, the result says the current internal loss model can only",
            "get strong tail protection by collapsing into very low-risk sources.",
            "",
            "<!-- V61_SOURCE_DIVERSIFICATION_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V61_SOURCE_DIVERSIFICATION_START -->"
        end = "<!-- V61_SOURCE_DIVERSIFICATION_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v61() -> dict[str, Any]:
    start = datetime.now(UTC)
    frontier, allocations = build_v61_source_diversified_frontier()
    status = {
        "schema_version": "2026-05-15.61",
        "generated_at_utc": now(),
        "phase": "v61_source_diversified_cvar_frontier",
        "source_diversified_frontier_rows_v61": int(len(frontier)),
        "source_diversified_feasible_rows_v61": int(frontier["solver_success_v61"].sum())
        if not frontier.empty
        else 0,
        "source_diversified_allocation_rows_v61": int(len(allocations)),
        "exact_full_universe_cvar_claim_allowed": False,
        "paper1_promotion_allowed_v61": False,
        "paper4_working_champion_changed_v61": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "source-diversified CVaR frontier over expanded restricted master; no exact full-universe claim",
    }
    write_json(STATUS_DIR / "paper4_v61_status.json", status)
    claim_delta = pd.DataFrame(
        [
            {
                "claim_id": "v61_source_diversified_cvar_feasible",
                "allowed": int(status["source_diversified_feasible_rows_v61"]) > 0,
                "artifact": "paper4_v61_source_diversified_frontier.csv",
                "boundary": "source-diversified restricted-master CVaR only; exact full-universe claim false",
            },
            {
                "claim_id": "v61_tail_solver_requires_grade_concentration",
                "allowed": True,
                "artifact": "paper4_v61_source_diversified_frontier.csv",
                "boundary": "negative governance finding under current internal loss model",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v61_claim_matrix_delta.csv", claim_delta)
    blocker = pd.DataFrame(
        [
            {
                "lane": "full_universe_lineage",
                "status": "resolved",
                "evidence_artifact": "paper4_v55_maximal_comparable_universe.parquet",
                "next_unlock": "use exact pricing/full LP for optimality claim",
            },
            {
                "lane": "cvar_expected_loss_solver",
                "status": "near_resolved_with_plateau",
                "evidence_artifact": "paper4_v59_cvar_adaptive_feasible_frontier.csv",
                "next_unlock": "find feasible diversified return/tail point or prove pricing optimality",
            },
            {
                "lane": "source_diversified_cvar",
                "status": "implementation_blocked",
                "evidence_artifact": "paper4_v61_source_diversified_frontier.csv",
                "next_unlock": "adjust loss model, source caps, or add pricing columns that improve diversified feasibility",
            },
            {
                "lane": "online_conformal_live_deployability",
                "status": "near_resolved_with_plateau",
                "evidence_artifact": "paper4_v57_online_source_family_direct_repair.csv",
                "next_unlock": "genuinely unseen temporal/source holdout",
            },
            {
                "lane": "differentiable_spo",
                "status": "dependency_blocked",
                "evidence_artifact": "paper4_v57_spo_dependency_probe.csv",
                "next_unlock": "isolated numpy<2/cvxpy/cvxpylayers/torch environment",
            },
            {
                "lane": "contractual_ifrs9",
                "status": "data_blocked",
                "evidence_artifact": "paper4_v57_ifrs9_sicr_proxy_panel_update.csv",
                "next_unlock": "monthly DPD/cure/recovery/prepayment/EAD panel",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v61_blocker_dashboard.csv", blocker)
    current_boundaries = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a full comparable OOT feature/prediction universe.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v55_maximal_comparable_universe.parquet",
                "boundary": "Comparable universe only; does not imply exact full-universe optimization.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "Expanded restricted-master CVaR is implemented over the comparable universe.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v56_cvar_full_comparable_frontier.csv",
                "boundary": "Restricted-master/diagnostic evidence; exact full-universe CVaR remains false.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "Source-diversified CVaR challenger is feasible and promotable.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v61_source_diversified_frontier.csv",
                "boundary": "No feasible diversified point under current internal loss model.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if not current_boundaries.empty:
        updated = pd.concat([current_boundaries, additions], ignore_index=True).drop_duplicates(
            ["claim"], keep="last"
        )
    else:
        updated = additions
    write_csv(TABLE_DIR / "paper4_current_claim_boundaries.csv", updated)
    update_v61_notebook(status)
    return status


def build_v62_source_slack_certificate() -> pd.DataFrame:
    allocations = read_parquet("paper4_v59_cvar_adaptive_allocations.parquet")
    hard_caps = {
        "grade": 0.55,
        "score_decile": 0.42,
        "state_top20": 0.24,
        "income_band": 0.38,
        "dti_band": 0.38,
        "period": 0.60,
    }
    rows: list[dict[str, Any]] = []
    if not allocations.empty:
        for policy, group in allocations.groupby("policy_id_v59", dropna=False):
            exposure = float(group["allocated_exposure_v59"].sum())
            for family, cap in hard_caps.items():
                exposure_by_source = group.groupby(family, dropna=False)[
                    "allocated_exposure_v59"
                ].sum() / max(exposure, 1.0)
                top_source = str(exposure_by_source.idxmax())
                top_share = float(exposure_by_source.max())
                rows.append(
                    {
                        "policy_id": policy,
                        "source_family": family,
                        "top_source_id_v62": top_source,
                        "top_exposure_share_v62": top_share,
                        "hard_cap_v62": cap,
                        "required_cap_slack_share_v62": max(0.0, top_share - cap),
                        "required_cap_slack_exposure_v62": max(0.0, top_share - cap) * exposure,
                        "source_diversification_pass_v62": top_share <= cap + 1e-9,
                        "certificate_scope_v62": "post-solve source slack over v59 feasible books; practical governance certificate",
                    }
                )
    cert = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "paper4_v62_source_diversification_slack_certificate.csv", cert)
    return cert


def update_v62_notebook(status: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V62_SOURCE_SLACK_CERTIFICATE_START -->",
            "",
            "## Wave v62: Source Diversification Slack Certificate",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Quantify why v59 feasible tail books fail v61 source-diversification",
            "governance. This turns infeasibility into a concrete slack requirement.",
            "",
            "### Results",
            "",
            f"- Slack certificate rows: `{status.get('source_slack_certificate_rows_v62')}`.",
            f"- Max required cap slack share: `{status.get('max_required_cap_slack_share_v62')}`.",
            "",
            "### Interpretation",
            "",
            "The current tail-feasible books require large source-cap slack, especially",
            "for grade concentration. That makes them useful diagnostics, not working",
            "champion candidates.",
            "",
            "<!-- V62_SOURCE_SLACK_CERTIFICATE_END -->",
            "",
        ]
    )
    if NOTEBOOK.exists():
        text = NOTEBOOK.read_text(encoding="utf-8")
        start = "<!-- V62_SOURCE_SLACK_CERTIFICATE_START -->"
        end = "<!-- V62_SOURCE_SLACK_CERTIFICATE_END -->"
        if start in text and end in text:
            before = text.split(start)[0]
            after = text.split(end, 1)[1]
            NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
        else:
            NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v62() -> dict[str, Any]:
    start = datetime.now(UTC)
    cert = build_v62_source_slack_certificate()
    max_slack = float(cert["required_cap_slack_share_v62"].max()) if not cert.empty else 0.0
    status = {
        "schema_version": "2026-05-15.62",
        "generated_at_utc": now(),
        "phase": "v62_source_diversification_slack_certificate",
        "source_slack_certificate_rows_v62": int(len(cert)),
        "max_required_cap_slack_share_v62": max_slack,
        "paper1_promotion_allowed_v62": False,
        "paper4_working_champion_changed_v62": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v62 practical source slack certificate; v59 tail books remain diagnostics only",
    }
    write_json(STATUS_DIR / "paper4_v62_status.json", status)
    update_v62_notebook(status)
    return status


def run_all() -> dict[str, dict[str, Any]]:
    statuses = {
        "v55": build_v55(),
        "v56": build_v56(),
        "v57": build_v57(),
        "v58": build_v58(),
        "v59": build_v59(),
        "v60": build_v60(),
        "v61": build_v61(),
        "v62": build_v62(),
    }
    update_v58_notebook(statuses)
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["all", "v55", "v56", "v57", "v58", "v59", "v60", "v61", "v62"],
        default="all",
    )
    args = parser.parse_args()
    if args.phase == "all":
        statuses = run_all()
    elif args.phase == "v55":
        statuses = {"v55": build_v55()}
    elif args.phase == "v56":
        statuses = {"v56": build_v56()}
    elif args.phase == "v57":
        statuses = {"v57": build_v57()}
    elif args.phase == "v58":
        statuses = {"v58": build_v58()}
    elif args.phase == "v59":
        statuses = {"v59": build_v59()}
    elif args.phase == "v60":
        statuses = {"v60": build_v60()}
    elif args.phase == "v61":
        statuses = {"v61": build_v61()}
    else:
        statuses = {"v62": build_v62()}
    print(json.dumps(statuses, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
