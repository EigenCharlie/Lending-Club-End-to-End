"""Compare baseline vs CATE-adjusted portfolio optimization.

Primary input for deployment is `data/processed/cate_estimates_oot.parquet`.
Training CATE estimates are only used as an explicit fallback when OOT estimates
are missing or partially incomplete.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.optimization.causal_portfolio import build_cate_adjusted_portfolio
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

CAUSAL_PORTFOLIO_SCHEMA_VERSION = "2026-03-07.1"


def _artifact_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    root = str(os.environ.get("GPU_REPLAY_ARTIFACT_ROOT", "")).strip()
    return (Path(root) / path) if root else path


def _parse_percent_series(s: pd.Series, default: float = 0.12) -> np.ndarray:
    """Convert a column that may be string '12.5%' or float 12.5 to decimal."""
    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy(dtype=float)
        if np.nanmedian(arr) > 1:
            arr = arr / 100.0
        return np.nan_to_num(arr, nan=default)
    return (
        s.astype(str)
        .str.strip()
        .str.rstrip("%")
        .pipe(pd.to_numeric, errors="coerce")
        .div(100)
        .fillna(default)
        .to_numpy(dtype=float)
    )


_CLI_RUN_TAG: str | None = None


def _resolve_run_tag() -> str:
    if _CLI_RUN_TAG:
        return _CLI_RUN_TAG
    candidates: list[str | None] = []
    for candidate_path in [
        Path("models/causal_effect_status.json"),
        Path("models/causal_policy_rule.json"),
        Path("models/causal_policy_oot_status.json"),
    ]:
        if not candidate_path.exists():
            continue
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.append(payload.get("run_tag"))
    primary = next((candidate for candidate in candidates if candidate), None)
    fallbacks = [candidate for candidate in candidates[1:] if candidate] if primary else []
    return resolve_run_tag(primary, fallback_candidates=fallbacks, require_explicit=True)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fallback_grade_map(train_cate: pd.DataFrame) -> dict[str, float]:
    if "grade" not in train_cate.columns or "cate" not in train_cate.columns:
        return {}
    frame = train_cate.copy()
    frame["grade"] = frame["grade"].astype(str).fillna("UNKNOWN")
    frame["cate"] = pd.to_numeric(frame["cate"], errors="coerce")
    frame = frame.dropna(subset=["cate"])
    if frame.empty:
        return {}
    return frame.groupby("grade", observed=True)["cate"].median().to_dict()


def _align_cate_to_test(test_df: pd.DataFrame, data_dir: Path) -> tuple[pd.Series, dict[str, Any]]:
    oot_path = data_dir / "cate_estimates_oot.parquet"
    train_path = data_dir / "cate_estimates.parquet"
    warning = ""
    alignment_strategy = "zero_fill_fallback"
    source_cate_artifact = None
    n_missing_cate = int(len(test_df))
    grade_map: dict[str, float] = {}
    train_cate = pd.DataFrame()

    if train_path.exists():
        train_cate = pd.read_parquet(train_path)
        grade_map = _fallback_grade_map(train_cate)

    if oot_path.exists():
        oot_cate = pd.read_parquet(oot_path)
        source_cate_artifact = str(oot_path)
        if "id" in test_df.columns and "id" in oot_cate.columns:
            aligned = (
                test_df[["id"]]
                .assign(id=lambda frame: frame["id"].astype(str))
                .merge(
                    oot_cate.assign(id=oot_cate["id"].astype(str))[["id", "cate"]],
                    on="id",
                    how="left",
                )
            )
            cate = pd.to_numeric(aligned["cate"], errors="coerce")
            n_missing_cate = int(cate.isna().sum())
            alignment_strategy = "id_join_oot_cate"
        elif len(oot_cate) == len(test_df):
            cate = pd.to_numeric(oot_cate["cate"], errors="coerce")
            n_missing_cate = int(cate.isna().sum())
            alignment_strategy = "row_index_oot_cate_fallback"
            warning = "OOT CATE artifact lacked ids; aligned by row index."
        else:
            cate = pd.Series(np.nan, index=test_df.index, dtype=float)
            warning = "OOT CATE artifact exists but could not be aligned to test candidates."
    elif not train_cate.empty:
        source_cate_artifact = str(train_path)
        if "id" in test_df.columns and "id" in train_cate.columns:
            aligned = (
                test_df[["id"]]
                .assign(id=lambda frame: frame["id"].astype(str))
                .merge(
                    train_cate.assign(id=train_cate["id"].astype(str))[["id", "cate"]],
                    on="id",
                    how="left",
                )
            )
            cate = pd.to_numeric(aligned["cate"], errors="coerce")
            n_direct = int(cate.notna().sum())
            n_missing_cate = int(cate.isna().sum())
            alignment_strategy = "id_join_train_cate_fallback"
            warning = (
                "Missing OOT CATE artifact; using training CATE fallback "
                f"({n_direct} direct id matches)."
            )
        else:
            cate = pd.Series(np.nan, index=test_df.index, dtype=float)
            warning = "Missing OOT CATE artifact; falling back to training-grade medians."
    else:
        cate = pd.Series(np.nan, index=test_df.index, dtype=float)
        warning = "Missing both OOT and training CATE artifacts; using zero CATE."

    if cate.isna().any() and grade_map and "grade" in test_df.columns:
        grade_fill = test_df["grade"].astype(str).map(grade_map)
        fill_mask = cate.isna() & grade_fill.notna()
        cate.loc[fill_mask] = grade_fill.loc[fill_mask]
        if alignment_strategy == "id_join_oot_cate":
            alignment_strategy = "id_join_oot_cate_plus_grade_fallback"
        elif "fallback" not in alignment_strategy:
            alignment_strategy = f"{alignment_strategy}_plus_grade_fallback"

    n_missing_cate = int(cate.isna().sum())
    if n_missing_cate:
        cate = cate.fillna(0.0)
        if not warning:
            warning = f"{n_missing_cate} candidates had missing CATE and were zero-filled."
        elif "zero" not in warning.lower():
            warning = f"{warning} Remaining missing CATE values were zero-filled."

    return cate.reset_index(drop=True), {
        "alignment_strategy": alignment_strategy,
        "source_cate_artifact": source_cate_artifact,
        "n_missing_cate": n_missing_cate,
        "warning": warning,
    }


def _constraint_binding_reason(result: dict[str, Any]) -> str | None:
    baseline = result.get("baseline", {})
    adjusted = result.get("cate_adjusted", {})
    if int(baseline.get("n_funded", 0)) == 0 and int(adjusted.get("n_funded", 0)) == 0:
        return "no_feasible_loans_under_current_budget_or_pd_constraints"
    if int(adjusted.get("n_funded", 0)) == 0:
        return "cate_adjusted_solution_empty_under_current_constraints"
    if int(baseline.get("n_funded", 0)) == 0:
        return "baseline_solution_empty_under_current_constraints"
    return None


def _build_segment_keys(test_df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for col in ("grade", "term", "verification_status"):
        if col in test_df.columns:
            parts.append(test_df[col].astype(str).fillna("UNKNOWN"))
    if not parts:
        return pd.Series(["GLOBAL"] * len(test_df), index=test_df.index, dtype="object")
    key = parts[0].astype(str)
    for part in parts[1:]:
        key = key.str.cat(part.astype(str), sep="__")
    return key


def _shrink_cate(test_df: pd.DataFrame, cate_series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    cate = pd.to_numeric(cate_series, errors="coerce").fillna(0.0).astype(float)
    if cate.empty:
        return cate, {
            "winsor_p05": 0.0,
            "winsor_p95": 0.0,
            "weak_signal_floor": 0.0,
            "shrink_weight_segment_median": 0.5,
        }

    p05 = float(np.quantile(cate, 0.05))
    p95 = float(np.quantile(cate, 0.95))
    clipped = cate.clip(lower=p05, upper=p95)

    segments = _build_segment_keys(test_df)
    segment_median = clipped.groupby(segments, observed=True).transform("median")
    shrunk = 0.5 * clipped + 0.5 * segment_median
    weak_signal_floor = float(max(abs(np.quantile(shrunk, 0.10)), 0.0025))
    shrunk = shrunk.mask(shrunk.abs() < weak_signal_floor, 0.0)

    return shrunk.reset_index(drop=True), {
        "winsor_p05": p05,
        "winsor_p95": p95,
        "weak_signal_floor": weak_signal_floor,
        "shrink_weight_segment_median": 0.5,
    }


def main(
    delta_rate: float = -1.0,
    total_budget: float = 1_000_000,
    max_portfolio_pd: float = 0.10,
    max_candidates: int = 5_000,
    uncertainty_aversion: float = 0.0,
    solver_backend: str = "highs",
) -> None:
    input_data_dir = Path("data/processed")
    output_data_dir = _artifact_path("data/processed")
    test_path = input_data_dir / "test_fe.parquet"
    intervals_path = input_data_dir / "conformal_intervals_mondrian.parquet"
    for path in [test_path, intervals_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    test_df = pd.read_parquet(test_path)
    intervals = pd.read_parquet(intervals_path)
    if "id" in test_df.columns:
        test_df = test_df.assign(id=test_df["id"].astype(str))
    cate_series, cate_meta = _align_cate_to_test(test_df, input_data_dir)
    shrunk_cate, shrink_meta = _shrink_cate(test_df, cate_series)
    logger.info(
        f"Loaded test={len(test_df):,} intervals={len(intervals):,} "
        f"with alignment={cate_meta['alignment_strategy']}"
    )

    max_candidates_norm = None if int(max_candidates) <= 0 else int(max_candidates)

    # Align intervals to test_df using _row_number when available; fall back to positional.
    if "_row_number" in intervals.columns:
        test_df = test_df.reset_index(drop=True).copy()
        test_df["_row_number"] = np.arange(len(test_df))
        merged_ints = test_df.merge(intervals, on="_row_number", how="inner", suffixes=("", "_int"))
        assert len(merged_ints) == len(test_df), (
            f"_row_number merge size mismatch: {len(merged_ints)} != {len(test_df)}"
        )
        intervals = merged_ints[
            [c for c in intervals.columns if c in merged_ints.columns]
        ].reset_index(drop=True)
        test_df = test_df.drop(columns=["_row_number"])
        logger.info(f"Aligned CATE portfolio test and intervals by _row_number: n={len(test_df):,}")

    n = min(len(test_df), len(intervals), len(cate_series))
    if max_candidates_norm is not None:
        n = min(n, max_candidates_norm)
    logger.info(
        f"Using {n:,} candidates "
        f"(max_candidates={'full' if max_candidates_norm is None else max_candidates_norm})"
    )
    test_df = test_df.iloc[:n].reset_index(drop=True)
    intervals = intervals.iloc[:n].reset_index(drop=True)
    cate_series = cate_series.iloc[:n].reset_index(drop=True)
    shrunk_cate = shrunk_cate.iloc[:n].reset_index(drop=True)

    pd_col = next(
        (c for c in ["pd_calibrated", "y_pred"] if c in intervals.columns), intervals.columns[0]
    )
    low_col = next((c for c in ["pd_low", "pd_low_90"] if c in intervals.columns), None)
    high_col = next((c for c in ["pd_high", "pd_high_90"] if c in intervals.columns), None)
    pd_point = intervals[pd_col].to_numpy(dtype=float)
    pd_low = intervals[low_col].to_numpy(dtype=float) if low_col else pd_point * 0.8
    pd_high = intervals[high_col].to_numpy(dtype=float) if high_col else pd_point * 1.3
    cate_raw = cate_series.to_numpy(dtype=float)
    cate = shrunk_cate.to_numpy(dtype=float)
    lgd = np.full(n, 0.45)
    int_rates = (
        _parse_percent_series(test_df["int_rate"])
        if "int_rate" in test_df.columns
        else np.full(n, 0.12)
    )

    result_raw = build_cate_adjusted_portfolio(
        loans=test_df,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        cate=cate_raw,
        lgd=lgd,
        int_rates=int_rates,
        delta_rate=delta_rate,
        total_budget=total_budget,
        max_portfolio_pd=max_portfolio_pd,
        uncertainty_aversion=uncertainty_aversion,
        solver_backend=solver_backend,
    )
    result = build_cate_adjusted_portfolio(
        loans=test_df,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        cate=cate,
        lgd=lgd,
        int_rates=int_rates,
        delta_rate=delta_rate,
        total_budget=total_budget,
        max_portfolio_pd=max_portfolio_pd,
        uncertainty_aversion=uncertainty_aversion,
        solver_backend=solver_backend,
    )

    baseline = result["baseline"]
    adjusted = result["cate_adjusted"]
    feasible_baseline = bool(int(baseline["n_funded"]) > 0)
    feasible_adjusted = bool(int(adjusted["n_funded"]) > 0)
    promotion_eligible = bool(
        feasible_baseline
        and feasible_adjusted
        and float(adjusted["objective_value"]) >= float(baseline["objective_value"])
        and int(adjusted["n_funded"]) >= 0.9 * int(baseline["n_funded"])
    )
    fallback_applied = not promotion_eligible
    selected_mode = "shrunk" if promotion_eligible else "research_only_fallback"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_data_dir / "cate_portfolio_comparison.parquet"
    comparison_df = pd.concat(
        [
            result_raw["comparison_df"].assign(cate_policy_mode="raw"),
            result["comparison_df"].assign(cate_policy_mode="shrunk"),
        ],
        ignore_index=True,
    )
    comparison_df.to_parquet(comparison_path, index=False)
    logger.info(f"Saved comparison: {comparison_path}")

    run_tag = _resolve_run_tag()
    binding_reason = _constraint_binding_reason(result)
    status = {
        "delta_rate": delta_rate,
        "baseline_objective": baseline["objective_value"],
        "cate_adjusted_objective": adjusted["objective_value"],
        "baseline_n_funded": baseline["n_funded"],
        "cate_adjusted_n_funded": adjusted["n_funded"],
        "objective_change_pct": float(
            (adjusted["objective_value"] - baseline["objective_value"])
            / (abs(baseline["objective_value"]) + 1e-6)
            * 100
        ),
        "n_candidates_available": int(min(len(test_df), len(intervals), len(cate_series))),
        "n_candidates_used": int(n),
        "max_candidates_requested": None if max_candidates_norm is None else max_candidates_norm,
        "dataset_scope": "full_candidates" if max_candidates_norm is None else "sampled_candidates",
        "solver_backend": str(solver_backend),
        "feasible_baseline": feasible_baseline,
        "feasible_adjusted": feasible_adjusted,
        "cate_policy_mode": selected_mode,
        "promotion_eligible": promotion_eligible,
        "fallback_applied": fallback_applied,
        "raw_objective_change_pct": float(
            (
                result_raw["cate_adjusted"]["objective_value"]
                - result_raw["baseline"]["objective_value"]
            )
            / (abs(result_raw["baseline"]["objective_value"]) + 1e-6)
            * 100
        ),
        "shrunk_objective_change_pct": float(
            (adjusted["objective_value"] - baseline["objective_value"])
            / (abs(baseline["objective_value"]) + 1e-6)
            * 100
        ),
        "alignment_strategy": cate_meta["alignment_strategy"],
        "source_cate_artifact": cate_meta["source_cate_artifact"],
        "n_missing_cate": int(cate_meta["n_missing_cate"]),
        "cate_shrink": shrink_meta,
        "constraint_binding_reason": binding_reason,
        "role": "insights_only" if not promotion_eligible else "operational_candidate",
        "promotion_decider": "formal_optimizer_objective",
        "policy_evaluation_consistent": bool(promotion_eligible),
        "warning": cate_meta["warning"]
        or (
            "Integracion causal no utilizable en este run."
            if binding_reason == "no_feasible_loans_under_current_budget_or_pd_constraints"
            else ("CATE portfolio left in research-only fallback mode." if fallback_applied else "")
        ),
        "source_effect_status_path": "models/causal_effect_status.json",
        **build_artifact_metadata(
            schema_version=CAUSAL_PORTFOLIO_SCHEMA_VERSION,
            run_tag=run_tag,
            require_explicit=True,
        ),
    }

    effect_status = _load_json_if_exists(Path("models/causal_effect_status.json"))
    if effect_status:
        status["effect_status_run_tag"] = effect_status.get("run_tag")

    status_path = _artifact_path("models/cate_portfolio_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    logger.info(f"Saved status: {status_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CATE-adjusted portfolio comparison")
    parser.add_argument("--delta_rate", type=float, default=-1.0)
    parser.add_argument("--total_budget", type=float, default=1_000_000)
    parser.add_argument("--max_portfolio_pd", type=float, default=0.10)
    parser.add_argument("--max_candidates", type=int, default=5_000)
    parser.add_argument("--uncertainty_aversion", type=float, default=0.0)
    parser.add_argument("--solver_backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument("--run-tag", default=None, help="Override run_tag on output artifacts")
    args = parser.parse_args()
    if args.run_tag:
        _CLI_RUN_TAG = args.run_tag
    main(
        delta_rate=args.delta_rate,
        total_budget=args.total_budget,
        max_portfolio_pd=args.max_portfolio_pd,
        max_candidates=args.max_candidates,
        uncertainty_aversion=args.uncertainty_aversion,
        solver_backend=args.solver_backend,
    )
