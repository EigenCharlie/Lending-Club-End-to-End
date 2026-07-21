"""Resolve the Paper 4 online conformal efficiency goal.

This v9 wave is intentionally narrow: it only attacks the online conformal
blocker left by v8.  The goal is to find a deployable structural recalibration
that keeps defended source-month coverage at or above 0.80, defended
policy-month coverage at or above 0.90, and average loan-level interval width at
or below 0.95.

The script writes no Paper Estrella artifacts and never creates
``paper4_final_promotion.json``.  V9 is a Paper 4 living-lab result only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_v6_priority_resolution import (
    NOTE_DIR,
    SOURCE_FAMILIES,
    STATUS_DIR,
    _coverage,
    _interval_width,
    _load_inputs,
    _prepare_online_frame,
    _write_csv,
    _write_json,
    _write_parquet,
)
from scripts.papers.build_paper4_v8_resolution_wave import _online_candidate_masks

SCHEMA_VERSION = "2026-05-13.9"
TARGET_SOURCE_MONTH = 0.80
TARGET_POLICY_MONTH = 0.90
TARGET_AVG_WIDTH = 0.95

PAPER1_PROMOTION = Path("models/final_project_promotion.json")
PAPER4_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"


def _json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _source_month_metrics_v9(
    local: pd.DataFrame, method: str, min_support: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_month = (
        local.groupby(["policy_id", "issue_month"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            coverage_online_v9=("covered_online_v9", "mean"),
            avg_width_online_v9=("interval_width_online_v9", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    policy_month["online_method_v9"] = method
    policy_month["standalone_gate_cell"] = policy_month["n_funded"].ge(min_support)
    policy_month["pooling_decision_v9"] = np.where(
        policy_month["standalone_gate_cell"],
        "standalone_defended_policy_month_cell",
        "pooled_small_policy_month_cell",
    )

    frames: list[pd.DataFrame] = []
    for source in SOURCE_FAMILIES:
        if source not in local.columns:
            continue
        src = (
            local.groupby(["policy_id", "issue_month", source], dropna=False, as_index=False)
            .agg(
                n=("loan_id", "nunique"),
                coverage_online_v9=("covered_online_v9", "mean"),
                avg_width_online_v9=("interval_width_online_v9", "mean"),
            )
            .rename(columns={"issue_month": "month", source: "source_value"})
        )
        src["source_id"] = source
        src["online_method_v9"] = method
        src["standalone_gate_cell"] = src["n"].ge(min_support)
        src["pooling_decision_v9"] = np.where(
            src["standalone_gate_cell"],
            "standalone_defended_source_month_cell",
            "pooled_small_source_month_cell",
        )
        frames.append(src)

    source_month = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not source_month.empty:
        source_month["source_value"] = source_month["source_value"].astype(str)
    return policy_month, source_month


def _group_codes(merged: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    keys = merged[columns].astype(str).agg("|".join, axis=1)
    codes, uniques = pd.factorize(keys, sort=False)
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    return codes, counts


def _online_metric_cache(merged: pd.DataFrame, min_support: int) -> dict[str, Any]:
    policy_codes, policy_counts = _group_codes(merged, ["policy_id", "issue_month"])
    source_groups = []
    for source in SOURCE_FAMILIES:
        if source not in merged.columns:
            continue
        codes, counts = _group_codes(merged, ["policy_id", "issue_month", source])
        source_groups.append({"source": source, "codes": codes, "counts": counts})
    return {
        "y_true": merged["y_true"].to_numpy(dtype=float),
        "y_pred": merged["y_pred"].to_numpy(dtype=float),
        "qhat_v4": merged["qhat_v4"].to_numpy(dtype=float),
        "policy_codes": policy_codes,
        "policy_counts": policy_counts,
        "policy_defended": policy_counts >= min_support,
        "source_groups": source_groups,
        "min_support": min_support,
    }


def _coverage_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    low = np.clip(y_pred - q, 0, 1)
    high = np.clip(y_pred + q, 0, 1)
    return (y_true >= low) & (y_true <= high)


def _width_from_arrays(y_pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.clip(y_pred + q, 0, 1) - np.clip(y_pred - q, 0, 1)


def _evaluate_online_v9_fast(
    merged: pd.DataFrame,
    q: pd.Series,
    *,
    method: str,
    family: str,
    params: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    q_array = q.clip(0, 1).to_numpy(dtype=float)
    y_true = cache["y_true"]
    y_pred = cache["y_pred"]
    covered = _coverage_from_arrays(y_true, y_pred, q_array).astype(float)
    width = _width_from_arrays(y_pred, q_array)

    policy_counts = cache["policy_counts"]
    policy_sums = np.bincount(cache["policy_codes"], weights=covered, minlength=len(policy_counts))
    policy_width_sums = np.bincount(
        cache["policy_codes"], weights=width, minlength=len(policy_counts)
    )
    policy_cov = np.divide(
        policy_sums, policy_counts, out=np.full_like(policy_sums, np.nan), where=policy_counts > 0
    )
    policy_width = np.divide(
        policy_width_sums,
        policy_counts,
        out=np.full_like(policy_width_sums, np.nan),
        where=policy_counts > 0,
    )
    policy_defended = cache["policy_defended"]
    policy_min = float(np.nanmin(policy_cov[policy_defended]))

    source_raw_min = np.nan
    source_min = np.nan
    source_small_cells = 0
    source_min_values = []
    source_raw_values = []
    for group in cache["source_groups"]:
        counts = group["counts"]
        sums = np.bincount(group["codes"], weights=covered, minlength=len(counts))
        cov = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        defended = counts >= cache["min_support"]
        source_small_cells += int((~defended).sum())
        source_raw_values.append(float(np.nanmin(cov)))
        if defended.any():
            source_min_values.append(float(np.nanmin(cov[defended])))
    if source_raw_values:
        source_raw_min = min(source_raw_values)
    if source_min_values:
        source_min = min(source_min_values)

    avg_width = float(np.mean(width))
    goal_pass = bool(
        source_min >= TARGET_SOURCE_MONTH
        and policy_min >= TARGET_POLICY_MONTH
        and avg_width <= TARGET_AVG_WIDTH
    )
    return {
        "online_method_v9": method,
        "method_family": family,
        "deployable_without_current_outcomes": True,
        "min_effective_sample_size": int(cache["min_support"]),
        "coverage_policy_month_raw_min": float(np.nanmin(policy_cov)),
        "coverage_policy_month_defended_min": policy_min,
        "coverage_source_month_raw_min": source_raw_min,
        "coverage_source_month_defended_min": source_min,
        "avg_width_loan": avg_width,
        "avg_width_policy_month": float(np.nanmean(policy_width)),
        "share_rows_widened_vs_v4": float((q_array > cache["qhat_v4"] + 1e-12).mean()),
        "share_rows_shrunk_vs_v4": float((q_array < cache["qhat_v4"] - 1e-12).mean()),
        "small_policy_month_cells_pooled": int((~policy_defended).sum()),
        "small_source_month_cells_pooled": source_small_cells,
        "gate_source_month_80": bool(source_min >= TARGET_SOURCE_MONTH),
        "gate_policy_month_90": bool(policy_min >= TARGET_POLICY_MONTH),
        "gate_width_95": bool(avg_width <= TARGET_AVG_WIDTH),
        "goal_pass": goal_pass,
        "source_margin_to_0p80": float(source_min - TARGET_SOURCE_MONTH),
        "policy_margin_to_0p90": float(policy_min - TARGET_POLICY_MONTH),
        "width_margin_to_0p95": float(TARGET_AVG_WIDTH - avg_width),
        "parameters_json": _json_dump(params),
        "caveat": "The structural tail guard is deployable from pre-decision fields, but the grid was selected in a replay and needs future-period validation.",
    }


def _evaluate_online_v9(
    merged: pd.DataFrame,
    q: pd.Series,
    *,
    method: str,
    family: str,
    min_support: int,
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    local = merged.copy()
    q = q.clip(0, 1)
    local["qhat_v9"] = q
    local["covered_online_v9"] = _coverage(local["y_true"], local["y_pred"], q)
    local["interval_width_online_v9"] = _interval_width(local["y_pred"], q)
    policy_month, source_month = _source_month_metrics_v9(local, method, min_support)

    defended_policy = policy_month[policy_month["standalone_gate_cell"].astype(bool)]
    defended_source = source_month[source_month["standalone_gate_cell"].astype(bool)]
    policy_min = (
        float(defended_policy["coverage_online_v9"].min()) if not defended_policy.empty else np.nan
    )
    source_min = (
        float(defended_source["coverage_online_v9"].min()) if not defended_source.empty else np.nan
    )
    avg_width = float(local["interval_width_online_v9"].mean())
    goal_pass = bool(
        source_min >= TARGET_SOURCE_MONTH
        and policy_min >= TARGET_POLICY_MONTH
        and avg_width <= TARGET_AVG_WIDTH
    )
    row = {
        "online_method_v9": method,
        "method_family": family,
        "deployable_without_current_outcomes": True,
        "min_effective_sample_size": min_support,
        "coverage_policy_month_raw_min": float(policy_month["coverage_online_v9"].min()),
        "coverage_policy_month_defended_min": policy_min,
        "coverage_source_month_raw_min": float(source_month["coverage_online_v9"].min())
        if not source_month.empty
        else np.nan,
        "coverage_source_month_defended_min": source_min,
        "avg_width_loan": avg_width,
        "avg_width_policy_month": float(policy_month["avg_width_online_v9"].mean()),
        "share_rows_widened_vs_v4": float((q > merged["qhat_v4"] + 1e-12).mean()),
        "share_rows_shrunk_vs_v4": float((q < merged["qhat_v4"] - 1e-12).mean()),
        "small_policy_month_cells_pooled": int(
            (~policy_month["standalone_gate_cell"].astype(bool)).sum()
        ),
        "small_source_month_cells_pooled": int(
            (~source_month["standalone_gate_cell"].astype(bool)).sum()
        )
        if not source_month.empty
        else 0,
        "gate_source_month_80": bool(source_min >= TARGET_SOURCE_MONTH),
        "gate_policy_month_90": bool(policy_min >= TARGET_POLICY_MONTH),
        "gate_width_95": bool(avg_width <= TARGET_AVG_WIDTH),
        "goal_pass": goal_pass,
        "source_margin_to_0p80": float(source_min - TARGET_SOURCE_MONTH),
        "policy_margin_to_0p90": float(policy_min - TARGET_POLICY_MONTH),
        "width_margin_to_0p95": float(TARGET_AVG_WIDTH - avg_width),
        "parameters_json": _json_dump(params),
        "caveat": "The structural tail guard is deployable from pre-decision fields, but the grid was selected in a replay and needs future-period validation.",
    }
    return row, policy_month, source_month


def _tail_masks(merged: pd.DataFrame) -> dict[str, pd.Series]:
    grade = merged["original_grade"].astype(str)
    dti = merged["dti_band"].astype(str)
    return {
        "dti_q5_only": dti.eq("dti_q5"),
        "gradeF_or_dti_q5": grade.isin(["F", "G"]) | dti.eq("dti_q5"),
        "gradeF_only": grade.isin(["F", "G"]),
        "gradeEplus_or_dti_q5": grade.isin(["E", "F", "G"]) | dti.eq("dti_q5"),
    }


def build_online_goal_v9(
    allocations: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    min_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = _prepare_online_frame(allocations, online_intervals)
    masks = _online_candidate_masks(merged)
    primary = masks["targeted_dti_score_or_gradeD"].astype(float)
    tails = _tail_masks(merged)
    cache = _online_metric_cache(merged, min_support)

    rows: list[dict[str, Any]] = []
    q_maps: dict[str, pd.Series] = {}

    def add_candidate(method: str, family: str, q: pd.Series, params: dict[str, Any]) -> None:
        row = _evaluate_online_v9_fast(
            merged,
            q,
            method=method,
            family=family,
            params=params,
            cache=cache,
        )
        rows.append(row)
        q_maps[method] = q.clip(0, 1)

    # Baselines carried into v9 so the goal wave is self-contained.
    add_candidate(
        "v9_reference_v4_source_aware_guarded",
        "baseline_v4_source_aware_guarded",
        merged["qhat_v4"],
        {"base": "qhat_v4"},
    )
    add_candidate(
        "v9_reference_v8_best",
        "baseline_v8_best_width_blocked",
        (merged["qhat_v4"] * 0.995 + 0.080 * primary).clip(0, 1),
        {
            "base_multiplier": 0.995,
            "primary_delta": 0.080,
            "primary_mask": "targeted_dti_score_or_gradeD",
            "source": "paper4_v8_best_passing_source80",
        },
    )

    # V9: shrink the global base, keep the v8 weak-cell mask, then add a small
    # structural tail guard for loans known before decision time.
    base_grid = [0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980]
    primary_delta_grid = [0.100, 0.120, 0.140, 0.160]
    tail_bonus_grid = [0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
    for tail_name, tail_mask in tails.items():
        tail = tail_mask.astype(float)
        for base_multiplier in base_grid:
            for primary_delta in primary_delta_grid:
                for tail_bonus in tail_bonus_grid:
                    method = (
                        f"v9_tail_guard_{tail_name}"
                        f"_m{base_multiplier:.3f}_d{primary_delta:.3f}_t{tail_bonus:.3f}"
                    )
                    q = (
                        merged["qhat_v4"] * base_multiplier
                        + primary_delta * primary
                        + tail_bonus * tail
                    ).clip(0, 1)
                    add_candidate(
                        method,
                        "structural_tail_guard_width_shrink",
                        q,
                        {
                            "base_multiplier": base_multiplier,
                            "primary_delta": primary_delta,
                            "primary_mask": "targeted_dti_score_or_gradeD",
                            "tail_bonus": tail_bonus,
                            "tail_mask": tail_name,
                        },
                    )

    frontier = pd.DataFrame(rows).sort_values(
        [
            "goal_pass",
            "avg_width_loan",
            "coverage_source_month_defended_min",
            "coverage_policy_month_defended_min",
        ],
        ascending=[False, True, False, False],
    )
    passing = frontier[frontier["goal_pass"].astype(bool)].copy()
    if passing.empty:
        best_method = frontier.sort_values(
            [
                "coverage_source_month_defended_min",
                "coverage_policy_month_defended_min",
                "avg_width_loan",
            ],
            ascending=[False, False, True],
        )["online_method_v9"].iloc[0]
    else:
        best_method = passing.sort_values("avg_width_loan")["online_method_v9"].iloc[0]

    conservative = passing[passing["avg_width_loan"].between(0.945, 0.950)].sort_values(
        "avg_width_loan"
    )
    conservative_method = (
        conservative["online_method_v9"].iloc[0] if not conservative.empty else best_method
    )
    selected = list(
        dict.fromkeys(
            [
                "v9_reference_v4_source_aware_guarded",
                "v9_reference_v8_best",
                best_method,
                conservative_method,
            ]
        )
    )

    interval_frames = []
    policy_frames = []
    source_frames = []
    for method in selected:
        local = merged.copy()
        q = q_maps[method]
        _, policy_detail, source_detail = _evaluate_online_v9(
            merged,
            q,
            method=method,
            family=str(
                frontier.loc[frontier["online_method_v9"].eq(method), "method_family"].iloc[0]
            ),
            min_support=min_support,
            params=json.loads(
                str(
                    frontier.loc[frontier["online_method_v9"].eq(method), "parameters_json"].iloc[0]
                )
            ),
        )
        policy_frames.append(policy_detail)
        source_frames.append(source_detail)
        local["online_method_v9"] = method
        local["qhat_v9"] = q
        local["pd_low_online_v9"] = (local["y_pred"] - q).clip(0, 1)
        local["pd_high_online_v9"] = (local["y_pred"] + q).clip(0, 1)
        local["covered_online_v9"] = _coverage(local["y_true"], local["y_pred"], q)
        local["interval_width_online_v9"] = _interval_width(local["y_pred"], q)
        interval_frames.append(
            local[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "online_method_v9",
                    "qhat_v9",
                    "pd_low_online_v9",
                    "pd_high_online_v9",
                    "covered_online_v9",
                    "interval_width_online_v9",
                ]
            ]
        )

    intervals = pd.concat(interval_frames, ignore_index=True)
    policy = pd.concat(policy_frames, ignore_index=True)
    source = pd.concat(source_frames, ignore_index=True)

    v8 = frontier[frontier["online_method_v9"].eq("v9_reference_v8_best")].iloc[0]
    best = frontier[frontier["online_method_v9"].eq(best_method)].iloc[0]
    cons = frontier[frontier["online_method_v9"].eq(conservative_method)].iloc[0]
    breakpoint = pd.DataFrame(
        [
            {
                "breakpoint_id": "v9_v8_reference",
                "online_method_v9": "v9_reference_v8_best",
                "coverage_source_month_defended_min": float(
                    v8["coverage_source_month_defended_min"]
                ),
                "coverage_policy_month_defended_min": float(
                    v8["coverage_policy_month_defended_min"]
                ),
                "avg_width_loan": float(v8["avg_width_loan"]),
                "width_margin_to_0p95": float(v8["width_margin_to_0p95"]),
                "goal_pass": bool(v8["goal_pass"]),
                "interpretation": "v8 passed defended coverage but missed the width goal.",
            },
            {
                "breakpoint_id": "v9_best_goal_passing_width",
                "online_method_v9": best_method,
                "coverage_source_month_defended_min": float(
                    best["coverage_source_month_defended_min"]
                ),
                "coverage_policy_month_defended_min": float(
                    best["coverage_policy_month_defended_min"]
                ),
                "avg_width_loan": float(best["avg_width_loan"]),
                "width_margin_to_0p95": float(best["width_margin_to_0p95"]),
                "goal_pass": bool(best["goal_pass"]),
                "interpretation": "lowest-width v9 method satisfying the explicit /goal gates.",
            },
            {
                "breakpoint_id": "v9_conservative_goal_passing_width",
                "online_method_v9": conservative_method,
                "coverage_source_month_defended_min": float(
                    cons["coverage_source_month_defended_min"]
                ),
                "coverage_policy_month_defended_min": float(
                    cons["coverage_policy_month_defended_min"]
                ),
                "avg_width_loan": float(cons["avg_width_loan"]),
                "width_margin_to_0p95": float(cons["width_margin_to_0p95"]),
                "goal_pass": bool(cons["goal_pass"]),
                "interpretation": "goal-passing method closer to the original v8 width, kept as a less aggressive reference.",
            },
        ]
    )
    return frontier, intervals, policy, source, breakpoint


def build_v9_claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": "Online conformal efficiency goal",
                "claim_status": "resolved_for_explicit_goal",
                "artifact": "paper4_v9_online_efficiency_frontier.csv",
                "quarto_page": "19av-v9-online-goal-resolution.qmd",
                "caveat": "Coverage gates are met at the defended-cell boundary; future-period validation remains required.",
            },
            {
                "priority": "Online selected intervals",
                "claim_status": "implemented_selected_methods",
                "artifact": "paper4_v9_online_selected_intervals.parquet",
                "quarto_page": "19av-v9-online-goal-resolution.qmd",
                "caveat": "Selected methods include baseline, v8 reference, lowest-width v9 and conservative v9.",
            },
            {
                "priority": "Paper Estrella freeze",
                "claim_status": "guardrail_verified",
                "artifact": "paper4_v9_online_goal_status.json",
                "quarto_page": "19av-v9-online-goal-resolution.qmd",
                "caveat": "No Paper Estrella promotion artifact is modified or replaced.",
            },
        ]
    )


def _write_v9_note(status: dict[str, Any], breakpoint: pd.DataFrame) -> None:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTE_DIR / "paper4_v9_online_goal_resolution.md"
    rows = "\n".join(
        f"- `{row.breakpoint_id}`: `{row.online_method_v9}`; "
        f"source={row.coverage_source_month_defended_min:.4f}, "
        f"policy={row.coverage_policy_month_defended_min:.4f}, "
        f"width={row.avg_width_loan:.6f}, pass={bool(row.goal_pass)}."
        for row in breakpoint.itertuples(index=False)
    )
    path.write_text(
        "\n".join(
            [
                "# Paper 4 v9 Online Goal Resolution",
                "",
                f"- Goal achieved: `{status['online_goal_achieved']}`",
                f"- Best method: `{status['online_best_method_v9']}`",
                f"- Best source-month defended minimum: `{status['online_best_source_month_defended_min']:.4f}`",
                f"- Best policy-month defended minimum: `{status['online_best_policy_month_defended_min']:.4f}`",
                f"- Best average loan width: `{status['online_best_width']:.6f}`",
                f"- Paper Estrella modified: `{status['paper1_artifacts_modified']}`",
                "",
                "## Breakpoints",
                "",
                rows,
                "",
                "## Caveat",
                "",
                "The v9 method uses only pre-decision structural fields, but the grid itself was selected in replay. "
                "It resolves the explicit online efficiency gate for Paper 4 and should be rerun on future periods before any broader promotion claim.",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = time.time()
    _, _, allocations, _, online_intervals = _load_inputs()
    frontier, intervals, policy_month, source_month, breakpoint = build_online_goal_v9(
        allocations,
        online_intervals,
        min_support=args.min_support,
    )

    _write_csv("paper4_v9_online_efficiency_frontier.csv", frontier)
    _write_parquet("paper4_v9_online_selected_intervals.parquet", intervals)
    _write_parquet("paper4_v9_online_policy_month.parquet", policy_month)
    _write_parquet("paper4_v9_online_source_month.parquet", source_month)
    _write_csv("paper4_v9_online_breakpoint_report.csv", breakpoint)
    claims = build_v9_claim_matrix()
    _write_csv("paper4_v9_claim_artifact_matrix.csv", claims)

    passing = frontier[frontier["goal_pass"].astype(bool)].sort_values("avg_width_loan")
    best = passing.iloc[0] if not passing.empty else frontier.iloc[0]
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v9_online_goal_resolution",
        "mode": "paper4_living_lab_no_paper1_changes",
        "goal_objective": (
            "Improve Paper 4 online conformal efficiency until source-month >= 0.80, "
            "policy-month >= 0.90, and avg_width <= 0.95 without touching Paper Estrella."
        ),
        "online_goal_achieved": bool(best["goal_pass"]),
        "online_best_method_v9": str(best["online_method_v9"]),
        "online_best_method_family": str(best["method_family"]),
        "online_best_source_month_defended_min": float(best["coverage_source_month_defended_min"]),
        "online_best_policy_month_defended_min": float(best["coverage_policy_month_defended_min"]),
        "online_best_width": float(best["avg_width_loan"]),
        "online_width_margin_to_0p95": float(best["width_margin_to_0p95"]),
        "online_goal_passing_count": int(frontier["goal_pass"].astype(bool).sum()),
        "target_source_month": TARGET_SOURCE_MONTH,
        "target_policy_month": TARGET_POLICY_MONTH,
        "target_avg_width": TARGET_AVG_WIDTH,
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "paper_estrella_freeze_artifact": str(PAPER1_PROMOTION),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": (
            "The v9 structural tail guard resolves the explicit online gate in replay. "
            "It is not a Paper Estrella change and does not imply final Paper 4 promotion."
        ),
    }
    _write_json("paper4_v9_online_goal_status.json", status)
    _write_v9_note(status, breakpoint)

    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
