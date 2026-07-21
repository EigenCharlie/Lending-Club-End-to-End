"""Generate journal-grade P1 evidence tables for Paper Estrella.

The outputs in ``reports/paper_material/paper1/tables`` are deliberately
derived from the current canonical champion artifacts. This script does not
reopen the champion search; it documents post-selection confirmation, segment
sensitivity, a CROMS-style decision-aware conformal screen, and synthetic shift
stress checks around the official economic champion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "paper_material" / "paper1" / "tables"
DOCS_OUT = ROOT / "docs" / "research"

PROMOTION_PATH = MODELS / "final_project_promotion.json"
STATUS_PATH = MODELS / "paper1_p1_evidence_status.json"
TEST_PATH = DATA / "test.parquet"
CONFORMAL_CANDIDATES_PATH = (
    DATA
    / "conformal_gap"
    / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612"
    / "conformal_reopen_phase1_final_candidates.parquet"
)
CONFORMAL_WINNER_INTERVALS_PATH = (
    DATA
    / "conformal_gap"
    / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612__phase1__final__rank-1"
    / "conformal_intervals_mondrian.parquet"
)
PORTFOLIO_FINALIST_PATH = DATA / "portfolio_tradeoff" / "conformal_finalist_comparison.parquet"
HARDENING_TABLES = {
    "funded_loans": OUT / "paper1_tableA7_funded_set_loans.csv",
    "funded_composition": OUT / "paper1_tableA8_funded_set_composition.csv",
    "strict_holdout": OUT / "paper1_tableA9_strict_temporal_holdout.csv",
    "finalist_exact": OUT / "paper1_tableA10_conformal_finalist_exact_bound_eval.csv",
    "enhanced_shift": OUT / "paper1_tableA11_enhanced_synthetic_shift.csv",
}
THEORY_APPENDIX_PATH = DOCS_OUT / "paper_estrella_conditional_tightening_appendix_2026-05-04.md"
FINALIST_INTERVALS = [
    {
        "rank": 1,
        "label": "rank1_score_decile_mondrian",
        "intervals_path": CONFORMAL_WINNER_INTERVALS_PATH,
        "policy_path": MODELS
        / "portfolio_tradeoff"
        / "conformal-finalist-rank1_score_decile_raw_bins5_mgs100"
        / "portfolio_research_policy.json",
    },
    {
        "rank": 2,
        "label": "rank2_grade_mgs100",
        "intervals_path": DATA
        / "conformal_gap"
        / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612__phase1__final__rank-2"
        / "conformal_intervals_mondrian.parquet",
        "policy_path": MODELS
        / "portfolio_tradeoff"
        / "conformal-finalist-rank2_grade_cal_bins10_mgs100"
        / "portfolio_research_policy.json",
    },
    {
        "rank": 3,
        "label": "rank3_grade_mgs1000",
        "intervals_path": DATA
        / "conformal_gap"
        / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612__phase1__final__rank-3"
        / "conformal_intervals_mondrian.parquet",
        "policy_path": MODELS
        / "portfolio_tradeoff"
        / "conformal-finalist-rank3_grade_cal_bins10_mgs1000"
        / "portfolio_research_policy.json",
    },
]

BOUND_STAGES = [
    {
        "stage": "bound_aware_5k",
        "role": "screening",
        "oot_rows": 5_000,
        "run_dir": "rank1_alpha01_bound_aware_5k_corrected_2026-04-05-1548",
    },
    {
        "stage": "bound_aware_25k",
        "role": "refinement",
        "oot_rows": 25_000,
        "run_dir": "rank1_alpha01_bound_aware_25k_gpu_2026-04-05-1611c",
    },
    {
        "stage": "bound_aware_276k",
        "role": "full_oot_confirmation",
        "oot_rows": 276_869,
        "run_dir": "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_table(name: str, frame: pd.DataFrame) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{name}.csv"
    tex_path = OUT / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    frame.to_latex(
        tex_path,
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.6f}",
    )
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {tex_path.relative_to(ROOT)}")
    return [csv_path, tex_path]


def _append_unique(paths: list[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)


def _relative_artifacts(paths: list[Path]) -> list[str]:
    return list(dict.fromkeys(str(path.relative_to(ROOT)) for path in paths))


def _safe_float(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _policy_matches(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    fields = ("risk_tolerance", "policy_mode", "gamma", "uncertainty_aversion")
    for field in fields:
        if field not in row or field not in policy:
            return False
        left = row[field]
        right = policy[field]
        if isinstance(right, str):
            if str(left) != right:
                return False
            continue
        left_float = _safe_float(left)
        right_float = _safe_float(right)
        if left_float is None or right_float is None or abs(left_float - right_float) > 1e-9:
            return False
    return True


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _effective_n(weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    denom = float((weights**2).sum())
    if denom <= 0:
        return 0.0
    return float(weights.sum() ** 2 / denom)


def _minmax_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    low = float(numeric.min())
    high = float(numeric.max())
    if abs(high - low) <= 1e-12:
        return numeric.notna().astype(float)
    scaled = (numeric - low) / (high - low)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.fillna(0.0)


def _coverage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["covered_90"] = (
        (work["y_true"] >= work["pd_low_90"]) & (work["y_true"] <= work["pd_high_90"])
    ).astype(float)
    work["covered_95"] = (
        (work["y_true"] >= work["pd_low_95"]) & (work["y_true"] <= work["pd_high_95"])
    ).astype(float)
    work["miscovered_90"] = 1.0 - work["covered_90"]
    work["loan_weight"] = pd.to_numeric(work["loan_amnt"], errors="coerce").fillna(1.0)
    return work


def _period_from_issue_d(issue_d: pd.Series, temporal_segment: pd.Series) -> pd.Series:
    dates = pd.to_datetime(issue_d, errors="coerce")
    period = pd.Series(pd.NA, index=issue_d.index, dtype="object")
    first_half = dates.dt.month.le(6)
    period.loc[dates.notna() & first_half] = dates.dt.year.astype(str) + "H1"
    period.loc[dates.notna() & ~first_half] = dates.dt.year.astype(str) + "H2"
    period.loc[dates.dt.year.eq(2020)] = "2020"

    missing = period.isna()
    if missing.any():
        extracted = temporal_segment.astype(str).str.extract(r"vintage=(\d{4})Q([1-4])")
        years = extracted[0]
        quarters = pd.to_numeric(extracted[1], errors="coerce")
        fallback = years + quarters.le(2).map({True: "H1", False: "H2"})
        fallback.loc[years.eq("2020")] = "2020"
        period.loc[missing] = fallback.loc[missing]
    return period.fillna("unknown")


def _load_joined_oot() -> pd.DataFrame:
    intervals = pd.read_parquet(
        CONFORMAL_WINNER_INTERVALS_PATH,
        columns=[
            "id",
            "y_true",
            "y_pred",
            "pd_low_90",
            "pd_high_90",
            "pd_low_95",
            "pd_high_95",
            "width_90",
            "width_95",
            "grade",
            "loan_amnt",
            "temporal_segment",
        ],
    )
    test = pd.read_parquet(
        TEST_PATH,
        columns=["id", "issue_d", "grade", "loan_amnt"],
    ).rename(columns={"grade": "original_grade", "loan_amnt": "loan_amnt_test"})
    merged = intervals.merge(test, on="id", how="left", validate="one_to_one")
    merged["loan_amnt"] = merged["loan_amnt"].fillna(merged["loan_amnt_test"])
    merged["original_grade"] = merged["original_grade"].fillna("unknown")
    merged["period"] = _period_from_issue_d(merged["issue_d"], merged["temporal_segment"])
    return _coverage_columns(merged)


def _build_nested_holdout_table(promotion: dict[str, Any]) -> pd.DataFrame:
    champion = promotion["final_champion"]
    rows: list[dict[str, Any]] = []
    for stage in BOUND_STAGES:
        run_dir = stage["run_dir"]
        selection_path = (
            MODELS / "portfolio_bound_aware" / run_dir / "portfolio_bound_aware_selection.json"
        )
        shortlist_path = (
            DATA / "portfolio_bound_aware" / run_dir / "portfolio_bound_aware_shortlist.parquet"
        )
        selection = _load_json(selection_path)
        shortlist = pd.read_parquet(shortlist_path)
        metrics = dict(selection["selected_metrics"])
        rows.append(
            {
                "stage": stage["stage"],
                "role": stage["role"],
                "run_label": selection["run_label"],
                "oot_rows": stage["oot_rows"],
                "candidate_count": len(shortlist),
                "alpha01_passers": int(shortlist["alpha01_exact_pass"].sum()),
                "alpha01_pass_rate": float(shortlist["alpha01_exact_pass"].mean()),
                "candidate_rank": int(metrics["candidate_rank"]),
                "risk_tolerance": float(metrics["risk_tolerance"]),
                "gamma": float(metrics["gamma"]),
                "uncertainty_aversion": float(metrics["uncertainty_aversion"]),
                "realized_total_return": float(metrics["realized_total_return"]),
                "alpha01_exact_pass": bool(metrics["alpha01_exact_pass"]),
                "alpha01_weighted_miscoverage_V": float(metrics["alpha01_weighted_miscoverage_V"]),
                "alpha01_gamma_cp": float(metrics["alpha01_gamma_cp"]),
                "alpha01_violation": float(metrics["alpha01_violation"]),
                "selected_matches_final_champion": _policy_matches(metrics, champion),
                "source_artifact": str(selection_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def _rank_from_text(value: Any) -> int | None:
    match = re.search(r"rank[-_]?(\d+)", str(value))
    return int(match.group(1)) if match else None


def _build_decision_aware_selector_table(promotion: dict[str, Any]) -> pd.DataFrame:
    conformal = pd.read_parquet(CONFORMAL_CANDIDATES_PATH).copy()
    tradeoff = pd.read_parquet(PORTFOLIO_FINALIST_PATH).copy()
    conformal["rank"] = conformal["namespace"].map(_rank_from_text)
    tradeoff["rank"] = tradeoff["label"].map(_rank_from_text)

    merged = conformal.merge(
        tradeoff[
            [
                "rank",
                "label",
                "realized_total_return",
                "price_of_robustness",
                "price_of_robustness_pct",
                "n_funded",
                "ab_pass",
            ]
        ],
        on="rank",
        how="left",
        validate="one_to_one",
    )
    champion = promotion["final_champion"]
    exact_path = HARDENING_TABLES["finalist_exact"]
    if exact_path.exists():
        exact = pd.read_csv(exact_path).rename(
            columns={
                "alpha01_exact_pass": "exact_alpha01_exact_pass",
                "alpha01_weighted_miscoverage_V": "exact_alpha01_weighted_miscoverage_V",
                "alpha01_gamma_cp": "exact_alpha01_gamma_cp",
                "alpha01_violation": "exact_alpha01_violation",
            }
        )
        merged = merged.merge(
            exact[
                [
                    "rank",
                    "exact_alpha01_exact_pass",
                    "exact_alpha01_weighted_miscoverage_V",
                    "exact_alpha01_gamma_cp",
                    "exact_alpha01_violation",
                ]
            ],
            on="rank",
            how="left",
            validate="one_to_one",
        )
        merged["exact_bound_available"] = merged["exact_alpha01_exact_pass"].notna()
        merged["alpha01_exact_pass"] = merged["exact_alpha01_exact_pass"]
        merged["alpha01_weighted_miscoverage_V"] = merged["exact_alpha01_weighted_miscoverage_V"]
        merged["alpha01_gamma_cp"] = merged["exact_alpha01_gamma_cp"]
        merged["alpha01_violation"] = merged["exact_alpha01_violation"]
    else:
        merged["exact_bound_available"] = merged["rank"].eq(1)
        merged["alpha01_exact_pass"] = merged["exact_bound_available"].map(
            {True: bool(champion["alpha01_exact_pass"]), False: pd.NA}
        )
        merged["alpha01_weighted_miscoverage_V"] = merged["exact_bound_available"].map(
            {True: champion["alpha01_weighted_miscoverage_V"], False: pd.NA}
        )
        merged["alpha01_gamma_cp"] = merged["exact_bound_available"].map(
            {True: champion["alpha01_gamma_cp"], False: pd.NA}
        )
        merged["alpha01_violation"] = merged["exact_bound_available"].map(
            {True: champion["alpha01_violation"], False: pd.NA}
        )
    merged["gate_pass"] = (
        merged["policy_overall_pass"].astype(bool)
        & merged["ab_pass"].fillna(False).astype(bool)
        & merged["min_group_coverage_90"].ge(0.90)
    )
    merged["coverage_margin_90"] = merged["coverage_90"] - 0.90
    merged["min_group_margin_90"] = merged["min_group_coverage_90"] - 0.90
    merged["return_score"] = _minmax_score(merged["realized_total_return"], higher_is_better=True)
    merged["width_score"] = _minmax_score(merged["avg_width_90"], higher_is_better=False)
    merged["coverage_score"] = _minmax_score(merged["coverage_margin_90"], higher_is_better=True)
    merged["group_score"] = _minmax_score(merged["min_group_margin_90"], higher_is_better=True)
    merged["tightness_score"] = _minmax_score(
        -pd.to_numeric(merged["alpha01_weighted_miscoverage_V"], errors="coerce"),
        higher_is_better=True,
    )
    raw_score = (
        0.30 * merged["return_score"]
        + 0.20 * merged["coverage_score"]
        + 0.20 * merged["group_score"]
        + 0.15 * merged["width_score"]
        + 0.15 * merged["tightness_score"]
    )
    merged["decision_aware_score"] = raw_score.where(merged["gate_pass"], -1.0)
    best_index = merged["decision_aware_score"].idxmax()
    merged["decision_aware_selected"] = False
    merged.loc[best_index, "decision_aware_selected"] = True
    keep = [
        "rank",
        "partition",
        "partition_probability_source",
        "n_score_bins",
        "fallback_mode",
        "min_group_size",
        "coverage_90",
        "avg_width_90",
        "min_group_coverage_90",
        "coverage_margin_90",
        "min_group_margin_90",
        "policy_overall_pass",
        "ab_pass",
        "realized_total_return",
        "price_of_robustness_pct",
        "n_funded",
        "exact_bound_available",
        "alpha01_exact_pass",
        "alpha01_weighted_miscoverage_V",
        "alpha01_gamma_cp",
        "alpha01_violation",
        "gate_pass",
        "decision_aware_score",
        "decision_aware_selected",
    ]
    return merged[keep].sort_values("rank").reset_index(drop=True)


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    loan_weights = group["loan_weight"]
    return {
        "n": int(len(group)),
        "loan_amount": float(loan_weights.sum()),
        "default_rate": float(group["y_true"].mean()),
        "coverage_90": float(group["covered_90"].mean()),
        "coverage_95": float(group["covered_95"].mean()),
        "avg_width_90": float(group["width_90"].mean()),
        "avg_width_95": float(group["width_95"].mean()),
        "weighted_miscoverage_90_proxy": _weighted_average(
            group["miscovered_90"],
            loan_weights,
        ),
    }


def _build_segment_period_table(oot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_loan_amount = float(oot["loan_weight"].sum())
    grouped = oot.groupby(["period", "original_grade"], dropna=False, observed=True)
    for (period, grade), group in grouped:
        row = {
            "period": str(period),
            "original_grade": str(grade),
            **_summarize_group(group),
        }
        row["loan_amount_share"] = row["loan_amount"] / total_loan_amount
        row["risk_flag"] = bool(row["coverage_90"] < 0.90)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["period", "original_grade"]).reset_index(drop=True)


def _scenario_weights(oot: pd.DataFrame) -> dict[str, pd.Series]:
    top_pd = float(oot["y_pred"].quantile(0.75))
    bottom_pd = float(oot["y_pred"].quantile(0.25))
    return {
        "baseline": pd.Series(1.0, index=oot.index),
        "high_pd_tail_3x": oot["y_pred"].ge(top_pd).map({True: 3.0, False: 1.0}),
        "grade_efg_3x": oot["original_grade"].isin(["E", "F", "G"]).map({True: 3.0, False: 1.0}),
        "late_period_3x": oot["period"].isin(["2019H2", "2020"]).map({True: 3.0, False: 1.0}),
        "low_pd_2020_3x": (oot["period"].eq("2020") | oot["y_pred"].le(bottom_pd)).map(
            {True: 3.0, False: 1.0}
        ),
    }


def _build_synthetic_shift_table(oot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario_descriptions = {
        "baseline": "Unweighted OOT 2018-2020.",
        "high_pd_tail_3x": "Triples the top predicted-PD quartile.",
        "grade_efg_3x": "Triples original grades E/F/G.",
        "late_period_3x": "Triples 2019H2 and 2020 originations.",
        "low_pd_2020_3x": "Triples 2020 or bottom predicted-PD quartile loans.",
    }
    for scenario, weights in _scenario_weights(oot).items():
        combined_weights = weights.astype(float)
        loan_weights = combined_weights * oot["loan_weight"]
        rows.append(
            {
                "scenario": scenario,
                "description": scenario_descriptions[scenario],
                "effective_n": _effective_n(combined_weights),
                "weighted_default_rate": _weighted_average(oot["y_true"], combined_weights),
                "weighted_coverage_90": _weighted_average(oot["covered_90"], combined_weights),
                "weighted_coverage_95": _weighted_average(oot["covered_95"], combined_weights),
                "weighted_avg_width_90": _weighted_average(oot["width_90"], combined_weights),
                "loan_weighted_miscoverage_90_proxy": _weighted_average(
                    oot["miscovered_90"],
                    loan_weights,
                ),
                "coverage90_pass": bool(
                    _weighted_average(oot["covered_90"], combined_weights) >= 0.90
                ),
                "coverage95_pass": bool(
                    _weighted_average(oot["covered_95"], combined_weights) >= 0.95
                ),
            }
        )
    return pd.DataFrame(rows)


def _normalise_policy(raw: dict[str, Any]) -> dict[str, Any]:
    selected = raw.get("selected_policy", raw)
    return {
        "risk_tolerance": float(selected.get("risk_tolerance", 0.10)),
        "policy_mode": str(selected.get("policy_mode", "blended_uncertainty")),
        "gamma": float(selected.get("gamma", 0.0)),
        "delta_cap_quantile": float(selected.get("delta_cap_quantile", 1.0)),
        "tail_focus_quantile": float(selected.get("tail_focus_quantile", 1.0)),
        "uncertainty_aversion": float(selected.get("uncertainty_aversion", 0.0)),
        "min_budget_utilization": float(selected.get("min_budget_utilization", 0.0)),
        "pd_cap_slack_penalty": float(selected.get("pd_cap_slack_penalty", 0.0)),
        "solver_backend": "highs",
    }


def _parse_rate_series(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if np.nanmax(values) > 1.5:
            values = values / 100.0
        return np.nan_to_num(values, nan=0.12)
    values = (
        series.astype(str)
        .str.strip()
        .str.rstrip("%")
        .pipe(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    return np.nan_to_num(values, nan=12.0) / 100.0


def _load_exact_aligned_dataset(intervals_path: Path) -> pd.DataFrame:
    from scripts.optimize_portfolio_tradeoff import _align_loans_and_intervals, _load_candidates
    from src.models.conformal_artifacts import load_conformal_intervals

    candidates = _load_candidates().reset_index(drop=True)
    intervals, _, _ = load_conformal_intervals(
        allow_legacy_fallback=False,
        override_path=str(intervals_path),
    )
    loans, ints = _align_loans_and_intervals(
        candidates=candidates,
        intervals=intervals.reset_index(drop=True),
        max_candidates=0,
        random_state=42,
    )
    aligned = loans.reset_index(drop=True).copy()
    if "grade" in aligned.columns:
        aligned["original_grade"] = aligned["grade"].astype(str)
    else:
        aligned["original_grade"] = "unknown"
    for column in ints.columns:
        target = column if column not in aligned.columns else f"interval_{column}"
        aligned[target] = ints[column].reset_index(drop=True)
    if "y_true" not in aligned and "interval_y_true" in aligned:
        aligned["y_true"] = aligned["interval_y_true"]
    if "y_pred" not in aligned and "interval_y_pred" in aligned:
        aligned["y_pred"] = aligned["interval_y_pred"]
    temporal = (
        aligned["temporal_segment"]
        if "temporal_segment" in aligned.columns
        else pd.Series(["unknown"] * len(aligned), index=aligned.index)
    )
    issue_d = (
        aligned["issue_d"]
        if "issue_d" in aligned.columns
        else pd.Series([pd.NA] * len(aligned), index=aligned.index)
    )
    aligned["period"] = _period_from_issue_d(issue_d, temporal)
    return aligned


def _interval_arrays_at_alpha(
    frame: pd.DataFrame, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pd_point = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=float)
    pd_low_90 = pd.to_numeric(frame["pd_low_90"], errors="coerce").to_numpy(dtype=float)
    pd_high_90 = pd.to_numeric(frame["pd_high_90"], errors="coerce").to_numpy(dtype=float)
    radius_90 = (pd_high_90 - pd_low_90) / 2.0
    sweep_path = DATA / "alpha_sweep_pareto_mondrian.parquet"
    if sweep_path.exists():
        sweep = pd.read_parquet(sweep_path)
        row_base = sweep[np.isclose(sweep["alpha"], 0.10)]
        row_target = sweep[np.isclose(sweep["alpha"], alpha)]
        if len(row_base) > 0 and len(row_target) > 0:
            scale = float(row_target["avg_width"].values[0]) / max(
                float(row_base["avg_width"].values[0]),
                1e-8,
            )
            radius = radius_90 * scale
            return (
                pd_point,
                np.clip(pd_point - radius, 0.0, 1.0),
                np.clip(
                    pd_point + radius,
                    0.0,
                    1.0,
                ),
            )
    return pd_point, pd_low_90, pd_high_90


def _segment_labels(frame: pd.DataFrame, policy_mode: str) -> np.ndarray | None:
    if str(policy_mode).strip().lower() not in {
        "segment_tail_blended_uncertainty",
        "segment_relative_tail_blended_uncertainty",
    }:
        return None
    grade = frame.get("original_grade", pd.Series(["unknown"] * len(frame))).fillna("unknown")
    term = frame.get("term", pd.Series(["unknown"] * len(frame))).fillna("unknown")
    verification = frame.get(
        "verification_status",
        pd.Series(["unknown"] * len(frame)),
    ).fillna("unknown")
    return (grade.astype(str) + "|" + term.astype(str) + "|" + verification.astype(str)).to_numpy(
        dtype=object
    )


def _realized_return(
    allocation: np.ndarray,
    loan_amounts: np.ndarray,
    int_rates: np.ndarray,
    y_true: np.ndarray,
    *,
    lgd: float = 0.45,
) -> float:
    nondefault_return = allocation * loan_amounts * int_rates * (1.0 - y_true)
    default_loss = allocation * loan_amounts * (-lgd) * y_true
    return float(np.sum(nondefault_return + default_loss))


def _solve_exact_policy(
    aligned: pd.DataFrame,
    policy: dict[str, Any],
    *,
    alpha: float = 0.01,
    budget: float = 1_000_000.0,
) -> dict[str, Any]:
    from src.optimization.portfolio_model import compute_effective_pd, optimize_portfolio_allocation

    pd_point, pd_low, pd_high = _interval_arrays_at_alpha(aligned, alpha)
    effective_pd = compute_effective_pd(
        pd_point=pd_point,
        pd_high=pd_high,
        policy_mode=str(policy["policy_mode"]),
        gamma=float(policy["gamma"]),
        delta_cap_quantile=float(policy["delta_cap_quantile"]),
        tail_focus_quantile=float(policy["tail_focus_quantile"]),
        segment_labels=_segment_labels(aligned, str(policy["policy_mode"])),
    )
    loan_amounts = (
        pd.to_numeric(aligned["loan_amnt"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    )
    int_rates = (
        _parse_rate_series(aligned["int_rate"])
        if "int_rate" in aligned
        else np.full(
            len(aligned),
            0.12,
        )
    )
    y_true = pd.to_numeric(aligned["y_true"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lgd = np.full(len(aligned), 0.45, dtype=float)
    solution = optimize_portfolio_allocation(
        loans=aligned,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        lgd=lgd,
        int_rates=int_rates,
        total_budget=budget,
        max_concentration=0.25,
        max_portfolio_pd=float(policy["risk_tolerance"]),
        robust=True,
        uncertainty_aversion=float(policy["uncertainty_aversion"]),
        min_budget_utilization=float(policy["min_budget_utilization"]),
        pd_cap_slack_penalty=float(policy["pd_cap_slack_penalty"]),
        pd_constraint_override=effective_pd,
        time_limit=300,
        threads=4,
        solver_backend=str(policy["solver_backend"]),
    )
    allocation = np.array(
        [float(solution["allocation"].get(i, 0.0)) for i in range(len(aligned))],
        dtype=float,
    )
    total_allocated = float(np.sum(allocation * loan_amounts))
    weights = (allocation * loan_amounts) / max(total_allocated, 1e-6)
    funded_mask = weights > 1e-8
    miscoverage = (y_true > pd_high).astype(float)
    weighted_pd_true = float(np.sum(weights * y_true))
    violation = max(0.0, weighted_pd_true - float(policy["risk_tolerance"]))
    weighted_miscoverage = float(np.sum(weights * miscoverage))
    sqrt_alpha = float(np.sqrt(alpha))
    return {
        "aligned": aligned,
        "allocation": allocation,
        "weights": weights,
        "pd_point": pd_point,
        "pd_high": pd_high,
        "effective_pd": effective_pd,
        "funded_mask": funded_mask,
        "metrics": {
            "alpha": float(alpha),
            "n_oot": int(len(aligned)),
            "n_funded": int(np.sum(funded_mask)),
            "total_allocated": total_allocated,
            "solver_status": str(solution.get("solver_status", "unknown")),
            "realized_total_return": _realized_return(allocation, loan_amounts, int_rates, y_true),
            "alpha01_gamma_cp": round(
                float(np.sum(weights * np.clip(pd_high - pd_point, 0.0, 1.0))),
                6,
            ),
            "alpha01_weighted_miscoverage_V": round(weighted_miscoverage, 6),
            "alpha01_weighted_pd_true": round(weighted_pd_true, 6),
            "alpha01_weighted_pd_constraint_used": round(float(np.sum(weights * effective_pd)), 6),
            "alpha01_weighted_pd_high": round(float(np.sum(weights * pd_high)), 6),
            "alpha01_violation": round(violation, 6),
            "alpha01_empirical_coverage_funded": round(
                float(1.0 - miscoverage[funded_mask].mean()) if funded_mask.any() else float("nan"),
                4,
            ),
            "alpha01_exact_pass": bool(
                violation <= alpha + 1e-8 and weighted_miscoverage <= sqrt_alpha + 1e-8
            ),
        },
    }


def _build_funded_set_tables(
    solve: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = solve["aligned"]
    allocation = solve["allocation"]
    weights = solve["weights"]
    funded_mask = solve["funded_mask"]
    loan_amounts = (
        pd.to_numeric(aligned["loan_amnt"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    )
    funded = aligned.loc[funded_mask].copy()
    funded["allocation_fraction"] = allocation[funded_mask]
    funded["funded_exposure"] = allocation[funded_mask] * loan_amounts[funded_mask]
    funded["portfolio_weight"] = weights[funded_mask]
    funded["pd_point"] = solve["pd_point"][funded_mask]
    funded["pd_high_alpha01"] = solve["pd_high"][funded_mask]
    funded["effective_pd_alpha01"] = solve["effective_pd"][funded_mask]
    funded["miscovered_alpha01"] = pd.to_numeric(funded["y_true"], errors="coerce").fillna(
        0.0
    ).to_numpy(dtype=float) > funded["pd_high_alpha01"].to_numpy(dtype=float)
    keep = [
        "id",
        "issue_d",
        "period",
        "original_grade",
        "sub_grade",
        "term",
        "loan_amnt",
        "int_rate",
        "y_true",
        "allocation_fraction",
        "funded_exposure",
        "portfolio_weight",
        "pd_point",
        "pd_high_alpha01",
        "effective_pd_alpha01",
        "miscovered_alpha01",
    ]
    funded_loans = funded[[col for col in keep if col in funded.columns]].sort_values(
        "funded_exposure",
        ascending=False,
    )
    rows: list[dict[str, Any]] = []
    total_exposure = float(funded["funded_exposure"].sum())
    for (period, grade), group in funded.groupby(["period", "original_grade"], dropna=False):
        exposure = float(group["funded_exposure"].sum())
        weights_group = group["funded_exposure"]
        rows.append(
            {
                "period": str(period),
                "original_grade": str(grade),
                "n_funded": int(len(group)),
                "funded_exposure": exposure,
                "exposure_share": exposure / max(total_exposure, 1e-6),
                "weighted_default_rate": _weighted_average(group["y_true"], weights_group),
                "weighted_pd_point": _weighted_average(group["pd_point"], weights_group),
                "weighted_pd_high_alpha01": _weighted_average(
                    group["pd_high_alpha01"],
                    weights_group,
                ),
                "portfolio_V_contribution": float(
                    group.loc[group["miscovered_alpha01"], "portfolio_weight"].sum()
                ),
            }
        )
    composition = (
        pd.DataFrame(rows).sort_values(["period", "original_grade"]).reset_index(drop=True)
    )
    return funded_loans.reset_index(drop=True), composition


def _build_strict_temporal_holdout_table(policy: dict[str, Any]) -> pd.DataFrame:
    aligned = _load_exact_aligned_dataset(CONFORMAL_WINNER_INTERVALS_PATH)
    slices = {
        "selection_slice_2018": aligned[aligned["period"].isin(["2018H1", "2018H2"])].copy(),
        "confirmation_slice_2019_2020": aligned[
            aligned["period"].isin(["2019H1", "2019H2", "2020"])
        ].copy(),
    }
    rows: list[dict[str, Any]] = []
    for name, frame in slices.items():
        result = _solve_exact_policy(frame.reset_index(drop=True), policy)
        metrics = result["metrics"]
        rows.append(
            {
                "holdout_slice": name,
                "role": "strict_disjoint_temporal_evaluation",
                "periods": ",".join(sorted(frame["period"].astype(str).unique())),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _build_finalist_exact_eval_table() -> pd.DataFrame:
    candidates = pd.read_parquet(CONFORMAL_CANDIDATES_PATH).copy()
    candidates["rank"] = candidates["namespace"].map(_rank_from_text)
    rows: list[dict[str, Any]] = []
    for finalist in FINALIST_INTERVALS:
        policy = _normalise_policy(_load_json(finalist["policy_path"]))
        aligned = _load_exact_aligned_dataset(finalist["intervals_path"])
        result = _solve_exact_policy(aligned, policy)
        candidate = candidates.loc[candidates["rank"].eq(finalist["rank"])].iloc[0]
        rows.append(
            {
                "rank": int(finalist["rank"]),
                "label": str(finalist["label"]),
                "partition": str(candidate["partition"]),
                "policy_overall_pass": bool(candidate["policy_overall_pass"]),
                "coverage_90": float(candidate["coverage_90"]),
                "avg_width_90": float(candidate["avg_width_90"]),
                "min_group_coverage_90": float(candidate["min_group_coverage_90"]),
                "policy_risk_tolerance": float(policy["risk_tolerance"]),
                "policy_gamma": float(policy["gamma"]),
                "policy_uncertainty_aversion": float(policy["uncertainty_aversion"]),
                **result["metrics"],
                "intervals_path": str(finalist["intervals_path"].relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def _flip_defaults(
    frame: pd.DataFrame,
    *,
    eligible_mask: pd.Series,
    fraction: float,
) -> tuple[pd.Series, int]:
    y = pd.to_numeric(frame["y_true"], errors="coerce").fillna(0.0).copy()
    candidates = frame.loc[eligible_mask & y.eq(0.0)].sort_values("y_pred", ascending=False)
    n_flip = int(math.ceil(len(candidates) * fraction))
    if n_flip <= 0:
        return y, 0
    flip_index = candidates.head(n_flip).index
    y.loc[flip_index] = 1.0
    return y, int(n_flip)


def _build_enhanced_synthetic_shift_table(oot: pd.DataFrame) -> pd.DataFrame:
    top_pd = float(oot["y_pred"].quantile(0.75))
    scenarios = {
        "baseline_observed": (oot["y_true"], 0, "Observed OOT labels."),
        "top_pd_nondefaults_to_default_5pct": (
            *_flip_defaults(oot, eligible_mask=oot["y_pred"].ge(top_pd), fraction=0.05),
            "Flips 5% of top-PD nondefaults to default.",
        ),
        "grade_efg_nondefaults_to_default_10pct": (
            *_flip_defaults(
                oot,
                eligible_mask=oot["original_grade"].isin(["E", "F", "G"]),
                fraction=0.10,
            ),
            "Flips 10% of grade E/F/G nondefaults to default.",
        ),
        "late_period_nondefaults_to_default_10pct": (
            *_flip_defaults(
                oot,
                eligible_mask=oot["period"].isin(["2019H2", "2020"]),
                fraction=0.10,
            ),
            "Flips 10% of late-period nondefaults to default.",
        ),
        "worst_segment_2018h1_b_nondefaults_to_default_5pct": (
            *_flip_defaults(
                oot,
                eligible_mask=oot["period"].eq("2018H1") & oot["original_grade"].eq("B"),
                fraction=0.05,
            ),
            "Flips 5% of nondefaults in the weakest observed period-grade segment.",
        ),
    }
    rows: list[dict[str, Any]] = []
    weights = oot["loan_weight"]
    for scenario, (y_stress, added_defaults, description) in scenarios.items():
        covered_90 = (
            (pd.to_numeric(y_stress, errors="coerce") >= oot["pd_low_90"])
            & (pd.to_numeric(y_stress, errors="coerce") <= oot["pd_high_90"])
        ).astype(float)
        covered_95 = (
            (pd.to_numeric(y_stress, errors="coerce") >= oot["pd_low_95"])
            & (pd.to_numeric(y_stress, errors="coerce") <= oot["pd_high_95"])
        ).astype(float)
        rows.append(
            {
                "scenario": scenario,
                "description": description,
                "added_defaults": int(added_defaults),
                "default_rate": float(pd.to_numeric(y_stress, errors="coerce").mean()),
                "coverage_90": float(covered_90.mean()),
                "coverage_95": float(covered_95.mean()),
                "loan_weighted_coverage_90": _weighted_average(covered_90, weights),
                "loan_weighted_coverage_95": _weighted_average(covered_95, weights),
                "coverage90_pass": bool(float(covered_90.mean()) >= 0.90),
            }
        )
    return pd.DataFrame(rows)


def build_p1_hardening_tables(promotion: dict[str, Any], oot: pd.DataFrame) -> list[Path]:
    policy = _normalise_policy({"selected_policy": promotion["final_champion"]})
    aligned = _load_exact_aligned_dataset(CONFORMAL_WINNER_INTERVALS_PATH)
    champion_solve = _solve_exact_policy(aligned, policy)
    funded_loans, funded_composition = _build_funded_set_tables(champion_solve)
    strict_holdout = _build_strict_temporal_holdout_table(policy)
    finalist_exact = _build_finalist_exact_eval_table()
    enhanced_shift = _build_enhanced_synthetic_shift_table(oot)

    artifacts: list[Path] = []
    artifacts += _write_table("paper1_tableA7_funded_set_loans", funded_loans)
    artifacts += _write_table("paper1_tableA8_funded_set_composition", funded_composition)
    artifacts += _write_table("paper1_tableA9_strict_temporal_holdout", strict_holdout)
    artifacts += _write_table(
        "paper1_tableA10_conformal_finalist_exact_bound_eval",
        finalist_exact,
    )
    artifacts += _write_table("paper1_tableA11_enhanced_synthetic_shift", enhanced_shift)
    return artifacts


def _attach_hardening_status(status: dict[str, Any], artifacts: list[Path]) -> None:
    existing: list[Path] = []
    for path in HARDENING_TABLES.values():
        if path.exists():
            existing.append(path)
        tex_path = path.with_suffix(".tex")
        if tex_path.exists():
            existing.append(tex_path)
    for path in existing:
        _append_unique(artifacts, path)
    if THEORY_APPENDIX_PATH.exists():
        _append_unique(artifacts, THEORY_APPENDIX_PATH)
        status["conditional_tightening"].update(
            {
                "status": "implemented_as_conditional_appendix",
                "appendix_artifact": str(THEORY_APPENDIX_PATH.relative_to(ROOT)),
                "main_theorem_role": "Markov remains the distribution-free guarantee.",
                "tightening_role": "Hoeffding/Bernstein require additional conditional independence.",
            }
        )
    if HARDENING_TABLES["funded_loans"].exists():
        funded = pd.read_csv(HARDENING_TABLES["funded_loans"])
        status["funded_set_export"] = {
            "status": "implemented",
            "n_funded_loans": int(len(funded)),
            "total_funded_exposure": float(funded["funded_exposure"].sum()),
            "artifact": str(HARDENING_TABLES["funded_loans"].relative_to(ROOT)),
        }
    if HARDENING_TABLES["funded_composition"].exists():
        composition = pd.read_csv(HARDENING_TABLES["funded_composition"])
        top = composition.sort_values("exposure_share", ascending=False).iloc[0]
        status["funded_set_composition"] = {
            "status": "implemented",
            "n_segments": int(len(composition)),
            "largest_segment": f"{top['period']}/{top['original_grade']}",
            "largest_segment_exposure_share": float(top["exposure_share"]),
            "artifact": str(HARDENING_TABLES["funded_composition"].relative_to(ROOT)),
        }
    if HARDENING_TABLES["strict_holdout"].exists():
        holdout = pd.read_csv(HARDENING_TABLES["strict_holdout"])
        status["strict_temporal_holdout"] = {
            "status": "implemented",
            "strict_disjoint_split": True,
            "all_alpha01_pass": bool(holdout["alpha01_exact_pass"].all()),
            "artifact": str(HARDENING_TABLES["strict_holdout"].relative_to(ROOT)),
        }
    if HARDENING_TABLES["finalist_exact"].exists():
        exact = pd.read_csv(HARDENING_TABLES["finalist_exact"])
        status["decision_aware_selector"]["exact_bound_available_for_all_ranks"] = bool(
            exact["alpha01_exact_pass"].notna().all()
        )
        status["conformal_finalist_exact_eval"] = {
            "status": "implemented",
            "n_finalists": int(len(exact)),
            "alpha01_pass_ranks": exact.loc[
                exact["alpha01_exact_pass"].astype(bool),
                "rank",
            ]
            .astype(int)
            .tolist(),
            "artifact": str(HARDENING_TABLES["finalist_exact"].relative_to(ROOT)),
        }
    if HARDENING_TABLES["enhanced_shift"].exists():
        shift = pd.read_csv(HARDENING_TABLES["enhanced_shift"])
        worst = shift.sort_values("coverage_90", ascending=True).iloc[0]
        status["enhanced_synthetic_shift"] = {
            "status": "implemented",
            "n_scenarios": int(len(shift)),
            "worst_scenario": str(worst["scenario"]),
            "worst_coverage_90": float(worst["coverage_90"]),
            "all_coverage90_pass": bool(shift["coverage90_pass"].all()),
            "artifact": str(HARDENING_TABLES["enhanced_shift"].relative_to(ROOT)),
        }


def _build_markdown_dossier(status: dict[str, Any]) -> Path:
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    path = DOCS_OUT / "paper_estrella_p1_evidence_2026-05-04.md"
    lines = [
        "# Paper Estrella P1 Evidence - 2026-05-04",
        "",
        "This dossier records the P1 evidence now materialized around the official",
        "`paper-thesis-final-economic-2026-04-06` champion. It does not reopen the",
        "champion search.",
        "",
        "## Generated artifacts",
        "",
    ]
    for artifact in status["generated_artifacts"]:
        lines.append(f"- `{artifact}`")
    lines += [
        "",
        "## Scope notes",
        "",
        "- The nested-holdout evidence is an artifact-level staged confirmation",
        "  chain: 5K screening, 25K refinement, and 276K full OOT confirmation. It",
        "  is complemented by a strict temporal funded-set confirmation split in",
        "  `paper1_tableA9_strict_temporal_holdout.csv`. That strict split evaluates",
        "  the frozen policy; it does not reopen the champion search.",
        "- The decision-aware conformal selector is a CROMS-style screen over the",
        "  three conformal finalists. Exact 276K bound-aware evaluations now exist",
        "  for ranks 1, 2 and 3, while ranks 2 and 3 still fail the conformal policy",
        "  gate through minimum group coverage.",
        "- Synthetic shift checks include both covariate reweighting and adversarial",
        "  label-flip stress scenarios on OOT labels. They are stronger than the",
        "  first pass, but they are still not an external dataset replacement.",
        "",
        "## Key status",
        "",
        f"- Nested final return: `{status['nested_holdout']['final_return']:.6f}`.",
        f"- Nested final V: `{status['nested_holdout']['final_V']:.6f}`.",
        f"- Decision-aware selected rank: `{status['decision_aware_selector']['selected_rank']}`.",
        f"- Worst segment coverage 90: `{status['segment_period']['worst_coverage_90']:.6f}`.",
        f"- Worst synthetic coverage 90: `{status['synthetic_shift']['worst_coverage_90']:.6f}`.",
        "",
    ]
    hardening_keys = [
        "strict_temporal_holdout",
        "funded_set_export",
        "funded_set_composition",
        "conformal_finalist_exact_eval",
        "enhanced_synthetic_shift",
        "conditional_tightening",
    ]
    if any(key in status for key in hardening_keys):
        lines += ["## Hardening status", ""]
        for key in hardening_keys:
            if key in status:
                payload = status[key]
                lines.append(f"- `{key}`: `{payload.get('status', 'unknown')}`.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_p1_evidence(*, include_hardening: bool = False) -> dict[str, Any]:
    promotion = _load_json(PROMOTION_PATH)
    nested = _build_nested_holdout_table(promotion)
    oot = _load_joined_oot()
    segment = _build_segment_period_table(oot)
    synthetic = _build_synthetic_shift_table(oot)

    artifacts: list[Path] = []
    if include_hardening:
        artifacts += build_p1_hardening_tables(promotion, oot)

    selector = _build_decision_aware_selector_table(promotion)
    artifacts += _write_table("paper1_tableA3_nested_holdout", nested)
    artifacts += _write_table("paper1_tableA4_segment_period_sensitivity", segment)
    artifacts += _write_table("paper1_tableA5_decision_aware_selector", selector)
    artifacts += _write_table("paper1_tableA6_synthetic_shift", synthetic)

    final_nested = nested.loc[nested["stage"].eq("bound_aware_276k")].iloc[0]
    selected_selector = selector.loc[selector["decision_aware_selected"]].iloc[0]
    worst_segment = segment.sort_values("coverage_90", ascending=True).iloc[0]
    worst_shift = synthetic.sort_values("weighted_coverage_90", ascending=True).iloc[0]
    status = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": 1,
        "run_tag": promotion["run_tag"],
        "champion_label": promotion["final_champion"]["label"],
        "generated_artifacts": _relative_artifacts(artifacts),
        "source_artifacts": [
            str(PROMOTION_PATH.relative_to(ROOT)),
            str(CONFORMAL_CANDIDATES_PATH.relative_to(ROOT)),
            str(CONFORMAL_WINNER_INTERVALS_PATH.relative_to(ROOT)),
            str(PORTFOLIO_FINALIST_PATH.relative_to(ROOT)),
            str(TEST_PATH.relative_to(ROOT)),
        ],
        "nested_holdout": {
            "scope": "staged_5k_25k_276k_post_selection_confirmation",
            "strict_disjoint_split": False,
            "final_return": float(final_nested["realized_total_return"]),
            "final_V": float(final_nested["alpha01_weighted_miscoverage_V"]),
            "final_gamma_cp": float(final_nested["alpha01_gamma_cp"]),
            "final_alpha01_exact_pass": bool(final_nested["alpha01_exact_pass"]),
            "final_matches_champion": bool(final_nested["selected_matches_final_champion"]),
        },
        "segment_period": {
            "n_segments": int(len(segment)),
            "worst_period": str(worst_segment["period"]),
            "worst_grade": str(worst_segment["original_grade"]),
            "worst_coverage_90": float(worst_segment["coverage_90"]),
            "flagged_segments": int(segment["risk_flag"].sum()),
        },
        "decision_aware_selector": {
            "scope": "croms_style_screen_existing_artifacts",
            "selected_rank": int(selected_selector["rank"]),
            "selected_partition": str(selected_selector["partition"]),
            "selected_score": float(selected_selector["decision_aware_score"]),
            "exact_bound_available_for_all_ranks": False,
        },
        "synthetic_shift": {
            "scope": "oot_covariate_reweighting",
            "n_scenarios": int(len(synthetic)),
            "worst_scenario": str(worst_shift["scenario"]),
            "worst_coverage_90": float(worst_shift["weighted_coverage_90"]),
            "all_coverage90_pass": bool(synthetic["coverage90_pass"].all()),
        },
        "conditional_tightening": {
            "documented_in": "book/chapters/14-paper-estrella/14b-theoretical-framework.qmd",
            "status": "conditional_lemma_under_additional_independence_assumptions",
        },
    }
    _attach_hardening_status(status, artifacts)
    _write_json(STATUS_PATH, status)
    artifacts.append(STATUS_PATH)
    dossier_path = _build_markdown_dossier(status)
    artifacts.append(dossier_path)
    status["generated_artifacts"] = _relative_artifacts(artifacts)
    _write_json(STATUS_PATH, status)
    _build_markdown_dossier(status)
    print(f"Wrote {STATUS_PATH.relative_to(ROOT)}")
    print(f"Wrote {dossier_path.relative_to(ROOT)}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-hardening",
        action="store_true",
        help="Regenerate exact-solver hardening artifacts A7-A11.",
    )
    args = parser.parse_args(argv)
    build_p1_evidence(include_hardening=bool(args.include_hardening))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
