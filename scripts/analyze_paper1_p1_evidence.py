"""Generate journal-grade P1 evidence tables for Paper Estrella.

The outputs in ``reports/paper_material/paper1/tables`` are deliberately
derived from the current canonical champion artifacts. This script does not
reopen the champion search; it documents post-selection confirmation, segment
sensitivity, a CROMS-style decision-aware conformal screen, and synthetic shift
stress checks around the official economic champion.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        "  is stronger than a single final table, but it is not a fresh strict",
        "  disjoint funded-set split.",
        "- The decision-aware conformal selector is a CROMS-style screen over the",
        "  three conformal finalists plus the final exact bound-aware champion.",
        "  Only rank 1 has final 276K exact bound-aware metrics because ranks 2 and",
        "  3 failed the conformal policy gate.",
        "- Synthetic shift checks are covariate-reweighting stress scenarios on OOT",
        "  labels; they are not an external dataset replacement.",
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
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_p1_evidence() -> dict[str, Any]:
    promotion = _load_json(PROMOTION_PATH)
    nested = _build_nested_holdout_table(promotion)
    selector = _build_decision_aware_selector_table(promotion)
    oot = _load_joined_oot()
    segment = _build_segment_period_table(oot)
    synthetic = _build_synthetic_shift_table(oot)

    artifacts: list[Path] = []
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
        "generated_artifacts": [str(path.relative_to(ROOT)) for path in artifacts],
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
    _write_json(STATUS_PATH, status)
    artifacts.append(STATUS_PATH)
    dossier_path = _build_markdown_dossier(status)
    artifacts.append(dossier_path)
    status["generated_artifacts"] = [str(path.relative_to(ROOT)) for path in artifacts]
    _write_json(STATUS_PATH, status)
    _build_markdown_dossier(status)
    print(f"Wrote {STATUS_PATH.relative_to(ROOT)}")
    print(f"Wrote {dossier_path.relative_to(ROOT)}")
    return status


def main() -> int:
    build_p1_evidence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
