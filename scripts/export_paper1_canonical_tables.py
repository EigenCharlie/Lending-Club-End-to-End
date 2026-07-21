"""Export Paper Estrella canonical tables from final promotion artifacts.

The paper-facing tables under reports/paper_material/paper1/tables must not
mix legacy robustness runs with the final bound-aware 276k closure. This script
regenerates those CSV/TeX tables from the explicit source hierarchy used by
chapter 14:

- models/final_project_promotion.json
- data/processed/final_project_summary.parquet
- data/processed/portfolio_bound_aware/...276k.../portfolio_bound_aware_*.parquet
- data/processed/conformal_gap/...rank-1/conformal_group_metrics_mondrian.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from analyze_paper1_p1_evidence import build_p1_evidence

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "paper_material" / "paper1" / "tables"

BOUND_AWARE_DIR = (
    DATA / "portfolio_bound_aware" / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
)
CONFORMAL_REOPEN_DIR = (
    DATA / "conformal_gap" / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612"
)
CONFORMAL_WINNER_DIR = (
    DATA
    / "conformal_gap"
    / "conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612__phase1__final__rank-1"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    return bool(value) if pd.notna(value) else False


def _policy_match(row: pd.Series, policy: dict[str, Any] | None) -> bool:
    if not policy:
        return False
    fields = [
        "risk_tolerance",
        "policy_mode",
        "gamma",
        "delta_cap_quantile",
        "tail_focus_quantile",
        "uncertainty_aversion",
        "min_budget_utilization",
        "pd_cap_slack_penalty",
    ]
    for field in fields:
        if field not in row or field not in policy:
            return False
        left = row[field]
        right = policy[field]
        if isinstance(right, str):
            if str(left) != right:
                return False
        elif abs(float(left) - float(right)) > 1e-9:
            return False
    return True


def _write_table(name: str, frame: pd.DataFrame) -> None:
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


def _table0_key_metrics(
    promotion: dict[str, Any],
    pipeline_summary: dict[str, Any],
    dvc_metrics: dict[str, Any],
) -> pd.DataFrame:
    champ = promotion["final_champion"]
    conformal = promotion["conformal_upstream"]["winner_metrics"]
    rows = [
        ("run_tag", promotion["run_tag"]),
        ("champion_label", champ["label"]),
        ("pd_auc", pipeline_summary["pd_auc"]),
        ("pd_brier", pipeline_summary["pd_brier"]),
        ("pd_ece_pipeline_summary", pipeline_summary["pd_ece"]),
        ("pd_ece_dvc_metrics", dvc_metrics["pd.ece"]),
        ("coverage_90", conformal["coverage_90"]),
        ("coverage_95", conformal["coverage_95"]),
        ("avg_width_90", conformal["avg_width_90"]),
        ("min_group_coverage_90", conformal["min_group_coverage_90"]),
        ("winkler_90", conformal["winkler_90"]),
        ("robust_return", champ["realized_total_return"]),
        ("price_of_robustness", champ["price_of_robustness"]),
        ("price_of_robustness_pct", champ["price_of_robustness_pct"]),
        ("alpha01_exact_pass", champ["alpha01_exact_pass"]),
        ("alpha01_weighted_miscoverage_V", champ["alpha01_weighted_miscoverage_V"]),
        ("alpha01_gamma_cp", champ["alpha01_gamma_cp"]),
        ("alpha01_violation", champ["alpha01_violation"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _table1_robustness_summary(
    shortlist: pd.DataFrame,
    promotion: dict[str, Any],
) -> pd.DataFrame:
    champ = promotion["final_champion"]
    work = shortlist.sort_values(
        ["risk_tolerance", "realized_total_return"],
        ascending=[True, False],
    )
    best = work.groupby("risk_tolerance", as_index=False).head(1).copy()
    best["selected_for_champion"] = best.apply(lambda row: _policy_match(row, champ), axis=1)
    columns = {
        "realized_total_return": "best_robust_realized_return",
        "policy_mode": "best_robust_policy_mode",
        "gamma": "best_robust_gamma",
        "uncertainty_aversion": "best_robust_uncertainty_aversion",
        "n_funded": "best_robust_funded",
        "ab_pass_all": "ab_pass",
    }
    keep = [
        "risk_tolerance",
        "realized_total_return",
        "policy_mode",
        "gamma",
        "uncertainty_aversion",
        "n_funded",
        "price_of_robustness",
        "price_of_robustness_pct",
        "ab_pass_all",
        "alpha01_exact_pass",
        "alpha01_weighted_miscoverage_V",
        "alpha01_gamma_cp",
        "alpha01_violation",
        "selected_for_champion",
    ]
    return best[keep].rename(columns=columns).reset_index(drop=True)


def _table2_conformal_benchmark(promotion: dict[str, Any]) -> pd.DataFrame:
    candidates = pd.read_parquet(
        CONFORMAL_REOPEN_DIR / "conformal_reopen_phase1_final_candidates.parquet"
    )
    conformal = promotion["conformal_upstream"]["winner_metrics"]
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        is_winner = str(row["namespace"]).endswith("rank-1")
        rows.append(
            {
                "variant": "score_decile_mondrian" if is_winner else str(row["partition"]),
                "namespace": row["namespace"],
                "partition": row["partition"],
                "coverage_90": conformal["coverage_90"] if is_winner else row["coverage_90"],
                "coverage_95": conformal["coverage_95"] if is_winner else pd.NA,
                "avg_width_90": conformal["avg_width_90"] if is_winner else row["avg_width_90"],
                "min_group_coverage_90": conformal["min_group_coverage_90"]
                if is_winner
                else row["min_group_coverage_90"],
                "winkler_90": conformal["winkler_90"] if is_winner else pd.NA,
                "policy_overall_pass": _as_bool(row["policy_overall_pass"]),
                "strict_overall_pass": _as_bool(row["strict_overall_pass"]),
                "methodological_justification_pass": _as_bool(
                    row["methodological_justification_pass"]
                ),
                "promotion_pass": bool(is_winner),
            }
        )
    return pd.DataFrame(rows)


def _table_a1_group_benchmark() -> pd.DataFrame:
    groups = pd.read_parquet(CONFORMAL_WINNER_DIR / "conformal_group_metrics_mondrian.parquet")
    groups = groups.copy()
    groups.insert(0, "variant", "score_decile_mondrian")
    return groups[
        [
            "variant",
            "group",
            "n",
            "coverage_90",
            "avg_width_90",
            "median_width_90",
            "coverage_95",
            "avg_width_95",
            "median_width_95",
            "coverage_before",
            "coverage_after",
            "multiplier",
            "adjusted",
        ]
    ]


def _table_a2_frontier(shortlist: pd.DataFrame, promotion: dict[str, Any]) -> pd.DataFrame:
    champion = promotion["final_champion"]
    theorem_tight = promotion.get("theorem_tight_comparator")
    balanced = promotion.get("balanced_comparator")
    frontier = shortlist.sort_values("realized_total_return", ascending=False).copy()
    frontier.insert(0, "policy", "robust")
    frontier["selected_for_champion"] = frontier.apply(
        lambda row: _policy_match(row, champion), axis=1
    )
    frontier["selected_for_theorem_tight"] = frontier.apply(
        lambda row: _policy_match(row, theorem_tight),
        axis=1,
    )
    frontier["selected_for_balanced"] = frontier.apply(
        lambda row: _policy_match(row, balanced), axis=1
    )
    keep = [
        "policy",
        "policy_mode",
        "gamma",
        "risk_tolerance",
        "uncertainty_aversion",
        "price_of_robustness",
        "price_of_robustness_pct",
        "realized_total_return",
        "ab_diff_total_return",
        "ab_pass_all",
        "solver_backend",
        "n_funded",
        "total_allocated",
        "expected_return_net_point",
        "worst_case_pd",
        "point_pd",
        "alpha01_exact_pass",
        "alpha01_gamma_cp",
        "alpha01_weighted_miscoverage_V",
        "alpha01_violation",
        "alpha03_exact_pass",
        "alpha10_exact_pass",
        "selected_for_champion",
        "selected_for_theorem_tight",
        "selected_for_balanced",
    ]
    return frontier[keep].rename(columns={"ab_pass_all": "ab_pass"}).reset_index(drop=True)


def main() -> int:
    promotion = _load_json(MODELS / "final_project_promotion.json")
    pipeline_summary = _load_json(DATA / "pipeline_summary.json")
    dvc_metrics = _load_json(ROOT / "reports" / "dvc" / "metrics_summary.json")["metrics"]
    shortlist = pd.read_parquet(BOUND_AWARE_DIR / "portfolio_bound_aware_shortlist.parquet")

    _write_table(
        "paper1_table0_key_metrics",
        _table0_key_metrics(promotion, pipeline_summary, dvc_metrics),
    )
    _write_table(
        "paper1_table1_robustness_summary",
        _table1_robustness_summary(shortlist, promotion),
    )
    _write_table(
        "paper1_table2_conformal_variant_benchmark", _table2_conformal_benchmark(promotion)
    )
    _write_table("paper1_tableA1_benchmark_by_group", _table_a1_group_benchmark())
    _write_table("paper1_tableA2_robustness_frontier", _table_a2_frontier(shortlist, promotion))
    build_p1_evidence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
