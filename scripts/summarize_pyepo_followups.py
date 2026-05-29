"""Summarize PyEPO follow-up runs for Paper 4 and Paper Estrella docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "reports" / "paper_material" / "paper4" / "tables"
STATUS_DIR = REPO_ROOT / "reports" / "paper_material" / "paper4" / "status"

PERIODS = {
    "2018H1": ("2018-01-01", "2018-07-01"),
    "2018H2": ("2018-07-01", "2019-01-01"),
    "2019H1": ("2019-01-01", "2019-07-01"),
    "2019H2": ("2019-07-01", "2020-01-01"),
    "2020": ("2020-01-01", "2021-01-01"),
}

RUNS = {
    "paper4_full_canonical": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_topk_full_20260528",
        "role": "canonical_full",
    },
    "paper4_temporal_canonical": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_topk_temporal_20260528",
        "role": "canonical_temporal",
    },
    "pfyl_risk_only_full": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_pfyl_risk_only_full_20260528",
        "role": "pfyl_ablation",
    },
    "cave_maxiter1_full": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_cave_maxiter1_full_20260528",
        "role": "cave_sensitivity",
    },
    "cave_maxiter5_medium": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_cave_maxiter5_medium_20260528",
        "role": "cave_sensitivity",
    },
    "spoplus_lr5e4_medium": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_spoplus_lr5e4_medium_20260528",
        "role": "spoplus_robustness",
    },
    "spoplus_topk50_medium": {
        "dir": STATUS_DIR / "paper4_pyepo137_wls_spoplus_topk50_medium_20260528",
        "role": "spoplus_robustness",
    },
}


def _load_status(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "pyepo_real_suite_status.json").open() as f:
        return json.load(f)


def _summary_rows(name: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    status = _load_status(meta["dir"])
    config = status.get("config", {})
    data = status.get("data", {})
    rows = []
    for row in status.get("results", {}).get("summary", []):
        rows.append(
            {
                "run_name": name,
                "run_tag": status.get("run_tag"),
                "role": meta["role"],
                "method": row["method"],
                "method_display": row["method_display"],
                "mean_regret": row["mean_regret"],
                "std_regret": row["std_regret"],
                "median_regret": row["median_regret"],
                "improvement_vs_two_stage_pct": row["improvement_vs_two_stage_pct"],
                "n_observations": row["n_observations"],
                "mode": status.get("mode"),
                "cost_target": config.get("cost_target", data.get("cost_target", "economic")),
                "cost_definition": data.get("cost_definition", "calibrated_pd * LGD - int_rate"),
                "n_items": config.get("n_items"),
                "budget": config.get("budget"),
                "n_train_instances": config.get("n_train_instances"),
                "n_test_instances": config.get("n_test_instances"),
                "epochs": config.get("epochs"),
                "seeds": config.get("seeds"),
                "lr": config.get("lr"),
                "cave_max_iter": config.get("cave_max_iter", 3),
                "runtime_seconds": status.get("runtime_seconds"),
                "artifact_dir": str(meta["dir"].relative_to(REPO_ROOT)),
            }
        )
    return rows


def _assign_periods(issue_d: pd.Series) -> pd.Series:
    dt = pd.to_datetime(issue_d)
    result = pd.Series("", index=dt.index, dtype=str)
    for period, (start, end) in PERIODS.items():
        mask = (dt >= pd.Timestamp(start)) & (dt < pd.Timestamp(end))
        result.loc[mask] = period
    return result


def _coverage_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ci = pd.read_parquet(REPO_ROOT / "data" / "processed" / "conformal_intervals_mondrian.parquet")
    test = pd.read_parquet(REPO_ROOT / "data" / "processed" / "test_fe.parquet")
    if len(ci) != len(test):
        raise RuntimeError(f"Conformal rows ({len(ci)}) do not match test rows ({len(test)})")

    work = ci.copy()
    work["period"] = _assign_periods(test["issue_d"]).values
    if "grade" not in work.columns and "grade" in test.columns:
        work["grade"] = test["grade"].astype(str).values
    if "default_flag" in test.columns:
        work["default_flag"] = test["default_flag"].astype(float).values

    work["_covered_90"] = (
        (work["y_true"] >= work["pd_low_90"]) & (work["y_true"] <= work["pd_high_90"])
    ).astype(float)
    work["_covered_95"] = (
        (work["y_true"] >= work["pd_low_95"]) & (work["y_true"] <= work["pd_high_95"])
    ).astype(float)
    work["_width_90"] = work["pd_high_90"] - work["pd_low_90"]
    work = work[work["period"].isin(PERIODS.keys())].copy()

    grade = (
        work.groupby(["period", "grade"], as_index=False)
        .agg(
            n_loans=("y_true", "size"),
            default_rate=("default_flag", "mean"),
            coverage_90=("_covered_90", "mean"),
            coverage_95=("_covered_95", "mean"),
            avg_width_90=("_width_90", "mean"),
        )
        .sort_values(["period", "grade"])
    )
    period = (
        work.groupby("period", as_index=False)
        .agg(
            n_loans=("y_true", "size"),
            default_rate=("default_flag", "mean"),
            coverage_90=("_covered_90", "mean"),
            coverage_95=("_covered_95", "mean"),
            avg_width_90=("_width_90", "mean"),
            min_grade_coverage_90=("_covered_90", "min"),
        )
        .merge(
            grade.groupby("period", as_index=False)["coverage_90"]
            .min()
            .rename(columns={"coverage_90": "min_grade_coverage_90"}),
            on="period",
            suffixes=("_loan_min", ""),
        )
        .drop(columns=["min_grade_coverage_90_loan_min"])
    )
    period["period"] = pd.Categorical(period["period"], categories=list(PERIODS), ordered=True)
    grade["period"] = pd.Categorical(grade["period"], categories=list(PERIODS), ordered=True)
    return period.sort_values("period").reset_index(drop=True), grade.reset_index(drop=True)


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, meta in RUNS.items():
        rows.extend(_summary_rows(name, meta))
    followups = pd.DataFrame(rows)
    followups.to_csv(TABLE_DIR / "pyepo_real_suite_followup_summary_20260528.csv", index=False)

    cave = followups[
        (followups["method"] == "cave")
        & (
            followups["run_name"].isin(
                ["paper4_full_canonical", "cave_maxiter1_full", "cave_maxiter5_medium"]
            )
        )
    ].copy()
    cave["scope"] = np.where(cave["n_train_instances"] == 2000, "full", "medium")
    cave.to_csv(TABLE_DIR / "pyepo_real_suite_cave_sensitivity_20260528.csv", index=False)

    pfyl = followups[
        (followups["method"] == "pfyl_mul")
        & (followups["run_name"].isin(["paper4_full_canonical", "pfyl_risk_only_full"]))
    ].copy()
    pfyl.to_csv(TABLE_DIR / "pyepo_real_suite_pfyl_ablation_20260528.csv", index=False)

    spo = followups[
        (followups["method"] == "spo_plus")
        & (
            followups["run_name"].isin(
                ["paper4_full_canonical", "spoplus_lr5e4_medium", "spoplus_topk50_medium"]
            )
        )
    ].copy()
    spo.to_csv(TABLE_DIR / "pyepo_real_suite_spoplus_robustness_20260528.csv", index=False)

    period_cov, grade_cov = _coverage_tables()
    period_cov.to_csv(TABLE_DIR / "pyepo_real_suite_temporal_coverage_20260528.csv", index=False)
    grade_cov.to_csv(
        TABLE_DIR / "pyepo_real_suite_temporal_coverage_by_period_grade_20260528.csv",
        index=False,
    )

    print(f"Wrote {len(followups)} follow-up summary rows")
    print(f"Wrote {len(grade_cov)} period-grade coverage rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
