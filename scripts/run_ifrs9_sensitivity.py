"""IFRS9 scenario and sensitivity analysis with stronger input governance.

Improvements over prior version:
- Uses real delinquency/status fields to derive DPD staging signals.
- Uses vintage-based origination PD (grade + issue quarter) from train history.
- Uses grade-level lifetime PD table when available.

Usage:
    uv run python scripts/run_ifrs9_sensitivity.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.conformal_artifacts import load_conformal_intervals
from src.evaluation.ifrs9 import assign_stage, compute_ecl, ecl_with_conformal_range


def _load_intervals() -> pd.DataFrame:
    df, path, is_legacy = load_conformal_intervals(allow_legacy_fallback=True)
    logger.info(f"Loaded intervals: {path} ({len(df):,}, legacy={is_legacy})")
    return df


def _load_raw_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path("data/processed/train.parquet")
    test_path = Path("data/processed/test.parquet")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train/test raw splits in data/processed/")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    logger.info(f"Loaded train raw: {train_path} ({len(train):,})")
    logger.info(f"Loaded test raw: {test_path} ({len(test):,})")
    return train, test


def _load_lifetime_table() -> pd.DataFrame | None:
    path = Path("data/processed/lifetime_pd_table.parquet")
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    if table.index.name is None and "Grade" in table.columns:
        table = table.set_index("Grade")
    logger.info(f"Loaded lifetime PD table: {path}")
    return table


def _to_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(arr, nan=default)


def _derive_dpd(df: pd.DataFrame) -> np.ndarray:
    """Derive DPD-style signal from status and delinquency columns."""
    n = len(df)
    dpd = np.zeros(n, dtype=float)
    status = df.get("loan_status", pd.Series(["unknown"] * n)).astype(str).str.lower()

    # Direct status mapping when available.
    dpd = np.where(status.str.contains("charged off|default", na=False), 120.0, dpd)
    dpd = np.where(status.str.contains("late \\(31-120 days\\)", na=False), np.maximum(dpd, 60.0), dpd)
    dpd = np.where(status.str.contains("late \\(16-30 days\\)|grace", na=False), np.maximum(dpd, 20.0), dpd)

    # Delinquency features as additional evidence.
    num_90 = _to_numeric(df, "num_tl_90g_dpd_24m", default=0.0)
    num_30 = _to_numeric(df, "num_tl_30dpd", default=0.0)
    acc_now = _to_numeric(df, "acc_now_delinq", default=0.0)
    charge12 = _to_numeric(df, "chargeoff_within_12_mths", default=0.0)

    dpd = np.where(num_30 > 0, np.maximum(dpd, 30.0), dpd)
    dpd = np.where(acc_now > 0, np.maximum(dpd, 30.0), dpd)
    dpd = np.where(num_90 > 0, np.maximum(dpd, 60.0), dpd)
    dpd = np.where(charge12 > 0, np.maximum(dpd, 90.0), dpd)
    return dpd


def _build_origination_pd(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Estimate origination PD from historical vintage rates (grade + issue quarter)."""
    if "issue_d" not in train.columns or "issue_d" not in test.columns or "grade" not in train.columns or "grade" not in test.columns:
        return np.full(len(test), float(train["default_flag"].mean()), dtype=float)

    tr = train[["issue_d", "grade", "default_flag"]].copy()
    te = test[["issue_d", "grade"]].copy()
    tr["issue_q"] = pd.to_datetime(tr["issue_d"], errors="coerce").dt.to_period("Q").astype(str)
    te["issue_q"] = pd.to_datetime(te["issue_d"], errors="coerce").dt.to_period("Q").astype(str)
    tr["grade"] = tr["grade"].astype(str)
    te["grade"] = te["grade"].astype(str)

    by_grade_q = tr.groupby(["grade", "issue_q"], observed=True)["default_flag"].mean().rename("pd_orig")
    by_grade = tr.groupby("grade", observed=True)["default_flag"].mean().rename("pd_grade")
    global_pd = float(tr["default_flag"].mean())

    te = te.join(by_grade_q, on=["grade", "issue_q"])
    te = te.join(by_grade, on="grade")
    pd_orig = te["pd_orig"].fillna(te["pd_grade"]).fillna(global_pd).to_numpy(dtype=float)
    return np.clip(pd_orig, 0.0001, 0.9999)


def _prepare_base_vectors(intervals: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    n = min(len(intervals), len(test))
    if len(intervals) != len(test):
        logger.warning(f"Length mismatch intervals={len(intervals):,}, test={len(test):,}. Using first {n:,} rows.")

    ints = intervals.iloc[:n].reset_index(drop=True)
    tst = test.iloc[:n].reset_index(drop=True)

    pd_point = ints["y_pred"].to_numpy(dtype=float) if "y_pred" in ints.columns else ints["pd_point"].to_numpy(dtype=float)
    pd_low = ints["pd_low_90"].to_numpy(dtype=float) if "pd_low_90" in ints.columns else ints["pd_low"].to_numpy(dtype=float)
    pd_high = ints["pd_high_90"].to_numpy(dtype=float) if "pd_high_90" in ints.columns else ints["pd_high"].to_numpy(dtype=float)

    loan_amnt = _to_numeric(tst, "loan_amnt", default=10_000.0)
    grade = tst.get("grade", pd.Series(["UNKNOWN"] * n)).astype(str).fillna("UNKNOWN").to_numpy(dtype=str)
    dpd = _derive_dpd(tst)
    pd_orig = _build_origination_pd(train, tst)

    base = {
        "pd_point": np.clip(pd_point, 0.0, 1.0),
        "pd_low": np.clip(pd_low, 0.0, 1.0),
        "pd_high": np.clip(pd_high, 0.0, 1.0),
        "loan_amnt": loan_amnt,
        "grade": grade,
        "dpd": dpd,
        "pd_orig": pd_orig,
    }
    quality = pd.DataFrame(
        [
            {
                "n_rows": int(n),
                "pd_orig_mean": float(np.mean(pd_orig)),
                "pd_current_mean": float(np.mean(base["pd_point"])),
                "dpd_ge_30_share": float(np.mean(dpd >= 30)),
                "dpd_ge_90_share": float(np.mean(dpd >= 90)),
                "stage3_status_share": float(np.mean(dpd >= 90)),
                "source_pd_orig": "train_vintage_grade_issue_quarter",
                "source_dpd": "loan_status_and_delinquency_features",
            }
        ]
    )
    return base, quality


def _scenario_config() -> dict[str, dict[str, float]]:
    return {
        "baseline": {"pd_mult": 1.00, "lgd_mult": 1.00, "ead_mult": 1.00, "discount_rate": 0.05},
        "mild_stress": {"pd_mult": 1.15, "lgd_mult": 1.05, "ead_mult": 1.02, "discount_rate": 0.06},
        "adverse": {"pd_mult": 1.30, "lgd_mult": 1.12, "ead_mult": 1.05, "discount_rate": 0.07},
        "severe": {"pd_mult": 1.55, "lgd_mult": 1.20, "ead_mult": 1.10, "discount_rate": 0.08},
    }


def _lifetime_pd(
    pd12: np.ndarray,
    grade: np.ndarray,
    pd_mult: float,
    lifetime_table: pd.DataFrame | None,
) -> tuple[np.ndarray, str]:
    if lifetime_table is not None and "PD_60m" in lifetime_table.columns:
        table = lifetime_table.copy()
        if table.index.name is None and "Grade" in table.columns:
            table = table.set_index("Grade")
        mapped = pd.Series(grade).map(table["PD_60m"]).fillna(np.nan).to_numpy(dtype=float)
        fallback = np.clip(1.0 - np.power(1.0 - pd12, 5.0), 0.0, 1.0)
        life = np.where(np.isfinite(mapped), np.clip(mapped * pd_mult, 0.0, 1.0), fallback)
        return life, "grade_lifetime_pd_table_scaled"
    return np.clip(1.0 - np.power(1.0 - pd12, 5.0), 0.0, 1.0), "formula_1_minus_1_minus_pd_power_5"


def _run_single_scenario(
    scenario: str,
    params: dict[str, float],
    base: dict[str, np.ndarray],
    base_lgd: float,
    lifetime_table: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pd12 = np.clip(base["pd_point"] * params["pd_mult"], 0.0, 1.0)
    pd_low = np.clip(base["pd_low"] * params["pd_mult"], 0.0, 1.0)
    pd_high = np.clip(base["pd_high"] * params["pd_mult"], 0.0, 1.0)
    lgd = np.clip(np.full_like(pd12, base_lgd) * params["lgd_mult"], 0.0, 1.0)
    ead = base["loan_amnt"] * params["ead_mult"]

    lifetime_pd, lifetime_source = _lifetime_pd(
        pd12=pd12,
        grade=base["grade"],
        pd_mult=float(params["pd_mult"]),
        lifetime_table=lifetime_table,
    )

    stages = assign_stage(
        pd_origination=base["pd_orig"],
        pd_current=pd12,
        dpd=base["dpd"],
        pd_high=pd_high,
    )
    ecl_df = compute_ecl(
        pd_values=pd12,
        lgd_values=lgd,
        ead_values=ead,
        stages=stages,
        lifetime_pd=lifetime_pd,
        discount_rate=float(params["discount_rate"]),
    )
    ecl_range = ecl_with_conformal_range(
        pd_low=pd_low,
        pd_point=pd12,
        pd_high=pd_high,
        lgd=lgd,
        ead=ead,
        stages=stages,
    )

    summary = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "pd_mult": float(params["pd_mult"]),
                "lgd_mult": float(params["lgd_mult"]),
                "ead_mult": float(params["ead_mult"]),
                "discount_rate": float(params["discount_rate"]),
                "n_loans": int(len(pd12)),
                "stage1_n": int(np.sum(stages == 1)),
                "stage2_n": int(np.sum(stages == 2)),
                "stage3_n": int(np.sum(stages == 3)),
                "stage1_share": float(np.mean(stages == 1)),
                "stage2_share": float(np.mean(stages == 2)),
                "stage3_share": float(np.mean(stages == 3)),
                "total_ecl": float(ecl_df["ecl"].sum()),
                "avg_ecl": float(ecl_df["ecl"].mean()),
                "total_ecl_low": float(ecl_range["ecl_low"].sum()),
                "total_ecl_point": float(ecl_range["ecl_point"].sum()),
                "total_ecl_high": float(ecl_range["ecl_high"].sum()),
                "total_ecl_range": float((ecl_range["ecl_high"] - ecl_range["ecl_low"]).sum()),
                "pd_orig_source": "train_vintage_grade_issue_quarter",
                "dpd_source": "loan_status_and_delinquency_features",
                "lifetime_pd_source": lifetime_source,
            }
        ]
    )

    grade_df = pd.DataFrame(
        {
            "scenario": scenario,
            "grade": base["grade"],
            "stage": stages,
            "ecl": ecl_df["ecl"].to_numpy(dtype=float),
            "ecl_low": ecl_range["ecl_low"].to_numpy(dtype=float),
            "ecl_high": ecl_range["ecl_high"].to_numpy(dtype=float),
        }
    )
    grade_summary = (
        grade_df.groupby(["scenario", "grade"], observed=True)
        .agg(
            n=("ecl", "size"),
            stage2_share=("stage", lambda s: float(np.mean(np.asarray(s) == 2))),
            stage3_share=("stage", lambda s: float(np.mean(np.asarray(s) == 3))),
            total_ecl=("ecl", "sum"),
            avg_ecl=("ecl", "mean"),
            total_ecl_low=("ecl_low", "sum"),
            total_ecl_high=("ecl_high", "sum"),
        )
        .reset_index()
        .sort_values(["scenario", "grade"])
    )
    grade_summary["total_ecl_range"] = grade_summary["total_ecl_high"] - grade_summary["total_ecl_low"]
    return summary, grade_summary


def _sensitivity_grid(base: dict[str, np.ndarray], base_lgd: float, lifetime_table: pd.DataFrame | None) -> pd.DataFrame:
    pd_mult_grid = [0.90, 1.00, 1.10, 1.20, 1.30]
    lgd_mult_grid = [0.90, 1.00, 1.10, 1.20]
    discount_grid = [0.04, 0.05, 0.06]
    rows = []

    for pd_mult in pd_mult_grid:
        for lgd_mult in lgd_mult_grid:
            for disc in discount_grid:
                pd12 = np.clip(base["pd_point"] * pd_mult, 0.0, 1.0)
                pd_low = np.clip(base["pd_low"] * pd_mult, 0.0, 1.0)
                pd_high = np.clip(base["pd_high"] * pd_mult, 0.0, 1.0)
                lgd = np.clip(np.full_like(pd12, base_lgd) * lgd_mult, 0.0, 1.0)
                ead = base["loan_amnt"]
                lifetime_pd, _ = _lifetime_pd(pd12, base["grade"], pd_mult=pd_mult, lifetime_table=lifetime_table)
                stages = assign_stage(base["pd_orig"], pd12, dpd=base["dpd"], pd_high=pd_high)
                ecl_df = compute_ecl(
                    pd_values=pd12,
                    lgd_values=lgd,
                    ead_values=ead,
                    stages=stages,
                    lifetime_pd=lifetime_pd,
                    discount_rate=float(disc),
                )
                ecl_range = ecl_with_conformal_range(pd_low, pd12, pd_high, lgd, ead, stages)
                rows.append(
                    {
                        "pd_mult": float(pd_mult),
                        "lgd_mult": float(lgd_mult),
                        "discount_rate": float(disc),
                        "stage2_share": float(np.mean(stages == 2)),
                        "stage3_share": float(np.mean(stages == 3)),
                        "total_ecl": float(ecl_df["ecl"].sum()),
                        "total_ecl_low": float(ecl_range["ecl_low"].sum()),
                        "total_ecl_high": float(ecl_range["ecl_high"].sum()),
                    }
                )

    out = pd.DataFrame(rows).sort_values(["pd_mult", "lgd_mult", "discount_rate"]).reset_index(drop=True)
    out["total_ecl_range"] = out["total_ecl_high"] - out["total_ecl_low"]
    return out


def main(base_lgd: float = 0.45):
    intervals = _load_intervals()
    train_raw, test_raw = _load_raw_splits()
    lifetime_table = _load_lifetime_table()
    base, quality = _prepare_base_vectors(intervals=intervals, train=train_raw, test=test_raw)

    # Keep console output concise during large sensitivity loops.
    logger.disable("src.evaluation.ifrs9")

    scenario_rows = []
    grade_rows = []
    for scenario, params in _scenario_config().items():
        s, g = _run_single_scenario(scenario, params, base, base_lgd=base_lgd, lifetime_table=lifetime_table)
        scenario_rows.append(s)
        grade_rows.append(g)
    scenario_summary = pd.concat(scenario_rows, ignore_index=True)
    grade_summary = pd.concat(grade_rows, ignore_index=True)
    sensitivity = _sensitivity_grid(base, base_lgd=base_lgd, lifetime_table=lifetime_table)

    logger.enable("src.evaluation.ifrs9")

    data_dir = Path("data/processed")
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = data_dir / "ifrs9_scenario_summary.parquet"
    grade_path = data_dir / "ifrs9_scenario_grade_summary.parquet"
    sensitivity_path = data_dir / "ifrs9_sensitivity_grid.parquet"
    quality_path = data_dir / "ifrs9_input_quality.parquet"
    scenario_summary.to_parquet(scenario_path, index=False)
    grade_summary.to_parquet(grade_path, index=False)
    sensitivity.to_parquet(sensitivity_path, index=False)
    quality.to_parquet(quality_path, index=False)

    with open(model_dir / "ifrs9_sensitivity_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "scenario_summary": scenario_summary.to_dict(orient="records"),
                "sensitivity_extremes": {
                    "min_total_ecl": float(sensitivity["total_ecl"].min()),
                    "max_total_ecl": float(sensitivity["total_ecl"].max()),
                },
                "input_quality": quality.to_dict(orient="records")[0],
            },
            f,
        )

    base_ecl = float(scenario_summary.loc[scenario_summary["scenario"] == "baseline", "total_ecl"].iloc[0])
    severe_ecl = float(scenario_summary.loc[scenario_summary["scenario"] == "severe", "total_ecl"].iloc[0])

    logger.info(f"Saved IFRS9 scenario summary: {scenario_path}")
    logger.info(f"Saved IFRS9 grade summary: {grade_path}")
    logger.info(f"Saved IFRS9 sensitivity grid: {sensitivity_path}")
    logger.info(f"Saved IFRS9 input quality: {quality_path}")
    logger.info(f"Baseline ECL={base_ecl:,.0f}, Severe ECL={severe_ecl:,.0f}, Uplift={(severe_ecl/base_ecl-1)*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_lgd", type=float, default=0.45)
    args = parser.parse_args()
    main(base_lgd=args.base_lgd)
