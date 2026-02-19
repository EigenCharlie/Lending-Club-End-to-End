"""Temporal backtesting for selected causal policy rule.

Evaluates month-by-month stability of the selected policy using issue-date vintages.

Usage:
    uv run python scripts/backtest_causal_policy_oot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _load_selected_rule(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected rule status: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rule = payload.get("selected_rule")
    if not rule:
        raise ValueError(f"selected_rule not found in {path}")
    return str(rule)


def _policy_mask(df: pd.DataFrame, rule_name: str, train_ref: pd.DataFrame) -> np.ndarray:
    if rule_name == "high_only":
        return df["segment"].eq("high_sensitivity").to_numpy()
    if rule_name == "high_plus_medium_all":
        return df["segment"].isin(["high_sensitivity", "medium_sensitivity"]).to_numpy()
    if rule_name == "high_plus_medium_positive":
        return (
            df["segment"].eq("high_sensitivity")
            | (df["segment"].eq("medium_sensitivity") & (df["net_value"] > 0))
        ).to_numpy()
    if rule_name == "top15_cate":
        q85 = float(train_ref["cate"].quantile(0.85))
        return (df["cate"] >= q85).to_numpy()
    if rule_name == "top10_cate":
        q90 = float(train_ref["cate"].quantile(0.90))
        return (df["cate"] >= q90).to_numpy()
    raise ValueError(f"Unsupported rule name: {rule_name}")


def main(
    min_history_months: int = 12,
    selected_rule_path: str = "models/causal_policy_rule.json",
) -> None:
    sim_path = Path("data/processed/causal_policy_simulation.parquet")
    train_path = Path("data/processed/train_fe.parquet")
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing policy simulation artifact: {sim_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training split with issue dates: {train_path}")

    sim = pd.read_parquet(sim_path)
    train = pd.read_parquet(train_path, columns=["id", "issue_d", "grade"])
    if "id" not in sim.columns:
        raise KeyError("causal_policy_simulation.parquet must include `id` for temporal backtesting.")

    sim["id"] = sim["id"].astype(str)
    train["id"] = train["id"].astype(str)
    train["issue_d"] = pd.to_datetime(train["issue_d"], errors="coerce")

    df = sim.merge(train[["id", "issue_d", "grade"]], on="id", how="left", suffixes=("", "_train"))
    if "grade_train" in df.columns:
        df["grade"] = df["grade"].fillna(df["grade_train"])
        df = df.drop(columns=["grade_train"])
    df = df.dropna(subset=["issue_d"]).copy()
    df["month"] = df["issue_d"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("month").reset_index(drop=True)

    selected_rule = _load_selected_rule(Path(selected_rule_path))
    months = sorted(df["month"].dropna().unique().tolist())
    if len(months) <= min_history_months:
        raise ValueError(
            f"Not enough months for temporal backtest. months={len(months)}, "
            f"min_history_months={min_history_months}"
        )

    rows: list[dict[str, float | int | str]] = []
    grade_rows: list[dict[str, float | int | str]] = []

    for month in months[min_history_months:]:
        train_ref = df[df["month"] < month]
        test_slice = df[df["month"] == month].copy()
        if train_ref.empty or test_slice.empty:
            continue

        mask = _policy_mask(test_slice, selected_rule, train_ref)
        selected = test_slice.loc[mask].copy()
        n_obs = int(len(test_slice))
        n_selected = int(mask.sum())
        action_rate = float(n_selected / max(n_obs, 1))

        total_net = float(selected["net_value"].sum()) if n_selected else 0.0
        total_loss_reduction = (
            float(selected["expected_loss_reduction"].sum())
            if n_selected and "expected_loss_reduction" in selected.columns
            else 0.0
        )
        total_revenue_impact = (
            float(selected["revenue_impact"].sum())
            if n_selected and "revenue_impact" in selected.columns
            else 0.0
        )
        avg_net = float(total_net / max(n_selected, 1))
        positive_share = (
            float((selected["net_value"] > 0).mean()) if n_selected else 0.0
        )

        rows.append(
            {
                "month": month,
                "rule_name": selected_rule,
                "n_obs": n_obs,
                "n_selected": n_selected,
                "action_rate": action_rate,
                "total_net_value": total_net,
                "avg_net_value_selected": avg_net,
                "total_loss_reduction": total_loss_reduction,
                "total_revenue_impact": total_revenue_impact,
                "positive_net_share_selected": positive_share,
            }
        )

        if n_selected and "grade" in selected.columns:
            by_grade = (
                selected.groupby("grade", observed=True)["net_value"]
                .agg(n_selected="size", total_net_value="sum", avg_net_value="mean")
                .reset_index()
            )
            by_grade["month"] = month
            by_grade["rule_name"] = selected_rule
            grade_rows.extend(by_grade.to_dict(orient="records"))

    backtest = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    if backtest.empty:
        raise ValueError("Temporal backtest produced no monthly slices.")

    backtest["total_net_roll3"] = backtest["total_net_value"].rolling(3, min_periods=1).sum()
    backtest["action_rate_roll3"] = backtest["action_rate"].rolling(3, min_periods=1).mean()
    by_grade = pd.DataFrame(grade_rows)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    backtest_path = out_dir / "causal_policy_oot_backtest.parquet"
    grade_path = out_dir / "causal_policy_oot_backtest_by_grade.parquet"
    backtest.to_parquet(backtest_path, index=False)
    by_grade.to_parquet(grade_path, index=False)

    status = {
        "rule_name": selected_rule,
        "min_history_months": int(min_history_months),
        "n_months_evaluated": int(len(backtest)),
        "avg_action_rate": float(backtest["action_rate"].mean()),
        "total_net_value": float(backtest["total_net_value"].sum()),
        "p05_monthly_net": float(backtest["total_net_value"].quantile(0.05)),
        "worst_month": str(backtest.loc[backtest["total_net_value"].idxmin(), "month"]),
        "best_month": str(backtest.loc[backtest["total_net_value"].idxmax(), "month"]),
    }
    status_path = Path("models/causal_policy_oot_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    logger.info(f"Saved causal OOT backtest: {backtest_path} ({len(backtest):,} rows)")
    logger.info(f"Saved causal OOT backtest by grade: {grade_path} ({len(by_grade):,} rows)")
    logger.info(f"Saved causal OOT status: {status_path}")
    logger.info(f"Causal OOT summary: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_history_months", type=int, default=12)
    parser.add_argument("--selected_rule_path", default="models/causal_policy_rule.json")
    args = parser.parse_args()
    main(min_history_months=args.min_history_months, selected_rule_path=args.selected_rule_path)
