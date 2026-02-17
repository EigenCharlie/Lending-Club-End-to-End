"""Validate and select a deployable causal policy rule.

Uses simulated row-level policy economics and selects a rule that:
- has positive downside (bootstrap p05 net value > 0)
- limits action rate
- avoids negative grade-level total net value

Usage:
    uv run python scripts/validate_causal_policy.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _bootstrap_total(
    values: np.ndarray, n_boot: int = 200, random_state: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_boot, n))
    totals = values[idx].sum(axis=1)
    return float(totals.mean()), float(np.quantile(totals, 0.05)), float(np.quantile(totals, 0.95))


def _evaluate_rule(
    df: pd.DataFrame, name: str, mask: np.ndarray, n_boot: int, random_state: int
) -> dict[str, float | int | str]:
    selected = df[mask].copy()
    values = selected["net_value"].to_numpy(dtype=float)
    boot_mean, boot_p05, boot_p95 = _bootstrap_total(
        values, n_boot=n_boot, random_state=random_state
    )

    if len(selected) > 0 and "grade" in selected.columns:
        grade_totals = selected.groupby("grade", observed=True)["net_value"].sum()
        min_grade_net = float(grade_totals.min())
        worst_grade = str(grade_totals.idxmin())
    else:
        min_grade_net = 0.0
        worst_grade = "NA"

    return {
        "rule_name": name,
        "n_selected": int(mask.sum()),
        "action_rate": float(mask.mean()),
        "total_net_value": float(values.sum()),
        "total_loss_reduction": float(
            selected["expected_loss_reduction"].sum() if len(selected) else 0.0
        ),
        "total_revenue_impact": float(selected["revenue_impact"].sum() if len(selected) else 0.0),
        "bootstrap_mean_net": boot_mean,
        "bootstrap_p05_net": boot_p05,
        "bootstrap_p95_net": boot_p95,
        "min_grade_total_net": min_grade_net,
        "worst_grade": worst_grade,
    }


def main(
    max_action_rate: float = 0.35,
    min_bootstrap_p05_net: float = 0.0,
    min_grade_total_net: float = 0.0,
    bootstrap_samples: int = 200,
    random_state: int = 42,
):
    input_path = Path("data/processed/causal_policy_simulation.parquet")
    if not input_path.exists():
        raise FileNotFoundError(
            "Missing causal policy simulation artifact. Run scripts/simulate_causal_policy.py first."
        )
    df = pd.read_parquet(input_path)
    if "segment" not in df.columns or "net_value" not in df.columns:
        raise KeyError("Required columns missing in causal_policy_simulation artifact.")

    q85 = float(df["cate"].quantile(0.85))
    q90 = float(df["cate"].quantile(0.90))
    rules = {
        "high_only": df["segment"].eq("high_sensitivity").to_numpy(),
        "high_plus_medium_all": df["segment"]
        .isin(["high_sensitivity", "medium_sensitivity"])
        .to_numpy(),
        "high_plus_medium_positive": (
            df["segment"].eq("high_sensitivity")
            | (df["segment"].eq("medium_sensitivity") & (df["net_value"] > 0))
        ).to_numpy(),
        "top15_cate": (df["cate"] >= q85).to_numpy(),
        "top10_cate": (df["cate"] >= q90).to_numpy(),
    }

    rows = []
    for rule_name, mask in rules.items():
        rows.append(
            _evaluate_rule(
                df=df,
                name=rule_name,
                mask=mask,
                n_boot=bootstrap_samples,
                random_state=random_state,
            )
        )
    candidates = (
        pd.DataFrame(rows).sort_values("bootstrap_p05_net", ascending=False).reset_index(drop=True)
    )
    candidates["pass_action_rate"] = candidates["action_rate"] <= max_action_rate
    candidates["pass_bootstrap"] = candidates["bootstrap_p05_net"] >= min_bootstrap_p05_net
    candidates["pass_grade_floor"] = candidates["min_grade_total_net"] >= min_grade_total_net
    candidates["pass_all"] = candidates[
        ["pass_action_rate", "pass_bootstrap", "pass_grade_floor"]
    ].all(axis=1)

    feasible = candidates[candidates["pass_all"]].copy()
    if feasible.empty:
        selected = candidates.iloc[[0]].copy()
        selection_reason = "fallback_best_bootstrap_p05"
    else:
        feasible = feasible.sort_values(
            by=["bootstrap_p05_net", "total_net_value", "action_rate"],
            ascending=[False, False, True],
        )
        selected = feasible.iloc[[0]].copy()
        selection_reason = "best_feasible"

    data_dir = Path("data/processed")
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = data_dir / "causal_policy_rule_candidates.parquet"
    selected_path = data_dir / "causal_policy_rule_selected.parquet"
    candidates.to_parquet(candidates_path, index=False)
    selected.to_parquet(selected_path, index=False)

    selected_row = selected.iloc[0].to_dict()
    status = {
        "selection_reason": selection_reason,
        "selected_rule": selected_row["rule_name"],
        "selected_metrics": {
            k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v
            for k, v in selected_row.items()
        },
        "constraints": {
            "max_action_rate": max_action_rate,
            "min_bootstrap_p05_net": min_bootstrap_p05_net,
            "min_grade_total_net": min_grade_total_net,
        },
    }
    status_path = model_dir / "causal_policy_rule.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    logger.info(f"Saved policy rule candidates: {candidates_path}")
    logger.info(f"Saved selected policy rule: {selected_path}")
    logger.info(f"Saved policy rule status: {status_path}")
    logger.info(
        "Selected rule: "
        f"{selected_row['rule_name']} | action_rate={selected_row['action_rate']:.4f}, "
        f"total_net={selected_row['total_net_value']:,.2f}, "
        f"bootstrap_p05={selected_row['bootstrap_p05_net']:,.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_action_rate", type=float, default=0.35)
    parser.add_argument("--min_bootstrap_p05_net", type=float, default=0.0)
    parser.add_argument("--min_grade_total_net", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(
        max_action_rate=args.max_action_rate,
        min_bootstrap_p05_net=args.min_bootstrap_p05_net,
        min_grade_total_net=args.min_grade_total_net,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
    )
