"""Validate and select a discrete causal pricing policy rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.causal import load_causal_config
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.pipeline_runtime import (
    atomic_write_json,
    atomic_write_parquet,
    write_last_valid_artifact,
    write_runtime_checkpoint,
    write_runtime_status,
)

_CLI_RUN_TAG: str | None = None

CAUSAL_POLICY_SCHEMA_VERSION = "2026-03-26.1"


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
    max_action_rate: float | None = None,
    min_bootstrap_p05_net: float | None = None,
    min_grade_total_net: float | None = None,
    bootstrap_samples: int | None = None,
    random_state: int = 42,
    config_path: str = "configs/causal_lane.yaml",
):
    stage_name = "causal_policy_validation"
    cfg = load_causal_config(config_path)
    policy_cfg = cfg.get("policy", {})
    input_path = Path("data/processed/causal_policy_simulation.parquet")
    effect_status_path = Path("models/causal_effect_status.json")
    if not input_path.exists():
        raise FileNotFoundError(
            "Missing causal policy simulation artifact. Run scripts/simulate_causal_policy.py first."
        )
    df = pd.read_parquet(input_path)
    if "recommended_action" not in df.columns or "net_value" not in df.columns:
        raise KeyError("Required columns missing in causal_policy_simulation artifact.")
    effect_status = (
        json.loads(effect_status_path.read_text(encoding="utf-8"))
        if effect_status_path.exists()
        else {}
    )
    run_tag = resolve_run_tag(_CLI_RUN_TAG or effect_status.get("run_tag"), require_explicit=True)
    write_runtime_status(
        stage_name,
        phase="loading_inputs",
        state="running",
        run_tag=run_tag,
    )

    max_action_rate = float(
        policy_cfg.get("max_action_rate", 0.35) if max_action_rate is None else max_action_rate
    )
    min_bootstrap_p05_net = float(
        policy_cfg.get("min_bootstrap_p05_net", 0.0)
        if min_bootstrap_p05_net is None
        else min_bootstrap_p05_net
    )
    min_grade_total_net = float(
        policy_cfg.get("min_grade_total_net", 0.0)
        if min_grade_total_net is None
        else min_grade_total_net
    )
    bootstrap_samples = int(
        policy_cfg.get("bootstrap_samples", 200) if bootstrap_samples is None else bootstrap_samples
    )

    write_runtime_checkpoint(
        stage_name,
        "inputs_loaded",
        {
            "rows": int(len(df)),
            "bootstrap_samples": int(bootstrap_samples),
        },
    )

    action_mask = df["recommended_action"].ne("hold_rate")
    q80 = (
        float(df.loc[action_mask, "policy_value_score"].quantile(0.80))
        if action_mask.any()
        else 0.0
    )
    q90 = (
        float(df.loc[action_mask, "policy_value_score"].quantile(0.90))
        if action_mask.any()
        else 0.0
    )
    rules = {
        "discount_100_only": df["recommended_action"].eq("decrease_100bps").to_numpy(),
        "discount_50_or_100": df["recommended_action"]
        .isin(["decrease_50bps", "decrease_100bps"])
        .to_numpy(),
        "positive_value_only": (action_mask & (df["net_value"] > 0)).to_numpy(),
        "top20_policy_value": (
            action_mask
            & (pd.to_numeric(df["policy_value_score"], errors="coerce").fillna(0.0) >= q80)
        ).to_numpy(),
        "top10_policy_value": (
            action_mask
            & (pd.to_numeric(df["policy_value_score"], errors="coerce").fillna(0.0) >= q90)
        ).to_numpy(),
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
    atomic_write_parquet(candidates, candidates_path, index=False)
    atomic_write_parquet(selected, selected_path, index=False)

    selected_row = selected.iloc[0].to_dict()
    policy_gate_pass = bool(
        selected_row.get("pass_all", False)
        and effect_status.get("overlap_pass", False)
        and effect_status.get("sensitivity_pass", False)
    )
    promotion_state = (
        "operational_candidate"
        if policy_gate_pass
        else "validated_research_policy"
        if bool(selected_row.get("pass_all", False))
        else "insights_only"
    )
    status = {
        "selection_reason": selection_reason,
        "selected_rule": selected_row["rule_name"],
        "selected_metrics": {
            k: bool(v)
            if isinstance(v, (bool, np.bool_))
            else float(v)
            if isinstance(v, (float, int, np.floating, np.integer))
            else v
            for k, v in selected_row.items()
        },
        "constraints": {
            "max_action_rate": max_action_rate,
            "min_bootstrap_p05_net": min_bootstrap_p05_net,
            "min_grade_total_net": min_grade_total_net,
        },
        "source_simulation_path": str(input_path),
        "source_effect_status_path": str(effect_status_path),
        "effect_status_run_tag": effect_status.get("run_tag"),
        "policy_semantics": cfg.get("defaults", {}).get(
            "policy_semantics", "research_grade_pricing_intervention"
        ),
        "policy_value_method": cfg.get("defaults", {}).get(
            "policy_value_method", "local_cate_discrete_grid"
        ),
        # Research-only policies are intentionally classified as insights_only so
        # downstream paper-grade closure can accept them without implying promotion.
        "role": "insights_only" if not policy_gate_pass else "operational_candidate",
        "promotion_eligible": policy_gate_pass,
        "promotion_state": promotion_state,
        "promotion_decider": "validate_causal_policy.py",
        "policy_evaluation_consistent": policy_gate_pass,
        "overlap_pass": bool(effect_status.get("overlap_pass", False)),
        "sensitivity_pass": bool(effect_status.get("sensitivity_pass", False)),
        **build_artifact_metadata(
            schema_version=CAUSAL_POLICY_SCHEMA_VERSION,
            run_tag=run_tag,
            require_explicit=True,
        ),
    }
    status_path = model_dir / "causal_policy_rule.json"
    atomic_write_json(status_path, status)
    write_last_valid_artifact(
        stage_name,
        artifact_key="causal_policy_rule",
        artifact_path=status_path,
        run_tag=run_tag,
        extra={"selected_rule": str(selected_row["rule_name"])},
    )
    write_runtime_status(
        stage_name,
        phase="completed",
        state="completed",
        run_tag=run_tag,
        extra={"status_path": str(status_path), "selected_rule": str(selected_row["rule_name"])},
    )

    logger.info("Saved policy rule candidates: {}", candidates_path)
    logger.info("Saved selected policy rule: {}", selected_path)
    logger.info("Saved policy rule status: {}", status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_action_rate", type=float, default=None)
    parser.add_argument("--min_bootstrap_p05_net", type=float, default=None)
    parser.add_argument("--min_grade_total_net", type=float, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--config", default="configs/causal_lane.yaml")
    parser.add_argument("--run-tag", default=None, help="Override run_tag on output artifacts")
    args = parser.parse_args()
    if args.run_tag:
        _CLI_RUN_TAG = args.run_tag
    main(
        max_action_rate=args.max_action_rate,
        min_bootstrap_p05_net=args.min_bootstrap_p05_net,
        min_grade_total_net=args.min_grade_total_net,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
        config_path=args.config,
    )
