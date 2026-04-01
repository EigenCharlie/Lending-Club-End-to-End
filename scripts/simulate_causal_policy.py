"""Simulate a discrete pricing policy from local CATE estimates."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.causal import load_causal_config
from src.utils.artifact_metadata import resolve_run_tag
from src.utils.pipeline_runtime import (
    atomic_write_parquet,
    atomic_write_pickle,
    write_last_valid_artifact,
    write_runtime_checkpoint,
    write_runtime_status,
)

POLICY_SEMANTICS = "local_cate_discrete_policy"
POLICY_VALUE_METHOD = "local_cate_discrete_grid"
_CLI_RUN_TAG: str | None = None


def _load_causal_inputs() -> tuple[pd.DataFrame, dict, dict]:
    cate_path = Path("data/processed/cate_estimates.parquet")
    if not cate_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/cate_estimates.parquet. Run estimate_causal_effects first."
        )
    df = pd.read_parquet(cate_path)

    summary_path = Path("models/causal_summary.pkl")
    if summary_path.exists():
        with open(summary_path, "rb") as f:
            summary = pickle.load(f)
    else:
        summary = {"treatment": "int_rate"}

    effect_status_path = Path("models/causal_effect_status.json")
    if effect_status_path.exists():
        effect_status = json.loads(effect_status_path.read_text(encoding="utf-8"))
    else:
        effect_status = {}
    return df, summary, effect_status


def _coerce_numeric(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    if pd.api.types.is_numeric_dtype(df[col]):
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    else:
        arr = (
            df[col]
            .astype(str)
            .str.strip()
            .str.rstrip("%")
            .pipe(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
    return np.nan_to_num(arr, nan=default)


def _action_label(action_bps: int) -> str:
    if action_bps == 0:
        return "hold_rate"
    return f"decrease_{abs(int(action_bps))}bps"


def main(lgd: float = 0.45, config_path: str = "configs/causal_lane.yaml"):
    stage_name = "causal_policy_simulation"
    cfg = load_causal_config(config_path)
    df, summary, effect_status = _load_causal_inputs()
    write_runtime_status(
        stage_name,
        phase="loading_inputs",
        state="running",
        run_tag=effect_status.get("run_tag"),
    )
    if "cate" not in df.columns:
        raise KeyError("CATE column not present in cate_estimates artifact.")

    cate = pd.to_numeric(df["cate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    treatment = summary.get("treatment", "int_rate")
    run_tag = resolve_run_tag(
        _CLI_RUN_TAG or effect_status.get("run_tag"),
        fallback_candidates=[summary.get("run_tag")],
        require_explicit=True,
    )
    write_runtime_checkpoint(
        stage_name,
        "inputs_loaded",
        {
            "rows": int(len(df)),
            "treatment": str(treatment),
        },
    )
    base_rate = _coerce_numeric(
        df, treatment if treatment in df.columns else "int_rate", default=12.0
    )
    loan_amnt = _coerce_numeric(df, "loan_amnt", default=10_000.0)
    grade = (
        df["grade"].astype(str).fillna("UNKNOWN")
        if "grade" in df.columns
        else pd.Series(["UNKNOWN"] * len(df))
    )

    if "default_flag" in df.columns:
        y = pd.to_numeric(df["default_flag"], errors="coerce").fillna(0.0)
        pd_base = y.groupby(grade).transform("mean").to_numpy(dtype=float)
    else:
        pd_base = np.full(len(df), 0.15, dtype=float)

    action_grid_bps = [
        int(v) for v in cfg.get("defaults", {}).get("action_grid_bps", [0, -50, -100])
    ]
    if 0 not in action_grid_bps:
        action_grid_bps = [0, *action_grid_bps]
    action_grid_bps = sorted(set(action_grid_bps), reverse=True)

    candidate_net_values: list[np.ndarray] = []
    candidate_pd_counterfactuals: list[np.ndarray] = []
    candidate_expected_loss_reduction: list[np.ndarray] = []
    candidate_revenue_impact: list[np.ndarray] = []
    candidate_rate_pp: list[np.ndarray] = []

    for action_bps in action_grid_bps:
        delta_pp = float(action_bps) / 100.0
        candidate_rate = np.clip(base_rate + delta_pp, 5.0, None)
        pd_shift = cate * delta_pp
        pd_counterfactual = np.clip(pd_base + pd_shift, 0.0, 1.0)
        avoided_pd = np.clip(pd_base - pd_counterfactual, 0.0, 1.0)
        expected_loss_reduction = avoided_pd * loan_amnt * lgd
        revenue_impact = loan_amnt * (delta_pp / 100.0)
        net_value = expected_loss_reduction + revenue_impact

        candidate_net_values.append(net_value)
        candidate_pd_counterfactuals.append(pd_counterfactual)
        candidate_expected_loss_reduction.append(expected_loss_reduction)
        candidate_revenue_impact.append(revenue_impact)
        candidate_rate_pp.append(candidate_rate)

    stacked_net = np.vstack(candidate_net_values)
    best_idx = np.argmax(stacked_net, axis=0)

    recommended_action_bps = np.take(np.asarray(action_grid_bps, dtype=int), best_idx)
    recommended_action = np.array([_action_label(v) for v in recommended_action_bps], dtype=object)
    recommended_delta_rate_pp = recommended_action_bps.astype(float) / 100.0

    pd_counterfactual = np.take_along_axis(
        np.vstack(candidate_pd_counterfactuals), best_idx[None, :], axis=0
    ).reshape(-1)
    expected_loss_reduction = np.take_along_axis(
        np.vstack(candidate_expected_loss_reduction), best_idx[None, :], axis=0
    ).reshape(-1)
    revenue_impact = np.take_along_axis(
        np.vstack(candidate_revenue_impact), best_idx[None, :], axis=0
    ).reshape(-1)
    new_rate = np.take_along_axis(np.vstack(candidate_rate_pp), best_idx[None, :], axis=0).reshape(
        -1
    )
    net_value = np.take_along_axis(stacked_net, best_idx[None, :], axis=0).reshape(-1)
    avoided_pd = np.clip(pd_base - pd_counterfactual, 0.0, 1.0)

    segment = np.where(
        recommended_action_bps <= -100,
        "high_sensitivity",
        np.where(recommended_action_bps <= -50, "medium_sensitivity", "low_sensitivity"),
    )

    out = pd.DataFrame(
        {
            "cate": cate,
            "segment": segment,
            "grade": grade.to_numpy(dtype=str),
            "base_rate_pp": base_rate,
            "recommended_delta_rate_bps": recommended_action_bps,
            "recommended_delta_rate_pp": recommended_delta_rate_pp,
            "counterfactual_rate_pp": new_rate,
            "pd_base_proxy": pd_base,
            "pd_counterfactual": pd_counterfactual,
            "expected_pd_reduction": avoided_pd,
            "loan_amnt": loan_amnt,
            "expected_loss_reduction": expected_loss_reduction,
            "revenue_impact": revenue_impact,
            "net_value": net_value,
            "policy_value_score": net_value,
            "recommended_action": recommended_action,
        }
    )
    if "id" in df.columns:
        out["id"] = df["id"].to_numpy()

    summary_segment = (
        out.groupby("segment", observed=True)
        .agg(
            n=("net_value", "size"),
            avg_cate=("cate", "mean"),
            avg_delta_rate_bps=("recommended_delta_rate_bps", "mean"),
            avg_pd_reduction=("expected_pd_reduction", "mean"),
            total_loss_reduction=("expected_loss_reduction", "sum"),
            total_revenue_impact=("revenue_impact", "sum"),
            total_net_value=("net_value", "sum"),
            action_rate=("recommended_action", lambda s: float((s != "hold_rate").mean())),
        )
        .reset_index()
        .sort_values("segment")
    )

    summary_grade = (
        out.groupby("grade", observed=True)
        .agg(
            n=("net_value", "size"),
            avg_cate=("cate", "mean"),
            avg_pd_reduction=("expected_pd_reduction", "mean"),
            total_net_value=("net_value", "sum"),
            action_rate=("recommended_action", lambda s: float((s != "hold_rate").mean())),
        )
        .reset_index()
        .sort_values("grade")
    )

    overall = {
        "n_obs": int(len(out)),
        "treatment": str(treatment),
        "run_tag": run_tag,
        "cate_mean": float(np.mean(cate)),
        "discount_share": float(np.mean(recommended_action_bps != 0)),
        "total_loss_reduction": float(np.sum(expected_loss_reduction)),
        "total_revenue_impact": float(np.sum(revenue_impact)),
        "total_net_value": float(np.sum(net_value)),
        "avg_pd_reduction": float(np.mean(avoided_pd)),
        "policy_semantics": POLICY_SEMANTICS,
        "policy_value_method": POLICY_VALUE_METHOD,
        "role": "insights_only",
        "promotion_eligible": False,
        "promotion_state": "insights_only",
        "promotion_decider": "optimize_cate_portfolio.py",
        "source_effect_status_path": "models/causal_effect_status.json",
        "action_grid_bps": action_grid_bps,
    }

    data_dir = Path("data/processed")
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    details_path = data_dir / "causal_policy_simulation.parquet"
    seg_path = data_dir / "causal_policy_segment_summary.parquet"
    grade_path = data_dir / "causal_policy_grade_summary.parquet"
    atomic_write_parquet(out, details_path, index=False)
    atomic_write_parquet(summary_segment, seg_path, index=False)
    atomic_write_parquet(summary_grade, grade_path, index=False)

    summary_path = model_dir / "causal_policy_summary.pkl"
    atomic_write_pickle(
        summary_path,
        {
            "overall": overall,
            "segment_summary": summary_segment.to_dict(orient="records"),
            "grade_summary": summary_grade.to_dict(orient="records"),
            "metadata": {
                "run_tag": run_tag,
                "policy_semantics": POLICY_SEMANTICS,
                "policy_value_method": POLICY_VALUE_METHOD,
                "source_effect_status_path": "models/causal_effect_status.json",
            },
        },
    )
    write_last_valid_artifact(
        stage_name,
        artifact_key="causal_policy_summary",
        artifact_path=summary_path,
        run_tag=run_tag,
        extra={"rows": int(len(out))},
    )
    write_runtime_status(
        stage_name,
        phase="completed",
        state="completed",
        run_tag=run_tag,
        extra={"summary_path": str(summary_path), "rows": int(len(out))},
    )

    logger.info("Saved policy simulation details: {} ({} rows)", details_path, len(out))
    logger.info("Saved segment policy summary: {}", seg_path)
    logger.info("Saved grade policy summary: {}", grade_path)
    logger.info("Overall policy impact: {}", overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgd", type=float, default=0.45)
    parser.add_argument("--config", default="configs/causal_lane.yaml")
    parser.add_argument("--run-tag", default=None, help="Override run_tag on output artifacts")
    args = parser.parse_args()
    if args.run_tag:
        _CLI_RUN_TAG = args.run_tag
    main(lgd=args.lgd, config_path=args.config)
