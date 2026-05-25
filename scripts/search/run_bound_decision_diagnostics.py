"""Post-selection diagnostics for bound-aware portfolio candidates.

The script does not promote a new theorem. It turns a selected portfolio policy
into reviewer-facing diagnostics: cluster concentration, temporal holdout slices,
and direct decision-loss summaries that can later motivate CRC/LTT follow-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_alpha_gamma_bound import (  # noqa: E402
    _compute_effective_pd_vector,
    _compute_exact_weights,
    _compute_intervals_at_alpha,
    _load_aligned_dataset,
)
from src.utils.pipeline_runtime import atomic_write_json, atomic_write_parquet  # noqa: E402


def _load_selected_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_policy", payload)
    return {
        "risk_tolerance": float(selected.get("risk_tolerance", 0.10)),
        "uncertainty_aversion": float(selected.get("uncertainty_aversion", 0.0)),
        "min_budget_utilization": float(selected.get("min_budget_utilization", 0.0)),
        "pd_cap_slack_penalty": float(selected.get("pd_cap_slack_penalty", 0.0)),
        "policy_mode": str(selected.get("policy_mode", "blended_uncertainty")),
        "gamma": float(selected.get("gamma", 0.5)),
        "delta_cap_quantile": float(selected.get("delta_cap_quantile", 1.0)),
        "tail_focus_quantile": float(selected.get("tail_focus_quantile", 1.0)),
        "solver_backend": str(selected.get("solver_backend", "highs")),
    }


def _period_series(frame: pd.DataFrame) -> pd.Series:
    if "issue_d" in frame.columns:
        parsed = pd.to_datetime(frame["issue_d"], errors="coerce")
        if parsed.notna().any():
            return parsed.dt.to_period("Q").astype(str).fillna("unknown")
    if "temporal_segment" in frame.columns:
        return frame["temporal_segment"].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(frame), index=frame.index)


def _safe_group_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(frame), index=frame.index)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _group_diagnostics(
    frame: pd.DataFrame,
    *,
    weights: np.ndarray,
    funded_mask: np.ndarray,
    y_true: np.ndarray,
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    effective_pd: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    work = frame.copy()
    work["period"] = _period_series(frame)
    work["portfolio_weight"] = weights
    work["funded"] = funded_mask
    work["y_true_numeric"] = y_true
    work["pd_point"] = pd_point
    work["pd_high"] = pd_high
    work["effective_pd"] = effective_pd
    work["miscovered"] = (y_true > pd_high).astype(float)
    work["uncertainty_width"] = np.clip(pd_high - pd_point, 0.0, 1.0)
    group_specs = [
        ("period", "period"),
        ("interval_grade_or_score_group", "grade"),
        ("state", "addr_state"),
        ("purpose", "purpose"),
        ("term", "term"),
        ("verification_status", "verification_status"),
    ]
    rows: list[dict[str, Any]] = []
    for group_family, column in group_specs:
        labels = _safe_group_series(work, column)
        for label, group in work.groupby(labels, dropna=False):
            idx = group.index.to_numpy()
            group_weights = weights[idx]
            funded_group = funded_mask[idx]
            funded_weights = group_weights[funded_group]
            funded_idx = idx[funded_group]
            weight_sum = float(np.sum(group_weights))
            funded_weight_sum = float(np.sum(funded_weights))
            rows.append(
                {
                    "group_family": group_family,
                    "group_value": str(label),
                    "n_rows": int(len(group)),
                    "n_funded": int(np.sum(funded_group)),
                    "portfolio_weight_sum": round(weight_sum, 8),
                    "funded_weight_sum": round(funded_weight_sum, 8),
                    "weighted_miscoverage": round(
                        float(np.sum(group_weights * work.loc[idx, "miscovered"].to_numpy())),
                        8,
                    ),
                    "funded_empirical_coverage": round(
                        float(
                            1.0 - work.loc[funded_idx, "miscovered"].mean()
                            if len(funded_idx)
                            else np.nan
                        ),
                        6,
                    ),
                    "funded_weighted_pd_true": round(
                        _weighted_mean(y_true[funded_idx], funded_weights)
                        if len(funded_idx)
                        else float("nan"),
                        8,
                    ),
                    "funded_weighted_pd_high": round(
                        _weighted_mean(pd_high[funded_idx], funded_weights)
                        if len(funded_idx)
                        else float("nan"),
                        8,
                    ),
                    "funded_weighted_effective_pd": round(
                        _weighted_mean(effective_pd[funded_idx], funded_weights)
                        if len(funded_idx)
                        else float("nan"),
                        8,
                    ),
                    "funded_weighted_gamma_cp": round(
                        _weighted_mean(
                            np.clip(pd_high[funded_idx] - pd_point[funded_idx], 0.0, 1.0),
                            funded_weights,
                        )
                        if len(funded_idx)
                        else float("nan"),
                        8,
                    ),
                    "sqrt_alpha": round(float(np.sqrt(alpha)), 8),
                }
            )
    return pd.DataFrame(rows).sort_values(
        by=["group_family", "funded_weight_sum", "portfolio_weight_sum"],
        ascending=[True, False, False],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--conformal-intervals-path", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--budget", type=float, default=1_000_000.0)
    args = parser.parse_args(argv)

    run_label = str(args.run_label).strip().replace("/", "_")
    data_dir = ROOT / "data" / "processed" / "bound_diagnostics" / run_label
    model_dir = ROOT / "models" / "bound_diagnostics" / run_label
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    policy = _load_selected_policy(Path(args.selection_path))
    policy["solver_backend"] = "highs"
    aligned = _load_aligned_dataset(
        conformal_intervals_path=str(args.conformal_intervals_path),
        max_candidates=int(args.max_candidates),
        random_state=int(args.random_state),
    )
    y_true = (
        pd.to_numeric(aligned["y_true"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "y_true" in aligned.columns
        else pd.to_numeric(aligned["default_flag"], errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    pd_point, pd_low, pd_high = _compute_intervals_at_alpha(aligned, float(args.alpha))
    effective_pd = _compute_effective_pd_vector(aligned, pd_point, pd_high, policy)
    weights, alloc_meta = _compute_exact_weights(
        aligned,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        effective_pd=effective_pd,
        policy=policy,
        budget=float(args.budget),
    )
    funded_mask = weights > 1e-8
    miscoverage = (y_true > pd_high).astype(float)
    weighted_pd_true = float(np.sum(weights * y_true))
    violation = max(0.0, weighted_pd_true - float(policy["risk_tolerance"]))
    weighted_miscoverage = float(np.sum(weights * miscoverage))
    gamma_cp = float(np.sum(weights * np.clip(pd_high - pd_point, 0.0, 1.0)))
    cluster = _group_diagnostics(
        aligned,
        weights=weights,
        funded_mask=funded_mask,
        y_true=y_true,
        pd_point=pd_point,
        pd_high=pd_high,
        effective_pd=effective_pd,
        alpha=float(args.alpha),
    )
    direct_loss = {
        "schema_version": "2026-05-24.1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_label": run_label,
        "selection_path": str(args.selection_path),
        "conformal_intervals_path": str(args.conformal_intervals_path),
        "alpha": float(args.alpha),
        "policy": policy,
        "n_rows": int(len(aligned)),
        "n_funded": int(np.sum(funded_mask)),
        "total_allocated": round(float(alloc_meta.get("total_allocated", 0.0)), 2),
        "weighted_pd_true": round(weighted_pd_true, 8),
        "risk_tolerance": float(policy["risk_tolerance"]),
        "decision_loss_pd_excess": round(violation, 8),
        "decision_loss_indicator": bool(violation > 0.0),
        "weighted_miscoverage_V": round(weighted_miscoverage, 8),
        "gamma_cp": round(gamma_cp, 8),
        "empirical_coverage_funded": round(
            float(1.0 - miscoverage[funded_mask].mean()) if funded_mask.any() else float("nan"),
            6,
        ),
        "sqrt_alpha": round(float(np.sqrt(float(args.alpha))), 8),
        "all_bounds_hold": bool(
            violation <= float(args.alpha) + 1e-8
            and weighted_miscoverage <= float(np.sqrt(float(args.alpha))) + 1e-8
        ),
        "diagnostic_role": "post_selection_crc_ltt_and_dependency_screen_not_formal_theorem",
    }
    max_cluster_v = (
        cluster.groupby("group_family")["weighted_miscoverage"].max().reset_index()
        if not cluster.empty
        else pd.DataFrame(columns=["group_family", "weighted_miscoverage"])
    )
    direct_loss["max_cluster_weighted_miscoverage_by_family"] = max_cluster_v.to_dict(
        orient="records"
    )
    atomic_write_parquet(cluster, data_dir / "bound_cluster_diagnostics.parquet", index=False)
    atomic_write_json(model_dir / "bound_decision_diagnostics_status.json", direct_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
