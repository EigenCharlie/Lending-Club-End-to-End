"""Return-aware exact rerank over an existing bound-aware portfolio frontier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.search.run_portfolio_bound_aware_search import (  # noqa: E402
    SCHEMA_VERSION,
    STAGE_NAME,
    _aggregate_exact_results,
    _build_stratified_shortlist,
    _policy_from_row,
    _selection_reason,
)
from scripts.validate_alpha_gamma_bound import (  # noqa: E402
    _load_aligned_dataset,
    _validate_single_alpha,
)
from src.utils.pipeline_runtime import atomic_write_json, atomic_write_parquet  # noqa: E402

DEFAULT_INCUMBENT_SELECTION = (
    ROOT
    / "models"
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
    / "portfolio_bound_aware_selection.json"
)


def _float_grid(raw: str | None, fallback: list[float]) -> list[float]:
    if not raw:
        return fallback
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _int_grid(raw: str | None, fallback: list[int]) -> list[int]:
    if not raw:
        return fallback
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _eta(elapsed: float, completed: int, total: int) -> float | None:
    if completed <= 0 or total <= 0:
        return None
    if completed >= total:
        return 0.0
    return elapsed / completed * (total - completed)


def _write_status(
    *,
    path: Path,
    run_label: str,
    phase: str,
    state: str,
    started: float,
    completed: int,
    total: int,
    extra: dict[str, Any] | None = None,
) -> None:
    elapsed = time.monotonic() - started
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "run_tag": run_label,
        "phase": phase,
        "state": state,
        "updated_at_utc": datetime.now(tz=UTC).isoformat(),
        "bound_completed_checks": int(completed),
        "bound_total_checks": int(total),
        "bound_pct_complete": float(completed / max(total, 1)),
        "elapsed_sec": float(elapsed),
        "eta_sec": _eta(elapsed, completed, total),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)


def _load_incumbent_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected_policy" in payload:
        return payload["selected_policy"]
    if "selected_metrics" in payload:
        selected = payload["selected_metrics"]
        return {
            "risk_tolerance": float(selected["risk_tolerance"]),
            "policy_mode": str(selected["policy_mode"]),
            "gamma": float(selected["gamma"]),
            "delta_cap_quantile": float(selected["delta_cap_quantile"]),
            "tail_focus_quantile": float(selected["tail_focus_quantile"]),
            "uncertainty_aversion": float(selected["uncertainty_aversion"]),
            "min_budget_utilization": float(selected["min_budget_utilization"]),
            "pd_cap_slack_penalty": float(selected["pd_cap_slack_penalty"]),
            "solver_backend": str(selected["solver_backend"]),
        }
    raise ValueError(f"Cannot infer incumbent policy from {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-label", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--alpha-grid", default="")
    parser.add_argument("--random-states", default="")
    parser.add_argument("--shortlist-top-k", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--incumbent-selection-path", default=str(DEFAULT_INCUMBENT_SELECTION))
    args = parser.parse_args(argv)

    source_run = str(args.source_run_label)
    run_label = str(args.run_label)
    source_data_dir = ROOT / "data" / "processed" / "portfolio_bound_aware" / source_run
    source_model_dir = ROOT / "models" / "portfolio_bound_aware" / source_run
    output_dir = ROOT / "data" / "processed" / "portfolio_bound_aware" / run_label
    model_dir = ROOT / "models" / "portfolio_bound_aware" / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    context_path = source_model_dir / "portfolio_bound_aware_exact_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    search_space = context.get("search_space", {})
    frontier_path = source_data_dir / "portfolio_bound_aware_frontier.parquet"
    frontier_raw_path = source_data_dir / "portfolio_bound_aware_frontier_raw.parquet"
    frontier = pd.read_parquet(frontier_path)
    frontier_raw = pd.read_parquet(frontier_raw_path) if frontier_raw_path.exists() else None

    alpha_grid = _float_grid(args.alpha_grid, [float(v) for v in context["alpha_grid"]])
    random_states = _int_grid(args.random_states, [int(v) for v in context["random_states"]])
    shortlist_top_k = (
        int(args.shortlist_top_k)
        if int(args.shortlist_top_k) > 0
        else int(search_space.get("shortlist_top_k", 240))
    )
    incumbent_policy = _load_incumbent_policy(Path(args.incumbent_selection_path))
    budget_profiles = search_space.get(
        "budget_profiles",
        [{"name": "free_budget", "min_budget_utilization": 0.0, "pd_cap_slack_penalty": 0.0}],
    )

    shortlist = _build_stratified_shortlist(
        frontier=frontier,
        shortlist_top_k=shortlist_top_k,
        bucket_return_k=int(search_space.get("bucket_return_k", 100)),
        bucket_proxy_k=int(search_space.get("bucket_proxy_k", 100)),
        bucket_family_k=int(search_space.get("bucket_family_k", 60)),
        bucket_region_k=int(search_space.get("bucket_region_k", 80)),
        incumbent_policy=incumbent_policy,
        incumbent_risk_neighbors=[float(v) for v in search_space["incumbent_risk_neighbors"]],
        incumbent_gamma_neighbors=[float(v) for v in search_space["incumbent_gamma_neighbors"]],
        incumbent_policy_modes=[str(v) for v in search_space["incumbent_policy_modes"]],
        budget_profiles=budget_profiles,
        solver_backend=str(context["frontier_solver_backend"]),
    )

    atomic_write_parquet(
        frontier, output_dir / "portfolio_bound_aware_frontier.parquet", index=False
    )
    if frontier_raw is not None:
        atomic_write_parquet(
            frontier_raw, output_dir / "portfolio_bound_aware_frontier_raw.parquet", index=False
        )
    atomic_write_parquet(
        shortlist, output_dir / "portfolio_bound_aware_shortlist.parquet", index=False
    )

    status_path = model_dir / "portfolio_bound_aware_runtime_status.json"
    total = len(shortlist) * len(alpha_grid) * len(random_states)
    started = time.monotonic()
    _write_status(
        path=status_path,
        run_label=run_label,
        phase="exact_bound_running",
        state="running",
        started=started,
        completed=0,
        total=total,
        extra={
            "source_run_label": source_run,
            "shortlist_size": int(len(shortlist)),
            "shortlist_buckets": shortlist["shortlist_bucket"].value_counts().to_dict(),
            "alpha_grid": alpha_grid,
            "random_states": random_states,
        },
    )

    aligned_by_seed = {
        int(seed): _load_aligned_dataset(
            conformal_intervals_path=str(context["conformal_intervals_path"]),
            max_candidates=int(context["max_candidates"]),
            random_state=int(seed),
        )
        for seed in random_states
    }

    rows: list[dict[str, Any]] = []
    completed = 0
    partial_path = output_dir / "portfolio_bound_aware_bound_eval_partial.parquet"
    for _, row in shortlist.iterrows():
        policy = _policy_from_row(
            row,
            solver_backend_override=str(context["exact_solver_backend"]),
        )
        candidate_payload = row.to_dict()
        for eval_seed in random_states:
            aligned = aligned_by_seed[int(eval_seed)]
            for alpha in alpha_grid:
                result = _validate_single_alpha(
                    aligned,
                    alpha=float(alpha),
                    policy=policy,
                    allocator_mode="exact",
                    budget=float(context["budget"]),
                    t_eval=float(context["t_eval"]),
                )
                rows.append(
                    {
                        "candidate_rank": int(candidate_payload["candidate_rank"]),
                        "eval_random_state": int(eval_seed),
                        "frontier_solver_backend": str(context["frontier_solver_backend"]),
                        "exact_solver_backend": str(context["exact_solver_backend"]),
                        **candidate_payload,
                        **result,
                    }
                )
                completed += 1
                if completed % max(1, int(args.checkpoint_every)) == 0 or completed == total:
                    atomic_write_parquet(pd.DataFrame(rows), partial_path, index=False)
                    _write_status(
                        path=status_path,
                        run_label=run_label,
                        phase="exact_bound_running",
                        state="running",
                        started=started,
                        completed=completed,
                        total=total,
                        extra={
                            "candidate_rank": int(candidate_payload["candidate_rank"]),
                            "eval_random_state": int(eval_seed),
                            "current_alpha": float(alpha),
                        },
                    )

    bound_eval = pd.DataFrame(rows)
    shortlist_eval = _aggregate_exact_results(shortlist=shortlist, bound_eval=bound_eval)
    selected = shortlist_eval.iloc[0].copy()
    selected_policy = _policy_from_row(
        selected,
        solver_backend_override=str(context["exact_solver_backend"]),
    )

    atomic_write_parquet(
        shortlist_eval, output_dir / "portfolio_bound_aware_shortlist.parquet", index=False
    )
    atomic_write_parquet(
        bound_eval, output_dir / "portfolio_bound_aware_bound_eval.parquet", index=False
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_label": run_label,
        "source_run_label": source_run,
        "rerank_strategy": "return_global_then_conservative_proxy_then_incumbent",
        "conformal_intervals_path": str(context["conformal_intervals_path"]),
        "search_space": search_space,
        "selection_policy": {
            "shortlist_strategy": "return_aware_exact_rerank",
            "rank_order": [
                "alpha01_exact_pass(desc)",
                "alpha03_exact_pass(desc)",
                "ab_pass_all(desc)",
                "realized_total_return(desc)",
                "price_of_robustness(desc)",
                "alpha01_weighted_miscoverage_V(asc)",
                "alpha01_gamma_cp(asc)",
            ],
        },
        "selected_policy": selected_policy,
        "selected_metrics": selected.to_dict(),
        "selection_reason": _selection_reason(selected),
        "frontier_path": str(output_dir / "portfolio_bound_aware_frontier.parquet"),
        "shortlist_path": str(output_dir / "portfolio_bound_aware_shortlist.parquet"),
        "bound_eval_path": str(output_dir / "portfolio_bound_aware_bound_eval.parquet"),
        "runtime_status_path": str(status_path),
        "frontier_solver_backend": str(context["frontier_solver_backend"]),
        "exact_solver_backend": str(context["exact_solver_backend"]),
    }
    atomic_write_json(model_dir / "portfolio_bound_aware_selection.json", payload)
    _write_status(
        path=status_path,
        run_label=run_label,
        phase="selection_complete",
        state="completed",
        started=started,
        completed=total,
        total=total,
        extra={
            "source_run_label": source_run,
            "selection_reason": str(payload["selection_reason"]),
            "selected_alpha01_exact_pass": bool(selected["alpha01_exact_pass"]),
            "selected_realized_total_return": float(selected["realized_total_return"]),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
