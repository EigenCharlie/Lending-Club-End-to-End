"""Exact bound reranking helper for bound-aware portfolio search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.search.run_portfolio_bound_aware_search import (  # noqa: E402
    SCHEMA_VERSION,
    STAGE_NAME,
    _aggregate_exact_results,
    _policy_from_row,
    _resource_snapshot,
    _selection_reason,
)
from scripts.validate_alpha_gamma_bound import (  # noqa: E402
    _load_aligned_dataset,
    _validate_single_alpha,
)
from src.utils.pipeline_runtime import (  # noqa: E402
    atomic_write_json,
    atomic_write_parquet,
    load_runtime_status,
    write_runtime_checkpoint,
    write_runtime_status,
)


def _eta_seconds(elapsed_sec: float, completed: int, total: int) -> float | None:
    if completed <= 0 or total <= 0 or completed >= total:
        return 0.0 if total > 0 and completed >= total else None
    return (elapsed_sec / max(completed, 1)) * max(total - completed, 0)


def _write_exact_status(
    *,
    context: dict[str, object],
    base_elapsed_sec: float,
    bound_completed_checks: int,
    state: str,
    phase: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    started = float(context["helper_started_monotonic"])
    frontier_total = int(context["frontier_total_units"])
    frontier_completed = int(context["frontier_completed_units"])
    bound_total = int(context["bound_total_checks"])
    elapsed_sec = base_elapsed_sec + (time.monotonic() - started)
    global_total = frontier_total + bound_total
    global_completed = frontier_completed + bound_completed_checks
    payload: dict[str, object] = {
        "frontier_total_units": frontier_total,
        "frontier_completed_units": frontier_completed,
        "frontier_pct_complete": float(frontier_completed / max(frontier_total, 1))
        if frontier_total > 0
        else 1.0,
        "bound_total_checks": bound_total,
        "bound_completed_checks": int(bound_completed_checks),
        "bound_pct_complete": float(bound_completed_checks / max(bound_total, 1))
        if bound_total > 0
        else 0.0,
        "global_total_units": global_total,
        "global_completed_units": global_completed,
        "global_pct_complete": float(global_completed / max(global_total, 1))
        if global_total > 0
        else 1.0,
        "elapsed_sec": float(elapsed_sec),
        "eta_sec": _eta_seconds(elapsed_sec, global_completed, global_total),
    }
    if extra:
        payload.update(extra)
    write_runtime_status(
        STAGE_NAME,
        phase=phase,
        state=state,
        run_tag=str(context["run_label"]),
        status_path=str(context["runtime_status_path"]),
        extra=payload,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-path", required=True)
    args = parser.parse_args(argv)

    context_path = Path(args.context_path).resolve()
    context = json.loads(context_path.read_text(encoding="utf-8"))
    status_path = Path(str(context["runtime_status_path"]))
    checkpoint_dir = Path(str(context["runtime_checkpoint_dir"]))
    resource_snapshot_path = Path(str(context["resource_snapshot_path"]))
    shortlist_path = Path(str(context["shortlist_path"]))
    bound_eval_path = Path(str(context["bound_eval_path"]))
    selection_path = Path(str(context["selection_path"]))

    prior_status = load_runtime_status(status_path)
    base_elapsed_sec = float(prior_status.get("elapsed_sec", 0.0))
    context["frontier_total_units"] = int(prior_status.get("frontier_total_units", 0))
    context["frontier_completed_units"] = int(prior_status.get("frontier_completed_units", 0))
    context["bound_total_checks"] = int(prior_status.get("bound_total_checks", 0))
    context["helper_started_monotonic"] = time.monotonic()

    shortlist = pd.read_parquet(shortlist_path)
    random_states = [int(v) for v in context["random_states"]]
    alpha_grid = [float(v) for v in context["alpha_grid"]]

    aligned_by_seed = {
        int(seed): _load_aligned_dataset(
            conformal_intervals_path=str(context["conformal_intervals_path"]),
            max_candidates=int(context["max_candidates"]),
            random_state=int(seed),
        )
        for seed in random_states
    }

    bound_rows: list[dict[str, object]] = []
    completed_checks = 0
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
                bound_rows.append(
                    {
                        "candidate_rank": int(candidate_payload["candidate_rank"]),
                        "eval_random_state": int(eval_seed),
                        "frontier_solver_backend": str(context["frontier_solver_backend"]),
                        "exact_solver_backend": str(context["exact_solver_backend"]),
                        **candidate_payload,
                        **result,
                    }
                )
                completed_checks += 1
                _write_exact_status(
                    context=context,
                    base_elapsed_sec=base_elapsed_sec,
                    bound_completed_checks=completed_checks,
                    phase="exact_bound_running",
                    state="running",
                    extra={
                        "candidate_rank": int(candidate_payload["candidate_rank"]),
                        "eval_random_state": int(eval_seed),
                        "current_alpha": float(alpha),
                    },
                )

    bound_eval = pd.DataFrame(bound_rows)
    shortlist_eval = _aggregate_exact_results(shortlist=shortlist, bound_eval=bound_eval)
    selected = shortlist_eval.iloc[0].copy()
    selected_policy = _policy_from_row(
        selected,
        solver_backend_override=str(context["exact_solver_backend"]),
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_label": str(context["run_label"]),
        "conformal_intervals_path": str(context["conformal_intervals_path"]),
        "search_space": context["search_space"],
        "selection_policy": context["selection_policy"],
        "selected_policy": selected_policy,
        "selected_metrics": selected.to_dict(),
        "selection_reason": _selection_reason(selected),
        "frontier_raw_path": str(context["frontier_raw_path"]),
        "frontier_path": str(context["frontier_path"]),
        "shortlist_path": str(context["shortlist_path"]),
        "bound_eval_path": str(context["bound_eval_path"]),
        "runtime_status_path": str(context["runtime_status_path"]),
        "runtime_checkpoint_dir": str(context["runtime_checkpoint_dir"]),
        "resource_snapshot_path": str(context["resource_snapshot_path"]),
        "frontier_solver_backend": str(context["frontier_solver_backend"]),
        "exact_solver_backend": str(context["exact_solver_backend"]),
    }

    atomic_write_parquet(shortlist_eval, shortlist_path, index=False)
    atomic_write_parquet(bound_eval, bound_eval_path, index=False)
    atomic_write_json(selection_path, payload)

    resource_payload = json.loads(resource_snapshot_path.read_text(encoding="utf-8"))
    resource_payload["exact_helper_python"] = sys.executable
    resource_payload["exact_helper_end"] = _resource_snapshot()
    atomic_write_json(resource_snapshot_path, resource_payload)

    final_payload = _write_exact_status(
        context=context,
        base_elapsed_sec=base_elapsed_sec,
        bound_completed_checks=int(context["bound_total_checks"]),
        phase="selection_complete",
        state="completed",
        extra={
            "selection_reason": str(payload["selection_reason"]),
            "selected_alpha01_exact_pass": bool(selected["alpha01_exact_pass"]),
            "selected_realized_total_return": float(selected["realized_total_return"]),
        },
    )
    write_runtime_checkpoint(
        STAGE_NAME,
        "003_selection_complete",
        final_payload,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info(
        "External exact bound stage complete: selected risk_tolerance={}, mode={}, gamma={}, alpha01_pass={}",
        selected["risk_tolerance"],
        selected["policy_mode"],
        selected["gamma"],
        selected["alpha01_exact_pass"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
