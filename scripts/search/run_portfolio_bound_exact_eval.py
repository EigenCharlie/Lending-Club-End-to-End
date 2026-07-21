"""Exact bound reranking helper for bound-aware portfolio search.

Implements a governed two-pass exact tournament:
1) pass1 at alpha=0.01 for the full shortlist;
2) pass2 full alpha grid for pass1 survivors plus fixed bucket quotas.

Optional shadow checking compares top-K finalists under an alternate exact
solver backend (e.g., gurobi) and marks selection as review-needed when
agreement exceeds configured tolerances.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def _float_equal(a: float, b: float, *, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def _semantic_bucket(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token.startswith("return_global"):
        return "return_global"
    if token.startswith("bound_proxy"):
        return "bound_proxy"
    if token.startswith("family::"):
        return "family_guardrail"
    if token.startswith("conservative_region"):
        return "conservative_region"
    if token.startswith(("forced_", "forced::", "incumbent_", "conservative_")):
        return "forced_incumbent_region"
    return "residual"


def _build_pass2_candidate_ranks(
    *,
    shortlist: pd.DataFrame,
    pass1_eval: pd.DataFrame,
    bucket_min: int,
) -> tuple[list[int], dict[str, Any]]:
    if shortlist.empty:
        return [], {"pass1_survivor_count": 0, "bucket_selected": {}, "selected_total": 0}
    pass1_ok = (
        pass1_eval.groupby("candidate_rank", dropna=False)["all_bounds_hold"]
        .all()
        .astype(bool)
        .to_dict()
    )
    work = shortlist.copy()
    work["candidate_rank"] = work["candidate_rank"].astype(int)
    work["pass1_exact_ok"] = work["candidate_rank"].map(pass1_ok).fillna(False).astype(bool)
    work["bucket_semantic"] = work["shortlist_bucket"].map(_semantic_bucket)
    work = work.sort_values(
        by=[
            "pass1_exact_ok",
            "realized_total_return",
            "bound_proxy_rank",
            "return_first_rank",
            "candidate_rank",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected: list[int] = sorted(
        {int(v) for v in work.loc[work["pass1_exact_ok"], "candidate_rank"].tolist()}
    )
    bucket_selected: dict[str, int] = {}
    if int(bucket_min) > 0:
        for bucket in sorted(work["bucket_semantic"].dropna().astype(str).unique()):
            bucket_rows = work[work["bucket_semantic"] == bucket]
            added = 0
            for _, row in bucket_rows.iterrows():
                rank = int(row["candidate_rank"])
                if rank in selected:
                    continue
                selected.append(rank)
                added += 1
                if added >= int(bucket_min):
                    break
            bucket_selected[bucket] = added
    selected = sorted({int(v) for v in selected})
    summary = {
        "pass1_survivor_count": int(sum(bool(v) for v in pass1_ok.values())),
        "bucket_selected": bucket_selected,
        "selected_total": int(len(selected)),
    }
    return selected, summary


def _task_row(
    *,
    candidate_payload: dict[str, Any],
    policy: dict[str, Any],
    eval_seed: int,
    alpha: float,
    aligned_by_seed: dict[int, pd.DataFrame],
    budget: float,
    t_eval: float,
    frontier_solver_backend: str,
    exact_solver_backend: str,
) -> dict[str, Any]:
    aligned = aligned_by_seed[int(eval_seed)]
    result = _validate_single_alpha(
        aligned,
        alpha=float(alpha),
        policy=policy,
        allocator_mode="exact",
        budget=float(budget),
        t_eval=float(t_eval),
    )
    return {
        "candidate_rank": int(candidate_payload["candidate_rank"]),
        "eval_random_state": int(eval_seed),
        "frontier_solver_backend": str(frontier_solver_backend),
        "exact_solver_backend": str(exact_solver_backend),
        **candidate_payload,
        **result,
    }


def _iter_task_results(
    *,
    tasks: list[dict[str, Any]],
    aligned_by_seed: dict[int, pd.DataFrame],
    budget: float,
    t_eval: float,
    frontier_solver_backend: str,
    exact_solver_backend: str,
    workers: int,
) -> Iterator[dict[str, Any]]:
    if not tasks:
        return
    if int(workers) <= 1:
        for task in tasks:
            yield _task_row(
                candidate_payload=task["candidate_payload"],
                policy=task["policy"],
                eval_seed=int(task["eval_seed"]),
                alpha=float(task["alpha"]),
                aligned_by_seed=aligned_by_seed,
                budget=float(budget),
                t_eval=float(t_eval),
                frontier_solver_backend=frontier_solver_backend,
                exact_solver_backend=exact_solver_backend,
            )
        return

    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="exact-bound") as executor:
        future_map: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        for task in tasks:
            fut = executor.submit(
                _task_row,
                candidate_payload=task["candidate_payload"],
                policy=task["policy"],
                eval_seed=int(task["eval_seed"]),
                alpha=float(task["alpha"]),
                aligned_by_seed=aligned_by_seed,
                budget=float(budget),
                t_eval=float(t_eval),
                frontier_solver_backend=frontier_solver_backend,
                exact_solver_backend=exact_solver_backend,
            )
            future_map[fut] = task
        for fut in as_completed(future_map):
            yield fut.result()


def _build_tasks(
    *,
    shortlist: pd.DataFrame,
    candidate_ranks: list[int],
    random_states: list[int],
    alphas: list[float],
    solver_backend: str,
) -> list[dict[str, Any]]:
    if not candidate_ranks or not alphas:
        return []
    frame = shortlist[shortlist["candidate_rank"].isin(candidate_ranks)].copy()
    tasks: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        policy = _policy_from_row(row, solver_backend_override=solver_backend)
        candidate_payload = row.to_dict()
        for eval_seed in random_states:
            for alpha in alphas:
                tasks.append(
                    {
                        "candidate_payload": candidate_payload,
                        "policy": policy,
                        "eval_seed": int(eval_seed),
                        "alpha": float(alpha),
                    }
                )
    return tasks


def _append_rows(
    *,
    rows_store: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    checkpoint_every: int,
    checkpoint_name: str,
    context: dict[str, object],
    base_elapsed_sec: float,
    completed_checks: int,
    bound_eval_partial_path: Path,
    checkpoint_dir: Path,
    phase: str,
    state: str,
    latest: dict[str, Any] | None,
    force_checkpoint: bool = False,
) -> int:
    if not new_rows:
        return completed_checks
    rows_store.extend(new_rows)
    completed_checks += len(new_rows)
    if force_checkpoint or completed_checks % max(1, int(checkpoint_every)) == 0:
        partial = pd.DataFrame(rows_store)
        if not partial.empty:
            atomic_write_parquet(partial, bound_eval_partial_path, index=False)
        payload = _write_exact_status(
            context=context,
            base_elapsed_sec=base_elapsed_sec,
            bound_completed_checks=completed_checks,
            phase=phase,
            state=state,
            extra=latest or {},
        )
        write_runtime_checkpoint(
            STAGE_NAME,
            checkpoint_name,
            payload,
            checkpoint_dir=checkpoint_dir,
        )
    return completed_checks


def _solver_agreement_report(
    *,
    primary_eval: pd.DataFrame,
    shadow_eval: pd.DataFrame,
    return_tol: float,
    v_tol: float,
    gamma_tol: float,
) -> dict[str, Any]:
    if primary_eval.empty or shadow_eval.empty:
        return {
            "status": "not_run",
            "max_abs_delta_return": None,
            "max_abs_delta_v": None,
            "max_abs_delta_gamma": None,
            "rows": [],
        }
    merge = primary_eval.merge(
        shadow_eval,
        on=["candidate_rank", "eval_random_state", "alpha"],
        suffixes=("_primary", "_shadow"),
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    max_return = 0.0
    max_v = 0.0
    max_gamma = 0.0
    for _, row in merge.iterrows():
        d_return = abs(
            float(row.get("weighted_pd_true_primary", 0.0))
            - float(row.get("weighted_pd_true_shadow", 0.0))
        )
        d_v = abs(
            float(row.get("weighted_miscoverage_V_primary", 0.0))
            - float(row.get("weighted_miscoverage_V_shadow", 0.0))
        )
        d_gamma = abs(
            float(row.get("gamma_cp_primary", 0.0)) - float(row.get("gamma_cp_shadow", 0.0))
        )
        max_return = max(max_return, d_return)
        max_v = max(max_v, d_v)
        max_gamma = max(max_gamma, d_gamma)
        rows.append(
            {
                "candidate_rank": int(row["candidate_rank"]),
                "eval_random_state": int(row["eval_random_state"]),
                "alpha": float(row["alpha"]),
                "abs_delta_weighted_pd_true": float(d_return),
                "abs_delta_weighted_miscoverage_V": float(d_v),
                "abs_delta_gamma_cp": float(d_gamma),
            }
        )
    status = "ok"
    if max_return > float(return_tol) or max_v > float(v_tol) or max_gamma > float(gamma_tol):
        status = "needs_review"
    return {
        "status": status,
        "max_abs_delta_return": float(max_return),
        "max_abs_delta_v": float(max_v),
        "max_abs_delta_gamma": float(max_gamma),
        "rows": rows,
    }


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
    bound_eval_partial_path = bound_eval_path.with_name(
        bound_eval_path.name.replace(".parquet", "_partial.parquet")
    )
    shadow_report_path = selection_path.with_name("portfolio_bound_solver_agreement.json")

    prior_status = load_runtime_status(status_path)
    base_elapsed_sec = float(prior_status.get("elapsed_sec", 0.0))
    context["frontier_total_units"] = int(prior_status.get("frontier_total_units", 0))
    context["frontier_completed_units"] = int(prior_status.get("frontier_completed_units", 0))
    context["bound_total_checks"] = int(prior_status.get("bound_total_checks", 0))
    context["helper_started_monotonic"] = time.monotonic()

    shortlist = pd.read_parquet(shortlist_path).copy()
    if shortlist.empty:
        raise RuntimeError("Shortlist is empty; exact tournament cannot run.")
    shortlist["candidate_rank"] = shortlist["candidate_rank"].astype(int)

    random_states = [int(v) for v in context["random_states"]]
    alpha_grid = sorted({float(v) for v in context["alpha_grid"]})
    pass1_alpha = float(context.get("exact_pass1_alpha", 0.01))
    pass2_bucket_min = int(context.get("exact_pass2_bucket_min", 8))
    exact_workers = max(1, int(context.get("exact_workers", 1)))
    checkpoint_every = max(1, int(context.get("exact_checkpoint_every", 32)))
    exact_shadow_backend = str(context.get("exact_shadow_backend", "none")).strip().lower()
    exact_shadow_top_k = max(0, int(context.get("exact_shadow_top_k", 0)))
    tol_return = float(context.get("exact_solver_agreement_return_abs", 250.0))
    tol_v = float(context.get("exact_solver_agreement_v_abs", 0.002))
    tol_gamma = float(context.get("exact_solver_agreement_gamma_abs", 0.005))

    aligned_by_seed = {
        int(seed): _load_aligned_dataset(
            conformal_intervals_path=str(context["conformal_intervals_path"]),
            max_candidates=int(context["max_candidates"]),
            random_state=int(seed),
        )
        for seed in random_states
    }

    # Pass 1: full shortlist at alpha=0.01.
    pass1_candidate_ranks = shortlist["candidate_rank"].astype(int).tolist()
    pass1_tasks = _build_tasks(
        shortlist=shortlist,
        candidate_ranks=pass1_candidate_ranks,
        random_states=random_states,
        alphas=[pass1_alpha],
        solver_backend=str(context["exact_solver_backend"]),
    )
    context["bound_total_checks"] = int(len(pass1_tasks))
    bound_rows: list[dict[str, Any]] = []
    completed_checks = 0
    _write_exact_status(
        context=context,
        base_elapsed_sec=base_elapsed_sec,
        bound_completed_checks=completed_checks,
        phase="exact_pass1_running",
        state="running",
        extra={
            "pass1_alpha": pass1_alpha,
            "pass1_candidates": len(pass1_candidate_ranks),
            "exact_workers": exact_workers,
        },
    )
    pass1_results = _iter_task_results(
        tasks=pass1_tasks,
        aligned_by_seed=aligned_by_seed,
        budget=float(context["budget"]),
        t_eval=float(context["t_eval"]),
        frontier_solver_backend=str(context["frontier_solver_backend"]),
        exact_solver_backend=str(context["exact_solver_backend"]),
        workers=exact_workers,
    )
    for idx, row in enumerate(pass1_results, start=1):
        completed_checks = _append_rows(
            rows_store=bound_rows,
            new_rows=[row],
            checkpoint_every=checkpoint_every,
            checkpoint_name=f"004_pass1_{idx:06d}",
            context=context,
            base_elapsed_sec=base_elapsed_sec,
            completed_checks=completed_checks,
            bound_eval_partial_path=bound_eval_partial_path,
            checkpoint_dir=checkpoint_dir,
            phase="exact_pass1_running",
            state="running",
            latest={
                "candidate_rank": int(row["candidate_rank"]),
                "eval_random_state": int(row["eval_random_state"]),
                "current_alpha": float(row["alpha"]),
            },
        )
    pass1_eval = pd.DataFrame(bound_rows)
    pass2_candidate_ranks, pass2_summary = _build_pass2_candidate_ranks(
        shortlist=shortlist,
        pass1_eval=pass1_eval,
        bucket_min=pass2_bucket_min,
    )
    write_runtime_checkpoint(
        STAGE_NAME,
        "010_pass1_complete",
        {
            "pass1_alpha": pass1_alpha,
            "pass1_checks": len(pass1_tasks),
            **pass2_summary,
        },
        checkpoint_dir=checkpoint_dir,
    )

    # Pass 2: full alpha grid (excluding pass1 alpha) on survivors + bucket quotas.
    pass2_alphas = [a for a in alpha_grid if not _float_equal(a, pass1_alpha)]
    pass2_tasks = _build_tasks(
        shortlist=shortlist,
        candidate_ranks=pass2_candidate_ranks,
        random_states=random_states,
        alphas=pass2_alphas,
        solver_backend=str(context["exact_solver_backend"]),
    )
    context["bound_total_checks"] = int(len(pass1_tasks) + len(pass2_tasks))
    _write_exact_status(
        context=context,
        base_elapsed_sec=base_elapsed_sec,
        bound_completed_checks=completed_checks,
        phase="exact_pass2_running",
        state="running",
        extra={
            "pass2_candidates": int(len(pass2_candidate_ranks)),
            "pass2_alphas": pass2_alphas,
            **pass2_summary,
        },
    )
    pass2_results = _iter_task_results(
        tasks=pass2_tasks,
        aligned_by_seed=aligned_by_seed,
        budget=float(context["budget"]),
        t_eval=float(context["t_eval"]),
        frontier_solver_backend=str(context["frontier_solver_backend"]),
        exact_solver_backend=str(context["exact_solver_backend"]),
        workers=exact_workers,
    )
    for idx, row in enumerate(pass2_results, start=1):
        completed_checks = _append_rows(
            rows_store=bound_rows,
            new_rows=[row],
            checkpoint_every=checkpoint_every,
            checkpoint_name=f"020_pass2_{idx:06d}",
            context=context,
            base_elapsed_sec=base_elapsed_sec,
            completed_checks=completed_checks,
            bound_eval_partial_path=bound_eval_partial_path,
            checkpoint_dir=checkpoint_dir,
            phase="exact_pass2_running",
            state="running",
            latest={
                "candidate_rank": int(row["candidate_rank"]),
                "eval_random_state": int(row["eval_random_state"]),
                "current_alpha": float(row["alpha"]),
            },
        )

    bound_eval = pd.DataFrame(bound_rows)
    if bound_eval.empty:
        raise RuntimeError("Exact tournament produced no rows.")
    shortlist_eval = _aggregate_exact_results(shortlist=shortlist, bound_eval=bound_eval)
    selected = shortlist_eval.iloc[0].copy()
    selected_policy = _policy_from_row(
        selected,
        solver_backend_override=str(context["exact_solver_backend"]),
    )

    # Optional shadow checks for top-K finalists at pass1 alpha.
    shadow_payload: dict[str, Any] = {"status": "not_run", "rows": []}
    if (
        exact_shadow_backend not in {"", "none"}
        and exact_shadow_backend != str(context["exact_solver_backend"]).strip().lower()
        and exact_shadow_top_k > 0
    ):
        top_k = shortlist_eval.head(int(exact_shadow_top_k)).copy()
        shadow_ranks = top_k["candidate_rank"].astype(int).tolist()
        shadow_tasks = _build_tasks(
            shortlist=shortlist_eval,
            candidate_ranks=shadow_ranks,
            random_states=random_states,
            alphas=[pass1_alpha],
            solver_backend=exact_shadow_backend,
        )
        _write_exact_status(
            context=context,
            base_elapsed_sec=base_elapsed_sec,
            bound_completed_checks=completed_checks,
            phase="exact_shadow_running",
            state="running",
            extra={
                "shadow_backend": exact_shadow_backend,
                "shadow_top_k": int(exact_shadow_top_k),
                "shadow_tasks": len(shadow_tasks),
            },
        )
        shadow_rows = list(
            _iter_task_results(
                tasks=shadow_tasks,
                aligned_by_seed=aligned_by_seed,
                budget=float(context["budget"]),
                t_eval=float(context["t_eval"]),
                frontier_solver_backend=str(context["frontier_solver_backend"]),
                exact_solver_backend=exact_shadow_backend,
                workers=max(1, min(exact_workers, 4)),
            )
        )
        shadow_eval = pd.DataFrame(shadow_rows)
        primary_eval = bound_eval[
            (bound_eval["candidate_rank"].isin(shadow_ranks))
            & bound_eval["alpha"].map(lambda x: _float_equal(float(x), pass1_alpha))
        ].copy()
        shadow_payload = _solver_agreement_report(
            primary_eval=primary_eval,
            shadow_eval=shadow_eval,
            return_tol=tol_return,
            v_tol=tol_v,
            gamma_tol=tol_gamma,
        )
        shadow_payload.update(
            {
                "shadow_backend": exact_shadow_backend,
                "shadow_top_k": int(exact_shadow_top_k),
                "tolerances": {
                    "return_abs": tol_return,
                    "v_abs": tol_v,
                    "gamma_abs": tol_gamma,
                },
            }
        )
        atomic_write_json(shadow_report_path, shadow_payload)
        write_runtime_checkpoint(
            STAGE_NAME,
            "030_shadow_complete",
            shadow_payload,
            checkpoint_dir=checkpoint_dir,
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
        "exact_shadow_backend": exact_shadow_backend,
        "exact_tournament": {
            "mode": "pass1_pass2",
            "pass1_alpha": pass1_alpha,
            "pass2_bucket_min": int(pass2_bucket_min),
            "pass2_candidate_count": int(len(pass2_candidate_ranks)),
            "workers": int(exact_workers),
            "checkpoint_every": int(checkpoint_every),
        },
        "solver_agreement_status": shadow_payload.get("status", "not_run"),
        "solver_agreement_report_path": str(shadow_report_path),
    }

    atomic_write_parquet(shortlist_eval, shortlist_path, index=False)
    atomic_write_parquet(bound_eval, bound_eval_path, index=False)
    if not bound_eval.empty:
        atomic_write_parquet(bound_eval, bound_eval_partial_path, index=False)
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
            "solver_agreement_status": shadow_payload.get("status", "not_run"),
            "exact_tournament_mode": "pass1_pass2",
        },
    )
    write_runtime_checkpoint(
        STAGE_NAME,
        "040_selection_complete",
        final_payload,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info(
        "External exact tournament complete: selected risk_tolerance={}, mode={}, gamma={}, alpha01_pass={}, agreement={}",
        selected["risk_tolerance"],
        selected["policy_mode"],
        selected["gamma"],
        selected["alpha01_exact_pass"],
        shadow_payload.get("status", "not_run"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
