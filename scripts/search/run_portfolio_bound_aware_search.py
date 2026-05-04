"""Focused bound-aware portfolio search with exact alpha checks and runtime monitoring."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.optimize_portfolio_tradeoff import (  # noqa: E402
    _align_loans_and_intervals,
    _load_candidates,
    _load_intervals,
    _parse_float_grid,
    _parse_percent_series,
    _solve_single,
)
from scripts.run_gpu_replay import _GpuSampler  # noqa: E402
from scripts.validate_alpha_gamma_bound import (  # noqa: E402
    _load_aligned_dataset,
    _validate_single_alpha,
)
from src.utils.pipeline_runtime import (  # noqa: E402
    atomic_write_json,
    atomic_write_parquet,
    write_runtime_checkpoint,
    write_runtime_status,
)

SCHEMA_VERSION = "2026-04-05.2"
STAGE_NAME = "portfolio_bound_aware"
DEFAULT_INCUMBENT_POLICY_PATH = (
    ROOT
    / "models"
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware"
    / "portfolio_bound_aware_selection.json"
)
DEFAULT_EXACT_HELPER_SCRIPT = ROOT / "scripts" / "search" / "run_portfolio_bound_exact_eval.py"
SEMANTIC_POLICY_FIELDS = [
    "risk_tolerance",
    "policy_mode",
    "gamma",
    "delta_cap_quantile",
    "tail_focus_quantile",
    "uncertainty_aversion",
    "min_budget_utilization",
    "pd_cap_slack_penalty",
    "solver_backend",
]


def _coerce_csv(raw: str | None) -> list[float]:
    if not raw:
        return []
    return _parse_float_grid(raw)


def _coerce_int_csv(raw: str | None, *, fallback: int) -> list[int]:
    if not raw:
        return [int(fallback)]
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    return values or [int(fallback)]


def _float_token(value: Any) -> float:
    return round(float(value), 10)


def _policy_semantic_key(policy_like: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for field in SEMANTIC_POLICY_FIELDS:
        value = policy_like[field]
        payload[field] = (
            str(value) if field in {"policy_mode", "solver_backend"} else _float_token(value)
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _policy_from_row(
    row: pd.Series,
    *,
    solver_backend_override: str | None = None,
) -> dict[str, Any]:
    return {
        "risk_tolerance": float(row["risk_tolerance"]),
        "policy_mode": str(row["policy_mode"]),
        "gamma": float(row["gamma"]),
        "delta_cap_quantile": float(row["delta_cap_quantile"]),
        "tail_focus_quantile": float(row["tail_focus_quantile"]),
        "uncertainty_aversion": float(row["uncertainty_aversion"]),
        "min_budget_utilization": float(row["min_budget_utilization"]),
        "pd_cap_slack_penalty": float(row["pd_cap_slack_penalty"]),
        "solver_backend": str(
            solver_backend_override
            if solver_backend_override is not None
            else row["solver_backend"]
        ),
    }


def _targeted_policy_grid(
    *,
    gamma_values: list[float],
    delta_cap_quantiles: list[float],
    tail_focus_quantiles: list[float],
    policy_modes: list[str] | None = None,
) -> list[tuple[str, float, float, float]]:
    allowed = {str(mode).strip() for mode in (policy_modes or []) if str(mode).strip()}
    use_all = not allowed
    grid: list[tuple[str, float, float, float]] = []
    for gamma in gamma_values:
        if use_all or "blended_uncertainty" in allowed:
            grid.append(("blended_uncertainty", float(gamma), 1.0, 1.0))
        if use_all or "capped_blended_uncertainty" in allowed:
            for delta_cap_quantile in delta_cap_quantiles:
                grid.append(
                    (
                        "capped_blended_uncertainty",
                        float(gamma),
                        float(delta_cap_quantile),
                        1.0,
                    )
                )
        if use_all or "tail_blended_uncertainty" in allowed:
            for tail_focus_quantile in tail_focus_quantiles:
                grid.append(
                    (
                        "tail_blended_uncertainty",
                        float(gamma),
                        1.0,
                        float(tail_focus_quantile),
                    )
                )
    return list(dict.fromkeys(grid))


def _eta_seconds(elapsed_sec: float, completed: int, total: int) -> float | None:
    if completed <= 0 or total <= 0 or completed >= total:
        return 0.0 if total > 0 and completed >= total else None
    return (elapsed_sec / max(completed, 1)) * max(total - completed, 0)


def _resource_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at_utc": datetime.now(tz=UTC).isoformat(),
        "cpu_count": int(os.cpu_count() or 0),
    }
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        parsed: dict[str, float] = {}
        for line in meminfo.splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            token = rest.strip().split()[0]
            parsed[key] = float(token)
        payload["memory_total_kib"] = int(parsed.get("MemTotal", 0))
        payload["memory_available_kib"] = int(parsed.get("MemAvailable", 0))
        payload["swap_free_kib"] = int(parsed.get("SwapFree", 0))
    except Exception as exc:  # pragma: no cover - best effort only
        payload["memory_probe_error"] = str(exc)
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
            if len(first) >= 7:
                payload["gpu"] = {
                    "name": first[0],
                    "driver_version": first[1],
                    "memory_total_mb": float(first[2]),
                    "memory_used_mb": float(first[3]),
                    "memory_free_mb": float(first[4]),
                    "gpu_util_pct": float(first[5]),
                    "memory_util_pct": float(first[6]),
                }
    except Exception as exc:  # pragma: no cover - best effort only
        payload["gpu_probe_error"] = str(exc)
    return payload


def _validate_cuopt_runtime() -> dict[str, Any]:
    modules = ["cuopt", "cudf", "cupy", "pyomo", "loguru"]
    payload: dict[str, Any] = {
        "validated_at_utc": datetime.now(tz=UTC).isoformat(),
        "python": sys.executable,
        "expected_release": "26.02",
    }
    for module_name in modules:
        module = importlib.import_module(module_name)
        payload[module_name] = getattr(module, "__version__", "ok")
    cuopt_version = str(payload.get("cuopt", ""))
    if not cuopt_version.startswith("26.02"):
        raise RuntimeError(
            f"cuOpt 26.02 is required for GPU bound-aware search; found {cuopt_version!r}."
        )
    return payload


class _ProgressTracker:
    def __init__(
        self,
        *,
        status_path: Path,
        checkpoint_dir: Path,
        run_tag: str,
        frontier_total_units: int,
    ) -> None:
        self.status_path = status_path
        self.checkpoint_dir = checkpoint_dir
        self.run_tag = run_tag
        self.frontier_total_units = int(frontier_total_units)
        self.bound_total_checks = 0
        self.frontier_completed_units = 0
        self.bound_completed_checks = 0
        self.started_at = time.monotonic()
        self._checkpoint_seq = 0

    def _elapsed(self) -> float:
        return float(time.monotonic() - self.started_at)

    def _payload(
        self, *, phase: str, state: str, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        elapsed_sec = self._elapsed()
        global_total = int(self.frontier_total_units + self.bound_total_checks)
        global_completed = int(self.frontier_completed_units + self.bound_completed_checks)
        payload: dict[str, Any] = {
            "frontier_total_units": int(self.frontier_total_units),
            "frontier_completed_units": int(self.frontier_completed_units),
            "frontier_pct_complete": (
                float(self.frontier_completed_units / max(self.frontier_total_units, 1))
                if self.frontier_total_units > 0
                else 1.0
            ),
            "bound_total_checks": int(self.bound_total_checks),
            "bound_completed_checks": int(self.bound_completed_checks),
            "bound_pct_complete": (
                float(self.bound_completed_checks / max(self.bound_total_checks, 1))
                if self.bound_total_checks > 0
                else 0.0
            ),
            "global_total_units": int(global_total),
            "global_completed_units": int(global_completed),
            "global_pct_complete": (
                float(global_completed / max(global_total, 1)) if global_total > 0 else 1.0
            ),
            "elapsed_sec": float(elapsed_sec),
            "eta_sec": _eta_seconds(elapsed_sec, global_completed, global_total),
        }
        if extra:
            payload.update(extra)
        write_runtime_status(
            STAGE_NAME,
            phase=phase,
            state=state,
            run_tag=self.run_tag,
            status_path=self.status_path,
            extra=payload,
        )
        return payload

    def checkpoint(self, name: str, payload: dict[str, Any]) -> None:
        self._checkpoint_seq += 1
        write_runtime_checkpoint(
            STAGE_NAME,
            f"{self._checkpoint_seq:03d}_{name}",
            payload,
            checkpoint_dir=self.checkpoint_dir,
        )

    def start(self, *, extra: dict[str, Any] | None = None) -> None:
        self._payload(phase="loading_inputs", state="running", extra=extra)

    def frontier_progress(
        self, *, completed_units: int, extra: dict[str, Any] | None = None
    ) -> None:
        self.frontier_completed_units = int(completed_units)
        self._payload(phase="frontier_running", state="running", extra=extra)

    def frontier_complete(self, *, extra: dict[str, Any] | None = None) -> None:
        self.frontier_completed_units = int(self.frontier_total_units)
        payload = self._payload(phase="frontier_complete", state="running", extra=extra)
        self.checkpoint("frontier_complete", payload)

    def set_bound_total(
        self, bound_total_checks: int, *, extra: dict[str, Any] | None = None
    ) -> None:
        self.bound_total_checks = int(bound_total_checks)
        payload = self._payload(phase="shortlist_building", state="running", extra=extra)
        self.checkpoint("shortlist_built", payload)

    def bound_progress(self, *, completed_checks: int, extra: dict[str, Any] | None = None) -> None:
        self.bound_completed_checks = int(completed_checks)
        self._payload(phase="exact_bound_running", state="running", extra=extra)

    def complete(self, *, extra: dict[str, Any] | None = None) -> None:
        self.bound_completed_checks = int(self.bound_total_checks)
        payload = self._payload(phase="selection_complete", state="completed", extra=extra)
        self.checkpoint("selection_complete", payload)


def _load_incumbent_policy(path: str | Path | None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_INCUMBENT_POLICY_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    selected = payload.get("selected_policy", payload)
    return {
        "risk_tolerance": float(selected.get("risk_tolerance", 0.16)),
        "policy_mode": str(selected.get("policy_mode", "blended_uncertainty")),
        "gamma": float(selected.get("gamma", 0.5)),
        "delta_cap_quantile": float(selected.get("delta_cap_quantile", 1.0)),
        "tail_focus_quantile": float(selected.get("tail_focus_quantile", 1.0)),
        "uncertainty_aversion": float(selected.get("uncertainty_aversion", 0.0)),
        "min_budget_utilization": float(selected.get("min_budget_utilization", 0.0)),
        "pd_cap_slack_penalty": float(selected.get("pd_cap_slack_penalty", 0.0)),
        "solver_backend": str(selected.get("solver_backend", "highs")),
        "source_path": str(target),
    }


def _policy_grid_size(
    *,
    gamma_values: list[float],
    delta_cap_quantiles: list[float],
    tail_focus_quantiles: list[float],
    aversion_values: list[float],
    budget_profiles: list[dict[str, float]],
    policy_modes: list[str] | None = None,
) -> int:
    policy_grid = _targeted_policy_grid(
        gamma_values=gamma_values,
        delta_cap_quantiles=delta_cap_quantiles,
        tail_focus_quantiles=tail_focus_quantiles,
        policy_modes=policy_modes,
    )
    return int(len(policy_grid) * len(aversion_values) * len(budget_profiles))


def _build_frontier_for_seed(
    *,
    config_path: str,
    conformal_intervals_path: str,
    risk_values: list[float],
    aversion_values: list[float],
    gamma_values: list[float],
    delta_cap_quantiles: list[float],
    tail_focus_quantiles: list[float],
    budget_profiles: list[dict[str, float]],
    max_candidates: int,
    random_state: int,
    solver_backend: str,
    cuopt_presolve: int | None,
    policy_modes: list[str] | None,
    progress_hook: callable,
) -> pd.DataFrame:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    candidates = _load_candidates().reset_index(drop=True)
    intervals = _load_intervals(conformal_intervals_path=conformal_intervals_path).reset_index(
        drop=True
    )
    loans, ints = _align_loans_and_intervals(
        candidates=candidates,
        intervals=intervals,
        max_candidates=max_candidates,
        random_state=random_state,
    )
    n = len(loans)
    col_point = "y_pred" if "y_pred" in ints.columns else "pd_point"
    col_low = "pd_low_90" if "pd_low_90" in ints.columns else "pd_low"
    col_high = "pd_high_90" if "pd_high_90" in ints.columns else "pd_high"
    pd_point = ints[col_point].to_numpy(dtype=float)
    pd_low = ints[col_low].to_numpy(dtype=float)
    pd_high = ints[col_high].to_numpy(dtype=float)
    lgd = pd.Series([0.45] * n, dtype=float).to_numpy(dtype=float)
    int_rates = (
        _parse_percent_series(loans["int_rate"])
        if "int_rate" in loans.columns
        else pd.Series([0.12] * n, dtype=float).to_numpy(dtype=float)
    )
    default_flag = (
        pd.to_numeric(loans["default_flag"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if "default_flag" in loans.columns
        else pd.Series([0] * n, dtype=int).to_numpy(dtype=int)
    )
    total_budget = float(config["portfolio"]["total_budget"])
    max_concentration = float(config["portfolio"]["max_concentration"])
    time_limit = int(config["optimization"]["time_limit"])
    threads = int(config["optimization"]["threads"])
    policy_grid = _targeted_policy_grid(
        gamma_values=gamma_values,
        delta_cap_quantiles=delta_cap_quantiles,
        tail_focus_quantiles=tail_focus_quantiles,
        policy_modes=policy_modes,
    )

    rows: list[dict[str, Any]] = []
    logger.info(
        "Running focused portfolio frontier on n={:,}, random_state={}, risk_values={}, aversion_values={}, policies={}",
        n,
        random_state,
        risk_values,
        aversion_values,
        len(policy_grid),
    )
    completed = 0
    for risk_tol in risk_values:
        baseline, _ = _solve_single(
            loans=loans,
            pd_point=pd_point,
            pd_low=pd_low,
            pd_high=pd_high,
            lgd=lgd,
            int_rates=int_rates,
            default_flag=default_flag,
            total_budget=total_budget,
            max_concentration=max_concentration,
            risk_tolerance=float(risk_tol),
            robust=False,
            uncertainty_aversion=0.0,
            min_budget_utilization=0.0,
            pd_cap_slack_penalty=0.0,
            time_limit=time_limit,
            threads=threads,
            solver_backend=solver_backend,
            random_seed=int(random_state),
            cuopt_presolve=cuopt_presolve,
        )
        completed += 1
        progress_hook(
            completed,
            {
                "phase_random_state": int(random_state),
                "latest_policy_mode": "point_estimate",
                "latest_risk_tolerance": float(risk_tol),
                "latest_gamma": 0.0,
            },
        )
        baseline_ret = float(baseline["expected_return_net_point"])
        baseline_realized = float(baseline["realized_total_return"])
        for policy_mode, gamma, delta_cap_quantile, tail_focus_quantile in policy_grid:
            for aversion in aversion_values:
                for budget_profile in budget_profiles:
                    robust_run, _ = _solve_single(
                        loans=loans,
                        pd_point=pd_point,
                        pd_low=pd_low,
                        pd_high=pd_high,
                        lgd=lgd,
                        int_rates=int_rates,
                        default_flag=default_flag,
                        total_budget=total_budget,
                        max_concentration=max_concentration,
                        risk_tolerance=float(risk_tol),
                        robust=True,
                        uncertainty_aversion=float(aversion),
                        min_budget_utilization=float(budget_profile["min_budget_utilization"]),
                        pd_cap_slack_penalty=float(budget_profile["pd_cap_slack_penalty"]),
                        time_limit=time_limit,
                        threads=threads,
                        solver_backend=solver_backend,
                        policy_mode=policy_mode,
                        gamma=float(gamma),
                        delta_cap_quantile=float(delta_cap_quantile),
                        tail_focus_quantile=float(tail_focus_quantile),
                        random_seed=int(random_state),
                        cuopt_presolve=cuopt_presolve,
                    )
                    completed += 1
                    progress_hook(
                        completed,
                        {
                            "phase_random_state": int(random_state),
                            "latest_policy_mode": str(policy_mode),
                            "latest_risk_tolerance": float(risk_tol),
                            "latest_gamma": float(gamma),
                        },
                    )
                    por = baseline_ret - float(robust_run["expected_return_net_point"])
                    por_pct = por / (abs(baseline_ret) + 1e-6) * 100.0
                    realized_total_return = float(robust_run["realized_total_return"])
                    ab_diff_total_return = float(realized_total_return - baseline_realized)
                    ab_pass = bool(ab_diff_total_return >= -(abs(baseline_realized) * 0.05))
                    rows.append(
                        {
                            "sample_random_state": int(random_state),
                            "risk_tolerance": float(risk_tol),
                            "policy_mode": str(policy_mode),
                            "gamma": float(gamma),
                            "delta_cap_quantile": float(delta_cap_quantile),
                            "tail_focus_quantile": float(tail_focus_quantile),
                            "uncertainty_aversion": float(aversion),
                            "min_budget_utilization": float(
                                budget_profile["min_budget_utilization"]
                            ),
                            "pd_cap_slack_penalty": float(budget_profile["pd_cap_slack_penalty"]),
                            "budget_profile_name": str(budget_profile["name"]),
                            "price_of_robustness": float(por),
                            "price_of_robustness_pct": float(por_pct),
                            "ab_diff_total_return": float(ab_diff_total_return),
                            "ab_pass": ab_pass,
                            **robust_run,
                        }
                    )
    return pd.DataFrame(rows).reset_index(drop=True)


def _aggregate_frontier(frontier_raw: pd.DataFrame) -> pd.DataFrame:
    grouped = frontier_raw.groupby(SEMANTIC_POLICY_FIELDS, dropna=False)
    frontier = grouped.agg(
        seed_count=("sample_random_state", "nunique"),
        sample_random_states=(
            "sample_random_state",
            lambda s: ",".join(map(str, sorted({int(v) for v in s}))),
        ),
        ab_pass_all=("ab_pass", "all"),
        ab_pass_rate=("ab_pass", "mean"),
        realized_total_return=("realized_total_return", "mean"),
        realized_total_return_max=("realized_total_return", "max"),
        price_of_robustness=("price_of_robustness", "mean"),
        price_of_robustness_pct=("price_of_robustness_pct", "mean"),
        ab_diff_total_return=("ab_diff_total_return", "mean"),
        objective_value=("objective_value", "mean"),
        n_funded=("n_funded", "mean"),
        total_allocated=("total_allocated", "mean"),
        expected_return_net_point=("expected_return_net_point", "mean"),
        worst_case_pd=("worst_case_pd", "mean"),
        point_pd=("point_pd", "mean"),
        pd_cap_slack=("pd_cap_slack", "max"),
    ).reset_index()
    frontier["semantic_policy_key"] = frontier.apply(_policy_semantic_key, axis=1)
    return frontier.reset_index(drop=True)


def _apply_rank(
    df: pd.DataFrame, *, by: list[str], ascending: list[bool], rank_col: str
) -> pd.DataFrame:
    ranked = df.sort_values(by=by, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked[rank_col] = range(1, len(ranked) + 1)
    return ranked.loc[:, ["semantic_policy_key", rank_col]]


def _rank_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    work = frontier.copy()
    if "semantic_policy_key" not in work.columns:
        work["semantic_policy_key"] = work.apply(_policy_semantic_key, axis=1)
    return_rank = _apply_rank(
        work,
        by=["ab_pass_all", "realized_total_return", "price_of_robustness"],
        ascending=[False, False, False],
        rank_col="return_first_rank",
    )
    proxy_rank = _apply_rank(
        work,
        by=[
            "ab_pass_all",
            "risk_tolerance",
            "worst_case_pd",
            "point_pd",
            "pd_cap_slack",
            "realized_total_return",
            "price_of_robustness",
        ],
        ascending=[False, True, True, True, True, False, False],
        rank_col="bound_proxy_rank",
    )
    work = work.merge(return_rank, on="semantic_policy_key", how="left")
    work = work.merge(proxy_rank, on="semantic_policy_key", how="left")
    return work


def _select_top_unique(
    frame: pd.DataFrame,
    *,
    limit: int,
    seen: set[str],
    bucket_name: str,
    selected_rows: list[dict[str, Any]],
) -> None:
    if limit <= 0 or frame.empty:
        return
    added = 0
    for _, row in frame.iterrows():
        key = row["semantic_policy_key"]
        if key in seen:
            continue
        payload = row.to_dict()
        payload["shortlist_bucket"] = bucket_name
        selected_rows.append(payload)
        seen.add(key)
        added += 1
        if added >= int(limit):
            break


def _build_forced_policy_keys(
    *,
    incumbent_policy: dict[str, Any],
    incumbent_risk_neighbors: list[float],
    incumbent_gamma_neighbors: list[float],
    incumbent_policy_modes: list[str],
    budget_profiles: list[dict[str, float]],
    solver_backend: str,
) -> set[str]:
    keys: set[str] = set()
    for risk in incumbent_risk_neighbors:
        for gamma in incumbent_gamma_neighbors:
            for policy_mode in incumbent_policy_modes:
                delta_cap_quantile = (
                    float(incumbent_policy["delta_cap_quantile"])
                    if policy_mode == "capped_blended_uncertainty"
                    else 1.0
                )
                tail_focus_quantile = (
                    float(incumbent_policy["tail_focus_quantile"])
                    if policy_mode == "tail_blended_uncertainty"
                    else 1.0
                )
                for budget_profile in budget_profiles:
                    keys.add(
                        _policy_semantic_key(
                            {
                                "risk_tolerance": risk,
                                "policy_mode": policy_mode,
                                "gamma": gamma,
                                "delta_cap_quantile": delta_cap_quantile,
                                "tail_focus_quantile": tail_focus_quantile,
                                "uncertainty_aversion": float(
                                    incumbent_policy["uncertainty_aversion"]
                                ),
                                "min_budget_utilization": float(
                                    budget_profile["min_budget_utilization"]
                                ),
                                "pd_cap_slack_penalty": float(
                                    budget_profile["pd_cap_slack_penalty"]
                                ),
                                "solver_backend": str(solver_backend),
                            }
                        )
                    )
    return keys


def _build_stratified_shortlist(
    *,
    frontier: pd.DataFrame,
    shortlist_top_k: int,
    bucket_return_k: int,
    bucket_proxy_k: int,
    bucket_family_k: int,
    bucket_region_k: int,
    incumbent_policy: dict[str, Any],
    incumbent_risk_neighbors: list[float],
    incumbent_gamma_neighbors: list[float],
    incumbent_policy_modes: list[str],
    budget_profiles: list[dict[str, float]],
    solver_backend: str,
) -> pd.DataFrame:
    ranked = _rank_frontier(frontier)
    forced_keys = _build_forced_policy_keys(
        incumbent_policy=incumbent_policy,
        incumbent_risk_neighbors=incumbent_risk_neighbors,
        incumbent_gamma_neighbors=incumbent_gamma_neighbors,
        incumbent_policy_modes=incumbent_policy_modes,
        budget_profiles=budget_profiles,
        solver_backend=solver_backend,
    )
    region_mask = (
        ranked["risk_tolerance"].isin([float(v) for v in incumbent_risk_neighbors])
        & ranked["gamma"].isin([float(v) for v in incumbent_gamma_neighbors])
        & ranked["policy_mode"].isin([str(v) for v in incumbent_policy_modes])
    )

    selected_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    forced = ranked[ranked["semantic_policy_key"].isin(forced_keys)].sort_values(
        by=["bound_proxy_rank", "return_first_rank"],
        ascending=[True, True],
        kind="mergesort",
    )
    _select_top_unique(
        forced,
        limit=len(forced),
        seen=seen,
        bucket_name="forced_incumbent_neighbors",
        selected_rows=selected_rows,
    )

    region = ranked[region_mask].sort_values(
        by=["bound_proxy_rank", "return_first_rank"],
        ascending=[True, True],
        kind="mergesort",
    )
    _select_top_unique(
        region,
        limit=int(bucket_region_k),
        seen=seen,
        bucket_name="incumbent_region",
        selected_rows=selected_rows,
    )

    conservative = ranked.sort_values(
        by=["bound_proxy_rank", "return_first_rank"],
        ascending=[True, True],
        kind="mergesort",
    )
    _select_top_unique(
        conservative,
        limit=int(bucket_proxy_k),
        seen=seen,
        bucket_name="conservative_proxy",
        selected_rows=selected_rows,
    )

    for family in sorted(ranked["policy_mode"].astype(str).unique()):
        family_frame = ranked[ranked["policy_mode"] == family].sort_values(
            by=["bound_proxy_rank", "return_first_rank"],
            ascending=[True, True],
            kind="mergesort",
        )
        _select_top_unique(
            family_frame,
            limit=int(bucket_family_k),
            seen=seen,
            bucket_name=f"family::{family}",
            selected_rows=selected_rows,
        )

    return_global = ranked.sort_values(
        by=["return_first_rank", "bound_proxy_rank"],
        ascending=[True, True],
        kind="mergesort",
    )
    _select_top_unique(
        return_global,
        limit=int(bucket_return_k),
        seen=seen,
        bucket_name="return_global",
        selected_rows=selected_rows,
    )

    shortlist = pd.DataFrame(selected_rows)
    if shortlist.empty:
        raise ValueError("Stratified shortlist is empty; cannot continue.")
    shortlist = shortlist.iloc[: int(shortlist_top_k)].copy().reset_index(drop=True)
    shortlist["candidate_rank"] = range(1, len(shortlist) + 1)
    return shortlist


def _aggregate_exact_results(
    *,
    shortlist: pd.DataFrame,
    bound_eval: pd.DataFrame,
) -> pd.DataFrame:
    alpha01 = (
        bound_eval[bound_eval["alpha"] == 0.01]
        .groupby(SEMANTIC_POLICY_FIELDS, dropna=False)
        .agg(
            alpha01_exact_pass=("all_bounds_hold", "all"),
            alpha01_pass_rate=("all_bounds_hold", "mean"),
            alpha01_gamma_cp=("gamma_cp", "mean"),
            alpha01_weighted_miscoverage_V=("weighted_miscoverage_V", "mean"),
            alpha01_violation=("violation", "max"),
            alpha01_weighted_pd_true=("weighted_pd_true", "mean"),
            alpha01_weighted_pd_constraint_used=("weighted_pd_constraint_used", "mean"),
            alpha01_empirical_coverage_funded=("empirical_coverage_funded", "mean"),
        )
        .reset_index()
    )
    alpha03 = (
        bound_eval[bound_eval["alpha"] == 0.03]
        .groupby(SEMANTIC_POLICY_FIELDS, dropna=False)
        .agg(
            alpha03_exact_pass=("all_bounds_hold", "all"),
            alpha03_weighted_miscoverage_V=("weighted_miscoverage_V", "mean"),
        )
        .reset_index()
    )
    alpha10 = (
        bound_eval[bound_eval["alpha"] == 0.10]
        .groupby(SEMANTIC_POLICY_FIELDS, dropna=False)
        .agg(
            alpha10_exact_pass=("all_bounds_hold", "all"),
            alpha10_weighted_miscoverage_V=("weighted_miscoverage_V", "mean"),
        )
        .reset_index()
    )

    work = shortlist.copy()
    work = work.merge(alpha01, on=SEMANTIC_POLICY_FIELDS, how="left")
    work = work.merge(alpha03, on=SEMANTIC_POLICY_FIELDS, how="left")
    work = work.merge(alpha10, on=SEMANTIC_POLICY_FIELDS, how="left")
    bool_cols = ["alpha01_exact_pass", "alpha03_exact_pass", "alpha10_exact_pass"]
    for col in bool_cols:
        work[col] = work[col].where(work[col].notna(), False).astype(bool)
    work = work.sort_values(
        by=[
            "alpha01_exact_pass",
            "alpha03_exact_pass",
            "ab_pass_all",
            "realized_total_return",
            "price_of_robustness",
            "alpha01_weighted_miscoverage_V",
            "alpha01_gamma_cp",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    )
    return work.reset_index(drop=True)


def _selection_reason(row: pd.Series) -> str:
    if bool(row["alpha01_exact_pass"]):
        return "selected_best_alpha01_exact_pass"
    if bool(row["alpha03_exact_pass"]):
        return "selected_fallback_alpha03_exact_pass"
    return "selected_best_available_without_alpha01_pass"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--conformal-intervals-path", required=True)
    parser.add_argument("--run-label", default="rank1_bound_aware")
    parser.add_argument("--risk-grid", default="0.14,0.15,0.16,0.17")
    parser.add_argument("--aversion-grid", default="0.0,0.25")
    parser.add_argument("--gamma-grid", default="0.2,0.35,0.5")
    parser.add_argument("--delta-cap-grid", default="0.5,0.75,1.0")
    parser.add_argument("--tail-focus-grid", default="0.9,0.95,1.0")
    parser.add_argument("--budget-profiles", default="free,floored")
    parser.add_argument("--shortlist-top-k", type=int, default=160)
    parser.add_argument("--bucket-return-k", type=int, default=40)
    parser.add_argument("--bucket-proxy-k", type=int, default=40)
    parser.add_argument("--bucket-family-k", type=int, default=20)
    parser.add_argument("--bucket-region-k", type=int, default=20)
    parser.add_argument("--alpha-grid", default="0.01,0.03,0.10")
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-states", default="")
    parser.add_argument("--solver-backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument("--exact-solver-backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument("--exact-python-executable", default="")
    parser.add_argument("--exact-helper-script", default=str(DEFAULT_EXACT_HELPER_SCRIPT))
    parser.add_argument("--cuopt-presolve", type=int, default=1)
    parser.add_argument("--policy-modes", default="")
    parser.add_argument(
        "--incumbent-policy-path",
        default=str(DEFAULT_INCUMBENT_POLICY_PATH),
    )
    parser.add_argument("--incumbent-risk-neighbors", default="0.155,0.16,0.165,0.17")
    parser.add_argument("--incumbent-gamma-neighbors", default="0.45,0.5,0.55")
    parser.add_argument(
        "--incumbent-policy-modes",
        default="blended_uncertainty,capped_blended_uncertainty",
    )
    parser.add_argument("--budget", type=float, default=1_000_000.0)
    parser.add_argument("--t-eval", type=float, default=0.05)
    args = parser.parse_args(argv)

    run_label = str(args.run_label).strip().replace("/", "_")
    output_dir = ROOT / "data" / "processed" / "portfolio_bound_aware" / run_label
    model_dir = ROOT / "models" / "portfolio_bound_aware" / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    status_path = model_dir / f"{STAGE_NAME}_runtime_status.json"
    checkpoint_dir = model_dir / f"{STAGE_NAME}_runtime_checkpoints"
    resource_path = model_dir / "resource_snapshot.json"
    gpu_csv_path = model_dir / "gpu_samples.csv"

    risk_values = _coerce_csv(args.risk_grid)
    aversion_values = _coerce_csv(args.aversion_grid)
    gamma_values = _coerce_csv(args.gamma_grid)
    delta_cap_quantiles = _coerce_csv(args.delta_cap_grid)
    tail_focus_quantiles = _coerce_csv(args.tail_focus_grid)
    alpha_grid = _coerce_csv(args.alpha_grid)
    random_states = _coerce_int_csv(args.random_states, fallback=int(args.random_state))
    incumbent_risk_neighbors = _coerce_csv(args.incumbent_risk_neighbors)
    incumbent_gamma_neighbors = _coerce_csv(args.incumbent_gamma_neighbors)
    incumbent_policy_modes = [
        part.strip() for part in str(args.incumbent_policy_modes).split(",") if part.strip()
    ]
    policy_modes = [part.strip() for part in str(args.policy_modes).split(",") if part.strip()]
    exact_python_executable = str(args.exact_python_executable).strip()
    exact_helper_script = Path(str(args.exact_helper_script)).resolve()

    budget_profiles: list[dict[str, float]] = []
    for token in [
        part.strip().lower() for part in str(args.budget_profiles).split(",") if part.strip()
    ]:
        if token == "free":
            budget_profiles.append(
                {"name": "free_budget", "min_budget_utilization": 0.0, "pd_cap_slack_penalty": 0.0}
            )
        elif token == "floored":
            budget_profiles.append(
                {
                    "name": "floored_budget",
                    "min_budget_utilization": 0.05,
                    "pd_cap_slack_penalty": 1.5,
                }
            )
        else:
            raise ValueError(f"Unsupported budget profile: {token}")
    if not budget_profiles:
        raise ValueError("At least one budget profile is required")

    incumbent_policy = _load_incumbent_policy(args.incumbent_policy_path)
    backend_validation: dict[str, Any] | None = None
    if (
        str(args.solver_backend).strip().lower() == "cuopt"
        or str(args.exact_solver_backend).strip().lower() == "cuopt"
    ):
        backend_validation = _validate_cuopt_runtime()

    policy_grid_count = _policy_grid_size(
        gamma_values=gamma_values,
        delta_cap_quantiles=delta_cap_quantiles,
        tail_focus_quantiles=tail_focus_quantiles,
        aversion_values=aversion_values,
        budget_profiles=budget_profiles,
        policy_modes=policy_modes,
    )
    frontier_total_units = int(len(random_states) * len(risk_values) * (1 + policy_grid_count))
    tracker = _ProgressTracker(
        status_path=status_path,
        checkpoint_dir=checkpoint_dir,
        run_tag=run_label,
        frontier_total_units=frontier_total_units,
    )
    resource_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_label": run_label,
        "solver_backend": str(args.solver_backend),
        "exact_solver_backend": str(args.exact_solver_backend),
        "start": _resource_snapshot(),
        "backend_validation": backend_validation,
    }
    atomic_write_json(resource_path, resource_payload)
    tracker.start(
        extra={
            "random_states": random_states,
            "search_space": {
                "risk_grid": risk_values,
                "aversion_grid": aversion_values,
                "gamma_grid": gamma_values,
                "delta_cap_grid": delta_cap_quantiles,
                "tail_focus_grid": tail_focus_quantiles,
                "budget_profiles": budget_profiles,
                "alpha_grid": alpha_grid,
                "max_candidates": int(args.max_candidates),
            },
        }
    )

    gpu_sampler: _GpuSampler | None = None
    if str(args.solver_backend).strip().lower() == "cuopt":
        gpu_sampler = _GpuSampler(gpu_csv_path)
        gpu_sampler.start()

    frontier_frames: list[pd.DataFrame] = []
    frontier_completed = 0
    try:
        for seed in random_states:
            seed_offset = frontier_completed

            def _progress_hook(
                local_completed: int,
                extra: dict[str, Any],
                _offset: int = seed_offset,
            ) -> None:
                tracker.frontier_progress(completed_units=_offset + local_completed, extra=extra)

            frontier_seed = _build_frontier_for_seed(
                config_path=args.config,
                conformal_intervals_path=args.conformal_intervals_path,
                risk_values=risk_values,
                aversion_values=aversion_values,
                gamma_values=gamma_values,
                delta_cap_quantiles=delta_cap_quantiles,
                tail_focus_quantiles=tail_focus_quantiles,
                budget_profiles=budget_profiles,
                max_candidates=int(args.max_candidates),
                random_state=int(seed),
                solver_backend=str(args.solver_backend),
                cuopt_presolve=int(args.cuopt_presolve)
                if str(args.solver_backend) == "cuopt"
                else None,
                policy_modes=policy_modes,
                progress_hook=_progress_hook,
            )
            frontier_frames.append(frontier_seed)
            frontier_completed += int(len(risk_values) * (1 + policy_grid_count))

        frontier_raw = (
            pd.concat(frontier_frames, ignore_index=True) if frontier_frames else pd.DataFrame()
        )
        if frontier_raw.empty:
            raise ValueError("Frontier search produced zero candidate rows.")
        frontier = _aggregate_frontier(frontier_raw)
        tracker.frontier_complete(
            extra={
                "frontier_policy_count": int(len(frontier)),
                "frontier_raw_rows": int(len(frontier_raw)),
            }
        )

        shortlist = _build_stratified_shortlist(
            frontier=frontier,
            shortlist_top_k=int(args.shortlist_top_k),
            bucket_return_k=int(args.bucket_return_k),
            bucket_proxy_k=int(args.bucket_proxy_k),
            bucket_family_k=int(args.bucket_family_k),
            bucket_region_k=int(args.bucket_region_k),
            incumbent_policy=incumbent_policy,
            incumbent_risk_neighbors=incumbent_risk_neighbors,
            incumbent_gamma_neighbors=incumbent_gamma_neighbors,
            incumbent_policy_modes=incumbent_policy_modes,
            budget_profiles=budget_profiles,
            solver_backend=str(args.solver_backend),
        )
        bound_total_checks = int(len(shortlist) * len(alpha_grid) * len(random_states))
        tracker.set_bound_total(
            bound_total_checks,
            extra={
                "shortlist_size": int(len(shortlist)),
                "shortlist_buckets": shortlist["shortlist_bucket"]
                .value_counts(dropna=False)
                .to_dict(),
            },
        )

        atomic_write_parquet(
            frontier_raw, output_dir / "portfolio_bound_aware_frontier_raw.parquet", index=False
        )
        atomic_write_parquet(
            frontier, output_dir / "portfolio_bound_aware_frontier.parquet", index=False
        )
        atomic_write_parquet(
            shortlist, output_dir / "portfolio_bound_aware_shortlist.parquet", index=False
        )

        selection_context = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "run_label": run_label,
            "conformal_intervals_path": str(args.conformal_intervals_path),
            "search_space": {
                "risk_grid": risk_values,
                "aversion_grid": aversion_values,
                "gamma_grid": gamma_values,
                "delta_cap_grid": delta_cap_quantiles,
                "tail_focus_grid": tail_focus_quantiles,
                "budget_profiles": budget_profiles,
                "alpha_grid": alpha_grid,
                "max_candidates": int(args.max_candidates),
                "random_states": random_states,
                "policy_modes": policy_modes,
                "bucket_return_k": int(args.bucket_return_k),
                "bucket_proxy_k": int(args.bucket_proxy_k),
                "bucket_family_k": int(args.bucket_family_k),
                "bucket_region_k": int(args.bucket_region_k),
                "incumbent_policy_path": str(args.incumbent_policy_path),
                "incumbent_risk_neighbors": incumbent_risk_neighbors,
                "incumbent_gamma_neighbors": incumbent_gamma_neighbors,
                "incumbent_policy_modes": incumbent_policy_modes,
            },
            "selection_policy": {
                "shortlist_strategy": "stratified_bound_first",
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
            "frontier_raw_path": str(output_dir / "portfolio_bound_aware_frontier_raw.parquet"),
            "frontier_path": str(output_dir / "portfolio_bound_aware_frontier.parquet"),
            "shortlist_path": str(output_dir / "portfolio_bound_aware_shortlist.parquet"),
            "bound_eval_path": str(output_dir / "portfolio_bound_aware_bound_eval.parquet"),
            "selection_path": str(model_dir / "portfolio_bound_aware_selection.json"),
            "runtime_status_path": str(status_path),
            "runtime_checkpoint_dir": str(checkpoint_dir),
            "resource_snapshot_path": str(resource_path),
            "frontier_solver_backend": str(args.solver_backend),
            "exact_solver_backend": str(args.exact_solver_backend),
            "budget": float(args.budget),
            "t_eval": float(args.t_eval),
            "max_candidates": int(args.max_candidates),
            "random_states": random_states,
            "alpha_grid": alpha_grid,
        }
        exact_context_path = model_dir / "portfolio_bound_aware_exact_context.json"
        atomic_write_json(exact_context_path, selection_context)

        if exact_python_executable:
            current_python = Path(sys.executable)
            requested_python = Path(exact_python_executable)
            if requested_python != current_python:
                cmd = [
                    str(requested_python),
                    str(exact_helper_script),
                    "--context-path",
                    str(exact_context_path),
                ]
                logger.info(
                    "Delegating exact bound stage to external Python: {}",
                    " ".join(cmd),
                )
                subprocess.run(cmd, cwd=str(ROOT), check=True)
                resource_payload["end"] = _resource_snapshot()
                atomic_write_json(resource_path, resource_payload)
                return 0

        aligned_by_seed = {
            int(seed): _load_aligned_dataset(
                conformal_intervals_path=args.conformal_intervals_path,
                max_candidates=int(args.max_candidates),
                random_state=int(seed),
            )
            for seed in random_states
        }
        bound_rows: list[dict[str, Any]] = []
        completed_checks = 0
        for _, row in shortlist.iterrows():
            policy = _policy_from_row(
                row,
                solver_backend_override=str(args.exact_solver_backend),
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
                        budget=float(args.budget),
                        t_eval=float(args.t_eval),
                    )
                    bound_rows.append(
                        {
                            "candidate_rank": int(candidate_payload["candidate_rank"]),
                            "eval_random_state": int(eval_seed),
                            "frontier_solver_backend": str(args.solver_backend),
                            "exact_solver_backend": str(args.exact_solver_backend),
                            **candidate_payload,
                            **result,
                        }
                    )
                    completed_checks += 1
                    tracker.bound_progress(
                        completed_checks=completed_checks,
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
            solver_backend_override=str(args.exact_solver_backend),
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "run_label": run_label,
            "conformal_intervals_path": str(args.conformal_intervals_path),
            "search_space": selection_context["search_space"],
            "selection_policy": selection_context["selection_policy"],
            "selected_policy": selected_policy,
            "selected_metrics": selected.to_dict(),
            "selection_reason": _selection_reason(selected),
            "frontier_raw_path": selection_context["frontier_raw_path"],
            "frontier_path": selection_context["frontier_path"],
            "shortlist_path": selection_context["shortlist_path"],
            "bound_eval_path": selection_context["bound_eval_path"],
            "runtime_status_path": selection_context["runtime_status_path"],
            "runtime_checkpoint_dir": selection_context["runtime_checkpoint_dir"],
            "resource_snapshot_path": selection_context["resource_snapshot_path"],
            "frontier_solver_backend": str(args.solver_backend),
            "exact_solver_backend": str(args.exact_solver_backend),
        }

        atomic_write_parquet(
            shortlist_eval, output_dir / "portfolio_bound_aware_shortlist.parquet", index=False
        )
        atomic_write_parquet(
            bound_eval, output_dir / "portfolio_bound_aware_bound_eval.parquet", index=False
        )
        atomic_write_json(model_dir / "portfolio_bound_aware_selection.json", payload)

        if gpu_sampler is not None:
            resource_payload["gpu_summary"] = gpu_sampler.stop()
        resource_payload["end"] = _resource_snapshot()
        atomic_write_json(resource_path, resource_payload)
        tracker.complete(
            extra={
                "selection_reason": str(payload["selection_reason"]),
                "selected_alpha01_exact_pass": bool(selected["alpha01_exact_pass"]),
                "selected_realized_total_return": float(selected["realized_total_return"]),
            }
        )

        logger.info(
            "Focused bound-aware search complete: selected risk_tolerance={}, mode={}, gamma={}, q_cap={}, q_tail={}, ab_pass_all={}, alpha01_pass={}",
            selected["risk_tolerance"],
            selected["policy_mode"],
            selected["gamma"],
            selected["delta_cap_quantile"],
            selected["tail_focus_quantile"],
            selected["ab_pass_all"],
            selected["alpha01_exact_pass"],
        )
        logger.info(
            "Saved frontier raw: {}", output_dir / "portfolio_bound_aware_frontier_raw.parquet"
        )
        logger.info(
            "Saved frontier aggregate: {}", output_dir / "portfolio_bound_aware_frontier.parquet"
        )
        logger.info("Saved shortlist: {}", output_dir / "portfolio_bound_aware_shortlist.parquet")
        logger.info(
            "Saved bound evaluations: {}", output_dir / "portfolio_bound_aware_bound_eval.parquet"
        )
        logger.info(
            "Saved selection payload: {}", model_dir / "portfolio_bound_aware_selection.json"
        )
        return 0
    finally:
        if gpu_sampler is not None:
            try:
                if "gpu_summary" not in resource_payload:
                    resource_payload["gpu_summary"] = gpu_sampler.stop()
                    resource_payload["end"] = _resource_snapshot()
                    atomic_write_json(resource_path, resource_payload)
            except Exception:  # pragma: no cover - best effort cleanup only
                pass


if __name__ == "__main__":
    raise SystemExit(main())
