"""Build Paper 4 v31 dynamic 512-path stress artifacts.

V31 is intentionally a continuation of the v27 dynamic stress engine.  It
either runs the larger common-path replay, or records a performance-blocked
alternative when the estimated runtime exceeds the caller's guardrail.  The
default guardrail is permissive enough for a 512-path run on the current local
machine, but the artifact always records whether the result is an actual replay
or a documented fallback.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
import scripts.papers.build_paper4_v19_dynamic_engine_v2 as v19
import scripts.papers.build_paper4_v23_dynamic_scale_and_paths as v23
import scripts.papers.build_paper4_v27_dynamic_scale_champion_stress as v27
from scripts.papers.build_paper4_extended_experiments import _safe_read_csv, _safe_read_json
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-15.31"
DEFAULT_CHALLENGER = "v13_cvar_mdcp_colgen_relaxed_k32000_floor105000_cap300000"


def _estimate_runtime(paths: int) -> tuple[float, str]:
    status = _safe_read_json(STATUS_DIR / "paper4_v27_status.json")
    base_paths = float(status.get("dynamic_path_count_v27", 256) or 256)
    base_runtime = float(status.get("runtime_seconds", np.nan))
    if not np.isfinite(base_runtime) or base_runtime <= 0:
        return float(paths * 6.0), "fallback_seconds_per_path"
    return float(base_runtime * paths / base_paths), "linear_from_v27"


def _copy_v27_as_performance_guard(
    paths: int, estimated_runtime: float, guard: float, challenger: str
) -> dict[str, Any]:
    trace = pd.read_parquet(TABLE_DIR / "paper4_v27_dynamic_policy_trace.parquet")
    summary = _safe_read_csv(TABLE_DIR / "paper4_v27_dynamic_policy_summary.csv")
    convergence = _safe_read_csv(TABLE_DIR / "paper4_v27_scale_convergence.csv")
    pairwise = _safe_read_csv(TABLE_DIR / "paper4_v27_policy_pairwise_common_path_ci.csv")
    stress = _safe_read_csv(TABLE_DIR / "paper4_v27_champion_vs_cvar_stress_memo.csv")
    performance = _safe_read_csv(TABLE_DIR / "paper4_v27_performance_report.csv")

    for frame in [summary, convergence, pairwise, stress, performance]:
        if not frame.empty:
            frame["version_v31"] = "performance_guard_reuse_v27_256_path_baseline"
            frame["requested_paths_v31"] = paths
            frame["actual_paths_v31"] = int(trace["path_id"].nunique()) if "path_id" in trace else 0
            frame["claim_boundary_v31"] = (
                "512-path replay performance-guarded; v27 256-path baseline reused"
            )
    trace = trace.copy()
    trace["version_v31"] = "performance_guard_reuse_v27_256_path_baseline"
    trace["requested_paths_v31"] = paths

    feasibility = pd.DataFrame(
        [
            {
                "requested_paths_v31": int(paths),
                "actual_paths_v31": int(trace["path_id"].nunique()),
                "estimated_runtime_seconds": float(estimated_runtime),
                "runtime_guard_seconds": float(guard),
                "execution_mode_v31": "performance_guard_reused_v27",
                "performance_blocked": True,
                "alternative_used": "v27_256_path_common_path_replay_plus_convergence_diagnostics",
                "claim_boundary": "not a 512-path result; documented fallback only",
            }
        ]
    )
    _write_parquet("paper4_v31_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v31_dynamic_policy_summary.csv", summary)
    _write_csv("paper4_v31_scale_convergence.csv", convergence)
    _write_csv("paper4_v31_policy_pairwise_common_path_ci.csv", pairwise)
    _write_csv("paper4_v31_champion_vs_cvar_stress_memo.csv", stress)
    _write_csv("paper4_v31_performance_report.csv", performance)
    _write_csv("paper4_v31_512_feasibility_guard.csv", feasibility)
    _write_csv(
        "paper4_v31_ranking_stability.csv",
        _safe_read_csv(TABLE_DIR / "paper4_v27_ranking_stability.csv"),
    )
    _write_csv(
        "paper4_v31_horizon_sensitivity.csv",
        _safe_read_csv(TABLE_DIR / "paper4_v27_horizon_sensitivity.csv"),
    )
    _write_csv(
        "paper4_v31_path_family_sensitivity.csv",
        _safe_read_csv(TABLE_DIR / "paper4_v27_path_family_sensitivity.csv"),
    )

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v31_dynamic_512_stress",
        "execution_mode_v31": "performance_guard_reused_v27",
        "requested_path_count_v31": int(paths),
        "dynamic_path_count_v31": int(trace["path_id"].nunique()),
        "dynamic_trace_rows_v31": int(len(trace)),
        "no_temporal_leakage_min_rate_v31": float(summary["no_temporal_leakage_rate"].min())
        if "no_temporal_leakage_rate" in summary
        else np.nan,
        "reference_policy_id_v31": str(
            _safe_read_json(STATUS_DIR / "paper4_v30_working_champion.json").get(
                "policy_id", "paper1_economic_champion"
            )
        ),
        "direct_challenger_policy_id_v31": challenger,
        "direct_challenger_decision_v31": str(stress["decision_v27"].iloc[0])
        if not stress.empty and "decision_v27" in stress
        else "not_evaluated",
        "estimated_runtime_seconds_v31": float(estimated_runtime),
        "runtime_guard_seconds_v31": float(guard),
        "performance_blocked_v31": True,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "performance-guarded fallback; no 512-path claim and no promotion",
    }
    _write_json("paper4_v31_status.json", status)
    _write_note(
        "paper4_v31_dynamic_512_stress.md",
        "\n".join(
            [
                "# Paper 4 v31 Dynamic 512 Stress",
                "",
                "- Execution mode: performance guard reused the v27 256-path replay.",
                f"- Requested paths: `{paths}`.",
                f"- Estimated runtime seconds: `{estimated_runtime:.1f}`.",
                "- Claim boundary: this is not a 512-path result.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def _run_replay(
    paths: int, horizon_months: int, solver_pool_n: int, challenger: str
) -> dict[str, Any]:
    start = time.time()
    v15.MONTHLY_REPAYMENT_HORIZON = horizon_months
    _, books, registry = v19._load_books(solver_pool_n)
    macro_source, macro = v19._external_macro_context()
    calibration = v23._observed_calibration(books)
    macro_events = v23._load_macro_events()
    path_design, scenarios, raw_paths, path_diag = v19._build_sample_paths_v19(
        books, n_paths=paths, horizon_months=horizon_months, macro=macro
    )
    sample_paths, path_v4_summary = v23._path_v3_annotations(raw_paths, calibration, macro_events)
    sample_paths["sample_path_version_v31"] = "v31_scaled_512_candidate_common_random_numbers"
    path_v4_summary["version_v31"] = "sample_path_scaled_512_candidate_diagnostics"

    trace, summary, _ = v15.build_dynamic_engine_v15(books, sample_paths, n_paths=paths)
    runtime = time.time() - start
    reference = str(
        _safe_read_json(STATUS_DIR / "paper4_v30_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    convergence = v27._scale_convergence_v27(trace, reference)
    ranking = v27._ranking_stability_v27(convergence)
    pairwise = v19._path_pairwise_ci(trace, reference)
    horizons = v19._summarize_horizons(trace, [12, 24, 36])
    family = v19._path_family_sensitivity(trace)
    paired, stress = v27._champion_stress(trace, summary, reference, challenger)
    if not stress.empty and "decision_v27" in stress:
        stress["decision_v31"] = stress["decision_v27"]
    committees = v27._committee_memos_v27(summary, reference, pairwise)

    summary["version_v31"] = "dynamic_512_path_scale_candidate"
    summary["claim_scope_v31"] = "larger internal replay; no forecast or production claim"
    trace["version_v31"] = "dynamic_512_path_scale_candidate"
    registry["version_v31"] = "adapter_reuse_for_v31_scaled_replay"
    macro_source["version_v31"] = "external_macro_context_retry_v31"
    path_design["version_v31"] = "sample_paths_v31"
    path_diag["version_v31"] = "sample_paths_v31_internal"

    _write_csv("paper4_v31_policy_adapter_registry.csv", registry)
    _write_csv("paper4_v31_external_macro_source_registry.csv", macro_source)
    if not macro.empty:
        _write_csv("paper4_v31_external_macro_context.csv", macro)
    _write_csv("paper4_v31_observed_calibration_by_month_grade.csv", calibration)
    _write_csv("paper4_v31_sample_path_design.csv", path_design)
    _write_csv("paper4_v31_path_regime_register.csv", scenarios)
    _write_parquet("paper4_v31_sample_paths.parquet", sample_paths)
    _write_csv("paper4_v31_path_calibration_diagnostics.csv", path_v4_summary)
    _write_parquet("paper4_v31_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v31_dynamic_policy_summary.csv", summary)
    _write_csv("paper4_v31_scale_convergence.csv", convergence)
    _write_csv("paper4_v31_ranking_stability.csv", ranking)
    _write_csv("paper4_v31_policy_pairwise_common_path_ci.csv", pairwise)
    _write_csv("paper4_v31_horizon_sensitivity.csv", horizons)
    _write_csv("paper4_v31_path_family_sensitivity.csv", family)
    _write_parquet("paper4_v31_champion_vs_cvar_paired_paths.parquet", paired)
    _write_csv("paper4_v31_champion_vs_cvar_stress_memo.csv", stress)
    _write_csv("paper4_v31_committee_profile_memos.csv", committees)
    performance = pd.DataFrame(
        [
            {
                "run_id": "v27_256",
                "n_paths": int(
                    _safe_read_json(STATUS_DIR / "paper4_v27_status.json").get(
                        "dynamic_path_count_v27", 256
                    )
                ),
                "trace_rows": int(
                    _safe_read_json(STATUS_DIR / "paper4_v27_status.json").get(
                        "dynamic_trace_rows_v27", 0
                    )
                ),
                "runtime_seconds": float(
                    _safe_read_json(STATUS_DIR / "paper4_v27_status.json").get(
                        "runtime_seconds", np.nan
                    )
                ),
                "bottleneck": "policy-path-month loan event replay",
            },
            {
                "run_id": "v31_scaled",
                "n_paths": int(paths),
                "trace_rows": int(len(trace)),
                "runtime_seconds": float(runtime),
                "bottleneck": "same replay; larger common random number panel",
            },
        ]
    )
    _write_csv("paper4_v31_performance_report.csv", performance)
    _write_csv(
        "paper4_v31_512_feasibility_guard.csv",
        pd.DataFrame(
            [
                {
                    "requested_paths_v31": int(paths),
                    "actual_paths_v31": int(paths),
                    "estimated_runtime_seconds": np.nan,
                    "runtime_guard_seconds": np.nan,
                    "execution_mode_v31": "actual_replay",
                    "performance_blocked": False,
                    "alternative_used": "",
                    "claim_boundary": "actual internal replay; no external forecast claim",
                }
            ]
        ),
    )

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v31_dynamic_512_stress",
        "execution_mode_v31": "actual_replay",
        "requested_path_count_v31": int(paths),
        "dynamic_path_count_v31": int(paths),
        "horizon_months_v31": int(horizon_months),
        "dynamic_trace_rows_v31": int(len(trace)),
        "no_temporal_leakage_min_rate_v31": float(summary["no_temporal_leakage_rate"].min()),
        "reference_policy_id_v31": reference,
        "direct_challenger_policy_id_v31": challenger,
        "direct_challenger_decision_v31": str(stress["decision_v27"].iloc[0])
        if not stress.empty
        else "not_evaluated",
        "external_macro_context_status_v31": str(
            macro_source.iloc[0].get("fetch_status", "not_attempted")
        ),
        "performance_blocked_v31": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "512-path internal replay and champion stress; not external forecast or final promotion",
        "runtime_seconds": round(runtime, 3),
    }
    _write_json("paper4_v31_status.json", status)
    _write_note(
        "paper4_v31_dynamic_512_stress.md",
        "\n".join(
            [
                "# Paper 4 v31 Dynamic 512 Stress",
                "",
                f"- Paths: `{paths}`.",
                f"- Horizon months: `{horizon_months}`.",
                f"- Reference: `{reference}`.",
                f"- Direct challenger: `{challenger}`.",
                f"- Decision: `{status['direct_challenger_decision_v31']}`.",
                "- Claim boundary: internal replay only; no Paper Estrella change.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def build_v31(
    paths: int,
    horizon_months: int,
    solver_pool_n: int,
    challenger: str,
    max_estimated_runtime_seconds: float,
    force: bool,
) -> dict[str, Any]:
    estimated_runtime, estimate_method = _estimate_runtime(paths)
    guard = float(max_estimated_runtime_seconds)
    if not force and estimated_runtime > guard:
        status = _copy_v27_as_performance_guard(paths, estimated_runtime, guard, challenger)
        status["runtime_estimate_method_v31"] = estimate_method
        _write_json("paper4_v31_status.json", status)
        return status
    return _run_replay(paths, horizon_months, solver_pool_n, challenger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    parser.add_argument("--challenger", type=str, default=DEFAULT_CHALLENGER)
    parser.add_argument("--max-estimated-runtime-seconds", type=float, default=3600.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_v31(
        args.paths,
        args.horizon_months,
        args.solver_pool_n,
        args.challenger,
        args.max_estimated_runtime_seconds,
        args.force,
    )


if __name__ == "__main__":
    main()
