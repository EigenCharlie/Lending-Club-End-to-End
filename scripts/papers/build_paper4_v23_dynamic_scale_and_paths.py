"""Build Paper 4 v23 dynamic scale-up and sample-path v3 artifacts."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
import scripts.papers.build_paper4_v19_dynamic_engine_v2 as v19
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

SCHEMA_VERSION = "2026-05-14.23"


def _load_macro_events() -> pd.DataFrame:
    path = Path("data/processed/macro_context.json")
    if not path.exists():
        return pd.DataFrame()
    try:
        events = pd.read_json(path)
    except Exception:
        return pd.DataFrame()
    if events.empty or "date" not in events:
        return pd.DataFrame()
    events["month"] = (
        pd.to_datetime(events["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    events["macro_event_label_v23"] = events["type"].astype(str) + ":" + events["event"].astype(str)
    return events[["month", "macro_event_label_v23", "type", "impact"]].dropna(subset=["month"])


def _observed_calibration(books: pd.DataFrame) -> pd.DataFrame:
    local = books.copy()
    local["issue_month"] = (
        pd.to_datetime(local["issue_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    local["prepayment_proxy"] = (
        local.get("loan_status", pd.Series("", index=local.index))
        .astype(str)
        .str.contains("Fully Paid", case=False, na=False)
        if "loan_status" in local
        else False
    )
    return (
        local.groupby(["issue_month", "original_grade"], as_index=False)
        .agg(
            observed_default_rate=("y_true", "mean"),
            observed_lgd_proxy=("lgd", "mean"),
            observed_prepayment_proxy=("prepayment_proxy", "mean"),
            loans=("loan_id", "nunique"),
            avg_pd_high=("pd_high_alpha01", "mean"),
            avg_width=("qhat_v4", "mean"),
        )
        .rename(columns={"issue_month": "month"})
        .assign(calibration_scope_v23="observed static book calibration proxy, not servicing panel")
    )


def _path_v3_annotations(
    paths: pd.DataFrame, calibration: pd.DataFrame, macro_events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = paths.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if not macro_events.empty:
        out = out.merge(macro_events, on="month", how="left")
        out["macro_event_label_v23"] = out["macro_event_label_v23"].fillna("none")
    else:
        out["macro_event_label_v23"] = "none"
    default_mean = (
        float(calibration["observed_default_rate"].mean()) if not calibration.empty else np.nan
    )
    lgd_mean = float(calibration["observed_lgd_proxy"].mean()) if not calibration.empty else np.nan
    out["default_factor_centered_v23"] = out["default_factor_v15"] / max(
        out["default_factor_v15"].mean(), 1e-9
    )
    out["lgd_factor_centered_v23"] = out["lgd_factor_v15"] / max(out["lgd_factor_v15"].mean(), 1e-9)
    out["observed_default_anchor_v23"] = default_mean
    out["observed_lgd_anchor_v23"] = lgd_mean
    out["sample_path_version_v23"] = "v3_internal_with_macro_event_context"
    out["claim_scope_v23"] = (
        "internal calibration with official/local macro context labels; not forecast validation"
    )
    summary = (
        out.groupby(["macro_regime_v15", "macro_event_label_v23"], as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            n_months=("month", "nunique"),
            default_factor_mean=("default_factor_v15", "mean"),
            default_factor_p95=("default_factor_v15", lambda s: float(np.quantile(s, 0.95))),
            lgd_factor_mean=("lgd_factor_v15", "mean"),
            prepay_factor_mean=("prepay_factor_v15", "mean"),
            observed_default_anchor_v23=("observed_default_anchor_v23", "mean"),
            observed_lgd_anchor_v23=("observed_lgd_anchor_v23", "mean"),
        )
        .assign(path_claim_boundary="internal paths; macro labels are contextual")
    )
    return out, summary


def _scale_convergence(
    trace: pd.DataFrame, reference: str, baseline_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    path_counts = sorted({32, 64, int(trace["path_id"].nunique())})
    for n in path_counts:
        local = trace[trace["path_id"].lt(n)]
        final = local.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
        base = final[final["policy_id"].eq(reference)][["path_id", "wealth"]].rename(
            columns={"wealth": "reference_wealth"}
        )
        for policy_id, grp in final.groupby("policy_id"):
            diff = grp.merge(base, on="path_id", how="inner")
            wealth = grp["wealth"]
            mean = float(wealth.mean())
            se = (
                float(wealth.std(ddof=1) / np.sqrt(max(len(wealth), 1))) if len(wealth) > 1 else 0.0
            )
            rows.append(
                {
                    "policy_id": policy_id,
                    "n_paths": n,
                    "wealth_mean": mean,
                    "wealth_p05": float(np.quantile(wealth, 0.05)),
                    "loss_p95": float(np.quantile(grp["cumulative_losses"], 0.95)),
                    "wealth_ci95_width": float(2 * 1.96 * se),
                    "prob_beats_reference": float(
                        (diff["wealth"] > diff["reference_wealth"]).mean()
                    )
                    if not diff.empty
                    else np.nan,
                    "no_temporal_leakage_rate": float(grp["no_temporal_leakage_flag"].mean()),
                }
            )
    convergence = pd.DataFrame(rows)
    ranks = []
    if not baseline_summary.empty:
        baseline_rank = baseline_summary[["policy_id", "paper4_champion_score_v15"]].rename(
            columns={"paper4_champion_score_v15": "score_64_v19"}
        )
        latest = convergence[convergence["n_paths"].eq(max(path_counts))].assign(
            score_proxy=lambda d: (
                d["wealth_mean"].rank(pct=True) - d["loss_p95"].rank(pct=True) * 0.35
            )
        )[["policy_id", "score_proxy"]]
        merged = baseline_rank.merge(latest, on="policy_id", how="inner")
        spearman = (
            float(
                merged["score_64_v19"].rank().corr(merged["score_proxy"].rank(), method="spearman")
            )
            if len(merged) > 2
            else np.nan
        )
        ranks.append(
            {
                "comparison": "v19_64_score_vs_v23_scaled_proxy",
                "n_common_policies": int(len(merged)),
                "spearman_rank_correlation": spearman,
                "baseline_top_policy": str(
                    baseline_summary.sort_values("paper4_champion_score_v15", ascending=False)[
                        "policy_id"
                    ].iloc[0]
                ),
                "scaled_top_policy": str(
                    latest.sort_values("score_proxy", ascending=False)["policy_id"].iloc[0]
                )
                if not latest.empty
                else "",
                "interpretation": "ranking stability diagnostic, not promotion rule",
            }
        )
    return convergence, pd.DataFrame(ranks)


def _performance_report(runtime: float, trace: pd.DataFrame, paths: int) -> pd.DataFrame:
    v19_status = _safe_read_json(STATUS_DIR / "paper4_v19_status.json")
    baseline_runtime = float(v19_status.get("runtime_seconds", np.nan))
    baseline_rows = float(v19_status.get("dynamic_trace_rows_v19", np.nan))
    rows = [
        {
            "run_id": "v19_baseline_64",
            "n_paths": int(v19_status.get("dynamic_path_count_v19", 64)),
            "trace_rows": int(v19_status.get("dynamic_trace_rows_v19", 0)),
            "runtime_seconds": baseline_runtime,
            "rows_per_second": baseline_rows / baseline_runtime
            if baseline_runtime and baseline_runtime > 0
            else np.nan,
            "memory_mb_trace_parquet": (TABLE_DIR / "paper4_v19_dynamic_policy_trace.parquet")
            .stat()
            .st_size
            / 1e6
            if (TABLE_DIR / "paper4_v19_dynamic_policy_trace.parquet").exists()
            else np.nan,
            "bottleneck": "policy-path-month simulation loops",
        },
        {
            "run_id": "v23_scaled",
            "n_paths": int(paths),
            "trace_rows": int(len(trace)),
            "runtime_seconds": float(runtime),
            "rows_per_second": float(len(trace) / runtime) if runtime > 0 else np.nan,
            "memory_mb_trace_parquet": np.nan,
            "bottleneck": "loan event replay; cache loan/path events before 256/512",
        },
    ]
    return pd.DataFrame(rows)


def build_v23(paths: int, horizon_months: int, solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    v15.MONTHLY_REPAYMENT_HORIZON = horizon_months
    _, books, registry = v19._load_books(solver_pool_n)
    macro_source, macro = v19._external_macro_context()
    calibration = _observed_calibration(books)
    macro_events = _load_macro_events()
    path_design, scenarios, raw_paths, path_diag = v19._build_sample_paths_v19(
        books, n_paths=paths, horizon_months=horizon_months, macro=macro
    )
    sample_paths, path_v3_summary = _path_v3_annotations(raw_paths, calibration, macro_events)
    trace, summary, _ = v15.build_dynamic_engine_v15(books, sample_paths, n_paths=paths)
    runtime = time.time() - start

    reference = str(
        _safe_read_json(STATUS_DIR / "paper4_v22_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    baseline_summary = _safe_read_csv(TABLE_DIR / "paper4_v19_dynamic_policy_summary.csv")
    convergence, ranking = _scale_convergence(trace, reference, baseline_summary)
    pairwise = v19._path_pairwise_ci(trace, reference)
    horizons = v19._summarize_horizons(trace, [12, 24, 36])
    family = v19._path_family_sensitivity(trace)
    performance = _performance_report(runtime, trace, paths)
    performance.loc[performance["run_id"].eq("v23_scaled"), "memory_mb_trace_parquet"] = 0.0

    summary = summary.copy()
    summary["version_v23"] = "dynamic_scale_up_128" if paths >= 128 else "dynamic_scale_up"
    summary["claim_scope_v23"] = "larger internal replay, not forecast"
    trace["version_v23"] = "dynamic_scale_up"
    registry = registry.copy()
    registry["version_v23"] = "adapter_reuse_for_scaled_replay"
    macro_source["version_v23"] = "sample_paths_v3_macro_attempt"
    path_design["version_v23"] = "sample_paths_v3"
    calibration["version_v23"] = "observed_static_book_calibration"
    path_v3_summary["version_v23"] = "sample_path_v3_diagnostics"

    _write_csv("paper4_v23_policy_adapter_registry.csv", registry)
    _write_csv("paper4_v23_external_macro_source_registry.csv", macro_source)
    if not macro.empty:
        _write_csv("paper4_v23_external_macro_context.csv", macro)
    _write_csv("paper4_v23_observed_calibration_by_month_grade.csv", calibration)
    _write_csv("paper4_v23_sample_path_design.csv", path_design)
    _write_csv("paper4_v23_path_regime_register.csv", scenarios)
    _write_parquet("paper4_v23_sample_paths.parquet", sample_paths)
    _write_csv("paper4_v23_path_calibration_diagnostics.csv", path_v3_summary)
    _write_parquet("paper4_v23_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v23_dynamic_policy_summary.csv", summary)
    _write_csv("paper4_v23_scale_convergence.csv", convergence)
    _write_csv("paper4_v23_ranking_stability.csv", ranking)
    _write_csv("paper4_v23_policy_pairwise_common_path_ci.csv", pairwise)
    _write_csv("paper4_v23_horizon_sensitivity.csv", horizons)
    _write_csv("paper4_v23_path_family_sensitivity.csv", family)
    _write_csv("paper4_v23_performance_report.csv", performance)

    # Update trace parquet size after writing.
    perf = pd.read_csv(TABLE_DIR / "paper4_v23_performance_report.csv")
    perf.loc[perf["run_id"].eq("v23_scaled"), "memory_mb_trace_parquet"] = (
        TABLE_DIR / "paper4_v23_dynamic_policy_trace.parquet"
    ).stat().st_size / 1e6
    _write_csv("paper4_v23_performance_report.csv", perf)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v23_dynamic_scale_and_paths",
        "dynamic_policy_count_v23": int(summary["policy_id"].nunique()),
        "dynamic_path_count_v23": int(paths),
        "horizon_months_v23": int(horizon_months),
        "dynamic_trace_rows_v23": int(len(trace)),
        "no_temporal_leakage_min_rate_v23": float(summary["no_temporal_leakage_rate"].min()),
        "reference_policy_id_v23": reference,
        "external_macro_context_status_v23": str(
            macro_source.iloc[0].get("fetch_status", "not_attempted")
        ),
        "ranking_stability_rows_v23": int(len(ranking)),
        "performance_report_rows_v23": int(len(perf)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "128-path internal replay and sample paths v3; no external forecast claim",
        "runtime_seconds": round(runtime, 3),
    }
    _write_json("paper4_v23_status.json", status)
    _write_note(
        "paper4_v23_dynamic_scale_and_paths.md",
        "\n".join(
            [
                "# Paper 4 v23 Dynamic Scale and Paths",
                "",
                f"- Policies: `{status['dynamic_policy_count_v23']}`.",
                f"- Paths: `{status['dynamic_path_count_v23']}`.",
                f"- Horizon months: `{status['horizon_months_v23']}`.",
                f"- No-leakage min rate: `{status['no_temporal_leakage_min_rate_v23']}`.",
                f"- Macro context status: `{status['external_macro_context_status_v23']}`.",
                "",
                "This wave scales dynamic replay and keeps sample paths as internal calibration.",
            ]
        ),
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_v23(args.paths, args.horizon_months, args.solver_pool_n)


if __name__ == "__main__":
    main()
