"""Build Paper 4 v27 dynamic scale-up and champion stress artifacts.

V27 extends the v23 128-path replay to a larger common-path evaluation and
turns the strongest v26 challenger into an explicit champion stress test.  The
artifacts remain Paper 4 lab evidence only.
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

SCHEMA_VERSION = "2026-05-14.27"
DEFAULT_CHALLENGER = "v13_cvar_mdcp_colgen_relaxed_k32000_floor105000_cap300000"


def _score01(series: pd.Series, *, high_is_good: bool) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    pct = x.rank(method="average", pct=True, ascending=True)
    out = pct if high_is_good else (1.0 - pct + 1.0 / len(x))
    return out.fillna(0.5)


def _scale_convergence_v27(trace: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_paths = int(trace["path_id"].nunique())
    for n in [32, 64, 128, max_paths]:
        if n > max_paths:
            continue
        local = trace[trace["path_id"].lt(n)]
        final = local.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
        base = final[final["policy_id"].eq(reference)][["path_id", "wealth"]].rename(
            columns={"wealth": "reference_wealth"}
        )
        for policy_id, grp in final.groupby("policy_id"):
            paired = grp.merge(base, on="path_id", how="inner")
            wealth = pd.to_numeric(grp["wealth"], errors="coerce")
            losses = pd.to_numeric(grp["cumulative_losses"], errors="coerce")
            se = (
                float(wealth.std(ddof=1) / np.sqrt(max(len(wealth), 1))) if len(wealth) > 1 else 0.0
            )
            rows.append(
                {
                    "policy_id": policy_id,
                    "n_paths": n,
                    "wealth_mean": float(wealth.mean()),
                    "wealth_p05": float(np.quantile(wealth, 0.05)),
                    "loss_p95": float(np.quantile(losses, 0.95)),
                    "wealth_ci95_width": float(2 * 1.96 * se),
                    "prob_beats_reference": float(
                        (paired["wealth"] > paired["reference_wealth"]).mean()
                    )
                    if not paired.empty
                    else np.nan,
                    "no_temporal_leakage_rate": float(grp["no_temporal_leakage_flag"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _ranking_stability_v27(convergence: pd.DataFrame) -> pd.DataFrame:
    base = _safe_read_csv(TABLE_DIR / "paper4_v23_scale_convergence.csv")
    latest = convergence[convergence["n_paths"].eq(convergence["n_paths"].max())].copy()
    if base.empty or latest.empty:
        return pd.DataFrame()
    base_latest = base[base["n_paths"].eq(base["n_paths"].max())].copy()
    for frame, score_col in [(base_latest, "score_v23"), (latest, "score_v27")]:
        frame[score_col] = (
            0.45 * _score01(frame["wealth_mean"], high_is_good=True)
            + 0.25 * _score01(frame["wealth_p05"], high_is_good=True)
            + 0.20 * _score01(frame["loss_p95"], high_is_good=False)
            + 0.10 * _score01(frame["prob_beats_reference"], high_is_good=True)
        )
    merged = base_latest[["policy_id", "score_v23"]].merge(
        latest[["policy_id", "score_v27"]], on="policy_id", how="inner"
    )
    return pd.DataFrame(
        [
            {
                "comparison": "v23_128_vs_v27_scaled_score",
                "n_common_policies": int(len(merged)),
                "spearman_rank_correlation": float(
                    merged["score_v23"].rank().corr(merged["score_v27"].rank(), method="spearman")
                )
                if len(merged) > 2
                else np.nan,
                "v23_top_policy": str(
                    base_latest.sort_values("score_v23", ascending=False)["policy_id"].iloc[0]
                ),
                "v27_top_policy": str(
                    latest.sort_values("score_v27", ascending=False)["policy_id"].iloc[0]
                ),
                "interpretation": "larger-path ranking stability diagnostic, not automatic promotion",
            }
        ]
    )


def _champion_stress(
    trace: pd.DataFrame,
    summary: pd.DataFrame,
    reference: str,
    challenger: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = trace.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    ref = final[final["policy_id"].eq(reference)][
        ["path_id", "wealth", "cumulative_losses", "ECL", "source_exposure_weak_share"]
    ].rename(
        columns={
            "wealth": "reference_wealth",
            "cumulative_losses": "reference_losses",
            "ECL": "reference_ecl",
            "source_exposure_weak_share": "reference_weak_source_share",
        }
    )
    chal = final[final["policy_id"].eq(challenger)][
        ["path_id", "wealth", "cumulative_losses", "ECL", "source_exposure_weak_share"]
    ].rename(
        columns={
            "wealth": "challenger_wealth",
            "cumulative_losses": "challenger_losses",
            "ECL": "challenger_ecl",
            "source_exposure_weak_share": "challenger_weak_source_share",
        }
    )
    paired = ref.merge(chal, on="path_id", how="inner")
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame()
    paired["wealth_delta_challenger_minus_reference"] = (
        paired["challenger_wealth"] - paired["reference_wealth"]
    )
    paired["loss_delta_challenger_minus_reference"] = (
        paired["challenger_losses"] - paired["reference_losses"]
    )
    paired["ecl_delta_challenger_minus_reference"] = (
        paired["challenger_ecl"] - paired["reference_ecl"]
    )
    paired["weak_source_delta_challenger_minus_reference"] = (
        paired["challenger_weak_source_share"] - paired["reference_weak_source_share"]
    )
    delta = paired["wealth_delta_challenger_minus_reference"]
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    s = summary.set_index("policy_id")
    ref_row = s.loc[reference] if reference in s.index else pd.Series(dtype=float)
    chal_row = s.loc[challenger] if challenger in s.index else pd.Series(dtype=float)
    decision = "retain_working_champion"
    if (
        float(delta.mean()) > 0
        and float((delta > 0).mean()) >= 0.60
        and float(chal_row.get("no_temporal_leakage_rate", 0.0)) >= 1.0
    ):
        decision = "review_champion_replacement_paper4_only"
    memo = pd.DataFrame(
        [
            {
                "reference_policy_id": reference,
                "challenger_policy_id": challenger,
                "n_common_paths": int(len(paired)),
                "mean_wealth_delta": float(delta.mean()),
                "ci95_low_wealth_delta": float(delta.mean() - 1.96 * se),
                "ci95_high_wealth_delta": float(delta.mean() + 1.96 * se),
                "prob_challenger_beats_reference": float((delta > 0).mean()),
                "mean_loss_delta": float(paired["loss_delta_challenger_minus_reference"].mean()),
                "prob_challenger_lower_loss": float(
                    (paired["loss_delta_challenger_minus_reference"] < 0).mean()
                ),
                "mean_ecl_delta": float(paired["ecl_delta_challenger_minus_reference"].mean()),
                "mean_weak_source_delta": float(
                    paired["weak_source_delta_challenger_minus_reference"].mean()
                ),
                "reference_wealth_mean": float(ref_row.get("final_wealth_mean", np.nan)),
                "challenger_wealth_mean": float(chal_row.get("final_wealth_mean", np.nan)),
                "reference_loss_p95": float(ref_row.get("cumulative_losses_p95", np.nan)),
                "challenger_loss_p95": float(chal_row.get("cumulative_losses_p95", np.nan)),
                "claim_safety_gate": bool(
                    float(chal_row.get("no_temporal_leakage_rate", 0.0)) >= 1.0
                ),
                "decision_v27": decision,
                "decision_scope": "Paper 4 working champion only; no Paper Estrella promotion",
            }
        ]
    )
    return paired, memo


def _committee_memos_v27(
    summary: pd.DataFrame, reference: str, pairwise: pd.DataFrame
) -> pd.DataFrame:
    profiles = [
        ("balanced", 0.30, 0.20, 0.20, 0.10, 0.10, 0.10),
        ("wealth_first", 0.55, 0.15, 0.10, 0.05, 0.05, 0.10),
        ("tail_first", 0.15, 0.20, 0.45, 0.05, 0.10, 0.05),
        ("auditability_first", 0.20, 0.15, 0.15, 0.25, 0.10, 0.15),
        ("ecl_aware", 0.25, 0.20, 0.20, 0.05, 0.25, 0.05),
        ("source_governance_aware", 0.25, 0.15, 0.15, 0.30, 0.05, 0.10),
        ("regret_aware", 0.25, 0.15, 0.15, 0.10, 0.05, 0.30),
    ]
    local = summary.merge(
        pairwise[["policy_id", "mean_wealth_diff", "prob_higher_wealth"]],
        on="policy_id",
        how="left",
    )
    local["robustness_gate"] = local["prob_higher_wealth"].fillna(0.0).ge(0.60) & local[
        "mean_wealth_diff"
    ].fillna(-1.0).gt(0.0)
    rows = []
    for name, w_mean, w_p05, w_tail, w_source, w_ecl, w_regret in profiles:
        d = local.copy()
        d["committee_score_v27"] = (
            w_mean * _score01(d["final_wealth_mean"], high_is_good=True)
            + w_p05 * _score01(d["final_wealth_p05"], high_is_good=True)
            + w_tail * _score01(d["cumulative_losses_p95"], high_is_good=False)
            + w_source * _score01(d["source_exposure_weak_share_final_mean"], high_is_good=False)
            + w_ecl * _score01(d["ECL_final_mean"], high_is_good=False)
            + w_regret * _score01(d["mean_wealth_diff"].fillna(0.0), high_is_good=True)
        )
        winner = d.sort_values("committee_score_v27", ascending=False).iloc[0]
        rows.append(
            {
                "committee_profile": name,
                "winning_policy_id": str(winner["policy_id"]),
                "reference_policy_id": reference,
                "committee_score": float(winner["committee_score_v27"]),
                "final_wealth_mean": float(winner["final_wealth_mean"]),
                "loss_p95": float(winner["cumulative_losses_p95"]),
                "prob_higher_wealth_vs_reference": float(winner.get("prob_higher_wealth", np.nan)),
                "robustness_gate_pass": bool(winner.get("robustness_gate", False)),
                "change_recommended": bool(
                    str(winner["policy_id"]) != reference
                    and bool(winner.get("robustness_gate", False))
                ),
                "decision_scope": "Paper 4 committee memo only; no final promotion",
                "consolidated_decision": "review_challenger"
                if str(winner["policy_id"]) != reference
                else "retain_working_champion",
            }
        )
    return pd.DataFrame(rows)


def _performance_report_v27(runtime: float, trace: pd.DataFrame, paths: int) -> pd.DataFrame:
    v23_status = _safe_read_json(STATUS_DIR / "paper4_v23_status.json")
    return pd.DataFrame(
        [
            {
                "run_id": "v23_128",
                "n_paths": int(v23_status.get("dynamic_path_count_v23", 128)),
                "trace_rows": int(v23_status.get("dynamic_trace_rows_v23", 0)),
                "runtime_seconds": float(v23_status.get("runtime_seconds", np.nan)),
                "trace_parquet_mb": (TABLE_DIR / "paper4_v23_dynamic_policy_trace.parquet")
                .stat()
                .st_size
                / 1e6
                if (TABLE_DIR / "paper4_v23_dynamic_policy_trace.parquet").exists()
                else np.nan,
                "bottleneck": "policy-path-month loan event replay",
            },
            {
                "run_id": "v27_scaled",
                "n_paths": int(paths),
                "trace_rows": int(len(trace)),
                "runtime_seconds": float(runtime),
                "trace_parquet_mb": (TABLE_DIR / "paper4_v27_dynamic_policy_trace.parquet")
                .stat()
                .st_size
                / 1e6
                if (TABLE_DIR / "paper4_v27_dynamic_policy_trace.parquet").exists()
                else np.nan,
                "bottleneck": "same replay; event cache remains next optimization",
            },
        ]
    )


def build_v27(
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
    sample_paths["sample_path_version_v27"] = "v4_internal_scaled_common_random_numbers"
    path_v4_summary["version_v27"] = "sample_path_v4_scaled_diagnostics"
    trace, summary, _ = v15.build_dynamic_engine_v15(books, sample_paths, n_paths=paths)
    runtime = time.time() - start

    reference = str(
        _safe_read_json(STATUS_DIR / "paper4_v26_working_champion.json").get(
            "policy_id", "paper1_economic_champion"
        )
    )
    convergence = _scale_convergence_v27(trace, reference)
    ranking = _ranking_stability_v27(convergence)
    pairwise = v19._path_pairwise_ci(trace, reference)
    horizons = v19._summarize_horizons(trace, [12, 24, 36])
    family = v19._path_family_sensitivity(trace)
    paired, stress = _champion_stress(trace, summary, reference, challenger)
    committees = _committee_memos_v27(summary, reference, pairwise)

    summary["version_v27"] = "dynamic_scale_up_256_or_more"
    summary["claim_scope_v27"] = "larger internal replay; no forecast or production claim"
    trace["version_v27"] = "dynamic_scale_up_256_or_more"
    registry["version_v27"] = "adapter_reuse_for_scaled_replay"
    macro_source["version_v27"] = "external_macro_context_retry"
    path_design["version_v27"] = "sample_paths_v4"
    path_diag["version_v27"] = "sample_paths_v4_internal"

    _write_csv("paper4_v27_policy_adapter_registry.csv", registry)
    _write_csv("paper4_v27_external_macro_source_registry.csv", macro_source)
    if not macro.empty:
        _write_csv("paper4_v27_external_macro_context.csv", macro)
    _write_csv("paper4_v27_observed_calibration_by_month_grade.csv", calibration)
    _write_csv("paper4_v27_sample_path_design.csv", path_design)
    _write_csv("paper4_v27_path_regime_register.csv", scenarios)
    _write_parquet("paper4_v27_sample_paths.parquet", sample_paths)
    _write_csv("paper4_v27_path_calibration_diagnostics.csv", path_v4_summary)
    _write_parquet("paper4_v27_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v27_dynamic_policy_summary.csv", summary)
    _write_csv("paper4_v27_scale_convergence.csv", convergence)
    _write_csv("paper4_v27_ranking_stability.csv", ranking)
    _write_csv("paper4_v27_policy_pairwise_common_path_ci.csv", pairwise)
    _write_csv("paper4_v27_horizon_sensitivity.csv", horizons)
    _write_csv("paper4_v27_path_family_sensitivity.csv", family)
    _write_parquet("paper4_v27_champion_vs_cvar_paired_paths.parquet", paired)
    _write_csv("paper4_v27_champion_vs_cvar_stress_memo.csv", stress)
    _write_csv("paper4_v27_committee_profile_memos.csv", committees)
    performance = _performance_report_v27(runtime, trace, paths)
    _write_csv("paper4_v27_performance_report.csv", performance)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v27_dynamic_scale_champion_stress",
        "dynamic_policy_count_v27": int(summary["policy_id"].nunique()),
        "dynamic_path_count_v27": int(paths),
        "horizon_months_v27": int(horizon_months),
        "dynamic_trace_rows_v27": int(len(trace)),
        "no_temporal_leakage_min_rate_v27": float(summary["no_temporal_leakage_rate"].min()),
        "reference_policy_id_v27": reference,
        "direct_challenger_policy_id_v27": challenger,
        "direct_challenger_decision_v27": str(stress["decision_v27"].iloc[0])
        if not stress.empty
        else "not_evaluated",
        "external_macro_context_status_v27": str(
            macro_source.iloc[0].get("fetch_status", "not_attempted")
        ),
        "ranking_stability_rows_v27": int(len(ranking)),
        "committee_profiles_v27": int(len(committees)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "256-path internal replay and champion stress; not external forecast or final promotion",
        "runtime_seconds": round(runtime, 3),
    }
    _write_json("paper4_v27_status.json", status)
    _write_note(
        "paper4_v27_dynamic_scale_champion_stress.md",
        "\n".join(
            [
                "# Paper 4 v27 Dynamic Scale and Champion Stress",
                "",
                f"- Paths: {paths}",
                f"- Horizon months: {horizon_months}",
                f"- Reference: `{reference}`",
                f"- Direct challenger: `{challenger}`",
                f"- Decision: `{status['direct_challenger_decision_v27']}`",
                "- Claim boundary: internal replay only; no Paper Estrella change.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    parser.add_argument("--challenger", type=str, default=DEFAULT_CHALLENGER)
    args = parser.parse_args()
    build_v27(args.paths, args.horizon_months, args.solver_pool_n, args.challenger)


if __name__ == "__main__":
    main()
