"""Build Paper 4 v19 dynamic engine v2 artifacts.

This wave expands the v15 monthly replay into a larger, horizon-aware dynamic
stress engine.  It remains a Paper 4 lab artifact:

* no Paper Estrella artifact is modified;
* no final Paper 4 promotion JSON is created;
* sample paths are internal replay/calibration paths, not forecasts.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import scripts.papers.build_paper4_v15_dynamic_stress_engine as v15
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    _load_inputs,
    _prepare_solver_pool,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.19"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE,FEDFUNDS,DROCLACBS,USREC"


def _load_books(solver_pool_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_universe, candidate_pool, _, _, online_intervals = _load_inputs()
    solver_source = base_universe if len(base_universe) > len(candidate_pool) else candidate_pool
    solver_pool = _prepare_solver_pool(
        solver_source,
        online_intervals,
        max_n=min(solver_pool_n, len(solver_source)),
    )
    books, adapter_registry = v15._load_policy_books(solver_pool)
    if books.empty:
        raise RuntimeError("No Paper 4 policy books were available for v19.")
    return solver_pool, books, adapter_registry


def _dynamic_state_schema_v19() -> dict[str, Any]:
    schema = v15.build_dynamic_state_schema()
    schema["schema_version"] = SCHEMA_VERSION
    schema["v19_extension"] = {
        "horizons_months": [12, 24, 36],
        "path_scale_targets": [64, 128, 256, 512],
        "engine_role": "monthly policy process comparison under common internal paths",
        "no_leakage_rule": "monthly decisions may only fund loans whose issue_month is not after the replay month",
        "claim_boundary": "larger internal stress replay, not production deployment and not external forecast",
    }
    return schema


def _external_macro_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = [
        {
            "source_id": "FRED_UNRATE_FEDFUNDS_DROCLACBS_USREC",
            "source_url": FRED_URL,
            "source_owner": "Federal Reserve Bank of St. Louis FRED",
            "use_in_v19": "optional_context_only",
            "claim_boundary": "official macro context can label stress regimes; it does not turn internal paths into forecasts",
        }
    ]
    try:
        macro = pd.read_csv(FRED_URL)
        macro = macro.rename(columns={"observation_date": "month"})
        macro["month"] = (
            pd.to_datetime(macro["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        )
        for col in ["UNRATE", "FEDFUNDS", "DROCLACBS", "USREC"]:
            if col in macro:
                macro[col] = pd.to_numeric(macro[col], errors="coerce")
        macro = macro.sort_values("month").ffill()
        rows[0]["fetch_status"] = "fetched"
        rows[0]["rows_fetched"] = int(len(macro))
    except Exception as exc:  # pragma: no cover - network dependent
        macro = pd.DataFrame(columns=["month", "UNRATE", "FEDFUNDS", "DROCLACBS", "USREC"])
        rows[0]["fetch_status"] = "fetch_failed"
        rows[0]["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        rows[0]["rows_fetched"] = 0
    return pd.DataFrame(rows), macro


def _build_sample_paths_v19(
    books: pd.DataFrame,
    *,
    n_paths: int,
    horizon_months: int,
    macro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v15.MONTHLY_REPAYMENT_HORIZON = int(horizon_months)
    design, scenarios, paths = v15.build_sample_paths_v15(books, n_paths=n_paths)
    design = design.copy()
    design["version_v19"] = "sample_paths_calibrated_v2"
    design["external_macro_use_v19"] = "context_join_if_available"
    design.loc[len(design)] = {
        "design_element": "official macro context",
        "data_anchor": "FRED UNRATE/FEDFUNDS/DROCLACBS/USREC",
        "implementation_v15": "v19 joins official macro labels where available; internal shocks remain primary",
        "overclaim_guardrail": "external macro context is descriptive, not forecast validation",
        "version_v19": "sample_paths_calibrated_v2",
        "external_macro_use_v19": "context_only",
    }

    paths = paths.copy()
    paths["path_family_v19"] = paths["macro_regime_v15"].astype(str)
    paths["default_dependence_v19"] = np.where(
        paths["systemic_factor_v15"].abs().ge(1.0), "high_common_factor", "moderate_common_factor"
    )
    paths["cyclic_lgd_v19"] = (
        paths["lgd_factor_v15"].sub(paths["lgd_factor_v15"].median()).fillna(0.0)
    )
    paths["claim_scope_v19"] = "internal_calibration_common_random_numbers_not_forecast"
    if not macro.empty:
        macro_keep = macro[
            [
                "month",
                *[col for col in ["UNRATE", "FEDFUNDS", "DROCLACBS", "USREC"] if col in macro],
            ]
        ].copy()
        paths = paths.merge(macro_keep, on="month", how="left").sort_values(["path_id", "month"])
        for col in ["UNRATE", "FEDFUNDS", "DROCLACBS", "USREC"]:
            if col in paths:
                paths[col] = paths.groupby("path_id")[col].ffill().bfill()
        paths["external_macro_joined_v19"] = True
    else:
        paths["external_macro_joined_v19"] = False

    diagnostics = (
        paths.groupby(["macro_regime_v15", "path_family_v19"], as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            n_months=("month", "nunique"),
            default_factor_mean=("default_factor_v15", "mean"),
            default_factor_p95=("default_factor_v15", lambda s: float(np.quantile(s, 0.95))),
            lgd_factor_mean=("lgd_factor_v15", "mean"),
            prepay_factor_mean=("prepay_factor_v15", "mean"),
            systemic_factor_sd=("systemic_factor_v15", "std"),
        )
        .assign(
            calibration_scope="internal_calibration_v2_with_common_random_numbers",
            external_macro_context_used=bool(not macro.empty),
        )
    )
    return design, scenarios, paths, diagnostics


def _summarize_horizons(trace: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    if trace.empty:
        return pd.DataFrame()
    for horizon in horizons:
        local = trace[trace["month_idx"].le(horizon - 1)].copy()
        if local.empty:
            continue
        final = local.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
        for policy_id, grp in final.groupby("policy_id"):
            rows.append(
                {
                    "policy_id": policy_id,
                    "horizon_months": horizon,
                    "n_paths": int(grp["path_id"].nunique()),
                    "wealth_mean": float(grp["wealth"].mean()),
                    "wealth_p05": float(np.quantile(grp["wealth"], 0.05)),
                    "loss_mean": float(grp["cumulative_losses"].mean()),
                    "loss_p95": float(np.quantile(grp["cumulative_losses"], 0.95)),
                    "defaults_mean": float(grp["cumulative_defaults"].mean()),
                    "realized_return_mean": float(grp["cumulative_realized_return"].mean()),
                    "no_temporal_leakage_rate": float(grp["no_temporal_leakage_flag"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _gate_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    variants = [
        ("base_committee", 225_000.0, 0.95, 0.30, 0.20, 0.25, 0.15),
        ("strict_tail", 175_000.0, 0.98, 0.20, 0.25, 0.35, 0.10),
        ("wealth_first", 260_000.0, 0.95, 0.45, 0.20, 0.15, 0.10),
        ("tail_first", 210_000.0, 0.97, 0.20, 0.20, 0.40, 0.10),
        ("review_relaxed", 300_000.0, 0.90, 0.35, 0.15, 0.15, 0.15),
    ]
    rows = []
    for variant, loss_cap, exposure_floor, w_wealth, w_p05, w_loss, w_source in variants:
        local = summary.copy()
        wealth_rank = v15._rank_score(local["final_wealth_mean"], high_is_good=True)
        p05_rank = v15._rank_score(local["final_wealth_p05"], high_is_good=True)
        loss_rank = v15._rank_score(local["cumulative_losses_p95"], high_is_good=False)
        source_rank = v15._rank_score(
            local["source_exposure_weak_share_final_mean"], high_is_good=False
        )
        default_rank = v15._rank_score(local["cumulative_defaults_mean"], high_is_good=False)
        local["gate_variant_score"] = (
            w_wealth * wealth_rank
            + w_p05 * p05_rank
            + w_loss * loss_rank
            + w_source * source_rank
            + 0.10 * default_rank
        )
        local["gate_variant_pass"] = (
            local["cumulative_funded_exposure_mean"].ge(exposure_floor * v15.BUDGET)
            & local["final_wealth_mean"].ge(v15.BUDGET)
            & local["cumulative_losses_p95"].le(loss_cap)
            & local["no_temporal_leakage_rate"].ge(1.0)
        )
        local["gate_variant_score"] = np.where(
            local["gate_variant_pass"], local["gate_variant_score"], -1.0
        )
        winner = local.sort_values("gate_variant_score", ascending=False).iloc[0]
        rows.append(
            {
                "gate_variant_v19": variant,
                "loss_cap": loss_cap,
                "exposure_floor_budget_share": exposure_floor,
                "winning_policy_id": winner["policy_id"],
                "winning_score": float(winner["gate_variant_score"]),
                "gate_pass_count": int(local["gate_variant_pass"].sum()),
                "interpretation": "committee score sensitivity; not final promotion",
            }
        )
    return pd.DataFrame(rows)


def _path_pairwise_ci(trace: pd.DataFrame, reference_policy_id: str) -> pd.DataFrame:
    final = trace.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    base = final[final["policy_id"].eq(reference_policy_id)][
        ["path_id", "wealth", "cumulative_losses"]
    ].rename(columns={"wealth": "reference_wealth", "cumulative_losses": "reference_loss"})
    if base.empty:
        reference_policy_id = str(
            final.groupby("policy_id")["wealth"].mean().sort_values(ascending=False).index[0]
        )
        base = final[final["policy_id"].eq(reference_policy_id)][
            ["path_id", "wealth", "cumulative_losses"]
        ].rename(columns={"wealth": "reference_wealth", "cumulative_losses": "reference_loss"})
    rows = []
    for policy_id, local in final.groupby("policy_id"):
        merged = local.merge(base, on="path_id", how="inner")
        if merged.empty:
            continue
        diff = merged["wealth"] - merged["reference_wealth"]
        loss_diff = merged["cumulative_losses"] - merged["reference_loss"]
        se = float(diff.std(ddof=1) / np.sqrt(max(len(diff), 1))) if len(diff) > 1 else 0.0
        rows.append(
            {
                "policy_id": policy_id,
                "reference_policy_id": reference_policy_id,
                "n_common_paths": int(len(diff)),
                "mean_wealth_diff": float(diff.mean()),
                "ci95_low_wealth_diff": float(diff.mean() - 1.96 * se),
                "ci95_high_wealth_diff": float(diff.mean() + 1.96 * se),
                "prob_higher_wealth": float((diff > 0).mean()),
                "mean_loss_diff": float(loss_diff.mean()),
                "prob_lower_loss": float((loss_diff < 0).mean()),
                "paired_scope": "common_random_numbers_internal_v19_paths",
            }
        )
    return pd.DataFrame(rows)


def _path_family_sensitivity(trace: pd.DataFrame) -> pd.DataFrame:
    final = (
        trace.sort_values("month")
        .groupby(["policy_id", "path_id", "macro_regime_v15"], as_index=False)
        .tail(1)
    )
    if final.empty:
        return pd.DataFrame()
    return (
        final.groupby(["policy_id", "macro_regime_v15"], as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            final_wealth_mean=("wealth", "mean"),
            final_wealth_p05=("wealth", lambda s: float(np.quantile(s, 0.05))),
            cumulative_losses_mean=("cumulative_losses", "mean"),
            cumulative_losses_p95=("cumulative_losses", lambda s: float(np.quantile(s, 0.95))),
            defaults_mean=("cumulative_defaults", "mean"),
        )
        .assign(claim_scope="path_family_sensitivity_internal_not_forecast")
    )


def build_v19(paths: int, horizon_months: int, solver_pool_n: int) -> dict[str, Any]:
    start = time.time()
    solver_pool, books, adapter_registry = _load_books(solver_pool_n)
    _write_json("paper4_v19_dynamic_state_schema.json", _dynamic_state_schema_v19())
    adapter_registry = adapter_registry.copy()
    adapter_registry["version_v19"] = "dynamic_engine_v2_adapter_reuse"
    _write_csv("paper4_v19_policy_adapter_registry.csv", adapter_registry)

    source_registry, macro = _external_macro_context()
    _write_csv("paper4_v19_external_macro_source_registry.csv", source_registry)
    if not macro.empty:
        _write_csv("paper4_v19_external_macro_context.csv", macro)

    design, scenario_register, sample_paths, calibration = _build_sample_paths_v19(
        books,
        n_paths=paths,
        horizon_months=horizon_months,
        macro=macro,
    )
    _write_csv("paper4_v19_sample_path_design.csv", design)
    _write_csv("paper4_v19_path_regime_register.csv", scenario_register)
    _write_parquet("paper4_v19_sample_paths.parquet", sample_paths)
    _write_csv("paper4_v19_path_calibration_diagnostics.csv", calibration)

    trace, summary, _ = v15.build_dynamic_engine_v15(books, sample_paths, n_paths=paths)
    summary = summary.copy()
    summary["version_v19"] = "dynamic_engine_v2"
    summary["claim_scope_v19"] = "larger internal monthly replay, not deployment"
    trace["version_v19"] = "dynamic_engine_v2"
    _write_parquet("paper4_v19_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v19_dynamic_policy_summary.csv", summary)

    horizons = sorted({12, 24, min(36, horizon_months), horizon_months})
    horizon_sensitivity = _summarize_horizons(trace, horizons)
    gate_sensitivity = _gate_sensitivity(summary)
    reference = _safe_working_champion()
    pairwise = _path_pairwise_ci(trace, reference_policy_id=reference)
    family = _path_family_sensitivity(trace)
    _write_csv("paper4_v19_horizon_sensitivity.csv", horizon_sensitivity)
    _write_csv("paper4_v19_gate_sensitivity.csv", gate_sensitivity)
    _write_csv("paper4_v19_champion_vs_challenger_dynamic_ci.csv", pairwise)
    _write_csv("paper4_v19_policy_pairwise_common_path_ci.csv", pairwise)
    _write_csv("paper4_v19_path_family_sensitivity.csv", family)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v19_dynamic_engine_v2",
        "dynamic_policy_count_v19": int(summary["policy_id"].nunique()),
        "dynamic_path_count_v19": int(paths),
        "horizon_months_v19": int(horizon_months),
        "dynamic_trace_rows_v19": int(len(trace)),
        "no_temporal_leakage_min_rate_v19": float(summary["no_temporal_leakage_rate"].min()),
        "working_reference_policy_id_v18": reference,
        "best_gate_variant_policy_id_v19": str(gate_sensitivity.iloc[0]["winning_policy_id"])
        if not gate_sensitivity.empty
        else "",
        "external_macro_context_status_v19": str(
            source_registry.iloc[0].get("fetch_status", "not_attempted")
        ),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "internal dynamic replay with common random numbers; no forecast or deployment claim",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v19_status.json", status)
    _write_note(
        "paper4_v19_dynamic_engine_v2.md",
        "\n".join(
            [
                "# Paper 4 v19 Dynamic Engine v2",
                "",
                f"- Policies: `{status['dynamic_policy_count_v19']}`.",
                f"- Paths: `{status['dynamic_path_count_v19']}`.",
                f"- Horizon months: `{status['horizon_months_v19']}`.",
                f"- No-leakage min rate: `{status['no_temporal_leakage_min_rate_v19']}`.",
                f"- External macro context: `{status['external_macro_context_status_v19']}`.",
                "",
                "This is a Paper 4 working/lab wave only.",
            ]
        ),
    )
    return status


def _safe_working_champion() -> str:
    path = STATUS_DIR / "paper4_v18_working_champion.json"
    if path.exists():
        try:
            return str(
                json.loads(path.read_text(encoding="utf-8")).get(
                    "policy_id", "paper1_economic_champion"
                )
            )
        except Exception:
            return "paper1_economic_champion"
    return "paper1_economic_champion"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--solver-pool-n", type=int, default=48_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    status = build_v19(args.paths, args.horizon_months, args.solver_pool_n)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
