"""Build Paper 4 v35 online holdout and macro/sample-path validation artifacts."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from io import BytesIO

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_csv
from scripts.papers.build_paper4_v6_priority_resolution import (
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-15.35"
FRED_SERIES = {
    "UNRATE": "unemployment_rate",
    "FEDFUNDS": "federal_funds_rate",
    "USREC": "nber_recession_indicator",
    "DRCLACBS": "delinquency_rate_consumer_loans_all_commercial_banks",
}


def _read_parquet(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _fetch_fred_series(series_id: str, timeout: int) -> tuple[pd.DataFrame, dict[str, str]]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    meta = {"series_id": series_id, "url": url, "fetch_status": "not_attempted", "error": ""}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        df = pd.read_csv(BytesIO(payload))
        df.columns = ["month", series_id]
        df["month"] = (
            pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        )
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        meta["fetch_status"] = "success"
        return df.dropna(subset=["month"]), meta
    except Exception as exc:
        meta["fetch_status"] = "fetch_failed"
        meta["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return pd.DataFrame(columns=["month", series_id]), meta


def _external_macro_context(timeout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    registry = []
    for series_id, meaning in FRED_SERIES.items():
        df, meta = _fetch_fred_series(series_id, timeout)
        meta["meaning"] = meaning
        meta["source"] = "Federal Reserve Economic Data (FRED), fred.stlouisfed.org"
        meta["claim_boundary_v35"] = "official macro context only; not forecast validation"
        registry.append(meta)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(registry), pd.DataFrame()
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="month", how="outer")
    macro = macro.sort_values("month")
    macro["macro_context_label_v35"] = np.select(
        [
            macro.get("USREC", pd.Series(0, index=macro.index)).fillna(0).gt(0),
            macro.get("UNRATE", pd.Series(np.nan, index=macro.index))
            .fillna(0)
            .ge(macro.get("UNRATE", pd.Series(np.nan, index=macro.index)).quantile(0.80)),
        ],
        ["recession_context", "high_unemployment_context"],
        default="normal_context",
    )
    macro["claim_boundary_v35"] = "external context labels only; no future forecast claim"
    return pd.DataFrame(registry), macro


def _online_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = _read_parquet("paper4_v9_online_policy_month.parquet")
    source = _read_parquet("paper4_v9_online_source_month.parquet")
    robustness = _safe_read_csv(TABLE_DIR / "paper4_v10_online_robustness_summary.csv")
    bootstrap = _safe_read_csv(TABLE_DIR / "paper4_v10_online_month_bootstrap.csv")
    min_support = _safe_read_csv(TABLE_DIR / "paper4_v10_online_min_support_sensitivity.csv")
    rows = []
    if not robustness.empty:
        best = robustness.iloc[0]
        rows.append(
            {
                "validation_item": "v10_best_nominal",
                "source_month_defended_min": best.get("source_month_defended_min", np.nan),
                "policy_month_defended_min": best.get("policy_month_defended_min", np.nan),
                "avg_width_loan": best.get("avg_width_loan", np.nan),
                "pass_gate": bool(best.get("pass_v10", False)),
                "claim_boundary_v35": "selection-replay robustness, not live deployment",
            }
        )
    if not policy.empty and not source.empty:
        policy["month"] = pd.to_datetime(policy["month"], errors="coerce")
        source["month"] = pd.to_datetime(source["month"], errors="coerce")
        last_months = sorted(policy["month"].dropna().unique())[-6:]
        p_last = policy[policy["month"].isin(last_months)]
        s_last = source[source["month"].isin(last_months)]
        rows.append(
            {
                "validation_item": "leave_last_six_months_temporal_holdout",
                "source_month_defended_min": float(
                    s_last.loc[
                        s_last.get("standalone_gate_cell", False).astype(bool), "coverage_online_v9"
                    ].min()
                )
                if "standalone_gate_cell" in s_last and not s_last.empty
                else np.nan,
                "policy_month_defended_min": float(
                    p_last.loc[
                        p_last.get("standalone_gate_cell", False).astype(bool), "coverage_online_v9"
                    ].min()
                )
                if "standalone_gate_cell" in p_last and not p_last.empty
                else np.nan,
                "avg_width_loan": float(p_last["avg_width_online_v9"].mean())
                if "avg_width_online_v9" in p_last
                else np.nan,
                "pass_gate": False,
                "claim_boundary_v35": "strict temporal holdout diagnostic",
            }
        )
        if "source_id" in source:
            for source_id, grp in source.groupby("source_id"):
                defended = (
                    grp[grp.get("standalone_gate_cell", False).astype(bool)]
                    if "standalone_gate_cell" in grp
                    else grp
                )
                rows.append(
                    {
                        "validation_item": f"leave_source_family_view::{source_id}",
                        "source_month_defended_min": float(defended["coverage_online_v9"].min())
                        if not defended.empty
                        else np.nan,
                        "policy_month_defended_min": np.nan,
                        "avg_width_loan": float(defended["avg_width_online_v9"].mean())
                        if "avg_width_online_v9" in defended and not defended.empty
                        else np.nan,
                        "pass_gate": bool(
                            (not defended.empty) and defended["coverage_online_v9"].min() >= 0.80
                        ),
                        "claim_boundary_v35": "source-family diagnostic, not new calibration",
                    }
                )
    if not bootstrap.empty:
        rows.append(
            {
                "validation_item": "month_bootstrap_min",
                "source_month_defended_min": float(
                    pd.to_numeric(bootstrap["source_month_defended_min"], errors="coerce").min()
                ),
                "policy_month_defended_min": float(
                    pd.to_numeric(bootstrap["policy_month_defended_min"], errors="coerce").min()
                ),
                "avg_width_loan": float(
                    pd.to_numeric(bootstrap["avg_width_loan"], errors="coerce").mean()
                ),
                "pass_gate": bool(
                    pd.to_numeric(bootstrap["source_month_defended_min"], errors="coerce").min()
                    >= 0.80
                    and pd.to_numeric(bootstrap["policy_month_defended_min"], errors="coerce").min()
                    >= 0.90
                    and pd.to_numeric(bootstrap["avg_width_loan"], errors="coerce").mean() <= 0.95
                ),
                "claim_boundary_v35": "bootstrap diagnostic over replay months",
            }
        )
    if not min_support.empty:
        sens = min_support.copy()
        sens["version_v35"] = "min_support_sensitivity_reaudit"
        sens["gate_pass_v35"] = (
            pd.to_numeric(sens["source_month_defended_min"], errors="coerce").ge(0.80)
            & pd.to_numeric(sens["policy_month_defended_min"], errors="coerce").ge(0.90)
            & pd.to_numeric(sens["avg_width_loan"], errors="coerce").le(0.95)
        )
    else:
        sens = pd.DataFrame()
    holdout = pd.DataFrame(rows)
    if not holdout.empty:
        holdout["pass_gate"] = (
            pd.to_numeric(holdout["source_month_defended_min"], errors="coerce").ge(0.80)
            & pd.to_numeric(holdout["policy_month_defended_min"].fillna(0.90), errors="coerce").ge(
                0.90
            )
            & pd.to_numeric(holdout["avg_width_loan"].fillna(0.0), errors="coerce").le(0.95)
        )
    return holdout, sens


def build_v35(timeout: int) -> dict:
    start = time.time()
    source_registry, macro = _external_macro_context(timeout)
    holdout, min_support = _online_holdout()
    paths = _read_parquet("paper4_v29_sample_paths.parquet")
    if not paths.empty:
        paths = paths.copy()
        paths["version_v35"] = "sample_paths_reaudited_with_external_context_available"
        paths["forecast_validation_claim_allowed"] = False
        _write_parquet("paper4_v35_sample_paths_reaudit.parquet", paths)
    _write_csv("paper4_v35_external_macro_source_registry.csv", source_registry)
    if not macro.empty:
        _write_csv("paper4_v35_external_macro_context.csv", macro)
    _write_csv("paper4_v35_online_temporal_holdout.csv", holdout)
    _write_csv("paper4_v35_online_min_support_sensitivity.csv", min_support)

    online_pass = bool(holdout["pass_gate"].all()) if not holdout.empty else False
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v35_online_macro_validation",
        "external_macro_success_count_v35": int(source_registry["fetch_status"].eq("success").sum())
        if not source_registry.empty
        else 0,
        "online_holdout_rows_v35": int(len(holdout)),
        "online_gate_survives_v35": online_pass,
        "external_forecast_validation_claim_allowed": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "online validation is replay/holdout only; macro data are context labels, not forecasts",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v35_status.json", status)
    _write_note(
        "paper4_v35_online_macro_validation.md",
        "\n".join(
            [
                "# Paper 4 v35 Online and Macro Validation",
                "",
                f"- External macro series fetched: `{status['external_macro_success_count_v35']}`.",
                f"- Online gate survives all v35 holdouts: `{status['online_gate_survives_v35']}`.",
                "- No external forecast claim is made.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    build_v35(args.timeout)


if __name__ == "__main__":
    main()
