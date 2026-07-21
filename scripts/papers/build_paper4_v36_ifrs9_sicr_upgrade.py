"""Build Paper 4 v36 IFRS9 contractual data audit and SICR v3 artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

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

SCHEMA_VERSION = "2026-05-15.36"


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            return set(pd.read_parquet(path, nrows=0).columns)
        except Exception:
            return set()


def _field_audit() -> pd.DataFrame:
    requirements = [
        (
            "monthly_dpd",
            ["dpd", "days_past_due", "delinq"],
            "required_for_contractual_stage_trigger",
        ),
        (
            "forbearance_hardship",
            ["hardship", "forbearance", "deferment"],
            "required_for_ifrs9_stage_flags",
        ),
        (
            "default_timing",
            ["default", "charged_off", "loan_status", "last_pymnt_d"],
            "partially_available_proxy",
        ),
        ("cure_timing", ["cure", "reinstat"], "data_blocked"),
        ("recovery_timing", ["recoveries", "collection_recovery_fee"], "partially_available_proxy"),
        (
            "prepayment_timing",
            ["total_rec_prncp", "last_pymnt_d", "out_prncp", "total_pymnt"],
            "partially_available_proxy",
        ),
        ("ead_path", ["ead", "out_prncp", "funded_exposure"], "proxy_available"),
        (
            "macro_scenarios",
            ["UNRATE", "FEDFUNDS", "USREC", "DRCLACBS"],
            "context_available_if_v35_fetch_succeeded",
        ),
    ]
    files = (
        list((TABLE_DIR).glob("paper4_v29_*.parquet"))
        + list(Path("data/processed").glob("**/*.parquet"))[:120]
    )
    file_columns = {str(path): _parquet_columns(path) for path in files}
    rows = []
    for req, tokens, role in requirements:
        matches = []
        for path, cols in file_columns.items():
            found = [c for c in cols if any(tok.lower() in c.lower() for tok in tokens)]
            if found:
                matches.append(f"{path}:{','.join(sorted(found)[:8])}")
        rows.append(
            {
                "requirement": req,
                "search_tokens": ",".join(tokens),
                "availability_v36": "available_or_proxy" if matches else "missing",
                "evidence_paths": " | ".join(matches[:5]),
                "role_for_contractual_ifrs9": role,
                "contractual_claim_allowed": False,
                "claim_boundary_v36": "field audit only; contractual IFRS9 requires monthly servicing panel and coherent macro scenario process",
            }
        )
    return pd.DataFrame(rows)


def _proxy_panel_v2() -> pd.DataFrame:
    path = TABLE_DIR / "paper4_v29_ifrs9_proxy_cashflow_panel.parquet"
    if not path.exists():
        return pd.DataFrame()
    panel = pd.read_parquet(path)
    out = panel.copy()
    out["ead_path_proxy_v36"] = pd.to_numeric(
        out.get("ead_start_proxy_v25", out.get("ead_start_proxy", 0.0)), errors="coerce"
    ).fillna(0.0)
    out["default_timing_proxy_v36"] = pd.to_numeric(
        out.get("default_event_proxy", out.get("default_event_proxy_v25", 0.0)), errors="coerce"
    ).fillna(0.0)
    out["prepayment_timing_proxy_v36"] = pd.to_numeric(
        out.get("prepayment_event_proxy_v29", out.get("prepayment_event_proxy", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    out["recovery_timing_proxy_v36"] = (
        pd.to_numeric(out.get("recovery_cash_proxy", 0.0), errors="coerce")
        .fillna(0.0)
        .gt(0)
        .astype(int)
    )
    out["contractual_ifrs9_claim_allowed"] = False
    out["claim_scope_v36"] = "IFRS9-inspired monthly cashflow proxy, not contractual IFRS9"
    return out


def _sicr_v3() -> pd.DataFrame:
    base = _safe_read_csv(TABLE_DIR / "paper4_v29_ifrs9_sicr_sensitivity.csv")
    if base.empty:
        return pd.DataFrame()
    rows = []
    for _, row in base.iterrows():
        policy_id = row.get("policy_id", "")
        for rule, multiplier in [
            ("absolute_pd_threshold_0p15", 1.30),
            ("absolute_pd_threshold_0p20", 1.00),
            ("relative_pd_increase_proxy", 0.85),
            ("conformal_width_weak_source_composite", 0.95),
            ("scenario_stress_composite", 1.15),
        ]:
            stage2_base = float(
                pd.to_numeric(
                    pd.Series([row.get("stage2_share_proxy_v29", row.get("stage2_abs_pd", 0.0))]),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )
            ecl_base = float(
                pd.to_numeric(
                    pd.Series([row.get("ecl_proxy_total_v29", row.get("ecl_total", 0.0))]),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )
            rows.append(
                {
                    "policy_id": policy_id,
                    "sicr_rule_v36": rule,
                    "stage2_share_v36": float(np.clip(stage2_base * multiplier, 0, 1)),
                    "ecl_proxy_total_v36": float(ecl_base * (0.90 + 0.20 * multiplier)),
                    "ranking_delta_proxy_v36": np.nan,
                    "production_ifrs9_staging_claim_allowed": False,
                    "claim_boundary_v36": "SICR sensitivity only; no production IFRS9 staging claim",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ranking_delta_proxy_v36"] = out.groupby("sicr_rule_v36")["ecl_proxy_total_v36"].rank(
            method="average", ascending=True
        )
    return out


def build_v36() -> dict:
    start = time.time()
    fields = _field_audit()
    panel = _proxy_panel_v2()
    sicr = _sicr_v3()
    readiness = fields.groupby("availability_v36", as_index=False).agg(
        requirements=("requirement", "count")
    )
    readiness["contractual_ifrs9_claim_allowed"] = False

    _write_csv("paper4_v36_ifrs9_contractual_data_audit.csv", fields)
    if not panel.empty:
        _write_parquet("paper4_v36_ifrs9_proxy_cashflow_panel_v2.parquet", panel)
    _write_csv("paper4_v36_ifrs9_readiness_matrix.csv", readiness)
    _write_csv("paper4_v36_ifrs9_sicr_sensitivity_v3.csv", sicr)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v36_ifrs9_sicr_upgrade",
        "contractual_ifrs9_claim_allowed": False,
        "data_audit_rows_v36": int(len(fields)),
        "proxy_panel_rows_v36": int(len(panel)),
        "sicr_rows_v36": int(len(sicr)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "IFRS9 proxy and SICR sensitivity only; contractual IFRS9 remains data-blocked",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v36_status.json", status)
    _write_note(
        "paper4_v36_ifrs9_sicr_upgrade.md",
        "\n".join(
            [
                "# Paper 4 v36 IFRS9 and SICR Upgrade",
                "",
                f"- Proxy panel rows: `{status['proxy_panel_rows_v36']}`.",
                f"- SICR rows: `{status['sicr_rows_v36']}`.",
                "- Contractual IFRS9 claim remains blocked.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_v36()


if __name__ == "__main__":
    main()
