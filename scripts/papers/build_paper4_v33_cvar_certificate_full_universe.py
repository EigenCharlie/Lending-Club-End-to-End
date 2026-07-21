"""Build Paper 4 v33 CVaR certificate and full-universe feasibility artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_csv
from scripts.papers.build_paper4_v6_priority_resolution import (
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-15.33"


def _pool_row_estimate() -> int:
    candidates = [
        Path(
            "data/processed/portfolio_bound_aware/rank1_alpha01_bound_aware_276k_full_2026-04-05-1734/portfolio_bound_aware_frontier_raw.parquet"
        ),
        TABLE_DIR / "paper4_v3_full_universe_all_policy_allocations.parquet",
        TABLE_DIR / "paper4_policy_loan_matrix.parquet",
    ]
    estimates = []
    for path in candidates:
        if path.exists():
            try:
                estimates.append(len(pd.read_parquet(path, columns=[])))
            except Exception:
                try:
                    estimates.append(len(pd.read_parquet(path)))
                except Exception:
                    continue
    return int(max(estimates) if estimates else 276_869)


def _full_universe_feasibility(max_rows_without_force: int, force: bool) -> pd.DataFrame:
    rows = _pool_row_estimate()
    scenarios = 8
    binary_vars = rows
    aux_vars = scenarios + 1
    estimated_constraint_rows = rows + scenarios + 30
    feasible_to_attempt = force or rows <= max_rows_without_force
    return pd.DataFrame(
        [
            {
                "attempt_id": "v33_full_universe_cvar_lp_feasibility_guard",
                "estimated_rows": rows,
                "estimated_binary_variables": binary_vars,
                "estimated_auxiliary_variables": aux_vars,
                "estimated_constraint_rows": estimated_constraint_rows,
                "max_rows_without_force": int(max_rows_without_force),
                "force_requested": bool(force),
                "attempt_executed": bool(feasible_to_attempt and force),
                "attempt_result": "not_executed_resource_guard"
                if not (feasible_to_attempt and force)
                else "force_path_not_implemented_in_v33_guardrail",
                "exact_full_universe_claim": False,
                "recommended_path": "column_generation_or_dual_certificate_after solver model is profiled",
                "claim_boundary": "resource feasibility audit, not exact optimization proof",
            }
        ]
    )


def _certificate_v3() -> pd.DataFrame:
    cert = _safe_read_csv(TABLE_DIR / "paper4_v28_cvar_infeasibility_certificate_formalized.csv")
    frontier = _safe_read_csv(TABLE_DIR / "paper4_v28_cvar_frontier_non_dominated.csv")
    if cert.empty:
        cert = _safe_read_csv(
            TABLE_DIR / "paper4_v24_cvar_infeasibility_certificate_formalized.csv"
        )
    if cert.empty:
        cert = pd.DataFrame(
            [
                {
                    "policy_id": "strict_cvar_unknown",
                    "certificate_type_v33": "missing_prior_certificate",
                    "mathematical_infeasibility_proof_claim": False,
                }
            ]
        )
    cert = cert.copy()
    nearest = pd.DataFrame()
    if not frontier.empty:
        feasible = frontier[
            ~frontier.get("regime_v28", pd.Series("", index=frontier.index))
            .astype(str)
            .str.contains("infeasible|no_solution", case=False, na=False)
        ].copy()
        if not feasible.empty and "scenario_loss_cvar90" in feasible:
            nearest = feasible.sort_values("scenario_loss_cvar90").head(1)
    cert["certificate_type_v33"] = cert.get(
        "certificate_type_v28", cert.get("certificate_type_v24", "restricted_master_diagnostic")
    )
    cert["broken_caps_v33"] = cert.get(
        "broken_caps_v28", "CVaR cap, source cap, auditability and return floor may conflict"
    )
    cert["required_cvar_slack_v33"] = pd.to_numeric(
        cert.get("required_cvar_slack_v28", cert.get("required_cvar_slack_proxy", np.nan)),
        errors="coerce",
    )
    cert["required_return_floor_relaxation_v33"] = pd.to_numeric(
        cert.get(
            "required_return_floor_relaxation_v28",
            cert.get("required_return_floor_relaxation_proxy", np.nan),
        ),
        errors="coerce",
    ).fillna(0.0)
    cert["nearest_feasible_relaxed_policy_id_v33"] = (
        str(nearest["policy_id"].iloc[0]) if not nearest.empty and "policy_id" in nearest else ""
    )
    cert["dual_slack_available_v33"] = False
    cert["exact_full_universe_claim_v33"] = False
    cert["mathematical_infeasibility_proof_claim"] = False
    cert["claim_boundary_v33"] = (
        "practical restricted-master/column-generation diagnostic, not exact mathematical proof"
    )
    return cert


def _frontier_v3() -> pd.DataFrame:
    frontier = _safe_read_csv(TABLE_DIR / "paper4_v28_cvar_frontier_non_dominated.csv")
    if frontier.empty:
        return pd.DataFrame()
    out = frontier.copy()
    out["version_v33"] = "cvar_certificate_v3"
    out["exact_full_universe_claim_v33"] = False
    out["frontier_claim_v33"] = "restricted-master non-dominated diagnostic"
    if {"objective_return", "scenario_loss_cvar90"}.issubset(out.columns):
        out["tail_champion_score_v33"] = (
            pd.to_numeric(out["objective_return"], errors="coerce").rank(pct=True)
            + (1 - pd.to_numeric(out["scenario_loss_cvar90"], errors="coerce").rank(pct=True))
        ) / 2
    else:
        out["tail_champion_score_v33"] = np.nan
    return out


def build_v33(max_rows_without_force: int, force_full_universe: bool) -> dict[str, Any]:
    start = time.time()
    feasibility = _full_universe_feasibility(max_rows_without_force, force_full_universe)
    cert = _certificate_v3()
    frontier = _frontier_v3()
    active = _safe_read_csv(TABLE_DIR / "paper4_v28_cvar_active_cap_diagnostics.csv")
    log = _safe_read_csv(TABLE_DIR / "paper4_v28_cvar_column_generation_log.csv")
    if not active.empty:
        active = active.copy()
        active["version_v33"] = "active_caps_v3"
        active["claim_boundary_v33"] = "active cap diagnostic; no exact dual certificate"
    if not log.empty:
        log = log.copy()
        log["version_v33"] = "column_generation_log_v3_reaudit"

    _write_csv("paper4_v33_cvar_full_universe_feasibility_attempt.csv", feasibility)
    _write_csv("paper4_v33_cvar_infeasibility_certificate_v3.csv", cert)
    _write_csv("paper4_v33_cvar_frontier_v3.csv", frontier)
    _write_csv("paper4_v33_cvar_active_cap_diagnostics_v3.csv", active)
    _write_csv("paper4_v33_cvar_column_generation_log_v3.csv", log)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v33_cvar_certificate_full_universe",
        "full_universe_attempt_executed_v33": bool(feasibility["attempt_executed"].iloc[0]),
        "exact_full_universe_claim_v33": False,
        "certificate_rows_v33": int(len(cert)),
        "frontier_rows_v33": int(len(frontier)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "CVaR v33 improves certificate wording but does not claim exact full-universe optimality",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v33_status.json", status)
    _write_note(
        "paper4_v33_cvar_certificate_full_universe.md",
        "\n".join(
            [
                "# Paper 4 v33 CVaR Certificate and Full-Universe Guard",
                "",
                f"- Full-universe attempt executed: `{status['full_universe_attempt_executed_v33']}`.",
                "- Exact full-universe CVaR claim remains false.",
                "- Strict infeasibility is documented as practical diagnostic unless exact solver evidence exists.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows-without-force", type=int, default=120_000)
    parser.add_argument("--force-full-universe", action="store_true")
    args = parser.parse_args()
    build_v33(args.max_rows_without_force, args.force_full_universe)


if __name__ == "__main__":
    main()
