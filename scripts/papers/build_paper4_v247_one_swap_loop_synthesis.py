#!/usr/bin/env python3
"""Build Paper 4 v247 one-swap loop synthesis artifacts."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 247
FIRST_PRICING_VERSION = 82
FIRST_REPAIR_VERSION = 83
FINAL_REPAIR_VERSION = 245
TERMINAL_REPRICE_VERSION = 246


def _version_from_path(path: Path) -> int:
    match = re.search(r"paper4_v(\d+)_status\.json$", path.name)
    if match is None:
        raise ValueError(f"Unexpected Paper 4 status path: {path}")
    return int(match.group(1))


def _version_value(status: dict[str, Any], version: int, stem: str) -> Any:
    return status.get(f"{stem}_v{version}")


def _load_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _wave_type(phase: str) -> str:
    if "one_swap_reprice" in phase or "one_swap_integer_pricing_probe" in phase:
        return "pricing"
    if "one_swap_repair" in phase:
        return "repair"
    return "other"


def _trajectory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    status_paths = [
        path
        for path in STATUS_DIR.glob("paper4_v*_status.json")
        if re.search(r"paper4_v\d+_status\.json$", path.name)
    ]
    for path in sorted(status_paths, key=_version_from_path):
        version = _version_from_path(path)
        if version < FIRST_PRICING_VERSION or version > TERMINAL_REPRICE_VERSION:
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        phase = str(status.get("phase", ""))
        wave_type = _wave_type(phase)
        if wave_type == "other":
            continue
        row = {
            "wave_version_v247": version,
            "wave_type_v247": wave_type,
            "phase_v247": phase,
            "selected_rows_v247": _version_value(status, version, "selected_rows"),
            "candidate_pair_rows_v247": _version_value(status, version, "candidate_pair_rows"),
            "one_swap_improving_rows_v247": _version_value(
                status, version, "one_swap_improving_rows"
            ),
            "best_one_swap_return_delta_v247": _version_value(
                status, version, "best_one_swap_return_delta"
            ),
            "best_one_swap_cvar90_after_v247": _version_value(
                status, version, "best_one_swap_cvar90_after"
            ),
            "objective_return_v247": _version_value(status, version, "objective_return"),
            "scenario_loss_cvar90_v247": _version_value(status, version, "scenario_loss_cvar90"),
            "added_loan_id_v247": _version_value(status, version, "added_loan_id"),
            "dropped_loan_id_v247": _version_value(status, version, "dropped_loan_id"),
            "budget_feasible_v247": _version_value(status, version, "budget_feasible"),
            "source_feasible_v247": _version_value(status, version, "source_feasible"),
            "cvar_feasible_v247": _version_value(status, version, "cvar_feasible"),
            "one_swap_local_optimality_cleared_v247": _version_value(
                status, version, "post_repair_one_swap_local_optimality_cleared"
            ),
            "paper1_promotion_allowed_v247": _version_value(
                status, version, "paper1_promotion_allowed"
            ),
            "full_universe_integer_optimality_allowed_v247": _version_value(
                status, version, "full_universe_integer_optimality_claim_allowed"
            ),
            "claim_boundary_v247": status.get("claim_boundary"),
        }
        if version == FIRST_PRICING_VERSION:
            row["one_swap_local_optimality_cleared_v247"] = status.get(
                "one_swap_local_optimality_cleared_v82"
            )
        rows.append(row)
    trajectory = pd.DataFrame(rows)
    for column in ("added_loan_id_v247", "dropped_loan_id_v247"):
        trajectory[column] = trajectory[column].fillna("not_applicable").astype(str)
    return trajectory


def _baseline_v80() -> pd.Series:
    summary = read_csv("paper4_v80_full_pool_milp_gap_summary.csv")
    return summary.loc[summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")].iloc[0]


def _append_or_update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v247 one-swap loop synthesis from v82-v246.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v247_one_swap_loop_synthesis.csv"
                ),
                "boundary": (
                    "Synthesizes pricing/repair waves only; not a multi-swap/global proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": ("v247 permits a one-swap local optimality claim for the v245 candidate."),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v246_post_repair_one_swap_summary.csv"
                ),
                "boundary": (
                    "Allowed only for one-drop/one-add whole-loan swaps in the v55 comparable "
                    "universe after exact source and CVaR checks."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v247 proves multi-swap or global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v247_claim_blockers.csv"
                ),
                "boundary": "Requires multi-loan exchange pricing or a global gap certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v247 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v247_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation, or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _append_or_update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v247 synthesizes the v82-v246 one-swap repair loop and redirects "
                    "remaining work to multi-swap/global gap evidence."
                ),
                "status": "multi_swap_global_gap_pending",
                "next_artifact": "paper4_v248_multi_swap_or_global_gap_probe.csv",
                "success_condition": (
                    "multi-loan exchange pricing or global gap certificate bounds the v245 candidate"
                ),
                "last_wave": "v247",
                "execution_result": "one_swap_loop_synthesized",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    key_cols = ["last_wave", "lane", "next_artifact"]
    merged_keys = set(map(tuple, additions[key_cols].astype(str).to_numpy()))
    keep = [tuple(row) not in merged_keys for row in current[key_cols].astype(str).to_numpy()]
    write_csv(path, pd.concat([current.loc[keep].copy(), additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V247_ONE_SWAP_LOOP_SYNTHESIS_START -->"
    end = "<!-- V247_ONE_SWAP_LOOP_SYNTHESIS_END -->"
    block = f"""
{start}

## Wave v247: One-Swap Loop Synthesis

Generated: {status["generated_at_utc"]}

### Objective

Synthesize the v82-v246 one-swap pricing/repair loop after v246 cleared the
post-v245 one-swap screen. This creates a compact evidence bridge from the
living-lab iterations to the next required multi-swap/global gap work.

### Results

- Trajectory rows: `{status["trajectory_rows_v247"]}`.
- Repair waves synthesized: `{status["repair_wave_rows_v247"]}`.
- Pricing waves synthesized: `{status["pricing_wave_rows_v247"]}`.
- Final repair candidate: `v{status["final_repair_version_v247"]}`.
- Terminal repricing wave: `v{status["terminal_reprice_version_v247"]}`.
- Final objective return: `{status["final_objective_return_v247"]}`.
- Return gain vs v80 focused MILP incumbent:
  `{status["return_gain_vs_v80_v247"]}`.
- Terminal one-swap improving rows:
  `{status["terminal_one_swap_improving_rows_v247"]}`.
- One-swap local optimality claim allowed:
  `{status["one_swap_local_optimality_claim_allowed_v247"]}`.

### Interpretation

v247 does not add a new portfolio. It packages the repair loop evidence:
the v245 candidate is locally clear under the implemented one-drop/one-add
screen, while multi-swap/global integer evidence and all promotion claims remain
blocked.

### Claim Impact

- Allowed: one-swap loop synthesis and v245 one-swap local screen clearance.
- Still prohibited: multi-swap/global optimality, Paper Estrella replacement,
  final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v247 in the living notebook. Promote only after multi-swap/global and
dynamic validation gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    trajectory = _trajectory()
    baseline = _baseline_v80()
    final_status = _load_status(FINAL_REPAIR_VERSION)
    terminal_status = _load_status(TERMINAL_REPRICE_VERSION)

    final_objective = float(final_status[f"objective_return_v{FINAL_REPAIR_VERSION}"])
    final_cvar = float(final_status[f"scenario_loss_cvar90_v{FINAL_REPAIR_VERSION}"])
    baseline_objective = float(baseline["objective_return_v80"])
    baseline_cvar = float(baseline["scenario_loss_cvar90_v80"])
    terminal_improving = int(
        terminal_status[f"one_swap_improving_rows_v{TERMINAL_REPRICE_VERSION}"]
    )
    terminal_local_cleared = bool(
        terminal_status[
            f"post_repair_one_swap_local_optimality_cleared_v{TERMINAL_REPRICE_VERSION}"
        ]
    )

    synthesis = pd.DataFrame(
        [
            {
                "synthesis_label_v247": "one_swap_loop_v82_v246",
                "first_pricing_version_v247": FIRST_PRICING_VERSION,
                "first_repair_version_v247": FIRST_REPAIR_VERSION,
                "final_repair_version_v247": FINAL_REPAIR_VERSION,
                "terminal_reprice_version_v247": TERMINAL_REPRICE_VERSION,
                "trajectory_rows_v247": int(len(trajectory)),
                "repair_wave_rows_v247": int(trajectory["wave_type_v247"].eq("repair").sum()),
                "pricing_wave_rows_v247": int(trajectory["wave_type_v247"].eq("pricing").sum()),
                "final_objective_return_v247": final_objective,
                "final_scenario_loss_cvar90_v247": final_cvar,
                "v80_objective_return_v247": baseline_objective,
                "v80_scenario_loss_cvar90_v247": baseline_cvar,
                "return_gain_vs_v80_v247": final_objective - baseline_objective,
                "cvar_delta_vs_v80_v247": final_cvar - baseline_cvar,
                "terminal_one_swap_improving_rows_v247": terminal_improving,
                "one_swap_local_optimality_claim_allowed_v247": terminal_local_cleared,
                "multi_swap_integer_optimality_claim_allowed_v247": False,
                "full_universe_integer_optimality_claim_allowed_v247": False,
                "paper1_promotion_allowed_v247": False,
                "paper4_final_promotion_created": False,
                "claim_boundary_v247": (
                    "one-swap local clearance only; multi-swap/global/dynamic promotion "
                    "claims remain blocked"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                "blocker_id_v247": "one_swap_local_screen_open",
                "blocking_v247": not terminal_local_cleared,
                "evidence_count_v247": terminal_improving,
                "required_next_artifact_v247": "paper4_v246_post_repair_one_swap_summary.csv",
                "claim_boundary_v247": "closed only because v246 finds zero improving one-swaps",
            },
            {
                "blocker_id_v247": "multi_swap_integer_pricing_missing",
                "blocking_v247": True,
                "evidence_count_v247": 1,
                "required_next_artifact_v247": "paper4_v248_multi_swap_or_global_gap_probe.csv",
                "claim_boundary_v247": "one-swap local clearance does not price multi-loan exchanges",
            },
            {
                "blocker_id_v247": "global_integer_gap_certificate_missing",
                "blocking_v247": True,
                "evidence_count_v247": 1,
                "required_next_artifact_v247": "paper4_v248_global_gap_certificate_protocol.csv",
                "claim_boundary_v247": "no global full-universe integer gap certificate",
            },
            {
                "blocker_id_v247": "paper4_final_promotion_forbidden",
                "blocking_v247": True,
                "evidence_count_v247": 1,
                "required_next_artifact_v247": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v247": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v247_one_swap_loop_synthesis_executed",
                "allowed": True,
                "artifact": "paper4_v247_one_swap_loop_synthesis.csv",
                "boundary": "summarizes v82-v246 one-swap loop only",
            },
            {
                "claim_id": "v247_v245_one_swap_local_optimality",
                "allowed": terminal_local_cleared,
                "artifact": "paper4_v246_post_repair_one_swap_summary.csv",
                "boundary": "one-drop/one-add whole-loan screen in v55 comparable universe",
            },
            {
                "claim_id": "v247_multi_swap_or_global_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v247_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": "v247_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v247_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v247_one_swap_loop_trajectory.csv", trajectory)
    write_csv(TABLE_DIR / "paper4_v247_one_swap_loop_synthesis.csv", synthesis)
    write_csv(TABLE_DIR / "paper4_v247_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v247_claim_matrix_delta.csv", claim_matrix)
    _append_or_update_claim_boundaries()
    _append_or_update_backlog()

    row = synthesis.iloc[0]
    status = {
        "phase": "v247_one_swap_loop_synthesis",
        "schema_version": "2026-05-15.247",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "trajectory_rows_v247": int(row["trajectory_rows_v247"]),
        "repair_wave_rows_v247": int(row["repair_wave_rows_v247"]),
        "pricing_wave_rows_v247": int(row["pricing_wave_rows_v247"]),
        "first_pricing_version_v247": FIRST_PRICING_VERSION,
        "first_repair_version_v247": FIRST_REPAIR_VERSION,
        "final_repair_version_v247": FINAL_REPAIR_VERSION,
        "terminal_reprice_version_v247": TERMINAL_REPRICE_VERSION,
        "final_objective_return_v247": float(row["final_objective_return_v247"]),
        "final_scenario_loss_cvar90_v247": float(row["final_scenario_loss_cvar90_v247"]),
        "return_gain_vs_v80_v247": float(row["return_gain_vs_v80_v247"]),
        "cvar_delta_vs_v80_v247": float(row["cvar_delta_vs_v80_v247"]),
        "terminal_one_swap_improving_rows_v247": terminal_improving,
        "one_swap_local_optimality_claim_allowed_v247": terminal_local_cleared,
        "multi_swap_integer_optimality_claim_allowed_v247": False,
        "full_universe_integer_optimality_claim_allowed_v247": False,
        "paper1_promotion_allowed_v247": False,
        "paper4_final_promotion_created": False,
        "claim_blocker_rows_v247": int(len(blockers)),
        "claim_matrix_rows_v247": int(len(claim_matrix)),
        "claim_boundary": str(row["claim_boundary_v247"]),
    }
    write_json(STATUS_DIR / "paper4_v247_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v247": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
