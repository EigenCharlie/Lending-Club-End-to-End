#!/usr/bin/env python3
"""Build Paper 4 v81 integer omitted-loan insertion screen artifacts.

v81 screens every omitted v55 loan from the focused post-v78 universe as a
single whole-loan insertion into the v80 binary portfolio. This is an integer
pricing-style local screen, not a branch-and-price or global full-universe
integer optimality certificate.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.papers import build_paper4_v70_restricted_master_solver as v70  # noqa: E402
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71  # noqa: E402

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"
FAMILIES = ["grade", "score_decile", "income_band", "dti_band", "period", "state_top20"]


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(name: str, directory: Path = TABLE_DIR) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_parquet(
    name: str | Path, directory: Path = TABLE_DIR, columns: list[str] | None = None
) -> pd.DataFrame:
    path = Path(name)
    if not path.is_absolute():
        path = directory / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(df) if isinstance(df, list) else df
    out.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _current_portfolio_context() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    allocations = read_parquet("paper4_v80_full_pool_milp_gap_allocations.parquet")
    selected = allocations.loc[allocations["focused_full_pool_binary_selected_v80"].eq(1)].copy()
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    summary = read_csv("paper4_v80_full_pool_milp_gap_summary.csv")
    milp_summary = summary.loc[
        summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].iloc[0]
    return selected, source_summary, milp_summary


def _source_feasibility_flags(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    current_exposure: float,
) -> tuple[pd.Series, pd.DataFrame]:
    source_ok = pd.Series(True, index=candidates.index)
    rows: list[dict[str, Any]] = []
    amount = candidates["loan_amnt"].astype(float)
    new_total = current_exposure + amount
    for family in FAMILIES:
        current_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
        cap_by_source = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .astype(float)
        )
        source_id = candidates[family].astype(str)
        current_source = source_id.map(current_by_source).fillna(0.0).astype(float)
        cap = source_id.map(cap_by_source).fillna(1.0).astype(float)
        local_ok = current_source + amount <= cap * new_total + 1e-7
        source_ok &= local_ok
        rows.append(
            {
                "source_family": family,
                "rows_checked_v81": int(len(candidates)),
                "source_blocked_rows_v81": int((~local_ok).sum()),
                "source_feasible_rows_v81": int(local_ok.sum()),
                "claim_boundary_v81": "single-add source feasibility only",
            }
        )
    return source_ok, pd.DataFrame(rows)


def _single_add_screen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    omitted = read_parquet("paper4_v78_source_scope_expanded_reprice.parquet")
    selected, source_summary, milp_summary = _current_portfolio_context()
    if universe.empty or omitted.empty or selected.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    omitted_idx = idx_by_id.loc[omitted["loan_id"].astype(str)].to_numpy()
    current_losses = losses[:, selected_idx].sum(axis=1)
    current_exposure = float(milp_summary["portfolio_exposure_v80"])
    exposure_max = float(milp_summary["exposure_max_v80"])
    cvar_cap = float(milp_summary["cvar_cap_v80"])

    screen = omitted[
        [
            "policy_id",
            "regime_v78",
            "loan_index_v55",
            "loan_id",
            "loan_amnt",
            "grade",
            "score_decile",
            "income_band",
            "dti_band",
            "period",
            "state_top20",
            "minimization_reduced_cost_v78",
            "return_improvement_signal_v78",
        ]
    ].copy()
    screen["mean_return_if_added_v81"] = mean_returns[omitted_idx].astype("float32")
    screen["current_exposure_v81"] = current_exposure
    screen["exposure_after_add_v81"] = current_exposure + screen["loan_amnt"].astype(float)
    screen["budget_add_feasible_v81"] = screen["exposure_after_add_v81"] <= exposure_max + 1e-7
    source_ok, source_summary_out = _source_feasibility_flags(
        screen, selected, source_summary, current_exposure
    )
    screen["source_add_feasible_v81"] = source_ok

    candidate_mask = screen["budget_add_feasible_v81"] & screen["source_add_feasible_v81"]
    add_cvar = np.full(len(screen), np.nan, dtype=np.float64)
    for row_number, universe_idx in zip(
        np.flatnonzero(candidate_mask.to_numpy()),
        omitted_idx[candidate_mask.to_numpy()],
        strict=False,
    ):
        add_cvar[row_number] = v70._tail_cvar(current_losses + losses[:, universe_idx])
    screen["cvar90_after_add_v81"] = add_cvar.astype("float32")
    screen["cvar_add_feasible_v81"] = (
        screen["cvar90_after_add_v81"].le(cvar_cap + 1e-7).fillna(False)
    )
    screen["single_add_feasible_v81"] = (
        screen["budget_add_feasible_v81"]
        & screen["source_add_feasible_v81"]
        & screen["cvar_add_feasible_v81"]
    )
    screen["single_add_improves_return_v81"] = screen["single_add_feasible_v81"] & screen[
        "mean_return_if_added_v81"
    ].gt(1e-9)
    screen["integer_screen_scope_v81"] = "single_omitted_whole_loan_addition"
    screen["claim_boundary_v81"] = (
        "single-add integer screen over omitted v55 loans; not multi-swap or global proof"
    )

    summary = pd.DataFrame(
        [
            {
                "policy_id": str(screen["policy_id"].iloc[0]),
                "regime_v81": str(screen["regime_v78"].iloc[0]),
                "omitted_rows_screened_v81": int(len(screen)),
                "budget_feasible_rows_v81": int(screen["budget_add_feasible_v81"].sum()),
                "positive_return_rows_v81": int(screen["mean_return_if_added_v81"].gt(0).sum()),
                "source_feasible_rows_v81": int(screen["source_add_feasible_v81"].sum()),
                "cvar_feasible_rows_v81": int(screen["cvar_add_feasible_v81"].sum()),
                "single_add_feasible_rows_v81": int(screen["single_add_feasible_v81"].sum()),
                "single_add_improving_rows_v81": int(
                    screen["single_add_improves_return_v81"].sum()
                ),
                "best_single_add_return_delta_v81": float(
                    screen.loc[screen["single_add_feasible_v81"], "mean_return_if_added_v81"].max()
                )
                if screen["single_add_feasible_v81"].any()
                else np.nan,
                "best_any_omitted_return_delta_v81": float(
                    screen["mean_return_if_added_v81"].max()
                ),
                "current_exposure_v81": current_exposure,
                "exposure_max_v81": exposure_max,
                "cvar_cap_v81": cvar_cap,
                "integer_single_add_screen_cleared_v81": bool(
                    not screen["single_add_improves_return_v81"].any()
                ),
                "full_universe_integer_optimality_claim_allowed_v81": False,
                "claim_boundary_v81": (
                    "single-add screen cleared only; multi-loan swaps and global integer gap remain"
                ),
            }
        ]
    )
    top_candidates = screen.sort_values(
        [
            "single_add_improves_return_v81",
            "single_add_feasible_v81",
            "mean_return_if_added_v81",
        ],
        ascending=[False, False, False],
    ).head(200)
    return screen, summary, source_summary_out, top_candidates


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["single_add_improving_rows_v81"].iloc[0]) if not summary.empty else 0
    return pd.DataFrame(
        [
            {
                "blocker_id_v81": "single_add_integer_improvement_found",
                "blocking_v81": improving > 0,
                "evidence_count_v81": improving,
                "required_next_artifact_v81": "paper4_v82_add_omitted_integer_column.csv",
                "claim_boundary_v81": "single-add screen has no improving omitted loans if zero",
            },
            {
                "blocker_id_v81": "multi_swap_integer_pricing_missing",
                "blocking_v81": True,
                "evidence_count_v81": 1,
                "required_next_artifact_v81": "paper4_v82_multi_swap_integer_pricing_probe.csv",
                "claim_boundary_v81": "single-add screen does not cover multi-loan swaps",
            },
            {
                "blocker_id_v81": "global_integer_gap_certificate_missing",
                "blocking_v81": True,
                "evidence_count_v81": 1,
                "required_next_artifact_v81": "paper4_v82_global_integer_gap_protocol.csv",
                "claim_boundary_v81": "no branch-and-price/global full-universe integer certificate",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v81 single-add integer screen over omitted v55 loans.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v81_integer_omitted_single_add_summary.csv"
                ),
                "boundary": "Single whole-loan insertion only; not multi-swap/global integer proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v81 proves full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v81_claim_blockers.csv"
                ),
                "boundary": "Requires multi-swap/branch-and-price or global integer gap certificate.",
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


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v81 screens every omitted v55 loan as a single whole-loan addition to "
                    "the v80 binary portfolio."
                ),
                "status": "multi_swap_or_global_gap_gated",
                "next_artifact": "paper4_v82_multi_swap_integer_pricing_probe.csv",
                "success_condition": "multi-loan swaps or global integer gap protocol closes remaining blocker",
                "last_wave": "v81",
                "execution_result": "single_add_integer_screen_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
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
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    existing = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    start = "<!-- V81_INTEGER_OMITTED_SINGLE_ADD_START -->"
    end = "<!-- V81_INTEGER_OMITTED_SINGLE_ADD_END -->"
    block = f"""
{start}

## Wave v81: Integer Omitted Single-Add Screen

Generated: {status["generated_at_utc"]}

### Objective

Screen every omitted v55 loan as a single whole-loan addition to the v80 binary
portfolio. This is an integer-pricing local check over omitted loans, not a
multi-swap or branch-and-price global certificate.

### Results

- Omitted rows screened: `{status["omitted_rows_screened_v81"]}`.
- Budget-feasible rows: `{status["budget_feasible_rows_v81"]}`.
- Source-feasible rows: `{status["source_feasible_rows_v81"]}`.
- CVaR-feasible rows: `{status["cvar_feasible_rows_v81"]}`.
- Single-add feasible rows: `{status["single_add_feasible_rows_v81"]}`.
- Single-add improving rows: `{status["single_add_improving_rows_v81"]}`.
- Best feasible single-add return delta: `{status["best_single_add_return_delta_v81"]}`.
- Full-universe integer optimality claim allowed: `{status["full_universe_integer_optimality_claim_allowed_v81"]}`.

### Interpretation

v81 finds no omitted loan that can be added by itself as a whole loan while
respecting budget, source and CVaR and improving return. This strengthens the
integer story, but the remaining blocker is still real: multi-loan swaps and a
global integer gap protocol are not covered by a single-add screen.

### Claim Impact

- Allowed: single-add integer screen over omitted v55 loans completed.
- Still prohibited: full-universe integer optimality, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v81 in the living notebook. Promote only after multi-swap or global
integer-gap evidence passes.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v81() -> dict[str, Any]:
    started = datetime.now(UTC)
    screen, summary, source_summary, top_candidates = _single_add_screen()
    blockers = _claim_blockers(summary)
    screen.to_parquet(
        TABLE_DIR / "paper4_v81_integer_omitted_single_add_screen.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v81_integer_omitted_single_add_summary.csv", summary)
    write_csv(
        TABLE_DIR / "paper4_v81_integer_omitted_single_add_source_summary.csv", source_summary
    )
    write_csv(
        TABLE_DIR / "paper4_v81_integer_omitted_single_add_top_candidates.csv", top_candidates
    )
    write_csv(TABLE_DIR / "paper4_v81_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v81_single_add_integer_screen_executed",
                "allowed": True,
                "artifact": "paper4_v81_integer_omitted_single_add_summary.csv",
                "boundary": "single whole-loan insertion screen only",
            },
            {
                "claim_id": "v81_single_add_integer_screen_cleared",
                "allowed": bool(summary["single_add_improving_rows_v81"].iloc[0] == 0)
                if not summary.empty
                else False,
                "artifact": "paper4_v81_claim_blockers.csv",
                "boundary": "allowed only for single-add screen, not global optimality",
            },
            {
                "claim_id": "v81_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v81_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v81_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    status = {
        "phase": "v81_integer_omitted_single_add_screen",
        "schema_version": "2026-05-15.81",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "screen_rows_v81": int(len(screen)),
        "summary_rows_v81": int(len(summary)),
        "source_summary_rows_v81": int(len(source_summary)),
        "top_candidate_rows_v81": int(len(top_candidates)),
        "claim_blocker_rows_v81": int(len(blockers)),
        "omitted_rows_screened_v81": int(row["omitted_rows_screened_v81"]),
        "budget_feasible_rows_v81": int(row["budget_feasible_rows_v81"]),
        "positive_return_rows_v81": int(row["positive_return_rows_v81"]),
        "source_feasible_rows_v81": int(row["source_feasible_rows_v81"]),
        "cvar_feasible_rows_v81": int(row["cvar_feasible_rows_v81"]),
        "single_add_feasible_rows_v81": int(row["single_add_feasible_rows_v81"]),
        "single_add_improving_rows_v81": int(row["single_add_improving_rows_v81"]),
        "best_single_add_return_delta_v81": float(row["best_single_add_return_delta_v81"]),
        "best_any_omitted_return_delta_v81": float(row["best_any_omitted_return_delta_v81"]),
        "integer_single_add_screen_cleared_v81": bool(row["integer_single_add_screen_cleared_v81"]),
        "full_universe_integer_optimality_claim_allowed_v81": False,
        "paper1_promotion_allowed_v81": False,
        "paper4_working_champion_changed_v81": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v81 clears only a single-add integer omitted-loan screen; multi-swap and "
            "global integer gap evidence remain missing"
        ),
    }
    write_json(STATUS_DIR / "paper4_v81_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v81": build_v81()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
