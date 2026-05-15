#!/usr/bin/env python3
"""Build Paper 4 v82 one-swap integer pricing probe artifacts.

v82 screens every one-drop/one-add whole-loan swap between the v80 binary
portfolio and omitted v55 loans. This is stronger than the v81 single-add
screen, but it remains a local integer-pricing probe rather than a
branch-and-price or global full-universe integer optimality certificate.
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


def _current_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    allocations = read_parquet("paper4_v80_full_pool_milp_gap_allocations.parquet")
    selected = allocations.loc[allocations["focused_full_pool_binary_selected_v80"].eq(1)].copy()
    selected = selected.reset_index(drop=True)
    source_summary = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_summary = source_summary.loc[
        source_summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    summary = read_csv("paper4_v80_full_pool_milp_gap_summary.csv")
    milp_summary = summary.loc[
        summary["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].iloc[0]
    omitted = read_parquet("paper4_v81_integer_omitted_single_add_screen.parquet").reset_index(
        drop=True
    )
    return selected, source_summary, milp_summary, omitted


def _source_maps(
    selected: pd.DataFrame, source_summary: pd.DataFrame
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    current_by_family: dict[str, dict[str, float]] = {}
    cap_by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        current_by_family[family] = (
            selected.groupby(family, dropna=False)["loan_amnt"].sum().astype(float).to_dict()
        )
        cap_by_family[family] = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .astype(float)
            .to_dict()
        )
    return current_by_family, cap_by_family


def _source_prefilter_mask(
    base_mask: np.ndarray,
    omitted: pd.DataFrame,
    selected: pd.DataFrame,
    source_summary: pd.DataFrame,
    current_exposure: float,
) -> np.ndarray:
    add_amount = omitted["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    new_total = current_exposure + add_amount[:, None] - drop_amount[None, :]
    mask = base_mask.copy()
    for family in FAMILIES:
        current_by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
        cap_by_source = (
            source_summary.loc[source_summary["source_family"].astype(str).eq(family)]
            .set_index("source_id")["cap_share_v80"]
            .astype(float)
        )
        add_source = omitted[family].astype(str).to_numpy()
        drop_source = selected[family].astype(str).to_numpy()
        current_add = np.array([current_by_source.get(x, 0.0) for x in add_source], dtype=float)
        current_drop = np.array([current_by_source.get(x, 0.0) for x in drop_source], dtype=float)
        cap_add = np.array([cap_by_source.get(x, 1.0) for x in add_source], dtype=float)
        cap_drop = np.array([cap_by_source.get(x, 1.0) for x in drop_source], dtype=float)
        same_source = add_source[:, None] == drop_source[None, :]
        add_ok = (
            current_add[:, None] + add_amount[:, None] - drop_amount[None, :] * same_source
        ) <= cap_add[:, None] * new_total + 1e-7
        drop_ok = (
            current_drop[None, :] - drop_amount[None, :] + add_amount[:, None] * same_source
        ) <= cap_drop[None, :] * new_total + 1e-7
        mask &= add_ok & drop_ok
    return mask


def _exact_source_metrics(
    add_row: pd.Series,
    drop_row: pd.Series,
    current_by_family: dict[str, dict[str, float]],
    cap_by_family: dict[str, dict[str, float]],
    new_total: float,
) -> tuple[bool, float, float, int, str, str]:
    min_slack = np.inf
    max_share = 0.0
    violations = 0
    first_family = ""
    first_source = ""
    add_amount = float(add_row["loan_amnt"])
    drop_amount = float(drop_row["loan_amnt"])
    for family in FAMILIES:
        add_source = str(add_row[family])
        drop_source = str(drop_row[family])
        sources = set(current_by_family[family]) | {add_source, drop_source}
        caps = cap_by_family[family]
        for source_id in sources:
            exposure = current_by_family[family].get(source_id, 0.0)
            if source_id == add_source:
                exposure += add_amount
            if source_id == drop_source:
                exposure -= drop_amount
            share = exposure / max(new_total, 1.0)
            cap = caps.get(source_id, 1.0)
            slack = cap - share
            min_slack = min(min_slack, slack)
            max_share = max(max_share, share)
            if share > cap + 1e-7:
                violations += 1
                if not first_family:
                    first_family = family
                    first_source = source_id
    return (
        violations == 0,
        float(min_slack),
        float(max_share),
        violations,
        first_family,
        first_source,
    )


def _candidate_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    selected, source_summary, milp_summary, omitted = _current_context()
    if universe.empty or selected.empty or source_summary.empty or omitted.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    selected_idx = idx_by_id.loc[selected["loan_id"].astype(str)].to_numpy()
    omitted_idx = idx_by_id.loc[omitted["loan_id"].astype(str)].to_numpy()
    selected["mean_return_if_dropped_v82"] = mean_returns[selected_idx]

    current_exposure = float(milp_summary["portfolio_exposure_v80"])
    exposure_min = float(milp_summary["exposure_min_v80"])
    exposure_max = float(milp_summary["exposure_max_v80"])
    cvar_cap = float(milp_summary["cvar_cap_v80"])
    current_objective_return = float(milp_summary["objective_return_v80"])
    current_losses = losses[:, selected_idx].sum(axis=1)

    add_amount = omitted["loan_amnt"].to_numpy(float)
    drop_amount = selected["loan_amnt"].to_numpy(float)
    add_return = omitted["mean_return_if_added_v81"].to_numpy(float)
    drop_return = selected["mean_return_if_dropped_v82"].to_numpy(float)

    return_mask = add_return[:, None] - drop_return[None, :] > 1e-9
    total_pairs = int(return_mask.size)
    return_pairs = int(return_mask.sum())
    exposure_after = current_exposure + add_amount[:, None] - drop_amount[None, :]
    budget_return_mask = (
        return_mask
        & (exposure_after >= exposure_min - 1e-7)
        & (exposure_after <= exposure_max + 1e-7)
    )
    budget_return_pairs = int(budget_return_mask.sum())
    source_prefilter_mask = _source_prefilter_mask(
        budget_return_mask, omitted, selected, source_summary, current_exposure
    )
    source_prefilter_pairs = int(source_prefilter_mask.sum())

    current_by_family, cap_by_family = _source_maps(selected, source_summary)
    rows: list[dict[str, Any]] = []
    for omitted_pos, selected_pos in np.argwhere(source_prefilter_mask):
        add_row = omitted.iloc[int(omitted_pos)]
        drop_row = selected.iloc[int(selected_pos)]
        new_total = float(current_exposure + add_row["loan_amnt"] - drop_row["loan_amnt"])
        source_ok, min_slack, max_share, violations, first_family, first_source = (
            _exact_source_metrics(add_row, drop_row, current_by_family, cap_by_family, new_total)
        )
        if not source_ok:
            continue
        swapped_losses = (
            current_losses
            + losses[:, omitted_idx[int(omitted_pos)]]
            - losses[:, selected_idx[int(selected_pos)]]
        )
        cvar_after = v70._tail_cvar(swapped_losses)
        loss_mean_after = float(swapped_losses.mean())
        return_delta = float(
            add_row["mean_return_if_added_v81"] - drop_row["mean_return_if_dropped_v82"]
        )
        row: dict[str, Any] = {
            "policy_id": str(add_row["policy_id"]),
            "regime_v82": str(add_row["regime_v78"]),
            "added_loan_id_v82": str(add_row["loan_id"]),
            "dropped_loan_id_v82": str(drop_row["loan_id"]),
            "added_loan_amount_v82": float(add_row["loan_amnt"]),
            "dropped_loan_amount_v82": float(drop_row["loan_amnt"]),
            "added_mean_return_v82": float(add_row["mean_return_if_added_v81"]),
            "dropped_mean_return_v82": float(drop_row["mean_return_if_dropped_v82"]),
            "return_delta_v82": return_delta,
            "objective_return_after_swap_v82": current_objective_return + return_delta,
            "exposure_after_swap_v82": new_total,
            "budget_swap_feasible_v82": True,
            "source_swap_feasible_v82": True,
            "source_min_slack_after_swap_v82": min_slack,
            "max_source_share_after_swap_v82": max_share,
            "source_cap_violations_after_swap_v82": violations,
            "first_source_block_family_v82": first_family,
            "first_source_block_id_v82": first_source,
            "loss_mean_after_swap_v82": loss_mean_after,
            "cvar90_after_swap_v82": cvar_after,
            "cvar_swap_feasible_v82": cvar_after <= cvar_cap + 1e-7,
            "one_swap_improves_return_v82": return_delta > 1e-9 and cvar_after <= cvar_cap + 1e-7,
            "integer_screen_scope_v82": "one_drop_one_add_whole_loan_swap",
            "claim_boundary_v82": (
                "one-swap integer pricing probe only; not multi-swap or global proof"
            ),
        }
        for family in FAMILIES:
            row[f"added_{family}_v82"] = str(add_row[family])
            row[f"dropped_{family}_v82"] = str(drop_row[family])
        rows.append(row)

    pairs = pd.DataFrame(rows)
    cvar_feasible_pairs = int(pairs["cvar_swap_feasible_v82"].sum()) if not pairs.empty else 0
    improving_pairs = int(pairs["one_swap_improves_return_v82"].sum()) if not pairs.empty else 0
    summary = pd.DataFrame(
        [
            {
                "policy_id": str(omitted["policy_id"].iloc[0]),
                "regime_v82": str(omitted["regime_v78"].iloc[0]),
                "selected_rows_v82": int(len(selected)),
                "omitted_rows_v82": int(len(omitted)),
                "total_pair_rows_screened_v82": total_pairs,
                "return_improving_pair_rows_v82": return_pairs,
                "budget_return_feasible_pair_rows_v82": budget_return_pairs,
                "source_prefilter_pair_rows_v82": source_prefilter_pairs,
                "source_exact_pair_rows_v82": int(len(pairs)),
                "cvar_feasible_pair_rows_v82": cvar_feasible_pairs,
                "one_swap_improving_rows_v82": improving_pairs,
                "best_one_swap_return_delta_v82": float(pairs["return_delta_v82"].max())
                if not pairs.empty
                else np.nan,
                "best_one_swap_cvar90_after_v82": float(
                    pairs.sort_values("return_delta_v82", ascending=False)[
                        "cvar90_after_swap_v82"
                    ].iloc[0]
                )
                if not pairs.empty
                else np.nan,
                "current_exposure_v82": current_exposure,
                "exposure_min_v82": exposure_min,
                "exposure_max_v82": exposure_max,
                "cvar_cap_v82": cvar_cap,
                "current_objective_return_v82": current_objective_return,
                "one_swap_local_optimality_cleared_v82": improving_pairs == 0,
                "full_universe_integer_optimality_claim_allowed_v82": False,
                "claim_boundary_v82": (
                    "one-swap screen only; feasible improving swaps require v83 repair/re-solve"
                ),
            }
        ]
    )
    stage_summary = pd.DataFrame(
        [
            {
                "stage_v82": "all_pairs",
                "pair_rows_v82": total_pairs,
                "claim_boundary_v82": "screening count only",
            },
            {
                "stage_v82": "return_improving",
                "pair_rows_v82": return_pairs,
                "claim_boundary_v82": "return delta only, before feasibility",
            },
            {
                "stage_v82": "budget_return_feasible",
                "pair_rows_v82": budget_return_pairs,
                "claim_boundary_v82": "budget plus return only",
            },
            {
                "stage_v82": "source_prefilter_feasible",
                "pair_rows_v82": source_prefilter_pairs,
                "claim_boundary_v82": "necessary source checks before exact full source audit",
            },
            {
                "stage_v82": "source_exact_feasible",
                "pair_rows_v82": int(len(pairs)),
                "claim_boundary_v82": "exact source caps after the swap",
            },
            {
                "stage_v82": "cvar_feasible_improving",
                "pair_rows_v82": cvar_feasible_pairs,
                "claim_boundary_v82": "one-swap feasible and improving only",
            },
        ]
    )
    return pairs, summary, stage_summary


def _claim_blockers(summary: pd.DataFrame) -> pd.DataFrame:
    improving = int(summary["one_swap_improving_rows_v82"].iloc[0]) if not summary.empty else 0
    return pd.DataFrame(
        [
            {
                "blocker_id_v82": "one_swap_integer_improvement_found",
                "blocking_v82": improving > 0,
                "evidence_count_v82": improving,
                "required_next_artifact_v82": "paper4_v83_apply_best_swap_or_reoptimize.csv",
                "claim_boundary_v82": (
                    "feasible improving one-swaps mean v80 binary portfolio is not locally optimal"
                ),
            },
            {
                "blocker_id_v82": "multi_swap_integer_pricing_missing",
                "blocking_v82": True,
                "evidence_count_v82": 1,
                "required_next_artifact_v82": "paper4_v83_iterated_swap_or_milp_repair.csv",
                "claim_boundary_v82": "one-swap screen does not cover multi-loan exchanges",
            },
            {
                "blocker_id_v82": "global_integer_gap_certificate_missing",
                "blocking_v82": True,
                "evidence_count_v82": 1,
                "required_next_artifact_v82": "paper4_v83_global_integer_gap_protocol.csv",
                "claim_boundary_v82": "no branch-and-price/global full-universe integer certificate",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v82 one-swap integer pricing probe over omitted v55 loans.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v82_one_swap_integer_pricing_probe.csv"
                ),
                "boundary": "One-drop/one-add whole-loan swaps only; not global integer proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v82 proves v80 is one-swap locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v82_claim_blockers.csv"
                ),
                "boundary": "v82 finds feasible improving one-swaps, so local optimality is false.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v82 proves full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v82_claim_blockers.csv"
                ),
                "boundary": "Requires iterated repair, multi-swap/branch-and-price or global gap certificate.",
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
                    "v82 screens one-drop/one-add whole-loan swaps and finds feasible "
                    "improving swaps against the v80 binary portfolio."
                ),
                "status": "improving_swap_found_ready_for_repair",
                "next_artifact": "paper4_v83_iterated_swap_or_reoptimization_repair.csv",
                "success_condition": (
                    "apply or re-optimize around the improving swaps and then rerun integer pricing"
                ),
                "last_wave": "v82",
                "execution_result": "one_swap_integer_pricing_probe_found_improvements",
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
    start = "<!-- V82_ONE_SWAP_INTEGER_PRICING_START -->"
    end = "<!-- V82_ONE_SWAP_INTEGER_PRICING_END -->"
    block = f"""
{start}

## Wave v82: One-Swap Integer Pricing Probe

Generated: {status["generated_at_utc"]}

### Objective

Screen every one-drop/one-add whole-loan swap between the v80 binary portfolio
and omitted v55 loans. This expands v81 from single-add insertions to local
integer exchanges, but it is still not a branch-and-price or global integer
optimality proof.

### Results

- Pair rows screened: `{status["total_pair_rows_screened_v82"]}`.
- Return-improving pairs: `{status["return_improving_pair_rows_v82"]}`.
- Budget+return feasible pairs: `{status["budget_return_feasible_pair_rows_v82"]}`.
- Exact source-feasible pairs: `{status["source_exact_pair_rows_v82"]}`.
- CVaR-feasible improving one-swaps: `{status["one_swap_improving_rows_v82"]}`.
- Best one-swap return delta: `{status["best_one_swap_return_delta_v82"]}`.
- Best one-swap CVaR after swap: `{status["best_one_swap_cvar90_after_v82"]}`.
- One-swap local optimality cleared: `{status["one_swap_local_optimality_cleared_v82"]}`.
- Full-universe integer optimality claim allowed:
  `{status["full_universe_integer_optimality_claim_allowed_v82"]}`.

### Interpretation

v82 finds feasible improving one-swaps against the v80 binary portfolio. That is
valuable negative evidence against claiming integer local optimality for v80 and
positive evidence for a v83 repair/re-optimization loop. It does not replace
Paper Estrella, change the official champion, or prove global full-universe
integer optimality.

### Claim Impact

- Allowed: one-swap integer pricing probe completed and improving swaps found.
- Still prohibited: v80 one-swap local optimality, full-universe integer
  optimality, Paper Estrella replacement, final Paper 4 promotion and live
  deployment.

### Quarto Promotion Decision

Keep v82 in the living notebook. Promote only after a repaired/re-solved
portfolio survives follow-up pricing, source, CVaR and dynamic gates.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v82() -> dict[str, Any]:
    started = datetime.now(UTC)
    pairs, summary, stage_summary = _candidate_pairs()
    top_candidates = pairs.sort_values(
        ["one_swap_improves_return_v82", "return_delta_v82"],
        ascending=[False, False],
    ).head(200)
    blockers = _claim_blockers(summary)

    write_csv(TABLE_DIR / "paper4_v82_one_swap_integer_pricing_probe.csv", pairs)
    write_csv(TABLE_DIR / "paper4_v82_one_swap_integer_pricing_top_candidates.csv", top_candidates)
    write_csv(TABLE_DIR / "paper4_v82_one_swap_integer_pricing_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v82_one_swap_screen_stage_summary.csv", stage_summary)
    write_csv(TABLE_DIR / "paper4_v82_claim_blockers.csv", blockers)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v82_one_swap_integer_probe_executed",
                "allowed": True,
                "artifact": "paper4_v82_one_swap_integer_pricing_summary.csv",
                "boundary": "one-drop/one-add whole-loan swaps only",
            },
            {
                "claim_id": "v82_feasible_improving_one_swaps_found",
                "allowed": bool(summary["one_swap_improving_rows_v82"].iloc[0] > 0),
                "artifact": "paper4_v82_one_swap_integer_pricing_probe.csv",
                "boundary": "lab-only repair signal, not champion replacement",
            },
            {
                "claim_id": "v82_v80_one_swap_local_optimality",
                "allowed": bool(summary["one_swap_local_optimality_cleared_v82"].iloc[0]),
                "artifact": "paper4_v82_claim_blockers.csv",
                "boundary": "false when feasible improving swaps are found",
            },
            {
                "claim_id": "v82_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v82_claim_blockers.csv",
                "boundary": "multi-swap/global gap certificate missing",
            },
            {
                "claim_id": "v82_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v82_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v82_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    status = {
        "phase": "v82_one_swap_integer_pricing_probe",
        "schema_version": "2026-05-15.82",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "summary_rows_v82": int(len(summary)),
        "stage_summary_rows_v82": int(len(stage_summary)),
        "candidate_pair_rows_v82": int(len(pairs)),
        "top_candidate_rows_v82": int(len(top_candidates)),
        "claim_blocker_rows_v82": int(len(blockers)),
        "selected_rows_v82": int(row["selected_rows_v82"]),
        "omitted_rows_v82": int(row["omitted_rows_v82"]),
        "total_pair_rows_screened_v82": int(row["total_pair_rows_screened_v82"]),
        "return_improving_pair_rows_v82": int(row["return_improving_pair_rows_v82"]),
        "budget_return_feasible_pair_rows_v82": int(row["budget_return_feasible_pair_rows_v82"]),
        "source_prefilter_pair_rows_v82": int(row["source_prefilter_pair_rows_v82"]),
        "source_exact_pair_rows_v82": int(row["source_exact_pair_rows_v82"]),
        "cvar_feasible_pair_rows_v82": int(row["cvar_feasible_pair_rows_v82"]),
        "one_swap_improving_rows_v82": int(row["one_swap_improving_rows_v82"]),
        "best_one_swap_return_delta_v82": float(row["best_one_swap_return_delta_v82"]),
        "best_one_swap_cvar90_after_v82": float(row["best_one_swap_cvar90_after_v82"]),
        "one_swap_local_optimality_cleared_v82": bool(row["one_swap_local_optimality_cleared_v82"]),
        "full_universe_integer_optimality_claim_allowed_v82": False,
        "paper1_promotion_allowed_v82": False,
        "paper4_working_champion_changed_v82": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v82 finds one-swap repair signals only; global integer optimality and promotion "
            "remain prohibited"
        ),
    }
    write_json(STATUS_DIR / "paper4_v82_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v82": build_v82()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
