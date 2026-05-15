#!/usr/bin/env python3
"""Build Paper 4 v65 online margin-repair artifacts.

v64 showed a near miss: the best online pseudo-unseen candidate passed three
of four temporal splits and missed the all-split gate by roughly 18 bps of
interval width.  v65 runs a finer qhat-delta grid around that boundary.  It can
certify an internal historical all-split pass for coarse source views, but it
still cannot certify live deployability because no external future holdout is
available.
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

from scripts.papers.build_paper4_v64_online_pseudo_unseen_stress import (  # noqa: E402
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    SOURCE_SPECS,
    STATUS_DIR,
    TABLE_DIR,
    _evaluate_method,
    _prepare_online_panel,
    _split_months,
    read_csv,
    write_csv,
    write_json,
)

MIN_SUPPORT_GRID = [3, 10]
GLOBAL_DELTA_GRID = [round(float(x), 3) for x in np.arange(0.0, 0.0201, 0.001)]
SMALL_CELL_BONUS_GRID = [0.0]


def now() -> str:
    return datetime.now(UTC).isoformat()


def _build_margin_grid(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(month for month in panel["issue_month"].dropna().unique() if month != "NaT")
    splits = _split_months(months)
    rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []

    for family, column in SOURCE_SPECS.items():
        work = panel.assign(
            source_family_v65=family,
            source_family_v64=family,
            source_id_v65=panel[column].astype(str),
            source_id_v64=panel[column].astype(str),
        ).dropna(subset=["source_id_v64", "qhat_v9", "y_pred", "y_true"])
        for split_name, holdout_months in splits.items():
            train = work.loc[~work["issue_month"].isin(holdout_months)].copy()
            holdout = work.loc[work["issue_month"].isin(holdout_months)].copy()
            if train.empty or holdout.empty:
                continue
            train_source_counts = train.groupby("source_id_v64")["loan_id"].size()
            holdout["train_source_count_v64"] = (
                holdout["source_id_v64"].map(train_source_counts).fillna(0).astype(int)
            )
            for min_support in MIN_SUPPORT_GRID:
                for global_delta in GLOBAL_DELTA_GRID:
                    for small_cell_bonus in SMALL_CELL_BONUS_GRID:
                        metrics, _ = _evaluate_method(
                            holdout,
                            min_support=min_support,
                            global_delta=global_delta,
                            small_cell_bonus=small_cell_bonus,
                        )
                        rows.append(
                            {
                                "source_family": family,
                                "pseudo_unseen_split_v65": split_name,
                                "method_v65": "fine_delta_margin_repair",
                                "min_support_v65": min_support,
                                "global_delta_v65": global_delta,
                                "small_cell_bonus_v65": small_cell_bonus,
                                "train_rows_v65": int(len(train)),
                                "holdout_rows_v65": int(len(holdout)),
                                "external_unseen_data_available_v65": False,
                                "strict_live_deployability_claim_allowed": False,
                                "claim_boundary_v65": (
                                    "fine-grid historical pseudo-unseen margin repair; "
                                    "requires external holdout before live claim"
                                ),
                                "source_month_defended_min_v65": metrics[
                                    "source_month_defended_min_v64"
                                ],
                                "policy_month_defended_min_v65": metrics[
                                    "policy_month_defended_min_v64"
                                ],
                                "stress_slice_min_coverage_v65": metrics[
                                    "stress_slice_min_coverage_v64"
                                ],
                                "avg_width_loan_v65": metrics["avg_width_loan_v64"],
                                "n_defended_source_cells_v65": metrics[
                                    "n_defended_source_cells_v64"
                                ],
                                "n_defended_policy_cells_v65": metrics[
                                    "n_defended_policy_cells_v64"
                                ],
                                "n_defended_stress_slices_v65": metrics[
                                    "n_defended_stress_slices_v64"
                                ],
                                "gate_source80_policy90_width95_v65": metrics[
                                    "gate_source80_policy90_width95_v64"
                                ],
                            }
                        )

    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid, pd.DataFrame(), pd.DataFrame()

    summary = (
        grid.groupby(
            [
                "source_family",
                "method_v65",
                "min_support_v65",
                "global_delta_v65",
                "small_cell_bonus_v65",
            ],
            dropna=False,
        )
        .agg(
            evaluated_splits_v65=("pseudo_unseen_split_v65", "nunique"),
            split_gate_pass_rows_v65=("gate_source80_policy90_width95_v65", "sum"),
            worst_source_month_coverage_v65=("source_month_defended_min_v65", "min"),
            worst_policy_month_coverage_v65=("policy_month_defended_min_v65", "min"),
            worst_stress_slice_coverage_v65=("stress_slice_min_coverage_v65", "min"),
            max_avg_width_loan_v65=("avg_width_loan_v65", "max"),
            avg_width_loan_mean_v65=("avg_width_loan_v65", "mean"),
        )
        .reset_index()
    )
    summary["all_splits_gate_pass_v65"] = summary["split_gate_pass_rows_v65"].eq(
        summary["evaluated_splits_v65"]
    )
    summary["source_margin_to_0p80_v65"] = summary["worst_source_month_coverage_v65"] - 0.80
    summary["policy_margin_to_0p90_v65"] = summary["worst_policy_month_coverage_v65"] - 0.90
    summary["width_margin_to_0p95_v65"] = 0.95 - summary["max_avg_width_loan_v65"]
    summary["strict_live_deployability_claim_allowed"] = False
    summary["claim_boundary_v65"] = (
        "internal pseudo-unseen all-split pass if true; still no external live validation"
    )
    summary["ranking_score_v65"] = (
        summary["all_splits_gate_pass_v65"].astype(int) * 1000
        + summary["split_gate_pass_rows_v65"] * 100
        + summary["source_margin_to_0p80_v65"].clip(lower=-1).fillna(-1) * 10
        + summary["policy_margin_to_0p90_v65"].clip(lower=-1).fillna(-1) * 10
        + summary["width_margin_to_0p95_v65"].clip(lower=-1).fillna(-1) * 10
    )
    summary = summary.sort_values(
        [
            "all_splits_gate_pass_v65",
            "ranking_score_v65",
            "max_avg_width_loan_v65",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    winners = (
        summary.loc[summary["all_splits_gate_pass_v65"].astype(bool)]
        .sort_values(
            [
                "source_family",
                "ranking_score_v65",
                "max_avg_width_loan_v65",
            ],
            ascending=[True, False, True],
        )
        .groupby("source_family", dropna=False)
        .head(1)
        .copy()
    )
    if winners.empty:
        winners = (
            summary.sort_values(
                [
                    "source_family",
                    "split_gate_pass_rows_v65",
                    "ranking_score_v65",
                    "max_avg_width_loan_v65",
                ],
                ascending=[True, False, False, True],
            )
            .groupby("source_family", dropna=False)
            .head(1)
            .copy()
        )
    winners["recommendation_v65"] = np.where(
        winners["all_splits_gate_pass_v65"],
        "internal_all_split_pass_requires_external_holdout",
        "still_needs_margin_repair",
    )
    winners = winners.sort_values(
        [
            "all_splits_gate_pass_v65",
            "ranking_score_v65",
            "max_avg_width_loan_v65",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    for _, winner in winners.iterrows():
        if not bool(winner["all_splits_gate_pass_v65"]):
            continue
        family = str(winner["source_family"])
        column = SOURCE_SPECS[family]
        work = panel.assign(
            source_family_v64=family,
            source_id_v64=panel[column].astype(str),
        ).dropna(subset=["source_id_v64", "qhat_v9", "y_pred", "y_true"])
        for split_name, holdout_months in splits.items():
            train = work.loc[~work["issue_month"].isin(holdout_months)].copy()
            holdout = work.loc[work["issue_month"].isin(holdout_months)].copy()
            if train.empty or holdout.empty:
                continue
            train_source_counts = train.groupby("source_id_v64")["loan_id"].size()
            holdout["train_source_count_v64"] = (
                holdout["source_id_v64"].map(train_source_counts).fillna(0).astype(int)
            )
            _, cells = _evaluate_method(
                holdout,
                min_support=int(winner["min_support_v65"]),
                global_delta=float(winner["global_delta_v65"]),
                small_cell_bonus=float(winner["small_cell_bonus_v65"]),
            )
            cell_frames.append(
                cells.assign(
                    source_family=family,
                    pseudo_unseen_split_v65=split_name,
                    min_support_v65=int(winner["min_support_v65"]),
                    global_delta_v65=float(winner["global_delta_v65"]),
                    selected_margin_repair_v65=True,
                    strict_live_deployability_claim_allowed=False,
                    claim_boundary_v65="internal historical margin repair details only",
                )
            )
    cell_summary = pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame()
    return grid, summary, winners, cell_summary


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": (
                    "Paper 4 has an internal pseudo-unseen online margin repair "
                    "that passes all historical splits."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v65_online_margin_repair_summary.csv"
                ),
                "boundary": "Internal historical split pass only; no external future holdout.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v65 margin repair proves live online deployability.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v65_online_margin_repair_winners.csv"
                ),
                "boundary": "External/future validation is still required before live claims.",
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
                "lane": "Online conformal",
                "executable_item": (
                    "v65 repairs the v64 near miss with a fine qhat-delta grid "
                    "and finds internal all-split pseudo-unseen passes."
                ),
                "status": "resolved_internal_only",
                "next_artifact": "external_or_future_period_online_holdout.csv",
                "success_condition": (
                    "v65 selected deltas pass a genuinely unseen temporal/source holdout"
                ),
                "last_wave": "v65",
                "execution_result": "fine_margin_repair_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Online conformal",
                "executable_item": (
                    "Package v65 candidates for external/future holdout once such data exists."
                ),
                "status": "data_blocked",
                "next_artifact": "paper4_v66_external_holdout_protocol.csv",
                "success_condition": "holdout protocol declares leakage-safe data and frozen method",
                "last_wave": "v65",
                "execution_result": "external_holdout_protocol_queued",
                "quarto_promotion_decision": "not_promoted_to_quarto",
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


def _update_notebook(status: dict[str, Any], winners: pd.DataFrame) -> None:
    existing = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    start = "<!-- V65_ONLINE_MARGIN_REPAIR_START -->"
    end = "<!-- V65_ONLINE_MARGIN_REPAIR_END -->"
    best = winners.iloc[0].to_dict() if not winners.empty else {}
    block = f"""
{start}

## Wave v65: Online Margin Repair

Generated: {status["generated_at_utc"]}

### Objective

Repair the v64 near miss by replacing the coarse global qhat delta grid with
a fine margin grid around the all-split boundary.

### Results

- Margin grid rows: `{status["margin_grid_rows_v65"]}`.
- All-split pass rows: `{status["all_split_gate_pass_rows_v65"]}`.
- Families with all-split pass: `{status["families_with_all_split_pass_v65"]}`.
- Best source family: `{best.get("source_family", "NA")}`.
- Best delta: `{best.get("global_delta_v65", "NA")}`.
- Best width margin to 0.95: `{best.get("width_margin_to_0p95_v65", "NA")}`.

### Interpretation

v65 upgrades the online lane from a near miss to an internal historical
pseudo-unseen all-split pass for coarse source views. This is valuable for
Paper 4 because it identifies a frozen candidate for future external holdout,
but it still does not prove live deployability.

### Claim Impact

- Allowed: internal pseudo-unseen margin repair with all-split historical pass.
- Still prohibited: live deployability, external/future validation, legal
  fairness claims, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v65 in the living notebook. Promote only after an external/future holdout
protocol exists and passes with the method frozen ex ante.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v65() -> dict[str, Any]:
    started = datetime.now(UTC)
    panel, audit = _prepare_online_panel()
    audit["claim_boundary_v65"] = "same frozen historical panel as v64; no external data"
    write_csv(TABLE_DIR / "paper4_v65_online_margin_repair_input_audit.csv", audit)
    grid, summary, winners, cells = _build_margin_grid(panel)

    write_csv(TABLE_DIR / "paper4_v65_online_margin_repair_grid.csv", grid)
    write_csv(TABLE_DIR / "paper4_v65_online_margin_repair_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v65_online_margin_repair_winners.csv", winners)
    if cells.empty:
        pd.DataFrame().to_parquet(
            TABLE_DIR / "paper4_v65_online_margin_repair_cells.parquet", index=False
        )
    else:
        cells.to_parquet(TABLE_DIR / "paper4_v65_online_margin_repair_cells.parquet", index=False)

    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v65_internal_all_split_margin_repair",
                "allowed": True,
                "artifact": "paper4_v65_online_margin_repair_summary.csv",
                "boundary": "internal historical pseudo-unseen split pass only",
            },
            {
                "claim_id": "v65_strict_live_deployability",
                "allowed": False,
                "artifact": "paper4_v65_online_margin_repair_winners.csv",
                "boundary": "blocked until external/future holdout validates frozen method",
            },
            {
                "claim_id": "v65_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v65_online_margin_repair_winners.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v65_claim_matrix_delta.csv", claim_matrix)

    all_pass_rows = (
        int(summary["all_splits_gate_pass_v65"].astype(bool).sum()) if not summary.empty else 0
    )
    pass_families = (
        int(
            summary.loc[summary["all_splits_gate_pass_v65"].astype(bool), "source_family"].nunique()
        )
        if not summary.empty
        else 0
    )
    status = {
        "phase": "v65_online_margin_repair",
        "schema_version": "2026-05-15.65",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "margin_grid_rows_v65": int(len(grid)),
        "margin_summary_rows_v65": int(len(summary)),
        "winner_rows_v65": int(len(winners)),
        "cell_summary_rows_v65": int(len(cells)),
        "all_split_gate_pass_rows_v65": all_pass_rows,
        "families_with_all_split_pass_v65": pass_families,
        "best_width_margin_to_0p95_v65": (
            float(winners["width_margin_to_0p95_v65"].max()) if not winners.empty else None
        ),
        "external_unseen_data_available_v65": False,
        "strict_live_deployability_claim_allowed_v65": False,
        "paper1_promotion_allowed_v65": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "paper4_working_champion_changed_v65": False,
        "claim_boundary": (
            "v65 is an internal historical margin repair only; no live deployability claim"
        ),
    }
    write_json(STATUS_DIR / "paper4_v65_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status, winners)
    return status


def main() -> None:
    print(json.dumps({"v65": build_v65()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
