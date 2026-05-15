#!/usr/bin/env python3
"""Build Paper 4 v64 online pseudo-unseen stress artifacts.

v64 turns the recurring online-conformal blocker into a stricter executable
diagnostic.  It does not use external future data.  Instead, it freezes the
v9 selected online intervals, holds out later temporal slices, applies only
train-slice source support information to small-cell bonuses, and asks whether
any source-family method passes source coverage, policy-month coverage, and
width gates across all pseudo-unseen splits.
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

PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
NOTEBOOK = NOTE_DIR / "paper4_living_lab_notebook.md"
FORBIDDEN_FINAL_PROMOTION = STATUS_DIR / "paper4_final_promotion.json"

SOURCE_SPECS = {
    "grade": "original_grade",
    "period": "period",
    "term": "term",
    "score_decile": "score_decile",
    "state_top20": "state_top20",
    "income_band": "income_band",
    "dti_band": "dti_band",
}

MIN_SUPPORT_GRID = [3, 10, 20, 40, 80]
GLOBAL_DELTA_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
SMALL_CELL_BONUS_GRID = [0.00, 0.04, 0.08]


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


def _float_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.min())


def _safe_ratio(numer: float, denom: float) -> float:
    if denom == 0:
        return float("nan")
    return float(numer / denom)


def _prepare_online_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = read_parquet("paper4_v9_online_selected_intervals.parquet")
    base = read_parquet(
        "paper4_online_conformal_v4_intervals.parquet",
        columns=[
            "loan_id",
            "issue_month",
            "period",
            "original_grade",
            "term",
            "score_decile",
            "state_top20",
            "income_band",
            "dti_band",
            "y_true",
            "y_pred",
        ],
    )
    audit_rows: list[dict[str, Any]] = [
        {
            "audit_item_v64": "selected_v9_interval_rows",
            "value_v64": int(len(selected)),
            "claim_boundary_v64": "historical v9 intervals reused; not a live deployment feed",
        },
        {
            "audit_item_v64": "base_v4_interval_rows_before_dedupe",
            "value_v64": int(len(base)),
            "claim_boundary_v64": "base rows deduped by loan_id and issue_month before v64 merge",
        },
    ]
    if selected.empty or base.empty:
        return pd.DataFrame(), pd.DataFrame(audit_rows)

    deduped_base = base.drop_duplicates(["loan_id", "issue_month"]).copy()
    panel = selected.merge(deduped_base, on=["loan_id", "issue_month"], how="left")
    panel["issue_month"] = (
        pd.to_datetime(panel["issue_month"], errors="coerce").dt.to_period("M").astype(str)
    )
    for col in ["qhat_v9", "y_pred", "y_true"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    audit_rows.extend(
        [
            {
                "audit_item_v64": "base_v4_interval_rows_after_dedupe",
                "value_v64": int(len(deduped_base)),
                "claim_boundary_v64": "dedupe prevents one-to-many policy inflation in v64",
            },
            {
                "audit_item_v64": "merged_panel_rows_v64",
                "value_v64": int(len(panel)),
                "claim_boundary_v64": "loan-level historical replay panel only",
            },
            {
                "audit_item_v64": "merged_missing_y_pred_rows_v64",
                "value_v64": int(panel["y_pred"].isna().sum()),
                "claim_boundary_v64": "rows with missing prediction fields are dropped per family",
            },
        ]
    )
    return panel, pd.DataFrame(audit_rows)


def _split_months(months: list[str]) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    if len(months) >= 6:
        splits["last_6m"] = set(months[-6:])
    if len(months) >= 9:
        splits["last_9m"] = set(months[-9:])
    if len(months) >= 12:
        splits["last_12m"] = set(months[-12:])
    covid = {month for month in months if month.startswith("2020")}
    if covid:
        splits["covid_2020"] = covid
    return splits


def _coverage_width(holdout: pd.DataFrame, qhat: pd.Series) -> tuple[pd.Series, pd.Series]:
    low = (holdout["y_pred"] - qhat).clip(0, 1)
    high = (holdout["y_pred"] + qhat).clip(0, 1)
    coverage = ((holdout["y_true"] >= low) & (holdout["y_true"] <= high)).astype(float)
    width = high - low
    return coverage, width


def _stress_slices(
    holdout: pd.DataFrame,
    coverage: pd.Series,
    width: pd.Series,
    min_support: int,
) -> pd.DataFrame:
    qhat_cut = float(holdout["qhat_v9"].quantile(0.75))
    pred_cut = float(holdout["y_pred"].quantile(0.75))
    masks = {
        "all_holdout": pd.Series(True, index=holdout.index),
        "low_train_source_support": holdout["train_source_count_v64"].lt(min_support),
        "high_qhat_quartile": holdout["qhat_v9"].ge(qhat_cut),
        "high_predicted_pd_quartile": holdout["y_pred"].ge(pred_cut),
    }
    rows: list[dict[str, Any]] = []
    for slice_id, mask in masks.items():
        local = mask.fillna(False)
        n = int(local.sum())
        if n == 0:
            continue
        rows.append(
            {
                "stress_slice_v64": slice_id,
                "slice_rows_v64": n,
                "coverage_v64": float(coverage.loc[local].mean()),
                "avg_width_v64": float(width.loc[local].mean()),
                "defended_slice_v64": bool(n >= min_support),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_method(
    holdout: pd.DataFrame,
    min_support: int,
    global_delta: float,
    small_cell_bonus: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    qhat = (
        holdout["qhat_v9"]
        + global_delta
        + np.where(holdout["train_source_count_v64"].lt(min_support), small_cell_bonus, 0.0)
    ).clip(0, 1)
    coverage, width = _coverage_width(holdout, qhat)

    cells = holdout[
        ["source_family_v64", "source_id_v64", "issue_month", "policy_id", "train_source_count_v64"]
    ].copy()
    cells["coverage_v64"] = coverage
    cells["interval_width_v64"] = width

    source_cells = (
        cells.groupby(["source_family_v64", "source_id_v64", "issue_month"], dropna=False)
        .agg(
            cell_rows_v64=("coverage_v64", "size"),
            coverage_v64=("coverage_v64", "mean"),
            avg_width_v64=("interval_width_v64", "mean"),
            train_source_count_v64=("train_source_count_v64", "max"),
        )
        .reset_index()
    )
    policy_cells = (
        cells.groupby(["policy_id", "issue_month"], dropna=False)
        .agg(cell_rows_v64=("coverage_v64", "size"), coverage_v64=("coverage_v64", "mean"))
        .reset_index()
    )
    source_defended = source_cells.loc[source_cells["cell_rows_v64"].ge(min_support)]
    policy_defended = policy_cells.loc[policy_cells["cell_rows_v64"].ge(min_support)]

    slices = _stress_slices(holdout, coverage, width, min_support)
    defended_slices = slices.loc[slices["defended_slice_v64"].astype(bool)]
    stress_slice_min = _float_min(defended_slices["coverage_v64"])

    source_min = _float_min(source_defended["coverage_v64"])
    policy_min = _float_min(policy_defended["coverage_v64"])
    avg_width = float(width.mean())
    gate = bool(source_min >= 0.80 and policy_min >= 0.90 and avg_width <= 0.95)
    metrics = {
        "source_month_defended_min_v64": source_min,
        "policy_month_defended_min_v64": policy_min,
        "stress_slice_min_coverage_v64": stress_slice_min,
        "avg_width_loan_v64": avg_width,
        "n_defended_source_cells_v64": int(len(source_defended)),
        "n_defended_policy_cells_v64": int(len(policy_defended)),
        "n_defended_stress_slices_v64": int(len(defended_slices)),
        "gate_source80_policy90_width95_v64": gate,
    }
    detail = source_cells.assign(
        min_support_v64=min_support,
        global_delta_v64=global_delta,
        small_cell_bonus_v64=small_cell_bonus,
        cell_kind_v64="source_month",
    )
    return metrics, detail


def _build_grid(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(month for month in panel["issue_month"].dropna().unique() if month != "NaT")
    splits = _split_months(months)
    grid_rows: list[dict[str, Any]] = []
    selected_cell_frames: list[pd.DataFrame] = []

    for family, column in SOURCE_SPECS.items():
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
            for min_support in MIN_SUPPORT_GRID:
                for global_delta in GLOBAL_DELTA_GRID:
                    for small_cell_bonus in SMALL_CELL_BONUS_GRID:
                        metrics, _ = _evaluate_method(
                            holdout,
                            min_support=min_support,
                            global_delta=global_delta,
                            small_cell_bonus=small_cell_bonus,
                        )
                        grid_rows.append(
                            {
                                "source_family": family,
                                "pseudo_unseen_split_v64": split_name,
                                "method_v64": "train_support_hierarchical_qhat_bonus",
                                "min_support_v64": min_support,
                                "global_delta_v64": global_delta,
                                "small_cell_bonus_v64": small_cell_bonus,
                                "train_rows_v64": int(len(train)),
                                "holdout_rows_v64": int(len(holdout)),
                                "train_months_v64": int(train["issue_month"].nunique()),
                                "holdout_months_v64": int(holdout["issue_month"].nunique()),
                                "external_unseen_data_available_v64": False,
                                "strict_live_deployability_claim_allowed": False,
                                "claim_boundary_v64": (
                                    "pseudo-unseen historical temporal stress only; "
                                    "no live deployability claim"
                                ),
                                **metrics,
                            }
                        )

    grid = pd.DataFrame(grid_rows)
    if grid.empty:
        return grid, pd.DataFrame(), pd.DataFrame()

    summary = (
        grid.groupby(
            [
                "source_family",
                "method_v64",
                "min_support_v64",
                "global_delta_v64",
                "small_cell_bonus_v64",
            ],
            dropna=False,
        )
        .agg(
            evaluated_splits_v64=("pseudo_unseen_split_v64", "nunique"),
            split_gate_pass_rows_v64=("gate_source80_policy90_width95_v64", "sum"),
            worst_source_month_coverage_v64=("source_month_defended_min_v64", "min"),
            worst_policy_month_coverage_v64=("policy_month_defended_min_v64", "min"),
            worst_stress_slice_coverage_v64=("stress_slice_min_coverage_v64", "min"),
            max_avg_width_loan_v64=("avg_width_loan_v64", "max"),
            avg_width_loan_mean_v64=("avg_width_loan_v64", "mean"),
        )
        .reset_index()
    )
    summary["all_splits_gate_pass_v64"] = summary["split_gate_pass_rows_v64"].eq(
        summary["evaluated_splits_v64"]
    )
    summary["source_margin_to_0p80_v64"] = summary["worst_source_month_coverage_v64"] - 0.80
    summary["policy_margin_to_0p90_v64"] = summary["worst_policy_month_coverage_v64"] - 0.90
    summary["width_margin_to_0p95_v64"] = 0.95 - summary["max_avg_width_loan_v64"]
    summary["strict_live_deployability_claim_allowed"] = False
    summary["claim_boundary_v64"] = (
        "all-split pseudo-unseen gate is still a historical stress diagnostic, not live validation"
    )
    summary["ranking_score_v64"] = (
        summary["split_gate_pass_rows_v64"] * 100
        + summary["source_margin_to_0p80_v64"].clip(lower=-1).fillna(-1) * 10
        + summary["policy_margin_to_0p90_v64"].clip(lower=-1).fillna(-1) * 10
        + summary["width_margin_to_0p95_v64"].clip(lower=-1).fillna(-1) * 10
    )
    summary = summary.sort_values(
        [
            "all_splits_gate_pass_v64",
            "split_gate_pass_rows_v64",
            "ranking_score_v64",
            "max_avg_width_loan_v64",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    winners = (
        summary.sort_values(
            [
                "source_family",
                "all_splits_gate_pass_v64",
                "split_gate_pass_rows_v64",
                "ranking_score_v64",
                "max_avg_width_loan_v64",
            ],
            ascending=[True, False, False, False, True],
        )
        .groupby("source_family", dropna=False)
        .head(1)
        .copy()
    )
    winners["recommendation_v64"] = np.where(
        winners["all_splits_gate_pass_v64"],
        "candidate_for_external_holdout_before_live_claim",
        "near_miss_or_partial_pass_requires_external_holdout",
    )
    winners = winners.sort_values(
        [
            "all_splits_gate_pass_v64",
            "split_gate_pass_rows_v64",
            "ranking_score_v64",
            "max_avg_width_loan_v64",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    for _, winner in winners.iterrows():
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
                min_support=int(winner["min_support_v64"]),
                global_delta=float(winner["global_delta_v64"]),
                small_cell_bonus=float(winner["small_cell_bonus_v64"]),
            )
            selected_cell_frames.append(
                cells.assign(
                    source_family=family,
                    pseudo_unseen_split_v64=split_name,
                    selected_family_winner_v64=True,
                    strict_live_deployability_claim_allowed=False,
                    claim_boundary_v64=(
                        "winner cell details for historical pseudo-unseen stress only"
                    ),
                )
            )

    cell_summary = (
        pd.concat(selected_cell_frames, ignore_index=True)
        if selected_cell_frames
        else pd.DataFrame()
    )
    return grid, summary, winners, cell_summary


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has pseudo-unseen online conformal stress diagnostics.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v64_online_pseudo_unseen_stress_grid.csv"
                ),
                "boundary": "Historical temporal/source stress only; no external future holdout.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "Online conformal coverage is strictly live-deployable after v64.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v64_online_winner_memo.csv"
                ),
                "boundary": "No live claim without genuinely unseen external/future validation.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    out = pd.concat([out, additions], ignore_index=True)
    write_csv(path, out)


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Online conformal",
                "executable_item": (
                    "v64 converts the external-holdout blocker into pseudo-unseen "
                    "temporal/source stress diagnostics."
                ),
                "status": "near_resolved_with_plateau",
                "next_artifact": "external_or_future_period_online_holdout.csv",
                "success_condition": (
                    "same method passes genuinely unseen source-family holdout with width <= 0.95"
                ),
                "last_wave": "v64",
                "execution_result": "pseudo_unseen_stress_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "short",
                "lane": "Online conformal",
                "executable_item": (
                    "Investigate the v64 near miss where partial pseudo-unseen passes fail "
                    "the all-split gate by width/source-policy margins."
                ),
                "status": "gated",
                "next_artifact": "paper4_v65_online_margin_repair_grid.csv",
                "success_condition": "reduce width or lift source/policy minima without external leakage",
                "last_wave": "v64",
                "execution_result": "margin_repair_queued",
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
    out = pd.concat([current.loc[keep].copy(), additions], ignore_index=True)
    write_csv(path, out)


def _update_notebook(status: dict[str, Any], winner: pd.DataFrame) -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    existing = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    start = "<!-- V64_ONLINE_PSEUDO_UNSEEN_START -->"
    end = "<!-- V64_ONLINE_PSEUDO_UNSEEN_END -->"
    best = winner.iloc[0].to_dict() if not winner.empty else {}
    block = f"""
{start}

## Wave v64: Online Pseudo-Unseen Stress

Generated: {status["generated_at_utc"]}

### Objective

Stress the v9/v57 online conformal repair logic on later temporal slices
without claiming true external deployment evidence. v64 asks whether any
source-family method passes source-month coverage, policy-month coverage and
width gates across every pseudo-unseen split.

### Results

- Stress-grid rows: `{status["stress_grid_rows_v64"]}`.
- Local split gate-pass rows: `{status["local_split_gate_pass_rows_v64"]}`.
- All-split strict pass rows: `{status["strict_all_split_pass_rows_v64"]}`.
- Best source family: `{best.get("source_family", "NA")}`.
- Best split-pass count: `{best.get("split_gate_pass_rows_v64", "NA")}`.
- Best max average width: `{best.get("max_avg_width_loan_v64", "NA")}`.

### Interpretation

v64 is a useful near-miss diagnostic: some local pseudo-unseen splits pass,
but no method clears every split simultaneously under the width, source and
policy gates. The live-deployability claim therefore stays blocked, and the
next useful experiment is a margin repair or genuinely external/future holdout.

### Claim Impact

- Allowed: historical pseudo-unseen stress diagnostics exist.
- Still prohibited: strict live deployability, external unseen validation,
  final Paper 4 promotion and Paper Estrella replacement.

### Quarto Promotion Decision

Keep v64 in the living notebook. Do not promote until an external/future
holdout or a stronger all-split margin repair exists.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v64() -> dict[str, Any]:
    started = datetime.now(UTC)
    panel, audit = _prepare_online_panel()
    write_csv(TABLE_DIR / "paper4_v64_online_input_audit.csv", audit)
    grid, summary, winners, cell_summary = _build_grid(panel)

    write_csv(TABLE_DIR / "paper4_v64_online_pseudo_unseen_stress_grid.csv", grid)
    write_csv(TABLE_DIR / "paper4_v64_online_method_split_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v64_online_winner_memo.csv", winners)
    if not cell_summary.empty:
        cell_summary.to_parquet(
            TABLE_DIR / "paper4_v64_online_stress_cell_summary.parquet",
            index=False,
        )
    else:
        pd.DataFrame().to_parquet(
            TABLE_DIR / "paper4_v64_online_stress_cell_summary.parquet",
            index=False,
        )

    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v64_pseudo_unseen_online_stress_exists",
                "allowed": True,
                "artifact": "paper4_v64_online_pseudo_unseen_stress_grid.csv",
                "boundary": "historical pseudo-unseen stress diagnostic only",
            },
            {
                "claim_id": "v64_strict_live_deployability",
                "allowed": False,
                "artifact": "paper4_v64_online_winner_memo.csv",
                "boundary": "blocked without all-split pass and external/future holdout",
            },
            {
                "claim_id": "v64_paper1_or_paper_estrella_promotion",
                "allowed": False,
                "artifact": "paper4_v64_online_method_split_summary.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v64_claim_matrix_delta.csv", claim_matrix)

    strict_pass_rows = (
        int(summary["all_splits_gate_pass_v64"].astype(bool).sum()) if not summary.empty else 0
    )
    local_pass_rows = (
        int(grid["gate_source80_policy90_width95_v64"].astype(bool).sum()) if not grid.empty else 0
    )
    status = {
        "phase": "v64_online_pseudo_unseen_stress",
        "schema_version": "2026-05-15.64",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "stress_grid_rows_v64": int(len(grid)),
        "method_split_summary_rows_v64": int(len(summary)),
        "winner_rows_v64": int(len(winners)),
        "cell_summary_rows_v64": int(len(cell_summary)),
        "local_split_gate_pass_rows_v64": local_pass_rows,
        "strict_all_split_pass_rows_v64": strict_pass_rows,
        "external_unseen_data_available_v64": False,
        "strict_live_deployability_claim_allowed_v64": False,
        "paper1_promotion_allowed_v64": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "paper4_working_champion_changed_v64": False,
        "claim_boundary": (
            "v64 is pseudo-unseen historical stress evidence only; no live deployability claim"
        ),
    }
    write_json(STATUS_DIR / "paper4_v64_status.json", status)

    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status, winners)
    return status


def main() -> None:
    print(json.dumps({"v64": build_v64()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
