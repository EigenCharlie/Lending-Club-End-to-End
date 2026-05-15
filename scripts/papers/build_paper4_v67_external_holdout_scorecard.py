#!/usr/bin/env python3
"""Build Paper 4 v67 external-holdout scorer artifacts.

The scorer is implemented before external/future holdout data exists.  When no
holdout file is present, it emits a blocked scorecard rather than inventing a
validation result.  If the expected holdout file is later added, the same script
scores the frozen v66 methods without changing parameters.
"""

from __future__ import annotations

import hashlib
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
    STATUS_DIR,
    TABLE_DIR,
    read_csv,
    write_csv,
    write_json,
)

HOLDOUT_PATH = TABLE_DIR / "external_or_future_period_online_holdout.csv"


def now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_holdout() -> pd.DataFrame:
    if not HOLDOUT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(HOLDOUT_PATH)


def _stress_slice_min(holdout: pd.DataFrame, coverage: pd.Series, min_support: int) -> float:
    qhat_cut = float(pd.to_numeric(holdout["qhat_v9"], errors="coerce").quantile(0.75))
    pred_cut = float(pd.to_numeric(holdout["y_pred"], errors="coerce").quantile(0.75))
    masks = [
        pd.Series(True, index=holdout.index),
        pd.to_numeric(holdout["qhat_v9"], errors="coerce").ge(qhat_cut),
        pd.to_numeric(holdout["y_pred"], errors="coerce").ge(pred_cut),
    ]
    values: list[float] = []
    for mask in masks:
        local = mask.fillna(False)
        if int(local.sum()) >= min_support:
            values.append(float(coverage.loc[local].mean()))
    return min(values) if values else float("nan")


def _score_method(holdout: pd.DataFrame, method: pd.Series) -> dict[str, Any]:
    family = str(method["source_family"])
    min_support = int(method["min_support_v65"])
    qhat = (
        pd.to_numeric(holdout["qhat_v9"], errors="coerce").fillna(0.0)
        + float(method["global_delta_v65"])
        + float(method["small_cell_bonus_v65"])
    ).clip(0, 1)
    y_pred = pd.to_numeric(holdout["y_pred"], errors="coerce").clip(0, 1)
    y_true = pd.to_numeric(holdout["y_true"], errors="coerce").clip(0, 1)
    low = (y_pred - qhat).clip(0, 1)
    high = (y_pred + qhat).clip(0, 1)
    coverage = ((y_true >= low) & (y_true <= high)).astype(float)
    width = high - low

    work = holdout.copy()
    work["issue_month"] = (
        pd.to_datetime(work["issue_month"], errors="coerce").dt.to_period("M").astype(str)
    )
    work["coverage_v67"] = coverage
    work["width_v67"] = width
    work["source_id_v67"] = work[family].astype(str)
    source_cells = (
        work.groupby(["source_id_v67", "issue_month"], dropna=False)
        .agg(n=("coverage_v67", "size"), coverage=("coverage_v67", "mean"))
        .reset_index()
    )
    policy_cells = (
        work.groupby(["policy_id", "issue_month"], dropna=False)
        .agg(n=("coverage_v67", "size"), coverage=("coverage_v67", "mean"))
        .reset_index()
    )
    source_defended = source_cells.loc[source_cells["n"].ge(min_support)]
    policy_defended = policy_cells.loc[policy_cells["n"].ge(min_support)]
    source_min = (
        float(source_defended["coverage"].min()) if not source_defended.empty else float("nan")
    )
    policy_min = (
        float(policy_defended["coverage"].min()) if not policy_defended.empty else float("nan")
    )
    avg_width = float(width.mean())
    stress_min = _stress_slice_min(work, coverage, min_support)
    gate = bool(
        source_min >= 0.80 and policy_min >= 0.90 and stress_min >= 0.80 and avg_width <= 0.95
    )
    return {
        "frozen_method_id_v66": method["frozen_method_id_v66"],
        "source_family": family,
        "holdout_data_available_v67": True,
        "holdout_rows_v67": int(len(holdout)),
        "source_month_coverage_min_v67": source_min,
        "policy_month_coverage_min_v67": policy_min,
        "stress_slice_coverage_min_v67": stress_min,
        "avg_interval_width_v67": avg_width,
        "all_gates_pass_v67": gate,
        "strict_live_deployability_claim_allowed": False,
        "score_status_v67": "scored_external_holdout_but_claim_still_requires_review",
        "claim_boundary_v67": "scored only if external data exists; live claim remains separately gated",
    }


def _build_readiness(manifest: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    required = read_csv("paper4_v66_required_holdout_schema.csv")
    expected_cols = set(required["column_name_v66"].astype(str)) if not required.empty else set()
    observed_cols = set(holdout.columns.astype(str)) if not holdout.empty else set()
    rows = [
        {
            "readiness_item_v67": "frozen_manifest_exists",
            "pass_v67": not manifest.empty,
            "detail_v67": "paper4_v66_frozen_method_manifest.csv",
        },
        {
            "readiness_item_v67": "selection_hash_matches_manifest",
            "pass_v67": bool(
                not manifest.empty
                and manifest["selection_artifact_sha256_v66"]
                .eq(_sha256(TABLE_DIR / "paper4_v65_online_margin_repair_winners.csv"))
                .all()
            ),
            "detail_v67": "v65 winner artifact hash check",
        },
        {
            "readiness_item_v67": "holdout_file_available",
            "pass_v67": HOLDOUT_PATH.exists(),
            "detail_v67": str(HOLDOUT_PATH.relative_to(ROOT)),
        },
        {
            "readiness_item_v67": "holdout_schema_complete",
            "pass_v67": bool(expected_cols and expected_cols.issubset(observed_cols)),
            "detail_v67": ",".join(sorted(expected_cols - observed_cols)),
        },
    ]
    out = pd.DataFrame(rows)
    out["claim_boundary_v67"] = "scorer readiness only; no validation claim"
    return out


def _build_scorecard(manifest: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    if holdout.empty:
        return pd.DataFrame(
            [
                {
                    "frozen_method_id_v66": row["frozen_method_id_v66"],
                    "source_family": row["source_family"],
                    "holdout_data_available_v67": False,
                    "holdout_rows_v67": 0,
                    "source_month_coverage_min_v67": np.nan,
                    "policy_month_coverage_min_v67": np.nan,
                    "stress_slice_coverage_min_v67": np.nan,
                    "avg_interval_width_v67": np.nan,
                    "all_gates_pass_v67": False,
                    "strict_live_deployability_claim_allowed": False,
                    "score_status_v67": "blocked_missing_external_holdout_data",
                    "claim_boundary_v67": "no holdout data, so no validation claim",
                }
                for _, row in manifest.iterrows()
            ]
        )
    return pd.DataFrame([_score_method(holdout, row) for _, row in manifest.iterrows()])


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has an executable frozen scorer for the v66 holdout protocol.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v67_external_holdout_scorecard.csv"
                ),
                "boundary": "Scorer exists; no external holdout has been scored yet.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v67 validates live online deployment.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v67_external_holdout_scorecard.csv"
                ),
                "boundary": "Current scorecard is blocked unless external/future data exists and passes.",
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
                "executable_item": "v67 implements the frozen external-holdout scorer.",
                "status": "data_blocked",
                "next_artifact": "external_or_future_period_online_holdout.csv",
                "success_condition": "provide leakage-safe holdout data and rerun v67 scorer",
                "last_wave": "v67",
                "execution_result": "scorer_implemented_scorecard_blocked_without_data",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Online conformal",
                "executable_item": "Rerun v67 scorer after the external holdout appears.",
                "status": "data_blocked",
                "next_artifact": "paper4_v68_external_holdout_scored_results.csv",
                "success_condition": "all frozen gates pass without changing v66/v67 code or parameters",
                "last_wave": "v67",
                "execution_result": "external_score_rerun_queued",
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


def _update_notebook(status: dict[str, Any]) -> None:
    existing = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    start = "<!-- V67_EXTERNAL_HOLDOUT_SCORER_START -->"
    end = "<!-- V67_EXTERNAL_HOLDOUT_SCORER_END -->"
    block = f"""
{start}

## Wave v67: External Holdout Scorer

Generated: {status["generated_at_utc"]}

### Objective

Implement the frozen v66 holdout scorer now, while external/future holdout
data is still absent, so the scoring path is executable and claim-safe later.

### Results

- Readiness rows: `{status["readiness_rows_v67"]}`.
- Scorecard rows: `{status["scorecard_rows_v67"]}`.
- Holdout data available: `{status["holdout_data_available_v67"]}`.
- Passing methods: `{status["passing_methods_v67"]}`.

### Interpretation

v67 is deliberately blocked by missing external data. That is the correct
result: the scorer exists and hashes the frozen method inputs, but it does not
pretend to validate live deployment.

### Claim Impact

- Allowed: executable frozen scorer and blocked scorecard.
- Still prohibited: live deployability, external validation, Paper Estrella
  replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v67 in the living notebook. Promote only after actual external/future
holdout rows are scored and pass the v66 gates without parameter edits.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v67() -> dict[str, Any]:
    started = datetime.now(UTC)
    manifest = read_csv("paper4_v66_frozen_method_manifest.csv")
    holdout = _read_holdout()
    readiness = _build_readiness(manifest, holdout)
    scorecard = _build_scorecard(manifest, holdout)

    write_csv(TABLE_DIR / "paper4_v67_scorer_readiness.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v67_external_holdout_scorecard.csv", scorecard)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v67_frozen_holdout_scorer_implemented",
                "allowed": True,
                "artifact": "paper4_v67_external_holdout_scorecard.csv",
                "boundary": "scorer exists and blocks when data is absent",
            },
            {
                "claim_id": "v67_live_deployability_validated",
                "allowed": False,
                "artifact": "paper4_v67_external_holdout_scorecard.csv",
                "boundary": "no validation unless external holdout rows pass frozen gates",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v67_claim_matrix_delta.csv", claim_matrix)

    passing_methods = (
        int(scorecard["all_gates_pass_v67"].astype(bool).sum()) if not scorecard.empty else 0
    )
    status = {
        "phase": "v67_external_holdout_scorer",
        "schema_version": "2026-05-15.67",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "readiness_rows_v67": int(len(readiness)),
        "scorecard_rows_v67": int(len(scorecard)),
        "holdout_data_available_v67": bool(not holdout.empty),
        "passing_methods_v67": passing_methods,
        "strict_live_deployability_claim_allowed_v67": False,
        "paper1_promotion_allowed_v67": False,
        "paper4_working_champion_changed_v67": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": "v67 implements scorer but blocks validation until external data exists",
    }
    write_json(STATUS_DIR / "paper4_v67_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v67": build_v67()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
