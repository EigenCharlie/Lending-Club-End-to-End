#!/usr/bin/env python3
"""Build Paper 4 v66 frozen external-holdout protocol artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_frozen_manifest() -> pd.DataFrame:
    winners = read_csv("paper4_v65_online_margin_repair_winners.csv")
    if winners.empty:
        return pd.DataFrame()
    manifest = winners[
        [
            "source_family",
            "method_v65",
            "min_support_v65",
            "global_delta_v65",
            "small_cell_bonus_v65",
            "worst_source_month_coverage_v65",
            "worst_policy_month_coverage_v65",
            "max_avg_width_loan_v65",
            "width_margin_to_0p95_v65",
        ]
    ].copy()
    manifest["frozen_method_id_v66"] = (
        "v65_"
        + manifest["source_family"].astype(str)
        + "_delta"
        + manifest["global_delta_v65"].map(lambda value: f"{value:.3f}".replace(".", "p"))
        + "_m"
        + manifest["min_support_v65"].astype(int).astype(str)
    )
    manifest["base_interval_artifact_v66"] = "paper4_v9_online_selected_intervals.parquet"
    manifest["selection_artifact_v66"] = "paper4_v65_online_margin_repair_winners.csv"
    manifest["selection_artifact_sha256_v66"] = _sha256(
        TABLE_DIR / "paper4_v65_online_margin_repair_winners.csv"
    )
    manifest["script_sha256_v66"] = _sha256(
        ROOT / "scripts" / "papers" / "build_paper4_v65_online_margin_repair.py"
    )
    manifest["frozen_after_utc_v66"] = now()
    manifest["allow_parameter_changes_before_external_holdout_v66"] = False
    manifest["strict_live_deployability_claim_allowed"] = False
    manifest["claim_boundary_v66"] = (
        "method frozen for future external holdout; no external validation yet"
    )
    ordered = [
        "frozen_method_id_v66",
        "source_family",
        "method_v65",
        "min_support_v65",
        "global_delta_v65",
        "small_cell_bonus_v65",
        "worst_source_month_coverage_v65",
        "worst_policy_month_coverage_v65",
        "max_avg_width_loan_v65",
        "width_margin_to_0p95_v65",
        "base_interval_artifact_v66",
        "selection_artifact_v66",
        "selection_artifact_sha256_v66",
        "script_sha256_v66",
        "frozen_after_utc_v66",
        "allow_parameter_changes_before_external_holdout_v66",
        "strict_live_deployability_claim_allowed",
        "claim_boundary_v66",
    ]
    return manifest[ordered]


def _build_required_schema() -> pd.DataFrame:
    rows = [
        ("loan_id", True, "unique loan identifier; must not overlap v9/v65 historical panel"),
        ("issue_month", True, "calendar month; must be after frozen historical window or external"),
        ("policy_id", True, "policy/book id used for policy-month coverage gates"),
        ("y_true", True, "realized binary outcome; unavailable during interval construction"),
        ("y_pred", True, "frozen model prediction produced before y_true is observed"),
        ("qhat_v9", True, "base conformal half-width from frozen v9-style method"),
        ("period", True, "period source value for the period frozen candidate"),
        ("term", True, "term source value for the term frozen candidate"),
        ("data_snapshot_id", True, "immutable source extract id"),
        ("prediction_timestamp", True, "timestamp proving prediction precedes observed outcome"),
        ("outcome_maturity_timestamp", True, "timestamp proving outcome maturity"),
    ]
    return pd.DataFrame(
        [
            {
                "column_name_v66": name,
                "required_v66": required,
                "validation_rule_v66": rule,
                "claim_boundary_v66": "schema gate only; no external data has been validated",
            }
            for name, required, rule in rows
        ]
    )


def _build_gate_spec() -> pd.DataFrame:
    rows = [
        (
            "source_month_coverage_min",
            "coverage by frozen source family and issue_month, defended cells only",
            0.80,
            ">=",
        ),
        ("policy_month_coverage_min", "coverage by policy_id and issue_month", 0.90, ">="),
        ("avg_interval_width", "mean clipped interval width across holdout loans", 0.95, "<="),
        (
            "stress_slice_coverage_min",
            "min over all/high-qhat/high-predicted-pd slices",
            0.80,
            ">=",
        ),
        ("min_support", "minimum rows for defended source/policy/slice cells", 3.0, ">="),
        ("loan_overlap_with_training", "loan_id overlap with v9/v65 historical rows", 0.0, "=="),
        ("parameter_changes_after_freeze", "any method or threshold edit after v66", 0.0, "=="),
    ]
    return pd.DataFrame(
        [
            {
                "gate_id_v66": gate_id,
                "metric_definition_v66": definition,
                "threshold_v66": threshold,
                "operator_v66": operator,
                "gate_required_for_live_claim_v66": True,
                "claim_boundary_v66": "future holdout gate spec; not yet a passed gate",
            }
            for gate_id, definition, threshold, operator in rows
        ]
    )


def _build_protocol() -> pd.DataFrame:
    steps = [
        (
            1,
            "freeze_methods",
            "Use only frozen_method_id_v66 rows from paper4_v66_frozen_method_manifest.csv.",
        ),
        (
            2,
            "load_holdout",
            "Load a future/external holdout satisfying paper4_v66_required_holdout_schema.csv.",
        ),
        (
            3,
            "prove_temporal_separation",
            "Reject any row with loan overlap or prediction timestamps after outcome maturity.",
        ),
        (
            4,
            "construct_intervals",
            "Apply qhat = clip(qhat_v9 + global_delta_v65 + small_cell_bonus_v65, 0, 1).",
        ),
        (
            5,
            "score_frozen_gates",
            "Compute source-month, policy-month, width and stress-slice gates exactly as v64/v65.",
        ),
        (
            6,
            "claim_decision",
            "Allow live-deployability language only if all gates pass without changing parameters.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "protocol_step_v66": step,
                "action_v66": action,
                "locked_instruction_v66": instruction,
                "editable_after_v66_freeze": False,
                "claim_boundary_v66": "ex ante protocol only; no holdout result exists",
            }
            for step, action, instruction in steps
        ]
    )


def _build_leakage_checklist() -> pd.DataFrame:
    checks = [
        ("no_loan_overlap", "loan_id not in v9/v65 historical interval panel", "hard_fail"),
        (
            "future_or_external_months",
            "issue_month outside frozen historical selection window",
            "hard_fail",
        ),
        (
            "prediction_before_outcome",
            "prediction_timestamp < outcome_maturity_timestamp",
            "hard_fail",
        ),
        (
            "immutable_snapshot",
            "data_snapshot_id is immutable and archived before scoring",
            "hard_fail",
        ),
        (
            "no_parameter_edits",
            "manifest hashes match before and after holdout scoring",
            "hard_fail",
        ),
        (
            "full_denominator_reported",
            "all eligible holdout rows included or excluded with reason",
            "hard_fail",
        ),
        ("source_values_present", "period and term values available before y_true", "hard_fail"),
    ]
    return pd.DataFrame(
        [
            {
                "leakage_check_id_v66": check_id,
                "rule_v66": rule,
                "failure_action_v66": action,
                "claim_boundary_v66": "checklist for future data only",
            }
            for check_id, rule, action in checks
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a frozen external-holdout protocol for v65 online candidates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v66_external_holdout_protocol.csv"
                ),
                "boundary": "Protocol only; no external holdout data has passed.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v66 protocol itself validates live online deployment.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v66_holdout_gate_spec.csv"
                ),
                "boundary": "A protocol is not validation; a future/external holdout must pass.",
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
                "executable_item": "v66 freezes the external/future holdout protocol for v65 candidates.",
                "status": "resolved_protocol_only",
                "next_artifact": "external_or_future_period_online_holdout.csv",
                "success_condition": "actual holdout data passes the frozen v66 gate spec",
                "last_wave": "v66",
                "execution_result": "external_holdout_protocol_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Online conformal",
                "executable_item": "Implement scorer once the v66 holdout dataset exists.",
                "status": "data_blocked",
                "next_artifact": "paper4_v67_external_holdout_scorecard.csv",
                "success_condition": "frozen v66 scorer runs without parameter changes",
                "last_wave": "v66",
                "execution_result": "holdout_scorer_queued",
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
    start = "<!-- V66_EXTERNAL_HOLDOUT_PROTOCOL_START -->"
    end = "<!-- V66_EXTERNAL_HOLDOUT_PROTOCOL_END -->"
    block = f"""
{start}

## Wave v66: External Holdout Protocol Freeze

Generated: {status["generated_at_utc"]}

### Objective

Freeze the external/future holdout protocol for the v65 online margin-repair
candidates before any such holdout data is available.

### Results

- Frozen method rows: `{status["frozen_method_rows_v66"]}`.
- Required schema rows: `{status["required_schema_rows_v66"]}`.
- Gate spec rows: `{status["gate_spec_rows_v66"]}`.
- Leakage checklist rows: `{status["leakage_check_rows_v66"]}`.
- External holdout data available: `{status["external_holdout_data_available_v66"]}`.

### Interpretation

v66 prevents post-hoc tuning. The v65 candidates are now frozen with hashes,
required columns, leakage checks and pass/fail gates. This is a protocol
artifact, not an empirical validation artifact.

### Claim Impact

- Allowed: a frozen external-holdout protocol exists for v65 candidates.
- Still prohibited: live deployability, external validation, final promotion
  and Paper Estrella replacement.

### Quarto Promotion Decision

Keep v66 in the living notebook until actual external/future holdout data
passes the frozen gate spec.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v66() -> dict[str, Any]:
    started = datetime.now(UTC)
    manifest = _build_frozen_manifest()
    schema = _build_required_schema()
    gates = _build_gate_spec()
    protocol = _build_protocol()
    leakage = _build_leakage_checklist()

    write_csv(TABLE_DIR / "paper4_v66_frozen_method_manifest.csv", manifest)
    write_csv(TABLE_DIR / "paper4_v66_required_holdout_schema.csv", schema)
    write_csv(TABLE_DIR / "paper4_v66_holdout_gate_spec.csv", gates)
    write_csv(TABLE_DIR / "paper4_v66_external_holdout_protocol.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v66_leakage_prevention_checklist.csv", leakage)

    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v66_external_holdout_protocol_frozen",
                "allowed": True,
                "artifact": "paper4_v66_external_holdout_protocol.csv",
                "boundary": "protocol exists before holdout; no validation claim",
            },
            {
                "claim_id": "v66_live_deployability_validated",
                "allowed": False,
                "artifact": "paper4_v66_holdout_gate_spec.csv",
                "boundary": "future/external data must pass frozen gates first",
            },
            {
                "claim_id": "v66_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v66_frozen_method_manifest.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v66_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v66_external_holdout_protocol",
        "schema_version": "2026-05-15.66",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "frozen_method_rows_v66": int(len(manifest)),
        "required_schema_rows_v66": int(len(schema)),
        "gate_spec_rows_v66": int(len(gates)),
        "protocol_rows_v66": int(len(protocol)),
        "leakage_check_rows_v66": int(len(leakage)),
        "method_frozen_for_future_holdout_v66": True,
        "external_holdout_data_available_v66": False,
        "strict_live_deployability_claim_allowed_v66": False,
        "paper1_promotion_allowed_v66": False,
        "paper4_working_champion_changed_v66": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": "v66 freezes the protocol only; no holdout validation exists",
    }
    write_json(STATUS_DIR / "paper4_v66_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v66": build_v66()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
