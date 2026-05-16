#!/usr/bin/env python3
"""Build Paper 4 v284 decomposition/branch-price prototype artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 284
BOUND_PROBE_VERSION = 283
INCUMBENT_REPAIR_VERSION = 279
RESTRICTED_POOL_VERSION = 281
NEXT_PRICING_VERSION = 285
TIGHT_SOURCE_SLACK_THRESHOLD = 1e-4


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v284 decomposition/branch-price prototype.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v284_decomposition_or_branch_price_prototype.csv"
                ),
                "boundary": (
                    "Prototype and source-tight pricing map only; no global bound or "
                    "branch-price termination certificate."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v284 proves a valid full-universe branch-price bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v284_claim_blockers.csv"
                ),
                "boundary": "Pricing blocks are identified but no dual-bound loop is executed.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v284 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v284_claim_blockers.csv"
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


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v284 identifies source-tight pricing blocks for a decomposition or "
                    "branch-price route after the direct full-v55 MIP was resource guarded."
                ),
                "status": "decomposition_branch_price_prototype_created",
                "next_artifact": "paper4_v285_source_tight_pricing_screen.csv",
                "success_condition": (
                    "source-tight pricing either finds entering candidates for a master loop "
                    "or records a sharper no-entering-column blocker"
                ),
                "last_wave": "v284",
                "execution_result": "source_tight_pricing_blocks_identified",
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


def _update_notebook(status: dict[str, object]) -> None:
    start = "<!-- V284_DECOMPOSITION_BRANCH_PRICE_PROTOTYPE_START -->"
    end = "<!-- V284_DECOMPOSITION_BRANCH_PRICE_PROTOTYPE_END -->"
    block = f"""
{start}

## Wave v284: Decomposition/Branch-Price Prototype

Generated: {status["generated_at_utc"]}

### Objective

Respond to the v283 direct-MIP resource guard by defining a decomposed
full-universe pricing route. The prototype identifies source constraints that
are effectively binding under the v279 incumbent and maps full-v55 omitted
candidates into source-tight pricing blocks.

### Results

- Full omitted candidate rows: `{status["full_omitted_candidate_rows_v284"]}`.
- Tight source rows: `{status["tight_source_rows_v284"]}`.
- Tight source candidate rows: `{status["tight_source_candidate_rows_v284"]}`.
- Positive-return tight source candidate rows:
  `{status["positive_return_tight_candidate_rows_v284"]}`.
- Decomposition prototype executed:
  `{status["decomposition_prototype_executed_v284"]}`.
- Valid branch-price bound produced:
  `{status["valid_branch_price_bound_v284"]}`.

### Interpretation

v284 is a useful bridge from resource blocker to executable decomposition work:
it shows that the binding source constraints are grade A and score decile 0,
and it scopes the next pricing screen. It still does not run a dual-bound
branch-price loop or certify global optimality.

### Claim Impact

- Allowed: decomposition/branch-price prototype and source-tight pricing map.
- Still prohibited: valid full-universe branch-price bound, Paper Estrella
  replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v284 in the living notebook. Promote only after a valid global bound,
dynamic validation and promotion gates pass.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v279_restricted_pool_milp_repair_allocations.parquet")
    source_summary = read_csv("paper4_v279_restricted_pool_milp_repair_source_summary.csv")
    bound_probe = read_csv("paper4_v283_full_universe_integer_bound_probe.csv")
    if universe.empty or selected.empty or source_summary.empty or bound_probe.empty:
        raise RuntimeError("Missing v55, v279, or v283 inputs for v284.")

    selected_ids = set(selected["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    _losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    candidates["mean_return_v284"] = mean_returns[candidates.index.to_numpy()]
    candidates["loan_id"] = candidates["loan_id"].astype(str)

    slack_col = f"source_slack_v{INCUMBENT_REPAIR_VERSION}"
    tight = source_summary.loc[
        pd.to_numeric(source_summary[slack_col], errors="coerce") <= TIGHT_SOURCE_SLACK_THRESHOLD
    ].copy()
    block_rows: list[dict[str, object]] = []
    for _, source_row in tight.iterrows():
        family = str(source_row["source_family"])
        source_id = str(source_row["source_id"])
        block_candidates = candidates.loc[candidates[family].astype(str).eq(source_id)].copy()
        positive = block_candidates.loc[block_candidates["mean_return_v284"].gt(0)].copy()
        top = block_candidates.sort_values("mean_return_v284", ascending=False).head(1)
        block_rows.append(
            {
                f"pricing_block_id_v{VERSION}": f"{family}={source_id}",
                "source_family": family,
                "source_id": source_id,
                f"cap_share_v{VERSION}": float(
                    source_row[f"cap_share_v{INCUMBENT_REPAIR_VERSION}"]
                ),
                f"incumbent_source_share_v{VERSION}": float(
                    source_row[f"source_share_v{INCUMBENT_REPAIR_VERSION}"]
                ),
                f"incumbent_source_slack_v{VERSION}": float(source_row[slack_col]),
                f"candidate_rows_v{VERSION}": int(len(block_candidates)),
                f"positive_return_candidate_rows_v{VERSION}": int(len(positive)),
                f"candidate_exposure_v{VERSION}": float(block_candidates["loan_amnt"].sum()),
                f"top_candidate_loan_id_v{VERSION}": str(top["loan_id"].iloc[0])
                if not top.empty
                else "",
                f"top_candidate_mean_return_v{VERSION}": float(top["mean_return_v284"].iloc[0])
                if not top.empty
                else float("nan"),
                f"pricing_role_v{VERSION}": (
                    "tight_source_pricing_block_requires_offsetting_drop_or_dual_penalty"
                ),
                f"claim_boundary_v{VERSION}": (
                    "pricing block diagnostic only; no entering-column certificate"
                ),
            }
        )
    pricing_blocks = pd.DataFrame(block_rows)
    tight_candidate_rows = int(pricing_blocks[f"candidate_rows_v{VERSION}"].sum())
    positive_tight_rows = int(pricing_blocks[f"positive_return_candidate_rows_v{VERSION}"].sum())
    prototype = pd.DataFrame(
        [
            {
                f"prototype_id_v{VERSION}": "source_tight_decomposition_branch_price_prototype",
                f"bound_probe_version_v{VERSION}": BOUND_PROBE_VERSION,
                f"incumbent_repair_version_v{VERSION}": INCUMBENT_REPAIR_VERSION,
                f"restricted_pool_version_v{VERSION}": RESTRICTED_POOL_VERSION,
                f"full_omitted_candidate_rows_v{VERSION}": int(len(candidates)),
                f"tight_source_threshold_v{VERSION}": TIGHT_SOURCE_SLACK_THRESHOLD,
                f"tight_source_rows_v{VERSION}": int(len(tight)),
                f"tight_source_candidate_rows_v{VERSION}": tight_candidate_rows,
                f"positive_return_tight_candidate_rows_v{VERSION}": positive_tight_rows,
                f"direct_full_mip_attempted_v{VERSION}": bool(
                    bound_probe["direct_full_mip_attempted_v283"].iloc[0]
                ),
                f"decomposition_prototype_executed_v{VERSION}": True,
                f"pricing_screen_executed_v{VERSION}": False,
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_PRICING_VERSION}_source_tight_pricing_screen.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "decomposition prototype only; no source-tight pricing or global bound yet"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "source_tight_pricing_screen_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(tight)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_PRICING_VERSION}_source_tight_pricing_screen.csv"
                ),
                f"claim_boundary_v{VERSION}": "pricing blocks exist but no pricing screen ran",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_PRICING_VERSION}_source_tight_pricing_screen.csv"
                ),
                f"claim_boundary_v{VERSION}": "no dual-bound loop or termination certificate",
            },
            {
                f"blocker_id_v{VERSION}": "global_integer_optimality_claim_blocked",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_PRICING_VERSION}_source_tight_pricing_screen.csv"
                ),
                f"claim_boundary_v{VERSION}": "prototype does not prove full-universe optimality",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v284_decomposition_branch_price_prototype_executed",
                "allowed": True,
                "artifact": "paper4_v284_decomposition_or_branch_price_prototype.csv",
                "boundary": "prototype only",
            },
            {
                "claim_id": "v284_source_tight_pricing_blocks_identified",
                "allowed": True,
                "artifact": "paper4_v284_source_tight_pricing_blocks.csv",
                "boundary": "diagnostic pricing blocks only",
            },
            {
                "claim_id": "v284_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v284_claim_blockers.csv",
                "boundary": "pricing and dual-bound loop missing",
            },
            {
                "claim_id": "v284_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v284_claim_blockers.csv",
                "boundary": "global bound missing",
            },
            {
                "claim_id": "v284_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v284_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v284_decomposition_or_branch_price_prototype.csv", prototype)
    write_csv(TABLE_DIR / "paper4_v284_source_tight_pricing_blocks.csv", pricing_blocks)
    write_csv(TABLE_DIR / "paper4_v284_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v284_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = prototype.iloc[0]
    status = {
        "phase": "v284_decomposition_branch_price_prototype",
        "schema_version": "2026-05-15.284",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "bound_probe_version_v284": BOUND_PROBE_VERSION,
        "incumbent_repair_version_v284": INCUMBENT_REPAIR_VERSION,
        "restricted_pool_version_v284": RESTRICTED_POOL_VERSION,
        "full_omitted_candidate_rows_v284": int(row["full_omitted_candidate_rows_v284"]),
        "tight_source_threshold_v284": float(row["tight_source_threshold_v284"]),
        "tight_source_rows_v284": int(row["tight_source_rows_v284"]),
        "tight_source_candidate_rows_v284": int(row["tight_source_candidate_rows_v284"]),
        "positive_return_tight_candidate_rows_v284": int(
            row["positive_return_tight_candidate_rows_v284"]
        ),
        "direct_full_mip_attempted_v284": bool(row["direct_full_mip_attempted_v284"]),
        "decomposition_prototype_executed_v284": bool(row["decomposition_prototype_executed_v284"]),
        "pricing_screen_executed_v284": False,
        "valid_branch_price_bound_v284": False,
        "full_universe_integer_optimality_claim_allowed_v284": False,
        "paper1_promotion_allowed_v284": False,
        "paper4_working_champion_changed_v284": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v284": int(len(blockers)),
        "claim_matrix_rows_v284": int(len(claim_matrix)),
        "next_artifact_v284": f"paper4_v{NEXT_PRICING_VERSION}_source_tight_pricing_screen.csv",
        "claim_boundary": (
            "v284 is a decomposition/branch-price prototype only; source-tight pricing, "
            "global bounds and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v284_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v284": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
