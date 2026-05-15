#!/usr/bin/env python3
"""Build Paper 4 v69 restricted-master expansion artifacts.

v69 consumes the v68 full-universe source/pricing screen and packages the
candidate columns into an executable restricted-master expansion.  It also
audits one-for-one source/pricing swaps against the v63 incumbent books.

This is still not an exact full-universe column-generation certificate.  The
protocol table states what evidence would be needed to replace the proxy
screen with a true pricing/termination proof.
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
BUDGET = 1_000_000.0
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


def normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(float(lo), float(hi)):
        out = pd.Series(0.5, index=series.index)
    else:
        out = (values - lo) / (hi - lo)
    if not higher_is_better:
        out = 1 - out
    return out.fillna(0.0)


def _prepare_universe() -> pd.DataFrame:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    if universe.empty:
        return universe
    u = universe.copy()
    u["expected_loss_proxy_v69"] = (
        u["loan_amnt"].astype(float)
        * u["lgd_proxy_v55"].astype(float)
        * u["pd_high_alpha01"].astype(float)
    )
    u["expected_return_proxy_v69"] = (
        u["base_return_vec"].astype(float)
        - u["expected_loss_proxy_v69"]
        - 0.012 * u["loan_amnt"].astype(float)
    )
    u["tail_score_proxy_v69"] = (
        u["expected_return_proxy_v69"]
        - 1.15 * u["expected_loss_proxy_v69"]
        - 1800.0 * u["qhat_v4"].astype(float)
        - 2000.0 * u["weak_source_proxy"].astype(float)
    )
    u["source_score_proxy_v69"] = (
        u["expected_return_proxy_v69"]
        - 0.80 * u["expected_loss_proxy_v69"]
        - 3300.0 * u["weak_source_proxy"].astype(float)
        - 1500.0 * u["qhat_v4"].astype(float)
    )
    u["return_norm_v69"] = normalize(u["expected_return_proxy_v69"])
    u["tail_norm_v69"] = normalize(u["tail_score_proxy_v69"])
    u["weak_source_norm_v69"] = normalize(u["weak_source_proxy"], higher_is_better=False)
    return u


def _policy_source_map(concentration: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    local = concentration.loc[concentration["policy_id"].eq(policy_id)].copy()
    if local.empty:
        return pd.DataFrame()
    local["over_cap_v69"] = pd.to_numeric(
        local["top_exposure_share_v63"], errors="coerce"
    ) > pd.to_numeric(local["target_cap_v63"], errors="coerce")
    return local


def _active_source_map(source_map: pd.DataFrame) -> pd.DataFrame:
    if source_map.empty:
        return source_map
    active = source_map.loc[source_map["over_cap_v69"].astype(bool)].copy()
    return active if not active.empty else source_map.copy()


def _source_relief(frame: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    relief = pd.Series(0.0, index=frame.index)
    active = _active_source_map(source_map)
    for _, row in active.iterrows():
        family = str(row["source_family"])
        if family not in frame:
            continue
        top_source = str(row["top_source_id_v63"])
        relief = relief + frame[family].astype(str).ne(top_source).astype(float)
    denom = max(int(len(active)), 1)
    return pd.DataFrame(
        {
            "source_relief_hits_v69": relief,
            "source_relief_share_v69": relief / denom,
            "active_source_constraints_v69": int(len(active)),
        },
        index=frame.index,
    )


def _drop_pressure(frame: pd.DataFrame, source_map: pd.DataFrame) -> pd.Series:
    pressure = pd.Series(0.0, index=frame.index)
    for _, row in _active_source_map(source_map).iterrows():
        family = str(row["source_family"])
        if family not in frame:
            continue
        pressure = pressure + frame[family].astype(str).eq(str(row["top_source_id_v63"]))
    return pressure


def _score_policy_rows(frame: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    relief = _source_relief(scored, source_map)
    scored = pd.concat([scored, relief], axis=1)
    scored["pricing_screen_score_v69"] = (
        0.42 * scored["return_norm_v69"]
        + 0.32 * scored["tail_norm_v69"]
        + 0.20 * scored["source_relief_share_v69"]
        + 0.06 * scored["weak_source_norm_v69"]
    )
    scored["drop_pressure_v69"] = _drop_pressure(scored, source_map)
    return scored


def _max_source_slack(frame: pd.DataFrame, source_map: pd.DataFrame) -> tuple[float, float]:
    exposure = float(pd.to_numeric(frame["loan_amnt"], errors="coerce").fillna(0.0).sum())
    max_share = 0.0
    max_slack = 0.0
    for _, row in source_map.iterrows():
        family = str(row["source_family"])
        if family not in frame:
            continue
        by_source = frame.groupby(family, dropna=False)["loan_amnt"].sum().astype(float) / max(
            exposure, 1.0
        )
        top_share = float(by_source.max()) if not by_source.empty else 0.0
        cap = float(row["target_cap_v63"])
        max_share = max(max_share, top_share)
        max_slack = max(max_slack, max(0.0, top_share - cap))
    return max_share, max_slack


def _policy_frames(
    universe: pd.DataFrame,
    books: pd.DataFrame,
    v68_candidates: pd.DataFrame,
    concentration: pd.DataFrame,
    policy_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_map = _policy_source_map(concentration, policy_id)
    if source_map.empty:
        return pd.DataFrame(), pd.DataFrame(), source_map

    book_ids = set(books.loc[books["policy_id_v63"].eq(policy_id), "loan_id"].astype(str))
    candidate_rows = v68_candidates.loc[v68_candidates["policy_id_v68"].eq(policy_id)].copy()
    candidate_ids = set(candidate_rows["loan_id"].astype(str))

    book = universe.loc[universe["loan_id"].astype(str).isin(book_ids)].copy()
    candidates = universe.loc[universe["loan_id"].astype(str).isin(candidate_ids)].copy()
    if not candidates.empty and not candidate_rows.empty:
        candidates = candidates.merge(
            candidate_rows[
                [
                    "loan_id",
                    "candidate_rank_v68",
                    "pricing_screen_score_v68",
                    "source_relief_share_v68",
                ]
            ].assign(loan_id=lambda df: df["loan_id"].astype(str)),
            on="loan_id",
            how="left",
        )

    book = _score_policy_rows(book, source_map)
    candidates = _score_policy_rows(candidates, source_map)
    book["policy_id_v69"] = policy_id
    candidates["policy_id_v69"] = policy_id
    return book, candidates, source_map


def _candidate_pack(candidates: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    out = candidates.copy()
    out["source_policy_id_v63"] = policy_id
    out["recommended_for_restricted_master_v69"] = out["source_relief_share_v69"].gt(0) & out[
        "candidate_rank_v68"
    ].le(50)
    out["claim_boundary_v69"] = (
        "v69 restricted-master candidate only; exact full-universe pricing remains unproven"
    )
    keep = [
        "policy_id_v69",
        "source_policy_id_v63",
        "loan_index_v55",
        "loan_id",
        "candidate_rank_v68",
        "loan_amnt",
        "grade",
        "score_decile",
        "income_band",
        "dti_band",
        "period",
        "state_top20",
        "expected_loss_proxy_v69",
        "expected_return_proxy_v69",
        "tail_score_proxy_v69",
        "source_score_proxy_v69",
        "source_relief_share_v69",
        "pricing_screen_score_v69",
        "pricing_screen_score_v68",
        "recommended_for_restricted_master_v69",
        "claim_boundary_v69",
    ]
    return out[[col for col in keep if col in out.columns]].sort_values(
        ["policy_id_v69", "candidate_rank_v68"]
    )


def _expanded_master(book: pd.DataFrame, candidates: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    incumbent = book.copy()
    incumbent["master_role_v69"] = "incumbent_v63_book"
    incumbent["candidate_rank_v68"] = np.nan
    incumbent["pricing_screen_score_v68"] = np.nan
    incumbent["source_policy_id_v63"] = policy_id
    challenger = candidates.copy()
    challenger["master_role_v69"] = "v68_pricing_candidate"
    challenger["source_policy_id_v63"] = policy_id
    master = pd.concat([incumbent, challenger], ignore_index=True)
    master["restricted_master_scope_v69"] = (
        "v63 incumbent book plus v68 out-of-book source/pricing candidates"
    )
    master["exact_column_generation_certificate_v69"] = False
    master["claim_boundary_v69"] = (
        "expanded restricted master input only; not exact full-universe CVaR proof"
    )
    keep = [
        "policy_id_v69",
        "source_policy_id_v63",
        "master_role_v69",
        "loan_index_v55",
        "loan_id",
        "candidate_rank_v68",
        "loan_amnt",
        "grade",
        "score_decile",
        "income_band",
        "dti_band",
        "period",
        "state_top20",
        "expected_loss_proxy_v69",
        "expected_return_proxy_v69",
        "tail_score_proxy_v69",
        "source_score_proxy_v69",
        "source_relief_share_v69",
        "drop_pressure_v69",
        "pricing_screen_score_v69",
        "pricing_screen_score_v68",
        "restricted_master_scope_v69",
        "exact_column_generation_certificate_v69",
        "claim_boundary_v69",
    ]
    return master[[col for col in keep if col in master.columns]]


def _swap_audit(
    book: pd.DataFrame,
    candidates: pd.DataFrame,
    source_map: pd.DataFrame,
    policy_id: str,
    max_swaps: int = 25,
) -> pd.DataFrame:
    if book.empty or candidates.empty:
        return pd.DataFrame()
    before_share, before_slack = _max_source_slack(book, source_map)
    current_exposure = float(book["loan_amnt"].astype(float).sum())
    eligible_drop = book.loc[book["drop_pressure_v69"].gt(0)].copy()
    if eligible_drop.empty:
        eligible_drop = book.copy()
    eligible_drop = eligible_drop.sort_values(
        ["drop_pressure_v69", "pricing_screen_score_v69", "expected_return_proxy_v69"],
        ascending=[False, True, True],
    )
    ranked_candidates = candidates.sort_values(
        ["candidate_rank_v68", "pricing_screen_score_v69"], ascending=[True, False]
    )
    rows: list[dict[str, Any]] = []
    used_drops: set[str] = set()
    working_exposure = current_exposure
    for _, cand in ranked_candidates.iterrows():
        drop_pool = eligible_drop.loc[~eligible_drop["loan_id"].astype(str).isin(used_drops)]
        if drop_pool.empty:
            break
        feasible = drop_pool.loc[
            working_exposure - drop_pool["loan_amnt"].astype(float) + float(cand["loan_amnt"])
            <= BUDGET
        ]
        if feasible.empty:
            feasible = drop_pool
        drop = feasible.iloc[0]
        after_exposure = working_exposure - float(drop["loan_amnt"]) + float(cand["loan_amnt"])
        swapped = pd.concat(
            [
                book.loc[book["loan_id"].astype(str).ne(str(drop["loan_id"]))],
                cand.to_frame().T,
            ],
            ignore_index=True,
        )
        after_share, after_slack = _max_source_slack(swapped, source_map)
        rows.append(
            {
                "policy_id": policy_id,
                "swap_rank_v69": len(rows) + 1,
                "add_loan_id_v69": cand["loan_id"],
                "drop_loan_id_v69": drop["loan_id"],
                "add_candidate_rank_v68": int(cand["candidate_rank_v68"]),
                "add_pricing_score_v69": float(cand["pricing_screen_score_v69"]),
                "drop_pricing_score_v69": float(drop["pricing_screen_score_v69"]),
                "delta_pricing_score_v69": float(
                    cand["pricing_screen_score_v69"] - drop["pricing_screen_score_v69"]
                ),
                "delta_expected_return_proxy_v69": float(
                    cand["expected_return_proxy_v69"] - drop["expected_return_proxy_v69"]
                ),
                "delta_expected_loss_proxy_v69": float(
                    cand["expected_loss_proxy_v69"] - drop["expected_loss_proxy_v69"]
                ),
                "delta_exposure_v69": float(cand["loan_amnt"] - drop["loan_amnt"]),
                "book_exposure_before_swap_v69": current_exposure,
                "book_exposure_after_sequential_swap_v69": after_exposure,
                "max_source_share_before_v69": before_share,
                "max_source_share_after_swap_v69": after_share,
                "max_source_slack_before_v69": before_slack,
                "max_source_slack_after_swap_v69": after_slack,
                "delta_max_source_slack_v69": after_slack - before_slack,
                "source_relief_gain_proxy_v69": float(
                    cand["source_relief_share_v69"] - drop["source_relief_share_v69"]
                ),
                "valid_budget_after_swap_v69": after_exposure <= BUDGET,
                "claim_boundary_v69": (
                    "heuristic one-for-one swap audit; exact solver must re-optimize jointly"
                ),
            }
        )
        used_drops.add(str(drop["loan_id"]))
        working_exposure = after_exposure
        if len(rows) >= max_swaps:
            break
    return pd.DataFrame(rows)


def _protocol() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_step_v69": 1,
                "step_name_v69": "freeze_inputs",
                "artifact_v69": "paper4_v69_expanded_restricted_master.parquet",
                "required_evidence_v69": "v63 incumbent books and v68 candidates are immutable inputs",
                "claim_if_missing_v69": "no exact pricing claim",
                "locked_v69": True,
            },
            {
                "protocol_step_v69": 2,
                "step_name_v69": "solve_expanded_restricted_master",
                "artifact_v69": "paper4_v70_restricted_master_solver_frontier.csv",
                "required_evidence_v69": "exact LP/MILP solution over expanded master with persisted primal allocation",
                "claim_if_missing_v69": "restricted-master improvement remains unproven",
                "locked_v69": True,
            },
            {
                "protocol_step_v69": 3,
                "step_name_v69": "persist_duals",
                "artifact_v69": "paper4_v70_restricted_master_duals.csv",
                "required_evidence_v69": "budget, source, return floor and CVaR duals are persisted",
                "claim_if_missing_v69": "cannot price omitted full-universe columns",
                "locked_v69": True,
            },
            {
                "protocol_step_v69": 4,
                "step_name_v69": "price_omitted_universe",
                "artifact_v69": "paper4_v70_full_universe_reduced_costs.parquet",
                "required_evidence_v69": "all omitted v55 comparable loans receive reduced-cost scores",
                "claim_if_missing_v69": "no full-universe termination claim",
                "locked_v69": True,
            },
            {
                "protocol_step_v69": 5,
                "step_name_v69": "iterate_negative_reduced_cost_columns",
                "artifact_v69": "paper4_v70_column_generation_iteration_log.csv",
                "required_evidence_v69": "new improving columns are added until tolerance is met",
                "claim_if_missing_v69": "column-generation loop is incomplete",
                "locked_v69": True,
            },
            {
                "protocol_step_v69": 6,
                "step_name_v69": "run_common_path_dynamic_gate",
                "artifact_v69": "paper4_v70_dynamic_gate_or_replay.csv",
                "required_evidence_v69": "candidate survives existing dynamic, online and source gates",
                "claim_if_missing_v69": "no working-champion or deployment claim",
                "locked_v69": True,
            },
            {
                "protocol_step_v69": 7,
                "step_name_v69": "claim_boundary_review",
                "artifact_v69": "paper4_v70_claim_boundary_review.csv",
                "required_evidence_v69": "editorial review maps each claim to exact artifacts",
                "claim_if_missing_v69": "keep result in living notebook only",
                "locked_v69": True,
            },
        ]
    )


def build_v69_expansion() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = _prepare_universe()
    books = read_parquet("paper4_v63_source_repair_candidate_books.parquet")
    v68_candidates = read_parquet("paper4_v68_full_universe_candidate_screen.parquet")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    if universe.empty or books.empty or v68_candidates.empty or concentration.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, _protocol()

    candidate_frames: list[pd.DataFrame] = []
    master_frames: list[pd.DataFrame] = []
    swap_frames: list[pd.DataFrame] = []
    policy_ids = sorted(v68_candidates["policy_id_v68"].dropna().astype(str).unique())
    for policy_id in policy_ids:
        book, candidates, source_map = _policy_frames(
            universe, books, v68_candidates, concentration, policy_id
        )
        if not candidates.empty:
            candidate_frames.append(_candidate_pack(candidates, policy_id))
        if not book.empty or not candidates.empty:
            master_frames.append(_expanded_master(book, candidates, policy_id))
        swaps = _swap_audit(book, candidates, source_map, policy_id)
        if not swaps.empty:
            swap_frames.append(swaps)

    candidates_out = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    )
    master_out = pd.concat(master_frames, ignore_index=True) if master_frames else pd.DataFrame()
    swaps_out = pd.concat(swap_frames, ignore_index=True) if swap_frames else pd.DataFrame()
    return candidates_out, master_out, swaps_out, _protocol()


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v69 restricted-master expansion pack for column-generation follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v69_expanded_restricted_master.parquet"
                ),
                "boundary": "Executable input/protocol only; not solver termination or full-universe proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v69 is an exact full-universe column-generation certificate.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v69_exact_column_generation_protocol.csv"
                ),
                "boundary": "Requires exact restricted-master solve, duals and reduced-cost termination.",
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
                    "v69 packages v68 source/pricing columns into an expanded restricted master "
                    "and freezes the exact column-generation protocol."
                ),
                "status": "ready_for_solver_attempt",
                "next_artifact": "paper4_v70_restricted_master_solver_frontier.csv",
                "success_condition": "exact expanded-master solver plus dual pricing certificate exists",
                "last_wave": "v69",
                "execution_result": "restricted_master_expansion_pack_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": (
                    "Use the v69 swap audit to seed exact source-aware re-optimization "
                    "without promoting heuristic swaps."
                ),
                "status": "gated",
                "next_artifact": "paper4_v70_source_aware_solver_swap_check.csv",
                "success_condition": "joint solver improves source slack or proves no feasible improvement",
                "last_wave": "v69",
                "execution_result": "heuristic_swap_audit_completed",
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
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    existing = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
    start = "<!-- V69_RESTRICTED_MASTER_EXPANSION_START -->"
    end = "<!-- V69_RESTRICTED_MASTER_EXPANSION_END -->"
    block = f"""
{start}

## Wave v69: Restricted-Master Expansion Pack

Generated: {status["generated_at_utc"]}

### Objective

Turn the v68 source/pricing screen into executable inputs for the next exact
solver attempt: an expanded restricted master, a source-aware swap audit and a
locked column-generation protocol.

### Results

- Candidate rows: `{status["candidate_rows_v69"]}`.
- Expanded restricted-master rows: `{status["expanded_master_rows_v69"]}`.
- Heuristic swap audit rows: `{status["swap_audit_rows_v69"]}`.
- Protocol rows: `{status["protocol_rows_v69"]}`.
- Restricted-master pack ready: `{status["restricted_master_pack_ready_v69"]}`.
- Exact column-generation certificate: `{status["exact_column_generation_certificate_v69"]}`.

### Interpretation

v69 converts the v68 negative/proxy finding into a concrete solver queue. It
does not certify global optimality, but it removes ambiguity about which
columns should enter the next restricted-master solve and what evidence would
be required to terminate a full-universe pricing loop.

### Claim Impact

- Allowed: v69 creates an executable restricted-master expansion pack.
- Still prohibited: exact full-universe CVaR optimality, column-generation
  termination, Paper Estrella replacement, final Paper 4 promotion and live
  deployment claims.

### Quarto Promotion Decision

Keep v69 in the living notebook. Promote only after v70 or later produces
exact solver, dual-pricing and claim-boundary evidence.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v69() -> dict[str, Any]:
    started = datetime.now(UTC)
    candidates, master, swaps, protocol = build_v69_expansion()
    candidates.to_parquet(
        TABLE_DIR / "paper4_v69_source_pricing_expansion_candidates.parquet",
        index=False,
        compression="zstd",
    )
    master.to_parquet(
        TABLE_DIR / "paper4_v69_expanded_restricted_master.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v69_candidate_swap_audit.csv", swaps)
    write_csv(TABLE_DIR / "paper4_v69_exact_column_generation_protocol.csv", protocol)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v69_restricted_master_expansion_pack_exists",
                "allowed": True,
                "artifact": "paper4_v69_expanded_restricted_master.parquet",
                "boundary": "input pack and protocol only",
            },
            {
                "claim_id": "v69_exact_column_generation_certificate",
                "allowed": False,
                "artifact": "paper4_v69_exact_column_generation_protocol.csv",
                "boundary": "requires exact solve, duals and reduced-cost termination",
            },
            {
                "claim_id": "v69_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v69_candidate_swap_audit.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v69_claim_matrix_delta.csv", claim_matrix)
    status = {
        "phase": "v69_restricted_master_expansion_protocol",
        "schema_version": "2026-05-15.69",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "candidate_rows_v69": int(len(candidates)),
        "expanded_master_rows_v69": int(len(master)),
        "swap_audit_rows_v69": int(len(swaps)),
        "protocol_rows_v69": int(len(protocol)),
        "policies_ready_for_restricted_solver_v69": int(
            candidates.groupby("policy_id_v69")["recommended_for_restricted_master_v69"].any().sum()
        )
        if not candidates.empty
        else 0,
        "positive_swap_score_rows_v69": int(swaps["delta_pricing_score_v69"].gt(0).sum())
        if not swaps.empty
        else 0,
        "restricted_master_pack_ready_v69": bool(not candidates.empty and not master.empty),
        "exact_column_generation_certificate_v69": False,
        "exact_full_universe_cvar_claim_allowed_v69": False,
        "paper1_promotion_allowed_v69": False,
        "paper4_working_champion_changed_v69": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v69 creates restricted-master inputs and protocol only; exact full-universe "
            "CVaR remains blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v69_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v69": build_v69()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
