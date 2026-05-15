#!/usr/bin/env python3
"""Build Paper 4 v63 source-governance repair artifacts.

This wave follows the v59-v62 finding: tail-feasible restricted-master CVaR
books exist, but they collapse into concentrated low-risk sources.  v63 does
not claim a new optimizer.  It builds auditable whole-loan repair candidates
that trade tail loss against source concentration and decides whether any
candidate is strong enough to deserve an expensive dynamic replay.
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


SPECS: list[dict[str, Any]] = [
    {
        "regime_v63": "return_guarded_repair",
        "target_exposure_v63": 850_000.0,
        "min_deployment_v63": 0.78,
        "loss_weight_v63": 0.90,
        "qhat_weight_v63": 1_100.0,
        "weak_weight_v63": 1_300.0,
        "caps_v63": {
            "grade": 0.85,
            "score_decile": 0.90,
            "income_band": 0.85,
            "dti_band": 0.85,
            "period": 0.82,
            "state_top20": 0.35,
        },
    },
    {
        "regime_v63": "tail_balanced_repair",
        "target_exposure_v63": 850_000.0,
        "min_deployment_v63": 0.70,
        "loss_weight_v63": 1.15,
        "qhat_weight_v63": 1_800.0,
        "weak_weight_v63": 2_000.0,
        "caps_v63": {
            "grade": 0.78,
            "score_decile": 0.82,
            "income_band": 0.72,
            "dti_band": 0.72,
            "period": 0.72,
            "state_top20": 0.30,
        },
    },
    {
        "regime_v63": "diversity_first_repair",
        "target_exposure_v63": 850_000.0,
        "min_deployment_v63": 0.58,
        "loss_weight_v63": 1.25,
        "qhat_weight_v63": 2_250.0,
        "weak_weight_v63": 2_600.0,
        "caps_v63": {
            "grade": 0.68,
            "score_decile": 0.72,
            "income_band": 0.62,
            "dti_band": 0.62,
            "period": 0.68,
            "state_top20": 0.26,
        },
    },
    {
        "regime_v63": "strict_source_probe",
        "target_exposure_v63": 850_000.0,
        "min_deployment_v63": 0.50,
        "loss_weight_v63": 1.35,
        "qhat_weight_v63": 2_600.0,
        "weak_weight_v63": 3_000.0,
        "caps_v63": {
            "grade": 0.58,
            "score_decile": 0.64,
            "income_band": 0.55,
            "dti_band": 0.55,
            "period": 0.62,
            "state_top20": 0.24,
        },
    },
]


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


def _select_master_pool(
    universe: pd.DataFrame, max_columns: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = universe.copy()
    u["expected_loss_proxy_v63"] = u["loan_amnt"] * u["lgd_proxy_v55"] * u["pd_high_alpha01"]
    u["expected_return_proxy_v63"] = (
        u["base_return_vec"] - u["expected_loss_proxy_v63"] - 0.012 * u["loan_amnt"]
    )
    u["return_score_v63"] = u["expected_return_proxy_v63"] - 0.25 * u["expected_loss_proxy_v63"]
    u["tail_score_v63"] = (
        u["expected_return_proxy_v63"]
        - 1.40 * u["expected_loss_proxy_v63"]
        - 1800 * u["qhat_v4"]
        - 2200 * u["weak_source_proxy"]
    )
    u["source_score_v63"] = (
        u["expected_return_proxy_v63"]
        - 0.80 * u["expected_loss_proxy_v63"]
        - 3300 * u["weak_source_proxy"]
        - 1500 * u["qhat_v4"]
    )
    initial = pd.concat(
        [
            u.nlargest(max_columns // 3, "return_score_v63"),
            u.nlargest(max_columns // 3, "tail_score_v63"),
            u.nlargest(max_columns // 3, "source_score_v63"),
        ],
        ignore_index=True,
    ).drop_duplicates("loan_id")
    if len(initial) < max_columns:
        initial = pd.concat(
            [initial, u.nlargest(max_columns, "expected_return_proxy_v63")],
            ignore_index=True,
        ).drop_duplicates("loan_id")
    initial = initial.head(max_columns).copy().reset_index(drop=True)
    logs = pd.DataFrame(
        [
            {
                "round_v63": 0,
                "selection_rule_v63": "tri_score_warm_start_reused_for_source_repair",
                "columns_before_v63": 0,
                "columns_after_v63": int(len(initial)),
                "new_columns_v63": int(len(initial)),
                "pricing_tolerance_v63": "heuristic_score_not_dual_exact",
            }
        ]
    )
    return initial, logs


def _scenario_factors(n_paths: int = 128) -> pd.DataFrame:
    paths = read_parquet("paper4_v31_sample_paths.parquet")
    if paths.empty:
        return pd.DataFrame()
    path_ids = sorted(paths["path_id"].drop_duplicates().astype(int).tolist())[:n_paths]
    p = paths.loc[paths["path_id"].isin(path_ids)].copy()
    p["issue_month"] = pd.to_datetime(p["month"], errors="coerce").dt.to_period("M").astype(str)
    keep = [
        "path_id",
        "issue_month",
        "macro_regime_v15",
        "path_family_v19",
        "default_factor_v15",
        "lgd_factor_v15",
        "prepay_factor_v15",
    ]
    p = p[keep].drop_duplicates(["path_id", "issue_month"])
    fallback = (
        p.sort_values("issue_month")
        .groupby("path_id", dropna=False)
        .head(1)
        .assign(issue_month="__fallback__")
    )
    return pd.concat([p, fallback], ignore_index=True)


def _expected_by_path_matrix(
    universe: pd.DataFrame, n_paths: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    factors = _scenario_factors(n_paths)
    path_ids = sorted(factors["path_id"].drop_duplicates().astype(int).tolist())[:n_paths]
    losses = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    returns = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    default_probs = np.zeros((len(path_ids), len(universe)), dtype=np.float64)
    fallback = factors.loc[factors["issue_month"].eq("__fallback__")].copy()
    base = universe[
        [
            "loan_id",
            "issue_month",
            "loan_amnt",
            "pd_high_alpha01",
            "lgd_proxy_v55",
            "base_return_vec",
        ]
    ].copy()
    for row_idx, path_id in enumerate(path_ids):
        f = factors.loc[factors["path_id"].eq(path_id)].drop(columns=["path_id"])
        merged = base.merge(f, on="issue_month", how="left")
        missing = merged["default_factor_v15"].isna()
        if missing.any():
            fb = fallback.loc[fallback["path_id"].eq(path_id)].head(1)
            for col in [
                "macro_regime_v15",
                "path_family_v19",
                "default_factor_v15",
                "lgd_factor_v15",
                "prepay_factor_v15",
            ]:
                merged.loc[missing, col] = fb[col].iloc[0] if not fb.empty else 1.0
        dp = (
            pd.to_numeric(merged["pd_high_alpha01"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["default_factor_v15"], errors="coerce").fillna(1.0)
        ).clip(0, 0.95)
        lgd_factor = (
            pd.to_numeric(merged["lgd_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        prepay_factor = (
            pd.to_numeric(merged["prepay_factor_v15"], errors="coerce").fillna(1.0).clip(0.25, 2.5)
        )
        expected_loss = (
            pd.to_numeric(merged["loan_amnt"], errors="coerce").fillna(0.0)
            * pd.to_numeric(merged["lgd_proxy_v55"], errors="coerce").fillna(0.45)
            * dp
            * lgd_factor
        )
        prepay_drag = (
            pd.to_numeric(merged["loan_amnt"], errors="coerce").fillna(0.0)
            * 0.012
            * (1 - dp)
            * prepay_factor
        )
        losses[row_idx, :] = expected_loss.to_numpy(float)
        returns[row_idx, :] = (
            pd.to_numeric(merged["base_return_vec"], errors="coerce").fillna(0.0)
            - expected_loss
            - prepay_drag
        ).to_numpy(float)
        default_probs[row_idx, :] = dp.to_numpy(float)
    return losses, returns, default_probs, path_ids


def _tail_cvar(values: np.ndarray, alpha: float = 0.90) -> float:
    if values.size == 0:
        return 0.0
    tail_n = max(1, int(np.ceil((1.0 - alpha) * values.size)))
    return float(np.sort(values)[-tail_n:].mean())


def _source_concentration(
    book: pd.DataFrame, exposure_col: str, caps: dict[str, float], policy_id: str
) -> tuple[pd.DataFrame, float, float]:
    rows: list[dict[str, Any]] = []
    exposure = float(book[exposure_col].sum())
    max_share = 0.0
    max_slack = 0.0
    for family in FAMILIES:
        if family not in book:
            continue
        by_source = book.groupby(family, dropna=False)[exposure_col].sum() / max(exposure, 1.0)
        top_source = str(by_source.idxmax())
        top_share = float(by_source.max())
        cap = float(caps.get(family, 1.0))
        slack = max(0.0, top_share - cap)
        max_share = max(max_share, top_share)
        max_slack = max(max_slack, slack)
        rows.append(
            {
                "policy_id": policy_id,
                "source_family": family,
                "top_source_id_v63": top_source,
                "top_exposure_share_v63": top_share,
                "target_cap_v63": cap,
                "required_cap_slack_share_v63": slack,
                "source_repair_pass_v63": slack <= 1e-9,
                "certificate_scope_v63": "whole-loan heuristic repair concentration certificate",
            }
        )
    return pd.DataFrame(rows), max_share, max_slack


def _build_book(
    pool: pd.DataFrame,
    losses: np.ndarray,
    returns_by_path: np.ndarray,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_exposure = float(spec["target_exposure_v63"])
    min_deployment = float(spec["min_deployment_v63"])
    caps = dict(spec["caps_v63"])
    returns = returns_by_path.mean(axis=0)
    loss_mean = losses.mean(axis=0)
    amounts = pool["loan_amnt"].to_numpy(float)
    score = (
        returns
        - float(spec["loss_weight_v63"]) * loss_mean
        - float(spec["qhat_weight_v63"]) * pool["qhat_v4"].to_numpy(float)
        - float(spec["weak_weight_v63"]) * pool["weak_source_proxy"].to_numpy(float)
    )
    ranked = np.argsort(-score)
    chosen: list[int] = []
    exposure_by_family: dict[tuple[str, str], float] = {}
    total = 0.0
    cap_relaxation = 1.0
    for relaxation in [1.0, 1.08, 1.18, 1.32]:
        chosen.clear()
        exposure_by_family.clear()
        total = 0.0
        cap_relaxation = relaxation
        for idx in ranked:
            amount = float(amounts[idx])
            if amount <= 0 or total + amount > target_exposure:
                continue
            violates = False
            for family, cap in caps.items():
                key = (family, str(pool.iloc[idx][family]))
                next_exposure = exposure_by_family.get(key, 0.0) + amount
                if next_exposure > cap * target_exposure * relaxation:
                    violates = True
                    break
            if violates:
                continue
            chosen.append(int(idx))
            total += amount
            for family in caps:
                key = (family, str(pool.iloc[idx][family]))
                exposure_by_family[key] = exposure_by_family.get(key, 0.0) + amount
            if total >= target_exposure * 0.995:
                break
        if total >= min_deployment * BUDGET:
            break

    policy_id = f"v63_source_repair_{spec['regime_v63']}"
    if not chosen:
        status = {
            "policy_id": policy_id,
            "regime_v63": spec["regime_v63"],
            "source_repair_success_v63": False,
            "target_exposure_v63": target_exposure,
            "allocated_exposure_v63": 0.0,
            "n_allocated_loans_v63": 0,
            "cap_relaxation_used_v63": cap_relaxation,
            "claim_boundary_v63": "failed heuristic source-governance repair; no optimization or promotion claim",
        }
        return status, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    selected = pool.iloc[chosen].copy()
    selected["allocation_fraction_v63"] = 1.0
    selected["allocated_exposure_v63"] = selected["loan_amnt"].astype(float)
    selected["policy_id_v63"] = policy_id
    selected["regime_v63"] = spec["regime_v63"]
    selected["claim_boundary_v63"] = (
        "whole-loan heuristic source-governance repair over expanded restricted master; "
        "not exact LP optimality"
    )
    scenario_losses = losses[:, chosen].sum(axis=1)
    scenario_returns = returns_by_path[:, chosen].sum(axis=1)
    concentration, max_share, max_slack = _source_concentration(
        selected, "allocated_exposure_v63", caps, policy_id
    )
    scenario = pd.DataFrame(
        {
            "policy_id": policy_id,
            "regime_v63": spec["regime_v63"],
            "scenario_row": np.arange(len(scenario_losses)),
            "scenario_loss_v63": scenario_losses,
            "scenario_return_v63": scenario_returns,
        }
    )
    status = {
        "policy_id": policy_id,
        "regime_v63": spec["regime_v63"],
        "source_repair_success_v63": total >= min_deployment * BUDGET,
        "target_exposure_v63": target_exposure,
        "allocated_exposure_v63": float(selected["allocated_exposure_v63"].sum()),
        "n_allocated_loans_v63": int(len(selected)),
        "cap_relaxation_used_v63": cap_relaxation,
        "objective_return_v63": float(scenario_returns.mean()),
        "scenario_return_p05_v63": float(np.quantile(scenario_returns, 0.05)),
        "scenario_loss_mean_v63": float(scenario_losses.mean()),
        "scenario_loss_p95_v63": float(np.quantile(scenario_losses, 0.95)),
        "scenario_loss_cvar90_v63": _tail_cvar(scenario_losses),
        "max_source_share_v63": max_share,
        "max_required_cap_slack_share_v63": max_slack,
        "grade_top_share_v63": float(
            concentration.loc[
                concentration["source_family"].eq("grade"), "top_exposure_share_v63"
            ].iloc[0]
        ),
        "score_top_share_v63": float(
            concentration.loc[
                concentration["source_family"].eq("score_decile"), "top_exposure_share_v63"
            ].iloc[0]
        ),
        "source_repair_pass_v63": max_slack <= 0.08,
        "exact_full_universe_claim_v63": False,
        "paper1_promotion_allowed_v63": False,
        "claim_boundary_v63": selected["claim_boundary_v63"].iloc[0],
    }
    return status, selected, scenario, concentration


def build_v63_source_repair(max_columns: int = 36_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet")
    if universe.empty:
        empty = pd.DataFrame()
        write_csv(TABLE_DIR / "paper4_v63_source_repair_frontier.csv", empty)
        empty.to_parquet(
            TABLE_DIR / "paper4_v63_source_repair_candidate_books.parquet", index=False
        )
        return empty, empty
    pool, logs = _select_master_pool(universe, max_columns=max_columns)
    logs = logs.assign(restricted_master_columns_v63=len(pool))
    write_csv(TABLE_DIR / "paper4_v63_source_repair_column_log.csv", logs)
    losses, returns_by_path, _default_probs, _path_ids = _expected_by_path_matrix(pool)
    rows: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    scenarios: list[pd.DataFrame] = []
    concentrations: list[pd.DataFrame] = []
    for spec in SPECS:
        status, book, scenario, concentration = _build_book(pool, losses, returns_by_path, spec)
        rows.append(status)
        if not book.empty:
            books.append(book)
        if not scenario.empty:
            scenarios.append(scenario)
        if not concentration.empty:
            concentrations.append(concentration)
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        frontier["return_norm_v63"] = normalize(frontier["objective_return_v63"])
        frontier["tail_norm_v63"] = normalize(
            frontier["scenario_loss_cvar90_v63"], higher_is_better=False
        )
        frontier["diversity_norm_v63"] = normalize(
            frontier["max_required_cap_slack_share_v63"], higher_is_better=False
        )
        frontier["deploy_norm_v63"] = normalize(frontier["allocated_exposure_v63"])
        frontier["repair_score_v63"] = (
            0.32 * frontier["return_norm_v63"]
            + 0.30 * frontier["tail_norm_v63"]
            + 0.28 * frontier["diversity_norm_v63"]
            + 0.10 * frontier["deploy_norm_v63"]
        )
        frontier["non_dominated_repair_v63"] = frontier["source_repair_success_v63"].astype(
            bool
        ) & (frontier["repair_score_v63"] >= frontier["repair_score_v63"].median())
    book_out = pd.concat(books, ignore_index=True) if books else pd.DataFrame()
    scenario_out = pd.concat(scenarios, ignore_index=True) if scenarios else pd.DataFrame()
    concentration_out = (
        pd.concat(concentrations, ignore_index=True) if concentrations else pd.DataFrame()
    )
    write_csv(TABLE_DIR / "paper4_v63_source_repair_frontier.csv", frontier)
    book_out.to_parquet(
        TABLE_DIR / "paper4_v63_source_repair_candidate_books.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(TABLE_DIR / "paper4_v63_source_repair_scenario_losses.csv", scenario_out)
    write_csv(TABLE_DIR / "paper4_v63_source_repair_concentration.csv", concentration_out)
    return frontier, book_out


def build_v63_gate_and_docs(frontier: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    gate = pd.DataFrame()
    if not frontier.empty:
        gate = frontier.loc[frontier["source_repair_success_v63"].astype(bool)].copy()
        if not gate.empty:
            gate["dynamic_512_or_1024_rerun_recommended_v63"] = (
                gate["objective_return_v63"].ge(35_000)
                & gate["scenario_loss_cvar90_v63"].le(85_000)
                & gate["max_required_cap_slack_share_v63"].le(0.08)
            )
            gate["working_champion_change_allowed_v63"] = False
            gate["rerun_decision_reason_v63"] = np.where(
                gate["dynamic_512_or_1024_rerun_recommended_v63"],
                "source-repaired candidate clears lab tail/diversity gate and merits dynamic replay",
                "source-repaired candidate remains diagnostic because return, tail, or source slack gate is insufficient",
            )
            gate = gate[
                [
                    "policy_id",
                    "regime_v63",
                    "objective_return_v63",
                    "scenario_loss_cvar90_v63",
                    "max_required_cap_slack_share_v63",
                    "dynamic_512_or_1024_rerun_recommended_v63",
                    "working_champion_change_allowed_v63",
                    "rerun_decision_reason_v63",
                    "claim_boundary_v63",
                ]
            ]
    write_csv(TABLE_DIR / "paper4_v63_dynamic_gate_memo.csv", gate)

    claims = pd.DataFrame(
        [
            {
                "claim_id": "v63_source_repair_frontier_exists",
                "allowed": True,
                "artifact": "paper4_v63_source_repair_frontier.csv",
                "boundary": "heuristic whole-loan repair over expanded restricted master; not exact optimizer",
            },
            {
                "claim_id": "v63_source_repaired_candidate_promotable",
                "allowed": False,
                "artifact": "paper4_v63_dynamic_gate_memo.csv",
                "boundary": "no Paper Estrella replacement and no final Paper 4 promotion from v63",
            },
            {
                "claim_id": "v63_exact_full_universe_cvar",
                "allowed": False,
                "artifact": "paper4_v63_source_repair_frontier.csv",
                "boundary": "candidate books are restricted-master heuristic repairs, not full-universe LP proof",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v63_claim_matrix_delta.csv", claims)

    backlog = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Source governance",
                "executable_item": "v63 builds whole-loan source-repair candidates after v59-v62 concentration diagnostics.",
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_v64_dynamic_replay_for_v63_if_gate_passes.csv",
                "success_condition": "a source-repaired candidate passes tail/diversity/return gate and then survives common-path dynamic replay",
                "last_wave": "v63",
                "execution_result": "source_repair_frontier_completed",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
            {
                "horizon": "short",
                "lane": "Dynamic stress",
                "executable_item": "Run expensive dynamic replay only if v63 dynamic gate recommends it.",
                "status": "gated",
                "next_artifact": "paper4_v64_dynamic_gate_or_replay.csv",
                "success_condition": "rerun is justified by candidate plausibly changing working champion decision",
                "last_wave": "v63",
                "execution_result": "dynamic_gate_created",
                "quarto_promotion_decision": "not_promoted_to_quarto",
            },
        ]
    )
    combined = (
        pd.concat([backlog, additions], ignore_index=True) if not backlog.empty else additions
    )
    combined = combined.drop_duplicates(["horizon", "lane", "executable_item"], keep="last")
    write_csv(TABLE_DIR / "paper4_living_lab_backlog.csv", combined)

    boundaries = read_csv("paper4_current_claim_boundaries.csv")
    boundary_additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has heuristic source-governance repair candidates after v59-v62.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v63_source_repair_frontier.csv",
                "boundary": "Lab-only whole-loan repairs over expanded restricted master; not exact CVaR optimality.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "A v63 source-repaired candidate replaces Paper Estrella.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v63_dynamic_gate_memo.csv",
                "boundary": "No replacement without dynamic replay, promotion memo and Paper Estrella gates.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if not boundaries.empty:
        boundaries = boundaries.loc[~boundaries["claim"].isin(boundary_additions["claim"])]
        boundaries = pd.concat([boundaries, boundary_additions], ignore_index=True)
    else:
        boundaries = boundary_additions
    write_csv(TABLE_DIR / "paper4_current_claim_boundaries.csv", boundaries)

    return gate


def update_notebook(status: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "<!-- V63_SOURCE_REPAIR_START -->",
            "",
            "## Wave v63: Source-Governance Repair Frontier",
            "",
            f"Generated: {now()}",
            "",
            "### Objective",
            "",
            "Turn the v61-v62 source-diversification blocker into an executable",
            "repair frontier: build whole-loan candidate books that reduce source",
            "concentration while measuring the cost in return and tail loss.",
            "",
            "### Results",
            "",
            f"- Frontier rows: `{status.get('source_repair_frontier_rows_v63')}`.",
            f"- Successful repair rows: `{status.get('source_repair_success_rows_v63')}`.",
            f"- Candidate book rows: `{status.get('source_repair_book_rows_v63')}`.",
            f"- Best repair score: `{status.get('best_repair_score_v63')}`.",
            f"- Dynamic rerun recommended rows: `{status.get('dynamic_rerun_recommended_rows_v63')}`.",
            "",
            "### Interpretation",
            "",
            "v63 shows whether the tail-feasible v59 idea can be made less trivial",
            "without pretending that a heuristic repair is an exact optimizer. If",
            "the dynamic gate stays closed, the value is still a publishable negative",
            "result: source governance imposes a measurable cost under the current",
            "internal loss model.",
            "",
            "### Claim Impact",
            "",
            "- Allowed: heuristic source-repair candidates and concentration certificates exist.",
            "- Still prohibited: Paper Estrella replacement, exact full-universe CVaR,",
            "  final Paper 4 promotion, live deployment and legal fairness claims.",
            "",
            "### Quarto Promotion Decision",
            "",
            "Keep v63 in the living notebook. Promote only after a candidate survives",
            "common-path dynamic replay and a separate editorial claim review.",
            "",
            "<!-- V63_SOURCE_REPAIR_END -->",
            "",
        ]
    )
    if not NOTEBOOK.exists():
        return
    text = NOTEBOOK.read_text(encoding="utf-8")
    start = "<!-- V63_SOURCE_REPAIR_START -->"
    end = "<!-- V63_SOURCE_REPAIR_END -->"
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end, 1)[1]
        NOTEBOOK.write_text(before.rstrip() + section + after.lstrip(), encoding="utf-8")
    else:
        NOTEBOOK.write_text(text.rstrip() + section, encoding="utf-8")


def build_v63() -> dict[str, Any]:
    start = datetime.now(UTC)
    frontier, books = build_v63_source_repair()
    gate = build_v63_gate_and_docs(frontier, books)
    best_score = (
        float(frontier["repair_score_v63"].max())
        if not frontier.empty and "repair_score_v63" in frontier
        else 0.0
    )
    status = {
        "schema_version": "2026-05-15.63",
        "generated_at_utc": now(),
        "phase": "v63_source_governance_repair_frontier",
        "source_repair_frontier_rows_v63": int(len(frontier)),
        "source_repair_success_rows_v63": int(frontier["source_repair_success_v63"].sum())
        if not frontier.empty
        else 0,
        "source_repair_book_rows_v63": int(len(books)),
        "best_repair_score_v63": best_score,
        "dynamic_rerun_recommended_rows_v63": int(
            gate["dynamic_512_or_1024_rerun_recommended_v63"].sum()
        )
        if not gate.empty
        else 0,
        "exact_full_universe_cvar_claim_allowed": False,
        "paper1_promotion_allowed_v63": False,
        "paper4_working_champion_changed_v63": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "runtime_seconds": round((datetime.now(UTC) - start).total_seconds(), 3),
        "claim_boundary": "v63 creates heuristic source-repair candidates only; no promotion or exact optimizer claim",
    }
    write_json(STATUS_DIR / "paper4_v63_status.json", status)
    update_notebook(status)
    return status


def main() -> None:
    status = build_v63()
    print(json.dumps({"v63": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
