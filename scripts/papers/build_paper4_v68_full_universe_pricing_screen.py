#!/usr/bin/env python3
"""Build Paper 4 v68 full-universe source/pricing screen artifacts.

This is deliberately not an exact CVaR optimizer.  It screens the full v55
comparable universe against the v63 source-repair books to detect whether
plausible out-of-book columns remain.  If such columns exist, the screen is
evidence that exact full-universe optimality remains unproven, not evidence of
promotion.
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
    u["expected_loss_proxy_v68"] = (
        u["loan_amnt"].astype(float)
        * u["lgd_proxy_v55"].astype(float)
        * u["pd_high_alpha01"].astype(float)
    )
    u["expected_return_proxy_v68"] = (
        u["base_return_vec"].astype(float)
        - u["expected_loss_proxy_v68"]
        - 0.012 * u["loan_amnt"].astype(float)
    )
    u["tail_score_proxy_v68"] = (
        u["expected_return_proxy_v68"]
        - 1.15 * u["expected_loss_proxy_v68"]
        - 1800.0 * u["qhat_v4"].astype(float)
        - 2000.0 * u["weak_source_proxy"].astype(float)
    )
    u["source_score_proxy_v68"] = (
        u["expected_return_proxy_v68"]
        - 0.80 * u["expected_loss_proxy_v68"]
        - 3300.0 * u["weak_source_proxy"].astype(float)
        - 1500.0 * u["qhat_v4"].astype(float)
    )
    u["return_norm_v68"] = normalize(u["expected_return_proxy_v68"])
    u["tail_norm_v68"] = normalize(u["tail_score_proxy_v68"])
    u["weak_source_norm_v68"] = normalize(u["weak_source_proxy"], higher_is_better=False)
    return u


def _policy_source_map(concentration: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    local = concentration.loc[concentration["policy_id"].eq(policy_id)].copy()
    if local.empty:
        return pd.DataFrame()
    local["over_cap_v68"] = pd.to_numeric(
        local["top_exposure_share_v63"], errors="coerce"
    ) > pd.to_numeric(local["target_cap_v63"], errors="coerce")
    return local


def _source_relief(universe: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    relief = pd.Series(0, index=universe.index, dtype=float)
    active = source_map.loc[source_map["over_cap_v68"].astype(bool)]
    if active.empty:
        active = source_map
    for _, row in active.iterrows():
        family = str(row["source_family"])
        if family not in universe:
            continue
        top_source = str(row["top_source_id_v63"])
        relief = relief + universe[family].astype(str).ne(top_source).astype(float)
    denom = max(int(len(active)), 1)
    out = pd.DataFrame(index=universe.index)
    out["source_relief_hits_v68"] = relief
    out["source_relief_share_v68"] = relief / denom
    out["active_source_constraints_v68"] = int(len(active))
    return out


def _screen_policy(
    universe: pd.DataFrame,
    books: pd.DataFrame,
    concentration: pd.DataFrame,
    policy_id: str,
    top_n: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    book = books.loc[books["policy_id_v63"].eq(policy_id)].copy()
    source_map = _policy_source_map(concentration, policy_id)
    if book.empty or source_map.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    in_book = set(book["loan_id"].astype(str))
    candidates = universe.loc[~universe["loan_id"].astype(str).isin(in_book)].copy()
    relief = _source_relief(candidates, source_map)
    candidates = pd.concat([candidates, relief], axis=1)
    candidates["pricing_screen_score_v68"] = (
        0.42 * candidates["return_norm_v68"]
        + 0.32 * candidates["tail_norm_v68"]
        + 0.20 * candidates["source_relief_share_v68"]
        + 0.06 * candidates["weak_source_norm_v68"]
    )
    candidates["policy_id_v68"] = policy_id
    candidates["screen_scope_v68"] = (
        "full comparable universe source/pricing screen; not exact dual pricing"
    )
    keep_cols = [
        "policy_id_v68",
        "loan_index_v55",
        "loan_id",
        "loan_amnt",
        "grade",
        "score_decile",
        "income_band",
        "dti_band",
        "period",
        "state_top20",
        "expected_loss_proxy_v68",
        "expected_return_proxy_v68",
        "tail_score_proxy_v68",
        "source_score_proxy_v68",
        "source_relief_hits_v68",
        "source_relief_share_v68",
        "active_source_constraints_v68",
        "pricing_screen_score_v68",
        "screen_scope_v68",
    ]
    top = (
        candidates.sort_values("pricing_screen_score_v68", ascending=False)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    top["candidate_rank_v68"] = np.arange(1, len(top) + 1)

    book_scored = universe.loc[universe["loan_id"].astype(str).isin(in_book)].copy()
    book_relief = _source_relief(book_scored, source_map)
    book_scored = pd.concat([book_scored, book_relief], axis=1)
    book_scored["pricing_screen_score_v68"] = (
        0.42 * book_scored["return_norm_v68"]
        + 0.32 * book_scored["tail_norm_v68"]
        + 0.20 * book_scored["source_relief_share_v68"]
        + 0.06 * book_scored["weak_source_norm_v68"]
    )
    candidate_max = float(top["pricing_screen_score_v68"].max()) if not top.empty else np.nan
    book_p25 = float(book_scored["pricing_screen_score_v68"].quantile(0.25))
    book_median = float(book_scored["pricing_screen_score_v68"].median())
    benchmark = {
        "policy_id": policy_id,
        "book_rows_v68": int(len(book_scored)),
        "screened_universe_rows_v68": int(len(universe)),
        "out_of_book_rows_v68": int(len(candidates)),
        "top_candidate_rows_v68": int(len(top)),
        "book_score_p25_v68": book_p25,
        "book_score_median_v68": book_median,
        "candidate_max_score_v68": candidate_max,
        "candidate_beats_book_p25_v68": bool(candidate_max > book_p25),
        "candidate_beats_book_median_v68": bool(candidate_max > book_median),
        "candidate_max_source_relief_share_v68": (
            float(top["source_relief_share_v68"].max()) if not top.empty else np.nan
        ),
        "screen_detects_unpriced_columns_v68": bool(candidate_max > book_p25),
        "exact_full_universe_cvar_claim_allowed_v68": False,
        "claim_boundary_v68": (
            "proxy screen only; out-of-book candidates block exact full-universe optimality claims"
        ),
    }

    family_rows: list[dict[str, Any]] = []
    for _, row in source_map.iterrows():
        family = str(row["source_family"])
        if family not in top:
            continue
        top_source = str(row["top_source_id_v63"])
        family_rows.append(
            {
                "policy_id": policy_id,
                "source_family": family,
                "top_source_id_v63": top_source,
                "top_candidate_rows_not_top_source_v68": int(
                    top[family].astype(str).ne(top_source).sum()
                ),
                "top_candidate_share_not_top_source_v68": float(
                    top[family].astype(str).ne(top_source).mean()
                ),
                "source_top_share_v63": row.get("top_exposure_share_v63"),
                "target_cap_v63": row.get("target_cap_v63"),
                "claim_boundary_v68": "source relief screen over top candidate columns",
            }
        )
    return top[keep_cols + ["candidate_rank_v68"]], benchmark, pd.DataFrame(family_rows)


def build_v68_screen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = _prepare_universe()
    books = read_parquet("paper4_v63_source_repair_candidate_books.parquet")
    concentration = read_csv("paper4_v63_source_repair_concentration.csv")
    frontier = read_csv("paper4_v63_source_repair_frontier.csv")
    if universe.empty or books.empty or concentration.empty or frontier.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    candidate_frames: list[pd.DataFrame] = []
    benchmark_rows: list[dict[str, Any]] = []
    relief_frames: list[pd.DataFrame] = []
    for policy_id in sorted(frontier["policy_id"].dropna().astype(str).unique()):
        top, benchmark, relief = _screen_policy(universe, books, concentration, policy_id)
        if not top.empty:
            candidate_frames.append(top)
        if benchmark:
            benchmark_rows.append(benchmark)
        if not relief.empty:
            relief_frames.append(relief)

    candidates = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    )
    benchmark = pd.DataFrame(benchmark_rows)
    relief = pd.concat(relief_frames, ignore_index=True) if relief_frames else pd.DataFrame()
    return candidates, benchmark, relief


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a full-universe source/pricing screen for v63 repair books.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v68_full_universe_candidate_screen.parquet"
                ),
                "boundary": "Proxy screen only; not exact dual pricing or full-universe CVaR.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v68 proves exact full-universe CVaR optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v68_screen_vs_book_benchmark.csv"
                ),
                "boundary": "Screen finds/prioritizes candidate columns but is not an exact LP certificate.",
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
                    "v68 screens the full comparable universe for out-of-book source/pricing "
                    "columns against v63 repair books."
                ),
                "status": "near_resolved_with_plateau",
                "next_artifact": "paper4_v69_exact_column_generation_protocol.csv",
                "success_condition": "exact pricing or column-generation certificate replaces proxy screen",
                "last_wave": "v68",
                "execution_result": "full_universe_proxy_screen_completed",
                "quarto_promotion_decision": "living_notebook_only",
            },
            {
                "horizon": "short",
                "lane": "Source governance",
                "executable_item": (
                    "Use v68 candidate columns to design an exact restricted-master expansion "
                    "without relaxing claim boundaries."
                ),
                "status": "gated",
                "next_artifact": "paper4_v69_source_pricing_expansion_candidates.parquet",
                "success_condition": "candidate expansion improves source/tail tradeoff under exact solver",
                "last_wave": "v68",
                "execution_result": "candidate_expansion_queued",
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
    start = "<!-- V68_FULL_UNIVERSE_PRICING_SCREEN_START -->"
    end = "<!-- V68_FULL_UNIVERSE_PRICING_SCREEN_END -->"
    block = f"""
{start}

## Wave v68: Full-Universe Source/Pricing Screen

Generated: {status["generated_at_utc"]}

### Objective

Audit the v63 source-repair books against the full v55 comparable universe
using a proxy source/pricing screen. The goal is to detect whether plausible
out-of-book columns remain, not to claim exact CVaR optimality.

### Results

- Screened policies: `{status["screened_policy_rows_v68"]}`.
- Top candidate rows: `{status["candidate_screen_rows_v68"]}`.
- Benchmark rows: `{status["benchmark_rows_v68"]}`.
- Policies with candidate columns above book p25: `{status["policies_with_unpriced_columns_v68"]}`.
- Exact full-universe CVaR claim allowed: `{status["exact_full_universe_cvar_claim_allowed_v68"]}`.

### Interpretation

v68 strengthens the source-governance audit: the comparable universe contains
plausible out-of-book columns under a proxy pricing/source-relief screen. That
keeps the exact full-universe CVaR claim blocked and gives v69 a concrete
candidate-expansion target.

### Claim Impact

- Allowed: proxy full-universe source/pricing screen exists.
- Still prohibited: exact full-universe optimality, Paper Estrella replacement,
  final Paper 4 promotion and live deployment claims.

### Quarto Promotion Decision

Keep v68 in the living notebook. Promote only after exact pricing or a
column-generation certificate replaces the proxy screen.

{end}
""".strip()
    if start in existing and end in existing:
        before = existing.split(start)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    NOTEBOOK.write_text(updated, encoding="utf-8")


def build_v68() -> dict[str, Any]:
    started = datetime.now(UTC)
    candidates, benchmark, relief = build_v68_screen()
    if not candidates.empty:
        candidates.to_parquet(
            TABLE_DIR / "paper4_v68_full_universe_candidate_screen.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(
            TABLE_DIR / "paper4_v68_full_universe_candidate_screen.parquet", index=False
        )
    write_csv(TABLE_DIR / "paper4_v68_screen_vs_book_benchmark.csv", benchmark)
    write_csv(TABLE_DIR / "paper4_v68_source_relief_summary.csv", relief)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v68_full_universe_proxy_screen_exists",
                "allowed": True,
                "artifact": "paper4_v68_full_universe_candidate_screen.parquet",
                "boundary": "proxy screen only",
            },
            {
                "claim_id": "v68_exact_full_universe_cvar_optimality",
                "allowed": False,
                "artifact": "paper4_v68_screen_vs_book_benchmark.csv",
                "boundary": "requires exact LP dual pricing or column-generation certificate",
            },
            {
                "claim_id": "v68_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v68_source_relief_summary.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    write_csv(TABLE_DIR / "paper4_v68_claim_matrix_delta.csv", claim_matrix)
    policies_with_unpriced = (
        int(benchmark["screen_detects_unpriced_columns_v68"].astype(bool).sum())
        if not benchmark.empty
        else 0
    )
    status = {
        "phase": "v68_full_universe_source_pricing_screen",
        "schema_version": "2026-05-15.68",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "screened_policy_rows_v68": int(len(benchmark)),
        "candidate_screen_rows_v68": int(len(candidates)),
        "benchmark_rows_v68": int(len(benchmark)),
        "source_relief_rows_v68": int(len(relief)),
        "policies_with_unpriced_columns_v68": policies_with_unpriced,
        "exact_dual_pricing_performed_v68": False,
        "exact_full_universe_cvar_claim_allowed_v68": False,
        "paper1_promotion_allowed_v68": False,
        "paper4_working_champion_changed_v68": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_boundary": (
            "v68 is a full-universe proxy screen only; exact full-universe CVaR remains blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v68_status.json", status)
    _update_claim_boundaries()
    _update_backlog()
    _update_notebook(status)
    return status


def main() -> None:
    print(json.dumps({"v68": build_v68()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
