"""Build extended Paper 4 living-lab experiments.

This script implements the next Paper 4 lanes after the foundation pack:

- P0: policy-loan sparse matrix, full IFRS9 policy overlays, monthly replay,
  diagnostic selector, and Pareto frontiers.
- P1: paired bootstrap, robust satisficing, online conformal prototype,
  MDCP/worst-source coverage, and fairness proxy stress.
- P2: toy multi-period SDAM/DLA replay and causal/CATE identification expansion.

The generator is intentionally diagnostic. It does not create
``paper4_final_promotion.json`` and it does not modify the Paper Estrella
champion. For non-champion policies, funded sets are reconstructed using a
policy-implied greedy proxy because the original bound-aware search preserved
aggregate metrics but not allocation vectors for every policy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.analyze_paper1_p1_evidence import (
    CONFORMAL_WINNER_INTERVALS_PATH,
    _interval_arrays_at_alpha,
    _load_exact_aligned_dataset,
    _parse_rate_series,
)
from scripts.papers.build_paper4_living_lab_artifacts import (
    DEFAULT_LGD,
    load_policy_universe,
)
from src.optimization.portfolio_model import compute_effective_pd

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
STATUS_DIR = OUT_ROOT / "status"
NOTE_DIR = OUT_ROOT / "notes"
SCHEMA_VERSION = "2026-05-12.2"
BUDGET = 1_000_000.0
RNG_SEED = 20260512


def _write_csv(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def _write_parquet(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_parquet(path, index=False)
    return path


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_note(name: str, text: str) -> Path:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTE_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = numeric.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float((numeric[mask] * weights[mask]).sum() / weights[mask].sum())


def _normalise(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    low = float(values.min())
    high = float(values.max())
    if abs(high - low) <= 1e-12:
        return values.notna().astype(float)
    out = (values - low) / (high - low)
    if not higher_is_better:
        out = 1.0 - out
    return out.fillna(0.0)


def _load_base_loan_frame() -> pd.DataFrame:
    aligned = _load_exact_aligned_dataset(CONFORMAL_WINNER_INTERVALS_PATH).copy()
    pd_point, pd_low, pd_high = _interval_arrays_at_alpha(aligned, 0.01)
    aligned["pd_point_alpha01"] = pd_point
    aligned["pd_low_alpha01"] = pd_low
    aligned["pd_high_alpha01"] = pd_high
    aligned["int_rate_decimal"] = _parse_rate_series(aligned["int_rate"])
    aligned["loan_amnt"] = pd.to_numeric(aligned["loan_amnt"], errors="coerce").fillna(0.0)
    aligned["y_true"] = pd.to_numeric(aligned["y_true"], errors="coerce").fillna(0.0)
    aligned["issue_month"] = (
        pd.to_datetime(aligned["issue_d"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    if "sub_grade" not in aligned:
        aligned["sub_grade"] = aligned["original_grade"].astype(str) + "_unknown"
    if "term" not in aligned:
        aligned["term"] = "unknown"
    if "purpose" not in aligned:
        aligned["purpose"] = "unknown"
    return aligned


def _policy_effective_pd(base: pd.DataFrame, policy: pd.Series) -> np.ndarray:
    return compute_effective_pd(
        pd_point=base["pd_point_alpha01"].to_numpy(dtype=float),
        pd_high=base["pd_high_alpha01"].to_numpy(dtype=float),
        policy_mode=str(policy["policy_mode"]),
        gamma=float(policy["gamma"]),
        delta_cap_quantile=1.0,
        tail_focus_quantile=1.0,
        segment_labels=None,
    )


def _select_proxy_indices(
    base: pd.DataFrame,
    policy: pd.Series,
    effective_pd: np.ndarray,
) -> np.ndarray:
    target_n = int(policy["n_funded"])
    tau = float(policy["risk_tolerance"])
    point = base["pd_point_alpha01"].to_numpy(dtype=float)
    high = base["pd_high_alpha01"].to_numpy(dtype=float)
    rates = base["int_rate_decimal"].to_numpy(dtype=float)
    uncertainty_aversion = float(policy.get("uncertainty_aversion", 0.0))
    objective_density = (
        rates
        - point * DEFAULT_LGD
        - uncertainty_aversion * np.clip(high - point, 0.0, 1.0) * DEFAULT_LGD
        - 0.25 * np.maximum(effective_pd - tau, 0.0)
    )
    amount = base["loan_amnt"].to_numpy(dtype=float)
    order = np.argsort(-objective_density)
    selected: list[int] = []
    selected_amount = 0.0
    selected_pd_amount = 0.0

    for tolerance_slack in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        selected.clear()
        selected_amount = 0.0
        selected_pd_amount = 0.0
        for idx in order:
            loan_amount = float(amount[idx])
            if loan_amount <= 0:
                continue
            next_amount = selected_amount + loan_amount
            next_pd = selected_pd_amount + loan_amount * float(effective_pd[idx])
            next_avg_pd = next_pd / max(next_amount, 1e-12)
            if next_avg_pd > tau + tolerance_slack:
                continue
            selected.append(int(idx))
            selected_amount = next_amount
            selected_pd_amount = next_pd
            if len(selected) >= target_n:
                break
        if len(selected) >= max(10, int(0.90 * target_n)):
            break
    return np.asarray(selected[:target_n], dtype=int)


def _canonical_champion_evidence() -> pd.DataFrame:
    funded = _safe_read_csv(
        ROOT
        / "reports"
        / "paper_material"
        / "paper1"
        / "tables"
        / "paper1_tableA7_funded_set_loans.csv"
    )
    if funded.empty:
        return pd.DataFrame()
    out = funded.copy()
    out["policy_id"] = "paper1_economic_champion"
    out["loan_id"] = out["id"]
    out["issue_month"] = (
        pd.to_datetime(out["issue_d"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    out["int_rate_decimal"] = _parse_rate_series(out["int_rate"])
    out["effective_pd_alpha01"] = pd.to_numeric(
        out["effective_pd_alpha01"], errors="coerce"
    ).fillna(out["pd_high_alpha01"])
    out["miscovered_alpha01"] = out["miscovered_alpha01"].astype(bool)
    out["reconstruction_method"] = "canonical_paper1_exact_funded_set"
    out["funded_flag"] = True
    return out[
        [
            "policy_id",
            "loan_id",
            "issue_d",
            "issue_month",
            "period",
            "original_grade",
            "sub_grade",
            "term",
            "loan_amnt",
            "int_rate",
            "int_rate_decimal",
            "y_true",
            "allocation_fraction",
            "funded_exposure",
            "portfolio_weight",
            "pd_point",
            "pd_high_alpha01",
            "effective_pd_alpha01",
            "miscovered_alpha01",
            "funded_flag",
            "reconstruction_method",
        ]
    ]


def build_policy_loan_evidence(
    base: pd.DataFrame,
    policy_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    champion = _canonical_champion_evidence()
    champion_ids = set(champion["loan_id"].astype(str)) if not champion.empty else set()
    if not champion.empty:
        rows.append(champion)

    for _, policy in policy_universe.iterrows():
        if str(policy["policy_id"]) == "paper1_economic_champion":
            continue
        effective_pd = _policy_effective_pd(base, policy)
        selected_idx = _select_proxy_indices(base, policy, effective_pd)
        selected = base.iloc[selected_idx].copy()
        selected["policy_id"] = str(policy["policy_id"])
        selected["loan_id"] = selected["id"]
        selected["effective_pd_alpha01"] = effective_pd[selected_idx]
        total_requested = float(selected["loan_amnt"].sum())
        allocation_fraction = min(1.0, BUDGET / max(total_requested, 1e-12))
        selected["allocation_fraction"] = allocation_fraction
        selected["funded_exposure"] = selected["loan_amnt"] * allocation_fraction
        selected["portfolio_weight"] = selected["funded_exposure"] / max(
            float(selected["funded_exposure"].sum()), 1e-12
        )
        selected["pd_point"] = selected["pd_point_alpha01"]
        selected["miscovered_alpha01"] = selected["y_true"].gt(selected["pd_high_alpha01"])
        selected["funded_flag"] = True
        selected["reconstruction_method"] = "policy_implied_greedy_proxy"
        rows.append(
            selected[
                [
                    "policy_id",
                    "loan_id",
                    "issue_d",
                    "issue_month",
                    "period",
                    "original_grade",
                    "sub_grade",
                    "term",
                    "loan_amnt",
                    "int_rate",
                    "int_rate_decimal",
                    "y_true",
                    "allocation_fraction",
                    "funded_exposure",
                    "portfolio_weight",
                    "pd_point",
                    "pd_high_alpha01",
                    "effective_pd_alpha01",
                    "miscovered_alpha01",
                    "funded_flag",
                    "reconstruction_method",
                ]
            ]
        )

    evidence = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    evidence["loan_id"] = evidence["loan_id"].astype(str)
    evidence["policy_id"] = evidence["policy_id"].astype(str)
    evidence["issue_d"] = pd.to_datetime(evidence["issue_d"], errors="coerce").astype(str)
    evidence["issue_month"] = pd.to_datetime(evidence["issue_month"], errors="coerce")
    evidence["period"] = evidence["period"].astype(str)
    evidence["original_grade"] = evidence["original_grade"].astype(str)
    evidence["sub_grade"] = evidence["sub_grade"].astype(str)
    evidence["term"] = evidence["term"].astype(str)
    evidence["is_champion_exact"] = evidence["policy_id"].eq("paper1_economic_champion")
    evidence["also_in_champion_funded_set"] = evidence["loan_id"].astype(str).isin(champion_ids)
    evidence["realized_return_proxy_lgd45"] = (
        evidence["funded_exposure"] * evidence["int_rate_decimal"] * (1.0 - evidence["y_true"])
        - evidence["funded_exposure"] * DEFAULT_LGD * evidence["y_true"]
    )
    evidence["ecl_baseline_lgd45"] = (
        evidence["pd_high_alpha01"] * DEFAULT_LGD * evidence["funded_exposure"]
    )

    sparse_matrix = evidence[
        [
            "policy_id",
            "loan_id",
            "funded_flag",
            "allocation_fraction",
            "funded_exposure",
            "portfolio_weight",
            "reconstruction_method",
            "is_champion_exact",
        ]
    ].copy()

    composition = (
        evidence.groupby(["policy_id", "period", "original_grade"], as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
            ecl_baseline_lgd45=("ecl_baseline_lgd45", "sum"),
            observed_default_rate=("y_true", "mean"),
            weighted_pd_high=(
                "pd_high_alpha01",
                lambda x: _weighted_average(x, evidence.loc[x.index, "funded_exposure"]),
            ),
        )
        .sort_values(["policy_id", "period", "original_grade"])
    )
    return evidence, sparse_matrix, composition


def build_ifrs9_full(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [("baseline", 1.0), ("adverse", 1.25), ("severe", 1.60)]
    rows = []
    for scenario, multiplier in scenarios:
        work = evidence.copy()
        work["scenario"] = scenario
        work["scenario_pd"] = np.clip(work["pd_high_alpha01"] * multiplier, 0.0, 1.0)
        work["ecl"] = work["scenario_pd"] * DEFAULT_LGD * work["funded_exposure"]
        sicr = (work["scenario_pd"] >= 0.30) | (
            work["scenario_pd"] / np.maximum(work["pd_point"], 1e-6) >= 1.50
        )
        work["ifrs9_stage"] = np.select(
            [work["y_true"].eq(1.0), sicr],
            ["Stage 3 observed", "Stage 2 proxy"],
            default="Stage 1 proxy",
        )
        work["provision"] = work["ecl"]
        rows.append(work)
    grid = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    stage_mix = (
        grid.groupby(["policy_id", "scenario", "ifrs9_stage"], as_index=False).agg(
            stage_exposure=("funded_exposure", "sum"), stage_ecl=("ecl", "sum")
        )
        if not grid.empty
        else pd.DataFrame()
    )
    if not stage_mix.empty:
        totals = stage_mix.groupby(["policy_id", "scenario"], as_index=False).agg(
            total_exposure=("stage_exposure", "sum"),
            total_ecl=("stage_ecl", "sum"),
        )
        stage_mix = stage_mix.merge(totals, on=["policy_id", "scenario"], how="left")
        stage_mix["stage_exposure_share"] = stage_mix["stage_exposure"] / stage_mix[
            "total_exposure"
        ].clip(lower=1e-12)
        stage_mix["stage_ecl_share"] = stage_mix["stage_ecl"] / stage_mix["total_ecl"].clip(
            lower=1e-12
        )

    summary = (
        grid.groupby(["policy_id", "scenario"], as_index=False).agg(
            n_funded=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
            ecl=("ecl", "sum"),
            provision=("provision", "sum"),
            mean_pd_scenario=("scenario_pd", "mean"),
            weighted_pd_scenario=(
                "scenario_pd",
                lambda x: _weighted_average(x, grid.loc[x.index, "funded_exposure"]),
            ),
            stage2_or_3_share=(
                "ifrs9_stage",
                lambda x: float(x.isin(["Stage 2 proxy", "Stage 3 observed"]).mean()),
            ),
            observed_default_rate=("y_true", "mean"),
        )
        if not grid.empty
        else pd.DataFrame()
    )
    if not summary.empty:
        summary["net_return_after_ecl_full"] = (
            summary["realized_return_proxy_lgd45"] - summary["provision"]
        )
        summary["status"] = "full_policy_loan_ifrs9_proxy"
    return grid, stage_mix, summary


def build_monthly_policy_replay(
    evidence: pd.DataFrame,
    ifrs9_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = ifrs9_grid[ifrs9_grid["scenario"].eq("baseline")][
        ["policy_id", "loan_id", "ecl", "provision", "ifrs9_stage"]
    ]
    work = evidence.merge(baseline, on=["policy_id", "loan_id"], how="left")
    replay = (
        work.groupby(["policy_id", "issue_month", "period"], as_index=False)
        .agg(
            funded_count=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
            ecl=("ecl", "sum"),
            provision=("provision", "sum"),
            observed_default_rate=("y_true", "mean"),
            coverage_alpha01=("miscovered_alpha01", lambda x: float(1.0 - x.astype(float).mean())),
            stage2_or_3_share=(
                "ifrs9_stage",
                lambda x: float(x.isin(["Stage 2 proxy", "Stage 3 observed"]).mean()),
            ),
        )
        .rename(columns={"issue_month": "month"})
    )
    replay["net_return_after_ecl"] = replay["realized_return_proxy_lgd45"] - replay["provision"]

    monthly = _safe_read_parquet(ROOT / "data" / "processed" / "conformal_backtest_monthly.parquet")
    if not monthly.empty:
        monthly = monthly.copy()
        monthly["month"] = pd.to_datetime(monthly["month"])
        replay = replay.merge(
            monthly[["month", "coverage_90", "coverage_95", "gap_90", "avg_width_90"]],
            on="month",
            how="left",
        )
    replay["status"] = "policy_implied_historical_replay"
    summary = (
        replay.groupby("policy_id", as_index=False)
        .agg(
            months=("month", "nunique"),
            funded_count_total=("funded_count", "sum"),
            funded_exposure_total=("funded_exposure", "sum"),
            realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
            ecl_total=("ecl", "sum"),
            net_return_after_ecl_total=("net_return_after_ecl", "sum"),
            min_monthly_coverage_alpha01=("coverage_alpha01", "min"),
            mean_stage2_or_3_share=("stage2_or_3_share", "mean"),
        )
        .sort_values("net_return_after_ecl_total", ascending=False)
    )
    return replay, summary


def _loan_tail_metrics(evidence: pd.DataFrame) -> pd.DataFrame:
    work = evidence.copy()
    work["loss_amount_lgd45"] = work["funded_exposure"] * DEFAULT_LGD * work["y_true"]
    rows: list[dict[str, Any]] = []
    for policy_id, group in work.groupby("policy_id"):
        loss_rate = group["loss_amount_lgd45"] / group["funded_exposure"].clip(lower=1e-12)
        threshold = float(loss_rate.quantile(0.95))
        cvar95 = float(loss_rate[loss_rate >= threshold].mean()) if len(loss_rate) else 0.0
        rows.append(
            {
                "policy_id": policy_id,
                "mean_loss_rate": _weighted_average(loss_rate, group["funded_exposure"]),
                "cvar_95_loss_rate_full": cvar95,
                "max_loan_loss_amount": float(group["loss_amount_lgd45"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostic_selector(
    policy_universe: pd.DataFrame,
    ifrs9_summary: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline = ifrs9_summary[ifrs9_summary["scenario"].eq("baseline")].copy()
    sat = _safe_read_csv(TABLE_DIR / "paper4_table4_satisficing_screen.csv")
    sat_agg = (
        sat.groupby("policy_id", as_index=False).agg(
            satisficing_pass_rate=("pass", "mean"),
            min_satisficing_margin=("margin", "min"),
        )
        if not sat.empty
        else pd.DataFrame()
    )
    tail = _loan_tail_metrics(evidence)
    selector = policy_universe.merge(baseline, on="policy_id", how="left")
    selector = selector.rename(
        columns={
            "n_funded_x": "n_funded_declared",
            "n_funded_y": "n_funded_reconstructed",
        }
    )
    selector = selector.merge(sat_agg, on="policy_id", how="left")
    selector = selector.merge(tail, on="policy_id", how="left")
    selector["score_return"] = _normalise(selector["realized_total_return"], higher_is_better=True)
    selector["score_net_ecl"] = _normalise(
        selector["net_return_after_ecl_full"], higher_is_better=True
    )
    selector["score_ecl"] = _normalise(selector["ecl"], higher_is_better=False)
    selector["score_tail"] = _normalise(selector["cvar_95_loss_rate_full"], higher_is_better=False)
    selector["score_satisficing"] = selector["satisficing_pass_rate"].fillna(0.0)
    selector["score_auditability"] = 0.50 * _normalise(
        selector["gamma_cp"], higher_is_better=False
    ) + 0.50 * _normalise(selector["weighted_miscoverage_V"], higher_is_better=False)
    weights = {
        "score_return": 0.25,
        "score_net_ecl": 0.30,
        "score_ecl": 0.10,
        "score_tail": 0.15,
        "score_satisficing": 0.10,
        "score_auditability": 0.10,
    }
    selector["diagnostic_selector_score"] = sum(
        selector[column] * weight for column, weight in weights.items()
    )
    selector["hard_gate_pass"] = selector["all_bounds_hold"].astype(bool) & selector[
        "ab_pass_all"
    ].astype(bool)
    selector["diagnostic_selector_rank"] = (
        selector["diagnostic_selector_score"].rank(ascending=False, method="first").astype(int)
    )
    selector["diagnostic_decision"] = np.select(
        [
            selector["policy_id"].eq("paper1_economic_champion"),
            selector["diagnostic_selector_rank"].le(5) & selector["hard_gate_pass"],
            selector["hard_gate_pass"],
        ],
        ["protected_paper1_champion", "promote_candidate_for_future_run", "keep"],
        default="park",
    )
    selector["status"] = "diagnostic_selector_no_promotion"
    config = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selector_id": "paper4_ifrs9_tail_satisficing_diagnostic_selector",
        "mode": "diagnostic_no_promotion",
        "weights": weights,
        "hard_gates": ["all_bounds_hold", "ab_pass_all"],
        "paper1_champion_protected": True,
        "promotion_json_created": False,
        "caveat": "Ranks reconstructed diagnostic policies; does not replace Paper Estrella champion.",
    }
    return config, selector.sort_values("diagnostic_selector_rank").reset_index(drop=True)


def write_pareto_frontiers(selector: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    frontier = selector[
        [
            "policy_id",
            "risk_tolerance",
            "gamma",
            "uncertainty_aversion",
            "realized_total_return",
            "net_return_after_ecl_full",
            "ecl",
            "cvar_95_loss_rate_full",
            "gamma_cp",
            "weighted_miscoverage_V",
            "diagnostic_selector_score",
            "diagnostic_selector_rank",
            "diagnostic_decision",
        ]
    ].copy()
    frontier["pareto_return_ecl"] = False
    for idx, row in frontier.iterrows():
        dominates = (
            frontier["realized_total_return"].ge(row["realized_total_return"])
            & frontier["ecl"].le(row["ecl"])
            & (
                frontier["realized_total_return"].gt(row["realized_total_return"])
                | frontier["ecl"].lt(row["ecl"])
            )
        )
        frontier.loc[idx, "pareto_return_ecl"] = not bool(dominates.any())

    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        scatter = ax.scatter(
            frontier["ecl"],
            frontier["realized_total_return"],
            c=frontier["cvar_95_loss_rate_full"],
            cmap="viridis_r",
            s=np.where(frontier["policy_id"].eq("paper1_economic_champion"), 130, 70),
            alpha=0.85,
        )
        ax.set_xlabel("Baseline ECL / provision")
        ax.set_ylabel("Robust realized return")
        ax.set_title("Paper 4 diagnostic frontier: return vs ECL")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="CVaR95 loss rate")
        path = FIGURE_DIR / "paper4_fig5_return_ecl_tail_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(
            frontier["gamma_cp"],
            frontier["realized_total_return"],
            c=frontier["diagnostic_selector_score"],
            cmap="plasma",
            s=np.where(frontier["policy_id"].eq("paper1_economic_champion"), 130, 70),
            alpha=0.85,
        )
        ax.set_xlabel("Gamma_CP")
        ax.set_ylabel("Robust realized return")
        ax.set_title("Paper 4 diagnostic frontier: return vs Gamma_CP")
        ax.grid(True, alpha=0.25)
        path = FIGURE_DIR / "paper4_fig6_gamma_return_frontier.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    except Exception:
        pass
    return frontier, paths


def build_bootstrap_ci(monthly_replay: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    if monthly_replay.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RNG_SEED)
    work = monthly_replay.copy()
    work["month"] = pd.to_datetime(work["month"])
    months = np.array(sorted(work["month"].dropna().unique()))
    pivot = (
        work.pivot_table(
            index="month", columns="policy_id", values="net_return_after_ecl", aggfunc="sum"
        )
        .reindex(months)
        .fillna(0.0)
    )
    champion = "paper1_economic_champion"
    rows = []
    for policy_id in pivot.columns:
        diffs = []
        for sample_id in range(n_boot):
            sampled = rng.choice(months, size=len(months), replace=True)
            policy_value = float(pivot.loc[sampled, policy_id].sum())
            champion_value = float(pivot.loc[sampled, champion].sum()) if champion in pivot else 0.0
            diffs.append(policy_value - champion_value)
        arr = np.asarray(diffs, dtype=float)
        rows.append(
            {
                "policy_id": policy_id,
                "baseline_policy_id": champion,
                "n_bootstrap_paths": n_boot,
                "mean_net_return_diff": float(arr.mean()),
                "p05_net_return_diff": float(np.quantile(arr, 0.05)),
                "p50_net_return_diff": float(np.quantile(arr, 0.50)),
                "p95_net_return_diff": float(np.quantile(arr, 0.95)),
                "prob_diff_positive": float((arr > 0).mean()),
                "status": "monthly_bootstrap_paired_diagnostic",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_net_return_diff", ascending=False)


def build_robust_satisficing_policy(selector: pd.DataFrame) -> pd.DataFrame:
    out = selector[
        [
            "policy_id",
            "satisficing_pass_rate",
            "min_satisficing_margin",
            "diagnostic_selector_rank",
            "diagnostic_selector_score",
            "hard_gate_pass",
            "diagnostic_decision",
        ]
    ].copy()
    out["robust_satisficing_decision"] = np.select(
        [
            out["policy_id"].eq("paper1_economic_champion"),
            out["satisficing_pass_rate"].ge(1.0) & out["diagnostic_selector_rank"].le(5),
            out["satisficing_pass_rate"].ge(1.0),
        ],
        ["protected_champion", "promote_candidate_for_future_selector", "keep"],
        default="park",
    )
    out["status"] = "robust_satisficing_policy_diagnostic"
    return out.sort_values(["diagnostic_selector_rank", "policy_id"])


def build_online_conformal_aci() -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = _safe_read_parquet(ROOT / "data" / "processed" / "conformal_backtest_monthly.parquet")
    grade = _safe_read_parquet(
        ROOT / "data" / "processed" / "conformal_backtest_monthly_grade.parquet"
    )
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()

    def aci(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, group in frame.sort_values("month").groupby(group_cols, dropna=False):
            alpha = 0.10
            eta = 0.05
            hist: list[float] = []
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            for _, row in group.iterrows():
                coverage = float(row["coverage_90"])
                hist.append(coverage)
                rows.append(
                    {
                        **{col: key_tuple[i] for i, col in enumerate(group_cols)},
                        "month": row["month"],
                        "n": int(row.get("n", 0)),
                        "coverage_90": coverage,
                        "target_coverage_90": 0.90,
                        "coverage_gap_90": coverage - 0.90,
                        "rolling_3m_coverage_90": float(np.mean(hist[-3:])),
                        "alpha_t": alpha,
                        "recommended_alpha_next": float(
                            np.clip(alpha + eta * (coverage - 0.90), 0.01, 0.30)
                        ),
                        "status": "aci_proxy_forward_update",
                    }
                )
                alpha = float(np.clip(alpha + eta * (coverage - 0.90), 0.01, 0.30))
        return pd.DataFrame(rows)

    monthly = monthly.copy()
    monthly["all"] = "all"
    monthly_aci = aci(monthly, ["all"])
    grade_aci = aci(grade.copy(), ["grade"]) if not grade.empty else pd.DataFrame()
    return monthly_aci, grade_aci


def build_mdcp_and_fairness(
    base: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_rows = []
    for source_family, columns in {
        "grade": ["original_grade"],
        "period": ["period"],
        "grade_period": ["original_grade", "period"],
    }.items():
        grouped = evidence.groupby(["policy_id", *columns], as_index=False)
        local = grouped.agg(
            n_funded=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            coverage_alpha01=("miscovered_alpha01", lambda x: float(1.0 - x.astype(float).mean())),
            ecl_baseline_lgd45=("ecl_baseline_lgd45", "sum"),
        )
        local["source_family"] = source_family
        local["source_value"] = local[columns].astype(str).agg("|".join, axis=1)
        source_rows.append(
            local[
                [
                    "policy_id",
                    "source_family",
                    "source_value",
                    "n_funded",
                    "funded_exposure",
                    "coverage_alpha01",
                    "ecl_baseline_lgd45",
                ]
            ]
        )
    source_registry = pd.concat(source_rows, ignore_index=True)
    worst = (
        source_registry.groupby(["policy_id", "source_family"], as_index=False)
        .agg(
            worst_source_coverage_alpha01=("coverage_alpha01", "min"),
            worst_source_ecl=("ecl_baseline_lgd45", "max"),
            sources=("source_value", "nunique"),
        )
        .sort_values(["policy_id", "source_family"])
    )
    worst["mdcp_proxy_pass"] = worst["worst_source_coverage_alpha01"].ge(0.90)
    worst["status"] = "mdcp_worst_source_proxy"

    universe = base.copy()
    universe["universe_exposure"] = universe["loan_amnt"]
    universe_grade = universe.groupby("original_grade", as_index=False).agg(
        universe_n=("id", "nunique"), universe_exposure=("universe_exposure", "sum")
    )
    universe_grade["universe_exposure_share"] = universe_grade["universe_exposure"] / max(
        float(universe_grade["universe_exposure"].sum()), 1e-12
    )
    funded_grade = evidence.groupby(["policy_id", "original_grade"], as_index=False).agg(
        funded_n=("loan_id", "nunique"),
        funded_exposure=("funded_exposure", "sum"),
        ecl_baseline_lgd45=("ecl_baseline_lgd45", "sum"),
        observed_default_rate=("y_true", "mean"),
    )
    funded_totals = funded_grade.groupby("policy_id", as_index=False).agg(
        total_funded_exposure=("funded_exposure", "sum"),
        total_ecl=("ecl_baseline_lgd45", "sum"),
    )
    fairness = funded_grade.merge(funded_totals, on="policy_id", how="left").merge(
        universe_grade, on="original_grade", how="left"
    )
    fairness["funded_exposure_share"] = fairness["funded_exposure"] / fairness[
        "total_funded_exposure"
    ].clip(lower=1e-12)
    fairness["ecl_share"] = fairness["ecl_baseline_lgd45"] / fairness["total_ecl"].clip(lower=1e-12)
    fairness["representation_gap_vs_universe"] = (
        fairness["funded_exposure_share"] - fairness["universe_exposure_share"]
    )
    fairness["stress_flag"] = fairness["representation_gap_vs_universe"].abs().gt(0.10)
    fairness["status"] = "proxy_fairness_grade_distribution_stress"
    return source_registry, worst, fairness


def build_multi_period_toy(
    monthly_replay: pd.DataFrame,
    selector: pd.DataFrame,
    ifrs9_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_ids = ["paper1_economic_champion"]
    selected_ids.extend(
        selector.loc[selector["diagnostic_selector_rank"].le(2), "policy_id"].astype(str).tolist()
    )
    selected_ids = list(dict.fromkeys(selected_ids))
    monthly = monthly_replay[monthly_replay["policy_id"].isin(selected_ids)].copy()
    monthly = monthly.sort_values(["policy_id", "month"])
    rows = []
    for policy_id, group in monthly.groupby("policy_id"):
        budget_start = BUDGET
        for step, (_, row) in enumerate(group.iterrows(), start=1):
            provision = float(row.get("provision", 0.0))
            realized = float(row.get("realized_return_proxy_lgd45", 0.0))
            budget_end = max(0.0, budget_start + realized - provision)
            rows.append(
                {
                    "policy_id": policy_id,
                    "month": row["month"],
                    "period": row["period"],
                    "step": step,
                    "budget_start": budget_start,
                    "funded_exposure": float(row.get("funded_exposure", 0.0)),
                    "capital_used": float(row.get("funded_exposure", 0.0)),
                    "ecl": provision,
                    "realized_return_proxy_lgd45": realized,
                    "stage2_or_3_share": float(row.get("stage2_or_3_share", np.nan)),
                    "coverage_alpha01": float(row.get("coverage_alpha01", np.nan)),
                    "budget_end": budget_end,
                    "transition_status": "toy_budget_return_minus_provision",
                }
            )
            budget_start = budget_end
    state = pd.DataFrame(rows)
    horizon_rows = []
    for horizon in (3, 6):
        for policy_id, group in state.groupby("policy_id"):
            group = group.sort_values("month")
            for start in range(0, max(len(group) - horizon + 1, 0)):
                window = group.iloc[start : start + horizon]
                horizon_rows.append(
                    {
                        "policy_id": policy_id,
                        "horizon_months": horizon,
                        "start_month": window["month"].iloc[0],
                        "end_month": window["month"].iloc[-1],
                        "window_ecl": float(window["ecl"].sum()),
                        "window_realized_return": float(
                            window["realized_return_proxy_lgd45"].sum()
                        ),
                        "window_net_after_ecl": float(
                            (window["realized_return_proxy_lgd45"] - window["ecl"]).sum()
                        ),
                        "min_coverage_alpha01": float(window["coverage_alpha01"].min()),
                        "mean_stage2_or_3_share": float(window["stage2_or_3_share"].mean()),
                        "status": "toy_dla_window_summary",
                    }
                )
    policy_json = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": "paper4_multi_period_dla_toy",
        "class": "DLA",
        "horizons_months": [3, 6],
        "state": ["budget", "funded_exposure", "ECL", "stage_mix", "coverage"],
        "transition": "budget_{t+1}=max(0,budget_t+realized_return_t-provision_t)",
        "scope": "toy_replay_for_learning_not_promotion",
        "selected_policy_ids": selected_ids,
        "promotion_eligible": False,
    }
    return state, pd.DataFrame(horizon_rows), policy_json


def build_causal_expansion(evidence: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    cate = _safe_read_parquet(ROOT / "data" / "processed" / "cate_estimates_oot.parquet")
    if cate.empty:
        return pd.DataFrame(), _write_note(
            "paper4_causal_identification_dossier.md",
            "# Paper 4 Causal Identification Dossier\n\nCATE artifact missing.\n",
        )
    merged = evidence.merge(
        cate[["id", "cate", "cate_lb", "cate_ub", "grade"]].rename(columns={"id": "loan_id"}),
        on="loan_id",
        how="left",
    )
    merged["toy_loss_reduction_value"] = (
        -merged["cate"].clip(upper=0.0) * DEFAULT_LGD * merged["funded_exposure"]
    )
    value = (
        merged.groupby("policy_id", as_index=False)
        .agg(
            n_funded=("loan_id", "nunique"),
            mean_cate=("cate", "mean"),
            mean_cate_lb=("cate_lb", "mean"),
            mean_cate_ub=("cate_ub", "mean"),
            share_negative_cate=("cate", lambda x: float((x < 0).mean())),
            toy_loss_reduction_value=("toy_loss_reduction_value", "sum"),
        )
        .sort_values("toy_loss_reduction_value", ascending=False)
    )
    causal_rule = _safe_read_json(ROOT / "models" / "causal_policy_rule.json")
    cate_status = _safe_read_json(ROOT / "models" / "cate_portfolio_status.json")
    text = f"""# Paper 4 Causal Identification Dossier

This dossier expands the causal lane for Paper 4 without promoting CATE into the
main objective.

## Current gate

- Causal rule: `{causal_rule.get("selected_rule", "N/D")}`
- Rule promotion state: `{causal_rule.get("promotion_state", "N/D")}`
- Overlap pass: `{causal_rule.get("overlap_pass", "N/D")}`
- Sensitivity pass: `{causal_rule.get("sensitivity_pass", "N/D")}`
- CATE portfolio state: `{cate_status.get("promotion_state", "N/D")}`

## Toy policy value

`paper4_cate_policy_value_toy.csv` computes a diagnostic loss-reduction proxy:

```text
toy_value = -min(CATE, 0) * LGD * funded_exposure
```

This reads negative CATE as a reduction in default probability. It is a toy
calculation, not a causal claim. It can become a policy objective only after a
clean treatment, outcome, overlap report, sensitivity report and policy-value
estimator are accepted.

## Decision

Keep CATE as `B_t`/future-intervention hypothesis. Do not use it as `C_t` or
`X^pi` in the Paper 4 selector yet.
"""
    return value, _write_note("paper4_causal_identification_dossier.md", text)


def build_artifact_registry(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "artifact": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "role": "paper4_extended_experiment_artifact",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": rows,
        "promotion_json_created": False,
    }


def main() -> None:
    generated: list[Path] = []
    base = _load_base_loan_frame()
    policy_universe = load_policy_universe()

    evidence, sparse_matrix, composition = build_policy_loan_evidence(base, policy_universe)
    generated.append(_write_parquet("paper4_policy_loan_level_evidence.parquet", evidence))
    generated.append(_write_parquet("paper4_policy_loan_matrix.parquet", sparse_matrix))
    generated.append(_write_csv("paper4_policy_funded_set_composition.csv", composition))

    ifrs9_grid, stage_mix, ifrs9_summary = build_ifrs9_full(evidence)
    generated.append(_write_parquet("paper4_ifrs9_loan_policy_grid.parquet", ifrs9_grid))
    generated.append(_write_csv("paper4_ifrs9_policy_stage_mix.csv", stage_mix))
    generated.append(_write_csv("paper4_table12_ifrs9_policy_full_eval.csv", ifrs9_summary))

    monthly_replay, monthly_summary = build_monthly_policy_replay(evidence, ifrs9_grid)
    generated.append(_write_parquet("paper4_monthly_policy_replay.parquet", monthly_replay))
    generated.append(
        _write_csv("paper4_table13_monthly_policy_replay_summary.csv", monthly_summary)
    )

    selector_config, selector = build_diagnostic_selector(policy_universe, ifrs9_summary, evidence)
    generated.append(_write_json("paper4_diagnostic_selector_config.json", selector_config))
    generated.append(_write_csv("paper4_diagnostic_selector_results.csv", selector))
    generated.append(_write_csv("paper4_table14_ifrs9_tail_satisficing_selector.csv", selector))

    frontier, figure_paths = write_pareto_frontiers(selector)
    generated.append(_write_csv("paper4_return_ecl_tail_frontier.csv", frontier))
    generated.extend(figure_paths)

    bootstrap = build_bootstrap_ci(monthly_replay)
    generated.append(_write_csv("paper4_policy_pairwise_bootstrap_ci.csv", bootstrap))
    robust_sat = build_robust_satisficing_policy(selector)
    generated.append(_write_csv("paper4_robust_satisficing_policy_eval.csv", robust_sat))

    monthly_aci, grade_aci = build_online_conformal_aci()
    generated.append(_write_parquet("paper4_online_conformal_aci_replay.parquet", monthly_aci))
    generated.append(_write_parquet("paper4_online_conformal_grade_replay.parquet", grade_aci))

    source_registry, mdcp_worst, fairness = build_mdcp_and_fairness(base, evidence)
    generated.append(_write_csv("paper4_source_segment_registry.csv", source_registry))
    generated.append(_write_csv("paper4_mdcp_worst_source_coverage.csv", mdcp_worst))
    generated.append(_write_csv("paper4_fairness_policy_stress.csv", fairness))
    fairness_screen = (
        fairness.groupby("policy_id", as_index=False)
        .agg(
            max_abs_representation_gap=(
                "representation_gap_vs_universe",
                lambda x: float(x.abs().max()),
            ),
            stressed_groups=("stress_flag", "sum"),
            total_groups=("original_grade", "nunique"),
        )
        .sort_values("max_abs_representation_gap")
    )
    fairness_screen["fairness_proxy_pass"] = fairness_screen["max_abs_representation_gap"].le(0.10)
    fairness_screen["status"] = "fairness_proxy_screen_grade_distribution"
    generated.append(_write_csv("paper4_fairness_constraint_screen.csv", fairness_screen))

    state, horizons, policy_json = build_multi_period_toy(monthly_replay, selector, ifrs9_grid)
    generated.append(_write_parquet("paper4_dla_toy_state_replay.parquet", state))
    generated.append(_write_parquet("paper4_state_transition_toy.parquet", state))
    generated.append(_write_csv("paper4_dla_toy_horizon_summary.csv", horizons))
    generated.append(_write_json("paper4_multi_period_toy_policy.json", policy_json))

    cate_value, causal_note = build_causal_expansion(evidence)
    generated.append(_write_csv("paper4_cate_policy_value_toy.csv", cate_value))
    generated.append(causal_note)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "extended_p0_p1_p2_lanes",
        "mode": "diagnostic_living_lab_not_promotion",
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "n_policies": int(policy_universe["policy_id"].nunique()),
        "n_policy_loan_rows": int(len(evidence)),
        "reconstruction_caveat": (
            "Champion funded set is exact from Paper Estrella; non-champion policy-loan rows "
            "are policy-implied greedy proxies because allocation vectors were not persisted."
        ),
        "generated_artifacts": [p.relative_to(ROOT).as_posix() for p in generated],
    }
    generated.append(_write_json("paper4_extended_lanes_status.json", status))
    registry_path = _write_json(
        "paper4_extended_artifact_registry.json", build_artifact_registry(generated)
    )
    generated.append(registry_path)
    print(json.dumps({"generated": [p.relative_to(ROOT).as_posix() for p in generated]}, indent=2))


if __name__ == "__main__":
    main()
