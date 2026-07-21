"""Validate the α-CP → Γ-robustness bound (Theorem 1, Paper Estrella)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.optimize_portfolio_tradeoff import (  # noqa: E402
    _align_loans_and_intervals,
    _load_candidates,
    _parse_percent_series,
)
from src.models.conformal_artifacts import load_conformal_intervals  # noqa: E402
from src.optimization.portfolio_model import (  # noqa: E402
    compute_effective_pd,
    optimize_portfolio_allocation,
)

DEFAULT_ALPHAS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
DEFAULT_MAX_CANDIDATES = 5_000
DEFAULT_RANDOM_STATE = 42
DEFAULT_T_EVAL = 0.05
DEFAULT_BUDGET = 1_000_000.0
DEFAULT_MAX_CONCENTRATION = 0.25
DEFAULT_LGD = 0.45
DEFAULT_TIME_LIMIT = 300
DEFAULT_THREADS = 4
DEFAULT_SOLVER = "highs"
FIG_DIR = ROOT / "reports" / "paper_material" / "figures_publication"
DEFAULT_FIG_PREFIX = FIG_DIR / "estrella_fig_alpha_gamma_bound"
DEFAULT_EXACT_JSON = ROOT / "data" / "processed" / "alpha_gamma_bound_validation_exact.json"
DEFAULT_PROXY_JSON = ROOT / "data" / "processed" / "alpha_gamma_bound_validation.json"
DEFAULT_COMPARISON_PATH = ROOT / "data" / "processed" / "alpha_gamma_bound_proxy_vs_exact.parquet"


def _coerce_alpha_grid(raw: str | None) -> list[float]:
    if not raw:
        return list(DEFAULT_ALPHAS)
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("alpha-grid cannot be empty")
    return values


def _load_policy(policy_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    selected = payload.get("selected_policy", payload)
    return {
        "source_path": str(policy_path),
        "risk_tolerance": float(selected.get("risk_tolerance", 0.10)),
        "uncertainty_aversion": float(selected.get("uncertainty_aversion", 0.0)),
        "min_budget_utilization": float(selected.get("min_budget_utilization", 0.0)),
        "pd_cap_slack_penalty": float(selected.get("pd_cap_slack_penalty", 0.0)),
        "policy_mode": str(selected.get("policy_mode", "hard_worst_case")),
        "gamma": float(selected.get("gamma", 1.0)),
        "delta_cap_quantile": float(selected.get("delta_cap_quantile", 1.0)),
        "tail_focus_quantile": float(selected.get("tail_focus_quantile", 1.0)),
        "solver_backend": str(selected.get("solver_backend", DEFAULT_SOLVER)),
    }


def _resolve_interval_columns(intervals: pd.DataFrame) -> tuple[str, str, str]:
    col_point = "y_pred" if "y_pred" in intervals.columns else "pd_point"
    col_low = "pd_low_90" if "pd_low_90" in intervals.columns else "pd_low"
    col_high = "pd_high_90" if "pd_high_90" in intervals.columns else "pd_high"
    return col_point, col_low, col_high


def _load_aligned_dataset(
    *,
    conformal_intervals_path: str | None,
    max_candidates: int,
    random_state: int,
) -> pd.DataFrame:
    candidates = _load_candidates().reset_index(drop=True)
    intervals, path, is_legacy = load_conformal_intervals(
        allow_legacy_fallback=False,
        override_path=conformal_intervals_path,
    )
    logger.info(
        "Loaded conformal intervals from {} (legacy={}, rows={:,})",
        path,
        is_legacy,
        len(intervals),
    )
    loans, ints = _align_loans_and_intervals(
        candidates=candidates,
        intervals=intervals.reset_index(drop=True),
        max_candidates=max_candidates,
        random_state=random_state,
    )
    aligned = loans.reset_index(drop=True).copy()
    for column in ints.columns:
        aligned[column] = ints[column].reset_index(drop=True)
    logger.info(
        "Aligned loans and intervals for bound validation: n={:,}, override={}",
        len(aligned),
        str(conformal_intervals_path or path),
    )
    return aligned


def _policy_segment_labels(loans: pd.DataFrame, policy_mode: str) -> np.ndarray | None:
    if str(policy_mode).strip().lower() not in {
        "segment_tail_blended_uncertainty",
        "segment_relative_tail_blended_uncertainty",
    }:
        return None
    grade = (
        loans["grade"].fillna("unknown").astype(str)
        if "grade" in loans.columns
        else pd.Series(["unknown"] * len(loans))
    )
    term = (
        loans["term"].fillna("unknown").astype(str)
        if "term" in loans.columns
        else pd.Series(["unknown"] * len(loans))
    )
    verification = (
        loans["verification_status"].fillna("unknown").astype(str)
        if "verification_status" in loans.columns
        else pd.Series(["unknown"] * len(loans))
    )
    return (grade + "|" + term + "|" + verification).to_numpy(dtype=object)


def _compute_intervals_at_alpha(
    frame: pd.DataFrame,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_point, col_low, col_high = _resolve_interval_columns(frame)
    pd_point = pd.to_numeric(frame[col_point], errors="coerce").to_numpy(dtype=float)
    pd_low_90 = pd.to_numeric(frame[col_low], errors="coerce").to_numpy(dtype=float)
    pd_high_90 = pd.to_numeric(frame[col_high], errors="coerce").to_numpy(dtype=float)
    radius_90 = (pd_high_90 - pd_low_90) / 2.0

    sweep_path = ROOT / "data" / "processed" / "alpha_sweep_pareto_mondrian.parquet"
    if sweep_path.exists():
        sweep = pd.read_parquet(sweep_path)
        row_base = sweep[np.isclose(sweep["alpha"], 0.10)]
        row_target = sweep[np.isclose(sweep["alpha"], alpha)]
        if len(row_base) > 0 and len(row_target) > 0:
            w_base = float(row_base["avg_width"].values[0])
            w_target = float(row_target["avg_width"].values[0])
            scale = w_target / max(w_base, 1e-8)
            radius = radius_90 * scale
            pd_high = np.clip(pd_point + radius, 0, 1)
            pd_low = np.clip(pd_point - radius, 0, 1)
            return pd_point, pd_low, pd_high
    return pd_point, pd_low_90, pd_high_90


def _compute_effective_pd_vector(
    loans: pd.DataFrame,
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    policy: dict[str, Any],
) -> np.ndarray:
    return compute_effective_pd(
        pd_point=pd_point,
        pd_high=pd_high,
        policy_mode=str(policy["policy_mode"]),
        gamma=float(policy["gamma"]),
        delta_cap_quantile=float(policy["delta_cap_quantile"]),
        tail_focus_quantile=float(policy["tail_focus_quantile"]),
        segment_labels=_policy_segment_labels(loans, str(policy["policy_mode"])),
    )


def _compute_proxy_weights(
    loans: pd.DataFrame,
    *,
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    effective_pd: np.ndarray,
    policy: dict[str, Any],
    budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    int_rates = (
        _parse_percent_series(loans["int_rate"])
        if "int_rate" in loans.columns
        else np.full(len(loans), 0.12)
    )
    loan_amounts = (
        pd.to_numeric(loans["loan_amnt"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if "loan_amnt" in loans.columns
        else np.ones(len(loans), dtype=float)
    )
    lgd = np.full(len(loans), DEFAULT_LGD, dtype=float)
    net_return = int_rates - effective_pd * lgd
    order = np.argsort(-net_return)
    alloc = np.zeros(len(loans), dtype=float)
    cum_exposure = 0.0
    cum_weighted_pd = 0.0

    for j in order:
        new_exposure = cum_exposure + loan_amounts[j]
        new_weighted_pd = cum_weighted_pd + loan_amounts[j] * effective_pd[j]
        if new_exposure > budget:
            break
        if new_weighted_pd / max(new_exposure, 1e-6) > float(policy["risk_tolerance"]):
            continue
        alloc[j] = 1.0
        cum_exposure = new_exposure
        cum_weighted_pd = new_weighted_pd

    if cum_exposure < 1e-6:
        top_n = min(100, len(order))
        alloc[order[:top_n]] = 1.0
        cum_exposure = float(np.sum(alloc * loan_amounts))

    weights = (alloc * loan_amounts) / max(cum_exposure, 1e-6)
    return weights, {
        "solver_status": "proxy_greedy",
        "total_allocated": float(cum_exposure),
        "n_funded": int(np.sum(weights > 1e-8)),
        "weighted_pd_constraint_used": float(np.sum(weights * effective_pd)),
        "weighted_pd_high": float(np.sum(weights * pd_high)),
        "weighted_pd_point": float(np.sum(weights * pd_point)),
    }


def _compute_exact_weights(
    loans: pd.DataFrame,
    *,
    pd_point: np.ndarray,
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    effective_pd: np.ndarray,
    policy: dict[str, Any],
    budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    int_rates = (
        _parse_percent_series(loans["int_rate"])
        if "int_rate" in loans.columns
        else np.full(len(loans), 0.12)
    )
    lgd = np.full(len(loans), DEFAULT_LGD, dtype=float)
    solution = optimize_portfolio_allocation(
        loans=loans,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        lgd=lgd,
        int_rates=int_rates,
        total_budget=budget,
        max_concentration=DEFAULT_MAX_CONCENTRATION,
        max_portfolio_pd=float(policy["risk_tolerance"]),
        robust=True,
        uncertainty_aversion=float(policy["uncertainty_aversion"]),
        min_budget_utilization=float(policy["min_budget_utilization"]),
        pd_cap_slack_penalty=float(policy["pd_cap_slack_penalty"]),
        pd_constraint_override=effective_pd,
        time_limit=DEFAULT_TIME_LIMIT,
        threads=DEFAULT_THREADS,
        solver_backend=str(policy["solver_backend"]),
    )
    loan_amounts = (
        pd.to_numeric(loans["loan_amnt"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if "loan_amnt" in loans.columns
        else np.ones(len(loans), dtype=float)
    )
    alloc = np.array(
        [float(solution["allocation"].get(i, 0.0)) for i in range(len(loans))], dtype=float
    )
    total_allocated = float(np.sum(alloc * loan_amounts))
    weights = (alloc * loan_amounts) / max(total_allocated, 1e-6)
    return weights, {
        "solver_status": str(solution.get("solver_status", "unknown")),
        "total_allocated": total_allocated,
        "n_funded": int(solution.get("n_funded", int(np.sum(weights > 1e-8)))),
        "weighted_pd_constraint_used": float(np.sum(weights * effective_pd)),
        "weighted_pd_high": float(np.sum(weights * pd_high)),
        "weighted_pd_point": float(np.sum(weights * pd_point)),
        "pd_cap_slack": float(solution.get("pd_cap_slack", 0.0)),
    }


def _validate_single_alpha(
    aligned: pd.DataFrame,
    *,
    alpha: float,
    policy: dict[str, Any],
    allocator_mode: str,
    budget: float,
    t_eval: float,
) -> dict[str, Any]:
    pd_point, pd_low, pd_high = _compute_intervals_at_alpha(aligned, alpha)
    y_true = (
        pd.to_numeric(aligned["y_true"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "y_true" in aligned.columns
        else pd.to_numeric(aligned["default_flag"], errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    effective_pd = _compute_effective_pd_vector(aligned, pd_point, pd_high, policy)
    mode = str(allocator_mode).strip().lower()
    if mode == "exact":
        weights, alloc_meta = _compute_exact_weights(
            aligned,
            pd_point=pd_point,
            pd_low=pd_low,
            pd_high=pd_high,
            effective_pd=effective_pd,
            policy=policy,
            budget=budget,
        )
    elif mode == "proxy":
        weights, alloc_meta = _compute_proxy_weights(
            aligned,
            pd_point=pd_point,
            pd_high=pd_high,
            effective_pd=effective_pd,
            policy=policy,
            budget=budget,
        )
    else:
        raise ValueError(f"Unsupported allocator-mode={allocator_mode!r}")

    miscoverage = (y_true > pd_high).astype(float)
    V = float(np.sum(weights * miscoverage))
    weighted_pd_true = float(np.sum(weights * y_true))
    violation = max(0.0, weighted_pd_true - float(policy["risk_tolerance"]))
    funded_mask = weights > 1e-8
    emp_coverage = (
        float(1.0 - miscoverage[funded_mask].mean()) if funded_mask.any() else float("nan")
    )
    sqrt_alpha = float(np.sqrt(alpha))
    bound_b_value = min(1.0, alpha / max(t_eval, 1e-8))

    return {
        "alpha": float(alpha),
        "confidence": float(1.0 - alpha),
        "gamma_cp": round(float(np.sum(weights * np.clip(pd_high - pd_point, 0.0, 1.0))), 6),
        "n_funded": int(np.sum(funded_mask)),
        "weighted_pd_true": round(weighted_pd_true, 6),
        "weighted_pd_constraint_used": round(float(alloc_meta["weighted_pd_constraint_used"]), 6),
        "weighted_pd_high": round(float(alloc_meta["weighted_pd_high"]), 6),
        "weighted_pd_point": round(float(alloc_meta["weighted_pd_point"]), 6),
        "tau": float(policy["risk_tolerance"]),
        "violation": round(violation, 6),
        "weighted_miscoverage_V": round(V, 6),
        "sqrt_alpha": round(sqrt_alpha, 6),
        "empirical_coverage_funded": round(emp_coverage, 4),
        "bound_a_expected_violation_leq_alpha": bool(violation <= alpha + 1e-8),
        "bound_b_prob_violation_gt_t": round(bound_b_value, 4),
        "bound_b_t_eval": float(t_eval),
        "bound_b_is_vacuous": bool(bound_b_value >= 1.0),
        "bound_c_V_leq_sqrt_alpha": bool(sqrt_alpha + 1e-8 >= V),
        "all_bounds_hold": bool((violation <= alpha + 1e-8) and (sqrt_alpha + 1e-8 >= V)),
        "allocator_mode": mode,
        "solver_status": str(alloc_meta.get("solver_status", "unknown")),
        "total_allocated": round(float(alloc_meta.get("total_allocated", 0.0)), 2),
        "pd_cap_slack": round(float(alloc_meta.get("pd_cap_slack", 0.0)), 6),
    }


def _plot_validation(results: list[dict[str, Any]], figure_prefix: Path) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 150,
        }
    )

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(df["alpha"], df["gamma_cp"], "o-", color="#2c3e50", linewidth=2, markersize=6)
    ax.set_xlabel(r"$\alpha$ (miscoverage level)")
    ax.set_ylabel(r"$\Gamma_{\mathrm{CP}}(\alpha)$")
    ax.set_title("(A) Presupuesto conformal de robustez")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(
        df["alpha"],
        df["violation"],
        width=0.012,
        color="#27ae60",
        alpha=0.7,
        label="Violación empírica",
    )
    ax.plot(
        df["alpha"],
        df["alpha"],
        "r--",
        linewidth=2,
        label=r"Cota teórica $\mathbb{E}[\mathrm{violación}] \leq \alpha$",
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Violación de restricción PD")
    ax.set_title(r"(B) Teorema 1(a): $\mathbb{E}[\mathrm{viol.}] \leq \alpha$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(
        df["alpha"],
        df["weighted_miscoverage_V"],
        width=0.012,
        color="#3498db",
        alpha=0.7,
        label=r"$V = \sum w_i Z_i$ (empírico)",
    )
    ax.plot(df["alpha"], df["sqrt_alpha"], "r--", linewidth=2, label=r"Cota $\sqrt{\alpha}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"No-cobertura ponderada $V$")
    ax.set_title(r"(C) Teorema 1(c): $V \leq \sqrt{\alpha}$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = figure_prefix.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        logger.info("Saved figure: {}", path)
    plt.close(fig)


def _print_summary(results: list[dict[str, Any]], allocator_mode: str) -> None:
    all_pass = all(bool(r["all_bounds_hold"]) for r in results)
    print("\n" + "=" * 96)
    print(f"VALIDATION SUMMARY: Theorem 1 ({allocator_mode.upper()} allocator)")
    print("=" * 96)
    header = (
        f"{'α':>6} {'1-α':>6} {'Γ_CP':>8} {'Violation':>10} {'V':>8} "
        f"{'√α':>8} {'Mode':>8} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        status = "  ✓" if r["all_bounds_hold"] else "  ✗"
        print(
            f"{r['alpha']:6.2f} {r['confidence']:6.2f} {r['gamma_cp']:8.4f} "
            f"{r['violation']:10.6f} {r['weighted_miscoverage_V']:8.4f} "
            f"{r['sqrt_alpha']:8.4f} {r['allocator_mode']:>8} {status}"
        )
    print("=" * 96)
    print(f"Result: {'ALL BOUNDS HOLD' if all_pass else 'SOME BOUNDS FAILED'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformal-intervals-path", default=None)
    parser.add_argument("--portfolio-policy-path", default="models/champion_portfolio_policy.json")
    parser.add_argument("--allocator-mode", choices=["proxy", "exact"], default="exact")
    parser.add_argument("--alpha-grid", default=None)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--comparison-output", default=str(DEFAULT_COMPARISON_PATH))
    parser.add_argument("--figure-prefix", default=str(DEFAULT_FIG_PREFIX))
    parser.add_argument("--t-eval", type=float, default=DEFAULT_T_EVAL)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)

    logger.info("=== Validating α-CP → Γ-robustness bound (Theorem 1) ===")
    policy = _load_policy(args.portfolio_policy_path)
    aligned = _load_aligned_dataset(
        conformal_intervals_path=args.conformal_intervals_path,
        max_candidates=args.max_candidates,
        random_state=args.random_state,
    )
    alpha_grid = _coerce_alpha_grid(args.alpha_grid)
    allocator_mode = str(args.allocator_mode)

    results = []
    for alpha in alpha_grid:
        logger.info("Validating α = {} with allocator={}", alpha, allocator_mode)
        result = _validate_single_alpha(
            aligned,
            alpha=float(alpha),
            policy=policy,
            allocator_mode=allocator_mode,
            budget=float(args.budget),
            t_eval=float(args.t_eval),
        )
        results.append(result)
        status = "✓" if result["all_bounds_hold"] else "✗"
        logger.info(
            "  α={:.2f}  Γ_CP={:.4f}  violation={:.6f}  V={:.4f}  √α={:.4f}  {}",
            alpha,
            result["gamma_cp"],
            result["violation"],
            result["weighted_miscoverage_V"],
            result["sqrt_alpha"],
            status,
        )

    all_pass = all(bool(r["all_bounds_hold"]) for r in results)
    output_json = Path(
        args.output_json
        or (DEFAULT_EXACT_JSON if allocator_mode == "exact" else DEFAULT_PROXY_JSON)
    )
    summary = {
        "theorem": "Conformal Feasibility Guarantee (Theorem 1)",
        "paper": "Paper Estrella (CRPTO)",
        "allocator_mode": allocator_mode,
        "n_test_observations": int(len(aligned)),
        "policy": policy,
        "conformal_intervals_path": str(args.conformal_intervals_path or ""),
        "alphas_tested": alpha_grid,
        "all_bounds_hold": all_pass,
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Saved validation artifact: {}", output_json)

    if allocator_mode == "exact":
        proxy_results = [
            _validate_single_alpha(
                aligned,
                alpha=float(alpha),
                policy=policy,
                allocator_mode="proxy",
                budget=float(args.budget),
                t_eval=float(args.t_eval),
            )
            for alpha in alpha_grid
        ]
        exact_df = pd.DataFrame(results).add_prefix("exact_")
        proxy_df = pd.DataFrame(proxy_results).add_prefix("proxy_")
        comparison = pd.concat([exact_df, proxy_df], axis=1)
        comparison_path = Path(args.comparison_output)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_parquet(comparison_path, index=False)
        logger.info("Saved proxy-vs-exact comparison: {}", comparison_path)
        _plot_validation(results, Path(args.figure_prefix))

    if all_pass:
        logger.success("ALL BOUNDS HOLD across all alpha levels.")
    else:
        failed = [r["alpha"] for r in results if not r["all_bounds_hold"]]
        logger.warning("Bounds failed at alpha = {}", failed)

    _print_summary(results, allocator_mode=allocator_mode)


if __name__ == "__main__":
    main()
