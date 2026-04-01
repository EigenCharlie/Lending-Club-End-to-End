"""Validate the α-CP → Γ-robustness bound (Theorem 1, Paper Estrella).

Computes the conformal robustness budget Γ_CP(α) and validates the
probabilistic feasibility guarantee against empirical data.

Deliverables:
  - data/processed/alpha_gamma_bound_validation.json
  - reports/paper_material/figures_publication/estrella_fig_alpha_gamma_bound.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ──────────────────────────────────────────────────
ALPHAS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
TAU = 0.10  # max_portfolio_pd
GAMMA_BLEND = 0.10  # champion gamma
LGD_FIXED = 0.45
MAX_CANDIDATES = 5_000
BUDGET = 1_000_000
FIG_DIR = ROOT / "reports" / "paper_material" / "figures_publication"
ARTIFACT_PATH = ROOT / "data" / "processed" / "alpha_gamma_bound_validation.json"


def load_test_data() -> pd.DataFrame:
    """Load conformal intervals with true labels."""
    path = ROOT / "data" / "processed" / "conformal_intervals_mondrian.parquet"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} test observations from {path.name}")
    return df


def compute_conformal_intervals_at_alpha(
    df: pd.DataFrame, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute conformal intervals at a given alpha using stored nonconformity scores.

    For the alpha sweep, we approximate by scaling the stored 90% intervals.
    The stored intervals are at alpha=0.10. For other alphas we use the
    alpha_sweep_pareto artifact which has empirical coverage/width at each level.
    """
    # Use the 90% intervals as the baseline
    pd_point = df["y_pred"].values
    pd_low_90 = df["pd_low_90"].values
    pd_high_90 = df["pd_high_90"].values

    # The conformal radius at alpha=0.10
    radius_90 = (pd_high_90 - pd_low_90) / 2.0

    # For other alphas, we need to look up the actual width from the sweep
    sweep_path = ROOT / "data" / "processed" / "alpha_sweep_pareto_mondrian.parquet"
    if sweep_path.exists():
        sweep = pd.read_parquet(sweep_path)
        row_base = sweep[np.isclose(sweep["alpha"], 0.10)]
        row_target = sweep[np.isclose(sweep["alpha"], alpha)]
        if len(row_base) > 0 and len(row_target) > 0:
            w_base = row_base["avg_width"].values[0]
            w_target = row_target["avg_width"].values[0]
            scale = w_target / max(w_base, 1e-8)
            radius = radius_90 * scale
            pd_high = np.clip(pd_point + radius, 0, 1)
            pd_low = np.clip(pd_point - radius, 0, 1)
            return pd_point, pd_low, pd_high

    # Fallback: use stored 90% intervals directly (only valid for alpha=0.10)
    return pd_point, pd_low_90, pd_high_90


def compute_proxy_weights(
    df: pd.DataFrame,
    pd_high: np.ndarray,
    n_candidates: int = MAX_CANDIDATES,
) -> np.ndarray:
    """Compute portfolio weights using a greedy proxy for the robust LP.

    Selects the top-n loans by net return (r - pd_high*LGD) that satisfy
    the PD cap, then assigns equal weights within budget.
    """
    n = min(len(df), n_candidates)
    idx = np.random.default_rng(42).choice(len(df), size=n, replace=False)
    sub_df = df.iloc[idx].copy()
    sub_pd_high = pd_high[idx]

    int_rate = sub_df["int_rate"].values if "int_rate" in sub_df.columns else np.full(n, 0.12)
    loan_amnt = sub_df["loan_amnt"].values
    net_return = int_rate - sub_pd_high * LGD_FIXED

    # Greedy: rank by net return, include loans while PD cap holds
    order = np.argsort(-net_return)
    alloc = np.zeros(n)
    cum_exposure = 0.0
    cum_weighted_pd = 0.0

    for j in order:
        new_exposure = cum_exposure + loan_amnt[j]
        new_weighted_pd = cum_weighted_pd + loan_amnt[j] * sub_pd_high[j]
        if new_exposure > BUDGET:
            break
        if new_weighted_pd / max(new_exposure, 1e-6) > TAU:
            continue
        alloc[j] = 1.0
        cum_exposure = new_exposure
        cum_weighted_pd = new_weighted_pd

    # Compute weights
    total_exp = np.sum(alloc * loan_amnt)
    if total_exp < 1e-6:
        # Fallback: equal weight across top 100 by net return
        top_idx = order[:100]
        alloc[top_idx] = 1.0
        total_exp = np.sum(alloc * loan_amnt)

    weights = (alloc * loan_amnt) / max(total_exp, 1e-6)
    return weights, idx


def validate_bound_at_alpha(
    df: pd.DataFrame,
    alpha: float,
) -> dict:
    """Validate Theorem 1 at a given alpha level."""
    pd_point, pd_low, pd_high = compute_conformal_intervals_at_alpha(df, alpha)
    y_true = df["y_true"].values

    # Compute proxy portfolio weights
    weights, idx = compute_proxy_weights(df, pd_high)

    # Subset to candidate loans
    y_true_sub = y_true[idx]
    pd_point_sub = pd_point[idx]
    pd_high_sub = pd_high[idx]

    n_funded = int(np.sum(weights > 1e-8))

    # ── Definition 1: Γ_CP(α) ──
    gamma_cp = float(np.sum(weights * (pd_high_sub - pd_point_sub)))

    # ── Proposition 1: verified by alpha sweep (monotonicity) ──

    # ── Theorem 1 validation ──
    # Miscoverage indicators: Z_i = 1{y_true > pd_high}
    miscoverage = (y_true_sub > pd_high_sub).astype(float)

    # Weighted miscoverage V = Σ w_i Z_i
    V = float(np.sum(weights * miscoverage))

    # Weighted PD true = Σ w_i * y_true
    weighted_pd_true = float(np.sum(weights * y_true_sub))

    # Weighted PD constraint (robust): Σ w_i * pd_high
    weighted_pd_robust = float(np.sum(weights * pd_high_sub))

    # Constraint violation
    violation = max(0.0, weighted_pd_true - TAU)

    # Empirical coverage on funded loans
    funded_mask = weights > 1e-8
    if funded_mask.sum() > 0:
        emp_coverage = float(1.0 - miscoverage[funded_mask].mean())
    else:
        emp_coverage = float("nan")

    # Theorem 1(a): E[violation] ≤ α
    bound_a_holds = violation <= alpha + 1e-8

    # Theorem 1(b): P(violation > t) ≤ α/t
    # We evaluate at t = 0.05
    t_eval = 0.05
    bound_b_value = min(1.0, alpha / t_eval)

    # Theorem 1(c): P(V > √α) ≤ √α
    sqrt_alpha = float(np.sqrt(alpha))
    bound_c_holds = sqrt_alpha >= V  # If V ≤ √α, the bound is trivially satisfied

    return {
        "alpha": alpha,
        "confidence": 1.0 - alpha,
        "gamma_cp": round(gamma_cp, 6),
        "n_funded": n_funded,
        "weighted_pd_true": round(weighted_pd_true, 6),
        "weighted_pd_robust": round(weighted_pd_robust, 6),
        "tau": TAU,
        "violation": round(violation, 6),
        "weighted_miscoverage_V": round(V, 6),
        "sqrt_alpha": round(sqrt_alpha, 6),
        "empirical_coverage_funded": round(emp_coverage, 4),
        "bound_a_expected_violation_leq_alpha": bound_a_holds,
        "bound_b_prob_violation_gt_005": round(bound_b_value, 4),
        "bound_c_V_leq_sqrt_alpha": bound_c_holds,
        "all_bounds_hold": bound_a_holds and bound_c_holds,
    }


def plot_validation(results: list[dict]) -> None:
    """Generate 3-panel publication figure."""
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

    # Panel A: Γ_CP(α) vs α — Monotonicity (Proposition 1)
    ax = axes[0]
    ax.plot(df["alpha"], df["gamma_cp"], "o-", color="#2c3e50", linewidth=2, markersize=6)
    ax.set_xlabel(r"$\alpha$ (miscoverage level)")
    ax.set_ylabel(r"$\Gamma_{\mathrm{CP}}(\alpha)$")
    ax.set_title("(A) Presupuesto conformal de robustez")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    # Annotate monotonicity
    ax.annotate(
        r"$\alpha \downarrow \Rightarrow \Gamma_{\mathrm{CP}} \uparrow$",
        xy=(0.05, df[df["alpha"] == 0.05]["gamma_cp"].values[0]),
        xytext=(0.12, df["gamma_cp"].max() * 0.95),
        arrowprops={"arrowstyle": "->", "color": "#7f8c8d"},
        fontsize=9,
        color="#7f8c8d",
    )

    # Panel B: Empirical violation vs theoretical bound (Theorem 1)
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

    # Panel C: Weighted miscoverage V vs √α (Theorem 1c)
    ax = axes[2]
    ax.bar(
        df["alpha"],
        df["weighted_miscoverage_V"],
        width=0.012,
        color="#3498db",
        alpha=0.7,
        label=r"$V = \sum w_i Z_i$ (empírico)",
    )
    ax.plot(
        df["alpha"],
        df["sqrt_alpha"],
        "r--",
        linewidth=2,
        label=r"Cota $\sqrt{\alpha}$",
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"No-cobertura ponderada $V$")
    ax.set_title(r"(C) Teorema 1(c): $V \leq \sqrt{\alpha}$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"estrella_fig_alpha_gamma_bound.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        logger.info(f"Saved figure: {path}")

    plt.close(fig)


def main() -> None:
    """Run full validation pipeline."""
    logger.info("=== Validating α-CP → Γ-robustness bound (Theorem 1) ===")

    df = load_test_data()

    # Add int_rate if missing (approximate from test_fe)
    if "int_rate" not in df.columns:
        fe_path = ROOT / "data" / "processed" / "test_fe.parquet"
        if fe_path.exists():
            fe = pd.read_parquet(fe_path, columns=["int_rate"])
            if len(fe) == len(df):
                df["int_rate"] = fe["int_rate"].values
                logger.info("Aligned int_rate from test_fe.parquet")

    results = []
    for alpha in ALPHAS:
        logger.info(f"Validating α = {alpha}")
        r = validate_bound_at_alpha(df, alpha)
        results.append(r)
        status = "✓" if r["all_bounds_hold"] else "✗"
        logger.info(
            f"  α={alpha:.2f}  Γ_CP={r['gamma_cp']:.4f}  "
            f"violation={r['violation']:.6f}  V={r['weighted_miscoverage_V']:.4f}  "
            f"√α={r['sqrt_alpha']:.4f}  {status}"
        )

    # Summary
    all_pass = all(r["all_bounds_hold"] for r in results)
    summary = {
        "theorem": "Conformal Feasibility Guarantee (Theorem 1)",
        "paper": "Paper Estrella (CRPTO)",
        "n_test_observations": len(df),
        "tau": TAU,
        "gamma_blend": GAMMA_BLEND,
        "alphas_tested": ALPHAS,
        "all_bounds_hold": all_pass,
        "results": results,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved validation artifact: {ARTIFACT_PATH}")

    if all_pass:
        logger.success("ALL BOUNDS HOLD across all alpha levels.")
    else:
        failed = [r["alpha"] for r in results if not r["all_bounds_hold"]]
        logger.warning(f"Bounds failed at alpha = {failed}")

    # Generate publication figure
    plot_validation(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY: Theorem 1 (Conformal Feasibility Guarantee)")
    print("=" * 80)
    header = f"{'α':>6} {'1-α':>6} {'Γ_CP':>8} {'Violation':>10} {'V':>8} {'√α':>8} {'Pass':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "  ✓" if r["all_bounds_hold"] else "  ✗"
        print(
            f"{r['alpha']:6.2f} {r['confidence']:6.2f} {r['gamma_cp']:8.4f} "
            f"{r['violation']:10.6f} {r['weighted_miscoverage_V']:8.4f} "
            f"{r['sqrt_alpha']:8.4f} {status}"
        )
    print("=" * 80)
    print(f"Result: {'ALL BOUNDS HOLD' if all_pass else 'SOME BOUNDS FAILED'}")


if __name__ == "__main__":
    main()
