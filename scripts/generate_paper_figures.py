"""Generate publication-quality matplotlib figures for all 3 papers.

Outputs PDF + PNG (300 DPI) under reports/paper_material/figures_publication/.

Papers:
  - Paper 3: Mondrian Conformal Prediction (COPA 2026)
  - Paper 2: IFRS9 E2E with CP (JBF/JORS)
  - Paper Estrella: Predict-then-Optimize (MS/OR/EJOR)

Usage:
    uv run python scripts/generate_paper_figures.py
    uv run python scripts/generate_paper_figures.py --paper 3
    uv run python scripts/generate_paper_figures.py --paper 2
    uv run python scripts/generate_paper_figures.py --paper estrella
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata

matplotlib.use("Agg")

# ── Publication style ─────────────────────────────────────────────────────────

# IEEE/Springer two-column format: 3.5" per column, 7" full-width
COL1 = 3.5  # single column (inches)
COL2 = 7.0  # double column (inches)
HEIGHT_S = 2.4  # short panel
HEIGHT_M = 3.2  # medium panel
HEIGHT_T = 4.2  # tall panel

# Colorblind-safe palette (Wong 2011 + Nature palette)
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
    "lgray": "#CCCCCC",
}
COLORS = list(PALETTE.values())

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.85,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.8,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
OUT_DIR = REPO_ROOT / "reports" / "paper_material" / "figures_publication"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BOOK_FIG_DIR = REPO_ROOT / "book" / "assets" / "figures" / "publication"
BOOK_FIG_DIR.mkdir(parents=True, exist_ok=True)
BOOK_OUTPUT_FIG_DIR = REPO_ROOT / "book" / "_output" / "assets" / "figures" / "publication"


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        shutil.copy2(path, BOOK_FIG_DIR / f"{name}.{ext}")
        if BOOK_OUTPUT_FIG_DIR.exists():
            BOOK_OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, BOOK_OUTPUT_FIG_DIR / f"{name}.{ext}")
    logger.info(f"Saved: {name}.pdf / .png")
    plt.close(fig)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Paper 3: Mondrian CP ──────────────────────────────────────────────────────

METHOD_LABELS = {
    "global_split": "Global Split-CP",
    "mondrian_scaled": "Mondrian (scaled)",
    "mondrian_unscaled": "Mondrian (unscaled)",
    "score_decile_mondrian": "Score-Decile Mondrian",
    "grade_x_scoreband_mondrian": "Grade×Scoreband",
    "cross_conformal_score_space": "Cross-Conformal",
}


def _paper3_fig1_variant_scatter() -> None:
    """Fig 1 — Efficiency–coverage Pareto scatter for 6 CP variants."""
    df = pd.read_parquet(DATA_DIR / "conformal_variant_benchmark.parquet")

    # Column is "variant", not "method"
    var_col = "variant" if "variant" in df.columns else "method"

    agg = (
        df.groupby(var_col)
        .agg(coverage=("coverage", "mean"), width=("avg_width", "mean"))
        .reset_index()
        .rename(columns={var_col: "method"})
    )

    fig, ax = plt.subplots(figsize=(COL2, HEIGHT_M))

    target_cov = 0.90
    ax.axvline(target_cov, color=PALETTE["red"], lw=1.0, ls="--", label="Target 90%")
    ax.axhline(agg["width"].min(), color=PALETTE["lgray"], lw=0.7, ls=":")

    offset_map = {
        "mondrian_selected_cfg": (6, 2),
        "grade_x_scoreband_mondrian": (6, 0),
        "score_decile_mondrian": (6, -8),
        "cross_conformal_score_space": (6, 4),
        "global_split": (6, -8),
        "mondrian_scaled": (6, -2),
        "mondrian_unscaled": (6, 8),
    }

    for _i, row in agg.iterrows():
        color = PALETTE["blue"] if row["coverage"] >= target_cov else PALETTE["gray"]
        ax.scatter(
            row["coverage"],
            row["width"],
            color=color,
            s=70,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )
        label = METHOD_LABELS.get(row["method"], row["method"])
        xoff, yoff = offset_map.get(row["method"], (6, 0))
        ax.annotate(
            label,
            (row["coverage"], row["width"]),
            textcoords="offset points",
            xytext=(xoff, yoff),
            fontsize=7.3,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
        )

    ax.set_xlabel("Empirical Coverage")
    ax.set_ylabel("Mean Interval Width")
    ax.set_title("Conformal Variants: Coverage–Efficiency Trade-off (OOT, $n=276{,}869$)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.893, 0.937)
    ax.set_ylim(max(0.71, agg["width"].min() - 0.01), agg["width"].max() + 0.02)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=PALETTE["blue"], label="Coverage ≥ 90% (valid)"),
            Patch(color=PALETTE["gray"], label="Coverage < 90% (invalid)"),
            plt.Line2D([0], [0], color=PALETTE["red"], ls="--", label="Target 90%"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    _save(fig, "p3_fig1_variant_scatter")


def _paper3_fig2_grade_coverage_heatmap() -> None:
    """Fig 2 — Per-grade coverage heatmap across 6 CP variants."""
    df = pd.read_parquet(DATA_DIR / "conformal_variant_benchmark_by_group.parquet")

    # Columns: group, variant (not method/grade)
    var_col = "variant" if "variant" in df.columns else "method"
    grp_col = "group" if "group" in df.columns else "grade"

    df = df[df[grp_col].astype(str).str.fullmatch(r"[A-G]")]
    pivot = df.pivot_table(index=var_col, columns=grp_col, values="coverage", aggfunc="mean")
    pivot.index = [METHOD_LABELS.get(m, m) for m in pivot.index]
    desired_order = [
        "mondrian_selected_cfg",
        "score_decile_mondrian",
        "grade_x_scoreband_mondrian",
        "cross_conformal_score_space",
        "mondrian_scaled",
        "mondrian_unscaled",
        "global_split",
    ]
    ordered_labels = [METHOD_LABELS.get(m, m) for m in desired_order if m in df[var_col].unique()]
    pivot = pivot.reindex(ordered_labels)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(COL2, HEIGHT_M))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0.70, vmax=1.00, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Conformal Method")
    ax.set_title("Per-Grade Coverage (nominal = 90%, OOT)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.isna(val):
                continue
            color = "white" if val < 0.80 or val > 0.97 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7.2, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Empirical Coverage", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "p3_fig2_grade_coverage_heatmap")


def _paper3_fig3_temporal_stability() -> None:
    """Fig 3 — Temporal coverage stability for top 3 variants."""
    df = pd.read_parquet(DATA_DIR / "conformal_temporal_diagnostics.parquet")

    if df.empty:
        logger.warning("conformal_temporal_diagnostics empty — skipping fig3")
        return

    # Pick top methods (by availability)
    method_col = "method" if "method" in df.columns else "variant"
    methods_available = df[method_col].dropna().unique().tolist()
    preferred = ["mondrian_selected_cfg", "score_decile_mondrian", "cross_conformal_score_space"]
    methods = [m for m in preferred if m in methods_available][:3]
    if not methods:
        methods = methods_available[:3]

    time_col = (
        "temporal_segment"
        if "temporal_segment" in df.columns
        else ("period" if "period" in df.columns else "month")
    )
    cov_col = (
        "coverage"
        if "coverage" in df.columns
        else ("coverage_90" if "coverage_90" in df.columns else df.columns[-1])
    )

    fig, ax = plt.subplots(figsize=(COL2, HEIGHT_M))
    ax.axhline(0.90, color=PALETTE["red"], lw=1.0, ls="--", label="Target 90%", zorder=1)

    color_cycle = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    for i, method in enumerate(methods):
        sub = df[df[method_col] == method].sort_values(time_col)
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            pd.to_datetime(sub[time_col]),
            sub[cov_col].values,
            marker="o",
            ms=4,
            color=color_cycle[i],
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Mes OOT")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Temporal Coverage Stability — Top Conformal Variants")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.70, 1.01)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "p3_fig3_temporal_stability")


# ── Paper 2: IFRS9 E2E ────────────────────────────────────────────────────────


def _paper2_fig4_sicr_grid() -> None:
    """Fig 4 — SICR trigger optimization: F1/recall/precision vs width threshold."""
    df = pd.read_parquet(DATA_DIR / "sicr_conformal_grid.parquet")

    if df.empty:
        logger.warning("sicr_conformal_grid empty — skipping fig4")
        return

    # Filter to one PD threshold (use the row with best F1)
    # Actual columns: f1_width, precision_width, recall_width_of_missed, ecl_additional
    f1_col = "f1_width" if "f1_width" in df.columns else "f1"
    prec_col = "precision_width" if "precision_width" in df.columns else "precision"
    rec_col = "recall_width_of_missed" if "recall_width_of_missed" in df.columns else "recall"

    best_pd_thr = (
        df.loc[df[f1_col].idxmax(), "pd_threshold"] if "pd_threshold" in df.columns else None
    )
    if best_pd_thr is not None:
        sub = df[df["pd_threshold"] == best_pd_thr].sort_values("width_threshold")
    else:
        sub = df.sort_values("width_threshold")

    fig, ax1 = plt.subplots(figsize=(COL2, HEIGHT_M))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    ax1.plot(
        sub["width_threshold"],
        sub[f1_col],
        color=PALETTE["blue"],
        marker="o",
        ms=4,
        label="F1 Score",
        zorder=5,
    )
    ax1.plot(
        sub["width_threshold"],
        sub[prec_col],
        color=PALETTE["orange"],
        marker="s",
        ms=4,
        ls="--",
        label="Precision",
    )
    ax1.plot(
        sub["width_threshold"],
        sub[rec_col],
        color=PALETTE["green"],
        marker="^",
        ms=4,
        ls=":",
        label="Recall",
    )

    if "ecl_additional" in sub.columns:
        ax2.fill_between(
            sub["width_threshold"], sub["ecl_additional"] / 1e6, alpha=0.12, color=PALETTE["red"]
        )
        ax2.plot(
            sub["width_threshold"],
            sub["ecl_additional"] / 1e6,
            color=PALETTE["red"],
            lw=1.0,
            ls="-.",
            label="ECL add. ($M)",
        )
        ax2.set_ylabel("Additional ECL (USD M)", color=PALETTE["red"], fontsize=9)
        ax2.tick_params(axis="y", colors=PALETTE["red"])

    # Mark optimal t*
    best_idx = sub[f1_col].idxmax()
    t_star = sub.loc[best_idx, "width_threshold"]
    ax1.axvline(
        t_star, color=PALETTE["purple"], lw=1.0, ls="--", alpha=0.7, label=f"$t^* = {t_star:.2f}$"
    )

    ax1.set_xlabel("Conformal Width Threshold $t$")
    ax1.set_ylabel("Score")
    ax1.set_title(f"SICR Trigger Optimization via Conformal Width (PD thr = {best_pd_thr:.2f})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "p2_fig4_sicr_grid")


def _paper2_fig5_ecl_alpha_sensitivity() -> None:
    """Fig 5 — ECL additional vs confidence level (alpha sensitivity)."""
    df = pd.read_parquet(DATA_DIR / "ecl_alpha_sensitivity.parquet")

    if df.empty:
        logger.warning("ecl_alpha_sensitivity empty — skipping fig5")
        return

    # Identify alpha/confidence and ECL columns
    alpha_col = next((c for c in df.columns if "alpha" in c.lower()), None)
    conf_col = next(
        (c for c in df.columns if "confidence" in c.lower() or "conf_level" in c.lower()), None
    )
    ecl_col = next((c for c in df.columns if "ecl" in c.lower() and "add" in c.lower()), None)

    if ecl_col is None:
        ecl_col = (
            [c for c in df.columns if "ecl" in c.lower()][0]
            if any("ecl" in c.lower() for c in df.columns)
            else df.columns[-1]
        )
    x_col = conf_col or alpha_col or df.columns[0]

    sub = df.sort_values(x_col)

    fig, ax = plt.subplots(figsize=(COL1 * 1.5, HEIGHT_M))

    ax.fill_between(sub[x_col], sub[ecl_col] / 1e6, alpha=0.15, color=PALETTE["blue"])
    ax.plot(
        sub[x_col],
        sub[ecl_col] / 1e6,
        color=PALETTE["blue"],
        marker="o",
        ms=5,
        label="ECL additional (SICR trigger)",
    )

    # Mark 90% level
    row_90 = sub[sub[x_col].between(0.89, 0.91)]
    if not row_90.empty:
        v90 = row_90.iloc[0][ecl_col] / 1e6
        ax.axhline(v90, color=PALETTE["gray"], lw=0.8, ls=":")
        ax.annotate(
            f"\\$${v90:.1f}M\n@ 90%",
            xy=(row_90.iloc[0][x_col], v90),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=7.5,
            color=PALETTE["gray"],
        )

    ax.set_xlabel("Confidence Level $1 - \\alpha$")
    ax.set_ylabel("Additional ECL (USD M)")
    ax.set_title("IFRS9: Additional ECL vs.\nConfidence Level")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, "p2_fig5_ecl_alpha_sensitivity")


def _paper2_fig6_bma_vs_cp() -> None:
    """Fig 6 — BMA vs Conformal Mondrian: coverage / width / min-group comparison."""
    status = _load_json(MODELS_DIR / "bma_comparison_status.json")
    # Actual structure: status["results"]["equal"]["overall_bma_coverage", ...]
    res = status.get("results", {}).get("equal", status.get("results", {}))

    labels = ["Empirical Coverage", "Mean Width", "Min-Grade Coverage"]
    cp_vals = [
        res.get("overall_cp_coverage", 0),
        res.get("mean_cp_width", 0),
        res.get("min_grade_cp_coverage", 0),
    ]
    bma_vals = [
        res.get("overall_bma_coverage", 0),
        res.get("mean_bma_width", 0),
        res.get("min_grade_bma_coverage", 0),
    ]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(COL1 * 1.5, HEIGHT_M))
    bars_cp = ax.bar(
        x - width / 2,
        cp_vals,
        width,
        label="Conformal Mondrian",
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_bm = ax.bar(
        x + width / 2,
        bma_vals,
        width,
        label="BMA",
        color=PALETTE["orange"],
        edgecolor="white",
        linewidth=0.5,
    )

    def _label_bars(bars: list, vals: list) -> None:
        for bar, v in zip(bars, vals, strict=False):
            if v:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    _label_bars(list(bars_cp), cp_vals)
    _label_bars(list(bars_bm), bma_vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("BMA vs. Conformal Mondrian:\nUncertainty Interval Comparison")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(cp_vals + bma_vals) * 1.18)
    fig.tight_layout()
    _save(fig, "p2_fig6_bma_vs_cp")


# ── Paper Estrella: Predict-then-Optimize ─────────────────────────────────────


def _estrella_fig7_uncertainty_baselines() -> None:
    """Fig 7 — 4 uncertainty set methods: coverage / width / min-group bar chart."""
    status = _load_json(MODELS_DIR / "uncertainty_baselines_status.json")

    results = status.get("results", {})
    methods_order = ["conformal_mondrian", "bootstrap", "parametric_gaussian", "ellipsoidal_grade"]
    method_labels_map = {
        "conformal_mondrian": "Mondrian CP",
        "bootstrap": "Bootstrap",
        "parametric_gaussian": "Parametric (Gauss.)",
        "ellipsoidal_grade": "Ellipsoidal",
    }

    metrics = ["empirical_coverage", "avg_width", "min_group_coverage"]
    metric_labels = ["Empirical Coverage", "Mean Width", "Min-Grade Coverage"]
    colors_m = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, HEIGHT_M))
    nominal = 0.90

    for ax, metric, mlabel in zip(axes, metrics, metric_labels, strict=False):
        vals = [results.get(m, {}).get(metric, 0) for m in methods_order]
        bars = ax.bar(
            range(len(methods_order)), vals, color=colors_m, edgecolor="white", linewidth=0.5
        )

        if metric == "empirical_coverage":
            ax.axhline(nominal, color=PALETTE["red"], lw=1.0, ls="--", label="Target 90%")
        if metric == "avg_width":
            ax.axhline(min(v for v in vals if v > 0), color=PALETTE["lgray"], lw=0.7, ls=":")

        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(range(len(methods_order)))
        ax.set_xticklabels(
            [method_labels_map[m] for m in methods_order], rotation=30, ha="right", fontsize=7.5
        )
        ax.set_ylabel(mlabel, fontsize=8)
        ax.set_title(mlabel, fontsize=9)
        if metric == "empirical_coverage":
            ax.legend(fontsize=7)

    fig.suptitle(
        "Uncertainty Set Methods: Coverage, Width, and Group Guarantee\n(OOT $n=276{,}869$, nominal 90%)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "estrella_fig7_uncertainty_baselines")


def _estrella_fig8_alpha_pareto() -> None:
    """Fig 8 — Alpha sweep Pareto: Mondrian vs Global (coverage × width × eligible loans)."""
    df = pd.read_parquet(DATA_DIR / "alpha_sweep_pareto_both.parquet")

    if df.empty:
        logger.warning("alpha_sweep_pareto_both empty — skipping fig8")
        return

    # Detect variant column
    variant_col = next(
        (c for c in df.columns if "variant" in c.lower() or "method" in c.lower()), None
    )
    alpha_col = next((c for c in df.columns if "alpha" in c.lower()), None)
    cov_col = next((c for c in df.columns if "coverage" in c.lower()), None)
    width_col = next((c for c in df.columns if "width" in c.lower()), None)
    elig_col = next(
        (c for c in df.columns if "eligible" in c.lower() or "n_eligible" in c.lower()), None
    )

    if not all([variant_col, alpha_col, cov_col, width_col]):
        logger.warning(f"alpha_sweep_pareto_both missing columns. Got: {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(COL2, HEIGHT_M))

    variants = df[variant_col].unique() if variant_col else ["global", "mondrian"]
    var_colors = {
        v: PALETTE["blue"] if "mond" in str(v).lower() else PALETTE["orange"] for v in variants
    }
    var_labels = {
        v: "Mondrian CP" if "mond" in str(v).lower() else "Global Split-CP" for v in variants
    }

    # Left: width vs coverage scatter (Pareto)
    ax = axes[0]
    ax.axvline(0.90, color=PALETTE["red"], lw=0.8, ls="--", alpha=0.6, label="90% target")
    for var in variants:
        sub = df[df[variant_col] == var].sort_values(alpha_col)
        ax.plot(
            sub[cov_col],
            sub[width_col],
            marker="o",
            ms=5,
            color=var_colors[var],
            label=var_labels[var],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"  α={row[alpha_col]:.2f}",
                (row[cov_col], row[width_col]),
                fontsize=6.5,
                va="center",
            )
    ax.set_xlabel("Empirical Coverage")
    ax.set_ylabel("Mean Interval Width")
    ax.set_title("Coverage–Width Pareto Frontier")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=7.5)

    # Right: eligible loans vs alpha
    ax2 = axes[1]
    if elig_col:
        for var in variants:
            sub = df[df[variant_col] == var].sort_values(alpha_col)
            ax2.bar(
                [str(round(a, 2)) for a in sub[alpha_col]],
                sub[elig_col],
                label=var_labels[var],
                color=var_colors[var],
                alpha=0.85,
                edgecolor="white",
            )
        ax2.set_xlabel("Alpha Level")
        ax2.set_ylabel("Eligible Loans (PD-high < 0.10)")
        ax2.set_title("Eligible Loans vs. Alpha")
        ax2.legend(loc="upper right", fontsize=7.5)
        ax2.tick_params(axis="x", rotation=30)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        "Alpha Sweep: Coverage–Width–Eligibility Trade-off\n(Mondrian vs. Global, $n_{cal}=237{,}584$)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "estrella_fig8_alpha_pareto")


def _estrella_fig9_spo_regret() -> None:
    """Fig 9 — SPO+ decision regret comparison (bar + violin from per-seed data)."""
    status = _load_json(MODELS_DIR / "spo_real_training_status.json")
    results = status.get("results", {})

    # Structure: results.{two_stage, spo_plus, conformal_robust}.{mean_regret, std_regret, per_seed_means}
    display = [
        ("Two-Stage", "two_stage", PALETTE["orange"]),
        ("SPO+", "spo_plus", PALETTE["blue"]),
        ("Conformal\nRobust", "conformal_robust", PALETTE["gray"]),
    ]

    names = [d[0] for d in display]
    means = [results.get(d[1], {}).get("mean_regret", 0) for d in display]
    stds = [results.get(d[1], {}).get("std_regret", 0) for d in display]
    colors = [d[2] for d in display]
    per_seed = [results.get(d[1], {}).get("per_seed_means", []) for d in display]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, HEIGHT_M))

    # Left: bar with error bars
    bars = ax.bar(
        names,
        means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    for _i, (bar, mean, std) in enumerate(zip(bars, means, stds, strict=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.015,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    if means[0] > 0 and means[1] > 0:
        improvement = (1 - means[1] / means[0]) * 100
        ax.annotate(
            f"−{improvement:.1f}%",
            xy=(0.95, (means[0] + means[1]) / 2),
            xytext=(15, 0),
            textcoords="offset points",
            fontsize=9,
            color=PALETTE["blue"],
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": PALETTE["blue"], "lw": 0.8},
        )
    ax.set_ylabel("Mean Decision Regret")
    ax.set_title("Decision Regret\n(mean ± std, 5 seeds)")

    # Right: per-seed violin (if available)
    if any(len(ps) > 1 for ps in per_seed):
        parts = ax2.violinplot(
            [ps for ps in per_seed if ps],
            positions=list(range(1, sum(1 for ps in per_seed if ps) + 1)),
            showmedians=True,
            showextrema=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.75)
        valid_names = [n for n, ps in zip(names, per_seed, strict=False) if ps]
        ax2.set_xticks(range(1, len(valid_names) + 1))
        ax2.set_xticklabels(valid_names, fontsize=8)
        ax2.set_ylabel("Per-Seed Mean Regret")
        ax2.set_title("Per-Seed Distribution\n(5 seeds)")
    else:
        ax2.set_visible(False)

    fig.suptitle(
        "SPO+ vs. Two-Stage vs. Conformal Robust: Decision Regret\n"
        "($n_{train}=1000$, $n_{test}=200$ instances/seed, 5 seeds, $n_{items}=100$)",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "estrella_fig9_spo_regret")


def _estrella_fig10_cqr_comparison() -> None:
    """Fig 10 — CQR vs Mondrian per-grade coverage comparison."""
    status = _load_json(MODELS_DIR / "cqr_comparison_status.json")
    per_group = status.get("per_group_coverage", {})

    methods_to_plot = {
        "mondrian_splitconf": "Mondrian Split-CP",
        "cqr_asymmetric": "CQR (Asymmetric)",
        "global_splitconf": "Global Split-CP",
    }
    grades = sorted(set().union(*[set(v.keys()) for v in per_group.values()]))

    fig, ax = plt.subplots(figsize=(COL2 * 0.75, HEIGHT_M))
    x = np.arange(len(grades))
    bar_w = 0.25
    colors_b = [PALETTE["blue"], PALETTE["orange"], PALETTE["gray"]]
    offset_map = {m: (i - 1) * bar_w for i, m in enumerate(methods_to_plot)}

    for (method_key, label), color in zip(methods_to_plot.items(), colors_b, strict=False):
        vals = [per_group.get(method_key, {}).get(g, 0) for g in grades]
        ax.bar(
            x + offset_map[method_key],
            vals,
            bar_w,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.axhline(0.90, color=PALETTE["red"], lw=1.0, ls="--", label="Target 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("CQR vs. Mondrian CP: Per-Grade Coverage\n(OOT, $\\alpha = 0.10$)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.40, 1.08)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "estrella_fig10_cqr_per_grade")


def _estrella_fig11_crpto_stability() -> None:
    """Fig 11 — CRPTO vs SPO+ stability under distributional shift."""
    detail_path = DATA_DIR / "crpto_vs_spo_stability_detail.parquet"
    if not detail_path.exists():
        logger.warning("Skipping fig11: {} not found", detail_path)
        return
    from scripts.run_crpto_vs_spo_stability import _generate_stability_figure

    detail_df = pd.read_parquet(detail_path)
    _generate_stability_figure(detail_df, OUT_DIR)


def _paper3_fig_nonconformity_scores_by_grade() -> None:
    """Fig — Nonconformity score distribution by grade with quantile thresholds.

    Illustrates WHY Mondrian is necessary: residual distributions differ
    substantially across grades, so a single global quantile leads to systematic
    under-coverage in high-risk grades and over-coverage in low-risk grades.
    """
    df = pd.read_parquet(DATA_DIR / "conformal_intervals_mondrian.parquet")
    required = {"y_true", "y_pred", "grade"}
    if not required.issubset(df.columns):
        logger.warning(
            "nonconformity_scores figure skipped — missing columns in intervals parquet."
        )
        return

    grades = [g for g in ["A", "B", "C", "D", "E", "F", "G"] if g in df["grade"].unique()]
    n_grades = len(grades)
    if n_grades == 0:
        return

    grade_colors = {
        "A": PALETTE["green"],
        "B": PALETTE["sky"],
        "C": PALETTE["blue"],
        "D": PALETTE["orange"],
        "E": PALETTE["yellow"],
        "F": PALETTE["red"],
        "G": PALETTE["purple"],
    }

    fig, axes = plt.subplots(1, n_grades, figsize=(COL2, HEIGHT_M), sharey=False)
    if n_grades == 1:
        axes = [axes]

    global_scores = np.abs(df["y_true"].to_numpy(float) - df["y_pred"].to_numpy(float))
    global_q90 = float(np.quantile(global_scores, 0.90))

    for ax, grade in zip(axes, grades, strict=False):
        sub = df[df["grade"] == grade]
        scores = np.abs(sub["y_true"].to_numpy(float) - sub["y_pred"].to_numpy(float))
        group_q90 = float(np.quantile(scores, 0.90))
        color = grade_colors.get(grade, PALETTE["blue"])

        ax.hist(scores, bins=30, color=color, alpha=0.75, density=True, edgecolor="none")
        ax.axvline(group_q90, color=color, lw=1.4, ls="-", label=f"q̂ᵍ={group_q90:.3f}")
        ax.axvline(global_q90, color=PALETTE["gray"], lw=1.0, ls="--", label=f"q̂={global_q90:.3f}")
        ax.set_title(f"Grade {grade}\n(n={len(sub):,})", fontsize=8)
        ax.set_xlabel("Score", fontsize=7)
        ax.tick_params(labelsize=6)
        if grade == grades[0]:
            ax.set_ylabel("Density", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper right")

    fig.suptitle(
        "Nonconformity Score Distributions by Grade — Motivation for Mondrian CP",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, "p3_fig_nonconformity_scores_by_grade")


def _paper3_fig_calibration_stat_tests() -> None:
    """Fig — Cumulative differences plot (MAPIE calibration hypothesis test).

    Reproduces the MAPIE tutorial figure with our Lending Club OOT data.
    Curve within ±2σ bands supports H₀: calibrated. Used in Paper 3 / Cap 06c.
    """
    path = DATA_DIR / "calibration_cumulative_diffs.parquet"
    if not path.exists():
        logger.warning(
            "calibration_cumulative_diffs.parquet not found — run train_pd_model.py first."
        )
        return

    df = pd.read_parquet(path)
    stat_path = DATA_DIR / "statistical_calibration_tests.json"
    stat_tests: dict = {}
    if stat_path.exists():
        import json as _json

        stat_tests = _json.loads(stat_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(COL2, HEIGHT_M))

    if "cum_diff_uncalibrated" in df.columns:
        ks_raw = stat_tests.get("uncalibrated", {}).get("ks_pvalue", float("nan"))
        ax.plot(
            df["k"],
            df["cum_diff_uncalibrated"],
            color=PALETTE["orange"],
            lw=1.4,
            alpha=0.85,
            label=f"Uncalibrated (KS p={ks_raw:.3f})",
        )
    if "cum_diff_calibrated" in df.columns:
        ks_cal = stat_tests.get("calibrated", {}).get("ks_pvalue", float("nan"))
        ax.plot(
            df["k"],
            df["cum_diff_calibrated"],
            color=PALETTE["blue"],
            lw=1.6,
            label=f"Venn-Abers calibrated (KS p={ks_cal:.3f})",
        )
    if "sigma_upper" in df.columns:
        ax.fill_between(
            df["k"],
            df["sigma_lower"],
            df["sigma_upper"],
            color=PALETTE["gray"],
            alpha=0.12,
            label="±2σ band (H₀: calibrated)",
        )
        ax.plot(df["k"], df["sigma_upper"], color=PALETTE["gray"], lw=0.8, ls="--")
        ax.plot(df["k"], df["sigma_lower"], color=PALETTE["gray"], lw=0.8, ls="--")
    ax.axhline(0, color=PALETTE["gray"], lw=0.7, alpha=0.5)

    ax.set_xlabel("Proportion of Sorted Scores")
    ax.set_ylabel("Cumulative Differences")
    ax.set_title("Calibration Hypothesis Test: Cumulative Differences (OOT, $n=276{,}869$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "p3_fig_calibration_stat_tests")


# ── Dispatch ──────────────────────────────────────────────────────────────────

PAPER3_FIGS = [
    _paper3_fig1_variant_scatter,
    _paper3_fig2_grade_coverage_heatmap,
    _paper3_fig3_temporal_stability,
    _paper3_fig_nonconformity_scores_by_grade,
    _paper3_fig_calibration_stat_tests,
]
PAPER2_FIGS = [
    _paper2_fig4_sicr_grid,
    _paper2_fig5_ecl_alpha_sensitivity,
    _paper2_fig6_bma_vs_cp,
]
ESTRELLA_FIGS = [
    _estrella_fig7_uncertainty_baselines,
    _estrella_fig8_alpha_pareto,
    _estrella_fig9_spo_regret,
    _estrella_fig10_cqr_comparison,
    _estrella_fig11_crpto_stability,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures.")
    parser.add_argument("--paper", choices=["2", "3", "estrella", "all"], default="all")
    args = parser.parse_args()

    figs_to_run: list = []
    if args.paper in ("3", "all"):
        figs_to_run.extend(PAPER3_FIGS)
    if args.paper in ("2", "all"):
        figs_to_run.extend(PAPER2_FIGS)
    if args.paper in ("estrella", "all"):
        figs_to_run.extend(ESTRELLA_FIGS)

    logger.info(f"Generating {len(figs_to_run)} figures → {OUT_DIR}")
    errors = []
    for fn in figs_to_run:
        try:
            fn()
        except Exception as exc:
            logger.error(f"{fn.__name__}: {exc}")
            errors.append((fn.__name__, str(exc)))

    logger.info(f"Done. {len(figs_to_run) - len(errors)} figures saved, {len(errors)} errors.")
    if errors:
        for name, err in errors:
            logger.warning(f"  FAILED: {name} — {err}")

    status = {
        **build_artifact_metadata(schema_version="2026-03-21.1"),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "output_dir": str(OUT_DIR),
        "figures_attempted": len(figs_to_run),
        "figures_succeeded": len(figs_to_run) - len(errors),
        "errors": dict(errors),
        "files": sorted(str(p.name) for p in OUT_DIR.glob("*.pdf")),
    }
    out_json = MODELS_DIR / "paper_figures_status.json"
    out_json.write_text(
        __import__("json").dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Status written → {out_json}")


if __name__ == "__main__":
    main()
