"""Generate book-friendly single-panel figures from canonical project artifacts.

The Quarto book should favor one figure per row and avoid collage-style dashboard
images. This script regenerates the most important visual evidence as standalone
figures directly from processed artifacts, so the book can reference clean,
readable images without depending on composite notebook dashboards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
from loguru import logger

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from book._helpers.plot_helpers import GRADE_ORDER, PALETTE, apply_style, save_figure  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
BOOK_FIG_DIR = REPO_ROOT / "book" / "assets" / "figures" / "editorial"
BOOK_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _method_label(name: str) -> str:
    mapping = {
        "conformal_mondrian": "Mondrian CP",
        "bootstrap": "Bootstrap",
        "parametric_gaussian": "Parametric (Gauss.)",
        "ellipsoidal_grade": "Ellipsoidal",
        "global": "Global Split-CP",
        "mondrian": "Mondrian CP",
    }
    return mapping.get(name, name)


def _save(fig: plt.Figure, stem: str) -> None:
    save_figure(fig, BOOK_FIG_DIR / f"{stem}.png", close=True)


def pipeline_pd_distribution_by_grade() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "conformal_intervals_mondrian.parquet")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for grade in GRADE_ORDER:
        sub = df[df["grade"] == grade]
        if sub.empty:
            continue
        ax.hist(
            sub["y_pred"],
            bins=28,
            density=True,
            alpha=0.28,
            label=f"Grade {grade}",
            color=PALETTE["grades"][grade],
        )
    ax.set_xlabel("PD calibrada")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de PD por grade (test OOT)")
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    _save(fig, "pipeline_pd_distribution_by_grade")


def pipeline_interval_sample() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "conformal_intervals_mondrian.parquet").sort_values("y_pred")
    sample = df.head(120).reset_index(drop=True)
    x = np.arange(len(sample))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.fill_between(
        x,
        sample["pd_low_90"],
        sample["pd_high_90"],
        color=PALETTE["info"],
        alpha=0.22,
        label="Intervalo conformal 90%",
    )
    ax.plot(x, sample["y_pred"], color=PALETTE["primary"], linewidth=1.6, label="PD puntual")
    ax.set_xlabel("Préstamo (ordenado por PD)")
    ax.set_ylabel("PD")
    ax.set_title("Muestra de intervalos conformales sobre PD calibrada")
    ax.legend()
    fig.tight_layout()
    _save(fig, "pipeline_interval_sample")


def pipeline_ecl_by_scenario() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "ifrs9_scenario_summary.parquet").copy()
    order = ["baseline", "mild_stress", "adverse", "severe"]
    df["scenario"] = pd.Categorical(df["scenario"], categories=order, ordered=True)
    df = df.sort_values("scenario")
    labels = ["Baseline", "Mild stress", "Adverse", "Severe"]
    values = df["total_ecl"] / 1e9
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = [PALETTE["success"], PALETTE["info"], PALETTE["warning"], PALETTE["danger"]]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}B",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("ECL total (miles de millones USD)")
    ax.set_title("Sensibilidad del ECL a escenarios IFRS9")
    fig.tight_layout()
    _save(fig, "pipeline_ecl_by_scenario")


def pd_roc_curve() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "roc_curve_data.parquet")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = {
        "logreg": PALETTE["muted"],
        "catboost_default": PALETTE["info"],
        "catboost_tuned": PALETTE["danger"],
    }
    labels = {
        "logreg": "LogReg baseline",
        "catboost_default": "CatBoost default",
        "catboost_tuned": "CatBoost tuned",
    }
    for model, sub in df.groupby("model"):
        ax.plot(
            sub["fpr"],
            sub["tpr"],
            label=labels.get(model, model),
            color=colors.get(model, PALETTE["primary"]),
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC OOT de modelos PD candidatos")
    ax.legend()
    fig.tight_layout()
    _save(fig, "pd_roc_curve")


def pd_pr_curve() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    y_true = df["y_true"].to_numpy()
    models = {
        "LogReg baseline": ("y_prob_lr", PALETTE["muted"]),
        "CatBoost default": ("y_prob_cb_default", PALETTE["info"]),
        "Champion calibrado": ("y_prob_final", PALETTE["danger"]),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for label, (col, color) in models.items():
        precision, recall, _ = precision_recall_curve(y_true, df[col].to_numpy())
        ap = average_precision_score(y_true, df[col].to_numpy())
        ax.plot(recall, precision, color=color, linewidth=1.8, label=f"{label} (AP={ap:.3f})")
    ax.axhline(
        y_true.mean(),
        color=PALETTE["muted"],
        linestyle="--",
        alpha=0.45,
        label=f"Base rate ({y_true.mean():.1%})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall en un problema desbalanceado")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "pd_pr_curve")


def pd_reliability_curve() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "calibration_curve_data.parquet")
    labels = {
        "catboost_uncalibrated": ("CatBoost raw", PALETTE["info"]),
        "isotonic": ("Isotónica", PALETTE["success"]),
        "platt": ("Platt", PALETTE["warning"]),
        "venn_abers": ("Venn-Abers", PALETTE["danger"]),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for model, sub in df.groupby("model"):
        label, color = labels.get(model, (model, PALETTE["primary"]))
        ax.plot(sub["predicted_prob"], sub["observed_freq"], "o-", color=color, label=label)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=PALETTE["muted"],
        alpha=0.45,
        label="Calibración perfecta",
    )
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Diagrama de confiabilidad del champion PD")
    ax.legend()
    fig.tight_layout()
    _save(fig, "pd_reliability_curve")


def conformal_coverage_by_grade() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "conformal_group_metrics_mondrian.parquet").copy()
    df["group"] = pd.Categorical(df["group"], categories=GRADE_ORDER, ordered=True)
    df = df.sort_values("group")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [PALETTE["grades"].get(g, PALETTE["muted"]) for g in df["group"].astype(str)]
    bars = ax.bar(df["group"].astype(str), df["coverage_90"], color=colors)
    ax.axhline(0.90, linestyle="--", color=PALETTE["warning"], alpha=0.7, label="Target 90%")
    for bar, value in zip(bars, df["coverage_90"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.82, 0.98)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Cobertura empírica")
    ax.set_title("Cobertura Mondrian por grade (90%)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "conformal_coverage_by_grade")


def conformal_width_vs_target() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "alpha_sweep_pareto_both.parquet").copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, color in [("global", PALETTE["warning"]), ("mondrian", PALETTE["primary"])]:
        sub = df[df["method"] == method].sort_values("target_coverage")
        ax.plot(
            sub["target_coverage"],
            sub["avg_width"],
            "o-",
            label=_method_label(method),
            color=color,
        )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Cobertura objetivo")
    ax.set_ylabel("Ancho medio del intervalo")
    ax.set_title("Costo en anchura al exigir más cobertura")
    ax.legend()
    fig.tight_layout()
    _save(fig, "conformal_width_vs_target")


def ts_default_rate_series() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "time_series.parquet").sort_values("ds")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(df["ds"], df["default_rate"], color=PALETTE["primary"], linewidth=1.8)
    ax.fill_between(df["ds"], 0, df["default_rate"], color=PALETTE["primary"], alpha=0.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa de default mensual")
    ax.set_title("Serie temporal agregada de default del portafolio")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "ts_default_rate_series")


def causal_policy_monthly_net() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "causal_policy_oot_backtest.parquet").sort_values("month")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(
        df["month"],
        df["total_net_value"],
        color=PALETTE["secondary"],
        linewidth=1.5,
        alpha=0.45,
        label="Valor neto mensual",
    )
    ax.plot(
        df["month"],
        df["total_net_roll3"],
        color=PALETTE["success"],
        linewidth=2.2,
        label="Promedio móvil 3 meses",
    )
    ax.axhline(0, color=PALETTE["muted"], linestyle="--", alpha=0.5)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Valor neto (USD)")
    ax.set_title("Backtest OOT de la política causal")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "causal_policy_monthly_net")


def shap_family_mass() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "explainability_global.parquet")
    agg = (
        df.groupby("feature_family", as_index=False)
        .agg(total_shap=("mean_abs_shap", "sum"), n_features=("feature", "count"))
        .sort_values("total_shap")
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.barh(agg["feature_family"], agg["total_shap"], color=PALETTE["primary"], alpha=0.85)
    total = agg["total_shap"].sum()
    for bar, (_, row) in zip(bars, agg.iterrows(), strict=False):
        pct = 100 * row["total_shap"] / total if total else 0
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{row['n_features']} vars · {pct:.1f}%",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Masa total |SHAP|")
    ax.set_ylabel("Familia de variables")
    ax.set_title("Contribución global por familia de variables")
    fig.tight_layout()
    _save(fig, "shap_family_mass")


def pd_feature_importance_top10() -> None:
    """Generate a compact top-10 feature importance figure for the PD landing page."""
    apply_style()
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover - optional runtime dep
        logger.warning(f"Skipping PD feature importance figure — catboost unavailable: {exc}")
        return

    contract_path = MODELS_DIR / "pd_model_contract.json"
    if not contract_path.exists():
        logger.warning("Skipping PD feature importance figure — model contract not found")
        return

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    feature_names = contract.get("feature_names", [])
    model_path = Path(contract.get("model_path", ""))
    if not feature_names or not model_path.exists():
        logger.warning("Skipping PD feature importance figure — missing model path or features")
        return

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    importances = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange"), dtype=float
    )
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bars = ax.barh(fi_df["feature"], fi_df["importance"], color=PALETTE["info"], alpha=0.9)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("Importancia (PredictionValuesChange)")
    ax.set_title("CatBoost Default — Top 10 Feature Importance")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    _save(fig, "pd_feature_importance_top10")


def ale_single_feature(feature: str, stem: str, title: str) -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "ale_curves.parquet")
    sub = df[df["feature"] == feature].sort_values("midpoint")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.fill_between(sub["midpoint"], 0, sub["ale_value"], color=PALETTE["info"], alpha=0.22)
    ax.plot(sub["midpoint"], sub["ale_value"], color=PALETTE["info"], linewidth=2)
    ax.scatter(sub["midpoint"], sub["ale_value"], color=PALETTE["primary"], s=22)
    ax.axhline(0, color=PALETTE["muted"], linestyle="--", alpha=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel("ALE")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, stem)


def uncertainty_baselines_tradeoff() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "uncertainty_baselines_comparison.parquet").copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    scatter = ax.scatter(
        df["empirical_coverage"],
        df["avg_width"],
        s=np.clip(df["min_group_coverage"] * 600, 80, 600),
        c=df["min_group_coverage"],
        cmap="RdYlGn",
        edgecolors="white",
        linewidths=0.7,
    )
    ax.axvline(0.90, linestyle="--", color=PALETTE["warning"], alpha=0.7, label="Target 90%")
    for _, row in df.iterrows():
        ax.annotate(
            _method_label(row["method"]),
            (row["empirical_coverage"], row["avg_width"]),
            textcoords="offset points",
            xytext=(6, 2),
            fontsize=8,
        )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    ax.set_xlabel("Cobertura empírica")
    ax.set_ylabel("Ancho medio")
    ax.set_title("Trade-off entre cobertura, anchura y cobertura mínima por grupo")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Cobertura mínima por grupo", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend()
    fig.tight_layout()
    _save(fig, "uncertainty_baselines_tradeoff")


def alpha_pareto_frontier() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "alpha_sweep_pareto_both.parquet").copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for method, color in [("global", PALETTE["warning"]), ("mondrian", PALETTE["primary"])]:
        sub = df[df["method"] == method].sort_values("empirical_coverage")
        ax.plot(
            sub["empirical_coverage"],
            sub["avg_width"],
            "o-",
            color=color,
            label=_method_label(method),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"α={row['alpha']:.2f}",
                (row["empirical_coverage"], row["avg_width"]),
                textcoords="offset points",
                xytext=(3, -10),
                fontsize=7,
            )
    ax.axvline(0.90, linestyle="--", color=PALETTE["muted"], alpha=0.5)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Cobertura empírica")
    ax.set_ylabel("Ancho medio del intervalo")
    ax.set_title("Frontera alpha-cobertura-anchura")
    ax.legend()
    fig.tight_layout()
    _save(fig, "alpha_pareto_frontier")


def alpha_eligible_loans() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "alpha_sweep_pareto_both.parquet")
    sub = df[df["method"] == "mondrian"].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(
        sub["alpha"].map(lambda x: f"{x:.2f}"), sub["n_eligible"], color=PALETTE["primary"]
    )
    for bar, val in zip(bars, sub["n_eligible"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(sub["n_eligible"].max() * 0.01, 1),
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Préstamos elegibles (PD_high < 0.10)")
    ax.set_title("Elegibilidad operativa al variar alpha")
    fig.tight_layout()
    _save(fig, "alpha_eligible_loans")


def fairness_threshold_frontier() -> None:
    apply_style()
    df = pd.read_parquet(DATA_DIR / "fairness_threshold_frontier.parquet")
    agg = (
        df.groupby(["threshold", "attribute_type"], as_index=False)
        .agg(max_dpd=("dpd", "max"), max_eo_gap=("eo_gap", "max"))
        .sort_values(["attribute_type", "threshold"])
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for attr_type, color in [("base", PALETTE["primary"]), ("intersectional", PALETTE["danger"])]:
        sub = agg[agg["attribute_type"] == attr_type]
        if sub.empty:
            continue
        label = "Base" if attr_type == "base" else "Interseccional"
        ax.plot(sub["threshold"], sub["max_dpd"], "o-", color=color, label=f"DPD máx. {label}")
        ax.plot(
            sub["threshold"],
            sub["max_eo_gap"],
            "s--",
            color=color,
            alpha=0.8,
            label=f"EO gap máx. {label}",
        )
    ax.axvline(0.35, linestyle="--", color=PALETTE["muted"], alpha=0.6, label="Threshold operativo")
    ax.set_xlabel("Threshold de decisión")
    ax.set_ylabel("Gap máximo observado")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Robustez de fairness al mover el threshold operativo")
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    _save(fig, "fairness_threshold_frontier")


def calibration_cumulative_diffs() -> None:
    """Cumulative differences plot: calibrated vs uncalibrated vs ±2σ band.

    Reproduces the MAPIE tutorial figure with our Lending Club data.
    Under H0 (perfect calibration) the curve stays within ±2σ (Brownian motion bands).
    """
    path = DATA_DIR / "calibration_cumulative_diffs.parquet"
    if not path.exists():
        logger.warning(
            "calibration_cumulative_diffs.parquet not found — skipping. Run train_pd_model.py first."
        )
        return
    apply_style()
    df = pd.read_parquet(path)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(
        df["k"],
        df["cum_diff_uncalibrated"],
        color=PALETTE["warning"],
        linewidth=1.6,
        alpha=0.85,
        label="CatBoost no calibrado",
    )
    ax.plot(
        df["k"],
        df["cum_diff_calibrated"],
        color=PALETTE["primary"],
        linewidth=1.8,
        label="Venn-Abers calibrado",
    )
    if "sigma_upper" in df.columns:
        ax.fill_between(
            df["k"],
            df["sigma_lower"],
            df["sigma_upper"],
            color=PALETTE["muted"],
            alpha=0.15,
            label="Banda ±2σ (H₀ calibrado)",
        )
        ax.plot(
            df["k"],
            df["sigma_upper"],
            color=PALETTE["muted"],
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(
            df["k"],
            df["sigma_lower"],
            color=PALETTE["muted"],
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Proporción de scores (ordenados)")
    ax.set_ylabel("Diferencias acumuladas")
    ax.set_title("Test de calibración: diferencias acumuladas vs bandas Brownian (OOT)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "calibration_cumulative_diffs")


def conformal_hsic_ssc_diagnostic() -> None:
    """Bar chart showing MAPIE diagnostic metrics: HSIC, SSC, MWI, CWC.

    HSIC ≈ 0 means interval width is independent of coverage (no systematic bias).
    SSC measures equity of coverage across interval width strata.
    """
    path = DATA_DIR / "conformal_diagnostic_metrics.json"
    if not path.exists():
        logger.warning(
            "conformal_diagnostic_metrics.json not found — skipping. Run backtest_conformal_coverage.py first."
        )
        return
    import json as _json

    diag = _json.loads(path.read_text(encoding="utf-8"))
    if not diag.get("mapie_diagnostics_available"):
        logger.warning("MAPIE diagnostics not available in the artifact — skipping.")
        return

    apply_style()
    metrics = {
        "HSIC\n(↓ mejor)": diag.get("hsic_90"),
        "SSC\n(↑ mejor)": diag.get("ssc_90"),
        "MWI\n(↓ mejor)": diag.get("mwi_90"),
        "CWC\n(↓ mejor)": diag.get("cwc_90"),
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}
    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = [
        PALETTE["success"] if "↓" in k and v < 0.5 else PALETTE["primary"]
        for k, v in metrics.items()
    ]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.55)
    for bar, val in zip(bars, metrics.values(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(abs(val) * 0.02, 0.001),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Métricas de diagnóstico MAPIE (Mondrian CP @ 90%, test OOT)")
    ax.set_ylabel("Valor de la métrica")
    n = diag.get("n", "?")
    ax.set_xlabel(f"n = {n:,}" if isinstance(n, int) else f"n = {n}")
    fig.tight_layout()
    _save(fig, "conformal_hsic_ssc_diagnostic")


def generate_all() -> None:
    pipeline_pd_distribution_by_grade()
    pipeline_interval_sample()
    pipeline_ecl_by_scenario()
    pd_feature_importance_top10()
    pd_roc_curve()
    pd_pr_curve()
    pd_reliability_curve()
    conformal_coverage_by_grade()
    conformal_width_vs_target()
    calibration_cumulative_diffs()
    conformal_hsic_ssc_diagnostic()
    ts_default_rate_series()
    causal_policy_monthly_net()
    shap_family_mass()
    ale_single_feature("int_rate", "ale_int_rate", "ALE de tasa de interés")
    ale_single_feature("fico_score", "ale_fico_score", "ALE de puntaje FICO")
    uncertainty_baselines_tradeoff()
    alpha_pareto_frontier()
    alpha_eligible_loans()
    fairness_threshold_frontier()


if __name__ == "__main__":
    generate_all()
