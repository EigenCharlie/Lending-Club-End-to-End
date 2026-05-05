"""Build journal-ready derived evidence for Paper Estrella.

The script uses frozen Paper Estrella artifacts only. It does not reopen the
champion search or replace the official economic champion. The outputs are
diagnostic tables and figures for the Quarto book, manuscript planning, and
appendix material.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "paper_material" / "paper1" / "tables"
FIG_DIR = ROOT / "reports" / "paper_material" / "figures_publication"
DOCS_DIR = ROOT / "docs" / "research"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

FUNDED_LOANS = TABLE_DIR / "paper1_tableA7_funded_set_loans.csv"
FUNDED_COMPOSITION = TABLE_DIR / "paper1_tableA8_funded_set_composition.csv"
PROMOTION_PATH = MODELS_DIR / "final_project_promotion.json"
STATUS_PATH = MODELS_DIR / "paper1_journal_package_status.json"
SHORTLIST_PATH = (
    DATA_DIR
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
    / "portfolio_bound_aware_shortlist.parquet"
)
BOUND_EVAL_PATH = (
    DATA_DIR
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
    / "portfolio_bound_aware_bound_eval.parquet"
)
ALPHA_SWEEP_PATH = DATA_DIR / "alpha_sweep_pareto_mondrian.parquet"

DEFAULT_LGD = 0.45
LGD_GRID = [0.35, 0.45, 0.60]
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260504


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_table(name: str, frame: pd.DataFrame) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{name}.csv"
    tex_path = TABLE_DIR / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    frame.to_latex(
        tex_path,
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.6f}",
    )
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {tex_path.relative_to(ROOT)}")
    return [csv_path, tex_path]


def _save_figure(name: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {png_path.relative_to(ROOT)}")
    print(f"Wrote {pdf_path.relative_to(ROOT)}")
    return [png_path, pdf_path]


def _relative(paths: list[Path]) -> list[str]:
    return list(dict.fromkeys(str(path.relative_to(ROOT)) for path in paths))


def _as_rate(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if float(values.max()) > 1.5:
        return values / 100.0
    return values


def _funded_frame() -> pd.DataFrame:
    funded = pd.read_csv(FUNDED_LOANS)
    funded["int_rate_decimal"] = _as_rate(funded["int_rate"])
    funded["y_true"] = pd.to_numeric(funded["y_true"], errors="coerce").fillna(0.0)
    funded["funded_exposure"] = pd.to_numeric(
        funded["funded_exposure"],
        errors="coerce",
    ).fillna(0.0)
    funded["portfolio_weight"] = funded["funded_exposure"] / max(
        float(funded["funded_exposure"].sum()),
        1e-9,
    )
    funded["miscovered_alpha01"] = funded["miscovered_alpha01"].astype(bool)
    return funded


def _return_from_frame(frame: pd.DataFrame, weights: pd.Series, *, lgd: float) -> float:
    total_exposure = 1_000_000.0
    exposure = weights.astype(float) * total_exposure
    nondefault_return = exposure * frame["int_rate_decimal"] * (1.0 - frame["y_true"])
    default_loss = exposure * (-float(lgd)) * frame["y_true"]
    return float((nondefault_return + default_loss).sum())


def _weighted_metric(frame: pd.DataFrame, weights: pd.Series, column: str) -> float:
    return float((weights.astype(float) * pd.to_numeric(frame[column], errors="coerce")).sum())


def _weighted_cvar(loss_rate: pd.Series, weights: pd.Series, tail: float) -> float:
    order = np.argsort(-loss_rate.to_numpy(dtype=float))
    sorted_loss = loss_rate.to_numpy(dtype=float)[order]
    sorted_weights = weights.to_numpy(dtype=float)[order]
    target = 1.0 - float(tail)
    used = 0.0
    total = 0.0
    for loss, weight in zip(sorted_loss, sorted_weights, strict=False):
        if used >= target:
            break
        take = min(float(weight), target - used)
        total += float(loss) * take
        used += take
    return total / max(used, 1e-12)


def _build_tail_risk_table(funded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weights = funded["portfolio_weight"]
    for lgd in LGD_GRID:
        loss_rate = (
            funded["y_true"] * float(lgd) - (1.0 - funded["y_true"]) * funded["int_rate_decimal"]
        )
        theta = 5.0
        oce = float(np.log(np.sum(weights * np.exp(theta * loss_rate))) / theta)
        rows.append(
            {
                "lgd": float(lgd),
                "mean_loss_rate": float(np.sum(weights * loss_rate)),
                "entropic_oce_theta5": oce,
                "cvar_90_loss_rate": _weighted_cvar(loss_rate, weights, 0.90),
                "cvar_95_loss_rate": _weighted_cvar(loss_rate, weights, 0.95),
                "cvar_99_loss_rate": _weighted_cvar(loss_rate, weights, 0.99),
                "funded_set_repriced_return": _return_from_frame(
                    funded,
                    weights,
                    lgd=float(lgd),
                ),
                "weighted_default_rate": _weighted_metric(funded, weights, "y_true"),
            }
        )
    return pd.DataFrame(rows)


def _build_satisficing_table(promotion: dict[str, Any]) -> pd.DataFrame:
    champion = promotion["final_champion"]
    theorem = promotion["theorem_tight_comparator"]
    robust_region = promotion["robust_region_summary"]
    rows = [
        {
            "criterion": "return_beats_theorem_tight",
            "observed": float(champion["realized_total_return"]),
            "threshold": float(theorem["realized_total_return"]),
            "margin": float(champion["realized_total_return"] - theorem["realized_total_return"]),
            "pass": bool(champion["realized_total_return"] >= theorem["realized_total_return"]),
            "interpretation": "Economic champion preserves more return than the tightness comparator.",
        },
        {
            "criterion": "V_below_sqrt_alpha01",
            "observed": float(champion["alpha01_weighted_miscoverage_V"]),
            "threshold": 0.10,
            "margin": 0.10 - float(champion["alpha01_weighted_miscoverage_V"]),
            "pass": bool(float(champion["alpha01_weighted_miscoverage_V"]) <= 0.10),
            "interpretation": "Weighted funded-set noncoverage stays below Markov sqrt(alpha).",
        },
        {
            "criterion": "gamma_cp_below_020",
            "observed": float(champion["alpha01_gamma_cp"]),
            "threshold": 0.20,
            "margin": 0.20 - float(champion["alpha01_gamma_cp"]),
            "pass": bool(float(champion["alpha01_gamma_cp"]) <= 0.20),
            "interpretation": "Conformal premium remains below an editorial 20pp diagnostic ceiling.",
        },
        {
            "criterion": "violation_zero",
            "observed": float(champion["alpha01_violation"]),
            "threshold": 0.0,
            "margin": -float(champion["alpha01_violation"]),
            "pass": bool(float(champion["alpha01_violation"]) <= 1e-12),
            "interpretation": "Exact funded-set risk violation is zero.",
        },
        {
            "criterion": "robust_region_all_pass",
            "observed": float(robust_region["alpha01_pass_rate"]),
            "threshold": 1.0,
            "margin": float(robust_region["alpha01_pass_rate"]) - 1.0,
            "pass": bool(float(robust_region["alpha01_pass_rate"]) >= 1.0),
            "interpretation": "The final 276K robust region is not a single-point result.",
        },
    ]
    return pd.DataFrame(rows)


def _build_dependency_table(funded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, label in [
        (["period"], "period"),
        (["original_grade"], "grade"),
        (["period", "original_grade"], "period_grade"),
    ]:
        grouped = funded.groupby(keys, dropna=False, observed=True)
        for values, group in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            weight = float(group["portfolio_weight"].sum())
            rows.append(
                {
                    "cluster_type": label,
                    "cluster": "/".join(str(value) for value in values),
                    "n_funded": int(len(group)),
                    "exposure_share": weight,
                    "weighted_default_rate": _weighted_metric(
                        group,
                        group["portfolio_weight"] / max(weight, 1e-12),
                        "y_true",
                    ),
                    "V_contribution": float(
                        group.loc[group["miscovered_alpha01"], "portfolio_weight"].sum()
                    ),
                    "gamma_cp_contribution": float(
                        (
                            group["portfolio_weight"]
                            * (
                                pd.to_numeric(group["pd_high_alpha01"], errors="coerce")
                                - pd.to_numeric(group["pd_point"], errors="coerce")
                            ).clip(lower=0.0)
                        ).sum()
                    ),
                    "max_loan_weight": float(group["portfolio_weight"].max()),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["cluster_type", "exposure_share"], ascending=[True, False])


def _scenario_metrics(frame: pd.DataFrame, weights: pd.Series, scenario: str) -> dict[str, Any]:
    weights = weights.astype(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    gamma_cp = float(
        (
            weights
            * (
                pd.to_numeric(frame["pd_high_alpha01"], errors="coerce")
                - pd.to_numeric(frame["pd_point"], errors="coerce")
            ).clip(lower=0.0)
        ).sum()
    )
    return {
        "scenario": scenario,
        "n_funded_effective": int((weights > 1e-10).sum()),
        "funded_set_repriced_return_lgd45": _return_from_frame(
            frame,
            weights,
            lgd=DEFAULT_LGD,
        ),
        "weighted_default_rate": _weighted_metric(frame, weights, "y_true"),
        "weighted_miscoverage_V": float(
            weights.loc[frame["miscovered_alpha01"].astype(bool)].sum()
        ),
        "gamma_cp": gamma_cp,
        "max_loan_weight": float(weights.max()),
    }


def _build_period_stress_table(funded: pd.DataFrame) -> pd.DataFrame:
    rows = [_scenario_metrics(funded, funded["portfolio_weight"], "baseline")]
    for period in sorted(funded["period"].astype(str).unique()):
        leave_weights = funded["portfolio_weight"].where(
            funded["period"].astype(str) != period, 0.0
        )
        rows.append(_scenario_metrics(funded, leave_weights, f"leave_out_{period}"))
        over_weights = funded["portfolio_weight"].where(
            funded["period"].astype(str) != period,
            funded["portfolio_weight"] * 2.0,
        )
        rows.append(_scenario_metrics(funded, over_weights, f"overweight_2x_{period}"))
    return pd.DataFrame(rows)


def _build_bootstrap_table(funded: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(funded)
    values: list[dict[str, float]] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample_index = rng.integers(0, n, size=n)
        sample = funded.iloc[sample_index].reset_index(drop=True)
        weights = sample["funded_exposure"] / max(float(sample["funded_exposure"].sum()), 1e-12)
        values.append(
            {
                "funded_set_repriced_return_lgd45": _return_from_frame(
                    sample,
                    weights,
                    lgd=DEFAULT_LGD,
                ),
                "weighted_default_rate": _weighted_metric(sample, weights, "y_true"),
                "weighted_miscoverage_V": float(
                    weights.loc[sample["miscovered_alpha01"].astype(bool)].sum()
                ),
                "n_default_loans": float(sample["y_true"].sum()),
                "n_miscovered_loans": float(sample["miscovered_alpha01"].sum()),
            }
        )
    draws = pd.DataFrame(values)
    rows = []
    for column in draws.columns:
        rows.append(
            {
                "metric": column,
                "mean": float(draws[column].mean()),
                "p025": float(draws[column].quantile(0.025)),
                "p50": float(draws[column].quantile(0.50)),
                "p975": float(draws[column].quantile(0.975)),
                "n_draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def _build_budget_cap_lgd_table(
    funded: pd.DataFrame,
    funded_composition: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weights = funded["portfolio_weight"]
    base_return = _return_from_frame(funded, weights, lgd=DEFAULT_LGD)
    for budget in [500_000.0, 1_000_000.0, 2_000_000.0]:
        rows.append(
            {
                "sensitivity_type": "budget_scaling_diagnostic",
                "scenario": f"budget_{budget:.0f}",
                "value": budget,
                "funded_set_repriced_return_lgd45": base_return * budget / 1_000_000.0,
                "weighted_default_rate": _weighted_metric(funded, weights, "y_true"),
                "weighted_miscoverage_V": float(
                    weights.loc[funded["miscovered_alpha01"].astype(bool)].sum()
                ),
                "diagnostic_pass": True,
                "note": "Linear funded-set rescale; not a re-optimized portfolio.",
            }
        )
    for lgd in LGD_GRID:
        rows.append(
            {
                "sensitivity_type": "lgd_sensitivity",
                "scenario": f"lgd_{lgd:.2f}",
                "value": float(lgd),
                "funded_set_repriced_return_lgd45": _return_from_frame(
                    funded,
                    weights,
                    lgd=float(lgd),
                ),
                "weighted_default_rate": _weighted_metric(funded, weights, "y_true"),
                "weighted_miscoverage_V": float(
                    weights.loc[funded["miscovered_alpha01"].astype(bool)].sum()
                ),
                "diagnostic_pass": True,
                "note": "Reprices realized funded set under alternative LGD.",
            }
        )
    largest_segment = float(funded_composition["exposure_share"].max())
    for cap in [0.25, 0.20, 0.15]:
        rows.append(
            {
                "sensitivity_type": "segment_cap_diagnostic",
                "scenario": f"segment_cap_{cap:.2f}",
                "value": float(cap),
                "funded_set_repriced_return_lgd45": base_return,
                "weighted_default_rate": _weighted_metric(funded, weights, "y_true"),
                "weighted_miscoverage_V": float(
                    weights.loc[funded["miscovered_alpha01"].astype(bool)].sum()
                ),
                "diagnostic_pass": bool(largest_segment <= cap),
                "note": "Checks period-grade funded-set concentration; not a re-optimized portfolio.",
            }
        )
    return pd.DataFrame(rows)


def _build_robust_region_table(shortlist: pd.DataFrame) -> pd.DataFrame:
    grouped = shortlist.groupby(["risk_tolerance", "gamma"], as_index=False)
    rows = grouped.agg(
        n_policies=("semantic_policy_key", "count"),
        alpha01_pass_rate=("alpha01_exact_pass", "mean"),
        best_return=("realized_total_return", "max"),
        median_return=("realized_total_return", "median"),
        min_V=("alpha01_weighted_miscoverage_V", "min"),
        median_gamma_cp=("alpha01_gamma_cp", "median"),
        max_n_funded=("n_funded", "max"),
    )
    rows["all_alpha01_pass"] = rows["alpha01_pass_rate"].eq(1.0)
    return rows.sort_values(["risk_tolerance", "gamma"]).reset_index(drop=True)


def _plot_conceptual_pipeline() -> list[Path]:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    boxes = [
        ("PD calibrada\nCatBoost + Venn-Abers", 0.04, "#DCEBFF"),
        ("Intervalo conformal\n[PD_low, PD_high]", 0.22, "#E6F4EA"),
        ("Uncertainty set\nu_i(alpha)", 0.40, "#FFF3CD"),
        ("LP robusto\nrisk-return + tau", 0.58, "#FCE4EC"),
        ("Funded set\n335 loans / $1M", 0.76, "#EDE7F6"),
        ("Bound eval\nV, Gamma_CP, violation", 0.91, "#E0F7FA"),
    ]
    for text, x, color in boxes:
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.05",
                "facecolor": color,
                "edgecolor": "#2F3A4A",
                "linewidth": 1.2,
            },
        )
    for start, end in zip(boxes[:-1], boxes[1:], strict=False):
        ax.annotate(
            "",
            xy=(end[1] - 0.075, 0.55),
            xytext=(start[1] + 0.075, 0.55),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#2F3A4A"},
        )
    ax.text(
        0.5,
        0.12,
        "CRPTO: uncertainty is not a diagnostic afterthought; it becomes a portfolio constraint.",
        ha="center",
        fontsize=11,
        color="#263238",
    )
    return _save_figure("estrella_fig12_crpto_conceptual_pipeline")


def _plot_alpha_gamma_funded_set(bound_eval: pd.DataFrame, promotion: dict[str, Any]) -> list[Path]:
    champion = promotion["final_champion"]
    mask = (
        bound_eval["risk_tolerance"].eq(float(champion["risk_tolerance"]))
        & bound_eval["gamma"].eq(float(champion["gamma"]))
        & bound_eval["uncertainty_aversion"].eq(float(champion["uncertainty_aversion"]))
        & bound_eval["policy_mode"].eq(str(champion["policy_mode"]))
    )
    data = bound_eval.loc[mask].sort_values("alpha")
    if data.empty:
        data = bound_eval.sort_values(["return_first_rank", "alpha"]).groupby("alpha").head(1)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.plot(data["alpha"], data["gamma_cp"], marker="o", label="Gamma_CP", color="#0B5CAD")
    ax1.plot(
        data["alpha"],
        data["weighted_miscoverage_V"],
        marker="s",
        label="V",
        color="#B00020",
    )
    ax1.plot(
        data["alpha"],
        data["sqrt_alpha"],
        linestyle="--",
        label="sqrt(alpha)",
        color="#616161",
    )
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Weighted bound quantities")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(data["alpha"], data["n_funded"], marker="^", color="#2E7D32", label="n_funded")
    ax2.set_ylabel("Funded loans")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    ax1.set_title("Alpha to Gamma_CP to funded-set behavior")
    return _save_figure("estrella_fig13_alpha_gamma_funded_set")


def _plot_robust_region_heatmap(shortlist: pd.DataFrame) -> list[Path]:
    pivot = shortlist.pivot_table(
        index="risk_tolerance",
        columns="gamma",
        values="realized_total_return",
        aggfunc="max",
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:.3f}" for x in pivot.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("risk_tolerance")
    ax.set_title("Robust region: best realized return by policy family")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value / 1000:.0f}K", ha="center", va="center", color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Realized return")
    return _save_figure("estrella_fig14_robust_region_heatmap")


def _write_markdown_dossier(status: dict[str, Any]) -> Path:
    path = DOCS_DIR / "paper_estrella_journal_package_2026-05-04.md"
    lines = [
        "# Paper Estrella Journal Package - 2026-05-04",
        "",
        "This dossier records the journal-oriented tables and figures generated from",
        "frozen Paper Estrella artifacts. It does not reopen the champion search.",
        "",
        "## Generated artifacts",
        "",
    ]
    for artifact in status["generated_artifacts"]:
        lines.append(f"- `{artifact}`")
    lines += [
        "",
        "## Scope notes",
        "",
        "- A12--A18 are diagnostic robustness and packaging tables.",
        "- Budget and segment-cap sensitivity are funded-set diagnostics, not",
        "  re-optimized portfolios.",
        "- Tail-risk and bootstrap return columns are funded-set repricing diagnostics;",
        "  the official champion return remains sourced from `final_project_promotion.json`.",
        "- The official champion remains `bound_aware_276k_economic_champion`.",
        "",
        "## Appendix map",
        "",
        "| Artifact | Purpose | Caveat |",
        "|---|---|---|",
        "| A12 tail risk OCE/CVaR | Adds funded-set tail-risk diagnostics under LGD alternatives. | Diagnostic repricing, not a new champion metric. |",
        "| A13 satisficing margins | Converts champion evidence into OR-style pass/margin checks. | Editorial thresholds must be justified if moved to body. |",
        "| A14 dependency clusters | Documents period/grade concentration for the tightening appendix. | Does not prove conditional independence. |",
        "| A15 leave-one-period stress | Reweights the funded set by period. | Not a re-optimized portfolio. |",
        "| A16 bootstrap funded-set metrics | Adds empirical intervals for realized funded-set quantities. | Not a conformal guarantee. |",
        "| A17 budget/LGD/cap sensitivity | Checks practical sensitivity to budget, LGD and segment caps. | Cap checks are diagnostics, not solver constraints. |",
        "| A18 robust region by family | Summarizes the `45/45` alpha01-safe region by `risk_tolerance x gamma`. | Bound-aware family only. |",
        "",
        "## Quarto integration",
        "",
        "- `book/chapters/14-paper-estrella/14g-manuscript-blueprint.qmd` uses",
        "  these artifacts to define the paper outline and final table/figure plan.",
        "- `book/chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd`",
        "  renders A12--A18 and Figures 12--14.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


def build_journal_package() -> dict[str, Any]:
    promotion = _load_json(PROMOTION_PATH)
    funded = _funded_frame()
    funded_composition = pd.read_csv(FUNDED_COMPOSITION)
    shortlist = pd.read_parquet(SHORTLIST_PATH)
    bound_eval = pd.read_parquet(BOUND_EVAL_PATH)

    artifacts: list[Path] = []
    artifacts += _write_table("paper1_tableA12_tail_risk_oce_cvar", _build_tail_risk_table(funded))
    artifacts += _write_table(
        "paper1_tableA13_satisficing_margins",
        _build_satisficing_table(promotion),
    )
    artifacts += _write_table(
        "paper1_tableA14_dependency_cluster_diagnostics",
        _build_dependency_table(funded),
    )
    artifacts += _write_table(
        "paper1_tableA15_leave_one_period_stress",
        _build_period_stress_table(funded),
    )
    artifacts += _write_table(
        "paper1_tableA16_bootstrap_funded_set_metrics",
        _build_bootstrap_table(funded),
    )
    artifacts += _write_table(
        "paper1_tableA17_budget_cap_lgd_sensitivity",
        _build_budget_cap_lgd_table(funded, funded_composition),
    )
    artifacts += _write_table(
        "paper1_tableA18_robust_region_policy_family",
        _build_robust_region_table(shortlist),
    )
    artifacts += _plot_conceptual_pipeline()
    artifacts += _plot_alpha_gamma_funded_set(bound_eval, promotion)
    artifacts += _plot_robust_region_heatmap(shortlist)

    status = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": 1,
        "run_tag": promotion["run_tag"],
        "champion_label": promotion["final_champion"]["label"],
        "generated_artifacts": _relative(artifacts),
        "source_artifacts": _relative(
            [
                FUNDED_LOANS,
                FUNDED_COMPOSITION,
                PROMOTION_PATH,
                SHORTLIST_PATH,
                BOUND_EVAL_PATH,
                ALPHA_SWEEP_PATH,
            ]
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "notes": [
            "No champion search was reopened.",
            "Budget and cap sensitivity are diagnostics on the exported funded set.",
            "Repriced funded-set return diagnostics are not replacements for the official champion return.",
            "Official champion metrics remain sourced from final_project_promotion.json.",
        ],
    }
    dossier = _write_markdown_dossier(status)
    artifacts.append(dossier)
    status["generated_artifacts"] = _relative(artifacts)
    _write_json(STATUS_PATH, status)
    print(f"Wrote {STATUS_PATH.relative_to(ROOT)}")
    return status


def main() -> int:
    build_journal_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
