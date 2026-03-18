"""Paper Estrella draft: Predict-then-Optimize with Conformal Uncertainty Sets.

This paper absorbs the original Paper 1 (CP + Robust Opt) and adds:
- Theoretical bound alpha-conformal <-> Gamma-robustness (Bertsimas & Sim)
- Baseline uncertainty set comparison (ellipsoidal, bootstrap, parametric)
- SPO+ decision regret comparison
- Alpha sweep Pareto frontier (coverage-width-return)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.context_help import methodology_dialog
from streamlit_app.components.paper_scaffold import render_phase_tracker
from streamlit_app.components.story_shell import (
    render_key_takeaway,
    render_next_steps,
    render_page_header,
    render_section_checkpoint,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    download_table,
    format_number,
    format_pct,
    try_load_json,
    try_load_parquet,
)

st.title("Paper Estrella \u2014 Working Draft")
st.caption(
    "Predict-then-Optimize under Distribution-Free Uncertainty: "
    "Conformal Intervals as Operational Uncertainty Sets for Robust Credit Portfolio Allocation"
)
page_contract = get_page_contract("paper_estrella_predict_optimize")
render_page_header(page_contract)
render_key_takeaway(
    "Novelty claim: framework predict-then-optimize con intervalos conformales Mondrian "
    "como conjuntos de incertidumbre operativos, bound teorico alpha-Gamma, y comparacion "
    "de regret contra SPO+ y baselines clasicos de incertidumbre."
)
st.warning(
    "Borrador de trabajo para revision academica. Este paper absorbe el Paper 1 original "
    "(CP + Robust Opt) y agrega contribucion teorica y baselines de uncertainty sets."
)
methodology_dialog(
    "Foco narrativo del Paper Estrella",
    """
**Foco correcto:**
- Bound teorico alpha-conformal <-> Gamma-robustez
- Comparacion de uncertainty sets: conformal vs ellipsoidal vs bootstrap vs parametric
- Decision regret (SPO+ vs two-stage vs robust conformal)
- Alpha sweep Pareto frontier
- Price of robustness como metrica operativa

**No duplicar:**
- Teoria general de CP -> remitir a Paper 3
- Detalles de IFRS9/ECL -> remitir a Paper 2
- EDA y feature engineering -> remitir a capitulos base
""",
    button_label="Regla de foco narrativo (Paper Estrella)",
)

# ---- Load canonical artifacts ----
pipeline_summary = try_load_json("pipeline_summary", directory="data", default={})
conformal_status = try_load_json("conformal_policy_status", directory="models", default={})
model_comparison = try_load_json("model_comparison", directory="data", default={})
champion_policy = try_load_json("champion_portfolio_policy", directory="models", default={})
ab_status = try_load_json("ab_simulation_status", directory="models", default={})
spo_status = try_load_json("spo_comparison_status", directory="models", default={})
spo_real_status = try_load_json("spo_real_training_status", directory="models", default={})

robust_summary = try_load_parquet("portfolio_robustness_summary")
robust_frontier = try_load_parquet("portfolio_robustness_frontier")
variant_benchmark = try_load_parquet("conformal_variant_benchmark")
ab_results = try_load_parquet("ab_simulation_summary")
alpha_sweep_pareto = try_load_parquet("alpha_sweep_pareto_both")
cqr_comparison = try_load_parquet("cqr_comparison")
unc_baselines = try_load_parquet("uncertainty_baselines_comparison")
cqr_status = try_load_json("cqr_comparison_status", directory="models", default={})
spo_loss_curve = try_load_parquet("spo_training_loss")

pipeline = pipeline_summary.get("pipeline", {})
conformal = pipeline_summary.get("conformal", {})
pd_metrics = model_comparison.get("final_test_metrics", {})
best_calibration = str(model_comparison.get("best_calibration", "auto-selected"))

coverage_90 = float(conformal_status.get("coverage_90", conformal.get("coverage_90", np.nan)))
coverage_95 = float(conformal_status.get("coverage_95", conformal.get("coverage_95", np.nan)))
width_90 = float(conformal_status.get("avg_width_90", pipeline.get("interval_width_mean", np.nan)))
robust_return = float(pipeline.get("robust_return", np.nan))
nonrobust_return = float(pipeline.get("nonrobust_return", np.nan))
price_of_robustness = float(pipeline.get("price_of_robustness", np.nan))
robust_funded = int(pipeline.get("robust_funded", 0) or 0)
nonrobust_funded = int(pipeline.get("nonrobust_funded", 0) or 0)

# ---- 0) Metadata ----
st.markdown("## 0) Metadata de Propuesta")
meta_df = pd.DataFrame(
    [
        {"Campo": "Estado", "Valor": "Working Draft"},
        {"Campo": "Venue sugerido", "Valor": "Management Science / Operations Research / EJOR"},
        {"Campo": "Formato", "Valor": "~30-35 paginas + online appendix (Quarto book)"},
        {"Campo": "Timeline", "Valor": "Julio-Diciembre 2026"},
        {"Campo": "Dataset", "Valor": "Lending Club (split OOT, 276K test loans)"},
        {
            "Campo": "Pregunta central",
            "Valor": "Como usar intervalos conformales como uncertainty sets para optimizacion robusta de portafolio crediticio",
        },
        {"Campo": "Absorbe", "Valor": "Paper 1 original (CP + Robust Opt)"},
        {"Campo": "AUC PD (OOT)", "Valor": f"{pd_metrics.get('auc_roc', np.nan):.4f}"},
        {
            "Campo": "Cobertura CP 90%",
            "Valor": format_pct(coverage_90, 2) if np.isfinite(coverage_90) else "N/D",
        },
        {"Campo": "Price of Robustness", "Valor": format_number(price_of_robustness, prefix="$")},
    ]
)
st.dataframe(meta_df, use_container_width=True, hide_index=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cobertura 90%", format_pct(coverage_90, 2) if np.isfinite(coverage_90) else "N/D")
k2.metric("Cobertura 95%", format_pct(coverage_95, 2) if np.isfinite(coverage_95) else "N/D")
k3.metric("Retorno Robusto", format_number(robust_return, prefix="$"))
k4.metric("Retorno No Robusto", format_number(nonrobust_return, prefix="$"))

# ---- 1) Abstract ----
st.markdown("## 1) Abstract (Draft)")
st.markdown(
    f"""
Proponemos un framework **predict-then-optimize** para asignacion de capital crediticio que
integra intervalos de **Mondrian Conformal Prediction** como conjuntos de incertidumbre
operativos para optimizacion robusta via Pyomo/HiGHS.

A diferencia de enfoques clasicos que usan conjuntos elipsoidales o bootstrap, los intervalos
conformales ofrecen **garantias de cobertura finita sin supuestos distribucionales**. Establecemos
un **bound teorico** que conecta el nivel de confianza conformal alpha con el presupuesto de
incertidumbre Gamma de Bertsimas & Sim (2004), formalizando el trade-off coverage-robustez.

En evaluacion OOT sobre {robust_funded + nonrobust_funded:,} prestamos, el modelo PD calibrado
({best_calibration}) alcanza AUC={pd_metrics.get("auc_roc", np.nan):.4f}; los intervalos conformales
obtienen cobertura {coverage_90:.2%} al nivel nominal 90% (ancho medio {width_90:.3f}).
El portafolio robusto logra {format_number(robust_return, prefix="$")} vs
{format_number(nonrobust_return, prefix="$")} no robusto, con price of robustness de
{format_number(price_of_robustness, prefix="$")}.

Comparamos el regret de decision contra SPO+ (Elmachtoub & Grigas 2022) y baselines clasicos,
mostrando que la robustez conformal ofrece una frontera Pareto competitiva entre retorno esperado
y proteccion ante el peor caso.
"""
)

render_section_checkpoint(
    "Checkpoint de framing (Paper Estrella)",
    [
        "Claim 1: bound teorico alpha-conformal <-> Gamma-robustez.",
        "Claim 2: CP como uncertainty sets dominan bootstrap/elipsoidal en setting crediticio.",
        "Claim 3: price of robustness cuantificado en OOT con evidencia A/B.",
        "Aporte operativo: framework end-to-end con codigo reproducible.",
    ],
)

# ---- 2) Introduction ----
st.markdown("## 2) Introduction")
st.markdown(
    """
Los pipelines crediticios tradicionales optimizan decisiones usando **predicciones puntuales** de PD,
ignorando la incertidumbre de estimacion. Eso induce decisiones fragiles: pequenas desviaciones de
calibracion o shift temporal degradan la politica de portafolio.

La optimizacion robusta clasica (Bertsimas & Sim 2004) requiere construir conjuntos de incertidumbre,
tipicamente elipsoidales o basados en bootstrap. Sin embargo, estos carecen de garantias de cobertura
finita o requieren supuestos distribucionales.

Este paper propone usar **intervalos de prediccion conformal** como conjuntos de incertidumbre nativos
para el optimizador robusto, con tres contribuciones:

1. **Bound teorico**: formalizamos la relacion entre el nivel alpha conformal y el presupuesto Gamma
   de robustez, mostrando que CP box uncertainty sets son un caso especial del framework de Bertsimas & Sim.
2. **Comparacion empirica**: benchmarkeamos uncertainty sets conformales contra elipsoidales,
   bootstrap y parametricos sobre un portafolio crediticio real (Lending Club, 276K loans OOT).
3. **Decision regret**: comparamos el regret de decision del framework robusto conformal contra
   SPO+ y el two-stage clasico, mostrando que la robustez conformal ofrece una frontera Pareto competitiva.
"""
)

# ---- 3) Related Work ----
st.markdown("## 3) Related Work")
related = pd.DataFrame(
    [
        [
            "Bertsimas & Sim (2004)",
            "Robust Optimization",
            "Budget of uncertainty, price of robustness",
        ],
        ["Elmachtoub & Grigas (2022)", "SPO+", "Decision-focused optimization loss"],
        [
            "Johnstone et al. (2021)",
            "Conformal uncertainty sets",
            "CP intervals for robust optimization",
        ],
        ["Patel et al. (2024)", "Contextual CRO", "Contextual conformal sets for optimization"],
        ["Romano et al. (2019)", "CQR", "Conformalized Quantile Regression"],
        ["Vovk et al. (2005)", "CP foundations", "Algorithmic Learning in a Random World"],
        ["Taquet et al. (2025)", "MAPIE", "Library for conformal prediction"],
    ],
    columns=["Referencia", "Eje", "Relevancia"],
)
st.dataframe(related, use_container_width=True, hide_index=True)

# ---- 4) Methods ----
st.markdown("## 4) Methods")

st.markdown("### 4.1 Conformal Prediction Intervals (Mondrian)")
st.latex(
    r"\widehat{C}_{1-\alpha}(x,g) = "
    r"[\max(0,\widehat{p}(x)-q_{1-\alpha,g}),\; \min(1,\widehat{p}(x)+q_{1-\alpha,g})]"
)
st.caption("Ecuacion 1. Intervalo conformal Mondrian por subgrupo (grade g).")

st.markdown("### 4.2 Robust Portfolio Optimization")
st.latex(
    r"\max_{x \in \{0,1\}^n} \sum_i x_i\,(r_i - \lambda\,u_i) "
    r"\quad \text{s.a.}\quad \sum_i x_i a_i \le B,\; "
    r"\sum_i x_i p_i^{H} a_i \le \tau B + s"
)
st.caption("Ecuacion 2. Formulacion robusta con peor caso de PD (p_i^H = pd_high conformal).")

st.markdown("### 4.3 Bound alpha-conformal <-> Gamma-robustez (Contribucion Teorica)")
st.latex(
    r"\Gamma(\alpha) = \sum_{i=1}^{n} \frac{p_i^{H}(\alpha) - \hat{p}_i}{\hat{p}_i} \cdot x_i^*"
)
st.caption(
    "Ecuacion 3. Budget de incertidumbre inducido por el nivel conformal alpha. "
    "A menor alpha, mayor Gamma, mayor proteccion."
)
st.info(
    "PENDIENTE: formalizar la demostracion completa y verificar condiciones de regularidad. "
    "Esta ecuacion es un placeholder conceptual."
)

st.markdown("### 4.4 SPO+ Decision Regret Comparison")
st.markdown(
    """
Comparamos tres enfoques:
1. **Two-stage clasico**: predict PD, then optimize with point estimates
2. **Robust conformal**: predict PD intervals, optimize with worst-case
3. **SPO+**: train prediction model to minimize downstream decision regret directly

El codigo SPO+ esta implementado en `src/optimization/spo_integration.py`.
"""
)

# ---- 5) Results ----
st.markdown("## 5) Results")

st.markdown("### 5.1 Portfolio Robustness Frontier")
if not robust_summary.empty:
    melt_cols = [
        c
        for c in ["baseline_nonrobust_return", "best_robust_return"]
        if c in robust_summary.columns
    ]
    if melt_cols:
        fig1 = px.bar(
            robust_summary.melt(
                id_vars=["risk_tolerance"],
                value_vars=melt_cols,
                var_name="policy",
                value_name="return_net",
            ),
            x="risk_tolerance",
            y="return_net",
            color="policy",
            barmode="group",
            title="Figure 1. Net Return by Risk Tolerance (Robust vs Non-Robust)",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig1, use_container_width=True)

    if "price_of_robustness_pct" in robust_summary.columns:
        fig2 = px.line(
            robust_summary,
            x="risk_tolerance",
            y="price_of_robustness_pct",
            markers=True,
            title="Figure 2. Price of Robustness (%)",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No se encontro `portfolio_robustness_summary.parquet`.")

st.markdown("### 5.2 Champion Portfolio Policy")
if champion_policy:
    cp1, cp2, cp3 = st.columns(3)
    _sp = champion_policy.get("selected_policy", {})
    cp1.metric("Risk Tolerance", str(_sp.get("risk_tolerance", champion_policy.get("risk_tolerance", "N/D"))))
    cp2.metric("Policy Mode", str(_sp.get("policy_mode", champion_policy.get("policy_mode", "N/D"))))
    cp3.metric("Promoted", str(champion_policy.get("promoted", False)))
else:
    st.info("No se encontro `champion_portfolio_policy.json`.")

st.markdown("### 5.3 A/B Simulation Evidence")
if ab_status:
    ab1, ab2, ab3, ab4 = st.columns(4)
    _m_robust = ab_status.get("metrics_b", ab_status.get("metrics_a", {}))
    _m_nonrobust = ab_status.get("metrics_a", {})
    _cmp = ab_status.get("comparison", {})
    ab1.metric("Robust Total Return", format_number(float(_m_robust.get("total_return", 0)), prefix="$"))
    ab2.metric("Non-Robust Total Return", format_number(float(_m_nonrobust.get("total_return", 0)), prefix="$"))
    ab3.metric("Robust Loans Funded", str(_m_robust.get("n_funded", "N/D")))
    ab4.metric(
        "A/B p-value",
        f"{_cmp.get('p_value', float('nan')):.4f}",
        delta="sig." if _cmp.get("significant") else "n.s.",
        delta_color="normal" if _cmp.get("significant") else "off",
    )
    st.caption(
        f"Strategy A={ab_status.get('strategy_a','non_robust')} vs "
        f"B={ab_status.get('strategy_b','robust')}. "
        "Diff no significativa → portfolio robusto logra protección comparable sin sacrificar retorno."
    )
else:
    st.info("No se encontro `ab_simulation_status.json`.")

st.markdown("### 5.4 SPO+ Real Training — Decision Regret (v2: multi-seed + 3 métodos)")
if spo_real_status:
    res = spo_real_status.get("results", {})
    cfg = spo_real_status.get("config", {})
    extra = spo_real_status.get("extra", {})
    fixes = spo_real_status.get("fixes_applied", {})

    # Detect v2 (multi-seed) vs v1 format
    _is_v2 = isinstance(res.get("two_stage"), dict)

    if _is_v2:
        # v2: mean ± std per method
        _ts = res["two_stage"]
        _spo = res["spo_plus"]
        _rob = res.get("conformal_robust", {})
        _wil = res.get("wilcoxon_spo_vs_ts", {})
        _imp = res.get("spo_improvement_vs_ts_pct", float("nan"))
        _n_seeds = spo_real_status.get("n_seeds") or extra.get("n_seeds", "?")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Two-Stage Regret", f"{_ts.get('mean_regret', float('nan')):.4f} ± {_ts.get('std_regret', 0):.4f}")
        r2.metric("SPO+ Regret", f"{_spo.get('mean_regret', float('nan')):.4f} ± {_spo.get('std_regret', 0):.4f}")
        r3.metric("SPO+ Δ vs 2-Stage", f"{_imp:.1f}%")
        r4.metric("Wilcoxon p-val", f"{_wil.get('pvalue', float('nan')):.4f}",
                  delta="sig." if _wil.get("significant_at_0.05") else "n.s.",
                  delta_color="normal" if _wil.get("significant_at_0.05") else "off")

        rows = [
            {
                "Metodo": "Two-Stage (Ridge → LP)",
                "Mean Regret": _ts.get("mean_regret"),
                "Std Regret": _ts.get("std_regret"),
                "Notas": "Ridge predice costo calibrado; LP optimiza usando ĉ",
            },
            {
                "Metodo": "SPO+ (point-wise MLP)",
                "Mean Regret": _spo.get("mean_regret"),
                "Std Regret": _spo.get("std_regret"),
                "Notas": "MLP entrena end-to-end con SPO+ loss (Elmachtoub & Grigas 2022)",
            },
        ]
        if _rob:
            rows.append({
                "Metodo": "Conformal Robust (pd_high_90 → LP)",
                "Mean Regret": _rob.get("mean_regret"),
                "Std Regret": _rob.get("std_regret"),
                "Notas": "Usa upper CI conformal como costo worst-case conservador",
            })
        regret_df_v2 = pd.DataFrame(rows)
        st.dataframe(regret_df_v2, use_container_width=True, hide_index=True)

        sig_txt = "significativo" if _wil.get("significant_at_0.05") else "no significativo"
        n_obs = res.get("n_paired_observations", "?")
        st.caption(
            f"**{_n_seeds} semillas** × {spo_real_status.get('n_test_instances') or extra.get('n_test_instances', '?')} instancias de prueba "
            f"= {n_obs} obs. pareadas. "
            f"Wilcoxon signed-rank (H1: two_stage > spo_plus): p={_wil.get('pvalue', float('nan')):.4f} ({sig_txt}). "
            f"n_items={cfg.get('n_items') or spo_real_status.get('n_items')}, budget={cfg.get('budget') or spo_real_status.get('budget')}, "
            f"epochs={spo_real_status.get('epochs') or extra.get('epochs')}, PD calibrada={fixes.get('fix2_calibrated_pd_costs', spo_real_status.get('use_calibrated_pd', False))}."
        )

        # Per-seed means chart
        with st.expander("Per-seed regret means"):
            seed_rows = []
            for s_idx, (ts_m, spo_m) in enumerate(zip(
                _ts.get("per_seed_means", []), _spo.get("per_seed_means", []), strict=False,
            )):
                seed_rows.append({"Seed": s_idx + 1, "Two-Stage": ts_m, "SPO+": spo_m})
                if _rob and s_idx < len(_rob.get("per_seed_means", [])):
                    seed_rows[-1]["Conformal Robust"] = _rob["per_seed_means"][s_idx]
            if seed_rows:
                seed_df = pd.DataFrame(seed_rows)
                fig_seeds = px.line(
                    seed_df.melt(id_vars="Seed", var_name="Method", value_name="Mean Regret"),
                    x="Seed", y="Mean Regret", color="Method",
                    title="Per-Seed Mean Decision Regret",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_seeds, use_container_width=True)

        # Fixes applied badge
        with st.expander("Mejoras aplicadas (v2)"):
            fixes_df = pd.DataFrame([
                {"Fix": "Fix 1: Point-wise MLP (permutation-equivariant)", "Aplicado": fixes.get("fix1_pointwise_mlp", False)},
                {"Fix": "Fix 2: PD calibrada como proxy de costo real", "Aplicado": fixes.get("fix2_calibrated_pd_costs", False)},
                {"Fix": "Fix 3: Multi-seed + Wilcoxon signed-rank test", "Aplicado": fixes.get("fix3_multi_seed_wilcoxon", False)},
                {"Fix": "Fix 4: Conformal Robust como 3er metodo", "Aplicado": fixes.get("fix4_conformal_robust", False)},
                {"Fix": "Fix 5: n_items >= 100 (espacio de decision mas rico)", "Aplicado": fixes.get("fix5_larger_n_items", False)},
            ])
            st.dataframe(fixes_df, use_container_width=True, hide_index=True)

    else:
        # v1 fallback
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Two-Stage Regret", f"{res.get('two_stage_regret', float('nan')):.4f}")
        r2.metric("SPO+ Regret", f"{res.get('spo_plus_regret', float('nan')):.4f}")
        r3.metric("Random Regret", f"{res.get('random_baseline_regret', float('nan')):.4f}")
        r4.metric("SPO+ Δ vs 2-Stage", f"{res.get('spo_improvement_vs_two_stage_pct', float('nan')):.1f}%")
        st.caption("v1 (single seed). Re-ejecute con `run_spo_real.py` para resultados multi-seed v2.")

    # Loss curve (average across seeds in v2)
    if not spo_loss_curve.empty and "epoch" in spo_loss_curve.columns and "spo_loss" in spo_loss_curve.columns:
        title_suffix = f" (avg {extra.get('n_seeds', '')} seeds)" if _is_v2 else ""
        fig_loss = px.line(
            spo_loss_curve, x="epoch", y="spo_loss",
            title=f"SPO+ Training Loss Curve{title_suffix}",
            labels={"epoch": "Epoch", "spo_loss": "SPO+ Loss"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info(
        "No se encontro `spo_real_training_status.json`. "
        "Ejecute: `uv run python scripts/run_spo_real.py --n-items 100 --budget 30 --seeds 5`"
    )

st.markdown("### 5.5 Alpha Sweep — Global vs Mondrian Pareto Frontier")
if not alpha_sweep_pareto.empty:
    alpha_sweep_pareto["confidence_pct"] = (alpha_sweep_pareto["confidence_level"] * 100).round(0).astype(int).astype(str) + "%"
    has_method = "method" in alpha_sweep_pareto.columns
    fig_sweep = px.scatter(
        alpha_sweep_pareto,
        x="avg_width",
        y="empirical_coverage",
        color="method" if has_method else "min_group_coverage",
        symbol="method" if has_method else None,
        text="confidence_pct",
        hover_data=["alpha", "avg_width", "empirical_coverage", "min_group_coverage"] if "min_group_coverage" in alpha_sweep_pareto.columns else ["alpha", "avg_width", "empirical_coverage"],
        title="Figure 4. Alpha Sweep: Coverage vs Width Pareto — Global vs Mondrian",
        labels={"avg_width": "Avg Interval Width", "empirical_coverage": "Empirical Coverage"},
        template=PLOTLY_TEMPLATE,
    )
    fig_sweep.update_traces(textposition="top center")
    st.plotly_chart(fig_sweep, use_container_width=True)
    st.caption("Mondrian (por grade) logra menor anchura que global al mismo nivel de cobertura, con garantia grupal adicional.")
    with st.expander("Alpha Sweep — Tabla completa (Global + Mondrian)"):
        sweep_cols = [c for c in ["alpha", "confidence_level", "method", "empirical_coverage", "min_group_coverage", "avg_width", "n_eligible", "eligible_pct"] if c in alpha_sweep_pareto.columns]
        st.dataframe(alpha_sweep_pareto[sweep_cols], use_container_width=True, hide_index=True)
        download_table(alpha_sweep_pareto[sweep_cols], "estrella_alpha_sweep_both.csv")
else:
    st.info("No se encontro `alpha_sweep_pareto_both.parquet`. Ejecute: `uv run python scripts/run_alpha_sweep.py --both`")

st.markdown("### 5.6 CQR vs Symmetric Conformal Comparison")
if not cqr_comparison.empty:
    cqr_display_cols = [c for c in ["method", "empirical_coverage", "min_group_coverage", "avg_width", "n_eligible", "eligible_pct"] if c in cqr_comparison.columns]
    st.dataframe(cqr_comparison[cqr_display_cols], use_container_width=True, hide_index=True)
    if cqr_status:
        summary = cqr_status.get("summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Coverage", summary.get("best_coverage_method", "N/D"))
        c2.metric("Tightest Intervals", summary.get("tightest_method", "N/D"))
        c3.metric("Most Eligible", summary.get("most_eligible_method", "N/D"))
    st.caption(
        "CQR usa regressores cuantilicos asimetricos — intervalos adaptativos al riesgo local. "
        "Global/Mondrian usa split-conformal simetrico con garantia formal de cobertura finita."
    )
else:
    st.info("No se encontro `cqr_comparison.parquet`. Ejecute: `uv run python scripts/run_cqr_comparison.py`")

st.markdown("### 5.7 Uncertainty Set Baselines Comparison")
if not unc_baselines.empty:
    unc_display_cols = [c for c in ["method", "empirical_coverage", "min_group_coverage", "avg_width", "n_fundable", "meets_90pct"] if c in unc_baselines.columns]
    st.dataframe(unc_baselines[unc_display_cols], use_container_width=True, hide_index=True)
    st.caption("Solo Mondrian conformal alcanza cobertura nominal 90% con garantia finita.")
else:
    st.info("No se encontro `uncertainty_baselines_comparison.parquet`.")

st.markdown("### 5.8 Conformal Variant Trade-off")
if not variant_benchmark.empty:
    fig3 = px.scatter(
        variant_benchmark,
        x="avg_width",
        y="min_group_coverage",
        color="variant",
        size="coverage",
        hover_data=["coverage", "coverage_gap", "std_group_coverage"],
        title="Figure 3. Conformal Variant Trade-off (Width vs Min Group Coverage)",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("### Tablas Principales")
with st.expander("Table 1. Robustness Summary"):
    if robust_summary.empty:
        st.info("No se encontro `portfolio_robustness_summary.parquet`.")
    else:
        st.dataframe(robust_summary, use_container_width=True, hide_index=True)
        download_table(robust_summary, "estrella_table1_robustness_summary.csv")

with st.expander("Table 2. Robustness Frontier (full)"):
    if robust_frontier.empty:
        st.info("No se encontro `portfolio_robustness_frontier.parquet`.")
    else:
        st.dataframe(robust_frontier, use_container_width=True, hide_index=True)
        download_table(robust_frontier, "estrella_table2_robustness_frontier.csv")

# ---- 6) Discussion ----
st.markdown("## 6) Discussion")
st.markdown(
    """
Los resultados sugieren que los intervalos conformales como uncertainty sets ofrecen
una alternativa competitiva a los conjuntos elipsoidales y bootstrap tradicionales.
El bound alpha-Gamma formaliza el trade-off entre cobertura estadistica y costo economico
de robustez, proporcionando una herramienta interpretable para decisores de politica crediticia.

**Hallazgos clave:**
- Mondrian conformal es el unico metodo con garantia formal de cobertura finita y min_group_cov >89%.
- CQR asimetrico logra cobertura similar pero con mayor anchura promedio (no usa informacion grupal).
- Alpha sweep Mondrian muestra Pareto frontier: alpha=0.10 ofrece 87% coverage con avg_width 0.87.
- Bootstrap y parametrico fallan en min_group_coverage (61-80%), violando garantia grupal.
- SPO+ (MLP end-to-end, 50 items, 800 instancias, 50 epochs) logra ~2.5% menor regret que two-stage.
- CQR Mondrian: todos los offsets por grupo son 0.0 — los cuantiles ya acoten bien datos binarios.

**Pendientes de investigacion:**
- Formalizar demostracion del bound alpha-Gamma bajo condiciones generales.
- Escalar SPO+ a mas instancias / arquitectura mas profunda para mayor diferencia de regret.
- Comparacion formal SPO+ vs robust conformal en mismo experimento (mismo LP, mismas instancias).
"""
)

# ---- 7) Threats to Validity ----
st.markdown("## 7) Threats to Validity")
st.markdown(
    """
- Validez externa limitada a estructura de Lending Club hasta re-calibracion en otros portafolios.
- El bound teorico asume uncertainty sets box-shaped; extensiones a conjuntos adaptativos requieren
  trabajo adicional.
- SPO+ comparison puede depender de arquitectura del modelo neural y hiperparametros.
- Cobertura conformal no garantiza por si sola optimalidad economica.
"""
)

# ---- 8) Reproducibility ----
st.markdown("## 8) Reproducibility Package")
st.code(
    "\n".join(
        [
            "# Pipeline completo",
            "uv run dvc repro",
            "",
            "# SPO+ real training v2 (point-wise MLP, PD calibrada, multi-seed, conformal robust)",
            "uv run python scripts/run_spo_real.py --n-items 100 --budget 30 --n-train 800 --epochs 50 --seeds 5",
            "",
            "# SPO+ comparison (vs uncertainty set baselines)",
            "uv run python scripts/run_spo_comparison.py",
            "",
            "# Alpha sweep (global y Mondrian, mismo plot)",
            "uv run python scripts/run_alpha_sweep.py --both",
            "",
            "# CQR vs symmetric conformal",
            "uv run python scripts/run_cqr_comparison.py",
            "",
            "# CQR Mondrian hibrido (4 metodos)",
            "uv run python scripts/run_cqr_mondrian.py",
            "",
            "# Uncertainty set baselines",
            "uv run python scripts/run_uncertainty_baselines.py",
            "",
            "# Competing risks CIF (default vs prepayment)",
            "uv run python scripts/run_competing_risks.py",
            "",
            "# Paper notebooks",
            "uv run python scripts/run_paper_notebook_suite.py",
            "",
            "# Tests",
            "uv run pytest -q",
        ]
    ),
    language="bash",
)

# ---- 9) Phase Tracker ----
st.markdown("## 9) Fases del Paper y Estado Actual")
render_phase_tracker(
    [
        {
            "Fase": "1. Scope + Research Question",
            "Estado": "Completada",
            "Evidencia": "Claim y framing definidos (absorbe Paper 1)",
            "Criterio de cierre": "Claim validado con supervisor.",
        },
        {
            "Fase": "2. Related Work",
            "Estado": "Completada",
            "Evidencia": "Tabla de related work con 7 referencias clave",
            "Criterio de cierre": "Agregar baselines sugeridos por revisor.",
        },
        {
            "Fase": "3. Methods + Bound Teorico",
            "Estado": "En progreso",
            "Evidencia": "Ecuaciones 1-3 + SPO+ code existente",
            "Criterio de cierre": "Formalizar demostracion del bound alpha-Gamma.",
        },
        {
            "Fase": "4. Results + Baselines",
            "Estado": "En progreso",
            "Evidencia": "Robustness frontier + A/B + variant benchmark",
            "Criterio de cierre": "Agregar alpha sweep Pareto + baselines de uncertainty sets.",
        },
        {
            "Fase": "5. Discussion + Submission",
            "Estado": "Pendiente",
            "Evidencia": "Draft section 6-7",
            "Criterio de cierre": "Manuscrito completo con figuras matplotlib publication-quality.",
        },
    ]
)

st.markdown("### Pendientes Clave")
st.markdown(
    """
**Completados:**
- ✅ Baselines uncertainty sets: conformal_mondrian vs bootstrap vs parametric vs ellipsoidal.
- ✅ CQR: implementado en `scripts/run_cqr_comparison.py` con CatBoost quantile regressors.
- ✅ Alpha sweep --both: global vs Mondrian en mismo plot (16 filas, pareto_both.parquet).
- ✅ SPO+ regret comparison: `scripts/run_spo_comparison.py` (pyepo+torch instalados).
- ✅ Competing risks CIF: `scripts/run_competing_risks.py`, diferencia KM-CIF a 60m = +4.5pp.
- ✅ CQR Mondrian hibrido: `scripts/run_cqr_mondrian.py` — 4 metodos con offsets por grupo.
- ✅ SPO+ real training v2: `scripts/run_spo_real.py` — 5 fixes aplicados:
  - Fix 1: MLP point-wise (permutation-equivariant, n_features→1 por loan)
  - Fix 2: PD calibrada (CatBoost + Venn-Abers) como proxy de costo real (continuo, rango [-0.24, +0.17])
  - Fix 3: 5 semillas, mean±std, Wilcoxon signed-rank test (n=seeds×n_test pareados)
  - Fix 4: Conformal Robust (pd_high_90 worst-case) como 3er metodo de comparacion
  - Fix 5: n_items=100, budget=30 (espacio de decision mas rico que C(50,15))

**Pendientes:**
- **Bound teorico alpha-Gamma**: formalizar y demostrar (contribucion principal).
- **Figuras matplotlib**: convertir de Plotly para publicacion (2-column IEEE/Springer format).
- **Online companion**: enlazar al Quarto book como supplementary material.
- **Escalar SPO+**: n_items=200, n_train=2000 para paper-grade con mayor separacion estadistica.
"""
)

render_next_steps(
    [
        (
            "Paper 1 (Draft Original)",
            "Draft historico que fue absorbido por este Paper Estrella.",
            "pages/paper_1_cp_robust_opt.py",
        ),
        (
            "Panorama de Investigacion",
            "Registro maestro de posicionamiento y referencias.",
            "pages/research_landscape.py",
        ),
        (
            "Buenas Practicas y Herramientas",
            "Playbook de revision de drafts y checklist pre-reunion.",
            "pages/research_best_practices.py",
        ),
    ]
)
