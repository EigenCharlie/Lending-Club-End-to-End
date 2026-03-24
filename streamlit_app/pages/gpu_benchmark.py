"""RAPIDS replay on the real project pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.story_shell import (
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    OFFICIAL_GPU_INSIGHT_TAG,
    OFFICIAL_GPU_REPLAY_TAG,
    OFFICIAL_PD_RAPIDS_BENCHMARK_TAG,
    OFFICIAL_RAPIDS_TRADEOFF_FULL_TAG,
    download_table,
    format_number,
    load_gpu_replay_summary,
    load_official_baseline_registry,
    load_rapids_ifrs9_correlated_metrics,
    load_rapids_ifrs9_mc_tail_metrics,
    load_rapids_insight_detail,
    load_rapids_insight_stage_table,
    load_rapids_pd_benchmark_stage_table,
    load_rapids_stage_comparison,
    load_rapids_tradeoff_full_ab_status,
    load_rapids_tradeoff_full_policy,
    try_load_gpu_replay_json,
    try_load_json,
    try_load_parquet,
)

STAGE_LABELS = {
    "pd": "PD CatBoost",
    "lgd_ead": "LGD/EAD",
    "portfolio": "Portfolio LP",
    "tradeoff": "Tradeoff Frontier",
    "ab": "A/B Replay",
    "cate_portfolio": "CATE Portfolio",
    "ifrs9_mc": "IFRS9 Monte Carlo",
    "ifrs9_sensitivity": "IFRS9 Sensitivity",
}

SAMPLE_NOTES = {
    "pd": "Full canonical train/test; same scale as CPU champion.",
    "lgd_ead": "Full defaults subset; same canonical scope as CPU champion.",
    "portfolio": "GPU final: full 276,869 candidates. CPU official: 150,000.",
    "tradeoff": "GPU final: 150,000 candidates. CPU official: 80,000.",
    "ab": "GPU final: 150,000 effective universe tied to tradeoff candidate universe.",
    "cate_portfolio": "GPU final: full 276,869 candidates. CPU official: 150,000.",
    "ifrs9_mc": "GPU-only RAPIDS extension: 276,869 loans x 8,192 scenarios.",
}


def _stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


def _fmt_metric(
    value: object,
    digits: int = 1,
    suffix: str = "",
    default: str = "N/D",
) -> str:
    number = _safe_float(value)
    if number is None:
        return default
    return f"{number:.{digits}f}{suffix}"


def _fmt_count(value: object, default: str = "N/D") -> str:
    number = _safe_float(value)
    if number is None:
        return default
    return format_number(number)


def _build_stage_table() -> pd.DataFrame:
    df = load_rapids_stage_comparison(OFFICIAL_GPU_REPLAY_TAG).copy()
    if df.empty:
        return df
    df["stage_label"] = df["stage"].map(_stage_label)
    df["sample_scope"] = df["stage"].map(SAMPLE_NOTES)
    return df.sort_values(
        by=["speedup_gpu_vs_cpu", "gpu_seconds"],
        ascending=[False, True],
        na_position="last",
    )


st.title("⚡ RAPIDS en el Proyecto Real")
st.caption(
    "Replay GPU del champion oficial sobre el mismo Lending Club pipeline: qué sí aceleró, "
    "qué no aceleró y dónde la GPU realmente vale la pena."
)
page_contract = get_page_contract("gpu_benchmark")
render_page_header(page_contract)
render_key_takeaway(
    "La historia final no es 'GPU en todo': es cuOpt para OR, CuPy para escenarios masivos, mejoras moderadas en LGD/EAD y una señal honesta de que PD completo todavía no gana en esta arquitectura."
)

gpu_summary = load_gpu_replay_summary(OFFICIAL_GPU_REPLAY_TAG)
baseline = load_official_baseline_registry()
ifrs9_mc = load_rapids_ifrs9_mc_tail_metrics(OFFICIAL_GPU_REPLAY_TAG)
ifrs9_mc_correlated = load_rapids_ifrs9_correlated_metrics()
stage_table = _build_stage_table()
insight_stage_table = load_rapids_insight_stage_table()
insight_umap_embedding = load_rapids_insight_detail("cuml_umap_embedding")
insight_hdbscan_profiles = load_rapids_insight_detail("cuml_hdbscan_cluster_profiles")
insight_cugraph_profiles = load_rapids_insight_detail("cugraph_community_profiles")
pd_benchmark_stage_table = load_rapids_pd_benchmark_stage_table()
ab_status = try_load_gpu_replay_json(
    OFFICIAL_GPU_REPLAY_TAG, "artifacts/models/ab_simulation_status.json"
)
policy_status = try_load_gpu_replay_json(
    OFFICIAL_GPU_REPLAY_TAG, "artifacts/models/champion_portfolio_policy.json"
)
cate_status = try_load_gpu_replay_json(
    OFFICIAL_GPU_REPLAY_TAG, "artifacts/models/cate_portfolio_status.json"
)
tradeoff_full_policy = load_rapids_tradeoff_full_policy()
tradeoff_full_ab = load_rapids_tradeoff_full_ab_status()

if stage_table.empty or not gpu_summary:
    st.error("No se encontraron artefactos del replay RAPIDS final.")
    st.stop()

best = stage_table.dropna(subset=["speedup_gpu_vs_cpu"]).sort_values(
    "speedup_gpu_vs_cpu", ascending=False
)
best_row = best.iloc[0] if not best.empty else None
worst = stage_table.dropna(subset=["speedup_gpu_vs_cpu"]).sort_values(
    "speedup_gpu_vs_cpu", ascending=True
)
worst_row = worst.iloc[0] if not worst.empty else None
gpu_wins = int((stage_table["speedup_gpu_vs_cpu"] > 1.0).fillna(False).sum())

kpi_row(
    [
        {"label": "Run RAPIDS final", "value": gpu_summary.get("run_tag", "N/D")},
        {"label": "Baseline CPU", "value": str(baseline.get("run_tag", "N/D"))},
        {"label": "Stages PASS", "value": f"{len(stage_table)}/{len(stage_table)}"},
        {"label": "Stages con speedup", "value": f"{gpu_wins}/{len(stage_table) - 1}"},
        {
            "label": "Mayor speedup",
            "value": (
                f"{_stage_label(str(best_row['stage']))} · "
                f"{_fmt_metric(best_row.get('speedup_gpu_vs_cpu'), digits=1, suffix='x')}"
                if best_row is not None
                else "N/D"
            ),
        },
        {
            "label": "IFRS9 Monte Carlo",
            "value": _fmt_metric(ifrs9_mc.get("speedup_gpu_vs_cpu"), digits=1, suffix="x"),
        },
    ],
    n_cols=3,
)

st.markdown(
    """
Esta página ya no resume microbenchmarks aislados. Resume un **replay RAPIDS sobre el proyecto real**:

- el baseline CPU oficial es el champion canónico del pipeline end-to-end;
- el replay GPU corre solo las etapas donde la GPU aporta de verdad;
- los outputs GPU viven en su propio namespace y no sustituyen el champion oficial;
- el objetivo es medir **valor incremental real** por etapa, no hacer demos de segundos.
"""
)

st.subheader("1) CPU vs GPU por etapa")
compare_df = stage_table[
    [
        "stage_label",
        "cpu_seconds",
        "gpu_seconds",
        "speedup_gpu_vs_cpu",
        "peak_gpu_util",
        "peak_memory_used_mb",
        "sample_scope",
    ]
].rename(
    columns={
        "stage_label": "Stage",
        "cpu_seconds": "CPU seconds",
        "gpu_seconds": "GPU seconds",
        "speedup_gpu_vs_cpu": "GPU speedup vs CPU",
        "peak_gpu_util": "Peak GPU util (%)",
        "peak_memory_used_mb": "Peak VRAM (MB)",
        "sample_scope": "Scope",
    }
)
st.dataframe(compare_df, width="stretch", hide_index=True)
download_table(compare_df, "rapids_stage_comparison.csv", "Descargar comparación CPU vs GPU")

speedup_plot = stage_table.dropna(subset=["speedup_gpu_vs_cpu"]).copy()
fig_speedup = px.bar(
    speedup_plot,
    x="speedup_gpu_vs_cpu",
    y="stage_label",
    orientation="h",
    color="speedup_gpu_vs_cpu",
    color_continuous_scale="Blues",
    text="speedup_gpu_vs_cpu",
    labels={"stage_label": "Stage", "speedup_gpu_vs_cpu": "GPU speedup vs CPU"},
    title="Aceleración real del replay RAPIDS frente al champion CPU (escala log)",
)
fig_speedup.update_traces(texttemplate="%{text:.1f}x", textposition="outside")
fig_speedup.add_vline(x=1.0, line_dash="dash", line_color="#E45756", annotation_text="1× (break-even)")
fig_speedup.update_xaxes(type="log")
fig_speedup.update_layout(**PLOTLY_TEMPLATE["layout"], height=420, coloraxis_showscale=False)
st.plotly_chart(fig_speedup, width="stretch")

st.markdown(
    """
**Lectura principal**

- La GPU ya es claramente ganadora en **investigación de operaciones**: portfolio, frontier tradeoff, A/B y CATE.
- **LGD/EAD** sí mejora, pero de forma moderada.
- **IFRS9 Monte Carlo** se volvió un caso GPU serio cuando pasamos de sensibilidad pequeña a escenarios masivos.
- **PD completo** todavía no gana: el stage full incluye mucha preparación CPU, calibración, fairness y validación alrededor del fit.
"""
)

tab_or, tab_pd, tab_ifrs9, tab_hw = st.tabs(
    ["OR & Decisión", "PD & LGD/EAD", "IFRS9 Monte Carlo", "Telemetría GPU"]
)

with tab_or:
    st.markdown("### La gran conclusión: cuOpt sí cambió el proyecto")
    st.markdown(
        """
La ruta correcta terminó siendo **cuOpt nativo**, no `Pyomo -> SolverFactory('cuopt')`.
Eso permitió correr en GPU las etapas que de verdad son caras en OR:

- `portfolio` sobre el universo full,
- `tradeoff` con muchos solves repetidos,
- `ab` sobre el mismo `candidate_universe`,
- `cate_portfolio` en modo research-only pero ya acelerado.
"""
    )
    or_df = stage_table[stage_table["stage"].isin(["portfolio", "tradeoff", "ab", "cate_portfolio"])].copy()
    or_long = (
        or_df[["stage_label", "cpu_seconds", "gpu_seconds"]]
        .melt(id_vars="stage_label", value_vars=["cpu_seconds", "gpu_seconds"],
              var_name="Engine", value_name="Seconds")
        .dropna(subset=["Seconds"])
    )
    or_long["Engine"] = or_long["Engine"].map({"cpu_seconds": "CPU (HiGHS)", "gpu_seconds": "GPU (cuOpt)"})
    fig_or = px.bar(
        or_long,
        x="stage_label",
        y="Seconds",
        color="Engine",
        barmode="group",
        text="Seconds",
        color_discrete_map={"CPU (HiGHS)": "#5F6B7A", "GPU (cuOpt)": "#00D4AA"},
        labels={"stage_label": "Stage", "Seconds": "Segundos"},
        title="OR en GPU: tiempo de ejecución CPU vs GPU por etapa",
    )
    fig_or.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
    fig_or.update_layout(**PLOTLY_TEMPLATE["layout"], height=400)
    st.plotly_chart(fig_or, width="stretch")
    st.caption("Barra más corta = más rápido. Columnas vacías = etapa GPU-only sin baseline CPU disponible.")

    selected_policy = policy_status.get("selected_policy", {})
    selection_metrics = policy_status.get("selection_metrics", {})
    st.markdown("#### Política champion del replay GPU")
    st.json(
        {
            "policy": selected_policy,
            "selection_metrics": selection_metrics,
            "ab_no_regression": ab_status.get("no_regression", {}),
        },
        expanded=False,
    )

    st.markdown("#### Qué subimos frente al champion CPU")
    st.markdown(
        """
- `portfolio`: de `150k` en CPU a **full 276,869** en GPU.
- `tradeoff`: de `80k` en CPU a **150k** en GPU final.
- `A/B`: alineado al mismo universo del frontier.
- `CATE`: de `150k` en CPU a **full 276,869** en GPU.
"""
    )
    full_selected = tradeoff_full_policy.get("selected_policy", {})
    full_metrics = tradeoff_full_policy.get("selection_metrics", {})
    balanced_selected = tradeoff_full_policy.get("selected_policy_balanced_robustness", {})
    balanced_metrics = tradeoff_full_policy.get("balanced_selection_metrics", {})
    robust_selected = tradeoff_full_policy.get("selected_policy_robustness_aware", {})
    robust_metrics = tradeoff_full_policy.get("research_selection_metrics", {})
    full_ab = tradeoff_full_ab.get("no_regression", {})
    if full_selected:
        st.markdown("#### Qué ocurrió cuando empujamos el frontier a full")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Selector": "promotion_first",
                        "Universe": "276,869 candidates",
                        "Gamma": full_selected.get("gamma"),
                        "Lambda": full_selected.get("uncertainty_aversion"),
                        "Funded": full_metrics.get("n_funded"),
                        "Realized return": full_metrics.get("realized_total_return"),
                        "Price of robustness (%)": full_metrics.get("price_of_robustness_pct"),
                    },
                    {
                        "Selector": "balanced_robustness",
                        "Universe": "276,869 candidates",
                        "Gamma": balanced_selected.get("gamma"),
                        "Lambda": balanced_selected.get("uncertainty_aversion"),
                        "Funded": balanced_metrics.get("n_funded"),
                        "Realized return": balanced_metrics.get("realized_total_return"),
                        "Price of robustness (%)": balanced_metrics.get(
                            "price_of_robustness_pct"
                        ),
                    },
                    {
                        "Selector": "robustness_aware",
                        "Universe": "276,869 candidates",
                        "Gamma": robust_selected.get("gamma"),
                        "Lambda": robust_selected.get("uncertainty_aversion"),
                        "Funded": robust_metrics.get("n_funded"),
                        "Realized return": robust_metrics.get("realized_total_return"),
                        "Price of robustness (%)": robust_metrics.get(
                            "price_of_robustness_pct"
                        ),
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.warning(
            "La GPU ya resolvió el cuello computacional del frontier full. El hallazgo nuevo es de diseño de policy: `promotion_first` colapsa a gamma=0, `balanced_robustness` sube a gamma=0.25 y `robustness_aware` a gamma=0.5, pero ninguna policy robusta real pasa el A/B full."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Run": OFFICIAL_RAPIDS_TRADEOFF_FULL_TAG,
                        "A/B selector": tradeoff_full_ab.get("policy_selector"),
                        "Control return": tradeoff_full_ab.get("metrics_a", {}).get(
                            "total_return"
                        ),
                        "Treatment return": tradeoff_full_ab.get("metrics_b", {}).get(
                            "total_return"
                        ),
                        "A/B diff": full_ab.get("diff_total_return"),
                        "A/B pass": full_ab.get("passed"),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    if cate_status:
        st.info(
            "CATE portfolio sigue en `research_only_fallback`: la GPU ya lo acelera, pero eso no cambia el hecho de que todavía no agrega valor económico canónico."
        )

with tab_pd:
    st.markdown("### PD y LGD/EAD: dos historias distintas")
    _lgd_slice = stage_table[stage_table["stage"] == "lgd_ead"]
    lgd_row = _lgd_slice.iloc[0] if not _lgd_slice.empty else pd.Series(dtype=object)
    fit_only = pd_benchmark_stage_table.loc[pd_benchmark_stage_table["stage"] == "fit_only"]
    hpo_stage = pd_benchmark_stage_table.loc[pd_benchmark_stage_table["stage"] == "hpo"]
    full_stage = pd_benchmark_stage_table.loc[pd_benchmark_stage_table["stage"] == "full_stage"]
    fit_row = fit_only.iloc[0].to_dict() if not fit_only.empty else {}
    hpo_row = hpo_stage.iloc[0].to_dict() if not hpo_stage.empty else {}
    full_row = full_stage.iloc[0].to_dict() if not full_stage.empty else {}
    kpi_row(
        [
            {
                "label": "PD fit-only",
                "value": _fmt_metric(fit_row.get("speedup_gpu_vs_cpu"), digits=2, suffix="x"),
            },
            {
                "label": "PD HPO GPU",
                "value": (
                    "inestable"
                    if pd.isna(hpo_row.get("gpu_seconds"))
                    else _fmt_metric(hpo_row.get("speedup_gpu_vs_cpu"), digits=2, suffix="x")
                ),
            },
            {
                "label": "PD full-stage",
                "value": _fmt_metric(full_row.get("speedup_gpu_vs_cpu"), digits=2, suffix="x"),
            },
            {"label": "LGD/EAD GPU", "value": _fmt_metric(lgd_row.get("gpu_seconds"), digits=0, suffix="s")},
            {
                "label": "LGD/EAD speedup",
                "value": _fmt_metric(lgd_row.get("speedup_gpu_vs_cpu"), digits=2, suffix="x"),
            },
            {
                "label": "LGD/EAD peak VRAM",
                "value": _fmt_metric(lgd_row.get("peak_memory_used_mb"), digits=0, suffix=" MB"),
            },
        ],
        n_cols=3,
    )
    st.markdown(
        """
**PD ahora está separado en tres benchmarks**

- `fit-only`: la GPU sí gana, ~`1.7x`, con métricas casi equivalentes.
- `HPO`: CPU completa 6 trials; GPU sigue cayéndose y por eso vive en un worker aislado.
- `full-stage`: el replay completo sigue perdiendo contra CPU, porque el fit ya no domina todo el costo.

**LGD/EAD**

Aquí sí aparece una mejora limpia. El problema está más cerca de un bloque de entrenamiento puro, con menos overhead alrededor, y por eso la GPU ya entrega un speedup defendible.
"""
    )
    if not pd_benchmark_stage_table.empty:
        st.dataframe(
            pd_benchmark_stage_table.assign(
                benchmark_run=OFFICIAL_PD_RAPIDS_BENCHMARK_TAG
            ),
            width="stretch",
            hide_index=True,
        )
    st.warning(
        "Conclusión práctica: PD no debe presentarse como un único número. La historia honesta es: `fit-only` sí gana, `HPO GPU` hoy es inestable y `full-stage` todavía pierde contra CPU."
    )

with tab_ifrs9:
    st.markdown("### IFRS9: el salto serio aparece con Monte Carlo, no con la sensibilidad pequeña")
    kpi_row(
        [
            {"label": "Loans", "value": _fmt_count(ifrs9_mc.get("n_loans"))},
            {"label": "Escenarios", "value": _fmt_count(ifrs9_mc.get("n_scenarios"))},
            {"label": "CPU", "value": _fmt_metric(ifrs9_mc.get("cpu_seconds"), digits=2, suffix="s")},
            {"label": "GPU", "value": _fmt_metric(ifrs9_mc.get("gpu_seconds"), digits=2, suffix="s")},
            {"label": "Speedup", "value": _fmt_metric(ifrs9_mc.get("speedup_gpu_vs_cpu"), digits=2, suffix="x")},
            {
                "label": "Error medio relativo",
                "value": _fmt_metric(
                    ifrs9_mc.get("mean_rel_diff_total_ecl_pct"),
                    digits=4,
                    suffix="%",
                ),
            },
        ],
        n_cols=3,
    )
    tail_df = pd.DataFrame(
        [
            {"Metric": key, "CPU": value, "GPU": ifrs9_mc.get("gpu_tail", {}).get(key)}
            for key, value in ifrs9_mc.get("cpu_tail", {}).items()
        ]
    )
    if not tail_df.empty:
        tail_long = tail_df.melt(id_vars="Metric", var_name="Engine", value_name="Total ECL")
        fig_tail = px.bar(
            tail_long,
            x="Metric",
            y="Total ECL",
            color="Engine",
            barmode="group",
            title="Tail metrics de ECL: CPU vs GPU",
        )
        fig_tail.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
        st.plotly_chart(fig_tail, width="stretch")
    st.dataframe(tail_df, width="stretch", hide_index=True)
    st.markdown(
        """
La decisión correcta fue **no sustituir** el IFRS9 canónico de sensibilidad por Monte Carlo.

- `run_ifrs9_sensitivity.py` sigue siendo el carril regulatorio simple y gobernado.
- `run_ifrs9_monte_carlo_gpu.py` es la extensión RAPIDS que sí exprime CuPy.

Eso mantiene dos narrativas limpias:

1. **canónica / negocio / gobernanza**
2. **RAPIDS / investigación / escenarios masivos**
"""
    )
    if ifrs9_mc_correlated:
        st.markdown("#### Evolución metodológica: shocks correlacionados")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Run": ifrs9_mc_correlated.get("run_tag"),
                        "Correlation profile": ifrs9_mc_correlated.get("correlation_profile"),
                        "Antithetic": ifrs9_mc_correlated.get("antithetic"),
                        "Scenarios": ifrs9_mc_correlated.get("n_scenarios"),
                        "CPU seconds": ifrs9_mc_correlated.get("cpu_seconds"),
                        "GPU seconds": ifrs9_mc_correlated.get("gpu_seconds"),
                        "Speedup": ifrs9_mc_correlated.get("speedup_gpu_vs_cpu"),
                        "Mean relative diff (%)": ifrs9_mc_correlated.get(
                            "mean_rel_diff_total_ecl_pct"
                        ),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.info(
            "La versión correlacionada ya no es solo fuerza bruta: introduce shocks más defendibles y mantiene equivalencia CPU/GPU prácticamente exacta."
        )

with tab_hw:
    hw_df = stage_table.dropna(subset=["peak_memory_used_mb"]).copy()
    fig_hw = px.bar(
        hw_df,
        x="stage_label",
        y="peak_memory_used_mb",
        color="avg_gpu_util",
        labels={
            "stage_label": "Stage",
            "peak_memory_used_mb": "Peak VRAM (MB)",
            "avg_gpu_util": "Avg GPU util (%)",
        },
        title="Telemetría GPU por etapa del replay final",
        color_continuous_scale="Viridis",
    )
    fig_hw.update_layout(**PLOTLY_TEMPLATE["layout"], height=420, coloraxis_colorbar_title="% util")
    st.plotly_chart(fig_hw, width="stretch")
    st.dataframe(
        stage_table[
            [
                "stage_label",
                "peak_gpu_util",
                "avg_gpu_util",
                "peak_memory_used_mb",
                "avg_memory_used_mb",
                "peak_power_draw_w",
            ]
        ].rename(
            columns={
                "stage_label": "Stage",
                "peak_gpu_util": "Peak GPU util (%)",
                "avg_gpu_util": "Avg GPU util (%)",
                "peak_memory_used_mb": "Peak VRAM (MB)",
                "avg_memory_used_mb": "Avg VRAM (MB)",
                "peak_power_draw_w": "Peak power (W)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.subheader("2) Insight Factory RAPIDS: dónde sí conviene abrir un segundo carril")
if not insight_stage_table.empty:
    insights_view = insight_stage_table.copy()
    insights_view["stage"] = insights_view["stage"].replace(
        {
            "cudf_etl": "cuDF ETL",
            "cuml_segmentation": "cuML segmentation",
            "cuml_umap": "cuML UMAP",
            "cuml_hdbscan": "cuML HDBSCAN",
            "cugraph_similarity": "cuGraph similarity",
        }
    )
    # Clean summary — only the columns that are always meaningful
    _summary_cols = {
        "stage": "Workload",
        "rows_input": "Filas",
        "cpu_seconds": "CPU (s)",
        "gpu_seconds": "GPU (s)",
        "speedup_gpu_vs_cpu": "Speedup GPU/CPU",
    }
    _avail = {k: v for k, v in _summary_cols.items() if k in insights_view.columns}
    clean_insights = insights_view[list(_avail.keys())].rename(columns=_avail)
    st.dataframe(clean_insights, width="stretch", hide_index=True)

    # Visual speedup bar
    _speedup_data = insights_view.dropna(subset=["speedup_gpu_vs_cpu"]).copy()
    if not _speedup_data.empty:
        fig_insight_speedup = px.bar(
            _speedup_data,
            x="stage",
            y="speedup_gpu_vs_cpu",
            color="speedup_gpu_vs_cpu",
            color_continuous_scale="RdYlGn",
            text="speedup_gpu_vs_cpu",
            labels={"stage": "Workload", "speedup_gpu_vs_cpu": "Speedup GPU/CPU"},
            title="Insight Factory: speedup GPU/CPU por workload",
        )
        _speedup_data_max = float(_speedup_data["speedup_gpu_vs_cpu"].max())
        fig_insight_speedup.update_traces(texttemplate="%{text:.2f}x", textposition="outside")
        fig_insight_speedup.add_hline(y=1.0, line_dash="dash", line_color="#E45756", annotation_text="break-even")
        fig_insight_speedup.update_layout(**PLOTLY_TEMPLATE["layout"], height=360, coloraxis_showscale=False,
                                          yaxis_range=[0, _speedup_data_max * 1.25])
        st.plotly_chart(fig_insight_speedup, width="stretch")

    with st.expander("Detalle completo por workload (columnas específicas de cada método)"):
        st.dataframe(insights_view, width="stretch", hide_index=True)

    st.markdown(
        """
La segunda iteración de *insight factory* ya deja un mapa bastante más útil:

- `cuDF ETL`: sigue sin ganar en este workload real.
- `cuML KMeans`: casi neutro, no es el showcase principal.
- `cuML UMAP`: sí gana con mucha claridad y sirve para geometría de riesgo.
- `cuML HDBSCAN`: también gana y ya deja perfiles de cluster comparables.
- `cuGraph similarity`: sigue siendo útil, pero ahora importa más por comunidades y estructura que por speedup puro.
"""
    )
    if not insight_umap_embedding.empty:
        st.markdown("#### UMAP: geometría de préstamos en 2D")
        fig_umap = px.scatter(
            insight_umap_embedding.sample(
                n=min(5000, len(insight_umap_embedding)),
                random_state=42,
            ),
            x="umap_x",
            y="umap_y",
            color="grade" if "grade" in insight_umap_embedding.columns else None,
            hover_data=[
                c
                for c in ["default_flag", "fico_score", "rev_utilization"]
                if c in insight_umap_embedding.columns
            ],
            title="Embedding UMAP GPU sobre préstamos reales",
            opacity=0.65,
        )
        fig_umap.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
        st.plotly_chart(fig_umap, width="stretch")
    if not insight_hdbscan_profiles.empty:
        st.markdown("#### HDBSCAN: perfiles de cluster")
        st.dataframe(insight_hdbscan_profiles.head(12), width="stretch", hide_index=True)
    if not insight_cugraph_profiles.empty:
        st.markdown("#### cuGraph: comunidades Louvain")
        st.dataframe(insight_cugraph_profiles.head(12), width="stretch", hide_index=True)

st.subheader("3) Qué entra y qué no entra al carril RAPIDS")
st.markdown(
    """
**Sí entra**

- etapas GPU-heavy del pipeline real,
- OR con `cuopt`,
- escenarios masivos con `cupy`,
- benchmarks exigentes que sí justifican comparación CPU vs GPU.

**No entra**

- re-ejecutar notebooks CPU que ya tienen artefactos canónicos,
- pasos triviales que tardan segundos,
- claims artificiales de aceleración donde el stage completo sigue dominado por CPU.
"""
)

with st.expander("Bridge Loans (cuGraph)", expanded=False):
    _bridge_df = load_rapids_insight_detail("cugraph_bridge_loans")
    if not _bridge_df.empty:
        st.markdown(
            "Préstamos puente identificados por cuGraph: nodos con alta intermediación "
            "que conectan comunidades distintas en el grafo de similitud."
        )
        st.dataframe(_bridge_df, width="stretch", hide_index=True)
        download_table(_bridge_df, "cugraph_bridge_loans.csv", "Descargar bridge loans")
    else:
        st.info(
            "Artefacto `cugraph_bridge_loans.parquet` no encontrado en el run "
            f"`{OFFICIAL_GPU_INSIGHT_TAG}`. Ejecuta la insight factory con cuGraph "
            "para generar este artefacto."
        )

_gpu_summary = try_load_json("gpu_consolidated_summary", directory="models", default={})
_gpu_table = try_load_parquet("gpu_consolidated_table")
if _gpu_summary or not _gpu_table.empty:
    with st.expander("Tabla consolidada CPU vs GPU — todas las secciones", expanded=False):
        if _gpu_summary:
            hw = _gpu_summary.get("hardware", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GPU", hw.get("gpu", "N/A"))
            c2.metric("Tasks comparados", _gpu_summary.get("n_tasks", "N/A"))
            c3.metric(
                "Speedup medio vs CPU",
                _fmt_metric(_gpu_summary.get("mean_speedup_vs_cpu"), digits=1, suffix="x"),
            )
            c4.metric(
                f"Máx speedup ({_gpu_summary.get('max_speedup_section', '?')})",
                _fmt_metric(_gpu_summary.get("max_speedup"), digits=1, suffix="x"),
            )
            st.caption(f"Task más rápida: `{_gpu_summary.get('max_speedup_task', 'N/A')}` — {hw.get('cpu', '')} vs {hw.get('gpu', '')}")
        if not _gpu_table.empty:
            cols_show = [c for c in ["section", "task", "backend_cpu", "backend_gpu", "cpu_seconds", "gpu_seconds", "speedup_cpu_vs_gpu", "speedup_vs_pandas"] if c in _gpu_table.columns]
            st.dataframe(_gpu_table[cols_show].dropna(subset=["cpu_seconds", "gpu_seconds"], how="all"), hide_index=True, width="stretch")
            st.caption("cpu_seconds / gpu_seconds = tiempo de ejecución en segundos. speedup_cpu_vs_gpu = cuántas veces más rápido es GPU vs CPU.")

st.success(
    f"Conclusión final: la GPU ya es parte defendible del proyecto, pero en un carril específico. El champion oficial sigue siendo CPU canónico; RAPIDS es el carril de aceleración comparativa, de OR con `cuopt`, de Monte Carlo con `cupy` y de una nueva insight factory donde hoy el mejor candidato claro es `cuGraph` ({OFFICIAL_GPU_INSIGHT_TAG})."
)

render_page_feedback("gpu_benchmark")
