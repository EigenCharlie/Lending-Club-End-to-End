"""Reusable applied conformal blocks for Streamlit narrative pages."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st


def build_cp_concept_matrix_rows() -> list[dict[str, str]]:
    """Return a canonical applied map: concept -> evidence -> decision."""
    return [
        {
            "Concepto aplicado": "Validez marginal finita",
            "Evidencia en artefactos": "`coverage_90`, `coverage_95`",
            "Decision que habilita": "Definir si la politica de incertidumbre es defendible.",
        },
        {
            "Concepto aplicado": "Eficiencia (ancho)",
            "Evidencia en artefactos": "`avg_width_90`, `median_width_90`",
            "Decision que habilita": "Cuantificar costo economico de robustez.",
        },
        {
            "Concepto aplicado": "Cobertura por particion (Mondrian)",
            "Evidencia en artefactos": "`min_group_coverage_90`, `conformal_group_metrics_mondrian`",
            "Decision que habilita": "Evitar decisiones ciegas por promedio global.",
        },
        {
            "Concepto aplicado": "Estabilidad temporal",
            "Evidencia en artefactos": "`conformal_backtest_monthly`, `conformal_backtest_alerts`",
            "Decision que habilita": "Monitorear, recalibrar o escalar metodo.",
        },
    ]


def render_cp_guarantees_and_limits() -> None:
    """Render a concise what-is-guaranteed and what-is-not block."""
    st.markdown("#### Que esta garantizado y que no")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Dimension": "Si garantiza",
                    "Lectura operativa": (
                        "Cobertura marginal en muestra finita bajo exchangeability "
                        "(promedio poblacional al nivel objetivo)."
                    ),
                },
                {
                    "Dimension": "No garantiza",
                    "Lectura operativa": (
                        "Cobertura condicional exacta para todo x sin supuestos adicionales."
                    ),
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Detalle tecnico minimo (formulas)", expanded=False):
        st.latex(r"\Pr\{Y_{n+1} \in \hat{C}_{1-\alpha}(X_{n+1})\} \ge 1-\alpha")
        st.latex(r"k_{1-\alpha} = \lceil (n_{cal}+1)(1-\alpha) \rceil")
        st.caption(
            "Lectura: la garantia es marginal y no elimina la necesidad de monitoreo por "
            "subgrupo y por tiempo."
        )


def _wilson_bounds(success_rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score bounds for a proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = min(max(float(success_rate), 0.0), 1.0)
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    spread = (z / den) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))


def build_coverage_stability_table(
    frame: pd.DataFrame,
    *,
    label_col: str,
    coverage_col: str,
    n_col: str,
    target: float = 0.90,
) -> pd.DataFrame:
    """Build coverage stability diagnostics with Wilson intervals."""
    if frame.empty or not {label_col, coverage_col, n_col}.issubset(frame.columns):
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    for _, row in frame.iterrows():
        label = str(row[label_col])
        coverage = float(row[coverage_col])
        n_obs = int(row[n_col])
        ci_low, ci_high = _wilson_bounds(coverage, n_obs)
        rows.append(
            {
                "segmento": label,
                "n": n_obs,
                "coverage": coverage,
                "target": float(target),
                "gap_vs_target": coverage - float(target),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "half_width_95": max(0.0, (ci_high - ci_low) / 2.0),
                "small_n_flag": n_obs < 1000,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def build_cp_method_menu_rows(canonical_method: str = "Mondrian selected cfg") -> list[dict[str, str]]:
    """Return a compact method-choice menu aligned to applied usage."""
    return [
        {
            "Metodo": "Split global",
            "Que garantiza": "Cobertura marginal",
            "Cuando conviene": "Baseline simple y rapido",
            "Riesgo de mala lectura": "Oculta subcobertura por segmento",
            "Estado actual": "Benchmark",
        },
        {
            "Metodo": "Mondrian por particion",
            "Que garantiza": "Cobertura por estrato/particion",
            "Cuando conviene": "Segmentos de riesgo con perfil distinto",
            "Riesgo de mala lectura": "Fragmentacion en grupos pequenos",
            "Estado actual": "Canónico" if "mondrian" in canonical_method.lower() else "Alternativa",
        },
        {
            "Metodo": "Mondrian + score normalizado",
            "Que garantiza": "Misma validez, mejor adaptacion por heteroscedasticidad",
            "Cuando conviene": "Ancho desigual por nivel de riesgo",
            "Riesgo de mala lectura": "Mayor complejidad de diagnostico",
            "Estado actual": "Benchmark",
        },
        {
            "Metodo": "CQR",
            "Que garantiza": "Cobertura marginal con intervalos adaptativos",
            "Cuando conviene": "Regresion heteroscedastica (LGD/EAD)",
            "Riesgo de mala lectura": "Confundir mejor ancho con mejor calibracion",
            "Estado actual": "Exploratorio",
        },
        {
            "Metodo": "Jackknife+ / CV+",
            "Que garantiza": "Reuso de datos con inferencia robusta",
            "Cuando conviene": "Cuando duele separar calibracion fija",
            "Riesgo de mala lectura": "Costo computacional y setup mas exigente",
            "Estado actual": "Exploratorio",
        },
    ]


def render_exchangeability_stress_checklist() -> None:
    """Render operational signs and actions for exchangeability stress."""
    st.markdown("#### Exchangeability stress checklist")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Senal": "Drift persistente en covariables o regimen",
                    "Accion recomendada": "MONITOR -> recalibrar thresholds y revisar mix.",
                },
                {
                    "Senal": "Alertas recurrentes por grupo/mes",
                    "Accion recomendada": "RECALIBRAR y revisar particion Mondrian.",
                },
                {
                    "Senal": "Caida sostenida de cobertura bajo target",
                    "Accion recomendada": "Escalar a metodo adaptativo y comite MRM.",
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def build_cp_evidence_ladder_rows() -> list[dict[str, str]]:
    """Return evidence ladder rows for thesis-style narrative."""
    return [
        {
            "Principio CP": "Validez",
            "Artefacto": "`coverage_90`, `coverage_95`",
            "Lectura": "La promesa estadistica se verifica en datos OOT.",
        },
        {
            "Principio CP": "Eficiencia",
            "Artefacto": "`avg_width_90`, `median_width_90`",
            "Lectura": "El costo de robustez es cuantificable, no narrativo.",
        },
        {
            "Principio CP": "Robustez temporal",
            "Artefacto": "`conformal_backtest_monthly`",
            "Lectura": "La estabilidad se monitorea por cohorte mensual.",
        },
        {
            "Principio CP": "Cobertura por particion",
            "Artefacto": "`conformal_group_metrics_mondrian`",
            "Lectura": "Evita sobreconfiar en el promedio global.",
        },
    ]
