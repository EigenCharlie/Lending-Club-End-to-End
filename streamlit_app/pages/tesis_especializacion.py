"""Página de tesis de especialización alineada con evidencia ejecutable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_app.components.story_shell import render_key_takeaway
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.utils import (
    format_pct,
    load_json,
    try_load_json,
    try_load_parquet,
)

st.title("Tesis de Especialización")
st.caption(
    "Conformal Prediction para riesgo crediticio: síntesis académica basada en artefactos "
    "ejecutables del pipeline."
)
page_contract = get_page_contract("tesis_especializacion")
render_key_takeaway(
    "Esta página prioriza evidencia verificable de la corrida actual. Los temas de investigación "
    "avanzada (fairness conformal condicional, online CP) se profundizan en páginas de maestría."
)

comparison = try_load_json("model_comparison", default={})
conformal_status = try_load_json("conformal_policy_status", directory="models", default={})
fairness_status = try_load_json("fairness_audit_status", directory="models", default={})
lgd_ead_conformal_status = try_load_json("conformal_lgd_ead_status", directory="models", default={})
pipeline_summary = try_load_json("pipeline_summary", default={})
ifrs9_summary = try_load_parquet("ifrs9_scenario_summary", default=pd.DataFrame())

final_metrics = comparison.get("final_test_metrics", {}) if isinstance(comparison, dict) else {}
cal_report = comparison.get("calibration_selection_report", {}) if isinstance(comparison, dict) else {}
best_calibration = str(comparison.get("best_calibration", "n/a")) if isinstance(comparison, dict) else "n/a"


def _safe_page_link(page_path: str, label: str, icon: str) -> None:
    try:
        st.page_link(page_path, label=label, icon=icon)
    except Exception:
        st.caption(f"{icon} {label} -> {page_path}")


def _status_badge(passed: bool) -> str:
    return "PASS" if passed else "WARN/FAIL"


tab_context, tab_results, tab_fairness, tab_roadmap = st.tabs(
    [
        "Propuesta y Alcance",
        "Resultados Ejecutables",
        "Fairness y Gobernanza",
        "Ruta de Investigación",
    ]
)

with tab_context:
    st.subheader("Contexto y pregunta de investigación")
    st.markdown(
        """
**Pregunta guía**:
¿Cómo mejorar calibración y cuantificación de incertidumbre en PD/LGD/EAD con Conformal Prediction,
y cómo se compara frente a calibración tradicional en un caso reproducible?

**Enfoque adoptado en esta versión**:
- PD con CatBoost + calibración seleccionada por validación temporal.
- Intervalos conformales para PD (Mondrian/split según artefacto vigente).
- Integración a decisiones de portafolio e IFRS9.
- LGD/EAD conformal como evidencia técnica cuando los artefactos están disponibles.
"""
    )

    st.subheader("Mapeo de objetivos de propuesta -> evidencia actual")
    rows = [
        {
            "Objetivo": "Revisión y framing CP en riesgo crediticio",
            "Evidencia ejecutable": "Páginas thesis_end_to_end y research_landscape",
            "Estado": "Activo",
        },
        {
            "Objetivo": "Construcción dataset y pipeline reproducible",
            "Evidencia ejecutable": "data/processed/* + scripts/run_long_pipeline.py",
            "Estado": "Activo",
        },
        {
            "Objetivo": "PD + calibración + CP",
            "Evidencia ejecutable": "model_comparison.json + conformal_policy_status.json",
            "Estado": _status_badge(bool(conformal_status)),
        },
        {
            "Objetivo": "CP en LGD/EAD",
            "Evidencia ejecutable": "models/conformal_lgd_ead_status.json",
            "Estado": _status_badge(bool(lgd_ead_conformal_status.get("lgd", {}).get("available", False))),
        },
        {
            "Objetivo": "Comparación y lectura regulatoria (IFRS9/Basilea)",
            "Evidencia ejecutable": "ifrs9_scenario_summary.parquet + policy checks",
            "Estado": "Activo",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tab_results:
    st.subheader("KPIs de la corrida actual")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AUC OOT", f"{float(final_metrics.get('auc_roc', 0.0)):.4f}")
    k2.metric("ECE", f"{float(final_metrics.get('ece', 0.0)):.4f}")
    k3.metric("Cobertura 90%", format_pct(float(conformal_status.get("coverage_90", 0.0))))
    k4.metric("Conformal gate", _status_badge(bool(conformal_status.get("overall_pass", False))))

    st.caption(
        "Nota: `conformal_policy_status.overall_pass` corresponde a política estricta; "
        "la promoción oficial se evalúa en `run_comparison.py` con reglas de negocio/gobernanza."
    )

    st.subheader("PD calibración")
    st.write(
        {
            "best_calibration": best_calibration,
            "selection_reason": cal_report.get("selection_reason"),
            "auc_drop_limit": cal_report.get("auc_drop_limit"),
        }
    )

    st.subheader("LGD/EAD conformal (estado real)")
    lgd_status = lgd_ead_conformal_status.get("lgd", {}) if isinstance(lgd_ead_conformal_status, dict) else {}
    ead_status = lgd_ead_conformal_status.get("ead", {}) if isinstance(lgd_ead_conformal_status, dict) else {}
    status_rows = [
        {
            "Componente": "LGD",
            "Disponible": bool(lgd_status.get("available", False)),
            "Artefacto": str(lgd_status.get("intervals_path", "n/a")),
            "Detalle": lgd_status.get("reason", "ok") if not lgd_status.get("available", False) else "ok",
        },
        {
            "Componente": "EAD",
            "Disponible": bool(ead_status.get("available", False)),
            "Artefacto": str(ead_status.get("intervals_path", "n/a")),
            "Detalle": ead_status.get("reason", "ok") if not ead_status.get("available", False) else "ok",
        },
    ]
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")

    if not ifrs9_summary.empty:
        st.subheader("IFRS9 escenarios (extracto)")
        st.dataframe(ifrs9_summary.head(10), hide_index=True, width="stretch")
    else:
        st.info("No hay `ifrs9_scenario_summary.parquet` en esta corrida.")

with tab_fairness:
    st.subheader("Fairness en especialización: alcance secundario")
    st.markdown(
        """
En esta versión de especialización, fairness conformal se conserva como **diagnóstico de gobernanza**
y no como claim central de contribución metodológica.
"""
    )

    st.write(
        {
            "overall_pass": bool(fairness_status.get("overall_pass", False)),
            "n_passed": int(fairness_status.get("n_passed", 0)),
            "n_attributes": int(fairness_status.get("n_attributes", 0)),
            "prediction_threshold": fairness_status.get("prediction_threshold"),
            "run_tag": fairness_status.get("run_tag"),
        }
    )
    attrs = fairness_status.get("attributes", [])
    if isinstance(attrs, list) and attrs:
        st.dataframe(pd.DataFrame(attrs), hide_index=True, width="stretch")
    else:
        st.info("No hay detalle de auditoría fairness para esta corrida.")

with tab_roadmap:
    st.subheader("Distribución de profundidad académica")
    st.markdown(
        """
- **Especialización (esta página)**: evidencia ejecutable, coherencia con propuesta, decisión técnica defendible.
- **End-to-end / contribución**: integración metodológica completa, trade-offs operativos.
- **Research landscape (maestría)**: fairness conformal condicional, adaptive/online CP, agenda de publicaciones.
"""
    )

    _safe_page_link("pages/thesis_end_to_end.py", "Ir a Thesis End-to-End", "🧭")
    _safe_page_link("pages/thesis_contribution.py", "Ir a Contribución de Tesis", "🎯")
    _safe_page_link("pages/research_landscape.py", "Ir a Panorama de Investigación", "🔬")

st.divider()
st.caption(
    f"Contrato narrativo: {page_contract.page_id} | "
    f"Último run_tag detectado: {conformal_status.get('run_tag', 'n/a')}"
)

# Guardrail visual for missing canonical outputs
missing = []
for rel in [
    Path("data/processed/model_comparison.json"),
    Path("models/conformal_policy_status.json"),
    Path("models/fairness_audit_status.json"),
]:
    if not rel.exists():
        missing.append(str(rel))
if missing:
    st.warning("Faltan artefactos canónicos: " + ", ".join(missing))
