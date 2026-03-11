"""Gobernanza de modelo: conformal policy, fairness, contrato y MRM."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pandas as pd
import streamlit as st

from streamlit_app.components.context_help import methodology_dialog, term_popover
from streamlit_app.components.dvc_kpi_spine import render_global_kpi_spine
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_decision_box,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.utils import try_load_json, try_load_parquet, page_error_boundary


def _artifact_health_rows() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    specs = [
        ("data/processed/pipeline_summary.json", "Resumen pipeline", "required"),
        ("data/processed/explainability_global.parquet", "Bundle explicabilidad global", "required"),
        ("data/processed/explainability_local_cases.parquet", "Casos locales explicados", "required"),
        ("data/processed/ale_curves.parquet", "Curvas ALE", "required"),
        (
            "data/processed/shap_interactions_or_redundancy.parquet",
            "Interacciones/redundancia SHAP",
            "required",
        ),
        ("data/processed/explanation_drift.parquet", "Drift de explicaciones", "required"),
        (
            "data/processed/fairness_threshold_frontier.parquet",
            "Fairness por umbral",
            "required",
        ),
        ("models/conformal_results_mondrian.pkl", "Conformal canónico (resultados)", "required"),
        (
            "data/processed/conformal_intervals_mondrian.parquet",
            "Conformal canónico (intervalos)",
            "required",
        ),
        ("models/conformal_policy_status.json", "Resumen conformal operativo", "required"),
        ("models/challenger_promotion_report.json", "Promoción challenger", "required"),
        (
            "data/processed/portfolio_robustness_summary.parquet",
            "Resumen robustez portafolio",
            "required",
        ),
        ("data/processed/ifrs9_scenario_summary.parquet", "Escenarios IFRS9", "required"),
    ]
    rows: list[dict[str, str]] = []
    for rel_path, label, level in specs:
        abs_path = project_root / rel_path
        exists = abs_path.exists()
        if exists:
            stat = abs_path.stat()
            modified = pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            size_mb = f"{stat.st_size / (1024 * 1024):.2f} MB"
        else:
            modified = "-"
            size_mb = "-"

        if level == "legacy":
            status = "Compatibilidad" if exists else "No usado"
        else:
            status = "OK" if exists else "Falta"

        rows.append(
            {
                "Artefacto": label,
                "Ruta": rel_path,
                "Estado": status,
                "Actualizado": modified,
                "Tamaño": size_mb,
            }
        )

    return pd.DataFrame(rows)


st.title("🛡️ Gobernanza del Modelo")
st.caption(
    "Validación integral de confiabilidad para riesgo de crédito: performance, "
    "estabilidad, sesgo y robustez operativa."
)
page_contract = get_page_contract("model_governance")
render_page_header(page_contract)
render_key_takeaway(
    "Gobernanza aquí significa decidir si el sistema completo sigue siendo operable, no solo si una métrica de performance se ve bien."
)
term_popover("canónico", label="Artefactos canónicos y compatibilidad")
storytelling_intro(
    page_goal=(
        "Verificar si el sistema de riesgo sigue siendo confiable para operar en producción."
    ),
    business_value=(
        "Evita pérdidas por drift, sesgos o degradación silenciosa que no se ven en una sola métrica de accuracy."
    ),
    key_decision=(
        "Decidir si el modelo puede seguir en operación, requiere recalibración o necesita intervención inmediata."
    ),
    how_to_read=[
        "Empieza por estado global y resultado de controles.",
        "Revisa fairness para detectar riesgos antes de que impacten negocio.",
        "Cierra con contrato de inputs y marco MRM para validar operación diaria.",
    ],
)
render_decision_box(
    "Usar semáforo de gobernanza + contratos + fairness + MRM para decidir operar, recalibrar o bloquear.",
    owner="Model Risk / MRM",
    cadence="batch diario + revisión mensual",
)
render_global_kpi_spine("governance")
methodology_dialog(
    "Qué cubre esta página de gobernanza",
    """
La página combina cuatro capas de control:

1. Existencia y frescura de artefactos canónicos.
2. Reglas de política conformal (cobertura/ancho/alertas).
3. Contrato de modelo e inputs.
4. Fairness + marco MRM (SR 11-7).

La decisión de operación debe leer todas las capas juntas.
""",
    button_label="Ver alcance del control de gobernanza",
)
st.markdown(
    """
Esta página representa la capa de control del proyecto. No evalúa "qué tan bonito sale el dashboard", sino si el sistema
es suficientemente confiable para sostener decisiones repetibles: cobertura conformal, sesgos
por subgrupo y cumplimiento de contrato de inputs. En una arquitectura seria de riesgo, esta etapa es tan importante como
la métrica de desempeño del modelo.
"""
)

summary = try_load_json("pipeline_summary")
governance = try_load_json("governance_status", directory="models", default={})
conformal_status = try_load_json("conformal_policy_status", directory="models", default={})
checks = try_load_parquet("conformal_policy_checks")
contract_val = try_load_parquet("pd_model_contract_validation")
fairness_audit = try_load_parquet("fairness_audit")
fairness_status = try_load_json("fairness_audit_status", directory="models", default={})
fairness_frontier = try_load_parquet("fairness_threshold_frontier")
explanation_drift = try_load_parquet("explanation_drift")
challenger_report = try_load_json("challenger_promotion_report", directory="models", default={})

passed = int(checks["passed"].sum()) if "passed" in checks.columns else 0
total = int(len(checks))
artifact_health = _artifact_health_rows()
required_health = artifact_health[artifact_health["Estado"].isin(["OK", "Falta"])]
required_ok = int((required_health["Estado"] == "OK").sum())
required_total = int(len(required_health))
missing_required = required_total - required_ok
pd_metrics = summary.get("pd_model", {})
test_auc = float(pd_metrics.get("final_auc", 0.0))

kpi_row(
    [
        {
            "label": "Estado global",
            "value": "OK" if governance.get("overall_pass", False) else "Revisión",
        },
        {
            "label": "Drift explicativo",
            "value": "PASS" if governance.get("explanation_drift_pass", False) else "REVISAR",
        },
        {
            "label": "Fairness",
            "value": "PASS" if fairness_status.get("overall_pass", False) else "REVISAR",
        },
        {
            "label": "Challenger",
            "value": "Promovible"
            if challenger_report.get("challenger_promotable", False)
            else "Benchmark",
        },
        {"label": "Artefactos OK", "value": f"{required_ok}/{required_total}"},
    ],
    n_cols=5,
)

st.subheader("0) Salud y detalle de artefactos")
if missing_required == 0:
    st.success(f"Artefactos críticos disponibles: {required_ok}/{required_total}.")
else:
    st.warning(f"Artefactos críticos faltantes: {missing_required} de {required_total}.")

with st.expander("Ver detalle de artefactos y rutas canónicas"):
    st.caption(
        "Fuente canónica conformal para storytelling: "
        "`models/conformal_results_mondrian.pkl` + `data/processed/conformal_intervals_mondrian.parquet`."
    )
    st.dataframe(artifact_health, width="stretch", hide_index=True)

st.subheader("1) Resultado de reglas de gobernanza")
if governance:
    kpi_row(
        [
            {
                "label": "Drift predictivo",
                "value": "PASS"
                if (governance.get("checks", {}) or {}).get("pass_predictive_drift", False)
                else "FAIL",
            },
            {
                "label": "Fairness primario",
                "value": "PASS"
                if (governance.get("checks", {}) or {}).get("pass_fairness", False)
                else "FAIL",
            },
            {
                "label": "Reason codes",
                "value": "PASS"
                if governance.get("reason_code_stability_pass", False)
                else "FAIL",
            },
            {"label": "Threshold principal", "value": f"{float(governance.get('primary_threshold', 0.5)):.2f}"},
        ],
        n_cols=4,
    )
    with st.expander("Ver resumen ejecutivo de gobernanza"):
        st.json(governance)

if checks.empty:
    st.info("No hay tabla de checks conformal disponible en este entorno.")
else:
    st.dataframe(checks, width="stretch", hide_index=True)

st.subheader("2) Contrato de modelo y validación de inputs")
if contract_val.empty:
    st.info("No hay tabla de validación del contrato (`pd_model_contract_validation.parquet`).")
else:
    st.dataframe(contract_val, width="stretch", hide_index=True)
with st.expander("Contrato completo del modelo (JSON)"):
    contract = try_load_json("pd_model_contract", directory="models", default={})
    st.json(contract)

st.subheader("3) Estabilidad de explicaciones")
if explanation_drift.empty:
    st.info("No hay `explanation_drift.parquet` disponible; la gobernanza explicativa queda incompleta.")
else:
    kpi_row(
        [
            {
                "label": "Segmentos evaluados",
                "value": str(len(explanation_drift)),
            },
            {
                "label": "Min overlap top-10",
                "value": f"{explanation_drift['rank_overlap_top10'].min():.3f}",
            },
            {
                "label": "Max SHAP PSI",
                "value": f"{explanation_drift['max_shap_psi_top5'].max():.3f}",
            },
            {
                "label": "Min estabilidad de razones",
                "value": f"{explanation_drift['reason_code_match_rate'].min():.3f}",
            },
        ],
        n_cols=4,
    )
    st.dataframe(explanation_drift, width="stretch", hide_index=True)
    st.caption(
        "Esta tabla responde 'que cambio en la explicacion del modelo' y no solo 'que cambio en la distribucion de inputs'."
    )

st.subheader("4) Auditoría de equidad multi-atributo (Fairness)")
if not fairness_audit.empty:
    n_pass = int(fairness_audit["passed_all"].sum())
    n_attr = len(fairness_audit)
    kpi_row(
        [
            {"label": "Atributos evaluados", "value": str(n_attr)},
            {"label": "Pasan todos", "value": f"{n_pass}/{n_attr}"},
            {"label": "Max DPD", "value": f"{fairness_audit['dpd'].max():.3f}"},
            {"label": "Min DIR", "value": f"{fairness_audit['dir'].min():.3f}"},
        ],
        n_cols=4,
    )
    st.dataframe(fairness_audit, width="stretch", hide_index=True)
    st.caption(
        "DPD = Demographic Parity Difference, DIR = Disparate Impact Ratio (4/5ths rule), "
        "EO = Equalized Odds gap. Umbrales configurables en `configs/fairness_policy.yaml`."
    )
    if not fairness_frontier.empty:
        frontier_attr = st.selectbox(
            "Atributo para frontera por umbral",
            options=sorted(fairness_frontier["attribute"].astype(str).unique().tolist()),
            index=0,
            key="governance_fairness_frontier_attribute",
        )
        frontier_view = fairness_frontier[
            fairness_frontier["attribute"].astype(str) == str(frontier_attr)
        ].copy()
        frontier_view = frontier_view.sort_values("threshold")
        frontier_chart = frontier_view.set_index("threshold")[["dpd", "eo_gap"]]
        st.line_chart(frontier_chart)
        st.dataframe(frontier_view, width="stretch", hide_index=True)
else:
    st.info(
        "Ejecuta `scripts/run_fairness_audit.py` para generar métricas de equidad multi-atributo."
    )

st.subheader("5) Challenger monotónico y promoción")
if not challenger_report:
    st.info("No hay `models/challenger_promotion_report.json` disponible.")
else:
    interp = challenger_report.get("interpretability", {})
    deltas = challenger_report.get("deltas", {})
    checks_ch = challenger_report.get("promotion_checks", {})
    kpi_row(
        [
            {
                "label": "Promovible",
                "value": "SI" if challenger_report.get("challenger_promotable", False) else "NO",
            },
            {"label": "AUC drop", "value": f"{float(deltas.get('auc_drop', 0.0)):.4f}"},
            {
                "label": "Brier increase",
                "value": f"{float(deltas.get('brier_increase_pct', 0.0)) * 100:.2f}%",
            },
            {
                "label": "Ganancias interpretabilidad",
                "value": str(int(interp.get("gain_count", 0))),
            },
        ],
        n_cols=4,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Check": key,
                    "Passed": bool(value),
                }
                for key, value in checks_ch.items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Ver reporte challenger completo"):
        st.json(challenger_report)

st.subheader("6) Marco regulatorio SR 11-7 (MRM)")
mrm_path = Path(__file__).resolve().parents[2] / "reports" / "mrm" / "mrm_validation_report.json"
if mrm_path.exists():
    import json

    mrm_report = json.loads(mrm_path.read_text())
    compliance = mrm_report.get("compliance_summary", {})
    kpi_row(
        [
            {
                "label": "Cumplimiento global",
                "value": "PASS" if compliance.get("overall_pass") else "REVISAR",
            },
            {
                "label": "Subsistemas OK",
                "value": f"{compliance.get('n_passing', 0)}/{compliance.get('n_subsystems', 0)}",
            },
            {"label": "Modelo campeón", "value": mrm_report.get("model", {}).get("name", "N/D")},
        ],
        n_cols=3,
    )
    with st.expander("Ver reporte MRM completo"):
        st.json(mrm_report)
    st.caption(
        "Reporte generado por `scripts/generate_mrm_report.py`. "
        "Documento completo en `docs/MODEL_RISK_MANAGEMENT.md`."
    )
else:
    st.info("Ejecuta `scripts/generate_mrm_report.py` para generar el reporte MRM (SR 11-7).")

st.subheader("7) Drift formal (KS/C2ST) y criterio de escalamiento")

with page_error_boundary("model_governance"):
    drift_checks = try_load_parquet("drift_checks")
    if drift_checks.empty:
        drift_checks = pd.DataFrame(
            [
                {
                    "test": "KS two-sample (univariado)",
                    "target": "covariate_shift",
                    "trigger_warn": "p-value < 0.05 en features críticas",
                    "trigger_fail": "p-value < 0.01 sostenido por cohorte",
                    "accion": "WARN: monitorear y revisar mix; FAIL: recalibrar o bloquear rollout",
                },
                {
                    "test": "C2ST (multivariado)",
                    "target": "concept_drift/proxy",
                    "trigger_warn": "AUC C2ST > 0.60",
                    "trigger_fail": "AUC C2ST > 0.70",
                    "accion": "WARN: investigar origen; FAIL: activar comité MRM",
                },
            ]
        )
    st.dataframe(drift_checks, width="stretch", hide_index=True)
    st.markdown(
        """
    **Escalamiento operativo sugerido**
    - `MONITOR`: señales aisladas sin deterioro de métricas de negocio.
    - `RECALIBRAR`: alertas persistentes o caída sostenida de calibración/cobertura.
    - `BLOQUEAR`: drift severo + incumplimiento de policy checks críticos.
    """
    )
    st.markdown("**Micro-panel CP: ruptura de supuestos y respuesta**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Señal CP": "Cobertura 90/95 bajo target por >=2 cohortes",
                    "Lectura": "Probable ruptura operativa de exchangeability",
                    "Respuesta": "Recalibrar y revisar partición Mondrian.",
                },
                {
                    "Señal CP": "Subgrupo crítico con under-coverage recurrente",
                    "Lectura": "Promedio global deja de ser representativo",
                    "Respuesta": "Escalar a comité MRM y ajustar política segmentada.",
                },
                {
                    "Señal CP": "Drift KS/C2ST + alertas conformal severas",
                    "Lectura": "Riesgo de degradación de validez útil",
                    "Respuesta": "Considerar variante adaptativa y bloquear despliegue si persiste.",
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
    **Lectura de control interno:**
    - La gobernanza ya no depende solo de drift de features: incorpora drift de explicaciones, fairness al umbral principal y promoción formal del challenger.
    - La trazabilidad por checks + contrato facilita auditoría técnica del pipeline.
    - El framework conformal mejora el control de cobertura por partición (Mondrian) bajo monitoreo.
    """
    )
    render_caveats(
        [
            "Un estado global 'OK' no elimina la necesidad de monitoreo continuo por lote/segmento.",
            "Fairness y MRM dependen de umbrales de política; cambios de threshold cambian el veredicto.",
            "La ausencia de artefactos puede ser un problema operativo aunque las métricas históricas sean buenas.",
        ]
    )
    render_page_feedback("model_governance")

    next_page_teaser(
        "Stack Tecnológico",
        "Librerías, versiones, decisiones de diseño y prácticas de ingeniería.",
        "pages/tech_stack.py",
    )
