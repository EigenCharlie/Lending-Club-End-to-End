"""Gobernanza de modelo: drift, fairness, robustez y contrato."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import try_load_json, try_load_parquet


def _artifact_health_rows() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    specs = [
        ("data/processed/pipeline_summary.json", "Resumen pipeline", "required"),
        ("models/conformal_results_mondrian.pkl", "Conformal canónico (resultados)", "required"),
        (
            "data/processed/conformal_intervals_mondrian.parquet",
            "Conformal canónico (intervalos)",
            "required",
        ),
        ("models/conformal_policy_status.json", "Estado de política conformal", "required"),
        (
            "data/processed/portfolio_robustness_summary.parquet",
            "Resumen robustez portafolio",
            "required",
        ),
        ("data/processed/ifrs9_scenario_summary.parquet", "Escenarios IFRS9", "required"),
        (
            "data/processed/conformal_intervals.parquet",
            "Conformal legacy (compatibilidad)",
            "legacy",
        ),
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
        "Empieza por estado global y checks aprobados.",
        "Revisa drift/fairness para detectar riesgos antes de que impacten negocio.",
        "Cierra con robustez y contrato de inputs para validar operación diaria.",
    ],
)
st.markdown(
    """
Esta página representa la capa de control del proyecto. No evalúa "qué tan bonito sale el dashboard", sino si el sistema
es suficientemente confiable para sostener decisiones repetibles: estabilidad poblacional, robustez frente a ruido, sesgos
por subgrupo y cumplimiento de contrato de inputs. En una arquitectura seria de riesgo, esta etapa es tan importante como
la métrica de desempeño del modelo.
"""
)

summary = try_load_json("pipeline_summary")
status = try_load_json("modeva_governance_status", directory="models", default={})
if not status:
    status = try_load_json("conformal_policy_status", directory="models", default={})

checks = try_load_parquet("modeva_governance_checks")
if checks.empty:
    checks = try_load_parquet("conformal_policy_checks")

metrics = try_load_parquet("modeva_governance_metrics")
drift_psi = try_load_parquet("modeva_governance_drift_psi")
drift_ks = try_load_parquet("modeva_governance_drift_ks")
fairness = try_load_parquet("modeva_governance_fairness")
robustness = try_load_parquet("modeva_governance_robustness")
slicing_acc = try_load_parquet("modeva_governance_slicing_accuracy")
slicing_rob = try_load_parquet("modeva_governance_slicing_robustness")
contract_val = try_load_parquet("pd_model_contract_validation")

passed = int(checks["passed"].sum()) if "passed" in checks.columns else 0
total = int(len(checks))
artifact_health = _artifact_health_rows()
required_health = artifact_health[artifact_health["Estado"].isin(["OK", "Falta"])]
required_ok = int((required_health["Estado"] == "OK").sum())
required_total = int(len(required_health))
missing_required = required_total - required_ok
metrics_row = metrics.iloc[0] if not metrics.empty else {}
pd_metrics = summary.get("pd_model", {})
test_auc = float(metrics_row.get("test_auc", pd_metrics.get("final_auc", 0.0)))
auc_gap = float(metrics_row.get("auc_gap_test_minus_train", 0.0))
fairness_air = float(metrics_row.get("fairness_air", 0.0))
max_drift_psi = float(metrics_row.get("max_drift_psi", 0.0))

kpi_row(
    [
        {
            "label": "Estado global",
            "value": "OK" if status.get("overall_pass", False) else "Revisión",
        },
        {"label": "Checks aprobados", "value": f"{passed}/{total}"},
        {"label": "AUC test", "value": f"{test_auc:.4f}"},
        {"label": "Gap train-test", "value": f"{auc_gap:.4f}"},
        {"label": "Fairness AIR", "value": f"{fairness_air:.3f}"},
        {"label": "Max PSI", "value": f"{max_drift_psi:.3f}"},
    ],
    n_cols=3,
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
    st.dataframe(artifact_health, use_container_width=True, hide_index=True)

st.subheader("1) Resultado de reglas de gobernanza")
if checks.empty:
    st.info("No hay tabla de checks de gobernanza disponible en este entorno.")
else:
    st.dataframe(checks, use_container_width=True, hide_index=True)

st.subheader("2) Drift de variables")
if drift_psi.empty or drift_ks.empty:
    st.info("No hay artefactos de drift PSI/KS; se omite esta sección.")
else:
    drift_psi_plot = drift_psi.copy()
    drift_psi_plot = drift_psi_plot.rename(columns={drift_psi_plot.columns[0]: "psi"})
    drift_psi_plot["feature_rank"] = range(1, len(drift_psi_plot) + 1)

    drift_ks_plot = drift_ks.copy()
    drift_ks_plot = drift_ks_plot.rename(columns={drift_ks_plot.columns[0]: "ks"})
    drift_ks_plot["feature_rank"] = range(1, len(drift_ks_plot) + 1)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            drift_psi_plot.sort_values("psi", ascending=False),
            x="feature_rank",
            y="psi",
            title="PSI por ranking de variable",
            labels={"feature_rank": "Índice de variable", "psi": "PSI"},
        )
        fig.add_hline(y=0.10, line_dash="dash", line_color="#FFD93D", annotation_text="Warning")
        fig.add_hline(y=0.25, line_dash="dash", line_color="#FF6B6B", annotation_text="Crítico")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=370)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: detectar drift poblacional por feature. Insight: PSI por encima de 0.10 activa vigilancia y por encima de 0.25 "
            "sugiere recalibración o revisión del pipeline."
        )

    with col2:
        fig = px.bar(
            drift_ks_plot.sort_values("ks", ascending=False),
            x="feature_rank",
            y="ks",
            title="KS drift por ranking de variable",
            labels={"feature_rank": "Índice de variable", "ks": "KS"},
        )
        fig.add_hline(y=0.20, line_dash="dash", line_color="#FF6B6B", annotation_text="Umbral")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=370)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: contrastar cambio de distribución train vs test con KS. Insight: valores altos indican potencial degradación "
            "de performance por cambio de régimen."
        )

st.subheader("3) Fairness y robustez")
col3, col4 = st.columns(2)
with col3:
    if fairness.empty:
        st.metric("AIR observado", "N/D")
        st.caption("No hay artefacto de fairness en este entorno.")
    else:
        fair_col = fairness.columns[0]
        fair_val = float(fairness.iloc[0][fair_col])
        st.metric("AIR observado", f"{fair_val:.3f}", help=f"Métrica: {fair_col}")
        st.caption("Referencia común: AIR cercano a 1 indica menor disparidad relativa.")

with col4:
    if robustness.empty:
        st.info("No hay artefacto de robustez a ruido; se omite boxplot.")
    else:
        robust_long = robustness.melt(var_name="ruido", value_name="auc")
        robust_long["ruido"] = robust_long["ruido"].astype(float)
        fig = px.box(
            robust_long,
            x="ruido",
            y="auc",
            points="all",
            title="Sensibilidad del AUC a ruido en features",
            labels={"ruido": "Nivel de ruido", "auc": "AUC"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propósito: medir estabilidad del AUC ante ruido sintético. Insight: caída abrupta del AUC a bajos niveles de ruido "
            "señala fragilidad del modelo."
        )

st.subheader("4) Slicing analysis (subpoblaciones débiles)")
weak_acc = (
    slicing_acc[slicing_acc["Weak"] == True].copy()  # noqa: E712
    if not slicing_acc.empty and "Weak" in slicing_acc.columns
    else pd.DataFrame()
)
weak_rob = (
    slicing_rob[slicing_rob["Weak"] == True].copy()  # noqa: E712
    if not slicing_rob.empty and "Weak" in slicing_rob.columns
    else pd.DataFrame()
)

if not slicing_acc.empty and not weak_acc.empty:
    fig = px.scatter(
        weak_acc,
        x="Size",
        y="AUC",
        color="Feature1",
        hover_data=["Segment1", "Feature2", "Segment2"],
        title="Slices débiles en accuracy",
        labels={"Size": "Tamaño del slice", "AUC": "AUC"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: ubicar subpoblaciones donde falla la discriminación. Insight: slices pequeños con AUC bajo pueden ocultarse "
        "en métricas globales y requieren controles específicos."
    )

if not slicing_rob.empty and not weak_rob.empty:
    fig = px.scatter(
        weak_rob,
        x="Size",
        y="AUC",
        color="Feature1",
        hover_data=["Segment1", "Feature2", "Segment2"],
        title="Slices débiles en robustez",
        labels={"Size": "Tamaño del slice", "AUC": "AUC"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: identificar segmentos sensibles a perturbaciones. Insight: robustez desigual por subgrupo implica riesgo "
        "operativo/fairness incluso con buen AUC agregado."
    )

st.subheader("5) Contrato de modelo y validación de inputs")
if contract_val.empty:
    st.info("No hay tabla de validación del contrato (`pd_model_contract_validation.parquet`).")
else:
    st.dataframe(contract_val, use_container_width=True, hide_index=True)
with st.expander("Contrato completo del modelo (JSON)"):
    contract = try_load_json("pd_model_contract", directory="models", default={})
    st.json(contract)

st.subheader("6) Auditoría de equidad multi-atributo (Fairness)")
fairness_audit = try_load_parquet("fairness_audit")
fairness_status = try_load_json("fairness_audit_status", directory="models", default={})
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
    st.dataframe(fairness_audit, use_container_width=True, hide_index=True)
    st.caption(
        "DPD = Demographic Parity Difference, DIR = Disparate Impact Ratio (4/5ths rule), "
        "EO = Equalized Odds gap. Umbrales configurables en `configs/fairness_policy.yaml`."
    )
else:
    st.info("Ejecuta `scripts/run_fairness_audit.py` para generar métricas de equidad multi-atributo.")

st.subheader("7) Marco regulatorio SR 11-7 (MRM)")
mrm_path = Path(__file__).resolve().parents[2] / "reports" / "mrm" / "mrm_validation_report.json"
if mrm_path.exists():
    import json

    mrm_report = json.loads(mrm_path.read_text())
    compliance = mrm_report.get("compliance_summary", {})
    kpi_row(
        [
            {"label": "Cumplimiento global", "value": "PASS" if compliance.get("overall_pass") else "REVISAR"},
            {"label": "Subsistemas OK", "value": f"{compliance.get('n_passing', 0)}/{compliance.get('n_subsystems', 0)}"},
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

st.markdown(
    """
**Lectura de control interno:**
- El marco detecta oportunidades de mejora en fairness y drift aunque el rendimiento base sea estable.
- La trazabilidad por checks + contrato facilita auditoría técnica del pipeline.
- La auditoría de equidad y el marco SR 11-7 elevan la gobernanza a estándar regulatorio.
"""
)

next_page_teaser(
    "Stack Tecnológico",
    "Librerías, versiones, decisiones de diseño y prácticas de ingeniería.",
    "pages/tech_stack.py",
)
