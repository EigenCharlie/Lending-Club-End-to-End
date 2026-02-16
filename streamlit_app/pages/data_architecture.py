"""Arquitectura y linaje de datos del proyecto."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import get_notebook_image_path, load_json, query_duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DBT_MANIFEST_PATH = PROJECT_ROOT / "dbt_project" / "target" / "manifest.json"
DBT_RUN_RESULTS_PATH = PROJECT_ROOT / "dbt_project" / "target" / "run_results.json"
FEAST_ONLINE_DB = PROJECT_ROOT / "data" / "feast_online.db"
FEATURE_VIEWS_FILE = PROJECT_ROOT / "feature_repo" / "feature_views.py"
FEATURE_SERVICES_FILE = PROJECT_ROOT / "feature_repo" / "feature_services.py"
DICTIONARY_PATH = PROJECT_ROOT / "docs" / "LCDataDictionary.xlsx"
DATASET_SHAPES_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_shapes_summary.json"


@st.cache_data(ttl=3600)
def load_dbt_metrics() -> dict:
    if not DBT_MANIFEST_PATH.exists():
        return {}

    manifest = json.loads(DBT_MANIFEST_PATH.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    models = [n for n in nodes.values() if n.get("resource_type") == "model"]
    tests = [n for n in nodes.values() if n.get("resource_type") == "test"]

    model_schema_counts: dict[str, int] = {}
    for node in models:
        schema = node.get("schema", "unknown")
        model_schema_counts[schema] = model_schema_counts.get(schema, 0) + 1

    run_status: dict[str, int] = {}
    if DBT_RUN_RESULTS_PATH.exists():
        run_results = json.loads(DBT_RUN_RESULTS_PATH.read_text(encoding="utf-8"))
        for result in run_results.get("results", []):
            status = result.get("status", "unknown")
            run_status[status] = run_status.get(status, 0) + 1

    return {
        "models_total": len(models),
        "tests_total": len(tests),
        "model_schema_counts": model_schema_counts,
        "run_status": run_status,
    }


@st.cache_data(ttl=3600)
def load_feast_metrics() -> dict:
    online_tables: list[str] = []
    if FEAST_ONLINE_DB.exists():
        conn = sqlite3.connect(str(FEAST_ONLINE_DB))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        online_tables = [row[0] for row in cur.fetchall()]
        conn.close()

    feature_views: list[str] = []
    if FEATURE_VIEWS_FILE.exists():
        text = FEATURE_VIEWS_FILE.read_text(encoding="utf-8")
        feature_views = re.findall(r'name="([^"]+)"', text)

    feature_services: list[str] = []
    if FEATURE_SERVICES_FILE.exists():
        text = FEATURE_SERVICES_FILE.read_text(encoding="utf-8")
        feature_services = re.findall(r'name="([^"]+)"', text)

    size_gb = FEAST_ONLINE_DB.stat().st_size / (1024**3) if FEAST_ONLINE_DB.exists() else 0.0
    return {
        "online_tables": online_tables,
        "feature_views": feature_views,
        "feature_services": feature_services,
        "online_size_gb": size_gb,
    }


@st.cache_data(ttl=3600)
def count_dictionary_variables() -> int:
    """Count variables from official XLSX dictionary without extra dependencies."""
    if not DICTIONARY_PATH.exists():
        return 0
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(DICTIONARY_PATH) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in rels.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        }
        sheet = wb.find("m:sheets", ns).find("m:sheet", ns)
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = "xl/" + rel_map[rid]

        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("m:si", ns):
                shared_strings.append("".join((t.text or "") for t in si.findall(".//m:t", ns)))

        ws = ET.fromstring(zf.read(target))
        values = []
        for row in ws.findall(".//m:sheetData/m:row", ns)[1:]:
            first_cell = row.find("m:c", ns)
            if first_cell is None:
                continue
            t = first_cell.attrib.get("t")
            v = first_cell.find("m:v", ns)
            if v is None:
                continue
            txt = v.text or ""
            if t == "s" and txt.isdigit() and int(txt) < len(shared_strings):
                txt = shared_strings[int(txt)]
            txt = txt.strip()
            if txt:
                values.append(txt)
        return len(values)


@st.cache_data(ttl=3600)
def load_dataset_shapes() -> pd.DataFrame:
    """Load dataset shape metadata without materializing full parquet files."""
    import pyarrow.parquet as pq

    summary: dict[str, dict[str, int | None]] = {}
    if DATASET_SHAPES_SUMMARY_PATH.exists():
        summary = json.loads(DATASET_SHAPES_SUMMARY_PATH.read_text(encoding="utf-8"))

    rows = []
    assets = [
        ("data/raw/Loan_status_2007-2020Q3.csv", "csv"),
        ("data/interim/lending_club_cleaned.parquet", "parquet"),
        ("data/processed/train.parquet", "parquet"),
        ("data/processed/calibration.parquet", "parquet"),
        ("data/processed/test.parquet", "parquet"),
        ("data/processed/train_fe.parquet", "parquet"),
        ("data/processed/loan_master.parquet", "parquet"),
        ("data/processed/time_series.parquet", "parquet"),
        ("data/processed/ead_dataset.parquet", "parquet"),
        ("data/processed/obt_loan_features.parquet", "parquet"),
    ]
    for rel_path, fmt in assets:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            try:
                if fmt == "csv":
                    cols = int(pd.read_csv(path, nrows=0, low_memory=False).shape[1])
                    fallback_rows = summary.get(rel_path, {}).get("rows")
                    rows.append({"dataset": rel_path, "rows": fallback_rows, "cols": cols})
                else:
                    metadata = pq.ParquetFile(path).metadata
                    rows.append(
                        {
                            "dataset": rel_path,
                            "rows": int(metadata.num_rows),
                            "cols": int(metadata.num_columns),
                        }
                    )
                continue
            except Exception:
                pass

        cached = summary.get(rel_path)
        if cached:
            rows.append(
                {
                    "dataset": rel_path,
                    "rows": cached.get("rows"),
                    "cols": int(cached.get("cols", 0)),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_duckdb_schema_counts() -> pd.DataFrame:
    return query_duckdb(
        """
        SELECT table_schema, count(*) AS n_tables
        FROM information_schema.tables
        GROUP BY 1
        ORDER BY 1
        """
    )


@st.cache_data(ttl=3600)
def load_duckdb_table_catalog() -> pd.DataFrame:
    return query_duckdb(
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        ORDER BY table_schema, table_name
        """
    )


def lineage_sankey(shapes: pd.DataFrame) -> go.Figure:
    col_map = {row["dataset"]: row["cols"] for _, row in shapes.iterrows()}
    labels = [
        f"Diccionario oficial ({count_dictionary_variables()} vars)",
        f"Raw CSV ({col_map.get('data/raw/Loan_status_2007-2020Q3.csv', 'N/D')} cols)",
        f"Interim cleaned ({col_map.get('data/interim/lending_club_cleaned.parquet', 'N/D')} cols)",
        f"Splits OOT train/cal/test ({col_map.get('data/processed/train.parquet', 'N/D')} cols)",
        f"Feature engineering ({col_map.get('data/processed/train_fe.parquet', 'N/D')} cols)",
        f"loan_master ({col_map.get('data/processed/loan_master.parquet', 'N/D')} cols)",
        f"time_series ({col_map.get('data/processed/time_series.parquet', 'N/D')} cols)",
        f"ead_dataset ({col_map.get('data/processed/ead_dataset.parquet', 'N/D')} cols)",
        f"obt_loan_features ({col_map.get('data/processed/obt_loan_features.parquet', 'N/D')} cols)",
        "DuckDB + dbt + Feast",
        "Streamlit / API / MCP",
    ]
    source = [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 10]
    value = [10, 10, 9, 8, 7, 5, 4, 4, 7, 3, 4, 5, 9]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": labels,
                    "pad": 18,
                    "thickness": 16,
                    "line": {"color": "#D9E2EC", "width": 1},
                    "color": [
                        "#7DA0FA",
                        "#4C78A8",
                        "#72B7B2",
                        "#54A24B",
                        "#ECA82C",
                        "#F58518",
                        "#EECA3B",
                        "#B279A2",
                        "#FF9DA6",
                        "#9D755D",
                        "#79706E",
                    ],
                },
                link={
                    "source": source,
                    "target": target,
                    "value": value,
                    "color": "rgba(11,94,215,0.18)",
                },
            )
        ]
    )
    layout = dict(PLOTLY_TEMPLATE["layout"])
    layout["font"] = {"color": "#000000", "family": "Arial, sans-serif"}
    fig.update_layout(
        **layout,
        title="Grafo de linaje de datos y transformación",
        height=520,
    )
    fig.update_traces(textfont={"size": 14, "color": "#000000"})
    return fig


def lineage_graphviz_dot(shapes: pd.DataFrame, dict_vars: int) -> str:
    col_map = {row["dataset"]: row["cols"] for _, row in shapes.iterrows()}
    return f"""
digraph G {{
    rankdir=LR;
    bgcolor="white";
    graph [pad="0.3", nodesep="0.55", ranksep="0.9", fontname="Arial", fontsize=16, label="Linaje de Datos: del Diccionario al Delivery", labelloc="t"];
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=13, color="#B8C7D9", fontcolor="#1F2937"];
    edge [color="#6B7A90", penwidth=1.6, arrowsize=0.8];

    dict [label="Diccionario oficial\\n({dict_vars} variables)", fillcolor="#EAF2FF"];
    raw [label="Raw CSV\\n({col_map.get('data/raw/Loan_status_2007-2020Q3.csv', 'N/D')} columnas)", fillcolor="#E7F3FF"];
    clean [label="Interim cleaned\\n({col_map.get('data/interim/lending_club_cleaned.parquet', 'N/D')} columnas)", fillcolor="#E8F8F1"];
    split [label="Split temporal OOT\\ntrain/cal/test", fillcolor="#F0F8E8"];
    fe [label="Feature engineering\\n({col_map.get('data/processed/train_fe.parquet', 'N/D')} columnas)", fillcolor="#FFF4E5"];
    loan [label="loan_master\\n({col_map.get('data/processed/loan_master.parquet', 'N/D')} cols)", fillcolor="#FFF8E1"];
    ts [label="time_series\\n({col_map.get('data/processed/time_series.parquet', 'N/D')} cols)", fillcolor="#FFF8E1"];
    ead [label="ead_dataset\\n({col_map.get('data/processed/ead_dataset.parquet', 'N/D')} cols)", fillcolor="#FFF8E1"];
    obt [label="obt_loan_features\\n({col_map.get('data/processed/obt_loan_features.parquet', 'N/D')} cols)", fillcolor="#FFF8E1"];
    plat [label="DuckDB + dbt + Feast", fillcolor="#F3E8FF"];
    app [label="Streamlit / API / MCP", fillcolor="#ECECEC"];

    dict -> raw [label="estandarización de campos"];
    raw -> clean [label="limpieza + tipos"];
    clean -> split [label="corte temporal"];
    split -> fe [label="transformaciones WOE/IV"];
    fe -> loan [label="vista riesgo"];
    fe -> ts [label="agregación mensual"];
    fe -> ead [label="dataset exposición"];
    fe -> obt [label="contrato feature store"];
    loan -> plat;
    ts -> plat;
    ead -> plat;
    obt -> plat;
    plat -> app;
}}
"""


st.title("🗂️ Arquitectura y Linaje de Datos")
st.caption(
    "Esta página prioriza el origen, la transformación y el propósito de cada dataset; "
    "la plataforma se entiende como consecuencia de una arquitectura de datos bien diseñada."
)

st.markdown(
    """
El proyecto parte del principio: **sin buena ingeniería y linaje de datos, no hay modelo confiable**.
Aquí se documenta de dónde sale cada variable, por qué se crean datasets separados y qué problema resuelve cada uno.
"""
)
st.markdown(
    """
El objetivo de esta página es demostrar que la arquitectura de datos no es un accesorio visual: es la condición para que
las capas de modelado, incertidumbre y optimización sean auditables. Aquí se explica por qué partimos de un diccionario
amplio de variables, cómo se depura hasta un `cleaned` reproducible y por qué después se separa en datasets con funciones
distintas (`loan_master`, `time_series`, `ead_dataset`, `obt_loan_features`), cada uno orientado a una decisión analítica
específica dentro del pipeline.
"""
)

shapes = load_dataset_shapes()
feature_iv = load_json("feature_importance_iv")
feast = load_feast_metrics()
dbt = load_dbt_metrics()

def _safe_cols(dataset_name: str) -> int:
    row = shapes.loc[shapes["dataset"] == dataset_name, "cols"]
    return int(row.iloc[0]) if not row.empty else 0


raw_cols = _safe_cols("data/raw/Loan_status_2007-2020Q3.csv")
clean_cols = _safe_cols("data/interim/lending_club_cleaned.parquet")
fe_cols = _safe_cols("data/processed/train_fe.parquet")
dict_vars = count_dictionary_variables()

kpi_row(
    [
        {"label": "Variables diccionario oficial", "value": str(dict_vars)},
        {"label": "Columnas raw CSV", "value": str(raw_cols)},
        {"label": "Columnas cleaned interim", "value": str(clean_cols)},
        {"label": "Columnas tras FE", "value": str(fe_cols)},
        {"label": "Features CatBoost", "value": str(feature_iv.get("n_features_total", 0))},
        {"label": "Features WOE", "value": str(feature_iv.get("n_woe_features", 0))},
    ],
    n_cols=3,
)

st.subheader("1) Grafo de linaje (DAG de datos)")
st.markdown("**Versión legible (Graphviz):**")
st.graphviz_chart(lineage_graphviz_dot(shapes, dict_vars), use_container_width=True)
st.caption(
    "Propósito: trazar origen y transformación de datos. Insight: el diseño desacopla limpieza, modelado y datasets "
    "especializados para reducir fuga y mejorar trazabilidad."
)

with st.expander("Ver Sankey de flujo (complementario)"):
    st.plotly_chart(lineage_sankey(shapes), use_container_width=True)
    st.caption("Sankey complementario para visualizar intensidad relativa del flujo entre capas.")

st.markdown(
    """
**Lectura del linaje:**
- Del diccionario oficial al raw CSV hay adaptación práctica de extracción histórica.
- `lending_club_cleaned.parquet` estandariza tipos, limpia campos y elimina ruido operacional.
- El split temporal OOT (`train/calibration/test`) evita fuga de información y simula despliegue real.
- A partir de ahí se especializan datasets por objetivo analítico (PD, IFRS9, forecast, EAD, feature-store).
"""
)

st.subheader("2) ¿Por qué se dividió en varios datasets?")
dataset_design = pd.DataFrame(
    [
        {
            "Dataset": "lending_club_cleaned",
            "Rol": "Base limpia unificada",
            "Problema que resuelve": "Inconsistencias de formato, nulos y columnas irrelevantes.",
            "Consumidor principal": "Todos los módulos aguas abajo",
        },
        {
            "Dataset": "train / calibration / test",
            "Rol": "Control temporal y calibración",
            "Problema que resuelve": "Validación realista y calibración sin leakage.",
            "Consumidor principal": "Modelado PD + Conformal",
        },
        {
            "Dataset": "train_fe / test_fe",
            "Rol": "Matriz de features modelables",
            "Problema que resuelve": "Transformar variables crudas en señales predictivas robustas.",
            "Consumidor principal": "CatBoost, logística y calibración",
        },
        {
            "Dataset": "loan_master",
            "Rol": "Vista compacta de riesgo de cartera",
            "Problema que resuelve": "Análisis EDA/segmentación sin columnas operativas de bajo valor.",
            "Consumidor principal": "Storytelling, survival, causal",
        },
        {
            "Dataset": "time_series",
            "Rol": "Serie mensual agregada",
            "Problema que resuelve": "Pronóstico de default y escenarios forward-looking.",
            "Consumidor principal": "Forecasting e IFRS9",
        },
        {
            "Dataset": "ead_dataset",
            "Rol": "Exposición para EAD/LGD",
            "Problema que resuelve": "Separar población con información de pérdida/exposición.",
            "Consumidor principal": "Cálculos IFRS9 y pérdida esperada",
        },
        {
            "Dataset": "obt_loan_features",
            "Rol": "Objeto tipo feature store",
            "Problema que resuelve": "Contrato consistente train/serve y consulta en Feast.",
            "Consumidor principal": "Feast + demostración plataforma",
        },
    ]
)
st.dataframe(dataset_design, use_container_width=True, hide_index=True)

st.subheader("3) Ingeniería de variables: WOE, IV y selección de features")
feature_sizes = {k: len(v) for k, v in feature_iv.get("feature_lists", {}).items() if isinstance(v, list)}
st.dataframe(
    pd.DataFrame(
        [{"familia": k, "n_variables": v} for k, v in feature_sizes.items()]
    ).sort_values("n_variables", ascending=False),
    use_container_width=True,
    hide_index=True,
)

iv_items = list(feature_iv.get("iv_scores", {}).items())[:20]
iv_df = pd.DataFrame(iv_items, columns=["feature", "iv"]).sort_values("iv", ascending=True)
fig = px.bar(
    iv_df,
    x="iv",
    y="feature",
    orientation="h",
    title="Ranking IV (top 20): poder predictivo de variables",
    labels={"iv": "Information Value (IV)", "feature": "Feature"},
    color="iv",
    color_continuous_scale="Blues",
)
fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=480, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: priorizar variables por poder predictivo (IV). Insight: sub_grade/int_rate/grade lideran señal; "
    "esto justifica su rol central en PD."
)

st.info(
    "Interpretación IV (regla práctica): <0.02 débil, 0.02-0.1 útil, 0.1-0.3 fuerte, >0.3 muy fuerte "
    "(revisar posible dependencia intensa con target y riesgo de sobreajuste semántico)."
)

col_a, col_b = st.columns(2)
with col_a:
    woe_img = get_notebook_image_path("02_feature_engineering", "cell_020_out_00.png")
    if woe_img.exists():
        st.image(str(woe_img), caption="Notebook 02: binning WOE en features de mayor IV", use_container_width=True)
with col_b:
    iv_img = get_notebook_image_path("02_feature_engineering", "cell_023_out_00.png")
    if iv_img.exists():
        st.image(str(iv_img), caption="Notebook 02: ranking IV y umbrales de interpretación", use_container_width=True)

st.markdown(
    """
**Por qué estas variables:**
- Se priorizan señales de capacidad de pago, calidad crediticia y carga financiera (`int_rate`, `fico`, `dti`, `installment_burden`).
- Se incluyen interacciones y buckets para capturar no linealidades con interpretación de riesgo.
- Se excluyen campos post-originación para mantener causalidad temporal del score.
"""
)

st.subheader("4) Plataforma de datos (DuckDB + dbt + Feast)")
duckdb_schema_count = 0
duckdb_table_count = 0
try:
    schema_df = load_duckdb_schema_counts()
    duckdb_schema_count = int(schema_df["table_schema"].nunique())
    duckdb_table_count = int(schema_df["n_tables"].sum())
except Exception as exc:
    st.warning(f"No se pudo cargar métricas de DuckDB: {exc}")

dbt_models = int(dbt.get("models_total", 0))
dbt_tests = int(dbt.get("tests_total", 0))
dbt_success = int(dbt.get("run_status", {}).get("success", 0))
total_run_results = sum(dbt.get("run_status", {}).values())
pass_rate = (dbt_success / max(1, total_run_results)) * 100

kpi_row(
    [
        {"label": "Schemas DuckDB", "value": str(duckdb_schema_count)},
        {"label": "Tablas DuckDB", "value": str(duckdb_table_count)},
        {"label": "Modelos/Tests dbt", "value": f"{dbt_models}/{dbt_tests}"},
        {"label": "Pass rate dbt", "value": f"{pass_rate:.0f}%"},
        {"label": "Feature Views/Services", "value": f"{len(feast.get('feature_views', []))}/{len(feast.get('feature_services', []))}"},
        {"label": "Online store Feast", "value": f"{feast.get('online_size_gb', 0.0):.2f} GB"},
    ],
    n_cols=3,
)

try:
    catalog = load_duckdb_table_catalog()
    with st.expander("Catálogo DuckDB (tablas disponibles)"):
        st.dataframe(catalog, use_container_width=True, height=300)
except Exception:
    pass

st.markdown(
    """
La capa de plataforma no reemplaza la ingeniería en Python; la formaliza:
- **DuckDB** acelera consulta local y visualización.
- **dbt** aporta linaje, tests y documentación reproducible.
- **Feast** expresa contrato de features y coherencia conceptual train/serve.
"""
)

# ── dbt Model Details ──
with st.expander("Detalle de modelos dbt (click para ver SQL)"):
    dbt_models_dir = PROJECT_ROOT / "dbt_project" / "models"
    if dbt_models_dir.exists():
        sql_files = sorted(dbt_models_dir.rglob("*.sql"))
        if sql_files:
            for sql_file in sql_files:
                rel = sql_file.relative_to(dbt_models_dir)
                layer = str(rel.parts[0]) if len(rel.parts) > 1 else "root"
                with st.expander(f"📄 `{rel}` ({layer})"):
                    st.code(sql_file.read_text(encoding="utf-8"), language="sql")
        else:
            st.info("No se encontraron archivos SQL en dbt_project/models/.")
    else:
        st.info("Directorio dbt_project/models/ no encontrado.")

# ── Feast Feature View Details ──
with st.expander("Detalle de Feast Feature Views"):
    fv_names = feast.get("feature_views", [])
    fs_names = feast.get("feature_services", [])
    col_fv, col_fs = st.columns(2)
    with col_fv:
        st.markdown("**Feature Views:**")
        for fv in fv_names:
            st.markdown(f"- `{fv}`")
        if not fv_names:
            st.info("No se detectaron feature views.")
    with col_fs:
        st.markdown("**Feature Services:**")
        for fs_name in fs_names:
            st.markdown(f"- `{fs_name}`")
        if not fs_names:
            st.info("No se detectaron feature services.")
    st.markdown(
        """
**Rol en el proyecto:** Feast no sirve features en tiempo real (no hay inferencia online),
sino que demuestra el patrón de **consistencia train/serve**: las mismas features definidas
en `feature_views.py` se usan tanto para entrenar como para scoring, eliminando el riesgo
de feature drift entre ambientes.
"""
    )

st.markdown(
    """
Como cierre narrativo, la lectura correcta es que “arquitectura” aquí significa trazabilidad total de transformación:
qué dato entra, cómo se modifica, qué artefacto produce y qué módulo lo consume. Esa trazabilidad es la que permite que
el relato metodológico del proyecto sea defendible de principio a fin y no dependa de supuestos implícitos.
"""
)

next_page_teaser(
    "Mapa Integrado de Métodos",
    "Cómo se conectan ML, estadística, causalidad y optimización sobre este linaje de datos.",
    "pages/thesis_defense.py",
)
