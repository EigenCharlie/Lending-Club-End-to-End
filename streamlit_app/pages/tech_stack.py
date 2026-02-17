"""Stack tecnologico: librerias, versiones, practicas de ingenieria."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_mermaid import st_mermaid

from streamlit_app.components.narrative import next_page_teaser


def _lib_df(rows: list[list[str]]) -> pd.DataFrame:
    """Build a library comparison DataFrame."""
    return pd.DataFrame(
        rows, columns=["Librería", "Versión", "Rol", "Alternativas", "Razón de elección"]
    )


st.title("🛠️ Stack Tecnológico")
st.caption(
    "Librerías, versiones, decisiones de diseño y prácticas de ingeniería "
    "del pipeline de riesgo de crédito."
)

st.markdown(
    """
Documentar el stack tecnológico no es un ejercicio cosmético: permite reproducibilidad,
facilita la colaboración, y demuestra que cada componente fue seleccionado con criterio
técnico. Esta página detalla **por qué** cada librería y no solo **cuál**.
"""
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Library Ecosystem
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("1) Ecosistema de librerías")

tabs = st.tabs(
    [
        "ML Core",
        "Conformal",
        "Series de Tiempo",
        "Supervivencia",
        "Causal",
        "Optimización",
        "Datos",
        "MLOps",
        "API & Dashboard",
        "Plataforma",
        "Dev",
    ]
)

with tabs[0]:
    st.dataframe(
        _lib_df(
            [
                [
                    "catboost",
                    ">=1.2",
                    "PD model principal",
                    "XGBoost, LightGBM",
                    "Manejo nativo de categorías y NaN — cero preprocesamiento",
                ],
                [
                    "scikit-learn",
                    ">=1.6",
                    "Framework base ML",
                    "—",
                    "Estándar de la industria; CalibratedClassifierCV para Platt",
                ],
                [
                    "lightgbm",
                    ">=4.5",
                    "Time series vía mlforecast",
                    "XGBoost",
                    "Más rápido en entrenamiento, integración Nixtla nativa",
                ],
                [
                    "optuna",
                    ">=4.1",
                    "Hyperparameter tuning",
                    "Hyperopt, Ray Tune",
                    "API pythónica, pruning inteligente, visualización integrada",
                ],
                [
                    "shap",
                    ">=0.47",
                    "Explicabilidad del modelo",
                    "LIME, ELI5",
                    "Teoría sólida (Shapley values), TreeSHAP para CatBoost",
                ],
                [
                    "optbinning",
                    ">=0.19",
                    "WOE binning",
                    "Binning manual",
                    "Binning óptimo con regularización; calcula IV automáticamente",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.dataframe(
        _lib_df(
            [
                [
                    "mapie",
                    ">=1.3.0,<2",
                    "Conformal prediction",
                    "crepes, nonconformist",
                    "API moderna: SplitConformalRegressor + Mondrian nativo",
                ],
                [
                    "crepes",
                    ">=0.7",
                    "CP complementario",
                    "mapie",
                    "Implementación alternativa para validación cruzada de resultados",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "**Decisión crítica**: MAPIE >=1.3.0 (no >=0.9). La API cambió "
        "completamente: `SplitConformalRegressor` reemplaza `MapieRegressor`, "
        "`conformalize()` reemplaza `fit()` con `cv='prefit'`."
    )

with tabs[2]:
    st.dataframe(
        _lib_df(
            [
                [
                    "statsforecast",
                    ">=2.0",
                    "Modelos estadísticos (ARIMA, ETS)",
                    "statsmodels, Prophet",
                    "10-100x más rápido que statsmodels; API consistente",
                ],
                [
                    "mlforecast",
                    ">=0.13",
                    "ML para time series (LightGBM)",
                    "Prophet, sktime",
                    "Integración nativa con LightGBM; feature engineering temporal",
                ],
                [
                    "hierarchicalforecast",
                    ">=1.0",
                    "Reconciliación jerárquica",
                    "—",
                    "Único ecosistema open-source para forecasting jerárquico",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Ecosistema Nixtla: las tres librerías comparten API y son interoperables.")

with tabs[3]:
    st.dataframe(
        _lib_df(
            [
                [
                    "lifelines",
                    ">=0.30",
                    "Cox PH, Kaplan-Meier",
                    "R survival",
                    "API pythónica, estimación semiparamétrica, Schoenfeld tests",
                ],
                [
                    "scikit-survival",
                    ">=0.24",
                    "Random Survival Forests",
                    "pycox (deep learning)",
                    "RSF interpretable; compatible con sklearn API",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[4]:
    st.dataframe(
        _lib_df(
            [
                [
                    "econml",
                    ">=0.16",
                    "DML, Causal Forest, CATE",
                    "CausalML (Uber)",
                    "Microsoft Research; DML + orthogonalization rigurosa",
                ],
                [
                    "dowhy",
                    ">=0.12",
                    "Grafos causales, refutación",
                    "—",
                    "Único framework Python con refutation tests formales",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[5]:
    st.dataframe(
        _lib_df(
            [
                [
                    "pyomo",
                    ">=6.8",
                    "Modelado algebraico (LP/MILP)",
                    "PuLP, CVXPY",
                    "Más expresivo que PuLP; soporta NLP y robust optimization",
                ],
                [
                    "highspy",
                    ">=1.10",
                    "Solver HiGHS (LP/MIP)",
                    "Gurobi (comercial), CBC",
                    "Open-source, rendimiento comparable a Gurobi en LPs medianos",
                ],
                [
                    "cvxpy",
                    ">=1.6",
                    "Optimización convexa",
                    "scipy.optimize",
                    "Disciplined convex programming; útil para formulaciones alternativas",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "**HiGHS vs Gurobi**: HiGHS es open-source y resuelve nuestro portfolio LP "
        "en <1s. Gurobi sería necesario solo para MILPs de >100K variables."
    )

with tabs[6]:
    st.dataframe(
        _lib_df(
            [
                [
                    "pandas",
                    ">=2.2",
                    "Data wrangling principal",
                    "polars",
                    "Ecosistema maduro; compatibilidad universal",
                ],
                [
                    "polars",
                    ">=1.20",
                    "Procesamiento high-performance",
                    "pandas",
                    "10x más rápido en operaciones pesadas; lazy evaluation",
                ],
                [
                    "duckdb",
                    ">=1.1",
                    "Queries analíticas in-process",
                    "SQLite",
                    "OLAP nativo; lee parquet directamente; backend para dbt",
                ],
                [
                    "pyarrow",
                    ">=18.0",
                    "I/O parquet, interop columnar",
                    "fastparquet",
                    "Estándar de facto; interop con pandas, polars, duckdb",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[7]:
    st.dataframe(
        _lib_df(
            [
                [
                    "dvc",
                    ">=3.56",
                    "Versionado de datos",
                    "git-lfs",
                    "Pipeline DAG + remote storage; git-lfs no soporta pipelines",
                ],
                [
                    "mlflow",
                    ">=2.20",
                    "Experiment tracking",
                    "W&B (comercial)",
                    "Open-source, UI integrada, model registry",
                ],
                [
                    "pandera",
                    ">=0.22",
                    "Validación de schemas",
                    "great_expectations",
                    "Ligero, decoradores, integración pandas nativa",
                ],
                [
                    "dagshub",
                    ">=0.4",
                    "Hub MLOps (Git + DVC + MLflow)",
                    "—",
                    "Unifica Git mirror, DVC remote y MLflow tracking en una plataforma",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[8]:
    st.dataframe(
        _lib_df(
            [
                [
                    "fastapi",
                    ">=0.115",
                    "API REST",
                    "Flask, Django",
                    "Async nativo, auto-docs OpenAPI, Pydantic validation",
                ],
                [
                    "streamlit",
                    ">=1.42",
                    "Dashboard interactivo",
                    "Dash, Gradio",
                    "Prototipado rápido; st.navigation() para multi-page apps",
                ],
                [
                    "mcp[cli]",
                    ">=1.2",
                    "Model Context Protocol",
                    "—",
                    "Protocolo estándar para LLM tool use; FastMCP server",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[9]:
    st.dataframe(
        _lib_df(
            [
                [
                    "dbt-duckdb",
                    ">=1.9",
                    "Transformación SQL gobernada",
                    "SQL scripts manuales",
                    "Lineage, tests, docs automáticos; read_parquet() en staging",
                ],
                [
                    "feast",
                    ">=0.55",
                    "Feature store",
                    "Tecton (comercial)",
                    "Feature views declarativos; offline/online serving",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[10]:
    st.dataframe(
        _lib_df(
            [
                [
                    "pytest",
                    ">=8.3",
                    "Test framework",
                    "unittest",
                    "Fixtures, markers, plugins; estándar de facto en Python moderno",
                ],
                [
                    "ruff",
                    ">=0.9",
                    "Linter + formatter",
                    "flake8+isort+black",
                    "Una herramienta reemplaza tres; 10-100x más rápido (Rust)",
                ],
                [
                    "pre-commit",
                    ">=4.0",
                    "Git hooks automáticos",
                    "husky (JS)",
                    "Estándar Python; ejecuta Ruff + nbstripout antes de commit",
                ],
                [
                    "nbstripout",
                    ">=0.8",
                    "Limpiar outputs de notebooks",
                    "—",
                    "Previene merge conflicts en notebooks",
                ],
                [
                    "uv",
                    "latest",
                    "Package manager",
                    "pip, poetry, conda",
                    "10-100x más rápido que pip; lock file determinístico",
                ],
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 2. Engineering Practices
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2) Prácticas de ingeniería")

TEST_SUITE_TOTAL = 195
TEST_BREAKDOWN = [
    ["test_api/test_router_normalization", 5, "Normalización de parámetros FastAPI"],
    ["test_config_consistency", 11, "Drift entre config YAML, contrato PD y DAG DVC"],
    ["test_data/test_make_dataset", 8, "Limpieza inicial y control de leakage"],
    ["test_data/test_prepare_dataset", 8, "Splits OOT y calibración temporal"],
    ["test_data/test_build_datasets", 13, "loan_master/time_series/ead + parsing robusto"],
    ["test_features/", 5, "Feature engineering + Pandera schemas"],
    ["test_models/test_pd_model", 9, "CatBoost, baseline LR, calibración"],
    ["test_models/test_conformal", 19, "Cobertura, grupos, Mondrian y edge cases"],
    ["test_evaluation/test_ifrs9", 19, "Staging ECL, rangos conformal y edge cases"],
    ["test_evaluation/test_metrics", 11, "Métricas clasificación y regresión"],
    ["test_optimization/test_portfolio", 15, "Pyomo, solver, constraints y escenarios"],
    ["test_scripts/", 15, "Suite de scripts (MLflow helpers + end_to_end + export artifacts)"],
    ["test_utils/test_mlflow_utils", 6, "init_dagshub + logging de experimentos"],
    ["test_streamlit/test_page_imports", 43, "Smoke AST/import de 20 páginas Streamlit"],
    ["test_integration", 8, "Pipeline completo sobre datos sintéticos"],
]

with st.expander(f"Testing: {TEST_SUITE_TOTAL} tests con pytest"):
    test_data = pd.DataFrame(
        TEST_BREAKDOWN,
        columns=["Módulo", "Tests", "Qué valida"],
    )
    st.dataframe(test_data, use_container_width=True, hide_index=True)
    st.markdown(
        """
**Patrones de testing**:
- Fixtures con datos sintéticos (no dependen de archivos reales)
- Markers: `@pytest.mark.slow` para tests pesados, `@pytest.mark.integration`
- Config consistency tests: previenen drift entre YAML, código y tesis
- `pytest -x`: falla rápido en el primer error
"""
    )

with st.expander("Linting: Ruff con reglas E, F, W, I, UP, B, SIM"):
    st.markdown(
        """
**Ruff** reemplaza tres herramientas (flake8 + isort + black) con una sola, escrita en
Rust y 10-100x más rápida.

**Reglas activas**:
| Código | Categoría | Qué detecta |
|--------|-----------|-------------|
| E | pycodestyle | Espacios, indentación, líneas largas |
| F | Pyflakes | Imports no usados, variables undefined |
| W | Warnings | Deprecations, estilo |
| I | isort | Orden de imports |
| UP | pyupgrade | Idiomas modernos de Python 3.11+ |
| B | flake8-bugbear | Pitfalls comunes |
| SIM | flake8-simplify | Simplificaciones de código |

**Config**: `line-length = 100`, `target-version = "py311"`, `known-first-party = ["src"]`
"""
    )

with st.expander("Pre-commit hooks"):
    st.markdown(
        """
Dos hooks ejecutan automáticamente antes de cada `git commit`:

1. **Ruff** (`ruff --fix` + `ruff format`): auto-corrige imports, formato y estilo.
2. **nbstripout**: elimina outputs y metadata de notebooks, previniendo merge conflicts
   y evitando que datos sensibles se filtren al repositorio.

Config en `.pre-commit-config.yaml` con `ruff-pre-commit v0.9.0` y `nbstripout 0.8.1`.
"""
    )

with st.expander("CI/CD: GitHub Actions (3 jobs)"):
    st.markdown(
        """
**Workflow**: `.github/workflows/ci.yml` — triggers en push a main/master y PRs.

| Job | Qué hace | Comando |
|-----|----------|---------|
| **Lint** | Verifica formato y estilo | `ruff check` + `ruff format --check` |
| **Test** | Ejecuta test suite completa | `pytest -x -m "not slow"` |
| **Streamlit Smoke** | Verifica que las 20 páginas importan sin error | `importlib.util.find_spec()` |

Setup: `actions/checkout@v4` + `astral-sh/setup-uv@v5` + `uv sync --extra dev`.
"""
    )

with st.expander("Containerización: Docker Compose (2 servicios)"):
    st.markdown(
        """
| Servicio | Puerto | Base | Propósito |
|----------|--------|------|-----------|
| **api** | 8000 | python:3.12-slim | FastAPI con CatBoost inference |
| **streamlit** | 8501 | python:3.12-slim | Dashboard de 20 páginas |

**Prácticas clave**:
- Data y modelos montados como **volúmenes read-only** (inmutabilidad)
- `uv sync --frozen` para instalación determinística
- Healthcheck HTTP en el servicio API
- Multi-stage build con `ghcr.io/astral-sh/uv:latest`
"""
    )

with st.expander("Gobernanza de datos: dbt + Feast"):
    st.markdown(
        """
**dbt-duckdb**: 19 modelos SQL (9 tables + 10 views), 16 tests de integridad.
- Staging: `read_parquet()` directo sobre archivos parquet
- Marts: tablas analíticas listas para consumo (risk assessment, ECL, conformal, causal)
- Tests: not_null, unique, accepted_values en columnas críticas

**Feast**: 3 feature views + 2 feature services.
- `loan_origination`: features de originación (monto, ingreso, DTI, tasa)
- `loan_credit_history`: historial crediticio (delinquencias, cuentas abiertas)
- `loan_demographics`: demografía (vivienda, propósito, empleo)
- Registry SQLite local; offline store en parquets
"""
    )

with st.expander("Configuración: YAML + Pandera"):
    st.markdown(
        """
**4 archivos YAML** en `configs/`:
- `pd_model.yaml`: hiperparámetros CatBoost, paths, features, calibración, conformal
- `conformal_policy.yaml`: umbrales de cobertura, alertas, artefactos
- `optimization.yaml`: tipo de solver, presupuesto, concentración, robustez
- `modeva_governance.yaml`: fairness, drift, robustness diagnostics

**Pandera schemas** en `src/features/schemas.py`: validación en boundaries del pipeline.
Cada DataFrame pasa por un schema check antes de persistirse o consumirse downstream.

**Tests de consistencia** (`tests/test_config_consistency.py`): 7 tests que verifican
que config YAML, código fuente y tesis estén alineados — previenen drift silencioso.
"""
    )

with st.expander("Logging: loguru (no print)"):
    st.markdown(
        """
`from loguru import logger` reemplaza `print()` en todo el codebase.

**Ventajas**: timestamps automáticos, niveles (INFO/WARNING/ERROR),
formato estructurado, rotación de archivos, stack traces enriquecidos.

Convención: `logger.info(f"...")` para progreso, `logger.warning(...)` para
condiciones recuperables, `logger.error(...)` para fallos.
"""
    )

with st.expander("Gestión de paquetes: uv"):
    st.markdown(
        """
**uv** (Astral) reemplaza pip, poetry y conda con una herramienta 10-100x más rápida
escrita en Rust.

- `uv sync --extra dev`: instala dependencias de desarrollo (pytest, ruff)
- `uv sync --extra platform`: instala dbt + Feast (incompatible con dev)
- `uv.lock`: lock file frozen para reproducibilidad exacta
- `.venv/`: virtualenv Linux nativo en WSL2

**Nota**: los extras `dev` y `platform` no pueden coexistir debido a conflictos
de versiones (feast requiere numpy>=2, mpi-sppy requiere numpy<2).
"""
    )

with st.expander("MLOps: DVC Pipeline + MLflow Tracking"):
    st.markdown(
        """
**DVC** (Data Version Control) define el pipeline completo como un DAG declarativo
en `dvc.yaml` con **17 stages**, desde la limpieza de datos hasta la capa de export/storytelling.

```text
make_dataset → prepare_dataset → build_datasets
  → train_pd_model → generate_conformal
  → forecast_default_rates
  → estimate_causal_effects → simulate_causal_policy → validate_causal_policy
  → run_ifrs9_sensitivity
  → optimize_portfolio → optimize_portfolio_tradeoff
  → run_survival_analysis
  → backtest_conformal_coverage → validate_conformal_policy
  → export_streamlit_artifacts → export_storytelling_snapshot
```

- `dvc.yaml`: define dependencias, comandos y salidas de cada stage
- `dvc.lock`: fija hashes de datos y artefactos para reproducibilidad exacta
- `dvc repro`: re-ejecuta solo los stages con dependencias modificadas
- Remote: **DagsHub** (almacenamiento centralizado de artefactos grandes)

**MLflow** registra 8 experimentos desde artefactos existentes:
1. `end_to_end` — métricas globales del pipeline
2. `pd_model` — AUC, Gini, KS, Brier, ECE del modelo CatBoost + Platt
3. `conformal` — cobertura Mondrian 90%/95%, ancho de intervalos
4. `causal_policy` — regla seleccionada, net value, bootstrap p05
5. `ifrs9` — ECL baseline/severe, uplift por escenario
6. `optimization` — portafolio robusto vs nominal, costo de robustez
7. `survival` — Cox C-index, RSF C-index, concordancia
8. `time_series` — MASE, RMSSE por modelo de forecast

**DagsHub** unifica las tres capas:
- **Git mirror**: sincronización automática con GitHub
- **DVC remote**: almacenamiento de parquets y modelos serializados
- **MLflow UI**: visualización de experimentos, comparación de runs, registro de modelos
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# 3. System Architecture
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3) Arquitectura del sistema")

st_mermaid(
    """
    graph LR
        subgraph Data["Capa de Datos"]
            A[CSV/Parquet] --> B[DuckDB]
            B --> C[dbt Models]
            C --> D[Feast Features]
        end
        subgraph ML["Capa ML"]
            D --> E[CatBoost PD]
            E --> F[Platt Calibration]
        end
        subgraph CP["Capa Conformal"]
            F --> G[MAPIE Mondrian]
            G --> H["[PD_low, PD_high]"]
        end
        subgraph OR["Capa Optimización"]
            H --> I[Box Uncertainty Sets]
            I --> J[Pyomo + HiGHS]
        end
        subgraph Serving["Capa Serving"]
            J --> K[FastAPI]
            J --> L[Streamlit]
            J --> M[MCP Server]
        end

        style Data fill:#1a1a2e,stroke:#00D4AA,color:#e0e0e0
        style ML fill:#16213e,stroke:#00D4AA,color:#e0e0e0
        style CP fill:#16213e,stroke:#FFD93D,color:#e0e0e0
        style OR fill:#0f3460,stroke:#FF6B6B,color:#e0e0e0
        style Serving fill:#1a1a2e,stroke:#00D4AA,color:#e0e0e0
    """,
    height=300,
)

st.caption(
    "Cinco capas independientes conectadas por artefactos (parquets, modelos serializados, JSON contracts). "
    "Cada capa se puede reemplazar sin afectar las demás."
)

next_page_teaser(
    "Chat con Datos",
    "Explora libremente marts y staging con SQL o consultas en lenguaje natural.",
    "pages/chat_with_data.py",
)
