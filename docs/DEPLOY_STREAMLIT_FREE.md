# Deploy Gratis en Streamlit Community Cloud

Guía operativa recomendada para publicar esta app con costo **$0**, siguiendo buenas prácticas de estabilidad y reproducibilidad.

## Objetivo

Publicar una app pública de showcase (tesis/portafolio) sin ejecutar entrenamiento en cloud.

## 1) Preflight local

Desde la raíz del repo:

```bash
uv sync --extra dev
uv run python scripts/export_streamlit_artifacts.py
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Salida esperada:

- bundle en `dist/streamlit_deploy/`
- dependencias para Streamlit Cloud:
  - `streamlit_app/requirements.txt`
  - `requirements.txt`

## 2) Publicar bundle en repo de showcase

Recomendado: repo separado (ejemplo: `lending-club-risk-showcase`) para evitar conflictos con `uv.lock`/dev deps del monorepo.

```bash
cd dist/streamlit_deploy
git init
git add .
git commit -m "streamlit showcase bundle"
git branch -M main
git remote add origin <TU_REPO_SHOWCASE>
git push -u origin main
```

## 3) Deploy en Streamlit Community Cloud

1. Entrar a Streamlit Community Cloud y conectar GitHub.
2. Seleccionar el repo de showcase.
3. Configurar:
   - Branch: `main`
   - Main file path: `streamlit_app/app.py`
   - Python version: **3.11** (alineado con el proyecto)
4. Deploy.

## 4) Secrets (opcional)

Solo para habilitar NL->SQL con Grok:

```toml
GROK_API_KEY = "tu_api_key"
```

Sin esa secret, la app funciona igual (la funcionalidad de chat se desactiva de forma segura).

## Buenas Prácticas Aplicadas en Este Repo

1. **Precomputar artefactos offline** (`export_streamlit_artifacts.py`) para evitar cómputo pesado en cloud.
2. **Cachear lecturas** (`st.cache_data` / `st.cache_resource`) para reducir RAM/CPU por rerun.
3. **Dependencias explícitas por app** con `streamlit_app/requirements.txt` para evitar que Streamlit Cloud tome `uv.lock` del monorepo.
4. **Degradación controlada**: páginas toleran artefactos opcionales ausentes.
5. **Repositorio de despliegue liviano**: publicar bundle, no todo el pipeline de entrenamiento.

## Límites Relevantes de Community Cloud (referencia oficial)

Según la documentación de Streamlit, los límites gratuitos pueden cambiar sin aviso. Valores de referencia publicados:

- CPU: `~0.078` a `2` cores
- RAM: `~690MB` a `2.7GB`
- Storage: hasta `50GB`
- Hibernación por inactividad: `12 horas` sin tráfico

Implicación práctica:
- Esta app debe tratarse como **showcase analítico** (artefactos precomputados), no como entorno de entrenamiento.

## Troubleshooting Rápido

| Síntoma | Causa típica | Acción |
|---|---|---|
| `ModuleNotFoundError` en deploy | Dependencias incorrectas detectadas | Verifica que exista `streamlit_app/requirements.txt` en el repo desplegado |
| App lenta al iniciar | Artefactos pesados o cold start | Esperar primer arranque, luego revisar tamaño de bundle |
| Error de memoria/recurso | Carga de datos grande en runtime | Regenerar bundle sin DuckDB (`--skip-duckdb`) y validar páginas SQL |
| `GROK_API_KEY` no detectada | Secret no configurada | Cargar secret en panel de Streamlit Cloud |

## Ciclo de actualización

Cuando cambies páginas o artefactos:

```bash
uv run python scripts/export_streamlit_artifacts.py
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Luego push al repo de showcase y redeploy.
