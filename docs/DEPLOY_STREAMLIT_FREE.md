# Deploy Streamlit Gratis (Perfil Light)

Guía práctica para publicar la app con costo **$0** en Streamlit Community Cloud.

## Objetivo

Publicar una versión pública de la app (portafolio/tesis) sin subir todo el peso del pipeline completo.

## 1) Preparar artefactos locales

Desde la raíz del proyecto:

```bash
uv sync --extra dev
uv run python scripts/export_streamlit_artifacts.py
```

## 2) Construir bundle de despliegue

```bash
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Salida esperada:

- Carpeta: `dist/streamlit_deploy/`
- Incluye solo código Streamlit + artefactos necesarios para la demo + `requirements.streamlit.txt`.

## 3) Crear repo GitHub para despliegue

Recomendado: repo separado para la app pública, por ejemplo `lending-club-risk-showcase`.

```bash
cd dist/streamlit_deploy
git init
git add .
git commit -m "Initial deploy bundle"
git branch -M main
git remote add origin <TU_REPO_SHOWCASE>
git push -u origin main
```

## 4) Publicar en Streamlit Community Cloud

1. Entrar a Streamlit Community Cloud.
2. Conectar GitHub y seleccionar el repo `lending-club-risk-showcase`.
3. Configurar:
   - Main file path: `streamlit_app/app.py`
   - Python dependencies: `requirements.streamlit.txt` (detección automática)
4. Deploy.

## 5) Variables opcionales (Secrets)

Solo si quieres habilitar NL->SQL con Grok:

```toml
GROK_API_KEY="tu_api_key"
```

Sin esa variable la app funciona igual (solo deshabilita esa función).

## Notas operativas

- Este perfil está optimizado para demo pública, no para reentrenar modelos en cloud.
- Si haces cambios en páginas o artefactos, vuelve a ejecutar `prepare_streamlit_deploy.py` y push al repo de showcase.
- Si quieres reducir aún más tamaño del bundle, usa `--skip-duckdb` y valida que las páginas SQL sigan respondiendo como esperas.
