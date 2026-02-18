# Integrations Setup (GitHub + DVC + DagsHub)

Fecha de verificación: 2026-02-17.

## Estado actual del repo

- `git` inicializado localmente en este directorio.
- `dvc` inicializado localmente (`.dvc/`).
- Remote DVC por defecto: **DagsHub**.
- `dvc.yaml` + `dvc.lock`: pipeline reproducible del proyecto.
- MLflow integrado con DagsHub vía `scripts/log_mlflow_experiment_suite.py`.

## Configuración recomendada (DagsHub-first)

Con variables exportadas en entorno:

```bash
bash scripts/configure_integrations.sh
```

Este comando configura:

1. identidad `git` local,
2. remoto `origin` (GitHub),
3. remoto `dagshub` (Git mirror),
4. remoto DVC **default** en DagsHub,
5. auth local para DVC DagsHub,
6. variables `.env` para MLflow/DagsHub.

## Variables requeridas (modo por defecto)

```bash
export GIT_USER_NAME="Tu Nombre"
export GIT_USER_EMAIL="tu@email.com"
export GITHUB_REPO_URL="https://github.com/<user>/<repo>.git"
export DAGSHUB_USER="<user>"
export DAGSHUB_REPO="<repo>"
export DAGSHUB_USER_TOKEN="<token>"
```

Variable opcional para no volver a pedir credenciales GitHub en terminal no interactiva:

```bash
export GITHUB_PAT="<github_pat>"
```

## Google Drive opcional (backup secundario)

Si quieres agregar Google Drive como **segundo** remoto DVC:

```bash
export GDRIVE_FOLDER_ID="<folder_id>"
bash scripts/configure_integrations.sh --enable-gdrive
```

Variables opcionales para auth de Google Drive (si OAuth default falla):

```bash
export GDRIVE_CLIENT_ID="..."
export GDRIVE_CLIENT_SECRET="..."
# o
export GDRIVE_SERVICE_ACCOUNT_JSON="/ruta/service-account.json"
```

## Data Pipeline en DagsHub

Si DagsHub muestra:

> "Your version controlled data pipeline could be here"

normalmente significa que falta `dvc.yaml` o que aún no está pusheado al remoto git.

Validación local:

```bash
uv run dvc dag
```

## Push recomendado de datos

```bash
# DagsHub (principal)
uv run dvc push -r dagshub

# opcional, backup en Google Drive
uv run dvc push -r gdrive
```

## MLflow Suite en DagsHub

Para registrar la suite completa desde artefactos existentes (sin reentrenar):

```bash
set -a
source .env
set +a
uv run python scripts/log_mlflow_experiment_suite.py \
  --repo-owner "$DAGSHUB_USER" \
  --repo-name "$DAGSHUB_REPO"
```

Experimentos creados:

- `lending_club/end_to_end`
- `lending_club/pd_model`
- `lending_club/conformal`
- `lending_club/causal_policy`
- `lending_club/ifrs9`
- `lending_club/optimization`
- `lending_club/survival`
- `lending_club/time_series`

Para backfill rápido sin subir artefactos pesados a MLflow:

```bash
export MLFLOW_MAX_ARTIFACT_MB=0
```

## Verificación rápida

```bash
uv run dvc version
uv run dvc remote list
uv run dvc dag
cat .dvc/config
cat .dvc/config.local
```
