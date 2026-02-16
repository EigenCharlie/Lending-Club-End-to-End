# Integrations Setup (GitHub + DVC + DagsHub + Google Drive)

Fecha de verificación: 2026-02-16.

## Estado actual del repo

- `git` inicializado localmente en este directorio.
- `dvc` inicializado localmente (`.dvc/`).
- Plugin `dvc-gdrive` agregado al proyecto para habilitar remoto Google Drive.
- `dvc.yaml` + `dvc.lock` creados con pipeline completo (13 etapas).
- MLflow configurado para DagsHub con script de backfill de experimentos.

## Comando único de configuración

Con variables exportadas en entorno, ejecuta:

```bash
bash scripts/configure_integrations.sh
```

El script configura:

1. identidad `git` local,
2. remoto `origin` (GitHub),
3. remoto `dagshub` (Git mirror),
4. remoto DVC por defecto en Google Drive,
5. remoto DVC secundario en DagsHub,
6. variables `.env` para MLflow/DagsHub.

## Variables requeridas

```bash
export GIT_USER_NAME="Tu Nombre"
export GIT_USER_EMAIL="tu@email.com"
export GITHUB_REPO_URL="https://github.com/<user>/<repo>.git"
export DAGSHUB_USER="<user>"
export DAGSHUB_REPO="<repo>"
export DAGSHUB_USER_TOKEN="<token>"
export GDRIVE_FOLDER_ID="<folder_id>"
```

## Variables opcionales (Google Drive auth)

Si Google bloquea OAuth de la app por defecto de DVC, usa una de estas opciones:

```bash
export GDRIVE_CLIENT_ID="..."
export GDRIVE_CLIENT_SECRET="..."
# o
export GDRIVE_SERVICE_ACCOUNT_JSON="/ruta/service-account.json"
```

## Nota importante sobre DagsHub + Google Drive

- DVC sí soporta remoto `gdrive://`.
- DagsHub (External Buckets UI) documenta soporte para `AWS S3`, `GCS`, `Azure Blob` y compatibles S3.
- Por lo tanto, la forma práctica de “usar Google Drive con DagsHub” es:
  - usar Google Drive como remoto DVC del proyecto, y
  - usar DagsHub para Git/MLflow y opcionalmente remoto DVC adicional (`.dvc`).

## Data Pipeline en DagsHub

Si DagsHub muestra:

> "Your version controlled data pipeline could be here"

normalmente significa que falta `dvc.yaml` (pipeline declarativo).  
En este repo ya existe `dvc.yaml`; la DAG se puede validar localmente con:

```bash
uv run dvc dag
```

## Primer push a Google Drive (OAuth)

La primera vez, DVC solicita autorización OAuth. Ejecuta:

```bash
uv run dvc push -r gdrive -v
```

Flujo esperado:

1. DVC imprime una URL de Google OAuth.
2. Abres esa URL en el navegador logueado con tu cuenta.
3. Aceptas permisos de Drive.
4. DVC guarda credenciales locales y continúa el push.

Después de ese primer login, `dvc push -r gdrive` y `dvc pull -r gdrive` funcionan sin repetir el flujo.

## MLflow Suite en DagsHub

Para registrar la suite completa de experimentos desde artefactos existentes (sin reentrenar):

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

Para un backfill rápido sin subir artifacts grandes a MLflow:

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
