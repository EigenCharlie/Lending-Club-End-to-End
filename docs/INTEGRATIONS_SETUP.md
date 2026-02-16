# Integrations Setup (GitHub + DVC + DagsHub + Google Drive)

Fecha de verificación: 2026-02-16.

## Estado actual del repo

- `git` inicializado localmente en este directorio.
- `dvc` inicializado localmente (`.dvc/`).
- Plugin `dvc-gdrive` agregado al proyecto para habilitar remoto Google Drive.

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

## Verificación rápida

```bash
uv run dvc version
uv run dvc remote list
cat .dvc/config
cat .dvc/config.local
```
