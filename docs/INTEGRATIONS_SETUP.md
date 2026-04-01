# Integrations Setup (GitHub + DVC + DagsHub)

Fecha de verificación: 2026-02-17.

## Estado actual del repo

- `git` inicializado localmente en este directorio.
- `dvc` inicializado localmente (`.dvc/`).
- Remote DVC por defecto: **DagsHub (S3-compatible recomendado)**.
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
4. remoto DVC **default** en DagsHub (S3-compatible por defecto, HTTP fallback),
5. auth local para DVC DagsHub,
6. variables `.env` para MLflow/DagsHub.
7. hooks `pre-commit` y `pre-push` (si existe `.pre-commit-config.yaml`).

## Variables requeridas (modo por defecto)

```bash
export GIT_USER_NAME="Tu Nombre"
export GIT_USER_EMAIL="tu@email.com"
export GITHUB_REPO_URL="https://github.com/<user>/<repo>.git"
export DAGSHUB_USER="<user>"
export DAGSHUB_REPO="<repo>"
export DAGSHUB_USER_TOKEN="<token>"
```

Variables opcionales de comportamiento:

```bash
# Default recomendado para evitar errores 413 con artefactos grandes
export DVC_REMOTE_BACKEND="s3"   # o "http" (legacy)

# Si no quieres que el script instale hooks locales
export SKIP_PRECOMMIT_INSTALL=1

# Paso cosmético opcional (onboarding DagsHub UI)
export DAGSHUB_CLIENT_BOOTSTRAP=1
```

Variable opcional para no volver a pedir credenciales GitHub en terminal no interactiva:

```bash
export GITHUB_PAT="<github_pat>"
```

## Data Pipeline en DagsHub

Si DagsHub muestra:

> "Your version controlled data pipeline could be here"

normalmente significa que falta `dvc.yaml` o que aún no está pusheado al remoto git.

Validación local:

```bash
uv run dvc dag
```

Si usas `DVC_REMOTE_BACKEND=s3`, asegúrate de tener soporte S3 en DVC:

```bash
uv run dvc doctor | rg "s3 \\("
```

Si no aparece, instala el plugin:

```bash
uv add "dvc[s3]>=3.56"
# o
uv add dvc-s3
```

## Push recomendado de datos

```bash
# DagsHub (principal)
uv run dvc push -r dagshub
```

El setup por defecto usa el endpoint S3-compatible de DagsHub (multipart upload), lo que evita el error `413 Request Entity Too Large` que sí puede ocurrir con el remoto HTTP `.dvc`.

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

## DVC Metrics / Plots (KPIs canónicos)

Este repo exporta un resumen canónico para `dvc metrics` y 2 CSVs para `dvc plots`:

```bash
uv run dvc repro core.governance.export_dvc_metrics
uv run dvc metrics show
uv run dvc plots show
```

Comparación rápida entre commits/branches:

```bash
uv run dvc metrics diff
```

## Paso opcional: completar onboarding visual de DagsHub

La casilla "Version your data with our client" en DagsHub puede quedar sin marcar aunque DVC/MLflow ya funcionen. Es un indicador UI, no un requisito funcional.

Si quieres intentar activarla, ejecuta una vez:

```bash
DAGSHUB_CLIENT_BOOTSTRAP=1 bash scripts/configure_integrations.sh
```

Esto corre `dagshub.init(..., dvc=True)` como paso opcional y puede modificar configuración local de DVC.

## Verificación rápida

```bash
uv run dvc version
uv run dvc remote list
uv run dvc dag
cat .dvc/config
cat .dvc/config.local
```
