#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: Missing required command: $1" >&2
    exit 1
  }
}

ensure_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: Required env var is missing: $name" >&2
    exit 1
  fi
}

require_cmd git
require_cmd uv

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: This directory is not a git repository." >&2
  exit 1
fi

if [[ ! -f .dvc/config ]]; then
  echo "DVC not initialized. Running: uv run dvc init"
  uv run dvc init
fi

ensure_var GIT_USER_NAME
ensure_var GIT_USER_EMAIL
ensure_var GITHUB_REPO_URL
ensure_var DAGSHUB_USER
ensure_var DAGSHUB_REPO
ensure_var DAGSHUB_USER_TOKEN
ensure_var GDRIVE_FOLDER_ID

echo "Configuring git identity (local repo)..."
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

echo "Configuring git remotes..."
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$GITHUB_REPO_URL"
else
  git remote add origin "$GITHUB_REPO_URL"
fi

DAGSHUB_GIT_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.git"
if git remote get-url dagshub >/dev/null 2>&1; then
  git remote set-url dagshub "$DAGSHUB_GIT_URL"
else
  git remote add dagshub "$DAGSHUB_GIT_URL"
fi

echo "Configuring DVC Google Drive remote..."
uv run dvc remote add -f -d gdrive "gdrive://${GDRIVE_FOLDER_ID}"

# Authentication mode for Google Drive remote.
# Priority: service account > custom OAuth app > default OAuth flow.
if [[ -n "${GDRIVE_SERVICE_ACCOUNT_JSON:-}" ]]; then
  if [[ ! -f "$GDRIVE_SERVICE_ACCOUNT_JSON" ]]; then
    echo "ERROR: GDRIVE_SERVICE_ACCOUNT_JSON path does not exist: $GDRIVE_SERVICE_ACCOUNT_JSON" >&2
    exit 1
  fi
  uv run dvc remote modify --local gdrive gdrive_use_service_account true
  uv run dvc remote modify --local gdrive gdrive_service_account_json_file_path "$GDRIVE_SERVICE_ACCOUNT_JSON"
  echo "Google Drive auth mode: service account"
elif [[ -n "${GDRIVE_CLIENT_ID:-}" && -n "${GDRIVE_CLIENT_SECRET:-}" ]]; then
  uv run dvc remote modify --local gdrive gdrive_client_id "$GDRIVE_CLIENT_ID"
  uv run dvc remote modify --local gdrive gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
  echo "Google Drive auth mode: custom OAuth app"
else
  echo "Google Drive auth mode: default OAuth app (may require custom app if blocked by Google policy)."
fi

echo "Configuring DVC DagsHub remote..."
DAGSHUB_DVC_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc"
uv run dvc remote add -f dagshub "$DAGSHUB_DVC_URL"
uv run dvc remote modify --local dagshub auth basic
uv run dvc remote modify --local dagshub user "$DAGSHUB_USER"
uv run dvc remote modify --local dagshub password "$DAGSHUB_USER_TOKEN"

echo "Syncing DagsHub/MLflow env vars in .env (without overwriting existing file)..."
if [[ ! -f .env ]]; then
  cp .env.example .env
fi

# Append only missing keys.
append_if_missing() {
  local key="$1"
  local value="$2"
  if ! grep -q "^${key}=" .env; then
    printf "%s=%s\n" "$key" "$value" >> .env
  fi
}

append_if_missing DAGSHUB_USER "$DAGSHUB_USER"
append_if_missing DAGSHUB_REPO "$DAGSHUB_REPO"
append_if_missing DAGSHUB_USER_TOKEN "$DAGSHUB_USER_TOKEN"
append_if_missing DAGSHUB_TOKEN "$DAGSHUB_USER_TOKEN"
append_if_missing GDRIVE_FOLDER_ID "$GDRIVE_FOLDER_ID"
append_if_missing MLFLOW_TRACKING_URI "https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.mlflow"
append_if_missing MLFLOW_TRACKING_USERNAME "$DAGSHUB_USER"
append_if_missing MLFLOW_TRACKING_PASSWORD "$DAGSHUB_USER_TOKEN"

echo
echo "Integration setup complete."
echo "- git origin:   $(git remote get-url origin)"
echo "- git dagshub:  $(git remote get-url dagshub)"
echo "- dvc default:  gdrive://$GDRIVE_FOLDER_ID"
echo "- dvc dagshub:  $DAGSHUB_DVC_URL"
echo
echo "Next manual steps:"
echo "1) Authenticate DVC Google Drive when prompted by: uv run dvc push -r gdrive"
echo "2) Push git to GitHub: git push -u origin main"
echo "3) Push git mirror to DagsHub: git push -u dagshub main"
echo "4) Push data artifacts: uv run dvc push -r gdrive"
echo "5) Optional backup to DagsHub remote: uv run dvc push -r dagshub"
