#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<USAGE
Usage:
  bash scripts/configure_integrations.sh [--enable-gdrive]

Default mode configures:
  - git identity + remotes (GitHub + DagsHub)
  - DVC remote default on DagsHub
  - DagsHub auth for DVC + MLflow env vars
  - persistent git HTTPS credentials for DagsHub
  - persistent git HTTPS credentials for GitHub (if GITHUB_PAT or GH_TOKEN is set)

Optional:
  --enable-gdrive   Configure Google Drive as secondary DVC remote (backup)
USAGE
}

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

upsert_env_key() {
  local key="$1"
  local value="$2"
  if [[ ! -f .env ]]; then
    cp .env.example .env
  fi

  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    printf "%s=%s\n" "$key" "$value" >> .env
  fi
}

configure_git_https_credential() {
  local host="$1"
  local username="$2"
  local password="$3"
  local label="$4"

  if [[ -z "$password" ]]; then
    echo "Skipping persistent git credential for $label (token not provided)."
    return
  fi

  # Persist HTTPS credentials locally in this machine so non-interactive git commands work.
  git config --global credential.helper store
  git credential approve <<EOF
protocol=https
host=$host
username=$username
password=$password
EOF
  echo "Configured persistent git credential for $label ($host)."
}

ENABLE_GDRIVE="false"
for arg in "$@"; do
  case "$arg" in
    --enable-gdrive)
      ENABLE_GDRIVE="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

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

if [[ "$ENABLE_GDRIVE" == "true" ]]; then
  ensure_var GDRIVE_FOLDER_ID
fi

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

echo "Configuring persistent git HTTPS credentials..."
GITHUB_TOKEN="${GITHUB_PAT:-${GH_TOKEN:-}}"
configure_git_https_credential "dagshub.com" "$DAGSHUB_USER" "$DAGSHUB_USER_TOKEN" "DagsHub"
configure_git_https_credential "github.com" "${GITHUB_USER_NAME:-$GIT_USER_NAME}" "$GITHUB_TOKEN" "GitHub"

echo "Configuring DVC DagsHub remote (default)..."
DAGSHUB_DVC_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc"
uv run dvc remote add -f -d dagshub "$DAGSHUB_DVC_URL"
uv run dvc remote modify --local dagshub auth basic
uv run dvc remote modify --local dagshub user "$DAGSHUB_USER"
uv run dvc remote modify --local dagshub password "$DAGSHUB_USER_TOKEN"

if [[ "$ENABLE_GDRIVE" == "true" ]]; then
  echo "Configuring DVC Google Drive remote (secondary backup)..."
  uv run dvc remote add -f gdrive "gdrive://${GDRIVE_FOLDER_ID}"

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
    echo "Google Drive auth mode: default OAuth app (may be blocked by Google policy)."
  fi
fi

echo "Syncing DagsHub/MLflow env vars in .env..."
upsert_env_key DAGSHUB_USER "$DAGSHUB_USER"
upsert_env_key DAGSHUB_REPO "$DAGSHUB_REPO"
upsert_env_key DAGSHUB_USER_TOKEN "$DAGSHUB_USER_TOKEN"
upsert_env_key DAGSHUB_TOKEN "$DAGSHUB_USER_TOKEN"
if [[ -n "${GITHUB_PAT:-}" ]]; then
  upsert_env_key GITHUB_PAT "$GITHUB_PAT"
fi
upsert_env_key MLFLOW_TRACKING_URI "https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.mlflow"
upsert_env_key MLFLOW_TRACKING_USERNAME "$DAGSHUB_USER"
upsert_env_key MLFLOW_TRACKING_PASSWORD "$DAGSHUB_USER_TOKEN"

if [[ "$ENABLE_GDRIVE" == "true" ]]; then
  upsert_env_key GDRIVE_FOLDER_ID "$GDRIVE_FOLDER_ID"
fi

echo
echo "Integration setup complete (DagsHub-first)."
echo "- git origin:    $(git remote get-url origin)"
echo "- git dagshub:   $(git remote get-url dagshub)"
echo "- dvc default:   $DAGSHUB_DVC_URL"
if [[ "$ENABLE_GDRIVE" == "true" ]]; then
  echo "- dvc secondary: gdrive://$GDRIVE_FOLDER_ID"
fi
echo
echo "Next steps:"
echo "1) Push git to GitHub: git push -u origin main"
echo "2) Push git mirror to DagsHub: git push -u dagshub main"
echo "3) Push data artifacts to DagsHub: uv run dvc push -r dagshub"
if [[ "$ENABLE_GDRIVE" == "true" ]]; then
  echo "4) Optional backup to Google Drive: uv run dvc push -r gdrive"
fi
