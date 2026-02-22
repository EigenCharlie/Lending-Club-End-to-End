#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DVC_REMOTE_BACKEND="${DVC_REMOTE_BACKEND:-s3}"
SKIP_PRECOMMIT_INSTALL="${SKIP_PRECOMMIT_INSTALL:-0}"
DAGSHUB_CLIENT_BOOTSTRAP="${DAGSHUB_CLIENT_BOOTSTRAP:-0}"

usage() {
  cat <<USAGE
Usage:
  bash scripts/configure_integrations.sh

Default mode configures:
  - git identity + remotes (GitHub + DagsHub)
  - DVC remote default on DagsHub (S3-compatible by default, HTTP fallback)
  - DagsHub auth for DVC + MLflow env vars
  - persistent git HTTPS credentials for DagsHub
  - persistent git HTTPS credentials for GitHub (if GITHUB_PAT or GH_TOKEN is set)
  - optional pre-commit hook install

Optional env flags:
  DVC_REMOTE_BACKEND=s3|http   (default: s3)
  SKIP_PRECOMMIT_INSTALL=1     (default: 0)
  DAGSHUB_CLIENT_BOOTSTRAP=1   (default: 0, cosmetic onboarding step)
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

require_dvc_s3_support() {
  if uv run dvc doctor | grep -q "s3 ("; then
    return
  fi

  cat >&2 <<'EOF'
ERROR: DVC S3 support is not available in the current uv environment.
This repo now defaults to DagsHub S3-compatible DVC remote to avoid HTTP 413 errors.

Install the plugin and retry:
  uv add "dvc[s3]>=3.56"
  # or
  uv add dvc-s3
EOF
  exit 1
}

configure_dvc_remote() {
  local backend="$1"
  local dvc_http_url="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc"
  local dvc_s3_endpoint="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.s3"

  if [[ "$backend" == "s3" ]]; then
    require_dvc_s3_support
    echo "Configuring DVC DagsHub remote (default, S3-compatible)..."
    uv run dvc remote add -f -d dagshub "s3://dvc"
    uv run dvc remote modify --local dagshub endpointurl "$dvc_s3_endpoint"
    uv run dvc remote modify --local dagshub access_key_id "$DAGSHUB_USER_TOKEN"
    uv run dvc remote modify --local dagshub secret_access_key "$DAGSHUB_USER_TOKEN"
    for key in auth user password; do
      uv run dvc remote modify --local -u dagshub "$key" >/dev/null 2>&1 || true
    done
    echo "DVC remote backend: dagshub (s3://dvc via $dvc_s3_endpoint)"
    return
  fi

  if [[ "$backend" == "http" ]]; then
    echo "Configuring DVC DagsHub remote (default, HTTP legacy mode)..."
    uv run dvc remote add -f -d dagshub "$dvc_http_url"
    uv run dvc remote modify --local dagshub auth basic
    uv run dvc remote modify --local dagshub user "$DAGSHUB_USER"
    uv run dvc remote modify --local dagshub password "$DAGSHUB_USER_TOKEN"
    for key in endpointurl access_key_id secret_access_key; do
      uv run dvc remote modify --local -u dagshub "$key" >/dev/null 2>&1 || true
    done
    echo "DVC remote backend: dagshub ($dvc_http_url)"
    return
  fi

  echo "ERROR: Invalid DVC_REMOTE_BACKEND='$backend' (expected 's3' or 'http')." >&2
  exit 1
}

maybe_install_precommit_hooks() {
  if [[ "$SKIP_PRECOMMIT_INSTALL" == "1" ]]; then
    echo "Skipping pre-commit install (SKIP_PRECOMMIT_INSTALL=1)."
    return
  fi

  if [[ ! -f .pre-commit-config.yaml ]]; then
    echo "No .pre-commit-config.yaml found; skipping hook install."
    return
  fi

  echo "Installing pre-commit hooks (pre-commit + pre-push)..."
  uv run --extra dev pre-commit install
  uv run --extra dev pre-commit install --hook-type pre-push
}

maybe_bootstrap_dagshub_client() {
  if [[ "$DAGSHUB_CLIENT_BOOTSTRAP" != "1" ]]; then
    return
  fi

  echo "Running optional DagsHub client bootstrap (dagshub.init dvc=True)..."
  uv run python - <<'PY'
from src.utils.mlflow_utils import init_dagshub

init_dagshub(enable_dvc=True)
print("DagsHub client bootstrap finished (dvc=True).")
PY
}

for arg in "$@"; do
  case "$arg" in
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

configure_dvc_remote "$DVC_REMOTE_BACKEND"

echo "Syncing DagsHub/MLflow env vars in .env..."
upsert_env_key DAGSHUB_USER "$DAGSHUB_USER"
upsert_env_key DAGSHUB_REPO "$DAGSHUB_REPO"
upsert_env_key DAGSHUB_USER_TOKEN "$DAGSHUB_USER_TOKEN"
upsert_env_key DAGSHUB_TOKEN "$DAGSHUB_USER_TOKEN"
if [[ -n "${GITHUB_PAT:-}" ]]; then
  upsert_env_key GITHUB_PAT "$GITHUB_PAT"
fi
upsert_env_key DVC_REMOTE_BACKEND "$DVC_REMOTE_BACKEND"
upsert_env_key MLFLOW_TRACKING_URI "https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.mlflow"
upsert_env_key MLFLOW_TRACKING_USERNAME "$DAGSHUB_USER"
upsert_env_key MLFLOW_TRACKING_PASSWORD "$DAGSHUB_USER_TOKEN"

maybe_install_precommit_hooks
maybe_bootstrap_dagshub_client

echo
echo "Integration setup complete (DagsHub-first)."
echo "- git origin:    $(git remote get-url origin)"
echo "- git dagshub:   $(git remote get-url dagshub)"
echo "- dvc backend:   $DVC_REMOTE_BACKEND"
echo "- dvc default:   $(uv run dvc remote list | awk '/\\(default\\)/{print $1, $2}')"
echo
echo "Next steps:"
echo "1) Push git to GitHub: git push -u origin main"
echo "2) Push git mirror to DagsHub: git push -u dagshub main"
echo "3) Push data artifacts to DagsHub: uv run dvc push -r dagshub"
