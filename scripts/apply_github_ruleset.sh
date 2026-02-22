#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-}"
RULESET_FILE="${2:-.github/rulesets/main-branch-protection.json}"
RULESET_NAME="${RULESET_NAME:-main-branch-protection}"

if [[ -z "$REPO" ]]; then
  echo "Usage: $0 <owner/repo> [ruleset_json_path]" >&2
  exit 1
fi
if [[ ! -f "$RULESET_FILE" ]]; then
  echo "Ruleset file not found: $RULESET_FILE" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run: gh auth login" >&2
  exit 1
fi

existing_id="$(gh api "repos/$REPO/rulesets" --jq ".[] | select(.name == \"$RULESET_NAME\") | .id" | head -n1 || true)"

if [[ -n "$existing_id" ]]; then
  echo "Updating existing ruleset id=$existing_id ($RULESET_NAME) on $REPO"
  gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    "repos/$REPO/rulesets/$existing_id" \
    --input "$RULESET_FILE" >/dev/null
else
  echo "Creating ruleset ($RULESET_NAME) on $REPO"
  gh api \
    --method POST \
    -H "Accept: application/vnd.github+json" \
    "repos/$REPO/rulesets" \
    --input "$RULESET_FILE" >/dev/null
fi

echo "Done. Current rulesets:"
gh api "repos/$REPO/rulesets" --jq '.[] | {id, name, target, enforcement}'
