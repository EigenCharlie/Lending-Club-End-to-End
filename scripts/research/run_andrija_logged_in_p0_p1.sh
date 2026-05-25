#!/usr/bin/env bash
set -euo pipefail

CDP_URL="${CDP_URL:-http://127.0.0.1:9222}"
PACK_DIR="reports/linkedin_credit_risk_andrija_djurovic/logged_in_review"

python scripts/research/build_andrija_logged_in_review_queue.py
python scripts/research/capture_linkedin_logged_in_cdp.py \
  --pack-dir "${PACK_DIR}" \
  --cdp-url "${CDP_URL}" \
  --items all \
  --sleep-seconds 1.0 \
  --expand-iterations 6
python scripts/research/analyze_linkedin_logged_in_review.py \
  --pack-dir "${PACK_DIR}" \
  --resolve
python scripts/research/build_andrija_logged_in_project_intake.py
