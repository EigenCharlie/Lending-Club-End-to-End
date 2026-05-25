#!/usr/bin/env bash
set -uo pipefail

# Governed Paper Estrella champion-replacement funnel:
# 1) stage minimal PD challenger artifacts,
# 2) run decision-wide conformal search,
# 3) run bound-aware portfolio search for selected conformal namespaces.
#
# Long-run knobs:
#   CANDIDATE_KEYS="bureau_behavior_15 canonical_4 affordability_rate_5"
#   MAX_CANDIDATES=0          # 0 = full available candidate pool
#   SHORTLIST_TOP_K=560
#   RANDOM_STATES=42
#   PORTFOLIO_VARIANTS=phase1,final
#   RUN_ROOT=paper1_champion_replacement_2026_05_23
#   RESUME_EXISTING_CONFORMAL=1
#   RESUME_EXISTING_PORTFOLIO=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ART_ROOT="${ART_ROOT:-/mnt/d/crpto_experiments/regret_auditability/regret_auditability_20260513_v3_resource_tuned}"
RUN_ROOT="${RUN_ROOT:-paper1_champion_replacement_2026_05_23}"
CONFORMAL_PROFILE="${CONFORMAL_PROFILE:-search_conformal_reopen_decision_wide}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"
SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-560}"
RANDOM_STATES="${RANDOM_STATES:-42}"
PORTFOLIO_VARIANTS="${PORTFOLIO_VARIANTS:-phase1,final}"
CANDIDATE_KEYS="${CANDIDATE_KEYS:-bureau_behavior_15 canonical_4 affordability_rate_5}"
RESUME_EXISTING_CONFORMAL="${RESUME_EXISTING_CONFORMAL:-1}"
RESUME_EXISTING_PORTFOLIO="${RESUME_EXISTING_PORTFOLIO:-1}"
CUOPT_BATCH_SIZE="${CUOPT_BATCH_SIZE:-8}"
CUOPT_METHOD="${CUOPT_METHOD:-Concurrent}"
CUOPT_NUM_CPU_THREADS="${CUOPT_NUM_CPU_THREADS:-24}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
mkdir -p "${LOG_DIR}"
EVENT_LOG="${EVENT_LOG:-${LOG_DIR}/${RUN_ROOT}_events.jsonl}"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"

emit_event() {
  local stage="$1"
  local key="$2"
  local state="$3"
  local detail="${4:-}"
  python - "$EVENT_LOG" "$RUN_ROOT" "$stage" "$key" "$state" "$detail" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

event_log, run_root, stage, key, state, detail = sys.argv[1:7]
payload = {
    "ts": datetime.now(UTC).isoformat(),
    "run_root": run_root,
    "stage": stage,
    "candidate_key": key,
    "state": state,
    "detail": detail,
}
path = Path(event_log)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

stage_candidate() {
  local key="$1"
  local tag="$2"
  local source_dir="$3"
  local target_dir="models/search_pd/${tag}"
  for file_name in pd_shadow_canonical.cbm pd_shadow_calibrator.pkl pd_model_contract.json pd_training_record.pkl; do
    if [[ ! -f "${source_dir}/${file_name}" ]]; then
      echo "Missing PD artifact for ${key}: ${source_dir}/${file_name}" >&2
      return 1
    fi
  done
  mkdir -p "${target_dir}"
  cp "${source_dir}/pd_shadow_canonical.cbm" "${target_dir}/pd_candidate_model.cbm" || return 1
  cp "${source_dir}/pd_shadow_calibrator.pkl" "${target_dir}/pd_candidate_calibrator.pkl" || return 1
  cp "${source_dir}/pd_model_contract.json" "${target_dir}/pd_model_contract.json" || return 1
  cp "${source_dir}/pd_training_record.pkl" "${target_dir}/pd_training_record.pkl" || return 1
  python - "$key" "$tag" "$source_dir" "${target_dir}/pd_training_status.json" <<'PY'
import json
import sys
from datetime import UTC, datetime

key, tag, source_dir, target = sys.argv[1:5]
payload = {
    "schema_version": "2026-05-23.1",
    "generated_at_utc": datetime.now(UTC).isoformat(),
    "run_tag": tag,
    "candidate_key": key,
    "source_dir": source_dir,
    "stage": "external_pd_candidate_staged_for_champion_reopen",
}
with open(target, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}

candidate_source_dir() {
  case "$1" in
    bureau_behavior_15)
      echo "${ART_ROOT}/pd/full_challenger_woe/bureau_behavior_15/pd-refine/models"
      ;;
    canonical_4)
      echo "${ART_ROOT}/pd/full_challenger/canonical_4/pd-refine/models"
      ;;
    affordability_rate_5)
      echo "${ART_ROOT}/pd/full_challenger_woe/affordability_rate_5/pd-refine/models"
      ;;
    *)
      echo "Unknown candidate key: $1" >&2
      return 2
      ;;
  esac
}

candidate_tag() {
  case "$1" in
    bureau_behavior_15) echo "regret_auditability_pd_bureau_behavior_15_2026_05_21" ;;
    canonical_4) echo "regret_auditability_pd_canonical_4_2026_05_23" ;;
    affordability_rate_5) echo "regret_auditability_pd_affordability_rate_5_2026_05_23" ;;
    *) return 2 ;;
  esac
}

status_value() {
  local path="$1"
  local key="$2"
  python - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print("")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload
for part in key.split("."):
    value = value.get(part, "") if isinstance(value, dict) else ""
print(value if value is not None else "")
PY
}

portfolio_gate() {
  local namespace="$1"
  python - "$namespace" <<'PY'
import json
import sys
from pathlib import Path

namespace = sys.argv[1]
status_path = Path("models/conformal_gap") / namespace / "conformal_policy_status.json"
if not status_path.exists():
    print("skip:missing_policy_status")
    raise SystemExit(0)
payload = json.loads(status_path.read_text(encoding="utf-8"))
coverage = float(payload.get("coverage_90", 0.0))
min_group = float(payload.get("min_group_coverage_90", 0.0))
width = float(payload.get("avg_width_90", 99.0))
total_alerts = int(payload.get("total_alerts", 99))
critical_alerts = int(payload.get("critical_alerts", 0))
if coverage >= 0.90 and min_group >= 0.88 and width <= 0.90 and critical_alerts == 0 and total_alerts <= 4:
    print("run")
else:
    print(
        "skip:"
        f"coverage90={coverage:.4f},"
        f"min_group90={min_group:.4f},"
        f"width90={width:.4f},"
        f"critical_alerts={critical_alerts},"
        f"total_alerts={total_alerts}"
    )
PY
}

run_portfolio() {
  local intervals_path="$1"
  local run_label="$2"
  conda run -n "${RAPIDS_ENV}" python scripts/search/run_portfolio_bound_aware_search.py \
    --conformal-intervals-path "${intervals_path}" \
    --run-label "${run_label}" \
    --risk-grid 0.150,0.155,0.160,0.165,0.170,0.175,0.180,0.185,0.190,0.195,0.200,0.205 \
    --gamma-grid 0.325,0.350,0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600,0.625,0.650,0.700 \
    --aversion-grid 0,0.02,0.05,0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00 \
    --policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty \
    --delta-cap-grid 0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,1.0 \
    --tail-focus-grid 0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,1.0 \
    --budget-profiles free \
    --shortlist-top-k "${SHORTLIST_TOP_K}" \
    --bucket-return-k 180 \
    --bucket-proxy-k 180 \
    --bucket-family-k 60 \
    --bucket-region-k 120 \
    --alpha-grid 0.01,0.02,0.03,0.05,0.10,0.15,0.20 \
    --max-candidates "${MAX_CANDIDATES}" \
    --random-states "${RANDOM_STATES}" \
    --solver-backend cuopt \
    --exact-solver-backend highs \
    --exact-python-executable "${MAIN_PYTHON}" \
    --cuopt-batch-size "${CUOPT_BATCH_SIZE}" \
    --cuopt-method "${CUOPT_METHOD}" \
    --cuopt-num-cpu-threads "${CUOPT_NUM_CPU_THREADS}" \
    --cuopt-dual-postsolve 0 \
    --incumbent-policy-path models/champion_portfolio_policy.json \
    --incumbent-risk-neighbors 0.160,0.165,0.170,0.175,0.180,0.185,0.190,0.195,0.200 \
    --incumbent-gamma-neighbors 0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600 \
    --incumbent-policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty
}

for key in ${CANDIDATE_KEYS}; do
  tag="$(candidate_tag "${key}")"
  if [[ -z "${tag}" ]]; then
    echo "[$(date -Is)] unknown candidate key: ${key}"
    emit_event "candidate" "${key}" "failed" "unknown candidate key"
    continue
  fi
  src="$(candidate_source_dir "${key}")"
  echo "[$(date -Is)] staging ${key} -> ${tag}"
  emit_event "staging" "${key}" "running" "${tag}"
  if ! stage_candidate "${key}" "${tag}" "${src}"; then
    echo "[$(date -Is)] staging failed ${key}; continuing"
    emit_event "staging" "${key}" "failed" "${src}"
    continue
  fi
  emit_event "staging" "${key}" "completed" "${tag}"

  conformal_run="${RUN_ROOT}__${key}__conformal"
  conformal_log="${LOG_DIR}/${conformal_run}.log"
  status_path="models/conformal_gap/${conformal_run}/conformal_reopen_status.json"
  if [[ "${RESUME_EXISTING_CONFORMAL}" == "1" && -f "${status_path}" ]]; then
    echo "[$(date -Is)] conformal ${key}: reuse ${conformal_run}"
    emit_event "conformal" "${key}" "reused" "${conformal_run}"
  else
    echo "[$(date -Is)] conformal ${key}: ${conformal_run}"
    emit_event "conformal" "${key}" "running" "${conformal_run}"
    if ! uv run python scripts/search/run_conformal_reopen_search.py \
      --run-tag "${conformal_run}" \
      --pipeline-profile "${CONFORMAL_PROFILE}" \
      --upstream-canonical-run-tag "${tag}" \
      >"${conformal_log}" 2>&1; then
      echo "[$(date -Is)] conformal failed ${key}; continuing"
      emit_event "conformal" "${key}" "failed" "${conformal_log}"
      continue
    fi
    emit_event "conformal" "${key}" "completed" "${conformal_run}"
  fi

  if [[ ! -f "${status_path}" ]]; then
    echo "[$(date -Is)] missing conformal status ${key}: ${status_path}; continuing"
    emit_event "conformal_status" "${key}" "failed" "${status_path}"
    continue
  fi
  phase1_ns="$(status_value "${status_path}" "phase1_oot_namespace")"
  final_ns="$(status_value "${status_path}" "final_namespace")"
  variants="${PORTFOLIO_VARIANTS}"

  if [[ "${variants}" == *"phase1"* && -n "${phase1_ns}" ]]; then
    intervals="data/processed/conformal_gap/${phase1_ns}/conformal_intervals_mondrian.parquet"
    portfolio_run="${RUN_ROOT}__${key}__phase1__portfolio"
    portfolio_log="${LOG_DIR}/${portfolio_run}.log"
    portfolio_selection="models/portfolio_bound_aware/${portfolio_run}/portfolio_bound_aware_selection.json"
    gate="$(portfolio_gate "${phase1_ns}")"
    echo "[$(date -Is)] portfolio gate phase1 ${key}: ${gate}"
    if [[ "${RESUME_EXISTING_PORTFOLIO}" == "1" && -f "${portfolio_selection}" ]]; then
      echo "[$(date -Is)] portfolio phase1 ${key}: reuse ${portfolio_run}"
      emit_event "portfolio_phase1" "${key}" "reused" "${portfolio_run}"
    elif [[ "${gate}" == "run" ]]; then
      echo "[$(date -Is)] portfolio phase1 ${key}: ${portfolio_run}"
      emit_event "portfolio_phase1" "${key}" "running" "${portfolio_run}"
      if run_portfolio "${intervals}" "${portfolio_run}" >"${portfolio_log}" 2>&1; then
        emit_event "portfolio_phase1" "${key}" "completed" "${portfolio_run}"
      else
        echo "[$(date -Is)] portfolio phase1 failed ${key}; continuing"
        emit_event "portfolio_phase1" "${key}" "failed" "${portfolio_log}"
      fi
    else
      emit_event "portfolio_phase1" "${key}" "skipped" "${gate}"
    fi
  fi

  if [[ "${variants}" == *"final"* && -n "${final_ns}" && "${final_ns}" != "${phase1_ns}" ]]; then
    intervals="data/processed/conformal_gap/${final_ns}/conformal_intervals_mondrian.parquet"
    portfolio_run="${RUN_ROOT}__${key}__final__portfolio"
    portfolio_log="${LOG_DIR}/${portfolio_run}.log"
    portfolio_selection="models/portfolio_bound_aware/${portfolio_run}/portfolio_bound_aware_selection.json"
    gate="$(portfolio_gate "${final_ns}")"
    echo "[$(date -Is)] portfolio gate final ${key}: ${gate}"
    if [[ "${RESUME_EXISTING_PORTFOLIO}" == "1" && -f "${portfolio_selection}" ]]; then
      echo "[$(date -Is)] portfolio final ${key}: reuse ${portfolio_run}"
      emit_event "portfolio_final" "${key}" "reused" "${portfolio_run}"
    elif [[ "${gate}" == "run" ]]; then
      echo "[$(date -Is)] portfolio final ${key}: ${portfolio_run}"
      emit_event "portfolio_final" "${key}" "running" "${portfolio_run}"
      if run_portfolio "${intervals}" "${portfolio_run}" >"${portfolio_log}" 2>&1; then
        emit_event "portfolio_final" "${key}" "completed" "${portfolio_run}"
      else
        echo "[$(date -Is)] portfolio final failed ${key}; continuing"
        emit_event "portfolio_final" "${key}" "failed" "${portfolio_log}"
      fi
    else
      emit_event "portfolio_final" "${key}" "skipped" "${gate}"
    fi
  fi
  emit_event "candidate" "${key}" "completed" "${conformal_run}"
done

echo "[$(date -Is)] champion replacement pipeline completed"
emit_event "pipeline" "all" "completed" "${RUN_ROOT}"
