#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

RUN_ROOT="${1:-paper1_bound_pareto_nextwave_2026_05_25_r4}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"

human_seconds_py='
def human_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return "0m"
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
'

echo "== nextwave: ${RUN_ROOT} =="
if [[ -f "${LOG_DIR}/${RUN_ROOT}.pid" ]]; then
  pid="$(cat "${LOG_DIR}/${RUN_ROOT}.pid")"
  ps -p "${pid}" -o pid,ppid,sid,etime,%cpu,%mem,rss,cmd || echo "driver_dead pid=${pid}"
else
  echo "missing driver pid: ${LOG_DIR}/${RUN_ROOT}.pid"
fi

echo
echo "== active related processes =="
pgrep -af "${RUN_ROOT}|run_portfolio_bound_aware_search.py|generate_conformal_intervals.py" \
  | grep -v "monitor_paper1_bound_pareto_nextwave" || true

echo
echo "== conformal variants =="
python - "${RUN_ROOT}" <<PY
from pathlib import Path
import pandas as pd
run_root = __import__("sys").argv[1]
for path in sorted(Path("data/processed/conformal_gap").glob(f"{run_root}__canonical4_*/conformal_intervals_mondrian.parquet")):
    df = pd.read_parquet(path)
    width = df["pd_high_90"] - df["pd_low_90"]
    cov = (df["y_true"] <= df["pd_high_90"]).mean()
    gamma = (df["pd_high_90"] - df["y_pred"]).clip(lower=0, upper=1).mean()
    print(f"{path.parent.name}: n={len(df):,} cov90={cov:.4f} width90={width.mean():.4f} gamma_mean={gamma:.4f}")
PY

echo
echo "== portfolio statuses =="
python - "${RUN_ROOT}" <<PY
from pathlib import Path
import json
import sys
${human_seconds_py}
run_root = sys.argv[1]
paths = sorted(Path("models/portfolio_bound_aware").glob(f"{run_root}__*/portfolio_bound_aware_runtime_status.json"))
if not paths:
    print("no portfolio status yet")
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        "run_tag",
        "phase",
        "state",
        "frontier_completed_units",
        "frontier_total_units",
        "frontier_pct_complete",
        "bound_completed_checks",
        "bound_total_checks",
        "global_pct_complete",
        "eta_sec",
        "latest_policy_mode",
        "latest_risk_tolerance",
        "latest_gamma",
        "selection_reason",
        "selected_realized_total_return",
        "selected_alpha01_exact_pass",
    ]
    compact = {key: payload.get(key) for key in keys if key in payload}
    if "eta_sec" in compact:
        compact["eta_human"] = human_seconds(compact["eta_sec"])
    print(path)
    print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo
echo "== summary table =="
summary="reports/paper_material/paper1/tables/paper1_bound_pareto_nextwave_summary_2026-05-25.csv"
if [[ -f "${summary}" ]]; then
  python - "${summary}" <<'PY'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
print(df.to_string(index=False))
PY
else
  echo "pending: ${summary}"
fi

echo
echo "== log tail =="
tail -60 "${LOG_DIR}/${RUN_ROOT}.log" 2>/dev/null || true
