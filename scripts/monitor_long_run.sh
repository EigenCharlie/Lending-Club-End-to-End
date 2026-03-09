#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${1:?Usage: bash scripts/monitor_long_run.sh <run_tag>}"
RUN_DIR="reports/run_logs/${RUN_TAG}"
INFO_FILE="${RUN_DIR}/run_info.json"
HB_FILE="${RUN_DIR}/heartbeat.json"

watch -n 20 "
cd ${REPO_ROOT@Q} || exit 1
echo '== ' \$(date) '=='
echo
echo 'run_tag: ${RUN_TAG}'
if [ -f '${INFO_FILE}' ]; then
  pid=\$(python3 -c \"import json; print(json.load(open('${INFO_FILE}'))['pid'])\" 2>/dev/null)
  if [ -n \"\${pid}\" ]; then
    echo \"pid: \${pid}\"
    if kill -0 \"\${pid}\" 2>/dev/null; then
      echo 'process: alive'
    else
      echo 'process: dead'
    fi
  else
    echo 'pid: (parse error)'
  fi
else
  echo 'pid: (no run_info.json yet)'
fi
echo
echo '== heartbeat =='
if [ -f '${HB_FILE}' ]; then
  python - <<'PY'
import json
from pathlib import Path
p = Path('${HB_FILE}')
d = json.loads(p.read_text())
for k in ['state','current_step','last_update_utc','orchestrator_pid','active_child_pid','final_exit_code']:
    if k in d:
        print(f'{k}: {d[k]}')
if 'failed_steps' in d:
    print('failed_steps:', d['failed_steps'])
PY
else
  echo '(missing)'
fi
echo
echo '== status files =='
for f in ${RUN_DIR}/status/*.exit; do
  [ -e \"\$f\" ] || continue
  printf '%-22s %s\n' \"\$(basename \"\$f\")\" \"\$(cat \"\$f\")\"
done
echo
echo '== master tail =='
tail -n 15 ${RUN_DIR}/master.log 2>/dev/null || true
echo
echo '== current log tails =='
for f in preflight main_pre heavy_main causal cate_portfolio post_core rapids notebooks; do
  lf=${RUN_DIR}/\${f}.log
  if [ -f \"\$lf\" ]; then
    echo \"--- \${f}.log\"
    tail -n 6 \"\$lf\"
  fi
done
echo
echo '== heavy processes =='
ps -eo pid,etimes,rss,%mem,%cpu,cmd --sort=-%cpu | rg -i 'train_pd_model|run_survival_analysis|run_all_benchmarks.py|estimate_causal_effects|run_all_notebooks.py|run_long_pipeline.py' | head -n 15 || true
"
