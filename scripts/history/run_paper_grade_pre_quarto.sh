#!/usr/bin/env bash
# run_paper_grade_pre_quarto.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs the 5 paper-grade cleanup tasks that must complete before migrating to
# Quarto.  Each task writes a clean artifact that the thesis narrative cites.
#
# Tasks (in execution order):
#   T3  update_cost_matrix_threshold.py      ~30 sec   → threshold_semantics.json
#   T5  export_pr_auc_metrics.py             ~30 sec   → model_comparison.json (patched)
#   T2  generate_slice_anomaly_report.py     ~10 min   → pd_slice_performance.json
#   T1  run_fairness_audit.py                ~10 min   → shap_fairness_status.json
#   T4  train_pd_model.py --hpo_enabled false ~30 min  → all PD artifacts (final canonical)
#
# Usage:
#   cd ~/projects/lending-club-risk-project
#   export PATH="$HOME/.local/bin:$PATH"
#   bash scripts/run_paper_grade_pre_quarto.sh 2>&1 | tee logs/pre_quarto_$(date +%Y%m%d_%H%M%S).log
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

LOG_FILE="logs/pre_quarto_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " Paper-Grade Pre-Quarto Pipeline"
echo " Started: $(date)"
echo " Log: $LOG_FILE"
echo "============================================================"

run_step() {
    local step_id="$1"
    local description="$2"
    shift 2
    echo ""
    echo "------------------------------------------------------------"
    echo " STEP $step_id: $description"
    echo " Command: $*"
    echo " Time: $(date)"
    echo "------------------------------------------------------------"
    if "$@"; then
        echo " ✓ STEP $step_id DONE ($(date))"
    else
        echo " ✗ STEP $step_id FAILED (exit $?) — aborting"
        exit 1
    fi
}

# T3 — Cost-matrix threshold on real OOT test set (~30 sec)
run_step T3 "Cost-matrix threshold → threshold_semantics.json" \
    uv run python scripts/update_cost_matrix_threshold.py

# T5 — PR-AUC / Recall@0.35 → model_comparison.json (~30 sec)
run_step T5 "PR-AUC / Recall export → model_comparison.json" \
    uv run python scripts/export_pr_auc_metrics.py

# T2 — IsolationForest + slicing → pd_slice_performance.json (~5-15 min)
run_step T2 "IsolationForest + slice metrics → pd_slice_performance.json" \
    uv run python scripts/generate_slice_anomaly_report.py

# T1 — Fairness audit with SHAP per-group → shap_fairness_status.json (~5-15 min)
run_step T1 "Fairness audit (SHAP per-group) → shap_fairness_status.json" \
    uv run python scripts/run_fairness_audit.py --run-tag paper-grade-pre-quarto

# T4 — Final confirmatory PD training (no HPO, uses existing best params) (~20-40 min)
run_step T4 "Final confirmatory PD run (hpo_enabled=false)" \
    uv run python scripts/train_pd_model.py \
        --config configs/pd_model.yaml \
        --hpo_enabled false

echo ""
echo "============================================================"
echo " ALL STEPS COMPLETE"
echo " Finished: $(date)"
echo " Artifacts to commit:"
echo "   models/threshold_semantics.json        (cost_matrix_result added)"
echo "   data/processed/model_comparison.json   (pr_auc + recall added)"
echo "   models/pd_slice_performance.json       (new)"
echo "   models/isolation_forest.pkl            (new)"
echo "   models/shap_fairness_status.json       (new)"
echo "   models/pd_canonical.cbm                (refreshed)"
echo "   models/pd_model_contract.json          (refreshed)"
echo "   data/processed/test_predictions.parquet (refreshed)"
echo "   data/processed/model_comparison.json   (refreshed by train)"
echo "============================================================"
