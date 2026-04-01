"""Guardrails for archived script wrappers."""

from __future__ import annotations

from pathlib import Path


def test_historical_wrappers_delegate_to_history_namespace() -> None:
    wrappers = {
        Path("scripts/export_pr_auc_metrics.py"): 'history" / "export_pr_auc_metrics.py',
        Path(
            "scripts/generate_slice_anomaly_report.py"
        ): 'history" / "generate_slice_anomaly_report.py',
        Path(
            "scripts/update_cost_matrix_threshold.py"
        ): 'history" / "update_cost_matrix_threshold.py',
    }
    for path, needle in wrappers.items():
        text = path.read_text(encoding="utf-8")
        assert "Historical compatibility wrapper" in text
        assert needle in text


def test_historical_shell_wrapper_delegates() -> None:
    text = Path("scripts/run_paper_grade_pre_quarto.sh").read_text(encoding="utf-8")
    assert "HISTORICAL wrapper" in text
    assert "history/run_paper_grade_pre_quarto.sh" in text
