"""Guardrails for active vs historical documentation layout."""

from __future__ import annotations

from pathlib import Path


def test_historical_docs_live_under_history_namespace() -> None:
    assert not Path("docs/PROMOTION_DOSSIER_2026-03-01.md").exists()
    assert not Path("docs/OFFICIAL_RERUN_MASTER_PLAN_2026-02-27.md").exists()
    assert not Path("docs/DEPLOY_STREAMLIT_FREE.md").exists()
    assert Path("docs/history/PROMOTION_DOSSIER_2026-03-01.md").exists()
    assert Path("docs/history/OFFICIAL_RERUN_MASTER_PLAN_2026-02-27.md").exists()
    assert Path("docs/history/DEPLOY_STREAMLIT_FREE.md").exists()


def test_research_docs_live_under_research_namespace() -> None:
    assert not Path("docs/conformal_prediction_research_2026.md").exists()
    assert not Path("docs/conformal_prediction_quick_reference.md").exists()
    assert not Path("docs/conformal_libraries_comparison.md").exists()
    assert Path("docs/research/conformal_prediction_research_2026.md").exists()
    assert Path("docs/research/conformal_prediction_quick_reference.md").exists()
    assert Path("docs/research/conformal_libraries_comparison.md").exists()
