"""Guardrails for the canonical documentation traceability refresh."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md")


def test_traceability_doc_exists() -> None:
    assert DOC_PATH.exists(), f"Missing canonical traceability document: {DOC_PATH}"


def test_traceability_doc_mentions_live_adsfcr_layers() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()
    required = [
        "monotonicity_audit_status.json",
        "pd_backtesting_status.json",
        "bootstrap_validation_status.json",
        "pd_validation_interpretation_status.json",
        "calibration_mapping_status.json",
        "ifrs9_diagnostics_status.json",
        "encoding_stability_status.json",
        "model_shift_status.json",
        "governance_status.json",
        "fairness_audit_status.json",
    ]
    missing = [token for token in required if token not in text]
    assert not missing, "Traceability document is missing live layers: " + ", ".join(missing)


def test_traceability_doc_marks_legacy_claims() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()
    for token in (
        "legacy claims",
        "monotonic challenger was useful but not promoted",
        "approval-based",
        "current canonical champion",
    ):
        assert token in text
