"""Guardrails for the executable ADSFCR backlog document."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/ADSFCR_EXECUTABLE_BACKLOG_2026-03-30.md")


def test_adsfcr_backlog_doc_exists() -> None:
    assert DOC_PATH.exists(), f"Missing ADSFCR backlog document: {DOC_PATH}"


def test_adsfcr_backlog_doc_mentions_top_remaining_items() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()
    required = [
        "bootstrap hypothesis tests",
        "calibration mapping diagnostics",
        "model shift",
        "lgd survival",
        "blockwise",
    ]
    missing = [token for token in required if token not in text]
    assert not missing, "ADSFCR backlog document is missing key remaining items: " + ", ".join(
        missing
    )
