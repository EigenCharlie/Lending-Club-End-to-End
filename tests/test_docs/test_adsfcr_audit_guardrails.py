"""Guardrails for the ADSFCR audit document."""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path("docs/ADSFCR_AUDIT_AND_MONOTONIC_CHALLENGER_PLAN_2026-03-29.md")

REQUIRED_LIVE_LINKS = [
    "https://leanpub.com/adsfcr",
    "https://leanpub.com/pdrmwr",
    "https://leanpub.com/crmwn",
    "https://andrijadj.shinyapps.io/vasicek_distribution/",
    "https://andrija-djurovic.github.io/adsfcr/model_dev_and_vld/bootstrap_ht.html",
    "https://andrija-djurovic.github.io/adsfcr/model_dev_and_vld/hl_vs_zscore.html#/",
    "https://andrija-djurovic.github.io/adsfcr/effective_interest_rate/eir.html",
    "https://andrija-djurovic.github.io/adsfcr/loan_repayment_plan/lrp.html",
]


def test_adsfcr_audit_doc_exists() -> None:
    assert DOC_PATH.exists(), f"Missing ADSFCR audit document: {DOC_PATH}"


def test_adsfcr_audit_doc_covers_all_blob_links() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    blob_links = re.findall(r"https://github\.com/andrija-djurovic/adsfcr/blob/main/[^\s)]+", text)
    assert len(blob_links) >= 90, (
        "ADSFCR audit document must include the 90 README blob links. "
        f"Found only {len(blob_links)}."
    )


def test_adsfcr_audit_doc_lists_required_live_links() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    missing = [url for url in REQUIRED_LIVE_LINKS if url not in text]
    assert not missing, "ADSFCR audit document is missing live links: " + ", ".join(missing)


def test_adsfcr_audit_doc_contains_decision_fields() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()
    for token in (
        "implementar ahora",
        "usar como referencia metodologica",
        "documentar pero no implementar",
        "descartar por ahora",
        "requiere reentrenamiento",
        "requiere rerun downstream",
    ):
        assert token in text
