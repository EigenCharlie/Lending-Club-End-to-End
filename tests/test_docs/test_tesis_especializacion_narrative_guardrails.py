"""Narrative guardrails for specialization thesis page and mirror report."""

from __future__ import annotations

from pathlib import Path

PAGE_PATH = Path("streamlit_app/pages/tesis_especializacion.py")
REPORT_PATH = Path("reports/tesis_especializacion.md")
CONTEXT_PATH = Path("streamlit_app/content/tesis_especializacion_context.py")

FORBIDDEN_SNAPSHOT_PATTERNS = [
    "AUC=0.7187",
    "Cobertura 90%=0.9197",
    "Cobertura 95%=0.9608",
    "7/7",
]


def test_tesis_assets_avoid_stale_snapshot_claims() -> None:
    violations: list[str] = []
    for path in [PAGE_PATH, REPORT_PATH]:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_SNAPSHOT_PATTERNS:
            if pattern in text:
                violations.append(f"{path}:{pattern}")
    assert not violations, (
        "Found stale snapshot claims in thesis assets: "
        + ", ".join(sorted(violations))
        + ". Keep thesis narrative tied to current canonical artifacts."
    )


def test_tesis_assets_require_run_tag_and_generated_at_utc_traceability() -> None:
    missing: list[str] = []
    for path in [PAGE_PATH, REPORT_PATH]:
        text = path.read_text(encoding="utf-8")
        for token in ["run_tag", "generated_at_utc"]:
            if token not in text:
                missing.append(f"{path}:{token}")
    assert not missing, "Missing required traceability tokens in thesis assets: " + ", ".join(
        sorted(missing)
    )


def test_tesis_context_has_absolute_dates_and_evaluation_concept() -> None:
    text = CONTEXT_PATH.read_text(encoding="utf-8")
    required_tokens = [
        "2025-12-11",
        "2026-02-25",
        "2026-02-28",
        "Aprobado con ajustes",
    ]
    missing = [token for token in required_tokens if token not in text]
    assert not missing, f"Missing required timeline/evaluation tokens: {missing}"
