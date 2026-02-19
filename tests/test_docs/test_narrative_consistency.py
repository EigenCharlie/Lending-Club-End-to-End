"""Tests that prevent stale hardcoded policy-check claims in docs/pages."""

from __future__ import annotations

from pathlib import Path

TARGETS = [
    Path("SESSION_STATE.md"),
    Path("docs/RUNBOOK.md"),
    Path("streamlit_app/pages/thesis_contribution.py"),
    Path("streamlit_app/pages/research_landscape.py"),
    Path("docs/conformal_prediction_README.md"),
]

UI_TARGETS = [
    Path("streamlit_app/pages/executive_summary.py"),
    Path("streamlit_app/pages/glossary_fundamentals.py"),
    Path("streamlit_app/pages/model_laboratory.py"),
    Path("streamlit_app/pages/survival_analysis.py"),
    Path("streamlit_app/pages/thesis_contribution.py"),
]

STALE_UI_PATTERNS = [
    "AUC=0.7187",
    "Cobertura 90%=0.9197",
    "Cobertura 95%=0.9608",
    "C=0.6769",
    "C=0.6838",
    "el 91.97% de las veces",
]


def test_no_stale_7_over_7_claims() -> None:
    violations: list[str] = []
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        if "7/7" in text:
            violations.append(str(path))

    assert not violations, (
        "Found stale hardcoded '7/7' policy claims in: "
        + ", ".join(sorted(violations))
        + ". Keep policy-check messaging dynamic or snapshot-neutral."
    )


def test_no_stale_ui_metric_snapshots() -> None:
    violations: list[str] = []
    for path in UI_TARGETS:
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_UI_PATTERNS:
            if pattern in text:
                violations.append(f"{path}:{pattern}")
    assert not violations, (
        "Found stale hardcoded UI metrics in: "
        + ", ".join(sorted(violations))
        + ". Load metrics from canonical artifacts instead of hardcoding snapshot numbers."
    )
