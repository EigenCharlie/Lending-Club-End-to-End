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
