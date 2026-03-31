"""Guardrails for the post-promotion Quarto narrative refresh."""

from __future__ import annotations

from pathlib import Path

PRIORITY_CHAPTERS = [
    Path("book/chapters/01-executive-map.qmd"),
    Path("book/chapters/06-pd-modeling/06b-catboost-tuned.qmd"),
    Path("book/chapters/06-pd-modeling/06d-model-comparison-champion.qmd"),
    Path("book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd"),
    Path("book/chapters/F-rerun-v2-refactor.qmd"),
]

FORBIDDEN_PATTERNS = [
    "no terminó promoviéndose al champion",
    "no sirvió para reemplazar al champion",
    "challenger monotónico no pasó sus guardrails",
    "paper-grade-2026-03-13-final-heavy-2026-03-13-230650",
]


def test_priority_chapters_avoid_stale_monotonic_narrative() -> None:
    violations: list[str] = []
    for path in PRIORITY_CHAPTERS:
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_PATTERNS:
            if token in text:
                violations.append(f"{path}:{token}")
    assert not violations, (
        "Found stale monotonic-champion narrative in priority Quarto chapters: "
        + ", ".join(sorted(violations))
    )


def test_quarto_refresh_mentions_new_diagnostic_layers() -> None:
    text = (
        Path("book/chapters/10-ifrs9-governance/10e-model-risk-management.qmd")
        .read_text(encoding="utf-8")
        .lower()
    )
    for token in (
        "c2st",
        "monotonicity audit",
        "pd backtesting",
        "bootstrap validation",
        "pd validation interpretation",
        "calibration mapping",
        "ifrs9 diagnostics",
        "encoding stability",
        "model shift",
    ):
        assert token in text
