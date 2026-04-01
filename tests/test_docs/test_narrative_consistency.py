"""Tests that prevent stale hardcoded policy-check claims in docs/pages."""

from __future__ import annotations

from pathlib import Path

TARGETS = [
    Path("SESSION_STATE.md"),
    Path("docs/RUNBOOK.md"),
    Path("streamlit_app/pages/model_laboratory.py"),
    Path("streamlit_app/pages/causal_intelligence.py"),
    Path("docs/conformal_prediction_README.md"),
    Path("docs/history/PROMOTION_DOSSIER_2026-03-01.md"),
]

UI_TARGETS = [
    Path("streamlit_app/pages/data_story.py"),
    Path("streamlit_app/pages/model_laboratory.py"),
    Path("streamlit_app/pages/model_interpretability.py"),
    Path("streamlit_app/pages/portfolio_optimizer.py"),
    Path("streamlit_app/pages/causal_intelligence.py"),
]

STALE_UI_PATTERNS = [
    "AUC=0.7187",
    "Cobertura 90%=0.9197",
    "Cobertura 95%=0.9608",
    "C=0.6769",
    "C=0.6838",
    "el 91.97% de las veces",
]


def test_session_state_points_to_current_official_baseline() -> None:
    text = Path("SESSION_STATE.md").read_text(encoding="utf-8")
    assert "champion-2026-03-12-mega-definitive" in text
    assert "configs/baselines/canonical_operational_baseline.json" in text


def test_threshold_narrative_separates_internal_vs_operational_roles() -> None:
    for path in (
        Path("SESSION_STATE.md"),
        Path("docs/RUNBOOK.md"),
        Path("docs/history/ANALISIS_TOBOML_VS_PROYECTO_2026-03-13.md"),
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "threshold interno" in text
        assert "threshold operativo" in text


def test_crepes_predict_p_not_described_as_probabilities() -> None:
    for path in (
        Path("docs/research/conformal_libraries_comparison.md"),
        Path("docs/research/conformal_prediction_quick_reference.md"),
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "predict_p" in text
        assert "p-values" in text


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


def test_historical_dossier_is_explicitly_marked_as_historical() -> None:
    text = Path("docs/history/PROMOTION_DOSSIER_2026-03-01.md").read_text(encoding="utf-8").lower()
    assert "historical snapshot" in text
    assert "do not treat it as the live canonical policy state" in text
