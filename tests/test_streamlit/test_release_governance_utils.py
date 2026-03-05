"""Tests for run-tag coherence helpers used by Streamlit release governance."""

from __future__ import annotations

from streamlit_app.utils import evaluate_run_tag_coherence


def test_evaluate_run_tag_coherence_passes_when_all_match() -> None:
    out = evaluate_run_tag_coherence(
        "run-abc",
        {
            "governance_status": {"run_tag": "run-abc"},
            "conformal_policy_status": {"run_tag": "run-abc"},
            "fairness_audit_status": {"run_tag": "run-abc"},
        },
    )
    assert out["coherent"] is True
    assert out["mismatched_artifacts"] == []
    assert out["missing_run_tag_artifacts"] == []


def test_evaluate_run_tag_coherence_detects_missing_and_mismatch() -> None:
    out = evaluate_run_tag_coherence(
        "run-abc",
        {
            "governance_status": {"run_tag": "run-abc"},
            "conformal_policy_status": {"run_tag": "run-xyz"},
            "fairness_audit_status": {},
        },
    )
    assert out["coherent"] is False
    assert "conformal_policy_status" in out["mismatched_artifacts"]
    assert "fairness_audit_status" in out["missing_run_tag_artifacts"]
