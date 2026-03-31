"""Unit tests for src/models/causal.py."""

from __future__ import annotations

from src.models.causal import summarize_refutation


def test_summarize_refutation_parses_p_value_from_result_text() -> None:
    class _Refutation:
        estimated_effect = 0.1
        new_effect = 0.02
        p_value = None

        def __str__(self) -> str:
            return (
                "Refute: Add a random common cause\n"
                "Estimated effect:0.1\n"
                "New effect:0.02\n"
                "p value:0.94\n"
            )

    refutation = _Refutation()

    summary = summarize_refutation("random_common_cause", refutation)

    assert summary["test"] == "random_common_cause"
    assert summary["p_value"] == 0.94
    assert summary["estimated_effect"] == 0.1
    assert summary["new_effect"] == 0.02
