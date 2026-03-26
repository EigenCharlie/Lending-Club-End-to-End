"""Guardrails for causal narrative pages.

The causal storytelling layer must read metrics from canonical artifacts instead of
embedding stale snapshot numbers or deprecated method names.
"""

from __future__ import annotations

from pathlib import Path

PRIORITY_PAGES = [
    Path("streamlit_app/pages/causal_intelligence.py"),
    Path("streamlit_app/pages/model_laboratory.py"),
    Path("streamlit_app/pages/portfolio_optimizer.py"),
]

FORBIDDEN_PATTERNS = [
    "0.787",
    "5.857M",
    "LinearDML",
    "simulación contrafactual",
]


def test_priority_pages_avoid_stale_causal_snapshot_claims() -> None:
    violations: list[str] = []
    for path in PRIORITY_PAGES:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_PATTERNS:
            if token in text:
                violations.append(f"{path}:{token}")
    assert not violations, (
        "Found stale or non-canonical causal claims in priority pages: "
        + ", ".join(sorted(violations))
    )


def test_priority_pages_reference_canonical_causal_artifacts() -> None:
    required_tokens = [
        "causal_effect_status",
        "causal_policy_rule",
    ]
    missing: list[str] = []
    for token in required_tokens:
        if not any(token in path.read_text(encoding="utf-8") for path in PRIORITY_PAGES):
            missing.append(token)
    assert not missing, "Priority pages must reference canonical causal artifacts: " + ", ".join(
        missing
    )
