"""Guardrails to keep current-facing scripts free of legacy pipeline semantics."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_current_scripts_do_not_reference_legacy_step_names() -> None:
    current_files = [
        PROJECT_ROOT / "scripts" / "run_paper_grade_final.py",
        PROJECT_ROOT / "scripts" / "run_insights_factory.py",
        PROJECT_ROOT / "scripts" / "build_clean_baseline_manifest.py",
        PROJECT_ROOT / "scripts" / "monitor_pipeline_eta.py",
        PROJECT_ROOT / "scripts" / "start_long_run.sh",
        PROJECT_ROOT / "scripts" / "recover_champion_search_run.sh",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "PROJECT_JUSTIFICATION.md",
    ]
    forbidden_tokens = (
        "main_pre",
        "heavy_main",
        "configs/run_profiles/",
    )

    violations: list[str] = []
    for path in current_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {token}")

    assert not violations, f"Found legacy references in current-facing files: {violations}"
