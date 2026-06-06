"""Guardrails for the retired local CRPTO mini-book surface."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_CRPTO = Path("/mnt/c/Users/carlos/Documents/Paper_CRPTO")
RETIREMENT_MEMO = REPO_ROOT / "docs/research/crpto_retirement_and_paper4_role_2026-06-06.md"


def test_local_crpto_mini_book_surface_is_retired() -> None:
    retired_paths = [
        REPO_ROOT / "papers/paper_crpto_book",
        REPO_ROOT / "papers/paper1_estrella",
        REPO_ROOT / "book/chapters/14-paper-estrella",
    ]
    for path in retired_paths:
        assert not path.exists(), f"Retired CRPTO surface still exists: {path}"

    quarto_config = (REPO_ROOT / "book/_quarto.yml").read_text(encoding="utf-8")
    assert "paper_crpto_book" not in quarto_config
    assert "paper1_estrella" not in quarto_config
    assert "14-paper-estrella" not in quarto_config


def test_external_crpto_source_of_truth_has_required_surfaces() -> None:
    required = [
        "book/_quarto.yml",
        "paper/CRPTO_ijds.qmd",
        "paper/supplement_ijds.qmd",
        "docs/research/papers_tesis_deep_audit_2026-06-06.md",
        "reports/crpto/literature/papers_tesis_source_matrix_2026-06-06.csv",
        "reports/crpto/literature/papers_tesis_figure_caption_index_2026-06-06.csv",
        "reports/crpto/literature/papers_tesis_curated_visual_sinks_2026-06-06.csv",
        "scripts/build_papers_tesis_deep_audit.py",
    ]
    for rel in required:
        assert (EXTERNAL_CRPTO / rel).exists(), f"Missing external CRPTO source: {rel}"


def test_retirement_memo_sets_boundary_for_paper4() -> None:
    text = RETIREMENT_MEMO.read_text(encoding="utf-8")

    assert "fuente de verdad para CRPTO" in text
    assert "Paper 4 queda como living lab" in text
    assert "No se reabre el champion CRPTO desde este repo" in text
    assert "No reconstruir `book/chapters/14-paper-estrella`" in text
