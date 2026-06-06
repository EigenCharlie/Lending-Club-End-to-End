"""Guardrails for Paper Estrella retirement in the parent book."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK_CONFIG = REPO_ROOT / "book/_quarto.yml"
RETIREMENT_MEMO = REPO_ROOT / "docs/research/crpto_retirement_and_paper4_role_2026-06-06.md"
EXTERNAL_CRPTO = Path("/mnt/c/Users/carlos/Documents/Paper_CRPTO")


def _book_chapter_entries(items: list[object]) -> set[str]:
    entries: set[str] = set()
    for item in items:
        if isinstance(item, str):
            entries.add(item)
        elif isinstance(item, dict):
            nested = item.get("chapters")
            if isinstance(nested, list):
                entries |= _book_chapter_entries(nested)
    return entries


def test_paper_estrella_is_no_longer_rendered_in_parent_book() -> None:
    config = yaml.safe_load(BOOK_CONFIG.read_text(encoding="utf-8"))
    chapters = _book_chapter_entries(config["book"]["chapters"])

    assert not any("14-paper-estrella" in chapter for chapter in chapters)
    assert not (REPO_ROOT / "book/chapters/14-paper-estrella").exists()


def test_paper_estrella_editorial_work_moved_to_external_crpto() -> None:
    required_external = [
        "book/chapters/06b-guia-editorial-claims.qmd",
        "book/chapters/14-release.qmd",
        "book/chapters/24-bibliografia-crpto-actualizada.qmd",
        "book/chapters/25-reviewer-map.qmd",
        "docs/research/papers_tesis_deep_audit_2026-06-06.md",
        "reports/crpto/literature/papers_tesis_source_matrix_2026-06-06.csv",
    ]
    for rel in required_external:
        assert (EXTERNAL_CRPTO / rel).exists(), f"Missing migrated CRPTO artifact: {rel}"

    memo = RETIREMENT_MEMO.read_text(encoding="utf-8")
    assert "autocontenido" in memo
    assert "no depende de este repo" in memo


def test_local_standalone_paper_estrella_manuscript_is_retired() -> None:
    assert not (REPO_ROOT / "papers/paper1_estrella").exists()
    assert not (REPO_ROOT / "papers/paper1_estrella/paper_estrella_manuscript.qmd").exists()
