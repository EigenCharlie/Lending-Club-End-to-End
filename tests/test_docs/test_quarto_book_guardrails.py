"""Guardrails for the Quarto book structure and editorial hygiene."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

BOOK_DIR = Path("book")
QUARTO_CONFIG = BOOK_DIR / "_quarto.yml"
CHAPTERS_DIR = BOOK_DIR / "chapters"
ARCHIVED_CHAPTER_MANIFEST = BOOK_DIR / "_archived_chapter_pages.yml"

PLACEHOLDER_PATTERNS = [
    "Contenido pendiente",
    "TODO",
    "FIXME",
]


def _walk_chapter_entries(items: list[object]) -> set[str]:
    paths: set[str] = set()
    for item in items:
        if isinstance(item, str):
            paths.add(item)
        elif isinstance(item, dict):
            for key in ("chapter", "part"):
                value = item.get(key)
                if isinstance(value, str):
                    paths.add(value)
            nested = item.get("chapters")
            if isinstance(nested, list):
                paths |= _walk_chapter_entries(nested)
    return paths


def _book_qmd_files() -> list[Path]:
    return sorted(CHAPTERS_DIR.rglob("*.qmd"))


def _archived_chapter_entries() -> dict[str, dict[str, object]]:
    if not ARCHIVED_CHAPTER_MANIFEST.exists():
        return {}
    payload = yaml.safe_load(ARCHIVED_CHAPTER_MANIFEST.read_text(encoding="utf-8")) or {}
    rows = payload.get("archived_chapter_pages", [])
    archive: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        if isinstance(path, str):
            archive[path] = row
    return archive


def test_quarto_book_has_no_placeholder_content() -> None:
    violations: list[str] = []
    for path in _book_qmd_files():
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in PLACEHOLDER_PATTERNS):
            violations.append(str(path))
    assert not violations, "Found placeholder content in book chapters: " + ", ".join(violations)


def test_quarto_includes_only_private_snippets() -> None:
    pattern = re.compile(r"\{\{<\s*include\s+([^ >]+)\s*>}}")
    violations: list[str] = []
    for path in BOOK_DIR.rglob("*.qmd"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.findall(text):
            include_target = match.strip("\"'")
            if not Path(include_target).name.startswith("_"):
                violations.append(f"{path}:{include_target}")
    assert not violations, (
        "Found include shortcodes pointing to non-private snippets: " + ", ".join(violations)
    )


def test_all_quarto_chapter_pages_are_referenced_in_book_config() -> None:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    chapter_entries = _walk_chapter_entries(config["book"]["chapters"])
    actual_files = {path.relative_to(BOOK_DIR).as_posix() for path in _book_qmd_files()}
    archived_entries = _archived_chapter_entries()
    archived_files = set(archived_entries)
    orphaned_archive_entries = sorted(archived_files - actual_files)
    registered_archive_entries = sorted(archived_files & chapter_entries)
    missing = sorted(actual_files - chapter_entries - archived_files)
    malformed_archive_entries = sorted(
        path
        for path, row in archived_entries.items()
        if row.get("render_policy") != "intentionally_not_rendered"
        or not row.get("archive_reason")
        or not row.get("source_decision_artifact")
    )

    assert not orphaned_archive_entries, (
        "Found archived Quarto entries without chapter files: "
        + ", ".join(orphaned_archive_entries)
    )
    assert not registered_archive_entries, (
        "Found archived Quarto entries also registered in book/_quarto.yml: "
        + ", ".join(registered_archive_entries)
    )
    assert not malformed_archive_entries, (
        "Found archived Quarto entries missing required archive metadata: "
        + ", ".join(malformed_archive_entries)
    )
    assert not missing, (
        "Found standalone Quarto chapter files missing from book/_quarto.yml: " + ", ".join(missing)
    )
