"""Guardrails for the Quarto book structure and editorial hygiene."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import yaml

BOOK_DIR = Path("book")
QUARTO_CONFIG = BOOK_DIR / "_quarto.yml"
CHAPTERS_DIR = BOOK_DIR / "chapters"
ARCHIVED_CHAPTER_MANIFEST = BOOK_DIR / "_archived_chapter_pages.yml"
PAPER4_PAGE_REGISTRY = Path("reports/paper_material/paper4/tables/paper4_quarto_page_registry.csv")
PAPER4_PAGE_REGISTRY_RELATIVE = PAPER4_PAGE_REGISTRY.as_posix()
LATEX_TABLE_FILTER = BOOK_DIR / "filters" / "latex-table-widths.lua"
LATEX_PREAMBLE = BOOK_DIR / "latex" / "preamble.tex"

PLACEHOLDER_PATTERNS = [
    "Contenido pendiente",
    "TODO",
    "FIXME",
]

RETIRED_QUARTO_SURFACES = [
    "chapters/13-advanced-topics/13f-gpu-edge-research.qmd",
    "chapters/13-advanced-topics/13g-gpu-edge-results.qmd",
    "chapters/16-paper-mondrian/index.qmd",
    "chapters/17-paper-gpu/index.qmd",
    "chapters/18-paper-quantum/index.qmd",
    "chapters/B-gpu-benchmarks.qmd",
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


def _paper4_page_registry_entries() -> dict[str, dict[str, str]]:
    assert PAPER4_PAGE_REGISTRY.exists(), (
        f"Missing canonical Paper 4 page registry: {PAPER4_PAGE_REGISTRY}"
    )
    with PAPER4_PAGE_REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["path"].removeprefix("book/"): row
        for row in rows
        if row.get("path", "").startswith("book/chapters/19-paper-mega-extension/")
    }


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


def test_pdf_tables_have_wrapping_and_breakable_identifier_guardrails() -> None:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    pdf_filters = config["format"]["pdf"].get("filters", [])
    assert "filters/latex-table-widths.lua" in pdf_filters
    assert LATEX_TABLE_FILTER.exists()
    filter_text = LATEX_TABLE_FILTER.read_text(encoding="utf-8")
    assert "ColWidthDefault" in filter_text
    assert "\\nolinkurl" in filter_text

    preamble = LATEX_PREAMBLE.read_text(encoding="utf-8")
    assert "\\usepackage{xurl}" in preamble
    assert "\\AtBeginEnvironment{longtable}" in preamble


def test_repo_artifact_links_are_rewritten_for_both_rendered_formats() -> None:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    filter_name = "filters/repo-artifact-links.lua"

    assert filter_name in config["format"]["html"]["filters"]
    assert filter_name in config["format"]["pdf"]["filters"]

    source = (BOOK_DIR / filter_name).read_text(encoding="utf-8")
    assert "https://github.com/EigenCharlie/Lending-Club-End-to-End" in source
    assert "source_ref" in source
    assert 'and "tree" or "blob"' in source


def test_active_pyepo_legacy_link_resolves_to_versioned_artifact() -> None:
    chapter = BOOK_DIR / "chapters" / "19-paper-mega-extension" / "19cc-v39-pyepo-real-suite.qmd"
    expected = Path(
        "reports/paper_material/paper_estrella/pyepo/"
        "paper_estrella_pyepo137_wls_topk_paired_20260528"
    )
    text = chapter.read_text(encoding="utf-8")
    assert expected.exists()
    assert expected.name in text
    assert "crpto_external_pyepo137_wls_topk_paired_20260528" not in text


def test_all_quarto_chapter_pages_are_referenced_in_book_config() -> None:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    chapter_entries = _walk_chapter_entries(config["book"]["chapters"])
    actual_files = {path.relative_to(BOOK_DIR).as_posix() for path in _book_qmd_files()}
    archived_entries = _archived_chapter_entries()
    paper4_registry = _paper4_page_registry_entries()
    archived_files = set(archived_entries)
    orphaned_archive_entries = sorted(archived_files - actual_files)
    registered_archive_entries = sorted(archived_files & chapter_entries)
    missing = sorted(actual_files - chapter_entries - archived_files)
    malformed_archive_entries = sorted(
        path
        for path, row in archived_entries.items()
        if row.get("render_policy") != "intentionally_not_rendered"
        or not row.get("archive_reason")
        or row.get("source_decision_artifact") != PAPER4_PAGE_REGISTRY_RELATIVE
    )
    missing_registry_membership = sorted(archived_files - set(paper4_registry))
    malformed_registry_membership = sorted(
        path
        for path in archived_files & set(paper4_registry)
        if paper4_registry[path].get("rendered_in_quarto", "").strip().lower() != "false"
        or paper4_registry[path].get("role") != "historical_archive"
        or paper4_registry[path].get("path_exists", "").strip().lower() != "true"
    )
    configured_paper4 = {
        path for path in chapter_entries if path.startswith("chapters/19-paper-mega-extension/")
    }
    official_registry_paper4 = {
        path
        for path, row in paper4_registry.items()
        if row.get("rendered_in_quarto", "").strip().lower() == "true"
    }
    malformed_official_paper4 = sorted(
        path
        for path in official_registry_paper4
        if paper4_registry[path].get("role") not in {"official_curated", "official_appendix"}
        or paper4_registry[path].get("target_surface") != "quarto_official_chapter"
        or paper4_registry[path].get("path_exists", "").strip().lower() != "true"
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
    assert not missing_registry_membership, (
        "Found archived Quarto entries absent from paper4_quarto_page_registry.csv: "
        + ", ".join(missing_registry_membership)
    )
    assert not malformed_registry_membership, (
        "Found archived Quarto entries without rendered=False, "
        "role=historical_archive and path_exists=True in the canonical registry: "
        + ", ".join(malformed_registry_membership)
    )
    assert len(configured_paper4) == 12
    assert official_registry_paper4 == configured_paper4, (
        "Paper 4 official registry differs from book/_quarto.yml: "
        f"registry_only={sorted(official_registry_paper4 - configured_paper4)}, "
        f"quarto_only={sorted(configured_paper4 - official_registry_paper4)}"
    )
    assert not malformed_official_paper4, "Malformed official Paper 4 registry rows: " + ", ".join(
        malformed_official_paper4
    )
    assert not missing, (
        "Found standalone Quarto chapter files missing from book/_quarto.yml: " + ", ".join(missing)
    )


def test_retired_research_surfaces_are_not_registered() -> None:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    chapter_entries = _walk_chapter_entries(config["book"]["chapters"])
    registered = sorted(set(RETIRED_QUARTO_SURFACES) & chapter_entries)
    assert not registered, (
        "Retired research surfaces are still registered in book/_quarto.yml: "
        + ", ".join(registered)
    )
