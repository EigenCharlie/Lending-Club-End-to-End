"""Guardrails for Paper Estrella retirement in the parent book."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK_CONFIG = REPO_ROOT / "book/_quarto.yml"
RETIREMENT_MEMO = REPO_ROOT / "docs/research/crpto_retirement_and_paper4_role_2026-06-06.md"
EXTERNAL_CONTRACT = REPO_ROOT / "docs/research/crpto_external_contract_2026-07-20.yml"
DEFAULT_EXTERNAL_CRPTO = Path("/mnt/c/Users/carlos/Documents/Paper_CRPTO")


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


def _external_crpto_or_skip() -> Path:
    configured = os.environ.get("CRPTO_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_EXTERNAL_CRPTO
    if not root.is_dir():
        pytest.skip("Set CRPTO_ROOT to inspect the external active companion")
    return root


def test_paper_estrella_is_no_longer_rendered_in_parent_book() -> None:
    config = yaml.safe_load(BOOK_CONFIG.read_text(encoding="utf-8"))
    chapters = _book_chapter_entries(config["book"]["chapters"])

    assert not any("14-paper-estrella" in chapter for chapter in chapters)
    assert not (REPO_ROOT / "book/chapters/14-paper-estrella").exists()
    assert not (REPO_ROOT / "book/_quarto-core.yml").exists()
    assert not (REPO_ROOT / "book/index-core.qmd").exists()
    assert not (REPO_ROOT / "scripts/serve_book_core.py").exists()


def test_paper_estrella_work_is_represented_by_a_stable_external_contract() -> None:
    contract = yaml.safe_load(EXTERNAL_CONTRACT.read_text(encoding="utf-8"))
    paths = {descriptor["path"] for descriptor in contract["surfaces"].values()}

    assert contract["repository"]["commit"] == "69095e05beae282701b4ea38aa69da26a209106f"
    assert {
        "paper/CRPTO_ijds.qmd",
        "paper/supplement_ijds.qmd",
        "docs/research/active_claims_2026-07-14.md",
        "configs/ijds_active_evidence_sources.yaml",
        "configs/ijds_claim_ledger.yaml",
        "reports/crpto/ijds_binary_geometry_frontier_v4_evidence.json",
    } <= paths
    assert contract["local_integration"]["substantive_imports_into_crpto"] == []


def test_optional_external_companion_renders_only_current_claim_surfaces() -> None:
    root = _external_crpto_or_skip()
    config = yaml.safe_load((root / "book/_quarto.yml").read_text(encoding="utf-8"))
    chapters = config["book"]["chapters"]

    assert chapters == [
        "index.qmd",
        {
            "part": "Companion del manuscrito IJDS activo",
            "chapters": [
                "chapters/06-blueprint-manuscrito.qmd",
                "chapters/06b-guia-editorial-claims.qmd",
            ],
        },
        "references.qmd",
    ]
    retired_contract = (root / "book/chapters/README.md").read_text(encoding="utf-8")
    assert "32 QMD" not in retired_contract  # inventory is explicit, not an unaudited prose count
    assert "Fuentes históricas no autoritativas" in retired_contract


def test_local_standalone_paper_estrella_manuscript_is_retired() -> None:
    assert not (REPO_ROOT / "papers/paper1_estrella").exists()
    memo = RETIREMENT_MEMO.read_text(encoding="utf-8")
    assert "autocontenido" in memo
    assert "no depende de este repo" in memo
