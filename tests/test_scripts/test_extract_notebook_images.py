"""Tests for scripts/extract_notebook_images.py."""

from __future__ import annotations

from scripts import extract_notebook_images as extractor


def test_resolve_notebook_dir_prefers_executed_dir(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    notebooks_dir = project_root / "notebooks"
    executed_dir = project_root / "reports" / "notebook_exec" / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    executed_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(extractor, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(extractor, "NOTEBOOK_DIR", notebooks_dir)
    monkeypatch.setattr(extractor, "EXECUTED_NOTEBOOK_DIR", executed_dir)

    assert extractor._resolve_notebook_dir(None) == executed_dir


def test_resolve_notebook_dir_uses_arg_when_provided(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    custom_dir = project_root / "custom_nb_dir"
    custom_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(extractor, "PROJECT_ROOT", project_root)

    resolved = extractor._resolve_notebook_dir("custom_nb_dir")
    assert resolved == custom_dir.resolve()
