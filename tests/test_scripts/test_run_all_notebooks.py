"""Tests for scripts/run_all_notebooks.py."""

from __future__ import annotations

from scripts import run_all_notebooks as nb_runner


def test_discover_notebooks_prefers_explicit_selection(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    notebooks_dir = project_root / "notebooks"
    side_projects = notebooks_dir / "side_projects"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    side_projects.mkdir(parents=True, exist_ok=True)

    (notebooks_dir / "01_alpha.ipynb").write_text("{}", encoding="utf-8")
    (side_projects / "10_gpu.ipynb").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(nb_runner, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(nb_runner, "NOTEBOOKS_DIR", notebooks_dir)

    selected = nb_runner._discover_notebooks(
        execute_all=False,
        include_side_projects=False,
        selected_notebooks=["01_alpha.ipynb", "side_projects/10_gpu.ipynb"],
    )

    assert [p.name for p in selected] == ["01_alpha.ipynb", "10_gpu.ipynb"]


def test_build_write_guard_source_contains_canonical_targets(tmp_path) -> None:
    project_root = tmp_path / "repo"
    guard_root = tmp_path / "repo" / "reports" / "notebook_exec" / "generated" / "01_test"
    source = nb_runner._build_write_guard_source(project_root=project_root, guard_root=guard_root)

    assert '_PROJECT_ROOT / "data" / "processed"' in source
    assert '_PROJECT_ROOT / "models"' in source
    assert '_PROJECT_ROOT / "reports" / "paper_material"' in source
    assert str(guard_root) in source


def test_sandbox_root_is_stable_by_notebook_path(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    notebooks_dir = project_root / "notebooks"
    nb_path = notebooks_dir / "03_pd_modeling.ipynb"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    nb_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(nb_runner, "PROJECT_ROOT", project_root)

    sandbox = nb_runner._sandbox_root_for_notebook(
        nb_path=nb_path,
        output_dir=project_root / "reports" / "notebook_exec",
    )
    expected = (
        project_root / "reports" / "notebook_exec" / "generated" / "notebooks" / "03_pd_modeling"
    )
    assert sandbox == expected.resolve()
