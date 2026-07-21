"""Tests for scripts/run_paper_notebook_suite.py."""

from __future__ import annotations

from scripts import run_paper_notebook_suite as paper_suite


def test_build_command_includes_expected_notebooks_and_flags(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    cmd = paper_suite.build_command(
        repo_root=repo_root,
        timeout_s=1200,
        output_dir="reports/notebook_exec",
    )

    cmd_str = " ".join(cmd)
    assert str(repo_root / "scripts" / "run_all_notebooks.py") in cmd_str
    assert "--notebook 11_paper2_ifrs9_e2e.ipynb" in cmd_str
    assert "10_paper1_cp_robust_opt.ipynb" not in cmd_str
    assert "12_paper3_mondrian.ipynb" not in cmd_str
    assert "--timeout 1200" in cmd_str
    assert "--inplace false" in cmd_str
    assert "--output-dir reports/notebook_exec" in cmd_str
