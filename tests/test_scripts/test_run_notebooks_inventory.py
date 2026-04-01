"""Tests for scripts/run_notebooks_inventory.py taxonomy."""

from __future__ import annotations

from scripts import run_notebooks_inventory as inventory_mod


def _classify(path_str: str) -> dict:
    return inventory_mod._classify_notebook(inventory_mod.PROJECT_ROOT / path_str)


def test_classify_reusable_evidence_notebook() -> None:
    record = _classify("notebooks/03_pd_modeling.ipynb")
    assert record["category"] == "reusable_evidence"
    assert record["reuse_status"] == "evidence_reusable"


def test_classify_causal_and_side_projects_as_research_labs() -> None:
    causal = _classify("notebooks/07_causal_inference.ipynb")
    side = _classify("notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb")
    assert causal["category"] == "research_labs"
    assert side["category"] == "research_labs"
    assert causal["reuse_status"] == "research_only"
    assert side["reuse_status"] == "research_only"


def test_classify_historical_and_paper_notebooks() -> None:
    historical = _classify("notebooks/09_end_to_end_pipeline.ipynb")
    paper = _classify("notebooks/11_paper2_ifrs9_e2e.ipynb")
    explainability = _classify("notebooks/13_model_explainability.ipynb")

    assert historical["category"] == "historical_demo"
    assert historical["reuse_status"] == "historical_reference"
    assert paper["category"] == "paper_notebooks"
    assert paper["reuse_status"] == "paper_material"
    assert explainability["category"] == "explainability_lab"
    assert explainability["reuse_status"] == "explainability_reference"
