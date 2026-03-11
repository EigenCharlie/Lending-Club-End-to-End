"""Coverage checks for TOBoML concept mapping across Streamlit pages."""

from __future__ import annotations

from pathlib import Path

from streamlit_app.content.concept_map import (
    CONCEPT_REGISTRY,
    PAGE_CONCEPT_MAP,
    REQUIRED_CONCEPT_KEYS,
    build_concept_index_rows,
)

PAGES_DIR = Path(__file__).resolve().parents[2] / "streamlit_app" / "pages"


def test_required_concepts_are_registered() -> None:
    assert set(REQUIRED_CONCEPT_KEYS) == set(CONCEPT_REGISTRY), (
        "REQUIRED_CONCEPT_KEYS must match CONCEPT_REGISTRY keys exactly"
    )
    assert len(REQUIRED_CONCEPT_KEYS) >= 18


def test_concept_map_covers_all_streamlit_pages() -> None:
    page_ids = {p.stem for p in PAGES_DIR.glob("*.py")}
    mapped_pages = set(PAGE_CONCEPT_MAP)
    assert page_ids == mapped_pages, (
        f"Concept map/page mismatch: missing={sorted(page_ids - mapped_pages)} "
        f"unknown={sorted(mapped_pages - page_ids)}"
    )
    empty_pages = sorted(page_id for page_id, keys in PAGE_CONCEPT_MAP.items() if not keys)
    assert not empty_pages, f"Pages without concept coverage: {empty_pages}"


def test_all_mapped_concepts_exist_in_registry() -> None:
    unknown_keys = sorted(
        {
            key
            for concept_keys in PAGE_CONCEPT_MAP.values()
            for key in concept_keys
            if key not in CONCEPT_REGISTRY
        }
    )
    assert not unknown_keys, f"Unknown concept keys in PAGE_CONCEPT_MAP: {unknown_keys}"


def test_each_registered_concept_is_used_by_at_least_one_page() -> None:
    mapped_keys = {key for concept_keys in PAGE_CONCEPT_MAP.values() for key in concept_keys}
    missing = sorted(set(CONCEPT_REGISTRY) - mapped_keys)
    assert not missing, f"Concepts never mapped to any page: {missing}"


def test_critical_gaps_have_explicit_page_targets() -> None:
    expected_min_targets = {
        "aleatoric": {"executive_summary", "uncertainty_quantification"},
        "epistemic": {"executive_summary", "uncertainty_quantification"},
        "confidence_interval": {"ab_testing_simulation", "paper_2_ifrs9_e2e"},
        "prediction_interval": {"uncertainty_quantification", "ifrs9_provisions"},
        "mcar_mar_mnar": {"data_architecture", "feature_engineering"},
        "nested_cv": {"model_laboratory", "research_landscape"},
        "class_imbalance": {"model_laboratory", "research_landscape"},
        "extrapolation": {"portfolio_optimizer", "paper_1_cp_robust_opt"},
        "convex_hull": {"portfolio_optimizer"},
        "optimizer_curse": {"model_laboratory", "tech_stack", "research_landscape"},
        "no_free_lunch": {"thesis_defense", "gpu_benchmark", "tech_stack"},
        "c2st": {"model_governance"},
        "concept_drift": {"model_governance"},
    }
    for concept_key, expected_pages in expected_min_targets.items():
        pages_for_key = {
            page_id
            for page_id, concept_keys in PAGE_CONCEPT_MAP.items()
            if concept_key in concept_keys
        }
        missing = sorted(expected_pages - pages_for_key)
        assert not missing, f"Concept `{concept_key}` is missing expected page targets: {missing}"


def test_glossary_page_is_master_index_for_all_concepts() -> None:
    glossary_keys = set(PAGE_CONCEPT_MAP.get("glossary_fundamentals", ()))
    assert glossary_keys == set(CONCEPT_REGISTRY), (
        "glossary_fundamentals should expose every concept in CONCEPT_REGISTRY"
    )


def test_concept_index_rows_are_traceable() -> None:
    rows = build_concept_index_rows()
    assert len(rows) == len(CONCEPT_REGISTRY)
    for row in rows:
        assert row["key"] in CONCEPT_REGISTRY
        assert row["paginas_objetivo"], f"Missing page traceability for concept `{row['key']}`"
        assert int(row["n_paginas"]) > 0
