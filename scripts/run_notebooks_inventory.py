"""Notebooks classification inventory.

Classifies notebooks into the pipeline-first editorial taxonomy:
  - reusable_evidence: reusable evidence notebooks (01-06, 08)
  - research_labs: causal + side projects
  - historical_demo: historical end-to-end notebook kept for provenance
  - paper_notebooks: active local paper-support notebooks
  - explainability_lab: explainability deep dive (13)

Outputs:
    models/notebooks_inventory.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
SCHEMA_VERSION = "2026-03-31.1"

# Canonical mapping of notebook number → chapter info
NOTEBOOK_META = {
    "01": {
        "chapter": "EDA y Dataset",
        "quarto_cap": 3,
        "artifacts": ["eda_summary.json", "dataset_dictionary.json"],
    },
    "02": {
        "chapter": "Feature Engineering",
        "quarto_cap": 4,
        "artifacts": ["feature_config.pkl", "feature_importance_iv.json"],
    },
    "03": {
        "chapter": "Modelado PD",
        "quarto_cap": 5,
        "artifacts": ["pd_canonical.cbm", "model_comparison.json"],
    },
    "04": {
        "chapter": "Conformal Prediction",
        "quarto_cap": 5,
        "artifacts": ["conformal_intervals_mondrian.parquet"],
    },
    "05": {
        "chapter": "Series Temporales",
        "quarto_cap": 6,
        "artifacts": ["ts_forecasts.parquet", "time_series_status.json"],
    },
    "06": {
        "chapter": "Análisis de Supervivencia",
        "quarto_cap": 6,
        "artifacts": ["km_curve_data.parquet"],
    },
    "07": {
        "chapter": "Inferencia Causal",
        "quarto_cap": 7,
        "artifacts": ["cate_estimates.parquet", "causal_effect_status.json"],
    },
    "08": {
        "chapter": "Optimización de Portafolio",
        "quarto_cap": 8,
        "artifacts": ["portfolio_allocations.parquet", "portfolio_robustness_frontier.parquet"],
    },
    "09": {
        "chapter": "Pipeline End-to-End",
        "quarto_cap": None,
        "artifacts": ["pipeline_summary.json"],
    },
    "10": {
        "chapter": "CRPTO frozen reference (external project)",
        "quarto_cap": None,
        "artifacts": ["spo_comparison_status.json"],
    },
    "11": {
        "chapter": "Paper 2: IFRS9 E2E",
        "quarto_cap": 12,
        "artifacts": ["ifrs9_scenario_summary.parquet", "sicr_trigger_optimization.parquet"],
    },
    "13": {
        "chapter": "Explicabilidad del Modelo",
        "quarto_cap": None,
        "artifacts": ["explainability_global.parquet", "shap_summary.parquet"],
    },
}


def _classify_notebook(path: Path) -> dict:
    name = path.stem
    # Extract number prefix
    parts = name.split("_")
    num = parts[0] if parts else "00"
    is_side = "side_project" in str(path)

    if is_side or num == "07":
        category = "research_labs"
    elif num in {"09", "10", "12"}:
        category = "historical_demo"
    elif num == "11":
        category = "paper_notebooks"
    elif num == "13":
        category = "explainability_lab"
    else:
        category = "reusable_evidence"

    meta = NOTEBOOK_META.get(num, {})

    reuse_status_map = {
        "reusable_evidence": "evidence_reusable",
        "research_labs": "research_only",
        "historical_demo": "historical_reference",
        "paper_notebooks": "paper_material",
        "explainability_lab": "explainability_reference",
    }

    return {
        "filename": path.name,
        "stem": name,
        "number": num,
        "category": category,
        "chapter": meta.get("chapter", name.replace("_", " ").title()),
        "quarto_chapter": meta.get("quarto_cap"),
        "key_artifacts": meta.get("artifacts", []),
        "reuse_status": reuse_status_map[category],
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    logger.info("Notebooks classification inventory")

    notebooks = sorted(NOTEBOOKS_DIR.glob("**/*.ipynb"))
    records = [_classify_notebook(nb) for nb in notebooks]

    by_category = {
        "reusable_evidence": [r for r in records if r["category"] == "reusable_evidence"],
        "research_labs": [r for r in records if r["category"] == "research_labs"],
        "historical_demo": [r for r in records if r["category"] == "historical_demo"],
        "paper_notebooks": [r for r in records if r["category"] == "paper_notebooks"],
        "explainability_lab": [r for r in records if r["category"] == "explainability_lab"],
    }

    logger.info(
        "Total: {} | reusable: {} | research: {} | historical: {} | paper: {} | explainability: {}".format(
            len(records),
            len(by_category["reusable_evidence"]),
            len(by_category["research_labs"]),
            len(by_category["historical_demo"]),
            len(by_category["paper_notebooks"]),
            len(by_category["explainability_lab"]),
        )
    )

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "total_notebooks": len(records),
        "by_category": {key: len(value) for key, value in by_category.items()},
        "notebooks": records,
        "classification_rules": {
            "reusable_evidence": "Notebooks 01-06 and 08: reusable evidence connected to the live thesis stack",
            "research_labs": "Notebook 07 and notebooks/side_projects/: research-only exploratory labs",
            "historical_demo": "Notebook 09: archived end-to-end demonstration notebook",
            "paper_notebooks": "Notebooks 10-12: paper-support notebooks executed in reference mode",
            "explainability_lab": "Notebook 13: explainability deep dive retained as a focused lab",
        },
    }

    out = MODELS_DIR / "notebooks_inventory.json"
    out.write_text(json.dumps(status, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Saved {out} ({len(records)} notebooks)")

    # Also write a parquet for Streamlit display
    df = pd.DataFrame(records)
    df["key_artifacts"] = df["key_artifacts"].apply(lambda x: ", ".join(x))
    df.to_parquet(PROJECT_ROOT / "data" / "processed" / "notebooks_inventory.parquet", index=False)
    logger.success("Saved notebooks_inventory.parquet")


if __name__ == "__main__":
    main()
