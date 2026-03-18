"""Notebooks classification inventory.

Classifies all notebooks into:
  - core_thesis: notebooks 01-09 (main pipeline chapters)
  - paper_research: notebooks 10-13 (paper materials)
  - side_projects: notebooks in side_projects/ (non-core exploratory)

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
SCHEMA_VERSION = "2026-03-17.1"

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
        "chapter": "Paper 1: CP + Robust Optimization",
        "quarto_cap": 11,
        "artifacts": ["spo_comparison_status.json"],
    },
    "11": {
        "chapter": "Paper 2: IFRS9 E2E",
        "quarto_cap": 12,
        "artifacts": ["ifrs9_scenario_summary.parquet", "sicr_trigger_optimization.parquet"],
    },
    "12": {
        "chapter": "Paper 3: Mondrian CP",
        "quarto_cap": 13,
        "artifacts": ["conformal_variant_benchmark.parquet"],
    },
    "13": {
        "chapter": "Explicabilidad del Modelo",
        "quarto_cap": None,
        "artifacts": ["explainability_global.parquet", "shap_summary.parquet"],
    },
}

SIDE_PROJECT_META = {
    "10_rapids": {
        "chapter": "GPU Benchmark RAPIDS",
        "type": "side_project",
        "artifacts": ["benchmark_summary_all_sections.parquet"],
    }
}


def _classify_notebook(path: Path) -> dict:
    name = path.stem
    # Extract number prefix
    parts = name.split("_")
    num = parts[0] if parts else "00"
    is_side = "side_project" in str(path)

    category = (
        "side_projects" if is_side else ("core_thesis" if int(num) <= 9 else "paper_research")
    )
    meta = NOTEBOOK_META.get(num, {})

    return {
        "filename": path.name,
        "stem": name,
        "number": num,
        "category": category,
        "chapter": meta.get("chapter", name.replace("_", " ").title()),
        "quarto_chapter": meta.get("quarto_cap"),
        "key_artifacts": meta.get("artifacts", []),
        "reuse_status": (
            "evidence_reusable"
            if category == "core_thesis"
            else "paper_material"
            if category == "paper_research"
            else "exploratory_side_project"
        ),
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    logger.info("Notebooks classification inventory")

    notebooks = sorted(NOTEBOOKS_DIR.glob("**/*.ipynb"))
    records = [_classify_notebook(nb) for nb in notebooks]

    core = [r for r in records if r["category"] == "core_thesis"]
    paper = [r for r in records if r["category"] == "paper_research"]
    side = [r for r in records if r["category"] == "side_projects"]

    logger.info(
        f"Total: {len(records)} | core: {len(core)} | paper: {len(paper)} | side: {len(side)}"
    )

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "total_notebooks": len(records),
        "by_category": {
            "core_thesis": len(core),
            "paper_research": len(paper),
            "side_projects": len(side),
        },
        "notebooks": records,
        "classification_rules": {
            "core_thesis": "Notebooks 01-09: main pipeline chapters, evidence reusable for thesis",
            "paper_research": "Notebooks 10-13: paper-specific material and deep dives",
            "side_projects": "Notebooks in side_projects/: exploratory, not part of core pipeline",
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
