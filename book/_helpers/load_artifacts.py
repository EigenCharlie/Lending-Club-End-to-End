"""Artifact loading utilities for Quarto book code cells.

Adapted from streamlit_app/utils.py — no Streamlit dependency.
All paths resolve relative to the repository root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
NOTEBOOK_IMAGE_DIR = REPORTS_DIR / "notebook_images"
PUBLICATION_FIGURES_DIR = REPORTS_DIR / "paper_material" / "figures_publication"
CONFIGS_DIR = REPO_ROOT / "configs"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file from ``data/processed/<name>.parquet``."""
    path = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(path)


def try_load_parquet(
    name: str, default: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Load parquet if available, otherwise return *default* or empty DF."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()


def load_json(name: str, directory: str = "data", search_models: bool = False) -> dict:
    """Load a JSON artifact.

    Args:
        name: File name **without** extension.
        directory: ``'data'`` → ``data/processed/``, ``'models'`` → ``models/``.
    """
    use_models = directory == "models" or search_models
    path = MODEL_DIR / f"{name}.json" if use_models else DATA_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(
    name: str,
    directory: str = "data",
    default: dict | None = None,
    search_models: bool = False,
) -> dict:
    """Load JSON if available, otherwise return *default* or empty dict."""
    use_models = directory == "models" or search_models
    path = MODEL_DIR / f"{name}.json" if use_models else DATA_DIR / f"{name}.json"
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def try_load_report_parquet(subdir: str, name: str) -> pd.DataFrame:
    """Load parquet from ``reports/<subdir>/<name>.parquet``."""
    path = REPORTS_DIR / subdir / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def load_yaml(name: str) -> dict:
    """Load a YAML config from ``configs/<name>.yaml``."""
    import yaml

    path = CONFIGS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a float as percentage string, e.g. 0.9257 → '92.57%'."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float | int, decimals: int = 0) -> str:
    """Format a number with thousands separator."""
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def format_money(value: float, decimals: int = 0) -> str:
    """Format as USD, e.g. 1003000 → '$1,003,000'."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.{decimals}f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.{decimals}f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.{decimals}f}K"
    return f"${value:,.{decimals}f}"
