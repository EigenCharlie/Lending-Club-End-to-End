"""Shared dependencies: model singletons, DuckDB connection, etc."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
from catboost import CatBoostClassifier
from loguru import logger

from api.config import (
    CALIBRATOR_FALLBACKS,
    CALIBRATOR_FILE,
    CONFORMAL_RESULTS_FILE,
    DUCKDB_PATH,
    FEATURE_CONFIG_FILE,
    MODEL_DIR,
    PD_MODEL_FALLBACKS,
    PD_MODEL_FILE,
)


@lru_cache(maxsize=1)
def get_pd_model() -> CatBoostClassifier:
    """Load the trained CatBoost PD model (cached singleton)."""
    model = CatBoostClassifier()
    candidates = [PD_MODEL_FILE, *PD_MODEL_FALLBACKS]
    model_path = next(
        (MODEL_DIR / name for name in candidates if (MODEL_DIR / name).exists()), None
    )
    if model_path is None:
        raise FileNotFoundError(f"No PD model found. Checked: {candidates}")
    model.load_model(str(model_path))
    logger.info(f"Loaded PD model from {model_path}")
    return model


@lru_cache(maxsize=1)
def get_calibrator():
    """Load the PD calibrator (Platt Sigmoid or Isotonic)."""
    candidates = [CALIBRATOR_FILE, *CALIBRATOR_FALLBACKS]
    path = next((MODEL_DIR / name for name in candidates if (MODEL_DIR / name).exists()), None)
    if path is None:
        raise FileNotFoundError(f"No calibrator found. Checked: {candidates}")
    with open(path, "rb") as f:
        cal = pickle.load(f)
    logger.info(f"Loaded calibrator from {path}")
    return cal


@lru_cache(maxsize=1)
def get_feature_config() -> dict:
    """Load feature configuration (feature lists, IV scores)."""
    path = Path(MODEL_DIR).parent / "data" / "processed" / FEATURE_CONFIG_FILE
    with open(path, "rb") as f:
        cfg = pickle.load(f)
    logger.info(f"Loaded feature config ({len(cfg.get('CATBOOST_FEATURES', []))} features)")
    return cfg


@lru_cache(maxsize=1)
def get_conformal_results() -> dict[str, Any] | None:
    """Load conformal serving artifact (Mondrian diagnostics + metrics)."""
    path = MODEL_DIR / CONFORMAL_RESULTS_FILE
    if not path.exists():
        logger.warning(
            f"Conformal artifact not found at {path}. Falling back to heuristic intervals."
        )
        return None

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        logger.warning(
            f"Unexpected conformal artifact type: {type(payload)}. "
            "Falling back to heuristic intervals."
        )
        return None

    logger.info(f"Loaded conformal artifact from {path}")
    return payload


def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """Create a read-only DuckDB connection."""
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)
