"""API configuration and paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DUCKDB_PATH = PROJECT_ROOT / "data" / "lending_club.duckdb"

# Model file names (canonical-first with legacy fallback)
PD_MODEL_FILE = "pd_canonical.cbm"
CALIBRATOR_FILE = "pd_canonical_calibrator.pkl"
PD_MODEL_FALLBACKS = ("pd_catboost.cbm", "pd_catboost_tuned.cbm")
CALIBRATOR_FALLBACKS = ("pd_calibrator.pkl",)
FEATURE_CONFIG_FILE = "feature_config.pkl"
CONFORMAL_RESULTS_FILE = "conformal_results_mondrian.pkl"

# API
API_TITLE = "Credit Risk API"
API_VERSION = "0.1.0"
CORS_ORIGINS = ["http://localhost:8501", "http://localhost:3000"]
