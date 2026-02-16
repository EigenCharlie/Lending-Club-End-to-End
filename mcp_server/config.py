"""MCP server configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DUCKDB_PATH = PROJECT_ROOT / "data" / "lending_club.duckdb"
