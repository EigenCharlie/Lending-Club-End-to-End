"""Shared utilities for Streamlit pages: data loading, DuckDB, caching."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DUCKDB_PATH = PROJECT_ROOT / "data" / "lending_club.duckdb"
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt_project"
NOTEBOOK_IMAGE_DIR = PROJECT_ROOT / "reports" / "notebook_images"
NOTEBOOK_IMAGE_MANIFEST = NOTEBOOK_IMAGE_DIR / "manifest.json"


@st.cache_data(ttl=3600)
def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file from data/processed/ with caching."""
    path = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(path)


def download_table(df: pd.DataFrame, filename: str, label: str = "Descargar CSV") -> None:
    """Render a download button for a DataFrame as CSV."""
    st.download_button(label, df.to_csv(index=False), filename, "text/csv")


@st.cache_data(ttl=3600)
def load_json(name: str, directory: str = "data") -> dict:
    """Load a JSON file with caching.

    Args:
        name: File name without extension.
        directory: 'data' for data/processed/, 'models' for models/.
    """
    path = MODEL_DIR / f"{name}.json" if directory == "models" else DATA_DIR / f"{name}.json"
    return json.loads(path.read_text())


@st.cache_resource
def get_duckdb():
    """Get a DuckDB connection (cached resource)."""
    import duckdb

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    # dbt views reference parquet paths relatively (e.g., ../data/processed/...).
    # Setting file_search_path keeps those views resolvable from Streamlit runtime.
    conn.execute(f"SET file_search_path='{DBT_PROJECT_DIR.as_posix()}'")
    return conn


def query_duckdb(sql: str) -> pd.DataFrame:
    """Execute a query against DuckDB and return a DataFrame."""
    conn = get_duckdb()
    return conn.execute(sql).fetchdf()


def suggest_sql_with_grok(
    question: str,
    schema_context: str,
    model: str = "grok-4-fast",
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """Generate a read-only SQL suggestion from a natural-language question.

    This function uses xAI's OpenAI-compatible endpoint and requires:
    - GROK_API_KEY in environment variables.
    """
    api_key = os.getenv("GROK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROK_API_KEY is not configured in environment variables.")

    system_prompt = (
        "You are a SQL assistant for DuckDB. "
        "Return only JSON with keys: sql, rationale. "
        "Rules: SQL must be read-only SELECT. No INSERT/UPDATE/DELETE/DDL. "
        "Use schema-qualified table names exactly as provided."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Schema context:\n{schema_context}\n\n"
        "Return JSON only."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(
            "https://api.x.ai/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    raw_text = data["choices"][0]["message"]["content"]
    parsed = json.loads(raw_text)
    return {
        "sql": str(parsed.get("sql", "")).strip(),
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def format_number(n: float, prefix: str = "", suffix: str = "") -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1_000_000_000:
        return f"{prefix}{n / 1_000_000_000:.1f}B{suffix}"
    if abs(n) >= 1_000_000:
        return f"{prefix}{n / 1_000_000:.1f}M{suffix}"
    if abs(n) >= 1_000:
        return f"{prefix}{n / 1_000:.1f}K{suffix}"
    return f"{prefix}{n:.1f}{suffix}"


def format_pct(n: float, decimals: int = 1) -> str:
    """Format a proportion as a percentage string."""
    return f"{n * 100:.{decimals}f}%"


@st.cache_data(ttl=3600)
def load_notebook_image_manifest() -> list[dict]:
    """Load extracted notebook image manifest."""
    if not NOTEBOOK_IMAGE_MANIFEST.exists():
        return []
    return json.loads(NOTEBOOK_IMAGE_MANIFEST.read_text(encoding="utf-8"))


def get_notebook_image_path(notebook_stem: str, file_name: str) -> Path:
    """Build absolute path to an extracted notebook figure."""
    return NOTEBOOK_IMAGE_DIR / notebook_stem / file_name
