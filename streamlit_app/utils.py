"""Shared utilities for Streamlit pages: data loading, DuckDB, caching."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DUCKDB_PATH = PROJECT_ROOT / "data" / "lending_club.duckdb"
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt_project"
REPORTS_DIR = PROJECT_ROOT / "reports"
DVC_REPORTS_DIR = REPORTS_DIR / "dvc"
NOTEBOOK_IMAGE_DIR = REPORTS_DIR / "notebook_images"
NOTEBOOK_IMAGE_MANIFEST = NOTEBOOK_IMAGE_DIR / "manifest.json"
BASELINE_REGISTRY_PATH = PROJECT_ROOT / "configs" / "baselines" / "core_official_baseline.json"


@st.cache_data(ttl=1800, max_entries=24)
def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file from data/processed/ with caching."""
    path = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(path)


def download_table(df: pd.DataFrame, filename: str, label: str = "Descargar CSV") -> None:
    """Render a download button for a DataFrame as CSV."""
    st.download_button(label, df.to_csv(index=False), filename, "text/csv")


@st.cache_data(ttl=1800, max_entries=64)
def load_json(name: str, directory: str = "data") -> dict:
    """Load a JSON file with caching.

    Args:
        name: File name without extension.
        directory: 'data' for data/processed/, 'models' for models/.
    """
    path = MODEL_DIR / f"{name}.json" if directory == "models" else DATA_DIR / f"{name}.json"
    return json.loads(path.read_text())


@st.cache_data(ttl=300, max_entries=32)
def try_load_parquet(name: str, default: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load parquet if available, otherwise return default/empty DataFrame."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=300, max_entries=64)
def try_load_json(name: str, directory: str = "data", default: dict | None = None) -> dict:
    """Load JSON if available, otherwise return default/empty dict."""
    path = MODEL_DIR / f"{name}.json" if directory == "models" else DATA_DIR / f"{name}.json"
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text())
    except Exception:
        return dict(default or {})


@st.cache_data(ttl=300, max_entries=16)
def try_load_report_parquet(subdir: str, name: str) -> pd.DataFrame:
    """Load parquet from reports/<subdir>/<name>.parquet."""
    path = REPORTS_DIR / subdir / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=16)
def try_load_report_json(subdir: str, name: str) -> dict:
    """Load JSON from reports/<subdir>/<name>.json."""
    path = REPORTS_DIR / subdir / f"{name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=300, max_entries=4)
def load_dvc_metrics_summary() -> dict[str, float]:
    """Load canonical DVC metrics summary exported under reports/dvc/."""
    path = DVC_REPORTS_DIR / "metrics_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    source = data.get("metrics", data) if isinstance(data, dict) else {}
    out: dict[str, float] = {}
    for key, value in source.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


@st.cache_data(ttl=300, max_entries=4)
def load_dvc_conformal_backtest() -> pd.DataFrame:
    """Load canonical conformal coverage backtest exported for DVC plots."""
    path = DVC_REPORTS_DIR / "conformal_coverage_backtest.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=4)
def load_dvc_robustness_frontier() -> pd.DataFrame:
    """Load canonical robustness frontier exported for DVC plots."""
    path = DVC_REPORTS_DIR / "robustness_frontier.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_metric_get(
    metrics: dict[str, float] | None, key: str, default: float | None = None
) -> float | None:
    """Safely fetch a metric key from DVC summary."""
    if not metrics:
        return default
    value = metrics.get(key, default)
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


@st.cache_data(ttl=300, max_entries=4)
def load_official_baseline_registry() -> dict:
    """Load official core baseline registry from configs/baselines."""
    if not BASELINE_REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(BASELINE_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def evaluate_run_tag_coherence(expected_run_tag: str | None, artifacts: dict[str, dict]) -> dict:
    """Check run_tag coherence across selected status artifacts."""
    expected = str(expected_run_tag or "").strip()
    observed: dict[str, str] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for name, payload in artifacts.items():
        if not isinstance(payload, dict):
            missing.append(name)
            continue
        tag = str(payload.get("run_tag", "")).strip()
        if not tag:
            missing.append(name)
            continue
        observed[name] = tag
        if expected and tag != expected:
            mismatched.append(name)
    unique_tags = sorted(set(observed.values()))
    coherent = bool(observed) and not missing and not mismatched and len(unique_tags) == 1
    if expected and not coherent:
        coherent = False
    return {
        "expected_run_tag": expected or None,
        "observed_run_tags": unique_tags,
        "observed_by_artifact": observed,
        "missing_run_tag_artifacts": sorted(missing),
        "mismatched_artifacts": sorted(mismatched),
        "coherent": coherent,
    }


def _collect_test_inventory() -> tuple[int, list[dict[str, int | str]]]:
    """Best-effort collected test inventory using pytest node IDs."""
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists() or shutil.which("uv") is None:
        return 0, []

    try:
        cmd = ["uv", "run", "pytest", "--collect-only", "-q", "-q"]
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        counts: Counter[str] = Counter()
        total = 0
        for raw in proc.stdout.splitlines():
            line = raw.strip()
            if not line.startswith("tests/") or "::" not in line:
                continue
            module_path = line.split("::", maxsplit=1)[0]
            module = module_path.removeprefix("tests/").removesuffix(".py")
            counts[module] += 1
            total += 1
        breakdown = [{"module": module, "tests": int(n)} for module, n in sorted(counts.items())]
        return total, breakdown
    except Exception:
        pass
    return 0, []


@st.cache_data(ttl=300)
def load_runtime_status() -> dict:
    """Load runtime status snapshot with resilient fallbacks."""
    status = try_load_json("runtime_status", directory="data", default={})
    # Always prefer filesystem truth for page count to avoid stale snapshots.
    status["streamlit_pages_total"] = len(
        list((PROJECT_ROOT / "streamlit_app" / "pages").glob("*.py"))
    )
    collected_total, collected_breakdown = _collect_test_inventory()
    if collected_total > 0:
        status["test_suite_total"] = int(collected_total)
        status["test_breakdown"] = collected_breakdown
    else:
        test_total = int(status.get("test_suite_total", 0) or 0)
        breakdown = status.get("test_breakdown", [])
        if not isinstance(breakdown, list):
            breakdown = []
        status["test_suite_total"] = test_total
        status["test_breakdown"] = breakdown
    return status


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
    # In Community Cloud, root-level secrets are injected via Secrets settings.
    # We still support env vars for local and non-Streamlit execution paths.
    api_key = str(st.secrets.get("GROK_API_KEY", os.getenv("GROK_API_KEY", ""))).strip()
    if not api_key:
        raise RuntimeError("GROK_API_KEY is not configured in environment variables.")

    system_prompt = (
        "You are a SQL assistant for DuckDB. "
        "Return only JSON with keys: sql, rationale. "
        "Rules: SQL must be read-only SELECT. No INSERT/UPDATE/DELETE/DDL. "
        "Use schema-qualified table names exactly as provided."
    )
    user_prompt = f"Question:\n{question}\n\nSchema context:\n{schema_context}\n\nReturn JSON only."

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
