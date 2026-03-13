"""Shared utilities for Streamlit pages: data loading, DuckDB, caching."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime
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
PRIMARY_BASELINE_REGISTRY_PATH = (
    PROJECT_ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
)
LEGACY_BASELINE_REGISTRY_PATH = (
    PROJECT_ROOT / "configs" / "baselines" / "core_official_baseline.json"
)
GPU_REPLAY_DIR = REPORTS_DIR / "gpu_replay"
GPU_INSIGHTS_DIR = REPORTS_DIR / "gpu_insights"
OFFICIAL_GPU_REPLAY_TAG = "2026-03-09-official-gpu-replay-rapids-final"
OFFICIAL_RAPIDS_TRADEOFF_FULL_TAG = "2026-03-10-tradeoff-full-v3"
OFFICIAL_RAPIDS_IFRS9_CORRELATED_TAG = "ifrs9-mc-correlated-smoke"
OFFICIAL_GPU_INSIGHT_TAG = "2026-03-10-rapids-insight-v4"
OFFICIAL_PD_RAPIDS_BENCHMARK_TAG = "2026-03-10-pd-rapids-benchmark-v1"


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


def _coerce_float(value: object) -> float | None:
    """Best-effort cast to float; return None for non-finite/non-numeric values."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if pd.notna(out) else None


def _get_nested(payload: dict, path: tuple[str, ...]) -> object | None:
    """Traverse nested dict by path, returning None when missing."""
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_first_float(payload: dict, candidates: tuple[tuple[str, ...], ...]) -> float | None:
    """Return first numeric value found across candidate paths."""
    for candidate in candidates:
        value = _coerce_float(_get_nested(payload, candidate))
        if value is not None:
            return value
    return None


def _load_pipeline_metrics_fallback() -> dict[str, float]:
    """Fallback mapping from pipeline_summary.json to DVC KPI keys."""
    path = DATA_DIR / "pipeline_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    metric_candidates: dict[str, tuple[tuple[str, ...], ...]] = {
        "pd.auc": (("pd_auc",), ("pd_model", "final_auc"), ("flattened_summary", "pd_auc")),
        "pd.gini": (("pd_gini",), ("pd_model", "final_gini"), ("flattened_summary", "pd_gini")),
        "pd.brier": (("pd_brier",), ("pd_model", "final_brier"), ("flattened_summary", "pd_brier")),
        "pd.ece": (("pd_ece",), ("pd_model", "final_ece"), ("flattened_summary", "pd_ece")),
        "conformal.coverage90": (
            ("conformal_coverage_90",),
            ("conformal", "coverage_90"),
            ("flattened_summary", "conformal_coverage_90"),
        ),
        "conformal.coverage95": (
            ("conformal_coverage_95",),
            ("conformal", "coverage_95"),
            ("flattened_summary", "conformal_coverage_95"),
        ),
        "conformal.avg_width90": (
            ("pipeline", "interval_width_mean"),
            ("interval_width_mean",),
        ),
        "ifrs9.ecl_baseline": (
            ("pipeline", "ecl_expected"),
            ("ifrs9", "ecl_baseline"),
        ),
        "ifrs9.ecl_severe": (
            ("pipeline", "ecl_conservative"),
            ("ifrs9", "ecl_severe"),
        ),
        "ifrs9.severe_uplift_pct": (
            ("ifrs9", "severe_uplift_pct"),
        ),
        "optimization.robust_return": (
            ("pipeline", "robust_return"),
            ("optimization", "robust_return"),
        ),
        "optimization.nonrobust_return": (
            ("pipeline", "nonrobust_return"),
            ("optimization", "nonrobust_return"),
        ),
        "optimization.price_of_robustness": (
            ("pipeline", "price_of_robustness"),
            ("optimization", "price_of_robustness"),
        ),
        "causal.ate": (
            ("causal", "ate"),
            ("causal_ate",),
            ("flattened_summary", "causal_ate"),
        ),
        "causal.cate_mean": (
            ("causal", "cate_mean"),
            ("causal_cate_mean",),
            ("flattened_summary", "causal_cate_mean"),
        ),
        "causal.bootstrap_p05_net": (
            ("causal", "bootstrap_p05_net"),
            ("causal_bootstrap_p05_net",),
            ("flattened_summary", "causal_bootstrap_p05_net"),
        ),
        "causal.oot_p05_monthly_net": (
            ("causal", "oot_p05_monthly_net"),
            ("causal_oot_p05_monthly_net",),
            ("flattened_summary", "causal_oot_p05_monthly_net"),
        ),
    }

    out: dict[str, float] = {}
    for metric_key, candidates in metric_candidates.items():
        value = _extract_first_float(payload, candidates)
        if value is not None:
            out[metric_key] = value

    if "ifrs9.severe_uplift_pct" not in out:
        baseline = out.get("ifrs9.ecl_baseline")
        severe = out.get("ifrs9.ecl_severe")
        if baseline is not None and severe is not None and baseline != 0:
            out["ifrs9.severe_uplift_pct"] = ((severe / baseline) - 1.0) * 100.0

    return out


@st.cache_data(ttl=300, max_entries=4)
def load_dvc_metrics_summary() -> dict[str, float]:
    """Load canonical DVC metrics summary exported under reports/dvc/."""
    path = DVC_REPORTS_DIR / "metrics_summary.json"
    out: dict[str, float] = {}

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        source = data.get("metrics", data) if isinstance(data, dict) else {}
        if isinstance(source, dict):
            for key, value in source.items():
                metric_value = _coerce_float(value)
                if metric_value is None:
                    continue
                out[str(key)] = metric_value

    if out:
        return out
    return _load_pipeline_metrics_fallback()


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
    """Load the official CPU baseline registry."""
    for path in (PRIMARY_BASELINE_REGISTRY_PATH, LEGACY_BASELINE_REGISTRY_PATH):
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


@st.cache_data(ttl=300, max_entries=8)
def load_gpu_replay_summary(run_tag: str = OFFICIAL_GPU_REPLAY_TAG) -> dict:
    """Load a GPU replay run summary."""
    path = GPU_REPLAY_DIR / run_tag / "run_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=300, max_entries=16)
def try_load_gpu_replay_json(run_tag: str, relative_path: str) -> dict:
    """Load a JSON artifact from reports/gpu_replay/<run_tag>/."""
    path = GPU_REPLAY_DIR / run_tag / relative_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=300, max_entries=16)
def try_load_gpu_replay_parquet(run_tag: str, relative_path: str) -> pd.DataFrame:
    """Load a parquet artifact from reports/gpu_replay/<run_tag>/."""
    path = GPU_REPLAY_DIR / run_tag / relative_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=16)
def try_load_gpu_insight_json(run_tag: str, relative_path: str) -> dict:
    """Load a JSON artifact from reports/gpu_insights/<run_tag>/."""
    path = GPU_INSIGHTS_DIR / run_tag / relative_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_command_stage(command: str) -> str | None:
    mappings = {
        "scripts/train_pd_model.py": "pd",
        "scripts/train_lgd_ead.py": "lgd_ead",
        "scripts/optimize_portfolio.py": "portfolio",
        "scripts/optimize_portfolio_tradeoff.py": "tradeoff",
        "scripts/simulate_ab_test.py": "ab",
        "scripts/optimize_cate_portfolio.py": "cate_portfolio",
        "scripts/run_ifrs9_sensitivity.py": "ifrs9_sensitivity",
    }
    for needle, stage in mappings.items():
        if needle in command:
            return stage
    return None


def _parse_ts(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


@st.cache_data(ttl=300, max_entries=4)
def load_cpu_stage_durations(run_tag: str | None = None) -> pd.DataFrame:
    """Parse subphase durations from the official CPU master log."""
    if not run_tag:
        registry = load_official_baseline_registry()
        run_tag = str(registry.get("official_run_tag") or registry.get("run_tag") or "")
    if not run_tag:
        return pd.DataFrame()

    path = REPORTS_DIR / "run_logs" / run_tag / "master.log"
    if not path.exists():
        return pd.DataFrame()

    pattern = re.compile(
        r"^\[(?P<ts>.+?)\] STEP_SUBPHASE_(?P<kind>START|END) name=(?P<name>\S+) idx=(?P<idx>\d+/\d+)(?: cmd=(?P<cmd>.*)| ec=(?P<ec>\d+))?$"
    )
    starts: dict[tuple[str, str], tuple[datetime, str]] = {}
    rows: list[dict[str, object]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        ts = _parse_ts(match.group("ts"))
        if ts is None:
            continue
        key = (match.group("name"), match.group("idx"))
        if match.group("kind") == "START":
            starts[key] = (ts, match.group("cmd") or "")
            continue
        start_info = starts.get(key)
        if not start_info:
            continue
        started_at, cmd = start_info
        stage = _extract_command_stage(cmd)
        if not stage:
            continue
        rows.append(
            {
                "stage": stage,
                "cpu_seconds": (ts - started_at).total_seconds(),
                "cpu_command": cmd,
                "cpu_run_tag": run_tag,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_stage_comparison(
    gpu_run_tag: str = OFFICIAL_GPU_REPLAY_TAG, cpu_run_tag: str | None = None
) -> pd.DataFrame:
    """Join official CPU timings with the RAPIDS replay stage timings."""
    gpu_summary = load_gpu_replay_summary(gpu_run_tag)
    stage_results = gpu_summary.get("stage_results", [])
    gpu_rows = []
    for result in stage_results:
        stage = str(result.get("stage", ""))
        gpu_seconds = _coerce_float(result.get("duration_seconds"))
        if stage and gpu_seconds is not None:
            gpu_rows.append(
                {
                    "stage": stage,
                    "gpu_seconds": gpu_seconds,
                    "gpu_run_tag": gpu_run_tag,
                    "peak_gpu_util": _coerce_float(
                        result.get("gpu_metrics", {}).get("peak_gpu_util")
                    ),
                    "avg_gpu_util": _coerce_float(
                        result.get("gpu_metrics", {}).get("avg_gpu_util")
                    ),
                    "peak_memory_used_mb": _coerce_float(
                        result.get("gpu_metrics", {}).get("peak_memory_used_mb")
                    ),
                    "avg_memory_used_mb": _coerce_float(
                        result.get("gpu_metrics", {}).get("avg_memory_used_mb")
                    ),
                    "peak_power_draw_w": _coerce_float(
                        result.get("gpu_metrics", {}).get("peak_power_draw_w")
                    ),
                }
            )

    gpu_df = pd.DataFrame(gpu_rows)
    cpu_df = load_cpu_stage_durations(cpu_run_tag)
    if gpu_df.empty:
        return cpu_df
    if cpu_df.empty or "stage" not in cpu_df.columns:
        merged = gpu_df.copy()
        merged["cpu_seconds"] = None
        merged["cpu_command"] = None
        merged["cpu_run_tag"] = None
    else:
        merged = gpu_df.merge(cpu_df, on="stage", how="left")
    if "cpu_seconds" in merged.columns:
        merged["speedup_gpu_vs_cpu"] = merged["cpu_seconds"] / merged["gpu_seconds"]
    else:
        merged["speedup_gpu_vs_cpu"] = None
    merged["speedup_gpu_vs_cpu"] = merged["speedup_gpu_vs_cpu"].replace([pd.NA], None)
    return merged


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_ifrs9_mc_tail_metrics(
    gpu_run_tag: str = OFFICIAL_GPU_REPLAY_TAG,
) -> dict:
    """Load the Monte Carlo IFRS9 GPU summary."""
    return try_load_gpu_replay_json(
        gpu_run_tag,
        "artifacts/data/processed/ifrs9_mc_tail_metrics.json",
    )


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_tradeoff_full_policy(
    run_tag: str = OFFICIAL_RAPIDS_TRADEOFF_FULL_TAG,
) -> dict:
    """Load champion policy from the full cuOpt tradeoff run."""
    return try_load_gpu_replay_json(run_tag, "artifacts/models/champion_portfolio_policy.json")


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_tradeoff_full_ab_status(
    run_tag: str = OFFICIAL_RAPIDS_TRADEOFF_FULL_TAG,
) -> dict:
    """Load A/B status from the full cuOpt tradeoff run."""
    return try_load_gpu_replay_json(run_tag, "artifacts/models/ab_simulation_status.json")


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_ifrs9_correlated_metrics(
    run_tag: str = OFFICIAL_RAPIDS_IFRS9_CORRELATED_TAG,
) -> dict:
    """Load the correlated Monte Carlo IFRS9 smoke summary."""
    return try_load_gpu_replay_json(
        run_tag,
        "artifacts/data/processed/ifrs9_mc_tail_metrics.json",
    )


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_insight_summary(
    run_tag: str = OFFICIAL_GPU_INSIGHT_TAG,
) -> dict:
    """Load the RAPIDS insight-factory summary."""
    return try_load_gpu_insight_json(run_tag, "artifacts/run_summary.json")


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_insight_stage_table(
    run_tag: str = OFFICIAL_GPU_INSIGHT_TAG,
) -> pd.DataFrame:
    """Return a normalized stage table for RAPIDS insight-factory results."""
    summary = load_rapids_insight_summary(run_tag)
    stages = summary.get("stages", [])
    if not isinstance(stages, list):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for row in stages:
        if not isinstance(row, dict):
            continue
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, max_entries=8)
def load_rapids_insight_detail(
    name: str,
    run_tag: str = OFFICIAL_GPU_INSIGHT_TAG,
) -> pd.DataFrame:
    """Load applied insight artifacts from RAPIDS insight-factory runs."""
    path = GPU_INSIGHTS_DIR / run_tag / "artifacts" / "data" / "processed" / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_pd_benchmark_summary(
    run_tag: str = OFFICIAL_PD_RAPIDS_BENCHMARK_TAG,
) -> dict:
    """Load the split PD RAPIDS benchmark summary."""
    path = GPU_REPLAY_DIR / run_tag / "artifacts" / "run_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=300, max_entries=4)
def load_rapids_pd_benchmark_stage_table(
    run_tag: str = OFFICIAL_PD_RAPIDS_BENCHMARK_TAG,
) -> pd.DataFrame:
    """Return the stage-level PD benchmark comparison table."""
    payload = load_rapids_pd_benchmark_summary(run_tag)
    stages = payload.get("stages", [])
    if not isinstance(stages, list):
        return pd.DataFrame()
    return pd.DataFrame(stages)


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


def page_error_boundary(page_name: str):
    """Context manager that catches data-loading errors and shows user-friendly messages."""
    from contextlib import contextmanager

    @contextmanager
    def _boundary():
        try:
            yield
        except FileNotFoundError as exc:
            st.error(
                f"Archivo de datos no encontrado en **{page_name}**. "
                "Ejecute el pipeline antes de usar esta pagina.",
                icon=":material/folder_off:",
            )
            st.caption(f"Detalle: `{exc}`")
            st.stop()
        except KeyError as exc:
            st.error(
                f"Clave o columna faltante en **{page_name}**: `{exc}`",
                icon=":material/key_off:",
            )
            st.caption(
                "Los artefactos pueden estar desactualizados. Ejecute el pipeline."
            )
            st.stop()
        except IndexError as exc:
            st.error(
                f"Datos insuficientes en **{page_name}**: `{exc}`",
                icon=":material/data_alert:",
            )
            st.caption(
                "El artefacto existe pero no contiene las filas esperadas. "
                "Re-ejecute el pipeline para regenerar los datos."
            )
            st.stop()
        except Exception as exc:
            st.error(
                f"Error inesperado en **{page_name}**: `{type(exc).__name__}: {exc}`",
                icon=":material/error:",
            )
            st.caption(
                "Sugerencia: `uv run python scripts/run_canonical_rebuild.py` para regenerar artefactos."
            )
            st.stop()

    return _boundary()


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
