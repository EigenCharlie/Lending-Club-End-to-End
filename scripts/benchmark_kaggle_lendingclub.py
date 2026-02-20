"""Build a public Kaggle + literature benchmark for Lending Club modeling.

This script does not require Kaggle login. It uses public web endpoints and
rendered notebook pages to collect:
- notebook metadata (kernel_id, run_id, title, author, URL)
- detected model families (LR, CatBoost, XGBoost, LightGBM, RF, NN)
- hyperparameter optimization methods (Optuna, GridSearch, RandomSearch, etc.)
- explainability methods (SHAP, LIME, PDP/ICE, permutation importance)
- reported AUC values parsed from rendered notebook HTML

Usage:
    uv run python scripts/benchmark_kaggle_lendingclub.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

KAGGLE_CODE_URL = "https://www.kaggle.com/code"
LIST_KERNELS_ENDPOINT = "https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels"
LIST_KERNEL_VERSIONS_ENDPOINT = (
    "https://www.kaggle.com/api/i/kernels.KernelsService/ListKernelVersions"
)

SEARCH_TERMS = [
    "lending club",
    "lending club credit risk",
    "lendingclub default prediction",
]

MODEL_PATTERNS = {
    "logistic_regression": r"\blogisticregression\b|\blogit\b|\blogistic\s+regression\b",
    "catboost": r"\bcatboost\b|\bcatboostclassifier\b",
    "xgboost": r"\bxgboost\b|\bxgbclassifier\b|\bxgb\b",
    "lightgbm": r"\blightgbm\b|\blgbmclassifier\b|\blgbm\b",
    "random_forest": r"\brandomforestclassifier\b|\brandom\s+forest\b",
    "neural_network": r"\bmlpclassifier\b|\bkeras\b|\btensorflow\b|\bpytorch\b",
}

HPO_PATTERNS = {
    "optuna": r"\boptuna\b",
    "grid_search": r"\bgridsearchcv\b|\bgrid\s+search\b",
    "random_search": r"\brandomizedsearchcv\b|\brandom\s+search\b",
    "bayesian": r"\bbayesian\b|\bhyperopt\b|\bskopt\b|\bbayes\b",
    "manual_tuning": r"\btune\b|\bhyperparameter\b",
}

XAI_PATTERNS = {
    "shap": r"\bshap\b|\btreeshap\b",
    "lime": r"\blime\b",
    "pdp_ice": r"\bpartial\s+dependence\b|\bpdp\b|\bice\b",
    "permutation_importance": r"\bpermutation\s+importance\b",
    "feature_importance": r"\bfeature\s+importance\b",
}

AUC_PATTERNS = [
    r"(?:auc|auroc|roc\s*auc|roc_auc)\D{0,20}(0\.\d{2,4})",
    r"(0\.\d{2,4})\D{0,20}(?:auc|auroc|roc\s*auc|roc_auc)",
]


@dataclass
class KernelRecord:
    search_term: str
    kernel_id: int
    title: str
    author: str
    script_url: str
    run_id: int | None
    run_date: str | None
    rendered_output_url: str | None
    detected_models: list[str]
    detected_hpo: list[str]
    detected_xai: list[str]
    auc_reported: float | None


class KagglePublicClient:
    """Minimal Kaggle web client using public endpoints and anonymous cookies."""

    def __init__(self, timeout: int = 30) -> None:
        self.session = requests.Session()
        self.timeout = timeout
        self._bootstrap()

    def _bootstrap(self) -> None:
        response = self.session.get(KAGGLE_CODE_URL, timeout=self.timeout)
        response.raise_for_status()
        xsrf = self.session.cookies.get("XSRF-TOKEN")
        if not xsrf:
            raise RuntimeError("Kaggle XSRF token was not found in anonymous session.")
        self.session.headers.update(
            {
                "x-xsrf-token": xsrf,
                "content-type": "application/json",
                "accept": "application/json",
                "origin": "https://www.kaggle.com",
                "referer": KAGGLE_CODE_URL,
            }
        )

    def list_kernels(self, search_term: str, page_size: int = 20) -> list[dict[str, Any]]:
        payload = {
            "kernelFilterCriteria": {
                "search": search_term,
                "listRequest": {
                    "search": search_term,
                    "sortBy": "HOTNESS",
                },
            },
            "pageSize": int(page_size),
        }
        response = self.session.post(
            LIST_KERNELS_ENDPOINT,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        kernels = body.get("kernels", [])
        if not isinstance(kernels, list):
            return []
        return kernels

    def list_kernel_versions(self, kernel_id: int) -> list[dict[str, Any]]:
        payload = {"kernelId": int(kernel_id)}
        response = self.session.post(
            LIST_KERNEL_VERSIONS_ENDPOINT,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            logger.warning(
                "ListKernelVersions failed for kernel_id={} with status={}",
                kernel_id,
                response.status_code,
            )
            return []
        body = response.json()
        items = body.get("items", [])
        return items if isinstance(items, list) else []

    def fetch_text(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text


def _extract_code_from_rendered_html(html: str) -> str:
    """Extract code text from Kaggle rendered notebook HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        candidates = soup.select("div.input_area pre")
        if not candidates:
            candidates = soup.select("div.input code")
        chunks = [c.get_text("\n", strip=False) for c in candidates]
        text = "\n\n".join(chunks)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback: strip tags coarsely and keep plain text.
    no_tags = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", no_tags)


def _detect_methods(code_text: str, patterns: dict[str, str]) -> list[str]:
    text = code_text.lower()
    found = [name for name, pattern in patterns.items() if re.search(pattern, text)]
    return sorted(found)


def _extract_auc_candidates(text: str) -> list[float]:
    low = text.lower()
    auc_values: list[float] = []
    for pattern in AUC_PATTERNS:
        for match in re.findall(pattern, low):
            try:
                value = float(match)
            except Exception:
                continue
            if 0.5 <= value <= 1.0:
                auc_values.append(value)
    return auc_values


def _pick_run_with_output(
    versions: list[dict[str, Any]],
) -> tuple[int | None, str | None, str | None]:
    """Pick a version with rendered output URL, preferring newest by run ID."""
    candidates: list[tuple[int, str | None, str | None]] = []
    for item in versions:
        run = item.get("run", {}) if isinstance(item, dict) else {}
        if not isinstance(run, dict):
            continue
        run_id = run.get("id")
        rendered = run.get("renderedOutputUrl")
        run_date = run.get("dateEvaluated") or run.get("dateCreated")
        if isinstance(run_id, int) and isinstance(rendered, str) and rendered:
            candidates.append((run_id, run_date, rendered))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda row: row[0], reverse=True)
    run_id, run_date, rendered = candidates[0]
    return int(run_id), run_date, rendered


def collect_kaggle_benchmark(
    max_kernels_per_query: int = 20,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Collect Kaggle notebook benchmark rows."""
    client = KagglePublicClient(timeout=30)
    rows: list[KernelRecord] = []
    seen_kernel_ids: set[int] = set()

    for search_term in SEARCH_TERMS:
        logger.info("Searching Kaggle notebooks for query='{}'", search_term)
        kernels = client.list_kernels(search_term=search_term, page_size=max_kernels_per_query)

        for kernel in kernels[:max_kernels_per_query]:
            kernel_id = kernel.get("id")
            if not isinstance(kernel_id, int):
                continue
            if kernel_id in seen_kernel_ids:
                continue
            seen_kernel_ids.add(kernel_id)

            title = str(kernel.get("title", ""))
            author = str((kernel.get("author") or {}).get("userName", ""))
            script_url = str(kernel.get("scriptUrl", ""))
            if script_url and not script_url.startswith("http"):
                script_url = f"https://www.kaggle.com{script_url}"

            versions = client.list_kernel_versions(kernel_id)
            run_id, run_date, rendered_output_url = _pick_run_with_output(versions)

            detected_models: list[str] = []
            detected_hpo: list[str] = []
            detected_xai: list[str] = []
            auc_reported: float | None = None

            if rendered_output_url:
                try:
                    html = client.fetch_text(rendered_output_url)
                    text_blob = _extract_code_from_rendered_html(html)
                    detected_models = _detect_methods(text_blob, MODEL_PATTERNS)
                    detected_hpo = _detect_methods(text_blob, HPO_PATTERNS)
                    detected_xai = _detect_methods(text_blob, XAI_PATTERNS)
                    auc_candidates = _extract_auc_candidates(text_blob)
                    if auc_candidates:
                        auc_reported = max(auc_candidates)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse rendered output for kernel_id={} run_id={}: {}",
                        kernel_id,
                        run_id,
                        exc,
                    )

            rows.append(
                KernelRecord(
                    search_term=search_term,
                    kernel_id=kernel_id,
                    title=title,
                    author=author,
                    script_url=script_url,
                    run_id=run_id,
                    run_date=run_date,
                    rendered_output_url=rendered_output_url,
                    detected_models=detected_models,
                    detected_hpo=detected_hpo,
                    detected_xai=detected_xai,
                    auc_reported=auc_reported,
                )
            )
            time.sleep(max(0.0, sleep_seconds))

    frame = pd.DataFrame([r.__dict__ for r in rows])
    if frame.empty:
        return frame

    frame = frame.sort_values(
        by=["auc_reported", "run_id"], ascending=[False, False], na_position="last"
    )
    frame = frame.reset_index(drop=True)
    return frame


def _flatten_counter(series: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for values in series.dropna():
        if isinstance(values, list):
            counter.update(str(v) for v in values)
    return counter


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build aggregate benchmark summary payload."""
    if df.empty:
        return {
            "n_notebooks": 0,
            "model_counts": {},
            "hpo_counts": {},
            "xai_counts": {},
            "auc_reported": {},
        }

    auc_values = pd.to_numeric(df["auc_reported"], errors="coerce").dropna()
    if len(auc_values) > 0:
        auc_stats = {
            "n_with_auc": int(len(auc_values)),
            "min": float(auc_values.min()),
            "median": float(auc_values.median()),
            "max": float(auc_values.max()),
            "p90": float(auc_values.quantile(0.90)),
        }
    else:
        auc_stats = {"n_with_auc": 0, "min": None, "median": None, "max": None, "p90": None}

    return {
        "n_notebooks": int(len(df)),
        "n_unique_authors": int(df["author"].nunique()),
        "n_with_rendered_output": int(df["rendered_output_url"].notna().sum()),
        "model_counts": dict(_flatten_counter(df["detected_models"])),
        "hpo_counts": dict(_flatten_counter(df["detected_hpo"])),
        "xai_counts": dict(_flatten_counter(df["detected_xai"])),
        "auc_reported": auc_stats,
    }


def literature_reference_table() -> pd.DataFrame:
    """Curated external references for expected performance ranges."""
    rows = [
        {
            "source": "Peer-reviewed benchmark (Lending Club, Li et al. 2022)",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9222552/",
            "model_family": "LightGBM / Gradient Boosting",
            "reported_auc": 0.7492,
            "notes": "Reports top AUC around 74.92% for LightGBM on Lending Club; verify split protocol before setting operational targets.",
        },
        {
            "source": "Peer-to-peer lending explainability survey",
            "url": "https://arxiv.org/abs/2103.00949",
            "model_family": "Tree ensembles + SHAP/LIME",
            "reported_auc": None,
            "notes": "Highlights explainability methods and governance considerations.",
        },
    ]
    return pd.DataFrame(rows)


def main(max_kernels_per_query: int = 20, sleep_seconds: float = 0.2) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_df = collect_kaggle_benchmark(
        max_kernels_per_query=max_kernels_per_query,
        sleep_seconds=sleep_seconds,
    )
    kaggle_out = DATA_DIR / "kaggle_lendingclub_notebooks.parquet"
    benchmark_df.to_parquet(kaggle_out, index=False)
    logger.info("Saved Kaggle notebook benchmark: {} (rows={})", kaggle_out, len(benchmark_df))

    summary = build_summary(benchmark_df)
    summary["generated_at_utc"] = pd.Timestamp.utcnow().isoformat()
    summary["search_terms"] = SEARCH_TERMS

    summary_out = DATA_DIR / "lendingclub_benchmark_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved benchmark summary: {}", summary_out)

    lit_df = literature_reference_table()
    lit_out = DATA_DIR / "lendingclub_literature_benchmark.parquet"
    lit_df.to_parquet(lit_out, index=False)
    logger.info("Saved literature benchmark table: {}", lit_out)

    # Human-readable compact CSV for quick inspection.
    compact_cols = [
        "kernel_id",
        "run_id",
        "title",
        "author",
        "script_url",
        "auc_reported",
        "detected_models",
        "detected_hpo",
        "detected_xai",
    ]
    compact = (
        benchmark_df[compact_cols] if not benchmark_df.empty else pd.DataFrame(columns=compact_cols)
    )
    compact_out = DATA_DIR / "kaggle_lendingclub_notebooks.csv"
    compact.to_csv(compact_out, index=False)
    logger.info("Saved compact benchmark CSV: {}", compact_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-kernels-per-query", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()
    main(max_kernels_per_query=args.max_kernels_per_query, sleep_seconds=args.sleep_seconds)
