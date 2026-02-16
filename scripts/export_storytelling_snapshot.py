"""Export a frozen storytelling snapshot for demos/defense sessions.

Usage:
    uv run python scripts/export_storytelling_snapshot.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_PATH = REPORTS_DIR / "storytelling_snapshot.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_meta(path: Path) -> dict[str, object]:
    exists = path.exists()
    if not exists:
        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": False,
            "updated_at_utc": None,
            "size_bytes": 0,
        }

    stat = path.stat()
    updated = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "updated_at_utc": updated,
        "size_bytes": int(stat.st_size),
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    pipeline_summary = _load_json(DATA_DIR / "pipeline_summary.json")
    conformal_status = _load_json(MODEL_DIR / "conformal_policy_status.json")
    model_comparison = _load_json(DATA_DIR / "model_comparison.json")

    ifrs9 = pd.read_parquet(DATA_DIR / "ifrs9_scenario_summary.parquet")
    robustness = pd.read_parquet(DATA_DIR / "portfolio_robustness_summary.parquet")

    pipeline = pipeline_summary.get("pipeline", {})
    final_metrics = model_comparison.get("final_test_metrics", {})

    baseline_row = ifrs9[ifrs9["scenario"] == "baseline"]
    severe_row = ifrs9[ifrs9["scenario"] == "severe"]
    baseline_ecl = _safe_float(baseline_row["total_ecl"].iloc[0]) if not baseline_row.empty else 0.0
    severe_ecl = _safe_float(severe_row["total_ecl"].iloc[0]) if not severe_row.empty else 0.0

    robust_row = robustness.sort_values("risk_tolerance").iloc[0] if not robustness.empty else None
    robust_price = _safe_float(robust_row["price_of_robustness"]) if robust_row is not None else 0.0
    robust_price_pct = _safe_float(robust_row["price_of_robustness_pct"]) if robust_row is not None else 0.0

    required_artifacts = [
        DATA_DIR / "pipeline_summary.json",
        MODEL_DIR / "conformal_results_mondrian.pkl",
        DATA_DIR / "conformal_intervals_mondrian.parquet",
        MODEL_DIR / "conformal_policy_status.json",
        DATA_DIR / "portfolio_robustness_summary.parquet",
        DATA_DIR / "ifrs9_scenario_summary.parquet",
    ]

    snapshot = {
        "snapshot_name": "storytelling_snapshot",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "headline_metrics": {
            "auc_oot": _safe_float(final_metrics.get("auc_roc"), _safe_float(pipeline.get("pd_auc"))),
            "coverage_90": _safe_float(conformal_status.get("coverage_90")),
            "coverage_95": _safe_float(conformal_status.get("coverage_95")),
            "price_of_robustness": _safe_float(pipeline.get("price_of_robustness"), robust_price),
            "price_of_robustness_pct": robust_price_pct,
            "robust_return": _safe_float(pipeline.get("robust_return")),
            "nonrobust_return": _safe_float(pipeline.get("nonrobust_return")),
            "baseline_ecl": baseline_ecl,
            "severe_ecl": severe_ecl,
            "severe_uplift_pct": ((severe_ecl / baseline_ecl - 1.0) * 100.0) if baseline_ecl > 0 else 0.0,
        },
        "artifact_health": [_artifact_meta(path) for path in required_artifacts],
        "notes": [
            "Snapshot con métricas para storytelling reproducible.",
            "Fuente conformal canónica: conformal_results_mondrian.pkl + conformal_intervals_mondrian.parquet.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info(f"Saved storytelling snapshot: {OUT_PATH}")


if __name__ == "__main__":
    main()
