"""Canonical PD model contract helpers.

Defines a single source of truth for:
- PD model artifact path
- calibrator artifact path
- feature contract (names, categorical subset)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

CANONICAL_MODEL_PATH = Path("models/pd_canonical.cbm")
CANONICAL_CALIBRATOR_PATH = Path("models/pd_canonical_calibrator.pkl")
CONTRACT_PATH = Path("models/pd_model_contract.json")


def _upstream_search_pd_candidates() -> list[Path]:
    run_tag = str(os.environ.get("UPSTREAM_CANONICAL_RUN_TAG", "") or "").strip()
    if not run_tag:
        return []
    base_dir = Path("models/search_pd") / run_tag
    return [
        base_dir / "pd_candidate_model.cbm",
        base_dir / "pd_local_hpo_tuned.cbm",
        base_dir / "pd_candidate_calibrator.pkl",
        base_dir / "pd_local_hpo_calibrator.pkl",
    ]


def resolve_model_path() -> Path:
    """Resolve canonical PD model path with fallback candidates."""
    upstream_candidates = _upstream_search_pd_candidates()
    candidates = [
        upstream_candidates[0] if len(upstream_candidates) > 0 else None,
        upstream_candidates[1] if len(upstream_candidates) > 1 else None,
        CANONICAL_MODEL_PATH,
        Path("models/pd_catboost.cbm"),
        Path("models/pd_catboost_tuned.cbm"),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("No PD model artifact found in models/.")
    return path


def resolve_calibrator_path() -> Path | None:
    """Resolve canonical calibrator path with fallback candidates."""
    upstream_candidates = _upstream_search_pd_candidates()
    candidates = [
        upstream_candidates[2] if len(upstream_candidates) > 2 else None,
        upstream_candidates[3] if len(upstream_candidates) > 3 else None,
        CANONICAL_CALIBRATOR_PATH,
        Path("models/pd_calibrator.pkl"),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    return next((p for p in candidates if p.exists()), None)


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any] | None:
    """Load persisted contract if present."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_contract_payload(
    model_path: Path,
    calibrator_path: Path | None,
    feature_names: list[str],
    categorical_features: list[str],
    split_shapes: dict[str, tuple[int, int]] | None = None,
    split_missing_features: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build serializable contract payload."""
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_path": model_path.as_posix(),
        "calibrator_path": calibrator_path.as_posix() if calibrator_path else None,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "n_features": len(feature_names),
    }
    if split_shapes is not None:
        payload["split_shapes"] = {k: [int(v[0]), int(v[1])] for k, v in split_shapes.items()}
    if split_missing_features is not None:
        payload["split_missing_features"] = split_missing_features
    return payload


def infer_model_feature_contract(model: CatBoostClassifier) -> tuple[list[str], list[str]]:
    """Extract model feature names and categorical feature names."""
    feature_names = list(getattr(model, "feature_names_", []) or [])
    cat_idx = set(model.get_cat_feature_indices())
    categorical = [f for i, f in enumerate(feature_names) if i in cat_idx]
    return feature_names, categorical


def validate_features_in_splits(
    feature_names: list[str],
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, tuple[int, int]], dict[str, list[str]]]:
    """Validate that required features exist in each split."""
    shapes: dict[str, tuple[int, int]] = {}
    missing: dict[str, list[str]] = {}
    for split_name, df in splits.items():
        shapes[split_name] = tuple(df.shape)
        missing[split_name] = [f for f in feature_names if f not in df.columns]
    return shapes, missing


def save_contract(payload: dict[str, Any], path: Path = CONTRACT_PATH) -> None:
    """Persist contract as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
