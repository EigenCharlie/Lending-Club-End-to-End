"""Utilities for loading conformal artifacts with canonical-path preference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

CANONICAL_INTERVALS_PATH = Path("data/processed/conformal_intervals_mondrian.parquet")
LEGACY_INTERVALS_PATH = Path("data/processed/conformal_intervals.parquet")


def resolve_intervals_path(allow_legacy_fallback: bool = True) -> tuple[Path, bool]:
    """Resolve conformal intervals artifact path.

    Returns:
        path: selected artifact path
        is_legacy: whether selected path is the legacy compatibility artifact
    """
    if CANONICAL_INTERVALS_PATH.exists():
        return CANONICAL_INTERVALS_PATH, False

    if allow_legacy_fallback and LEGACY_INTERVALS_PATH.exists():
        return LEGACY_INTERVALS_PATH, True

    raise FileNotFoundError(
        "Conformal intervals artifact not found. Expected canonical path "
        f"'{CANONICAL_INTERVALS_PATH}'"
        + (f" or legacy fallback '{LEGACY_INTERVALS_PATH}'." if allow_legacy_fallback else ".")
    )


def load_conformal_intervals(allow_legacy_fallback: bool = True) -> tuple[pd.DataFrame, Path, bool]:
    """Load conformal interval artifact and return dataframe + selected path metadata."""
    path, is_legacy = resolve_intervals_path(allow_legacy_fallback=allow_legacy_fallback)
    if is_legacy:
        logger.warning(
            f"Using legacy conformal artifact for compatibility: {path}. "
            f"Prefer canonical artifact: {CANONICAL_INTERVALS_PATH}"
        )
    else:
        logger.info(f"Using canonical conformal artifact: {path}")

    df = pd.read_parquet(path)
    return df, path, is_legacy
