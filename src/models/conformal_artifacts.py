"""Utilities for loading conformal artifacts with canonical-path preference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

CANONICAL_INTERVALS_PATH = Path("data/processed/conformal_intervals_mondrian.parquet")


def resolve_intervals_path(
    allow_legacy_fallback: bool = False,
    override_path: str | Path | None = None,
) -> tuple[Path, bool]:
    """Resolve conformal intervals artifact path.

    Returns:
        path: selected artifact path
        is_legacy: whether selected path is the legacy compatibility artifact
    """
    if override_path is not None:
        path = Path(override_path)
        if not path.exists():
            raise FileNotFoundError(f"Conformal intervals override not found: {path}")
        return path, False

    if CANONICAL_INTERVALS_PATH.exists():
        return CANONICAL_INTERVALS_PATH, False

    raise FileNotFoundError(
        "Conformal intervals artifact not found. Expected canonical path "
        f"'{CANONICAL_INTERVALS_PATH}'."
    )


def load_conformal_intervals(
    allow_legacy_fallback: bool = False,
    override_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, bool]:
    """Load conformal interval artifact and return dataframe + selected path metadata."""
    if allow_legacy_fallback:
        logger.warning(
            "allow_legacy_fallback=True is deprecated; only canonical conformal artifact is supported."
        )
    path, is_legacy = resolve_intervals_path(
        allow_legacy_fallback=allow_legacy_fallback,
        override_path=override_path,
    )
    if override_path is not None:
        logger.info(f"Using conformal intervals override artifact: {path}")
    else:
        logger.info(f"Using canonical conformal artifact: {path}")

    df = pd.read_parquet(path)
    return df, path, is_legacy
