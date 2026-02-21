"""Shared I/O utilities for data loading with fallback logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def read_with_fallback(primary_path: str | Path, fallback_path: str | Path) -> pd.DataFrame:
    """Read parquet from primary path; fall back to alternate if missing.

    Args:
        primary_path: Preferred file path.
        fallback_path: Alternate file path if primary is unavailable.

    Returns:
        DataFrame loaded from whichever path exists.

    Raises:
        FileNotFoundError: If neither path exists.
    """
    primary = Path(primary_path)
    fallback = Path(fallback_path)
    if primary.exists():
        return pd.read_parquet(primary)
    if fallback.exists():
        logger.warning(f"{primary} not found. Falling back to {fallback}")
        return pd.read_parquet(fallback)
    raise FileNotFoundError(f"Neither {primary} nor {fallback} exists")


def read_split_with_fe_fallback(path: str | Path) -> pd.DataFrame:
    """Read a data split, trying *_fe variant and base variant as fallback.

    Handles the common pattern where feature-engineered splits (*_fe.parquet)
    may or may not exist alongside base splits (*.parquet).

    Args:
        path: Configured path (either *_fe.parquet or *.parquet).

    Returns:
        DataFrame from whichever variant exists.

    Raises:
        FileNotFoundError: If neither variant exists.
    """
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)

    alt = None
    if p.name.endswith("_fe.parquet"):
        alt = p.with_name(p.name.replace("_fe.parquet", ".parquet"))
    elif p.name.endswith(".parquet"):
        alt = p.with_name(p.name.replace(".parquet", "_fe.parquet"))

    if alt is not None and alt.exists():
        logger.warning(f"Configured path not found: {p}. Falling back to {alt}")
        return pd.read_parquet(alt)

    raise FileNotFoundError(f"Neither configured path nor fallback exists: {p}")
