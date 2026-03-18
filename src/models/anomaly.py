"""Isolation Forest anomaly detection for out-of-distribution credit risk screening.

Trains an IsolationForest on PD model training features to flag loans that fall
outside the training distribution at inference time. Wide conformal intervals
correlate with high IsolationForest anomaly scores — together they provide two
complementary OOD signals: one supervised (conformal width) and one unsupervised.

O(n log n) complexity makes this viable on the full 1.3M training set.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest

DEFAULT_MODEL_PATH = Path("models/isolation_forest.pkl")
DEFAULT_CONTAMINATION = 0.05


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = 42,
    n_jobs: int = -1,
) -> IsolationForest:
    """Train an IsolationForest anomaly detector on training features.

    Args:
        X_train: Feature matrix from the training split (no target column).
        contamination: Expected proportion of anomalies. 0.05 = flag bottom 5%.
        random_state: Random seed for reproducibility.
        n_jobs: Parallel jobs for fitting (-1 = all cores).

    Returns:
        Fitted IsolationForest instance.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    iso.fit(X_train)
    logger.info(
        f"IsolationForest trained on {len(X_train):,} rows, "
        f"contamination={contamination}, features={X_train.shape[1]}"
    )
    return iso


def score_anomaly(iso: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Predict anomaly flags for each row.

    Args:
        iso: Fitted IsolationForest instance.
        X: Feature matrix to score.

    Returns:
        Integer array: -1 (anomaly) or 1 (normal) per row.
    """
    return iso.predict(X)


def anomaly_score(iso: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Return continuous anomaly scores (lower = more anomalous).

    The decision function returns the mean anomaly score of the input samples.
    Negative values indicate anomalies; the threshold is approximately 0.

    Args:
        iso: Fitted IsolationForest instance.
        X: Feature matrix to score.

    Returns:
        Float array of decision function values, one per row.
    """
    return iso.decision_function(X)


def save_isolation_forest(
    iso: IsolationForest,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Persist a fitted IsolationForest to disk as a pickle file.

    Args:
        iso: Fitted IsolationForest instance.
        path: Destination file path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(iso, f)
    logger.info(f"Saved IsolationForest to {target}")


def load_isolation_forest(
    path: str | Path = DEFAULT_MODEL_PATH,
) -> IsolationForest | None:
    """Load a persisted IsolationForest from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        Loaded IsolationForest, or None if the file does not exist.
    """
    target = Path(path)
    if not target.exists():
        logger.warning(f"IsolationForest artifact not found at {target}")
        return None
    with open(target, "rb") as f:
        iso = pickle.load(f)
    logger.info(f"Loaded IsolationForest from {target}")
    return iso


def compute_anomaly_report(
    iso: IsolationForest,
    X: pd.DataFrame,
    label: str = "test",
) -> dict[str, Any]:
    """Compute summary statistics for anomaly detection on a dataset.

    Args:
        iso: Fitted IsolationForest instance.
        X: Feature matrix (same feature space as training).
        label: Dataset label for reporting (e.g., "train", "test", "oot").

    Returns:
        Dict with anomaly_rate, n_anomalies, n_total, and score statistics.
    """
    predictions = score_anomaly(iso, X)
    scores = anomaly_score(iso, X)
    n_total = len(predictions)
    n_anomalies = int((predictions == -1).sum())
    return {
        "label": label,
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_rate": float(n_anomalies / n_total) if n_total > 0 else 0.0,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_p5": float(np.percentile(scores, 5)),
        "score_p95": float(np.percentile(scores, 95)),
    }
