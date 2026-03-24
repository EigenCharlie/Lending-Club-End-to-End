"""Benchmark LAC vs APS vs RAPS conformity scores for binary PD classification sets.

MAPIE's SplitConformalClassifier supports three conformity score methods:
- LAC (Least Ambiguous Classification-based): simpler, deterministic threshold
- APS (Adaptive Prediction Sets): adaptive, class-conditional scores
- RAPS (Regularized APS): adds regularization λ to penalize large sets

For binary PD classification, this produces prediction *sets* rather than
intervals — useful for abstention/rejection when the model is uncertain:
  - Set = {} → empty (very confident: predict non-default)
  - Set = {1} or {0} → single-class (confident)
  - Set = {0,1} → ambiguous (abstain / flag for review)

The benchmark computes coverage and mean set size for each method.

Usage:
    uv run python scripts/run_classification_set_benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

CONFIDENCE_LEVELS = [0.80, 0.85, 0.90, 0.95]
CONFORMITY_SCORES = ["lac", "aps", "raps"]
MAX_CAL_ROWS = 50_000
MAX_TEST_ROWS = 50_000


def _load_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load y_true, y_prob_cal, y_prob_test from canonical artifacts."""
    test_path = DATA_DIR / "test_predictions.parquet"
    cal_path = DATA_DIR / "calibration_predictions.parquet"

    if not test_path.exists():
        raise FileNotFoundError(f"test_predictions.parquet not found at {test_path}")

    test_df = pd.read_parquet(test_path)
    y_test = pd.to_numeric(test_df["y_true"], errors="coerce").to_numpy(float)
    prob_col = next(
        (c for c in ["y_prob_final", "pd_calibrated", "y_prob"] if c in test_df.columns), None
    )
    if prob_col is None:
        raise ValueError(
            f"No probability column found in test_predictions.parquet. Columns: {test_df.columns.tolist()}"
        )
    y_prob_test = pd.to_numeric(test_df[prob_col], errors="coerce").to_numpy(float)

    # Use calibration predictions if available, otherwise split test for conformalization
    if cal_path.exists():
        cal_df = pd.read_parquet(cal_path)
        y_cal = pd.to_numeric(cal_df["y_true"], errors="coerce").to_numpy(float)
        prob_col_cal = next(
            (c for c in ["y_prob_final", "pd_calibrated", "y_prob"] if c in cal_df.columns), None
        )
        y_prob_cal = pd.to_numeric(cal_df[prob_col_cal], errors="coerce").to_numpy(float)
    else:
        # Fallback: use first half of test as calibration
        n = len(y_test)
        split = n // 2
        y_cal, y_prob_cal = y_test[:split], y_prob_test[:split]
        y_test, y_prob_test = y_test[split:], y_prob_test[split:]
        logger.warning(
            "calibration_predictions.parquet not found — using first half of test as conformalization set."
        )

    # Cap sizes for speed
    if len(y_cal) > MAX_CAL_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_cal), MAX_CAL_ROWS, replace=False)
        y_cal, y_prob_cal = y_cal[idx], y_prob_cal[idx]
    if len(y_test) > MAX_TEST_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_test), MAX_TEST_ROWS, replace=False)
        y_test, y_prob_test = y_test[idx], y_prob_test[idx]

    logger.info(
        f"Loaded: cal n={len(y_cal):,}, test n={len(y_test):,}, "
        f"default_rate_cal={y_cal.mean():.3f}, default_rate_test={y_test.mean():.3f}"
    )
    return y_cal.astype(int), y_prob_cal, y_test.astype(int), y_prob_test


def _build_softmax_proba(y_prob: np.ndarray) -> np.ndarray:
    """Convert scalar P(default) to (n, 2) softmax matrix expected by MAPIE."""
    p_pos = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p_pos, p_pos])


class _ScorelessClassifier:
    """Minimal sklearn-compatible wrapper that serves pre-computed probabilities.

    MAPIE's SplitConformalClassifier needs a fitted estimator with predict_proba.
    Since we already have calibrated scores, we just replay them.
    """

    def __init__(self, proba_matrix: np.ndarray) -> None:
        self._proba = proba_matrix
        self.classes_ = np.array([0, 1])

    def fit(self, _X: np.ndarray, _y: np.ndarray) -> _ScorelessClassifier:
        return self

    def predict(self, _X: np.ndarray) -> np.ndarray:
        return (self._proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, _X: np.ndarray) -> np.ndarray:
        return self._proba


def _run_benchmark(
    y_cal: np.ndarray,
    y_prob_cal: np.ndarray,
    y_test: np.ndarray,
    y_prob_test: np.ndarray,
) -> pd.DataFrame:
    """Run LAC/APS/RAPS benchmark across confidence levels.

    Returns a DataFrame with columns:
    method, confidence_level, empirical_coverage, mean_set_size,
    pct_empty, pct_ambiguous, pct_single
    """
    from mapie.classification import SplitConformalClassifier

    proba_cal = _build_softmax_proba(y_prob_cal)
    proba_test = _build_softmax_proba(y_prob_test)

    # Feature matrix: just a dummy 1D index (MAPIE needs X but we use prefit=True)
    X_cal_dummy = np.arange(len(y_cal)).reshape(-1, 1).astype(float)
    X_test_dummy = np.arange(len(y_test)).reshape(-1, 1).astype(float)

    rows: list[dict[str, object]] = []

    for score in CONFORMITY_SCORES:
        if score != "lac":
            logger.warning(
                f"[{score}] skipped: MAPIE binary SplitConformalClassifier only supports 'lac'."
            )
            continue
        clf_wrapper = _ScorelessClassifier(proba_cal)
        for conf in CONFIDENCE_LEVELS:
            try:
                mapie_clf = SplitConformalClassifier(
                    estimator=clf_wrapper,
                    confidence_level=conf,
                    conformity_score=score,
                    prefit=True,
                )
                mapie_clf.conformalize(X_cal_dummy, y_cal)

                # Swap wrapper so predict_proba returns test probabilities
                mapie_clf.estimator_ = _ScorelessClassifier(proba_test)
                _, y_pred_set = mapie_clf.predict_set(X_test_dummy)
                pred_set = y_pred_set[:, :, 0]  # shape (n, 2)

                covered = pred_set[np.arange(len(y_test)), y_test.astype(int)]
                cov = float(np.mean(covered))
                mean_width = float(np.mean(pred_set.sum(axis=1)))

                n = len(y_test)
                n_empty = int(np.sum(~pred_set[:, 0] & ~pred_set[:, 1]))
                n_ambig = int(np.sum(pred_set[:, 0] & pred_set[:, 1]))
                n_single = n - n_empty - n_ambig

                rows.append(
                    {
                        "method": score,
                        "confidence_level": conf,
                        "empirical_coverage": cov,
                        "mean_set_size": mean_width,
                        "pct_empty": 100.0 * n_empty / n,
                        "pct_single": 100.0 * n_single / n,
                        "pct_ambiguous": 100.0 * n_ambig / n,
                        "n": n,
                    }
                )
                logger.info(
                    f"[{score}] conf={conf:.2f} → coverage={cov:.4f} "
                    f"mean_width={mean_width:.3f} ambiguous={100 * n_ambig / n:.1f}%"
                )
            except Exception as exc:
                logger.error(f"[{score}] conf={conf:.2f} failed: {exc}")

    columns = [
        "method",
        "confidence_level",
        "empirical_coverage",
        "mean_set_size",
        "pct_empty",
        "pct_single",
        "pct_ambiguous",
        "n",
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    logger.info("Starting LAC/APS/RAPS classification set benchmark.")
    y_cal, y_prob_cal, y_test, y_prob_test = _load_predictions()
    results = _run_benchmark(y_cal, y_prob_cal, y_test, y_prob_test)

    out_path = DATA_DIR / "classification_set_benchmark.parquet"
    results.to_parquet(out_path, index=False)
    logger.info(f"Saved benchmark results: {out_path} ({len(results):,} rows)")

    # Also save a concise status JSON
    if results.empty:
        best_per_score = {}
    else:
        best_per_score = (
            results[results["confidence_level"] == 0.90]
            .set_index("method")[["empirical_coverage", "mean_set_size", "pct_ambiguous"]]
            .to_dict(orient="index")
        )
    status = {
        **build_artifact_metadata(schema_version="2026-03-21.1"),
        "benchmark_rows": len(results),
        "confidence_levels": CONFIDENCE_LEVELS,
        "methods": CONFORMITY_SCORES,
        "supported_methods": ["lac"],
        "skipped_methods": [score for score in CONFORMITY_SCORES if score != "lac"],
        "summary_at_90": best_per_score,
    }
    status_path = DATA_DIR / "classification_set_benchmark_status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved status: {status_path}")

    # Print summary table
    if not results.empty:
        summary = results[results["confidence_level"] == 0.90][
            ["method", "empirical_coverage", "mean_set_size", "pct_empty", "pct_ambiguous"]
        ]
        logger.info(f"\n{summary.to_string(index=False)}")
    else:
        logger.warning("No benchmark rows were generated.")


if __name__ == "__main__":
    main()
