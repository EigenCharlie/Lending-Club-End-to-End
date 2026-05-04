"""Conformal prediction utilities using MAPIE >= 1.3.

Includes:
- Global split-conformal intervals for PD probabilities.
- Group-conditional (Mondrian-style) split conformal by segment (e.g., grade).
- Classification set wrappers (LAC/APS/RAPS via MAPIE).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin


class ProbabilityRegressor(BaseEstimator, RegressorMixin):
    """Wrap classifier predict_proba as a regression predictor.

    Optionally applies a probability calibrator after raw predictions.
    """

    def __init__(self, classifier, calibrator: Any | None = None):
        self.classifier = classifier
        self.calibrator = calibrator
        self.is_fitted_ = True  # required for MAPIE prefit checks

    def fit(self, X, y):
        """Already fitted - no-op for MAPIE interface."""
        return self

    def predict(self, X):
        """Return calibrated P(default) in [0, 1]."""
        raw = self.classifier.predict_proba(X)[:, 1]
        return apply_probability_calibrator(self.calibrator, raw)


class PrefitClassifierAdapter(BaseEstimator):
    """Small sklearn-style adapter for prefit classifiers inside MAPIE checks."""

    def __init__(self, classifier, n_features_in: int | None = None):
        self.classifier = classifier
        classes = getattr(classifier, "classes_", np.array([0, 1]))
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in or getattr(classifier, "n_features_in_", 0) or 0)
        self.feature_names_in_ = np.asarray(
            [f"f{i}" for i in range(self.n_features_in_)], dtype=object
        )
        self.is_fitted_ = True

    def fit(self, X, y):
        return self

    def _is_minimal_probe(self, X: pd.DataFrame) -> bool:
        if X.shape[0] != 1 or X.shape[1] != self.n_features_in_:
            return False
        numeric = X.apply(pd.to_numeric, errors="coerce")
        return bool(np.isfinite(numeric.to_numpy()).all() and np.allclose(numeric.to_numpy(), 0.0))

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if self._is_minimal_probe(X_df):
            return np.zeros(len(X_df), dtype=int)
        return self.classifier.predict(X_df)

    def predict_proba(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if self._is_minimal_probe(X_df):
            return np.column_stack([np.ones(len(X_df)), np.zeros(len(X_df))])
        return self.classifier.predict_proba(X_df)


class PrefitCalibratedClassifierAdapter(PrefitClassifierAdapter):
    """Prefit classifier adapter that applies a probability calibrator."""

    def __init__(self, classifier, calibrator: Any | None = None, n_features_in: int | None = None):
        super().__init__(classifier, n_features_in=n_features_in)
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = super().predict_proba(X)
        if self.calibrator is None:
            return raw
        p_pos = apply_probability_calibrator(self.calibrator, raw[:, 1])
        p_neg = np.clip(1.0 - p_pos, 0.0, 1.0)
        return np.column_stack([p_neg, p_pos])


def apply_probability_calibrator(calibrator: Any, scores: np.ndarray) -> np.ndarray:
    """Apply calibrator robustly across sklearn calibrator API variants."""
    scores = np.asarray(scores, dtype=float)
    if calibrator is None:
        return np.clip(scores, 0.0, 1.0)

    if hasattr(calibrator, "transform"):
        out = calibrator.transform(scores)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    if hasattr(calibrator, "predict_proba"):
        out = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    try:
        out = calibrator.predict(scores)
        out = np.asarray(out, dtype=float)
        if out.shape[0] != scores.shape[0]:
            out = np.asarray(calibrator.predict(scores.reshape(-1, 1)), dtype=float)
    except (ValueError, TypeError, IndexError):
        out = np.asarray(calibrator.predict(scores.reshape(-1, 1)), dtype=float)

    return np.clip(out, 0.0, 1.0)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile with 'higher' interpolation."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    n = scores.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def _resolve_score_scale_family(*, scaled_scores: bool, score_scale_family: str | None) -> str:
    family = str(score_scale_family or "").strip().lower()
    if family in {"", "auto"}:
        family = "bernoulli_sqrt" if scaled_scores else "none"
    valid = {
        "none",
        "bernoulli_sqrt",
        "bernoulli_sqrt_clipped_0.02",
        "bernoulli_sqrt_clipped_0.05",
    }
    if family not in valid:
        raise ValueError(f"Unsupported score_scale_family: {score_scale_family}")
    return family


def _compute_score_scale(y_prob: np.ndarray, score_scale_family: str) -> np.ndarray:
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    if score_scale_family == "none":
        return np.ones_like(y_prob_arr)
    if score_scale_family == "bernoulli_sqrt":
        return np.sqrt(np.clip(y_prob_arr * (1.0 - y_prob_arr), 1e-6, None))
    if score_scale_family == "bernoulli_sqrt_clipped_0.02":
        clipped = np.clip(y_prob_arr, 0.02, 0.98)
        return np.sqrt(np.clip(clipped * (1.0 - clipped), 1e-6, None))
    if score_scale_family == "bernoulli_sqrt_clipped_0.05":
        clipped = np.clip(y_prob_arr, 0.05, 0.95)
        return np.sqrt(np.clip(clipped * (1.0 - clipped), 1e-6, None))
    raise ValueError(f"Unsupported score_scale_family: {score_scale_family}")


def create_pd_intervals(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
    calibrator: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate global split-conformal PD intervals via MAPIE."""
    from mapie.regression import SplitConformalRegressor

    prob_reg = ProbabilityRegressor(classifier, calibrator=calibrator)
    mapie = SplitConformalRegressor(
        estimator=prob_reg,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal.astype(float))

    # MAPIE output shape: (n, 2, n_confidence_levels)
    y_pred, y_intervals_raw = mapie.predict_interval(X_test)
    y_intervals = y_intervals_raw[:, :, 0]  # -> (n, 2)

    y_intervals = np.clip(y_intervals, 0, 1)
    y_pred = np.clip(y_pred, 0, 1)

    avg_width = float((y_intervals[:, 1] - y_intervals[:, 0]).mean())
    logger.info(
        f"Conformal PD intervals (global, alpha={alpha}): "
        f"avg_width={avg_width:.4f}, target_coverage={1 - alpha:.0%}"
    )
    return y_pred, y_intervals


def create_pd_intervals_mondrian(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    group_cal: pd.Series,
    group_test: pd.Series,
    alpha: float = 0.1,
    min_group_size: int = 500,
    calibrator: Any | None = None,
    scaled_scores: bool = False,
    score_scale_family: str = "none",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create group-conditional split-conformal PD intervals.

    This follows a Mondrian-style approach:
    - compute nonconformity scores on calibration split
    - estimate quantiles within each group
    - apply group-specific radius to test predictions

    Args:
        classifier: fitted classifier with predict_proba.
        X_cal, y_cal: calibration data.
        X_test: test features.
        group_cal, group_test: segmentation labels (e.g., grade).
        alpha: significance level.
        min_group_size: fallback to global quantile for small groups.
        calibrator: optional probability calibrator.
        scaled_scores: legacy shortcut for Bernoulli scaling.
        score_scale_family: explicit scaling family for residual scores.

    Returns:
        y_pred_test, y_intervals, diagnostics
    """
    y_cal_pred_raw = classifier.predict_proba(X_cal)[:, 1]
    y_test_pred_raw = classifier.predict_proba(X_test)[:, 1]
    y_cal_pred = apply_probability_calibrator(calibrator, y_cal_pred_raw)
    y_test_pred = apply_probability_calibrator(calibrator, y_test_pred_raw)

    y_cal_arr = np.asarray(y_cal, dtype=float)
    g_cal = pd.Series(group_cal).fillna("UNKNOWN").astype(str).to_numpy()
    g_test = pd.Series(group_test).fillna("UNKNOWN").astype(str).to_numpy()

    scores = np.abs(y_cal_arr - y_cal_pred)
    resolved_scale_family = _resolve_score_scale_family(
        scaled_scores=scaled_scores,
        score_scale_family=score_scale_family,
    )
    cal_scale = _compute_score_scale(y_cal_pred, resolved_scale_family)
    test_scale = _compute_score_scale(y_test_pred, resolved_scale_family)
    if resolved_scale_family != "none":
        scores = scores / cal_scale

    global_q = _conformal_quantile(scores, alpha)
    group_quantiles: dict[str, float] = {}
    group_cal_counts: dict[str, int] = {}
    fallback_groups: list[str] = []

    # Use union so unseen test groups still get a valid quantile.
    all_groups = sorted(set(g_cal).union(set(g_test)))
    for g in all_groups:
        mask = g_cal == g
        n_g = int(mask.sum())
        group_cal_counts[g] = n_g
        if n_g >= min_group_size:
            group_quantiles[g] = _conformal_quantile(scores[mask], alpha)
        else:
            group_quantiles[g] = global_q
            fallback_groups.append(g)

    radii = np.array([group_quantiles[str(g)] for g in g_test], dtype=float) * test_scale
    low = np.clip(y_test_pred - radii, 0.0, 1.0)
    high = np.clip(y_test_pred + radii, 0.0, 1.0)
    y_intervals = np.column_stack([low, high])

    diagnostics = {
        "alpha": alpha,
        "global_quantile": global_q,
        "group_quantiles": group_quantiles,
        "group_cal_counts": group_cal_counts,
        "fallback_groups": fallback_groups,
        "scaled_scores": bool(resolved_scale_family != "none"),
        "score_scale_family": resolved_scale_family,
        "min_group_size": min_group_size,
        "avg_width": float((high - low).mean()),
        "median_width": float(np.median(high - low)),
    }
    logger.info(
        "Conformal PD intervals (mondrian): "
        f"groups={len(all_groups)}, avg_width={diagnostics['avg_width']:.4f}, "
        f"fallback_groups={len(fallback_groups)}"
    )
    return y_test_pred, y_intervals, diagnostics


def conditional_coverage_by_group(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compute conditional coverage and width per segment."""
    g = pd.Series(groups).fillna("UNKNOWN").astype(str)
    y_true_arr = np.asarray(y_true, dtype=float)
    low = y_intervals[:, 0]
    high = y_intervals[:, 1]
    covered = (y_true_arr >= low) & (y_true_arr <= high)
    widths = high - low

    df = pd.DataFrame(
        {
            "group": g,
            "covered": covered.astype(float),
            "width": widths,
        }
    )
    out = (
        df.groupby("group", observed=True)
        .agg(
            n=("covered", "size"),
            coverage=("covered", "mean"),
            avg_width=("width", "mean"),
            median_width=("width", "median"),
        )
        .reset_index()
        .sort_values("group")
    )
    return out


def create_regression_intervals(
    regressor,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate regression intervals (for LGD/EAD) using MAPIE."""
    from mapie.regression import SplitConformalRegressor

    mapie = SplitConformalRegressor(
        estimator=regressor,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)

    y_pred, y_intervals_raw = mapie.predict_interval(X_test)
    y_intervals = y_intervals_raw[:, :, 0]

    avg_width = (y_intervals[:, 1] - y_intervals[:, 0]).mean()
    logger.info(f"Conformal regression intervals (alpha={alpha}): avg_width={avg_width:.4f}")
    return y_pred, y_intervals


def create_classification_sets(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
    method: str = "lac",
    calibrator: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate conformal prediction sets for classification."""
    method_key = str(method or "lac").strip().lower()
    if method_key == "margin":
        y_pred, y_sets = _create_margin_classification_sets(
            classifier=classifier,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            alpha=alpha,
            calibrator=calibrator,
        )
    else:
        from mapie.classification import SplitConformalClassifier

        adapted = PrefitCalibratedClassifierAdapter(
            classifier,
            calibrator=calibrator,
            n_features_in=X_cal.shape[1],
        )
        mapie = SplitConformalClassifier(
            estimator=adapted,
            confidence_level=1 - alpha,
            conformity_score=method_key,
            prefit=True,
        )
        mapie.conformalize(X_cal, y_cal)

        y_pred = mapie.predict(X_test)
        _, y_sets_raw = mapie.predict_set(X_test)
        y_sets = np.asarray(y_sets_raw[:, :, 0], dtype=int)

    singleton_rate = (y_sets.sum(axis=1) == 1).mean()
    logger.info(
        f"Conformal sets (alpha={alpha}, method={method_key}): singleton_rate={singleton_rate:.2%}"
    )
    return y_pred, y_sets


def summarize_prediction_sets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_sets: np.ndarray,
) -> dict[str, float]:
    """Summarize binary conformal prediction sets for abstention analysis."""
    true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=int).reshape(-1)
    sets = np.asarray(y_sets, dtype=int)
    if sets.ndim != 2:
        raise ValueError(f"Expected y_sets to be 2D, got shape={sets.shape}")
    if len(true_arr) != len(sets):
        raise ValueError("y_true and y_sets must have the same length.")

    set_size = sets.sum(axis=1)
    singleton_mask = set_size == 1
    ambiguous_mask = set_size > 1
    empty_mask = set_size == 0
    covered_mask = sets[np.arange(len(true_arr)), true_arr] == 1
    positive_singleton_mask = singleton_mask & (pred_arr == 1)

    return {
        "n_obs": float(len(true_arr)),
        "set_coverage": float(covered_mask.mean()) if len(true_arr) else float("nan"),
        "singleton_rate": float(singleton_mask.mean()) if len(true_arr) else float("nan"),
        "ambiguity_rate": float(ambiguous_mask.mean()) if len(true_arr) else float("nan"),
        "empty_set_rate": float(empty_mask.mean()) if len(true_arr) else float("nan"),
        "default_rate_ambiguous": float(true_arr[ambiguous_mask].mean())
        if ambiguous_mask.any()
        else float("nan"),
        "default_rate_singleton_positive": float(true_arr[positive_singleton_mask].mean())
        if positive_singleton_mask.any()
        else float("nan"),
        "default_rate_overall": float(true_arr.mean()) if len(true_arr) else float("nan"),
    }


def _create_margin_classification_sets(
    *,
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
    calibrator: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary margin-style conformal sets over calibrated probabilities."""
    p_cal = apply_probability_calibrator(calibrator, classifier.predict_proba(X_cal)[:, 1])
    p_test = apply_probability_calibrator(calibrator, classifier.predict_proba(X_test)[:, 1])
    y_cal_arr = np.asarray(y_cal, dtype=int).reshape(-1)

    p_true = np.where(y_cal_arr == 1, p_cal, 1.0 - p_cal)
    nonconformity = 2.0 * (1.0 - p_true)
    q_alpha = _conformal_quantile(nonconformity, alpha)

    include_pos = (2.0 * (1.0 - p_test)) <= q_alpha
    include_neg = (2.0 * p_test) <= q_alpha
    y_sets = np.column_stack([include_neg.astype(int), include_pos.astype(int)])
    y_pred = (p_test >= 0.5).astype(int)
    return y_pred, y_sets


def create_classification_sets_mondrian(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    group_cal: pd.Series,
    group_test: pd.Series,
    alpha: float = 0.1,
    method: str = "lac",
    min_group_size: int = 500,
    calibrator: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate group-conditional conformal prediction sets for binary classification."""
    g_cal = pd.Series(group_cal).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    g_test = pd.Series(group_test).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    all_groups = sorted(set(g_cal).union(set(g_test)))
    group_counts = {group: int((g_cal == group).sum()) for group in all_groups}
    fallback_groups: list[str] = []

    global_pred, global_sets = create_classification_sets(
        classifier=classifier,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        alpha=alpha,
        method=method,
        calibrator=calibrator,
    )
    y_pred = np.asarray(global_pred, dtype=int).copy()
    y_sets = np.asarray(global_sets, dtype=int).copy()

    for group in all_groups:
        cal_mask = g_cal == group
        test_mask = g_test == group
        if not test_mask.any():
            continue
        if int(cal_mask.sum()) < int(min_group_size):
            fallback_groups.append(group)
            continue
        group_pred, group_sets = create_classification_sets(
            classifier=classifier,
            X_cal=X_cal.loc[cal_mask].reset_index(drop=True),
            y_cal=y_cal.loc[cal_mask].reset_index(drop=True),
            X_test=X_test.loc[test_mask].reset_index(drop=True),
            alpha=alpha,
            method=method,
            calibrator=calibrator,
        )
        y_pred[np.asarray(test_mask)] = np.asarray(group_pred, dtype=int)
        y_sets[np.asarray(test_mask)] = np.asarray(group_sets, dtype=int)

    diagnostics = {
        "alpha": float(alpha),
        "method": str(method),
        "min_group_size": int(min_group_size),
        "group_cal_counts": group_counts,
        "fallback_groups": sorted(set(fallback_groups)),
    }
    return y_pred, y_sets, diagnostics


def build_mondrian_partition_labels(
    *,
    y_prob_cal: np.ndarray,
    y_prob_eval: np.ndarray,
    partition: str,
    base_groups_cal: pd.Series | np.ndarray | None = None,
    base_groups_eval: pd.Series | np.ndarray | None = None,
    n_score_bins: int = 10,
    min_group_size: int = 500,
    fallback_mode: str = "grade_then_global",
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Build partition labels for Mondrian-style calibration.

    Supported partitions:
    - grade: original subgroup labels.
    - score_decile_mondrian: deciles of calibrated/raw score.
    - grade_x_scoreband_mondrian: grade crossed with score bands, with fallback to
      grade/global when calibration support is too small.
    """

    partition_key = str(partition).strip().lower()
    if partition_key in {"grade", "group", "default"}:
        if base_groups_cal is None or base_groups_eval is None:
            raise ValueError("grade partition requires base group labels for calibration/eval.")
        g_cal = pd.Series(base_groups_cal).fillna("UNKNOWN").astype(str).reset_index(drop=True)
        g_eval = pd.Series(base_groups_eval).fillna("UNKNOWN").astype(str).reset_index(drop=True)
        return (
            g_cal,
            g_eval,
            {
                "partition": "grade",
                "score_band_count": 0,
                "fallback_groups": [],
            },
        )

    cal_scores = pd.Series(np.asarray(y_prob_cal, dtype=float).reshape(-1)).clip(0.0, 1.0)
    eval_scores = pd.Series(np.asarray(y_prob_eval, dtype=float).reshape(-1)).clip(0.0, 1.0)
    rank_source = cal_scores.rank(method="first")
    n_bins_effective = int(max(1, min(int(n_score_bins), int(rank_source.nunique()))))
    if n_bins_effective <= 1:
        cal_band = pd.Series(["score_q0"] * len(cal_scores), dtype="string")
        eval_band = pd.Series(["score_q0"] * len(eval_scores), dtype="string")
        edges = np.array([0.0, 1.0], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins_effective + 1)
        edges = np.unique(np.quantile(cal_scores.to_numpy(dtype=float), quantiles))
        if len(edges) <= 2:
            cal_band = pd.Series(["score_q0"] * len(cal_scores), dtype="string")
            eval_band = pd.Series(["score_q0"] * len(eval_scores), dtype="string")
        else:
            labels = [f"score_q{i:02d}" for i in range(len(edges) - 1)]
            cal_band = pd.cut(
                cal_scores,
                bins=edges,
                labels=labels,
                include_lowest=True,
                duplicates="drop",
            ).astype("string")
            eval_band = pd.cut(
                eval_scores,
                bins=edges,
                labels=labels,
                include_lowest=True,
                duplicates="drop",
            ).astype("string")
            cal_band = cal_band.fillna(labels[0])
            eval_band = eval_band.fillna(labels[0])

    if partition_key in {"score_decile_mondrian", "score_decile", "scoreband"}:
        return (
            cal_band.astype(str).reset_index(drop=True),
            eval_band.astype(str).reset_index(drop=True),
            {
                "partition": "score_decile_mondrian",
                "score_band_count": int(cal_band.nunique()),
                "score_band_edges": [float(x) for x in np.asarray(edges, dtype=float)],
                "fallback_groups": [],
                "fallback_mode": "score_only",
            },
        )

    if partition_key not in {"grade_x_scoreband_mondrian", "grade_scoreband", "hybrid"}:
        raise ValueError(f"Unsupported partition mode: {partition}")

    if base_groups_cal is None or base_groups_eval is None:
        raise ValueError("grade_x_scoreband_mondrian requires base group labels.")

    grade_cal = pd.Series(base_groups_cal).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    grade_eval = pd.Series(base_groups_eval).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    hybrid_cal = (grade_cal + "|" + cal_band.astype(str)).reset_index(drop=True)
    hybrid_eval = (grade_eval + "|" + eval_band.astype(str)).reset_index(drop=True)

    counts = hybrid_cal.value_counts(dropna=False).to_dict()
    grade_counts = grade_cal.value_counts(dropna=False).to_dict()
    fallback_groups: list[str] = []
    fallback_mode_key = str(fallback_mode or "grade_then_global").strip().lower()
    if fallback_mode_key not in {"grade_then_global", "global_only"}:
        raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")

    def _resolve_label(label: str, base_grade: str) -> str:
        n_label = int(counts.get(label, 0))
        if n_label >= int(min_group_size):
            return label
        fallback_groups.append(label)
        if fallback_mode_key == "grade_then_global" and int(grade_counts.get(base_grade, 0)) >= int(
            min_group_size
        ):
            return base_grade
        return "GLOBAL"

    resolved_cal = pd.Series(
        [
            _resolve_label(str(label), str(grade))
            for label, grade in zip(hybrid_cal, grade_cal, strict=False)
        ],
        dtype="string",
    )
    resolved_eval = pd.Series(
        [
            _resolve_label(str(label), str(grade))
            for label, grade in zip(hybrid_eval, grade_eval, strict=False)
        ],
        dtype="string",
    )
    return (
        resolved_cal.astype(str).reset_index(drop=True),
        resolved_eval.astype(str).reset_index(drop=True),
        {
            "partition": "grade_x_scoreband_mondrian",
            "score_band_count": int(cal_band.nunique()),
            "score_band_edges": [float(x) for x in np.asarray(edges, dtype=float)],
            "fallback_groups": sorted(set(fallback_groups)),
            "hybrid_group_count_cal": int(hybrid_cal.nunique()),
            "resolved_group_count_cal": int(resolved_cal.nunique()),
            "fallback_mode": fallback_mode_key,
        },
    )


def create_cross_conformal_score_intervals(
    y_cal: pd.Series | np.ndarray,
    y_prob_cal: np.ndarray,
    y_prob_test: np.ndarray,
    *,
    alpha: float = 0.1,
    method: str = "plus",
    cv: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a lightweight cross conformal benchmark on raw score space.

    This is intentionally a score-space benchmark, not a full feature-space
    retraining of the upstream CatBoost classifier. It lets the repo compare a
    cross conformal variant with low incremental cost while keeping the current
    canonical model frozen.
    """
    from mapie.regression import CrossConformalRegressor
    from sklearn.linear_model import LinearRegression

    y_cal_arr = np.asarray(y_cal, dtype=float).reshape(-1)
    X_cal = np.asarray(y_prob_cal, dtype=float).reshape(-1, 1)
    X_test = np.asarray(y_prob_test, dtype=float).reshape(-1, 1)

    regressor = CrossConformalRegressor(
        estimator=LinearRegression(),
        confidence_level=1 - alpha,
        method=method,
        cv=cv,
    )
    regressor.fit_conformalize(X_cal, y_cal_arr)
    y_pred, y_intervals_raw = regressor.predict_interval(X_test)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float).reshape(-1), 0.0, 1.0)
    y_intervals = np.asarray(y_intervals_raw[:, :, 0], dtype=float)
    y_intervals = np.clip(y_intervals, 0.0, 1.0)
    return y_pred_arr, y_intervals


def create_pd_intervals_venn_abers(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Venn-Abers multi-probability PD intervals.

    Venn-Abers predictors produce automatically well-calibrated probability
    intervals [p0, p1] with theoretical guarantees (Vovk & Petej, 2014).

    Uses MAPIE's native VennAbersCalibrator (mapie.calibration) when available,
    falling back to the external venn_abers library otherwise.

    Args:
        classifier: Fitted classifier with predict_proba.
        X_cal: Calibration features (required for MAPIE path).
        y_cal: Calibration labels.
        X_test: Test features (required for MAPIE path).

    Returns:
        Tuple of (y_pred_point, p0_array, p1_array) where:
        - y_pred_point: positive-class probability as point estimate.
        - p0_array: lower probability bound per observation.
        - p1_array: upper probability bound per observation.
    """
    y_cal_arr = np.asarray(y_cal.values if hasattr(y_cal, "values") else y_cal, dtype=int).reshape(
        -1
    )

    # ── MAPIE native path (preferred) ─────────────────────────────────────────
    try:
        from mapie.calibration import VennAbersCalibrator

        va = VennAbersCalibrator(estimator=classifier, inductive=True, random_state=42)
        va.fit(
            X_cal,
            y_cal_arr,
            X_calib=X_cal,
            y_calib=y_cal_arr,
        )
        probs = va.predict_proba(X_test)  # shape (n, 2): [p_neg, p_pos]
        p_low_raw = np.clip(np.asarray(probs[:, 0], dtype=float), 0.0, 1.0)
        p_high_raw = np.clip(np.asarray(probs[:, 1], dtype=float), 0.0, 1.0)
        # VennAbersCalibrator returns [p0, p1] bounds for positive class
        p_low = np.minimum(p_low_raw, p_high_raw)
        p_high = np.maximum(p_low_raw, p_high_raw)
        y_pred_point = np.clip((p_low + p_high) / 2.0, 0.0, 1.0)
        avg_width = float((p_high - p_low).mean())
        logger.info(
            f"Venn-Abers PD intervals [MAPIE]: avg_width={avg_width:.4f}, n_test={len(X_test)}"
        )
        return y_pred_point, p_low, p_high

    except (ImportError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
        logger.warning(
            f"MAPIE VennAbersCalibrator failed ({exc}) — falling back to venn_abers library."
        )

    # ── External venn_abers fallback ──────────────────────────────────────────
    from venn_abers import VennAbers

    p_cal_pos = np.asarray(classifier.predict_proba(X_cal)[:, 1], dtype=float).reshape(-1)
    p_cal = np.column_stack([1.0 - p_cal_pos, p_cal_pos])
    p_test_pos = np.asarray(classifier.predict_proba(X_test)[:, 1], dtype=float).reshape(-1)
    p_test = np.column_stack([1.0 - p_test_pos, p_test_pos])

    wrapped = VennAbers()
    wrapped.fit(p_cal, y_cal_arr)
    y_pred_binary, p_result = wrapped.predict_proba(p_test)
    p0 = np.clip(np.asarray(p_result[:, 0], dtype=float), 0.0, 1.0)
    p1 = np.clip(np.asarray(p_result[:, 1], dtype=float), 0.0, 1.0)

    p_low = np.minimum(p0, p1)
    p_high = np.maximum(p0, p1)
    y_pred_point = np.clip(np.asarray(y_pred_binary[:, 1], dtype=float), 0.0, 1.0)

    avg_width = float((p_high - p_low).mean())
    logger.info(
        f"Venn-Abers PD intervals [fallback]: avg_width={avg_width:.4f}, n_test={len(X_test)}"
    )
    return y_pred_point, p_low, p_high


def create_residual_intervals(
    regressor,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate naive residual-based prediction intervals (benchmark).

    Uses the calibration residual distribution to build percentile-based
    intervals. Does NOT have conformal coverage guarantees — serves as a
    non-conformal baseline for comparison.

    Args:
        regressor: Fitted model with predict (or predict_proba for classifiers).
        X_cal: Calibration features.
        y_cal: Calibration labels (float).
        X_test: Test features.
        alpha: Significance level (e.g., 0.10 for 90% intervals).

    Returns:
        Tuple of (y_pred, y_intervals) where y_intervals is (n, 2).
    """
    # Get calibration predictions
    if hasattr(regressor, "predict_proba"):
        cal_preds = regressor.predict_proba(X_cal)[:, 1]
        test_preds = regressor.predict_proba(X_test)[:, 1]
    else:
        cal_preds = regressor.predict(X_cal)
        test_preds = regressor.predict(X_test)

    cal_preds = np.asarray(cal_preds, dtype=float)
    test_preds = np.asarray(test_preds, dtype=float)
    y_cal_arr = np.asarray(y_cal, dtype=float)

    # Residuals on calibration set
    residuals = y_cal_arr - cal_preds

    # Percentile-based interval from residual distribution
    q_low = np.percentile(residuals, 100 * (alpha / 2))
    q_high = np.percentile(residuals, 100 * (1 - alpha / 2))

    low = test_preds + q_low
    high = test_preds + q_high
    y_intervals = np.column_stack([low, high])

    avg_width = float((high - low).mean())
    logger.info(f"Residual intervals (bootstrap-style, alpha={alpha}): avg_width={avg_width:.4f}")
    return test_preds, y_intervals


def validate_coverage(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Validate empirical coverage against target."""
    low = y_intervals[:, 0]
    high = y_intervals[:, 1]
    covered = ((y_true >= low) & (y_true <= high)).mean()
    target = 1 - alpha

    metrics = {
        "empirical_coverage": float(covered),
        "target_coverage": float(target),
        "coverage_gap": float(abs(covered - target)),
        "avg_interval_width": float((high - low).mean()),
        "median_interval_width": float(np.median(high - low)),
    }
    logger.info(f"Coverage validation: empirical={covered:.4f} vs target={target:.4f}")
    return metrics
