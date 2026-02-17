"""Survival analysis: time-to-default modeling.

Cox PH (lifelines), Random Survival Forest (scikit-survival).
Handles right-censored loans and generates lifetime PD curves for IFRS9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def train_cox_ph(
    df: pd.DataFrame,
    duration_col: str = "time_to_event",
    event_col: str = "event_observed",
    feature_cols: list[str] | None = None,
) -> tuple:
    """Train Cox Proportional Hazards model."""
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=0.01)
    cols = feature_cols + [duration_col, event_col] if feature_cols else df.columns.tolist()
    cph.fit(df[cols], duration_col=duration_col, event_col=event_col)

    logger.info(f"Cox PH — Concordance: {cph.concordance_index_:.4f}")
    cph.print_summary()
    return cph, {"concordance_index": cph.concordance_index_}


def check_cox_assumptions(cph, df: pd.DataFrame) -> None:
    """Check proportional hazards assumption."""
    cph.check_assumptions(df, show_plots=True)


def train_random_survival_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_estimators: int = 500,
) -> tuple:
    """Train Random Survival Forest.

    Args:
        y_train/y_test: Structured array with fields (event, time).
            Create with: np.array([(bool, float), ...], dtype=[('event', bool), ('time', float)])
    """
    from sksurv.ensemble import RandomSurvivalForest

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rsf.fit(X_train, y_train)

    c_index = rsf.score(X_test, y_test)
    logger.info(f"Random Survival Forest — C-index: {c_index:.4f}")
    return rsf, {"c_index": c_index}


def make_survival_target(
    df: pd.DataFrame,
    event_col: str = "default_flag",
    time_col: str = "time_to_event",
) -> np.ndarray:
    """Create structured array for scikit-survival."""
    return np.array(
        list(zip(df[event_col].astype(bool), df[time_col].astype(float), strict=False)),
        dtype=[("event", bool), ("time", float)],
    )


def generate_lifetime_pd_curve(
    model,
    X: pd.DataFrame,
    times: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate lifetime PD curves (1 - survival function) at given times.

    Useful for IFRS9 Stage 2/3 lifetime ECL calculation.
    """
    if times is None:
        times = np.arange(30, 1830, 30)  # Every month for 5 years

    surv_funcs = model.predict_survival_function(X)
    pd_curves = []
    for fn in surv_funcs:
        pds = 1 - fn(times)
        pd_curves.append(pds)

    result = pd.DataFrame(pd_curves, columns=[f"PD_t{t}" for t in times])
    logger.info(f"Generated lifetime PD curves: {result.shape}")
    return result
