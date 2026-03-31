from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.monotonicity import (
    adjacent_monotonicity_report,
    monotonicity_status,
    pd_band_summary,
)


def test_monotonicity_summary_detects_clean_monotone_structure() -> None:
    y_true = np.array([0] * 40 + [1] * 60, dtype=float)
    pd_scores = np.linspace(0.05, 0.95, 100)

    summary = pd_band_summary(y_true=y_true, pd_scores=pd_scores, n_bands=5)
    pairs = adjacent_monotonicity_report(summary)
    status = monotonicity_status(summary, pairs)

    assert len(summary) >= 2
    assert int(status["n_disruptions"]) == 0
    assert bool(status["overall_pass"]) is True


def test_monotonicity_summary_detects_adjacent_disruption() -> None:
    summary = pd.DataFrame(
        {
            "band": ["B1", "B2", "B3"],
            "n_obs": [100, 100, 100],
            "mean_predicted_pd": [0.10, 0.20, 0.30],
            "observed_default_rate": [0.08, 0.25, 0.20],
            "rate_gap": [-0.02, 0.05, -0.10],
        }
    )

    pairs = adjacent_monotonicity_report(summary)
    status = monotonicity_status(summary, pairs)

    assert pairs["disrupted"].astype(bool).sum() == 1
    assert int(status["n_disruptions"]) == 1
    assert bool(status["overall_pass"]) is False
