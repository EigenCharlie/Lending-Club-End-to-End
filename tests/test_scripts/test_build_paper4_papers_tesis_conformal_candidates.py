from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.papers.build_paper4_papers_tesis_conformal_candidates import (
    CALIBRATION_END,
    _split_replay,
    finite_sample_quantile,
    weighted_quantile,
)


def test_weighted_quantile_moves_toward_upweighted_tail() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    unweighted = weighted_quantile(scores, np.ones_like(scores), alpha=0.25)
    weighted = weighted_quantile(scores, np.array([1.0, 1.0, 1.0, 20.0]), alpha=0.25)

    assert unweighted == pytest.approx(0.3)
    assert weighted == pytest.approx(0.9)


def test_finite_sample_quantile_uses_higher_order_statistic() -> None:
    assert finite_sample_quantile(np.array([0.1, 0.2, 0.3, 0.4]), alpha=0.25) == pytest.approx(0.4)


def test_split_replay_uses_2018_for_calibration_and_later_holdout() -> None:
    frame = pd.DataFrame(
        {
            "issue_month": [
                CALIBRATION_END - pd.Timedelta(days=1),
                CALIBRATION_END + pd.Timedelta(days=1),
            ],
            "y_true": [0.0, 1.0],
            "y_pred": [0.2, 0.7],
            "score_abs": [0.2, 0.3],
        }
    )

    calibration, holdout = _split_replay(frame)

    assert len(calibration) == 1
    assert len(holdout) == 1
    assert calibration["issue_month"].max() <= CALIBRATION_END
    assert holdout["issue_month"].min() > CALIBRATION_END
