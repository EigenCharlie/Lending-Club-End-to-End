from __future__ import annotations

import pandas as pd

from scripts.run_pd_rapids_benchmark import _sample_train_val, _score_binary


def test_sample_train_val_respects_caps() -> None:
    df = pd.DataFrame(
        {
            "issue_d": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "default_flag": [0, 1] * 500,
        }
    )
    fit, val = _sample_train_val(df, fit_sample_size=200, val_sample_size=50)
    assert len(fit) == 200
    assert len(val) == 50
    assert fit["issue_d"].max() < val["issue_d"].min()


def test_score_binary_has_expected_keys() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_prob = pd.Series([0.1, 0.9, 0.3, 0.8, 0.7, 0.2]).to_numpy()
    scores = _score_binary(y_true, y_prob)
    assert set(scores) == {"auc", "brier", "ece"}
    assert 0.0 <= scores["auc"] <= 1.0
