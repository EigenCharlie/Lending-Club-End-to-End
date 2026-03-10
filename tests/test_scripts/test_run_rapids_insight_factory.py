from __future__ import annotations

import pandas as pd

from scripts.run_rapids_insight_factory import _prepare_issue_quarter, _select_numeric_features


def test_select_numeric_features_prefers_core_numeric_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "default_flag": [0, 1],
            "loan_to_income": [0.2, 0.3],
            "fico_score": [700, 710],
            "random_num": [1.0, 2.0],
            "purpose": ["credit_card", "debt_consolidation"],
        }
    )
    cols = _select_numeric_features(df, max_features=4)
    assert "id" not in cols
    assert "default_flag" not in cols
    assert cols[:2] == ["loan_to_income", "fico_score"]


def test_prepare_issue_quarter_formats_quarter_strings() -> None:
    series = pd.Series(["2018-01-15", "2019-09-01", None])
    out = _prepare_issue_quarter(series)
    assert out.iloc[0].startswith("2018Q")
    assert out.iloc[1].startswith("2019Q")
