"""Tests for scripts/estimate_causal_effects.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import estimate_causal_effects as estimate_mod


class DummyEstimator:
    def const_marginal_effect(self, X: pd.DataFrame) -> np.ndarray:
        return np.linspace(0.01, 0.02, len(X))

    def const_marginal_effect_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        effect = self.const_marginal_effect(X)
        return effect - 0.001, effect + 0.001

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        return self.const_marginal_effect(X)

    def effect_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.const_marginal_effect_interval(X, alpha=alpha)


def test_estimate_causal_effects_writes_official_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "int_rate": [10.0, 12.0, 14.0, 16.0],
            "default_flag": [0, 1, 0, 1],
            "loan_amnt": [5000, 7000, 9000, 11000],
            "annual_inc": [50000, 55000, 60000, 65000],
            "dti": [10.0, 12.0, 14.0, 16.0],
            "fico_range_low": [680, 670, 660, 650],
            "grade_woe": [0.2, 0.3, 0.4, 0.5],
            "purpose_woe": [0.1, 0.2, 0.3, 0.4],
            "home_ownership_woe": [0.1, 0.1, 0.2, 0.2],
            "grade": ["A", "B", "C", "D"],
            "purpose": ["debt_consolidation"] * 4,
            "home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE"],
        }
    )
    test = train.assign(id=["10", "11", "12", "13"], default_flag=[0, 0, 1, 1])
    train.to_parquet(data_dir / "train_fe.parquet", index=False)
    test.to_parquet(data_dir / "test_fe.parquet", index=False)

    monkeypatch.setattr(
        estimate_mod,
        "estimate_ate_dowhy",
        lambda **kwargs: {
            "ate": 0.0123,
            "ate_ci": [0.0101, 0.0145],
            "identified_estimand": "backdoor estimand",
            "identification_strategy": "backdoor",
            "refutation_summary": [
                {"test": "placebo_treatment", "result": "ok"},
                {"test": "random_common_cause", "result": "ok"},
                {"test": "data_subset", "result": "ok"},
            ],
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "estimate_cate",
        lambda Y, T, X, W, **kwargs: (
            DummyEstimator(),
            np.linspace(0.01, 0.02, len(X)),
            (np.linspace(0.009, 0.019, len(X)), np.linspace(0.011, 0.021, len(X))),
        ),
    )

    estimate_mod.main(treatment="int_rate", run_tag="run-causal-test")

    status = json.loads((model_dir / "causal_effect_status.json").read_text(encoding="utf-8"))
    cate_train = pd.read_parquet(data_dir / "cate_estimates.parquet")
    cate_oot = pd.read_parquet(data_dir / "cate_estimates_oot.parquet")
    overlap = pd.read_parquet(data_dir / "causal_overlap_diagnostics.parquet")

    assert status["run_tag"] == "run-causal-test"
    assert status["treatment"] == "int_rate"
    assert status["identification_strategy"] == "backdoor"
    assert status["identification_valid"] is True
    assert status["role"] == "insights_only"
    assert status["promotion_eligible"] is False
    assert status["continuous_treatment_semantics"]["estimand"] == "const_marginal_effect"
    assert len(status["refutation_summary"]) == 3
    assert list(cate_train.columns[:4]) == ["id", "cate", "cate_lb", "cate_ub"]
    assert len(cate_oot) == len(test)
    assert {"segment_type", "segment_value", "support_ok"}.issubset(overlap.columns)


def test_estimate_causal_effects_fails_when_required_columns_are_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "id": ["1", "2"],
            "int_rate": [10.0, 12.0],
            "default_flag": [0, 1],
            "loan_amnt": [5000, 7000],
            "annual_inc": [50000, 55000],
            "dti": [10.0, 12.0],
            "grade_woe": [0.2, 0.3],
        }
    )
    train.to_parquet(data_dir / "train_fe.parquet", index=False)
    train.to_parquet(data_dir / "test_fe.parquet", index=False)

    try:
        estimate_mod.main(treatment="int_rate", run_tag="run-causal-missing-cols")
    except ValueError as exc:
        assert "Missing required causal columns" in str(exc)
    else:
        raise AssertionError("estimate_causal_effects should fail when required columns are absent")
