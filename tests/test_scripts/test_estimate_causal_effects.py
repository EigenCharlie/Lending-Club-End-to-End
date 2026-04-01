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
        "estimate_ate_linear_dml",
        lambda **kwargs: {
            "ate": 0.0111,
            "ate_ci": [0.0091, 0.0131],
            "estimator": DummyEstimator(),
            "estimator_family": "linear_dml",
        },
    )
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
        "build_sensitivity_status",
        lambda estimator, **kwargs: {
            "sensitivity_supported": True,
            "sensitivity_pass": True,
            "robustness_value": 0.2,
            "sensitivity_interval": [0.001, 0.02],
            "sensitivity_summary": "ok",
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "estimate_cate_candidates",
        lambda **kwargs: {
            "selected_name": "causal_forest_dml",
            "selected": {
                "estimator": DummyEstimator(),
                "cate": np.linspace(0.01, 0.02, len(kwargs["X"])),
                "cate_lb": np.linspace(0.009, 0.019, len(kwargs["X"])),
                "cate_ub": np.linspace(0.011, 0.021, len(kwargs["X"])),
            },
            "candidates": {
                "causal_forest_dml": {
                    "estimator_family": "causal_forest_dml",
                    "cate_mean": 0.015,
                    "cate_std": 0.003,
                    "selection_score": 0.1,
                }
            },
            "failures": {},
            "selection_reason": "rscorer",
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "inspect_causal_environment",
        lambda: {"compatible": True, "packages": {"econml": {"installed": None}}},
    )
    monkeypatch.setattr(
        estimate_mod,
        "evaluate_overlap_status",
        lambda overlap, **kwargs: {
            "overlap_pass": True,
            "support_ok_share": 1.0,
            "failing_segments": [],
        },
    )

    estimate_mod.main(treatment="int_rate", run_tag="run-causal-test")

    status = json.loads((model_dir / "causal_effect_status.json").read_text(encoding="utf-8"))
    cate_train = pd.read_parquet(data_dir / "cate_estimates.parquet")
    cate_oot = pd.read_parquet(data_dir / "cate_estimates_oot.parquet")
    overlap = pd.read_parquet(data_dir / "causal_overlap_diagnostics.parquet")

    overlap_status = json.loads(
        (model_dir / "causal_overlap_status.json").read_text(encoding="utf-8")
    )
    sensitivity = json.loads(
        (model_dir / "causal_sensitivity_status.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (model_dir / "causal_estimator_selection_status.json").read_text(encoding="utf-8")
    )

    assert status["run_tag"] == "run-causal-test"
    assert status["treatment"] == "int_rate"
    assert status["identification_strategy"] == "orthogonal_dml_with_dowhy_audit"
    assert status["identification_valid"] is True
    assert status["role"] == "insights_only"
    assert status["promotion_eligible"] is False
    assert status["continuous_treatment_semantics"]["estimand"] == "const_marginal_effect"
    assert len(status["refutation_summary"]) == 3
    assert status["overlap_pass"] is True
    assert status["sensitivity_pass"] is True
    assert list(cate_train.columns[:4]) == ["id", "cate", "cate_lb", "cate_ub"]
    assert len(cate_oot) == len(test)
    assert {"segment_type", "segment_value", "support_ok"}.issubset(overlap.columns)
    assert overlap_status["overlap_pass"] is True
    assert sensitivity["sensitivity_pass"] is True
    assert selection["selected_estimator_family"] == "causal_forest_dml"


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


def test_estimate_causal_effects_sanitizes_covariate_nans_consistently(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "int_rate": [10.0, 12.0, 14.0],
            "default_flag": [0, 1, 0],
            "loan_amnt": [5000, 7000, 9000],
            "annual_inc": [50000, 55000, 60000],
            "dti": [10.0, np.nan, 14.0],
            "fico_range_low": [680, 670, 660],
            "grade_woe": [0.2, 0.3, 0.4],
            "purpose_woe": [0.1, 0.2, 0.3],
            "home_ownership_woe": [0.1, 0.1, 0.2],
            "grade": ["A", "B", "C"],
            "purpose": ["debt_consolidation"] * 3,
            "home_ownership": ["RENT", "OWN", "RENT"],
        }
    )
    test = train.assign(id=["10", "11", "12"], default_flag=[0, 0, 1])
    train.to_parquet(data_dir / "train_fe.parquet", index=False)
    test.to_parquet(data_dir / "test_fe.parquet", index=False)

    monkeypatch.setattr(
        estimate_mod,
        "estimate_ate_linear_dml",
        lambda **kwargs: {
            "ate": 0.0111,
            "ate_ci": [0.0091, 0.0131],
            "estimator": DummyEstimator(),
            "estimator_family": "linear_dml",
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "estimate_ate_dowhy",
        lambda **kwargs: {
            "ate": 0.0123,
            "ate_ci": [0.0101, 0.0145],
            "identified_estimand": "backdoor estimand",
            "identification_strategy": "backdoor",
            "refutation_summary": [],
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "build_sensitivity_status",
        lambda estimator, **kwargs: {
            "sensitivity_supported": True,
            "sensitivity_pass": True,
            "robustness_value": 0.2,
            "sensitivity_interval": [0.001, 0.02],
            "sensitivity_summary": "ok",
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "estimate_cate_candidates",
        lambda **kwargs: {
            "selected_name": "causal_forest_dml",
            "selected": {
                "estimator": DummyEstimator(),
                "cate": np.linspace(0.01, 0.02, len(kwargs["X"])),
                "cate_lb": np.linspace(0.009, 0.019, len(kwargs["X"])),
                "cate_ub": np.linspace(0.011, 0.021, len(kwargs["X"])),
            },
            "candidates": {
                "causal_forest_dml": {
                    "estimator_family": "causal_forest_dml",
                    "cate_mean": 0.015,
                    "cate_std": 0.003,
                    "selection_score": 0.1,
                }
            },
            "failures": {},
            "selection_reason": "rscorer",
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "inspect_causal_environment",
        lambda: {"compatible": True, "packages": {"econml": {"installed": None}}},
    )
    monkeypatch.setattr(
        estimate_mod,
        "evaluate_overlap_status",
        lambda overlap, **kwargs: {
            "overlap_pass": True,
            "support_ok_share": 1.0,
            "failing_segments": [],
        },
    )
    monkeypatch.setattr(
        estimate_mod,
        "load_causal_config",
        lambda *args, **kwargs: {
            "data": {
                "max_covariate_missing_rate": 0.5,
                "max_row_drop_rate": 0.5,
                "impute_covariates": "median",
            },
            "overlap": {"min_support_ok_share": 0.5},
            "estimators": {
                "linear_dml": {"cv": 3, "mc_iters": 1},
                "cate_candidates": ["causal_forest_dml", "linear_dml"],
                "causal_forest_dml": {"cv": 3, "mc_iters": 1},
            },
            "sensitivity": {
                "min_robustness_value": 0.05,
                "alpha": 0.05,
                "c_y": 0.05,
                "c_t": 0.05,
                "rho": 1.0,
            },
            "defaults": {
                "treatment_unit": "percentage_points",
                "policy_value_method": "local_cate_discrete_grid",
            },
        },
    )

    estimate_mod.main(treatment="int_rate", run_tag="run-causal-nan")
    status = json.loads((model_dir / "causal_effect_status.json").read_text(encoding="utf-8"))

    assert status["n_rows_input"] == 3
    assert status["n_rows_dropped_nonfinite"] == 0
    assert status["drop_rate"] == 0.0
