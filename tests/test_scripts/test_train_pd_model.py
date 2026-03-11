"""Unit tests for train_pd_model helpers (walk-forward + seed replay)."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from scripts import train_pd_model as train_mod


class _FakeState:
    def __init__(self, name: str = "COMPLETE") -> None:
        self.name = name


class _FakeTrial:
    def __init__(self, number: int, value: float, params: dict, user_attrs: dict | None = None):
        self.number = number
        self.value = value
        self.params = params
        self.user_attrs = user_attrs or {}
        self.state = _FakeState("COMPLETE")


class _FakeStudy:
    def __init__(self, trials):
        self.trials = trials


def test_build_walk_forward_splits_missing_date_col() -> None:
    n_rows = 40_000
    df = pd.DataFrame({"x": np.arange(n_rows), "default_flag": np.random.randint(0, 2, n_rows)})
    splits = train_mod._build_walk_forward_splits(
        df,
        n_windows=2,
        min_train_rows=12_000,
        window_rows=12_000,
        date_col="missing_col",
        max_rows=30_000,
    )
    assert len(splits) >= 1
    idx_fit, idx_eval = splits[0]
    assert len(idx_fit) >= 12_000
    assert len(idx_eval) >= 10_000


def test_replay_top_optuna_trials_prioritizes_gate_pass(monkeypatch) -> None:
    trials = [
        _FakeTrial(1, 0.72, {"trial_id": "A"}, user_attrs={"fairness_pass": False}),
        _FakeTrial(
            2,
            0.719,
            {"trial_id": "B"},
            user_attrs={"fairness_pass": True, "conformal_pass": True, "governance_pass": True},
        ),
    ]

    fake_optuna = types.SimpleNamespace(load_study=lambda **_kwargs: _FakeStudy(trials))
    monkeypatch.setitem(__import__("sys").modules, "optuna", fake_optuna)

    class _FakeModel:
        def __init__(self, trial_id: str) -> None:
            self.trial_id = trial_id

        def predict_proba(self, X):
            n = len(X)
            p1 = np.full(n, 0.95) if self.trial_id == "A" else np.full(n, 0.65)
            return np.column_stack([1.0 - p1, p1])

    def _fake_train(*_args, params=None, **_kwargs):
        trial_id = str((params or {}).get("trial_id", "A"))
        model = _FakeModel(trial_id)
        return model, {"validation_auc": 0.70 if trial_id == "A" else 0.69, "best_iteration": 10}

    monkeypatch.setattr(train_mod, "train_catboost_default", _fake_train)

    X = pd.DataFrame({"f": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])
    report = train_mod._replay_top_optuna_trials(
        hpo_cfg={"enabled": True, "study_storage": "sqlite:///tmp.db", "study_name": "x"},
        base_params={},
        X_train_fit_cb=X,
        y_train_fit=y,
        X_val_cb=X,
        y_val=y,
        cat_features=[],
        seeds=[42],
        top_k_trials=2,
        prioritize_gate_pass=True,
    )

    assert report["enabled"] is True
    assert report["selected_trial"] == 2


def test_replay_top_optuna_trials_uses_ece_when_gate_not_prioritized(monkeypatch) -> None:
    trials = [
        _FakeTrial(1, 0.72, {"trial_id": "A"}),
        _FakeTrial(2, 0.719, {"trial_id": "B"}),
    ]

    fake_optuna = types.SimpleNamespace(load_study=lambda **_kwargs: _FakeStudy(trials))
    monkeypatch.setitem(__import__("sys").modules, "optuna", fake_optuna)

    class _FakeModel:
        def __init__(self, trial_id: str) -> None:
            self.trial_id = trial_id

        def predict_proba(self, X):
            n = len(X)
            if self.trial_id == "A":
                p1 = np.array([0.05, 0.95, 0.05, 0.95][:n])
            else:
                p1 = np.array([0.40, 0.60, 0.40, 0.60][:n])
            return np.column_stack([1.0 - p1, p1])

    def _fake_train(*_args, params=None, **_kwargs):
        trial_id = str((params or {}).get("trial_id", "A"))
        model = _FakeModel(trial_id)
        return model, {"validation_auc": 0.69 if trial_id == "A" else 0.68, "best_iteration": 9}

    monkeypatch.setattr(train_mod, "train_catboost_default", _fake_train)

    X = pd.DataFrame({"f": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])
    report = train_mod._replay_top_optuna_trials(
        hpo_cfg={"enabled": True, "study_storage": "sqlite:///tmp.db", "study_name": "x"},
        base_params={},
        X_train_fit_cb=X,
        y_train_fit=y,
        X_val_cb=X,
        y_val=y,
        cat_features=[],
        seeds=[42],
        top_k_trials=2,
        prioritize_gate_pass=False,
    )

    assert report["enabled"] is True
    assert report["selected_trial"] == 1


def test_apply_training_regime_recent_window_keeps_latest_quarters() -> None:
    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(
                [
                    "2019-01-01",
                    "2019-04-01",
                    "2019-07-01",
                    "2019-10-01",
                    "2020-01-01",
                ]
            ),
            "default_flag": [0, 1, 0, 1, 0],
        }
    )
    out, meta = train_mod._apply_training_regime(
        df,
        {"mode": "recent_12q", "recent_window_quarters": 2},
        date_col="issue_d",
    )

    assert len(out) == 2
    assert meta["mode"] == "recent_12q"
    assert meta["recent_window_quarters"] == 2


def test_apply_training_regime_full_weighted_emits_recency_weights() -> None:
    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2019-01-01", "2019-04-01", "2020-01-01"]),
            "default_flag": [0, 1, 0],
        }
    )
    out, meta = train_mod._apply_training_regime(
        df,
        {"mode": "full_weighted", "half_life_quarters": 4},
        date_col="issue_d",
    )

    assert "_recency_weight" in out.columns
    assert float(out["_recency_weight"].iloc[-1]) >= float(out["_recency_weight"].iloc[0])
    assert meta["mode"] == "full_weighted"
