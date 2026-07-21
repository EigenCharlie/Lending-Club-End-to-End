"""Tests for scripts/run_pyepo_real_suite.py.

The tests use tiny synthetic LP instances so they validate PyEPO 1.3.7 loss
wiring without touching the full Lending Club data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pyepo = pytest.importorskip("pyepo", reason="pyepo is installed only with the spo extra")
torch = pytest.importorskip("torch", reason="torch is installed only with the spo extra")

if getattr(pyepo, "__version__", "") != "1.3.7":
    pytest.skip("PyEPO real suite tests require pyepo==1.3.7", allow_module_level=True)


def _tiny_instances(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(8, 6, 3)).astype(np.float32)
    # Continuous costs reduce tie risk in top-k LP solutions.
    costs = rng.normal(loc=-0.05, scale=0.2, size=(8, 6)).astype(np.float32)
    return X, costs


def test_pfyl_positive_shift_preserves_fixed_budget_solutions() -> None:
    from scripts.run_pyepo_real_suite import CreditPortfolioTopKOracle, _shift_costs_positive

    _, costs = _tiny_instances(seed=11)
    shifted, shifts = _shift_costs_positive(costs)

    assert shifted.min() > 0
    assert shifts.shape == (len(costs),)

    lp = CreditPortfolioTopKOracle(n_items=6, budget=2)
    for original, positive in zip(costs, shifted, strict=True):
        lp.setObj(original)
        sol_original, _ = lp.solve()
        lp.setObj(positive)
        sol_shifted, _ = lp.solve()
        np.testing.assert_allclose(sol_shifted, sol_original, atol=1e-6)


def test_topk_oracle_matches_fixed_budget_argmin() -> None:
    from scripts.run_pyepo_real_suite import CreditPortfolioTopKOracle

    costs = np.array([0.7, -0.3, -0.3, 0.2, 1.1, -0.1], dtype=np.float32)
    oracle = CreditPortfolioTopKOracle(n_items=6, budget=3)
    oracle.setObj(costs)
    sol, obj = oracle.solve()

    # Ties are broken by index so the solution is deterministic across runs.
    assert sol == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert obj == pytest.approx(float(costs[[1, 2, 5]].sum()))


@pytest.mark.parametrize("method", ["spo_plus", "rfyl", "pfyl_mul", "pairwise_ltr", "cave"])
def test_each_pyepo_loss_smoke_trains_one_epoch(method: str) -> None:
    from scripts.run_pyepo_real_suite import (
        _build_opt_dataset,
        _predict_model,
        _shift_costs_positive,
        _train_pyepo_method,
    )

    X, costs = _tiny_instances(seed=21)
    train_costs = costs
    if method == "pfyl_mul":
        train_costs, _ = _shift_costs_positive(costs)

    dataset_kind = "constrs" if method == "cave" else "standard"
    dataset = _build_opt_dataset(
        X,
        train_costs,
        n_items=6,
        budget=2,
        label=f"test-{method}",
        kind=dataset_kind,
    )
    model, losses = _train_pyepo_method(
        method,
        dataset,
        n_features=3,
        n_items=6,
        budget=2,
        epochs=1,
        lr=1e-3,
        batch_size=4,
        seed=123,
    )
    pred = _predict_model(model, X)

    assert len(losses) == 1
    assert np.isfinite(losses[0])
    assert pred.shape == (8, 6)
    assert np.isfinite(pred).all()


def test_tiny_spo_smoke_reproducible_with_same_seed() -> None:
    from scripts.run_pyepo_real_suite import (
        CreditPortfolioTopKOracle,
        _build_opt_dataset,
        _predict_model,
        _train_pyepo_method,
    )
    from scripts.run_spo_real import _compute_regret, _compute_true_optima

    X, costs = _tiny_instances(seed=31)
    dataset = _build_opt_dataset(X, costs, n_items=6, budget=2, label="repro")

    def run_once() -> tuple[float, float]:
        model, losses = _train_pyepo_method(
            "spo_plus",
            dataset,
            n_features=3,
            n_items=6,
            budget=2,
            epochs=1,
            lr=1e-3,
            batch_size=4,
            seed=999,
        )
        pred = _predict_model(model, X)
        lp = CreditPortfolioTopKOracle(n_items=6, budget=2)
        true_optima = _compute_true_optima(costs, lp)
        regret = _compute_regret(pred, costs, lp.copy(), true_optima).mean()
        return float(losses[-1]), float(regret)

    first = run_once()
    second = run_once()
    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


def test_summary_and_artifact_schema_roundtrip(tmp_path) -> None:
    from scripts.run_pyepo_real_suite import _summarize_regrets

    rows = []
    for seed in [42, 1042]:
        for instance_id in range(3):
            rows.append(
                {
                    "seed": seed,
                    "period": "pooled",
                    "instance_id": instance_id,
                    "method": "two_stage",
                    "method_display": "Two-stage Ridge",
                    "regret": 0.20 + 0.01 * instance_id,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "period": "pooled",
                    "instance_id": instance_id,
                    "method": "spo_plus",
                    "method_display": "SPO+",
                    "regret": 0.10 + 0.01 * instance_id,
                }
            )
    regrets = pd.DataFrame(rows)
    summary = _summarize_regrets(regrets)

    required_summary_cols = {
        "method",
        "mean_regret",
        "std_regret",
        "improvement_vs_two_stage_pct",
        "auditability_score",
    }
    assert required_summary_cols.issubset(summary.columns)
    assert summary.loc[summary["method"] == "spo_plus", "improvement_vs_two_stage_pct"].iloc[0] > 0

    parquet_path = tmp_path / "regrets.parquet"
    summary_path = tmp_path / "summary.csv"
    status_path = tmp_path / "status.json"
    regrets.to_parquet(parquet_path, index=False)
    summary.to_csv(summary_path, index=False)
    status_path.write_text(json.dumps({"results": summary.to_dict(orient="records")}))

    assert set(pd.read_parquet(parquet_path).columns) == set(regrets.columns)
    assert set(pd.read_csv(summary_path).columns) == set(summary.columns)
    assert json.loads(status_path.read_text())["results"]
