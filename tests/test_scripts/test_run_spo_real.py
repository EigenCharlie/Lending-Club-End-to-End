"""Tests for scripts/run_spo_real.py — unit tests for key components.

Tests the building blocks independently of real data files (which are too
large to load in unit tests). Covers the LP model, point-wise MLP architecture,
regret computation, and instance sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (needs spo extra)")

# ── CreditPortfolioLP ────────────────────────────────────────────────────────


def test_portfolio_lp_selects_exactly_budget_items() -> None:
    from scripts.run_spo_real import CreditPortfolioLP

    n_items, budget = 10, 3
    lp = CreditPortfolioLP(n_items=n_items, budget=budget)
    # Costs: items 0-2 cheapest
    costs = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.95], dtype=float)
    lp.setObj(costs)
    sol, obj = lp.solve()

    assert len(sol) == n_items
    assert abs(sum(sol) - budget) < 1e-6, f"Expected {budget} selected, got {sum(sol):.4f}"


def test_portfolio_lp_selects_cheapest_items() -> None:
    from scripts.run_spo_real import CreditPortfolioLP

    n_items, budget = 6, 2
    lp = CreditPortfolioLP(n_items=n_items, budget=budget)
    costs = np.array([-0.5, -0.3, 0.1, 0.2, 0.3, 0.4], dtype=float)
    lp.setObj(costs)
    sol, _ = lp.solve()

    selected = [i for i, v in enumerate(sol) if v > 0.5]
    # Items 0 and 1 are cheapest (most negative → best)
    assert 0 in selected
    assert 1 in selected


def test_portfolio_lp_copy_is_independent() -> None:
    from scripts.run_spo_real import CreditPortfolioLP

    lp = CreditPortfolioLP(n_items=5, budget=2)
    lp_copy = lp.copy()
    assert lp_copy.n_items == 5
    assert lp_copy.budget == 2
    assert lp_copy is not lp


# ── PDPredictorMLP ────────────────────────────────────────────────────────────


def test_mlp_output_shape() -> None:
    """Point-wise MLP must output (B, n_items) from (B, n_items * n_features)."""
    from scripts.run_spo_real import PDPredictorMLP

    n_features, n_items, batch = 10, 20, 8
    model = PDPredictorMLP(n_features=n_features, n_items=n_items)
    x = torch.randn(batch, n_items * n_features)
    out = model(x)
    assert out.shape == (batch, n_items), f"Expected ({batch}, {n_items}), got {out.shape}"


def test_mlp_permutation_equivariance() -> None:
    """Shuffling items within an instance should consistently shuffle predictions."""
    from scripts.run_spo_real import PDPredictorMLP

    n_features, n_items = 8, 15
    model = PDPredictorMLP(n_features=n_features, n_items=n_items)
    model.eval()

    rng = np.random.RandomState(0)
    feats = rng.randn(n_items, n_features).astype(np.float32)  # (n_items, n_features)
    perm = np.random.permutation(n_items)

    x_orig = torch.tensor(feats.reshape(1, -1))
    x_perm = torch.tensor(feats[perm].reshape(1, -1))

    with torch.no_grad():
        out_orig = model(x_orig)[0].numpy()  # (n_items,)
        out_perm = model(x_perm)[0].numpy()  # (n_items,)

    # out_perm should equal out_orig reordered by perm
    np.testing.assert_allclose(
        out_perm, out_orig[perm], atol=1e-5, err_msg="Point-wise MLP is not permutation-equivariant"
    )


def test_mlp_shared_weights_across_items() -> None:
    """Same loan features at different positions should get same cost prediction."""
    from scripts.run_spo_real import PDPredictorMLP

    n_features, n_items = 6, 10
    model = PDPredictorMLP(n_features=n_features, n_items=n_items)
    model.eval()

    # Build instance where item 0 and item 3 have identical features
    feats = torch.randn(n_items, n_features)
    feats[3] = feats[0]
    x = feats.reshape(1, -1)

    with torch.no_grad():
        out = model(x)[0].numpy()

    assert abs(out[0] - out[3]) < 1e-5, "Identical loan features should produce identical costs"


# ── Instance sampling ─────────────────────────────────────────────────────────


def test_sample_instances_shapes() -> None:
    from scripts.run_spo_real import _sample_instances

    n_loans, n_feats, n_items, n_inst = 500, 10, 20, 50
    rng = np.random.RandomState(42)
    X = np.random.randn(n_loans, n_feats).astype(np.float32)
    c = np.random.randn(n_loans).astype(np.float32)

    feats, costs, indices = _sample_instances(X, c, n_items, n_inst, rng)

    assert feats.shape == (n_inst, n_items, n_feats)
    assert costs.shape == (n_inst, n_items)
    assert indices.shape == (n_inst, n_items)


def test_sample_instances_no_duplicates_within_instance() -> None:
    from scripts.run_spo_real import _sample_instances

    n_loans, n_feats, n_items, n_inst = 200, 5, 30, 20
    rng = np.random.RandomState(7)
    X = np.random.randn(n_loans, n_feats).astype(np.float32)
    c = np.random.randn(n_loans).astype(np.float32)

    _, _, indices = _sample_instances(X, c, n_items, n_inst, rng)

    for i in range(n_inst):
        assert len(set(indices[i])) == n_items, f"Instance {i} has duplicate loan indices"


# ── Regret computation ────────────────────────────────────────────────────────


def test_zero_regret_with_true_costs() -> None:
    """Regret must be 0 when predicted costs == true costs (optimal prediction)."""
    from scripts.run_spo_real import CreditPortfolioLP, _compute_regret, _compute_true_optima

    n_items, budget, n_inst = 8, 3, 5
    rng = np.random.RandomState(99)
    lp = CreditPortfolioLP(n_items=n_items, budget=budget)

    c_true = rng.randn(n_inst, n_items).astype(np.float32)
    true_optima = _compute_true_optima(c_true, lp)
    regrets = _compute_regret(c_true, c_true, lp.copy(), true_optima)

    assert np.allclose(regrets, 0.0, atol=1e-5), f"Expected zero regret, got {regrets}"


def test_nonnegative_regret() -> None:
    """Regret is always >= 0 (any prediction is at least as bad as optimal)."""
    from scripts.run_spo_real import CreditPortfolioLP, _compute_regret, _compute_true_optima

    n_items, budget, n_inst = 10, 4, 20
    rng = np.random.RandomState(123)
    lp = CreditPortfolioLP(n_items=n_items, budget=budget)

    c_true = rng.randn(n_inst, n_items).astype(np.float32)
    c_pred = rng.randn(n_inst, n_items).astype(np.float32)  # random prediction

    true_optima = _compute_true_optima(c_true, lp)
    regrets = _compute_regret(c_pred, c_true, lp.copy(), true_optima)

    assert (regrets >= -1e-6).all(), f"Found negative regret: {regrets.min():.6f}"


# ── index_costs helper ────────────────────────────────────────────────────────


def test_index_costs_shape_and_values() -> None:
    from scripts.run_spo_real import _index_costs

    n_loans, n_items, n_inst = 100, 10, 15
    rng = np.random.RandomState(0)
    c_all = rng.randn(n_loans).astype(np.float32)
    indices = rng.choice(n_loans, size=(n_inst, n_items), replace=True)

    result = _index_costs(c_all, indices)
    assert result.shape == (n_inst, n_items)
    # Verify a specific value
    assert result[3, 7] == c_all[indices[3, 7]]
