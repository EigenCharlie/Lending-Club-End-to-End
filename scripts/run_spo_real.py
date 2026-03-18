"""SPO+ Real Training for Paper Estrella — v2.

Implements Smart Predict-then-Optimize (Elmachtoub & Grigas 2022) using PyEPO
with OR-Tools as the LP backend for credit portfolio selection.

Fixes from v1:
  Fix 1: Point-wise permutation-equivariant MLP (each loan processed independently).
  Fix 2: Calibrated PD (CatBoost + Venn-Abers) as true cost proxy, not binary default_flag.
  Fix 3: Multi-seed evaluation → mean±std + Wilcoxon signed-rank test vs two-stage.
  Fix 4: Conformal robust optimization as 3rd method (pd_high_90 worst-case PD cost).
  Fix 5: Default n_items=100 for richer combinatorial decision space.

Usage:
    uv run python scripts/run_spo_real.py
    uv run python scripts/run_spo_real.py --n-items 100 --budget 30 --epochs 50 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyepo
import torch
import torch.nn as nn
from loguru import logger
from ortools.linear_solver import pywraplp
from pyepo.model.opt import optModel
from scipy import stats

from src.models.conformal import apply_probability_calibrator
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

SCHEMA_VERSION = "2026-03-17.2"
LGD = 0.40
RANDOM_SEED = 42

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
    "installment",
    "emp_length",
    "pub_rec",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
]


# ── 1. Portfolio LP optModel ────────────────────────────────────────────────


class CreditPortfolioLP(optModel):
    """Portfolio selection LP: select exactly `budget` of `n_items` loans to minimize
    expected loss  c^T x = sum(x_i * (PD_i * LGD - r_i)).

    Args:
        n_items: Loans per problem instance.
        budget: Loans to select (equality constraint).
    """

    def __init__(self, n_items: int, budget: int) -> None:
        self.n_items = n_items
        self.budget = budget
        self.modelSense = pyepo.EPO.MINIMIZE
        super().__init__()

    def _getModel(self) -> tuple:
        solver = pywraplp.Solver.CreateSolver("GLOP")
        solver.SuppressOutput()
        x = {i: solver.NumVar(0.0, 1.0, f"x{i}") for i in range(self.n_items)}
        ct = solver.Constraint(float(self.budget), float(self.budget))
        for i in range(self.n_items):
            ct.SetCoefficient(x[i], 1.0)
        return solver, x

    def setObj(self, c: np.ndarray | torch.Tensor) -> None:
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        c = np.asarray(c, dtype=float)
        obj = self._model.Objective()
        obj.Clear()
        for i in range(self.n_items):
            obj.SetCoefficient(self.x[i], float(c[i]))
        obj.SetMinimization()

    def solve(self) -> tuple[list[float], float]:
        status = self._model.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            logger.warning("LP solver failed (status={}), greedy fallback", status)
            sol = [0.0] * self.n_items
            for i in range(self.budget):
                sol[i] = 1.0
            return sol, 0.0
        sol = [self.x[i].solution_value() for i in range(self.n_items)]
        return sol, self._model.Objective().Value()

    def copy(self) -> CreditPortfolioLP:
        return CreditPortfolioLP(self.n_items, self.budget)


# ── 2. Fix 1: Point-wise permutation-equivariant MLP ────────────────────────


class PDPredictorMLP(nn.Module):
    """Point-wise MLP that processes each loan independently.

    Input: (B, n_items * n_features) — flattened instance features (optDataset compat).
    Internally: reshapes to (B * n_items, n_features), processes each loan with shared weights.
    Output: (B, n_items) — predicted cost per loan.

    Permutation-equivariant: shuffling loans within an instance consistently
    shuffles predictions. Much lower input dimensionality than flat MLP.
    """

    def __init__(self, n_features: int, n_items: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_items = n_items
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # (B, n_items * n_features) → (B * n_items, n_features)
        x_per_loan = x.view(B * self.n_items, self.n_features)
        out = self.net(x_per_loan)  # (B * n_items, 1)
        return out.view(B, self.n_items)


# ── 3. Fix 2: Calibrated PD as true cost ────────────────────────────────────


def _load_pd_artifacts() -> tuple | None:
    """Load CatBoost model, calibrator, and feature contract. Returns None if unavailable."""
    paths = [
        Path("models/pd_canonical.cbm"),
        Path("models/pd_canonical_calibrator.pkl"),
        Path("models/pd_model_contract.json"),
    ]
    if not all(p.exists() for p in paths):
        return None
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(paths[0]))
        with open(paths[1], "rb") as f:
            calibrator = pickle.load(f)
        with open(paths[2]) as f:
            contract = json.load(f)
        return (
            model,
            calibrator,
            contract["feature_names"],
            contract.get("categorical_features", []),
        )
    except Exception as e:
        logger.warning("PD artifacts unavailable: {} — using binary default_flag costs", e)
        return None


def _predict_calibrated_costs(
    df: pd.DataFrame,
    model: object,
    calibrator: object,
    feature_names: list[str],
    cat_feats: list[str],
) -> np.ndarray:
    """Predict calibrated PD and return cost_i = pd_i * LGD - int_rate_i."""
    avail = [f for f in feature_names if f in df.columns]
    X = df[avail].copy()
    for col in avail:
        if col in cat_feats:
            X[col] = X[col].astype(str).fillna("UNKNOWN")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    raw_pd = model.predict_proba(X)[:, 1]
    cal_pd = apply_probability_calibrator(calibrator, raw_pd)
    int_rate = pd.to_numeric(df["int_rate"], errors="coerce").fillna(12.0).values / 100.0
    return (cal_pd * LGD - int_rate).astype(np.float32)


def _binary_costs(df: pd.DataFrame) -> np.ndarray:
    int_rate = pd.to_numeric(df["int_rate"], errors="coerce").fillna(12.0).values / 100.0
    return (df["default_flag"].astype(float).values * LGD - int_rate).astype(np.float32)


# ── 4. Data loading (done once, outside seed loop) ──────────────────────────


def _prep_features(
    df: pd.DataFrame,
    avail: list[str],
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features. Returns (X, mu, sigma)."""
    X = df[avail].copy()
    for col in avail:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    X_arr = X.values.astype(np.float32)
    if mu is None:
        mu = X_arr.mean(axis=0)
        sigma = X_arr.std(axis=0) + 1e-8
    return ((X_arr - mu) / sigma).astype(np.float32), mu, sigma


def _load_conformal_pd_high() -> np.ndarray | None:
    """Load pd_high_90 from conformal intervals (aligned with test_fe rows)."""
    path = Path("data/processed/conformal_intervals_mondrian.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["pd_high_90"])
    return df["pd_high_90"].values.astype(np.float32)


def _load_all_data(avail: list[str]) -> tuple:
    """Load and pre-process all loan-level data. Returns arrays for train and test."""
    train = pd.read_parquet("data/processed/train_fe.parquet")
    test = pd.read_parquet("data/processed/test_fe.parquet")
    logger.info("Loaded: train {:,} | test {:,} rows", len(train), len(test))

    X_tr, mu, sigma = _prep_features(train, avail)
    X_te, _, _ = _prep_features(test, avail, mu=mu, sigma=sigma)

    pd_arts = _load_pd_artifacts()
    if pd_arts is not None:
        cb_model, calibrator, feat_names, cat_feats = pd_arts
        logger.info(
            "Predicting calibrated PD on {:,} train + {:,} test loans...", len(train), len(test)
        )
        t0 = time.time()
        c_tr = _predict_calibrated_costs(train, cb_model, calibrator, feat_names, cat_feats)
        c_te = _predict_calibrated_costs(test, cb_model, calibrator, feat_names, cat_feats)
        logger.info(
            "  Done in {:.1f}s | cost range train [{:.4f}, {:.4f}]",
            time.time() - t0,
            c_tr.min(),
            c_tr.max(),
        )
        use_calibrated = True
    else:
        c_tr = _binary_costs(train)
        c_te = _binary_costs(test)
        logger.warning("Using binary default_flag costs (fallback)")
        use_calibrated = False

    # Fix 3: Two-stage baseline — Ridge regression fitted on full training data
    from sklearn.linear_model import Ridge

    logger.info("Fitting Ridge regression for two-stage baseline...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, c_tr)
    c_ts_te = ridge.predict(X_te).astype(np.float32)
    logger.info("  Two-stage Ridge: train R²={:.4f}", ridge.score(X_tr, c_tr))

    # Fix 4: Conformal robust costs for test set
    pd_high = _load_conformal_pd_high()
    if pd_high is not None and len(pd_high) == len(test):
        int_rate_te = pd.to_numeric(test["int_rate"], errors="coerce").fillna(12.0).values / 100.0
        c_robust_te = (pd_high * LGD - int_rate_te).astype(np.float32)
        logger.info(
            "Conformal robust costs loaded (pd_high_90 ≤ 0.20 for {:,} loans)",
            (pd_high < 0.20).sum(),
        )
    else:
        c_robust_te = None
        logger.warning("Conformal intervals not available — skipping conformal robust method")

    return X_tr, c_tr, X_te, c_te, c_ts_te, c_robust_te, use_calibrated


# ── 5. Instance sampling ─────────────────────────────────────────────────────


def _sample_instances(
    X_all: np.ndarray,
    c_all: np.ndarray,
    n_items: int,
    n_inst: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_inst problem instances of n_items loans each.

    Returns:
        feats: (n_inst, n_items, n_features) — per-loan features per instance
        costs: (n_inst, n_items) — per-loan costs per instance
        indices: (n_inst, n_items) — row indices into source arrays
    """
    n = len(X_all)
    feats, costs, indices = [], [], []
    for _ in range(n_inst):
        idx = rng.choice(n, size=n_items, replace=False)
        feats.append(X_all[idx])
        costs.append(c_all[idx])
        indices.append(idx)
    return np.stack(feats), np.stack(costs), np.array(indices)


def _index_costs(c_all: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Look up pre-computed costs for sampled indices: (n_inst, n_items)."""
    return np.array([c_all[idx] for idx in indices])


# ── 6. SPO+ training ──────────────────────────────────────────────────────────


def _train_spo(
    X_inst_train: np.ndarray,
    c_inst_train: np.ndarray,
    optmodel: CreditPortfolioLP,
    n_features: int,
    n_items: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[PDPredictorMLP, list[float]]:
    """Train SPO+ MLP using optDataset pre-solved instances.

    Args:
        X_inst_train: (n_inst, n_items, n_features) — instance features.
        c_inst_train: (n_inst, n_items) — true costs per instance.
    """
    from pyepo.data.dataset import optDataset
    from pyepo.func import SPOPlus
    from torch.utils.data import DataLoader

    n_input = n_items * n_features
    X_flat = X_inst_train.reshape(len(X_inst_train), n_input)

    logger.info("  Pre-solving {} instances with LP...", len(X_flat))
    t_pre = time.time()
    dataset = optDataset(optmodel, X_flat, c_inst_train)
    logger.info("  optDataset built in {:.1f}s", time.time() - t_pre)

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = PDPredictorMLP(n_features=n_features, n_items=n_items)
    spo_loss_fn = SPOPlus(optmodel, processes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for feats_b, costs_b, sols_b, objs_b in loader:
            optimizer.zero_grad()
            c_hat = model(feats_b.float())  # (B, n_items) via point-wise reshape
            loss = spo_loss_fn(c_hat, costs_b.float(), sols_b.float(), objs_b.float())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("    Epoch {:3d}/{} | SPO+ loss = {:.6f}", epoch + 1, epochs, avg)

    return model, losses


# ── 7. Evaluation ─────────────────────────────────────────────────────────────


def _compute_true_optima(c_true: np.ndarray, optmodel: CreditPortfolioLP) -> list[tuple]:
    """Pre-compute optimal LP decisions for all instances (cached for all methods)."""
    optima = []
    for i in range(len(c_true)):
        optmodel.setObj(c_true[i])
        x_star, _ = optmodel.solve()
        optima.append((np.array(x_star), float(np.dot(c_true[i], x_star))))
    return optima


def _compute_regret(
    c_pred: np.ndarray,
    c_true: np.ndarray,
    optmodel: CreditPortfolioLP,
    true_optima: list[tuple],
) -> np.ndarray:
    """Per-instance regret using pre-computed optima (no redundant LP solves)."""
    regrets = []
    for i in range(len(c_pred)):
        _, true_opt = true_optima[i]
        optmodel.setObj(c_pred[i])
        x_pred, _ = optmodel.solve()
        regrets.append(float(np.dot(c_true[i], x_pred)) - true_opt)
    return np.array(regrets)


# ── 8. Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="SPO+ real training — v2 (5 fixes)")
    parser.add_argument(
        "--n-items", type=int, default=100, help="Loans per problem instance (Fix 5: default 100)"
    )
    parser.add_argument(
        "--budget", type=int, default=30, help="Loans to select per instance (default 30%%)"
    )
    parser.add_argument(
        "--n-train", type=int, default=800, help="Training instances per seed (default 800)"
    )
    parser.add_argument(
        "--n-test", type=int, default=200, help="Test instances per seed (default 200)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="SPO+ training epochs (default 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32)")
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds for multi-seed eval (Fix 3: default 5)",
    )
    args = parser.parse_args()

    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info(
        "SPO+ v2 | n_items={} budget={} n_train={} n_test={} epochs={} seeds={} run_tag={}",
        args.n_items,
        args.budget,
        args.n_train,
        args.n_test,
        args.epochs,
        args.seeds,
        run_tag,
    )

    # ── Load all data once (outside seed loop) ───────────────────────────────
    avail = [f for f in NUMERIC_FEATURES if True]  # all checked per-dataset below
    train_check = pd.read_parquet("data/processed/train_fe.parquet", columns=NUMERIC_FEATURES[:1])
    test_check = pd.read_parquet("data/processed/test_fe.parquet", columns=NUMERIC_FEATURES[:1])
    del train_check, test_check

    # Load train_fe to get available columns
    _tr_cols = pd.read_parquet("data/processed/train_fe.parquet").columns
    _te_cols = pd.read_parquet("data/processed/test_fe.parquet").columns
    avail = [f for f in NUMERIC_FEATURES if f in _tr_cols and f in _te_cols]
    n_features = len(avail)
    n_input = args.n_items * n_features
    logger.info("Using {} features: {}", n_features, avail)

    t_load = time.time()
    X_tr_all, c_tr_all, X_te_all, c_te_all, c_ts_te_all, c_robust_te_all, use_cal = _load_all_data(
        avail
    )
    logger.info("Data loaded in {:.1f}s | use_calibrated_pd={}", time.time() - t_load, use_cal)

    # ── Multi-seed evaluation loop ───────────────────────────────────────────
    all_regrets: dict[str, list[np.ndarray]] = {
        "two_stage": [],
        "spo_plus": [],
    }
    if c_robust_te_all is not None:
        all_regrets["conformal_robust"] = []

    per_seed_means: dict[str, list[float]] = {k: [] for k in all_regrets}
    all_spo_losses: list[list[float]] = []
    t_total = time.time()

    for seed_idx in range(args.seeds):
        seed = RANDOM_SEED + seed_idx * 1000
        logger.info("=== Seed {}/{} (seed={}) ===", seed_idx + 1, args.seeds, seed)
        rng = np.random.RandomState(seed)

        # Sample train instances (for SPO+ training)
        X_tr_inst, c_tr_inst, _ = _sample_instances(
            X_tr_all, c_tr_all, args.n_items, args.n_train, rng
        )

        # Sample test instances (shared across all methods — paired comparison)
        X_te_inst, c_te_inst, idx_te = _sample_instances(
            X_te_all, c_te_all, args.n_items, args.n_test, rng
        )

        # Look up pre-computed two-stage and robust costs for these test indices
        c_ts_inst = _index_costs(c_ts_te_all, idx_te)  # (n_test, n_items)
        c_robust_inst = (
            _index_costs(c_robust_te_all, idx_te) if c_robust_te_all is not None else None
        )

        # Compute true optima ONCE (shared cache for all methods)
        optmodel_eval = CreditPortfolioLP(n_items=args.n_items, budget=args.budget)
        logger.info("  Pre-computing true optima for {} test instances...", args.n_test)
        t_opt = time.time()
        true_optima = _compute_true_optima(c_te_inst, optmodel_eval)
        logger.info("  True optima computed in {:.1f}s", time.time() - t_opt)

        # SPO+ training
        optmodel_train = CreditPortfolioLP(n_items=args.n_items, budget=args.budget)
        spo_model, spo_losses = _train_spo(
            X_tr_inst,
            c_tr_inst,
            optmodel_train,
            n_features=n_features,
            n_items=args.n_items,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=seed,
        )
        all_spo_losses.append(spo_losses)

        # SPO+ predictions for test instances
        X_te_flat = X_te_inst.reshape(args.n_test, n_input)
        spo_model.eval()
        with torch.no_grad():
            c_spo_inst = spo_model(torch.tensor(X_te_flat, dtype=torch.float32)).numpy()

        # Evaluate all methods
        t_eval = time.time()
        regrets_ts = _compute_regret(c_ts_inst, c_te_inst, optmodel_eval.copy(), true_optima)
        regrets_spo = _compute_regret(c_spo_inst, c_te_inst, optmodel_eval.copy(), true_optima)
        logger.info(
            "  Regret — two_stage: {:.4f}±{:.4f} | spo_plus: {:.4f}±{:.4f} (eval {:.1f}s)",
            regrets_ts.mean(),
            regrets_ts.std(),
            regrets_spo.mean(),
            regrets_spo.std(),
            time.time() - t_eval,
        )

        all_regrets["two_stage"].append(regrets_ts)
        all_regrets["spo_plus"].append(regrets_spo)
        per_seed_means["two_stage"].append(float(regrets_ts.mean()))
        per_seed_means["spo_plus"].append(float(regrets_spo.mean()))

        if c_robust_inst is not None:
            regrets_robust = _compute_regret(
                c_robust_inst, c_te_inst, optmodel_eval.copy(), true_optima
            )
            all_regrets["conformal_robust"].append(regrets_robust)
            per_seed_means["conformal_robust"].append(float(regrets_robust.mean()))
            logger.info(
                "  conformal_robust: {:.4f}±{:.4f}", regrets_robust.mean(), regrets_robust.std()
            )

    total_time = time.time() - t_total
    logger.info("All seeds done in {:.1f}s", total_time)

    # ── Aggregate across seeds ───────────────────────────────────────────────
    pooled: dict[str, np.ndarray] = {k: np.concatenate(v) for k, v in all_regrets.items()}
    agg: dict[str, dict[str, float]] = {}
    for method, arr in pooled.items():
        agg[method] = {
            "mean_regret": float(arr.mean()),
            "std_regret": float(arr.std()),
            "median_regret": float(np.median(arr)),
            "per_seed_means": per_seed_means[method],
        }

    # Wilcoxon signed-rank test: H1 = two_stage regrets > spo_plus regrets
    wil_stat, wil_pval = stats.wilcoxon(
        pooled["two_stage"],
        pooled["spo_plus"],
        alternative="greater",
    )
    improvement_pct = (
        (agg["two_stage"]["mean_regret"] - agg["spo_plus"]["mean_regret"])
        / (abs(agg["two_stage"]["mean_regret"]) + 1e-9)
        * 100
    )

    logger.info("=== Final Results ===")
    logger.info(
        "Two-stage:          {:.4f} ± {:.4f}",
        agg["two_stage"]["mean_regret"],
        agg["two_stage"]["std_regret"],
    )
    logger.info(
        "SPO+:               {:.4f} ± {:.4f}",
        agg["spo_plus"]["mean_regret"],
        agg["spo_plus"]["std_regret"],
    )
    if "conformal_robust" in agg:
        logger.info(
            "Conformal robust:   {:.4f} ± {:.4f}",
            agg["conformal_robust"]["mean_regret"],
            agg["conformal_robust"]["std_regret"],
        )
    logger.info("SPO+ improvement:   {:.2f}%", improvement_pct)
    logger.info("Wilcoxon p-value:   {:.4f} (significant={})", wil_pval, wil_pval < 0.05)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Average loss curve across seeds
    avg_losses = np.mean(all_spo_losses, axis=0).tolist()
    loss_df = pd.DataFrame(
        {
            "epoch": range(1, len(avg_losses) + 1),
            "spo_loss": avg_losses,
        }
    )
    loss_path = Path("data/processed/spo_training_loss.parquet")
    loss_df.to_parquet(loss_path, index=False)

    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={
            "n_items": args.n_items,
            "budget": args.budget,
            "n_train_instances": args.n_train,
            "n_test_instances": args.n_test,
            "epochs": args.epochs,
            "n_seeds": args.seeds,
            "n_features": n_features,
            "n_input": n_input,
            "feature_names": avail,
            "use_calibrated_pd": use_cal,
        },
    )
    status: dict[str, object] = {
        **metadata,
        "results": {
            **agg,
            "spo_improvement_vs_ts_pct": improvement_pct,
            "wilcoxon_spo_vs_ts": {
                "statistic": float(wil_stat),
                "pvalue": float(wil_pval),
                "alternative": "two_stage > spo_plus",
                "significant_at_0.05": bool(wil_pval < 0.05),
            },
            "n_paired_observations": int(args.seeds * args.n_test),
        },
        "config": {
            "n_items": args.n_items,
            "budget": args.budget,
            "selection_rate": args.budget / args.n_items,
            "lgd": LGD,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "fixes_applied": {
            "fix1_pointwise_mlp": True,
            "fix2_calibrated_pd_costs": use_cal,
            "fix3_multi_seed_wilcoxon": True,
            "fix4_conformal_robust": "conformal_robust" in agg,
            "fix5_larger_n_items": args.n_items >= 100,
        },
        "train_time_seconds": total_time,
        "note": (
            "SPO+ v2: point-wise MLP + calibrated PD costs + multi-seed Wilcoxon test "
            "+ conformal robust as 3rd comparison method. "
            "Two-stage uses Ridge regression to predict calibrated cost. "
            "SPO+ trains end-to-end to minimize decision regret directly."
        ),
    }
    status_path = Path("models/spo_real_training_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info("Saved status: {}", status_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
