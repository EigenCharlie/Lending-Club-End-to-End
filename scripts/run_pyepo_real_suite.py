"""PyEPO 1.3.7 real training suite for Paper Estrella and Paper 4.

This runner keeps the historical SPO+ baseline intact and adds a formal
decision-focused learning suite:

* two-stage Ridge baseline
* SPO+
* Regularized Frank-Wolfe Fenchel-Young (RFYL)
* multiplicative perturbed Fenchel-Young (PFYL-Mul)
* pairwise learning-to-rank (LTR) with PyEPO cache
* CRPTO robust costs as a non-trained auditable comparator

The economic cost is always evaluated as calibrated_pd * LGD - int_rate.
PFYL-Mul is trained on per-instance shifted positive costs; the fixed-budget
LP is invariant to adding the same constant to every item in an instance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyepo
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats
from torch.utils.data import DataLoader

from scripts.run_spo_real import (
    LGD,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    CreditPortfolioLP,
    PDPredictorMLP,
    _binary_costs,
    _compute_regret,
    _compute_true_optima,
    _index_costs,
    _load_pd_artifacts,
    _predict_calibrated_costs,
    _prep_features,
    _sample_instances,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

try:
    import gurobipy as gp
    from gurobipy import GRB
    from pyepo.model.grb.grbmodel import optGrbModel

    _HAS_GUROBI = True
except Exception:
    gp = None
    GRB = None
    optGrbModel = object
    _HAS_GUROBI = False

SCHEMA_VERSION = "2026-05-26.1"
EXPECTED_PYEPO_VERSION = "1.3.7"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"
PAPER4_TABLE_DIR = REPO_ROOT / "reports" / "paper_material" / "paper4" / "tables"
PAPER4_NOTE_DIR = REPO_ROOT / "reports" / "paper_material" / "paper4" / "notes"
FIG_DIR = REPO_ROOT / "reports" / "paper_material" / "figures_publication"
BOOK_FIG_DIR = REPO_ROOT / "book" / "assets" / "figures" / "publication"

STATUS_PATH = MODEL_DIR / "pyepo_real_suite_status.json"
REGRETS_PATH = DATA_DIR / "pyepo_real_suite_regrets.parquet"
LOSSES_PATH = DATA_DIR / "pyepo_real_suite_losses.parquet"
SUMMARY_PATH = PAPER4_TABLE_DIR / "pyepo_real_suite_summary.csv"

HISTORICAL_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "open_acc",
    "revol_bal",
    "total_acc",
    "installment",
    "pub_rec",
    "inq_last_6mths",
]

PERIODS = OrderedDict(
    [
        ("2018H1", ("2018-01-01", "2018-07-01")),
        ("2018H2", ("2018-07-01", "2019-01-01")),
        ("2019H1", ("2019-01-01", "2019-07-01")),
        ("2019H2", ("2019-07-01", "2020-01-01")),
        ("2020", ("2020-01-01", "2021-01-01")),
    ]
)

PAPER_SOURCE_LOG = [
    "Elmachtoub and Grigas (2021/2022), Smart predict, then optimize",
    "Mandi, Stuckey, Guns (2020), Smart predict-and-optimize for combinatorial optimization",
    "Vlastelica et al. (2019), Differentiation of blackbox combinatorial solvers",
    "Sahoo et al. (2022), Backpropagation through combinatorial algorithms",
    "Berthet et al. (2020), Learning with differentiable perturbed optimizers",
    "Dalle et al. (2022), Learning with Combinatorial Optimization Layers",
    "Mulamba et al. (2021), Contrastive losses and solution caching",
    "Mandi et al. (2022), Decision-focused learning through the lens of learning to rank",
    "Niepert et al. (2021) and Minervini et al. (2023), implicit MLE and perturbation gradients",
    "Gupta and Huang (2024), Decision-Focused Learning with Directional Gradients",
    "Schutte, Postek, Yorke-Smith (2023), Robust losses for DFL",
    "Tang and Khalil (2024), CaVE",
]

METHOD_DISPLAY = {
    "two_stage": "Two-stage Ridge",
    "spo_plus": "SPO+",
    "rfyl": "RFYL",
    "pfyl_mul": "PFYL-Mul",
    "pairwise_ltr": "Pairwise LTR",
    "cave": "CaVE",
    "crpto_robust": "CRPTO robust",
}

AUDITABILITY_SCORE = {
    "two_stage": 1.0,
    "spo_plus": 1.0,
    "rfyl": 1.0,
    "pfyl_mul": 1.0,
    "pairwise_ltr": 1.0,
    "cave": 1.0,
    "crpto_robust": 3.0,
}

PALETTE = {
    "two_stage": "#E69F00",
    "spo_plus": "#0072B2",
    "rfyl": "#009E73",
    "pfyl_mul": "#CC79A7",
    "pairwise_ltr": "#56B4E9",
    "cave": "#6A3D9A",
    "crpto_robust": "#D55E00",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


@dataclass(frozen=True)
class SuiteConfig:
    mode: str
    n_items: int
    budget: int
    n_train: int
    n_test: int
    epochs: int
    seeds: int
    batch_size: int
    lr: float
    feature_set: str
    cost_target: str
    methods: tuple[str, ...]
    include_crpto: bool
    run_tag: str
    torch_num_threads: int
    cave_max_iter: int
    archive_dir: Path | None


@dataclass
class SuiteData:
    train: pd.DataFrame
    test: pd.DataFrame
    ci: pd.DataFrame | None
    feature_names: list[str]
    X_tr: np.ndarray
    X_te: np.ndarray
    c_tr: np.ndarray
    c_te: np.ndarray
    c_ts_te: np.ndarray
    c_robust_te: np.ndarray | None
    use_calibrated_pd: bool
    cost_target: str
    cost_definition: str


class PositivePDPredictorMLP(nn.Module):
    """Point-wise MLP with strictly positive outputs for PFYL-Mul."""

    def __init__(self, n_features: int, n_items: int) -> None:
        super().__init__()
        self.base = PDPredictorMLP(n_features=n_features, n_items=n_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.base(x)) + 1e-6


class CreditPortfolioTopKOracle(CreditPortfolioLP):
    """Exact fixed-budget minimization oracle for the PyEPO real suite.

    The portfolio subproblem in this suite has only one equality constraint:
    select exactly ``budget`` loans out of ``n_items``. Its continuous LP and
    binary MILP optimum is therefore the same deterministic top-k solution:
    choose the loans with the smallest predicted costs. Using this oracle keeps
    SPO+/RFYL/PFYL/LTR training numerically stable while CaVE still uses the
    Gurobi binary model needed for tight-constraint extraction.
    """

    def __init__(self, n_items: int, budget: int) -> None:
        self._cost: np.ndarray | None = None
        super().__init__(n_items=n_items, budget=budget)

    def setObj(self, c: np.ndarray | torch.Tensor) -> None:
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        c_arr = np.asarray(c, dtype=float).reshape(-1)
        if len(c_arr) != self.n_items:
            raise ValueError(f"Expected {self.n_items} costs, got {len(c_arr)}")
        self._cost = c_arr

    def solve(self) -> tuple[list[float], float]:
        if self._cost is None:
            raise RuntimeError("Objective costs must be set before solve()")
        order = np.lexsort((np.arange(self.n_items), self._cost))
        chosen = order[: self.budget]
        sol = np.zeros(self.n_items, dtype=float)
        sol[chosen] = 1.0
        obj = float(np.dot(self._cost, sol))
        return sol.tolist(), obj

    def copy(self) -> CreditPortfolioTopKOracle:
        return CreditPortfolioTopKOracle(self.n_items, self.budget)


class CreditPortfolioBinaryGurobi(optGrbModel):
    """Binary top-k portfolio model required by PyEPO CaVE."""

    def __init__(self, n_items: int, budget: int) -> None:
        if not _HAS_GUROBI:
            raise ImportError("gurobipy is required for CreditPortfolioBinaryGurobi")
        self.n_items = n_items
        self.budget = budget
        super().__init__()

    def _getModel(self) -> tuple:
        assert gp is not None
        assert GRB is not None
        model = gp.Model("credit_portfolio_binary")
        model.Params.OutputFlag = 0
        x = {
            i: model.addVar(vtype=GRB.BINARY, lb=0.0, ub=1.0, name=f"x{i}")
            for i in range(self.n_items)
        }
        model.addConstr(gp.quicksum(x[i] for i in range(self.n_items)) == self.budget)
        model.ModelSense = GRB.MINIMIZE
        model.update()
        return model, x

    def copy(self) -> CreditPortfolioBinaryGurobi:
        return CreditPortfolioBinaryGurobi(self.n_items, self.budget)


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_versions() -> dict[str, str | None]:
    names = [
        "pyepo",
        "gurobipy",
        "torch",
        "ortools",
        "numpy",
        "pandas",
        "scipy",
        "catboost",
        "mapie",
        "venn-abers",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["cuda_available"] = str(torch.cuda.is_available())
    versions["torch_cuda"] = torch.version.cuda
    return versions


def _validate_pyepo_version(*, allow_mismatch: bool) -> dict[str, Any]:
    actual = getattr(pyepo, "__version__", importlib.metadata.version("pyepo"))
    ok = actual == EXPECTED_PYEPO_VERSION
    if not ok and not allow_mismatch:
        raise RuntimeError(
            f"PyEPO version mismatch: expected {EXPECTED_PYEPO_VERSION}, got {actual}. "
            "Use --allow-pyepo-version-mismatch only for exploratory reruns."
        )
    return {
        "expected": EXPECTED_PYEPO_VERSION,
        "actual": actual,
        "ok": ok,
        "docs_note": "Public docs may be labeled 1.3.8; this suite pins executable PyPI 1.3.7.",
    }


def _check_cave_availability() -> dict[str, Any]:
    candidates = [
        Path.home() / "gurobi.lic",
        Path("/opt/gurobi/gurobi.lic"),
        Path(os.environ["GRB_LICENSE_FILE"]) if os.environ.get("GRB_LICENSE_FILE") else None,
    ]
    license_candidate_count = sum(1 for p in candidates if p is not None and p.exists())
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return {
            "available": False,
            "reason": f"gurobipy import failed: {type(exc).__name__}: {exc}",
            "gurobi_version": None,
            "license_file_candidate_count": license_candidate_count,
            "main_suite_included": False,
        }

    try:
        model = gp.Model()
        model.Params.OutputFlag = 0
        x = model.addVar(vtype=GRB.BINARY, name="x")
        model.setObjective(x, GRB.MAXIMIZE)
        model.optimize()
        ok = model.Status == GRB.OPTIMAL
        return {
            "available": bool(ok),
            "reason": "gurobipy imports and solves a minimal binary model"
            if ok
            else "minimal solve failed",
            "gurobi_version": ".".join(map(str, gp.gurobi.version())),
            "license_file_candidate_count": license_candidate_count,
            "license_note": (
                "No gurobi.lic found in default locations; gurobipy may be using its restricted "
                "built-in license until grbgetkey or WLS is configured."
                if license_candidate_count == 0
                else "Found a gurobi.lic candidate in a default/project-visible location."
            ),
            "main_suite_included": True,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"gurobipy minimal solve failed: {type(exc).__name__}: {exc}",
            "gurobi_version": ".".join(map(str, gp.gurobi.version())),
            "license_file_candidate_count": license_candidate_count,
            "main_suite_included": False,
        }


def _mode_defaults(mode: str) -> dict[str, Any]:
    defaults = {
        "smoke": {
            "n_items": 20,
            "budget": 6,
            "n_train": 64,
            "n_test": 32,
            "epochs": 2,
            "seeds": 1,
            "feature_set": "full",
            "methods": "spo_plus,rfyl,pfyl_mul,pairwise_ltr,cave",
        },
        "paired": {
            "n_items": 100,
            "budget": 30,
            "n_train": 800,
            "n_test": 200,
            "epochs": 50,
            "seeds": 5,
            "feature_set": "historical",
            "methods": "spo_plus",
        },
        "paper4_full": {
            "n_items": 100,
            "budget": 30,
            "n_train": 2000,
            "n_test": 500,
            "epochs": 75,
            "seeds": 10,
            "feature_set": "full",
            "methods": "spo_plus,rfyl,pfyl_mul,pairwise_ltr,cave",
        },
        "temporal": {
            "n_items": 50,
            "budget": 15,
            "n_train": 1600,
            "n_test": 80,
            "epochs": 75,
            "seeds": 10,
            "feature_set": "full",
            "methods": "spo_plus,rfyl,pfyl_mul,pairwise_ltr,cave",
        },
    }
    return defaults[mode]


def _parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(m.strip() for m in text.split(",") if m.strip())
    valid = {"spo_plus", "rfyl", "pfyl_mul", "pairwise_ltr", "cave"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {sorted(valid)}")
    return methods


def _select_features(feature_set: str, train_cols: list[str], test_cols: list[str]) -> list[str]:
    requested = HISTORICAL_FEATURES if feature_set == "historical" else NUMERIC_FEATURES
    cols = [c for c in requested if c in train_cols and c in test_cols]
    if len(cols) != len(requested):
        missing = sorted(set(requested) - set(cols))
        logger.warning("Feature set {} missing columns: {}", feature_set, missing)
    return cols


def _ensure_finite_features(X: np.ndarray) -> np.ndarray:
    """Keep downstream estimators from seeing NaN/inf after normalization."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _load_conformal_intervals(test_len: int) -> pd.DataFrame | None:
    path = DATA_DIR / "conformal_intervals_mondrian.parquet"
    if not path.exists():
        logger.warning("Conformal intervals not found: {}", path)
        return None
    ci = pd.read_parquet(path)
    if len(ci) != test_len:
        logger.warning(
            "Conformal intervals length {} != test length {}; skipping CRPTO", len(ci), test_len
        )
        return None
    return ci


def _int_rate_array(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["int_rate"], errors="coerce").fillna(12.0).values / 100.0


def _load_suite_data(
    feature_set: str,
    *,
    cost_target: str,
    allow_binary_cost_fallback: bool = False,
) -> SuiteData:
    train = pd.read_parquet(DATA_DIR / "train_fe.parquet")
    test = pd.read_parquet(DATA_DIR / "test_fe.parquet")
    logger.info("Loaded train {:,} rows | test {:,} rows", len(train), len(test))

    feature_names = _select_features(feature_set, list(train.columns), list(test.columns))
    if not feature_names:
        raise RuntimeError(f"No usable features found for feature_set={feature_set}")
    logger.info("Using {} {} features: {}", len(feature_names), feature_set, feature_names)

    X_tr, mu, sigma = _prep_features(train, feature_names)
    X_te, _, _ = _prep_features(test, feature_names, mu=mu, sigma=sigma)
    X_tr = _ensure_finite_features(X_tr)
    X_te = _ensure_finite_features(X_te)

    pd_arts = _load_pd_artifacts()
    if pd_arts is None and not allow_binary_cost_fallback:
        raise RuntimeError(
            "PD artifacts are required for the PyEPO real suite. "
            "Use --allow-binary-cost-fallback only for local debugging."
        )

    if pd_arts is not None:
        cb_model, calibrator, feat_names, cat_feats = pd_arts
        logger.info("Predicting calibrated economic costs...")
        t0 = time.time()
        c_tr_economic = _predict_calibrated_costs(
            train, cb_model, calibrator, feat_names, cat_feats
        )
        c_te_economic = _predict_calibrated_costs(test, cb_model, calibrator, feat_names, cat_feats)
        if cost_target == "economic":
            c_tr = c_tr_economic
            c_te = c_te_economic
            cost_definition = "calibrated_pd * LGD - int_rate"
        elif cost_target == "risk_only":
            c_tr = (c_tr_economic + _int_rate_array(train)).astype(np.float32)
            c_te = (c_te_economic + _int_rate_array(test)).astype(np.float32)
            cost_definition = "calibrated_pd * LGD"
        else:
            raise ValueError(f"Unsupported cost_target={cost_target}")
        logger.info(
            "Calibrated costs ready in {:.1f}s | train range [{:.4f}, {:.4f}]",
            time.time() - t0,
            float(c_tr.min()),
            float(c_tr.max()),
        )
        use_calibrated_pd = True
    else:
        c_tr = _binary_costs(train)
        c_te = _binary_costs(test)
        cost_definition = "default_flag * LGD - int_rate"
        use_calibrated_pd = False
        logger.warning("Using binary default_flag costs: this is not valid for the paper run")

    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, c_tr)
    c_ts_te = ridge.predict(X_te).astype(np.float32)
    logger.info("Two-stage Ridge train R2={:.4f}", ridge.score(X_tr, c_tr))

    ci = _load_conformal_intervals(len(test))
    c_robust_te = None
    if ci is not None and "pd_high_90" in ci.columns:
        robust_risk = ci["pd_high_90"].values.astype(np.float32) * LGD
        if cost_target == "economic":
            c_robust_te = (robust_risk - _int_rate_array(test)).astype(np.float32)
        else:
            c_robust_te = robust_risk.astype(np.float32)
        logger.info("CRPTO robust costs loaded from conformal pd_high_90")

    return SuiteData(
        train=train,
        test=test,
        ci=ci,
        feature_names=feature_names,
        X_tr=X_tr,
        X_te=X_te,
        c_tr=c_tr,
        c_te=c_te,
        c_ts_te=c_ts_te,
        c_robust_te=c_robust_te,
        use_calibrated_pd=use_calibrated_pd,
        cost_target=cost_target,
        cost_definition=cost_definition,
    )


def _flatten_instances(X_inst: np.ndarray) -> np.ndarray:
    return X_inst.reshape(len(X_inst), -1).astype(np.float32)


def _shift_costs_positive(costs: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Shift every instance by a common constant so all costs are positive."""
    min_per_instance = costs.min(axis=1, keepdims=True)
    shifts = np.maximum(0.0, eps - min_per_instance)
    shifted = (costs + shifts).astype(np.float32)
    return shifted, shifts.reshape(-1).astype(np.float32)


def _build_opt_dataset(
    X_inst: np.ndarray,
    c_inst: np.ndarray,
    *,
    n_items: int,
    budget: int,
    label: str,
    kind: str = "standard",
) -> Any:
    from pyepo.data.dataset import optDataset, optDatasetConstrs

    dataset_cls = optDatasetConstrs if kind == "constrs" else optDataset
    optmodel = (
        CreditPortfolioBinaryGurobi(n_items=n_items, budget=budget)
        if kind == "constrs"
        else CreditPortfolioTopKOracle(n_items=n_items, budget=budget)
    )
    logger.info(
        "Pre-solving {} {} training instances with PyEPO {}",
        len(X_inst),
        label,
        dataset_cls.__name__,
    )
    t0 = time.time()
    dataset = dataset_cls(
        optmodel,
        _flatten_instances(X_inst),
        c_inst.astype(np.float32),
    )
    logger.info("optDataset({}) built in {:.1f}s", label, time.time() - t0)
    return dataset


def _build_loss(
    method: str,
    optmodel: Any,
    dataset: Any,
    seed: int,
    *,
    cave_max_iter: int,
) -> nn.Module:
    from pyepo import func as epo_func

    if method == "spo_plus":
        return epo_func.SPOPlus(optmodel, processes=1, dataset=dataset)
    if method == "rfyl":
        return epo_func.regularizedFrankWolfeFenchelYoung(
            optmodel, lambd=1.0, max_iter=20, processes=1, dataset=dataset
        )
    if method == "pfyl_mul":
        return epo_func.perturbedFenchelYoungMul(
            optmodel, n_samples=10, sigma=1.0, processes=1, seed=seed, dataset=dataset
        )
    if method == "pairwise_ltr":
        return epo_func.pairwiseLTR(optmodel, processes=1, dataset=dataset)
    if method == "cave":
        return epo_func.coneAlignedCosine(
            optmodel, max_iter=cave_max_iter, solve_ratio=1.0, processes=1
        )
    raise ValueError(f"Unsupported method: {method}")


def _loss_forward(
    method: str,
    loss_fn: nn.Module,
    c_hat: torch.Tensor,
    costs_b: torch.Tensor,
    sols_b: torch.Tensor,
    objs_b: torch.Tensor,
) -> torch.Tensor:
    if method == "spo_plus":
        return loss_fn(c_hat, costs_b.float(), sols_b.float(), objs_b.float())
    if method in {"rfyl", "pfyl_mul"}:
        return loss_fn(c_hat, sols_b.float())
    if method == "pairwise_ltr":
        return loss_fn(c_hat, costs_b.float())
    if method == "cave":
        return loss_fn(c_hat, costs_b.float())
    raise ValueError(f"Unsupported method: {method}")


def _train_pyepo_method(
    method: str,
    dataset: Any,
    *,
    n_features: int,
    n_items: int,
    budget: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    cave_max_iter: int = 3,
) -> tuple[nn.Module, list[float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if method == "pfyl_mul":
        model: nn.Module = PositivePDPredictorMLP(n_features=n_features, n_items=n_items)
    else:
        model = PDPredictorMLP(n_features=n_features, n_items=n_items)

    optmodel = (
        CreditPortfolioBinaryGurobi(n_items=n_items, budget=budget)
        if method == "cave"
        else CreditPortfolioTopKOracle(n_items=n_items, budget=budget)
    )
    loss_fn = _build_loss(method, optmodel, dataset, seed, cave_max_iter=cave_max_iter)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator()
    generator.manual_seed(seed)
    collate_fn = None
    if method == "cave":
        from pyepo.data.dataset import collate_tight_constraints

        collate_fn = collate_tight_constraints
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            if method == "cave":
                feats_b, costs_b, sols_b, objs_b, tight_ctrs_b = batch
            else:
                feats_b, costs_b, sols_b, objs_b = batch
                tight_ctrs_b = None
            optimizer.zero_grad(set_to_none=True)
            c_hat = model(feats_b.float())
            loss_arg = tight_ctrs_b if tight_ctrs_b is not None else costs_b
            loss = _loss_forward(method, loss_fn, c_hat, loss_arg, sols_b, objs_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            logger.info("  {} epoch {:3d}/{} loss={:.6f}", method, epoch + 1, epochs, avg_loss)

    return model, losses


def _predict_model(model: nn.Module, X_inst: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(_flatten_instances(X_inst), dtype=torch.float32)).cpu().numpy()


def _regret_rows(
    *,
    config: SuiteConfig,
    seed_idx: int,
    seed: int,
    method: str,
    regrets: np.ndarray,
    true_optima: list[tuple],
    period: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for instance_id, regret in enumerate(regrets):
        true_obj = float(true_optima[instance_id][1])
        rows.append(
            {
                "mode": config.mode,
                "run_tag": config.run_tag,
                "seed_index": seed_idx,
                "seed": seed,
                "period": period or "pooled",
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "instance_id": instance_id,
                "regret": float(regret),
                "true_optimal_objective": true_obj,
                "chosen_true_objective": true_obj + float(regret),
                "n_items": config.n_items,
                "budget": config.budget,
                "feature_set": config.feature_set,
                "cost_target": config.cost_target,
            }
        )
    return rows


def _loss_rows(
    *,
    config: SuiteConfig,
    seed_idx: int,
    seed: int,
    method: str,
    losses: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "mode": config.mode,
            "run_tag": config.run_tag,
            "seed_index": seed_idx,
            "seed": seed,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "epoch": epoch,
            "loss": float(loss),
            "cost_target": config.cost_target,
        }
        for epoch, loss in enumerate(losses, start=1)
    ]


def _train_all_methods_for_seed(
    data: SuiteData,
    config: SuiteConfig,
    seed: int,
) -> tuple[dict[str, nn.Module], dict[str, list[float]], dict[str, Any]]:
    rng = np.random.RandomState(seed)
    X_tr_inst, c_tr_inst, _ = _sample_instances(
        data.X_tr, data.c_tr, config.n_items, config.n_train, rng
    )

    standard_dataset = _build_opt_dataset(
        X_tr_inst,
        c_tr_inst,
        n_items=config.n_items,
        budget=config.budget,
        label="standard-cost",
    )
    shifted_costs, pfyl_shifts = _shift_costs_positive(c_tr_inst)
    shifted_dataset = None
    if "pfyl_mul" in config.methods:
        shifted_dataset = _build_opt_dataset(
            X_tr_inst,
            shifted_costs,
            n_items=config.n_items,
            budget=config.budget,
            label="pfyl-positive-cost",
        )
    cave_dataset = None
    if "cave" in config.methods:
        if not _HAS_GUROBI:
            raise RuntimeError("CaVE requested but gurobipy is not installed")
        cave_dataset = _build_opt_dataset(
            X_tr_inst,
            c_tr_inst,
            n_items=config.n_items,
            budget=config.budget,
            label="cave-binary-constrs",
            kind="constrs",
        )

    models: dict[str, nn.Module] = {}
    losses_by_method: dict[str, list[float]] = {}
    method_meta: dict[str, Any] = {
        "pfyl_mul_shift": {
            "min_shift": float(pfyl_shifts.min()),
            "max_shift": float(pfyl_shifts.max()),
            "mean_shift": float(pfyl_shifts.mean()),
            "positive_min_cost": float(shifted_costs.min()),
        },
        "method_runtime_seconds": {},
    }

    for method in config.methods:
        logger.info("Training {}", METHOD_DISPLAY[method])
        if method == "pfyl_mul":
            dataset = shifted_dataset
        elif method == "cave":
            dataset = cave_dataset
        else:
            dataset = standard_dataset
        if dataset is None:
            raise RuntimeError(f"Dataset for {method} was not built")
        method_t0 = time.time()
        model, losses = _train_pyepo_method(
            method,
            dataset,
            n_features=len(data.feature_names),
            n_items=config.n_items,
            budget=config.budget,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            seed=seed + _stable_int(method) % 100_000,
            cave_max_iter=config.cave_max_iter,
        )
        method_meta["method_runtime_seconds"][method] = time.time() - method_t0
        models[method] = model
        losses_by_method[method] = losses

    return models, losses_by_method, method_meta


def _run_paired_like(
    data: SuiteData, config: SuiteConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    regret_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    method_meta_by_seed: dict[str, Any] = {}

    for seed_idx in range(config.seeds):
        seed = RANDOM_SEED + seed_idx * 1000
        logger.info("=== Seed {}/{} ({}) ===", seed_idx + 1, config.seeds, seed)
        rng = np.random.RandomState(seed)

        models, losses_by_method, method_meta = _train_all_methods_for_seed(data, config, seed)
        method_meta_by_seed[str(seed)] = method_meta
        for method, losses in losses_by_method.items():
            loss_rows.extend(
                _loss_rows(
                    config=config, seed_idx=seed_idx, seed=seed, method=method, losses=losses
                )
            )

        X_te_inst, c_te_inst, idx_te = _sample_instances(
            data.X_te, data.c_te, config.n_items, config.n_test, rng
        )
        c_ts_inst = _index_costs(data.c_ts_te, idx_te)
        c_robust_inst = (
            _index_costs(data.c_robust_te, idx_te)
            if config.include_crpto and data.c_robust_te is not None
            else None
        )

        optmodel_eval = CreditPortfolioTopKOracle(n_items=config.n_items, budget=config.budget)
        true_optima = _compute_true_optima(c_te_inst, optmodel_eval)

        regrets_ts = _compute_regret(c_ts_inst, c_te_inst, optmodel_eval.copy(), true_optima)
        regret_rows.extend(
            _regret_rows(
                config=config,
                seed_idx=seed_idx,
                seed=seed,
                method="two_stage",
                regrets=regrets_ts,
                true_optima=true_optima,
            )
        )

        if c_robust_inst is not None:
            regrets_crpto = _compute_regret(
                c_robust_inst, c_te_inst, optmodel_eval.copy(), true_optima
            )
            regret_rows.extend(
                _regret_rows(
                    config=config,
                    seed_idx=seed_idx,
                    seed=seed,
                    method="crpto_robust",
                    regrets=regrets_crpto,
                    true_optima=true_optima,
                )
            )

        for method, model in models.items():
            c_pred_inst = _predict_model(model, X_te_inst)
            regrets = _compute_regret(c_pred_inst, c_te_inst, optmodel_eval.copy(), true_optima)
            regret_rows.extend(
                _regret_rows(
                    config=config,
                    seed_idx=seed_idx,
                    seed=seed,
                    method=method,
                    regrets=regrets,
                    true_optima=true_optima,
                )
            )
            logger.info(
                "  {} regret mean={:.6f} std={:.6f}",
                METHOD_DISPLAY[method],
                float(regrets.mean()),
                float(regrets.std()),
            )

        logger.info("  Two-stage regret mean={:.6f}", float(regrets_ts.mean()))
        if c_robust_inst is not None:
            logger.info("  CRPTO robust regret mean={:.6f}", float(regrets_crpto.mean()))

    return pd.DataFrame(regret_rows), pd.DataFrame(loss_rows), method_meta_by_seed


def _assign_periods(issue_d: pd.Series) -> pd.Series:
    dt = pd.to_datetime(issue_d)
    result = pd.Series("", index=dt.index, dtype=str)
    for name, (start, end) in PERIODS.items():
        mask = (dt >= pd.Timestamp(start)) & (dt < pd.Timestamp(end))
        result[mask] = name
    return result


def _evaluate_period_coverage(ci_slice: pd.DataFrame) -> dict[str, float | None]:
    y = ci_slice["y_true"].values
    low_90 = ci_slice["pd_low_90"].values
    high_90 = ci_slice["pd_high_90"].values
    covered_90 = ((y >= low_90) & (y <= high_90)).astype(float)
    coverage_95 = None
    if "pd_low_95" in ci_slice.columns and "pd_high_95" in ci_slice.columns:
        covered_95 = (y >= ci_slice["pd_low_95"].values) & (y <= ci_slice["pd_high_95"].values)
        coverage_95 = float(covered_95.mean())
    min_grade_coverage = None
    if "grade" in ci_slice.columns:
        grade_cov = ci_slice.assign(_covered=covered_90).groupby("grade")["_covered"].mean()
        if len(grade_cov) > 0:
            min_grade_coverage = float(grade_cov.min())
    return {
        "coverage_90": float(covered_90.mean()),
        "coverage_95": coverage_95,
        "avg_width_90": float((high_90 - low_90).mean()),
        "min_grade_coverage_90": min_grade_coverage,
    }


def _run_temporal(data: SuiteData, config: SuiteConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if data.c_robust_te is None or data.ci is None:
        raise RuntimeError("Temporal mode requires conformal_intervals_mondrian.parquet")

    test_periods = _assign_periods(data.test["issue_d"])
    regret_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    temporal_meta: dict[str, Any] = {"periods": {}}

    for name in PERIODS:
        mask = test_periods.values == name
        temporal_meta["periods"][name] = {
            "n_loans": int(mask.sum()),
            "default_rate": float(data.test.loc[mask, "default_flag"].mean())
            if mask.sum()
            else None,
            "coverage": _evaluate_period_coverage(data.ci.loc[mask]) if mask.sum() else {},
        }

    for seed_idx in range(config.seeds):
        seed = RANDOM_SEED + seed_idx * 1000
        logger.info("=== Temporal seed {}/{} ({}) ===", seed_idx + 1, config.seeds, seed)
        models, losses_by_method, method_meta = _train_all_methods_for_seed(data, config, seed)
        temporal_meta.setdefault("method_meta_by_seed", {})[str(seed)] = method_meta

        for method, losses in losses_by_method.items():
            loss_rows.extend(
                _loss_rows(
                    config=config, seed_idx=seed_idx, seed=seed, method=method, losses=losses
                )
            )

        for period_name in PERIODS:
            mask = test_periods.values == period_name
            period_idx = np.flatnonzero(mask)
            if len(period_idx) < config.n_items:
                logger.warning("{} has only {} loans; skipping", period_name, len(period_idx))
                continue

            X_period = data.X_te[period_idx]
            c_period = data.c_te[period_idx]
            c_ts_period = data.c_ts_te[period_idx]
            c_robust_period = data.c_robust_te[period_idx]

            n_period_test = min(config.n_test, max(5, len(period_idx) // config.n_items))
            rng = np.random.RandomState(seed + _stable_int(period_name) % 100_000)
            X_inst, c_inst, idx_inst = _sample_instances(
                X_period, c_period, config.n_items, n_period_test, rng
            )
            c_ts_inst = _index_costs(c_ts_period, idx_inst)
            c_robust_inst = _index_costs(c_robust_period, idx_inst)

            optmodel_eval = CreditPortfolioTopKOracle(n_items=config.n_items, budget=config.budget)
            true_optima = _compute_true_optima(c_inst, optmodel_eval)

            regrets_ts = _compute_regret(c_ts_inst, c_inst, optmodel_eval.copy(), true_optima)
            regrets_crpto = _compute_regret(
                c_robust_inst, c_inst, optmodel_eval.copy(), true_optima
            )
            regret_rows.extend(
                _regret_rows(
                    config=config,
                    seed_idx=seed_idx,
                    seed=seed,
                    method="two_stage",
                    regrets=regrets_ts,
                    true_optima=true_optima,
                    period=period_name,
                )
            )
            regret_rows.extend(
                _regret_rows(
                    config=config,
                    seed_idx=seed_idx,
                    seed=seed,
                    method="crpto_robust",
                    regrets=regrets_crpto,
                    true_optima=true_optima,
                    period=period_name,
                )
            )

            for method, model in models.items():
                c_pred_inst = _predict_model(model, X_inst)
                regrets = _compute_regret(c_pred_inst, c_inst, optmodel_eval.copy(), true_optima)
                regret_rows.extend(
                    _regret_rows(
                        config=config,
                        seed_idx=seed_idx,
                        seed=seed,
                        method=method,
                        regrets=regrets,
                        true_optima=true_optima,
                        period=period_name,
                    )
                )

            logger.info(
                "{} seed {} regrets: TS={:.6f} CRPTO={:.6f}",
                period_name,
                seed,
                float(regrets_ts.mean()),
                float(regrets_crpto.mean()),
            )

    return pd.DataFrame(regret_rows), pd.DataFrame(loss_rows), temporal_meta


def _summarize_regrets(regrets_df: pd.DataFrame, coverage_meta: dict | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_mean = float(regrets_df.loc[regrets_df["method"] == "two_stage", "regret"].mean())

    pivot_index = ["seed", "period", "instance_id"]
    pivot = regrets_df.pivot_table(index=pivot_index, columns="method", values="regret")

    for method, group in regrets_df.groupby("method", sort=False):
        arr = group["regret"].to_numpy(dtype=float)
        row = {
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "mean_regret": float(np.mean(arr)),
            "std_regret": float(np.std(arr)),
            "median_regret": float(np.median(arr)),
            "q25_regret": float(np.quantile(arr, 0.25)),
            "q75_regret": float(np.quantile(arr, 0.75)),
            "min_regret": float(np.min(arr)),
            "max_regret": float(np.max(arr)),
            "n_observations": int(len(arr)),
            "improvement_vs_two_stage_pct": (
                float((baseline_mean - np.mean(arr)) / (abs(baseline_mean) + 1e-9) * 100)
                if method != "two_stage"
                else 0.0
            ),
            "auditability_score": AUDITABILITY_SCORE[method],
        }
        if method != "two_stage" and method in pivot.columns and "two_stage" in pivot.columns:
            paired = pivot[["two_stage", method]].dropna()
            if len(paired) > 0 and not np.allclose(paired["two_stage"], paired[method]):
                stat, pval = stats.wilcoxon(
                    paired["two_stage"],
                    paired[method],
                    alternative="greater",
                )
                row["wilcoxon_vs_two_stage_statistic"] = float(stat)
                row["wilcoxon_vs_two_stage_pvalue"] = float(pval)
                row["wilcoxon_alternative"] = "two_stage > method"
            else:
                row["wilcoxon_vs_two_stage_statistic"] = np.nan
                row["wilcoxon_vs_two_stage_pvalue"] = np.nan
                row["wilcoxon_alternative"] = "not_applicable"
        else:
            row["wilcoxon_vs_two_stage_statistic"] = np.nan
            row["wilcoxon_vs_two_stage_pvalue"] = np.nan
            row["wilcoxon_alternative"] = "baseline"

        if method == "crpto_robust" and coverage_meta:
            row.update(coverage_meta)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_regret").reset_index(drop=True)


def _coverage_meta_for_status(data: SuiteData) -> dict[str, float | None]:
    if data.ci is None:
        return {}
    return _evaluate_period_coverage(data.ci)


def _plot_regret_summary(summary_df: pd.DataFrame) -> list[Path]:
    ordered = summary_df.sort_values("mean_regret")
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colors = [PALETTE.get(m, "#999999") for m in ordered["method"]]
    ax.bar(
        ordered["method_display"],
        ordered["mean_regret"],
        yerr=ordered["std_regret"],
        color=colors,
        alpha=0.85,
        capsize=3,
    )
    ax.set_ylabel("Mean Decision Regret")
    ax.set_title("PyEPO Real Suite: Regret by Method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    paths = []
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"pyepo_real_suite_regret.{ext}"
        fig.savefig(path)
        shutil.copy2(path, BOOK_FIG_DIR / path.name)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_frontier(summary_df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for _, row in summary_df.iterrows():
        method = row["method"]
        ax.scatter(
            row["auditability_score"],
            row["mean_regret"],
            s=80,
            color=PALETTE.get(method, "#999999"),
            label=row["method_display"],
        )
        ax.annotate(
            row["method_display"],
            (row["auditability_score"], row["mean_regret"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Auditability Score")
    ax.set_ylabel("Mean Decision Regret")
    ax.set_title("Regret-Auditability Frontier")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.6, 3.4)
    fig.tight_layout()

    paths = []
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"pyepo_real_suite_regret_auditability.{ext}"
        fig.savefig(path)
        shutil.copy2(path, BOOK_FIG_DIR / path.name)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_temporal(regrets_df: pd.DataFrame) -> list[Path]:
    temporal = regrets_df[regrets_df["period"] != "pooled"]
    if temporal.empty:
        return []
    agg = (
        temporal.groupby(["period", "method", "method_display"], as_index=False)["regret"]
        .agg(["mean", "std"])
        .reset_index()
    )
    periods = list(PERIODS.keys())
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for method in agg["method"].unique():
        mdf = agg[agg["method"] == method].set_index("period").reindex(periods)
        ax.plot(
            periods,
            mdf["mean"],
            marker="o",
            label=METHOD_DISPLAY[method],
            color=PALETTE.get(method, "#999999"),
        )
        ax.fill_between(periods, mdf["mean"] - mdf["std"], mdf["mean"] + mdf["std"], alpha=0.12)
    ax.set_ylabel("Mean Decision Regret")
    ax.set_title("PyEPO Real Suite: Temporal OOT Stability")
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()

    paths = []
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"pyepo_real_suite_temporal_stability.{ext}"
        fig.savefig(path)
        shutil.copy2(path, BOOK_FIG_DIR / path.name)
        paths.append(path)
    plt.close(fig)
    return paths


def _ensure_output_dirs() -> None:
    for path in [DATA_DIR, MODEL_DIR, PAPER4_TABLE_DIR, PAPER4_NOTE_DIR, FIG_DIR, BOOK_FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    *,
    config: SuiteConfig,
    data: SuiteData,
    regrets_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pyepo_validation: dict[str, Any],
    cave_status: dict[str, Any],
    runtime_seconds: float,
    extra_meta: dict[str, Any],
) -> None:
    _ensure_output_dirs()

    regrets_df.to_parquet(REGRETS_PATH, index=False)
    losses_df.to_parquet(LOSSES_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    fig_paths = []
    fig_paths.extend(_plot_regret_summary(summary_df))
    fig_paths.extend(_plot_frontier(summary_df))
    fig_paths.extend(_plot_temporal(regrets_df))

    artifact_paths = [REGRETS_PATH, LOSSES_PATH, SUMMARY_PATH, *fig_paths]
    artifact_hashes = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in artifact_paths}

    required_methods = {"two_stage", *config.methods}
    if config.include_crpto and data.c_robust_te is not None:
        required_methods.add("crpto_robust")
    observed_methods = set(regrets_df["method"].unique())
    min_regret = float(regrets_df["regret"].min()) if len(regrets_df) else float("nan")
    spo_row = summary_df.loc[summary_df["method"] == "spo_plus"]
    spo_improvement = (
        float(spo_row["improvement_vs_two_stage_pct"].iloc[0]) if len(spo_row) else None
    )

    acceptance_checks = {
        "smoke_gate_passed": (
            min_regret >= -1e-6
            and required_methods.issubset(observed_methods)
            and data.use_calibrated_pd
            and pyepo_validation["ok"]
        ),
        "nonnegative_regret_tolerance": min_regret >= -1e-6,
        "min_regret": min_regret,
        "required_methods_present": required_methods.issubset(observed_methods),
        "required_methods": sorted(required_methods),
        "observed_methods": sorted(observed_methods),
        "spo_plus_improvement_required": config.mode in {"paired", "paper4_full"},
        "spo_plus_improves_two_stage": spo_improvement is not None and spo_improvement > 0,
        "spo_plus_reproduction_gate_passed": (
            (spo_improvement is not None and spo_improvement > 0)
            if config.mode in {"paired", "paper4_full"}
            else None
        ),
        "spo_plus_improvement_vs_two_stage_pct": spo_improvement,
        "use_calibrated_pd": data.use_calibrated_pd,
        "pyepo_version_ok": pyepo_validation["ok"],
    }

    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=config.run_tag,
        allow_untracked=True,
        extra={
            "mode": config.mode,
            "config": {
                "n_items": config.n_items,
                "budget": config.budget,
                "selection_rate": config.budget / config.n_items,
                "n_train_instances": config.n_train,
                "n_test_instances": config.n_test,
                "epochs": config.epochs,
                "seeds": config.seeds,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "feature_set": config.feature_set,
                "feature_names": data.feature_names,
                "cost_target": config.cost_target,
                "methods": list(config.methods),
                "include_crpto": config.include_crpto,
                "lgd": LGD,
                "torch_num_threads": config.torch_num_threads,
                "cave_max_iter": config.cave_max_iter,
                "archive_dir": str(config.archive_dir) if config.archive_dir else None,
            },
            "data": {
                "train_rows": int(len(data.train)),
                "test_rows": int(len(data.test)),
                "conformal_rows": int(len(data.ci)) if data.ci is not None else None,
                "cost_target": data.cost_target,
                "cost_definition": data.cost_definition,
                "use_calibrated_pd": data.use_calibrated_pd,
                "standard_oracle_backend": "exact_topk_numpy_lexicographic",
                "cave_oracle_backend": "gurobi_binary_optDatasetConstrs",
            },
            "versions": _collect_versions(),
            "pyepo_validation": pyepo_validation,
            "cave_status": cave_status,
            "source_log": PAPER_SOURCE_LOG,
            "runtime_seconds": runtime_seconds,
            "results": {
                "summary": json.loads(summary_df.to_json(orient="records")),
                "n_regret_rows": int(len(regrets_df)),
                "n_loss_rows": int(len(losses_df)),
            },
            "acceptance_checks": acceptance_checks,
            "artifacts": artifact_hashes,
            "extra_meta": extra_meta,
        },
    )

    with STATUS_PATH.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved {}", STATUS_PATH)
    logger.info("Saved {}", REGRETS_PATH)
    logger.info("Saved {}", LOSSES_PATH)
    logger.info("Saved {}", SUMMARY_PATH)

    if config.archive_dir is not None:
        archive_dir = config.archive_dir
        if not archive_dir.is_absolute():
            archive_dir = REPO_ROOT / archive_dir
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in [STATUS_PATH, REGRETS_PATH, LOSSES_PATH, SUMMARY_PATH, *fig_paths]:
            shutil.copy2(path, archive_dir / path.name)
        logger.info("Archived PyEPO outputs to {}", archive_dir)


def _build_config(args: argparse.Namespace) -> SuiteConfig:
    defaults = _mode_defaults(args.mode)
    n_items = args.n_items if args.n_items is not None else defaults["n_items"]
    budget = args.budget if args.budget is not None else defaults["budget"]
    n_train = args.n_train if args.n_train is not None else defaults["n_train"]
    n_test = args.n_test if args.n_test is not None else defaults["n_test"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    seeds = args.seeds if args.seeds is not None else defaults["seeds"]
    feature_set = args.feature_set or defaults["feature_set"]
    methods = _parse_methods(args.methods or defaults["methods"])
    run_tag = resolve_run_tag(args.run_tag, allow_untracked=True)
    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    return SuiteConfig(
        mode=args.mode,
        n_items=n_items,
        budget=budget,
        n_train=n_train,
        n_test=n_test,
        epochs=epochs,
        seeds=seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_set=feature_set,
        cost_target=args.cost_target,
        methods=methods,
        include_crpto=not args.skip_crpto,
        run_tag=run_tag,
        torch_num_threads=args.torch_num_threads,
        cave_max_iter=args.cave_max_iter,
        archive_dir=archive_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PyEPO 1.3.7 real DFL suite")
    parser.add_argument(
        "--mode",
        choices=["smoke", "paired", "paper4_full", "temporal"],
        default="smoke",
        help="Experiment gate/mode to run.",
    )
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-set", choices=["historical", "full"], default=None)
    parser.add_argument(
        "--cost-target",
        choices=["economic", "risk_only"],
        default="economic",
        help="Cost target optimized and evaluated by the DFL suite.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset: spo_plus,rfyl,pfyl_mul,pairwise_ltr,cave",
    )
    parser.add_argument("--skip-crpto", action="store_true")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--cave-max-iter", type=int, default=3)
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--allow-pyepo-version-mismatch", action="store_true")
    parser.add_argument("--allow-binary-cost-fallback", action="store_true")
    args = parser.parse_args()

    config = _build_config(args)
    if config.budget > config.n_items:
        raise ValueError("budget cannot exceed n_items")
    if config.n_items <= 0 or config.budget <= 0:
        raise ValueError("n_items and budget must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", str(config.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(config.torch_num_threads))
    torch.set_num_threads(config.torch_num_threads)

    pyepo_validation = _validate_pyepo_version(allow_mismatch=args.allow_pyepo_version_mismatch)
    cave_status = _check_cave_availability()
    logger.info(
        "PyEPO suite | mode={} n_items={} budget={} n_train={} n_test={} epochs={} seeds={} cost_target={} methods={} run_tag={}",
        config.mode,
        config.n_items,
        config.budget,
        config.n_train,
        config.n_test,
        config.epochs,
        config.seeds,
        config.cost_target,
        config.methods,
        config.run_tag,
    )

    t0 = time.time()
    data = _load_suite_data(
        config.feature_set,
        cost_target=config.cost_target,
        allow_binary_cost_fallback=args.allow_binary_cost_fallback,
    )

    if config.mode == "temporal":
        regrets_df, losses_df, extra_meta = _run_temporal(data, config)
    else:
        regrets_df, losses_df, extra_meta = _run_paired_like(data, config)

    coverage_meta = _coverage_meta_for_status(data) if config.include_crpto else {}
    summary_df = _summarize_regrets(regrets_df, coverage_meta=coverage_meta)
    runtime_seconds = time.time() - t0

    _write_outputs(
        config=config,
        data=data,
        regrets_df=regrets_df,
        losses_df=losses_df,
        summary_df=summary_df,
        pyepo_validation=pyepo_validation,
        cave_status=cave_status,
        runtime_seconds=runtime_seconds,
        extra_meta=extra_meta,
    )

    logger.info("Done in {:.1f}s", runtime_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
