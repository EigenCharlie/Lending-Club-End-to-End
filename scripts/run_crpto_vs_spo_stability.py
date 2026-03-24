"""CRPTO vs SPO+ stability under distributional shift — Paper Estrella.

Evaluates decision regret (SPO+, two-stage, conformal robust) AND conformal
coverage per temporal period in the OOT test set (2018-2020).  The natural
regime change (default rate 24.5% → 1.35%) provides a quasi-experiment for
distributional shift stability.

Key design: SPO+ is trained ONCE per seed on full training data (2007-2017),
then evaluated on each period separately — mirroring production deployment.

Output artifacts:
  - data/processed/crpto_vs_spo_stability.json
  - data/processed/crpto_vs_spo_stability_detail.parquet
  - reports/paper_material/figures_publication/estrella_fig11_crpto_stability.{pdf,png}

Usage:
    uv run python scripts/run_crpto_vs_spo_stability.py
    uv run python scripts/run_crpto_vs_spo_stability.py --n-items 50 --budget 15
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from scripts.run_spo_real import (
    LGD,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    CreditPortfolioLP,
    _compute_regret,
    _compute_true_optima,
    _index_costs,
    _load_pd_artifacts,
    _predict_calibrated_costs,
    _prep_features,
    _sample_instances,
    _train_spo,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

matplotlib.use("Agg")

SCHEMA_VERSION = "2026-03-22.1"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "reports" / "paper_material" / "figures_publication"
BOOK_FIG_DIR = REPO_ROOT / "book" / "assets" / "figures" / "publication"

PERIODS = OrderedDict(
    [
        ("2018H1", ("2018-01-01", "2018-07-01")),
        ("2018H2", ("2018-07-01", "2019-01-01")),
        ("2019H1", ("2019-01-01", "2019-07-01")),
        ("2019H2", ("2019-07-01", "2020-01-01")),
        ("2020", ("2020-01-01", "2021-01-01")),
    ]
)

# ── Publication style (matches generate_paper_figures.py) ────────────────────

COL2 = 7.0
HEIGHT_M = 3.2

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "gray": "#999999",
    "sky": "#56B4E9",
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
        "legend.framealpha": 0.85,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
    }
)


# ── Period assignment ────────────────────────────────────────────────────────


def _assign_periods(issue_d: pd.Series) -> pd.Series:
    """Assign each loan to a temporal period based on issue_d."""
    dt = pd.to_datetime(issue_d)
    result = pd.Series("", index=dt.index, dtype=str)
    for name, (start, end) in PERIODS.items():
        mask = (dt >= pd.Timestamp(start)) & (dt < pd.Timestamp(end))
        result[mask] = name
    return result


# ── Conformal coverage evaluation ────────────────────────────────────────────


def _evaluate_period_coverage(ci_slice: pd.DataFrame) -> dict:
    """Compute conformal coverage metrics for a slice of conformal intervals."""
    y = ci_slice["y_true"].values
    low_90 = ci_slice["pd_low_90"].values
    high_90 = ci_slice["pd_high_90"].values

    covered_90 = ((y >= low_90) & (y <= high_90)).astype(float)
    coverage_90 = float(covered_90.mean())
    avg_width_90 = float((high_90 - low_90).mean())

    # Coverage 95 if available
    coverage_95 = None
    if "pd_low_95" in ci_slice.columns and "pd_high_95" in ci_slice.columns:
        low_95 = ci_slice["pd_low_95"].values
        high_95 = ci_slice["pd_high_95"].values
        covered_95 = ((y >= low_95) & (y <= high_95)).astype(float)
        coverage_95 = float(covered_95.mean())

    # Min coverage by grade (Mondrian guarantee)
    min_grade_coverage_90 = None
    if "grade" in ci_slice.columns:
        grade_cov = ci_slice.assign(covered=covered_90).groupby("grade")["covered"].mean()
        if len(grade_cov) > 0:
            min_grade_coverage_90 = float(grade_cov.min())

    return {
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "avg_width_90": avg_width_90,
        "min_grade_coverage_90": min_grade_coverage_90,
    }


# ── Figure generation ────────────────────────────────────────────────────────


def _generate_stability_figure(detail_df: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel figure: Panel A = regret by period, Panel B = coverage by period."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, HEIGHT_M))

    periods = detail_df["period"].values
    x = np.arange(len(periods))

    # ── Panel A: Decision Regret by Period ──────────────────────────────────
    methods = [
        ("two_stage_mean_regret", "two_stage_std_regret", "Two-Stage", PALETTE["orange"], "-"),
        ("spo_plus_mean_regret", "spo_plus_std_regret", "SPO+", PALETTE["blue"], "-"),
        (
            "conformal_robust_mean_regret",
            "conformal_robust_std_regret",
            "CRPTO",
            PALETTE["green"],
            "-",
        ),
    ]

    for mean_col, std_col, label, color, ls in methods:
        y_mean = detail_df[mean_col].values
        y_std = detail_df[std_col].values
        ax1.plot(x, y_mean, ls, color=color, label=label, marker="o", markersize=4)
        ax1.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15, color=color)

    # Annotate default rates above
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(x)
    default_labels = [f"{r:.1f}%" for r in detail_df["default_rate"].values * 100]
    ax1_top.set_xticklabels(default_labels, fontsize=7, color=PALETTE["gray"])
    ax1_top.set_xlabel("Default Rate", fontsize=8, color=PALETTE["gray"])
    ax1_top.spines["top"].set_visible(True)
    ax1_top.spines["top"].set_color(PALETTE["gray"])
    ax1_top.spines["top"].set_alpha(0.3)
    ax1_top.tick_params(axis="x", colors=PALETTE["gray"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, fontsize=8)
    ax1.set_ylabel("Mean Decision Regret")
    ax1.set_title("A. Decision Regret by Period")
    ax1.legend(loc="upper right", fontsize=7)

    # ── Panel B: Conformal Coverage by Period ───────────────────────────────
    cov = detail_df["coverage_90"].values * 100
    width = detail_df["avg_width_90"].values

    ax2.plot(x, cov, "-o", color=PALETTE["blue"], label="Coverage 90%", markersize=5)
    ax2.axhline(y=90, color=PALETTE["red"], linestyle="--", linewidth=1.0, label="Target 90%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, fontsize=8)
    ax2.set_ylabel("Coverage (%)", color=PALETTE["blue"])
    ax2.set_ylim(85, 105)
    ax2.tick_params(axis="y", labelcolor=PALETTE["blue"])

    # Secondary axis: width
    ax2r = ax2.twinx()
    ax2r.plot(x, width, "--s", color=PALETTE["orange"], label="Avg Width", markersize=4)
    ax2r.set_ylabel("Avg Interval Width", color=PALETTE["orange"])
    ax2r.tick_params(axis="y", labelcolor=PALETTE["orange"])
    ax2r.spines["right"].set_visible(True)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=7)

    ax2.set_title("B. Conformal Coverage by Period")

    fig.suptitle(
        "CRPTO vs SPO+ Stability Under Distributional Shift (OOT 2018–2020)",
        fontsize=10,
        y=1.05,
    )
    fig.tight_layout()

    name = "estrella_fig11_crpto_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    BOOK_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        shutil.copy2(path, BOOK_FIG_DIR / f"{name}.{ext}")
    logger.info(f"Saved: {name}.pdf / .png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CRPTO vs SPO+ stability under distributional shift"
    )
    parser.add_argument("--n-items", type=int, default=50, help="Loans per instance (default 50)")
    parser.add_argument("--budget", type=int, default=15, help="Loans to select (default 15)")
    parser.add_argument(
        "--n-train", type=int, default=800, help="Training instances per seed (default 800)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="SPO+ training epochs (default 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default 5)")
    args = parser.parse_args()

    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info(
        "CRPTO stability | n_items={} budget={} epochs={} seeds={} run_tag={}",
        args.n_items,
        args.budget,
        args.epochs,
        args.seeds,
        run_tag,
    )

    # ── 1. Load data ────────────────────────────────────────────────────────
    train = pd.read_parquet(DATA_DIR / "train_fe.parquet")
    test = pd.read_parquet(DATA_DIR / "test_fe.parquet")
    ci = pd.read_parquet(DATA_DIR / "conformal_intervals_mondrian.parquet")
    logger.info("Loaded: train {:,} | test {:,} | ci {:,}", len(train), len(test), len(ci))

    # Verify alignment
    assert len(ci) == len(test), f"ci ({len(ci)}) != test ({len(test)}) — row alignment broken"

    # Assign periods
    test_periods = _assign_periods(test["issue_d"])
    ci["_period"] = test_periods.values

    period_counts = test_periods.value_counts().to_dict()
    logger.info("Period counts: {}", period_counts)

    # ── 2. Prepare features and costs ───────────────────────────────────────
    _tr_cols = train.columns
    _te_cols = test.columns
    avail = [f for f in NUMERIC_FEATURES if f in _tr_cols and f in _te_cols]
    n_features = len(avail)
    logger.info("Using {} features: {}", n_features, avail)

    X_tr_all, mu, sigma = _prep_features(train, avail)
    X_te_all, _, _ = _prep_features(test, avail, mu=mu, sigma=sigma)

    # Calibrated costs
    pd_arts = _load_pd_artifacts()
    if pd_arts is None:
        logger.error("PD artifacts required — run train_pd_model.py first")
        return 1
    cb_model, calibrator, feat_names, cat_feats = pd_arts

    logger.info("Predicting calibrated PD costs...")
    t0 = time.time()
    c_tr_all = _predict_calibrated_costs(train, cb_model, calibrator, feat_names, cat_feats)
    c_te_all = _predict_calibrated_costs(test, cb_model, calibrator, feat_names, cat_feats)
    logger.info("  Done in {:.1f}s", time.time() - t0)

    # Two-stage: Ridge on full train
    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_all, c_tr_all)
    c_ts_te_all = ridge.predict(X_te_all).astype(np.float32)
    logger.info("Two-stage Ridge: train R²={:.4f}", ridge.score(X_tr_all, c_tr_all))

    # Conformal robust costs
    pd_high = ci["pd_high_90"].values.astype(np.float32)
    int_rate_te = pd.to_numeric(test["int_rate"], errors="coerce").fillna(12.0).values / 100.0
    c_robust_te_all = (pd_high * LGD - int_rate_te).astype(np.float32)

    # ── 3. Build period masks ───────────────────────────────────────────────
    period_masks = {}
    for name in PERIODS:
        mask = test_periods.values == name
        period_masks[name] = mask
        n_loans = int(mask.sum())
        default_rate = float(test.loc[mask, "default_flag"].mean()) if n_loans > 0 else 0.0
        logger.info("  {} : {:,} loans, default rate {:.2%}", name, n_loans, default_rate)

    # ── 4. Multi-seed × multi-period evaluation ─────────────────────────────
    # Structure: per_period[period][method] = list of per-seed mean regrets
    per_period_regrets: dict[str, dict[str, list[float]]] = {
        name: {"two_stage": [], "spo_plus": [], "conformal_robust": []} for name in PERIODS
    }

    t_total = time.time()

    for seed_idx in range(args.seeds):
        seed = RANDOM_SEED + seed_idx * 1000
        logger.info("=== Seed {}/{} (seed={}) ===", seed_idx + 1, args.seeds, seed)
        rng = np.random.RandomState(seed)

        # Sample training instances (from FULL train, same as original)
        X_tr_inst, c_tr_inst, _ = _sample_instances(
            X_tr_all, c_tr_all, args.n_items, args.n_train, rng
        )

        # Train SPO+ ONCE per seed on full training data
        optmodel_train = CreditPortfolioLP(n_items=args.n_items, budget=args.budget)
        spo_model, _ = _train_spo(
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
        spo_model.eval()

        # Evaluate on each temporal period
        for period_name, mask in period_masks.items():
            n_period = int(mask.sum())
            if n_period < args.n_items:
                logger.warning("  {} : only {} loans, skipping", period_name, n_period)
                for method in per_period_regrets[period_name]:
                    per_period_regrets[period_name][method].append(float("nan"))
                continue

            n_test_period = min(80, n_period // args.n_items)
            if n_test_period < 5:
                n_test_period = 5

            # Slice arrays to this period
            period_idx = np.where(mask)[0]
            X_period = X_te_all[period_idx]
            c_period = c_te_all[period_idx]
            c_ts_period = c_ts_te_all[period_idx]
            c_robust_period = c_robust_te_all[period_idx]

            # Sample test instances within this period
            rng_period = np.random.RandomState(seed + hash(period_name) % (2**31))
            X_inst, c_inst, idx_inst = _sample_instances(
                X_period, c_period, args.n_items, n_test_period, rng_period
            )
            c_ts_inst = _index_costs(c_ts_period, idx_inst)
            c_robust_inst = _index_costs(c_robust_period, idx_inst)

            # SPO+ predictions
            import torch

            n_input = args.n_items * n_features
            X_flat = X_inst.reshape(n_test_period, n_input)
            with torch.no_grad():
                c_spo_inst = spo_model(torch.tensor(X_flat, dtype=torch.float32)).numpy()

            # Compute true optima + regrets
            optmodel_eval = CreditPortfolioLP(n_items=args.n_items, budget=args.budget)
            true_optima = _compute_true_optima(c_inst, optmodel_eval)

            regrets_ts = _compute_regret(c_ts_inst, c_inst, optmodel_eval.copy(), true_optima)
            regrets_spo = _compute_regret(c_spo_inst, c_inst, optmodel_eval.copy(), true_optima)
            regrets_robust = _compute_regret(
                c_robust_inst, c_inst, optmodel_eval.copy(), true_optima
            )

            per_period_regrets[period_name]["two_stage"].append(float(regrets_ts.mean()))
            per_period_regrets[period_name]["spo_plus"].append(float(regrets_spo.mean()))
            per_period_regrets[period_name]["conformal_robust"].append(float(regrets_robust.mean()))

            logger.info(
                "  {} (n_test={}): ts={:.4f} spo={:.4f} crpto={:.4f}",
                period_name,
                n_test_period,
                regrets_ts.mean(),
                regrets_spo.mean(),
                regrets_robust.mean(),
            )

    total_time = time.time() - t_total
    logger.info("All seeds done in {:.1f}s", total_time)

    # ── 5. Conformal coverage per period (deterministic, no seed) ───────────
    period_coverage: dict[str, dict] = {}
    for period_name, mask in period_masks.items():
        ci_slice = ci.loc[mask]
        if len(ci_slice) == 0:
            continue
        period_coverage[period_name] = _evaluate_period_coverage(ci_slice)
        logger.info(
            "  {} coverage: 90%={:.2%} width={:.4f} min_grade={:.2%}",
            period_name,
            period_coverage[period_name]["coverage_90"],
            period_coverage[period_name]["avg_width_90"],
            period_coverage[period_name].get("min_grade_coverage_90", 0) or 0,
        )

    # ── 6. Aggregate into detail DataFrame ──────────────────────────────────
    rows = []
    for period_name in PERIODS:
        mask = period_masks[period_name]
        n_loans = int(mask.sum())
        default_rate = float(test.loc[mask, "default_flag"].mean()) if n_loans > 0 else 0.0

        regrets = per_period_regrets[period_name]
        cov = period_coverage.get(period_name, {})

        ts_vals = [v for v in regrets["two_stage"] if not np.isnan(v)]
        spo_vals = [v for v in regrets["spo_plus"] if not np.isnan(v)]
        cr_vals = [v for v in regrets["conformal_robust"] if not np.isnan(v)]

        ts_mean = float(np.mean(ts_vals)) if ts_vals else float("nan")
        spo_mean = float(np.mean(spo_vals)) if spo_vals else float("nan")
        cr_mean = float(np.mean(cr_vals)) if cr_vals else float("nan")
        ts_std = float(np.std(ts_vals)) if ts_vals else float("nan")
        spo_std = float(np.std(spo_vals)) if spo_vals else float("nan")
        cr_std = float(np.std(cr_vals)) if cr_vals else float("nan")

        spo_improvement = (
            ((ts_mean - spo_mean) / (abs(ts_mean) + 1e-9) * 100) if ts_vals and spo_vals else None
        )

        rows.append(
            {
                "period": period_name,
                "n_loans": n_loans,
                "default_rate": default_rate,
                "two_stage_mean_regret": ts_mean,
                "two_stage_std_regret": ts_std,
                "spo_plus_mean_regret": spo_mean,
                "spo_plus_std_regret": spo_std,
                "conformal_robust_mean_regret": cr_mean,
                "conformal_robust_std_regret": cr_std,
                "spo_improvement_pct": spo_improvement,
                "coverage_90": cov.get("coverage_90"),
                "coverage_95": cov.get("coverage_95"),
                "avg_width_90": cov.get("avg_width_90"),
                "min_grade_coverage_90": cov.get("min_grade_coverage_90"),
            }
        )

    detail_df = pd.DataFrame(rows)
    detail_path = DATA_DIR / "crpto_vs_spo_stability_detail.parquet"
    detail_df.to_parquet(detail_path, index=False)
    logger.info("Saved: {}", detail_path)

    # ── 7. Summary JSON ─────────────────────────────────────────────────────
    coverages_90 = [r["coverage_90"] for r in rows if r["coverage_90"] is not None]
    spo_improvements = [
        r["spo_improvement_pct"] for r in rows if r["spo_improvement_pct"] is not None
    ]

    per_period_json = {}
    for r in rows:
        regrets = per_period_regrets[r["period"]]
        per_period_json[r["period"]] = {
            "n_loans": r["n_loans"],
            "default_rate": round(r["default_rate"], 4),
            "regret": {
                "two_stage": {
                    "mean": round(r["two_stage_mean_regret"], 6),
                    "std": round(r["two_stage_std_regret"], 6),
                    "per_seed": regrets["two_stage"],
                },
                "spo_plus": {
                    "mean": round(r["spo_plus_mean_regret"], 6),
                    "std": round(r["spo_plus_std_regret"], 6),
                    "per_seed": regrets["spo_plus"],
                },
                "conformal_robust": {
                    "mean": round(r["conformal_robust_mean_regret"], 6),
                    "std": round(r["conformal_robust_std_regret"], 6),
                    "per_seed": regrets["conformal_robust"],
                },
            },
            "spo_improvement_vs_ts_pct": (
                round(r["spo_improvement_pct"], 2) if r["spo_improvement_pct"] is not None else None
            ),
            "coverage_90": round(r["coverage_90"], 4) if r["coverage_90"] is not None else None,
            "coverage_95": round(r["coverage_95"], 4) if r["coverage_95"] is not None else None,
            "avg_width_90": round(r["avg_width_90"], 4) if r["avg_width_90"] is not None else None,
            "min_grade_coverage_90": (
                round(r["min_grade_coverage_90"], 4)
                if r["min_grade_coverage_90"] is not None
                else None
            ),
        }

    summary = {
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION, run_tag=run_tag, allow_untracked=True
        ),
        "config": {
            "n_items": args.n_items,
            "budget": args.budget,
            "n_train_instances": args.n_train,
            "epochs": args.epochs,
            "n_seeds": args.seeds,
            "n_features": n_features,
            "feature_names": avail,
            "lgd": LGD,
        },
        "per_period": per_period_json,
        "stability_summary": {
            "coverage_always_above_target": all(c >= 0.90 for c in coverages_90),
            "coverage_range": [round(min(coverages_90), 4), round(max(coverages_90), 4)],
            "spo_improvement_range_pct": (
                [round(min(spo_improvements), 2), round(max(spo_improvements), 2)]
                if spo_improvements
                else None
            ),
        },
        "train_time_seconds": round(total_time, 1),
    }

    json_path = DATA_DIR / "crpto_vs_spo_stability.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: {}", json_path)

    # ── 8. Generate publication figure ──────────────────────────────────────
    _generate_stability_figure(detail_df, OUT_DIR)

    logger.info("Done. Artifacts: {} + {} + fig11", json_path, detail_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
