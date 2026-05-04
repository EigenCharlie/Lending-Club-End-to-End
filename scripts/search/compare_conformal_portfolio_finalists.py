"""Run focused portfolio tradeoff comparisons across conformal finalists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.optimize_portfolio_tradeoff import main as tradeoff_main  # noqa: E402


def _coerce_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _default_label(path: str) -> str:
    candidate = Path(path).stem or "finalist"
    return candidate.replace("/", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-paths", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--risk-grid", default="0.06,0.08,0.10,0.12")
    parser.add_argument("--aversion-grid", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--robust-min-budget-utilization", type=float, default=0.05)
    parser.add_argument("--strict-risk-threshold", type=float, default=0.12)
    parser.add_argument("--robust-pd-slack-penalty", type=float, default=1.5)
    parser.add_argument("--grid-profile", default="custom")
    parser.add_argument("--solver-backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument(
        "--candidate-universe-path",
        default="data/processed/champion_candidate_universe.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/portfolio_tradeoff/conformal_finalist_comparison.parquet",
    )
    parser.add_argument(
        "--output-json",
        default="models/portfolio_tradeoff/conformal_finalist_comparison.json",
    )
    args = parser.parse_args(argv)

    interval_paths = _coerce_csv(args.interval_paths)
    labels = _coerce_csv(args.labels)
    if labels and len(labels) != len(interval_paths):
        raise ValueError("labels and interval-paths must have the same number of entries")
    if not labels:
        labels = [_default_label(path) for path in interval_paths]

    rows: list[dict[str, object]] = []
    for label, interval_path in zip(labels, interval_paths, strict=True):
        artifact_namespace = f"conformal-finalist-{label}"
        tradeoff_main(
            config_path=args.config,
            risk_grid=args.risk_grid,
            aversion_grid=args.aversion_grid,
            max_candidates=args.max_candidates,
            random_state=args.random_state,
            robust_min_budget_utilization=args.robust_min_budget_utilization,
            strict_risk_threshold=args.strict_risk_threshold,
            robust_pd_slack_penalty=args.robust_pd_slack_penalty,
            grid_profile=args.grid_profile,
            solver_backend=args.solver_backend,
            candidate_universe_path=args.candidate_universe_path,
            conformal_intervals_path=interval_path,
            artifact_namespace=artifact_namespace,
            run_tag=f"portfolio-finalist-{label}",
        )
        research_policy_path = (
            Path("models/portfolio_tradeoff")
            / artifact_namespace
            / "portfolio_research_policy.json"
        )
        frontier_path = (
            Path("data/processed/portfolio_tradeoff")
            / artifact_namespace
            / "portfolio_robustness_frontier.parquet"
        )
        summary_path = (
            Path("data/processed/portfolio_tradeoff")
            / artifact_namespace
            / "portfolio_robustness_summary.parquet"
        )
        research_payload = json.loads(research_policy_path.read_text(encoding="utf-8"))
        selection_metrics = dict(research_payload.get("selection_metrics", {}) or {})
        selected_policy = dict(research_payload.get("selected_policy", {}) or {})
        rows.append(
            {
                "label": label,
                "conformal_intervals_path": interval_path,
                "artifact_namespace": artifact_namespace,
                "frontier_path": str(frontier_path),
                "summary_path": str(summary_path),
                "research_policy_path": str(research_policy_path),
                "risk_tolerance": float(selected_policy.get("risk_tolerance", 0.0)),
                "policy_mode": str(selected_policy.get("policy_mode", "")),
                "gamma": float(selected_policy.get("gamma", 0.0)),
                "delta_cap_quantile": float(selected_policy.get("delta_cap_quantile", 0.0)),
                "tail_focus_quantile": float(selected_policy.get("tail_focus_quantile", 0.0)),
                "uncertainty_aversion": float(selected_policy.get("uncertainty_aversion", 0.0)),
                "ab_pass": bool(selection_metrics.get("ab_pass", False)),
                "ab_diff_total_return": float(selection_metrics.get("ab_diff_total_return", 0.0)),
                "realized_total_return": float(selection_metrics.get("realized_total_return", 0.0)),
                "price_of_robustness": float(selection_metrics.get("price_of_robustness", 0.0)),
                "price_of_robustness_pct": float(
                    selection_metrics.get("price_of_robustness_pct", 0.0)
                ),
                "n_funded": int(selection_metrics.get("n_funded", 0)),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["ab_pass", "realized_total_return", "price_of_robustness_pct"],
        ascending=[False, False, False],
    )
    output_parquet = Path(args.output_parquet)
    output_json = Path(args.output_json)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(output_parquet, index=False)
    output_json.write_text(
        json.dumps(
            {
                "n_finalists": int(len(comparison)),
                "rows": comparison.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
