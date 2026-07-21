"""Run alpha-gamma bound validation across multiple conformal finalists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_alpha_gamma_bound import main as validate_bound_main  # noqa: E402


def _coerce_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _default_label(path: str) -> str:
    return (Path(path).stem or "finalist").replace("/", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-paths", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--portfolio-policy-path", default="models/champion_portfolio_policy.json")
    parser.add_argument("--alpha-grid", default=None)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-parquet",
        default="data/processed/alpha_gamma_bound_finalist_sweep.parquet",
    )
    parser.add_argument(
        "--output-json",
        default="data/processed/alpha_gamma_bound_finalist_sweep.json",
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
        output_json = (
            Path("data/processed/alpha_gamma_bound")
            / f"{label}_alpha_gamma_bound_validation_exact.json"
        )
        validate_bound_main(
            [
                "--conformal-intervals-path",
                interval_path,
                "--portfolio-policy-path",
                args.portfolio_policy_path,
                "--allocator-mode",
                "exact",
                "--output-json",
                str(output_json),
                "--alpha-grid",
                str(args.alpha_grid or ""),
                "--max-candidates",
                str(int(args.max_candidates)),
                "--random-state",
                str(int(args.random_state)),
                "--figure-prefix",
                str(
                    Path("reports/paper_material/figures_publication")
                    / f"{label}_alpha_gamma_bound"
                ),
                "--comparison-output",
                str(Path("data/processed/alpha_gamma_bound") / f"{label}_proxy_vs_exact.parquet"),
            ]
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            rows.append(
                {
                    "label": label,
                    "conformal_intervals_path": interval_path,
                    **result,
                }
            )

    sweep = pd.DataFrame(rows)
    output_parquet = Path(args.output_parquet)
    output_json_path = Path(args.output_json)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(output_parquet, index=False)
    output_json_path.write_text(
        json.dumps(
            {
                "n_rows": int(len(sweep)),
                "rows": sweep.to_dict(orient="records"),
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
