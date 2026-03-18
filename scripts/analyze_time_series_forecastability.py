"""Generate a governed forecastability report for the time-series stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from scripts.forecast_default_rates import _load_history
from src.models.time_series import compute_forecastability_report, load_time_series_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Saved {}", path)


def main(config_path: str = "configs/time_series.yaml") -> None:
    config = load_time_series_config(config_path)
    outputs = config.get("outputs", {})
    report_path = Path(
        str(
            outputs.get(
                "forecastability_report_path", "data/processed/ts_forecastability_report.parquet"
            )
        )
    )
    status_path = Path(
        str(
            outputs.get(
                "forecastability_status_path", "models/time_series_forecastability_status.json"
            )
        )
    )

    _, panel_history = _load_history()
    report, status = compute_forecastability_report(
        panel_history,
        season_length=int(config.get("season_length", 12)),
        forecastability_cfg=config.get("forecastability", {}),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(report_path, index=False)
    _write_json(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/time_series.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
