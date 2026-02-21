"""Generate consolidated Model Risk Management (MRM) validation report.

Aggregates status JSON files from pipeline, conformal, governance,
and fairness subsystems into a single MRM report following SR 11-7.

Usage:
    uv run python scripts/generate_mrm_report.py
    uv run python scripts/generate_mrm_report.py --config configs/mrm_policy.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger


def _load_status(path: str | Path) -> dict:
    """Load a JSON status file, returning empty dict if missing."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Status file not found: {p}")
        return {}
    with open(p) as f:
        return json.load(f)


def _check_pass(status: dict) -> bool:
    """Check if a status dict indicates overall pass."""
    if not status:
        return False
    # Try common patterns for overall pass
    for key in ["overall_pass", "all_passed", "pass"]:
        if key in status:
            return bool(status[key])
    # For conformal: check if all checks passed
    if "checks" in status:
        checks = status["checks"]
        if isinstance(checks, list):
            return all(c.get("passed", False) for c in checks)
    return False


def _overall_compliance(statuses: dict[str, dict]) -> dict:
    """Compute top-level compliance summary."""
    subsystem_pass = {}
    for name, status in statuses.items():
        subsystem_pass[name] = _check_pass(status)

    return {
        "overall_pass": all(subsystem_pass.values()) if subsystem_pass else False,
        "subsystems": subsystem_pass,
        "n_subsystems": len(statuses),
        "n_passing": sum(subsystem_pass.values()),
    }


def main(config_path: str = "configs/mrm_policy.yaml") -> None:
    """Generate the MRM validation report."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    artifacts = cfg["artifacts"]

    # Load all status files
    statuses = {
        "pipeline": _load_status(artifacts["pipeline_summary"]),
        "conformal": _load_status(artifacts["conformal_status"]),
        "governance": _load_status(artifacts["governance_status"]),
        "fairness": _load_status(artifacts["fairness_status"]),
    }

    compliance = _overall_compliance(statuses)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": cfg["model"],
        "governance_policy": cfg["governance"],
        "retraining_triggers": cfg["retraining_triggers"],
        "challenger_criteria": cfg["challenger"],
        "pipeline_summary": statuses["pipeline"],
        "conformal_status": statuses["conformal"],
        "governance_status": statuses["governance"],
        "fairness_status": statuses["fairness"],
        "compliance_summary": compliance,
    }

    output_path = Path(cfg["output"]["mrm_report_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    pass_label = "PASS" if compliance["overall_pass"] else "FAIL"
    logger.info(
        f"MRM report: {pass_label} "
        f"({compliance['n_passing']}/{compliance['n_subsystems']} subsystems). "
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MRM validation report")
    parser.add_argument("--config", default="configs/mrm_policy.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
