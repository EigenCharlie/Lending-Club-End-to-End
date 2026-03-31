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
from typing import Any

import yaml
from loguru import logger
from skops.card import Card
from skops.io import dump as skops_dump

from src.utils.artifact_metadata import resolve_run_tag as resolve_artifact_run_tag
from src.utils.io_utils import load_pickle_compat


def _load_status(path: str | Path) -> dict:
    """Load a JSON status file, returning empty dict if missing."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists() or p.is_dir():
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
        if isinstance(checks, dict):
            return all(bool(v) for v in checks.values())
        if isinstance(checks, list):
            return all(c.get("passed", False) for c in checks)
    # Pipeline summary artifact has no explicit overall pass flag.
    return "pipeline" in status and "pd_model" in status


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


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_skops_sidecar(
    cfg: dict[str, Any], statuses: dict[str, dict], output_path: Path
) -> dict[str, Any]:
    model_cfg = cfg.get("model", {}) or {}
    champion_artifact = str(model_cfg.get("champion_artifact", ""))
    model_name = str(model_cfg.get("name", "CorePD"))
    model_version = str(model_cfg.get("version", "unknown"))

    skops_dir = output_path.parent / "skops"
    skops_dir.mkdir(parents=True, exist_ok=True)

    exports: list[dict[str, Any]] = []
    logreg_path = Path("models/pd_logreg_baseline.pkl")
    logreg_payload = None
    if logreg_path.exists():
        try:
            logreg_payload = load_pickle_compat(logreg_path)
            target_path = skops_dir / "pd_logreg_baseline.skops"
            skops_dump(logreg_payload, target_path)
            exports.append(
                {
                    "name": "pd_logreg_baseline",
                    "source_path": str(logreg_path),
                    "export_path": str(target_path),
                    "status": "exported",
                }
            )
        except Exception as exc:
            exports.append(
                {
                    "name": "pd_logreg_baseline",
                    "source_path": str(logreg_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    card_summary = {
        "model_name": model_name,
        "model_version": model_version,
        "champion_artifact": champion_artifact,
        "subsystem_pass": {
            name: bool(status.get("overall_pass", False)) for name, status in statuses.items()
        },
        "limitations": [
            "Champion artifact remains CatBoost/CBM; skops export is limited to sklearn-compatible artifacts.",
            "Fairlearn is integrated as a sidecar audit and does not replace the canonical fairness gate in this phase.",
            "Observational causal outputs remain insights_only and are not promotion-eligible.",
        ],
        "exports": exports,
    }

    if isinstance(logreg_payload, dict) and "model" in logreg_payload:
        try:
            card = Card(logreg_payload["model"])
            card.add_metrics(
                section="Validation",
                description="Pipeline-level subsystem gates consumed by MRM.",
                metrics={
                    "pipeline_overall_pass": bool(
                        statuses.get("pipeline", {}).get("overall_pass", False)
                    ),
                    "conformal_overall_pass": bool(
                        statuses.get("conformal", {}).get("overall_pass", False)
                    ),
                    "governance_overall_pass": bool(
                        statuses.get("governance", {}).get("overall_pass", False)
                    ),
                    "fairness_overall_pass": bool(
                        statuses.get("fairness", {}).get("overall_pass", False)
                    ),
                },
            )
            card.add_hyperparams()
            model_card_md = card.render()
        except Exception as exc:
            model_card_md = (
                f"# {model_name} model card\n\n"
                f"Skops card rendering failed: {exc}\n\n"
                f"Exports:\n{json.dumps(_safe_jsonable(exports), indent=2)}\n"
            )
            card_summary["card_render_status"] = "fallback"
            card_summary["card_render_error"] = str(exc)
        else:
            card_summary["card_render_status"] = "rendered"
    else:
        model_card_md = (
            f"# {model_name} model card\n\n"
            "No sklearn-compatible baseline artifact was available for skops rendering.\n"
        )
        card_summary["card_render_status"] = "no_supported_model"

    model_card_html = (
        "<html><body><pre>"
        + model_card_md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre></body></html>"
    )
    model_card_json_path = output_path.parent / "corepd_model_card.json"
    model_card_html_path = output_path.parent / "corepd_model_card.html"
    model_card_json_path.write_text(
        json.dumps(_safe_jsonable(card_summary), indent=2, default=str),
        encoding="utf-8",
    )
    model_card_html_path.write_text(model_card_html, encoding="utf-8")

    return {
        "exports": exports,
        "model_card_json": str(model_card_json_path),
        "model_card_html": str(model_card_html_path),
        "card_render_status": card_summary.get("card_render_status", "unknown"),
    }


def _resolve_run_tag(run_tag_arg: str | None) -> str:
    """Resolve run_tag with official pipeline env fallback before pipeline_summary."""
    pipeline_path = Path("data/processed/pipeline_summary.json")
    pipeline_tag = None
    if pipeline_path.exists():
        try:
            data = json.loads(pipeline_path.read_text(encoding="utf-8"))
            tag = data.get("run_tag")
            if tag:
                pipeline_tag = str(tag)
        except Exception:
            pipeline_tag = None
    return resolve_artifact_run_tag(
        run_tag_arg,
        fallback_candidates=[pipeline_tag],
        allow_untracked=True,
    )


def main(config_path: str = "configs/mrm_policy.yaml", run_tag: str | None = None) -> None:
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
    diagnostic_statuses = {
        "pd_backtesting": _load_status(artifacts.get("pd_backtesting_status", "")),
        "bootstrap_validation": _load_status(artifacts.get("bootstrap_validation_status", "")),
        "monotonicity": _load_status(artifacts.get("monotonicity_status", "")),
        "ifrs9_diagnostics": _load_status(artifacts.get("ifrs9_diagnostics_status", "")),
        "encoding_stability": _load_status(artifacts.get("encoding_stability_status", "")),
        "pd_validation_interpretation": _load_status(
            artifacts.get("pd_validation_interpretation_status", "")
        ),
        "calibration_mapping": _load_status(artifacts.get("calibration_mapping_status", "")),
        "model_shift": _load_status(artifacts.get("model_shift_status", "")),
    }

    compliance = _overall_compliance(statuses)
    resolved_run_tag = _resolve_run_tag(run_tag)
    now_iso = datetime.now(tz=UTC).isoformat()

    report = {
        "schema_version": "2026-03-14.1",
        "generated_at_utc": now_iso,
        "generated_at": now_iso,
        "run_tag": resolved_run_tag,
        "overall_pass": compliance["overall_pass"],
        "model": cfg["model"],
        "governance_policy": cfg["governance"],
        "retraining_triggers": cfg["retraining_triggers"],
        "challenger_criteria": cfg["challenger"],
        "pipeline_summary": statuses["pipeline"],
        "conformal_status": statuses["conformal"],
        "governance_status": statuses["governance"],
        "fairness_status": statuses["fairness"],
        "diagnostic_statuses": diagnostic_statuses,
        "compliance_summary": compliance,
    }

    output_path = Path(cfg["output"]["mrm_report_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["skops_governance"] = _build_skops_sidecar(cfg, statuses, output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Write models/mrm_report_status.json wrapper for Streamlit pages
    status_path = Path("models/mrm_report_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "2026-03-14.1",
                "generated_at_utc": now_iso,
                "run_tag": resolved_run_tag,
                "overall_pass": compliance["overall_pass"],
                "compliance_summary": compliance,
                "report_path": str(output_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pass_label = "PASS" if compliance["overall_pass"] else "FAIL"
    logger.info(
        f"MRM report: {pass_label} "
        f"({compliance['n_passing']}/{compliance['n_subsystems']} subsystems). "
        f"Saved: {output_path}"
    )
    logger.info(f"MRM status wrapper: {status_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MRM validation report")
    parser.add_argument("--config", default="configs/mrm_policy.yaml")
    parser.add_argument("--run-tag", default=None, help="Run tag to stamp on the report")
    args = parser.parse_args()
    main(config_path=args.config, run_tag=args.run_tag)
