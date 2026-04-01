"""Run calibration mapping diagnostics plus shadow conformal validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.backtest_conformal_coverage import main as backtest_conformal_main
from scripts.benchmark_conformal_variants import main as benchmark_conformal_main
from scripts.generate_conformal_intervals import main as generate_conformal_main
from scripts.run_calibration_mapping_diagnostics import main as mapping_diagnostics_main
from scripts.validate_conformal_policy import main as validate_conformal_policy_main
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag

MODEL_DIR = Path("models")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _namespaced_paths(namespace: str) -> dict[str, Path]:
    ns = str(namespace).strip().replace("/", "_")
    data_dir = Path("data/processed/conformal_gap") / ns
    models_dir = Path("models/conformal_gap") / ns
    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "results": models_dir / "conformal_results_mondrian.pkl",
        "variant_status": models_dir / "conformal_variant_selection_status.json",
        "policy_status": models_dir / "conformal_policy_status.json",
        "intervals": data_dir / "conformal_intervals_mondrian.parquet",
    }


def _evaluate_conformal_no_regression(
    canonical_policy: dict[str, Any],
    shadow_policy: dict[str, Any],
    canonical_variant_status: dict[str, Any],
    shadow_variant_status: dict[str, Any],
) -> dict[str, Any]:
    canonical_width = float(canonical_policy.get("avg_width_90", 0.0))
    shadow_width = float(shadow_policy.get("avg_width_90", 0.0))
    canonical_group = float(canonical_policy.get("min_group_coverage_90", 0.0))
    shadow_group = float(shadow_policy.get("min_group_coverage_90", 0.0))
    canonical_stability = float(
        (canonical_variant_status.get("selected_metrics", {}) or {}).get("stability_over_time", 0.0)
    )
    shadow_stability = float(
        (shadow_variant_status.get("selected_metrics", {}) or {}).get("stability_over_time", 0.0)
    )
    checks = {
        "policy_overall_pass": bool(shadow_policy.get("overall_pass", False)),
        "avg_width_90_non_regression": bool((shadow_width - canonical_width) <= 0.02),
        "min_group_coverage_90_non_regression": bool((shadow_group - canonical_group) >= -0.005),
        "stability_over_time_non_regression": bool(
            (shadow_stability - canonical_stability) <= 0.01
        ),
    }
    return {
        "checks": checks,
        "deltas": {
            "avg_width_90_delta": shadow_width - canonical_width,
            "min_group_coverage_90_delta": shadow_group - canonical_group,
            "stability_over_time_delta": shadow_stability - canonical_stability,
        },
        "pass": bool(all(checks.values())),
    }


def _final_recommendation(
    diagnostics_status: dict[str, Any],
    shadow_variant_status: dict[str, Any],
    conformal_gate: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not bool(diagnostics_status.get("stage_a_pass", False)):
        reasons.append("No shadow candidate passed Gate PD.")
        return "keep_current_calibrator", reasons

    if conformal_gate is None:
        reasons.append("Shadow conformal validation did not complete.")
        return "manual_review_required", reasons

    if not bool(conformal_gate.get("pass", False)):
        failed = [
            name for name, passed in (conformal_gate.get("checks") or {}).items() if not passed
        ]
        reasons.append(
            "Shadow candidate passed Gate PD but failed conformal no-regression/policy checks."
        )
        reasons.extend(failed)
        return "shadow_candidate_pd_only", reasons

    selected_variant = str(shadow_variant_status.get("selected_variant", "")).strip()
    if selected_variant and selected_variant != "score_decile_mondrian":
        reasons.append(
            f"Shadow conformal selector changed champion family to {selected_variant}; requires manual review."
        )
        return "manual_review_required", reasons

    reasons.append("Shadow candidate passed Gate PD and conformal no-regression checks.")
    return "shadow_candidate_pd_and_conformal", reasons


def main(run_tag: str | None = None) -> None:
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[resolve_official_baseline_run_tag()],
        require_explicit=True,
    )
    mapping_diagnostics_main(run_tag=resolved_run_tag)

    diagnostics_status = _load_json(MODEL_DIR / "calibration_mapping_status.json")
    output_path = MODEL_DIR / "calibration_mapping_shadow_impact_status.json"
    status_payload: dict[str, Any] = {
        "diagnostic_only": True,
        "run_tag": resolved_run_tag,
        "pd_diagnostics": diagnostics_status,
    }

    if not bool(diagnostics_status.get("stage_a_pass", False)):
        recommendation, reasons = _final_recommendation(diagnostics_status, {}, None)
        status_payload.update(
            {
                "recommendation": recommendation,
                "reasons": reasons,
                "shadow_validation_executed": False,
                **build_artifact_metadata(
                    schema_version="2026-03-31.1",
                    run_tag=resolved_run_tag,
                    require_explicit=True,
                ),
            }
        )
        output_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        logger.info("Calibration mapping shadow validation complete without downstream rerun.")
        return

    shadow_namespace = str(diagnostics_status.get("shadow_namespace", "")).strip()
    calibrator_override_path = str(diagnostics_status.get("shadow_candidate_path", "")).strip()
    if not shadow_namespace or not calibrator_override_path:
        raise ValueError("Stage A passed but shadow namespace/calibrator path is missing.")
    ns_paths = _namespaced_paths(shadow_namespace)

    generate_conformal_main(
        artifact_namespace=shadow_namespace,
        calibrator_override_path=calibrator_override_path,
    )
    benchmark_conformal_main(
        selected_config_path=str(ns_paths["results"]),
        artifact_namespace=shadow_namespace,
        calibrator_override_path=calibrator_override_path,
    )
    backtest_conformal_main(
        intervals_path=str(ns_paths["intervals"]),
        output_dir=str(ns_paths["data_dir"]),
    )
    validate_conformal_policy_main(
        run_tag=resolved_run_tag,
        artifact_namespace=shadow_namespace,
    )

    canonical_policy = _load_json(MODEL_DIR / "conformal_policy_status.json")
    shadow_policy = _load_json(ns_paths["policy_status"])
    canonical_variant = _load_json(MODEL_DIR / "conformal_variant_selection_status.json")
    shadow_variant = _load_json(ns_paths["variant_status"])
    conformal_gate = _evaluate_conformal_no_regression(
        canonical_policy,
        shadow_policy,
        canonical_variant,
        shadow_variant,
    )
    recommendation, reasons = _final_recommendation(
        diagnostics_status,
        shadow_variant,
        conformal_gate,
    )

    status_payload.update(
        {
            "shadow_validation_executed": True,
            "shadow_namespace": shadow_namespace,
            "shadow_candidate_path": calibrator_override_path,
            "recommendation": recommendation,
            "reasons": reasons,
            "canonical_pd_reference": {
                "current_candidate": diagnostics_status.get("current_candidate", {}),
            },
            "shadow_pd_candidate": diagnostics_status.get("best_candidate", {}),
            "canonical_conformal_reference": {
                "policy_status": canonical_policy,
                "variant_selection_status": canonical_variant,
            },
            "shadow_conformal_candidate": {
                "policy_status": shadow_policy,
                "variant_selection_status": shadow_variant,
            },
            "conformal_gate": conformal_gate,
            **build_artifact_metadata(
                schema_version="2026-03-31.1",
                run_tag=resolved_run_tag,
                require_explicit=True,
            ),
        }
    )
    output_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    logger.info("Calibration mapping shadow validation saved: {}", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(run_tag=args.run_tag)
