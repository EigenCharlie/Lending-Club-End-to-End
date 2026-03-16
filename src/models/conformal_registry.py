"""Canonical registry of conformal methods and libraries used in the project."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_conformal_method_registry(*, run_tag: str = "untracked") -> dict[str, Any]:
    """Return the canonical registry of conformal/calibration methods.

    The registry is intentionally explicit about what is:
    - operationally adopted,
    - research-only,
    - rejected for canonical use,
    so downstream docs and bundles can explain project choices consistently.
    """
    return {
        "schema_version": "2026-03-13.1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": run_tag,
        "project_scope": "credit_risk_tabular",
        "decision_principles": [
            "roi_high",
            "baseline_compatibility",
            "reproducibility",
            "release_stable_over_colab_notebooks",
        ],
        "libraries": {
            "mapie": {
                "version_family": "1.3.x",
                "role": [
                    "primary_conformal_intervals",
                    "classification_sets",
                    "time_series_adaptive_benchmark",
                ],
                "status": "adopted",
                "notes": "Primary conformal library for PD intervals, LGD/EAD candidates, and ACI/EnbPI benchmarks.",
            },
            "crepes": {
                "version_family": "0.9.x",
                "role": ["p_values", "predictive_systems", "research_only"],
                "status": "research_only",
                "notes": "Kept for p-values/predictive systems. Not used as a Venn-Abers substitute.",
            },
            "venn_abers": {
                "version_family": "1.5.x",
                "role": ["probability_calibration"],
                "status": "adopted",
                "notes": "Canonical implementation for Venn-Abers calibration.",
            },
            "nonconformist": {
                "status": "rejected",
                "reason": "legacy_unmaintained_api",
            },
            "fortuna": {
                "status": "rejected_for_canonical_use",
                "reason": "archived_project_and_extra_dependency_surface",
            },
            "neuralprophet": {
                "status": "excluded_from_canonical_stack",
                "reason": "beta_ts_stack_outside_current_tabular_credit_scope",
            },
        },
        "methods": {
            "pd_core": {
                "calibration_default": "venn_abers",
                "calibration_candidates": ["platt", "isotonic", "venn_abers"],
                "binary_classification_sets": {
                    "status": "research_sidecar",
                    "method": "lac",
                    "artifact": "models/pd_set_prediction_status.json",
                },
            },
            "pd_conformal": {
                "operational_family": "mondrian_group_conditional",
                "implemented_variants": [
                    "global_split",
                    "mondrian_unscaled",
                    "mondrian_scaled",
                    "mondrian_selected_cfg",
                    "cross_conformal_score_space",
                ],
                "strict_diagnostics": ["kupiec", "christoffersen"],
                "selector_artifact": "models/conformal_variant_selection_status.json",
            },
            "lgd_ead": {
                "current_operational_variants": [
                    "two_stage_split",
                    "direct_split",
                    "direct_cqr",
                    "direct_adaptive_grade_temporal",
                ],
                "research_extensions": [
                    "mapie_cqr_short_benchmark",
                    "jackknife_after_bootstrap_short_benchmark",
                ],
            },
            "time_series": {
                "official_baseline": "statsforecast_native_intervals",
                "research_shortlist": ["enbpi", "aci", "online_conformal"],
                "status_artifact": "models/time_series_status.json",
            },
            "multiclass_extension": {
                "status": "backlog_p2",
                "candidate_targets": [
                    "loan_status_multiclass",
                    "stage_migration",
                    "delinquency_bucket",
                ],
            },
        },
    }
