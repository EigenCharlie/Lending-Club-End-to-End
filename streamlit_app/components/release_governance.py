"""Release governance summary block for Streamlit pages."""

from __future__ import annotations

import json

import streamlit as st

from streamlit_app.utils import (
    REPORTS_DIR,
    evaluate_run_tag_coherence,
    load_official_baseline_registry,
    try_load_json,
)


def _load_comparison(current_run_tag: str) -> dict:
    path = REPORTS_DIR / "run_comparisons" / current_run_tag / "comparison.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_first_available_comparison(run_tags: list[str]) -> tuple[dict, str]:
    for tag in run_tags:
        clean_tag = str(tag or "").strip()
        if not clean_tag:
            continue
        payload = _load_comparison(clean_tag)
        if payload:
            return payload, clean_tag
    return {}, ""


def render_release_governance(
    *,
    current_run_tag: str | None = None,
    governance_status: dict | None = None,
    conformal_status: dict | None = None,
    show_title: bool = True,
) -> None:
    """Render a compact release-governance block with baseline + gate traceability."""
    baseline_registry = load_official_baseline_registry()
    pipeline_summary = try_load_json("pipeline_summary", directory="data", default={})
    official_run_tag = str(
        baseline_registry.get("official_run_tag")
        or pipeline_summary.get("official_baseline_run_tag")
        or ""
    ).strip()

    governance = governance_status or try_load_json(
        "governance_status", directory="models", default={}
    )
    conformal = conformal_status or try_load_json(
        "conformal_policy_status", directory="models", default={}
    )
    fairness = try_load_json("fairness_audit_status", directory="models", default={})
    ab_status = try_load_json("ab_simulation_status", directory="models", default={})

    resolved_run_tag = str(
        current_run_tag
        or pipeline_summary.get("run_tag")
        or governance.get("run_tag")
        or conformal.get("run_tag")
        or fairness.get("run_tag")
        or ""
    ).strip()

    comparison_candidates = [resolved_run_tag, official_run_tag]
    comparison, comparison_run_tag = _load_first_available_comparison(comparison_candidates)
    coherence = evaluate_run_tag_coherence(
        official_run_tag,
        {
            "governance_status": governance,
            "conformal_policy_status": conformal,
            "fairness_audit_status": fairness,
            "ab_simulation_status": ab_status,
        },
    )

    with st.container(border=True):
        if show_title:
            st.markdown("### Release Governance")

        if comparison:
            gates_label = "PASS" if bool(comparison.get("overall_pass", False)) else "REVISION"
        elif isinstance(governance, dict) and "overall_pass" in governance:
            gates_label = "PARCIAL"
        else:
            gates_label = "SIN DATOS"

        if official_run_tag:
            coherence_label = "OK" if bool(coherence.get("coherent", False)) else "REVISION"
        else:
            coherence_label = "PARCIAL" if bool(coherence.get("coherent", False)) else "REVISION"

        cols = st.columns(4)
        cols[0].metric("Baseline oficial", official_run_tag or "N/D")
        cols[1].metric("Run actual", resolved_run_tag or "N/D")
        cols[2].metric("Gates", gates_label)
        cols[3].metric("Coherencia run_tag", coherence_label)

        if comparison:
            st.caption(
                f"Fuente gates: reports/run_comparisons/{comparison_run_tag}/comparison.json. "
                "comparison.json: "
                f"artifact_coherence={comparison.get('artifact_coherence_pass')}, "
                f"conformal_promotion={comparison.get('conformal_promotion_pass')}, "
                f"fairness_absolute_business={comparison.get('fairness_absolute_business_pass')}, "
                f"ab_no_regression={comparison.get('ab_no_regression_pass')}"
            )
            if comparison.get("conformal_statistical_warning", False):
                st.warning(
                    "Conformal estricto mantiene warning estadístico (Kupiec/Christoffersen); "
                    "la promoción operacional sigue en PASS."
                )
        else:
            expected_paths = []
            if resolved_run_tag:
                expected_paths.append(f"reports/run_comparisons/{resolved_run_tag}/comparison.json")
            if official_run_tag and official_run_tag != resolved_run_tag:
                expected_paths.append(f"reports/run_comparisons/{official_run_tag}/comparison.json")
            if expected_paths:
                st.info(
                    "No se encontró comparison.json. Rutas intentadas: " + "; ".join(expected_paths)
                )
            else:
                st.info(
                    "No se pudo resolver run_tag/baseline para buscar comparison.json. "
                    "Revisa pipeline_summary.json y baseline registry."
                )

        if not official_run_tag:
            st.caption(
                "Baseline oficial N/D: falta configs/baselines/canonical_operational_baseline.json "
                "(o el fallback legacy configs/baselines/core_official_baseline.json) "
                "o pipeline_summary.official_baseline_run_tag."
            )

        if not coherence.get("coherent", False):
            missing = coherence.get("missing_run_tag_artifacts", [])
            mismatched = coherence.get("mismatched_artifacts", [])
            if missing:
                st.caption(f"Artifacts sin run_tag: {', '.join(missing)}")
            if mismatched:
                st.caption(f"Artifacts con run_tag distinto: {', '.join(mismatched)}")
