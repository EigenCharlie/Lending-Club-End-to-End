"""Shared helpers for canonical threshold semantics across artifacts and UI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("models/threshold_semantics.json")
SCHEMA_VERSION = "2026-03-13.1"

DEFAULT_BUSINESS_MEANING = {
    "pd_internal_selected_threshold": (
        "Threshold interno de screening/seleccion PD usado en busqueda y analisis tecnico."
    ),
    "pd_internal_fallback_threshold": (
        "Fallback interno tomado de la policy de fairness durante la busqueda de threshold PD."
    ),
    "fairness_primary_threshold": (
        "Threshold operativo principal para auditoria de fairness y narrativa de aprobacion."
    ),
    "decision_policy_global_threshold": (
        "Threshold operativo global aplicado por la policy de decision/aprobacion."
    ),
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def load_threshold_semantics(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_threshold_semantics(
    *,
    pd_internal_selected_threshold: float | None = None,
    pd_internal_fallback_threshold: float | None = None,
    fairness_primary_threshold: float | None = None,
    decision_policy_global_threshold: float | None = None,
    source_artifacts: dict[str, str] | None = None,
    business_meaning: dict[str, str] | None = None,
    run_tag: str | None = None,
    path: str | Path = DEFAULT_PATH,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    payload = load_threshold_semantics(target)
    resolved_run_tag = str(run_tag or payload.get("run_tag") or "untracked").strip() or "untracked"
    current_sources = payload.get("source_artifacts", {}) if isinstance(payload, dict) else {}
    merged_sources = {
        **(current_sources if isinstance(current_sources, dict) else {}),
        **(source_artifacts or {}),
    }
    meanings = {
        **DEFAULT_BUSINESS_MEANING,
        **(
            payload.get("business_meaning", {})
            if isinstance(payload.get("business_meaning", {}), dict)
            else {}
        ),
        **(business_meaning or {}),
    }

    resolved = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "pd_internal_selected_threshold": _safe_float(
            pd_internal_selected_threshold
            if pd_internal_selected_threshold is not None
            else payload.get("pd_internal_selected_threshold")
        ),
        "pd_internal_fallback_threshold": _safe_float(
            pd_internal_fallback_threshold
            if pd_internal_fallback_threshold is not None
            else payload.get("pd_internal_fallback_threshold")
        ),
        "fairness_primary_threshold": _safe_float(
            fairness_primary_threshold
            if fairness_primary_threshold is not None
            else payload.get("fairness_primary_threshold")
        ),
        "decision_policy_global_threshold": _safe_float(
            decision_policy_global_threshold
            if decision_policy_global_threshold is not None
            else payload.get("decision_policy_global_threshold")
        ),
        "source_artifacts": merged_sources,
        "business_meaning": meanings,
    }
    if extra:
        resolved.update(extra)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def resolve_operational_threshold(
    semantics: dict[str, Any] | None = None,
    default: float = 0.5,
) -> float:
    payload = semantics or {}
    for key in ("decision_policy_global_threshold", "fairness_primary_threshold"):
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return float(default)


def resolve_pd_internal_threshold(
    semantics: dict[str, Any] | None = None,
    default: float = 0.5,
) -> float:
    payload = semantics or {}
    for key in ("pd_internal_selected_threshold", "pd_internal_fallback_threshold"):
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return float(default)
