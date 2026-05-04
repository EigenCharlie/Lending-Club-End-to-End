"""Declarative pipeline-family and profile registry helpers.

This module centralizes the semantic topology used by the orchestration
wrappers. The goal is to keep the new pipeline-first contract in tracked YAML
files instead of hardcoding family semantics inside the compatibility launcher.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG_DIR = REPO_ROOT / "configs" / "pipelines"
PROFILE_CONFIG_DIR = REPO_ROOT / "configs" / "profiles"
LEGACY_PIPELINE_ALIASES = {
    "canonical_rebuild": "core_canonical",
    "champion_search": "search_pd",
    "challenger_promotion": "search_pd",
    "insights_factory": "research_labs",
}

LEGACY_PROFILE_DEFAULTS = {
    "canonical_rebuild": "canonical_operational",
    "champion_search": "champion_search_max",
    "challenger_promotion": "canonical_monotonic_promotion_full",
}

FAMILY_DEFAULTS = {
    "core_canonical": "core_canonical_cpu",
    "search_pd": "search_pd_default",
    "search_conformal": "search_conformal_default",
    "search_portfolio": "search_portfolio_default",
    "search_paper2_ifrs9": "search_paper2_ifrs9_exhaustive",
    "paper1_e2e": "paper1_e2e_default",
    "paper2_e2e": "paper2_e2e_default",
    "diagnostics_governance": "diagnostics_governance_default",
    "research_labs": "research_labs_default",
}


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def resolve_pipeline_family(name: str | None) -> str:
    raw = str(name or "").strip().lower()
    if not raw:
        return "paper1_e2e"
    return LEGACY_PIPELINE_ALIASES.get(raw, raw)


def legacy_family_alias(name: str | None) -> str | None:
    raw = str(name or "").strip().lower()
    if not raw:
        return None
    return raw if raw in LEGACY_PIPELINE_ALIASES else None


def load_pipeline_definition(name: str | None) -> dict[str, object]:
    family = resolve_pipeline_family(name)
    payload = _load_yaml(PIPELINE_CONFIG_DIR / f"{family}.yaml")
    if not payload:
        return {
            "schema_version": "2026-03-31.1",
            "pipeline_family": family,
            "description": "",
            "artifact_scope": "research",
            "promotion_state": "research_open",
            "writes_canonical_artifacts": False,
            "upstream_required": False,
            "default_profile": FAMILY_DEFAULTS.get(family, "search_pd_default"),
            "allowed_step_groups": ["preflight"],
            "forbidden_step_groups": [],
            "required_envs": [],
            "produced_registries": [],
            "papers_served": [],
        }
    payload["pipeline_family"] = family
    if "default_profile" not in payload:
        payload["default_profile"] = FAMILY_DEFAULTS.get(family, "search_pd_default")
    return payload


def default_profile_for_family(name: str | None, sampling_profile: str | None = None) -> str:
    raw = str(name or "").strip().lower()
    if raw in LEGACY_PROFILE_DEFAULTS:
        return LEGACY_PROFILE_DEFAULTS[raw]
    family = resolve_pipeline_family(raw)
    default = FAMILY_DEFAULTS.get(family)
    if default:
        return default
    return str(sampling_profile or "").strip() or "search_pd_default"


def load_profile_config(profile_name: str | None) -> dict[str, object]:
    name = str(profile_name or "").strip()
    if not name:
        return {}
    return _load_yaml(PROFILE_CONFIG_DIR / f"{name}.yaml")


def build_pipeline_contract(
    *,
    pipeline_family: str,
    pipeline_profile_arg: str | None,
    sampling_profile: str,
    writes_canonical_artifacts_arg: bool | None,
    upstream_canonical_run_tag: str | None,
) -> dict[str, object]:
    requested_family = str(pipeline_family).strip().lower() or "paper1_e2e"
    family = resolve_pipeline_family(requested_family)
    pipeline_def = load_pipeline_definition(family)
    pipeline_profile = str(pipeline_profile_arg or "").strip() or default_profile_for_family(
        requested_family, sampling_profile
    )
    writes_canonical_artifacts = bool(pipeline_def.get("writes_canonical_artifacts", False))
    if writes_canonical_artifacts_arg is not None:
        writes_canonical_artifacts = bool(writes_canonical_artifacts_arg)
    return {
        "requested_pipeline_family": requested_family,
        "legacy_pipeline_alias": legacy_family_alias(requested_family),
        "pipeline_family": family,
        "pipeline_profile": pipeline_profile,
        "description": str(pipeline_def.get("description", "") or ""),
        "artifact_scope": str(pipeline_def.get("artifact_scope", "research") or "research"),
        "promotion_state": str(
            pipeline_def.get("promotion_state", "research_open") or "research_open"
        ),
        "writes_canonical_artifacts": writes_canonical_artifacts,
        "upstream_required": bool(pipeline_def.get("upstream_required", False)),
        "allowed_step_groups": list(pipeline_def.get("allowed_step_groups", []) or []),
        "forbidden_step_groups": list(pipeline_def.get("forbidden_step_groups", []) or []),
        "required_envs": list(pipeline_def.get("required_envs", []) or []),
        "produced_registries": list(pipeline_def.get("produced_registries", []) or []),
        "papers_served": list(pipeline_def.get("papers_served", []) or []),
        "upstream_canonical_run_tag": upstream_canonical_run_tag,
    }
