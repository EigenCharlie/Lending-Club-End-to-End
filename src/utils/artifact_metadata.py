"""Shared helpers for run-tag resolution and artifact metadata contracts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any


class MissingRunTagError(ValueError):
    """Raised when an official artifact writer lacks a resolvable run tag."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _candidate_strings(values: Iterable[object | None]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return out


def resolve_run_tag(
    explicit: object | None = None,
    *,
    env_var: str = "PIPELINE_RUN_TAG",
    fallback_candidates: Iterable[object | None] | None = None,
    require_explicit: bool = False,
    allow_untracked: bool = False,
) -> str:
    candidates = _candidate_strings(
        [explicit, os.environ.get(env_var, ""), *(fallback_candidates or ())]
    )
    for candidate in candidates:
        if candidate.lower() not in {"untracked", "unknown", "none"}:
            return candidate
    if allow_untracked:
        return "untracked"
    if require_explicit:
        raise MissingRunTagError(f"Missing official run_tag. Provide --run-tag or set {env_var}.")
    return "untracked"


def build_artifact_metadata(
    *,
    schema_version: str,
    run_tag: object | None = None,
    fallback_candidates: Iterable[object | None] | None = None,
    require_explicit: bool = False,
    allow_untracked: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "generated_at_utc": utc_now_iso(),
        "run_tag": resolve_run_tag(
            run_tag,
            fallback_candidates=fallback_candidates,
            require_explicit=require_explicit,
            allow_untracked=allow_untracked,
        ),
    }
    if extra:
        payload.update(extra)
    return payload
