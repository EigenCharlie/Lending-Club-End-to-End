"""Helpers for resolving the current official baseline registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_BASELINE = ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
LEGACY_BASELINE = ROOT / "configs" / "baselines" / "core_official_baseline.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_official_baseline_registry() -> dict[str, Any]:
    for path in (PRIMARY_BASELINE, LEGACY_BASELINE):
        payload = _load_json(path)
        if payload:
            payload = dict(payload)
            payload["_path"] = str(path)
            return payload
    return {}


def resolve_official_baseline_run_tag(default: str | None = None) -> str | None:
    payload = load_official_baseline_registry()
    value = str(payload.get("official_run_tag", "")).strip()
    if value:
        return value
    return default
