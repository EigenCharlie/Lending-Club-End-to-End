"""Helpers for frozen baseline replay manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = ROOT / "configs" / "baselines" / "clean_baseline_manifest.json"


def resolve_manifest_path(path_like: str | Path | None = None) -> Path:
    if path_like is None:
        return DEFAULT_MANIFEST_PATH
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_replay_manifest(path_like: str | Path | None = None) -> dict[str, Any]:
    path = resolve_manifest_path(path_like)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_replay_manifest(payload: dict[str, Any], path_like: str | Path | None = None) -> Path:
    path = resolve_manifest_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def manifest_section(
    manifest: dict[str, Any] | None,
    section: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    payload = manifest.get(section, {})
    return dict(payload) if isinstance(payload, dict) else {}


def sha256_for_path(path_like: str | Path) -> str | None:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_descriptor(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    rel = (
        str(path.relative_to(ROOT))
        if path.is_absolute() and ROOT in path.parents
        else str(path_like)
    )
    return {
        "path": rel,
        "exists": bool(path.exists()),
        "sha256": sha256_for_path(path),
    }
