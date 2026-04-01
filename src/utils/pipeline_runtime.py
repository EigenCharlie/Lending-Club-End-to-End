"""Runtime status helpers for long-running thesis pipeline phases."""

from __future__ import annotations

import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(target)
    return target


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
    )


def atomic_write_pickle(path: str | Path, obj: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return target


def atomic_write_parquet(df: Any, path: str | Path, *, index: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    df.to_parquet(tmp, index=index)
    tmp.replace(target)
    return target


def runtime_status_path(stage_name: str, *, base_dir: str | Path = "models") -> Path:
    return Path(base_dir) / f"{stage_name}_runtime_status.json"


def runtime_checkpoint_dir(stage_name: str, *, base_dir: str | Path = "models") -> Path:
    return Path(base_dir) / f"{stage_name}_runtime_checkpoints"


def runtime_last_artifact_path(stage_name: str, *, base_dir: str | Path = "models") -> Path:
    return Path(base_dir) / f"{stage_name}_last_valid_artifact.json"


def write_runtime_status(
    stage_name: str,
    *,
    phase: str,
    state: str,
    run_tag: str | None = None,
    status_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "stage_name": stage_name,
        "phase": phase,
        "state": state,
        "updated_at_utc": utc_now_iso(),
    }
    if run_tag is not None:
        payload["run_tag"] = str(run_tag)
    if extra:
        payload.update(extra)
    target = Path(status_path) if status_path is not None else runtime_status_path(stage_name)
    return atomic_write_json(target, payload)


def write_runtime_checkpoint(
    stage_name: str,
    checkpoint_name: str,
    payload: dict[str, Any],
    *,
    checkpoint_dir: str | Path | None = None,
) -> Path:
    target_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else runtime_checkpoint_dir(stage_name)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    enriched = {
        "stage_name": stage_name,
        "checkpoint_name": checkpoint_name,
        "updated_at_utc": utc_now_iso(),
        **payload,
    }
    return atomic_write_json(target_dir / f"{checkpoint_name}.json", enriched)


def write_last_valid_artifact(
    stage_name: str,
    *,
    artifact_key: str,
    artifact_path: str | Path,
    run_tag: str | None = None,
    manifest_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "stage_name": stage_name,
        "artifact_key": artifact_key,
        "artifact_path": str(artifact_path),
        "validated_at_utc": utc_now_iso(),
    }
    if run_tag is not None:
        payload["run_tag"] = str(run_tag)
    if extra:
        payload.update(extra)
    target = (
        Path(manifest_path) if manifest_path is not None else runtime_last_artifact_path(stage_name)
    )
    return atomic_write_json(target, payload)


def load_runtime_status(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
