#!/usr/bin/env python3
"""Freeze an official core baseline run tag for future core/offical launches.

Usage:
    uv run python scripts/freeze_core_baseline.py \
      --run-tag 2026-03-04-C-core-balanced-cert2 \
      --refresh-snapshot \
      --set-current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "baselines" / "core_official_baseline.json"
PRIMARY_REGISTRY_PATH = ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
RUN_COMPARISONS = ROOT / "reports" / "run_comparisons"
SCHEMA_VERSION = "2026-03-05.1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path_for(run_tag: str) -> Path:
    return RUN_COMPARISONS / run_tag / "baseline_snapshot.json"


def _refresh_snapshot(run_tag: str) -> None:
    cmd = [sys.executable, "scripts/run_comparison.py", "snapshot", "--run-tag", run_tag]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "official_run_tag": "",
            "baseline_snapshot_path": "",
            "baseline_snapshot_sha256": "",
            "baseline_snapshot_mtime_utc": "",
            "frozen_at_utc": "",
            "notes": "",
            "history": [],
        }
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "official_run_tag": "",
            "baseline_snapshot_path": "",
            "baseline_snapshot_sha256": "",
            "baseline_snapshot_mtime_utc": "",
            "frozen_at_utc": "",
            "notes": "",
            "history": [],
        }


def _write_registry(payload: dict[str, Any]) -> None:
    for path in (REGISTRY_PATH, PRIMARY_REGISTRY_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _snapshot_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze canonical operational baseline run tag.")
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Run tag to freeze as canonical operational baseline.",
    )
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="Refresh reports/run_comparisons/<run_tag>/baseline_snapshot.json before freezing.",
    )
    parser.add_argument(
        "--set-current",
        action="store_true",
        help=(
            "Set this run as current official baseline in "
            "configs/baselines/canonical_operational_baseline.json "
            "and dual-write the legacy registry."
        ),
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional note stored in registry history.",
    )
    args = parser.parse_args()

    run_tag = str(args.run_tag).strip()
    if not run_tag:
        raise ValueError("--run-tag must be non-empty")

    if args.refresh_snapshot:
        _refresh_snapshot(run_tag)

    snapshot_path = _snapshot_path_for(run_tag)
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Baseline snapshot not found for run_tag '{run_tag}': {snapshot_path}"
        )

    snapshot_rel = str(snapshot_path.relative_to(ROOT))
    snapshot_sha = _sha256(snapshot_path)
    snapshot_mtime = _snapshot_mtime_iso(snapshot_path)
    now = _utc_now()

    registry = _load_registry()
    history = registry.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "frozen_at_utc": now,
            "run_tag": run_tag,
            "baseline_snapshot_path": snapshot_rel,
            "baseline_snapshot_sha256": snapshot_sha,
            "baseline_snapshot_mtime_utc": snapshot_mtime,
            "notes": str(args.notes or "").strip(),
        }
    )
    registry["history"] = history
    registry["schema_version"] = SCHEMA_VERSION
    registry["generated_at_utc"] = now

    if args.set_current:
        registry["official_run_tag"] = run_tag
        registry["baseline_snapshot_path"] = snapshot_rel
        registry["baseline_snapshot_sha256"] = snapshot_sha
        registry["baseline_snapshot_mtime_utc"] = snapshot_mtime
        registry["frozen_at_utc"] = now
        registry["notes"] = str(args.notes or "").strip()

    _write_registry(registry)

    print(
        "[freeze] registries: "
        f"{REGISTRY_PATH.relative_to(ROOT)}, {PRIMARY_REGISTRY_PATH.relative_to(ROOT)}"
    )
    print(f"[freeze] run_tag: {run_tag}")
    print(f"[freeze] snapshot: {snapshot_rel}")
    print(f"[freeze] sha256: {snapshot_sha}")
    print(f"[freeze] set_current: {bool(args.set_current)}")


if __name__ == "__main__":
    main()
