#!/usr/bin/env python3
"""Cleanup workspace run artifacts with explicit retention profiles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_LOGS = ROOT / "reports" / "run_logs"
RUN_COMPARISONS = ROOT / "reports" / "run_comparisons"
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
MLRUNS_DIR = ROOT / "reports" / "mlruns"
ARCHIVE_DIR = ROOT / "reports" / "archive"


RETENTION_PROFILES = {
    "core_closure_6": {
        "run_logs_keep": {
            "champion-2026-03-12-mega-definitive",
            "2026-03-11-C-official-selector-v3-freeze",
            "2026-03-04-C-core-balanced-cert2",
            "2026-03-04-C-core-balanced-e2e-cert1",
            "2026-03-04-C-core-balanced-pass2",
            "2026-03-01-C-official-smart",
            "2026-03-04-C-rapids-annex-pass3",
            "2026-03-04-C-notebooks-annex-pass2",
        },
        "run_comparisons_keep": {
            "champion-2026-03-12-mega-definitive",
            "2026-03-11-C-official-selector-v3-freeze",
            "2026-03-04-C-core-balanced-cert2",
            "2026-03-04-C-core-balanced-e2e-cert1",
            "2026-03-04-C-core-balanced-pass2",
            "2026-03-03-C-core-balanced-pass1",
            "2026-03-03-C-core-balanced-ws1-manual",
            "2026-03-01-C-official-smart",
        },
    }
}


TMP_FILE_PATTERNS = [
    "models/ab_simulation_status_tmp_custom*.json",
    "models/ab_tmp_sweep.json",
    "data/processed/tmp_*ab*.parquet",
    "models/conformal_policy_status_v2.json",
    "models/fairness_audit_status_v2.json",
    "models/governance_status_v2.json",
    "data/processed/drift_monitoring_v2.parquet",
]


@dataclass
class CleanupPlan:
    run_logs_to_delete: list[Path]
    run_comparisons_to_delete: list[Path]
    run_log_files_to_delete: list[Path]
    run_comparison_files_to_delete: list[Path]
    tmp_files_to_delete: list[Path]
    stale_running_runs: list[str]


def _list_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def _list_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file()], key=lambda p: p.name)


def _stale_running_runs() -> list[str]:
    stale: list[str] = []
    for run_dir in _list_dirs(RUN_LOGS):
        hb_path = run_dir / "heartbeat.json"
        if not hb_path.exists():
            continue
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(hb.get("state", "")).strip().lower() != "running":
            continue
        pid = hb.get("orchestrator_pid")
        if not isinstance(pid, int) or pid <= 0:
            stale.append(run_dir.name)
            continue
        try:
            os.kill(pid, 0)
        except Exception:
            stale.append(run_dir.name)
    return sorted(set(stale))


def _collect_tmp_files(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(ROOT.glob(pattern))
    return sorted(set(out), key=lambda p: str(p))


def build_cleanup_plan(profile: str) -> CleanupPlan:
    cfg = RETENTION_PROFILES[profile]
    run_logs_keep = set(cfg["run_logs_keep"])
    run_comparisons_keep = set(cfg["run_comparisons_keep"])

    run_logs_to_delete = [p for p in _list_dirs(RUN_LOGS) if p.name not in run_logs_keep]
    run_log_files_to_delete = _list_files(RUN_LOGS)
    run_comparisons_to_delete = [
        p for p in _list_dirs(RUN_COMPARISONS) if p.name not in run_comparisons_keep
    ]
    run_comparison_files_to_delete = _list_files(RUN_COMPARISONS)
    tmp_files_to_delete = [p for p in _collect_tmp_files(TMP_FILE_PATTERNS) if p.exists()]
    stale_runs = _stale_running_runs()

    return CleanupPlan(
        run_logs_to_delete=run_logs_to_delete,
        run_comparisons_to_delete=run_comparisons_to_delete,
        run_log_files_to_delete=run_log_files_to_delete,
        run_comparison_files_to_delete=run_comparison_files_to_delete,
        tmp_files_to_delete=tmp_files_to_delete,
        stale_running_runs=stale_runs,
    )


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _backup_mlruns() -> Path | None:
    if not MLRUNS_DIR.exists():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    tar_path = ARCHIVE_DIR / f"mlruns_backup_{stamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(MLRUNS_DIR, arcname="mlruns")
    return tar_path


def _reconcile_stale_heartbeat(run_dir: Path) -> bool:
    hb_path = run_dir / "heartbeat.json"
    if not hb_path.exists():
        return False
    try:
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(hb.get("state", "")).strip().lower() != "running":
        return False
    hb["state"] = "stale"
    hb["stale_reconciled_at_utc"] = datetime.now(UTC).isoformat()
    hb.setdefault("stale_reason", "orchestrator_pid_dead_or_missing")
    hb_path.write_text(json.dumps(hb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cleanup workspace artifacts with retention profile."
    )
    parser.add_argument(
        "--retention-profile",
        choices=sorted(RETENTION_PROFILES.keys()),
        default="core_closure_6",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions and stale-heartbeat reconciliation.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cleanup actions without deleting files (default mode).",
    )
    parser.add_argument(
        "--purge-mlruns-local",
        action="store_true",
        help="Backup and purge reports/mlruns local store.",
    )
    args = parser.parse_args()

    plan = build_cleanup_plan(args.retention_profile)
    summary = {
        "schema_version": "2026-03-05.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "retention_profile": args.retention_profile,
        "mode": "apply" if args.apply else "dry_run",
        "stale_running_runs": plan.stale_running_runs,
        "run_logs_to_delete": [str(p.relative_to(ROOT)) for p in plan.run_logs_to_delete],
        "run_log_files_to_delete": [str(p.relative_to(ROOT)) for p in plan.run_log_files_to_delete],
        "run_comparisons_to_delete": [
            str(p.relative_to(ROOT)) for p in plan.run_comparisons_to_delete
        ],
        "run_comparison_files_to_delete": [
            str(p.relative_to(ROOT)) for p in plan.run_comparison_files_to_delete
        ],
        "tmp_files_to_delete": [str(p.relative_to(ROOT)) for p in plan.tmp_files_to_delete],
        "stale_heartbeats_reconciled": [],
        "mlruns_backup": None,
        "mlruns_purged": False,
    }

    if args.apply:
        for path in plan.run_logs_to_delete:
            _remove_path(path)
        for path in plan.run_comparisons_to_delete:
            _remove_path(path)
        for path in plan.tmp_files_to_delete:
            _remove_path(path)
        for path in plan.run_log_files_to_delete:
            _remove_path(path)
        for path in plan.run_comparison_files_to_delete:
            _remove_path(path)

        run_logs_deleted = {p.name for p in plan.run_logs_to_delete}
        reconciled: list[str] = []
        for run_name in plan.stale_running_runs:
            if run_name in run_logs_deleted:
                continue
            run_dir = RUN_LOGS / run_name
            if _reconcile_stale_heartbeat(run_dir):
                reconciled.append(run_name)
        summary["stale_heartbeats_reconciled"] = reconciled

        if args.purge_mlruns_local:
            backup_path = _backup_mlruns()
            summary["mlruns_backup"] = str(backup_path.relative_to(ROOT)) if backup_path else None
            if MLRUNS_DIR.exists():
                shutil.rmtree(MLRUNS_DIR)
                summary["mlruns_purged"] = True

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
