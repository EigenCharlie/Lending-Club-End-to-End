"""Tests for scripts/cleanup_workspace_artifacts.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import cleanup_workspace_artifacts as cleanup_mod


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_layout(tmp_path: Path) -> None:
    # Kept run
    keep_run = tmp_path / "reports" / "run_logs" / "2026-03-04-C-core-balanced-cert2"
    keep_run.mkdir(parents=True, exist_ok=True)
    _touch(
        keep_run / "heartbeat.json", json.dumps({"state": "running", "orchestrator_pid": 999999})
    )

    # Purged stale run
    stale_run = tmp_path / "reports" / "run_logs" / "stale-run"
    stale_run.mkdir(parents=True, exist_ok=True)
    _touch(
        stale_run / "heartbeat.json", json.dumps({"state": "running", "orchestrator_pid": 999999})
    )

    # Kept comparison
    keep_cmp = tmp_path / "reports" / "run_comparisons" / "2026-03-04-C-core-balanced-cert2"
    keep_cmp.mkdir(parents=True, exist_ok=True)
    _touch(keep_cmp / "comparison.json", "{}")

    # Purged comparison
    purge_cmp = tmp_path / "reports" / "run_comparisons" / "obsolete-run"
    purge_cmp.mkdir(parents=True, exist_ok=True)
    _touch(purge_cmp / "comparison.json", "{}")

    # Loose files to be removed
    _touch(tmp_path / "reports" / "run_logs" / "scratch.log", "log")
    _touch(tmp_path / "reports" / "run_comparisons" / "legacy.md", "legacy")

    # Temporary files
    _touch(tmp_path / "models" / "ab_tmp_sweep.json", "{}")
    _touch(tmp_path / "data" / "processed" / "tmp_ab_policy_sweep.parquet", "tmp")


def _monkeypatch_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cleanup_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cleanup_mod, "RUN_LOGS", tmp_path / "reports" / "run_logs")
    monkeypatch.setattr(cleanup_mod, "RUN_COMPARISONS", tmp_path / "reports" / "run_comparisons")
    monkeypatch.setattr(cleanup_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cleanup_mod, "DATA_PROCESSED", tmp_path / "data" / "processed")
    monkeypatch.setattr(cleanup_mod, "MLRUNS_DIR", tmp_path / "reports" / "mlruns")
    monkeypatch.setattr(cleanup_mod, "ARCHIVE_DIR", tmp_path / "reports" / "archive")


def test_cleanup_plan_detects_stale_runs_and_tmp_files(tmp_path, monkeypatch) -> None:
    _prepare_layout(tmp_path)
    _monkeypatch_paths(monkeypatch, tmp_path)

    plan = cleanup_mod.build_cleanup_plan("core_closure_6")
    assert any(p.name == "stale-run" for p in plan.run_logs_to_delete)
    assert any(p.name == "obsolete-run" for p in plan.run_comparisons_to_delete)
    assert any(p.name == "scratch.log" for p in plan.run_log_files_to_delete)
    assert any(p.name == "legacy.md" for p in plan.run_comparison_files_to_delete)
    assert "2026-03-04-C-core-balanced-cert2" in plan.stale_running_runs
    assert "stale-run" in plan.stale_running_runs
    assert any(p.name == "ab_tmp_sweep.json" for p in plan.tmp_files_to_delete)


def test_cleanup_apply_deletes_targets(tmp_path, monkeypatch) -> None:
    _prepare_layout(tmp_path)
    _monkeypatch_paths(monkeypatch, tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["cleanup_workspace_artifacts.py", "--retention-profile", "core_closure_6", "--apply"],
    )
    cleanup_mod.main()

    assert not (tmp_path / "reports" / "run_logs" / "stale-run").exists()
    assert not (tmp_path / "reports" / "run_comparisons" / "obsolete-run").exists()
    assert not (tmp_path / "reports" / "run_logs" / "scratch.log").exists()
    assert not (tmp_path / "reports" / "run_comparisons" / "legacy.md").exists()
    assert not (tmp_path / "models" / "ab_tmp_sweep.json").exists()
    keep_hb = (
        tmp_path / "reports" / "run_logs" / "2026-03-04-C-core-balanced-cert2" / "heartbeat.json"
    )
    assert keep_hb.exists()
    hb = json.loads(keep_hb.read_text(encoding="utf-8"))
    assert hb["state"] == "stale"
