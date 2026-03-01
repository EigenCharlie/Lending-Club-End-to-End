"""Tests for long-run orchestrator behavior."""

from __future__ import annotations

import argparse
import json

from scripts import run_long_pipeline as lp


def test_build_steps_post_core_runs_governance_before_mrm() -> None:
    steps = lp.build_steps("run-x", include_rapids=False, include_notebooks=False)
    post_core_cmd = next(cmd for name, _required, cmd in steps if name == "post_core")

    assert "generate_governance_status.py" in post_core_cmd
    assert "generate_mrm_report.py" in post_core_cmd
    assert post_core_cmd.index("generate_governance_status.py") < post_core_cmd.index(
        "generate_mrm_report.py"
    )


def test_main_optional_failure_with_stop_flag_sets_nonzero_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-stop",
            resume=False,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=True,
            sampling_profile="full",
        ),
    )
    monkeypatch.setattr(lp, "load_completed_ok", lambda *args, **kwargs: False)

    def fake_run_step(_run_tag: str, step: str, _command: str, *, required: bool) -> int:
        if required:
            return 0
        if step == "heavy_main":
            return 2
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()
    summary_path = tmp_path / "reports" / "run_logs" / "test-run-stop" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert summary["final_exit_code"] == 1
    assert summary["failed_required"] is False
    assert summary["stopped_on_optional_failure"] is True
    assert summary["failed_steps"] == ["heavy_main"]


def test_main_optional_failure_without_stop_flag_keeps_zero_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-continue",
            resume=False,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            sampling_profile="full",
        ),
    )
    monkeypatch.setattr(lp, "load_completed_ok", lambda *args, **kwargs: False)

    def fake_run_step(_run_tag: str, step: str, _command: str, *, required: bool) -> int:
        if required:
            return 0
        if step == "heavy_main":
            return 2
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()
    summary_path = tmp_path / "reports" / "run_logs" / "test-run-continue" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["final_exit_code"] == 0
    assert summary["stopped_on_optional_failure"] is False
    assert summary["failed_steps"] == ["heavy_main"]
