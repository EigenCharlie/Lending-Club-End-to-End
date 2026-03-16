"""Tests for scripts/finalize_paper_grade_run.py."""

from __future__ import annotations

import json

from scripts import finalize_paper_grade_run as final_mod


def test_finalize_paper_grade_run_promotes_and_refreshes(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    baseline_dir = tmp_path / "configs" / "baselines"
    comparison_dir = report_dir / "run_comparisons" / "final-run"
    run_log_dir = report_dir / "run_logs" / "final-run"
    model_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    run_log_dir.mkdir(parents=True)

    (comparison_dir / "comparison.json").write_text(
        json.dumps({"overall_pass": True}),
        encoding="utf-8",
    )
    (model_dir / "paper_grade_protocol_status.json").write_text(
        json.dumps(
            {
                "run_tag": "final-run",
                "final_checks": {
                    "pd_conformal": True,
                    "time_series": True,
                    "causal_cate": True,
                    "ab_evidence": True,
                    "ab_no_regression": True,
                    "governance": True,
                    "protocol_frozen": True,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in final_mod.REQUIRED_RUN_TAG_ARTIFACTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.name == "paper_grade_protocol_status.json":
            continue
        target.write_text(json.dumps({"run_tag": "final-run"}), encoding="utf-8")
    for path in (
        baseline_dir / "canonical_operational_baseline.json",
        baseline_dir / "core_official_baseline.json",
    ):
        path.write_text(json.dumps({"official_run_tag": "baseline-old"}), encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd, cwd=None, env=None, check=None):
        calls.append(list(cmd))
        if "freeze_core_baseline.py" in cmd[1]:
            for path in (
                baseline_dir / "canonical_operational_baseline.json",
                baseline_dir / "core_official_baseline.json",
            ):
                path.write_text(json.dumps({"official_run_tag": "final-run"}), encoding="utf-8")
        return None

    monkeypatch.setattr(final_mod, "ROOT", tmp_path)
    monkeypatch.setattr(final_mod, "MODELS", model_dir)
    monkeypatch.setattr(final_mod, "REPORTS", report_dir)
    monkeypatch.setattr(final_mod, "BASELINES", baseline_dir)
    monkeypatch.setattr(
        final_mod,
        "REQUIRED_RUN_TAG_ARTIFACTS",
        {name: tmp_path / name for name in final_mod.REQUIRED_RUN_TAG_ARTIFACTS},
    )
    monkeypatch.setattr(
        final_mod,
        "REGISTRY_PATHS",
        [
            baseline_dir / "canonical_operational_baseline.json",
            baseline_dir / "core_official_baseline.json",
        ],
    )
    monkeypatch.setattr(final_mod.subprocess, "run", _fake_run)

    assert final_mod.main(["--run-tag", "final-run"]) == 0

    payload = json.loads(
        (run_log_dir / "paper_grade_finalization.json").read_text(encoding="utf-8")
    )
    assert payload["promoted"] is True
    assert payload["rebuild_verified"] is True
    assert payload["baseline_registry_official_run_tag"] == "final-run"
    assert any("freeze_core_baseline.py" in cmd[1] for cmd in calls)
    assert any("run_canonical_rebuild.py" in cmd[1] for cmd in calls)


def test_finalize_paper_grade_run_refuses_promotion_with_untracked_artifacts(
    tmp_path, monkeypatch
) -> None:
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    comparison_dir = report_dir / "run_comparisons" / "final-run"
    run_log_dir = report_dir / "run_logs" / "final-run"
    model_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    run_log_dir.mkdir(parents=True)

    (comparison_dir / "comparison.json").write_text(
        json.dumps({"overall_pass": True}),
        encoding="utf-8",
    )
    (model_dir / "paper_grade_protocol_status.json").write_text(
        json.dumps(
            {
                "run_tag": "final-run",
                "final_checks": {
                    "pd_conformal": True,
                    "time_series": True,
                    "causal_cate": True,
                    "ab_evidence": True,
                    "ab_no_regression": True,
                    "governance": True,
                    "protocol_frozen": True,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in final_mod.REQUIRED_RUN_TAG_ARTIFACTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.name == "paper_grade_protocol_status.json":
            continue
        run_tag = "untracked" if target.name == "time_series_status.json" else "final-run"
        target.write_text(json.dumps({"run_tag": run_tag}), encoding="utf-8")

    monkeypatch.setattr(final_mod, "ROOT", tmp_path)
    monkeypatch.setattr(final_mod, "MODELS", model_dir)
    monkeypatch.setattr(final_mod, "REPORTS", report_dir)
    monkeypatch.setattr(
        final_mod,
        "REQUIRED_RUN_TAG_ARTIFACTS",
        {name: tmp_path / name for name in final_mod.REQUIRED_RUN_TAG_ARTIFACTS},
    )

    assert final_mod.main(["--run-tag", "final-run", "--skip-rebuild"]) == 1

    payload = json.loads(
        (run_log_dir / "paper_grade_finalization.json").read_text(encoding="utf-8")
    )
    assert payload["promoted"] is False
    assert payload["failure_reason"] == "run_tag_incoherent_artifacts"
