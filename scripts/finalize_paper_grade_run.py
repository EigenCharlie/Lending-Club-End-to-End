"""Finalize a paper-grade final run: promote if gates pass, then confirm rebuild."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
BASELINES = ROOT / "configs" / "baselines"
SCHEMA_VERSION = "2026-03-14.1"
REQUIRED_RUN_TAG_ARTIFACTS = {
    "models/conformal_policy_status.json": MODELS / "conformal_policy_status.json",
    "models/governance_status.json": MODELS / "governance_status.json",
    "models/fairness_audit_status.json": MODELS / "fairness_audit_status.json",
    "models/time_series_status.json": MODELS / "time_series_status.json",
    "models/ab_simulation_status.json": MODELS / "ab_simulation_status.json",
    "models/paper_grade_protocol_status.json": MODELS / "paper_grade_protocol_status.json",
    "models/champion_registry.json": MODELS / "champion_registry.json",
    "models/champion_search_bundle.json": MODELS / "champion_search_bundle.json",
    "models/pd_set_prediction_status.json": MODELS / "pd_set_prediction_status.json",
    "models/pd_rare_event_calibration_status.json": MODELS
    / "pd_rare_event_calibration_status.json",
    "models/conformal_method_registry.json": MODELS / "conformal_method_registry.json",
}
REGISTRY_PATHS = [
    BASELINES / "canonical_operational_baseline.json",
    BASELINES / "core_official_baseline.json",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _report_path(run_tag: str) -> Path:
    return REPORTS / "run_logs" / run_tag / "paper_grade_finalization.json"


def _comparison_path(run_tag: str) -> Path:
    return REPORTS / "run_comparisons" / run_tag / "comparison.json"


def _check_protocol_final_checks(payload: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    final_checks = payload.get("final_checks", {}) if isinstance(payload, dict) else {}
    checks = {
        name: bool(final_checks.get(name, False))
        for name in (
            "pd_conformal",
            "time_series",
            "causal_cate",
            "ab_evidence",
            "ab_no_regression",
            "governance",
            "protocol_frozen",
        )
    }
    return bool(all(checks.values())), checks


def _check_required_run_tags(run_tag: str) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    passed = True
    for name, path in REQUIRED_RUN_TAG_ARTIFACTS.items():
        payload = _load_json(path)
        observed = str(payload.get("run_tag", "")).strip()
        row = {
            "artifact": name,
            "exists": path.exists(),
            "observed_run_tag": observed or None,
            "expected_run_tag": run_tag,
            "passed": bool(path.exists() and observed == run_tag),
        }
        rows.append(row)
        if not row["passed"]:
            passed = False
    return passed, rows


def _run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _snapshot_registries() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in REGISTRY_PATHS:
        if path.exists():
            snapshot[str(path)] = path.read_text(encoding="utf-8")
    return snapshot


def _restore_registries(snapshot: dict[str, str]) -> None:
    for path_str, content in snapshot.items():
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _refresh_paper_grade_artifacts(
    run_tag: str,
    *,
    promoted: bool,
    comparison_path: Path,
    rebuild_verified: bool,
) -> None:
    env = os.environ.copy()
    env["PIPELINE_RUN_TAG"] = run_tag
    env["PAPER_GRADE_FINAL_RUN_TAG"] = run_tag
    env["PAPER_GRADE_FINAL_RUN_PROMOTED"] = "true" if promoted else "false"
    env["PAPER_GRADE_FINAL_RUN_COMPARISON_PATH"] = str(comparison_path.relative_to(ROOT))
    env["PAPER_GRADE_FINAL_REBUILD_VERIFIED"] = "true" if rebuild_verified else "false"
    for script in (
        "scripts/generate_paper_grade_protocol.py",
        "scripts/export_storytelling_snapshot.py",
        "scripts/update_champion_registry.py",
        "scripts/build_champion_search_bundle.py",
    ):
        _run_command([sys.executable, script], env=env)


def _confirm_rebuild(run_tag: str) -> None:
    _run_command(
        [
            sys.executable,
            "scripts/run_canonical_rebuild.py",
            "--run-tag",
            f"{run_tag}-rebuild",
            "--comparison-baseline-run-tag",
            run_tag,
            "--pipeline-profile",
            "canonical_operational",
            "--sampling-profile",
            "champion64safe",
            "--no-notebooks",
            "--no-rapids",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize paper-grade final run.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="Do not execute the confirmatory canonical_rebuild step.",
    )
    args = parser.parse_args(argv)

    run_tag = str(args.run_tag).strip()
    comparison_path = _comparison_path(run_tag)
    protocol_path = MODELS / "paper_grade_protocol_status.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_tag": run_tag,
        "comparison_path": str(comparison_path.relative_to(ROOT)),
        "promotion_attempted": True,
        "promoted": False,
        "rebuild_verified": False,
    }

    if not comparison_path.exists():
        report["failure_reason"] = "missing_comparison_json"
        _write_json(_report_path(run_tag), report)
        return 1

    comparison = _load_json(comparison_path)
    protocol = _load_json(protocol_path)
    protocol_ok, final_checks = _check_protocol_final_checks(protocol)
    run_tag_ok, run_tag_rows = _check_required_run_tags(run_tag)

    report["comparison_overall_pass"] = bool(comparison.get("overall_pass", False))
    report["protocol_final_checks"] = final_checks
    report["required_run_tag_checks"] = run_tag_rows

    if not bool(comparison.get("overall_pass", False)):
        report["failure_reason"] = "comparison_overall_pass_false"
        _write_json(_report_path(run_tag), report)
        return 1
    if not protocol_ok:
        report["failure_reason"] = "protocol_final_checks_incomplete"
        _write_json(_report_path(run_tag), report)
        return 1
    if not run_tag_ok:
        report["failure_reason"] = "run_tag_incoherent_artifacts"
        _write_json(_report_path(run_tag), report)
        return 1

    registry_snapshot = _snapshot_registries()
    try:
        _run_command(
            [
                sys.executable,
                "scripts/freeze_core_baseline.py",
                "--run-tag",
                run_tag,
                "--refresh-snapshot",
                "--set-current",
                "--notes",
                "Paper-grade final heavy run promoted after final comparison gates.",
            ]
        )
        if not args.skip_rebuild:
            _confirm_rebuild(run_tag)
            report["rebuild_verified"] = True
        _refresh_paper_grade_artifacts(
            run_tag,
            promoted=True,
            comparison_path=comparison_path,
            rebuild_verified=bool(report["rebuild_verified"]),
        )
    except Exception as exc:
        _restore_registries(registry_snapshot)
        report["failure_reason"] = "promotion_or_rebuild_failed"
        report["error"] = str(exc)
        _write_json(_report_path(run_tag), report)
        raise

    report["promoted"] = True
    report["baseline_registry_official_run_tag"] = _load_json(REGISTRY_PATHS[0]).get(
        "official_run_tag"
    )
    _write_json(_report_path(run_tag), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
