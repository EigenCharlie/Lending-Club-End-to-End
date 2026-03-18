"""Run complementary insight-factory workloads from canonical artifacts.

`insights_factory` never promotes a champion and is intended to consume the
current canonical baseline as upstream evidence. The `canonical` profile keeps
to lightweight evidence generation, while `research` adds heavier notebooks and
GPU-centric exploratory workloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUN_LOGS = REPORTS / "run_logs"
PRIMARY_BASELINE = ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
LEGACY_BASELINE = ROOT / "configs" / "baselines" / "core_official_baseline.json"
SCHEMA_VERSION = "2026-03-12.1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_upstream_canonical_run_tag(explicit: str | None) -> str:
    value = str(explicit or "").strip()
    if value:
        return value
    for path in (PRIMARY_BASELINE, LEGACY_BASELINE):
        payload = _load_json(path)
        run_tag = str(payload.get("official_run_tag", "")).strip()
        if run_tag:
            return run_tag
    raise FileNotFoundError(
        "No canonical baseline registry found. Expected "
        "`configs/baselines/canonical_operational_baseline.json` or legacy fallback."
    )


def _build_profile_commands(profile: str, run_tag: str) -> list[str]:
    if profile == "canonical":
        return [
            "uv run python -u scripts/export_conformal_method_registry.py",
            "uv run python -u scripts/benchmark_pd_set_prediction.py",
            "uv run python -u scripts/analyze_pd_rare_event_calibration.py",
            "uv run python -u scripts/run_paper_notebook_suite.py --output-dir reports/notebook_exec",
            "uv run python -u scripts/extract_notebook_images.py",
            "uv run python -u scripts/export_storytelling_snapshot.py",
        ]
    return [
        "uv run python -u scripts/export_conformal_method_registry.py",
        "uv run python -u scripts/benchmark_pd_set_prediction.py",
        "uv run python -u scripts/analyze_pd_rare_event_calibration.py",
        f"uv run python -u scripts/run_spo_comparison.py --run-tag {run_tag}-spo",
        "uv run python -u scripts/run_spo_real.py --n-items 100 --budget 30 --n-train 800 --n-test 200 --epochs 50 --seeds 5",
        f"uv run python -u scripts/run_alpha_sweep.py --mondrian --run-tag {run_tag}-alpha-mondrian",
        f"uv run python -u scripts/run_alpha_sweep.py --both --run-tag {run_tag}-alpha-both",
        f"uv run python -u scripts/run_uncertainty_baselines.py --run-tag {run_tag}-unc-baselines",
        "uv run python -u scripts/run_cqr_comparison.py",
        "uv run python -u scripts/run_competing_risks.py",
        "uv run python -u scripts/run_cif_ecl_impact.py",
        "uv run python -u scripts/run_sicr_conformal.py",
        "uv run python -u scripts/run_bma_comparison.py",
        "uv run python -u scripts/run_ts_ecl_intervals.py",
        "uv run python -u scripts/run_stage_misclassification_cost.py",
        "uv run python -u scripts/run_paper_notebook_suite.py --output-dir reports/notebook_exec",
        "uv run python -u scripts/run_all_notebooks.py --execute-all --include-side-projects --timeout 3600 --inplace false --output-dir reports/notebook_exec",
        "uv run python -u scripts/extract_notebook_images.py",
        f"uv run python -u scripts/run_pd_rapids_benchmark.py --run-tag {run_tag}-pd-rapids",
        f"uv run python -u scripts/run_rapids_insight_factory.py --run-tag {run_tag}-rapids",
        "uv run python -u scripts/export_storytelling_snapshot.py",
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the insights factory pipeline.")
    parser.add_argument(
        "--run-tag", default=f"insights-{datetime.now(UTC).strftime('%Y-%m-%d-%H%M%S')}"
    )
    parser.add_argument("--profile", choices=["canonical", "research"], default="canonical")
    parser.add_argument("--upstream-canonical-run-tag", default=None)
    args = parser.parse_args()

    run_tag = str(args.run_tag).strip()
    profile = str(args.profile).strip()
    upstream = _resolve_upstream_canonical_run_tag(args.upstream_canonical_run_tag)
    commands = _build_profile_commands(profile, run_tag)

    run_dir = RUN_LOGS / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "pipeline_family": "insights_factory",
        "pipeline_profile": f"insights_factory_{profile}",
        "artifact_scope": "insight",
        "promotion_state": "insight_only",
        "writes_canonical_artifacts": False,
        "upstream_canonical_run_tag": upstream,
        "started_at_utc": _utc_now(),
        "commands": commands,
        "python": sys.executable,
    }
    _write_json(run_dir / "run_info.json", run_info)

    results: list[dict[str, object]] = []
    env = {
        **os.environ,
        "PIPELINE_RUN_TAG": run_tag,
        "PIPELINE_FAMILY": "insights_factory",
        "PIPELINE_PROFILE": f"insights_factory_{profile}",
        "PIPELINE_ARTIFACT_SCOPE": "insight",
        "PIPELINE_PROMOTION_STATE": "insight_only",
        "WRITES_CANONICAL_ARTIFACTS": "0",
        "UPSTREAM_CANONICAL_RUN_TAG": upstream,
    }
    final_exit_code = 0
    for idx, command in enumerate(commands, start=1):
        log_path = run_dir / f"step_{idx:02d}.log"
        started_at = _utc_now()
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=ROOT,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        result = {
            "step_index": idx,
            "command": command,
            "log_path": str(log_path.relative_to(ROOT)),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "exit_code": int(proc.returncode),
        }
        results.append(result)
        if proc.returncode != 0:
            final_exit_code = int(proc.returncode)
            break

    run_summary = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "pipeline_family": "insights_factory",
        "pipeline_profile": f"insights_factory_{profile}",
        "artifact_scope": "insight",
        "promotion_state": "insight_only",
        "writes_canonical_artifacts": False,
        "upstream_canonical_run_tag": upstream,
        "ended_at_utc": _utc_now(),
        "final_exit_code": final_exit_code,
        "steps": results,
    }
    _write_json(run_dir / "run_summary.json", run_summary)
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
