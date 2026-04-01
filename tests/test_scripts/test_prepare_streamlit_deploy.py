"""Contract tests for scripts/prepare_streamlit_deploy.py."""

from __future__ import annotations

import re
from pathlib import Path

from scripts import prepare_streamlit_deploy as deploy_mod

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = PROJECT_ROOT / "streamlit_app" / "pages"

PARQUET_LOADER_PATTERN = re.compile(r'(?:try_load_parquet|load_parquet)\("([^"]+)"')
JSON_LOADER_PATTERN = re.compile(
    r'(?:try_load_json|load_json)\("([^"]+)"(?:,\s*directory\s*=\s*"([^"]+)")?'
)


def _artifact_contract_paths() -> set[str]:
    return set(deploy_mod.REQUIRED_FILES) | set(deploy_mod.OPTIONAL_FILES)


def test_prepare_streamlit_deploy_includes_reports_dirs() -> None:
    assert "reports/dvc" in deploy_mod.REQUIRED_DIRS
    assert "reports/gpu_benchmark" in deploy_mod.REQUIRED_DIRS
    assert (
        "reports/gpu_replay/2026-03-09-official-gpu-replay-rapids-final" in deploy_mod.REQUIRED_DIRS
    )
    assert "reports/gpu_insights/2026-03-10-rapids-insight-v4" in deploy_mod.REQUIRED_DIRS


def test_prepare_streamlit_deploy_includes_baseline_registry() -> None:
    assert "configs/baselines/canonical_operational_baseline.json" in deploy_mod.REQUIRED_FILES
    assert "configs/baselines/core_official_baseline.json" in deploy_mod.REQUIRED_FILES
    assert "docs/backlog-papers-unified.md" in deploy_mod.REQUIRED_FILES


def test_discover_release_governance_comparison_tags_collects_current_and_official(
    tmp_path,
) -> None:
    repo = tmp_path
    (repo / "data/processed").mkdir(parents=True, exist_ok=True)
    (repo / "configs/baselines").mkdir(parents=True, exist_ok=True)

    (repo / "data/processed/pipeline_summary.json").write_text(
        '{"run_tag":"run-current","official_baseline_run_tag":"run-official"}',
        encoding="utf-8",
    )
    (repo / "configs/baselines/core_official_baseline.json").write_text(
        '{"official_run_tag":"run-official"}',
        encoding="utf-8",
    )
    (repo / "configs/baselines/canonical_operational_baseline.json").write_text(
        '{"official_run_tag":"run-official"}',
        encoding="utf-8",
    )

    tags = deploy_mod._discover_release_governance_comparison_tags(repo)
    assert tags == ["run-current", "run-official"]


def test_discover_release_governance_run_log_files_matches_tags(tmp_path) -> None:
    repo = tmp_path
    (repo / "data/processed").mkdir(parents=True, exist_ok=True)
    (repo / "configs/baselines").mkdir(parents=True, exist_ok=True)

    (repo / "data/processed/pipeline_summary.json").write_text(
        '{"run_tag":"run-current","official_baseline_run_tag":"run-official"}',
        encoding="utf-8",
    )
    (repo / "configs/baselines/core_official_baseline.json").write_text(
        '{"official_run_tag":"run-official"}',
        encoding="utf-8",
    )
    (repo / "configs/baselines/canonical_operational_baseline.json").write_text(
        '{"official_run_tag":"run-official"}',
        encoding="utf-8",
    )

    rel_files = deploy_mod._discover_release_governance_run_log_files(repo)
    assert rel_files == [
        "reports/run_logs/run-current/master.log",
        "reports/run_logs/run-official/master.log",
    ]


def test_prepare_streamlit_deploy_covers_streamlit_loader_artifacts() -> None:
    contract_paths = _artifact_contract_paths()
    missing_contract_entries: list[str] = []

    for page_path in sorted(PAGES_DIR.glob("*.py")):
        text = page_path.read_text(encoding="utf-8")

        for match in PARQUET_LOADER_PATTERN.finditer(text):
            rel = f"data/processed/{match.group(1)}.parquet"
            if rel not in contract_paths:
                missing_contract_entries.append(f"{page_path.name}:{rel}")

        for match in JSON_LOADER_PATTERN.finditer(text):
            directory = "models" if match.group(2) == "models" else "data/processed"
            rel = f"{directory}/{match.group(1)}.json"
            if rel not in contract_paths:
                missing_contract_entries.append(f"{page_path.name}:{rel}")

    assert not missing_contract_entries, (
        "Missing artifact contract entries in prepare_streamlit_deploy.py:\n"
        + "\n".join(sorted(missing_contract_entries))
    )
