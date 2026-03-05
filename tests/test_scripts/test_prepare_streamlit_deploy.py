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
