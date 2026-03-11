from __future__ import annotations

from scripts import git_dirty_guard as guard


def test_parse_porcelain_line_handles_modified_and_rename() -> None:
    assert guard._parse_porcelain_line(" M foo/bar.txt") == "foo/bar.txt"
    assert guard._parse_porcelain_line("R  old/name.txt -> new/name.txt") == "new/name.txt"


def test_split_dirty_paths_respects_allowlist() -> None:
    dirty = [
        "configs/baselines/core_official_baseline.json",
        "models/governance_status.json",
        "reports/gpu_benchmark/cuml_benchmark.csv",
        "src/models/pd_model.py",
        "README.md",
    ]
    allowed, blocked = guard.split_dirty_paths(dirty)

    assert "configs/baselines/core_official_baseline.json" in allowed
    assert "models/governance_status.json" in allowed
    assert "reports/gpu_benchmark/cuml_benchmark.csv" in allowed
    assert "src/models/pd_model.py" in blocked
    assert "README.md" in blocked
