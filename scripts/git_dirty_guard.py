"""Classify dirty git paths for official-launch guardrails."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Generated outputs that should not block an official launch when they are the
# only dirty paths left in the working tree.
ALLOWED_PATTERNS = (
    "configs/baselines/core_official_baseline.json",
    "models/*_status.json",
    "models/*_policy.json",
    "models/*_registry.json",
    "models/*_report.json",
    "reports/gpu_benchmark/",
    "reports/gpu_benchmark/**",
    "reports/paper_material/**/*",
    "reports/pd_candidate_metrics/",
    "reports/pd_candidate_metrics/**",
    "reports/gpu_replay/",
    "reports/gpu_replay/**",
)


def _normalize_path(value: str) -> str:
    return value.strip().replace("\\", "/")


def _parse_porcelain_line(line: str) -> str | None:
    text = line.rstrip("\n")
    if not text:
        return None
    if len(text) < 4:
        return None
    payload = text[3:]
    if " -> " in payload:
        payload = payload.split(" -> ", 1)[1]
    return _normalize_path(payload)


def list_dirty_paths(repo_root: Path = REPO_ROOT) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git status --porcelain failed")
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        path = _parse_porcelain_line(line)
        if path:
            paths.append(path)
    return paths


def is_allowed_dirty_path(path: str) -> bool:
    candidate = _normalize_path(path)
    return any(fnmatch.fnmatch(candidate, pattern) for pattern in ALLOWED_PATTERNS)


def split_dirty_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    blocked: list[str] = []
    for path in paths:
        if is_allowed_dirty_path(path):
            allowed.append(path)
        else:
            blocked.append(path)
    return allowed, blocked


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify dirty paths for official launch policy.")
    parser.add_argument(
        "--mode",
        choices=("blocked-only", "all"),
        default="blocked-only",
        help="Whether to print only blocked paths or all dirty paths with labels.",
    )
    args = parser.parse_args()

    dirty_paths = list_dirty_paths()
    allowed, blocked = split_dirty_paths(dirty_paths)

    if args.mode == "all":
        for path in allowed:
            print(f"ALLOWED\t{path}")
        for path in blocked:
            print(f"BLOCKED\t{path}")
    else:
        for path in blocked:
            print(path)

    return 0 if not blocked else 1


if __name__ == "__main__":
    sys.exit(main())
