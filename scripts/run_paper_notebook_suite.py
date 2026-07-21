"""Execute active local paper-support notebooks in non-destructive reference mode.

This script delegates execution to scripts/run_all_notebooks.py and keeps
canonical pipeline artifacts untouched by routing notebook writes into the
notebook execution sandbox.

Usage:
    uv run python scripts/run_paper_notebook_suite.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

NOTEBOOKS = ("11_paper2_ifrs9_e2e.ipynb",)


def build_command(repo_root: Path, *, timeout_s: int, output_dir: str) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        str(repo_root / "scripts" / "run_all_notebooks.py"),
    ]
    for notebook in NOTEBOOKS:
        cmd.extend(["--notebook", notebook])
    cmd.extend(
        [
            "--timeout",
            str(timeout_s),
            "--inplace",
            "false",
            "--output-dir",
            output_dir,
        ]
    )
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute paper notebooks safely.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", default="reports/notebook_exec")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    notebooks_dir = repo_root / "notebooks"
    missing = [name for name in NOTEBOOKS if not (notebooks_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing paper notebooks: {missing}")

    cmd = build_command(
        repo_root,
        timeout_s=int(args.timeout),
        output_dir=str(args.output_dir),
    )
    print("Running paper notebook suite in reference mode...")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print(
        "Done. Executed notebooks are under reports/notebook_exec and canonical "
        "pipeline outputs were protected from notebook writes."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
