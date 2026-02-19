"""Execute paper-support notebooks and refresh paper material outputs.

Usage:
    uv run python scripts/run_paper_notebook_suite.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "10_paper1_cp_robust_opt.ipynb",
    "11_paper2_ifrs9_e2e.ipynb",
    "12_paper3_mondrian.ipynb",
]


def execute_notebook(path: Path, cwd: Path, timeout_s: int = 900) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout_s, kernel_name="python3")
    client.execute(cwd=str(cwd))
    nbformat.write(nb, path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebooks_dir = repo_root / "notebooks"

    for name in NOTEBOOKS:
        nb_path = notebooks_dir / name
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")

    print("Running paper notebook suite...")
    for name in NOTEBOOKS:
        nb_path = notebooks_dir / name
        print(f"- Executing {nb_path.name}")
        execute_notebook(nb_path, notebooks_dir)

    print("Done. Outputs refreshed under reports/paper_material/.")


if __name__ == "__main__":
    main()
