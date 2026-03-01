"""Execute notebooks in batch with reproducible outputs and manifest summary.

Usage:
    uv run python scripts/run_all_notebooks.py --execute-all --timeout 3600 \
      --inplace false --output-dir reports/notebook_exec
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


@dataclass
class NotebookRunResult:
    notebook: str
    source_path: str
    output_path: str
    success: bool
    duration_seconds: float
    error: str | None = None


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _discover_notebooks(execute_all: bool, include_side_projects: bool) -> list[Path]:
    paths: list[Path] = []
    if execute_all:
        paths.extend(sorted(NOTEBOOKS_DIR.glob("*.ipynb")))
    if include_side_projects:
        paths.extend(sorted((NOTEBOOKS_DIR / "side_projects").glob("*.ipynb")))
    # Stable unique order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _execute_notebook(
    nb_path: Path,
    *,
    timeout: int,
    inplace: bool,
    output_dir: Path | None,
) -> NotebookRunResult:
    started = time.perf_counter()
    rel = nb_path.relative_to(PROJECT_ROOT)
    target_path = nb_path
    if not inplace:
        if output_dir is None:
            raise ValueError("output_dir is required when inplace=False")
        target_path = output_dir / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")

    try:
        client.execute(cwd=str(nb_path.parent))
        nbformat.write(nb, target_path)
        return NotebookRunResult(
            notebook=nb_path.name,
            source_path=str(rel),
            output_path=str(target_path.relative_to(PROJECT_ROOT)),
            success=True,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
    except CellExecutionError as exc:
        nbformat.write(nb, target_path)
        return NotebookRunResult(
            notebook=nb_path.name,
            source_path=str(rel),
            output_path=str(target_path.relative_to(PROJECT_ROOT)),
            success=False,
            duration_seconds=round(time.perf_counter() - started, 3),
            error=str(exc).splitlines()[-1][:500],
        )
    except Exception as exc:  # pragma: no cover - safety net for runtime environments
        return NotebookRunResult(
            notebook=nb_path.name,
            source_path=str(rel),
            output_path=str(target_path.relative_to(PROJECT_ROOT)),
            success=False,
            duration_seconds=round(time.perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute notebooks in batch.")
    parser.add_argument(
        "--execute-all", action="store_true", help="Run all notebooks in notebooks/"
    )
    parser.add_argument(
        "--include-side-projects",
        action="store_true",
        help="Include notebooks under notebooks/side_projects/",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--inplace", default="false", help="Write outputs back to source notebooks")
    parser.add_argument("--output-dir", default="reports/notebook_exec")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    inplace = _parse_bool(args.inplace)
    output_dir = None if inplace else (PROJECT_ROOT / args.output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    notebooks = _discover_notebooks(
        execute_all=args.execute_all,
        include_side_projects=args.include_side_projects,
    )
    if not notebooks:
        raise SystemExit(
            "No notebooks selected. Use --execute-all and optionally --include-side-projects."
        )

    results: list[NotebookRunResult] = []
    for nb_path in notebooks:
        print(f"[nb] Executing {nb_path.relative_to(PROJECT_ROOT)}")
        res = _execute_notebook(
            nb_path,
            timeout=int(args.timeout),
            inplace=inplace,
            output_dir=output_dir,
        )
        results.append(res)
        status = "OK" if res.success else "FAIL"
        print(f"  -> {status} ({res.duration_seconds:.1f}s)")
        if (not res.success) and args.stop_on_error:
            break

    manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": "2026-02-26.1",
        "inplace": bool(inplace),
        "timeout_seconds": int(args.timeout),
        "notebooks_total": len(notebooks),
        "executed": len(results),
        "success_count": int(sum(1 for r in results if r.success)),
        "failure_count": int(sum(1 for r in results if not r.success)),
        "results": [asdict(r) for r in results],
    }

    manifest_dir = (
        output_dir if output_dir is not None else (PROJECT_ROOT / "reports" / "notebook_exec")
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[nb] Manifest saved: {manifest_path.relative_to(PROJECT_ROOT)}")

    if manifest["failure_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
