"""Execute notebooks in batch with reproducible outputs and manifest summary.

Usage:
    uv run python scripts/run_all_notebooks.py --execute-all --timeout 3600 \
      --inplace false --output-dir reports/notebook_exec
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_SANDBOX_ROOT = PROJECT_ROOT / "reports" / "notebook_exec" / "generated"


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


def _resolve_selected_notebook(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    direct = NOTEBOOKS_DIR / candidate
    if direct.exists():
        return direct
    from_repo_root = PROJECT_ROOT / candidate
    if from_repo_root.exists():
        return from_repo_root
    return direct


def _discover_notebooks(
    *,
    execute_all: bool,
    include_side_projects: bool,
    selected_notebooks: list[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    selected = [s for s in (selected_notebooks or []) if str(s).strip()]

    if selected:
        paths.extend(_resolve_selected_notebook(s) for s in selected)
    elif execute_all:
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


@contextmanager
def _temporary_environ(values: dict[str, str]):
    old_values: dict[str, str | None] = {}
    for key, value in values.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _build_write_guard_source(project_root: Path, guard_root: Path) -> str:
    project_root_abs = project_root.resolve()
    guard_root_abs = guard_root.resolve()
    return f"""
# Injected by scripts/run_all_notebooks.py to prevent notebooks from overwriting
# canonical pipeline artifacts under data/processed, models, and reports outputs.
import builtins as _nb_builtins
import os as _nb_os
from pathlib import Path as _NBPath

_PROJECT_ROOT = _NBPath(r"{project_root_abs}")
_GUARD_ROOT = _NBPath(r"{guard_root_abs}")
_GUARD_ROOT.mkdir(parents=True, exist_ok=True)
_CANONICAL_BLOCK_PREFIXES = [
    _PROJECT_ROOT / "data" / "processed",
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "reports" / "paper_material",
    _PROJECT_ROOT / "reports" / "figures",
]
_CANONICAL_BLOCK_PREFIXES = [p.resolve() for p in _CANONICAL_BLOCK_PREFIXES]
_ORIG_OPEN = _nb_builtins.open
_GUARD_TAG = "[nb-guard]"


def _resolve_path(path_like):
    p = _NBPath(path_like)
    if not p.is_absolute():
        p = (_NBPath.cwd() / p)
    return p.resolve()


def _redirect_if_blocked(path_like):
    try:
        p_abs = _resolve_path(path_like)
    except Exception:
        return path_like
    for prefix in _CANONICAL_BLOCK_PREFIXES:
        try:
            p_abs.relative_to(prefix)
            rel = p_abs.relative_to(_PROJECT_ROOT)
            redirected = (_GUARD_ROOT / rel).resolve()
            redirected.parent.mkdir(parents=True, exist_ok=True)
            _nb_builtins.print(f"{{_GUARD_TAG}} redirected write: {{p_abs}} -> {{redirected}}")
            return redirected
        except Exception:
            continue
    return path_like


def _guard_open(file, mode="r", *args, **kwargs):
    if any(flag in str(mode) for flag in ("w", "a", "x", "+")) and isinstance(
        file, (str, _NBPath)
    ):
        file = _redirect_if_blocked(file)
    return _ORIG_OPEN(file, mode, *args, **kwargs)


_nb_builtins.open = _guard_open

try:
    import pandas as _nb_pd

    def _patch_df_writer(name):
        original = getattr(_nb_pd.DataFrame, name, None)
        if original is None:
            return

        def _wrapped(self, path_or_buf=None, *args, **kwargs):
            if isinstance(path_or_buf, (str, _NBPath)):
                path_or_buf = _redirect_if_blocked(path_or_buf)
            return original(self, path_or_buf, *args, **kwargs)

        setattr(_nb_pd.DataFrame, name, _wrapped)

    for _method in (
        "to_parquet",
        "to_csv",
        "to_pickle",
        "to_json",
        "to_feather",
        "to_excel",
    ):
        _patch_df_writer(_method)
except Exception as _guard_exc:
    _nb_builtins.print(f"{{_GUARD_TAG}} pandas patch skipped: {{_guard_exc}}")

try:
    import plotly.graph_objects as _nb_go

    _orig_write_html = _nb_go.Figure.write_html
    _orig_write_image = _nb_go.Figure.write_image
    _orig_write_json = _nb_go.Figure.write_json

    def _wrap_plotly_writer(original):
        def _wrapped(self, file, *args, **kwargs):
            if isinstance(file, (str, _NBPath)):
                file = _redirect_if_blocked(file)
            return original(self, file, *args, **kwargs)

        return _wrapped

    _nb_go.Figure.write_html = _wrap_plotly_writer(_orig_write_html)
    _nb_go.Figure.write_image = _wrap_plotly_writer(_orig_write_image)
    _nb_go.Figure.write_json = _wrap_plotly_writer(_orig_write_json)
except Exception as _guard_exc:
    _nb_builtins.print(f"{{_GUARD_TAG}} plotly patch skipped: {{_guard_exc}}")

try:
    from matplotlib.figure import Figure as _NBFigure

    _orig_savefig = _NBFigure.savefig

    def _guard_savefig(self, fname, *args, **kwargs):
        if isinstance(fname, (str, _NBPath)):
            fname = _redirect_if_blocked(fname)
        return _orig_savefig(self, fname, *args, **kwargs)

    _NBFigure.savefig = _guard_savefig
except Exception as _guard_exc:
    _nb_builtins.print(f"{{_GUARD_TAG}} matplotlib patch skipped: {{_guard_exc}}")

_orig_write_text = _NBPath.write_text
_orig_write_bytes = _NBPath.write_bytes
_orig_touch = _NBPath.touch


def _guard_write_text(self, *args, **kwargs):
    target = _redirect_if_blocked(self)
    if isinstance(target, _NBPath):
        return _orig_write_text(target, *args, **kwargs)
    return _orig_write_text(self, *args, **kwargs)


def _guard_write_bytes(self, *args, **kwargs):
    target = _redirect_if_blocked(self)
    if isinstance(target, _NBPath):
        return _orig_write_bytes(target, *args, **kwargs)
    return _orig_write_bytes(self, *args, **kwargs)


def _guard_touch(self, *args, **kwargs):
    target = _redirect_if_blocked(self)
    if isinstance(target, _NBPath):
        return _orig_touch(target, *args, **kwargs)
    return _orig_touch(self, *args, **kwargs)


_NBPath.write_text = _guard_write_text
_NBPath.write_bytes = _guard_write_bytes
_NBPath.touch = _guard_touch
_nb_builtins.print(f"{{_GUARD_TAG}} active; sandbox={{_GUARD_ROOT}}")
""".strip()


def _is_fully_executed(nb_path: Path) -> bool:
    """Return True if all code cells in the notebook have a non-None execution_count.

    A cell with execution_count=None means it was added or edited after the last
    run (nbformat sets it to null on edit). A notebook with every code cell executed
    has not changed since the last run.
    """
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception:
        return False
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if not code_cells:
        return True  # no code → nothing to execute
    return all(c.get("execution_count") is not None for c in code_cells)


def _sandbox_root_for_notebook(nb_path: Path, output_dir: Path | None) -> Path:
    rel = nb_path.relative_to(PROJECT_ROOT).with_suffix("")
    base = output_dir if output_dir is not None else NOTEBOOK_SANDBOX_ROOT
    return (base / "generated" / rel).resolve()


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
    guard_root = _sandbox_root_for_notebook(nb_path, output_dir)
    guard_root.mkdir(parents=True, exist_ok=True)
    guard_cell = nbformat.v4.new_code_cell(
        source=_build_write_guard_source(PROJECT_ROOT, guard_root),
        metadata={"tags": ["injected-write-guard"]},
    )
    nb.cells.insert(0, guard_cell)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")

    try:
        with _temporary_environ(
            {
                "NOTEBOOK_EXECUTION_MODE": "reference",
                "NOTEBOOK_GUARD_ROOT": str(guard_root),
            }
        ):
            client.execute(cwd=str(nb_path.parent))
        if nb.cells and "injected-write-guard" in nb.cells[0].get("metadata", {}).get("tags", []):
            nb.cells.pop(0)
        nbformat.write(nb, target_path)
        return NotebookRunResult(
            notebook=nb_path.name,
            source_path=str(rel),
            output_path=str(target_path.relative_to(PROJECT_ROOT)),
            success=True,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
    except CellExecutionError as exc:
        if nb.cells and "injected-write-guard" in nb.cells[0].get("metadata", {}).get("tags", []):
            nb.cells.pop(0)
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
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help=(
            "Notebook path/name to execute (repeatable). "
            "When provided, overrides --execute-all/--include-side-projects."
        ),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--inplace", default="true", help="Write outputs back to source notebooks")
    parser.add_argument("--output-dir", default="reports/notebook_exec")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--skip-if-executed",
        action="store_true",
        default=True,
        help="Skip notebooks where all code cells are already executed (no edits since last run)",
    )
    parser.add_argument(
        "--no-skip-if-executed",
        dest="skip_if_executed",
        action="store_false",
        help="Force re-execution of all notebooks regardless of execution state",
    )
    args = parser.parse_args()

    inplace = _parse_bool(args.inplace)
    output_dir = None if inplace else (PROJECT_ROOT / args.output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    notebooks = _discover_notebooks(
        execute_all=args.execute_all,
        include_side_projects=args.include_side_projects,
        selected_notebooks=list(args.notebook or []),
    )
    if not notebooks:
        raise SystemExit(
            "No notebooks selected. Use --execute-all, --include-side-projects, or --notebook."
        )

    results: list[NotebookRunResult] = []
    for nb_path in notebooks:
        rel = nb_path.relative_to(PROJECT_ROOT)
        if args.skip_if_executed and _is_fully_executed(nb_path):
            print(f"[nb] Skipping {rel} (all cells already executed, no edits detected)")
            results.append(
                NotebookRunResult(
                    notebook=nb_path.name,
                    source_path=str(rel),
                    output_path=str(rel),
                    success=True,
                    duration_seconds=0.0,
                    error=None,
                )
            )
            continue
        print(f"[nb] Executing {rel}")
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

    skipped_count = int(sum(1 for r in results if r.success and r.duration_seconds == 0.0))
    manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": "2026-02-26.2",
        "inplace": bool(inplace),
        "skip_if_executed": bool(args.skip_if_executed),
        "timeout_seconds": int(args.timeout),
        "notebooks_total": len(notebooks),
        "executed": len(results) - skipped_count,
        "skipped": skipped_count,
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
