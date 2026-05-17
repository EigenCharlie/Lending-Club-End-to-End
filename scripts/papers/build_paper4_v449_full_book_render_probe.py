#!/usr/bin/env python3
"""Build Paper 4 v449 full-book Quarto render probe artifacts."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yaml

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    ROOT,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    write_csv,
    write_json,
)

VERSION = 449
PRIOR_QUARTO_RENDER_VERSION = 448
NEXT_ARTIFACT = "paper4_v450_post_full_book_render_pytest_probe.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v449_full_book_render_probe.md"
BOOK_DIR = ROOT / "book"
QUARTO_CONFIG = BOOK_DIR / "_quarto.yml"
ARCHIVE_MANIFEST = BOOK_DIR / "_archived_chapter_pages.yml"
PAPER4_PREFIX = "chapters/19-paper-mega-extension/"
PAPER4_DIR = BOOK_DIR / "chapters" / "19-paper-mega-extension"
OUTPUT_INDEX = BOOK_DIR / "_output" / "index.html"
RENDER_COMMAND = [
    "bash",
    "scripts/render_quarto.sh",
    "render",
    "book/",
    "--to",
    "html",
    "--execute-daemon-restart",
]
SUMMARY_ARTIFACT = (
    "reports/paper_material/paper4/tables/paper4_v449_full_book_render_probe_summary.csv"
)
REGISTERED_ARTIFACT = (
    "reports/paper_material/paper4/tables/paper4_v449_full_book_registered_pages.csv"
)
PAPER4_SURFACE_ARTIFACT = (
    "reports/paper_material/paper4/tables/paper4_v449_paper4_surface_in_full_book.csv"
)
BLOCKERS_ARTIFACT = "reports/paper_material/paper4/tables/paper4_v449_claim_blockers.csv"


def _walk_chapter_entries(items: list[object]) -> list[str]:
    pages: list[str] = []
    for item in items:
        if isinstance(item, str):
            pages.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("chapter", "part"):
            value = item.get(key)
            if isinstance(value, str) and key == "chapter":
                pages.append(value)
        nested = item.get("chapters")
        if isinstance(nested, list):
            pages.extend(_walk_chapter_entries(nested))
    return pages


def _registered_pages() -> list[str]:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    return _walk_chapter_entries(config["book"]["chapters"])


def _archived_paper4_pages() -> set[str]:
    if not ARCHIVE_MANIFEST.exists():
        return set()
    payload = yaml.safe_load(ARCHIVE_MANIFEST.read_text(encoding="utf-8")) or {}
    rows = payload.get("archived_chapter_pages", [])
    return {
        str(row["path"])
        for row in rows
        if isinstance(row, dict) and str(row.get("path", "")).startswith(PAPER4_PREFIX)
    }


def _paper4_qmd_pages() -> set[str]:
    return {
        path.relative_to(BOOK_DIR).as_posix()
        for path in PAPER4_DIR.glob("*.qmd")
        if path.is_file()
    }


def _run_render() -> dict[str, Any]:
    started = datetime.now(UTC)
    result = subprocess.run(
        RENDER_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    runtime = round((datetime.now(UTC) - started).total_seconds(), 3)
    stdout = (result.stdout or "").replace("\r", "\n")
    stderr = (result.stderr or "").replace("\r", "\n")
    combined = f"{stdout}\n{stderr}"
    rendered_pages_raw = re.findall(r"\[\s*\d+/\d+\]\s+([^\n]+)", combined)
    rendered_pages = list(dict.fromkeys(page.strip() for page in rendered_pages_raw))
    output_match = re.search(r"Output created:\s+(.+)", combined)
    return {
        "command": " ".join(RENDER_COMMAND),
        "exit_code": int(result.returncode),
        "passed": result.returncode == 0,
        "runtime_seconds": runtime,
        "rendered_pages": rendered_pages,
        "rendered_page_count": len(rendered_pages),
        "stdout_tail": "\n".join(stdout.splitlines()[-60:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-60:]),
        "reported_output": output_match.group(1).strip() if output_match else "",
    }


def _summary_table(render: dict[str, Any], registered_pages: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v449": "full_book_quarto_render",
                "command_v449": render["command"],
                "exit_code_v449": int(render["exit_code"]),
                "passed_v449": bool(render["passed"]),
                "runtime_seconds_v449": float(render["runtime_seconds"]),
                "registered_page_count_v449": len(registered_pages),
                "rendered_page_count_v449": int(render["rendered_page_count"]),
                "output_index_v449": OUTPUT_INDEX.relative_to(ROOT).as_posix(),
                "reported_output_v449": str(render["reported_output"]),
                "claim_boundary_v449": "full official Quarto book render only",
            }
        ]
    )


def _registered_pages_table(
    registered_pages: list[str],
    rendered_pages: list[str],
) -> pd.DataFrame:
    rendered_set = set(rendered_pages)
    return pd.DataFrame(
        [
            {
                "page_v449": page,
                "registered_in_book_v449": True,
                "rendered_in_v449": page in rendered_set,
                "paper4_page_v449": page.startswith(PAPER4_PREFIX),
                "claim_boundary_v449": "official Quarto book registry",
            }
            for page in registered_pages
        ]
    )


def _paper4_surface_table(
    *,
    registered_pages: list[str],
    rendered_pages: list[str],
    archived_pages: set[str],
    qmd_pages: set[str],
) -> pd.DataFrame:
    paper4_registered = [page for page in registered_pages if page.startswith(PAPER4_PREFIX)]
    rendered_set = set(rendered_pages)
    unregistered_nonarchived = sorted(qmd_pages - set(paper4_registered) - archived_pages)
    rows = [
        {
            "surface_metric_v449": "paper4_registered_pages_in_full_book",
            "count_v449": len(paper4_registered),
            "passed_v449": len(paper4_registered) == 10,
            "claim_boundary_v449": "Paper 4 official compact surface",
        },
        {
            "surface_metric_v449": "paper4_rendered_pages_in_full_book",
            "count_v449": sum(1 for page in paper4_registered if page in rendered_set),
            "passed_v449": set(paper4_registered).issubset(rendered_set),
            "claim_boundary_v449": "Paper 4 pages rendered during full-book render",
        },
        {
            "surface_metric_v449": "paper4_qmd_files_on_disk",
            "count_v449": len(qmd_pages),
            "passed_v449": True,
            "claim_boundary_v449": "includes historical archive files",
        },
        {
            "surface_metric_v449": "intentionally_archived_paper4_pages",
            "count_v449": len(archived_pages),
            "passed_v449": True,
            "claim_boundary_v449": "preserved on disk but excluded from official render",
        },
        {
            "surface_metric_v449": "unregistered_nonarchived_paper4_pages",
            "count_v449": len(unregistered_nonarchived),
            "passed_v449": len(unregistered_nonarchived) == 0,
            "claim_boundary_v449": "should remain zero under archive policy",
        },
    ]
    return pd.DataFrame(rows)


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v449": "post_render_full_repository_pytest_not_rerun",
                "blocking_v449": True,
                "evidence_count_v449": 1,
                "required_next_artifact_v449": NEXT_ARTIFACT,
                "claim_boundary_v449": "new v448-v449 guardrails need a post-render pytest refresh",
            },
            {
                "blocker_id_v449": "paper4_final_promotion_forbidden",
                "blocking_v449": True,
                "evidence_count_v449": 1,
                "required_next_artifact_v449": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v449": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix(*, render_passed: bool, counts_match: bool, paper4_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v449_full_book_quarto_render_run",
                "allowed": True,
                "artifact": "paper4_v449_full_book_render_probe_summary.csv",
                "boundary": "full official Quarto book render command executed",
            },
            {
                "claim_id": "v449_full_book_quarto_render_clean",
                "allowed": render_passed,
                "artifact": "paper4_v449_full_book_render_probe_summary.csv",
                "boundary": "true only when Quarto exits 0",
            },
            {
                "claim_id": "v449_registered_page_count_matches_render",
                "allowed": counts_match,
                "artifact": "paper4_v449_full_book_registered_pages.csv",
                "boundary": "rendered pages must match registered book pages",
            },
            {
                "claim_id": "v449_paper4_surface_renders_inside_full_book",
                "allowed": paper4_passed,
                "artifact": "paper4_v449_paper4_surface_in_full_book.csv",
                "boundary": "registered Paper 4 pages render as part of full book",
            },
            {
                "claim_id": "v449_post_render_full_repository_pytest_clean",
                "allowed": False,
                "artifact": "paper4_v449_claim_blockers.csv",
                "boundary": "post-render full pytest deferred to v450",
            },
            {
                "claim_id": "v449_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *,
    render_passed: bool,
    counts_match: bool,
    paper4_passed: bool,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v449 renders the full official Quarto book.",
                "allowed": render_passed,
                "evidence_artifact": SUMMARY_ARTIFACT,
                "boundary": "Full book render only; not a deployment or paper promotion.",
                "prohibited_claim_flag": not render_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v449 rendered pages match the official Quarto book registry.",
                "allowed": counts_match,
                "evidence_artifact": REGISTERED_ARTIFACT,
                "boundary": "Registered page count equals observed render page count.",
                "prohibited_claim_flag": not counts_match,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v449 renders Paper 4 inside the full book.",
                "allowed": paper4_passed,
                "evidence_artifact": PAPER4_SURFACE_ARTIFACT,
                "boundary": "Registered Paper 4 pages only.",
                "prohibited_claim_flag": not paper4_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v449 proves post-render full repository pytest is clean.",
                "allowed": False,
                "evidence_artifact": BLOCKERS_ARTIFACT,
                "boundary": "Full pytest refresh is deferred to v450.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v449 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": BLOCKERS_ARTIFACT,
                "boundary": (
                    "No final promotion artifact, champion replacement or deployment gate "
                    "is created."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(render_passed: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Publication",
                "executable_item": "v449 renders the full official Quarto book.",
                "status": "full_book_render_passed" if render_passed else "full_book_render_failed",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v450 reruns full pytest after v448-v449 guardrail additions",
                "last_wave": "v449",
                "execution_result": (
                    "full_official_quarto_book_render_passed"
                    if render_passed
                    else "full_official_quarto_book_render_failed"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v449")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], render: dict[str, Any]) -> str:
    return f"""# Paper 4 Full-Book Render Probe v449

Generated: {status["generated_at_utc"]}

v449 runs the full official Quarto book render after v448 established that the
registered Paper 4 chapter renders cleanly on its own.

## Result

- Command: `{render["command"]}`.
- Exit code: `{status["full_book_render_exit_code_v449"]}`.
- Full book render passed: `{status["full_book_render_clean_v449"]}`.
- Runtime seconds: `{status["full_book_render_runtime_seconds_v449"]}`.
- Registered book pages: `{status["registered_book_page_count_v449"]}`.
- Observed rendered pages: `{status["rendered_page_count_v449"]}`.
- Paper 4 rendered pages inside full book: `{status["paper4_rendered_page_count_v449"]}`.
- Output index exists: `{status["output_index_exists_v449"]}`.
- Post-render full pytest run: `{status["post_render_full_repository_pytest_run_v449"]}`.

## Stdout Tail

```text
{render["stdout_tail"]}
```

## Stderr Tail

```text
{render["stderr_tail"]}
```

## Required Caveat

v449 proves the full official Quarto book renders, including the registered
Paper 4 compact surface. It does not claim a post-render full-pytest refresh,
champion replacement, Paper Estrella replacement, or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v449"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V449_FULL_BOOK_RENDER_PROBE_START -->"
    end = "<!-- V449_FULL_BOOK_RENDER_PROBE_END -->"
    block = f"""
{start}

## Wave v449: Full-Book Render Probe

Generated: {status["generated_at_utc"]}

### Objective

v449 renders the full official Quarto book after v448 proved the registered
Paper 4 chapter render.

### Results

- Full book render command:
  `{status["full_book_render_command_v449"]}`.
- Full book render exit code:
  `{status["full_book_render_exit_code_v449"]}`.
- Full book render clean:
  `{status["full_book_render_clean_v449"]}`.
- Registered book pages:
  `{status["registered_book_page_count_v449"]}`.
- Observed rendered pages:
  `{status["rendered_page_count_v449"]}`.
- Paper 4 rendered pages inside full book:
  `{status["paper4_rendered_page_count_v449"]}`.
- Paper 4 archive policy preserved:
  `{status["paper4_archive_policy_preserved_v449"]}`.
- Output index exists:
  `{status["output_index_exists_v449"]}`.
- Post-render full pytest run:
  `{status["post_render_full_repository_pytest_run_v449"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v449"]}`.

### Interpretation

The full book render is clean and includes the compact Paper 4 registered
surface. The remaining publication validation gap is a post-render full pytest
refresh after adding the v448-v449 guardrails and documentation artifacts.

### Claim Impact

- Allowed: full official Quarto book render passed; Paper 4 renders inside it.
- Still prohibited: post-render full pytest clean, champion replacement and
  final promotion claims.

### Quarto Promotion Decision

Keep v449 in the living notebook. v450 should rerun full repository pytest.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v448_status = json.loads((STATUS_DIR / "paper4_v448_status.json").read_text(encoding="utf-8"))
    if v448_status["next_artifact_v448"] != "paper4_v449_full_book_render_probe.md":
        raise RuntimeError("v449 expects v448 to route to full-book render probe.")
    if v448_status["paper4_official_quarto_render_clean_v448"] is not True:
        raise RuntimeError("v449 expects v448 Paper 4 render to be clean.")

    render = _run_render()
    registered_pages = _registered_pages()
    rendered_pages = list(render["rendered_pages"])
    rendered_set = set(rendered_pages)
    paper4_registered = [page for page in registered_pages if page.startswith(PAPER4_PREFIX)]
    paper4_rendered = [page for page in paper4_registered if page in rendered_set]
    qmd_pages = _paper4_qmd_pages()
    archived_pages = _archived_paper4_pages()
    unregistered_nonarchived = sorted(qmd_pages - set(paper4_registered) - archived_pages)
    counts_match = set(rendered_pages) == set(registered_pages)
    paper4_passed = set(paper4_registered).issubset(rendered_set)
    archive_clean = len(unregistered_nonarchived) == 0 and not (
        set(paper4_registered) & archived_pages
    )

    summary = _summary_table(render, registered_pages)
    registered = _registered_pages_table(registered_pages, rendered_pages)
    paper4_surface = _paper4_surface_table(
        registered_pages=registered_pages,
        rendered_pages=rendered_pages,
        archived_pages=archived_pages,
        qmd_pages=qmd_pages,
    )
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix(
        render_passed=bool(render["passed"]),
        counts_match=counts_match,
        paper4_passed=paper4_passed,
    )
    _update_claim_boundaries(
        render_passed=bool(render["passed"]),
        counts_match=counts_match,
        paper4_passed=paper4_passed,
    )
    _update_backlog(bool(render["passed"]))

    write_csv(TABLE_DIR / "paper4_v449_full_book_render_probe_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v449_full_book_registered_pages.csv", registered)
    write_csv(TABLE_DIR / "paper4_v449_paper4_surface_in_full_book.csv", paper4_surface)
    write_csv(TABLE_DIR / "paper4_v449_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v449_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v449_full_book_render_probe",
        "schema_version": "2026-05-17.449",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_quarto_render_version_v449": PRIOR_QUARTO_RENDER_VERSION,
        "full_book_render_command_v449": str(render["command"]),
        "full_book_render_exit_code_v449": int(render["exit_code"]),
        "full_book_render_runtime_seconds_v449": float(render["runtime_seconds"]),
        "full_book_render_run_v449": True,
        "full_book_render_clean_v449": bool(render["passed"]),
        "registered_book_page_count_v449": len(registered_pages),
        "rendered_page_count_v449": int(render["rendered_page_count"]),
        "registered_page_count_matches_render_v449": counts_match,
        "paper4_registered_page_count_v449": len(paper4_registered),
        "paper4_rendered_page_count_v449": len(paper4_rendered),
        "paper4_surface_renders_inside_full_book_v449": paper4_passed,
        "paper4_qmd_files_on_disk_v449": len(qmd_pages),
        "paper4_archived_page_count_v449": len(archived_pages),
        "paper4_unregistered_nonarchived_page_count_v449": len(unregistered_nonarchived),
        "paper4_archive_policy_preserved_v449": archive_clean,
        "output_index_v449": OUTPUT_INDEX.relative_to(ROOT).as_posix(),
        "output_index_exists_v449": OUTPUT_INDEX.exists(),
        "post_render_full_repository_pytest_run_v449": False,
        "post_render_full_repository_pytest_clean_v449": False,
        "working_champion_claim_allowed_v449": False,
        "paper1_promotion_allowed_v449": False,
        "paper4_working_champion_changed_v449": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v449": NEXT_ARTIFACT,
        "claim_boundary": (
            "v449 is a full official Quarto book render probe; post-render "
            "pytest and final promotion remain blocked"
        ),
    }
    if not status["full_book_render_clean_v449"]:
        raise RuntimeError("v449 expected the full book render to pass.")
    if not counts_match:
        raise RuntimeError("v449 expected rendered pages to match registered book pages.")
    if not paper4_passed:
        raise RuntimeError("v449 expected Paper 4 pages to render inside the full book.")
    if not archive_clean:
        raise RuntimeError("v449 expected the Paper 4 archive policy to remain clean.")
    if not status["output_index_exists_v449"]:
        raise RuntimeError("v449 expected the book output index to exist.")

    PROBE_MD.write_text(_probe_markdown(status, render), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v449": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
