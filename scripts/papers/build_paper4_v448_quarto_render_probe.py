#!/usr/bin/env python3
"""Build Paper 4 v448 Quarto render probe artifacts."""

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

VERSION = 448
PRIOR_PYTEST_PROBE_VERSION = 447
NEXT_ARTIFACT = "paper4_v449_full_book_render_probe.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v448_quarto_render_probe.md"
BOOK_DIR = ROOT / "book"
QUARTO_CONFIG = BOOK_DIR / "_quarto.yml"
ARCHIVE_MANIFEST = BOOK_DIR / "_archived_chapter_pages.yml"
PAPER4_PREFIX = "chapters/19-paper-mega-extension/"
PAPER4_DIR = BOOK_DIR / "chapters" / "19-paper-mega-extension"
OUTPUT_INDEX = BOOK_DIR / "_output" / "chapters" / "19-paper-mega-extension" / "index.html"
RENDER_COMMAND = [
    "bash",
    "scripts/render_quarto.sh",
    "render",
    "book/chapters/19-paper-mega-extension",
    "--to",
    "html",
    "--execute-daemon-restart",
]
SUMMARY_ARTIFACT = (
    "reports/paper_material/paper4/tables/paper4_v448_quarto_render_probe_summary.csv"
)
REGISTERED_ARTIFACT = (
    "reports/paper_material/paper4/tables/paper4_v448_quarto_registered_pages.csv"
)
ARCHIVE_ARTIFACT = (
    "reports/paper_material/paper4/tables/paper4_v448_quarto_archive_surface.csv"
)
BLOCKERS_ARTIFACT = "reports/paper_material/paper4/tables/paper4_v448_claim_blockers.csv"


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
            if isinstance(value, str):
                pages.append(value)
        nested = item.get("chapters")
        if isinstance(nested, list):
            pages.extend(_walk_chapter_entries(nested))
    return pages


def _registered_paper4_pages() -> list[str]:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    pages = _walk_chapter_entries(config["book"]["chapters"])
    return sorted(page for page in pages if page.startswith(PAPER4_PREFIX))


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
        timeout=1200,
    )
    runtime = round((datetime.now(UTC) - started).total_seconds(), 3)
    stdout = (result.stdout or "").replace("\r", "\n")
    stderr = (result.stderr or "").replace("\r", "\n")
    combined = f"{stdout}\n{stderr}"
    rendered_pages_raw = re.findall(
        r"\[\s*\d+/\d+\]\s+(chapters/19-paper-mega-extension/[^\n]+)",
        combined,
    )
    rendered_pages = list(dict.fromkeys(page.strip() for page in rendered_pages_raw))
    output_match = re.search(r"Output created:\s+(.+)", combined)
    return {
        "command": " ".join(RENDER_COMMAND),
        "exit_code": int(result.returncode),
        "passed": result.returncode == 0,
        "runtime_seconds": runtime,
        "rendered_pages": [page.strip() for page in rendered_pages],
        "rendered_page_count": len(rendered_pages),
        "stdout_tail": "\n".join(stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-40:]),
        "reported_output": output_match.group(1).strip() if output_match else "",
    }


def _summary_table(render: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v448": "paper4_official_quarto_chapter_render",
                "command_v448": render["command"],
                "exit_code_v448": int(render["exit_code"]),
                "passed_v448": bool(render["passed"]),
                "runtime_seconds_v448": float(render["runtime_seconds"]),
                "rendered_page_count_v448": int(render["rendered_page_count"]),
                "output_index_v448": OUTPUT_INDEX.relative_to(ROOT).as_posix(),
                "reported_output_v448": str(render["reported_output"]),
                "claim_boundary_v448": "official Paper 4 registered Quarto chapter only",
            }
        ]
    )


def _registered_pages_table(pages: list[str], rendered_pages: list[str]) -> pd.DataFrame:
    rendered_set = set(rendered_pages)
    return pd.DataFrame(
        [
            {
                "page_v448": page,
                "registered_in_book_v448": True,
                "rendered_in_v448": page in rendered_set,
                "render_policy_v448": "official_paper4_chapter_page",
                "claim_boundary_v448": "registered Paper 4 Quarto surface",
            }
            for page in pages
        ]
    )


def _archive_surface_table(
    *,
    registered_pages: list[str],
    archived_pages: set[str],
    qmd_pages: set[str],
) -> pd.DataFrame:
    unregistered_nonarchived = sorted(qmd_pages - set(registered_pages) - archived_pages)
    return pd.DataFrame(
        [
            {
                "surface_metric_v448": "registered_official_paper4_pages",
                "count_v448": len(registered_pages),
                "claim_boundary_v448": "pages rendered by the official chapter command",
            },
            {
                "surface_metric_v448": "paper4_qmd_files_on_disk",
                "count_v448": len(qmd_pages),
                "claim_boundary_v448": "includes historical archive files",
            },
            {
                "surface_metric_v448": "intentionally_archived_paper4_pages",
                "count_v448": len(archived_pages),
                "claim_boundary_v448": "preserved on disk but not rendered officially",
            },
            {
                "surface_metric_v448": "unregistered_nonarchived_paper4_pages",
                "count_v448": len(unregistered_nonarchived),
                "claim_boundary_v448": "should remain zero under archive policy",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v448": "full_book_render_not_run",
                "blocking_v448": True,
                "evidence_count_v448": 1,
                "required_next_artifact_v448": NEXT_ARTIFACT,
                "claim_boundary_v448": "Paper 4 chapter render is not a full book render claim",
            },
            {
                "blocker_id_v448": "paper4_final_promotion_forbidden",
                "blocking_v448": True,
                "evidence_count_v448": 1,
                "required_next_artifact_v448": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v448": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix(*, render_passed: bool, counts_match: bool, archive_clean: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v448_paper4_official_quarto_render_run",
                "allowed": True,
                "artifact": "paper4_v448_quarto_render_probe_summary.csv",
                "boundary": "official Paper 4 chapter render command executed",
            },
            {
                "claim_id": "v448_paper4_official_quarto_render_clean",
                "allowed": render_passed,
                "artifact": "paper4_v448_quarto_render_probe_summary.csv",
                "boundary": "true only when Quarto exits 0",
            },
            {
                "claim_id": "v448_registered_page_count_matches_render",
                "allowed": counts_match,
                "artifact": "paper4_v448_quarto_registered_pages.csv",
                "boundary": "rendered pages must match registered Paper 4 pages",
            },
            {
                "claim_id": "v448_archive_policy_preserved",
                "allowed": archive_clean,
                "artifact": "paper4_v448_quarto_archive_surface.csv",
                "boundary": "historical files remain archived, not official rendered pages",
            },
            {
                "claim_id": "v448_full_book_render_clean",
                "allowed": False,
                "artifact": "paper4_v448_claim_blockers.csv",
                "boundary": "full book render deferred to v449",
            },
            {
                "claim_id": "v448_working_champion_or_final_promotion",
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
    archive_clean: bool,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v448 renders the official registered Paper 4 Quarto chapter.",
                "allowed": render_passed,
                "evidence_artifact": SUMMARY_ARTIFACT,
                "boundary": "Limited to the registered Paper 4 chapter surface.",
                "prohibited_claim_flag": not render_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v448 rendered pages match the official Paper 4 registry.",
                "allowed": counts_match,
                "evidence_artifact": REGISTERED_ARTIFACT,
                "boundary": "Registered page count equals observed render page count.",
                "prohibited_claim_flag": not counts_match,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v448 preserves the historical Paper 4 archive policy.",
                "allowed": archive_clean,
                "evidence_artifact": ARCHIVE_ARTIFACT,
                "boundary": "Archived QMD files stay out of the official rendered chapter.",
                "prohibited_claim_flag": not archive_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v448 proves full book render cleanliness.",
                "allowed": False,
                "evidence_artifact": BLOCKERS_ARTIFACT,
                "boundary": "Full book render is deferred to v449.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v448 replaces Paper Estrella or finalizes Paper 4.",
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
                "executable_item": "v448 renders the official Paper 4 Quarto chapter surface.",
                "status": (
                    "paper4_quarto_render_passed"
                    if render_passed
                    else "paper4_quarto_render_failed"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v449 runs a full book render probe without promoting Paper 4",
                "last_wave": "v448",
                "execution_result": (
                    "official_paper4_quarto_chapter_render_passed"
                    if render_passed
                    else "official_paper4_quarto_chapter_render_failed"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v448")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], render: dict[str, Any]) -> str:
    return f"""# Paper 4 Quarto Render Probe v448

Generated: {status["generated_at_utc"]}

v448 runs the official registered Paper 4 Quarto chapter render after v447
established a clean full-pytest and repository-Ruff baseline.

## Result

- Command: `{render["command"]}`.
- Exit code: `{status["quarto_render_exit_code_v448"]}`.
- Render passed: `{status["paper4_official_quarto_render_clean_v448"]}`.
- Runtime seconds: `{status["quarto_render_runtime_seconds_v448"]}`.
- Registered Paper 4 pages: `{status["registered_paper4_page_count_v448"]}`.
- Observed rendered pages: `{status["rendered_page_count_v448"]}`.
- Output index exists: `{status["output_index_exists_v448"]}`.
- Full book render run: `{status["full_book_render_run_v448"]}`.

## Stdout Tail

```text
{render["stdout_tail"]}
```

## Stderr Tail

```text
{render["stderr_tail"]}
```

## Required Caveat

v448 proves only the official Paper 4 registered chapter render. It does not
claim a full-book render, champion replacement, Paper Estrella replacement, or
final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v448"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V448_QUARTO_RENDER_PROBE_START -->"
    end = "<!-- V448_QUARTO_RENDER_PROBE_END -->"
    block = f"""
{start}

## Wave v448: Quarto Render Probe

Generated: {status["generated_at_utc"]}

### Objective

v448 renders the official registered Paper 4 Quarto chapter after v447 made the
repository clean under full pytest and Ruff.

### Results

- Quarto command:
  `{status["quarto_render_command_v448"]}`.
- Quarto exit code:
  `{status["quarto_render_exit_code_v448"]}`.
- Paper 4 official render clean:
  `{status["paper4_official_quarto_render_clean_v448"]}`.
- Registered Paper 4 pages:
  `{status["registered_paper4_page_count_v448"]}`.
- Observed rendered pages:
  `{status["rendered_page_count_v448"]}`.
- Historical Paper 4 QMD files on disk:
  `{status["paper4_qmd_files_on_disk_v448"]}`.
- Intentionally archived Paper 4 pages:
  `{status["archived_paper4_page_count_v448"]}`.
- Unregistered non-archived Paper 4 pages:
  `{status["unregistered_nonarchived_page_count_v448"]}`.
- Output index exists:
  `{status["output_index_exists_v448"]}`.
- Full book render run:
  `{status["full_book_render_run_v448"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v448"]}`.

### Interpretation

The official Paper 4 Quarto chapter renders cleanly with the compact registered
surface. This closes the v447 Quarto-render blocker at the Paper 4 chapter level
while keeping the historical archive out of the official rendered surface.

### Claim Impact

- Allowed: official Paper 4 registered Quarto chapter render passed.
- Still prohibited: full-book render clean, champion replacement and final
  promotion claims.

### Quarto Promotion Decision

Keep v448 in the living notebook. v449 should probe the full book render.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v447_status = json.loads((STATUS_DIR / "paper4_v447_status.json").read_text(encoding="utf-8"))
    if v447_status["next_artifact_v447"] != "paper4_v448_quarto_render_probe.md":
        raise RuntimeError("v448 expects v447 to route to Quarto render probe.")
    if v447_status["repository_ruff_clean_v447"] is not True:
        raise RuntimeError("v448 expects repository Ruff to be clean in v447.")
    if v447_status["full_repository_pytest_passed_v447"] is not True:
        raise RuntimeError("v448 expects full repository pytest to pass in v447.")

    render = _run_render()
    registered_pages = _registered_paper4_pages()
    rendered_pages = list(render["rendered_pages"])
    qmd_pages = _paper4_qmd_pages()
    archived_pages = _archived_paper4_pages()
    unregistered_nonarchived = sorted(qmd_pages - set(registered_pages) - archived_pages)
    counts_match = set(rendered_pages) == set(registered_pages)
    archive_clean = len(unregistered_nonarchived) == 0 and not (
        set(registered_pages) & archived_pages
    )

    summary = _summary_table(render)
    registered = _registered_pages_table(registered_pages, rendered_pages)
    archive_surface = _archive_surface_table(
        registered_pages=registered_pages,
        archived_pages=archived_pages,
        qmd_pages=qmd_pages,
    )
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix(
        render_passed=bool(render["passed"]),
        counts_match=counts_match,
        archive_clean=archive_clean,
    )
    _update_claim_boundaries(
        render_passed=bool(render["passed"]),
        counts_match=counts_match,
        archive_clean=archive_clean,
    )
    _update_backlog(bool(render["passed"]))

    write_csv(TABLE_DIR / "paper4_v448_quarto_render_probe_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v448_quarto_registered_pages.csv", registered)
    write_csv(TABLE_DIR / "paper4_v448_quarto_archive_surface.csv", archive_surface)
    write_csv(TABLE_DIR / "paper4_v448_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v448_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v448_quarto_render_probe",
        "schema_version": "2026-05-17.448",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_pytest_probe_version_v448": PRIOR_PYTEST_PROBE_VERSION,
        "quarto_render_command_v448": str(render["command"]),
        "quarto_render_exit_code_v448": int(render["exit_code"]),
        "quarto_render_runtime_seconds_v448": float(render["runtime_seconds"]),
        "paper4_official_quarto_render_run_v448": True,
        "paper4_official_quarto_render_clean_v448": bool(render["passed"]),
        "full_quarto_render_run_v448": True,
        "full_quarto_render_clean_v448": bool(render["passed"]),
        "full_book_render_run_v448": False,
        "full_book_render_clean_v448": False,
        "registered_paper4_page_count_v448": len(registered_pages),
        "rendered_page_count_v448": int(render["rendered_page_count"]),
        "registered_page_count_matches_render_v448": counts_match,
        "paper4_qmd_files_on_disk_v448": len(qmd_pages),
        "archived_paper4_page_count_v448": len(archived_pages),
        "unregistered_nonarchived_page_count_v448": len(unregistered_nonarchived),
        "archive_policy_preserved_v448": archive_clean,
        "output_index_v448": OUTPUT_INDEX.relative_to(ROOT).as_posix(),
        "output_index_exists_v448": OUTPUT_INDEX.exists(),
        "working_champion_claim_allowed_v448": False,
        "paper1_promotion_allowed_v448": False,
        "paper4_working_champion_changed_v448": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v448": NEXT_ARTIFACT,
        "claim_boundary": (
            "v448 is an official Paper 4 Quarto chapter render probe; "
            "full-book render and final promotion remain blocked"
        ),
    }
    if not status["paper4_official_quarto_render_clean_v448"]:
        raise RuntimeError("v448 expected the official Paper 4 Quarto chapter render to pass.")
    if not counts_match:
        raise RuntimeError("v448 expected rendered Paper 4 pages to match registered pages.")
    if not archive_clean:
        raise RuntimeError("v448 expected the Paper 4 archive policy to remain clean.")
    if not status["output_index_exists_v448"]:
        raise RuntimeError("v448 expected the Paper 4 output index to exist.")

    PROBE_MD.write_text(_probe_markdown(status, render), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v448": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
