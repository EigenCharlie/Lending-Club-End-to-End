#!/usr/bin/env python3
"""Build a logged-in LinkedIn review queue from all Denis Burakov ingest packs."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT_PACK = Path("reports/linkedin_credit_risk_denis_burakov")
REVIEW_PACK = ROOT_PACK / "logged_in_review"
QUEUE_PATH = REVIEW_PACK / "data" / "logged_in_review_queue.csv"

QUEUE_FIELDS = [
    "queue_id",
    "source_pack",
    "source_row_id",
    "activity_id",
    "post_url",
    "title",
    "theme",
    "prior_status",
    "logged_in_capture_status",
    "comments_status",
    "stop_condition",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_post(
    rows: list[dict[str, str]],
    seen: set[str],
    *,
    source_pack: str,
    source_row_id: str,
    activity_id: str,
    post_url: str,
    title: str,
    theme: str,
    prior_status: str,
) -> None:
    if not activity_id or activity_id in seen:
        return
    seen.add(activity_id)
    rows.append(
        {
            "queue_id": f"LI-LOGIN-{len(rows) + 1:03d}",
            "source_pack": source_pack,
            "source_row_id": source_row_id,
            "activity_id": activity_id,
            "post_url": post_url,
            "title": title,
            "theme": theme,
            "prior_status": prior_status,
            "logged_in_capture_status": "pending_cdp_capture",
            "comments_status": "pending_logged_in_comment_expansion",
            "stop_condition": (
                "Close when logged-in rendered text, visible comments, comment links, "
                "and any newly exposed attachments are captured or explicitly blocked."
            ),
        }
    )


def build_queue() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for post in read_csv(ROOT_PACK / "data" / "posts_index.csv"):
        add_post(
            rows,
            seen,
            source_pack="first_ingest",
            source_row_id=post.get("n", ""),
            activity_id=post.get("activity_id", ""),
            post_url=post.get("post_url", ""),
            title=post.get("title", ""),
            theme=post.get("theme", ""),
            prior_status="first_ingest_public_assets_processed",
        )

    for child in read_csv(ROOT_PACK / "data" / "external_linkedin_child_post_backlog.csv"):
        add_post(
            rows,
            seen,
            source_pack="first_ingest_child_post",
            source_row_id=child.get("backlog_id", ""),
            activity_id=child.get("activity_id", ""),
            post_url=child.get("post_url", ""),
            title=child.get("title", ""),
            theme=child.get("theme", ""),
            prior_status=child.get("source_status", "first_ingest_child_source"),
        )

    for post in read_csv(ROOT_PACK / "second_ingest" / "data" / "posts_index.csv"):
        add_post(
            rows,
            seen,
            source_pack="second_ingest",
            source_row_id=post.get("n", ""),
            activity_id=post.get("activity_id", ""),
            post_url=post.get("post_url", ""),
            title=post.get("title", ""),
            theme=post.get("theme", ""),
            prior_status="second_ingest_public_assets_processed",
        )

    return rows


def write_docs(row_count: int) -> None:
    docs_dir = REVIEW_PACK / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Logged-In LinkedIn Review Queue - 2026-05-21

This pack is for a third review pass using the user's own visible logged-in
LinkedIn access. It is intentionally separate from the public first and second
ingests.

- Queue rows: {row_count}
- Capture method: Chrome DevTools Protocol against a user-owned browser session.
- Scope: rendered post text, comments, comment links, and any newly exposed
  attachment controls.

## Guardrails

- No fake accounts, captcha bypass, stealth, or rate evasion.
- Do not print cookie values or credentials.
- Treat comments and LinkedIn-only materials as private research intake.
- Promote no paper/book claims from comments alone.

## Stop Condition

Close each row when logged-in rendered text, visible comments, comment links,
and newly exposed attachments are either captured/read or assigned an explicit
blocker.
"""
    (docs_dir / "logged_in_review_plan_2026-05-21.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = build_queue()
    write_csv(QUEUE_PATH, rows, QUEUE_FIELDS)
    write_docs(len(rows))
    print(f"Logged-in review queue written: {len(rows)} rows")


if __name__ == "__main__":
    main()
