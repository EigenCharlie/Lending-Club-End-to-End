#!/usr/bin/env python3
"""Build the P0/P1 logged-in review queue for Andrija Djurovic LinkedIn material."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT_PACK = Path("reports/linkedin_credit_risk_andrija_djurovic")
REVIEW_PACK = ROOT_PACK / "logged_in_review"
QUEUE_PATH = REVIEW_PACK / "data" / "logged_in_review_queue.csv"
PRIORITY_PATH = ROOT_PACK / "data" / "andrija_login_only_priority_queue.csv"
POST_INDEX = ROOT_PACK / "data" / "posts_index.csv"

QUEUE_FIELDS = [
    "queue_id",
    "source_pack",
    "source_row_id",
    "priority",
    "target_kind",
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
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def activity_route(activity_id: str) -> str:
    return f"https://www.linkedin.com/feed/update/urn:li:activity:{activity_id}/"


def add_row(
    rows: list[dict[str, str]],
    seen: set[str],
    *,
    source_row_id: str,
    priority: str,
    target_kind: str,
    activity_id: str,
    post_url: str,
    title: str,
    theme: str,
    prior_status: str,
    stop_condition: str,
) -> None:
    if not activity_id or activity_id in seen:
        return
    seen.add(activity_id)
    rows.append(
        {
            "queue_id": f"ANDRIJA-LOGIN-{len(rows) + 1:03d}",
            "source_pack": "andrija_login_only_p0_p1",
            "source_row_id": source_row_id,
            "priority": priority,
            "target_kind": target_kind,
            "activity_id": activity_id,
            "post_url": post_url or activity_route(activity_id),
            "title": title,
            "theme": theme,
            "prior_status": prior_status,
            "logged_in_capture_status": "pending_cdp_capture",
            "comments_status": "pending_logged_in_comment_expansion",
            "stop_condition": stop_condition,
        }
    )


def build_queue() -> list[dict[str, str]]:
    posts = {row["activity_id"]: row for row in read_csv(POST_INDEX)}
    priority_rows = read_csv(PRIORITY_PATH)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for source_row in priority_rows:
        if source_row["priority"] not in {"P0", "P1"}:
            continue
        target_kind = source_row["target_kind"]
        target_ids = [item.strip() for item in source_row["target_id"].split(",") if item.strip()]
        for target_id in target_ids:
            post = posts.get(target_id, {})
            target_url = activity_route(target_id)
            add_row(
                rows,
                seen,
                source_row_id=source_row["target_id"],
                priority=source_row["priority"],
                target_kind=target_kind,
                activity_id=target_id,
                post_url=target_url,
                title=post.get("title", f"Profile activity {target_id}"),
                theme=post.get("theme", target_kind),
                prior_status=source_row["reason_to_try_logged_in"],
                stop_condition=source_row["stop_condition"],
            )
    return rows


def write_docs(row_count: int) -> None:
    docs_dir = REVIEW_PACK / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Andrija Logged-In Review Queue - 2026-05-25

This queue contains P0 and P1 targets from
`reports/linkedin_credit_risk_andrija_djurovic/data/andrija_login_only_priority_queue.csv`.

- Queue rows: {row_count}
- Intended method: Playwright/CDP against a user-owned visible logged-in browser
  session.
- Scope: rendered post text, visible comments, comment links, and classification
  of profile activity IDs that did not expose public own-post permalinks.

## Guardrails

- No fake accounts, captcha bypass, stealth, or rate evasion.
- Do not print cookie values or credentials.
- Treat comments and LinkedIn-only materials as private research intake.
- Promote no paper/book claims from comments alone.

## Stop Condition

Close each row when logged-in rendered text, visible comments, comment links,
and newly exposed attachments are captured/read, or when the page is classified
as non-credit-risk, reaction-only, inaccessible, or already covered by stronger
external sources.
"""
    (docs_dir / "logged_in_review_plan_2026-05-25.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = build_queue()
    write_csv(QUEUE_PATH, rows, QUEUE_FIELDS)
    write_docs(len(rows))
    print(f"Andrija logged-in review queue written: {len(rows)} rows")


if __name__ == "__main__":
    main()
