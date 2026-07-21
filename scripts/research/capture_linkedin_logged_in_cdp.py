#!/usr/bin/env python3
"""Capture rendered LinkedIn posts/comments through a user-owned Chrome CDP session.

The script assumes Chrome is already running with a DevTools endpoint, for
example on http://127.0.0.1:9222. It does not read or print cookie values. It
uses the browser's own authenticated state to capture rendered post text,
visible comments, links inside the post/comment surface, and a screenshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT_PACK = Path("reports/linkedin_credit_risk_denis_burakov")
REVIEW_PACK = ROOT_PACK / "logged_in_review"
QUEUE_REL = Path("data/logged_in_review_queue.csv")
LOG_REL = Path("data/logged_in_capture_log.csv")
LINKS_REL = Path("data/logged_in_comment_link_candidates.csv")
COMMENTS_REL = Path("data/logged_in_visible_comments.csv")
TEXT_ROOT = Path("rendered_text")
HTML_ROOT = Path("rendered_html")
SCREEN_ROOT = Path("screenshots")

USER_WAIT_MS = 1800
CLICK_PATTERNS = [
    "see more",
    "show more",
    "load more",
    "view more",
    "more comments",
    "previous comments",
    "replies",
    "ver mas",
    "ver más",
    "mostrar mas",
    "mostrar más",
    "cargar mas",
    "cargar más",
    "mas comentarios",
    "más comentarios",
    "respuestas",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cdp_available(cdp_url: str) -> bool:
    try:
        with urlopen(f"{cdp_url.rstrip('/')}/json/version", timeout=2) as resp:
            return resp.status == 200
    except (OSError, URLError):
        return False


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def click_expandable_text(page) -> int:
    pattern_json = json.dumps(CLICK_PATTERNS)
    return page.evaluate(
        """(patterns) => {
            const lowerPatterns = patterns;
            let clicked = 0;
            const candidates = Array.from(document.querySelectorAll(
              'button, [role="button"], a, span, div'
            ));
            for (const el of candidates) {
              const text = (el.innerText || el.textContent || '').trim().toLowerCase();
              if (!text || text.length > 120) continue;
              if (!lowerPatterns.some(p => text.includes(p))) continue;
              const rect = el.getBoundingClientRect();
              if (rect.width <= 0 || rect.height <= 0) continue;
              try {
                el.click();
                clicked += 1;
              } catch (_) {}
              if (clicked >= 20) break;
            }
            return clicked;
        }""",
        json.loads(pattern_json),
    )


def expand_post_surface(page, iterations: int) -> int:
    total_clicks = 0
    for _ in range(iterations):
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(USER_WAIT_MS)
        total_clicks += click_expandable_text(page)
        page.wait_for_timeout(USER_WAIT_MS)
    page.mouse.wheel(0, -1200)
    page.wait_for_timeout(800)
    return total_clicks


def extract_rendered_payload(page) -> dict[str, object]:
    return page.evaluate(
        """() => {
            const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
            const bodyText = clean(document.body ? document.body.innerText : '');
            const anchors = Array.from(document.querySelectorAll('a[href]')).map((a) => {
                const rect = a.getBoundingClientRect();
                let context = '';
                let node = a;
                for (let i = 0; i < 4 && node; i += 1) {
                    const text = clean(node.innerText || node.textContent || '');
                    if (text.length > context.length) context = text;
                    node = node.parentElement;
                }
                return {
                    href: a.href,
                    text: clean(a.innerText || a.textContent || ''),
                    context: context.slice(0, 900),
                    visible: rect.width > 0 && rect.height > 0,
                };
            });
            const seen = new Set();
            const comments = [];
            for (const el of document.querySelectorAll('article.comments-comment-entity')) {
                const contentEl = el.querySelector('.comments-comment-item__main-content') ||
                    el.querySelector('.comments-comment-entity__content') ||
                    el;
                const text = clean(contentEl.innerText || contentEl.textContent || '');
                if (text.length < 2 || seen.has(text)) continue;
                seen.add(text);
                const authorEl = el.querySelector('.comments-comment-meta__description-title');
                const headlineEl = el.querySelector('.comments-comment-meta__description-subtitle');
                const timeEl = el.querySelector('time.comments-comment-meta__data');
                const links = Array.from(contentEl.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: clean(a.innerText || a.textContent || ''),
                }));
                comments.push({
                    selector: 'article.comments-comment-entity',
                    comment_urn: el.getAttribute('data-id') || '',
                    is_reply: el.classList.contains('comments-comment-entity--reply'),
                    author: clean(authorEl ? (authorEl.innerText || authorEl.textContent) : ''),
                    author_headline: clean(headlineEl ? (headlineEl.innerText || headlineEl.textContent) : ''),
                    relative_time: clean(timeEl ? (timeEl.innerText || timeEl.textContent) : ''),
                    text: text.slice(0, 3500),
                    links
                });
            }
            return {url: location.href, title: document.title, bodyText, anchors, comments};
        }"""
    )


def looks_authenticated(payload: dict[str, object]) -> bool:
    url = str(payload.get("url", "")).lower()
    body = str(payload.get("bodyText", "")).lower()
    if "/login" in url or "checkpoint" in url or "captcha" in body:
        return False
    login_markers = ["sign in", "join now", "iniciar sesión", "unirte ahora"]
    return not any(marker in body[:2000] for marker in login_markers)


def capture_one(
    page,
    row: dict[str, str],
    pack_dir: Path,
    expand_iterations: int,
    skip_screenshot: bool = False,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    activity_id = row["activity_id"]
    html_dir = pack_dir / HTML_ROOT
    text_dir = pack_dir / TEXT_ROOT
    shot_dir = pack_dir / SCREEN_ROOT
    for directory in (html_dir, text_dir, shot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log = {
        "queue_id": row["queue_id"],
        "activity_id": activity_id,
        "post_url": row["post_url"],
        "capture_status": "",
        "final_url": "",
        "rendered_text_path": "",
        "html_path": "",
        "screenshot_path": "",
        "body_text_chars": "0",
        "comment_blocks": "0",
        "comment_link_count": "0",
        "expand_clicks": "0",
        "error": "",
    }
    link_rows: list[dict[str, str]] = []
    comment_rows: list[dict[str, str]] = []

    try:
        page.goto(row["post_url"], wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(3500)
        expand_clicks = expand_post_surface(page, expand_iterations)
        payload = extract_rendered_payload(page)
        html = page.content()
        body_text = str(payload.get("bodyText", ""))
        authenticated = looks_authenticated(payload)

        html_path = html_dir / f"{activity_id}.html"
        text_path = text_dir / f"{activity_id}.txt"
        screenshot_path = shot_dir / f"{activity_id}.png"
        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(
            "\n".join(
                [
                    f"activity_id: {activity_id}",
                    f"source_url: {row['post_url']}",
                    f"final_url: {payload.get('url', '')}",
                    f"title: {payload.get('title', '')}",
                    f"authenticated_surface: {authenticated}",
                    "",
                    body_text,
                ]
            ),
            encoding="utf-8",
        )
        if not skip_screenshot:
            page.screenshot(path=str(screenshot_path), full_page=True)

        comments = payload.get("comments", []) or []
        anchors = payload.get("anchors", []) or []
        for idx, comment in enumerate(comments, start=1):
            if not isinstance(comment, dict):
                continue
            comment_links = comment.get("links") or []
            comment_rows.append(
                {
                    "activity_id": activity_id,
                    "queue_id": row["queue_id"],
                    "comment_idx": str(idx),
                    "comment_urn": normalize_space(str(comment.get("comment_urn", ""))),
                    "is_reply": str(bool(comment.get("is_reply", False))).lower(),
                    "author": normalize_space(str(comment.get("author", ""))),
                    "author_headline": normalize_space(str(comment.get("author_headline", ""))),
                    "relative_time": normalize_space(str(comment.get("relative_time", ""))),
                    "comment_text": normalize_space(str(comment.get("text", ""))),
                    "link_count": str(len(comment_links)),
                    "links_json": json.dumps(comment_links, ensure_ascii=False),
                }
            )
        link_idx = 0
        for source_kind, items in (("anchor", anchors), ("comment_block", comments)):
            for item in items:
                item_links = item.get("links") if isinstance(item, dict) else None
                if item_links is None:
                    item_links = [item]
                for link in item_links or []:
                    href = normalize_space(str(link.get("href", "")))
                    if not href or href.startswith("javascript:"):
                        continue
                    if (
                        "linkedin.com" not in href
                        and "lnkd.in" not in href
                        and not href.startswith("http")
                    ):
                        continue
                    link_idx += 1
                    context = (
                        normalize_space(str(item.get("context", item.get("text", ""))))
                        if isinstance(item, dict)
                        else ""
                    )
                    link_rows.append(
                        {
                            "activity_id": activity_id,
                            "queue_id": row["queue_id"],
                            "link_idx": str(link_idx),
                            "source_kind": source_kind,
                            "href": href,
                            "anchor_text": normalize_space(str(link.get("text", ""))),
                            "context_excerpt": context[:600],
                        }
                    )

        log.update(
            {
                "capture_status": "logged_in_rendered_capture_complete"
                if authenticated
                else "not_authenticated_or_checkpoint",
                "final_url": str(payload.get("url", "")),
                "rendered_text_path": str(text_path.relative_to(pack_dir)),
                "html_path": str(html_path.relative_to(pack_dir)),
                "screenshot_path": ""
                if skip_screenshot
                else str(screenshot_path.relative_to(pack_dir)),
                "body_text_chars": str(len(body_text)),
                "comment_blocks": str(len(comments)),
                "comment_link_count": str(len(link_rows)),
                "expand_clicks": str(expand_clicks),
            }
        )
        if not authenticated:
            log["error"] = "Rendered page looked unauthenticated, checkpointed, or captcha-gated."
    except (PlaywrightTimeoutError, PlaywrightError, OSError) as exc:
        log["capture_status"] = "capture_error"
        log["error"] = f"{type(exc).__name__}: {exc}"

    return log, link_rows, comment_rows


def parse_limit(raw: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if raw == "all":
        return rows
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    return [row for row in rows if row["queue_id"] in wanted or row["activity_id"] in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=REVIEW_PACK)
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
    parser.add_argument("--items", default="all", help="all, queue ids, or activity ids")
    parser.add_argument("--sleep-seconds", type=float, default=1.25)
    parser.add_argument("--expand-iterations", type=int, default=6)
    parser.add_argument("--skip-screenshot", action="store_true")
    args = parser.parse_args()

    queue_path = args.pack_dir / QUEUE_REL
    rows, queue_fields = read_csv(queue_path)
    selected = parse_limit(args.items, rows)
    if not selected:
        raise SystemExit("No queue rows selected.")

    if not cdp_available(args.cdp_url):
        raise SystemExit(
            f"CDP endpoint unavailable at {args.cdp_url}. Launch a user-owned Chrome session "
            "with --remote-debugging-port=9222 first."
        )

    existing_logs: list[dict[str, str]] = []
    log_path = args.pack_dir / LOG_REL
    if log_path.exists():
        existing_logs, _ = read_csv(log_path)
    selected_ids = {row["activity_id"] for row in selected}
    existing_logs = [row for row in existing_logs if row["activity_id"] not in selected_ids]

    existing_links: list[dict[str, str]] = []
    links_path = args.pack_dir / LINKS_REL
    if links_path.exists():
        existing_links, _ = read_csv(links_path)
    existing_links = [row for row in existing_links if row["activity_id"] not in selected_ids]

    existing_comments: list[dict[str, str]] = []
    comments_path = args.pack_dir / COMMENTS_REL
    if comments_path.exists():
        existing_comments, _ = read_csv(comments_path)
    existing_comments = [row for row in existing_comments if row["activity_id"] not in selected_ids]

    logs: list[dict[str, str]] = []
    links: list[dict[str, str]] = []
    comments: list[dict[str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(args.cdp_url)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        page.set_viewport_size({"width": 1440, "height": 1200})
        for row in selected:
            log, link_rows, comment_rows = capture_one(
                page,
                row,
                args.pack_dir,
                args.expand_iterations,
                skip_screenshot=args.skip_screenshot,
            )
            logs.append(log)
            links.extend(link_rows)
            comments.extend(comment_rows)
            print(
                f"{row['queue_id']} {log['capture_status']} "
                f"comments={log.get('comment_blocks', '0')} "
                f"links={log.get('comment_link_count', '0')}",
                flush=True,
            )
            time.sleep(args.sleep_seconds)
        page.close()
        browser.close()

    write_csv(
        log_path,
        existing_logs + logs,
        [
            "queue_id",
            "activity_id",
            "post_url",
            "capture_status",
            "final_url",
            "rendered_text_path",
            "html_path",
            "screenshot_path",
            "body_text_chars",
            "comment_blocks",
            "comment_link_count",
            "expand_clicks",
            "error",
        ],
    )
    write_csv(
        links_path,
        existing_links + links,
        [
            "activity_id",
            "queue_id",
            "link_idx",
            "source_kind",
            "href",
            "anchor_text",
            "context_excerpt",
        ],
    )
    write_csv(
        comments_path,
        existing_comments + comments,
        [
            "activity_id",
            "queue_id",
            "comment_idx",
            "comment_urn",
            "is_reply",
            "author",
            "author_headline",
            "relative_time",
            "comment_text",
            "link_count",
            "links_json",
        ],
    )

    # Update queue status without losing row order.
    log_by_id = {row["activity_id"]: row for row in logs}
    for row in rows:
        log = log_by_id.get(row["activity_id"])
        if not log:
            continue
        row["logged_in_capture_status"] = log["capture_status"]
        row["comments_status"] = (
            "comments_or_links_captured"
            if int(log.get("comment_blocks") or 0) or int(log.get("comment_link_count") or 0)
            else "no_visible_comment_blocks_after_expansion"
        )
    write_csv(queue_path, rows, queue_fields)
    print(
        f"Captured {len(logs)} logged-in rows; "
        f"extracted {len(comments)} comments and {len(links)} candidate links."
    )


if __name__ == "__main__":
    main()
