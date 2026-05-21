#!/usr/bin/env python3
"""Resolve and snapshot external sources referenced by the LinkedIn research pack.

Input is `data/attachment_manifest.csv`, not the deduplicated source log, so the
109 post-level link references remain traceable to their parent posts. The script
handles LinkedIn shortlink interstitial pages by extracting the actual outbound
target embedded in the page HTML, then writes a per-link backlog and local raw
artifacts for pages that can be fetched without authentication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
EXCLUDED_LINK_HOSTS = {
    "static.licdn.com",
    "www.linkedin.com",
    "linkedin.com",
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag in {"br", "p", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.parts))
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch(url: str, timeout: int = 25) -> tuple[bytes, str, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("content-type", ""), resp.status, resp.geturl()


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_tags(page_html: str) -> str:
    parser = TextExtractor()
    parser.feed(page_html)
    return parser.text()


def clean_url(raw_url: str) -> str:
    url = html.unescape(raw_url)
    url = url.replace("\\u002F", "/").replace("\\/", "/")
    for sep in ['"', "'", "<", ">", "\\\\"]:
        if sep in url:
            url = url.split(sep, 1)[0]
    return url.rstrip(".,)]}")


def extract_title(page_html: str) -> str:
    patterns = [
        r'<meta\s+property="og:title"\s+content="([^"]*)"',
        r'<meta\s+name="twitter:title"\s+content="([^"]*)"',
        r"<title[^>]*>(.*?)</title>",
    ]
    for pattern in patterns:
        match = re.search(pattern, page_html, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
    return ""


def extract_shortlink_target(page_html: str, source_url: str) -> str:
    candidates: list[str] = []
    for raw in re.findall(r"https?://[^\"'<>\s]+", page_html):
        url = clean_url(raw)
        host = urlparse(url).netloc.lower()
        if url == source_url:
            continue
        if host in {"static.licdn.com"}:
            continue
        if "linkedin.com/help/" in url:
            continue
        if (
            host in {"www.linkedin.com", "linkedin.com"}
            and url.rstrip("/") == "https://www.linkedin.com"
        ):
            continue
        if url not in candidates:
            candidates.append(url)

    non_linkedin = [url for url in candidates if "linkedin.com" not in urlparse(url).netloc.lower()]
    if non_linkedin:
        return non_linkedin[0]
    if candidates:
        return candidates[0]
    return source_url


def resolve_interstitial_chain(
    source_url: str, page_html: str, max_hops: int = 4
) -> tuple[str, bytes, str, str, str]:
    canonical_url = extract_shortlink_target(page_html, source_url)
    body = page_html.encode("utf-8")
    content_type = "text/html"
    status = ""
    error = ""
    seen = {source_url}
    hops = 0

    while canonical_url not in seen and hops < max_hops:
        seen.add(canonical_url)
        hops += 1
        try:
            body, content_type, status_code, final_url = fetch(canonical_url)
            status = str(status_code)
            canonical_url = final_url if final_url else canonical_url
            if urlparse(canonical_url).netloc.lower() == "lnkd.in":
                next_html = body.decode("utf-8", errors="replace")
                next_url = extract_shortlink_target(next_html, canonical_url)
                if next_url == canonical_url:
                    break
                canonical_url = next_url
                continue
            break
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    return canonical_url, body, content_type, status, error


def source_type(url: str, content_type: str = "") -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host == "lnkd.in":
        return "linkedin_shortlink_unresolved"
    if "github.com" in host or "raw.githubusercontent.com" in host:
        return "github"
    if "medium.com" in host:
        return "blog_medium"
    if "linktr.ee" in host or host == "tr.ee":
        return "link_aggregator"
    if "matplotlib.org" in host:
        return "official_documentation"
    if "arxiv.org" in host:
        return "preprint"
    if "doi.org" in host:
        return "doi"
    if "linkedin.com" in host:
        if "/posts/" in path or "/feed/update/" in path or "/pulse/" in path:
            return "linkedin_post_or_article"
        if "/in/" in path:
            return "linkedin_profile"
        if "/company/" in path or "/showcase/" in path:
            return "linkedin_company"
        return "linkedin_web"
    if "pdf" in content_type.lower() or path.endswith(".pdf"):
        return "pdf"
    return "external_web"


def handling_for(canonical_url: str, kind: str) -> tuple[str, str, str]:
    if kind == "linkedin_post_or_article":
        return (
            "associate_or_spawn_linkedin_child",
            "Capture/read as child post if not already in the 59-post index; LinkedIn-only unless external sources are supplied.",
            "Stop when the linked post is either matched to an indexed activity or logged as child backlog with accessible text/assets.",
        )
    if kind in {"linkedin_profile", "linkedin_company", "linkedin_web"}:
        return (
            "archive_identity_or_context_source",
            "Do not use as evidence for technical claims; keep only for provenance/context.",
            "Stop after canonical URL, status, and non-claim role are recorded.",
        )
    if kind in {"github", "official_documentation", "preprint", "doi", "pdf"}:
        return (
            "read_as_potential_evidence",
            "Read before promotion; may support appendix, implementation, or related work depending on content.",
            "Stop when local snapshot exists, source status is labeled, and parent post decision is updated.",
        )
    if kind in {"blog_medium", "external_web", "link_aggregator"}:
        return (
            "read_or_triage_as_non_peer_reviewed_context",
            "Use only as implementation/context unless it links to canonical academic or official material.",
            "Stop when resolved links are either attached to parent post, expanded as child sources, or archived as low-evidence context.",
        )
    return (
        "manual_review",
        "Unknown source class; review before evidence use.",
        "Stop when source is classified or marked inaccessible.",
    )


def safe_name(asset_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", asset_id)


def resolve_sources(pack_dir: Path, sleep_seconds: float) -> None:
    data_dir = pack_dir / "data"
    manifest_rows, manifest_fields = read_csv(data_dir / "attachment_manifest.csv")
    post_rows, _ = read_csv(data_dir / "posts_index.csv")
    post_number_by_activity = {row["activity_id"]: row["n"] for row in post_rows}
    existing_ids = {row["activity_id"] for row in post_rows}
    external_rows = [row for row in manifest_rows if row["asset_type"] == "external_link"]

    raw_root = pack_dir / "external_sources" / "raw"
    rows: list[dict[str, str]] = []
    child_posts: dict[str, dict[str, str]] = {}

    for row in external_rows:
        source_url = row["source_url"]
        canonical_url = source_url
        status = ""
        content_type = ""
        title = ""
        local_html_path = ""
        local_text_path = ""
        local_binary_path = ""
        access_status = ""
        error = ""
        body = b""

        try:
            body, content_type, status_code, final_url = fetch(source_url)
            status = str(status_code)
            page_html = body.decode("utf-8", errors="replace")
            canonical_url = final_url
            if urlparse(source_url).netloc.lower() == "lnkd.in":
                canonical_url, body, content_type, chained_status, chained_error = (
                    resolve_interstitial_chain(source_url, page_html)
                )
                if chained_status:
                    status = chained_status
                page_html = body.decode("utf-8", errors="replace")
                if chained_error:
                    access_status = "target_resolution_only_fetch_blocked"
                    error = chained_error
            kind = source_type(canonical_url, content_type)
            if not access_status:
                access_status = "reachable"

            target_dir = raw_root / safe_name(row["asset_id"])
            target_dir.mkdir(parents=True, exist_ok=True)
            if "pdf" in content_type.lower() or canonical_url.lower().split("?", 1)[0].endswith(
                ".pdf"
            ):
                suffix = ".pdf"
                binary_path = target_dir / f"{safe_name(row['asset_id'])}{suffix}"
                binary_path.write_bytes(body)
                local_binary_path = str(binary_path.relative_to(pack_dir))
            else:
                html_path = target_dir / "source.html"
                text_path = target_dir / "source_text.txt"
                html_path.write_text(page_html, encoding="utf-8")
                text_path.write_text(strip_tags(page_html), encoding="utf-8")
                local_html_path = str(html_path.relative_to(pack_dir))
                local_text_path = str(text_path.relative_to(pack_dir))
                title = extract_title(page_html)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            kind = source_type(canonical_url)
            access_status = "blocked_or_http_error"
            error = f"{type(exc).__name__}: {exc}"

        handling, use_rule, stop_condition = handling_for(canonical_url, kind)
        checksum_value = ""
        for rel in [local_binary_path, local_text_path, local_html_path]:
            if rel:
                checksum_value = checksum(pack_dir / rel)
                break

        link_row = {
            "link_asset_id": row["asset_id"],
            "parent_activity_id": row["activity_id"],
            "parent_post_number": post_number_by_activity.get(row["activity_id"], ""),
            "source_url": source_url,
            "canonical_url": canonical_url,
            "http_status": status,
            "source_type": kind,
            "access_status": access_status,
            "title": title,
            "local_html_path": local_html_path,
            "local_text_path": local_text_path,
            "local_binary_path": local_binary_path,
            "checksum": checksum_value,
            "handling_decision": handling,
            "claim_use_rule": use_rule,
            "stop_condition": stop_condition,
            "error": error,
        }
        rows.append(link_row)

        if kind == "linkedin_post_or_article":
            activity_match = re.search(
                r"activity[-:]([0-9]{12,})|activity-([0-9]{12,})", canonical_url
            )
            activity_id = (
                next((group for group in activity_match.groups() if group), "")
                if activity_match
                else ""
            )
            if activity_id and activity_id not in existing_ids and activity_id not in child_posts:
                child_posts[activity_id] = {
                    "backlog_id": f"EXT-LI-{len(child_posts) + 1:03d}",
                    "source_kind": "external_linkedin_child_post",
                    "parent_activity_id": row["activity_id"],
                    "activity_id": activity_id,
                    "post_url": canonical_url,
                    "title": title,
                    "theme": "linked external LinkedIn post",
                    "source_status": "linkedin_only_child_source_pending_capture",
                    "handling_decision": "capture_if_relevant_or_archive",
                    "stop_condition": "Stop when child post is captured/read or explicitly archived as context-only.",
                }
        time.sleep(sleep_seconds)

    fields = [
        "link_asset_id",
        "parent_activity_id",
        "parent_post_number",
        "source_url",
        "canonical_url",
        "http_status",
        "source_type",
        "access_status",
        "title",
        "local_html_path",
        "local_text_path",
        "local_binary_path",
        "checksum",
        "handling_decision",
        "claim_use_rule",
        "stop_condition",
        "error",
    ]
    write_csv(data_dir / "external_link_backlog.csv", rows, fields)
    write_csv(
        data_dir / "external_linkedin_child_post_backlog.csv",
        list(child_posts.values()),
        [
            "backlog_id",
            "source_kind",
            "parent_activity_id",
            "activity_id",
            "post_url",
            "title",
            "theme",
            "source_status",
            "handling_decision",
            "stop_condition",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_sources(args.pack_dir, args.sleep_seconds)


if __name__ == "__main__":
    main()
