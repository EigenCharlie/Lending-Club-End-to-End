#!/usr/bin/env python3
"""Capture public LinkedIn article pages listed in a research pack.

This is intentionally separate from the post capture script because LinkedIn
articles do not expose numeric activity IDs in the same way as feed posts. The
script stores raw HTML, cleaned article text, and public article cover/inline
images for later manual/visual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov/second_ingest")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag in {"br", "p", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4"}:
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


def fetch(url: str, timeout: int = 30) -> tuple[bytes, str, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("content-type", ""), resp.status, resp.geturl()


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "article"


def clean_url(raw_url: str) -> str:
    url = html.unescape(raw_url)
    url = url.replace("\\u002F", "/").replace("\\/", "/")
    for sep in ['"', "'", "<", ">", "\\\\"]:
        if sep in url:
            url = url.split(sep, 1)[0]
    return url.rstrip(".,)]}")


def strip_tags(page_html: str) -> str:
    parser = TextExtractor()
    parser.feed(page_html)
    return parser.text()


def trim_article_text(full_text: str, title: str) -> str:
    lines = full_text.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        if line == "Report this article":
            start = max(0, idx - 1)
            break
    if start == 0:
        title_matches = [idx for idx, line in enumerate(lines) if line == title]
        if title_matches:
            start = title_matches[-1]

    end = len(lines)
    for idx, line in enumerate(lines[start:], start=start):
        if line in {"Explore topics", "Others also viewed", "More articles by this author"}:
            end = idx
            break

    kept = lines[start:end]
    while kept and kept[0] in {"Report this article", "LinkedIn"}:
        kept.pop(0)
    return "\n".join(kept).strip()


def extract_article_image_urls(page_html: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r"https://media\.licdn\.com/dms/image/[^\"<\\]+", page_html):
        url = clean_url(match.group(0))
        if not any(token in url for token in ("article-cover_image", "article-inline_image")):
            continue
        if url not in urls:
            urls.append(url)
    return urls


def extension_from_content_type(content_type: str, default: str = ".jpg") -> str:
    lowered = content_type.lower()
    if "png" in lowered:
        return ".png"
    if "webp" in lowered:
        return ".webp"
    if "jpeg" in lowered or "jpg" in lowered:
        return ".jpg"
    return default


def relative(path: Path, pack_dir: Path) -> str:
    return str(path.relative_to(pack_dir))


def capture_article(
    article: dict[str, str], pack_dir: Path, sleep_seconds: float
) -> tuple[dict[str, str], list[dict[str, str]]]:
    article_id = article["article_id"]
    raw_dir = pack_dir / "articles" / "raw" / safe_name(article_id)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "article_id": article_id,
        "source_url": article["source_url"],
        "title": article["title"],
        "parent_activity_id": article.get("parent_activity_id", ""),
        "capture_status": "",
        "final_url": "",
        "local_html_path": "",
        "local_text_path": "",
        "text_length": "0",
        "image_count": "0",
        "error": "",
    }
    assets: list[dict[str, str]] = []
    try:
        body, content_type, status, final_url = fetch(article["source_url"])
        page_html = body.decode("utf-8", errors="replace")
        html_path = raw_dir / "article.html"
        html_path.write_text(page_html, encoding="utf-8")
        full_text = strip_tags(page_html)
        article_text = trim_article_text(full_text, article["title"])
        text_path = raw_dir / "article_text.txt"
        text_path.write_text(article_text, encoding="utf-8")

        log.update(
            {
                "capture_status": f"captured_http_{status}_{content_type.split(';')[0]}",
                "final_url": final_url,
                "local_html_path": relative(html_path, pack_dir),
                "local_text_path": relative(text_path, pack_dir),
                "text_length": str(len(article_text)),
            }
        )

        image_urls = extract_article_image_urls(page_html)
        for idx, image_url in enumerate(image_urls, start=1):
            image_bytes, image_type, _, _ = fetch(image_url)
            image_path = (
                raw_dir / f"article_image_{idx:02d}{extension_from_content_type(image_type)}"
            )
            image_path.write_bytes(image_bytes)
            assets.append(
                {
                    "article_id": article_id,
                    "asset_id": f"{article_id}_image_{idx:02d}",
                    "asset_type": "linkedin_article_image",
                    "source_url": image_url,
                    "local_path": relative(image_path, pack_dir),
                    "checksum": checksum(image_path),
                    "visual_read_status": "pending_manual_visual_read",
                    "analytic_memo": "Public LinkedIn article cover/inline image captured; use with article text for interpretation.",
                }
            )
            time.sleep(sleep_seconds)
        log["image_count"] = str(len(assets))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        log["capture_status"] = "capture_error"
        log["error"] = f"{type(exc).__name__}: {exc}"
    return log, assets


def capture_articles(pack_dir: Path, sleep_seconds: float) -> None:
    articles, _ = read_csv(pack_dir / "data" / "article_candidates.csv")
    logs: list[dict[str, str]] = []
    assets: list[dict[str, str]] = []
    for article in articles:
        log, article_assets = capture_article(article, pack_dir, sleep_seconds)
        logs.append(log)
        assets.extend(article_assets)
        time.sleep(sleep_seconds)

    write_csv(
        pack_dir / "data" / "article_capture_log.csv",
        logs,
        [
            "article_id",
            "source_url",
            "title",
            "parent_activity_id",
            "capture_status",
            "final_url",
            "local_html_path",
            "local_text_path",
            "text_length",
            "image_count",
            "error",
        ],
    )
    write_csv(
        pack_dir / "data" / "article_asset_manifest.csv",
        assets,
        [
            "article_id",
            "asset_id",
            "asset_type",
            "source_url",
            "local_path",
            "checksum",
            "visual_read_status",
            "analytic_memo",
        ],
    )
    print(
        f"Captured {sum(1 for row in logs if row['capture_status'].startswith('captured'))} articles and {len(assets)} images"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_articles(args.pack_dir, args.sleep_seconds)


if __name__ == "__main__":
    main()
