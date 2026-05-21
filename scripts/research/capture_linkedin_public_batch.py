#!/usr/bin/env python3
"""Capture public LinkedIn permalink artifacts for selected Denis Burakov posts.

This uses only public permalink HTML and public `media.licdn.com` assets exposed
by those pages. It does not read browser cookies, credentials, or private Chrome
state. Raw HTML, post text, LinkedIn document transcripts/PDFs, and feedshare
images are saved under the private research pack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov")
DEFAULT_POST_NUMBERS = "1,3,4,5,6,8,14,15,20,22,31,32,35,45,55,58"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag in {"br", "p", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "li"}:
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


def fetch_bytes(url: str, timeout: int = 30) -> tuple[bytes, str, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("content-type", ""), resp.status, resp.geturl()


def fetch_text(url: str, timeout: int = 30) -> tuple[str, str, int, str]:
    data, content_type, status, final_url = fetch_bytes(url, timeout=timeout)
    return data.decode("utf-8", errors="replace"), content_type, status, final_url


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_tags(fragment: str) -> str:
    parser = TextExtractor()
    parser.feed(fragment)
    return parser.text()


def extract_post_text(page_html: str) -> str:
    match = re.search(
        r'<p[^>]+data-test-id="main-feed-activity-card__commentary"[^>]*>(.*?)</p>',
        page_html,
        flags=re.DOTALL,
    )
    if match:
        return strip_tags(match.group(1))

    meta = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page_html, flags=re.DOTALL)
    if meta:
        return html.unescape(meta.group(1)).strip()
    return ""


def extract_main_article(page_html: str) -> str:
    commentary = page_html.find('data-test-id="main-feed-activity-card__commentary"')
    if commentary == -1:
        return page_html
    start = page_html.rfind("<article", 0, commentary)
    if start == -1:
        start = commentary
    end = page_html.find("</article>", commentary)
    if end == -1:
        return page_html[start:]
    return page_html[start : end + len("</article>")]


def clean_url(raw_url: str) -> str:
    url = html.unescape(raw_url)
    for sep in ['"', "<", "\\\\"]:
        if sep in url:
            url = url.split(sep, 1)[0]
    return url.rstrip("}])")


def extract_document_manifest_urls(page_html: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(
        r"https://media\.licdn\.com/dms/document/[^\"<\\]+feedshare-document-master-manifest[^\"<\\]+",
        page_html,
    ):
        url = clean_url(match.group(0))
        if url not in urls:
            urls.append(url)
    return urls


def extract_feedshare_image_urls(page_html: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r"https://media\.licdn\.com/dms/image/[^\"<\\]+", page_html):
        url = clean_url(match.group(0))
        if not any(
            token in url for token in ["feedshare-shrink", "feedshare-document-cover-images"]
        ):
            continue
        if url not in urls:
            urls.append(url)
    return urls


def extension_from_content_type(content_type: str, default: str) -> str:
    lowered = content_type.lower()
    if "png" in lowered:
        return ".png"
    if "jpeg" in lowered or "jpg" in lowered:
        return ".jpg"
    if "webp" in lowered:
        return ".webp"
    if "pdf" in lowered:
        return ".pdf"
    if "json" in lowered:
        return ".json"
    return default


def update_manifest_row(
    manifest_rows: list[dict[str, str]],
    activity_id: str,
    local_path: str,
    source_url: str,
    asset_type: str,
    page_or_slide_count: str,
    ocr_status: str,
    text_extract_status: str,
    analytic_memo: str,
    pack_dir: Path,
) -> None:
    primary = next(
        (
            row
            for row in manifest_rows
            if row["activity_id"] == activity_id and row["asset_id"].endswith("_primary")
        ),
        None,
    )
    row = primary or {
        "activity_id": activity_id,
        "asset_id": f"{activity_id}_primary",
        "asset_type": asset_type,
        "source_url": source_url,
        "local_path": "",
        "page_or_slide_count": "",
        "ocr_status": "",
        "checksum": "",
        "text_extract_status": "",
        "analytic_memo": "",
    }
    existing_manual_read = row.get("ocr_status") == "manual_visual_read_completed"
    existing_count = row.get("page_or_slide_count", "")
    existing_text_status = row.get("text_extract_status", "")
    existing_memo = row.get("analytic_memo", "")
    row["asset_type"] = asset_type
    row["source_url"] = source_url
    row["local_path"] = local_path
    row["page_or_slide_count"] = (
        existing_count if existing_manual_read and existing_count else page_or_slide_count
    )
    row["ocr_status"] = "manual_visual_read_completed" if existing_manual_read else ocr_status
    row["checksum"] = checksum(pack_dir / local_path)
    row["text_extract_status"] = (
        existing_text_status
        if existing_manual_read and existing_text_status
        else text_extract_status
    )
    row["analytic_memo"] = (
        existing_memo if existing_manual_read and existing_memo else analytic_memo
    )
    if primary is None:
        manifest_rows.append(row)


def append_aux_asset(
    manifest_rows: list[dict[str, str]],
    activity_id: str,
    asset_id: str,
    local_path: str,
    source_url: str,
    asset_type: str,
    pack_dir: Path,
    page_or_slide_count: str = "",
    ocr_status: str = "not_applicable",
    text_extract_status: str = "captured",
    analytic_memo: str = "",
) -> None:
    row = next((item for item in manifest_rows if item["asset_id"] == asset_id), None)
    if row is None:
        row = {
            "activity_id": activity_id,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "source_url": source_url,
            "local_path": local_path,
            "page_or_slide_count": page_or_slide_count,
            "ocr_status": ocr_status,
            "checksum": "",
            "text_extract_status": text_extract_status,
            "analytic_memo": analytic_memo,
        }
        manifest_rows.append(row)
    existing_manual_read = row.get("ocr_status") == "manual_visual_read_completed"
    existing_count = row.get("page_or_slide_count", "")
    existing_text_status = row.get("text_extract_status", "")
    existing_memo = row.get("analytic_memo", "")
    row.update(
        {
            "asset_type": asset_type,
            "source_url": source_url,
            "local_path": local_path,
            "page_or_slide_count": existing_count
            if existing_manual_read and existing_count
            else page_or_slide_count,
            "ocr_status": "manual_visual_read_completed" if existing_manual_read else ocr_status,
            "checksum": checksum(pack_dir / local_path),
            "text_extract_status": existing_text_status
            if existing_manual_read and existing_text_status
            else text_extract_status,
            "analytic_memo": existing_memo
            if existing_manual_read and existing_memo
            else analytic_memo,
        }
    )


def relative(path: Path, pack_dir: Path) -> str:
    return str(path.relative_to(pack_dir))


def capture_post(
    post: dict[str, str],
    pack_dir: Path,
    manifest_rows: list[dict[str, str]],
    sleep_seconds: float,
) -> dict[str, str]:
    activity_id = post["activity_id"]
    raw_dir = pack_dir / "attachments" / "raw" / activity_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log: dict[str, str] = {
        "n": post["n"],
        "activity_id": activity_id,
        "post_url": post["post_url"],
        "capture_status": "",
        "post_text_path": "",
        "html_path": "",
        "document_count": "0",
        "image_count": "0",
        "error": "",
    }
    try:
        page_html, content_type, status, final_url = fetch_text(post["post_url"])
        html_path = raw_dir / "post.html"
        html_path.write_text(page_html, encoding="utf-8")
        log["html_path"] = relative(html_path, pack_dir)

        post_text = extract_post_text(page_html)
        text_path = raw_dir / "post_text.txt"
        text_path.write_text(post_text, encoding="utf-8")
        log["post_text_path"] = relative(text_path, pack_dir)

        main_article = extract_main_article(page_html)
        document_urls = extract_document_manifest_urls(main_article)
        image_urls = extract_feedshare_image_urls(main_article)
        log["document_count"] = str(len(document_urls))
        log["image_count"] = str(len(image_urls))

        if document_urls:
            for idx, manifest_url in enumerate(document_urls, start=1):
                manifest_text, _, _, _ = fetch_text(manifest_url)
                manifest = json.loads(manifest_text)
                manifest_path = raw_dir / f"document_manifest_{idx:02d}.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                transcript_pages: list[str] = []
                transcript_url = manifest.get("transcriptManifestUrl", "")
                if transcript_url:
                    transcript_text, _, _, _ = fetch_text(transcript_url)
                    transcript = json.loads(transcript_text)
                    transcript_path = raw_dir / f"document_transcript_{idx:02d}.json"
                    transcript_path.write_text(
                        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    transcript_pages = [str(page) for page in transcript.get("pages", [])]
                    transcript_txt_path = raw_dir / f"document_transcript_{idx:02d}.txt"
                    transcript_txt_path.write_text(
                        "\n\n--- PAGE BREAK ---\n\n".join(transcript_pages),
                        encoding="utf-8",
                    )
                    append_aux_asset(
                        manifest_rows,
                        activity_id,
                        f"{activity_id}_document_transcript_{idx:02d}",
                        relative(transcript_txt_path, pack_dir),
                        transcript_url,
                        "linkedin_document_transcript",
                        pack_dir,
                        page_or_slide_count=str(len(transcript_pages)),
                        ocr_status="not_needed_linkedin_transcript",
                        text_extract_status="transcript_captured",
                        analytic_memo="LinkedIn document transcript captured from public media manifest.",
                    )

                pdf_url = manifest.get("transcribedDocumentUrl", "")
                if pdf_url:
                    pdf_bytes, pdf_type, _, _ = fetch_bytes(pdf_url)
                    pdf_path = (
                        raw_dir
                        / f"document_{idx:02d}{extension_from_content_type(pdf_type, '.pdf')}"
                    )
                    pdf_path.write_bytes(pdf_bytes)
                    update_manifest_row(
                        manifest_rows,
                        activity_id,
                        relative(pdf_path, pack_dir),
                        pdf_url,
                        "linkedin_document_pdf",
                        str(len(transcript_pages)) if transcript_pages else "",
                        "not_needed_linkedin_transcript",
                        "pdf_captured_from_public_manifest",
                        "PDF captured from LinkedIn public document media manifest; transcript sidecar also captured when available.",
                        pack_dir,
                    )
                else:
                    update_manifest_row(
                        manifest_rows,
                        activity_id,
                        relative(manifest_path, pack_dir),
                        manifest_url,
                        "linkedin_document_manifest_only",
                        str(len(transcript_pages)) if transcript_pages else "",
                        "transcript_captured" if transcript_pages else "pending",
                        "document_manifest_captured_without_pdf_url",
                        "Document manifest captured, but no PDF URL was exposed.",
                        pack_dir,
                    )
                time.sleep(sleep_seconds)
        elif image_urls:
            first_image_path: Path | None = None
            for idx, image_url in enumerate(image_urls, start=1):
                image_bytes, image_type, _, _ = fetch_bytes(image_url)
                image_path = (
                    raw_dir / f"image_{idx:02d}{extension_from_content_type(image_type, '.jpg')}"
                )
                if first_image_path is None:
                    first_image_path = image_path
                image_path.write_bytes(image_bytes)
                append_aux_asset(
                    manifest_rows,
                    activity_id,
                    f"{activity_id}_image_{idx:02d}",
                    relative(image_path, pack_dir),
                    image_url,
                    "linkedin_feedshare_image",
                    pack_dir,
                    page_or_slide_count="1 image",
                    ocr_status="pending_image_reading_or_ocr",
                    text_extract_status="image_captured",
                    analytic_memo="Feedshare image captured from public permalink HTML.",
                )
                time.sleep(sleep_seconds)
            update_manifest_row(
                manifest_rows,
                activity_id,
                relative(first_image_path, pack_dir)
                if first_image_path
                else relative(raw_dir, pack_dir),
                image_urls[0],
                "linkedin_image_carousel",
                str(len(image_urls)),
                "pending_image_reading_or_ocr",
                "images_captured_from_public_permalink",
                "Feedshare image assets captured; OCR/manual image reading still required before evidence use.",
                pack_dir,
            )
        else:
            update_manifest_row(
                manifest_rows,
                activity_id,
                relative(text_path, pack_dir),
                post["post_url"],
                "linkedin_post_text_only",
                "",
                "not_applicable",
                "post_text_captured_no_media_found",
                "Public permalink text captured; no document/image asset detected in HTML.",
                pack_dir,
            )

        log["capture_status"] = f"captured_http_{status}_{content_type.split(';')[0]}"
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        log["capture_status"] = "capture_error"
        log["error"] = f"{type(exc).__name__}: {exc}"
    return log


def update_inventory(pack_dir: Path, logs: list[dict[str, str]]) -> None:
    inventory_path = pack_dir / "data" / "linkedin_corpus_inventory.csv"
    rows, fieldnames = read_csv(inventory_path)
    by_id = {row["activity_id"]: row for row in rows}
    for log in logs:
        row = by_id.get(log["activity_id"])
        if not row:
            continue
        if log["capture_status"].startswith("captured"):
            row["capture_status"] = "public_permalink_captured"
            row["completeness_status"] = (
                "complete_public_document_transcript_pending_reading_memo"
                if int(log.get("document_count") or 0) > 0
                else "partial_images_captured_pending_image_reading_or_ocr"
                if int(log.get("image_count") or 0) > 0
                else "partial_post_text_captured_no_media"
            )
            row["blocker_or_next_action"] = (
                "Read extracted post text/assets and write concept-level Spanish analytic memo."
            )
        else:
            row["capture_status"] = "public_permalink_capture_error"
            row["blocker_or_next_action"] = log["error"]
    write_csv(inventory_path, rows, fieldnames)


def parse_numbers(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def capture_batch(pack_dir: Path, post_numbers: set[str], sleep_seconds: float) -> None:
    posts, _ = read_csv(pack_dir / "data" / "posts_index.csv")
    selected = (
        posts if "all" in post_numbers else [post for post in posts if post["n"] in post_numbers]
    )
    selected_ids = {post["activity_id"] for post in selected}
    manifest_path = pack_dir / "data" / "attachment_manifest.csv"
    manifest_rows, manifest_fields = read_csv(manifest_path)
    manifest_rows = [
        row
        for row in manifest_rows
        if not (
            row["activity_id"] in selected_ids
            and (
                row["asset_id"].endswith("_primary")
                or "_document_transcript_" in row["asset_id"]
                or "_image_" in row["asset_id"]
            )
            and row.get("ocr_status") != "manual_visual_read_completed"
        )
    ]

    logs = [
        capture_post(post, pack_dir, manifest_rows, sleep_seconds=sleep_seconds)
        for post in selected
    ]
    write_csv(manifest_path, manifest_rows, manifest_fields)
    update_inventory(pack_dir, logs)

    log_path = pack_dir / "data" / "public_permalink_capture_log.csv"
    if log_path.exists():
        existing_logs, _ = read_csv(log_path)
    else:
        existing_logs = []
    combined_logs = [row for row in existing_logs if row["activity_id"] not in selected_ids] + logs
    combined_logs.sort(key=lambda row: int(row["n"]))
    write_csv(
        log_path,
        combined_logs,
        [
            "n",
            "activity_id",
            "post_url",
            "capture_status",
            "post_text_path",
            "html_path",
            "document_count",
            "image_count",
            "error",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--post-numbers", default=DEFAULT_POST_NUMBERS)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_batch(args.pack_dir, parse_numbers(args.post_numbers), args.sleep_seconds)


if __name__ == "__main__":
    main()
