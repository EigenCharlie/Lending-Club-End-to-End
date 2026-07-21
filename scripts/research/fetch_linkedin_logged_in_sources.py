#!/usr/bin/env python3
"""Fetch and extract high-priority sources from the logged-in LinkedIn review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

PACK = Path("reports/linkedin_credit_risk_denis_burakov/logged_in_review")
INVENTORY = Path("data/logged_in_external_link_inventory.csv")
READING_STATUS = Path("data/logged_in_source_reading_status.csv")
RAW_DIR = Path("sources/raw")
TEXT_DIR = Path("sources/text")
MEMO_PATH = Path("docs/logged_in_source_reading_memo_2026-05-21.md")


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip = True
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip = False
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        text = html.unescape(" ".join(self.parts))
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def slug(value: str, max_len: int = 72) -> str:
    parsed = urlparse(value)
    base = f"{parsed.netloc}{parsed.path}".lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if not base:
        base = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return base[:max_len].strip("-")


def arxiv_pdf_url(url: str) -> str | None:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", url)
    if not match:
        return None
    arxiv_id = match.group(1).removesuffix(".pdf")
    return f"https://arxiv.org/pdf/{arxiv_id}"


def google_export_url(url: str) -> str | None:
    if "docs.google.com/spreadsheets/d/" in url:
        match = re.search(r"/spreadsheets/d/([^/]+)", url)
        if match:
            return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"
    if "docs.google.com/presentation/d/" in url:
        match = re.search(r"/presentation/d/([^/]+)", url)
        if match:
            return f"https://docs.google.com/presentation/d/{match.group(1)}/export/pdf"
    if "drive.google.com/file/d/" in url:
        match = re.search(r"/file/d/([^/]+)", url)
        if match:
            return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return None


def target_url(url: str) -> str:
    return arxiv_pdf_url(url) or google_export_url(url) or url


def extract_pdf(raw_path: Path, text_path: Path) -> tuple[str, str]:
    info = ""
    try:
        pdfinfo = subprocess.run(
            ["pdfinfo", str(raw_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        pages = ""
        for line in pdfinfo.stdout.splitlines():
            if line.startswith("Pages:"):
                pages = line.split(":", 1)[1].strip()
                break
        info = f"pages={pages}" if pages else ""
    except Exception as exc:  # noqa: BLE001
        info = f"pdfinfo_error={type(exc).__name__}: {exc}"
    try:
        subprocess.run(
            ["pdftotext", "-layout", str(raw_path), str(text_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return "text_extracted", info
    except Exception as exc:  # noqa: BLE001
        return "pdf_text_error", f"{info}; {type(exc).__name__}: {exc}"


def extract_html(raw_bytes: bytes, text_path: Path) -> str:
    parser = TextExtractor()
    parser.feed(raw_bytes.decode("utf-8", errors="ignore"))
    text = parser.text()
    text_path.write_text(text, encoding="utf-8")
    return text


def fetch_one(row: dict[str, str], pack: Path, timeout: float) -> dict[str, str]:
    url = target_url(row["canonical_url"])
    raw_dir = pack / RAW_DIR
    text_dir = pack / TEXT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    base = f"{row['link_id']}_{slug(row['canonical_url'])}"
    result = {
        "link_id": row["link_id"],
        "queue_id": row["queue_id"],
        "activity_id": row["activity_id"],
        "source_type": row["source_type"],
        "canonical_url": row["canonical_url"],
        "fetch_url": url,
        "http_status": "",
        "content_type": "",
        "raw_path": "",
        "text_path": "",
        "byte_count": "0",
        "text_chars": "0",
        "extraction_status": "",
        "evidence_status": "",
        "notes": "",
    }
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        result["http_status"] = str(response.status_code)
        result["content_type"] = response.headers.get("content-type", "")
        content = response.content
        result["byte_count"] = str(len(content))
        if response.status_code >= 400:
            result["extraction_status"] = "fetch_blocked_or_missing"
            result["evidence_status"] = "blocked"
            result["notes"] = response.reason
            return result

        content_type = result["content_type"].lower()
        is_pdf = (
            "application/pdf" in content_type
            or urlparse(url).path.lower().endswith(".pdf")
            or content.startswith(b"%PDF")
        )
        raw_path = raw_dir / f"{base}.{'pdf' if is_pdf else 'html'}"
        raw_path.write_bytes(content)
        result["raw_path"] = str(raw_path.relative_to(pack))
        text_path = text_dir / f"{base}.txt"
        if is_pdf:
            status, notes = extract_pdf(raw_path, text_path)
            result["extraction_status"] = status
            result["notes"] = notes
        else:
            text = extract_html(content, text_path)
            result["extraction_status"] = "html_text_extracted"
            result["notes"] = f"title_or_text_start={text[:90]}"
        if text_path.exists():
            result["text_path"] = str(text_path.relative_to(pack))
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            result["text_chars"] = str(len(text))
            result["evidence_status"] = (
                "readable" if len(text.strip()) >= 500 else "thin_or_unreadable"
            )
        else:
            result["evidence_status"] = "not_readable"
    except Exception as exc:  # noqa: BLE001
        result["extraction_status"] = "fetch_error"
        result["evidence_status"] = "error"
        result["notes"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=PACK)
    parser.add_argument("--priority", default="high")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    inventory = read_csv(args.pack_dir / INVENTORY)
    seen: set[str] = set()
    selected: list[dict[str, str]] = []
    for row in inventory:
        if row.get("priority") != args.priority:
            continue
        key = row["canonical_url"]
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)

    rows = [fetch_one(row, args.pack_dir, args.timeout) for row in selected]
    write_csv(
        args.pack_dir / READING_STATUS,
        rows,
        [
            "link_id",
            "queue_id",
            "activity_id",
            "source_type",
            "canonical_url",
            "fetch_url",
            "http_status",
            "content_type",
            "raw_path",
            "text_path",
            "byte_count",
            "text_chars",
            "extraction_status",
            "evidence_status",
            "notes",
        ],
    )

    readable = [row for row in rows if row["evidence_status"] == "readable"]
    blocked = [row for row in rows if row["evidence_status"] in {"blocked", "error"}]
    thin = [row for row in rows if row["evidence_status"] == "thin_or_unreadable"]
    memo = [
        "# Logged-In Source Reading Memo - 2026-05-21",
        "",
        f"- High-priority unique sources attempted: {len(rows)}",
        f"- Readable local extractions: {len(readable)}",
        f"- Thin/unreadable extractions: {len(thin)}",
        f"- Blocked/errors: {len(blocked)}",
        "",
        "## Readable Sources",
        "",
    ]
    for row in readable:
        memo.append(
            f"- `{row['queue_id']}` [{row['source_type']}]({row['canonical_url']}) "
            f"text_chars={row['text_chars']} path=`{row['text_path']}`"
        )
    memo.extend(["", "## Blocked Or Thin", ""])
    for row in blocked + thin:
        memo.append(
            f"- `{row['queue_id']}` [{row['source_type']}]({row['canonical_url']}) "
            f"status={row['http_status']} evidence={row['evidence_status']} notes={row['notes'][:140]}"
        )
    (args.pack_dir / MEMO_PATH).write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"attempted {len(rows)} high-priority sources")
    print(f"readable {len(readable)}; thin {len(thin)}; blocked/errors {len(blocked)}")
    print(args.pack_dir / READING_STATUS)


if __name__ == "__main__":
    main()
