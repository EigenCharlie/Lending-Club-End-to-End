#!/usr/bin/env python3
"""Process captured LinkedIn research-pack assets.

This script consumes `data/attachment_manifest.csv`. Rows with a `local_path`
are checked for existence, hashed, and processed into text artifacts under
`attachments/extracted/`. PDFs use `pdftotext` when available and fall back to
`pypdf`; images use Tesseract when available and otherwise record an explicit
OCR blocker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov")
MANIFEST_REL = Path("data/attachment_manifest.csv")
EXTRACTED_REL = Path("attachments/extracted")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(pack_dir: Path, local_path: str) -> Path | None:
    if not local_path:
        return None
    path = Path(local_path)
    if path.is_absolute():
        return path
    return pack_dir / path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def extract_pdf_text(path: Path, output_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        result = subprocess.run(
            [pdftotext, "-layout", str(path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return "pdf_text_extracted_pdftotext"

    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    output_path.write_text(text, encoding="utf-8")
    if text:
        return "pdf_text_extracted_pypdf"
    return "pdf_text_empty_ocr_needed"


def process_image(path: Path, output_path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        width, height = image.size
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return "ocr_tool_missing", f"1 image ({width}x{height})"

    result = subprocess.run(
        [tesseract, str(path), str(output_path.with_suffix(""))],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output_path.write_text(result.stderr.strip(), encoding="utf-8")
        return "ocr_error", f"1 image ({width}x{height})"
    return "ocr_text_extracted_tesseract", f"1 image ({width}x{height})"


def process_manifest(pack_dir: Path) -> int:
    manifest_path = pack_dir / MANIFEST_REL
    rows, fieldnames = read_csv(manifest_path)
    extracted_dir = pack_dir / EXTRACTED_REL
    extracted_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for row in rows:
        local_path = resolve_path(pack_dir, row.get("local_path", ""))
        if not local_path:
            continue
        if not local_path.exists():
            row["text_extract_status"] = "local_file_missing"
            row["analytic_memo"] = f"Missing local_path: {local_path}"
            continue

        row["checksum"] = checksum(local_path)
        asset_id = row["asset_id"]
        output_path = extracted_dir / f"{asset_id}.txt"
        suffix = local_path.suffix.lower()
        manually_read = row.get("ocr_status") == "manual_visual_read_completed"

        if manually_read:
            if suffix == ".pdf" and not row.get("page_or_slide_count"):
                row["page_or_slide_count"] = str(pdf_page_count(local_path))
            elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"} and not row.get(
                "page_or_slide_count"
            ):
                with Image.open(local_path) as image:
                    width, height = image.size
                row["page_or_slide_count"] = f"1 image ({width}x{height})"
            row["text_extract_status"] = row.get("text_extract_status") or "manual_visual_read"
            row["analytic_memo"] = row.get("analytic_memo") or (
                "Manual visual read completed; preserved without OCR overwrite."
            )
            processed += 1
            continue

        if suffix == ".pdf":
            row["page_or_slide_count"] = str(pdf_page_count(local_path))
            row["text_extract_status"] = extract_pdf_text(local_path, output_path)
            row["ocr_status"] = (
                "not_needed_pdf_text_layer"
                if "extracted" in row["text_extract_status"]
                else "ocr_needed"
            )
            row["analytic_memo"] = f"Extracted text path: {output_path.relative_to(pack_dir)}"
            processed += 1
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            ocr_status, count_label = process_image(local_path, output_path)
            row["page_or_slide_count"] = count_label
            row["ocr_status"] = ocr_status
            row["text_extract_status"] = "image_ocr_attempted"
            row["analytic_memo"] = f"OCR/text path: {output_path.relative_to(pack_dir)}"
            processed += 1
        elif suffix == ".txt":
            row["ocr_status"] = (
                row.get("ocr_status")
                if row.get("ocr_status") and not row.get("ocr_status", "").startswith("unsupported")
                else "not_needed_text_sidecar"
            )
            row["text_extract_status"] = (
                row.get("text_extract_status")
                if row.get("text_extract_status")
                and not row.get("text_extract_status", "").startswith("unsupported")
                else "text_sidecar_available"
            )
            row["analytic_memo"] = f"Text sidecar available: {local_path.relative_to(pack_dir)}"
            processed += 1
        else:
            row["text_extract_status"] = f"unsupported_asset_suffix:{suffix}"
            row["analytic_memo"] = (
                "Add PDF/image support or extract manually before using this asset as evidence."
            )

    write_csv(manifest_path, rows, fieldnames)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = process_manifest(args.pack_dir)
    print(f"Processed {processed} local assets")


if __name__ == "__main__":
    main()
