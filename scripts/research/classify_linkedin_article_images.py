#!/usr/bin/env python3
"""Classify captured LinkedIn article images by likely analytic value."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov/second_ingest")


FIELDS = [
    "article_id",
    "asset_id",
    "local_path",
    "source_url",
    "width",
    "height",
    "image_role",
    "visual_priority",
    "read_decision",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_role(source_url: str, asset_id: str) -> str:
    if "article-inline_image" in source_url:
        return "article_inline_figure"
    if "article-cover_image" in source_url and asset_id.endswith("_image_01"):
        return "article_primary_cover"
    if "article-cover_image" in source_url:
        return "recommended_or_related_article_cover"
    return "other_linkedin_article_image"


def priority_for(role: str, width: int, height: int) -> tuple[str, str]:
    if role == "article_inline_figure":
        return "high", "manual_visual_read_with_article_text"
    if role == "article_primary_cover" and width >= 400 and height >= 180:
        return "medium", "skim_for_context_not_claim_evidence"
    if role == "recommended_or_related_article_cover":
        return "archive", "not_parent_article_content"
    return "low", "archive_unless_text_indicates_relevance"


def classify(pack_dir: Path) -> None:
    rows = read_csv(pack_dir / "data" / "article_asset_manifest.csv")
    outputs: list[dict[str, str]] = []
    for row in rows:
        path = pack_dir / row["local_path"]
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError:
            width, height = 0, 0
        role = classify_role(row["source_url"], row["asset_id"])
        priority, decision = priority_for(role, width, height)
        outputs.append(
            {
                "article_id": row["article_id"],
                "asset_id": row["asset_id"],
                "local_path": row["local_path"],
                "source_url": row["source_url"],
                "width": str(width),
                "height": str(height),
                "image_role": role,
                "visual_priority": priority,
                "read_decision": decision,
            }
        )
    write_csv(pack_dir / "data" / "article_visual_priority.csv", outputs, FIELDS)
    print(f"Classified {len(outputs)} article images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classify(args.pack_dir)


if __name__ == "__main__":
    main()
