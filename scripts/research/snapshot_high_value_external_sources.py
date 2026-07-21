#!/usr/bin/env python3
"""Create readable snapshots for high-value external sources.

The first resolver intentionally preserves each post-level link reference. This
second pass is narrower: it reads only sources already marked
`read_as_potential_evidence` and tries to replace noisy HTML snapshots with a
stable, readable artifact such as a raw GitHub file or repository README.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


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


def fetch_text(url: str, timeout: int = 30) -> tuple[str, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return body.decode("utf-8", errors="replace"), resp.status, resp.geturl()


def fetch_bytes(url: str, timeout: int = 30) -> tuple[bytes, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.status, resp.geturl()


def checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_pdf_snapshot(
    pack_dir: Path, row: dict[str, str], target: Path
) -> tuple[str, int, str, str]:
    """Extract a resolver-downloaded PDF into a readable text snapshot."""
    local_binary_path = row.get("local_binary_path", "")
    if not local_binary_path:
        return "", 0, "", "resolver did not record a local PDF path"

    pdf_path = pack_dir / local_binary_path
    if not pdf_path.exists():
        return "", 0, "", f"local PDF path missing: {pdf_path}"

    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return "", 0, "", "pdftotext is not installed"

    extracted = target.with_suffix(".body.txt")
    result = subprocess.run(
        [pdftotext, "-layout", str(pdf_path), str(extracted)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "", 0, "", result.stderr.strip()

    text = clean_readable_text(extracted.read_text(encoding="utf-8", errors="replace"))
    extracted.unlink(missing_ok=True)
    header = (
        f"source_asset_id: {row['link_asset_id']}\n"
        f"parent_activity_id: {row['parent_activity_id']}\n"
        f"canonical_url: {row['canonical_url']}\n"
        f"readable_url: {row['canonical_url']}\n"
        f"source_type: {row['source_type']}\n"
        f"local_binary_path: {local_binary_path}\n\n"
    )
    target.write_text(header + text + "\n", encoding="utf-8")
    return row["canonical_url"], len(text), checksum_text(header + text + "\n"), ""


def extract_pdf_url_snapshot(
    row: dict[str, str], pdf_url: str, target: Path
) -> tuple[str, int, str, str]:
    """Fetch a real PDF URL and extract it into a readable text snapshot."""
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return "", 0, "", "pdftotext is not installed"

    try:
        pdf_bytes, status, final_url = fetch_bytes(pdf_url)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return "", 0, "", f"{type(exc).__name__}: {exc}"
    if not pdf_bytes.startswith(b"%PDF"):
        return "", 0, "", f"HTTP {status}: fetched target is not a PDF"

    temp_pdf = target.with_suffix(".source.pdf")
    temp_body = target.with_suffix(".body.txt")
    temp_pdf.write_bytes(pdf_bytes)
    result = subprocess.run(
        [pdftotext, "-layout", str(temp_pdf), str(temp_body)],
        check=False,
        capture_output=True,
        text=True,
    )
    temp_pdf.unlink(missing_ok=True)
    if result.returncode != 0:
        temp_body.unlink(missing_ok=True)
        return "", 0, "", result.stderr.strip()

    text = clean_readable_text(temp_body.read_text(encoding="utf-8", errors="replace"))
    temp_body.unlink(missing_ok=True)
    readable_url = final_url or pdf_url
    header = (
        f"source_asset_id: {row['link_asset_id']}\n"
        f"parent_activity_id: {row['parent_activity_id']}\n"
        f"canonical_url: {row['canonical_url']}\n"
        f"readable_url: {readable_url}\n"
        f"source_type: {row['source_type']}\n\n"
    )
    target.write_text(header + text + "\n", encoding="utf-8")
    return readable_url, len(text), checksum_text(header + text + "\n"), ""


def clean_readable_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def github_candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    if "github.com" not in parsed.netloc.lower():
        return []

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    candidates: list[str] = []

    if len(parts) >= 5 and parts[2] == "blob":
        branch = parts[3]
        rel_path = "/".join(parts[4:])
        candidates.append(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel_path}")
    elif len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
        subdir = "/".join(parts[4:])
        if subdir:
            for name in ("README.md", "README.rst", "readme.md"):
                candidates.append(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{subdir}/{name}"
                )
        for name in ("README.md", "README.rst", "readme.md"):
            candidates.append(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}")
    else:
        for branch in ("main", "master"):
            for name in ("README.md", "README.rst", "readme.md"):
                candidates.append(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}"
                )

    return list(dict.fromkeys(candidates))


def project_use_for(row: dict[str, str], text_length: int) -> tuple[str, str]:
    url = row["canonical_url"].lower()
    source_type = row["source_type"]
    if "credit-risk-modeling-on-aws" in url:
        return (
            "governance_context",
            "Use as implementation context for model lineage, deployment separation, and reproducibility only.",
        )
    if "on-credit-book" in url:
        return (
            "source_discovery",
            "Keep as code/source trail for the author's book materials; do not promote as independent evidence.",
        )
    if "fastwoe" in url:
        return (
            "prototype_candidate",
            "Park as optional WOE tooling/prototype; no dependency or claim without project-local benchmark.",
        )
    if "xbooster" in url:
        return (
            "prototype_candidate",
            "Park as boosted-scorecard or SHAP-scorecard prototype candidate; no champion change.",
        )
    if "pearsonify" in url:
        return (
            "related_work_context",
            "Use for interval-method contrast only; keep separate from project empirical claims.",
        )
    if "woeboost" in url:
        return (
            "prototype_candidate",
            "Park as future WOE/boosting benchmark candidate; no current implementation lane.",
        )
    if "bankingsupervision.europa.eu" in url:
        return (
            "official_supervisory_guidance",
            "Use as official governance context for internal-model ML controls, reproducibility, and explainability.",
        )
    if "fisher-scoring" in url:
        return (
            "prototype_candidate",
            "Park as optional logistic inference/robust-variant tooling; no dependency now.",
        )
    if "calibration_visualization.py" in url:
        return (
            "teaching_visual_context",
            "Use only as visualization/source-trail context for calibration concepts.",
        )
    if "fastwoe_finetune.py" in url:
        return (
            "prototype_candidate",
            "Use as source trail for GBDT/WOE maintenance concept if raw file is readable.",
        )
    if "arxiv.org/html/2401.06091" in url:
        return (
            "preprint_metric_caveat",
            "Use as preprint support for AUROC/AUPRC imbalance caveat, clearly labeled as preprint.",
        )
    if "arxiv.org/html/2412.15321" in url:
        return (
            "out_of_scope_preprint",
            "Archive as low-relevance generative-AI context; no credit-risk implementation.",
        )
    if "matplotlib.org" in url:
        return (
            "official_tool_documentation",
            "Keep only as official plotting documentation provenance.",
        )
    if "patch-gpt" in url:
        return (
            "out_of_scope_code_context",
            "Archive as non-credit-risk language-model code context.",
        )
    if text_length < 500:
        return ("insufficient_readable_text", "Readable snapshot is too short for evidence use.")
    if source_type == "github":
        return ("code_context", "Use as implementation context only unless locally benchmarked.")
    if source_type == "preprint":
        return ("preprint_context", "Use only with preprint label and no peer-reviewed claim.")
    return ("context", "Use after manual source-status review.")


def snapshot_sources(pack_dir: Path, sleep_seconds: float) -> None:
    data_dir = pack_dir / "data"
    rows, _ = read_csv(data_dir / "external_link_backlog.csv")
    post_rows, _ = read_csv(data_dir / "posts_index.csv")
    post_number_by_activity = {row["activity_id"]: row["n"] for row in post_rows}

    high_value_rows = [
        row for row in rows if row["handling_decision"] == "read_as_potential_evidence"
    ]
    readable_root = pack_dir / "external_sources" / "readable"
    readable_root.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, str]] = []
    for row in high_value_rows:
        asset_id = row["link_asset_id"]
        source_type = row["source_type"]
        canonical_url = row["canonical_url"]
        candidates = github_candidates(canonical_url)
        if not candidates:
            candidates = [canonical_url]

        status = "blocked_or_unreadable"
        readable_url = ""
        local_readable_path = ""
        text_length = 0
        sha256 = ""
        error = ""

        target = readable_root / f"{asset_id}.txt"
        is_pdf_like = bool(row.get("local_binary_path", "")) or canonical_url.lower().split("?", 1)[
            0
        ].endswith(".pdf")
        if is_pdf_like:
            readable_url, text_length, sha256, error = extract_pdf_snapshot(pack_dir, row, target)
            if not text_length:
                for candidate in candidates:
                    if not candidate.lower().split("?", 1)[0].endswith(".pdf"):
                        continue
                    readable_url, text_length, sha256, error = extract_pdf_url_snapshot(
                        row, candidate, target
                    )
                    if text_length:
                        break
            if text_length:
                local_readable_path = str(target.relative_to(pack_dir))
                status = "pdf_text_extracted_pdftotext"
            else:
                status = "blocked_or_unreadable"
        else:
            for candidate in candidates:
                try:
                    text, http_status, final_url = fetch_text(candidate)
                    text = clean_readable_text(text)
                    if len(re.findall(r"\w+", text)) < 25 and source_type == "github":
                        error = f"HTTP {http_status}: readable text too short"
                        continue
                    readable_url = final_url or candidate
                    header = (
                        f"source_asset_id: {asset_id}\n"
                        f"parent_activity_id: {row['parent_activity_id']}\n"
                        f"canonical_url: {canonical_url}\n"
                        f"readable_url: {readable_url}\n"
                        f"source_type: {source_type}\n\n"
                    )
                    target.write_text(header + text + "\n", encoding="utf-8")
                    local_readable_path = str(target.relative_to(pack_dir))
                    text_length = len(text)
                    sha256 = checksum_text(header + text + "\n")
                    status = f"readable_http_{http_status}"
                    error = ""
                    break
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    continue
                finally:
                    if sleep_seconds:
                        time.sleep(sleep_seconds)

        project_use, decision = project_use_for(row, text_length)
        if status == "blocked_or_unreadable":
            decision = (
                f"Readable snapshot unavailable; keep resolver metadata only. Last error: {error}"
            )

        outputs.append(
            {
                "link_asset_id": asset_id,
                "parent_activity_id": row["parent_activity_id"],
                "parent_post_number": post_number_by_activity.get(row["parent_activity_id"], ""),
                "source_type": source_type,
                "canonical_url": canonical_url,
                "readable_url": readable_url,
                "local_readable_path": local_readable_path,
                "reading_status": status,
                "readable_text_length": str(text_length),
                "checksum": sha256,
                "project_use": project_use,
                "decision": decision,
                "error": error,
            }
        )

    write_csv(
        data_dir / "high_value_external_source_reading.csv",
        outputs,
        [
            "link_asset_id",
            "parent_activity_id",
            "parent_post_number",
            "source_type",
            "canonical_url",
            "readable_url",
            "local_readable_path",
            "reading_status",
            "readable_text_length",
            "checksum",
            "project_use",
            "decision",
            "error",
        ],
    )
    print(f"Wrote {len(outputs)} high-value source reading rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    args = parser.parse_args()
    snapshot_sources(args.pack_dir, args.sleep_seconds)


if __name__ == "__main__":
    main()
