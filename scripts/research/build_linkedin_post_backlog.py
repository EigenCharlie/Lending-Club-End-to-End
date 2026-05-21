#!/usr/bin/env python3
"""Build a post-indexed execution backlog for the Denis Burakov LinkedIn corpus."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov")
P1_ACTIVITY_IDS = {
    "7458410505825685505",
    "7454786558127161344",
    "7452612214328537088",
    "7449713105552482304",
    "7439581405879103488",
    "7436682459037089793",
    "7426897722655277074",
    "7423998736009375745",
    "7416750932140552192",
    "7410227928887783424",
    "7391383804583645184",
    "7389209473522999296",
    "7363453428431224832",
    "7297638535648481280",
    "7239520617035640833",
    "7226833197983006722",
}


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


def word_count(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    if path.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".json", ".csv"}:
        return 0
    return len(re.findall(r"\w+", path.read_text(encoding="utf-8", errors="ignore")))


def destination_and_executable(post: dict[str, str]) -> tuple[str, str, str, str]:
    text = " ".join(
        [
            post.get("title", ""),
            post.get("theme", ""),
            post.get("summary_es", ""),
            post.get("tesis_use", ""),
        ]
    ).lower()
    if post.get("relevance") == "Baja":
        return (
            "Research archive",
            "No direct implementation unless a linked canonical source becomes surprisingly relevant.",
            "archive",
            "Stop after source/content is logged and low-relevance rationale is explicit.",
        )
    if any(
        term in text for term in ["woe", "scorecard", "fastwoe", "fisher scoring", "naive bayes"]
    ):
        return (
            "Book Ch05/Ch06; Paper 4 appendix only if experiment-ready",
            "WOE binning/recalibration diagnostic, scorecard comparison, or interpretable-ML sidebar.",
            "append_or_prototype",
            "Stop when the WOE/scorecard idea is either mapped to an existing chapter note or has a bounded experiment spec with data, metric, and rejection rule.",
        )
    if any(
        term in text
        for term in [
            "calibr",
            "brier",
            "gini",
            "somers",
            "ece",
            "ranking",
            "precision/recall",
            "log loss",
        ]
    ):
        return (
            "Book Ch06; Paper 4 metric governance; Paper Estrella reviewer-defense language",
            "Metric-governance text, calibration caveat, or diagnostic table/prototype.",
            "append",
            "Stop when discrimination/calibration/economic metric implication is classified as promote, append, park, or archive and no LinkedIn-only claim remains unlabeled.",
        )
    if any(term in text for term in ["lgd", "multiclass", "default", "softmax"]):
        return (
            "Book Ch07/Ch10; Paper 4 IFRS/LGD appendix candidate",
            "Distributional LGD or multiclass-default caveat; proxy-only unless Lending Club evidence exists.",
            "append_or_park",
            "Stop when data-feasibility is decided: implementable proxy appendix, related-work note, or parked due missing LGD/default-state evidence.",
        )
    if any(term in text for term in ["shap", "explain", "distillation", "catboost"]):
        return (
            "Book Ch06/Ch10; Paper 4 governance appendix",
            "SHAP distillation/governance memo or explainable-scorecard comparison.",
            "append_or_prototype",
            "Stop when explanation target is separated from causal claims and either added as governance language or queued as a bounded prototype.",
        )
    if any(
        term in text
        for term in ["aws", "sagemaker", "mlflow", "localstack", "pipeline", "mlops", "docker"]
    ):
        return (
            "Book Ch10/Ch14/Ch19; implementation companion",
            "MLOps/source-governance checklist or reproducibility appendix; not a core empirical claim.",
            "append",
            "Stop when implementation relevance is mapped to book infrastructure/governance text or archived as tool-context only.",
        )
    if any(term in text for term in ["conformal", "interval", "confidence"]):
        return (
            "Book Ch07; Paper Estrella related-work contrast",
            "Uncertainty-interval framing; distinguish conformal, classical, and Pearson-style intervals.",
            "append",
            "Stop when interval type, evidence status, and manuscript-safe contrast are explicit.",
        )
    if any(term in text for term in ["economic", "portfolio", "approval", "profit", "acceptance"]):
        return (
            "Book Ch09; Paper Estrella introduction/discussion",
            "Decision-value framing for Gini/acceptance-rate and calibrated PD use.",
            "append",
            "Stop when business-value language is tied to existing project artifacts without inventing empirical claims.",
        )
    return (
        "Research archive / source discovery",
        "Read for source trail; implement only if it changes a claim, appendix, or reviewer response.",
        "park",
        "Stop after a promote/append/park/archive decision and blocker/evidence note are recorded.",
    )


def status_for(primary: dict[str, str], activity_id: str) -> tuple[str, str]:
    text_status = primary.get("text_extract_status", "")
    ocr_status = primary.get("ocr_status", "")
    if activity_id in P1_ACTIVITY_IDS:
        return (
            "p1_analysis_memo_completed",
            "Verify external/canonical sources before manuscript promotion.",
        )
    if text_status == "pdf_text_extracted_pdftotext":
        return (
            "ready_for_pdf_text_reading",
            "Read post text, extracted PDF, transcript sidecar, and relevant external links.",
        )
    if text_status == "post_text_captured_no_media_found":
        return (
            "ready_for_post_text_reading",
            "Read public post text and resolve any external links.",
        )
    if ocr_status == "ocr_tool_missing":
        return (
            "blocked_pending_image_ocr_or_manual_read",
            "Run OCR/manual visual reading before claim use.",
        )
    return "needs_triage", "Inspect manifest paths and source blockers."


def build_backlog(pack_dir: Path) -> None:
    data_dir = pack_dir / "data"
    posts, _ = read_csv(data_dir / "posts_index.csv")
    inventory, _ = read_csv(data_dir / "linkedin_corpus_inventory.csv")
    manifest, _ = read_csv(data_dir / "attachment_manifest.csv")
    link_rows, _ = read_csv(data_dir / "external_link_backlog.csv")
    child_rows, child_fields = read_csv(data_dir / "external_linkedin_child_post_backlog.csv")
    capture_log, _ = read_csv(data_dir / "public_permalink_capture_log.csv")

    inv_by_id = {row["activity_id"]: row for row in inventory}
    capture_by_id = {row["activity_id"]: row for row in capture_log}
    manifest_by_activity: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        manifest_by_activity.setdefault(row["activity_id"], []).append(row)
    links_by_activity: dict[str, list[dict[str, str]]] = {}
    for row in link_rows:
        links_by_activity.setdefault(row["parent_activity_id"], []).append(row)

    backlog: list[dict[str, str]] = []
    for post in posts:
        activity_id = post["activity_id"]
        assets = manifest_by_activity.get(activity_id, [])
        primary = next((row for row in assets if row["asset_id"].endswith("_primary")), {})
        post_text_paths = [capture_by_id.get(activity_id, {}).get("post_text_path", "")]
        document_paths = [
            row["local_path"]
            for row in assets
            if row["asset_type"] in {"linkedin_document_pdf", "linkedin_document_transcript"}
        ]
        image_paths = [row["local_path"] for row in assets if "image" in row["asset_type"]]
        extracted_paths = []
        for row in assets:
            if row["asset_id"].endswith("_primary"):
                candidate = pack_dir / "attachments" / "extracted" / f"{row['asset_id']}.txt"
                if candidate.exists():
                    extracted_paths.append(str(candidate.relative_to(pack_dir)))
        readable_paths = [
            path
            for path in post_text_paths + document_paths + extracted_paths
            if path
            and Path(path).suffix.lower() in {".txt", ".md", ".html", ".htm", ".json", ".csv"}
        ]
        total_words = sum(word_count(pack_dir / path) for path in readable_paths)
        destination, executable, default_decision, stop_condition = destination_and_executable(post)
        execution_status, next_action = status_for(primary, activity_id)
        link_refs = links_by_activity.get(activity_id, [])
        source_types = sorted({row["source_type"] for row in link_refs})
        high_value_links = [
            row["link_asset_id"]
            for row in link_refs
            if row["handling_decision"]
            in {"read_as_potential_evidence", "associate_or_spawn_linkedin_child"}
        ]

        backlog.append(
            {
                "backlog_id": f"POST-{int(post['n']):03d}",
                "source_kind": "indexed_denis_post",
                "post_number": post["n"],
                "activity_id": activity_id,
                "parent_activity_id": "",
                "title": post["title"],
                "relevance": post["relevance"],
                "theme": post["theme"],
                "summary_es": post["summary_es"],
                "post_url": post["post_url"],
                "capture_status": inv_by_id.get(activity_id, {}).get("capture_status", ""),
                "primary_asset_type": primary.get("asset_type", ""),
                "primary_text_status": primary.get("text_extract_status", ""),
                "primary_ocr_status": primary.get("ocr_status", ""),
                "post_text_paths": "; ".join(sorted(set(post_text_paths))),
                "document_paths": "; ".join(sorted(set(document_paths))),
                "image_paths": "; ".join(sorted(set(image_paths))),
                "extracted_text_paths": "; ".join(sorted(set(extracted_paths))),
                "local_word_count_estimate": str(total_words),
                "external_link_count": str(len(link_refs)),
                "external_link_asset_ids": "; ".join(row["link_asset_id"] for row in link_refs),
                "external_source_types": "; ".join(source_types),
                "high_value_external_link_asset_ids": "; ".join(high_value_links),
                "project_destination": destination,
                "possible_executable_or_implementable": executable,
                "default_decision": default_decision,
                "execution_status": execution_status,
                "next_action": next_action,
                "post_stop_condition": stop_condition,
                "claim_gate": "No manuscript/book claim can be promoted from LinkedIn alone; attached/canonical evidence must be labeled first.",
            }
        )

    for child in child_rows:
        backlog.append(
            {
                "backlog_id": child["backlog_id"],
                "source_kind": child["source_kind"],
                "post_number": "",
                "activity_id": child["activity_id"],
                "parent_activity_id": child["parent_activity_id"],
                "title": child["title"],
                "relevance": "External",
                "theme": child["theme"],
                "summary_es": "LinkedIn post externo enlazado desde el corpus; se analiza solo si refuerza el tema del post padre.",
                "post_url": child["post_url"],
                "capture_status": "pending_child_capture_or_archive",
                "primary_asset_type": "",
                "primary_text_status": "",
                "primary_ocr_status": "",
                "post_text_paths": "",
                "document_paths": "",
                "image_paths": "",
                "extracted_text_paths": "",
                "local_word_count_estimate": "0",
                "external_link_count": "0",
                "external_link_asset_ids": "",
                "external_source_types": "linkedin_post_or_article",
                "high_value_external_link_asset_ids": "",
                "project_destination": "Parent-post dependent",
                "possible_executable_or_implementable": "Capture/read only if it can change the parent post decision or source trail.",
                "default_decision": "park",
                "execution_status": "pending_child_capture_or_archive",
                "next_action": child["handling_decision"],
                "post_stop_condition": child["stop_condition"],
                "claim_gate": "Child LinkedIn posts are context unless canonical evidence is found.",
            }
        )

    fields = [
        "backlog_id",
        "source_kind",
        "post_number",
        "activity_id",
        "parent_activity_id",
        "title",
        "relevance",
        "theme",
        "summary_es",
        "post_url",
        "capture_status",
        "primary_asset_type",
        "primary_text_status",
        "primary_ocr_status",
        "post_text_paths",
        "document_paths",
        "image_paths",
        "extracted_text_paths",
        "local_word_count_estimate",
        "external_link_count",
        "external_link_asset_ids",
        "external_source_types",
        "high_value_external_link_asset_ids",
        "project_destination",
        "possible_executable_or_implementable",
        "default_decision",
        "execution_status",
        "next_action",
        "post_stop_condition",
        "claim_gate",
    ]
    write_csv(data_dir / "post_execution_backlog.csv", backlog, fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_backlog(args.pack_dir)


if __name__ == "__main__":
    main()
