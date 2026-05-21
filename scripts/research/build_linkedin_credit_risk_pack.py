#!/usr/bin/env python3
"""Build the Denis Burakov LinkedIn credit-risk research intake pack.

The script intentionally does not scrape LinkedIn. It reprocesses the prior
human/browser-assisted index, records the official API access gate, and creates
auditable intake artifacts for later manual capture of posts, PDFs, decks, and
images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib import error, parse, request

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_denis_burakov")
POSTS_INDEX = Path("data/posts_index.csv")
RELEVANT_POSTS = Path("data/relevant_posts.csv")
EXTERNAL_LINKS = Path("data/external_links_resolved.csv")
ATTACHMENTS_MD = Path("attachments/ATTACHMENTS.md")

OFFICIAL_LINKEDIN_DOCS = [
    "https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api",
    "https://learn.microsoft.com/en-us/linkedin/marketing/community-management/community-management-overview",
    "https://www.linkedin.com/help/linkedin/answer/a1341387",
    "https://www.linkedin.com/legal/crawling-terms",
    "https://playwright.dev/python/docs/auth",
    "https://playwright.dev/python/docs/api/class-download",
    "https://playwright.dev/python/docs/screenshots",
    "https://chromedevtools.github.io/devtools-protocol/",
]


@dataclass(frozen=True)
class Post:
    n: str
    activity_id: str
    post_url: str
    title: str
    relevance: str
    theme: str
    summary_es: str
    tesis_use: str
    attachment_type: str
    external_links: str
    short_snippet: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def post_from_row(row: dict[str, str]) -> Post:
    return Post(
        n=row.get("n", ""),
        activity_id=row.get("activity_id", ""),
        post_url=row.get("post_url", ""),
        title=row.get("title", ""),
        relevance=row.get("relevance", ""),
        theme=row.get("theme", ""),
        summary_es=row.get("summary_es", ""),
        tesis_use=row.get("tesis_use", ""),
        attachment_type=row.get("attachment_type", ""),
        external_links=row.get("external_links", ""),
        short_snippet=row.get("short_snippet_under_25_words", ""),
    )


def parse_external_links(raw: str) -> list[str]:
    if not raw:
        return []
    urls = re.findall(r"https?://[^\s|,]+", raw)
    cleaned: list[str] = []
    for url in urls:
        normalized = url.strip().rstrip(").]:")
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def attachment_asset_type(attachment_type: str) -> str:
    lowered = (attachment_type or "").lower()
    if "document" in lowered or "deck" in lowered:
        return "linkedin_document_deck"
    if "image" in lowered or "carousel" in lowered:
        return "linkedin_image_carousel"
    if "video" in lowered:
        return "linkedin_video"
    if "link" in lowered:
        return "external_link_preview"
    if not lowered:
        return "none_recorded"
    return lowered.replace(" ", "_").replace("/", "_")


def status_for_post(post: Post) -> tuple[str, str, str]:
    if post.attachment_type:
        return (
            "prior_index_reprocessed",
            "partial_pending_attachment_capture",
            "prior scrape has post metadata, analytic summary, short snippet, and attachment type; full text and asset OCR are pending.",
        )
    if post.external_links:
        return (
            "prior_index_reprocessed",
            "partial_pending_external_link_review",
            "prior scrape has post metadata and links; full post text and external source reading are pending.",
        )
    return (
        "prior_index_reprocessed",
        "partial_pending_full_post_text",
        "prior scrape has metadata and a short snippet only; full post text remains pending.",
    )


def decision_for_post(post: Post) -> str:
    if post.relevance.lower().startswith("alta"):
        return "append_candidate_after_source_verification"
    if post.relevance.lower().startswith("media"):
        return "park_or_context_after_source_verification"
    return "archive_low_priority"


def destination_for_post(post: Post) -> str:
    text = f"{post.theme} {post.title} {post.tesis_use}".lower()
    destinations: list[str] = []
    if any(k in text for k in ["woe", "scorecard", "logistic", "weight of evidence"]):
        destinations.append("Book Ch05/Ch06")
    if any(
        k in text for k in ["brier", "gini", "calibr", "somers", "recall", "precision", "imbalance"]
    ):
        destinations.append("Book Ch06; Paper4 metric governance")
    if any(k in text for k in ["pearsonify", "interval", "conformal", "confidence"]):
        destinations.append("Book Ch07; Paper Estrella framing")
    if any(k in text for k in ["lgd", "ifrs", "ecl", "loss given default"]):
        destinations.append("Book Ch07/Ch10; Paper4 IFRS/LGD appendix")
    if any(k in text for k in ["aws", "mlflow", "sagemaker", "localstack", "mlops", "lifecycle"]):
        destinations.append("Book Ch10; implementation companion")
    if any(k in text for k in ["portfolio", "economics", "acceptance", "point worth"]):
        destinations.append("Book Ch09; Paper Estrella reviewer defense")
    if any(k in text for k in ["shap", "explainability", "distillation"]):
        destinations.append("Book Ch06/Ch10; Paper4 governance appendix")
    if not destinations:
        destinations.append("Research archive / source discovery")
    return "; ".join(dict.fromkeys(destinations))


def build_inventory(posts: list[Post]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for post in posts:
        capture_status, completeness_status, blocker = status_for_post(post)
        rows.append(
            {
                "activity_id": post.activity_id,
                "post_url": post.post_url,
                "date": "",
                "author": "Denis Burakov",
                "post_type": "linkedin_activity_post",
                "attachment_type": post.attachment_type or "none_recorded",
                "capture_status": capture_status,
                "completeness_status": completeness_status,
                "blocker_or_next_action": blocker,
                "source_status": "linkedin_member_post_prior_index_only",
            }
        )
    return rows


def checksum_for_existing_file(local_path: str, pack_dir: Path) -> str:
    if not local_path:
        return ""
    path = Path(local_path)
    if not path.is_absolute():
        path = pack_dir / path
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_attachment_manifest(posts: list[Post], pack_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for post in posts:
        if post.attachment_type:
            asset_id = f"{post.activity_id}_primary"
            rows.append(
                {
                    "activity_id": post.activity_id,
                    "asset_id": asset_id,
                    "asset_type": attachment_asset_type(post.attachment_type),
                    "source_url": post.post_url,
                    "local_path": "",
                    "page_or_slide_count": "",
                    "ocr_status": "pending_human_assisted_capture",
                    "checksum": "",
                    "text_extract_status": "not_available_prior_scrape",
                    "analytic_memo": "Prior scrape captured type only. Capture/download the asset, then OCR or extract full text before using as evidence.",
                }
            )
        for idx, url in enumerate(parse_external_links(post.external_links), start=1):
            rows.append(
                {
                    "activity_id": post.activity_id,
                    "asset_id": f"{post.activity_id}_external_{idx:02d}",
                    "asset_type": "external_link",
                    "source_url": url,
                    "local_path": "",
                    "page_or_slide_count": "",
                    "ocr_status": "not_applicable_until_source_downloaded",
                    "checksum": checksum_for_existing_file("", pack_dir),
                    "text_extract_status": "pending_external_source_review",
                    "analytic_memo": "Resolve and read the canonical external source before promoting claims.",
                }
            )
    return rows


def source_type_from_url(url: str) -> str:
    if not url:
        return "unknown"
    host = parse.urlparse(url).netloc.lower()
    path = parse.urlparse(url).path.lower()
    if "linkedin.com" in host and "/feed/update/" in path:
        return "linkedin_post"
    if "linkedin.com" in host and "/posts/" in path:
        return "linkedin_post"
    if "linkedin.com" in host and "/in/" in path:
        return "linkedin_profile"
    if "linkedin.com" in host and "/company/" in path:
        return "linkedin_company"
    if host == "lnkd.in":
        return "linkedin_shortlink_unresolved"
    if "github.com" in host:
        return "github_repository_or_file"
    if "medium.com" in host:
        return "blog_medium"
    if "arxiv.org" in host:
        return "preprint_arxiv"
    if "amazon.com" in host or "kdp.amazon" in host:
        return "book_or_marketplace"
    if any(
        x in host
        for x in ["microsoft.com", "linkedin.com/help", "ifrs.org", "bis.org", "federalreserve.gov"]
    ):
        return "official_guidance_or_docs"
    return "external_web"


def access_status(row: dict[str, str]) -> str:
    status = (row.get("status") or "").strip()
    error_msg = (row.get("error") or "").strip()
    resolved = row.get("resolved_url") or ""
    source = row.get("source_url") or ""
    if error_msg:
        return "blocked_or_http_error"
    if (
        status == "200"
        and resolved
        and resolved != source
        and "lnkd.in" not in parse.urlparse(resolved).netloc
    ):
        return "resolved"
    if status == "200" and "lnkd.in" in parse.urlparse(resolved).netloc:
        return "shortlink_gate_or_unresolved"
    if status == "200":
        return "reachable"
    if not status:
        return "not_checked_or_not_public"
    return f"http_{status}"


def build_external_source_log(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        canonical = row.get("resolved_url") or row.get("source_url") or ""
        out.append(
            {
                "activity_id": (row.get("post_urn", "") or "").replace("urn:li:activity:", ""),
                "post_url": row.get("post_url", ""),
                "label": row.get("label", ""),
                "source_url": row.get("source_url", ""),
                "canonical_url": canonical,
                "http_status": row.get("status", ""),
                "source_type": source_type_from_url(canonical),
                "access_status": access_status(row),
                "title": row.get("title", ""),
                "error": row.get("error", ""),
                "claim_use_rule": "Use as source only after reading canonical content; LinkedIn-only material remains non-peer-reviewed evidence.",
            }
        )
    return out


CONCEPT_SPECS = [
    {
        "concept": "Brier vs Gini/ECE separation",
        "keywords": ["brier", "calibration", "model calibration", "gini", "ece"],
        "required_any": [
            "brier",
            "model calibration",
            "calibracion",
            "calibration toolkit",
            "class imbalance",
        ],
        "method_family": "Calibration and metric governance",
        "novelty": "Medium",
        "project_destination": "Book Ch06; Paper4 metric governance; Paper Estrella reviewer defense",
        "implementation_difficulty": "Low",
        "claim_risk": "Medium: requires validated source beyond LinkedIn before manuscript claim.",
    },
    {
        "concept": "Observation-level Gini contribution diagnostics",
        "keywords": ["observations help", "hurt", "gini"],
        "required_any": ["observations help", "hurt your model's gini", "debug de gini"],
        "method_family": "Ranking diagnostics",
        "novelty": "Medium-high",
        "project_destination": "Book Ch06; Paper4 appendix diagnostic",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium: useful as diagnostic, not a new guarantee.",
    },
    {
        "concept": "Economic value of Gini and acceptance-rate framing",
        "keywords": ["gini point", "model economics", "acceptance", "portfolio", "bi-normal"],
        "required_any": [
            "gini point",
            "model economics",
            "economia de gini",
            "valor economico de gini",
        ],
        "method_family": "Credit decision economics",
        "novelty": "High for narrative integration",
        "project_destination": "Book Ch09; Paper Estrella introduction/discussion",
        "implementation_difficulty": "Low-medium",
        "claim_risk": "Medium: must tie to project artifacts, not generic LinkedIn examples.",
    },
    {
        "concept": "WOE recalibration under drift",
        "keywords": ["woe recalibration", "fine-tuning", "drift"],
        "required_any": ["woe recalibration"],
        "method_family": "Scorecard maintenance",
        "novelty": "High",
        "project_destination": "Book Ch05; Paper4 bounded prototype candidate",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium-high: needs reproducible Lending Club drift experiment.",
    },
    {
        "concept": "GBDT leaf WOE and boosted scorecards",
        "keywords": ["boosted scorecards", "xbooster", "woeboost", "gbdt", "main effects"],
        "required_any": ["boosted scorecards", "xbooster", "woeboost", "main effects", "gbdt"],
        "method_family": "Interpretable ML / scorecards",
        "novelty": "High",
        "project_destination": "Book Ch05/Ch06; future benchmark",
        "implementation_difficulty": "High",
        "claim_risk": "High: avoid adding dependency-heavy benchmark without gate.",
    },
    {
        "concept": "Probabilistic LGD via quantile or multiclass bins",
        "keywords": ["lgd", "loss given default", "quantile", "multiclass"],
        "required_any": ["lgd", "loss given default", "multiclass"],
        "method_family": "Risk parameter uncertainty",
        "novelty": "High",
        "project_destination": "Book Ch07/Ch10; Paper4 IFRS/LGD appendix",
        "implementation_difficulty": "Medium-high",
        "claim_risk": "High: project data limitations and ECL proxy boundaries apply.",
    },
    {
        "concept": "SHAP distillation into scorecard-style explanations",
        "keywords": ["shap", "distillation", "explainability", "scorecards"],
        "required_any": ["shap", "distillation", "explainability and scorecards"],
        "method_family": "Explainability and model governance",
        "novelty": "High",
        "project_destination": "Book Ch06/Ch10; Paper4 governance appendix",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium: explanation of predictions is not causal mechanism.",
    },
    {
        "concept": "Class imbalance and probability correction",
        "keywords": ["imbalanced", "resampling", "class imbalance", "fraud"],
        "required_any": ["imbalanced", "class imbalance", "resampling"],
        "method_family": "Calibration under rare events",
        "novelty": "Medium",
        "project_destination": "Book Ch06 calibration caveats",
        "implementation_difficulty": "Low",
        "claim_risk": "Low-medium: align with existing calibration evidence.",
    },
    {
        "concept": "Somers D / Dxy for ordinal or continuous outcomes",
        "keywords": ["somers", "dxy", "ordinal", "lgd"],
        "required_any": ["somers", "dxy"],
        "method_family": "Credit-risk metrics",
        "novelty": "Medium",
        "project_destination": "Book Ch06; possible LGD metric extension",
        "implementation_difficulty": "Low-medium",
        "claim_risk": "Low: as metric addition, not a core contribution.",
    },
    {
        "concept": "Classification/probability intervals and Pearsonify contrast",
        "keywords": [
            "pearsonify",
            "classification intervals",
            "confidence intervals",
            "probability intervals",
        ],
        "required_any": [
            "pearsonify",
            "classification intervals",
            "confidence intervals",
            "probability intervals",
        ],
        "method_family": "Probability uncertainty",
        "novelty": "Medium-high",
        "project_destination": "Book Ch07; Paper Estrella related-work contrast",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium: distinguish classical intervals, Venn-Abers, and conformal intervals.",
    },
    {
        "concept": "Credit-risk MLOps with MLflow/SageMaker/LocalStack",
        "keywords": ["aws", "mlflow", "sagemaker", "localstack", "catboost lifecycle"],
        "required_any": ["aws", "mlflow", "sagemaker", "localstack", "catboost lifecycle"],
        "method_family": "MLOps and reproducibility",
        "novelty": "Medium",
        "project_destination": "Book Ch10; implementation companion",
        "implementation_difficulty": "Medium-high",
        "claim_risk": "Low for engineering appendix; high if framed as research contribution.",
    },
    {
        "concept": "Fine-tuning GBDT when new data sources arrive",
        "keywords": ["new data source", "fine-tuning", "init_model"],
        "required_any": ["new data source", "init_model"],
        "method_family": "Model maintenance",
        "novelty": "Medium-high",
        "project_destination": "Book Ch10; Paper4 drift/maintenance candidate",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium-high: needs stable train/update protocol.",
    },
    {
        "concept": "Robust logistic regression for noisy labels/outliers",
        "keywords": ["robust logistic regression", "contaminated", "outliers"],
        "required_any": ["robust logistic regression"],
        "method_family": "Robust statistics",
        "novelty": "Medium",
        "project_destination": "Book Ch06 baseline robustness note",
        "implementation_difficulty": "Medium",
        "claim_risk": "Medium: baseline only unless empirically material.",
    },
    {
        "concept": "External resource collections and On Credit book trail",
        "keywords": ["on credit", "notes on lending", "resource collection", "foundations"],
        "required_any": ["on credit", "notes on lending", "resource collection"],
        "method_family": "Source discovery",
        "novelty": "Low-medium",
        "project_destination": "Bibliography triage; Book foundations",
        "implementation_difficulty": "Low",
        "claim_risk": "Medium: source status must be labeled; not all resources are peer reviewed.",
    },
]


def build_concept_atlas(posts: list[Post]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in CONCEPT_SPECS:
        matches = []
        for post in posts:
            haystack = " ".join([post.title, post.theme, post.summary_es, post.tesis_use]).lower()
            required_any = spec.get("required_any", spec["keywords"])
            if any(keyword in haystack for keyword in required_any):
                matches.append(post)
        if not matches:
            continue
        evidence_strength = "LinkedIn-only intake; source verification pending"
        if any(post.external_links for post in matches):
            evidence_strength = (
                "LinkedIn intake plus unresolved/resolved external leads; full reading pending"
            )
        rows.append(
            {
                "concept": spec["concept"],
                "method_family": spec["method_family"],
                "matched_posts": "; ".join(f"{post.n}:{post.activity_id}" for post in matches),
                "post_count": len(matches),
                "novelty": spec["novelty"],
                "evidence_strength": evidence_strength,
                "project_destination": spec["project_destination"],
                "implementation_difficulty": spec["implementation_difficulty"],
                "claim_risk": spec["claim_risk"],
                "recommended_decision": recommended_decision_for_concept(spec, matches),
            }
        )
    return rows


def recommended_decision_for_concept(spec: dict[str, str], matches: list[Post]) -> str:
    if any(post.relevance.lower().startswith("alta") for post in matches):
        if (
            "Paper4" in spec["project_destination"]
            and "candidate" in spec["project_destination"].lower()
        ):
            return "append_or_prototype_only_after_source_verification"
        return "append_to_atlas_after_source_verification"
    return "park_for_context"


def build_api_probe(posts: list[Post], limit: int) -> list[dict[str, object]]:
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "").strip()
    rows: list[dict[str, object]] = []
    for post in posts[:limit]:
        api_urns = [
            f"urn:li:activity:{post.activity_id}",
            f"urn:li:share:{post.activity_id}",
            f"urn:li:ugcPost:{post.activity_id}",
        ]
        for api_urn in api_urns:
            encoded = parse.quote(api_urn, safe="")
            endpoint = f"https://api.linkedin.com/rest/posts/{encoded}"
            headers = {
                "LinkedIn-Version": "202505",
                "X-Restli-Protocol-Version": "2.0.0",
                "Accept": "application/json",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = request.Request(
                endpoint,
                headers=headers,
            )
            try:
                with request.urlopen(req, timeout=15) as resp:
                    status = resp.status
                    body = resp.read(512).decode("utf-8", errors="replace")
                rows.append(
                    {
                        "activity_id": post.activity_id,
                        "attempted_urn": api_urn,
                        "endpoint": endpoint,
                        "http_status": status,
                        "result": "reachable" if token else "unexpected_reachable_without_oauth",
                        "blocker": body[:200],
                    }
                )
            except error.HTTPError as exc:
                body = exc.read(512).decode("utf-8", errors="replace")
                rows.append(
                    {
                        "activity_id": post.activity_id,
                        "attempted_urn": api_urn,
                        "endpoint": endpoint,
                        "http_status": exc.code,
                        "result": "blocked_unauthenticated_no_oauth_token"
                        if not token
                        else "blocked_or_not_authorized",
                        "blocker": body[:240]
                        or "Official LinkedIn Posts API requires OAuth; member post retrieval requires restricted r_member_social access.",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - probe log should preserve failure class.
                rows.append(
                    {
                        "activity_id": post.activity_id,
                        "attempted_urn": api_urn,
                        "endpoint": endpoint,
                        "http_status": "",
                        "result": "probe_error_without_oauth_token" if not token else "probe_error",
                        "blocker": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def build_capture_queue(posts: list[Post]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for post in posts:
        priority = "P1" if post.relevance == "Alta" else "P2" if post.relevance == "Media" else "P3"
        if (
            "document" in (post.attachment_type or "").lower()
            or "deck" in (post.attachment_type or "").lower()
        ):
            action = "Open post in visible Windows Chrome session, expand text, try document download, otherwise capture every page/slide and OCR."
        elif (
            "image" in (post.attachment_type or "").lower()
            or "carousel" in (post.attachment_type or "").lower()
        ):
            action = "Open post in visible Windows Chrome session, expand text, capture every carousel image/slide and OCR."
        elif post.external_links:
            action = "Open post, expand text, resolve and save/read relevant external links."
        else:
            action = (
                "Open post, expand full text, save private screenshot/text transcript if available."
            )
        rows.append(
            {
                "priority": priority,
                "activity_id": post.activity_id,
                "post_url": post.post_url,
                "title": post.title,
                "attachment_type": post.attachment_type or "none_recorded",
                "next_action": action,
                "done_definition": "Full post text captured; all attachments/links have extracted text or explicit blocker; analytic memo added.",
                "safety_rule": "Use user-owned visible session only; no fake accounts, captcha bypass, stealth, or rate evasion.",
            }
        )
    return rows


def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cleaned = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(lines)


def build_claim_map(
    pack_dir: Path,
    posts: list[Post],
    concept_rows: list[dict[str, object]],
    api_rows: list[dict[str, object]],
    run_date: str,
) -> str:
    rel_counts = Counter(post.relevance or "Sin clasificar" for post in posts)
    attach_counts = Counter(post.attachment_type or "none_recorded" for post in posts)
    api_summary = Counter(str(row["result"]) for row in api_rows)
    top_concepts = sorted(concept_rows, key=lambda row: int(row["post_count"]), reverse=True)
    concept_table = markdown_table(
        [
            [
                row["concept"],
                row["method_family"],
                row["post_count"],
                row["project_destination"],
                row["recommended_decision"],
                row["claim_risk"],
            ]
            for row in top_concepts
        ],
        ["Concept", "Family", "Posts", "Destination", "Decision", "Claim Risk"],
    )

    post_table = markdown_table(
        [
            [
                post.n,
                post.activity_id,
                post.relevance,
                post.theme,
                post.attachment_type or "none_recorded",
                decision_for_post(post),
                destination_for_post(post),
                "partial_prior_index_pending_full_text_assets",
            ]
            for post in posts
        ],
        ["#", "Activity", "Rel", "Theme", "Attachment", "Decision", "Destination", "Source Status"],
    )

    docs = "\n".join(f"- {url}" for url in OFFICIAL_LINKEDIN_DOCS)
    rel_summary = ", ".join(f"{key}: {value}" for key, value in sorted(rel_counts.items()))
    attach_summary = ", ".join(f"{key}: {value}" for key, value in sorted(attach_counts.items()))
    api_summary_text = (
        ", ".join(f"{key}: {value}" for key, value in sorted(api_summary.items())) or "not run"
    )

    return f"""# LinkedIn Credit Risk Claim-Evidence Map: Denis Burakov

Generated: {run_date}

## Executive Status

This pack reprocesses the prior 59-post index into an auditable intake surface.
It does not claim that posts, PDFs, decks, or images have been fully read yet.
The prior scrape preserved metadata, short snippets, analytic summaries, visible
attachment types, and external-link leads; full post text and attachments remain
queued for human-assisted capture.

- Posts reprocessed: {len(posts)}
- Relevance mix: {rel_summary}
- Attachment mix: {attach_summary}
- Official API probe status: {api_summary_text}
- Governance rule: LinkedIn material is intake evidence only until the attached
  PDF/image/deck/external source is read and source status is labeled.

## API And Capture Decision

The implementation records the official API path first in
`data/linkedin_api_probe_log.csv`. When no OAuth token is supplied, the probe is
sent without an Authorization header and should return a 401 authorization
blocker. If a token is supplied later, the script will attempt the documented
Posts API endpoint for activity/share/UGC URN variants. Current working
assumption remains that third-party member post retrieval is blocked by
restricted `r_member_social` access, so the approved fallback is a visible,
user-owned, human-assisted Chrome workflow.

## Concept Atlas

{concept_table}

## Post-Level Claim Intake

{post_table}

## Source Status Rules

- `append_candidate_after_source_verification`: relevant to the project, but not
  manuscript evidence until full post and attachment/source content are read.
- `park_or_context_after_source_verification`: useful context or future lane,
  but not first-wave Quarto/Paper material.
- `archive_low_priority`: keep permalink and summary only unless a reviewer or
  later source trail makes it relevant.

## Official/Workflow References Used For Feasibility

{docs}

## Files In This Pack

- `data/linkedin_corpus_inventory.csv`
- `data/attachment_manifest.csv`
- `data/external_source_log.csv`
- `data/concept_atlas.csv`
- `data/linkedin_api_probe_log.csv`
- `data/human_assisted_capture_queue.csv`
- `docs/linkedin_claim_evidence_map.md`
"""


def validate_outputs(pack_dir: Path, expected_posts: int) -> list[str]:
    errors: list[str] = []
    required_files = [
        "data/linkedin_corpus_inventory.csv",
        "data/attachment_manifest.csv",
        "data/external_source_log.csv",
        "data/concept_atlas.csv",
        "data/linkedin_api_probe_log.csv",
        "data/human_assisted_capture_queue.csv",
        "docs/linkedin_claim_evidence_map.md",
    ]
    for rel in required_files:
        if not (pack_dir / rel).exists():
            errors.append(f"Missing required output: {rel}")

    inventory_path = pack_dir / "data/linkedin_corpus_inventory.csv"
    if inventory_path.exists():
        inventory_rows = read_csv(inventory_path)
        if len(inventory_rows) != expected_posts:
            errors.append(f"Inventory row count {len(inventory_rows)} != expected {expected_posts}")
        for row in inventory_rows:
            if not row.get("capture_status") or not row.get("completeness_status"):
                errors.append(f"Missing status for activity_id={row.get('activity_id')}")
                break

    manifest_path = pack_dir / "data/attachment_manifest.csv"
    if manifest_path.exists():
        for row in read_csv(manifest_path):
            if not row.get("ocr_status") and row.get("asset_type") != "external_link":
                errors.append(f"Missing OCR status for asset_id={row.get('asset_id')}")
                break
    return errors


def build_pack(pack_dir: Path, probe_api: bool, api_probe_limit: int, run_date: str) -> None:
    posts_path = pack_dir / POSTS_INDEX
    external_path = pack_dir / EXTERNAL_LINKS
    if not posts_path.exists():
        raise FileNotFoundError(f"Missing prior post index: {posts_path}")
    if not external_path.exists():
        raise FileNotFoundError(f"Missing prior external link index: {external_path}")

    posts = [post_from_row(row) for row in read_csv(posts_path)]
    external_rows = read_csv(external_path)

    inventory_rows = build_inventory(posts)
    manifest_rows = build_attachment_manifest(posts, pack_dir)
    external_log_rows = build_external_source_log(external_rows)
    concept_rows = build_concept_atlas(posts)
    api_rows = build_api_probe(posts, api_probe_limit) if probe_api else []
    capture_queue_rows = build_capture_queue(posts)

    write_csv(
        pack_dir / "data/linkedin_corpus_inventory.csv",
        inventory_rows,
        [
            "activity_id",
            "post_url",
            "date",
            "author",
            "post_type",
            "attachment_type",
            "capture_status",
            "completeness_status",
            "blocker_or_next_action",
            "source_status",
        ],
    )
    write_csv(
        pack_dir / "data/attachment_manifest.csv",
        manifest_rows,
        [
            "activity_id",
            "asset_id",
            "asset_type",
            "source_url",
            "local_path",
            "page_or_slide_count",
            "ocr_status",
            "checksum",
            "text_extract_status",
            "analytic_memo",
        ],
    )
    write_csv(
        pack_dir / "data/external_source_log.csv",
        external_log_rows,
        [
            "activity_id",
            "post_url",
            "label",
            "source_url",
            "canonical_url",
            "http_status",
            "source_type",
            "access_status",
            "title",
            "error",
            "claim_use_rule",
        ],
    )
    write_csv(
        pack_dir / "data/concept_atlas.csv",
        concept_rows,
        [
            "concept",
            "method_family",
            "matched_posts",
            "post_count",
            "novelty",
            "evidence_strength",
            "project_destination",
            "implementation_difficulty",
            "claim_risk",
            "recommended_decision",
        ],
    )
    write_csv(
        pack_dir / "data/linkedin_api_probe_log.csv",
        api_rows,
        ["activity_id", "attempted_urn", "endpoint", "http_status", "result", "blocker"],
    )
    write_csv(
        pack_dir / "data/human_assisted_capture_queue.csv",
        capture_queue_rows,
        [
            "priority",
            "activity_id",
            "post_url",
            "title",
            "attachment_type",
            "next_action",
            "done_definition",
            "safety_rule",
        ],
    )

    docs_dir = pack_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "linkedin_claim_evidence_map.md").write_text(
        build_claim_map(pack_dir, posts, concept_rows, api_rows, run_date=run_date),
        encoding="utf-8",
    )

    errors = validate_outputs(pack_dir, expected_posts=len(posts))
    if errors:
        raise SystemExit("\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument(
        "--probe-api", action="store_true", help="Record official LinkedIn Posts API probe rows."
    )
    parser.add_argument(
        "--api-probe-limit",
        type=int,
        default=3,
        help="Number of posts to probe across activity/share/UGC URN variants.",
    )
    parser.add_argument(
        "--run-date", default=os.environ.get("LCRP_RUN_DATE", date.today().isoformat())
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pack(
        args.pack_dir,
        probe_api=args.probe_api,
        api_probe_limit=args.api_probe_limit,
        run_date=args.run_date,
    )


if __name__ == "__main__":
    main()
