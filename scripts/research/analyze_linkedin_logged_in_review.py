#!/usr/bin/env python3
"""Build logged-in LinkedIn comment/link intake tables and memo.

This script is intentionally conservative: it keeps raw captured artifacts in
place, deduplicates external links, resolves shortlinks when possible, and
creates an incremental research memo without promoting LinkedIn comments to
public-facing claims.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse, urlunparse
from urllib.request import Request, urlopen

PACK = Path("reports/linkedin_credit_risk_denis_burakov/logged_in_review")
QUEUE = Path("data/logged_in_review_queue.csv")
CAPTURE_LOG = Path("data/logged_in_capture_log.csv")
COMMENTS = Path("data/logged_in_visible_comments.csv")
LINKS = Path("data/logged_in_comment_link_candidates.csv")
EXTERNAL_INVENTORY = Path("data/logged_in_external_link_inventory.csv")
POST_SUMMARY = Path("data/logged_in_post_comment_summary.csv")
MEMO = Path("docs/logged_in_review_findings_2026-05-21.md")

NOISE_EXTERNAL_DOMAINS = {"about.linkedin.com"}
SHORTLINK_DOMAINS = {"lnkd.in", "tr.ee"}
HIGH_VALUE_DOMAINS = {
    "arxiv.org",
    "doi.org",
    "dx.doi.org",
    "papers.ssrn.com",
    "bis.org",
    "github.com",
    "gking.harvard.edu",
    "cer.business-school.ed.ac.uk",
    "communities.sas.com",
    "blogs.sas.com",
    "selfexplainml.github.io",
    "modeva.ai",
    "risk.net",
    "files.wmich.edu",
    "stats.oarc.ucla.edu",
    "search.r-project.org",
    "cran.r-project.org",
    "kdd.org",
    "aaai.org",
    "journals.sagepub.com",
    "pages.cs.wisc.edu",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def clean_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        return url.strip()
    query = parse_qs(parsed.query, keep_blank_values=True)
    drop_prefixes = ("utm_",)
    kept = []
    for key, vals in query.items():
        if any(key.lower().startswith(prefix) for prefix in drop_prefixes):
            continue
        for val in vals:
            kept.append((key, val))
    query_text = "&".join(f"{key}={val}" if val else key for key, val in kept)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query_text, ""))


def is_external(url: str) -> bool:
    host = domain(url)
    return bool(host) and "linkedin.com" not in host and host not in NOISE_EXTERNAL_DOMAINS


def classify_source(url: str) -> str:
    host = domain(url)
    path = urlparse(url).path.lower()
    if host in {"arxiv.org", "papers.ssrn.com", "doi.org", "dx.doi.org"}:
        return "academic_or_preprint"
    if host in {"bis.org", "gking.harvard.edu", "files.wmich.edu"}:
        return "official_or_academic_pdf"
    if host in {"github.com", "selfexplainml.github.io", "modeva.ai"}:
        return "code_or_tool"
    if "google.com" in host:
        if "presentation" in path:
            return "google_slides"
        if "spreadsheet" in path:
            return "google_sheet"
        return "google_drive_or_doc"
    if host in {
        "medium.com",
        "deburky.medium.com",
        "blogs.sas.com",
        "communities.sas.com",
        "multithreaded.stitchfix.com",
        "btelligent.com",
        "baeldung.com",
    }:
        return "blog_or_tutorial"
    if host in {"cran.r-project.org", "search.r-project.org"}:
        return "software_documentation"
    if host in {"risk.net", "risk-practitioner.com", "shop.elsevier.com"}:
        return "book_or_publisher"
    if path.endswith(".pdf"):
        return "pdf"
    if host in SHORTLINK_DOMAINS or host == "linktr.ee":
        return "shortlink_or_link_hub"
    return "web_source"


def priority_for(url: str, context: str, source_kind: str) -> str:
    host = domain(url)
    context_l = context.lower()
    if host in HIGH_VALUE_DOMAINS:
        return "high"
    if source_kind == "comment_block" and any(
        term in context_l
        for term in [
            "paper",
            "calibration",
            "gini",
            "woe",
            "scorecard",
            "logistic",
            "model validation",
            "psi",
            "rulefit",
            "shap",
            "naive bayes",
        ]
    ):
        return "high"
    if host in SHORTLINK_DOMAINS or host in {"docs.google.com", "drive.google.com"}:
        return "medium"
    return "low"


def resolve_url(url: str, timeout: float) -> tuple[str, str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return str(response.status), response.geturl(), ""
    except HTTPError as exc:
        return str(exc.code), exc.geturl() or url, str(exc.reason)
    except (URLError, TimeoutError, OSError) as exc:
        return "error", url, f"{type(exc).__name__}: {exc}"


@dataclass
class ExternalRow:
    activity_id: str
    queue_id: str
    source_kind: str
    original_url: str
    anchor_text: str
    context_excerpt: str


def build_external_rows(pack: Path) -> list[ExternalRow]:
    rows: dict[tuple[str, str, str], ExternalRow] = {}
    for link in read_csv(pack / LINKS):
        original = clean_url(link.get("href", ""))
        if not is_external(original):
            continue
        key = (link.get("activity_id", ""), link.get("source_kind", ""), original)
        if key in rows:
            continue
        rows[key] = ExternalRow(
            activity_id=link.get("activity_id", ""),
            queue_id=link.get("queue_id", ""),
            source_kind=link.get("source_kind", ""),
            original_url=original,
            anchor_text=link.get("anchor_text", ""),
            context_excerpt=link.get("context_excerpt", ""),
        )
    return list(rows.values())


def write_outputs(pack: Path, resolve: bool, timeout: float, pause: float) -> None:
    queue = {row["activity_id"]: row for row in read_csv(pack / QUEUE)}
    logs = {row["activity_id"]: row for row in read_csv(pack / CAPTURE_LOG)}
    comments = read_csv(pack / COMMENTS)
    external_rows = build_external_rows(pack)

    inventory: list[dict[str, str]] = []
    for idx, item in enumerate(external_rows, start=1):
        resolved_status = "not_resolved"
        canonical_url = item.original_url
        resolve_error = ""
        host = domain(item.original_url)
        if resolve and (
            host in SHORTLINK_DOMAINS
            or item.source_kind == "comment_block"
            or priority_for(item.original_url, item.context_excerpt, item.source_kind) == "high"
        ):
            resolved_status, canonical_url, resolve_error = resolve_url(item.original_url, timeout)
            time.sleep(pause)
        canonical_host = domain(canonical_url)
        context = re.sub(r"\s+", " ", item.context_excerpt).strip()
        inventory.append(
            {
                "link_id": f"LI-EXT-{idx:03d}",
                "activity_id": item.activity_id,
                "queue_id": item.queue_id,
                "source_kind": item.source_kind,
                "original_url": item.original_url,
                "canonical_url": canonical_url,
                "domain": canonical_host or host,
                "source_type": classify_source(canonical_url),
                "http_status": resolved_status,
                "access_status": "resolved"
                if resolved_status.isdigit() and resolved_status.startswith(("2", "3"))
                else ("unresolved" if resolved_status == "not_resolved" else "blocked_or_error"),
                "priority": priority_for(canonical_url, context, item.source_kind),
                "anchor_text": item.anchor_text,
                "context_excerpt": context[:900],
                "resolve_error": resolve_error[:250],
            }
        )

    write_csv(
        pack / EXTERNAL_INVENTORY,
        inventory,
        [
            "link_id",
            "activity_id",
            "queue_id",
            "source_kind",
            "original_url",
            "canonical_url",
            "domain",
            "source_type",
            "http_status",
            "access_status",
            "priority",
            "anchor_text",
            "context_excerpt",
            "resolve_error",
        ],
    )

    comments_by_activity: dict[str, list[dict[str, str]]] = defaultdict(list)
    for comment in comments:
        comments_by_activity[comment["activity_id"]].append(comment)
    external_by_activity: dict[str, list[dict[str, str]]] = defaultdict(list)
    for link in inventory:
        external_by_activity[link["activity_id"]].append(link)

    summary: list[dict[str, str]] = []
    for activity_id, qrow in queue.items():
        crows = comments_by_activity.get(activity_id, [])
        erows = external_by_activity.get(activity_id, [])
        high = [row for row in erows if row["priority"] == "high"]
        domains = ", ".join(sorted({row["domain"] for row in erows if row["domain"]})[:12])
        high_context = " | ".join(row["context_excerpt"][:180] for row in high[:4])
        if high:
            action = "promote_to_source_reading_queue"
        elif crows:
            action = "append_comment_context_or_archive"
        else:
            action = "close_no_visible_comment_delta"
        summary.append(
            {
                "queue_id": qrow["queue_id"],
                "activity_id": activity_id,
                "title": qrow.get("title", ""),
                "theme": qrow.get("theme", ""),
                "capture_status": logs.get(activity_id, {}).get(
                    "capture_status", qrow.get("logged_in_capture_status", "")
                ),
                "comment_count": str(len(crows)),
                "external_link_count": str(len(erows)),
                "high_priority_external_count": str(len(high)),
                "external_domains": domains,
                "logged_in_incremental_value": "yes" if crows or high else "low",
                "recommended_action": action,
                "high_priority_context_excerpt": high_context,
                "stop_condition_status": "closed_logged_in_surface_captured",
            }
        )

    write_csv(
        pack / POST_SUMMARY,
        summary,
        [
            "queue_id",
            "activity_id",
            "title",
            "theme",
            "capture_status",
            "comment_count",
            "external_link_count",
            "high_priority_external_count",
            "external_domains",
            "logged_in_incremental_value",
            "recommended_action",
            "high_priority_context_excerpt",
            "stop_condition_status",
        ],
    )

    docs_dir = pack / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    queue_status = Counter(row["capture_status"] for row in summary)
    action_counts = Counter(row["recommended_action"] for row in summary)
    priority_counts = Counter(row["priority"] for row in inventory)
    domain_counts = Counter(row["domain"] for row in inventory if row["domain"])
    high_links = [row for row in inventory if row["priority"] == "high"][:35]
    dense_posts = sorted(
        summary,
        key=lambda r: (int(r["comment_count"]), int(r["high_priority_external_count"])),
        reverse=True,
    )[:20]

    memo = [
        "# Logged-In LinkedIn Review Findings - 2026-05-21",
        "",
        "This memo summarizes the logged-in pass over the Denis Burakov LinkedIn backlog. It captures visible comments and comment-shared links using the user's own authenticated browser session. Comments remain private research intake; they are not cited as standalone evidence for public claims.",
        "",
        "## Coverage",
        "",
        f"- Queue rows captured: {len(summary)}",
        f"- Capture status: {dict(queue_status)}",
        f"- Visible comments captured: {len(comments)} across {len(comments_by_activity)} posts",
        f"- External link rows after dedupe by post/source/url: {len(inventory)}",
        f"- Link priority counts: {dict(priority_counts)}",
        f"- Post action counts: {dict(action_counts)}",
        "",
        "## High-Priority Domains",
        "",
    ]
    for host, count in domain_counts.most_common(30):
        memo.append(f"- `{host}`: {count}")
    memo.extend(["", "## High-Priority Links To Read", ""])
    for row in high_links:
        memo.append(
            f"- `{row['queue_id']}` {row['source_type']} [{row['domain']}]({row['canonical_url']}): "
            f"{row['context_excerpt'][:220]}"
        )
    memo.extend(["", "## Dense Comment Threads", ""])
    for row in dense_posts:
        memo.append(
            f"- `{row['queue_id']}` comments={row['comment_count']} high_links={row['high_priority_external_count']} "
            f"action={row['recommended_action']}: {row['title'] or row['theme']}"
        )
    memo.extend(
        [
            "",
            "## Immediate Research Intake",
            "",
            "- Calibration/model selection: Brier decomposition critique, LOESS-smoothed calibration curves with uncertainty, rare-event logistic prior correction.",
            "- Scorecards/WOE: boosted scorecards, WOE after XGBoost, Bayesian/Good WOE formula, WOE reference bibliography.",
            "- Explainability/governance: SHAP-as-WOE framing, intrinsically interpretable ML via EBM/GAM/GamiNet, conceptual soundness/outcome analysis framing.",
            "- Portfolio value: Gini to bad-rate/loss/profit links, Somers D/AR equivalence, Lorenz/CVaR note.",
            "- Drift/monitoring: PSI reference, model validation and monitoring sources, comment-level cautions about sampling and calibration drift.",
            "",
            "## Next Stop Conditions",
            "",
            "- Read or park each high-priority external source in `logged_in_external_link_inventory.csv`.",
            "- Promote only sources with independent evidence status; keep LinkedIn comments as trace/context.",
            "- Use dense threads for appendix caveats, reviewer-defense language, or backlog lanes only when they map to existing paper/book claims.",
        ]
    )
    (pack / MEMO).write_text("\n".join(memo) + "\n", encoding="utf-8")

    print(f"wrote {len(summary)} post summaries")
    print(f"wrote {len(inventory)} external link rows")
    print(f"memo: {pack / MEMO}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=PACK)
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--pause", type=float, default=0.25)
    args = parser.parse_args()
    write_outputs(args.pack_dir, resolve=args.resolve, timeout=args.timeout, pause=args.pause)


if __name__ == "__main__":
    main()
