#!/usr/bin/env python3
"""Build the public LinkedIn seed pack for Andrija Djurovic.

The script uses only public LinkedIn/profile pages and public external sources.
It prepares the same intake structure used for the Denis Burakov pack, but keeps
Andrija's corpus separate so later captures, source resolution, and manuscript
decisions remain traceable post-by-post.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_PACK_DIR = Path("reports/linkedin_credit_risk_andrija_djurovic")
PROFILE_URL = "https://se.linkedin.com/in/andrija-djurovic"
RECENT_ACTIVITY_URL = "https://www.linkedin.com/in/andrija-djurovic/recent-activity/posts/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"

SEARCH_SEED_POST_URLS = [
    "https://fr.linkedin.com/posts/andrija-djurovic_python-r-rstats-activity-6965364290866282496-JXRW",
    "https://www.linkedin.com/posts/andrija-djurovic_%F0%9D%90%88%F0%9D%90%85%F0%9D%90%91%F0%9D%90%92%F0%9D%9F%97-%F0%9D%90%85%F0%9D%90%A8%F0%9D%90%AB%F0%9D%90%B0%F0%9D%90%9A%F0%9D%90%AB%F0%9D%90%9D-%F0%9D%90%8B%F0%9D%90%A8%F0%9D%90%A8%F0%9D%90%A4%F0%9D%90%A2%F0%9D%90%A7%F0%9D%90%A0-activity-7392821942728273920-sYiG",
    "https://www.linkedin.com/posts/andrija-djurovic_modelbasedpdratingscalecalibration-activity-7435946096050339840-VDj7",
    "https://www.linkedin.com/posts/andrija-djurovic_%F0%9D%90%91-%F0%9D%90%88%F0%9D%90%91%F0%9D%90%81-%F0%9D%90%AD%F0%9D%90%A8%F0%9D%90%A8%F0%9D%90%A5%F0%9D%90%A4%F0%9D%90%A2%F0%9D%90%AD-my-r-irb-toolkit-activity-7330472362141761538-5UJ8",
    "https://www.linkedin.com/posts/andrija-djurovic_probability-of-default-rating-modeling-with-activity-7112329977517215745-eeri",
    "https://www.linkedin.com/posts/andrija-djurovic_creditrisk-irb-ifrs9-activity-7420725800745754625-CRhr",
    "https://www.linkedin.com/posts/andrija-djurovic_github-andrija-djurovicpdtoolkit-collection-activity-6886388764965392384-fZDY",
    "https://www.linkedin.com/posts/andrija-djurovic_i-have-created-a-github-repository-dedicated-activity-7163079427961090049-iXzW",
    "https://www.linkedin.com/posts/andrija-djurovic_our-paper-drifts-shifts-and-instabilities-activity-7342064984069173248-DTrS",
    "https://www.linkedin.com/posts/andrija-djurovic_woeencodinginstability-activity-7296422990588579840-7lwj",
]

POST_INDEX_FIELDS = [
    "n",
    "activity_id",
    "post_url",
    "date",
    "author",
    "title",
    "relevance",
    "theme",
    "summary_es",
    "tesis_use",
    "attachment_type",
    "external_links",
    "short_snippet_under_25_words",
    "discovery_source",
]


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


def fetch_text(url: str, timeout: int = 30) -> tuple[str, str, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return data, resp.headers.get("content-type", ""), resp.status, resp.geturl()


def strip_tags(page_html: str) -> str:
    parser = TextExtractor()
    parser.feed(page_html)
    return parser.text()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_url(raw_url: str) -> str:
    url = html.unescape(raw_url).replace("\\u002F", "/").replace("\\/", "/")
    url = url.split("\\n", 1)[0].split("\\r", 1)[0]
    url = url.split("?trk=", 1)[0].split("&trk=", 1)[0]
    url = url.split("?", 1)[0] if "linkedin.com/posts/" in url else url
    for sep in ['"', "'", "<", ">", "\\\\"]:
        if sep in url:
            url = url.split(sep, 1)[0]
    return url.rstrip(".,)]}")


def normalize_post_url(raw_url: str) -> str:
    return clean_url(raw_url).split("?", 1)[0]


def activity_id_from_url(url: str) -> str:
    match = re.search(r"activity-(\d+)", url)
    return match.group(1) if match else ""


def extract_public_post_urls(page_html: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(
        r"https://www\.linkedin\.com/posts/andrija-djurovic[^\"<\\\s]+", page_html
    ):
        url = normalize_post_url(match.group(0))
        if url not in urls:
            urls.append(url)
    return urls


def extract_article_urls(page_html: str) -> list[dict[str, str]]:
    urls: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r'href="([^"]+)"[^>]*>(.*?)</a>', page_html, flags=re.DOTALL):
        href = html.unescape(match.group(1))
        if "linkedin.com/pulse/" not in href:
            continue
        if "xenith" in href.lower():
            continue
        title = re.sub(r"\s+", " ", strip_tags(match.group(2))).strip()
        if href not in seen:
            seen.add(href)
            urls.append(
                {
                    "article_id": safe_id(href.rsplit("/", 1)[-1]),
                    "source_url": href,
                    "title": title or href.rsplit("/", 1)[-1].replace("-", " ").title(),
                    "parent_activity_id": "",
                    "source_status": "public_linkedin_article",
                }
            )
    return urls


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def extract_jsonld_post(page_html: str) -> dict[str, object]:
    for script in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        page_html,
        flags=re.DOTALL,
    ):
        try:
            data = json.loads(html.unescape(script).strip())
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if item.get("@type") == "DiscussionForumPosting":
                    return item
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return {}


def extract_title(page_html: str, post_text: str) -> str:
    jsonld = extract_jsonld_post(page_html)
    text = str(jsonld.get("text") or post_text)
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line and not line.startswith("#"):
            return line[:180]
    title_match = re.search(r"<title[^>]*>(.*?)</title>", page_html, flags=re.DOTALL)
    if title_match:
        return re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip()[:180]
    return "LinkedIn post"


def extract_main_post_text(page_html: str) -> str:
    jsonld = extract_jsonld_post(page_html)
    if jsonld.get("text"):
        return str(jsonld["text"]).strip()
    match = re.search(
        r'<p[^>]+data-test-id="main-feed-activity-card__commentary"[^>]*>(.*?)</p>',
        page_html,
        flags=re.DOTALL,
    )
    if match:
        return strip_tags(match.group(1))
    meta = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page_html, flags=re.DOTALL)
    return html.unescape(meta.group(1)).strip() if meta else ""


def extract_visible_comments(page_html: str) -> list[str]:
    lines = strip_tags(page_html).splitlines()
    comments: list[str] = []
    for idx, line in enumerate(lines):
        if line != "Report this comment":
            continue
        chunk: list[str] = []
        for next_line in lines[idx + 1 : idx + 8]:
            if next_line in {"Like", "Reply", "To view or add a comment, sign in"}:
                break
            if next_line.startswith("Like ") or next_line.endswith("Reactions"):
                break
            if next_line and not next_line.endswith("linkedin.com"):
                chunk.append(next_line)
        text = " ".join(chunk).strip()
        if text and text not in comments:
            comments.append(text)
    return comments


def extract_external_links(page_html: str, post_text: str, comments: list[str]) -> list[str]:
    haystacks = [post_text, "\n".join(comments), page_html]
    links: list[str] = []
    for haystack_idx, haystack in enumerate(haystacks):
        for raw in re.findall(r"https?://[^\s\"'<>)]+", haystack):
            url = clean_url(raw)
            host = urlparse(url).netloc.lower()
            if not host:
                continue
            if host in {"static.licdn.com", "media.licdn.com", "schema.org"}:
                continue
            if host == "www.linkedin.com" and "/feed/update/" in url:
                continue
            if host.endswith("linkedin.com") and "/posts/" in url:
                continue
            if host.endswith("linkedin.com") and "/pulse/" in url and haystack_idx == 2:
                continue
            if host.endswith("linkedin.com") and not any(token in url for token in ["/pulse/"]):
                continue
            if url not in links:
                links.append(url)
    return links


def attachment_type(page_html: str, external_links: list[str]) -> str:
    if "feedshare-document-master-manifest" in page_html:
        return "linkedin_document_deck"
    if "feedshare-document-cover-images" in page_html:
        return "linkedin_document_preview"
    if "feedshare-shrink" in page_html or 'data-test-id="feedshare-image"' in page_html:
        return "linkedin_image_or_carousel"
    if external_links:
        return "external_links"
    return "none_recorded"


def theme_for(text: str, title: str) -> tuple[str, str, str, str]:
    lowered = f"{title} {text}".lower()
    if any(k in lowered for k in ["backtesting", "binomial", "z-score", "normal test"]):
        return (
            "Alta",
            "PD backtesting and validation",
            "Post sobre validacion/backtesting de PD; revisar si refuerza la capa de materialidad, z-score, binomial exacto o pruebas multi-periodo.",
            "Paper CRPTO / libro Ch06-Ch10; Paper 4 si cambia una tabla de validacion o una respuesta a revisores.",
        )
    if any(k in lowered for k in ["ifrs9", "forward-looking", "macroeconomic", "smi"]):
        return (
            "Alta",
            "IFRS9 forward-looking modeling",
            "Material de FLI/IFRS9 y modelos macro; sirve para diagnostico y narrativa de limitaciones, no para mover el champion PD.",
            "Libro Ch10 y mini libro CRPTO; posible apendice de tesis sobre FLI/escenarios.",
        )
    if any(k in lowered for k in ["heterogeneity", "representativeness", "c2st", "monotonicity"]):
        return (
            "Alta",
            "Representativeness, heterogeneity, monotonicity",
            "Material de gobierno/model risk que puede fortalecer auditoria de estructura, C2ST y monotonicidad.",
            "Libro Ch10, paper estrella defensa de gobernanza, Paper 4 si hay diagnostico reproducible.",
        )
    if any(k in lowered for k in ["woe", "scorecard", "binning", "weight of evidence"]):
        return (
            "Alta",
            "WOE, binning and scorecard stability",
            "Material de binning/WOE/scorecards; util para estabilidad, interpretabilidad y caveats de scorecards.",
            "Libro Ch05-Ch06; Paper 4 solo como apendice/prototipo acotado.",
        )
    if any(k in lowered for k in ["lgd", "ead", "somers", "cure"]):
        return (
            "Media",
            "LGD/EAD validation and discriminatory power",
            "Material LGD/EAD; valioso para frontera futura pero menos directo para Lending Club PD actual.",
            "Libro Ch07-Ch10 y tesis; aparcar para Paper 4 salvo que cierre un claim LGD/ECL.",
        )
    if any(k in lowered for k in ["pdtoolkit", "github", "cran", "package"]):
        return (
            "Media",
            "Tooling and reproducible credit-risk packages",
            "Paquetes/repos de credito; leer para ideas implementables y separarlos de evidencia academica.",
            "Repositorio intelectual, MLOps/gobernanza; no promover como evidencia peer-reviewed.",
        )
    return (
        "Media",
        "Credit-risk modeling source discovery",
        "Post candidato a lectura; clasificar despues de resolver adjuntos/enlaces.",
        "Research pack; promover solo si cambia libro, paper o apendice.",
    )


def short_snippet(text: str, limit: int = 24) -> str:
    words = re.findall(r"\S+", re.sub(r"\s+", " ", text).strip())
    return " ".join(words[:limit])


def build_profile_candidates(
    profile_html: str, recent_html: str, post_urls: list[str]
) -> list[dict[str, str]]:
    activity_ids = sorted(
        set(re.findall(r"activity-(\d+)", profile_html + recent_html)), reverse=True
    )
    post_activity_ids = {activity_id_from_url(url) for url in post_urls}
    rows: list[dict[str, str]] = []
    for activity_id in activity_ids:
        rows.append(
            {
                "activity_id": activity_id,
                "classification": "own_post_permalink_visible"
                if activity_id in post_activity_ids
                else "profile_activity_without_own_post_permalink",
                "post_url": next((url for url in post_urls if activity_id in url), ""),
                "handling": "capture_public_permalink"
                if activity_id in post_activity_ids
                else "archive_reaction_or_comment_candidate_until_login_review",
                "stop_condition": "Stop when permalink text/assets/links are captured."
                if activity_id in post_activity_ids
                else "Stop unless logged-in review confirms this is Andrija-authored and credit-risk relevant.",
            }
        )
    return rows


def build_pack(pack_dir: Path, sleep_seconds: float) -> None:
    data_dir = pack_dir / "data"
    raw_dir = pack_dir / "raw_public_profile"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    profile_html, _, profile_status, profile_final_url = fetch_text(PROFILE_URL)
    recent_error = ""
    try:
        recent_html, _, recent_status, recent_final_url = fetch_text(RECENT_ACTIVITY_URL)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        recent_html = profile_html
        recent_status = ""
        recent_final_url = RECENT_ACTIVITY_URL
        recent_error = f"{type(exc).__name__}: {exc}"
    profile_path = raw_dir / "profile.html"
    recent_path = raw_dir / "recent_activity_posts.html"
    profile_text_path = raw_dir / "profile_text.txt"
    profile_path.write_text(profile_html, encoding="utf-8")
    recent_path.write_text(recent_html, encoding="utf-8")
    profile_text_path.write_text(strip_tags(profile_html), encoding="utf-8")

    profile_discovered_urls = extract_public_post_urls(profile_html) + extract_public_post_urls(
        recent_html
    )
    discovered_urls = list(profile_discovered_urls)
    for url in SEARCH_SEED_POST_URLS:
        if url not in discovered_urls:
            discovered_urls.append(url)
    deduped_urls: list[str] = []
    for url in discovered_urls:
        normalized = normalize_post_url(url)
        if normalized and normalized not in deduped_urls:
            deduped_urls.append(normalized)

    post_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    comment_rows: list[dict[str, object]] = []
    capture_seed_rows: list[dict[str, object]] = []

    for idx, post_url in enumerate(deduped_urls, start=1):
        activity_id = activity_id_from_url(post_url)
        source = (
            "public_profile_recent_activity"
            if post_url in profile_discovered_urls
            else "search_seed"
        )
        capture_status = ""
        error = ""
        post_text = ""
        comments: list[str] = []
        links: list[str] = []
        date_published = ""
        title = "LinkedIn post"
        post_attachment_type = "unknown_pending_capture"

        try:
            page_html, content_type, status, final_url = fetch_text(post_url)
            post_text = extract_main_post_text(page_html)
            comments = extract_visible_comments(page_html)
            links = extract_external_links(page_html, post_text, comments)
            jsonld = extract_jsonld_post(page_html)
            date_published = str(jsonld.get("datePublished") or "")
            title = extract_title(page_html, post_text)
            post_attachment_type = attachment_type(page_html, links)
            capture_status = f"seed_checked_http_{status}_{content_type.split(';')[0]}"
            capture_seed_rows.append(
                {
                    "n": idx,
                    "activity_id": activity_id,
                    "post_url": post_url,
                    "status": status,
                    "final_url": final_url,
                    "content_type": content_type,
                    "post_text_length": len(post_text),
                    "visible_comment_count": len(comments),
                    "external_link_count": len(links),
                    "error": "",
                }
            )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            capture_status = "seed_check_error"
            error = f"{type(exc).__name__}: {exc}"
            capture_seed_rows.append(
                {
                    "n": idx,
                    "activity_id": activity_id,
                    "post_url": post_url,
                    "status": "",
                    "final_url": "",
                    "content_type": "",
                    "post_text_length": 0,
                    "visible_comment_count": 0,
                    "external_link_count": 0,
                    "error": error,
                }
            )

        relevance, theme, summary_es, tesis_use = theme_for(post_text, title)
        external_links = " | ".join(links)
        post_rows.append(
            {
                "n": str(idx),
                "activity_id": activity_id,
                "post_url": post_url,
                "date": date_published,
                "author": "Andrija Djurovic",
                "title": title,
                "relevance": relevance,
                "theme": theme,
                "summary_es": summary_es,
                "tesis_use": tesis_use,
                "attachment_type": post_attachment_type,
                "external_links": external_links,
                "short_snippet_under_25_words": short_snippet(post_text),
                "discovery_source": source,
            }
        )
        inventory_rows.append(
            {
                "activity_id": activity_id,
                "post_url": post_url,
                "date": date_published,
                "author": "Andrija Djurovic",
                "post_type": "linkedin_activity_post",
                "attachment_type": post_attachment_type,
                "capture_status": capture_status,
                "completeness_status": "seed_only_pending_public_permalink_capture",
                "blocker_or_next_action": error
                or "Run public permalink capture, resolve external links, and write post-level decision.",
                "source_status": "public_linkedin_seed",
            }
        )
        if post_attachment_type not in {"external_links", "none_recorded"}:
            manifest_rows.append(
                {
                    "activity_id": activity_id,
                    "asset_id": f"{activity_id}_primary",
                    "asset_type": post_attachment_type,
                    "source_url": post_url,
                    "local_path": "",
                    "page_or_slide_count": "",
                    "ocr_status": "pending_public_capture_or_manual_read",
                    "checksum": "",
                    "text_extract_status": "pending_public_capture",
                    "analytic_memo": "Primary LinkedIn attachment inferred from public page; capture assets before evidence use.",
                }
            )
        for link_idx, url in enumerate(links, start=1):
            manifest_rows.append(
                {
                    "activity_id": activity_id,
                    "asset_id": f"{activity_id}_external_{link_idx:02d}",
                    "asset_type": "external_link",
                    "source_url": url,
                    "local_path": "",
                    "page_or_slide_count": "",
                    "ocr_status": "not_applicable_until_source_downloaded",
                    "checksum": "",
                    "text_extract_status": "pending_external_source_review",
                    "analytic_memo": "Resolve and read canonical source before promotion.",
                }
            )
        for comment_idx, comment in enumerate(comments, start=1):
            comment_rows.append(
                {
                    "activity_id": activity_id,
                    "comment_index": comment_idx,
                    "comment_text": comment,
                    "contains_link": "yes" if re.search(r"https?://", comment) else "no",
                    "source_status": "public_visible_comment",
                    "use_rule": "Use only for source discovery or context; do not promote comment-only claims.",
                }
            )
        time.sleep(sleep_seconds)

    article_rows = extract_article_urls(profile_html)
    profile_candidate_rows = build_profile_candidates(profile_html, recent_html, deduped_urls)

    write_csv(data_dir / "posts_index.csv", post_rows, POST_INDEX_FIELDS)
    write_csv(
        data_dir / "linkedin_corpus_inventory.csv",
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
        data_dir / "attachment_manifest.csv",
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
        data_dir / "public_seed_capture_log.csv",
        capture_seed_rows,
        [
            "n",
            "activity_id",
            "post_url",
            "status",
            "final_url",
            "content_type",
            "post_text_length",
            "visible_comment_count",
            "external_link_count",
            "error",
        ],
    )
    write_csv(
        data_dir / "visible_comment_log.csv",
        comment_rows,
        [
            "activity_id",
            "comment_index",
            "comment_text",
            "contains_link",
            "source_status",
            "use_rule",
        ],
    )
    write_csv(
        data_dir / "article_candidates.csv",
        article_rows,
        ["article_id", "source_url", "title", "parent_activity_id", "source_status"],
    )
    write_csv(
        data_dir / "profile_public_activity_candidates.csv",
        profile_candidate_rows,
        ["activity_id", "classification", "post_url", "handling", "stop_condition"],
    )
    write_csv(
        data_dir / "profile_source_log.csv",
        [
            {
                "source_url": PROFILE_URL,
                "final_url": profile_final_url,
                "http_status": profile_status,
                "local_path": str(profile_path.relative_to(pack_dir)),
                "text_path": str(profile_text_path.relative_to(pack_dir)),
                "checksum": checksum(profile_path),
                "notes": "Public profile snapshot used only for post/article discovery.",
            },
            {
                "source_url": RECENT_ACTIVITY_URL,
                "final_url": recent_final_url,
                "http_status": recent_status,
                "local_path": str(recent_path.relative_to(pack_dir)),
                "text_path": "",
                "checksum": checksum(recent_path),
                "notes": recent_error
                or "Public recent-activity/posts snapshot used for permalink discovery.",
            },
        ],
        [
            "source_url",
            "final_url",
            "http_status",
            "local_path",
            "text_path",
            "checksum",
            "notes",
        ],
    )

    docs_dir = pack_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "ingest_plan_2026-05-25.md").write_text(
        f"""# Andrija Djurovic LinkedIn Intake Plan

Date: {date.today().isoformat()}

## Scope

- Profile: {PROFILE_URL}
- Public recent activity page: {RECENT_ACTIVITY_URL}
- Public seed posts discovered: {len(post_rows)}
- Public article candidates discovered: {len(article_rows)}
- Profile activity IDs observed: {len(profile_candidate_rows)}

## Method

1. Use public LinkedIn/profile pages and public permalink pages only.
2. Save all post/comment/source traces under this private research pack.
3. Resolve `lnkd.in` shortlinks and classify canonical sources before any claim promotion.
4. Read GitHub/PDF/package sources as canonical materials when available.
5. Treat LinkedIn-only posts/comments as intake/context, not public scholarly evidence.

## Stop Rules

- Each post stops when text, attachments, visible comments, links, and canonical source status are captured or explicitly blocked.
- Each link stops when it is resolved to a canonical source, archived as low relevance, or marked blocked.
- Implementation happens only when the item changes a Quarto chapter, Paper 4 appendix/prototype, Paper CRPTO defense, or a thesis-roadmap gap.

## Known Access Note

No local Chrome CDP session was available at pack creation time, so this first pass uses public pages. Logged-in review can be appended later without changing source IDs.
""",
        encoding="utf-8",
    )

    print(
        f"Built Andrija seed pack at {pack_dir}: {len(post_rows)} posts, "
        f"{len(article_rows)} articles, {len(manifest_rows)} manifest rows."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pack(args.pack_dir, args.sleep_seconds)


if __name__ == "__main__":
    main()
