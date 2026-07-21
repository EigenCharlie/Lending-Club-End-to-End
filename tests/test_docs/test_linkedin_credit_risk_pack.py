import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "reports" / "linkedin_credit_risk_denis_burakov"
DATA_DIR = PACK_DIR / "data"


def _read_csv(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing CSV: {path}"
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_linkedin_pack_reprocesses_prior_59_posts() -> None:
    posts = _read_csv(DATA_DIR / "posts_index.csv")
    inventory = _read_csv(DATA_DIR / "linkedin_corpus_inventory.csv")
    capture_queue = _read_csv(DATA_DIR / "human_assisted_capture_queue.csv")
    capture_log = _read_csv(DATA_DIR / "public_permalink_capture_log.csv")

    assert len(posts) == 59
    assert len(inventory) == len(posts)
    assert len(capture_queue) == len(posts)
    assert len(capture_log) == 59
    assert {row["author"] for row in inventory} == {"Denis Burakov"}
    assert {row["capture_status"] for row in inventory} == {"public_permalink_captured"}
    assert all(
        row["completeness_status"].startswith(("partial_", "complete_")) for row in inventory
    )
    assert {row["capture_status"] for row in capture_log} == {"captured_http_200_text/html"}
    assert sum(int(row["document_count"]) for row in capture_log) == 28


def test_linkedin_attachment_manifest_has_status_for_each_visible_asset() -> None:
    posts = _read_csv(DATA_DIR / "posts_index.csv")
    manifest = _read_csv(DATA_DIR / "attachment_manifest.csv")
    primary_assets = [row for row in manifest if row["asset_id"].endswith("_primary")]
    primary_statuses = {row["text_extract_status"] for row in primary_assets}

    assert len(primary_assets) == len(posts)
    assert {"pdf_text_extracted_pdftotext", "image_pdf_manual_visual_read"} & primary_statuses
    assert {"images_captured_from_public_permalink", "image_manual_visual_read"} & primary_statuses
    assert any(row["ocr_status"] == "ocr_tool_missing" for row in primary_assets)
    assert {"linkedin_document_pdf", "linkedin_image_carousel"}.issubset(
        {row["asset_type"] for row in primary_assets}
    )


def test_linkedin_api_probe_records_official_blocker_without_token() -> None:
    probe_rows = _read_csv(DATA_DIR / "linkedin_api_probe_log.csv")

    assert len(probe_rows) == 9
    assert {row["result"] for row in probe_rows} == {"blocked_unauthenticated_no_oauth_token"}
    assert {row["http_status"] for row in probe_rows} == {"401"}
    assert {"urn:li:activity", "urn:li:share", "urn:li:ugcPost"}.issubset(
        {row["attempted_urn"].rsplit(":", 1)[0] for row in probe_rows}
    )


def test_linkedin_concept_atlas_and_claim_map_are_governed() -> None:
    concepts = _read_csv(DATA_DIR / "concept_atlas.csv")
    concept_names = {row["concept"] for row in concepts}
    claim_map = (PACK_DIR / "docs" / "linkedin_claim_evidence_map.md").read_text(encoding="utf-8")

    expected = {
        "Brier vs Gini/ECE separation",
        "WOE recalibration under drift",
        "Probabilistic LGD via quantile or multiclass bins",
        "SHAP distillation into scorecard-style explanations",
        "Classification/probability intervals and Pearsonify contrast",
    }
    assert expected.issubset(concept_names)
    assert all("after_source_verification" in row["recommended_decision"] for row in concepts)
    assert "LinkedIn material is intake evidence only" in claim_map
    assert (
        "https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api"
        in claim_map
    )


def test_linkedin_overnight_backlog_covers_posts_and_links() -> None:
    post_backlog = _read_csv(DATA_DIR / "post_execution_backlog.csv")
    link_backlog = _read_csv(DATA_DIR / "external_link_backlog.csv")
    high_value_sources = _read_csv(DATA_DIR / "high_value_external_source_reading.csv")
    child_posts = _read_csv(DATA_DIR / "external_linkedin_child_post_backlog.csv")
    decisions = _read_csv(DATA_DIR / "post_execution_decisions.csv")
    plan = (PACK_DIR / "docs" / "overnight_goal_backlog_plan_2026-05-21.md").read_text(
        encoding="utf-8"
    )
    external_memo = (
        PACK_DIR / "docs" / "external_high_value_sources_memo_2026-05-21.md"
    ).read_text(encoding="utf-8")
    visual_memo = (PACK_DIR / "docs" / "manual_visual_reread_memo_2026-05-21.md").read_text(
        encoding="utf-8"
    )

    assert len(post_backlog) == 67
    assert len([row for row in post_backlog if row["source_kind"] == "indexed_denis_post"]) == 59
    assert len(child_posts) == 8
    assert len(link_backlog) == 109
    assert len(high_value_sources) == 21
    assert {row["reading_status"] for row in high_value_sources} == {"readable_http_200"}
    assert all(row["local_readable_path"] for row in high_value_sources)
    assert len(decisions) == len(post_backlog)
    assert {row["backlog_id"] for row in decisions} == {row["backlog_id"] for row in post_backlog}
    assert all(row["post_stop_condition"] for row in post_backlog)
    assert all(row["possible_executable_or_implementable"] for row in post_backlog)
    assert all(row["handling_decision"] and row["stop_condition"] for row in link_backlog)
    assert not any(row["source_type"] == "linkedin_shortlink_unresolved" for row in link_backlog)
    assert "Global Stop Rule" in plan
    assert "21 external links marked `read_as_potential_evidence`" in external_memo
    assert "direct visual reading" in visual_memo
    assert "POST-053" in visual_memo


def test_second_linkedin_ingest_closes_discovered_posts_articles_and_official_source() -> None:
    second_dir = PACK_DIR / "second_ingest"
    second_data = second_dir / "data"
    posts = _read_csv(second_data / "posts_index.csv")
    articles = _read_csv(second_data / "article_candidates.csv")
    backlog = _read_csv(second_data / "second_ingest_execution_backlog.csv")
    concepts = _read_csv(second_data / "second_ingest_concept_atlas.csv")
    high_value_sources = _read_csv(second_data / "high_value_external_source_reading.csv")
    visuals = _read_csv(second_data / "second_ingest_visual_read_log.csv")
    memo = (second_dir / "docs" / "second_ingest_execution_memo_2026-05-21.md").read_text(
        encoding="utf-8"
    )
    gap_report = (second_dir / "docs" / "second_ingest_profile_gap_report_2026-05-21.md").read_text(
        encoding="utf-8"
    )

    assert len(posts) == 15
    assert {"7152948849597132801", "7168870006380740608"}.issubset(
        {row["activity_id"] for row in posts}
    )
    assert len(articles) == 12
    assert len(backlog) == 27
    assert {row["closure_status"] for row in backlog} == {"closed"}
    assert all(row["stop_condition"] for row in backlog)
    assert all(row["possible_executable_or_implementable"] for row in backlog)
    assert len(concepts) >= 10
    assert {
        "WOE as centered log odds with uncertainty",
        "Profit scoring beyond PD",
        "Official ML governance for internal credit risk models",
    }.issubset({row["concept"] for row in concepts})
    assert any(
        row["source_type"] == "pdf"
        and row["project_use"] == "official_supervisory_guidance"
        and row["reading_status"] == "pdf_text_extracted_pdftotext"
        for row in high_value_sources
    )
    assert any(row["visual_read_status"] == "manual_visual_read_completed" for row in visuals)
    assert "Backlog items closed: 27/27" in memo
    assert "logged-in visible-browser" in gap_report


def test_logged_in_review_closes_comments_links_sources_and_project_decisions() -> None:
    logged_dir = PACK_DIR / "logged_in_review"
    logged_data = logged_dir / "data"

    queue = _read_csv(logged_data / "logged_in_review_queue.csv")
    comments = _read_csv(logged_data / "logged_in_visible_comments.csv")
    inventory = _read_csv(logged_data / "logged_in_external_link_inventory.csv")
    source_status = _read_csv(logged_data / "logged_in_source_reading_status.csv")
    recoveries = _read_csv(logged_data / "logged_in_alternate_source_recoveries.csv")
    findings = (logged_dir / "docs" / "logged_in_review_findings_2026-05-21.md").read_text(
        encoding="utf-8"
    )
    decisions = (
        logged_dir / "docs" / "logged_in_project_intake_decisions_2026-05-21.md"
    ).read_text(encoding="utf-8")

    assert len(queue) == 80
    assert {row["logged_in_capture_status"] for row in queue} == {
        "logged_in_rendered_capture_complete"
    }
    assert len(comments) >= 500
    assert len({row["activity_id"] for row in comments}) >= 60
    assert len(inventory) >= 200
    assert sum(1 for row in inventory if row["priority"] == "high") >= 70
    assert sum(1 for row in source_status if row["evidence_status"] == "readable") >= 25
    assert sum(1 for row in recoveries if row["evidence_status"] == "readable") >= 6
    assert "Visible comments captured: 503" in findings
    assert "Promoted To The Quarto Book" in decisions
    assert "Paper Estrella" in decisions
    assert "The logged-in pass is closed for the current corpus" in decisions
