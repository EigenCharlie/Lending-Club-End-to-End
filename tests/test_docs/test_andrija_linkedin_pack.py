import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "reports" / "linkedin_credit_risk_andrija_djurovic"
DATA_DIR = PACK_DIR / "data"
LOGGED_DIR = PACK_DIR / "logged_in_review"
LOGGED_DATA_DIR = LOGGED_DIR / "data"
BIG_BOOK = REPO_ROOT / "book"
EXTERNAL_CRPTO = Path("/mnt/c/Users/carlos/Documents/Paper_CRPTO")
RETIREMENT_MEMO = REPO_ROOT / "docs" / "research" / "crpto_retirement_and_paper4_role_2026-06-06.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing CSV: {path}"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_andrija_pack_captures_posts_articles_assets_and_sources() -> None:
    posts = _read_csv(DATA_DIR / "posts_index.csv")
    articles = _read_csv(DATA_DIR / "article_capture_log.csv")
    manifest = _read_csv(DATA_DIR / "attachment_manifest.csv")
    external_links = _read_csv(DATA_DIR / "external_link_backlog.csv")
    high_value_sources = _read_csv(DATA_DIR / "high_value_external_source_reading.csv")

    assert len(posts) == 21
    assert {row["author"] for row in posts} == {"Andrija Djurovic"}
    assert len(articles) == 4
    assert {row["capture_status"] for row in articles} == {"captured_http_200_text/html"}
    assert len(external_links) == 58
    assert len(high_value_sources) == 50
    assert (
        sum(1 for row in manifest if row["text_extract_status"] == "pdf_text_extracted_pdftotext")
        == 9
    )
    assert (
        sum(
            1
            for row in high_value_sources
            if row["reading_status"] == "pdf_text_extracted_pdftotext"
        )
        >= 40
    )
    assert any(row["source_type"] == "github" for row in high_value_sources)


def test_andrija_backlog_closes_every_post_article_and_visual_asset() -> None:
    post_backlog = _read_csv(DATA_DIR / "andrija_post_execution_backlog.csv")
    article_backlog = _read_csv(DATA_DIR / "andrija_article_execution_backlog.csv")
    visuals = _read_csv(DATA_DIR / "andrija_visual_read_log.csv")

    assert len(post_backlog) == 21
    assert len(article_backlog) == 4
    assert {row["closure_status"] for row in article_backlog} == {"closed"}
    assert (
        sum(row["closure_status"] == "closed_blocked_public_capture" for row in post_backlog) == 2
    )
    assert all(row["stop_condition"] for row in post_backlog + article_backlog)
    assert all(
        row["possible_executable_or_implementable"] for row in post_backlog + article_backlog
    )
    assert len(visuals) == 52
    assert sum(row["visual_read_status"] == "manual_visual_read_completed" for row in visuals) == 4
    assert (
        sum(row["visual_read_status"] == "manual_visual_triaged_context_only" for row in visuals)
        == 48
    )


def test_andrija_topic_atlas_and_claim_map_are_governed() -> None:
    atlas = _read_csv(DATA_DIR / "andrija_source_topic_atlas.csv")
    concepts = {row["concept"] for row in atlas}
    claim_map = (PACK_DIR / "docs" / "andrija_linkedin_claim_evidence_map_2026-05-25.md").read_text(
        encoding="utf-8"
    )
    decisions = (PACK_DIR / "docs" / "andrija_project_intake_decisions_2026-05-25.md").read_text(
        encoding="utf-8"
    )

    assert {
        "Multi-period average PD backtesting",
        "Model-based discrete PD rating scale calibration",
        "WoE encoding instability",
        "Supervised Macroeconomic Index for IFRS9 FLI",
        "Selective ML support for IRB models",
        "Monotonic binning tooling for credit risk factors",
    }.issubset(concepts)
    assert "LinkedIn material is intake evidence only" in claim_map
    assert "ANDRIJA-POST-021" in claim_map
    assert "Backlog closure counts" in decisions
    assert "High-value readable source rows: 50" in decisions


def test_andrija_intake_is_reflected_in_crpto_book_and_research_memo() -> None:
    decisions = (PACK_DIR / "docs" / "andrija_project_intake_decisions_2026-05-25.md").read_text(
        encoding="utf-8"
    )
    memo = (
        REPO_ROOT / "docs" / "research" / "linkedin_backlog_paper4_estrella_intake_2026-05-21.md"
    ).read_text(encoding="utf-8")
    retirement = RETIREMENT_MEMO.read_text(encoding="utf-8")

    assert "PD backtesting multi-period average testing" in decisions
    assert "WoE encoding instability" in decisions
    assert "Monotonic binning tooling" in decisions
    assert "Paper CRPTO reviewer-defense" in decisions
    assert "Andrija Djurovic / ADSFCR Addendum" in memo
    assert "58 external links" in memo
    assert "fuente de verdad para CRPTO" in retirement
    assert (EXTERNAL_CRPTO / "book/_quarto.yml").exists()


def test_andrija_logged_in_opera_pass_is_closed_and_curated() -> None:
    capture = _read_csv(LOGGED_DATA_DIR / "logged_in_capture_log.csv")
    comments = _read_csv(LOGGED_DATA_DIR / "logged_in_visible_comments.csv")
    links = _read_csv(LOGGED_DATA_DIR / "logged_in_external_link_inventory.csv")
    decisions = _read_csv(LOGGED_DATA_DIR / "logged_in_project_intake_decisions.csv")
    source_queue = _read_csv(LOGGED_DATA_DIR / "logged_in_source_reading_queue.csv")

    assert len(capture) == 37
    assert (
        sum(row["capture_status"] == "logged_in_rendered_capture_complete" for row in capture) == 34
    )
    assert sum(row["capture_status"] == "not_authenticated_or_checkpoint" for row in capture) == 2
    assert sum(row["capture_status"] == "capture_error" for row in capture) == 1
    assert len(comments) == 121
    assert len(links) == 72
    assert sum(row["priority"] == "high" for row in links) == 15
    assert len(decisions) == 37
    assert len(source_queue) == 6
    assert {
        "promote_to_crpto_metric_governance",
        "promote_to_crpto_woe_stability_caveat",
        "park_residual_tree_validation_prototype",
        "promote_pd_backtesting_dependence_caveat",
        "append_model_shift_to_thesis_mrm",
    }.issubset({row["decision"] for row in decisions})
    assert any(
        row["source_title"] == "Statistical Hypothesis Testing for Information Value (IV)"
        for row in source_queue
    )


def test_andrija_logged_in_findings_are_reflected_in_crpto_book() -> None:
    findings = (LOGGED_DIR / "docs" / "andrija_logged_in_review_findings_2026-05-25.md").read_text(
        encoding="utf-8"
    )
    decisions = (
        LOGGED_DIR / "docs" / "andrija_logged_in_project_intake_decisions_2026-05-25.md"
    ).read_text(encoding="utf-8")
    iv_memo = (LOGGED_DIR / "docs" / "iv_hypothesis_testing_source_memo_2026-05-25.md").read_text(
        encoding="utf-8"
    )
    memo = (
        REPO_ROOT / "docs" / "research" / "linkedin_backlog_paper4_estrella_intake_2026-05-21.md"
    ).read_text(encoding="utf-8")
    retirement = RETIREMENT_MEMO.read_text(encoding="utf-8")

    assert "Opera GX / Windows Playwright" in findings
    assert "Visible comments captured: 121" in findings
    assert "J-Divergence" in iv_memo
    assert "preprint_not_peer_reviewed" in findings
    assert "promote_to_crpto_metric_governance" in decisions
    assert "Information Value" in iv_memo
    assert "promote_pd_backtesting_dependence_caveat" in decisions
    assert "Andrija Logged-In P0/P1 Addendum" in memo
    assert "No reconstruir `book/chapters/14-paper-estrella`" in retirement


def test_andrija_logged_in_findings_are_propagated_to_main_book() -> None:
    woe = (
        BIG_BOOK / "chapters" / "05-feature-engineering" / "05a-woe-iv-optbinning.qmd"
    ).read_text(encoding="utf-8")
    backtesting = (
        BIG_BOOK / "chapters" / "07-conformal" / "07d-backtest-monitoring.qmd"
    ).read_text(encoding="utf-8")
    mrm = (
        BIG_BOOK / "chapters" / "10-ifrs9-governance" / "10e-model-risk-management.qmd"
    ).read_text(encoding="utf-8")
    retirement = RETIREMENT_MEMO.read_text(encoding="utf-8")
    thesis = (
        BIG_BOOK / "chapters" / "18-research-agenda" / "18b-thesis-contributions.qmd"
    ).read_text(encoding="utf-8")
    references = (BIG_BOOK / "references.bib").read_text(encoding="utf-8")

    assert "J-Divergence" in woe
    assert "Caveat de reemplazo WOE" in woe
    assert "tamaño efectivo de muestra" in backtesting
    assert "segment.vld" in mrm
    assert "umbral heredado" in mrm
    assert "fuente de verdad para CRPTO" in retirement
    assert "Gobernanza de heuristicas crediticias" in thesis
    assert "rojas2026_iv_hypothesis_testing" in references
    assert "djurovic2026_pdtoolkit" in references
