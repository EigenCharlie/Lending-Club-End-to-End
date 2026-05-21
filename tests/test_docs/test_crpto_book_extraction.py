from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CRPTO_BOOK = REPO_ROOT / "papers" / "paper_crpto_book"


def test_paper_crpto_book_scaffold_exists_and_names_public_surface() -> None:
    required = [
        "_quarto.yml",
        "index.qmd",
        "chapters/01-ijds-target-and-claim.qmd",
        "chapters/02-book-to-crpto-intake.qmd",
        "chapters/03-manuscript-body.qmd",
        "chapters/04-online-supplement.qmd",
        "chapters/05-thesis-chapter.qmd",
        "chapters/06-open-boundaries.qmd",
        "chapters/07-project-expansion-map.qmd",
        "chapters/08-roadmap-and-gates.qmd",
        "references.qmd",
    ]

    for rel in required:
        assert (CRPTO_BOOK / rel).exists(), f"Missing Paper CRPTO file: {rel}"

    index_text = (CRPTO_BOOK / "index.qmd").read_text(encoding="utf-8")
    target_text = (CRPTO_BOOK / "chapters/01-ijds-target-and-claim.qmd").read_text(encoding="utf-8")
    body_text = (CRPTO_BOOK / "chapters/03-manuscript-body.qmd").read_text(encoding="utf-8")
    supplement_text = (CRPTO_BOOK / "chapters/04-online-supplement.qmd").read_text(encoding="utf-8")

    assert "Paper CRPTO" in index_text
    assert "Paper Estrella` queda como alias interno histórico" in index_text
    assert "INFORMS Journal on Data Science" in target_text
    assert "25 páginas" in target_text
    assert "Page budget inicial" in body_text
    assert "Checklist de release doble anónimo" in supplement_text


def test_paper_crpto_book_records_expansion_roadmap() -> None:
    config_text = (CRPTO_BOOK / "_quarto.yml").read_text(encoding="utf-8")
    expansion_text = (CRPTO_BOOK / "chapters/07-project-expansion-map.qmd").read_text(
        encoding="utf-8"
    )
    roadmap_text = (CRPTO_BOOK / "chapters/08-roadmap-and-gates.qmd").read_text(encoding="utf-8")

    assert "chapters/07-project-expansion-map.qmd" in config_text
    assert "chapters/08-roadmap-and-gates.qmd" in config_text
    assert "Evidence spine claim -> artifact -> test" in expansion_text
    assert "Paper 4 living lab" in expansion_text
    assert "Roadmap IJDS: 6 meses" in roadmap_text
    assert "Roadmap tesis: 12 meses" in roadmap_text
    assert "Reviewer-defense bank inicial" in roadmap_text
    assert "Promotion gate" in roadmap_text


def test_crpto_audit_records_book_to_paper_decisions() -> None:
    audit = REPO_ROOT / "docs" / "research" / "quarto_book_crpto_full_audit_2026-05-21.md"
    text = audit.read_text(encoding="utf-8")

    assert "Paper CRPTO mini-book" in text
    assert "Must enter IJDS body" in text
    assert "Must enter IJDS supplement" in text
    assert "Must enter thesis chapter" in text
    assert "No LinkedIn-only claim is promoted as public evidence" in text


def test_crpto_expansion_audit_records_new_editorial_controls() -> None:
    audit = REPO_ROOT / "docs" / "research" / "crpto_mini_book_expansion_audit_2026-05-21.md"
    text = audit.read_text(encoding="utf-8")

    assert "IJDS paper" in text
    assert "master's thesis" in text
    assert "evidence spine" in text
    assert "page-budget ledger" in text
    assert "negative-results registry" in text
