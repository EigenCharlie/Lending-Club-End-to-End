#!/usr/bin/env python3
"""Fetch alternate source URLs for high-value blocked logged-in links."""

from __future__ import annotations

from fetch_linkedin_logged_in_sources import PACK, fetch_one, write_csv

RECOVERY_ROWS = [
    {
        "link_id": "LI-REC-001",
        "queue_id": "LI-LOGIN-037",
        "activity_id": "7421223494132314112",
        "source_type": "academic_or_preprint",
        "canonical_url": "https://gking.harvard.edu/publication/logistic-regression-in-rare-events-data/",
        "blocked_original": "https://gking.harvard.edu/files/abs/0s-abs.shtml",
        "recovery_note": "Harvard DASH copy of King and Zeng rare-events logistic regression.",
    },
    {
        "link_id": "LI-REC-002",
        "queue_id": "LI-LOGIN-013",
        "activity_id": "7429434643583725571",
        "source_type": "academic_or_preprint",
        "canonical_url": "https://www.bundesbank.de/resource/blob/704150/b9fa10a16dfff3c98842581253f6d141/472B63F073F071307366337C94F8C870/2003-10-01-dkp-01-data.pdf",
        "blocked_original": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2793951",
        "recovery_note": "EconStor PDF for Engelmann, Hayden, and Tasche discriminative power paper.",
    },
    {
        "link_id": "LI-REC-003",
        "queue_id": "LI-LOGIN-013",
        "activity_id": "7429434643583725571",
        "source_type": "academic_or_preprint",
        "canonical_url": "https://tzin.bgu.ac.il/~shalit/Publications/Portfolio%20Risk%20Management%20Using%20The%20Lorenz%20CurveJPM.pdf",
        "blocked_original": "https://www.researchgate.net/publication/228906308_Portfolio_Risk_Management_Using_the_Lorenz_Curve",
        "recovery_note": "Author-hosted PDF for Portfolio Risk Management Using the Lorenz Curve.",
    },
    {
        "link_id": "LI-REC-004",
        "queue_id": "LI-LOGIN-060",
        "activity_id": "7140373482889961472",
        "source_type": "academic_or_preprint",
        "canonical_url": "https://pdfs.semanticscholar.org/e12b/afcfe7b29cb6af7bae5f5615770712272f8c.pdf",
        "blocked_original": "https://www.hindawi.com/journals/jmath/2013/848271/",
        "recovery_note": "Hindawi direct PDF for Metric Divergence Measures and Information Value in Credit Scoring.",
    },
    {
        "link_id": "LI-REC-005",
        "queue_id": "LI-LOGIN-075",
        "activity_id": "7117177687510233088",
        "source_type": "academic_or_preprint",
        "canonical_url": "https://cdn.aaai.org/KDD/1998/KDD98-015.pdf",
        "blocked_original": "https://aaai.org/papers/00101-KDD98-015-interpretable-boosted-naive-bayes-classification/",
        "recovery_note": "AAAI CDN PDF for Interpretable Boosted Naive Bayes Classification.",
    },
    {
        "link_id": "LI-REC-006",
        "queue_id": "LI-LOGIN-048",
        "activity_id": "7386904709726797824",
        "source_type": "academic_or_preprint",
        "canonical_url": "https://www.reacfin.com/wp-content/uploads/2016/12/2023-07-26-XAI-Conference-Cost-of-explainability-with-credit-scoring-JDE-presentation-v2.0-2.pdf",
        "blocked_original": "https://scholar.google.fr/citations?view_op=view_citation&hl=fr&user=S9OI0xUAAAAJ&citation_for_view=S9OI0xUAAAAJ:2osOgNQ5qMEC",
        "recovery_note": "Public presentation for Cost of Explainability in AI: An Example with Credit Scoring Models.",
    },
    {
        "link_id": "LI-REC-007",
        "queue_id": "LI-LOGIN-013",
        "activity_id": "7429434643583725571",
        "source_type": "web_source",
        "canonical_url": "https://datascience.stackexchange.com/questions/134389/stackprinter?service=datascience.stackexchange&language=en&hideAnswers=false&width=700",
        "blocked_original": "https://datascience.stackexchange.com/questions/134389/is-class-imbalance-really-a-problem-in-machine-learning",
        "recovery_note": "StackPrinter attempt for class imbalance discussion.",
    },
]


def main() -> None:
    pack = PACK
    rows: list[dict[str, str]] = []
    for seed in RECOVERY_ROWS:
        fetched = fetch_one(seed, pack, timeout=35.0)
        fetched["blocked_original"] = seed["blocked_original"]
        fetched["recovery_note"] = seed["recovery_note"]
        rows.append(fetched)

    out = pack / "data" / "logged_in_alternate_source_recoveries.csv"
    fields = [
        "link_id",
        "queue_id",
        "activity_id",
        "source_type",
        "canonical_url",
        "fetch_url",
        "http_status",
        "content_type",
        "raw_path",
        "text_path",
        "byte_count",
        "text_chars",
        "extraction_status",
        "evidence_status",
        "notes",
        "blocked_original",
        "recovery_note",
    ]
    write_csv(out, rows, fields)

    readable = [row for row in rows if row["evidence_status"] == "readable"]
    memo = [
        "# Logged-In Blocker Recovery Memo - 2026-05-21",
        "",
        f"- Alternate sources attempted: {len(rows)}",
        f"- Recovered readable sources: {len(readable)}",
        "",
    ]
    for row in rows:
        memo.append(
            f"- `{row['link_id']}` `{row['queue_id']}` evidence={row['evidence_status']} "
            f"status={row['http_status']} [{row['canonical_url']}]({row['canonical_url']}) "
            f"path=`{row['text_path']}` note={row['recovery_note']}"
        )
    (pack / "docs" / "logged_in_blocker_recovery_memo_2026-05-21.md").write_text(
        "\n".join(memo) + "\n",
        encoding="utf-8",
    )
    print(f"attempted {len(rows)} alternate sources; recovered {len(readable)} readable")
    print(out)


if __name__ == "__main__":
    main()
