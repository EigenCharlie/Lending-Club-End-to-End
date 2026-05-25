#!/usr/bin/env python3
"""Build Andrija logged-in project intake decisions.

The raw logged-in capture is intentionally treated as private research intake:
comments can explain why a source matters, but public claims need an external
source or a local project artifact. This script converts the P0/P1 logged-in
pass into stop-rule-backed rows and a short source reading queue.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path("reports/linkedin_credit_risk_andrija_djurovic")
PACK = ROOT / "logged_in_review"
DATA = PACK / "data"
DOCS = PACK / "docs"

SUMMARY = DATA / "logged_in_post_comment_summary.csv"
CAPTURE = DATA / "logged_in_capture_log.csv"
COMMENTS = DATA / "logged_in_visible_comments.csv"
EXTERNAL = DATA / "logged_in_external_link_inventory.csv"

DECISIONS = DATA / "logged_in_project_intake_decisions.csv"
SOURCE_QUEUE = DATA / "logged_in_source_reading_queue.csv"
FINDINGS = DOCS / "andrija_logged_in_review_findings_2026-05-25.md"
DECISION_MEMO = DOCS / "andrija_logged_in_project_intake_decisions_2026-05-25.md"
LEGACY_FINDINGS = DOCS / "logged_in_review_findings_2026-05-21.md"

IV_PAPER_TXT = (
    PACK
    / "external_sources"
    / "iv_hypothesis_testing"
    / "rojas_alvarez_rojas_2026_iv_hypothesis_testing.txt"
)

DECISION_FIELDS = [
    "queue_id",
    "activity_id",
    "title",
    "capture_status",
    "comment_count",
    "high_priority_external_count",
    "decision",
    "project_destination",
    "possible_executable_or_implementable",
    "stop_condition",
    "evidence_status",
]

SOURCE_FIELDS = [
    "source_id",
    "queue_id",
    "activity_id",
    "source_title",
    "canonical_url",
    "source_status",
    "evidence_status",
    "project_use",
    "decision",
    "stop_condition",
    "local_path",
]

OVERRIDES = {
    "7464369310325702657": {
        "decision": "promote_to_crpto_metric_governance",
        "project_destination": "Mini libro CRPTO; thesis validation chapter; Paper 4 only as a gated appendix prototype.",
        "possible_executable_or_implementable": (
            "Add IV threshold uncertainty and inherited-heuristic governance; optionally benchmark "
            "J-Divergence IV testing against current WOE/IV filters without changing the champion."
        ),
        "stop_condition": (
            "Closed when the arXiv IV paper is logged as preprint evidence, CRPTO caveat text is "
            "updated, and no new public claim relies on LinkedIn comments alone."
        ),
        "evidence_status": "external_preprint_read_plus_private_comment_context",
    },
    "7296422990588579840": {
        "decision": "promote_to_crpto_woe_stability_caveat",
        "project_destination": "Mini libro CRPTO; Paper 4 candidate lane; thesis feature-governance chapter.",
        "possible_executable_or_implementable": (
            "Keep WoE replacement instability as a preprocessing/monitoring caveat; prototype only if "
            "a KL-limit or iterative-WoE experiment changes an appendix table or reviewer response."
        ),
        "stop_condition": (
            "Closed for IJDS body; reopen only under a separate Paper 4 protocol with a local "
            "experiment and no champion replacement."
        ),
        "evidence_status": "external_adsfcr_source_plus_private_comment_context",
    },
    "7458765378664427520": {
        "decision": "park_residual_tree_validation_prototype",
        "project_destination": "Paper 4 prototype; thesis validation and feature-engineering chapter.",
        "possible_executable_or_implementable": (
            "Use residual tree segment validation as an omitted-risk-factor diagnostic, with "
            "monotonicity and minimum-leaf constraints; keep ML as enhancement layer."
        ),
        "stop_condition": (
            "Closed when PDtoolkit segment.vld is logged as software documentation; prototype only if "
            "it changes validation language or an appendix diagnostic."
        ),
        "evidence_status": "software_documentation_read_plus_private_comment_context",
    },
    "7453691768476405760": {
        "decision": "promote_pd_backtesting_dependence_caveat",
        "project_destination": "Mini libro CRPTO reviewer-defense; thesis PD validation chapter.",
        "possible_executable_or_implementable": (
            "Add effective-sample-size/autocorrelation caveat for multi-period PD tests; keep "
            "Blumke/order-statistic tests as source-discovery for thesis."
        ),
        "stop_condition": (
            "Closed for IJDS unless a cited PD-validation paper materially changes a reviewer-defense "
            "answer; no new test is implemented without an artifact gate."
        ),
        "evidence_status": "private_comment_context_plus_external_source_discovery",
    },
    "7342064984069173248": {
        "decision": "append_model_shift_to_thesis_mrm",
        "project_destination": "Thesis MRM chapter; IJDS limitations/future work only.",
        "possible_executable_or_implementable": (
            "Use model shift as a governance analogy for specification-risk monitoring, not as a new "
            "IJDS contribution."
        ),
        "stop_condition": (
            "Closed because model-shift sources are already captured; reopen only for thesis chapter "
            "writing or MRM appendix alignment."
        ),
        "evidence_status": "external_presentation_and_prospectus_read",
    },
    "7398299296388771840": {
        "decision": "archive_non_credit_risk_context",
        "project_destination": "Archive.",
        "possible_executable_or_implementable": "No CRPTO action; financial-stability/hybrid-threat context is out of scope.",
        "stop_condition": "Closed as non-credit-risk for current CRPTO/Paper 4 scope.",
        "evidence_status": "blocked_or_low_relevance_external_context",
    },
    "7373943955094319104": {
        "decision": "archive_financial_crime_ai_context",
        "project_destination": "Archive; possible future regulated-AI governance reading only.",
        "possible_executable_or_implementable": (
            "No credit-risk implementation; agentic AI for FCC is governance-adjacent but outside "
            "CRPTO scope."
        ),
        "stop_condition": "Closed unless the thesis opens a regulated-AI governance appendix.",
        "evidence_status": "external_preprint_title_abstract_read",
    },
    "7464473809778212864": {
        "decision": "archive_dense_irrelevant_comment_thread",
        "project_destination": "Archive.",
        "possible_executable_or_implementable": "No project action; dense thread is not credit-risk material.",
        "stop_condition": "Closed as humorous/non-substantive for credit-risk modeling.",
        "evidence_status": "private_comment_context_archived",
    },
}

BLOCKED_DECISIONS = {
    "not_authenticated_or_checkpoint": (
        "archive_login_blocked_surface",
        "Archive.",
        "No action; LinkedIn returned signup/checkpoint surface in the owned logged-in profile copy.",
        "Closed with no bypass; do not attempt captcha/checkpoint evasion.",
        "logged_in_blocked_no_bypass",
    ),
    "capture_error": (
        "archive_capture_error_surface",
        "Archive.",
        "No action; HTTP error remained after logged-in Opera pass.",
        "Closed unless a normal visible permalink becomes available later.",
        "capture_error_logged",
    ),
}

SOURCE_ROWS = [
    {
        "source_id": "ANDRIJA-LI-SRC-001",
        "queue_id": "ANDRIJA-LOGIN-008",
        "activity_id": "7464369310325702657",
        "source_title": "Statistical Hypothesis Testing for Information Value (IV)",
        "canonical_url": "https://arxiv.org/abs/2309.13183",
        "source_status": "arxiv_preprint_v3_read",
        "evidence_status": "preprint_not_peer_reviewed",
        "project_use": "IV threshold governance; Paper 4 feature-selection appendix candidate.",
        "decision": "promote_as_caveat_and_source_reading",
        "stop_condition": "Use for language and optional prototype only; no IJDS body claim without local benchmark.",
        "local_path": str(IV_PAPER_TXT),
    },
    {
        "source_id": "ANDRIJA-LI-SRC-002",
        "queue_id": "ANDRIJA-LOGIN-012",
        "activity_id": "7458765378664427520",
        "source_title": "PDtoolkit package manual: segment.vld",
        "canonical_url": "https://cran.r-project.org/web/packages/PDtoolkit/PDtoolkit.pdf",
        "source_status": "cran_software_documentation_read",
        "evidence_status": "software_documentation",
        "project_use": "Residual-tree segment validation and omitted-risk-factor diagnostics.",
        "decision": "append_to_paper4_prototype_queue",
        "stop_condition": "Prototype only if it changes an appendix diagnostic or reviewer response.",
        "local_path": "",
    },
    {
        "source_id": "ANDRIJA-LI-SRC-003",
        "queue_id": "ANDRIJA-LOGIN-011",
        "activity_id": "7296422990588579840",
        "source_title": "ADSFCR repository",
        "canonical_url": "https://github.com/andrija-djurovic/adsfcr",
        "source_status": "github_source_anchor_read",
        "evidence_status": "code_or_tool_source",
        "project_use": "WoE instability and scorecard/MRM source trace.",
        "decision": "append_as_context_not_public_claim",
        "stop_condition": "No champion reopen; use only for caveat/prototype scoping.",
        "local_path": "",
    },
    {
        "source_id": "ANDRIJA-LI-SRC-004",
        "queue_id": "ANDRIJA-LOGIN-015",
        "activity_id": "7342064984069173248",
        "source_title": "Model shift prospectus",
        "canonical_url": "https://www.crc.business-school.ed.ac.uk/sites/crc/files/2024-06/model-shift-prospectus_0.pdf",
        "source_status": "pdf_read",
        "evidence_status": "working_paper_or_prospectus",
        "project_use": "Thesis MRM and model-specification-risk discussion.",
        "decision": "append_to_thesis_mrm",
        "stop_condition": "Keep outside IJDS body except as limitation/future work.",
        "local_path": "",
    },
    {
        "source_id": "ANDRIJA-LI-SRC-005",
        "queue_id": "ANDRIJA-LOGIN-013",
        "activity_id": "7453691768476405760",
        "source_title": "Probability of default validation: Basel score and order statistic methodology",
        "canonical_url": "https://www.risk.net/journal-of-risk-model-validation/2186764/probability-of-default-validation-a-single-year-and-a-multiyear-methodology-for-the-basel-framework",
        "source_status": "source_discovered_title_abstract",
        "evidence_status": "peer_reviewed_source_discovery_not_full_text",
        "project_use": "PD backtesting caveat around serial/cross-sectional dependence and order statistic tests.",
        "decision": "park_for_thesis_source_retrieval",
        "stop_condition": "Retrieve full text only if thesis PD-validation chapter needs it.",
        "local_path": "",
    },
    {
        "source_id": "ANDRIJA-LI-SRC-006",
        "queue_id": "ANDRIJA-LOGIN-030",
        "activity_id": "7373943955094319104",
        "source_title": "Agentic AI for Financial Crime Compliance",
        "canonical_url": "https://arxiv.org/abs/2509.13137",
        "source_status": "arxiv_title_abstract_read",
        "evidence_status": "preprint_or_forthcoming_conference_out_of_scope",
        "project_use": "Archive; regulated-AI governance adjacent only.",
        "decision": "archive_for_crpto",
        "stop_condition": "Closed for credit-risk CRPTO scope.",
        "local_path": "",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_decision_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(SUMMARY):
        activity_id = row["activity_id"]
        if activity_id in OVERRIDES:
            decision = OVERRIDES[activity_id]
        elif row["capture_status"] in BLOCKED_DECISIONS:
            vals = BLOCKED_DECISIONS[row["capture_status"]]
            decision = {
                "decision": vals[0],
                "project_destination": vals[1],
                "possible_executable_or_implementable": vals[2],
                "stop_condition": vals[3],
                "evidence_status": vals[4],
            }
        elif int(row["comment_count"] or 0) > 0:
            decision = {
                "decision": "append_private_context_or_close",
                "project_destination": "Private intake archive; prior public Andrija decisions remain controlling.",
                "possible_executable_or_implementable": (
                    "No immediate project change; keep comment context for source discovery and reviewer-language nuance."
                ),
                "stop_condition": "Closed unless a comment-shared external source changes a claim, appendix, or thesis chapter.",
                "evidence_status": "private_comment_context_only",
            }
        else:
            decision = {
                "decision": "close_no_logged_in_delta",
                "project_destination": "No new action beyond public Andrija intake.",
                "possible_executable_or_implementable": "No immediate project change.",
                "stop_condition": "Closed because logged-in pass exposed no comments or high-priority external sources.",
                "evidence_status": "logged_in_surface_checked_no_delta",
            }
        rows.append(
            {
                "queue_id": row["queue_id"],
                "activity_id": activity_id,
                "title": row["title"],
                "capture_status": row["capture_status"],
                "comment_count": row["comment_count"],
                "high_priority_external_count": row["high_priority_external_count"],
                **decision,
            }
        )
    return rows


def md_table(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        cells = []
        for field in fields:
            cell = row.get(field, "").replace("\n", " ").replace("|", "/")
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_memos(decisions: list[dict[str, str]], sources: list[dict[str, str]]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    summary = read_csv(SUMMARY)
    capture = read_csv(CAPTURE)
    comments = read_csv(COMMENTS)
    external = read_csv(EXTERNAL)
    capture_counts = Counter(row["capture_status"] for row in capture)
    action_counts = Counter(row["decision"] for row in decisions)
    priority_counts = Counter(row["priority"] for row in external)

    findings = [
        "# Andrija Logged-In Review Findings - 2026-05-25",
        "",
        "This memo summarizes the Opera GX / Windows Playwright pass over the Andrija Djurovic P0/P1 queue. It used the user's own visible logged-in browser state through a non-destructive profile copy. Comments remain private research intake and are not public evidence.",
        "",
        "## Coverage",
        "",
        f"- Queue rows captured: {len(summary)}",
        f"- Capture status: {dict(capture_counts)}",
        f"- Visible comments captured: {len(comments)} across {len({row['activity_id'] for row in comments})} posts",
        f"- External link rows after dedupe by post/source/url: {len(external)}",
        f"- Link priority counts: {dict(priority_counts)}",
        f"- Project decision rows: {len(decisions)}",
        "",
        "## Why Opera Worked",
        "",
        "The previous Chrome path was blocked because the Windows browser did not expose a reachable DevTools endpoint from the WSL process. The working path was to launch Opera GX on Windows with remote debugging against a copied Opera profile that already held the authenticated LinkedIn state, then run Playwright from Windows so `127.0.0.1` referred to the Windows browser.",
        "",
        "## High-Value Logged-In Deltas",
        "",
        "- `ANDRIJA-LOGIN-008`: comments exposed an arXiv preprint on statistical hypothesis testing for Information Value. This directly strengthens CRPTO metric-governance language around IV thresholds, inherited heuristics, class imbalance and p-value-based feature screening.",
        "- `ANDRIJA-LOGIN-011`: the WoE instability thread clarified that the issue is not only binning or drift; iterative replacement using model-predicted outcomes can force a new model even under perfect replication, with a possible KL-convergent limit as a Paper 4 prototype only.",
        "- `ANDRIJA-LOGIN-012`: residual-tree validation is useful as an omitted-risk-factor and over/underestimation diagnostic, but monotonicity and splitting-node governance keep it in the prototype/thesis lane.",
        "- `ANDRIJA-LOGIN-013`: multi-period PD testing needs dependence-aware language; autocorrelation changes effective sample size and can make default-rate tests reject for assumption failure rather than true miscalibration.",
        "- `ANDRIJA-LOGIN-015`: model shift stays thesis/MRM material, useful for specification-risk governance but not as a new IJDS claim.",
        "",
        "## Blocked Or Archived",
        "",
        "- Three queue rows remained blocked/error after the logged-in Opera pass; they are closed without captcha/checkpoint bypass.",
        "- The dense non-credit-risk thread, hybrid-threat link, and financial-crime-compliance AI preprint are archived for CRPTO because they do not change the credit-risk decision pipeline.",
        "",
        "## Source Reading Queue",
        "",
    ]
    findings.extend(
        md_table(sources, ["source_id", "queue_id", "source_title", "evidence_status", "decision"])
    )
    findings.extend(
        [
            "",
            "## Stop Rule",
            "",
            "The logged-in Andrija pass is closed for P0/P1 when every row has a capture status, comment/link count, decision, implementable path, evidence status and stop condition. Reopen only if a newly accessible independent source or local experiment can change a claim, appendix table, reviewer response or thesis chapter.",
        ]
    )
    FINDINGS.write_text("\n".join(findings) + "\n", encoding="utf-8")
    LEGACY_FINDINGS.write_text("\n".join(findings) + "\n", encoding="utf-8")

    high_rows = [row for row in decisions if row["activity_id"] in OVERRIDES]
    decision_lines = [
        "# Andrija Logged-In Project Intake Decisions - 2026-05-25",
        "",
        "LinkedIn comments are private intake and source-discovery context. The only promotable public evidence in this pass is external material with explicit source status, or local project artifacts already governed by tests.",
        "",
        "## Counts",
        "",
        f"- Decision rows: {len(decisions)}",
        f"- Comment rows: {len(comments)}",
        f"- External link rows: {len(external)}",
        f"- Decision counts: {dict(action_counts)}",
        "",
        "## Promote / Append / Park / Archive",
        "",
    ]
    decision_lines.extend(
        md_table(
            decisions,
            [
                "queue_id",
                "activity_id",
                "decision",
                "project_destination",
                "evidence_status",
                "stop_condition",
            ],
        )
    )
    decision_lines.extend(
        [
            "",
            "## Highest-Value Decisions",
            "",
        ]
    )
    decision_lines.extend(
        md_table(
            high_rows,
            [
                "queue_id",
                "title",
                "decision",
                "possible_executable_or_implementable",
                "stop_condition",
            ],
        )
    )
    DECISION_MEMO.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> None:
    decisions = build_decision_rows()
    write_csv(DECISIONS, decisions, DECISION_FIELDS)
    write_csv(SOURCE_QUEUE, SOURCE_ROWS, SOURCE_FIELDS)
    write_memos(decisions, SOURCE_ROWS)
    print(f"Andrija logged-in decisions written: {len(decisions)} rows")
    print(f"Andrija logged-in source queue written: {len(SOURCE_ROWS)} rows")


if __name__ == "__main__":
    main()
