#!/usr/bin/env python3
"""Prioritize Andrija LinkedIn items that are worth a logged-in pass.

This does not use credentials. It only reads the already captured public pack
and writes a governed queue for a future visible-browser / cookie-state pass.
"""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent

PACK_DIR = Path("reports/linkedin_credit_risk_andrija_djurovic")
DATA_DIR = PACK_DIR / "data"
DOCS_DIR = PACK_DIR / "docs"


FIELDS = [
    "priority",
    "target_kind",
    "target_id",
    "url_or_route",
    "reason_to_try_logged_in",
    "expected_incremental_value",
    "project_destination_if_useful",
    "recommended_access_mode",
    "stop_condition",
    "do_not_use_for",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def post_url(activity_id: str, posts: dict[str, dict[str, str]]) -> str:
    row = posts.get(activity_id, {})
    return (
        row.get("post_url")
        or f"https://www.linkedin.com/feed/update/urn:li:activity:{activity_id}/"
    )


def main() -> None:
    posts = {row["activity_id"]: row for row in read_csv(DATA_DIR / "posts_index.csv")}
    profile_candidates = [
        row
        for row in read_csv(DATA_DIR / "profile_public_activity_candidates.csv")
        if row["classification"] == "profile_activity_without_own_post_permalink"
    ]

    rows: list[dict[str, str]] = [
        {
            "priority": "P0",
            "target_kind": "blocked_post_full_capture",
            "target_id": "7463579851317239808",
            "url_or_route": post_url("7463579851317239808", posts),
            "reason_to_try_logged_in": "Public capture returned only generic LinkedIn text; activity is recent and could contain a new credit-risk post/deck.",
            "expected_incremental_value": "Potentially high because current content is zero and it sits in the recent-profile window.",
            "project_destination_if_useful": "CRPTO/Paper 4/thesis triage after content is known.",
            "recommended_access_mode": "Visible logged-in browser/CDP capture first; avoid password-in-chat and avoid internal API scraping.",
            "stop_condition": "Stop when full text, attachments, outbound links and visible comments are captured, or the page is confirmed non-credit-risk/reaction-only.",
            "do_not_use_for": "Do not infer topic from activity ID or snippets.",
        },
        {
            "priority": "P0",
            "target_kind": "blocked_post_full_capture",
            "target_id": "7462756612538073090",
            "url_or_route": post_url("7462756612538073090", posts),
            "reason_to_try_logged_in": "Public capture returned only generic LinkedIn text; activity is recent and could contain a new credit-risk post/deck.",
            "expected_incremental_value": "Potentially high because current content is zero and it sits in the recent-profile window.",
            "project_destination_if_useful": "CRPTO/Paper 4/thesis triage after content is known.",
            "recommended_access_mode": "Visible logged-in browser/CDP capture first; avoid password-in-chat and avoid internal API scraping.",
            "stop_condition": "Stop when full text, attachments, outbound links and visible comments are captured, or the page is confirmed non-credit-risk/reaction-only.",
            "do_not_use_for": "Do not infer topic from activity ID or snippets.",
        },
        {
            "priority": "P0",
            "target_kind": "profile_activity_discovery_recent",
            "target_id": ",".join(row["activity_id"] for row in profile_candidates[:6]),
            "url_or_route": "https://www.linkedin.com/in/andrija-djurovic/recent-activity/all/",
            "reason_to_try_logged_in": "Six most recent public profile activity IDs lack own-post permalinks; login can classify authored post vs reaction/comment.",
            "expected_incremental_value": "High discovery value; this is where missing current posts would appear.",
            "project_destination_if_useful": "New rows in Andrija post backlog only if authored and credit-risk relevant.",
            "recommended_access_mode": "Visible logged-in recent-activity scan; capture only pages the account can normally view.",
            "stop_condition": "Stop after each ID is classified as authored post, reaction/comment, non-credit-risk, or inaccessible.",
            "do_not_use_for": "Do not treat reactions/comments as Andrija-authored technical content.",
        },
    ]

    comment_targets = [
        (
            "P0",
            "7461301929135022080",
            "Multi-period average PD testing has high CRPTO reviewer-defense value and currently has zero public comments.",
            "Possible links or clarifications on estimator, effective sample size, or regulatory backtesting practice.",
            "CRPTO reviewer-defense; Paper 4 validation appendix candidate.",
        ),
        (
            "P0",
            "7435946096050339840",
            "Model-based PD rating-scale calibration is directly relevant to calibration language and possible appendix prototype.",
            "Possible references/code/comments on intercept optimization and grade-level PD tests.",
            "CRPTO calibration defense; Paper 4 candidate.",
        ),
        (
            "P0",
            "7296422990588579840",
            "WoE encoding instability is one of the strongest bridge concepts for project governance and Paper 4.",
            "Possible discussion around replication risk, monitoring, and better WoE alternatives.",
            "Mini-book caveat; Paper 4 encoding-stability lane.",
        ),
        (
            "P0",
            "7458765378664427520",
            "Tree-based interactions is a concrete prototype lane under selective ML.",
            "Possible implementation links or constraints for monotone/business-valid interactions.",
            "Paper 4 prototype; thesis feature-engineering.",
        ),
        (
            "P1",
            "7453691768476405760",
            "Normal-test reliability is useful, but it already has extracted deck content.",
            "Possible clarifications on simulation settings and autocorrelation caveats.",
            "CRPTO validation caveat; thesis.",
        ),
        (
            "P1",
            "7451153797881618432",
            "SMI is valuable for thesis IFRS9, but outside IJDS body.",
            "Possible package/docs links or use-case clarifications.",
            "Thesis IFRS9 chapter.",
        ),
        (
            "P1",
            "7342064984069173248",
            "Model shift/MRM is useful for thesis and limitations; public comments already exist but login may reveal all.",
            "Possible paper/prospectus links or conference discussion.",
            "Thesis MRM; CRPTO limitations.",
        ),
        (
            "P1",
            "7455503720936693760",
            "Working Notes bundle could have comment-only links, but the main external sources are already resolved.",
            "Possible extra references not in the post body.",
            "Source-discovery only.",
        ),
        (
            "P1",
            "7443560044412968960",
            "ADSFCR umbrella post has many public comments and links, but the core source bundle is already read.",
            "Possible missing references or clarifications.",
            "Source-discovery only.",
        ),
        (
            "P2",
            "7441019641449050113",
            "Exact binomial is already implemented locally.",
            "Low; comments may only repeat implementation details.",
            "Archive/source trace unless a stronger reference appears.",
        ),
        (
            "P2",
            "6965364290866282496",
            "monobinpy is tooling context, not a central claim.",
            "Low; README is already read.",
            "Tooling appendix only.",
        ),
    ]
    for priority, activity_id, reason, value, destination in comment_targets:
        rows.append(
            {
                "priority": priority,
                "target_kind": "logged_in_comment_and_link_scan",
                "target_id": activity_id,
                "url_or_route": post_url(activity_id, posts),
                "reason_to_try_logged_in": reason,
                "expected_incremental_value": value,
                "project_destination_if_useful": destination,
                "recommended_access_mode": "Visible logged-in browser/CDP capture of expanded comments and outbound links.",
                "stop_condition": "Stop when visible comments are expanded, links resolved, and no new source changes claim/table/reviewer-defense status.",
                "do_not_use_for": "Do not promote comment-only claims without independent source status.",
            }
        )

    if len(profile_candidates) > 6:
        rows.append(
            {
                "priority": "P1",
                "target_kind": "profile_activity_discovery_remaining",
                "target_id": ",".join(row["activity_id"] for row in profile_candidates[6:]),
                "url_or_route": "https://www.linkedin.com/in/andrija-djurovic/recent-activity/all/",
                "reason_to_try_logged_in": "Remaining activity IDs may include older authored posts hidden from public extraction, but likely include reactions/comments.",
                "expected_incremental_value": "Medium; scan only after P0 because current public corpus already has main technical decks and source bundles.",
                "project_destination_if_useful": "New backlog rows only if authored, credit-risk relevant and not already covered by external source bundles.",
                "recommended_access_mode": "Visible logged-in recent-activity scan, sorted newest first.",
                "stop_condition": "Stop when each ID is classified; only capture authored credit-risk posts.",
                "do_not_use_for": "Do not expand non-authored reactions unless they link to a canonical technical source.",
            }
        )

    rows.append(
        {
            "priority": "NO",
            "target_kind": "external_non_linkedin_blocker",
            "target_id": "7342064984069173248_external_05",
            "url_or_route": "https://crc.business-school.ed.ac.uk/conference-2025",
            "reason_to_try_logged_in": "Not a LinkedIn blocker; LinkedIn cookies will not fix CRC 403.",
            "expected_incremental_value": "Low because the model-shift prospectus/PDF was already captured.",
            "project_destination_if_useful": "Archive.",
            "recommended_access_mode": "Do not spend LinkedIn session; use normal browser manually only if needed.",
            "stop_condition": "Already closed unless the conference page exposes proceedings unavailable elsewhere.",
            "do_not_use_for": "Do not route to LinkedIn cookie/API pass.",
        }
    )

    write_csv(DATA_DIR / "andrija_login_only_priority_queue.csv", rows)

    p0_count = sum(row["priority"] == "P0" for row in rows)
    p1_count = sum(row["priority"] == "P1" for row in rows)
    p2_count = sum(row["priority"] == "P2" for row in rows)

    doc = dedent(
        f"""
        # Andrija Login-Only/API Triage - 2026-05-25

        ## Recommendation

        Do **not** start with LinkedIn's official API for this pass. For third-party
        member activity, the official Posts API is permission-limited and is unlikely
        to read arbitrary posts/comments. The safest useful route is a visible,
        user-owned logged-in browser session, controlled through CDP/Playwright
        storage state if available. Do not paste passwords into chat; if cookies are
        used, keep them in a local file outside git and treat them as temporary
        secrets.

        ## Priority Counts

        - P0: {p0_count}
        - P1: {p1_count}
        - P2: {p2_count}
        - No LinkedIn-cookie value: 1

        ## P0 Targets

        1. Full capture of `7463579851317239808` and `7462756612538073090`.
           These are the only indexed posts with zero useful public content.
        2. Logged-in classification of the six newest profile activity IDs without
           public own-post permalinks. These are the best place to discover missing
           current authored posts.
        3. Expanded comments/link scan for multi-period PD testing, model-based PD
           rating-scale calibration, WoE encoding instability, and tree-based
           interactions.

        ## P1 Targets

        Scan comments for normal-test reliability, SMI/IFRS9, model shift/MRM and
        the two ADSFCR/Working Notes umbrella posts. These can add references or
        reviewer-defense language, but most primary assets are already read.

        ## P2 / Skip

        Exact binomial, monobinpy, PDtoolkit, R toolkit, book-release and repo
        announcement posts should not consume much session time. They are already
        implemented, tooling context, or source-discovery only. The CRC 403 page is
        not a LinkedIn blocker and should not be routed through LinkedIn cookies.

        ## Guardrails

        - Use only user-owned visible access; no fake accounts, captcha bypass,
          stealth, or rate evasion.
        - Capture only what the account can normally view.
        - Stop each target when it is classified, captured, resolved, or shown to
          be non-credit-risk.
        - LinkedIn comments remain intake/source-discovery only unless backed by
          an independent source or local result.
        """
    ).strip()
    (DOCS_DIR / "andrija_login_only_api_triage_2026-05-25.md").write_text(
        doc + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(rows)} login-only triage rows: P0={p0_count}, P1={p1_count}, P2={p2_count}.")


if __name__ == "__main__":
    main()
