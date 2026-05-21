# Second LinkedIn Ingest Plan - 2026-05-21

## Scope

This second ingest preserves the closed first corpus and creates a separate
sub-pack at `reports/linkedin_credit_risk_denis_burakov/second_ingest/`.

- Missing public LinkedIn posts queued: 13
- Public LinkedIn articles queued as associated sources: 12
- Discovery basis: public search-result snippets, public LinkedIn permalink
  pages, and prior first-ingest blocker audit.

## Rules

- Public pages only; no fake accounts, captcha bypass, stealth, rate evasion, or
  browser-session extraction.
- LinkedIn-only material can inform project framing and implementation ideas but
  cannot promote paper/book claims by itself.
- Academic, official, preprint, GitHub, blog, Medium, and LinkedIn-only sources
  remain separated in the logs.
- Stop each row when post text, attachments, external links, and article/source
  status are either read or assigned a concrete blocker.

## Stop Condition

The second ingest closes when every queued post/article has a capture/read
status, every link has a resolved or blocked source row, and every concept is
mapped to one of: promote, append, prototype, park, archive, or blocked.
