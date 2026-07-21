# Overnight Goal Backlog Plan: Denis Burakov LinkedIn Corpus

Generated: 2026-05-21

## Goal

Traverse every indexed post and every associated external source without waiting
for further user input. Each post is its own unit of work because the topics are
not guaranteed to be mutually dependent.

## Master Files

- `data/post_execution_backlog.csv`: one row per indexed post plus child
  LinkedIn posts discovered through external links.
- `data/external_link_backlog.csv`: one row per external-link reference from
  the manifest, preserving duplicates across parent posts.
- `data/external_linkedin_child_post_backlog.csv`: LinkedIn posts/articles that
  should be captured as child units only if they can change the parent decision.

## Backlog Counts

- Indexed Denis posts: 59.
- External LinkedIn child posts discovered: 8.
- Total post backlog rows: 67.
- External-link references: 109.
- Unique canonical external URLs after resolution: 78.

## Source Resolution Rules

- GitHub, official docs, preprints, PDFs, and DOI-style sources can support
  implementation or scholarly context after reading.
- LinkedIn posts/articles are child context unless they point to canonical
  evidence.
- LinkedIn profiles/companies are provenance/context only.
- Blogs, Medium, Linktree, and general web pages are implementation/context
  unless they lead to canonical sources.
- Unresolved shortlinks and HTTP-blocked sources remain blockers but do not stop
  the parent post analysis if local post/PDF/image evidence is sufficient for
  a park/archive/append decision.

## Per-Post Stop Rule

For each post, stop when all applicable items are true:

1. Public post text is read or marked missing.
2. PDF/transcript/image asset is read, OCRed, or explicitly blocked.
3. External links are resolved and classified, with relevant sources read or
   marked as blocked/context-only.
4. The post has a promote/append/park/archive decision.
5. The executable/implementable outcome is either applied, queued as a bounded
   experiment, or rejected with a reason.
6. No LinkedIn-only claim is promoted into public prose without source-status
   labeling.

## Global Stop Rule

The overnight goal is complete when all 67 backlog rows have a closed status, all
109 link references have a resolution/handling decision, and any useful changes
to the Quarto book, Paper 4, Paper Estrella, or project research memos have been
implemented or queued with a concrete rejection/acceptance condition.

## Initial Execution Order

1. Close ready PDF/text posts with high project value:
   posts 16, 18, 26, 30, 34, 39, 42, 44, 46, 48, 54, and 59.
2. Close high-relevance image posts via manual visual reading where OCR is not
   available:
   posts 2, 7, 13, 21, 24, 28, and 49.
3. Read or archive source-discovery and medium/low value posts.
4. Inspect high-value GitHub/official/preprint links.
5. Apply manuscript/book changes only where the evidence gate is satisfied.
