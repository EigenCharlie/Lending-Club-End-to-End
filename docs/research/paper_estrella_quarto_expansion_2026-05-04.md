# Paper Estrella Quarto Expansion - 2026-05-04

This note records the book-side expansion added after the P1 hardening work. The
purpose is to keep the Quarto book richer than the eventual paper manuscript:
more explanation, more reviewer-facing context, and a local numeric reference
guide for the Paper Estrella section.

## Book Changes

- Added `book/chapters/14-paper-estrella/14f-editorial-claims-references.qmd`.
- Added the new page to `book/_quarto.yml` under Part IV.
- Reworked the Paper Estrella landing page so it no longer depends on hard-coded
  chapter numbers and now explains how to read the book as an editorial dossier.
- Updated the introduction scope: the conditional Hoeffding/Bernstein tightening
  is now documented as appendix-level material, while Markov remains the main
  distribution-free theorem.
- Added a methodology table that maps every P1 evidence layer to its artifact and
  reviewer question.
- Linked the discussion back to the new editorial guide.

## Why This Matters

The manuscript version should eventually be compressed, but the book should keep
the reasoning that justifies compression decisions. The new page separates:

- claims that belong in the paper body;
- robustness checks that belong in appendix;
- future work that should not be sold as current evidence;
- local numeric references `[1]`, `[2]`, ... for the Paper Estrella narrative.

## Guardrails

The documentation tests should verify that:

- the new Quarto page is registered in `book/_quarto.yml`;
- the page contains a claim ladder, reviewer Q&A, paper-placement table and local
  numbered references;
- the Paper Estrella docs still point to the official economic champion and do
  not reopen the champion search.
