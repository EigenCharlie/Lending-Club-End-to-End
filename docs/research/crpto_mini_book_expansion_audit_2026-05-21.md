# CRPTO Mini-Book Expansion Audit - 2026-05-21

## Decision

The Paper CRPTO mini-book should remain the shared spine for both the IJDS paper
and the master's thesis during the next 6--12 months. It should not duplicate
the full Quarto book. It should instead add extraction controls: evidence spine,
page budget, reviewer-defense bank, thesis expansion ledger, release checklist
and negative-results registry.

## Project Read

- The main Quarto book has 123 configured chapters and all referenced chapter
  files exist.
- Chapter 14 is the canonical long-form CRPTO dossier.
- Paper 1 artifacts A0--A18 are the strongest manuscript-ready evidence pack.
- Paper 4 is useful as bounded governance/future-work evidence, not as a
  replacement for the official CRPTO champion.
- The LinkedIn intake is useful for language and caveats, but no LinkedIn-only
  content should become public scholarly evidence.

## Additions Made

- Added `papers/paper_crpto_book/chapters/07-project-expansion-map.qmd`.
- Added `papers/paper_crpto_book/chapters/08-roadmap-and-gates.qmd`.
- Registered both chapters in `papers/paper_crpto_book/_quarto.yml`.
- Updated the CRPTO extraction tests so the roadmap and gates are guarded.

## Editorial Additions Recommended

1. Evidence spine C1--C5 -> artifact -> test.
2. IJDS page-budget ledger.
3. Reviewer-defense bank.
4. Thesis expansion ledger.
5. Double-anonymous release checklist.
6. Negative-results registry from Paper 4 blockers.
7. Figure/table decision log for the 3-figure, 2--3-table IJDS body.

## Stop Rule

Do not reopen champion selection or Paper 4 promotion. A parked item may enter
the mini-book only if it changes a claim, appendix table, figure, reviewer
response or thesis defense section and remains consistent with
`models/final_project_promotion.json`.
