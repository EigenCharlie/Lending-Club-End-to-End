# Paper 4 Methods/Results Draft v453

Generated: 2026-05-17T14:51:55.385385+00:00

## Methods Draft

Paper 4 is maintained as a living-lab protocol in which each executable wave
produces a versioned status file, tabular evidence, a notebook entry, and a
guardrail test. The current validation surface is defined by four operational
gates: full repository pytest, repository Ruff, Quarto rendering of the compact
registered Paper 4 chapter, and rendering of the full official Quarto book. A
negative promotion gate is also enforced: the final Paper 4 promotion artifact
must remain absent.

The manuscript extraction is therefore evidence-first. Claims are admitted only
when they can be traced to generated artifacts, and every claim boundary is kept
in `paper4_current_claim_boundaries.csv`. Historical Paper 4 pages remain on
disk as an archive, while the official book renders only the compact registered
Paper 4 surface.

## Results Draft

After the full-book render probe, the repository passed 1188 tests with 2
skipped tests, 13 warnings, and zero Ruff diagnostics. The official Paper 4
chapter rendered as 10 registered pages, and the full Quarto book rendered 122
registered pages. The archive policy remained clean: historical Paper 4 files
were preserved on disk but excluded from the official rendered chapter.

Together, these gates support a bounded readiness statement: the living-lab
evidence package is internally reproducible and ready for manuscript extraction.
They do not support final-paper, submission, deployment, legal fairness,
external-validation, or champion-replacement claims.

## Required Caveat

v453 is a partial Methods/Results draft. Discussion, limitations, abstract,
conclusion, external validation, and final promotion remain blocked.

## Next Executable Wave

Build `paper4_v454_discussion_limitations_draft.md`.
