# Paper 4 Quarto Archive Guardrail Patch v387

Generated: 2026-05-17T06:31:25.362589+00:00

v387 implements the v386 decision by creating a stable archived-pages manifest
and narrowing the Quarto book guardrail around that manifest.

## Patch

- Stable archive manifest: `book/_archived_chapter_pages.yml`.
- Archived historical pages: `70`.
- Historical pages registered in `book/_quarto.yml`: `0`.
- Book config mutated: `False`.

## Required Caveat

This is a registration-guardrail repair only. v387 does not claim a full Quarto
render, full regression-suite cleanliness, champion replacement or Paper 4 final
promotion.

## Next Executable Wave

Build `paper4_v388_full_regression_probe_plan.md` to probe broader regression readiness.
