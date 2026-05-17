# Paper 4 Quarto Registration Gap Decision v386

Generated: 2026-05-17T06:26:41.809328+00:00

v386 turns the v385 validation gap into an explicit governance decision.

## Selected Policy

`archive_in_place_with_manifested_guardrail_exemption`

The 70 historical standalone pages remain on disk for provenance, but they should
not be rendered as official Paper 4 book chapters. The official rendered surface
stays limited to the curated Paper 4 pages.

## Options Reviewed

- `register_all_historical_pages_in_book`: `rejected`
- `move_pages_outside_book_chapters`: `deferred`
- `delete_historical_pages`: `rejected`
- `archive_in_place_with_manifested_guardrail_exemption`: `selected` (selected)
- `ignore_without_manifest`: `rejected`

## Required Caveat

v386 is a decision packet only. It does not mutate `book/_quarto.yml`, does not
patch the Quarto book guardrail, does not make the full regression suite clean,
and does not create Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v387_quarto_archive_guardrail_patch.csv` to write the stable archive manifest and
patch the book guardrail narrowly around explicitly archived historical pages.
