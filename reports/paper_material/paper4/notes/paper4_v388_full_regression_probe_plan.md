# Paper 4 Full Regression Probe Plan v388

Generated: 2026-05-17T06:40:07.183199+00:00

v388 captures the first broader regression result after the v387 Quarto archive
guardrail repair.

## Observed Clean Surface

- Documentation tests: `440` /
  `440` passed.
- Documentation runtime: `76.0` seconds.
- Paper 4 focal guardrails selected: `10`.
- Quarto book guardrails passed: `3`.

## Required Caveat

v388 does not claim full repository pytest, global ruff, full Quarto render,
champion replacement or Paper 4 final promotion.

## Next Executable Wave

Build `paper4_v389_full_repository_pytest_probe.md` by running `uv run pytest -q --maxfail=10`
and classifying the full repository pass/failure frontier.
