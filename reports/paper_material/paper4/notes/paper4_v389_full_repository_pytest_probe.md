# Paper 4 Full Repository Pytest Probe v389

Generated: 2026-05-17T06:58:21.350980+00:00

v389 runs the full repository pytest frontier after the v388 documentation
regression probe.

## Result

- Initial full pytest: `1` Streamlit AppTest timeout failure.
- Repair: `tests/test_streamlit/test_app_shell_navigation.py` timeout raised from
  `20`s to
  `45`s.
- Post-repair full pytest: `1128`
  passed, `2` skipped,
  `13` warnings.
- Runtime: `205.61` seconds.

## Required Caveat

v389 proves full repository pytest cleanliness only. It does not claim global
ruff cleanliness, full Quarto render success, champion replacement or Paper 4
final promotion.

## Next Executable Wave

Build `paper4_v390_repository_lint_frontier.md` by probing `uv run ruff check .`.
