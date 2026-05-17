# Paper 4 Full-Book Render Probe v449

Generated: 2026-05-17T14:30:49.967975+00:00

v449 runs the full official Quarto book render after v448 established that the
registered Paper 4 chapter renders cleanly on its own.

## Result

- Command: `bash scripts/render_quarto.sh render book/ --to html --execute-daemon-restart`.
- Exit code: `0`.
- Full book render passed: `True`.
- Runtime seconds: `454.026`.
- Registered book pages: `122`.
- Observed rendered pages: `122`.
- Paper 4 rendered pages inside full book: `10`.
- Output index exists: `True`.
- Post-render full pytest run: `False`.

## Stdout Tail

```text

```

## Stderr Tail

```text
[ 94/122] chapters/19-paper-mega-extension/19c-integrated-architecture.qmd

[ 95/122] chapters/19-paper-mega-extension/19f-sequential-decision-framework.qmd

[ 96/122] chapters/19-paper-mega-extension/19h-mvp-evidence-pack.qmd

[ 97/122] chapters/19-paper-mega-extension/19i-regret-auditability-frontier.qmd

[ 98/122] chapters/19-paper-mega-extension/19n-online-mdcp-fairness.qmd

[ 99/122] chapters/19-paper-mega-extension/19t-multi-period-solver.qmd

[100/122] chapters/19-paper-mega-extension/19ca-v38-final-synthesis.qmd

[101/122] chapters/17-paper-gpu/index.qmd

[102/122] chapters/17-paper-gpu/17a-positioning.qmd

[103/122] chapters/17-paper-gpu/17b-theoretical-framework.qmd

[104/122] chapters/17-paper-gpu/17c-existing-evidence.qmd

[105/122] chapters/17-paper-gpu/17d-gap-and-experiments.qmd

[106/122] chapters/18-paper-quantum/index.qmd

[107/122] chapters/18-paper-quantum/18a-context-hypothesis.qmd

[108/122] chapters/18-paper-quantum/18b-state-of-art.qmd

[109/122] chapters/18-paper-quantum/18c-proposed-methodology.qmd

[110/122] chapters/18-paper-quantum/18d-gap-execution-strategy.qmd

[111/122] chapters/17-specialization-bridge.qmd

[112/122] chapters/19-specialization-snapshot.qmd

[113/122] chapters/18-research-agenda/index.qmd

[114/122] chapters/18-research-agenda/18a-state-of-the-art.qmd

[115/122] chapters/18-research-agenda/18b-thesis-contributions.qmd

[116/122] chapters/18-research-agenda/18c-future-directions.qmd

[117/122] chapters/A-notebook-atlas.qmd

[118/122] chapters/B-gpu-benchmarks.qmd

[119/122] chapters/C-artifact-catalog.qmd

[120/122] chapters/D-configuration-reference.qmd

[121/122] chapters/E-streamlit-companion.qmd

[122/122] chapters/F-rerun-v2-refactor.qmd

Output created: _output/index.html

```

## Required Caveat

v449 proves the full official Quarto book renders, including the registered
Paper 4 compact surface. It does not claim a post-render full-pytest refresh,
champion replacement, Paper Estrella replacement, or final Paper 4 promotion.

## Next Executable Wave

Build `paper4_v450_post_full_book_render_pytest_probe.md`.
