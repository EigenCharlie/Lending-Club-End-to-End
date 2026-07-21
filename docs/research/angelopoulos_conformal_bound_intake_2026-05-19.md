# Angelopoulos Conformal Bound Intake - 2026-05-19

This memo records how the Angelopoulos/Bates/Barber line updates Paper Estrella
and Paper 4 without reopening the champion search.

## Paper Estrella

Paper Estrella should use Angelopoulos and coauthors as the modern conformal
foundation:

- `angelopoulos2023`: readable entry point and pedagogical reference for split
  conformal, distribution-free uncertainty and examples.
- `angelopoulos2026_foundations`: proof-audit source for exchangeability,
  conformal quantiles, split conformal and the logic of distribution-free
  inference.
- `bates2021rcps`, `angelopoulos2024risk`, `angelopoulos2025ltt`: the direct
  lineage for the funded-set bound, because CRPTO is best framed as a
  portfolio-aware instantiation of bounded monotone risk control.

The central claim remains unchanged: CRPTO is a post-hoc auditable bridge from
calibrated PD and conformal upper bounds to a robust credit portfolio constraint.
The Markov bound remains the main distribution-free statement; exact `276k`
evidence remains empirical validation of the frozen policy, not a stronger
post-selection theorem.

## Paper 4

The same source pack opens four bounded Paper 4 lanes:

| Source | Paper 4 lane | Decision |
| --- | --- | --- |
| CRC/LTT | lane gates for monotone losses | append |
| Label-noise robustness | default/charge-off label governance | append/future |
| Non-monotonic CRC | multi-objective decision-risk gate | park until implemented |
| Gradient equilibrium / time-series repo | online drift recalibration | park unless prospective split exists |

These are research-program improvements, not new proof that the current Paper
Estrella champion should change.

## Implementation Rule

Promote to Quarto only when a source changes claim boundaries, proof language or
future-work prioritization. Do not create new v-numbered scripts or new runs
unless the experiment can change a specific claim.
