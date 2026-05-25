# IV Hypothesis Testing Source Memo - 2026-05-25

## Source Status

- Source: Rojas, Alvarez and Rojas, "Statistical Hypothesis Testing for Information Value (IV)."
- Canonical URL: <https://arxiv.org/abs/2309.13183>
- Status: arXiv preprint, v3 dated 2026-01-27; not treated as peer-reviewed evidence.
- Local text: `reports/linkedin_credit_risk_andrija_djurovic/logged_in_review/external_sources/iv_hypothesis_testing/rojas_alvarez_rojas_2026_iv_hypothesis_testing.txt`
- Implementation trace: PyPI package `statistical-iv` exists at <https://pypi.org/project/statistical-iv/>; current observed version during intake was 0.3.2.

## What The Paper Adds

The paper formalizes Information Value as a Jeffreys-divergence estimator and
derives a nonparametric J-Divergence hypothesis test for whether the class-
conditional feature-bin distributions differ. Its practical contribution for
credit-risk feature screening is not "use this package tomorrow"; it is the
governance point that fixed IV thresholds such as 0.02 or 0.1 are operational
heuristics, not statistical guarantees.

The most CRPTO-relevant claim is that imbalance can distort fixed IV threshold
rules. In the paper's simulations, the proposed test is framed as more robust
under imbalanced targets because the decision is tied to an estimated sampling
distribution and a p-value, rather than a universal IV cutoff.

## Project Intake

- **Mini libro CRPTO**: add IV threshold uncertainty to metric-governance
  language. This strengthens the distinction between auditable feature
  engineering and metric fetishism.
- **Paper IJDS**: do not add as body evidence unless it supports a short
  reviewer-defense caveat. The paper is a preprint and IV/WOE is not the core
  CRPTO contribution.
- **Tesis**: useful in the validation/governance chapter as an example of
  replacing inherited credit-risk thresholds with estimable uncertainty.
- **Paper 4**: optional prototype only: compare current WOE/IV filters with
  J-Divergence selection on the existing Lending Club feature universe, and
  close the lane unless it changes an appendix table, reviewer response or
  thesis section.

## Stop Rule

Closed for the current IJDS manuscript. Reopen only if a local benchmark shows
that J-Divergence IV testing materially changes feature-governance conclusions
without reopening the official champion model.
