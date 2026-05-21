# Quarto Book and Paper CRPTO Full Audit - 2026-05-21

## Executive Decision

The main Quarto book should remain the intellectual repository for the whole
credit-risk research program. It is too broad to become the public IJDS paper or
the central thesis chapter directly. The right architecture is now three-layered:

1. **Main Quarto book**: full intellectual archive, implementation narrative,
   side papers, exploratory lanes, and research governance.
2. **Paper CRPTO mini-book**: focused extraction layer for IJDS, online
   supplement, and thesis chapter design.
3. **IJDS manuscript/supplement**: anonymized, compact, 25-page paper plus
   reproducibility package.

This audit created the second layer at `papers/paper_crpto_book/`.

## Scope Read

- Book `_quarto.yml`: 123 declared chapters, all present.
- Chapter 14 dossier: `14a`--`14o` remains the canonical long-form CRPTO/Paper
  Estrella dossier.
- Standalone scaffold: `papers/paper1_estrella/paper_estrella_manuscript.qmd`
  remains an extraction scaffold, now labeled Paper CRPTO.
- LinkedIn research pack: final logged-in pass is closed; relevant concepts were
  already promoted to book chapters, not treated as public bibliography by
  themselves.

## Consistency Fixes Applied

- Updated book landing language that still described Part IV as only "three
  drafts"; it now reflects CRPTO plus paper satellites and research extensions.
- Marked `Paper Estrella` as an internal historical alias and `Paper CRPTO` as
  the public name for IJDS/thesis extraction.
- Updated the chapter 14 blueprint from a generic MS/OR/EJOR target to the
  explicit IJDS target and its submission constraints.
- Removed high-risk duplicate Quarto labels for ECE, Mondrian quantile,
  temporal splits, feature-family figures, calibration-impact tables,
  categorical-feature tables, and claim-artifact-test tables.
- Rewrote internal theorem/proposition references from citation-like handles
  into section/equation cross-references so they no longer look like missing
  bibliography keys.

## Material Confirmed As Incorporated

### From LinkedIn Intake

- **Brier/Yates and reliability diagrams**: incorporated in Chapter 06
  calibration and champion-selection language.
- **Gini/Somers and rare events**: incorporated in Chapter 06 model comparison.
- **WOE/IV uncertainty and Naive Bayes analogy**: incorporated in Chapter 05.
- **PSI caution**: incorporated in Chapter 10 model risk management.
- **RuleFit and GBDT leaf one-hot scorecards**: incorporated in Chapter 11
  explanation-drift language.

No LinkedIn-only claim is promoted as public evidence. The useful material is
either backed by independent sources or parked as intake.

### From The Main Book Into Paper CRPTO

- **Must enter IJDS body**: CRPTO pipeline, calibrated PD gate, Mondrian
  conformal intervals, `Gamma_CP`, `V`, robust LP, champion, exact region,
  SPO+ boundary, compact governance implications.
- **Must enter IJDS supplement**: A3--A18 robustness, Mondrian ablation, funded
  set composition, fair-lending proxy, MRM, claim-artifact-test map, data/code
  disclosure.
- **Must enter thesis chapter**: foundations from Chapters 02--10, WOE/IV
  governance, metric-governance caveats, full Paper 4 living-lab closure, and
  research stop rules.
- **Should stay out of IJDS body**: GPU, quantum, deep IFRS9 contractual claims,
  CATE policy value, online conformal, MDCP, DLA, and OCE/CVaR as a new
  optimized objective.

## Narrative Issues Found

1. The book is coherent as a research repository, but its landing page was
   undercounting the paper surface. Fixed.
2. Chapter 14 had a mature extraction manifest, but the venue target was still
   generic. Fixed toward IJDS.
3. The name `Paper Estrella` is useful internally but should not survive into
   public-facing manuscript text. The mini-book now defines the alias boundary.
4. Several duplicate Quarto labels could confuse cross-reference resolution in a
   full render. Fixed.
5. Some theorem references were written as `@thm-*`/`@lem-*` without Quarto
   theorem environments. Fixed by pointing to section/equation labels.

## IJDS Fit

CRPTO fits IJDS better than a pure OR or credit-scoring venue because the core
claim combines:

- real data and a decision-making environment;
- a data science method/approach, not just an application;
- managerial/industrial relevance through portfolio selection;
- practical and ethical implications through reproducibility, MRM, and fairness
  proxy governance.

The main pressure point is page economy. The manuscript should use three figures
and two or three tables in the body, pushing proof details, robustness and
governance tables into the online supplement.

## Thesis Fit

For a laureate-oriented master's thesis, the strongest structure is not to submit
the entire Quarto book as the thesis. The thesis should make CRPTO the central
chapter and use the full book as intellectual infrastructure. Recommended thesis
logic:

1. Credit risk and decision uncertainty.
2. PD calibration and conformal prediction.
3. Robust optimization and CRPTO.
4. Empirical Lending Club design and artifacts.
5. Results and robust region.
6. Governance, fairness proxy, MRM and reproducibility.
7. Extensions and research agenda.

This preserves ambition without making the thesis look like many disconnected
papers.

## Stop Rule

The full-audit loop is closed when:

- the Paper CRPTO mini-book renders;
- duplicate labels stay cleared;
- paper/book guardrail tests pass;
- the final manuscript extraction uses `Paper CRPTO`, not `Paper Estrella`, in
  public-facing surfaces.
