# Paper 4 to Paper Estrella Export Memo - 2026-05-17 (SUPERSEDED)

**Status: `superseded` as of 2026-07-20.**

This file is retained only as historical provenance. The proposed export below
was not transferred into the active standalone CRPTO IJDS paper. That paper is
now a retrospective identification audit: it does not select a learner,
residual window, taxonomy, gamma, ruler, coordinate, cap, comparator, or policy.
Paper 4 therefore treats CRPTO as an autonomous external claim surface and does
not use F03, F04, F05, SDAM, CVaR, or PyEPO to extend its active claims.
The observed external authority is pinned in
`docs/research/crpto_external_contract_2026-07-20.yml` at
`Paper_CRPTO@69095e05beae282701b4ea38aa69da26a209106f`; this historical memo is
not evidence for finding F02.

The active Paper 4 interpretation is:

- F03 is an execution/provenance fact under one internal simulator;
- F04 is conditional descriptive simulation evidence;
- F05 consists of post-selection slices, not strict holdouts;
- SDAM is a design and governance checklist, not an optimizer;
- PyEPO is a teacher-cost benchmark, not realized-loss or CRPTO policy evidence.

Everything below this notice records the May 2026 decision and is not active.

## Historical decision (not active)

Close the Paper 4 procedural loop and export only the pieces that strengthen the
official Paper Estrella. Paper Estrella remains the official publication and
champion surface; Paper 4 remains a long-horizon living lab.

The 14 pending Paper 4 review outcomes are accepted in
`reports/paper_material/paper4/tables/paper4_loop_closure_accept_outcomes_2026-05-17.csv`
for a bounded four-chapter patch. This is an owner closure decision, not external
peer review and not a Paper 4 final promotion.

## Historical proposed export (not executed in the active CRPTO IJDS paper)

| ID | Paper 4 finding | Paper Estrella use | Boundary |
|---|---|---|---|
| F03 | Dynamic stress replay reached 512 common paths and 494,592 trace rows. | Extra robustness context for the official champion. | Internal replay only; not a forecast or new champion search. |
| F04 | CVaR/OCE challengers reduced tail loss but did not robustly beat Paper Estrella on paired wealth (`prob_beats = 47.65625%`). | Justifies retaining the economic champion over a tail-risk-first challenger. | Challenger evidence only; no CVaR/OCE promotion. |
| F05 | Online conformal monitoring passed internal proxy gates but lacks external/source holdout proof. | Strengthens limitations and future work. | No live deployment claim. |
| SDAM | Powell/SDAM cleanly classifies CRPTO as a CFA-style policy with a single-period decision state. | Improves theoretical positioning in the introduction/discussion. | Framing only; not DLA/Bellman optimality. |

## What Stays In Paper 4

| Lane | Keep | Why |
|---|---|---|
| Source governance | `paper4_v468_tight_source_rankings.csv` and source pressure diagnostics | Useful if Paper Estrella later needs an operational governance paragraph. |
| Formal claim matrix | Current claim-boundary table, v467-v478 export/provenance packets and the closure artifacts | Useful as thesis/lab provenance, not as manuscript body. |
| Full living-lab history before v490 | Official v1-v38 Quarto evidence plus curated v467-v478 export evidence | Scientific diagnostics retained only where they support the rendered book or bounded Paper Estrella export. |

## What Is Deleted

The v490--v526 builders, status JSONs, notes and procedural CSVs are retired.
They documented repeated review-gate requests, candidate nomination packets,
dispatch evidence requests and missing-evidence follow-ups. They did not add
scientific evidence beyond the accepted four-chapter patch scope.

The deletion manifest is:
`reports/paper_material/paper4/tables/paper4_loop_closure_cleanup_manifest_2026-05-17.csv`.

A second cleanup on 2026-05-18 retired generated v39-v466 and v479-v489 lab
artifacts plus one-shot v39-v489 builders. Its manifest is
`reports/paper_material/paper4/tables/paper4_deep_cleanup_manifest_2026-05-18.csv`.

## Chapter Patch Scope

The accepted Paper 4 patch updates four Quarto chapters:

| Chapter | Patch role |
|---|---|
| `19bv-v33-cvar-certificate.qmd` | Accepts F04 as a CVaR/OCE caveat that supports retaining the economic champion. |
| `19bx-v35-online-macro-validation.qmd` | Accepts F05 as an online-conformal limitation rather than deployment evidence. |
| `19ca-v38-final-synthesis.qmd` | Records Paper 4 closure and the accepted/deleted boundary. |
| `19f-sequential-decision-framework.qmd` | Clarifies SDAM as framing/contract rather than optimizer. |

Two Paper Estrella chapters are also updated as the export destination:
`14h-journal-appendix-robustness.qmd` receives the F03/F04/F05/SDAM export table,
and `14e-discussion-conclusions.qmd` receives the limitation/future-work
language.

## Guardrails

- Do not create `reports/paper_material/paper4/status/paper4_final_promotion.json`.
- Do not reopen `models/final_project_promotion.json`.
- Do not cite Paper 4 CVaR/OCE or online findings as Paper Estrella champion
  evidence.
- Use F03/F04/F05/SDAM as robustness, limitations and framing only.

## Validation Note

The six patched chapters rendered successfully with targeted Quarto renders
during the closure pass. In the 2026-05-18 deep-cleanup pass, the full book
render still stops before the Paper Estrella/Paper 4 surface at
`chapters/06-pd-modeling/06c-calibration-selection.qmd` with the known Quarto
kernel-cleanup `AssertionError`. The validated evidence for this cleanup is
therefore `ruff`, `git diff --check`, absence of `paper4_final_promotion.json`,
the compact Paper 4 guardrail test suite, and explicit checks that every
Paper 4 CSV read by the retained Quarto/Paper Estrella surfaces still exists.
