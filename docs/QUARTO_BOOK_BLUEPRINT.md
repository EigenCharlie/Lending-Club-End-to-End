# Quarto Book Blueprint

## Objective
- Leave the repository future-ready for a Quarto book without migrating the main narrative yet.
- Keep Streamlit as the interactive/exploratory layer and Quarto as the long-form editorial layer.

## Official narrative axes
- `Pipeline Operativo`
  Data -> PD -> calibration -> conformal -> survival/time series -> IFRS9 -> portfolio policy -> governance.
- `Insight Factory`
  Explainability, causal extensions, notebooks, RAPIDS benchmarks, side projects, extended figures and paper drafts.

## Artifact contract for future Quarto chapters
- Every narrative page should declare:
  - `narrative_axis`
  - `pipeline_role`
  - `artifact_scope`
  - `book_chapter`
- Quarto chapters must consume frozen artifacts only.
- Interactive widgets remain in Streamlit unless they become static evidence tables/figures.

## Initial chapter map
- `00-executive-map`
- `01-glossary-and-foundations`
- `02-operational-pipeline`
- `03-pd-modeling-and-calibration`
- `04-conformal-and-uncertainty`
- `05-survival-time-series-and-causal`
- `06-portfolio-policy-and-selection`
- `07-ifrs9-and-governance`
- `08-explainability-and-insights`
- `09-specialization-to-masters-bridge`
- `10-research-agenda-and-contributions`
- `11-paper-cp-robust-opt`
- `12-paper-ifrs9-e2e`
- `13-paper-mondrian`
- `14-appendix-notebook-atlas`
- `15-streamlit-exploration`
- `16-quarto-and-publication-contracts`

## Streamlit -> Quarto mapping rules
- Canonical pipeline pages become core book chapters.
- Insight factory pages become appendices, sidebars, research chapters, or optional companion notes.
- Pages marked `artifact_scope=shared` can feed both layers.
- Pages marked `artifact_scope=research` should never be presented as operationally frozen evidence.
