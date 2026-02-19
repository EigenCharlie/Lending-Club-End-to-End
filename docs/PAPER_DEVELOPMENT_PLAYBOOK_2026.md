# Paper Development Playbook (2026)

## Objective
Convert project artifacts into publication-ready papers with repeatable, auditable workflows.

## 1) Canonical Paper Lifecycle
1. Research question + claim definition
2. Related work and positioning
3. Methods and experimental protocol
4. Main results and ablations
5. Threats to validity and limitations
6. Reproducibility package (code + data + commands)
7. Venue-specific checklist and formatting

## 2) Section Checklist (IMRaD+)
- Title and Abstract
- Introduction
- Related Work
- Data and Methods
- Results
- Robustness/Sensitivity/Ablations
- Threats to Validity
- Conclusion and Future Work
- Reproducibility Appendix

## 3) Practical Toolchain
### Authoring and executable manuscripts
- Quarto: https://quarto.org/docs/guide/
- Jupyter Book: https://jupyterbook.org/en/stable/intro.html
- MyST markdown: https://mystmd.org/guide

### Notebook execution and parameterization
- Papermill: https://papermill.readthedocs.io/en/latest/
- nbclient: https://nbclient.readthedocs.io/en/latest/
- Jupytext: https://jupytext.readthedocs.io/en/latest/

### Figure/table export for papers
- Plotly static export: https://plotly.com/python/static-image-export/
- Pandas Styler to LaTeX:
  https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html

### Bibliography automation and literature API access
- Crossref REST API: https://api.crossref.org/
- Semantic Scholar API: https://www.semanticscholar.org/product/api
- OpenAlex client (PyAlex): https://github.com/J535D165/pyalex

### Reproducibility and citation
- DVC docs: https://dvc.org/doc
- Zenodo API/docs: https://developers.zenodo.org/
- CITATION on GitHub:
  https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content

## 4) Venue and Reporting Checklists
- ICMJE recommendations: https://www.icmje.org/recommendations/
- NeurIPS checklist reference (Call for Papers): https://neurips.cc/Conferences/2025/CallForPapers
- ICLR Author Guide (reproducibility statement/checklist): https://iclr.cc/Conferences/2025/AuthorGuide
- ACM artifact review and badging: https://www.acm.org/publications/policies/artifact-review-and-badging-current

## 5) Recommended Workflow in This Repo
1. Rebuild canonical artifacts:
   - `uv run dvc repro build_pipeline_results export_streamlit_artifacts`
2. Generate paper-specific evidence tables from processed artifacts.
3. Keep manuscript claims tied to concrete artifact paths.
4. Run tests before snapshotting manuscript numbers:
   - `uv run pytest -q`
5. Refresh DVC/remote status before freezing a paper draft:
   - `uv run dvc status --json`
   - `uv run dvc status -c --json`

## 6) Current Research Workspaces in Streamlit
- `Paper 1: CP + Robust Opt`
- `Paper 2: IFRS9 E2E`
- `Paper 3: Mondrian`
- `Buenas Prácticas y Herramientas` (guía transversal para los 3 drafts)

Paper pages include:
- full draft structure (metadata, abstract, introduction, related work, methods, results, discussion, threats, reproducibility),
- numbered equations (`st.latex`) and numbered figures/tables,
- live results from canonical artifacts and downloadable tables,
- final section with phase tracker + bullet list of pending improvements by section/figure/table.

The shared best-practices page includes:
- writing and reproducibility standards,
- Streamlit patterns for LaTeX, figures, and tables,
- references to tooling and reporting checklists.

## 7) Paper Notebook Suite (Repository)
- `notebooks/10_paper1_cp_robust_opt.ipynb`
- `notebooks/11_paper2_ifrs9_e2e.ipynb`
- `notebooks/12_paper3_mondrian.ipynb`

Generated outputs:
- `reports/paper_material/paper1/*`
- `reports/paper_material/paper2/*`
- `reports/paper_material/paper3/*`

Batch execution command:
- `uv run python scripts/run_paper_notebook_suite.py`
