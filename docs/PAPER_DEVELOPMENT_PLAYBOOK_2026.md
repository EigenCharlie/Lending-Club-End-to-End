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
1. Refresh the operational baseline when needed with `scripts/run_canonical_rebuild.py`.
2. Use `scripts/run_champion_search.py` only when searching for a better champion or new frozen parameters.
3. Run `scripts/run_insights_factory.py --profile canonical|research` for complementary figures, notebook evidence, and paper material.
4. Keep manuscript claims tied to concrete artifact paths.
5. Run tests before snapshotting manuscript numbers:
   - `uv run pytest -q`
6. Refresh DVC/remote status before freezing a paper draft:
   - `uv run dvc status --json`
   - `uv run dvc status -c --json`

## 6) Current Research Workspaces
- `Paper 2: IFRS9 E2E`
- `Paper 4: Sequential Credit Decision Analytics Living Lab`
- `Buenas Prácticas y Herramientas` (guía transversal)

CRPTO/Paper IJDS ya no se desarrolla como paper local en este repositorio:
su fuente de verdad vive autocontenida en `C:/Users/carlos/Documents/Paper_CRPTO`.
Paper 3, Paper GPU y Paper Quantum fueron retirados como superficies
editoriales el 2026-06-06.

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
- `notebooks/11_paper2_ifrs9_e2e.ipynb`

Paper 4 no tiene notebook único: se gobierna mediante
`reports/paper_material/paper4/notes/paper4_living_lab_notebook.md` y scripts
`scripts/papers/build_paper4_*.py`.

Generated outputs:
- `reports/paper_material/paper2/*`
- `reports/paper_material/paper4/*`

Batch execution command:
- `uv run python scripts/run_paper_notebook_suite.py`
