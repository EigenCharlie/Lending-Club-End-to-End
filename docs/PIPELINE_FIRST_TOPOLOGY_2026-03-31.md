# Pipeline-First Topology

> **HISTORICAL EXECUTION TOPOLOGY (current boundary added 2026-07-20).** The
> family names below remain useful for rebuilding the local freeze. They do not
> define current paper status: external CRPTO has no selected learner or policy,
> Paper 2 is `parked_ifrs9`, and Paper 4 is a living lab without promotion.

## Familias oficiales
- `core_canonical`: rebuild reproducible del champion stack, CPU-only por defecto, sin causal, survival derivado, GPU ni notebooks.
- `search_pd`: HPO PD, challenger/monotonic y sidecars de evaluación del propio modelo.
- `search_conformal`: benchmark, tuning y sensibilidad conformal sobre artefactos PD ya congelados.
- `search_portfolio`: selector económico, tradeoff frontier y A/B guarded sin reentrenar PD.
- `search_paper2_ifrs9`: búsqueda exhaustiva derivada para survival/LGD-EAD/PoC e IFRS9 sobre upstream canónico congelado.
- `paper1_e2e`: stack end-to-end del paper estrella: PD + conformal + portfolio + comparadores paper-grade.
- `paper2_e2e`: stack derivado TS + survival/LGD-EAD + IFRS9/SICR.
- `diagnostics_governance`: backtesting, bootstrap, interpretación, fairness, governance y MRM.
- `research_labs`: causal, GPU/RAPIDS, notebooks y side projects.

## Entry points oficiales
- `scripts/run_canonical_rebuild.py` -> `core_canonical`
- `scripts/run_champion_search.py` -> `search_pd`
- `scripts/run_insights_factory.py` -> `research_labs`

## Entry points organizados
- `scripts/search/run_pd_search.py`
- `scripts/search/run_conformal_search.py`
- `scripts/search/run_conformal_reopen_search.py`
- `scripts/search/run_portfolio_search.py`
- `scripts/search/run_paper2_ifrs9_search.py`
- `scripts/papers/run_paper2_e2e.py`
- `scripts/diagnostics/run_governance_diagnostics.py`
- `scripts/labs/run_research_labs.py`

## Compatibilidad corta
- `scripts/run_long_pipeline.py` y `scripts/end_to_end_pipeline.py` quedan como entrypoints de compatibilidad.
- Los aliases legacy siguen aceptados en CLI:
  - `canonical_rebuild` -> `core_canonical`
  - `champion_search` -> `search_pd`
  - `challenger_promotion` -> `search_pd`

## Contratos declarativos
- Familias: `configs/pipelines/*.yaml`
- Perfiles: `configs/profiles/*.yaml`
- Registros vivos:
  - `configs/pipeline_registry/pipeline_matrix.yaml`
  - `configs/pipeline_registry/script_role_registry.yaml`
  - `configs/pipeline_registry/artifact_flow_registry.yaml`
  - `configs/pipeline_registry/search_registry.yaml`

## Regla operativa
- Ningún lane `research_only` debe sobrescribir artefactos canónicos.
- `search_pd` no debe disparar survival, causal, GPU ni notebooks por defecto.
- `search_conformal` puede abrir workflows exhaustivos namespaced como `run_conformal_reopen_search.py`, pero no debe sobrescribir artefactos canónicos durante la fase de búsqueda.
- `search_portfolio` consume PD/conformal congelados; no reentrena PD.
- `paper2_e2e` declara survival explícitamente; ya no lo hereda del core.
- `search_paper2_ifrs9` reutiliza la semántica de `paper2_e2e`, pero como lane de búsqueda exhaustiva no-canónica.
