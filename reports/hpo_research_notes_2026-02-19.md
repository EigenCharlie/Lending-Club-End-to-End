# HPO Research Notes (CatBoost + Optuna) — 2026-02-19

## Fuentes revisadas
- CatBoost docs: Parameter tuning
  - https://catboost.ai/docs/en/concepts/parameter-tuning.html
- CatBoost docs: `bagging_temperature`
  - https://catboost.ai/docs/en/references/training-parameters/common#bagging_temperature
- CatBoost docs: `bootstrap_type`
  - https://catboost.ai/docs/en/references/training-parameters/common#bootstrap_type
- CatBoost docs: `random_strength`
  - https://catboost.ai/docs/en/references/training-parameters/common#random_strength
- CatBoost docs: `has_time`
  - https://catboost.ai/docs/en/references/training-parameters/common#has_time
- Optuna docs: Efficient optimization + pruners
  - https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- Optuna docs: TPESampler
  - https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- Optuna docs: CatBoostPruningCallback
  - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.CatBoostPruningCallback.html
- Optuna example (raw): CatBoost + pruning callback
  - https://raw.githubusercontent.com/optuna/optuna-examples/main/catboost/catboost_pruning.py
- Kaggle Lending Club benchmark (script público del repo)
  - script: `scripts/benchmark_kaggle_lendingclub.py`
  - output: `data/processed/lendingclub_benchmark_summary.json`
- Lending Club peer-reviewed benchmark
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9222552/
  - Nota: se corrigió una referencia previa mal enlazada (`PMC9533764`), que corresponde a un artículo clínico de COVID-19 no relacionado con Lending Club.

## Hallazgos accionables
1. El pruning de Optuna solo aporta si hay reportes intermedios del trial.
   - En CatBoost, esto se habilita con `CatBoostPruningCallback` y luego `check_pruned()`.
2. TPE multivariado es útil cuando hay interacción entre hiperparámetros (común en CatBoost).
3. En CatBoost para tabular de crédito, conviene priorizar tuning de:
   - `learning_rate`, `depth`, `l2_leaf_reg`, `rsm`, `random_strength`, `bootstrap_type` (+ `bagging_temperature` o `subsample` condicional), `border_count`.
4. Con datasets temporales, `has_time=true` ayuda a mantener consistencia temporal en el tratamiento del orden.
5. Kaggle Lending Club (49 notebooks públicos analizados en este run) confirma:
   - familias dominantes: LR/RF/XGBoost;
   - CatBoost aparece menos, pero SHAP/XAI aparece recurrentemente;
   - AUC reportado parseado máximo: 0.7336 (con riesgo de variación por protocolo de split).

## Cambios aplicados al pipeline
- `src/models/pd_model.py`
  - HPO con TPE multivariado + startup trials.
  - Pruner median con warmup/startup.
  - `CatBoostPruningCallback` operativo (si integración disponible).
  - Espacio de búsqueda condicional (`bootstrap_type` -> `bagging_temperature` o `subsample`).
  - Parámetros adicionales: `random_strength`, `border_count`.
  - `random_seed` fijo por trial para reducir ruido en comparación de hiperparámetros.
  - Refit opcional del modelo final en `train+val` usando `best_iteration`.
- `scripts/train_pd_model.py`
  - Pasa nuevos parámetros HPO desde config.
- `configs/pd_model.yaml`
  - Run largo con `n_trials=400`, TPE multivariado, pruning callback, estudio persistente SQLite.

## Riesgos/metodología
- AUC muy altos de notebooks públicos no se toman como objetivo operativo si no hay validación temporal OOT estricta.
- Métricas objetivo del proyecto deben juzgarse en OOT y con calibración (Brier/ECE), no solo ranking.
