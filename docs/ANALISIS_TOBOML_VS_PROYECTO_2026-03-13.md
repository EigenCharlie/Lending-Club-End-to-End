# Analisis integral del proyecto vs. *The Orange Book of Machine Learning* (Green edition, v2.4.0)

Fecha: 2026-03-13

Libro analizado:
- `thesis_poster/TOBoML2 (2).pdf`

Fuentes vivas principales usadas para este analisis:
- `configs/baselines/canonical_operational_baseline.json`
- `models/threshold_semantics.json`
- `reports/run_comparisons/champion-2026-03-12-mega-definitive/comparison.md`
- `data/processed/pipeline_summary.json`
- `data/processed/model_comparison.json`
- `models/conformal_policy_status.json`
- `models/time_series_status.json`
- `models/causal_effect_status.json`
- `models/cate_portfolio_status.json`
- `models/ab_simulation_status.json`
- `models/governance_status.json`
- `models/fairness_audit_status.json`
- `models/champion_policy_selection_status.json`
- `models/challenger_promotion_report.json`
- `scripts/run_long_pipeline.py`
- `scripts/run_insights_factory.py`
- `scripts/run_rapids_insight_factory.py`
- `docs/backlog-13-03.md`

Jerarquia de verdad aplicada:
1. status artifacts y metrics del baseline oficial actual
2. scripts y modulos fuente
3. backlog actual
4. documentacion historica

## 1. Estado real del proyecto hoy

### 1.1 Baseline oficial y mapa de pipelines

Estado operativo oficial:
- baseline oficial actual: `champion-2026-03-12-mega-definitive`
- source of truth del baseline: `configs/baselines/canonical_operational_baseline.json`
- pipeline families reales:
  - `canonical_rebuild`: reconstruccion congelada y reproducible
  - `champion_search`: carril pesado de busqueda, seleccion y promocion
  - `insights_factory`: carril research y evidencia complementaria

Lectura:
- la arquitectura del repo ya esta conceptualmente alineada con el espiritu del libro: tabular ML pragmatico, validacion seria, y foco en prediccion util antes que sofisticacion innecesaria.
- el proyecto ya no esta en fase de "armar baseline". Esta en fase de "cerrar metodo, promotion logic y narrativa regulatoria".

### 1.2 Metricas vigentes del baseline oficial

PD core:
- mejor modelo: `CatBoost (tuned + calibrated)`
- calibracion oficial: `Isotonic Regression`
- AUC: `0.7128`
- Gini: `0.4256`
- Brier: `0.1545`
- D2 Brier: `0.0988`
- ECE: ~`0.0062-0.0064`
- HPO: `295` trials, mejor validation AUC historico `0.7226`

Conformal PD:
- coverage_90: `0.9257`
- coverage_95: `0.9516`
- min_group_coverage_90: `0.8992`
- avg_width_90: `0.7550`
- promotion gate: `PASS`
- strict policy `overall_pass`: `false`
- warnings fuertes: `Kupiec` y `Christoffersen` rechazan cobertura exacta

Time series:
- point champion: `AutoARIMA`, `promotable=true`
- interval champion: `AutoARIMA`, `promotable=false`
- coverage_90 de intervalos: `0.8102`
- warning adicional: exogenous future contract ausente o deshabilitado

Causal / CATE:
- ATE: `0.00975`
- IC cruza cero: `[-0.0331, 0.0525]`
- refutaciones: no disponibles
- CATE portfolio: `promotion_eligible=false`
- cambio objetivo CATE ajustado: `-4.47%`

A/B economico:
- control no robusto: retorno `221,297.45`, funded `209`
- champion robusto: retorno `227,629.59`, funded `233`
- delta total return: `+6,332.14`
- p-value bootstrap: `0.4495`
- gate real: `no_regression`

Governance:
- `overall_pass=true`
- `max_psi=0.1394`
- `score_psi=0.0145`
- `c2st_auc=0.9853`
- warnings: `warn_c2st=true`, `warn_distribution_tests=true`

Fairness:
- `overall_pass=true`
- `6/6` atributos pasan
- threshold operativo de negocio/fairness: `0.35`

Portfolio champion:
- selector: `economic_actual_ab_v3`
- policy champion: `risk_tolerance=0.18`, `policy_mode=capped_blended_uncertainty`, `gamma=0.05`, `delta_cap_quantile=0.9`

Survival:
- `cox_concordance=0.6643`
- `rsf_concordance=0.6797`

### 1.3 Contradicciones y documentos desfasados

Hay contradicciones reales entre documentos historicos y estado vigente:
- `SESSION_STATE.md` todavia mezcla snapshots previos y no refleja por completo la promocion del 2026-03-13.
- `reports/project_audit_snapshot.json` contiene metricas historicas que no corresponden al baseline oficial actual.
- `reports/INFORME_INTEGRAL_PROYECTO_RIESGO_CREDITO_ES.md` es util como contexto, pero varias cifras y afirmaciones quedaron viejas.
- la cantidad de pruebas reportada en docs historicos (`5`, `76`, `463`) ya no debe usarse como verdad viva; hoy el arbol contiene `81` archivos `test_*.py` y `351` funciones `test_`.

Conclusion operativa:
- el repo necesita seguir usando una regla estricta de narrativa: artefactos vivos primero, documentos historicos despues.

## 2. Lectura del libro, capitulo por capitulo, y que cambia en este proyecto

## Capitulo 1. Introduction

Clasificacion:
- aplica fuerte
- ya cubierto bien, pero con dos mejoras narrativas claras

Que aporta el capitulo:
- diferencia entre estimacion explicativa y prediccion
- prediccion vs forecast
- incertidumbre aleatorica vs epistemica
- intervalos de confianza vs intervalos de prediccion
- explainability vs interpretability
- correlacion vs causalidad

Que ya existe en el repo:
- tesis central basada en decision bajo incertidumbre
- separacion real entre PD, forecasting, causalidad y optimizacion
- Streamlit y docs ya cuentan historia de correlacion vs causalidad y de point estimate vs interval

Que falta o esta debil:
- la narrativa aun mezcla a veces threshold interno de modelado PD con threshold operativo de fairness (`0.05` vs `0.35`)
- faltan disclaimers mas consistentes sobre "prediction interval" vs "confidence interval", especialmente en time series y conformal
- causalidad sigue presentada como capacidad instalada, pero operativamente aun no cumple criterio canonico

Que si vale la pena cambiar o agregar:
- blindar en `canonical_rebuild` una narrativa unica para:
  - PD probabilistico
  - intervalos conformal
  - threshold operativo oficial
  - estatus research-only de causal/CATE
- quick win: estandarizar en docs/UI el lenguaje de incertidumbre aleatorica vs epistemica

Que no vale la pena tocar ahora:
- reescribir la tesis conceptual de alto nivel; ya esta bien encaminada

## Capitulo 2. Statistics

Clasificacion:
- aplica parcialmente
- cubierto, pero mejorable en EDA y governance

Que aporta el capitulo:
- dispersion robusta, cuantiles, IQR, skewness, kurtosis, no normalidad, desigualdades y familias de distribucion

Que ya existe en el repo:
- EDA basica, drift tests, PSI, KS, CvM, monitoreo de score y distribuciones

Que falta o esta debil:
- falta una capa mas robusta de estadistica descriptiva para explicar por que `KS/CvM/C2ST` explotan con muestras grandes mientras `PSI` sigue razonable
- no hay una politica explicita de materialidad estadistica vs materialidad operacional

Que si vale la pena cambiar o agregar:
- `canonical_rebuild`: agregar tabla canonica de materialidad con cuantiles, IQR, MAD y efecto en score/decision para top drift features
- `champion_search`: convertir governance de "drift test p-value" a "materialidad + impacto en score + impacto en decision"

Que no vale la pena tocar ahora:
- extender EDA con mas tests de normalidad solo por completitud

## Capitulo 3. Exploratory Data Analysis (EDA)

Clasificacion:
- cubierto pero mejorable

Que aporta el capitulo:
- calidad de datos
- descriptivos utiles
- outliers e inliers
- correlacion mas alla de Pearson
- visualizaciones que realmente revelan estructura

Que ya existe en el repo:
- notebook de EDA ejecutado
- resumenes de dataset y diccionario
- visualizacion narrativa en Streamlit

Que falta o esta debil:
- no veo un uso sistematico de `Theil's U`, `PhiK` o `MI` para dependencias mixtas y redundancia de features
- falta una auditoria mas formal de redundancia entre engineered features del champion
- el proyecto ya esta mas alla del pairplot; necesita analisis de dependencia mas util para seleccion y governance

Que si vale la pena cambiar o agregar:
- `champion_search`: agregar auditoria de redundancia tipo `MI` o `mRMR` como filtro diagnostico para el challenger PD
- `insights_factory`: crear una nota/artifact de dependencias no lineales y mixtas para storytelling tecnico
- quick win: enriquecer la narrativa de top drift features con eCDF y cuantiles, no solo p-values

Que no vale la pena tocar ahora:
- pairplots masivos o expansion cosmetica del notebook EDA

## Capitulo 4. Data cleaning

Clasificacion:
- cubierto pero mejorable

Que aporta el capitulo:
- missingness MCAR/MAR/MNAR
- estrategias de imputacion
- outliers
- duplicados
- zero variance
- encoding
- PII

Que ya existe en el repo:
- CatBoost aprovecha NaN y categoricas nativas
- hay contratos de features y pipeline de preparacion estable
- existe limpieza de datos y feature engineering consolidado

Que falta o esta debil:
- la historia de missingness esta implicita en el codigo, no explicita en la narrativa de metodo
- el baseline fuerte depende de CatBoost; la capa interpretativa/challenger podria beneficiarse de indicadores de missingness mas explicitos
- no aparece una auditoria canonica de PII como guardrail visible, aunque el dataset sea historico y de Lending Club

Que si vale la pena cambiar o agregar:
- `canonical_rebuild`: agregar evidencia explicita de tratamiento de missingness y por que CatBoost + baseline LR no requieren la misma estrategia
- `champion_search`: probar missingness indicators solo en challengers interpretables, no en el champion actual
- `backlog`: incorporar un checklist de PII/data minimization para tesis y anexos

Que no vale la pena tocar ahora:
- reemplazar la logica central de limpieza; el beneficio esperado es bajo frente a otros huecos

## Capitulo 5. Cross-validation

Clasificacion:
- aplica fuerte
- uno de los capitulos mas importantes para el proyecto

Que aporta el capitulo:
- three-way split
- CV serio
- nested CV
- leakage
- covariate shift
- concept drift
- two-sample tests y C2ST

Que ya existe en el repo:
- validacion temporal real
- split train/cal/test temporal
- leakage guard en conformal tuning
- drift monitoring con PSI, KS, CvM y C2ST

Que falta o esta debil:
- HPO sigue muy atado a una logica de mejor estudio/seed, no a robustez inter-seed
- governance usa `C2ST` y p-values, pero no cierra del todo la pregunta de materialidad
- conformal PD tiene promotion gate pragmatica y strict gate estadistica, pero aun sin una narrativa final totalmente cerrada

Que si vale la pena cambiar o agregar:
- P0 en `champion_search`: agregar robustez a random seed para HPO y calibracion, no solo mas trials
- P0 en `canonical_rebuild`: explicitar por que `Kupiec/Christoffersen` son diagnosticos no bloqueantes en promocion pero siguen vivos en MRM
- P1 en `governance`: crear una politica de materialidad que combine PSI, delta score, delta AUC/Brier y cambio de decision

Que no vale la pena tocar ahora:
- nested CV completa sobre todo el stack PD; seria cara y no necesariamente de mejor ROI que repeated temporal validation focalizada

## Capitulo 6. Interpolation and smoothing

Clasificacion:
- baja prioridad / no aplica fuerte

Que aporta el capitulo:
- smoothing e interpolacion para series y señales

Que ya existe en el repo:
- forecasting y escenarios IFRS9

Que falta o esta debil:
- poco uso explicito de smoothing diagnostico en series residuales o escenarios

Que si vale la pena cambiar o agregar:
- `insights_factory`: usar smoothing solo como diagnostico para residual drift y visualizaciones de tasas de default
- posible uso acotado en IFRS9 temporal storytelling

Que no vale la pena tocar ahora:
- abrir una linea de desarrollo dedicada a interpolacion; no mueve el cuello de botella actual

## Capitulo 7. Regression

Clasificacion:
- aplica parcialmente, pero muy fuerte en intervalos y monotonicidad

Que aporta el capitulo:
- baselines simples
- overfitting
- bias-variance
- monotonic constraints
- quantile regression
- conformal prediction intervals
- CQR
- locally weighted conformal
- Winkler score

Que ya existe en el repo:
- Winkler score ya esta en la politica conformal
- monotonic challenger ya existe
- conformal tuning ya tiene split leakage-free, Pareto front y guardbands
- LGD/EAD conformal ya muestra que el proyecto sabe seleccionar variantes con criterio

Que falta o esta debil:
- en PD el challenger monotono es promotable, pero aun no esta integrado como candidato serio a decision core
- en conformal PD no hay selector de variante tan rico como el que ya implicitamente existe en LGD/EAD
- en time series intervals no hay aun una estrategia madura tipo CQR/ACI/EnbPI/online conformal

Que si vale la pena cambiar o agregar:
- P0 en `champion_search`: formalizar selector conformal PD por variantes, inspirado en la disciplina ya usada en LGD/EAD
- P1 en `champion_search`: promover benchmarking serio de monotonic challenger contra champion, porque hoy gana mucho en consistencia monotona pero pierde poco menos de 1 punto AUC
- P0 en `time_series`: atacar intervalos con `ACI`, `EnbPI`, `OnlineConformal` y revisar si el problema es forecast base, no conformal solamente

Que no vale la pena tocar ahora:
- mover el champion PD a una familia cuantile/CQR como si fuera problema de regresion; el target central sigue siendo clasificacion probabilistica

## Capitulo 8. Classification

Clasificacion:
- aplica fuerte
- es el corazon del proyecto

Que aporta el capitulo:
- logistic regression
- tree classifiers
- proper scoring rules
- calibration
- reliability diagrams
- Venn-Abers
- thresholding
- confusion metrics
- imbalance

Que ya existe en el repo:
- baseline LR y final CatBoost
- calibracion temporal multi-candidato
- metricas probabilisticas y threshold governance
- fairness threshold operativo separado

Que falta o esta debil:
- `venn_abers` sigue listado como candidato, pero en `model_comparison.json` su evaluacion es claramente defectuosa: AUC colapsa de ~0.71 a ~0.28 y ECE/Brier empeoran fuertemente
- aun hay riesgo narrativo entre threshold interno de busqueda PD (`0.05`) y threshold operativo de fairness/negocio (`0.35`)
- el proyecto ya esta muy bien en AUC/Brier/ECE, por lo que seguir exprimiendo ranking puro probablemente no sea el mejor uso del tiempo

Que si vale la pena cambiar o agregar:
- P0 en `champion_search`: sacar `venn_abers` del selector oficial hasta corregirlo o relegarlo a research-only
- P0 en `canonical_rebuild`: dejar un contrato narrativo unico para thresholds internos vs threshold operativo
- P1 en `insights_factory`: añadir una comparativa compacta de proper scoring rules y calibration-refinement decomposition para explicar por que el valor actual esta en probabilidades utiles, no en AUC marginal

Que no vale la pena tocar ahora:
- oversampling o balanceos artificiales del target; el libro es prudente con imbalanced data y el repo ya opera mejor con validacion temporal real

## Capitulo 9. GLM and GAM

Clasificacion:
- aplica parcialmente

Que aporta el capitulo:
- GLM y GAM como punto medio entre interpretabilidad y flexibilidad

Que ya existe en el repo:
- LR baseline regulatoria
- challenger de interpretabilidad basado en seleccion de features y monotonicidad

Que falta o esta debil:
- no hay un challenger estilo GAM/EBM que pueda servir como puente entre "champion fuerte pero complejo" y "baseline interpretable"

Que si vale la pena cambiar o agregar:
- P2 en `insights_factory` o `champion_search`: evaluar un GAM/EBM monotono como challenger interpretativo si se quiere fortalecer defensa de tesis/libro

Que no vale la pena tocar ahora:
- intentar reemplazar CatBoost champion por un GLM/GAM; con las metricas actuales no parece la mejor apuesta

## Capitulo 10. Ensemble estimators

Clasificacion:
- cubierto bien, pero mejorable en forecasting

Que aporta el capitulo:
- random forest
- extra trees
- boosting
- histogram gradient boosting
- stacking
- combinacion convexa de modelos

Que ya existe en el repo:
- CatBoost ya captura gran parte de la leccion del capitulo
- RSF en survival ya dio una mejora clara
- time series usa familia estadistica + ML challengers

Que falta o esta debil:
- no hay evidencia de que stacking en PD vaya a comprar suficiente uplift para justificar complejidad
- en time series si podria valer mas una combinacion simple de modelos puntuales que seguir apostando a un unico campeon puntual

Que si vale la pena cambiar o agregar:
- P1 en `time_series`: evaluar mezcla convexa simple de forecasts puntuales, si mejora estabilidad antes de trabajar intervalos
- P2 en `insights_factory`: dejar stacking de PD solo como exploracion, no como prioridad operativa

Que no vale la pena tocar ahora:
- abrir una carrera de ensambles para el PD champion; el plateau actual de AUC no justifica esa deuda adicional

## Capitulo 11. Hyperparameter optimization (HPO)

Clasificacion:
- aplica fuerte

Que aporta el capitulo:
- optimizer's curse
- early stopping
- limites de tuning
- cambiar random_state

Que ya existe en el repo:
- Optuna serio
- estudio persistente
- pruning
- limpieza de stale trials
- 295 trials ejecutados en el champion vigente

Que falta o esta debil:
- el mayor riesgo ya no es "pocos trials". Es `selection optimism` y sobrelectura de un estudio exitoso
- no hay una narrativa suficientemente fuerte sobre robustez inter-seed y estabilidad del mejor trial

Que si vale la pena cambiar o agregar:
- P0 en `champion_search`: agregar chequeo formal de robustez inter-seed o repeated temporal validation para el mejor bloque de configuraciones, no para todo el search space
- P1 en `champion_search`: convertir el criterio final de seleccion PD en objetivo multi-metrica mas explicito: AUC, Brier, ECE, estabilidad, no solo mejor val AUC
- P1 en `canonical_rebuild`: dejar muy claro que el carril canonico nunca reabre HPO

Que no vale la pena tocar ahora:
- simplemente subir el presupuesto de trials por inercia

## Capitulo 12. Feature engineering and selection

Clasificacion:
- aplica fuerte

Que aporta el capitulo:
- interacciones
- bucketing
- power transforms
- temporal features
- external secondary features
- permutation importance
- SHAP
- LASSO
- mRMR
- Boruta
- chi-square
- PCA

Que ya existe en el repo:
- ratios, buckets, interacciones y features temporales
- auditoria challenger con `MI`, permutation importance y proxy tipo Boruta
- taxonomy de explainability y monotonicidad
- stable core y recent-window regime

Que falta o esta debil:
- `external secondary features` encaja perfecto con la deuda actual de time series: exogenous covariates siguen apagadas o sin contrato futuro
- feature selection del challenger existe, pero no esta aun cerrando el loop con governance y promotion
- no hay una auditoria compacta de redundancia para el champion core

Que si vale la pena cambiar o agregar:
- P0 en `time_series` y `canonical_rebuild`: construir contrato de covariables exogenas futuras para forecasting/IFRS9
- P1 en `champion_search`: sumar `mRMR` o una auditoria de redundancia MI/PhiK para refinar challengers interpretables
- P1 en `canonical_rebuild`: exportar un resumen canonico de familias de features, drivers estables y features excluidas del stable core

Que no vale la pena tocar ahora:
- PCA para el champion PD; seria malo para interpretabilidad y probablemente innecesario con CatBoost tabular

## Capitulo 13. Why no neural networks / deep learning?

Clasificacion:
- aplica fuerte como conclusion estrategica

Que aporta el capitulo:
- una defensa pragmatica del tabular ML clasico frente a DL innecesario

Que ya existe en el repo:
- el stack ya esta correctamente sesgado hacia CatBoost, conformal, RSF, causal forest, forecasting estadistico y optimizacion

Que falta o esta debil:
- nada central

Que si vale la pena cambiar o agregar:
- quick win narrativo: dejar explicito en tesis/libro/presentacion que el cuello de botella actual no es capacidad del modelo, sino:
  - cierre conformal estricto
  - intervalos temporales
  - materialidad de drift
  - evidencia causal y A/B

Que no vale la pena tocar ahora:
- cualquier intento de deep learning en el camino canonico

## 3. Consolidacion por pipeline

## 3.1 champion_search

### P0

1. Formalizar un selector de variantes conformal PD.
Intencion:
- pasar de "promocion operativa con warning" a politica defendible tambien para narrativa paper-grade
Justificacion:
- PD conformal ya tiene cobertura util, pero sigue fallando tests estrictos
- LGD/EAD ya demostro que el repo sabe elegir variantes con criterio
Impacto esperado:
- alto
Costo:
- medio
Dependencia:
- `generate_conformal_intervals.py`, `conformal_tuning.py`, `validate_conformal_policy.py`

2. Corregir o sacar `venn_abers` del selector oficial.
Intencion:
- evitar que un candidato claramente roto siga figurando como calibrador serio
Justificacion:
- hoy colapsa AUC y empeora Brier/ECE
Impacto esperado:
- medio-alto por higiene metodologica
Costo:
- bajo
Dependencia:
- `train_pd_model.py`, `configs/pd_model*.yaml`

3. Rehacer la agenda de time series alrededor de intervalos, no del point forecast.
Intencion:
- atacar el mayor hueco tecnico abierto del proyecto
Justificacion:
- point champion ya es promotable; interval champion no lo es
- el libro enfatiza intervalos y scoring de intervalos, no solo forecast puntual
Impacto esperado:
- alto
Costo:
- medio
Dependencia:
- `forecast_default_rates.py`, `src/models/time_series.py`

4. Introducir robustez inter-seed para el top tranche del HPO/calibration search.
Intencion:
- reducir optimizer's curse y sobrelectura del mejor run
Justificacion:
- el siguiente gran riesgo ya no es falta de HPO, sino optimismo de seleccion
Impacto esperado:
- alto
Costo:
- medio
Dependencia:
- `train_pd_model.py`, Optuna study policy

### P1

5. Promover benchmark serio del challenger monotono.
Intencion:
- aprovechar una mejora real de interpretabilidad/consistencia sin asumir que debe reemplazar al champion
Justificacion:
- monotonic violation rate del champion ~`0.292`; challenger `0.0`
- el challenger ya sale `promotable`
Impacto esperado:
- medio-alto
Costo:
- medio
Dependencia:
- `build_pd_challenger_artifacts.py`, `challenger_promotion_report.json`

6. Rehacer governance con materialidad, no solo sensibilidad estadistica.
Intencion:
- separar drift benigno de drift operacionalmente relevante
Justificacion:
- `PSI` y `score_psi` pasan, pero `KS/CvM/C2ST` disparan warnings muy fuertes
Impacto esperado:
- alto narrativamente
Costo:
- medio
Dependencia:
- `generate_governance_status.py`, `configs/mrm_policy.yaml`

7. Endurecer la evaluacion A/B.
Intencion:
- pasar de `no_regression` a evidencia economica mas defendible
Justificacion:
- uplift existe, significancia no
Impacto esperado:
- medio
Costo:
- medio
Dependencia:
- `simulate_ab_test.py`, `select_economic_portfolio_policy.py`

### P2

8. Evaluar GAM/EBM o stacking solo como challengers de interpretabilidad o research.

## 3.2 canonical_rebuild

### P0

1. Congelar narrativa unica de thresholds.
Intencion:
- evitar confundir threshold interno PD (`0.05`) con threshold operativo fairness (`0.35`)
Justificacion:
- hoy ambos existen y cumplen roles distintos
Impacto esperado:
- alto para tesis, demo y defensa metodologica
Costo:
- bajo
Dependencia:
- artifacts de threshold y fairness

2. Blindar que el carril canonico no reabra search.
Intencion:
- mantener reconstruccion barata, reproducible y audit-friendly
Justificacion:
- el libro favorece pipelines claros y evaluacion fuera de muestra, no search permanente
Impacto esperado:
- alto
Costo:
- bajo
Dependencia:
- `run_long_pipeline.py`, perfiles de corrida, docs

### P1

3. Agregar politica canonica de materialidad de drift.
4. Exportar resumen canonico de features, drivers y exclusiones del stable core.
5. Documentar explicitamente missingness, leakage guards y rol de conformal strict vs promotion gate.

## 3.3 insights_factory

### P1

1. Dejar causal/CATE como carril research formal.
Intencion:
- limpiar la narrativa: valioso como insight, no canonico todavia
Justificacion:
- ATE pequeno, IC cruza cero, refutaciones no disponibles, CATE portfolio empeora objetivo
Impacto esperado:
- alto en claridad
Costo:
- bajo
Dependencia:
- artifacts causales actuales

2. Usar insights_factory para anexos metodologicos del libro:
- proper scoring rules y calibration
- redundancia de features
- visualizaciones de drift material vs drift estadistico
- smoothing diagnostico en time series

### P2

3. Mantener RAPIDS como anexo tecnico reusable, no como pieza core.
Justificacion:
- algunos benchmarks muestran speedups fuertes, otros no
- el valor real es evidencia tecnica, no canonicidad

## 3.4 backlog reordenado por impacto

### P0

1. PD conformal estricto con selector de variantes
2. Time series intervals + exogenous future contract
3. Governance materiality policy
4. HPO/calibration robustness inter-seed
5. Claridad total de thresholds operativos vs internos

### P1

6. Monotonic challenger benchmark y ruta de promocion
7. A/B economico mas fuerte
8. Causal/CATE research lane formalizado
9. Redundancia de features con MI/mRMR/PhiK segun costo

### P2

10. GAM/EBM challenger
11. stacking research-only
12. anexos GPU y visualizaciones extra de EDA

## 4. Quick wins, mejoras tesis/paper y research-only

### Quick wins

1. Sacar `venn_abers` del selector oficial o marcarlo `research_only`.
2. Estandarizar narrativa de threshold `0.05` vs `0.35`.
3. Documentar por que `Kupiec/Christoffersen` no bloquean promocion operativa.
4. Agregar tabla de materialidad para top drift features.
5. Declarar causal/CATE como `insights_only` hasta nuevo aviso.

### Mejoras metodologicas defendibles para tesis/paper

1. selector formal de variantes conformal PD
2. agenda de intervalos time series con ACI / EnbPI / OnlineConformal
3. robustez inter-seed en HPO/calibracion
4. politica de materialidad de drift
5. benchmark monotono como challenger serio

### Ideas research utiles pero no canonicas

1. GAM/EBM challenger
2. stacking de PD
3. benchmarks GPU adicionales
4. smoothing diagnostico extendido

## 5. Conclusiones ejecutivas

1. El libro confirma que la direccion estrategica del repo es correcta. El proyecto ya esta bien sesgado hacia tabular ML fuerte, calibracion, validacion temporal, incertidumbre y decision.

2. El mayor upside ya no esta en subir unas milesimas de AUC. Esta en cerrar mejor:
- conformal PD
- intervalos de time series
- materialidad de drift
- semantica de thresholds
- evidencia causal/A-B

3. El mejor cambio nuevo sugerido por el libro y respaldado por el estado real del repo es doble:
- tratar intervalos como producto principal y no como subproducto
- reducir deuda metodologica de seleccion y governance, no perseguir complejidad de modelo por si misma

4. El capitulo 13 tambien aplica: no vale la pena meter deep learning en el camino canonico. Los cuellos de botella actuales son de validacion, incertidumbre, policy selection y narrativa regulatoria.
