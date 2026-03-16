# Auditoría integral de hardening, gates y protocolo paper-grade

Fecha: 2026-03-13

Baseline operativo de referencia:
- `champion-2026-03-12-mega-definitive`

Fuentes vivas principales auditadas:
- `configs/conformal_policy.yaml`
- `configs/fairness_policy.yaml`
- `configs/mrm_policy.yaml`
- `configs/run_profiles/champion_search_max.yaml`
- `models/conformal_policy_status.json`
- `models/time_series_status.json`
- `models/fairness_audit_status.json`
- `models/fairness_decision_policy.json`
- `models/governance_status.json`
- `models/champion_policy_selection_status.json`
- `models/paper_grade_protocol_status.json`
- `models/threshold_semantics.json`
- `models/champion_registry.json`
- `models/champion_search_bundle.json`
- `reports/storytelling_snapshot.json`
- `scripts/validate_conformal_policy.py`
- `scripts/run_comparison.py`
- `scripts/run_fairness_audit.py`
- `scripts/generate_governance_status.py`
- `scripts/forecast_default_rates.py`
- `scripts/generate_paper_grade_protocol.py`
- `scripts/build_champion_search_bundle.py`
- `scripts/update_champion_registry.py`
- `docs/RUNBOOK.md`
- `docs/backlog-13-03.md`
- `docs/backlog-papers-unified.md`
- `tests/test_docs/test_narrative_consistency.py`
- `tests/test_scripts/test_run_comparison.py`
- `tests/test_scripts/test_validate_conformal_policy.py`

## Resumen ejecutivo

Conclusión general:
- el proyecto ya tiene una arquitectura de hardening bastante madura;
- la separación entre `strict policy`, `promotion gate`, `paper-grade closure` y `research-only` existe de forma real en código;
- el problema principal ya no es ausencia de reglas, sino normalización final de contratos y narrativa.

Addendum de cierre ejecutado el mismo 2026-03-13:
- `run_tag` normalizado en artifacts refreshed post-hoc;
- `threshold_semantics` propagado a `champion_registry` y `champion_search_bundle`;
- fallback de `configs/fairness_policy.yaml` alineado a `0.35`;
- `storytelling_snapshot` refrescado con schema vigente;
- tests de coherencia semántica añadidos;
- dossier histórico de promoción marcado explícitamente como histórico.

Estado actual:
- `PD conformal` está correctamente separado en tres capas:
  - strict MRM: `overall_pass=false`
  - promotion operativo: `promotion_pass=false`
  - paper-grade closure: `methodological_justification_pass=true`
- `time_series` está correctamente separado entre:
  - forecast puntual oficial
  - intervalos `research_only`
- `A/B` usa una política consistente: `no_regression` bloquea, `significance` diagnostica.
- `governance` ya interpreta sensibilidad estadística como warning contextualizado, no como bloqueo bruto.

Hallazgo principal:
- las reglas centrales son razonables;
- lo que sigue abierto es la consistencia entre artifacts, bundle, registry y documentación histórica.

Clasificación global:
- contratos canónicos correctos: sí
- duplicados coherentes: sí, varios
- duplicados incoherentes: sí, algunos relevantes
- reglas ad hoc aún por formalizar o reducir: sí, pocas, pero importantes

## 1. Mapa de verdad y contratos vivos

| Artifact | Productor principal | Consumidores principales | Rol contractual | Significado de promoción |
|---|---|---|---|---|
| `models/conformal_policy_status.json` | `scripts/validate_conformal_policy.py` | `run_comparison`, `storytelling_snapshot`, Streamlit, paper-grade protocol | strict conformal policy + closure metadata | `overall_pass` es strict; no equivale a promoción |
| `models/conformal_variant_selection_status.json` | `scripts/benchmark_conformal_variants.py` | `storytelling_snapshot`, champion bundle, backlog técnico | selector de variante y `promotion_pass` | `promotion_pass` es promoción operativa relativa del carril conformal |
| `models/time_series_status.json` | `scripts/forecast_default_rates.py` | Streamlit, MLflow, paper-grade protocol, bundle, registry | contrato canónico TS | `interval_champion.promotable` define si el intervalo es oficial o `research_only` |
| `models/fairness_audit_status.json` | `scripts/run_fairness_audit.py` | governance, comparison, Streamlit, registry, storytelling | gate fairness y threshold primario | `overall_pass=true` significa que la auditoría canónica pasa con threshold operativo |
| `models/fairness_decision_policy.json` | `scripts/run_fairness_audit.py` | threshold semantics, fairness audit, docs/UI | threshold operativo global de decisión | policy operativa de aprobación, no threshold PD interno |
| `models/governance_status.json` | `scripts/generate_governance_status.py` | comparison, registry, protocol, Streamlit | governance/MRM gate | `overall_pass=true` con warnings contextualizados |
| `models/threshold_semantics.json` | `train_pd_model.py`, `refresh_pd_calibration_artifacts.py`, `run_fairness_audit.py` | governance, Streamlit, storytelling, docs | semántica canónica de thresholds | separa explícitamente `0.05` interno y `0.35` operativo |
| `models/champion_registry.json` | `scripts/update_champion_registry.py` | lectura rápida de champion, reporting | resumen ejecutivo del champion | hoy resume estado, pero no es la fuente más rica de semántica |
| `models/champion_search_bundle.json` | `scripts/build_champion_search_bundle.py` | futuras corridas `champion_search` y revisión | bundle promotion-ready del carril de búsqueda | siembra artifacts y criterios, no promueve por sí mismo |
| `models/paper_grade_protocol_status.json` | `scripts/generate_paper_grade_protocol.py` | backlog final, storytelling técnico, revisión paper-grade | cierre metodológico final | documenta si un frente quedó `promoted`, `research_only` o cerrado por justificación |
| `reports/storytelling_snapshot.json` | `scripts/export_storytelling_snapshot.py` | demos, defensa, narrativa consolidada | snapshot narrativo congelado | publica interpretación ya derivada de los artifacts |

Jerarquía de verdad recomendada:
1. artifacts `models/*_status.json`, `models/threshold_semantics.json`, `reports/storytelling_snapshot.json`
2. scripts que los generan
3. configs de policy
4. bundle/registry
5. documentación
6. dossiers y reportes históricos

## 2. Cómo está realmente gobernado hoy

### 2.1 Conformal

Hay tres políticas distintas y no deben mezclarse:

1. Política strict / MRM:
- fuente: `configs/conformal_policy.yaml` + `scripts/validate_conformal_policy.py`
- salida: `models/conformal_policy_status.json`
- blocking rules:
  - cobertura
  - subgroup coverage
  - width
  - `winkler_90` con banda compensada
  - `Kupiec/Christoffersen`

2. Política de promoción operativa:
- fuente: `scripts/run_comparison.py`
- salida: `reports/run_comparisons/<run_tag>/comparison.json`
- blocking rules:
  - cobertura relativa vs baseline
  - subgroup coverage relativa
  - `winkler_90` relativo
  - `critical_alerts`
- `Kupiec/Christoffersen` son warning diagnóstico, no bloqueo

3. Política paper-grade:
- fuente: `scripts/generate_paper_grade_protocol.py`
- salida: `models/paper_grade_protocol_status.json`
- cierre aceptado si:
  - `strict_overall_pass=true`, o
  - `methodological_justification_pass=true` sin fallos no estadísticos, o
  - existe cierre válido por sensibilidad

Conclusión:
- la triple semántica es correcta y defendible;
- el riesgo es narrativo, no metodológico.

### 2.2 Fairness y thresholds

Hoy existen tres capas:
- `models/decision_threshold.json.selected_threshold = 0.05`
- `models/fairness_audit_status.json.primary_threshold = 0.35`
- `models/fairness_decision_policy.json.global_threshold = 0.35`

La separación correcta está formalizada en `models/threshold_semantics.json`.

Conclusión:
- la separación ya es canónica;
- el proyecto no tiene un problema de lógica aquí;
- sí tiene un problema potencial de lectura ambigua en artifacts secundarios.

### 2.3 Time series

Política actual:
- `point_champion.promotable=true`
- `interval_champion.promotable=false`
- `final_interval_decision.status=research_only`

Conclusión:
- la semántica es correcta;
- no significa “pipeline roto”;
- significa “forecast puntual oficial, intervalos aún diagnósticos”.

### 2.4 A/B

Política real:
- blocking gate: `no_regression`
- significance: diagnóstico

Conclusión:
- la implementación es coherente con el discurso actual;
- no hay contradicción metodológica seria;
- sí hay que seguir dejando explícito que `significant=false` no invalida un `PASS` operativo bajo `no_regression`.

### 2.5 Governance

Política real:
- `overall_pass=true`
- warnings de drift estadístico pueden convivir con pass si materialidad, PSI, score drift y performance siguen aceptables

Conclusión:
- el criterio actual ya es de materialidad, no de p-values desnudos;
- esto es correcto para paper-grade y MRM pragmático.

## 3. Hallazgos clasificados

### 3.1 Contrato canónico correcto

- `threshold_semantics.json` ya es la mejor fuente de verdad para `0.05` interno vs `0.35` operativo.
- `validate_conformal_policy.py`, `run_comparison.py` y `generate_paper_grade_protocol.py` sí representan tres capas distintas y compatibles.
- `time_series_status.json` expresa bien `research_only` como estado del intervalo y no como falla del pipeline completo.
- `simulate_ab_test.py` ya trata `significance` como diagnóstico y `no_regression` como gate operativo.

### 3.2 Duplicado coherente

- `fairness_audit_status.primary_threshold` y `fairness_decision_policy.global_threshold` hoy apuntan al mismo threshold operativo `0.35`.
- `reports/storytelling_snapshot.json` replica correctamente la semántica desde `threshold_semantics.json`.
- `paper_grade_protocol_status.json` resume reglas existentes sin redefinirlas.

### 3.3 Duplicado incoherente

#### Hallazgo P0-1
- `configs/fairness_policy.yaml` conserva `prediction_threshold: 0.50`, mientras el contrato operativo vigente es `0.35`.
- En código esto no rompe porque `threshold_policy.use_artifact=true`, pero sí deja un fallback engañoso.
- Clasificación: `duplicado incoherente`.

#### Hallazgo P0-2
- `models/champion_registry.json` y `models/champion_search_bundle.json` muestran `pd.decision_threshold.selected_threshold = 0.05` sin exponer al mismo nivel la semántica operativa de `0.35`.
- Esto favorece lecturas equivocadas de negocio.
- Clasificación: `duplicado incoherente`.

#### Hallazgo P0-3
- `models/champion_search_bundle.json` tiene `run_tag = untracked` aunque agrega artifacts derivados del baseline oficial.
- `models/conformal_policy_status.json` y `models/governance_status.json` también están hoy en `run_tag = untracked`.
- Esto no invalida la lógica, pero sí debilita la coherencia de artifacts post-hoc.
- Clasificación: `contract incoherente`.

### 3.4 Regla ad hoc que debe formalizarse o eliminarse

#### Hallazgo P1-1
- `reports/storytelling_snapshot.json` sigue con `SCHEMA_VERSION = 2026-02-26.1` aunque hoy expresa semánticas de 2026-03-13.
- No rompe nada, pero comunica mal madurez contractual.

#### Hallazgo P1-2
- `run_comparison` verifica metadata crítica, pero no verifica coherencia de semántica entre:
  - `threshold_semantics`
  - `fairness_decision_policy`
  - `storytelling_snapshot`
  - `paper_grade_protocol_status`
- Hoy solo verifica presencia, `run_tag` y timestamps para artifacts críticos del core.

#### Hallazgo P1-3
- Existen documentos históricos que todavía comunican `conformal_promotion_pass=true` como si fuera el estado actual.
- Ejemplo: `docs/PROMOTION_DOSSIER_2026-03-01.md`.
- El contenido histórico es válido como snapshot, no como verdad vigente.

## 4. Inconsistencias narrativas

### Críticas

- `run_tag` no coherente entre artifacts canónicos refreshed post-hoc:
  - `models/conformal_policy_status.json`
  - `models/governance_status.json`
  - `models/champion_search_bundle.json`
- Riesgo:
  - dificulta defender qué artifacts pertenecen realmente al baseline oficial versus a refreshes posteriores.

### Confusas pero no bloqueantes

- fallback `0.50` en `configs/fairness_policy.yaml`
- `champion_registry` y `champion_search_bundle` mostrando `0.05` sin semántica explícita
- `storytelling_snapshot` con schema version vieja
- `paper_grade_protocol_status` considera `time_series` como cerrado si la decisión está documentada (`promoted` o `research_only`), lo cual es correcto pero puede malinterpretarse si no se explica.

### Editoriales

- algunos backlogs y dossiers históricos siguen mezclando:
  - baseline de mega run
  - estado vigente post-P0
- esto debe marcarse siempre como histórico o vigente, nunca dejarse implícito.

## 5. Lo que falta para dejarlo óptimo

### P0 contract fix

1. Normalizar `run_tag` de los artifacts refreshed post-hoc:
- `conformal_policy_status.json`
- `governance_status.json`
- `champion_search_bundle.json`

2. Llevar `threshold_semantics` al mismo nivel de `champion_registry` y `champion_search_bundle`.
- No basta con mostrar `selected_threshold=0.05`;
- deben mostrar también el threshold operativo `0.35` o referenciar explícitamente `threshold_semantics`.

3. Reetiquetar en docs históricos cualquier claim vigente de promoción que ya no aplique.

### P1 policy normalization

1. Cambiar `configs/fairness_policy.yaml` para que el fallback narrativo no choque con la policy vigente.
- opción recomendada:
  - dejar `prediction_threshold: 0.35`, o
  - marcarlo explícitamente como fallback técnico no operativo.

2. Expandir la noción de `artifact_coherence` con una segunda capa opcional:
- `semantic_coherence`
- checks mínimos:
  - threshold operativo consistente entre `threshold_semantics`, `fairness_audit_status` y `fairness_decision_policy`
  - status de TS consistente entre `time_series_status`, `paper_grade_protocol_status` y `storytelling_snapshot`
  - conformal closure consistente entre `conformal_policy_status`, `comparison.json` y `paper_grade_protocol_status`

### P1 test hardening

Agregar tests de:
- coherencia de thresholds entre artifacts
- coherencia de `run_tag` entre artifacts refreshed
- bundle/registry no ambiguos sobre thresholds
- `research_only` vs `promoted` en `time_series`
- separación `strict` / `promotion` / `paper-grade`

### P2 editorial cleanup

- banner histórico en dossiers viejos
- unificar wording de:
  - `promotion_pass`
  - `strict_overall_pass`
  - `methodological_justification_pass`
  - `research_only`
- refrescar schema version de snapshots narrativos

## 6. Matriz de policy declarada vs ejecutada vs publicada

| Tema | Policy declarada | Policy ejecutada | Policy publicada | Estado |
|---|---|---|---|---|
| Conformal strict | `configs/conformal_policy.yaml` | `validate_conformal_policy.py` | `conformal_policy_status.json`, `RUNBOOK` | coherente |
| Conformal promotion | comparación relativa vs baseline | `run_comparison.py` | `comparison.json`, `storytelling_snapshot.json` | coherente |
| Conformal paper-grade | cierre metodológico válido | `generate_paper_grade_protocol.py` | `paper_grade_protocol_status.json`, backlog | coherente |
| Fairness threshold operativo | artifact-driven | `run_fairness_audit.py` | `fairness_audit_status.json`, `fairness_decision_policy.json`, `threshold_semantics.json` | coherente |
| Fairness fallback YAML | `0.50` | casi nunca usado | puede confundir | incoherente |
| Threshold en registry/bundle | no declarado con semántica completa | expone `0.05` | puede interpretarse como threshold de negocio | incoherente |
| Time series closure | `research_only` aceptable | `forecast_default_rates.py` + protocol | docs/backlogs/Runbook | coherente |
| A/B | `no_regression` | `simulate_ab_test.py`, `run_comparison.py` | backlog/protocol/storytelling | coherente |
| Governance | materialidad + warnings | `generate_governance_status.py` | `governance_status.json`, docs | coherente |

## 7. Juicio final

El proyecto no está en una situación de “reglas mal diseñadas”. Está en una situación de:
- ruleset central ya bastante bueno;
- artifacts secundarios y narrativa aún no completamente normalizados.

Mi juicio:
- el núcleo metodológico ya está en nivel paper-grade;
- lo que falta para dejarlo óptimo es `contract hardening`, no rediseño de policy.

Prioridad real restante:
1. coherencia de `run_tag`
2. semántica de thresholds en registry/bundle
3. fallback fairness YAML
4. tests de coherencia contractual
5. limpieza editorial de documentos históricos

Si estos cinco frentes se cierran, el stack de gates/hardening quedaría esencialmente limpio para la corrida final paper-grade.
