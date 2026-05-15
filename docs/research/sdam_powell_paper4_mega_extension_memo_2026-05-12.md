# Powell SDAM → Paper 4 Mega Extension Memo - 2026-05-12

## Objetivo

Este memo analiza el libro *Sequential Decisions Analytics and Modeling*,
2nd ed., Warren B. Powell, febrero 2026, a partir del markdown optimizado en:

`/mnt/c/Users/carlos/Documents/Claude Code/lending-club-risk-project/reports/powell_kindle_sdam_2nd_ed_feb_9_2026.pdf-inspector.md`

La pregunta es cómo usar sus conceptos, fórmulas y ejemplos para fortalecer el
Paper 4 / mega extensión:

`book/chapters/19-paper-mega-extension/`

La lectura corta: Powell le da a Paper 4 una columna vertebral formal para
dejar de ser "CRPTO + IFRS9 + CATE + online + fairness" como lista de módulos,
y convertirse en un problema secuencial completo con estado, decisión,
información exógena, transición, política, evaluación por sample paths y
taxonomía explícita de políticas.

## Síntesis ejecutiva

Powell complementa la mega extensión en cinco frentes:

1. **Formalización UMF**. Paper 4 debería adoptar el Universal Modeling
   Framework como lenguaje principal: estado `S_t`, decisión `x_t`, información
   exógena `W_{t+1}`, transición `S^M`, objetivo y política `X^\pi(S_t)`.
   Esto convierte la extensión en sequential decision analytics, no solo en
   ML + optimización.

2. **Separación model-first, solve-second**. El libro insiste en modelar el
   sistema antes de elegir solución. Esa regla calza perfecto con los
   guardrails actuales: diagnóstico, selector y solver deben ser cosas
   diferentes.

3. **Taxonomía de políticas**. Las cuatro clases de Powell organizan todo Paper
   4:
   - PFA: reglas y thresholds de comité.
   - CFA: CRPTO/IFRS9-aware LP, OCE/CVaR y satisficing como optimizadores
     parametrizados.
   - VFA: valor futuro de capital, provisión, stage capacity y oportunidades de
     originación.
   - DLA: rolling/receding horizon con forecasts macro, ECL, defaults,
     funding demand y cobertura.

4. **Online vs offline objectives**. Paper 4 debe declarar si está optimizando
   recompensa acumulada en operación o recompensa final tras entrenamiento. Esta
   distinción evita el overclaim de llamar "online" a un replay offline.

5. **Estado post-decisión**. La pieza más útil técnicamente: después de aprobar
   préstamos, pero antes de observar defaults, prepagos y drift, existe un
   `S_t^x`. Esto permite ligar el LP one-shot actual con multi-period CRPTO,
   VFA y DLA sin romper el bound del Paper Estrella.

## Conceptos centrales del libro y traducción a Paper 4

### 1. Universal Modeling Framework

Powell modela cualquier problema secuencial como:

```text
(S_0, x_0, W_1, S_1, x_1, W_2, ..., S_t, x_t, W_{t+1}, ..., S_T)
```

con transición:

```text
S_{t+1} = S^M(S_t, x_t, W_{t+1})
```

y objetivo típico:

```text
max_pi E[ sum_t C_t(S_t, X^pi(S_t)) | S_0 ].
```

**Traducción a crédito**:

```text
S_t = (R_t, I_t, B_t)

R_t = recursos / exposiciones / presupuesto / capital / stage balances /
      funded book / remaining risk capacity.

I_t = información observable actual: loan applications, PD calibrada,
      intervalos conformales, macro actual, escenario IFRS9, period, grade,
      fairness slice, constraints operativas.

B_t = creencias o estimaciones: quantile conformal vigente, forecasts macro,
      ECL esperado, distribución de LGD/EAD, drift, source/regime beliefs,
      CATE o uplift si el gate causal pasa.
```

La decisión puede ser:

```text
x_t = approvals / funded-set vector / pricing or term intervention /
      alpha-gamma policy / stage-SICR rule / recalibration trigger.
```

La información posterior:

```text
W_{t+1} = observed defaults, repayments, recoveries, prepayments,
          realized macro, new applications, residuals, miscoverage,
          stage transitions, updated forecasts.
```

La transición:

```text
S_{t+1} = update(
  S_t,
  x_t,
  observed defaults,
  repayments,
  ECL/stage update,
  conformal recalibration,
  drift/regime update,
  remaining budget/capital
)
```

Clave para causalidad: en crédito, `W_{t+1}` puede depender de `x_t`. Solo
observamos muchos outcomes si el préstamo fue aprobado/fundado. Esa lectura de
Powell justifica tratar causal/CATE como un problema de información
state/action-dependent, no como un apéndice de predicción.

### 2. Los tres tipos de estado

Powell separa:

- `R_t`: physical/resource state.
- `I_t`: information state.
- `B_t`: belief state.

Paper 4 ya tiene piezas, pero no las nombra así. Conviene hacerlo explícito.

| Powell | En Paper 4 | Ejemplos concretos |
|---|---|---|
| Physical state `R_t` | balance de recursos y exposiciones | budget, outstanding loans, capital, EAD, Stage 1/2/3 book, funded set |
| Information state `I_t` | información observable para decidir | PD, conformal interval, grade, decile, current macro, scenario, fairness proxy |
| Belief state `B_t` | incertidumbre aprendida/actualizable | conformal quantile, forecast distribution, drift/regime posterior, CATE posterior, LGD/ECL beliefs |

Esto también evita un error frecuente: decir que el estado es solo "la cartera"
o solo "la PD". Para Paper 4, el estado tiene recursos, información y creencias.

### 3. Estado pre-decisión y post-decisión

Powell usa `S_t^x` para el estado inmediatamente después de decidir y antes de
que llegue nueva información. En Paper 4:

```text
S_t       = solicitudes + PD + CP + ECL + budget + macro + beliefs
x_t       = funded-set / policy action
S_t^x     = exposures committed + remaining budget + stage mix +
            post-funding conformal/robust status
W_{t+1}   = defaults + payments + macro + residuals + new information
S_{t+1}   = updated book + updated beliefs + next applications
```

Esto sugiere una política VFA:

```text
X^VFA(S_t) =
  argmax_{x in X_t} C_t(S_t, x) + Vbar_t(S_t^x)
```

donde `Vbar_t(S_t^x)` aproxima el valor futuro de conservar capital, no
sobrecargar Stage 2, mantener cobertura o preservar capacidad para originaciones
futuras.

### 4. Cuatro clases de políticas

Powell organiza todas las políticas en cuatro familias. Paper 4 debería incluir
una tabla así:

| Clase | Forma | Paper 4 |
|---|---|---|
| PFA | función directa `x = f(S_t | theta)` | PD threshold, alpha schedule, approval rules, committee thresholds |
| CFA | optimización parametrizada | CRPTO LP, IFRS9-aware selector, OCE/CVaR-constrained solver, robust satisficing |
| VFA | decisión actual + valor futuro aproximado | value of remaining capital, value of Stage 2 capacity, future default/regime value |
| DLA | modelo de horizonte futuro aproximado | rolling horizon policy with macro/ECL/default forecasts and reserve buffers |

El Paper Estrella actual es principalmente **CFA one-shot auditable**. Paper 4
sería **DLA/CFA/VFA híbrido** si entra en multi-period portfolio.

### 5. Online vs offline objectives

Powell distingue:

```text
Online/cumulative reward:
max_theta E[ sum_t C_t(S_t, X^pi(S_t | theta)) | S_0 ]

Offline/final reward:
max_theta E[ F(x^{pi,N}(theta), W^c) ]
```

En Paper 4:

- **Offline/final reward**: entrenar/calibrar/selectar policy usando artifacts
  congelados, luego evaluar en OOT o confirmation.
- **Online/cumulative reward**: operar mes a mes, tomando decisiones mientras
  aprende/recalibra, acumulando retorno neto, ECL, coverage regret y tail risk.

Recomendación editorial: toda claim temporal debe declarar cuál objetivo usa.
Un monthly replay sin actualización real es offline evaluation over historical
sample paths; no es online learning salvo que actualice estado/policy
forward-only.

### 6. Evaluación por sample paths

Powell reemplaza expectativas imposibles por sample paths:

```text
Fhat^pi(omega | S_0) = sum_t C_t(S_t(omega), X^pi(S_t(omega)))
```

y luego promedia sobre `omega^1, ..., omega^N`.

Aplicación inmediata:

- `monthly_policy_replay.parquet` debería tratar cada periodo, bootstrap o
  stress como sample path.
- Comparar policies con los mismos sample paths reduce varianza y hace más
  defendible el ranking diagnóstico.
- El MVP de Paper 4 debería reportar intervalos/confidence bands para
  diferencias de policy, no solo números puntuales.

### 7. Uncertainty styles

El libro separa variabilidad fina, shifts, bursts, spikes, eventos espaciales,
sistémicos, raros y contingencias. Para Paper 4 esto da un mapa de stress:

| Estilo de incertidumbre | Crédito / Paper 4 |
|---|---|
| fine-grained variability | defaults normales, repayments, application flow |
| shifts | cambios macro, underwriting drift, post-2019 regime |
| bursts | ola de defaults por segmento |
| spikes | shock de unemployment, LGD/recovery shock |
| spatial events | geografía / estado / región |
| systemic events | recesión, crisis crediticia, pandemia |
| rare events | extreme tail loss, severe scenario |
| contingencies | escenario sin historia suficiente |

Esto fortalece la justificación de MDCP, stress y tail-risk-aware CRPTO.

## Lectura capítulo por capítulo

| Capítulo Powell | Concepto útil | Cómo complementa Paper 4 |
|---|---|---|
| 1. Modeling sequential decision problems | UMF, `S_t`, `x_t`, `W_{t+1}`, `S^M`, objetivo, policy testing | Nueva sección metodológica: "Sequential credit decision model" |
| 2. Asset selling | optimal stopping, PFAs, sample paths, CIs, AR processes, correlated basket | analogía para stop/recalibrate/fund-now-vs-wait; common random numbers para comparar policies |
| 3. Adaptive market planning | newsvendor, stochastic gradient, stepsize como decisión, cumulative vs final reward, censored demand | approval/funding como asignación bajo demanda incierta; outcomes censurados si no se financia |
| 4. Diabetes/bandits | belief state, UCB, interval estimation, Thompson, contextual bandit | CATE/causal layer e interventions; exploration vs exploitation en crédito |
| 5. Static stochastic shortest path | Bellman, post-decision state, ADP | formalizar `S_t^x` del funded set; evitar curse of dimensionality |
| 6. Dynamic shortest path | deterministic lookahead, percentile cost, rolling/receding horizon | DLA mensual para crédito con percentiles de ECL/default/macro |
| 7. Applications revisited | taxonomía PFA/CFA/VFA/DLA, online/offline, SPSA, derivative-free search | organizar todo Paper 4 y declarar objetivo de entrenamiento/evaluación |
| 8. Energy storage I | heavy tails, jump diffusion, empirical quantiles, transformed normal, ADP cautions | tail risk, OCE/CVaR, LGD/default spikes, evitar VFA/RL overclaim |
| 9. Energy storage II | rolling forecasts, martingale model of forecast evolution, GPR, hidden-state Markov, parameterized lookahead | macro/default/ECL forecasts; forecast multipliers by horizon; regime/crossing-time models |
| 10. Two-agent newsvendor | private information, decentralized objectives, policies interacting | risk committee vs business line vs accounting/MRM; multi-agent governance |
| 11. Beer game | information delays, backlogs, bullwhip, anchor-adjustment, multiagent lookahead | lending cycle feedback: approval changes future applicant mix, capital, monitoring and model updates |
| 12. Ad-click optimization | logistic response, Bayesian sampled parameter, excitation, value of information | field trial / counterfactual deployment / causal exploration; value of approving marginal loans |
| 13. Blood management | multiattribute LP, myopic vs VFA, separable piecewise-linear future value | credit portfolio as multiattribute resource allocation; value of capacity by grade/stage/period |
| 14. Clinical trials | stopping, enrollment, active learning, lookahead Models A/B/C | champion promotion/recalibration/stopping; causal proof package and policy-within-policy design |

## Direct additions recommended for Paper 4

### A. Add a Powell-style UMF subsection

Suggested title:

```text
Sequential Decision Model for IFRS9-Aware CRPTO
```

Minimal content:

```text
S_t = (R_t, I_t, B_t)
x_t in X_t(S_t)
W_{t+1} = W_{t+1}(S_t, x_t)
S_{t+1} = S^M(S_t, x_t, W_{t+1})
X^pi(S_t | theta) = policy class with tunable parameters
```

Then list variables as a table. This is stronger than the current text-only
state sketch.

### B. Reclassify current and future modules by policy class

Add a table:

| Module | Policy class | Status |
|---|---|---|
| Paper Estrella CRPTO | CFA | current one-shot champion |
| robust region thresholds | PFA/CFA diagnostic | implemented |
| IFRS9-aware selector | CFA | next MVP |
| satisficing screen | PFA/CFA | diagnostic now, selector later |
| OCE/CVaR constrained search | CFA | future solver |
| monthly replay | evaluation protocol | MVP |
| online conformal | transition/belief update | future |
| multi-period CRPTO | DLA/CFA or VFA/CFA hybrid | future |
| CATE policy value | belief/value layer | gated future |

### C. Add pre/post-decision notation

This is the most valuable formula-level complement:

```text
S_t^x = S^{M,x}(S_t, x_t)
```

with:

```text
C_t(S_t, x_t)
Vbar_t(S_t^x)
```

and:

```text
X(S_t) = argmax_x C_t(S_t, x) + Vbar_t(S_t^x)
```

Even if Paper 4 MVP does not estimate `Vbar`, this notation shows exactly what
is missing before claiming multi-period optimality.

### D. Define DLA for rolling credit policy

Suggested rolling-horizon objective:

```text
X_t^DLA(S_t | theta) =
  first-period decision from

  max_{x_t, ..., x_{t+H}}
    sum_{h=0}^H [
      robust_return_{t+h}(x_{t+h})
      - lambda_ecl ECL_{t+h}(x_{t+h})
      - lambda_tail CVaR_{t+h}(x_{t+h})
      - lambda_fair disparity_{t+h}(x_{t+h})
    ]
```

subject to forecast-adjusted gates:

```text
coverage_{t+h,g} >= target_g
Stage2Share_{t+h} <= limit_stage(theta)
Capital_{t+h} <= limit_capital
CVaRLoss_{t+h} <= limit_tail
Budget_{t+h} <= budget_{t+h}
```

Powell's energy-storage chapters justify forecast multipliers by horizon:

```text
tilde_f^X_{t,t+h} = theta^X_h f^X_{t,t+h}
```

For Paper 4, `X` can be default rate, prepayment, LGD, application volume,
macro overlay or ECL.

### E. Turn OCE/CVaR and satisficing into Powell-style CFAs

Current A12/A13 are diagnostic. Powell's CFA language makes the next step clean:

```text
X^CFA(S_t | theta) =
  argmax_x robust_return(x)
          - lambda_ecl ECL(x)
          - lambda_tail CVaR_loss(x)
```

or:

```text
argmax_x robust_return(x)
subject to CVaR_loss(x) <= theta_tail
           Stage2Share(x) <= theta_stage
           Gamma_CP(x) <= theta_gamma
```

The parameters `theta` are not "magic weights"; they are policy parameters that
must be tuned/evaluated by sample paths.

### F. Use common random numbers for policy comparison

Powell's asset-selling evaluation recommends comparing policies on the same
sample paths. For Paper 4:

- Use identical months/stress paths/bootstrap replicates for all policies.
- Report differences policy A minus policy B.
- Compute standard error of the paired difference.

This is low-cost and high-value for the MVP.

### G. Reframe CATE as state/action-dependent information

Causal CRPTO should be framed as:

```text
W_{t+1}(S_t, x_t)
```

because funding/approval/intervention changes both observed outcomes and future
information. This supports the current gate: no central causal claim without
identification, overlap, sensitivity and policy value.

## Proposed Paper 4 structure after Powell integration

1. **Problem framing**
   - metrics, decisions, uncertainties.
   - explicitly separate accounting, robust-return, coverage, tail, fairness,
     causal and temporal metrics.

2. **Sequential decision model**
   - `S_t = (R_t, I_t, B_t)`.
   - `x_t`, `W_{t+1}`, `S^M`, `C_t`.
   - state/action-dependent observation issue.

3. **Policy classes**
   - PFA/CFA/VFA/DLA taxonomy.
   - current CRPTO as CFA one-shot.
   - Paper 4 extension path as CFA selector, then DLA/VFA solver.

4. **Evaluation objectives**
   - offline final reward vs online cumulative reward.
   - sample path evaluation.
   - common-random-path policy comparisons.

5. **MVP diagnostic pack**
   - policy × ECL × scenario.
   - net return after ECL.
   - tail/OCE/CVaR.
   - satisficing.
   - monthly replay.

6. **Sequential extension**
   - post-decision state.
   - rolling horizon DLA.
   - online conformal coverage regret.
   - VFA value of capital/stage capacity.

7. **Gated future layers**
   - MDCP.
   - fairness constraints.
   - CATE/causal policy value.
   - external validation.

## Implementation roadmap influenced by Powell

### Phase 0: Documentation and modeling, no new champion

Create:

```text
docs/research/paper4_sequential_decision_model.md
```

or add a Quarto page:

```text
book/chapters/19-paper-mega-extension/19f-sequential-decision-framework.qmd
```

Contents:

- five UMF elements for Paper 4;
- state decomposition `R/I/B`;
- pre/post-decision state;
- policy-class table;
- online/offline objective statement.

Acceptance criteria:

- no new metrics;
- no champion change;
- only formalizes the problem.

### Phase 1: Powell-style MVP evaluation

Extend planned artifacts:

```text
paper4_policy_sample_path_eval.parquet
paper4_policy_pairwise_differences.csv
paper4_policy_class_registry.csv
paper4_post_decision_state_schema.json
```

Minimal columns:

```text
policy_id
sample_path_id
period
scenario
segment
robust_return
ecl
stage2_share
cvar_loss
oce_loss
coverage
miscoverage_count
gamma_cp
V
fairness_gap_proxy
capital_used
budget_remaining
```

### Phase 2: CFA selector

Use frozen policies first:

```text
Score(policy) =
  robust_return
  - lambda_ecl * ECL
  - lambda_tail * CVaR_loss
  - lambda_prov * adverse_provision
  - lambda_fair * disparity_penalty
```

subject to current conformal and governance gates.

This remains a selector over existing policies, not a new solver.

### Phase 3: DLA prototype

Build a rolling-horizon LP using forecasts/scenarios:

```text
horizon H = 3, 6, 12 months
forecast inputs = application volume, default rate, LGD, macro, ECL
theta = horizon-specific forecast multipliers or reserve buffers
```

Report it as prototype until:

- coverage by horizon passes;
- forward-only replay is respected;
- DVC/MLflow lineage exists;
- no leakage from future periods.

### Phase 4: VFA/ADP only if structure is exploitable

Powell is cautious about ADP/RL. Paper 4 should only use VFA if the value
function has clear structure, for example:

- separable value by grade/stage/month;
- monotonic value of remaining capital;
- concavity in budget/exposure capacity;
- piecewise-linear value of Stage 2 headroom.

Avoid deep RL as a central claim unless the paper is explicitly about that
method and has serious benchmarks.

## The strongest conceptual bridge

The cleanest Powell-inspired claim is:

> Paper 4 extends CRPTO from a one-shot cost function approximation into a
> sequential decision architecture where robust conformal uncertainty,
> accounting state, tail risk, committee thresholds and future opportunity value
> are modeled as a unified state-transition-policy system.

This is stronger than saying "we add IFRS9 and CATE." It says the class of
problem changed.

## What Powell warns us not to overclaim

- Solving a deterministic lookahead model optimally is not an optimal policy for
  the stochastic base model.
- A VFA/RL approximation is not automatically credible; it needs structure,
  tuning and empirical validation.
- A forecast in a lookahead model is often a latent simplification, not a full
  uncertainty model.
- Policy parameters depend on `S_0`; if the data/regime changes, old parameters
  may not remain optimal.
- Online learning and offline final reward are different objectives.
- Diagnostics are not selectors unless they change the policy through declared
  parameters, objectives or constraints.

These warnings match the current Paper 4 guardrails almost exactly.

## Concrete edits to consider in the Quarto chapter

1. In `19c-integrated-architecture.qmd`, add a subsection after "Estado y
   transición":

```text
Sequential Decision Analytics Formulation
```

with UMF, `R/I/B`, `S_t^x` and sample-path evaluation.

2. In `19d-implementation-roadmap.qmd`, add Phase 0.25:

```text
Powell/UMF formal model
```

Output:

```text
paper4_policy_class_registry.csv
paper4_post_decision_state_schema.json
```

3. In `19a-proposal-and-scope.qmd`, revise the thesis line from only "policy
crediticia moderna" to "sequential decision policy" with state-transition
language.

4. In `19e-insights-factory-integration.qmd`, require every insight to map not
only to state/gate/objective/constraint/selector, but specifically to one of:

```text
R_t, I_t, B_t, x_t, W_{t+1}, S^M, C_t, X^pi
```

That mapping will make the Insights Factory truly end-to-end.

## Bottom line

Powell's book does not merely add citations. It supplies the grammar of Paper 4.

The mega extension should be framed as:

```text
Sequential IFRS9-Aware Conformal Robust Credit Decision Analytics
```

with CRPTO as the current one-shot CFA base, IFRS9/tail/satisficing as the MVP
selector layer, and online conformal/multi-period/CATE/MDCP as gated sequential
extensions.

The next safest action is to add a Powell/UMF framework page to Chapter 19
before implementing new solvers. That gives the whole extension a rigorous
decision-analytic spine while preserving the Paper Estrella champion freeze.
