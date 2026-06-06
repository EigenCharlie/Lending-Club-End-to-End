<!-- cspell:disable -->
<!-- markdownlint-disable -->

# Backlog Vivo: Paper 2 + Paper 4

Fecha de curaduría: 2026-06-06

Este documento reemplaza el backlog unificado de marzo. La superficie local
activa del repositorio queda reducida a:

- **Paper 2**: IFRS9 end-to-end con incertidumbre conformal.
- **Paper 4**: Sequential Credit Decision Analytics Living Lab.

CRPTO/Paper IJDS vive autocontenido en
`C:/Users/carlos/Documents/Paper_CRPTO`. Este repositorio puede conservar
artefactos congelados que Paper 4 usa como referencia, pero no debe volver a
tratar CRPTO/Paper Estrella como paper local activo.

## Retirado

| Frente | Decisión | Qué se conserva |
|---|---|---|
| Paper 3 / Mondrian | Retirado como paper separado | La capa Mondrian queda en @sec-mondrian, backtesting, MRM y Paper 4 |
| Paper GPU | Retirado como paper y como capítulo avanzado | Solo evidencia histórica de infraestructura si ya existe en `reports/history` |
| Paper Quantum | Retirado | Ningún claim local; no había experimento ejecutado defendible |
| Paper Estrella local | Retirado del libro grande | Referencia congelada para Paper 4; desarrollo real en `Paper_CRPTO` |

## Paper 2

Estado actual:

- Mantener el capítulo 15 como superficie editorial oficial.
- Usar la evidencia de IFRS9, ECL, escenarios, SICR conformal, survival y
  sensibilidad como paquete principal.
- No abrir líneas contractuales IFRS9 completas sin datos nuevos de pagos,
  DPD, EAD paths y macro escenarios formales.

Pendiente real:

1. compactar la narrativa para que sea paper-first;
2. fortalecer threats to validity con límites de proxy IFRS9;
3. verificar que tablas y figuras citadas sigan sincronizadas con artefactos;
4. decidir si Paper 2 permanece independiente o se convierte en componente de
   Paper 4 cuando el living lab madure.

## Paper 4

Estado actual:

- Paper 4 queda como laboratorio vivo y superconjunto experimental.
- La superficie oficial en Quarto debe ser compacta; la bitácora viva queda en
  `reports/paper_material/paper4/notes/paper4_living_lab_notebook.md`.
- Las olas históricas se mantienen archivadas solo cuando tienen manifest y
  guardrail explícito.

Frentes abiertos con valor:

1. online/source conformal con boundaries fuertes;
2. DFL/SPO/PyEPO como comparador, no como reemplazo del champion congelado;
3. CVaR/OCE y tail-risk como appendix o diagnóstico;
4. IFRS9/SICR proxy cuando cambie una tabla o claim de governance;
5. fairness/source governance como proxy explícito, no fair-lending legal;
6. multi-period/DLA/SDAM como framing y simulación, no Bellman exacto.

## Regla De Priorización

Solo se ejecutan experimentos si tienen:

- claim target;
- evidence gate;
- artifact sink;
- stop rule;
- destino editorial concreto en Paper 2, Paper 4 o el proyecto externo CRPTO.

## Stop Rule

No reabrir Paper 3, Paper GPU o Paper Quantum sin instrucción explícita del
usuario y sin un protocolo evidence-gated que cambie una figura, tabla,
appendix o claim publicable.
