<!-- cspell:disable -->
<!-- markdownlint-disable -->

# Backlog Vivo: Paper 2 + Paper 4

Fecha de curaduría: 2026-07-20

Este documento reemplaza el backlog unificado de marzo. La superficie local
activa del repositorio queda reducida a:

- **Paper 2**: auditoría diagnóstica `parked_ifrs9`; no paper IFRS9 activo.
- **Paper 4**: Sequential Credit Decision Analytics Living Lab.

CRPTO/Paper IJDS vive en un repositorio autocontenido. Su commit y hashes
observados están en `docs/research/crpto_external_contract_2026-07-20.yml`.
Este repositorio puede conservar artefactos congelados, pero no debe volver a
tratarlos como el estado actual de CRPTO/Paper Estrella.

## Retirado

| Frente | Decisión | Qué se conserva |
|---|---|---|
| Paper 3 / Mondrian | Retirado como paper separado | La capa Mondrian queda en @sec-mondrian, backtesting, MRM y Paper 4 |
| Paper GPU | Retirado como paper y como capítulo avanzado | Solo evidencia histórica de infraestructura si ya existe en `reports/history` |
| Paper Quantum | Retirado | Ningún claim local; no había experimento ejecutado defendible |
| Paper Estrella local | Retirado del libro grande | Referencia congelada para Paper 4; desarrollo real en `Paper_CRPTO` |

## Paper 2

Estado actual:

- Mantener el capítulo 15 como superficie de auditoría metodológica.
- Conservar ECL, escenarios, SICR, survival y sensibilidad como diagnósticos
  históricos del software, no como resultados prudenciales.
- No reestimar umbrales o cifras con el archivo actual.

Pendiente real:

1. obtener un panel préstamo–fecha-de-reporte point-in-time;
2. definir PD 12m/lifetime, LGD, EAD y SICR con horizontes coherentes;
3. separar desarrollo, calibración, selección de regla y test temporal final;
4. reabrir solo si se satisfacen todos los requisitos de
   `reports/paper_material/paper2/paper2_claim_contract.yml`.

## Paper 4

Estado actual:

- Paper 4 queda como laboratorio vivo acotado, no como contenedor que pueda
  absorber y promover automáticamente cualquier resultado local.
- La superficie oficial en Quarto debe ser compacta; la bitácora viva queda en
  `reports/paper_material/paper4/notes/paper4_living_lab_notebook.md`.
- Las olas históricas se mantienen archivadas solo cuando tienen manifest y
  guardrail explícito.

Frentes abiertos con valor:

1. online/source conformal con boundaries fuertes;
2. DFL/SPO/PyEPO como benchmark teacher-cost hasta disponer de costos
   realizados, clases comparables e inferencia agrupada;
3. CVaR/OCE y tail-risk como appendix o diagnóstico;
4. auditoría negativa de la frontera $Y\to PD/ECL/SICR$ y especificación del
   contrato de datos faltante; no reestimar ni promover proxies monetarios con
   el archivo actual;
5. fairness/source governance como proxy explícito, no fair-lending legal;
6. multi-period/DLA/SDAM como framing y simulación, no Bellman exacto.

## Regla De Priorización

Solo se ejecutan experimentos si tienen:

- claim target;
- evidence gate;
- artifact sink;
- stop rule;
- destino editorial concreto en Paper 2 o Paper 4. Importar a CRPTO exige un
  protocolo nuevo ejecutado dentro del repo externo.

## Stop Rule

No reabrir Paper 3, Paper GPU o Paper Quantum sin instrucción explícita del
usuario y sin un protocolo evidence-gated que cambie una figura, tabla,
appendix o claim publicable.
