# Retiro de Superficies Locales: Paper 3, GPU y Quantum

Fecha: 2026-06-06

## Decisión

El libro grande de Quarto deja de presentar Paper 3, Paper GPU y Paper Quantum
como papers activos. La superficie local activa queda en Paper 2 y Paper 4.
CRPTO/Paper IJDS vive autocontenido en `C:/Users/carlos/Documents/Paper_CRPTO`.

## Qué Se Absorbió

| Frente retirado | Valor que sí aporta | Destino |
|---|---|---|
| Paper 3 / Mondrian | cobertura por subgrupo, alerta de subcobertura, estabilidad temporal, límite de cobertura marginal | capítulo 7, MRM, Paper 4 |
| Paper GPU | lección de aceleración selectiva: útil solo si cambia costo/tiempo sin degradar fidelidad | evidencia histórica, no paper |
| Paper Quantum | revisión conceptual de QML para crédito, pero sin experimento local | cerrado; no entra al libro |

## Qué Se Eliminó

- rutas Quarto de `16-paper-mondrian`, `17-paper-gpu` y `18-paper-quantum`;
- capítulos avanzados GPU `13f` y `13g`;
- apéndice `B-gpu-benchmarks`;
- notebook y paquete activo de Paper 3;
- manuscrito LaTeX `papers/paper3_copa2026`;
- figuras editoriales/publication dedicadas a Paper 3/GPU/Quantum cuando no
  alimentan capítulos activos;
- bibliografía quantum que solo existía para el paper retirado.

## Qué No Se Debe Borrar

- la implementación conformal Mondrian del pipeline;
- artefactos de backtesting y MRM que sostienen capítulos activos;
- evidencia congelada de CRPTO/Paper Estrella usada por Paper 4 como referencia;
- archivos en `reports/history` que sean claramente históricos y no backlog
  activo.

## Regla De Reapertura

Estos frentes no se reabren por curiosidad. Solo vuelven si el usuario lo pide
explícitamente y si existe un protocolo con:

- claim target;
- evidence gate;
- artifact sink;
- stop rule;
- destino editorial concreto.
