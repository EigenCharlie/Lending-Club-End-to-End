# Streamlit Storytelling Guide

Guía editorial y de UX para mantener consistencia narrativa en el dashboard de tesis.

## Objetivo
- Reducir carga cognitiva sin perder rigor técnico.
- Alinear cómo se presentan métricas, gráficos y decisiones entre páginas.
- Mantener tono experto en `Investigación` y `paper_*`.
- Reorganizar la experiencia alrededor de dos ejes explícitos:
  `Pipeline Operativo` e `Insight Factory`.

## Ejes narrativos oficiales
- `Pipeline Operativo`: datos -> PD -> calibración -> conformal -> survival/time series -> IFRS9 -> portfolio policy -> governance.
- `Insight Factory`: explicabilidad, causalidad extendida, notebooks, benchmarks RAPIDS, drafts de paper y exploración.
- `Libro/Quarto future-ready`: cada página debe declarar `book_chapter`, `artifact_scope` y `pipeline_role` para permitir migración editorial posterior sin rehacer el contenido.

## Estructura recomendada por tipo de página

## Páginas no-research (ejecutivo / analítica / decisión / gobernanza)
1. Hook / idea clave (qué demuestra la página)
2. Por qué importa en negocio
3. Decisión que habilita
4. Evidencia guiada (gráficos/tablas con lectura)
5. Caveats / límites
6. Siguiente paso (continuidad narrativa)

## Páginas de investigación y borradores de paper
1. Claim técnico / pregunta
2. Método / diseño experimental
3. Resultados clave
4. Amenazas a validez
5. Posición frente a literatura
6. Qué falta para publication-ready

## Reglas de carga cognitiva
- 1 idea principal por bloque visual.
- El texto principal debe interpretar la evidencia, no repetir la leyenda del gráfico.
- Expanders: solo detalle secundario, no contenido crítico.
- Tablas largas: siempre precedidas por "qué mirar".
- Insertar checkpoints/resúmenes en páginas largas cada 2–3 bloques.

## Estándar de visualizaciones
Para cada gráfico importante:
- Título interpretativo (no solo descriptivo)
- "Qué mirar primero"
- Implicación de decisión
- Error común de lectura (si aplica)

## Glosario contextual
- No repetir definiciones largas en múltiples páginas.
- Usar popovers/context help para términos recurrentes:
  - canónico
  - calibración
  - conformal
  - coverage
  - ECE / Brier / KS / Gini
  - Price of Robustness

## Sinergia DVC + Streamlit (patrón recomendado)
- DVC genera snapshots canónicos (`reports/dvc/*.json|csv`)
- Streamlit consume esos artefactos con loaders cacheados
- CLI (`dvc metrics show`, `dvc plots diff`) se usa fuera del runtime de la app
- La app comunica el snapshot actual y su interpretación

## Tono editorial por sección
- `Inicio`, `Recorrido`, `Analítica`, `Decisiones`: mixto (negocio + técnico), escaneable.
- `Gobernanza`: técnico-operativo, accionable.
- `Investigación` y `paper_*`: experto-first, sin pedagogía redundante.
- `Anexos`: técnico, directo, con menos narrativa si es benchmark/evidencia.

## Checklist de revisión antes de merge
- ¿La página tiene una idea clave explícita?
- ¿Las métricas están interpretadas, no solo mostradas?
- ¿La decisión habilitada está escrita?
- ¿Hay caveats suficientes?
- ¿Se evitó repetir definiciones que ya están en glosario?
- ¿La narrativa mantiene el tono correcto para la audiencia de esa sección?
