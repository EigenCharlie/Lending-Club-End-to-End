# CRPTO Retirement and Paper 4 Boundary - 2026-06-06

## Decisión

`C:\Users\carlos\Documents\Paper_CRPTO` (`/mnt/c/Users/carlos/Documents/Paper_CRPTO`
en WSL) es la fuente de verdad para CRPTO, Paper Estrella, el paper IJDS y la
tesis de maestría. Ese proyecto debe ser autocontenido: no depende de este repo
padre ni debe referenciar rutas, capítulos o tests de aquí.

Este repo conserva conocimiento histórico de CRPTO, pero deja de poseer
superficies editoriales activas. Por eso se retiran:

- `book/chapters/14-paper-estrella/`
- `papers/paper_crpto_book/`
- `papers/paper1_estrella/`

## Qué se migró a Paper_CRPTO

Antes del retiro se verificó que `Paper_CRPTO` ya contenía el libro Quarto
enfocado, el paper IJDS (`paper/CRPTO_ijds.qmd`), el supplement
(`paper/supplement_ijds.qmd`), los artefactos A0--A34 y la réplica
multidataset Prosper/Freddie. Desde este repo se llevó lo que faltaba y sí
aportaba:

- auditoría exhaustiva de los 61 PDFs de `Papers_tesis/`;
- matriz fuente paper-by-paper para `promote`, `append`, `park` y `future_gate`;
- índice de captions y visual sinks;
- generador reproducible `scripts/build_papers_tesis_deep_audit.py`;
- hooks editoriales en la guía de claims, release, bibliografía y reviewer map.

## Rol de Paper 4 en este repo

Paper 4 queda como living lab y superconjunto metodológico: puede explorar
source/shift conformal, DFL/PyEPO, IFRS9 proxy, fairness proxy, online
conformal, CATE/policy value, DLA/sequential decision y tail risk. Su salida no
es automáticamente material CRPTO.

La regla de frontera es:

1. Paper 4 puede producir diagnósticos, negativos y evidencia exploratoria.
2. Solo material con claim target, evidence gate, artifact sink y stop rule
   puede aspirar a importarse a `Paper_CRPTO`.
3. Si algo pasa el gate, se reescribe dentro de `Paper_CRPTO` como artefacto
   autocontenido. No se referencia este repo padre como dependencia editorial.
4. No se reabre el champion CRPTO desde este repo.

## Stop rule

No reconstruir `book/chapters/14-paper-estrella`, `papers/paper_crpto_book` ni
`papers/paper1_estrella` sin una instrucción explícita. Cualquier nuevo aporte
CRPTO/IJDS/tesis debe entrar directamente a `Paper_CRPTO` y pasar sus propios
guardrails.
