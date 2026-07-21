# CRPTO Retirement and Paper 4 Boundary — 2026-06-06

## Actualización auditada — 2026-07-20

`C:\Users\carlos\Documents\Paper_CRPTO`
(`/mnt/c/Users/carlos/Documents/Paper_CRPTO` en WSL) sigue siendo la autoridad
autocontenida para CRPTO y el paper IJDS. La revisión cruzada de julio observó su
`main` en el commit
`69095e05beae282701b4ea38aa69da26a209106f`; el contrato portable, con hashes
de las superficies activas, está en
`docs/research/crpto_external_contract_2026-07-20.yml`.

El CRPTO vigente es una **auditoría retrospectiva de identificación**. Predice
el resultado binario observado $Y$ y no selecciona learner ni policy. Tampoco
selecciona ventana residual, taxonomía, $\gamma$, ruler, coordenada, cap o
comparador. La cobertura candidata de $Y$ no es un intervalo para la PD
individual latente y no se transporta automáticamente a ECL, SICR, expected
loss o selected-set validity.

Las superficies externas activas son:

- `paper/CRPTO_ijds.qmd` y `paper/supplement_ijds.qmd`;
- `docs/research/active_claims_2026-07-14.md`;
- `configs/ijds_active_evidence_sources.yaml` y
  `configs/ijds_claim_ledger.yaml`;
- `reports/crpto/ijds_binary_geometry_frontier_v4_evidence.json`;
- el TeX generado y el paquete de reproducibilidad de `paper/submission/`;
- el companion Quarto reducido a `index`, capítulos `06` y `06b`, y
  referencias.

Los otros 32 QMD bajo `Paper_CRPTO/book/chapters/` son procedencia histórica
no autoritativa y llevan un marcador explícito de retiro. En particular, los
materiales `pool93`, `compact-v7`, selected-policy y A1--A40 no sostienen los
claims actuales.

## Decisión original y alcance local

El proyecto externo debe ser autocontenido: no depende de este repo padre ni
debe referenciar rutas, capítulos o tests de aquí. Este repo conserva
conocimiento histórico de CRPTO, pero no posee una superficie editorial CRPTO
activa. Permanecen retirados:

- `book/chapters/14-paper-estrella/`;
- `papers/paper_crpto_book/`;
- `papers/paper1_estrella/`.

La auditoría de `Papers_tesis/`, su matriz de fuentes y el generador histórico
que se transfirieron en junio explican procedencia, pero ya no forman parte del
registro activo de evidencia del CRPTO. No deben volver a usarse como prueba de
un claim vigente solo porque alguna vez existieron en el repo externo.

## Rol de Paper 2 y Paper 4 en este repo

Paper 2 queda `parked_ifrs9`: conserva un prototipo diagnóstico, pero no hay un
panel a fecha de reporte, PD de originación y actual con horizonte comparable,
SICR validado ni estimador ECL publicable. Sus números históricos no se
promueven como resultado IFRS9.

Paper 4 queda como living lab acotado. Puede explorar source/shift conformal,
DFL/PyEPO, auditorías negativas del transporte hacia IFRS9, fairness proxy,
online conformal, CATE/policy value, DLA/sequential decision y tail risk. Las
transformaciones históricas de Paper 2 hacia ECL/SICR solo pueden conservarse
como sensibilidades mecánicas o fallas de estimando no citables; llamarles
`proxy` no repara la frontera $Y\ne PD$. Sus salidas no son automáticamente
material CRPTO.

La regla de frontera es:

1. Paper 4 puede producir diagnósticos, negativos y evidencia exploratoria.
2. Solo material con claim target, evidence gate, artifact sink y stop rule
   puede aspirar a una evaluación de importación.
3. Cualquier aporte que pase ese gate debe reestimarse y escribirse dentro de
   `Paper_CRPTO` como artefacto autocontenido; este repo no se vuelve una
   dependencia editorial.
4. No se reabre ni se inventa un champion CRPTO desde este repo: el contrato
   vigente no contiene uno.
5. El benchmark PyEPO teacher-cost, los slices post-selección, los simuladores
   condicionales y los proxies FICO/IFRS9 no se exportan como claims CRPTO.

## Stop rule

No reconstruir `book/chapters/14-paper-estrella`, `papers/paper_crpto_book` ni
`papers/paper1_estrella` sin una instrucción explícita. Cualquier nuevo aporte
CRPTO/IJDS debe entrar directamente a `Paper_CRPTO`, declarar el estimando y
pasar sus propios registros, tests, renders y guardrails. Mientras eso no
ocurra, el arreglo correcto es **no importar ningún claim sustantivo** desde
este repo.
