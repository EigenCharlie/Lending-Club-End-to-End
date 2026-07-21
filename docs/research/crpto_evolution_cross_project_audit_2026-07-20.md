# Auditoría de evolución CRPTO y reconciliación entre proyectos

Fecha de corte: 2026-07-20 (America/Bogota)

## Veredicto ejecutivo

La evolución del antiguo *Paper Estrella* hacia el CRPTO/IJDS actual fue, en lo
esencial, una corrección científica necesaria: pasó de una narrativa de
`champion` económico y policy ganadora a una auditoría de identificación que
separa score, cobertura candidata, endpoint, comparador y decisión. El cambio
no debilitó un resultado bien identificado; retiró conclusiones que los datos y
el protocolo ya no podían sostener después de auditar madurez, disponibilidad
de labels, selección y soporte común.

El contrato externo observado en
`EigenCharlie/Paper_CRPTO@69095e05beae282701b4ea38aa69da26a209106f`
no selecciona learner, ventana, taxonomía, $\gamma$, ruler, coordenada, cap,
comparador ni policy. Su objeto conformal es el resultado binario observado
$Y$, no una PD individual latente. En consecuencia:

- no conviene reimportar al CRPTO actual los resultados sustantivos de Paper 2
  o Paper 4;
- sí conviene transferir al libro y a ambos papers su disciplina de estimandos,
  separación outcome-free/evaluation, partial identification y registros de
  claims;
- del proyecto padre solo entraron ahora a CRPTO controles editoriales:
  frontera explícita $Y$ versus PD/ECL/SICR y retiro inequívoco de 32 capítulos
  históricos. No se añadió ningún número ni claim empírico nuevo.

Paper 2 debe permanecer `parked_ifrs9`. Paper 4 puede continuar como living
lab, pero sus simuladores, slices post-selección, proxy FICO y benchmark PyEPO
teacher-cost no son evidencia para CRPTO ni una ruta de promoción automática.

## Alcance, repositorios y método

### Repositorios observados

| Superficie | Estado auditado |
|---|---|
| CRPTO externo | `Paper_CRPTO`, `main`, 289 commits alcanzables, commit final `69095e05...` |
| Proyecto padre | `Lending-Club-End-to-End`, rama `sync/paper-estrella-economic-champion-pipeline-freeze-2026-05-04`, parent previo `3c220f4405...` |
| Contrato portable | `docs/research/crpto_external_contract_2026-07-20.yml`, 13 superficies con SHA-256 |

La rama del proyecto padre contiene una historia extensa posterior a `main` y
no se trató como una rama corta apta para merge automático. Esta auditoría no
reescribe esa historia ni mezcla el repo padre con el CRPTO autocontenido.

### Procedimiento

La revisión combinó:

1. inventario completo de Git, ramas, commits, tags, archivos eliminados y
   diferencias en los pivotes que cambiaron claims;
2. inspección de paper, supplement, TeX generado, companion Quarto, registros
   de fuentes, claim ledger, manifests JSON, DVC pointers, tablas y figuras;
3. reconciliación de números paper-facing contra las autoridades registradas;
4. auditorías adversariales independientes de CRPTO, Paper 2, Paper 4 y del
   libro padre;
5. tests semánticos y de integridad, lint, tipos, hooks, auditoría de
   dependencias, estado DVC local/remoto y renders HTML/PDF;
6. inspección visual de las 24 páginas del companion CRPTO después del cambio.

No se reestimaron Paper 2 o Paper 4 para fabricar una conclusión más atractiva.
Cuando el diseño no identificaba el claim, el resultado de la auditoría fue
reducirlo, parkearlo o convertirlo en diagnóstico.

## Evolución del CRPTO por pivotes verificables

Los 289 commits no representan 289 cambios científicos independientes. La
tabla agrupa los commits que alteraron el estimando, la población, el
comparador, la selección o la autoridad de artefactos; los commits intermedios
son principalmente implementación, documentación, QA o mantenimiento de esas
fases.

| Fecha | Commit | Evolución | Juicio de auditoría |
|---|---|---|---|
| 2026-05-10 | `70b5ea7` | Bootstrap del repo autónomo con score/policy congelados y narrativa heredada de champion. | Correcto separar el repo; prematuro tratar el freeze como evidencia definitiva. |
| 2026-05-12 | `166f6d5` | Dossier y companion Quarto orientados a paper/tesis. | Útil para trazabilidad; mezclaba todavía narrativa editorial con autoridad científica. |
| 2026-05-29 | `8ac4d6c` | Integra evidencia P2/CVaR/online y amplía el libro manteniendo el champion. | Valioso como exploración; incorrecto exportar automáticamente estimandos distintos al mismo claim CRPTO. |
| 2026-06-03 | `be182ec` | Corrige el signo de *price of robustness* y añade ledger monetario. | Corrección obligatoria; prueba por qué una tabla favorable no debe sobrevivir sin tests de signo. |
| 2026-06-09 | `9576ba6` | Formaliza Assumption/Theorem/Remark de la lectura anterior. | Buena disciplina formal, pero una demostración correcta no rescata un estimando mal interpretado. |
| 2026-06-14 | `8ce7aaa` | Obtiene la lectura exacta de $V$ para outcome binario. | Pivote teórico correcto: hace visible la geometría discreta que la lectura continua ocultaba. |
| 2026-07-02 | `5da0849` | Promueve `pool93` como evidencia champion. | Históricamente reproducible, pero hoy retirado: selección y endpoint no soportan el lenguaje de winner. |
| 2026-07-09 | `e0daf55` | Promueve selector determinista `compact-v7`. | Mejoró reproducibilidad del selector, no su identificación; correctamente relegado a procedencia. |
| 2026-07-10 | `b4bbf48` | Reconstrucción *maturity-safe* revierte la lectura económica anterior. | Cambio científico correcto: la madurez/resolución del outcome era una condición material, no un detalle. |
| 2026-07-12 | `0cd8e79` | V4 convierte el manuscrito en auditoría de geometría binaria e identificación. | Pivote central correcto; alinea el claim con lo observado y abandona la obligación de encontrar ganador. |
| 2026-07-13 | `930c49f` | Añade cinco especificaciones de riesgo y controles WOE/IV/taxonomía. | Correcto como recurrencia/control; no autoriza seleccionar el mejor modelo con el mismo OOT. |
| 2026-07-15 | `e71cd98` | Auditoría de endpoint, razones de no resolución y diseño outcome-safe. | Correcto: evita llamar snapshot point-in-time a un archivo que no lo verifica. |
| 2026-07-16 | `fd7d61d` | Consolida la cápsula activa y elimina una gran superficie legacy del checkout. | Correcto porque preserva procedencia en Git/DVC y reduce autoridades rivales; exigía marcar también el libro dormido. |
| 2026-07-16 | `d68e806` | Integra seis familias de sensibilidad y QA pre-freeze. | Correcto: reporta el grid completo y conserva resultados mixtos, sin escoger escenario favorable. |
| 2026-07-20 | `c8d8c9e` | Simplifica solver y prueba reproducibilidad desde clean clone. | Correcto y necesario para una submission verificable. |
| 2026-07-20 | `9427ddc` | Sincroniza paper, supplement, TeX, preview, registry y companion activo. | Correcto: elimina drift entre superficies y fija 33 DVC pointers/20 claims. |
| 2026-07-20 | `f48457a` / `69095e0` | Declara la frontera $Y$–PD/ECL/SICR y marca 32 QMD no renderizados como históricos. | Correcto: cierra una fuente silenciosa de claims refutados sin borrar procedencia. |

### Qué cambió realmente en la teoría

La cadena antigua trataba aproximadamente el ancho o upper endpoint alrededor
de un score PD como una cantidad continua de incertidumbre de PD y después lo
inyectaba en optimización. La evolución auditada separó cuatro objetos:

1. un score calibrado para el outcome binario;
2. un intervalo de residuos cuyo claim candidato es cobertura de $Y$;
3. un coeficiente de guardrail $q_i(\gamma)=(1-\gamma)p_i+\gamma u_i$;
4. una asignación definida respecto de un ruler, soporte y coordenada concretos.

Para $Y\in\{0,1\}$ la geometría residual tiene umbrales y discontinuidades que
no admiten la intuición de una banda continua de PD. Además, la validez de
candidatos no se transporta al conjunto elegido después de optimizar. El
comparador es parte del estimando: cambiar ruler o coordenada puede cambiar la
dirección del contraste. Por eso el resultado defendible es una auditoría de
transporte e identificación, no una policy ganadora.

## Contrato científico vigente y evidencia dura

El registro activo observado contiene:

- 640,543 préstamos elegibles de 36 meses dentro de los roles temporales
  declarados, provenientes de 2,925,493 filas del archivo;
- 376,890 candidatos OOT primarios: 364,814 resueltos y 12,076 no resueltos;
- 8 ventanas residuales, 5 especificaciones de riesgo y 9 policies reportadas;
- 2 rulers outcome-blind, 3 coordenadas interiores y 3,067 caps de frontera;
- 33 punteros DVC y 20 claims activos;
- 6 familias de sensibilidad: endpoint, estructura de portafolio, segundo
  origen, codificación de missingness, completitud de labels de fit y
  granularidad de asignación.

Resultados que sí pueden citarse bajo ese contrato:

- los 40 upper bounds agudos de cobertura candidato–ventana están por debajo
  de 0.90 bajo el endpoint principal;
- el patrón CatBoost reaparece bajo tres codificaciones de missingness y un
  segundo origen retrospectivo;
- cuatro escenarios declarados para 215 labels de fit no disponibles dejan los
  32 bounds escenario–ventana por debajo de nominal;
- el cruce geométrico W7–W8 es compatible con un mecanismo de umbral, pero el
  escenario all-default lo elimina: no es universal ni causal;
- los 216 envelopes amplios point-cap incluyen cero;
- los resultados estructurales son mixtos y la dirección depende del ruler y
  de la coordenada;
- el redondeo determinista a USD 25 altera tasas evaluadas a lo sumo 0.0013
  puntos porcentuales;
- no hay `policy_winner`, preferred ruler, preferred coordinate ni garantía de
  selected-set coverage.

## Auditoría de decisiones: agregar, eliminar y reformular

### Cambios correctos

- **Agregar endpoint reconstruction y unresolved outcomes.** Evita sesgo de
  madurez y permite partial identification en lugar de completar labels sin
  declarar reglas.
- **Separar outcome-free freeze de evaluation.** Hace auditable qué información
  estuvo disponible al crear menús, rulers y asignaciones.
- **Reportar todas las familias y no un escenario favorable.** Reduce
  cherry-picking y hace visibles las direcciones mixtas.
- **Agregar controles de crédito sin promover ganador.** Las cinco
  especificaciones prueban recurrencia acotada, no superioridad.
- **Tratar el comparador como parte del estimando.** Es la corrección decisiva
  para interpretar los contrastes de portafolio.
- **Eliminar superficies editoriales rivales.** La cápsula activa, claim ledger,
  hashes y tests reducen drift entre paper, supplement, TeX y artefactos.
- **Conservar procedencia pero marcarla.** Los 32 QMD retirados ya no parecen
  evidencia actual y siguen disponibles para reconstruir la historia.

### Eliminaciones o retiros correctos

- `pool93`, `compact-v7`, selected-policy y A1--A40 como autoridad activa;
- lenguaje de champion/winner, dominancia universal y mejora económica general;
- lectura de endpoints de $Y$ como intervalos de PD;
- inferencias ECL/SICR/IFRS9 sin panel y horizonte correspondientes;
- afirmaciones de validación prospectiva, causalidad o snapshot point-in-time;
- transferencias Paper 4 → CRPTO basadas solo en que un resultado parecía
  favorable.

### Aspectos que solo deben conservarse como diagnóstico

- score/calibrador local y sus nombres `pd_*`;
- simulación CVaR/MDCP bajo paths internos;
- slices F05 posteriores a selección;
- comparación legacy-root versus proxy FICO;
- PyEPO con costos teacher y menús muestreados solapados;
- escenarios mecánicos de ECL y staging reconstruido desde estado terminal.

## Hallazgo de implementación local: Venn--Abers y validez split

La revisión adversarial del proyecto padre encontró dos problemas que no
pertenecen al CRPTO externo y que, precisamente por eso, no deben contaminar su
freeze:

1. `create_pd_intervals_venn_abers` prefería una clase de MAPIE cuyo
   `predict_proba` devuelve probabilidades de clase $[P(Y=0),P(Y=1)]$ y las
   reinterpretaba como el par multiprobabilístico $[p_0,p_1]$. Esa ruta podía
   fabricar un punto cercano a 0.5 y un ancho sin semántica Venn--Abers.
2. El calibrador canónico local resumía el par real con el punto medio
   $(p_0+p_1)/2$. Para pérdida logarítmica, el punto minimax publicado por la
   implementación `venn-abers` es $p_1/(1-p_0+p_1)$.

La corrección elimina la ruta MAPIE ambigua y usa el API público de
`venn-abers` 1.5.1 para punto y multiprobabilidades. Se versionó la regla:
fits nuevos llevan `point_rule=log_loss_minimax`; un pickle histórico que no
tiene ese atributo conserva `midpoint_legacy`. El modo replay también congela
la regla: manifiestos nuevos escriben `selected_calibration_point_rule` y los
manifiestos Venn--Abers antiguos que carecen del campo se interpretan como
`midpoint_legacy`. Así, ni cargar el mismo pickle ni reproducir un manifiesto
histórico cambia silenciosamente sus predicciones. No se reescribieron el
pickle, el modelo, las tablas ni las decisiones congeladas.

Comparación directa sobre las 276,869 filas de la partición OOT local:

| Salida | Media | Brier | Log-loss | AUC |
|---|---:|---:|---:|---:|
| score raw | 0.4398 | 0.2088904 | 0.6018832 | 0.7125226 |
| punto medio histórico | 0.2171240 | 0.1546305 | 0.4770611 | 0.7124382 |
| minimax log-loss | 0.2172380 | 0.1546292 | 0.4770409 | 0.7124382 |

El efecto agregado es pequeño, pero no nulo: 25.449% de filas cambia más de
$10^{-4}$, 1.546% más de $10^{-3}$, 14 filas más de 0.01 y 3,590 observaciones
cruzan el threshold exploratorio 0.10. Esto confirma que un refreeze silencioso
sería incorrecto aunque Brier y AUC apenas se muevan.

La misma auditoría mostró una limitación más fundamental: los labels de la
partición de calibración ajustaron el calibrador probabilístico y después se
reutilizaron para conformalizar; además, el OOT fue inspeccionado y reutilizado
en comparaciones y selección. El runtime sigue siendo reproducible, pero sus
tasas son cobertura empírica retrospectiva de $Y$, no una garantía split
conformal activa ni un holdout confirmatorio. Para recuperar ese claim se
necesitan muestras separadas o un diseño anidado/cross-fitted preespecificado y
una evaluación final intacta.

Este hallazgo refuerza, no modifica, la transferencia decidida: Paper 2 debe
seguir parqueado, Paper 4 debe mantener sus resultados como descriptivos y el
libro debe enseñar la diferencia entre una interfaz ejecutable y un teorema
aplicable. CRPTO no recibe las métricas locales ni una nueva policy.

## Impacto sobre el libro Quarto

El libro mezclaba tres capas: ingeniería ejecutable, resultados históricos y
claims de papers. Esa mezcla produjo saltos semánticos repetidos: $Y\to$PD,
PD$\to$ECL, intervalo candidato$\to$selected set y simulación$\to$policy.
La reconciliación mantiene los artefactos como historia técnica, pero añade un
contrato científico transversal y reescribe las páginas que promovían esos
saltos.

Cambios principales:

- fecha reproducible en lugar de `date: today`;
- retiro de `_quarto-core.yml`, `index-core.qmd` y `serve_book_core.py`, que
  constituían una segunda navegación rota;
- abstract, mapa ejecutivo y contribuciones reescritos como síntesis auditada;
- capítulos conformal delimitados a cobertura de $Y$;
- portafolio convertido en sensibilidad/selección histórica, no policy actual;
- IFRS9, SICR, ECL y series temporales convertidos en prototipos diagnósticos;
- Paper 2 marcado `parked_ifrs9`;
- Paper 4 alineado a findings y boundaries actuales;
- IDs de figuras reparados y registros de archivo apuntando al page registry
  existente.

La mejora principal que el libro recibe de CRPTO no es una tabla nueva: es una
jerarquía de autoridad. Nombres de archivos como `champion`, `pd_high` o
`strict_holdout` dejan de determinar por sí solos el claim editorial.

## Paper 2: qué sobrevive y qué falta

### Diagnóstico

Paper 2 no tiene hoy el diseño mínimo para un paper IFRS9:

- no existe un panel a fecha de reporte;
- DPD y Stage 3 se reconstruyen desde estado terminal;
- no hay PD de originación y PD actual comparables al mismo horizonte;
- el threshold se seleccionó y evaluó sobre el mismo OOT;
- el filtro a préstamos resueltos induce sesgo de madurez;
- conviven estimadores centrales de ECL inconsistentes;
- el llamado BMA no implementa una posterior BMA identificable;
- la serie temporal es de vintage, no un forecast calendario de cartera.

### Uso correcto de la evolución CRPTO

Paper 2 sí debe adoptar:

- cohortes y endpoints explícitos;
- outcomes no resueltos y bounds en lugar de imputación silenciosa;
- separación desarrollo/selección/evaluación;
- contrato de horizonte para PD, LGD y EAD;
- tests que impidan llamar PD/ECL a cobertura de $Y$.

Hasta reestimar con un panel apropiado, su contribución defendible es una nota
metodológica negativa: mostrar por qué un pipeline binario de Lending Club no
se convierte automáticamente en IFRS9. Requisitos de reapertura: reporting
dates, estados longitudinales, horizontes coherentes, SICR preespecificado,
validación externa o temporal intacta, una definición única de ECL y análisis
de censura/madurez.

## Paper 4: qué sobrevive y qué falta

Paper 4 conserva valor como laboratorio gobernado, no como colección de
ganadores. El registro actual permite:

- F03 como hecho de ejecución de 512 paths y 494,592 filas;
- F04 como contraste descriptivo condicional a un simulador interno;
- F05 como slices post-selección, no holdouts estrictos;
- PyEPO como benchmark descriptivo de teacher-regret sobre menús/semillas
  solapados, sin el Wilcoxon fila-a-fila inválido ni el score manual de
  auditabilidad en la vista activa;
- F14 como diagnóstico histórico legacy-root vs proxy FICO.

La antigua “absorción de Paper 2” se corrigió de manera sustantiva: F08 y F13,
los anchors y el appendix ya no retienen rangos ECL, umbrales SICR, costos de
staging o importes monetarios como evidencia. Ahora forman una auditoría
negativa de estimando/readiness y procedencia no citable. Llamar `proxy` a la
transformación no bastaba para reparar $Y\ne PD$.

No permite dominancia, inferencia independiente por instancia, performance de
pérdida realizada, atribución a la clase de modelo, garantía conformal del set
seleccionado, fair-lending legal ni resultado CRPTO.

La evolución CRPTO ayuda a Paper 4 de tres maneras: obliga a declarar el ruler,
separar candidates de selección y congelar decisiones antes de outcomes. Para
que PyEPO/DFL se convierta en contribución publicable harían falta costos
realizados o un estimando económico defendible, clases de modelo comparables,
menús o unidades independientes, inferencia agrupada y una evaluación
temporal intacta. Para DLA/online se requieren estados y transiciones observadas,
no solo paths sintéticos.

## Matriz de transferencia

| Origen → destino | Transferir ahora | Condición o frontera |
|---|---|---|
| CRPTO → libro | Estimando $Y$, endpoint contract, partial identification, outcome-free/evaluation, candidate vs selected, comparator-as-estimand, registries/hashes | Sí; ya integrado como contrato transversal. |
| CRPTO → Paper 2 | Madurez, unresolved outcomes, horizonte explícito, no transporte $Y\to$PD/ECL | Sí como metodología; no copiar cifras CRPTO a IFRS9. |
| CRPTO → Paper 4 | Freeze antes de outcomes, soporte común, rulers múltiples, no winner por selección ex post | Sí como gobernanza del living lab. |
| Proyecto padre → CRPTO | Marcadores de retiro, guardrails cross-surface y stop rules | Sí; se incorporó gobernanza, no resultados. |
| Paper 2 → CRPTO | SICR, ECL, BMA, TS-vintage | No ahora; estimandos y datos incompatibles. |
| Paper 4 → CRPTO | PyEPO/SPO+, CVaR winner, FICO, fairness, CATE, online, DLA, SDAM | No ahora; evidencia diagnóstica o framing solamente. |
| Papers/artefactos históricos → CRPTO | `pool93`, champion, A1--A40, selected-policy | No; procedencia únicamente. |

## Oportunidades futuras para CRPTO

El proyecto padre contiene ideas que podrían aportar después, pero solo bajo un
nuevo protocolo dentro de `Paper_CRPTO`:

1. **Ledger de promoción genérico.** Reutilizar la taxonomía
   `official_with_boundaries`/`diagnostic_only`/`parked` y exigir sink, gate y
   stop rule para cada claim.
2. **Comparador predict-then-optimize genuino.** Rehacer PyEPO con el mismo
   estimando, misma clase de modelo, costos defendibles, menús independientes y
   evaluación temporal no usada para selección.
3. **Extensión dinámica.** Solo con estados/transiciones observados, decisión
   recurrente y evaluación de política preespecificada.
4. **Extensión prudencial.** Solo con panel de reporting dates, PD/LGD/EAD de
   horizontes definidos y SICR validado; sería otro paper, no una reinterpretación
   de la cobertura binaria existente.
5. **Fairness/FICO.** Solo con atributos, score bureau y pregunta regulatoria
   adecuados; el proxy actual no basta.

Ninguna de estas oportunidades justifica ampliar hoy el abstract o los claims
del CRPTO. La decisión correcta en el corte auditado fue mejorar su contrato de
interpretación y dejar el resultado sustantivo intacto.

## Verificación y trazabilidad

### CRPTO externo

- PR #103: sincronización del dossier activo; commit `9427ddc`, 29 archivos,
  488 inserciones y 94 eliminaciones; merge `e25ac2e63...`.
- PR #104: frontera de estimando y retiro de fuentes dormidas; merge
  `69095e05beae282701b4ea38aa69da26a209106f`; 39 archivos, 455 inserciones y
  11 eliminaciones.
- Suite pytest completa, gates IJDS activo/drift/integridad/manifest, Ruff,
  formato, mypy, pre-commit, auditoría de dependencias y TeX determinista:
  pasan.
- PDF oficial: 29 páginas; body 21; supplement 31; sin páginas vacías, identidad
  expuesta, tamaños no-letter ni fallos de referencias.
- Companion: HTML/PDF completo, 24 páginas inspeccionadas visualmente.
- DVC: cápsula activa disponible y sincronizada con remoto `dagshub`.

### Proyecto padre

- Suite completa: 826 tests pasan; las 13 advertencias son de Optuna,
  statsmodels y KPSS y no son fallos de los contratos añadidos.
- Ruff pasa sobre el repositorio y los 16 Python modificados satisfacen
  `ruff format --check`; `git diff --check` queda limpio.
- El notebook aparcado de Paper 2 valida con `nbformat`: 9 celdas y 0 outputs;
  no se ejecutó para fabricar claims nuevos.
- Quarto inspecciona un único libro de 90 inputs y dos formatos. El render
  conjunto HTML/PDF pasa; el PDF final tiene 402 páginas letter, fuentes
  embebidas y 0 errores estructurales de `qpdf`.
- Se retiró la salida secundaria versionada `book/_output_pdf` (PDF obsoleto de
  322 páginas y una figura) y se añadió al ignore; la autoridad material es el
  render reproducible de `_output`, no un binario histórico rival.
- El HTML final contiene 11,304 referencias locales verificadas y ninguna rota.
  Los artefactos Git se enlazan al branch fuente; los outputs DVC o local-only
  se presentan como rutas de procedencia, no como descargas inexistentes.
- `reports/dependency_summary.json` coincide exactamente con las 62 filas
  recalculadas. El cache DVC requerido está sincronizado con remoto.
- DVC local conserva cuatro stages científicos deliberadamente sucios:
  `core.pd.train_model`, `core.conformal.generate_intervals`,
  `search.portfolio.tradeoff` y `diagnostic.conformal.validate_policy`. No se
  refijaron porque hacerlo habría reescrito resultados fuera del protocolo de
  esta auditoría.
- Auditoría visual adversarial: `APPROVE_RENDER` para el SHA-256
  `065d863e6aaa70290bf6a6fae9724edc2ce84383ce8ee07e72f0a070eaf9107a`;
  402/402 páginas rasterizadas, 0 recortes, overflows o colisiones, 46 páginas
  blancas pares/intencionales y ninguna figura o etiqueta legacy retirada.

Los checks cross-repo son portables: siempre validan el contrato local y solo
inspeccionan el checkout externo cuando existe `CRPTO_ROOT` o la ruta WSL
declarada.

## Riesgos residuales y stop rules

- El libro conserva muchos nombres legacy por compatibilidad de artefactos; no
  deben usarse como semántica científica.
- Un render congelado puede estar técnicamente limpio y científicamente
  obsoleto; por eso los tests revisan también lenguaje y contratos.
- Los artefactos históricos de Paper 4 siguen en disco: solo las 12 páginas
  registradas y los findings/boundaries actuales gobiernan claims. El page
  registry y las vistas descriptivas saneadas gobiernan pertenencia y evidencia;
  los demás archivos preservan procedencia o inputs acotados.
- Actualizar CRPTO exige refrescar explícitamente commit y hashes del contrato;
  nunca aceptar drift silencioso.
- Reabrir Paper 2, promover Paper 4 o importar un claim a CRPTO exige nuevos
  datos/protocolo, no una relectura más optimista de los artefactos actuales.
