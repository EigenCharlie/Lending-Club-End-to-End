#!/usr/bin/env python3
"""Build the governed project-intake layer for Andrija Djurovic LinkedIn material.

The capture scripts create raw inventory, downloaded assets, and extracted text.
This script is intentionally curated: it turns that material into executable
research decisions with stop conditions, while keeping LinkedIn as intake
evidence rather than public bibliographic evidence.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

PACK_DIR = Path("reports/linkedin_credit_risk_andrija_djurovic")
DATA_DIR = PACK_DIR / "data"
DOCS_DIR = PACK_DIR / "docs"


@dataclass(frozen=True)
class CuratedDecision:
    topic_family: str
    key_takeaway_es: str
    possible_executable_or_implementable: str
    project_destination: str
    handling_decision: str
    stop_condition: str
    closure_status: str
    evidence_status: str
    claim_use_rule: str
    notes: str


POST_DECISIONS: dict[str, CuratedDecision] = {
    "7455503720936693760": CuratedDecision(
        "ADSFCR / IFRS9 / validation source bundle",
        "Paquete de notas con SMI, C2ST, heterogeneidad, PD calibration, Somers D y WoE; sirve como mapa de fuentes e implementaciones, no como evidencia academica final.",
        "Cruzar el bundle con lanes ya existentes: C2ST, IFRS9, encoding-stability y PD validation; no abrir codigo nuevo salvo que una fuente cambie tabla o respuesta de reviewer.",
        "Mini libro CRPTO como intake; tesis para FLI/IFRS9; Paper 4 solo por appendices gobernados.",
        "append_as_source_map",
        "Cerrado cuando cada enlace de alto valor tenga texto local, source status, y destino paper/tesis/archive; no se reabre por otro link comercial.",
        "closed",
        "linkedin_intake_plus_external_sources_read",
        "LinkedIn material is intake evidence only; GitHub/PDFs son contexto tecnico hasta ser benchmarkeados localmente.",
        "Tercer update de Working Notes; varias piezas ya fueron absorbidas en auditorias ADSFCR previas.",
    ),
    "7443560044412968960": CuratedDecision(
        "Applied Data Science for Credit Risk umbrella",
        "Post paraguas del repositorio ADSFCR con backtesting, calibration, MoC, Somers D y heterogeneidad.",
        "Mantenerlo como indice de fuentes; reutilizar solo las piezas que ya tienen PDF/codigo leido y trazado.",
        "Mini libro CRPTO; docs de investigacion; tesis.",
        "append_as_governed_intake",
        "Cerrado cuando las fuentes externas resueltas queden separadas por software, PDF tecnico, LinkedIn-only y bloqueado.",
        "closed",
        "external_sources_read_as_software_context",
        "No promover claims desde el post; usarlo como puerta de entrada a fuentes externas.",
        "Complementa el audit ADSFCR ya existente.",
    ),
    "7461301929135022080": CuratedDecision(
        "PD backtesting multi-period average testing",
        "El z-test multi-periodo debe alinear su varianza con el promedio aritmetico anual de tasas de default; el tamano efectivo cambia la lectura de p-values.",
        "Prototipo Paper 4 o appendix: recomputar backtesting multi-periodo con promedio aritmetico y N efectivo si puede cambiar una tabla de validacion o una defensa de reviewer.",
        "Paper CRPTO reviewer-defense; Paper 4 candidate lane; tesis validation chapter.",
        "promote_to_crpto_defense_and_park_prototype",
        "Parar si la prueba no cambia un claim, tabla de appendix o respuesta probable; no sustituye el champion.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "Usar como caveat tecnico con source status; buscar soporte academico/regulatorio antes de bibliografia publica.",
        "Deck PDF local extraido.",
    ),
    "7458765378664427520": CuratedDecision(
        "Tree-based PD risk-factor interactions",
        "Arboles pueden descubrir interacciones utiles para scorecards si respetan minimos de observaciones/defaults, missing treatment, monotonicidad y validacion de negocio.",
        "Prototipo acotado: generar interacciones candidatas y comparar contra inputs actuales solo como challenger explicable.",
        "Paper 4 prototype; tesis feature-engineering; libro Ch05/Ch06.",
        "park_with_executable_prototype_rule",
        "Parar cuando la interaccion no mejore evidencia reproducible o rompa interpretabilidad/monotonicidad; no abrir reemplazo del champion.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "No venderlo como ML champion; ubicarlo como soporte selectivo.",
        "Deck PDF local extraido.",
    ),
    "7453691768476405760": CuratedDecision(
        "Reliability limits of multi-period normal tests",
        "Simulaciones del deck muestran inflacion de error tipo I con pocos periodos y bajo autocorrelacion; la normalidad multi-periodo no es un pase automatico.",
        "Agregar caveat a PD validation/reviewer bank; simular localmente solo si cambia appendix de backtesting.",
        "Mini libro CRPTO reviewer-defense; tesis validation chapter.",
        "promote_as_caveat",
        "Cerrado cuando el caveat quede documentado y no se use para sobreprometer una prueba alternativa.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "Necesita fuente independiente antes de ser cita publica fuerte.",
        "Deck PDF local extraido.",
    ),
    "7451153797881618432": CuratedDecision(
        "Supervised Macroeconomic Index for IFRS9 FLI",
        "SMI aborda series macro cortas mediante estructura supervisada, promedios de modelos/constrained OLS y explicabilidad para IFRS9.",
        "Mantener como lane de tesis IFRS9; no mover body IJDS salvo como limitacion/agenda.",
        "Tesis IFRS9 chapter; CRPTO limitations.",
        "append_to_thesis_lane",
        "Parar al registrar SMI como expansion futura; no convertir IJDS en paper IFRS9.",
        "closed",
        "linkedin_deck_pdf_extracted_plus_github_manual",
        "Software/contexto tecnico; usar resultados solo si se reproducen localmente.",
        "Deck PDF local extraido y manual externo leido.",
    ),
    "7448618361606873088": CuratedDecision(
        "LGD/EAD Somers D under conservatism",
        "La conservatividad regulatoria puede distorsionar la lectura de poder discriminante en LGD/EAD si se evalua el output final ajustado.",
        "Guardar para tesis LGD/EAD y MRM; no impacta el champion PD.",
        "Tesis; Paper 4 si se abre LGD/EAD diagnostic lane.",
        "park_for_thesis",
        "Cerrado al separar PD del lane LGD/EAD y registrar que no cambia claims IJDS actuales.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "No extrapolar a PD sin experimento especifico.",
        "Deck PDF local extraido.",
    ),
    "7446081646761738240": CuratedDecision(
        "MoC Type C aggregate conservatism",
        "La cuantificacion de MoC Type C puede variar mucho segun metodo y supuestos; es una advertencia de gobernanza mas que un algoritmo unico.",
        "Usar como caveat de MRM/calibration; implementar solo si se define appendix de MoC con claim verificable.",
        "Tesis governance; Paper 4 optional appendix.",
        "park_with_claim_gate",
        "Parar si no cambia un claim de conservatism/materiality; no inflar IJDS.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "Requiere fuentes regulatorias/academicas para cita publica.",
        "Deck PDF local extraido.",
    ),
    "7441019641449050113": CuratedDecision(
        "Two-sided exact binomial PD backtesting",
        "El test exacto bilateral binomial ya esta alineado con la suite de validacion local.",
        "Mantener como source trace; no se requiere nueva implementacion porque el proyecto ya cubre exact binomial/Jeffreys/z/HL.",
        "Mini libro CRPTO validation appendix; docs ADSFCR.",
        "already_implemented_append_trace",
        "Cerrado cuando los tests locales sigan cubriendo binomial exacto y el post no se use como evidencia unica.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "Usar solo como contexto de implementacion, no como cita publica final.",
        "Deck PDF local extraido.",
    ),
    "7463579851317239808": CuratedDecision(
        "Blocked public LinkedIn post",
        "La captura publica devolvio texto generico de LinkedIn; no hay contenido tecnico verificable.",
        "Ninguno; reabrir solo con permalink legible, captura logueada o fuente canonica externa.",
        "Archive.",
        "archive_blocked",
        "Cerrado por blocker publico documentado; no inferir contenido desde snippets.",
        "closed_blocked_public_capture",
        "public_capture_generic_linkedin_text",
        "No usar en claims.",
        "Bloqueado sin sesion visible desde Linux.",
    ),
    "7462756612538073090": CuratedDecision(
        "Blocked public LinkedIn post",
        "La captura publica devolvio texto generico de LinkedIn; no hay contenido tecnico verificable.",
        "Ninguno; reabrir solo con permalink legible, captura logueada o fuente canonica externa.",
        "Archive.",
        "archive_blocked",
        "Cerrado por blocker publico documentado; no inferir contenido desde snippets.",
        "closed_blocked_public_capture",
        "public_capture_generic_linkedin_text",
        "No usar en claims.",
        "Bloqueado sin sesion visible desde Linux.",
    ),
    "6965364290866282496": CuratedDecision(
        "Monotonic binning tooling for PD/LGD/EAD",
        "monobinpy replica en Python la logica de monotonic binning para factores numericos en modelos PD, LGD y EAD.",
        "Usar como tooling/source-discovery para comparar principios de binning monotono; no adoptar dependencia ni claim de performance sin benchmark local.",
        "Libro Ch05; tesis feature-engineering; archive/tooling context.",
        "archive_tooling_context",
        "Cerrado cuando el README de monobinpy quede leido y la idea se clasifique como tooling, no como evidencia del champion.",
        "closed",
        "external_github_readme_read",
        "Software externo es contexto tecnico; no evidencia publica de rendimiento CRPTO.",
        "Post antiguo recuperado por busqueda web adicional.",
    ),
    "7392821942728273920": CuratedDecision(
        "IFRS9 forward-looking modeling source bundle",
        "Bundle IFRS9 con PCA, ADF power, OLS importance, dynamic/recursive regressions, Vasicek transition/Z-factor y LGD-as-function-of-DR.",
        "Usar como mapa de tesis IFRS9; solo importar a CRPTO como limitacion o agenda, no como segunda contribucion.",
        "Tesis; CRPTO limitations; docs research.",
        "append_to_thesis_lane",
        "Parar cuando cada PDF externo tenga status local y destino; no crear experimentos IFRS9 sin claim nuevo aprobado.",
        "closed",
        "external_pdfs_read",
        "Contexto tecnico/source discovery; evidencia publica exige fuentes independientes.",
        "Diez PDFs externos extraidos.",
    ),
    "7435946096050339840": CuratedDecision(
        "Model-based discrete PD rating scale calibration",
        "Optimizar intercepto logistico con offset de log-odds puede preservar el vinculo entre score raw y PD calibrada por grados.",
        "Prototipo appendix: comparar calibracion discreta model-based contra calibracion actual solo si mejora explicacion o diagnostico de PD.",
        "Paper CRPTO reviewer-defense; Paper 4 candidate; tesis calibration.",
        "promote_to_defense_and_park_prototype",
        "Parar si no cambia calibracion, tabla de appendix o respuesta de reviewer; no reabrir champion.",
        "closed",
        "linkedin_deck_pdf_extracted",
        "Necesita reproducibilidad local antes de afirmacion fuerte.",
        "Deck PDF local extraido.",
    ),
    "7330472362141761538": CuratedDecision(
        "R IRB toolkit",
        "Inventario de herramientas R para binning, PD/LGD validation y scorecard workflows.",
        "No adoptar dependencia; usar solo como orientacion de funcionalidades que el proyecto ya cubre o podria comparar.",
        "Archive/tooling context; tesis appendix if needed.",
        "archive_tooling_context",
        "Cerrado al registrar que no hay dependencia nueva ni claim que cambie.",
        "closed",
        "external_github_readme_read",
        "Software externo no es evidencia de performance del proyecto.",
        "PDtoolkit/monobin/LGDtoolkit son contexto de ecosistema.",
    ),
    "7112329977517215745": CuratedDecision(
        "Probability of Default Rating Modeling with R book release",
        "El post apunta a temas valiosos: binning monotono/U-shape, ranking uncertainty, MoC, ML soporte y calibration.",
        "Usar como source-discovery; no citar el post como evidencia; revisar libro/fuentes formales si se requiere.",
        "Tesis source-discovery; archive.",
        "archive_source_discovery",
        "Cerrado cuando los temas queden en atlas y no se promocionen sin fuente formal.",
        "closed",
        "linkedin_post_text_only_public",
        "No usar como evidencia bibliografica publica.",
        "Book/commercial context.",
    ),
    "7420725800745754625": CuratedDecision(
        "Credit Risk Modeling Working Notes second update",
        "Bundle con dynamic regression, constrained threshold LR, heterogeneity p-values, WoE importance, LGD survival, recursive regression y PoC calibration.",
        "Cruzar con lanes ya implementados y tesis; no abrir nueva linea salvo claim/test reproducible.",
        "Tesis; docs research; CRPTO limitations.",
        "append_as_source_map",
        "Cerrado cuando links externos esten leidos y la imagen quede marcada como contexto visual.",
        "closed",
        "external_pdfs_read_plus_image_manual_read",
        "Los PDFs son contexto tecnico; imagen solo confirma material del post.",
        "Imagen leida manualmente; no contenia claim tecnico adicional.",
    ),
    "6886388764965392384": CuratedDecision(
        "PDtoolkit package",
        "PDtoolkit agrupa binning, WoE, IV, calibration, validation, heterogeneity y Monte Carlo power para PD.",
        "No adoptar R package; usar como checklist comparativo de funcionalidades.",
        "Archive/tooling context; thesis appendix.",
        "archive_tooling_context",
        "Cerrado cuando README local quede leido y no haya dependency adoption.",
        "closed",
        "external_github_readme_read",
        "Software/contexto; no evidence claim sin benchmark.",
        "README externo capturado.",
    ),
    "7163079427961090049": CuratedDecision(
        "ADSFCR repository source anchor",
        "Post inicial del repositorio ADSFCR; importante como ancla de procedencia.",
        "Mantener como fuente de trazabilidad hacia el repo; no implementa nada por si solo.",
        "Archive/source trace.",
        "archive_source_trace",
        "Cerrado cuando el repo ADSFCR quede leido y ya exista audit/backlog local.",
        "closed",
        "external_github_readme_read",
        "No usar el post como claim.",
        "ADSFCR ya tiene auditorias locales.",
    ),
    "7342064984069173248": CuratedDecision(
        "Model shift and scorecard model risk",
        "Model shift trata perturbaciones/drifts de inputs como sensibilidad de parametros/output y herramienta MRM para scorecards.",
        "Agregar a reviewer-defense/tesis como marco de model risk y drift; no cambiar champion.",
        "Tesis MRM; CRPTO limitations; Paper 4 governance.",
        "promote_to_thesis_and_defense",
        "Cerrado cuando paper/prospectus esten leidos y la imagen de programa quede marcada como confirmatoria.",
        "closed",
        "external_pdf_read_plus_image_manual_read",
        "Buscar/citar fuente formal si se usa en texto publico.",
        "Imagen de conferencia leida; PDF/prospectus extraidos.",
    ),
    "7296422990588579840": CuratedDecision(
        "WoE encoding instability",
        "Recalcular WoE sobre datos etiquetados por prediccion puede impedir replicacion exacta y revelar riesgo de preprocessing bajo drift/self-labeling.",
        "Incorporar como caveat de encoding stability y posible appendix Paper 4 si cambia monitoring claim.",
        "Mini libro CRPTO; Paper 4 candidate; tesis feature governance.",
        "promote_to_crpto_caveat",
        "Cerrado cuando el caveat quede documentado y no se use para reemplazar el pipeline actual.",
        "closed",
        "linkedin_deck_pdf_extracted_plus_external_pdf_read",
        "Usar con fuente formal o experimento propio si se vuelve claim fuerte.",
        "Deck PDF local extraido.",
    ),
}


ARTICLE_DECISIONS: dict[str, CuratedDecision] = {
    "irb-models-uncertainty-conformal-inference-andrija-djurovic": CuratedDecision(
        "Conformal inference for IRB model uncertainty",
        "Conformal inference aparece como lenguaje natural para diagnosticar incertidumbre y segmentos con menor desempeno en IRB.",
        "Usar como defensa conceptual de CRPTO: intervalos/cobertura como capa de decision, sin prometer cobertura condicional perfecta.",
        "Paper CRPTO discussion/reviewer-defense; tesis.",
        "promote_to_reviewer_defense",
        "Cerrado cuando quede como argumento de framing y no como evidencia academica unica.",
        "closed",
        "linkedin_article_text_captured",
        "LinkedIn material is intake evidence only; citar literatura CP formal para el paper.",
        "Articulo publico capturado completo.",
    ),
    "irb-model-calibration-andrija-djurovic": CuratedDecision(
        "IRB calibration and risk quantification",
        "Diferencia calibracion IRB de ML generico: risk differentiation no sustituye risk quantification ni pruebas de subestimacion.",
        "Agregar lenguaje al mini libro sobre gate de PD calibrada antes del LP y sobre ranking/calibracion/valor economico como dimensiones separadas.",
        "Mini libro CRPTO; IJDS discussion; tesis.",
        "promote_to_crpto_language",
        "Cerrado cuando el lenguaje quede en reviewer-defense y no cree una seccion paralela de IRB regulation.",
        "closed",
        "linkedin_article_text_captured",
        "Usar fuente formal para claims regulatorios.",
        "Articulo publico capturado completo.",
    ),
    "selective-use-machine-learning-irb-models-andrija-djurovic": CuratedDecision(
        "Selective ML support for IRB models",
        "ML puede apoyar imputacion, seleccion/ingenieria de factores, residual diagnostics y two-stage modeling sin sustituir el contrato auditable.",
        "Usar para defender que CRPTO no es anti-ML: ML entra como soporte gobernado, no como reemplazo opaco.",
        "Paper CRPTO reviewer-defense; Paper 4 prototypes; tesis.",
        "promote_to_reviewer_defense",
        "Cerrado cuando quede como principio de diseno y todo prototipo tenga claim gate.",
        "closed",
        "linkedin_article_text_captured",
        "LinkedIn-only para framing; evidencia tecnica debe venir de benchmark local o literatura.",
        "Articulo publico capturado completo.",
    ),
    "machine-learning-irb-models-andrija-djurovic": CuratedDecision(
        "Machine learning for IRB models",
        "ML en IRB debe justificar mejora predictiva significativa, valor economico, plausibilidad de relaciones y trazabilidad.",
        "Agregar a reviewer bank: AUC/Gini no bastan; decision value, calibracion y plausibilidad gobiernan complejidad.",
        "Mini libro CRPTO; IJDS discussion; tesis.",
        "promote_to_crpto_defense",
        "Cerrado cuando el punto quede enlazado al evidence spine y no se use para reclamar compliance legal.",
        "closed",
        "linkedin_article_text_captured",
        "No usar como cita academica central; buscar EBA/fuentes oficiales si se cita.",
        "Articulo publico capturado completo.",
    ),
}


TOPIC_ATLAS = [
    {
        "concept": "Multi-period average PD backtesting",
        "method_family": "PD validation",
        "novelty_for_project": "medium_high",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "CRPTO reviewer-defense; Paper 4 appendix candidate",
        "implementation_difficulty": "medium",
        "claim_risk": "medium",
        "decision": "append_or_prototype_only_if_validation_claim_changes",
    },
    {
        "concept": "Reliability limits of multi-period normal tests",
        "method_family": "PD validation",
        "novelty_for_project": "medium",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "CRPTO limitations; thesis validation chapter",
        "implementation_difficulty": "low_medium",
        "claim_risk": "medium",
        "decision": "promote_as_caveat_not_as_new_result",
    },
    {
        "concept": "Two-sided exact binomial PD backtesting",
        "method_family": "PD validation",
        "novelty_for_project": "low",
        "evidence_strength": "already_implemented_locally",
        "project_destination": "Validation appendix/source trace",
        "implementation_difficulty": "low",
        "claim_risk": "low",
        "decision": "already_implemented",
    },
    {
        "concept": "Model-based discrete PD rating scale calibration",
        "method_family": "PD calibration",
        "novelty_for_project": "medium_high",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "Reviewer-defense; Paper 4 prototype",
        "implementation_difficulty": "medium",
        "claim_risk": "medium",
        "decision": "park_with_appendix_gate",
    },
    {
        "concept": "Risk differentiation vs risk quantification",
        "method_family": "IRB calibration governance",
        "novelty_for_project": "medium",
        "evidence_strength": "linkedin_article_intake",
        "project_destination": "CRPTO mini-book language; IJDS discussion",
        "implementation_difficulty": "low",
        "claim_risk": "low_medium",
        "decision": "promote_as_language_with_formal_sources",
    },
    {
        "concept": "Conformal inference as IRB uncertainty diagnostic",
        "method_family": "Uncertainty quantification",
        "novelty_for_project": "medium",
        "evidence_strength": "linkedin_article_plus_project_core",
        "project_destination": "CRPTO framing; thesis",
        "implementation_difficulty": "low",
        "claim_risk": "medium",
        "decision": "promote_as_framing_only",
    },
    {
        "concept": "Selective ML support for IRB models",
        "method_family": "ML governance",
        "novelty_for_project": "medium",
        "evidence_strength": "linkedin_article_intake",
        "project_destination": "Reviewer-defense; Paper 4 prototypes",
        "implementation_difficulty": "low_medium",
        "claim_risk": "medium",
        "decision": "promote_as_governance_principle",
    },
    {
        "concept": "Tree-based interactions under credit constraints",
        "method_family": "Feature engineering",
        "novelty_for_project": "medium_high",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "Paper 4 prototype; thesis feature chapter",
        "implementation_difficulty": "medium",
        "claim_risk": "medium",
        "decision": "prototype_only_with_monotonicity_business_gate",
    },
    {
        "concept": "Monotonic binning tooling for credit risk factors",
        "method_family": "Feature engineering",
        "novelty_for_project": "low_medium",
        "evidence_strength": "software_readme_intake",
        "project_destination": "Book Ch05; thesis tooling appendix",
        "implementation_difficulty": "low_medium",
        "claim_risk": "low_medium",
        "decision": "archive_as_tooling_context_until_local_benchmark",
    },
    {
        "concept": "WoE encoding instability",
        "method_family": "Scorecard stability",
        "novelty_for_project": "high",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "CRPTO caveat; Paper 4 appendix; thesis",
        "implementation_difficulty": "medium",
        "claim_risk": "medium",
        "decision": "promote_as_caveat_and_candidate_appendix",
    },
    {
        "concept": "Supervised Macroeconomic Index for IFRS9 FLI",
        "method_family": "IFRS9 macro modeling",
        "novelty_for_project": "medium_high",
        "evidence_strength": "software_pdf_intake",
        "project_destination": "Thesis IFRS9 expansion",
        "implementation_difficulty": "high",
        "claim_risk": "high",
        "decision": "thesis_only_until_dedicated_protocol",
    },
    {
        "concept": "Model shift for scorecard model risk",
        "method_family": "MRM and drift",
        "novelty_for_project": "medium",
        "evidence_strength": "external_pdf_intake",
        "project_destination": "CRPTO limitations; thesis governance",
        "implementation_difficulty": "medium",
        "claim_risk": "medium",
        "decision": "append_as_governance_frame",
    },
    {
        "concept": "LGD/EAD Somers D under conservatism",
        "method_family": "LGD/EAD validation",
        "novelty_for_project": "medium",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "Thesis LGD/EAD chapter",
        "implementation_difficulty": "medium",
        "claim_risk": "medium_high",
        "decision": "park_for_thesis",
    },
    {
        "concept": "MoC Type C aggregate conservatism",
        "method_family": "Regulatory model risk",
        "novelty_for_project": "medium",
        "evidence_strength": "technical_deck_intake",
        "project_destination": "Thesis governance; optional Paper 4 appendix",
        "implementation_difficulty": "medium",
        "claim_risk": "medium_high",
        "decision": "park_with_claim_gate",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_manifest() -> tuple[dict[str, list[dict[str, str]]], Counter[str]]:
    rows = read_csv(DATA_DIR / "attachment_manifest.csv")
    by_activity: dict[str, list[dict[str, str]]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    for row in rows:
        activity_id = row.get("activity_id", "")
        by_activity[activity_id].append(row)
        statuses[row.get("text_extract_status", "") or row.get("ocr_status", "")] += 1
    return by_activity, statuses


def summarize_external_sources() -> tuple[dict[str, list[dict[str, str]]], Counter[str]]:
    rows = read_csv(DATA_DIR / "high_value_external_source_reading.csv")
    by_activity: dict[str, list[dict[str, str]]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    for row in rows:
        parent = row.get("parent_activity_id", "")
        by_activity[parent].append(row)
        statuses[row.get("reading_status", "")] += 1
    return by_activity, statuses


def attachment_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No visible attachment captured."
    type_counts = Counter(row.get("asset_type", "unknown") for row in rows)
    status_counts = Counter(row.get("text_extract_status") or row.get("ocr_status") for row in rows)
    return "; ".join(
        [
            "asset_types=" + "|".join(f"{k}:{v}" for k, v in sorted(type_counts.items())),
            "asset_statuses=" + "|".join(f"{k}:{v}" for k, v in sorted(status_counts.items())),
        ]
    )


def external_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No high-value external source captured."
    source_counts = Counter(row.get("source_type", "unknown") for row in rows)
    status_counts = Counter(row.get("reading_status", "unknown") for row in rows)
    return "; ".join(
        [
            f"readable_sources={len(rows)}",
            "source_types=" + "|".join(f"{k}:{v}" for k, v in sorted(source_counts.items())),
            "reading_statuses=" + "|".join(f"{k}:{v}" for k, v in sorted(status_counts.items())),
        ]
    )


def build_post_backlog() -> list[dict[str, str]]:
    posts = read_csv(DATA_DIR / "posts_index.csv")
    manifest_by_activity, _ = summarize_manifest()
    externals_by_activity, _ = summarize_external_sources()
    rows: list[dict[str, str]] = []
    for post in posts:
        activity_id = post["activity_id"]
        decision = POST_DECISIONS[activity_id]
        rows.append(
            {
                "backlog_id": f"ANDRIJA-POST-{int(post['n']):03d}",
                "source_kind": "indexed_andrija_post",
                "activity_id": activity_id,
                "post_url": post["post_url"],
                "title": post["title"],
                "capture_status": decision.evidence_status,
                "attachment_summary": attachment_summary(manifest_by_activity.get(activity_id, [])),
                "external_source_summary": external_summary(
                    externals_by_activity.get(activity_id, [])
                ),
                "topic_family": decision.topic_family,
                "key_takeaway_es": decision.key_takeaway_es,
                "possible_executable_or_implementable": decision.possible_executable_or_implementable,
                "project_destination": decision.project_destination,
                "handling_decision": decision.handling_decision,
                "stop_condition": decision.stop_condition,
                "closure_status": decision.closure_status,
                "claim_use_rule": decision.claim_use_rule,
                "notes": decision.notes,
            }
        )
    return rows


def build_article_backlog() -> list[dict[str, str]]:
    article_rows = read_csv(DATA_DIR / "article_capture_log.csv")
    rows: list[dict[str, str]] = []
    for index, article in enumerate(article_rows, start=1):
        article_id = article["article_id"]
        decision = ARTICLE_DECISIONS[article_id]
        rows.append(
            {
                "backlog_id": f"ANDRIJA-ARTICLE-{index:03d}",
                "source_kind": "andrija_public_linkedin_article",
                "article_id": article_id,
                "source_url": article["source_url"],
                "title": article["title"],
                "capture_status": article["capture_status"],
                "text_length": article["text_length"],
                "image_count": article["image_count"],
                "topic_family": decision.topic_family,
                "key_takeaway_es": decision.key_takeaway_es,
                "possible_executable_or_implementable": decision.possible_executable_or_implementable,
                "project_destination": decision.project_destination,
                "handling_decision": decision.handling_decision,
                "stop_condition": decision.stop_condition,
                "closure_status": decision.closure_status,
                "claim_use_rule": decision.claim_use_rule,
                "notes": decision.notes,
            }
        )
    return rows


def build_visual_log() -> list[dict[str, str]]:
    manifest_rows = read_csv(DATA_DIR / "attachment_manifest.csv")
    article_assets = read_csv(DATA_DIR / "article_asset_manifest.csv")
    rows: list[dict[str, str]] = []

    post_image_notes = {
        "7420725800745754625": (
            "manual_visual_read_completed",
            "Imagen de Credit Risk Modeling Working Notes: portada/landing visual; no agrega claim tecnico adicional frente al texto y links.",
        ),
        "7342064984069173248": (
            "manual_visual_read_completed",
            "Imagen de programa CSCC XIX: confirma titulo/abstract de model shift; el contenido sustantivo se leyo en PDF/prospectus.",
        ),
    }
    for row in manifest_rows:
        asset_type = row.get("asset_type", "")
        activity_id = row.get("activity_id", "")
        if "image" not in asset_type:
            continue
        status, memo = post_image_notes.get(
            activity_id,
            (
                "manual_visual_triaged_context_only",
                "Asset visual capturado; el contenido analitico principal se lee desde texto/PDF/link asociado.",
            ),
        )
        rows.append(
            {
                "source_kind": "post_attachment",
                "parent_id": activity_id,
                "asset_id": row["asset_id"],
                "asset_type": asset_type,
                "local_path": row.get("local_path", ""),
                "visual_read_status": status,
                "analytic_memo": memo,
            }
        )

    for row in article_assets:
        rows.append(
            {
                "source_kind": "article_image",
                "parent_id": row["article_id"],
                "asset_id": row["asset_id"],
                "asset_type": row["asset_type"],
                "local_path": row["local_path"],
                "visual_read_status": "manual_visual_triaged_context_only",
                "analytic_memo": "Imagen publica de articulo/cobertura/recomendacion; el texto del articulo es la fuente analitica. No se promueve ningun claim desde la imagen.",
            }
        )
    return rows


def markdown_table(
    rows: list[dict[str, str]], columns: list[str], max_rows: int | None = None
) -> str:
    selected = rows[:max_rows] if max_rows else rows
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected:
        cells = []
        for col in columns:
            text = str(row.get(col, "")).replace("\n", " ").replace("|", "/")
            cells.append(text)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def clean_markdown(text: str) -> str:
    """Normalize indented f-string markdown blocks that contain unindented tables."""
    lines = dedent(text).strip().splitlines()
    return "\n".join(line[8:] if line.startswith("        ") else line for line in lines)


def write_docs(
    post_rows: list[dict[str, str]],
    article_rows: list[dict[str, str]],
    visual_rows: list[dict[str, str]],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _, manifest_statuses = summarize_manifest()
    _, source_statuses = summarize_external_sources()
    link_rows = read_csv(DATA_DIR / "external_link_backlog.csv")

    decision_counts = Counter(row["handling_decision"] for row in post_rows + article_rows)
    closure_counts = Counter(row["closure_status"] for row in post_rows + article_rows)
    visual_counts = Counter(row["visual_read_status"] for row in visual_rows)

    memo = clean_markdown(
        f"""
        # Andrija Djurovic LinkedIn Intake Decisions - 2026-05-25

        ## Status

        This pack covers the public material captured from
        <https://www.linkedin.com/in/andrija-djurovic/> plus public articles and
        resolved external links. LinkedIn material is intake evidence only: it
        can suggest concepts, implementation ideas, reviewer-defense language or
        source-discovery paths, but no public paper/book claim is promoted from
        LinkedIn alone.

        - Indexed posts: {len(post_rows)}
        - Public articles: {len(article_rows)}
        - External link backlog rows: {len(link_rows)}
        - High-value readable source rows: {sum(source_statuses.values())}
        - Asset manifest status counts: {dict(sorted(manifest_statuses.items()))}
        - External source reading status counts: {dict(sorted(source_statuses.items()))}
        - Backlog closure counts: {dict(sorted(closure_counts.items()))}
        - Decision counts: {dict(sorted(decision_counts.items()))}
        - Visual read/triage counts: {dict(sorted(visual_counts.items()))}

        ## Promote / Append / Park / Archive

        {markdown_table(post_rows + article_rows, ["backlog_id", "topic_family", "handling_decision", "project_destination", "closure_status"])}

        ## High-Value Decisions

        - **PD backtesting**: multi-period average testing, reliability limits of
          normal tests, and exact binomial testing strengthen reviewer-defense
          around calibration/backtesting. Exact binomial is already implemented;
          multi-period average testing is parked as an appendix/prototype only
          if it changes a validation claim.
        - **PD calibration**: the model-based discrete PD rating-scale deck and
          IRB calibration article sharpen the distinction between risk
          differentiation, risk quantification and decision value.
        - **Selective ML**: tree-based interactions and selective ML articles
          support a controlled Paper 4 lane: ML as risk-factor engineering,
          residual diagnostics or challenger support, not as uncontrolled
          champion replacement.
        - **WOE/encoding stability**: WoE instability is promoted as a CRPTO
          caveat and thesis/Paper 4 candidate because it connects preprocessing,
          monitoring and self-labeled replication risk.
        - **IFRS9/LGD/EAD/MRM**: SMI, PCA/ADF/recursive-regression, LGD/EAD
          Somers D and model shift are thesis/governance material. They should
          remain outside the IJDS body except as limitations or future work.

        ## Stop Rule

        The Andrija intake is closed when each indexed post/article has a local
        content status, a destination, a possible implementable, and a stop
        condition. Reopen only if a newly visible logged-in comment/link, a
        canonical paper, or a local experiment can change a claim, appendix
        table, reviewer response or thesis chapter.
        """
    )
    (DOCS_DIR / "andrija_project_intake_decisions_2026-05-25.md").write_text(
        memo + "\n", encoding="utf-8"
    )

    evidence_map = clean_markdown(
        f"""
        # Andrija LinkedIn Claim Evidence Map - 2026-05-25

        ## Governance

        LinkedIn material is intake evidence only. Public-facing claims require
        one of: local project result, peer-reviewed/official source, verified
        software artifact with local benchmark, or explicitly labeled
        exploratory/theory framing.

        ## Post-By-Post Map

        {markdown_table(post_rows, ["backlog_id", "title", "evidence_status", "handling_decision", "stop_condition"])}

        ## Article Map

        {markdown_table(article_rows, ["backlog_id", "title", "capture_status", "handling_decision", "stop_condition"])}

        ## Visual Material

        Feed images were read manually when they carried potential technical
        content. Article images were triaged as context because the captured
        article text contains the analytic material and the images are cover,
        header, or recommendation assets.

        {markdown_table(visual_rows, ["source_kind", "parent_id", "asset_id", "visual_read_status", "analytic_memo"], max_rows=12)}
        """
    )
    (DOCS_DIR / "andrija_linkedin_claim_evidence_map_2026-05-25.md").write_text(
        evidence_map + "\n", encoding="utf-8"
    )

    readme = clean_markdown(
        f"""
        # Andrija Djurovic LinkedIn Credit Risk Pack

        This folder contains the second LinkedIn credit-risk intake, focused on
        Andrija Djurovic public posts, public LinkedIn articles and external
        sources. It is private research material and should not be redistributed
        casually.

        Core outputs:

        - `data/andrija_post_execution_backlog.csv`
        - `data/andrija_article_execution_backlog.csv`
        - `data/andrija_source_topic_atlas.csv`
        - `data/andrija_visual_read_log.csv`
        - `docs/andrija_project_intake_decisions_2026-05-25.md`
        - `docs/andrija_linkedin_claim_evidence_map_2026-05-25.md`

        Closure: {len(post_rows)} posts and {len(article_rows)} articles have a
        decision, destination and stop condition. Claims remain governed by the
        rule: LinkedIn material is intake evidence only.
        """
    )
    (PACK_DIR / "README.md").write_text(readme + "\n", encoding="utf-8")


def main() -> None:
    post_rows = build_post_backlog()
    article_rows = build_article_backlog()
    visual_rows = build_visual_log()

    post_fields = [
        "backlog_id",
        "source_kind",
        "activity_id",
        "post_url",
        "title",
        "capture_status",
        "attachment_summary",
        "external_source_summary",
        "topic_family",
        "key_takeaway_es",
        "possible_executable_or_implementable",
        "project_destination",
        "handling_decision",
        "stop_condition",
        "closure_status",
        "claim_use_rule",
        "notes",
    ]
    article_fields = [
        "backlog_id",
        "source_kind",
        "article_id",
        "source_url",
        "title",
        "capture_status",
        "text_length",
        "image_count",
        "topic_family",
        "key_takeaway_es",
        "possible_executable_or_implementable",
        "project_destination",
        "handling_decision",
        "stop_condition",
        "closure_status",
        "claim_use_rule",
        "notes",
    ]
    visual_fields = [
        "source_kind",
        "parent_id",
        "asset_id",
        "asset_type",
        "local_path",
        "visual_read_status",
        "analytic_memo",
    ]
    atlas_fields = [
        "concept",
        "method_family",
        "novelty_for_project",
        "evidence_strength",
        "project_destination",
        "implementation_difficulty",
        "claim_risk",
        "decision",
    ]

    write_csv(DATA_DIR / "andrija_post_execution_backlog.csv", post_rows, post_fields)
    write_csv(DATA_DIR / "andrija_article_execution_backlog.csv", article_rows, article_fields)
    write_csv(DATA_DIR / "andrija_visual_read_log.csv", visual_rows, visual_fields)
    write_csv(DATA_DIR / "andrija_source_topic_atlas.csv", TOPIC_ATLAS, atlas_fields)
    write_docs(post_rows, article_rows, visual_rows)

    print(
        "Wrote Andrija intake: "
        f"{len(post_rows)} posts, {len(article_rows)} articles, "
        f"{len(visual_rows)} visual rows, {len(TOPIC_ATLAS)} concepts."
    )


if __name__ == "__main__":
    main()
