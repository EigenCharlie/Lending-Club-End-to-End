"""Contexto academico estructurado para la pagina de tesis de especializacion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEvent:
    date_iso: str
    milestone: str
    detail: str
    status: str


@dataclass(frozen=True)
class EvaluatorComment:
    criterion: str
    score: float
    evaluator_comment: str
    project_response: str


THESIS_TITLE = "Conformal Prediction para la Calibracion y Cuantificacion de Incertidumbre en Modelos de Riesgo Crediticio"
THESIS_AUTHOR = "Carlos Alfredo Vergara Rojas"
THESIS_PROGRAM = "Especializacion en Analitica y Ciencia de Datos Aplicada - UTP"

RESEARCH_QUESTION = (
    "Como mejorar la calibracion y la cuantificacion de la incertidumbre en modelos de riesgo "
    "crediticio (PD, LGD, EAD) mediante tecnicas de prediccion conformal y cual es su desempeno "
    "frente a metodos tradicionales de calibracion."
)

GENERAL_OBJECTIVE = (
    "Implementar y evaluar tecnicas de prediccion conformal para mejorar la calibracion y "
    "cuantificacion de incertidumbre en PD, LGD y EAD, utilizando Lending Club como caso de estudio."
)

SPECIFIC_OBJECTIVES: tuple[str, ...] = (
    "Revisar el estado del arte de calibracion, cuantificacion de incertidumbre y conformal prediction aplicada a riesgo crediticio.",
    "Construir y preparar dataset Lending Club 2007-2020 con variables objetivo PD, LGD y EAD.",
    "Disenar e implementar experimentos con Split Conformal, Mondrian, Venn-Abers y CQR.",
    "Comparar desempeno frente a metodos tradicionales de calibracion (Platt, Isotonic y benchmarks residuales).",
    "Evaluar impacto potencial en provisiones IFRS9, asignacion de capital y backtesting regulatorio.",
    "Proponer lineamientos practicos de adopcion de conformal prediction en contexto bancario real.",
)

TIMELINE_EVENTS: tuple[TimelineEvent, ...] = (
    TimelineEvent(
        date_iso="2025-12-11",
        milestone="Orientaciones institucionales",
        detail=(
            "Correo con el paso a paso para evaluacion de anteproyecto y tramite de trabajo de grado."
        ),
        status="Completado",
    ),
    TimelineEvent(
        date_iso="2026-02-25",
        milestone="Evaluacion del anteproyecto",
        detail=(
            "Concepto: APROBADO CON AJUSTES. Se recomienda reducir alcance para mayor viabilidad."
        ),
        status="Completado",
    ),
    TimelineEvent(
        date_iso="2026-02-28",
        milestone="Cierre del periodo lectivo",
        detail=(
            "Fecha de cierre academico reportada en orientaciones institucionales."
        ),
        status="Completado",
    ),
    TimelineEvent(
        date_iso="2026-03-01",
        milestone="Run canonico del proyecto",
        detail="Generacion de artefactos `run_tag=2026-03-01-C-smart-v2` para narrativa oficial.",
        status="Completado",
    ),
    TimelineEvent(
        date_iso="2026-03-02",
        milestone="Inicio de revision con directora",
        detail="Etapa actual enfocada en consolidar historia, resultados y argumentos de defensa.",
        status="En curso",
    ),
)

EVALUATOR_COMMENTS: tuple[EvaluatorComment, ...] = (
    EvaluatorComment(
        criterion="Pertinencia",
        score=4.8,
        evaluator_comment=(
            "La propuesta es pertinente en utilidad real y alineacion con las lineas del programa."
        ),
        project_response=(
            "Se mantiene enfoque aplicado a riesgo crediticio con evidencia ejecutable y trazabilidad."
        ),
    ),
    EvaluatorComment(
        criterion="Planteamiento",
        score=4.3,
        evaluator_comment=(
            "Problema bien argumentado, pero los objetivos pueden mejorar en funcion de alcance realista."
        ),
        project_response=(
            "Se acota narrativa a evidencia actual del pipeline y se explicita alcance por componente."
        ),
    ),
    EvaluatorComment(
        criterion="Metodologia",
        score=4.6,
        evaluator_comment=(
            "Diseno en fases adecuado, pero con alcance amplio y riesgoso para un solo investigador."
        ),
        project_response=(
            "Se prioriza estructura CRISP-DM con artefactos canonicamente disponibles y gaps transparentes."
        ),
    ),
    EvaluatorComment(
        criterion="Innovacion",
        score=4.7,
        evaluator_comment=(
            "No totalmente original, pero novedosa por integracion metodologica."
        ),
        project_response=(
            "Se destaca integracion PD/LGD/EAD + conformal + IFRS9 con evidencia reproducible."
        ),
    ),
    EvaluatorComment(
        criterion="Viabilidad",
        score=3.5,
        evaluator_comment=(
            "Acceso a datos garantizado, pero cronograma ajustado para el numero de modelos/simulaciones."
        ),
        project_response=(
            "Se reporta estado real por componente, diferenciando logros consolidados y pendientes."
        ),
    ),
    EvaluatorComment(
        criterion="Comunicacion",
        score=4.8,
        evaluator_comment="Redaccion y forma general adecuadas.",
        project_response=(
            "Se fortalece narrativa con trazabilidad artefactual y lectura por capitulos."
        ),
    ),
)

FINAL_EVALUATION_SCORE = 4.5
FINAL_EVALUATION_CONCEPT = "Aprobado con ajustes"
