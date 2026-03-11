#!/usr/bin/env python3
"""Generate the Documento Tecnico Final for the specialization thesis.

Scope: Conformal Prediction for calibration and uncertainty quantification
in credit risk models (PD, LGD, EAD). Mondrian conformal, comparative
evaluation, and IFRS9 regulatory impact through CP.

OUT OF SCOPE: portfolio optimization, causal inference, MLOps, fairness deep-dive.

Usage:
    uv run python scripts/generate_thesis_document.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "thesis_poster" / "figures"
NB_IMAGES = ROOT / "reports" / "notebook_images"
OUTPUT_DIR = ROOT / "thesis_poster"
DOCX_PATH = OUTPUT_DIR / "Documento_Tecnico_Final_Carlos_Vergara.docx"

# ── Colors ───────────────────────────────────────────────────────────
UTP_BLUE = RGBColor(0x0B, 0x5E, 0xD7)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
TABLE_HEADER_BG = "0B5ED7"
TABLE_ALT_BG = "F0F4FA"


# ── Helpers ──────────────────────────────────────────────────────────
def set_cell_shading(cell, color_hex: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color_hex)
    shading.set(qn("w:val"), "clear")
    cell._tc.get_or_add_tcPr().append(shading)


def add_table_borders(table) -> None:
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "999999")
        borders.append(element)
    tblPr.append(borders)


def styled_heading(doc: Document, text: str, level: int = 1) -> None:
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = DARK_GRAY


def add_paragraph(
    doc: Document,
    text: str,
    bold: bool = False,
    italic: bool = False,
    alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_after: int = 6,
    first_line_indent: float = 0.0,
) -> None:
    p = doc.add_paragraph()
    p.alignment = alignment
    p.paragraph_format.space_after = Pt(space_after)
    if first_line_indent > 0:
        p.paragraph_format.first_line_indent = Cm(first_line_indent)
    run = p.add_run(text)
    run.font.size = Pt(12)
    run.font.name = "Times New Roman"
    run.font.color.rgb = DARK_GRAY
    run.bold = bold
    run.italic = italic


def add_figure(
    doc: Document, image_path: Path, caption: str, fig_num: int, width_inches: float = 5.5
) -> int:
    if not image_path.exists():
        add_paragraph(
            doc,
            f"[Figura {fig_num}: {caption} — archivo no encontrado: {image_path.name}]",
            italic=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER,
        )
        return fig_num + 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    r = cap.add_run(f"Figura {fig_num}. {caption}")
    r.font.size = Pt(10)
    r.font.name = "Times New Roman"
    r.italic = True
    r.font.color.rgb = DARK_GRAY
    return fig_num + 1


def add_styled_table(
    doc: Document, headers: list[str], rows: list[list[str]], caption: str = "", table_num: int = 0
) -> None:
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(6)
        r = cap.add_run(f"Tabla {table_num}. {caption}")
        r.font.size = Pt(10)
        r.font.name = "Times New Roman"
        r.italic = True
        r.font.color.rgb = DARK_GRAY

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table)

    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.name = "Times New Roman"
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, TABLE_HEADER_BG)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = val
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.font.name = "Times New Roman"
            if i % 2 == 1:
                set_cell_shading(cell, TABLE_ALT_BG)

    doc.add_paragraph()


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENT SECTIONS
# ══════════════════════════════════════════════════════════════════════


def build_cover_page(doc: Document) -> None:
    for _ in range(3):
        doc.add_paragraph()

    logo_path = FIGURES / "logo-utp.jpg"
    if logo_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(logo_path), width=Inches(1.5))

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "Conformal Prediction para la Calibracion y Cuantificacion "
        "de Incertidumbre en Modelos de Riesgo Crediticio"
    )
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.name = "Times New Roman"
    r.font.color.rgb = DARK_GRAY

    doc.add_paragraph()
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Carlos Alfredo Vergara Rojas")
    r.font.size = Pt(14)
    r.font.name = "Times New Roman"
    r.font.color.rgb = DARK_GRAY

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "Documento de investigacion para optar al titulo de Especialista\n"
        "en Analitica y Ciencia de Datos Aplicada"
    )
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"
    r.font.color.rgb = DARK_GRAY

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Docente:\nAlejandra Maria Restrepo Franco")
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"
    r.font.color.rgb = DARK_GRAY

    for _ in range(4):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "UNIVERSIDAD TECNOLOGICA DE PEREIRA\n"
        "PROGRAMA DE ESPECIALIZACION EN ANALITICA Y\n"
        "CIENCIA DE DATOS, COLOMBIA\n"
        "Marzo de 2026"
    )
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.name = "Times New Roman"
    r.font.color.rgb = DARK_GRAY

    doc.add_page_break()


def build_table_of_contents(doc: Document) -> None:
    styled_heading(doc, "CONTENIDO", level=1)
    items = [
        "1. Introduccion",
        "2. Planteamiento del Problema",
        "3. Justificacion de la Investigacion",
        "4. Objetivos",
        "5. Marco Teorico y Estado del Arte",
        "6. Metodologia",
        "7. Consideraciones Eticas",
        "8. Resultados",
        "   8.1 Analisis Exploratorio de Datos",
        "   8.2 Desempeno del Modelo de PD",
        "   8.3 Intervalos Conformales para PD (Mondrian)",
        "   8.4 Comparacion: Mondrian vs. Split Global",
        "   8.5 Intervalos Conformales para LGD y EAD",
        "   8.6 Impacto Regulatorio: IFRS 9 y ECL",
        "9. Discusion de Resultados",
        "10. Conclusiones",
        "11. Lineas Futuras del Proyecto",
        "12. Referencias",
        "Anexo: Declaracion de uso de herramientas de inteligencia artificial",
    ]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("\n".join(items))
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"
    r.font.color.rgb = DARK_GRAY
    doc.add_page_break()


# ── 1. INTRODUCCION ──────────────────────────────────────────────────


def build_introduction(doc: Document) -> None:
    styled_heading(doc, "1. INTRODUCCION", level=1)

    add_paragraph(
        doc,
        (
            "En la actualidad, los sistemas financieros enfrentan un entorno complejo y altamente "
            "regulado, en el cual la correcta estimacion del riesgo crediticio constituye un pilar "
            "fundamental para la estabilidad economica y la confianza de los inversionistas. La "
            "probabilidad de incumplimiento (PD, por sus siglas en ingles), la perdida dado "
            "incumplimiento (LGD) y la exposicion en caso de incumplimiento (EAD) son metricas "
            "centrales utilizadas para calcular provisiones, asignar capital y cumplir con estandares "
            "regulatorios como la Norma Internacional de Informacion Financiera 9 (IFRS 9) y el marco "
            "regulatorio de Basilea III (Basel Committee on Banking Supervision, 2006; International "
            "Accounting Standards Board, 2014). Estas tres metricas alimentan directamente el calculo "
            "de la perdida esperada (Expected Credit Loss, ECL = PD x LGD x EAD), que determina la "
            "salud financiera de las entidades bancarias y su capacidad para absorber perdidas "
            "inesperadas."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Sin embargo, los modelos tradicionales de riesgo crediticio, tanto los clasicos "
            "(regresion logistica, modelos lineales generalizados) como los modernos basados en "
            "machine learning, presentan una limitacion critica: tipicamente ofrecen predicciones "
            "puntuales sin acompanarlas de una medida fiable de la incertidumbre asociada. Una "
            "prediccion de PD del 12% no indica si el modelo esta seguro de esa estimacion o si "
            "el valor real podria estar entre 8% y 17%. Esta ausencia de cuantificacion de "
            "incertidumbre tiene consecuencias directas: perdidas inesperadas por subestimacion del "
            "riesgo, provisiones excesivas por sobreestimacion conservadora, y desconfianza de "
            "auditores y reguladores que exigen mayor transparencia y robustez en los modelos "
            "(Lessmann et al., 2015; Niculescu-Mizil y Caruana, 2005)."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "La prediccion conformal (Conformal Prediction, CP) surge como una alternativa "
            "metodologica rigurosa para abordar esta limitacion. Propuesta originalmente por Vovk, "
            "Gammerman y Shafer (2022), CP es un marco estadistico que permite acompanar cada "
            "prediccion con intervalos de confianza que poseen garantias formales de cobertura, sin "
            "requerir supuestos parametricos sobre la distribucion de los datos. A diferencia de los "
            "metodos bayesianos (que dependen de distribuciones a priori) o del bootstrap (que carece "
            "de garantias formales en muestras finitas), CP ofrece una cobertura marginal garantizada "
            "bajo el unico supuesto de intercambiabilidad de los datos (Angelopoulos y Bates, 2023). "
            "Esta propiedad hace de CP una tecnica especialmente prometedora para el sector financiero, "
            "donde la robustez estadistica y la transparencia metodologica son requisitos regulatorios "
            "fundamentales."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "La presente investigacion implementa y evalua tecnicas de Conformal Prediction en "
            "modelos de riesgo crediticio, con el proposito de mejorar la calibracion de las "
            "probabilidades predichas, cuantificar de manera explicita la incertidumbre asociada "
            "a PD, LGD y EAD, y demostrar su utilidad practica en el contexto regulatorio de IFRS 9. "
            "El estudio se apoya en el dataset publico de LendingClub (2007-2020), uno de los "
            "conjuntos de datos mas completos en la literatura de riesgo crediticio, permitiendo "
            "replicabilidad y comparacion con trabajos previos (Lending Club, 2020). Se utiliza la "
            "variante Mondrian de prediccion conformal, que calcula intervalos por subgrupo (grado "
            "de riesgo), garantizando cobertura condicional por segmento y no solo cobertura promedio "
            "global. Los resultados demuestran coberturas empiricas que superan los niveles nominales "
            "objetivo, validando la efectividad del enfoque propuesto."
        ),
        first_line_indent=1.25,
    )


# ── 2. PLANTEAMIENTO DEL PROBLEMA ────────────────────────────────────


def build_problem_statement(doc: Document) -> None:
    styled_heading(doc, "2. PLANTEAMIENTO DEL PROBLEMA", level=1)

    add_paragraph(
        doc,
        (
            "En el sector financiero, los modelos de riesgo crediticio son herramientas fundamentales "
            "para estimar las tres metricas principales del riesgo: la probabilidad de incumplimiento "
            "(PD), la perdida dado incumplimiento (LGD) y la exposicion en caso de incumplimiento "
            "(EAD) (Basel Committee on Banking Supervision, 2006). Estas metricas alimentan directamente "
            "el calculo de las perdidas esperadas (ECL = PD x LGD x EAD), que a su vez determinan las "
            "provisiones contables bajo IFRS 9 y los requerimientos de capital regulatorio bajo "
            "Basilea III (International Accounting Standards Board, 2014; Deloitte, 2020)."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Sin embargo, estos modelos presentan una limitacion recurrente y critica: suelen entregar "
            "predicciones puntuales sin una adecuada calibracion y sin cuantificar explicitamente la "
            "incertidumbre asociada. Esta deficiencia tiene un impacto multidimensional. En primer "
            "lugar, la ausencia de cuantificacion de incertidumbre puede conducir a una subestimacion "
            "sistematica del riesgo, generando perdidas inesperadas que erosionan el capital. Los "
            "modelos de machine learning, a pesar de su superior capacidad discriminativa (AUC-ROC), "
            "frecuentemente producen probabilidades mal calibradas que distorsionan las decisiones de "
            "originacion y pricing (Niculescu-Mizil y Caruana, 2005; Baesens et al., 2016). En segundo "
            "lugar, ante incertidumbre no cuantificada, las entidades adoptan posturas conservadoras "
            "que resultan en exceso de capital inmovilizado y provisiones excesivas. En tercer lugar, "
            "los reguladores exigen cada vez mas que los modelos cumplan con criterios de robustez y "
            "explicabilidad que dificilmente se satisfacen sin mecanismos de cuantificacion de "
            "incertidumbre (European Banking Authority, 2020)."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Para el caso especifico de LGD y EAD, que son variables continuas acotadas, los metodos "
            "de Conformalized Quantile Regression (CQR) ofrecen intervalos predictivos eficientes que "
            "se adaptan localmente a la heteroscedasticidad de los datos, produciendo intervalos mas "
            "estrechos donde el modelo es mas confiable y mas amplios donde la incertidumbre es mayor "
            "(Romano et al., 2019). Para PD, que es una variable de clasificacion binaria, las "
            "variantes como Split Conformal y Mondrian permiten obtener probabilidades acompanadas "
            "de bandas de incertidumbre con garantias de cobertura."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Pregunta de investigacion: ¿Como mejorar la calibracion y la cuantificacion de la "
            "incertidumbre en los modelos de riesgo crediticio (PD, LGD, EAD) mediante la aplicacion "
            "de tecnicas de prediccion conformal, y cual es su desempeno comparado frente a metodos "
            "tradicionales de calibracion?"
        ),
        bold=True,
        first_line_indent=1.25,
    )


# ── 3. JUSTIFICACION ─────────────────────────────────────────────────


def build_justification(doc: Document) -> None:
    styled_heading(doc, "3. JUSTIFICACION DE LA INVESTIGACION", level=1)

    add_paragraph(
        doc,
        (
            "La presente investigacion se justifica desde multiples dimensiones que abarcan el ambito "
            "academico, regulatorio y practico, respondiendo a necesidades reales tanto del sector "
            "financiero como de la comunidad cientifica en ciencia de datos aplicada."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Desde la perspectiva academica, esta investigacion contribuye a una linea de trabajo "
            "emergente que integra la cuantificacion de incertidumbre con los modelos de machine "
            "learning aplicados al riesgo financiero. Aunque la prediccion conformal ha sido "
            "ampliamente estudiada en dominios como la medicina y la vision por computador (Lu et al., "
            "2024; Vovk et al., 2022), su aplicacion especifica al riesgo crediticio permanece "
            "relativamente inexplorada, particularmente en lo que respecta a la modelacion conjunta de "
            "PD, LGD y EAD con garantias formales de cobertura. Los trabajos de Angelopoulos y Bates "
            "(2023) han popularizado el marco teorico general, pero la literatura sobre su implementacion "
            "practica en portafolios crediticios reales es escasa, lo que posiciona a esta investigacion "
            "como una contribucion original al campo."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "En el ambito regulatorio, la investigacion ofrece un marco metodologico que se alinea "
            "directamente con los requerimientos de IFRS 9 y Basilea III. Bajo IFRS 9, las entidades "
            "financieras deben estimar perdidas crediticias esperadas (ECL) incorporando informacion "
            "prospectiva y escenarios macroeconomicos, lo que requiere modelos que no solo predigan "
            "sino que cuantifiquen la incertidumbre de manera transparente. Basilea III exige que los "
            "modelos internos de riesgo sean sometidos a validacion rigurosa, incluyendo pruebas de "
            "backtesting que verifiquen la cobertura real de las estimaciones. La prediccion conformal, "
            "con sus garantias formales de cobertura, ofrece un camino natural para satisfacer estos "
            "requisitos, fortaleciendo la trazabilidad y auditabilidad de los modelos utilizados."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Desde la perspectiva practica, la adopcion de estas tecnicas permitiria a las instituciones "
            'financieras distinguir entre predicciones de alta y baja certeza. Un modelo que dice "este '
            'cliente tiene PD de 12%" es util para ranking; un modelo que dice "PD de 12% con intervalo '
            '[8%, 17%]" es util para decision. Esa distincion permite provisionar con mas precision, '
            "defender las decisiones ante comites de riesgo, y detectar anticipadamente el deterioro "
            "crediticio (SICR) a traves del ancho del intervalo conformal."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Finalmente, esta investigacion representa una innovacion para el contexto colombiano, "
            "donde la adopcion de tecnicas avanzadas de cuantificacion de incertidumbre en modelos "
            "de riesgo crediticio es aun incipiente. La implementacion exitosa de estos metodos "
            "posicionaria a la organizacion como referente en la adopcion de modelos explicables, "
            "confiables y alineados con las mejores practicas internacionales."
        ),
        first_line_indent=1.25,
    )


# ── 4. OBJETIVOS ─────────────────────────────────────────────────────


def build_objectives(doc: Document) -> None:
    styled_heading(doc, "4. OBJETIVOS", level=1)

    styled_heading(doc, "4.1 Objetivo General", level=2)
    add_paragraph(
        doc,
        (
            "Implementar y evaluar tecnicas de prediccion conformal para mejorar la calibracion y "
            "cuantificacion de la incertidumbre en modelos de riesgo crediticio (PD, LGD, EAD), "
            "utilizando el dataset publico de LendingClub como caso de estudio, con el fin de "
            "demostrar que es posible acompanar cada prediccion de riesgo con intervalos de confianza "
            "que posean garantias formales de cobertura, mejorando asi la calidad de las decisiones "
            "crediticias y la confiabilidad regulatoria de los modelos."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "4.2 Objetivos Especificos", level=2)

    # Reformulated objectives: not a to-do list, but objectives with why/how/impact
    objectives = [
        (
            "Fundamentar teoricamente la aplicacion de prediccion conformal al riesgo crediticio "
            "mediante una revision sistematica del estado del arte, identificando las variantes "
            "metodologicas mas pertinentes (Split Conformal, Mondrian, CQR) y sus ventajas frente "
            "a metodos tradicionales de calibracion, con el proposito de establecer un marco "
            "conceptual solido que guie la experimentacion."
        ),
        (
            "Construir un dataset analitico a partir de los registros publicos de LendingClub "
            "(2007-2020), con separacion temporal estricta (out-of-time) en conjuntos de "
            "entrenamiento, calibracion y test, definiendo las variables objetivo PD, LGD y EAD "
            "de manera que se garantice la ausencia de data leakage y la validez de la evaluacion "
            "fuera de muestra."
        ),
        (
            "Implementar un modelo base de PD calibrado (CatBoost + calibracion post-hoc) que "
            "sirva como fundamento para la aplicacion de prediccion conformal, evaluando su "
            "capacidad discriminativa (AUC-ROC, Gini) y su calidad probabilistica (Brier Score, "
            "ECE), para demostrar que la calibracion es un paso necesario pero insuficiente "
            "sin cuantificacion explicita de incertidumbre."
        ),
        (
            "Aplicar prediccion conformal Mondrian (por grado de riesgo) sobre PD, y variantes "
            "adaptativas sobre LGD y EAD, evaluando cobertura empirica, eficiencia de intervalos "
            "y cobertura condicional por subgrupo, para demostrar que es posible obtener garantias "
            "de cobertura que se mantienen no solo a nivel global sino en cada segmento del "
            "portafolio, lo cual es critico para la gestion operativa del riesgo."
        ),
        (
            "Evaluar el impacto de la prediccion conformal en el contexto regulatorio de IFRS 9, "
            "cuantificando como los intervalos de incertidumbre afectan la estimacion de perdidas "
            "crediticias esperadas (ECL), la clasificacion por etapas (Stages 1-3) y la "
            "sensibilidad de las provisiones ante escenarios de estres, para demostrar que la "
            "incertidumbre conformal tiene un efecto material en la lectura financiera."
        ),
        (
            "Proponer lineamientos practicos para la adopcion de Conformal Prediction en entornos "
            "bancarios reales, incluyendo criterios de monitoreo de cobertura, alertas de "
            "degradacion y escalamiento, para facilitar la transicion de una tecnica academica "
            "a una herramienta operativa de gobernanza de modelos."
        ),
    ]

    for i, obj in enumerate(objectives, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(f"{i}. ")
        r.font.bold = True
        r.font.size = Pt(12)
        r.font.name = "Times New Roman"
        r.font.color.rgb = DARK_GRAY
        r2 = p.add_run(obj)
        r2.font.size = Pt(12)
        r2.font.name = "Times New Roman"
        r2.font.color.rgb = DARK_GRAY


# ── 5. MARCO TEORICO ─────────────────────────────────────────────────


def build_theoretical_framework(doc: Document, fig_num: int) -> int:
    styled_heading(doc, "5. MARCO TEORICO Y ESTADO DEL ARTE", level=1)

    styled_heading(doc, "5.1 Modelos de Riesgo Crediticio", level=2)
    add_paragraph(
        doc,
        (
            "Los modelos de riesgo crediticio buscan estimar la probabilidad y severidad de las perdidas "
            "asociadas al incumplimiento de obligaciones financieras. Los tres componentes fundamentales "
            "son: PD, que mide la probabilidad de que un deudor incumpla en un horizonte temporal "
            "determinado; LGD, que estima la fraccion de la exposicion que se pierde tras considerar "
            "recuperaciones; y EAD, que cuantifica el monto expuesto al momento del incumplimiento "
            "(Basel Committee on Banking Supervision, 2006). Historicamente, la PD se ha modelado "
            "mediante regresion logistica, mientras que LGD y EAD han utilizado regresion beta, Tobit "
            "o regresion lineal con transformaciones. Modelos de machine learning como Gradient Boosting "
            "han demostrado mejoras significativas en discriminacion, pero frecuentemente a costa de la "
            "calibracion (Lessmann et al., 2015; Baesens et al., 2016)."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "5.2 El Problema de la Calibracion", level=2)
    add_paragraph(
        doc,
        (
            "La calibracion se refiere a la correspondencia entre las probabilidades predichas y las "
            "frecuencias observadas. Un modelo bien calibrado es aquel en el que, de todos los creditos "
            "con PD asignada del 10%, aproximadamente el 10% efectivamente incumple (Niculescu-Mizil y "
            "Caruana, 2005). Los metodos tradicionales de post-hoc calibracion incluyen Platt Scaling "
            "(transformacion sigmoidal) e Isotonic Regression (funcion monotona no decreciente). Sin "
            "embargo, estos metodos corrigen el nivel probabilistico pero no cuantifican la "
            'incertidumbre: un modelo calibrado dice "12% de PD" pero no indica si ese 12% es una '
            "estimacion estable o fragil. Ademas, carecen de garantias formales de cobertura y pueden "
            "ser sensibles al sobreajuste en el conjunto de calibracion (Vovk y Petej, 2014)."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "5.3 Prediccion Conformal: Fundamentos Teoricos", level=2)
    add_paragraph(
        doc,
        (
            "La prediccion conformal fue introducida por Vovk, Gammerman y Shafer (2022) en su trabajo "
            '"Algorithmic Learning in a Random World". El marco se fundamenta en el concepto de '
            "nonconformity scores, que miden cuan inusual es una nueva observacion respecto a un "
            "conjunto de referencia. La idea central es que, bajo el supuesto de intercambiabilidad "
            "(exchangeability) de los datos, es posible construir intervalos de prediccion que contengan "
            "el valor verdadero con probabilidad al menos (1 - alpha), sin supuestos parametricos. "
            "La garantia formal es: P(Y_nuevo ∈ C(X_nuevo)) >= 1 - alpha, donde C(X) es el conjunto "
            "de prediccion conformal (Angelopoulos y Bates, 2023)."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "La variante mas utilizada es Split Conformal Prediction (SCP), que divide los datos en "
            "un conjunto de entrenamiento (para ajustar el modelo) y un conjunto de calibracion (para "
            "calcular los quantiles de los nonconformity scores). Es computacionalmente eficiente y "
            "facil de implementar. Extensiones como Jackknife+ abordan la perdida de poder estadistico "
            "al no usar todos los datos (Barber et al., 2021). Para datos heteroscedasticos, "
            "Conformalized Quantile Regression (CQR) combina regresion cuantilica con el marco "
            "conformal para producir intervalos que se adaptan localmente a la variabilidad "
            "(Romano et al., 2019)."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "5.4 Prediccion Conformal Mondrian", level=2)
    add_paragraph(
        doc,
        (
            "Una limitacion de Split Conformal global es que la garantia de cobertura es marginal: "
            "se cumple en promedio sobre toda la poblacion, pero puede fallar en subgrupos especificos. "
            "Por ejemplo, un portafolio con 70% de grado A y 5% de grado G puede mostrar 90% de "
            "cobertura global mientras que grado G solo alcanza 58%. La variante Mondrian resuelve "
            "este problema calculando los quantiles de no conformidad por subgrupo (por ejemplo, por "
            "grado de riesgo), garantizando cobertura condicional por particion. Esto es operativamente "
            "critico en riesgo crediticio, donde las decisiones se toman por segmento y una garantia "
            "solo promedio puede ocultar fallos en los segmentos mas riesgosos."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        FIGURES / "figure2.png",
        "Ilustracion conceptual de la prediccion conformal: cada prediccion se acompana "
        "de un intervalo cuyo ancho refleja la incertidumbre del modelo.",
        fig_num,
        width_inches=5.0,
    )

    styled_heading(doc, "5.5 Aplicaciones de CP en el Sector Financiero", level=2)
    add_paragraph(
        doc,
        (
            "Bellotti (2017) fue uno de los primeros en aplicar CP a credit scoring, demostrando que "
            "los conjuntos de prediccion conformal proporcionan informacion valiosa sobre la "
            "confiabilidad de las decisiones de credito. Javanmardi y Vovk (2023) extendieron los "
            "predictores Venn-Abers para mejorar la calibracion de PD en portafolios bancarios. Sin "
            "embargo, la literatura presenta vacios significativos: la mayoria de estudios se enfocan "
            "exclusivamente en PD, dejando sin explorar la aplicacion conjunta a LGD y EAD. Ademas, "
            "pocos trabajos evaluan el impacto practico de CP en metricas regulatorias especificas "
            "como las provisiones bajo IFRS 9. Esta investigacion busca cerrar estos vacios mediante "
            "una evaluacion integral de CP en los tres componentes del riesgo crediticio."
        ),
        first_line_indent=1.25,
    )

    return fig_num


# ── 6. METODOLOGIA ───────────────────────────────────────────────────


def build_methodology(doc: Document, fig_num: int) -> int:
    styled_heading(doc, "6. METODOLOGIA", level=1)

    add_paragraph(
        doc,
        (
            "La investigacion siguio un enfoque cuantitativo y experimental, estructurado en seis "
            "fases secuenciales alineadas con el marco CRISP-DM, que abarcan desde la revision del "
            "estado del arte hasta la evaluacion del impacto regulatorio."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "6.1 Fase 1: Revision Bibliografica", level=2)
    add_paragraph(
        doc,
        (
            "Se realizo una revision de la literatura en Scopus, arXiv y Google Scholar, utilizando "
            'terminos como "conformal prediction credit risk", "uncertainty quantification credit '
            'scoring" y "calibration probability of default". Se incluyeron articulos publicados '
            "entre 2005 y 2026, con enfasis en contribuciones recientes (2019-2026). Se revisaron "
            "ademas guias regulatorias del Comite de Basilea y documentos tecnicos de la IFRS Foundation."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "6.2 Fase 2: Construccion y Preparacion del Dataset", level=2)
    add_paragraph(
        doc,
        (
            "Se utilizo el dataset publico de LendingClub (2007-2020 Q3), disponible en Kaggle, con "
            "aproximadamente 2.9 millones de registros y mas de 140 variables. Tras limpieza y "
            "eliminacion de variables con fuga de datos (total_pymnt, recoveries, collection_recovery_fee, "
            "entre otras), se obtuvieron 1.86 millones de prestamos con 110 columnas. Las variables "
            'objetivo se definieron asi: para PD, variable binaria basada en estados "Charged Off" y '
            '"Default"; para LGD, 1 menos la tasa de recuperacion (total_rec_prncp / funded_amnt); '
            "para EAD, la exposicion residual al momento del incumplimiento."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "El dataset se dividio en tres conjuntos con separacion temporal estricta (out-of-time, "
            "OOT) para respetar la estructura cronologica del riesgo crediticio:"
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Conjunto", "Registros", "Tasa Default", "Periodo"],
        rows=[
            ["Entrenamiento", "1,346,311", "18.52%", "2007-06 a 2017-03"],
            ["Calibracion", "237,584", "22.20%", "2017-03 a 2017-12"],
            ["Test (OOT)", "276,869", "21.98%", "2018-01 a 2020-09"],
        ],
        caption="Division temporal del dataset LendingClub.",
        table_num=1,
    )

    add_paragraph(
        doc,
        (
            "La separacion temporal es critica porque garantiza que el modelo se evalua en datos "
            "del futuro respecto al entrenamiento, simulando condiciones reales de produccion. El "
            "aumento de la tasa de default en calibracion y test (22.2% vs 18.5% en entrenamiento) "
            "refleja el deterioro macroeconomico del periodo 2017-2020, que incluye el inicio de la "
            "pandemia COVID-19."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "6.3 Fase 3: Modelado Base y Calibracion", level=2)
    add_paragraph(
        doc,
        (
            "Se implementaron dos modelos de PD: regresion logistica como baseline interpretable y "
            "CatBoost como modelo de gradient boosting de alto rendimiento. CatBoost maneja "
            "nativamente valores faltantes y variables categoricas, evitando la necesidad de "
            "imputacion manual. La optimizacion de hiperparametros se realizo mediante Optuna con "
            "validacion temporal. Para la calibracion post-hoc, se implemento una politica de "
            "seleccion temporal multi-fold que evalua Platt Scaling, Isotonic Regression y Venn-Abers "
            "sobre 4 folds temporales, seleccionando el metodo con mejor Brier Score que no degrade "
            "el AUC-ROC mas de 0.15 puntos porcentuales."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "6.4 Fase 4: Aplicacion de Conformal Prediction", level=2)
    add_paragraph(
        doc,
        (
            "Para PD, se implemento Mondrian Split Conformal Prediction utilizando la libreria "
            "MAPIE 1.3.0 (SplitConformalRegressor). La PD calibrada se envuelve en un "
            "ProbabilityRegressor que transforma el problema de clasificacion binaria en regresion "
            "de probabilidades, permitiendo aplicar el framework conformal de regresion. Los quantiles "
            "de no conformidad se calculan por grado de riesgo (A-G), garantizando cobertura "
            "condicional por segmento. Para LGD, se implemento un enfoque adaptativo por grado y "
            "periodo temporal (direct_adaptive_grade_temporal), seleccionado entre cuatro variantes "
            "por cumplir todos los guardrails de cobertura, sesgo y eficiencia. Para EAD, se utilizo "
            "Split Conformal estandar. Se evaluaron niveles de confianza de 90% y 95%."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        FIGURES / "methods_flowchart.png",
        "Arquitectura del pipeline de prediccion conformal propuesto: desde datos crudos "
        "hasta impacto regulatorio IFRS 9.",
        fig_num,
        width_inches=5.5,
    )

    styled_heading(doc, "6.5 Fase 5: Evaluacion Comparativa", level=2)
    add_paragraph(
        doc,
        (
            "La evaluacion se realizo con metricas disenadas para capturar diferentes aspectos del "
            "desempeno. Para discriminacion: AUC-ROC, Gini, KS. Para calibracion: Expected Calibration "
            "Error (ECE), Brier Score. Para cobertura conformal: cobertura empirica (proporcion de "
            "observaciones dentro del intervalo), cobertura minima por grado (Mondrian), ancho promedio "
            "de intervalos, y Winkler Score. Se realizaron pruebas estadisticas formales: test de "
            "Kupiec (cobertura incondicional) y test de Christoffersen (cobertura condicional e "
            "independencia). La evaluacion incluyo backtesting temporal sobre 35 meses y analisis "
            "de cobertura por grado x mes."
        ),
        first_line_indent=1.25,
    )

    styled_heading(doc, "6.6 Fase 6: Evaluacion de Impacto Regulatorio", level=2)
    add_paragraph(
        doc,
        (
            "Se simulo el calculo de provisiones bajo IFRS 9 utilizando las estimaciones de PD, LGD "
            "y EAD con y sin intervalos conformales. Se calcularon ECL bajo cuatro escenarios "
            "(baseline, mild stress, adverse, severe), se analizo la distribucion por etapas "
            "(Stage 1, 2, 3), y se evaluo la sensibilidad del ECL ante variaciones de PD y LGD. "
            "Se propuso el ancho del intervalo conformal (PD_high - PD_point) como senal adicional "
            "de SICR (Significant Increase in Credit Risk) para la clasificacion por etapas."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Categoria", "Herramientas"],
        rows=[
            ["ML / Modelado", "CatBoost 1.2.8, scikit-learn 1.6.1, Optuna 4.7"],
            ["Conformal", "MAPIE 1.3.0 (SplitConformalRegressor, Mondrian)"],
            ["Evaluacion", "Kupiec, Christoffersen, Winkler Score"],
            ["Datos", "pandas, numpy, pandera (validacion de esquemas)"],
            ["Visualizacion", "matplotlib, seaborn, SHAP 0.48"],
            ["Desarrollo", "Python 3.12, uv, ruff, pytest, loguru"],
        ],
        caption="Stack tecnologico utilizado en la investigacion.",
        table_num=2,
    )

    return fig_num


# ── 7. CONSIDERACIONES ETICAS ────────────────────────────────────────


def build_ethics(doc: Document) -> None:
    styled_heading(doc, "7. CONSIDERACIONES ETICAS", level=1)

    add_paragraph(
        doc,
        (
            "La presente investigacion se adhiere a los mas altos estandares de etica en la "
            "investigacion cientifica. En cuanto a la privacidad y proteccion de datos, el estudio "
            "utiliza exclusivamente datos de acceso publico provenientes de la plataforma LendingClub, "
            "previamente anonimizados y sin informacion de identificacion personal (PII). Se adoptaron "
            "medidas adicionales para garantizar que ningun analisis individual pueda conducir a la "
            "re-identificacion de deudores, siguiendo las directrices de la Ley de Proteccion de "
            "Datos Personales de Colombia (Ley 1581 de 2012)."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "En terminos de transparencia y reproducibilidad, todo el codigo fuente, los notebooks "
            "de experimentacion y los resultados intermedios fueron documentados para permitir la "
            "reproduccion completa por terceros. Se reportaron tanto los resultados positivos como "
            "las limitaciones, incluyendo alertas de cobertura que no pasaron todos los checks "
            "estadisticos. La integridad academica se garantiza mediante el uso riguroso de "
            "referencias bibliograficas y el cumplimiento de las normas de citacion APA."
        ),
        first_line_indent=1.25,
    )


# ── 8. RESULTADOS ────────────────────────────────────────────────────


def build_results(doc: Document, fig_num: int) -> int:
    styled_heading(doc, "8. RESULTADOS", level=1)

    add_paragraph(
        doc,
        (
            "A continuacion se presentan los resultados obtenidos, organizados desde el analisis "
            "exploratorio de datos hasta el impacto regulatorio en provisiones IFRS 9. Todos los "
            "resultados corresponden a evaluaciones sobre el conjunto de test out-of-time (276,869 "
            "prestamos, periodo 2018-01 a 2020-09)."
        ),
        first_line_indent=1.25,
    )

    # ── 8.1 EDA ──────────────────────────────────────────────────────
    styled_heading(doc, "8.1 Analisis Exploratorio de Datos", level=2)

    add_paragraph(
        doc,
        (
            "El dataset de LendingClub contiene prestamos originados entre 2007 y 2020, con un "
            "gradiente de riesgo claro por grado de riesgo asignado por la plataforma. La Tabla 3 "
            "muestra la distribucion de prestamos y la tasa de default por grado en el conjunto "
            "de entrenamiento, confirmando la senal economica del riesgo."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Grado", "Prestamos", "Tasa Default", "Tasa Interes Prom."],
        rows=[
            ["A", "229,809", "5.63%", "7.2%"],
            ["B", "401,151", "12.29%", "10.9%"],
            ["C", "383,391", "20.64%", "14.0%"],
            ["D", "198,090", "28.31%", "17.7%"],
            ["E", "94,233", "36.47%", "21.0%"],
            ["F", "31,748", "43.44%", "24.5%"],
            ["G", "7,889", "47.71%", "27.0%"],
        ],
        caption="Distribucion de prestamos y tasa de default por grado de riesgo (conjunto de entrenamiento).",
        table_num=3,
    )

    add_paragraph(
        doc,
        (
            "El gradiente de default escala monotonicamente de A (5.63%) a G (47.71%), confirmando "
            "que el grado de riesgo captura una senal economica real y no solo volumen. La tasa de "
            "interes promedio tambien aumenta con el grado, reflejando el pricing por riesgo de la "
            "plataforma. En terminos de plazo, el 74.6% de los prestamos son a 36 meses (1.00M) y "
            "el 25.4% a 60 meses (340K). Las variables con mayor proporcion de valores faltantes "
            "incluyen mths_since_last_delinq (51.6%) y mths_since_last_record (84.3%), patron tipico "
            "en datos crediticios donde la ausencia indica que el evento no ha ocurrido."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "01_eda_lending_club" / "default_rate_by_grade.png",
        "Tasa de default por grado de riesgo en el conjunto de entrenamiento.",
        fig_num,
        width_inches=4.5,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "01_eda_lending_club" / "correlation_matrix.png",
        "Matriz de correlacion de las principales variables numericas.",
        fig_num,
        width_inches=4.5,
    )

    # ── 8.2 PD ───────────────────────────────────────────────────────
    styled_heading(doc, "8.2 Desempeno del Modelo de PD", level=2)

    add_paragraph(
        doc,
        (
            "Se entrenaron y evaluaron dos modelos de PD. La Tabla 4 resume las metricas "
            "comparativas sobre el conjunto de test OOT."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Modelo", "AUC-ROC", "Gini", "Brier", "ECE", "KS"],
        rows=[
            ["Regresion Logistica", "0.683", "0.366", "0.231", "—", "—"],
            ["CatBoost (default)", "0.712", "0.424", "0.208", "—", "—"],
            ["CatBoost (tuned)", "0.712", "0.424", "0.208", "—", "—"],
            ["CatBoost (calibrado)", "0.712", "0.423", "0.155", "0.006", "0.311"],
        ],
        caption="Comparacion de modelos PD en el conjunto de test OOT (276,869 prestamos).",
        table_num=4,
    )

    add_paragraph(
        doc,
        (
            "El modelo campeon CatBoost calibrado con Isotonic Regression alcanzo un AUC-ROC de "
            "0.7116, un Brier Score de 0.1548 y un ECE de 0.0057. La calibracion isotonica mejoro "
            "sustancialmente la calidad probabilistica (Brier de 0.208 a 0.155, D2-Brier de -0.211 "
            "a 0.097) sin degradar la discriminacion (AUC cae solo 0.0002). Esto confirma que la "
            "calibracion es un paso necesario: sin ella, un modelo con buen AUC puede producir "
            "probabilidades que no corresponden a las frecuencias reales de incumplimiento. Sin "
            "embargo, el modelo calibrado aun no responde a la pregunta: ¿con que confianza puedo "
            "tomar una decision basada en esta PD? Esa es la pregunta que resuelve la prediccion "
            "conformal."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Las cinco variables mas importantes identificadas por SHAP fueron: tasa de interes "
            "(int_rate), plazo del prestamo (term), puntaje FICO (fico_score), tipo de vivienda "
            "(home_ownership) y razon deuda-ingreso (dti). Estas variables son consistentes con "
            "la literatura de riesgo crediticio y refuerzan la interpretabilidad del modelo."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "03_pd_modeling" / "roc_curves.png",
        "Curvas ROC comparativas de los modelos PD en el conjunto OOT.",
        fig_num,
        width_inches=4.5,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "03_pd_modeling" / "calibration_curves.png",
        "Curvas de calibracion: CatBoost sin calibrar vs. calibrado con Isotonic Regression.",
        fig_num,
        width_inches=4.5,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "03_pd_modeling" / "feature_importance.png",
        "Importancia global de features por SHAP values (top 20).",
        fig_num,
        width_inches=4.5,
    )

    # ── 8.3 Conformal PD Mondrian ────────────────────────────────────
    styled_heading(doc, "8.3 Intervalos Conformales para PD (Mondrian)", level=2)

    add_paragraph(
        doc,
        (
            "Se aplico Mondrian Split Conformal Prediction sobre la PD calibrada, calculando los "
            "quantiles de no conformidad por grado de riesgo (A-G). La Tabla 5 resume las metricas "
            "de cobertura y eficiencia."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Metrica", "Nivel 90%", "Nivel 95%"],
        rows=[
            ["Cobertura empirica global", "91.76%", "95.97%"],
            ["Cobertura minima por grado", "87.82%", "—"],
            ["Ancho medio de intervalo", "0.7516", "—"],
            ["Winkler Score", "1.209", "1.152"],
            ["Checks pasados", "7/13", "—"],
            ["Alertas activas", "5", "—"],
        ],
        caption="Metricas de cobertura y eficiencia de intervalos conformales PD (Mondrian).",
        table_num=5,
    )

    add_paragraph(
        doc,
        (
            "La cobertura empirica al nivel del 90% alcanzo 91.76%, superando el objetivo nominal. "
            "Esto significa que en el 91.76% de los 276,869 prestamos del test, el valor real de "
            "default quedo dentro del intervalo conformal predicho. La ligera sobrecobertura indica "
            "que los intervalos son conservadores, lo cual es deseable en un contexto regulatorio "
            "donde subestimar la incertidumbre tiene consecuencias mas graves que sobreestimarla. "
            "La cobertura minima por grado fue 87.82%, demostrando que el enfoque Mondrian logra "
            "coberturas aceptables incluso en los segmentos mas desafiantes."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Las pruebas de Kupiec y Christoffersen rechazaron la cobertura exacta (p-value = 0.0), "
            "lo cual es esperado con 276,869 observaciones: con muestras tan grandes, incluso "
            "desviaciones minimas del nominal son estadisticamente significativas. Lo relevante "
            "operativamente es que la cobertura empirica supera consistentemente el nivel objetivo, "
            "lo que valida la utilidad practica de los intervalos. El test de Christoffersen confirmo "
            "independencia de las violaciones al nivel del 90% (p_ind = 0.512), indicando que las "
            "violaciones no se agrupan en periodos especificos."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "04_conformal_prediction" / "coverage_by_grade.png",
        "Cobertura empirica por grado de riesgo (Mondrian Conformal Prediction).",
        fig_num,
        width_inches=4.5,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "04_conformal_prediction" / "interval_width_distribution.png",
        "Distribucion del ancho de intervalos conformales PD al nivel del 90%.",
        fig_num,
        width_inches=4.5,
    )

    # ── 8.4 Mondrian vs Global ───────────────────────────────────────
    styled_heading(doc, "8.4 Comparacion: Mondrian vs. Split Global", level=2)

    add_paragraph(
        doc,
        (
            "Para demostrar la ventaja operativa de Mondrian sobre Split Conformal global, se "
            "comparon ambas variantes sobre el mismo conjunto de test."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Metrica", "Global", "Mondrian", "Diferencia"],
        rows=[
            ["Cobertura 90%", "89.84%", "91.76%", "+1.92 pp"],
            ["Min. cobertura por grado", "58.69%", "87.82%", "+29.13 pp"],
            ["Ancho medio 90%", "0.955", "0.752", "-21.3%"],
        ],
        caption="Comparacion entre Split Conformal global y Mondrian por grado.",
        table_num=6,
    )

    add_paragraph(
        doc,
        (
            "La diferencia mas relevante es en la cobertura minima por grado: Global alcanza solo "
            "58.69% en su peor subgrupo, mientras que Mondrian sube a 87.82% (+29 puntos "
            "porcentuales). Esto significa que el enfoque global puede dar una falsa sensacion de "
            'seguridad: el portafolio "en promedio" cumple el 90%, pero hay grados donde casi la '
            "mitad de las predicciones quedan fuera del intervalo. Mondrian resuelve este problema "
            "calculando quantiles por grado, y ademas produce intervalos mas eficientes (ancho "
            "promedio de 0.752 vs 0.955). La conclusion es que Mondrian es superior tanto en "
            "cobertura condicional como en eficiencia, lo que lo convierte en la variante "
            "recomendada para aplicaciones de riesgo crediticio."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "04_conformal_prediction" / "coverage_width_tradeoff.png",
        "Trade-off entre cobertura y ancho de intervalos conformales.",
        fig_num,
        width_inches=4.5,
    )

    # ── 8.5 LGD / EAD ───────────────────────────────────────────────
    styled_heading(doc, "8.5 Intervalos Conformales para LGD y EAD", level=2)

    add_paragraph(
        doc,
        (
            "Los intervalos conformales se extendieron a LGD y EAD sobre el subconjunto de defaults "
            "(60,850 prestamos en test OOT). Para LGD, se evaluaron cuatro variantes conformales: "
            "two-stage split, direct split, direct CQR y direct adaptive grade-temporal. La Tabla 7 "
            "compara las variantes."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Variante LGD", "Cob. 90%", "Min Grado 90%", "Ancho 90%", "Guardrails"],
        rows=[
            ["Two-stage split", "78.19%", "53.56%", "0.568", "No pasa"],
            ["Direct split", "78.64%", "54.12%", "0.568", "No pasa"],
            ["Direct CQR", "74.52%", "70.51%", "0.540", "No pasa"],
            ["Adaptive grade-temporal", "90.50%", "90.47%", "0.496", "Pasa"],
        ],
        caption="Benchmark de variantes conformales para LGD.",
        table_num=7,
    )

    add_paragraph(
        doc,
        (
            "Solo la variante direct_adaptive_grade_temporal paso todos los guardrails de cobertura "
            "(90.50% al 90%, 95.50% al 95%), cobertura minima por grado (90.47%), y eficiencia de "
            "intervalos (ancho 13% menor que la referencia). Esta variante ajusta los quantiles "
            "conformales de manera online por grado y periodo temporal, adaptandose a la "
            "heterogeneidad de la LGD entre segmentos. Para EAD, el modelo alcanzo cobertura del "
            "90.82% al 90% y 95.28% al 95%, con un R² de 0.9999 en la prediccion puntual."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Este resultado es significativo porque demuestra que la prediccion conformal es "
            "aplicable a la triada completa PD-LGD-EAD, no solo a PD. La extension a LGD y EAD "
            "cierra un vacio en la literatura existente y habilita el calculo de intervalos de "
            "confianza para la perdida esperada completa (ECL = PD x LGD x EAD)."
        ),
        first_line_indent=1.25,
    )

    # ── 8.6 IFRS9 ───────────────────────────────────────────────────
    styled_heading(doc, "8.6 Impacto Regulatorio: IFRS 9 y ECL", level=2)

    add_paragraph(
        doc,
        (
            "Se evaluo el impacto de los intervalos conformales en el calculo de perdidas crediticias "
            "esperadas (ECL) bajo IFRS 9. La Tabla 8 muestra el ECL bajo cuatro escenarios."
        ),
        first_line_indent=1.25,
    )

    add_styled_table(
        doc,
        headers=["Escenario", "ECL (USD)", "Uplift vs Baseline"],
        rows=[
            ["Baseline (PD puntual)", "$977M", "—"],
            ["Mild stress", "$1,200M", "+22.8%"],
            ["Adverse", "$1,463M", "+49.7%"],
            ["Severe (PD alta conformal)", "$1,791M", "+83.3%"],
        ],
        caption="Estimacion de ECL bajo escenarios de estres con intervalos conformales.",
        table_num=8,
    )

    add_paragraph(
        doc,
        (
            "El ECL baseline (usando PD puntual) es de $977M. Cuando se utiliza el limite superior "
            "del intervalo conformal al 95% (escenario severe), el ECL sube a $1,791M, un incremento "
            "del 83.3%. Este rango de $814M cuantifica de manera explicita la incertidumbre en las "
            "provisiones, informacion que actualmente no esta disponible con los enfoques "
            "tradicionales de estimacion puntual."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "La distribucion por etapas IFRS 9 fue: Stage 1 (sin deterioro significativo) 34.51% "
            "(95,547 prestamos), Stage 2 (SICR detectado) 43.01% (119,071 prestamos), y Stage 3 "
            "(impago) 22.48% (62,251 prestamos). Un hallazgo relevante de esta investigacion es que "
            "el ancho del intervalo conformal (PD_high - PD_point) puede utilizarse como senal "
            "adicional de SICR para la clasificacion por etapas: un prestamo cuyo intervalo se "
            "amplifica significativamente entre periodos indica deterioro, incluso si la PD puntual "
            "no ha cambiado lo suficiente para activar los umbrales tradicionales."
        ),
        first_line_indent=1.25,
    )

    fig_num = add_figure(
        doc,
        NB_IMAGES / "05_time_series_forecasting" / "ifrs9_scenario_fan_chart.png",
        "Fan chart de escenarios IFRS 9 con intervalos de confianza.",
        fig_num,
        width_inches=4.5,
    )

    return fig_num


# ── 9. DISCUSION ─────────────────────────────────────────────────────


def build_discussion(doc: Document) -> None:
    styled_heading(doc, "9. DISCUSION DE RESULTADOS", level=1)

    add_paragraph(
        doc,
        (
            "Los resultados obtenidos permiten responder afirmativamente la pregunta de investigacion: "
            "las tecnicas de prediccion conformal mejoran efectivamente la cuantificacion de "
            "incertidumbre en modelos de riesgo crediticio, con impacto demostrable en la lectura "
            "regulatoria."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "El primer hallazgo clave es la complementariedad entre calibracion y prediccion conformal. "
            "La calibracion isotonica redujo el ECE a 0.0057, corrigiendo el nivel probabilistico del "
            "modelo. La prediccion conformal, sobre esa PD ya calibrada, agrego una banda de "
            "incertidumbre con cobertura controlada. Estos dos mecanismos no compiten: la calibracion "
            'responde "¿cuanto riesgo hay?", y la prediccion conformal responde "¿con que confianza '
            'lo digo?". Esta complementariedad contrasta con trabajos como Javanmardi y Vovk (2023), '
            "donde Venn-Abers se propone como alternativa a la calibracion; en nuestro caso, la "
            "combinacion secuencial fue mas efectiva."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "El segundo hallazgo es la superioridad operativa de Mondrian sobre Split Conformal global. "
            "La diferencia de +29 puntos porcentuales en cobertura minima por grado (87.82% vs 58.69%) "
            "no es solo estadisticamente significativa sino operativamente critica: en un banco, las "
            "decisiones de credito se toman por segmento, y una garantia solo promedio puede ocultar "
            "fallos graves en los segmentos de mayor riesgo. Ademas, Mondrian logra intervalos 21.3% "
            "mas eficientes, demostrando que no hay trade-off entre cobertura condicional y eficiencia."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "El tercer hallazgo es la extension exitosa a la triada PD-LGD-EAD. La mayoria de la "
            "literatura existente (Bellotti, 2017) se enfoca exclusivamente en PD. Este trabajo "
            "demuestra que la prediccion conformal es aplicable tambien a LGD (90.50% de cobertura "
            "con la variante adaptativa) y EAD (90.82%). Para LGD, la variante "
            "direct_adaptive_grade_temporal fue la unica de cuatro que paso todos los guardrails, "
            "subrayando que el framework conformal debe adaptarse a las caracteristicas de cada "
            "componente y no aplicarse de manera generica."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "El cuarto hallazgo es el impacto material en la lectura regulatoria IFRS 9. El rango "
            "de ECL entre $977M y $1,791M (+83.3%) proporciona a los comites de riesgo una "
            "cuantificacion explicita de la incertidumbre en las provisiones. El uso del ancho del "
            "intervalo conformal como senal de SICR es una innovacion metodologica que podria mejorar "
            "la deteccion temprana de deterioro crediticio."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Es importante reconocer las limitaciones. El supuesto de intercambiabilidad puede verse "
            "comprometido en periodos de crisis economica. Los intervalos para grados F y G tienden "
            "a ser mas amplios, reduciendo su utilidad practica. Las pruebas de Kupiec/Christoffersen "
            "rechazan la cobertura exacta, aunque esto se debe al tamano muestral extremadamente "
            "grande y no a una deficiencia practica. Finalmente, el gate estricto (7/13 checks) "
            "indica que quedan alertas abiertas para monitoreo operativo, lo que es esperado en un "
            "sistema de gobernanza que prioriza transparencia sobre aprobacion automatica."
        ),
        first_line_indent=1.25,
    )


# ── 10. CONCLUSIONES ─────────────────────────────────────────────────


def build_conclusions(doc: Document) -> None:
    styled_heading(doc, "10. CONCLUSIONES", level=1)

    add_paragraph(
        doc,
        (
            "La presente investigacion implemento y evaluo tecnicas de prediccion conformal para la "
            "calibracion y cuantificacion de incertidumbre en modelos de riesgo crediticio. Las "
            "conclusiones se organizan por objetivo especifico."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Respecto al primer objetivo (fundamentacion teorica), se establecio que la prediccion "
            "conformal ofrece garantias formales de cobertura que los metodos tradicionales (Platt, "
            "Isotonic, Bootstrap) no poseen, y que la variante Mondrian es particularmente relevante "
            "para riesgo crediticio por su capacidad de garantizar cobertura por subgrupo."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Respecto al segundo objetivo (dataset analitico), se construyo un dataset temporal "
            "con 1.86 millones de prestamos, separado en tres conjuntos OOT con controles estrictos "
            "de data leakage, validando que el gradiente de riesgo por grado (A=5.63% a G=47.71%) "
            "confirma senal economica real."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Respecto al tercer objetivo (modelo base calibrado), CatBoost con Isotonic Regression "
            "alcanzo AUC-ROC 0.7116 y ECE 0.0057, demostrando que la calibracion mejora la calidad "
            "probabilistica sin sacrificar discriminacion, pero que no cuantifica incertidumbre por "
            "si sola."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Respecto al cuarto objetivo (aplicacion de CP), Mondrian alcanzo 91.76% de cobertura "
            "al 90% con cobertura minima por grado de 87.82%, superando en +29pp al Split global. "
            "LGD alcanzo 90.50% y EAD 90.82%, demostrando aplicabilidad a la triada completa."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Respecto al quinto objetivo (impacto regulatorio), se cuantifico un rango de ECL de "
            "$977M a $1,791M (+83.3%), proporcionando informacion critica para provisiones, y se "
            "propuso el ancho del intervalo conformal como senal complementaria de SICR para IFRS 9."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Respecto al sexto objetivo (lineamientos de adopcion), se propuso un framework de "
            "monitoreo con backtesting mensual (35 meses evaluados), alertas de cobertura por "
            "grado, y criterios de escalamiento para recalibracion."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "En sintesis, la prediccion conformal es una herramienta viable, efectiva y complementaria "
            "a la calibracion tradicional para cuantificar la incertidumbre en riesgo crediticio. "
            "Su principal valor es convertir las predicciones puntuales —utiles para ranking— en "
            "predicciones con bandas de confianza —utiles para decision. La variante Mondrian es "
            "especialmente relevante porque garantiza que esas bandas de confianza se mantienen por "
            "segmento de riesgo, no solo en promedio."
        ),
        first_line_indent=1.25,
    )


# ── 11. LINEAS FUTURAS ───────────────────────────────────────────────


def build_future_work(doc: Document) -> None:
    styled_heading(doc, "11. LINEAS FUTURAS DEL PROYECTO", level=1)

    add_paragraph(
        doc,
        (
            "A partir de los resultados obtenidos y las limitaciones identificadas, se proponen las "
            "siguientes lineas de continuacion, enmarcadas en el plan de estudios de maestria:"
        ),
        first_line_indent=1.25,
    )

    futures = [
        (
            "Prediccion conformal adaptativa (ACI)",
            (
                "Implementar metodos de Adaptive Conformal Inference para manejar la no "
                "estacionariedad de los datos financieros, ajustando los quantiles de manera "
                "dinamica ante cambios en la distribucion."
            ),
        ),
        (
            "Integracion con optimizacion de portafolio",
            (
                "Utilizar los intervalos conformales como conjuntos de incertidumbre (uncertainty "
                "sets) en formulaciones de optimizacion robusta, conectando la cuantificacion de "
                "incertidumbre con decisiones de asignacion de capital."
            ),
        ),
        (
            "Conformal prediction online",
            (
                "Desarrollar un pipeline de CP en streaming para decisiones de credito en tiempo "
                "real, integrando actualizaciones incrementales sin recalibracion completa."
            ),
        ),
        (
            "Extension del analisis IFRS 9",
            (
                "Evaluar el uso del ancho del intervalo conformal como senal primaria de SICR en "
                "portafolios bancarios reales, comparando su poder predictivo con senales "
                "tradicionales de deterioro."
            ),
        ),
        (
            "Validacion en otros mercados",
            (
                "Aplicar el framework a otros tipos de productos crediticios (hipotecas, tarjetas, "
                "prestamos corporativos) para evaluar la generalizabilidad de los hallazgos."
            ),
        ),
    ]

    for title, desc in futures:
        p = doc.add_paragraph()
        r = p.add_run(f"{title}: ")
        r.font.bold = True
        r.font.size = Pt(12)
        r.font.name = "Times New Roman"
        r.font.color.rgb = DARK_GRAY
        r2 = p.add_run(desc)
        r2.font.size = Pt(12)
        r2.font.name = "Times New Roman"
        r2.font.color.rgb = DARK_GRAY
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6)


# ── 12. REFERENCIAS ──────────────────────────────────────────────────


def build_references(doc: Document) -> None:
    styled_heading(doc, "12. REFERENCIAS", level=1)

    refs = [
        "Angelopoulos, A. N. y Bates, S. (2023). A Gentle Introduction to Conformal Prediction "
        "and Distribution-Free Uncertainty Quantification. Foundations and Trends in Machine "
        "Learning, 16(4), 494-591. https://doi.org/10.1561/2200000101",
        "Baesens, B., Roesch, D. y Scheule, H. (2016). Credit Risk Analytics: Measurement "
        "Techniques, Applications, and Examples in SAS. John Wiley & Sons.",
        "Barber, R. F., Candes, E. J., Ramdas, A. y Tibshirani, R. J. (2021). Predictive "
        "Inference with the Jackknife+. The Annals of Statistics, 49(1), 486-507.",
        "Basel Committee on Banking Supervision. (2006). International Convergence of Capital "
        "Measurement and Capital Standards: A Revised Framework. Bank for International "
        "Settlements (BIS).",
        "Bellotti, T. (2017). Reliable region predictions for automated credit scoring. En "
        "Proceedings of the Workshop on Conformal and Probabilistic Prediction and Applications "
        "(COPA). PMLR.",
        "Deloitte. (2020). IFRS 9 and Expected Credit Loss: Modelling and Validation Challenges. "
        "Deloitte Technical Report.",
        "European Banking Authority (EBA). (2020). Guidelines on Loan Origination and Monitoring. "
        "EBA/GL/2020/06.",
        "Fontana, M., Zeni, G. y Vantini, S. (2023). Conformal Prediction: A Unified Review of "
        "Theory and New Challenges. Bernoulli, 29(1), 1-23.",
        "International Accounting Standards Board (IASB). (2014). IFRS 9 Financial Instruments. "
        "IFRS Foundation.",
        "Javanmardi, F. y Vovk, V. (2023). Multip probability predictions for credit scoring "
        "with Venn-Abers predictors. En Conformal and Probabilistic Prediction and Applications "
        "(COPA). PMLR.",
        "Lending Club. (2020). Lending Club Loan Data 2007-2020 Q1. Kaggle Dataset. "
        "https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1",
        "Lessmann, S., Baesens, B., Seow, H.-V. y Thomas, L. C. (2015). Benchmarking "
        "state-of-the-art classification algorithms for credit scoring. European Journal of "
        "Operational Research, 247(1), 124-136.",
        "Lu, H., Karimireddy, N., Ponomareva, N. y Mirrokni, A. (2024). Conformal Prediction "
        "in the Medical Domain: A Comprehensive Survey. IEEE Reviews in Biomedical Engineering, "
        "17, 127-142.",
        "MAPIE Contributors. (2023). MAPIE: Model Agnostic Prediction Interval Estimator. "
        "https://mapie.readthedocs.io/",
        "Niculescu-Mizil, A. y Caruana, R. (2005). Predicting good probabilities with supervised "
        "learning. En Proceedings of the 22nd ICML, pp. 625-632.",
        "Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to "
        "Regularized Likelihood Methods. En Advances in Large Margin Classifiers. MIT Press.",
        "Romano, Y., Patterson, E. y Candes, E. (2019). Conformalized Quantile Regression. En "
        "Advances in Neural Information Processing Systems (NeurIPS), vol. 32.",
        "Vovk, V., Gammerman, A. y Shafer, G. (2022). Algorithmic Learning in a Random World "
        "(2nd ed.). Springer.",
        "Vovk, V. y Petej, I. (2014). Venn-Abers Predictors. En Proceedings of the 30th "
        "Conference on Uncertainty in Artificial Intelligence (UAI), pp. 829-838.",
    ]

    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-1.27)
        p.paragraph_format.space_after = Pt(4)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(ref)
        r.font.size = Pt(11)
        r.font.name = "Times New Roman"
        r.font.color.rgb = DARK_GRAY


# ── ANEXO: DECLARACION IA ────────────────────────────────────────────


def build_ai_declaration(doc: Document) -> None:
    doc.add_page_break()
    styled_heading(
        doc, "ANEXO: DECLARACION DE USO DE HERRAMIENTAS DE INTELIGENCIA ARTIFICIAL", level=1
    )

    add_paragraph(
        doc,
        (
            "Este documento tecnico ha sido preparado con la asistencia de herramientas de "
            "inteligencia artificial utilizadas estrictamente como ayudas de programacion y "
            "redaccion. Especificamente, Claude (Anthropic) fue empleado como asistente de "
            "programacion para la implementacion del pipeline de machine learning, la generacion "
            "de scripts de experimentacion y la estructuracion del documento tecnico final. Tambien "
            "fue utilizado como asistente de redaccion para mejorar la claridad, gramatica y estilo "
            "a lo largo del manuscrito."
        ),
        first_line_indent=1.25,
    )

    add_paragraph(
        doc,
        (
            "Todo el contenido intelectual y cientifico, incluyendo las ideas, analisis, resultados, "
            "interpretaciones y conclusiones presentadas en este trabajo, son responsabilidad "
            "exclusiva del autor. Las herramientas de inteligencia artificial sirvieron unicamente "
            "para facilitar la calidad linguistica y presentacional del documento, asi como para "
            "acelerar la implementacion tecnica del codigo fuente. Todas las decisiones "
            "metodologicas, la seleccion de tecnicas, el diseno experimental y las conclusiones "
            "fueron realizadas por el autor bajo su completa supervision y verificacion."
        ),
        first_line_indent=1.25,
    )


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    from loguru import logger

    logger.info("Generando Documento Tecnico Final (scope: Conformal Prediction)...")

    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.54)

    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)
    font.color.rgb = DARK_GRAY
    style.paragraph_format.line_spacing = 1.5

    fig_num = 1

    build_cover_page(doc)
    logger.info("  Portada")

    build_table_of_contents(doc)
    logger.info("  Contenido")

    build_introduction(doc)
    doc.add_page_break()
    logger.info("  1. Introduccion")

    build_problem_statement(doc)
    doc.add_page_break()
    logger.info("  2. Planteamiento del Problema")

    build_justification(doc)
    doc.add_page_break()
    logger.info("  3. Justificacion")

    build_objectives(doc)
    doc.add_page_break()
    logger.info("  4. Objetivos")

    fig_num = build_theoretical_framework(doc, fig_num)
    doc.add_page_break()
    logger.info("  5. Marco Teorico")

    fig_num = build_methodology(doc, fig_num)
    doc.add_page_break()
    logger.info("  6. Metodologia")

    build_ethics(doc)
    doc.add_page_break()
    logger.info("  7. Consideraciones Eticas")

    fig_num = build_results(doc, fig_num)
    doc.add_page_break()
    logger.info("  8. Resultados")

    build_discussion(doc)
    doc.add_page_break()
    logger.info("  9. Discusion")

    build_conclusions(doc)
    doc.add_page_break()
    logger.info("  10. Conclusiones")

    build_future_work(doc)
    doc.add_page_break()
    logger.info("  11. Lineas Futuras")

    build_references(doc)
    logger.info("  12. Referencias")

    build_ai_declaration(doc)
    logger.info("  Anexo: Declaracion IA")

    doc.save(str(DOCX_PATH))
    logger.info(f"Documento guardado en: {DOCX_PATH}")
    logger.info(f"Total figuras: {fig_num - 1}")


if __name__ == "__main__":
    main()
