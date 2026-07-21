#!/usr/bin/env python3
"""Build the second LinkedIn ingest pack for missing Denis Burakov posts.

The first ingest intentionally preserved the original 59-post corpus. This
script creates a separate sub-pack for public posts discovered later through
search-result snippets and public LinkedIn permalink pages. It also records
public LinkedIn articles that should be read as associated sources when a post
only introduces or links to the deeper article.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT_PACK = Path("reports/linkedin_credit_risk_denis_burakov")
SECOND_PACK = ROOT_PACK / "second_ingest"


POST_FIELDS = [
    "n",
    "activity_id",
    "post_url",
    "title",
    "relevance",
    "theme",
    "summary_es",
    "tesis_use",
    "attachment_type",
    "external_links",
    "short_snippet_under_25_words",
]

MANIFEST_FIELDS = [
    "activity_id",
    "asset_id",
    "asset_type",
    "source_url",
    "local_path",
    "page_or_slide_count",
    "ocr_status",
    "checksum",
    "text_extract_status",
    "analytic_memo",
]

INVENTORY_FIELDS = [
    "activity_id",
    "post_url",
    "date",
    "author",
    "post_type",
    "attachment_type",
    "capture_status",
    "completeness_status",
    "blocker_or_next_action",
    "source_status",
]

DISCOVERY_FIELDS = [
    "candidate_kind",
    "activity_id",
    "source_url",
    "title",
    "theme",
    "discovery_source",
    "already_in_first_ingest",
    "capture_priority",
    "why_relevant",
    "stop_condition",
]

ARTICLE_FIELDS = [
    "article_id",
    "source_url",
    "title",
    "parent_activity_id",
    "theme",
    "source_status",
    "why_relevant",
    "stop_condition",
]


MISSING_POSTS = [
    {
        "activity_id": "7447901245949792256",
        "post_url": "https://www.linkedin.com/posts/denisburakov_datascience-machinelearning-nlp-activity-7447901245949792256-Fxkf",
        "title": "You already understand LLMs better than you think",
        "relevance": "Media",
        "theme": "MLE as bridge between scorecards, NLP, and LLMs",
        "summary_es": "Conecta WOE/logistic regression con conteos NLP y entrenamiento de LLMs via maximum likelihood; util como marco pedagogico, no como evidencia de credito.",
        "tesis_use": "Libro Quarto: contextualizar MLE/logit/WOE; Paper Estrella: solo framing si se mantiene fuente secundaria.",
        "attachment_type": "Image/link/article",
        "external_links": "https://lnkd.in/dhxUaTt4",
        "short_snippet_under_25_words": "MLE connects scorecards, word-frequency models, bigrams, and GPT-style training loops.",
        "why_relevant": "Ayuda a explicar por que scorecards, NLP y modelos generativos comparten una base estadistica.",
        "stop_condition": "Cerrar cuando el post, imagen y articulo enlazado queden leidos y clasificados como framing o archivo.",
    },
    {
        "activity_id": "7376148352284901376",
        "post_url": "https://www.linkedin.com/posts/denisburakov_explainable-credit-risk-models-activity-7376148352284901376-Mj2m",
        "title": "Explainable Credit Risk Models",
        "relevance": "Alta",
        "theme": "WOE, logistic regression, boosting, explainability workshop",
        "summary_es": "Deck corto del workshop con WOE, regresion logistica y boosting como puente entre scorecards reguladas y ML interpretable.",
        "tesis_use": "Libro capitulos 05/10/11; Paper 4 como contexto de distilacion, WOE y governance, sin promover claims desde LinkedIn solamente.",
        "attachment_type": "LinkedIn document/deck",
        "external_links": "",
        "short_snippet_under_25_words": "Workshop deck on WOE, logistic regression, ensemble learning, and practical credit-risk explainability.",
        "why_relevant": "Es una sintesis directa de las tecnicas centrales del proyecto.",
        "stop_condition": "Cerrar cuando el deck completo, transcripcion/PDF e imagenes queden leidos y mapeados a destinos del proyecto.",
    },
    {
        "activity_id": "7211748798618824704",
        "post_url": "https://www.linkedin.com/posts/denisburakov_creditriskmodeling-lending-modelriskmanagement-activity-7211748798618824704-BeWw",
        "title": "Credit Scoring with Trended Data",
        "relevance": "Alta",
        "theme": "Trended data, transactional sequences, neural representations",
        "summary_es": "Argumenta que atributos historicos y secuencias transaccionales pueden mejorar Gini al aportar informacion temporal frente a snapshots puntuales.",
        "tesis_use": "Paper 4: limitacion/future-work por falta de secuencias; Libro: capitulo de feature engineering y drift.",
        "attachment_type": "Image/link",
        "external_links": "https://linktr.ee/deburky",
        "short_snippet_under_25_words": "Trended scoring uses historical behavior rather than point-in-time snapshots.",
        "why_relevant": "Marca una frontera importante frente al dataset Lending Club estatico del proyecto.",
        "stop_condition": "Cerrar cuando se archive como limitacion accionable o se convierta en appendix/future-work sin abrir experimento imposible.",
    },
    {
        "activity_id": "7328316894258577408",
        "post_url": "https://www.linkedin.com/posts/denisburakov_poisson-models-activity-7328316894258577408-5klk",
        "title": "Poisson Models",
        "relevance": "Media",
        "theme": "Count models versus default probability models",
        "summary_es": "Distingue modelos de conteo de modelos binarios de default; incluye Fisher scoring y discusion de zero inflation/negative binomial.",
        "tesis_use": "Libro: nota conceptual GLM; Proyecto: no implementar para PD binaria salvo futura linea de eventos repetidos/cobranza.",
        "attachment_type": "Image/carousel/link",
        "external_links": "https://lnkd.in/dnfgP4AJ",
        "short_snippet_under_25_words": "Poisson fits event counts; PD default remains a one-time binary event in most credit settings.",
        "why_relevant": "Evita aplicar GLMs de conteo al problema equivocado y delimita LGD/EAD/cobranza.",
        "stop_condition": "Cerrar cuando se decida si queda como nota pedagogica o como parking para eventos repetidos.",
    },
    {
        "activity_id": "7381584253375025152",
        "post_url": "https://www.linkedin.com/posts/denisburakov_enhancing-explainability-in-credit-scoring-activity-7381584253375025152-ilAM",
        "title": "Enhancing Explainability in Credit Scoring",
        "relevance": "Alta",
        "theme": "Constrained tree scorecards and responsible ML",
        "summary_es": "Presenta blueprint de modelos tree-based restringidos que buscan competir con scorecards sin perder calibracion ni interpretabilidad.",
        "tesis_use": "Paper 4/Paper Estrella: related-work framing; Libro capitulo 11.",
        "attachment_type": "Image/link/GitHub",
        "external_links": "https://lnkd.in/dVjjSfeJ",
        "short_snippet_under_25_words": "Constrained tree-based models as a responsible bridge between scorecards and ML.",
        "why_relevant": "Conecta interpretabilidad, calibracion y regulacion, tres ejes del proyecto.",
        "stop_condition": "Cerrar cuando repo/articulo enlazado quede clasificado y no se promueva como evidencia sin fuente primaria.",
    },
    {
        "activity_id": "7346062624972754944",
        "post_url": "https://www.linkedin.com/posts/denisburakov_woe-log-odds-and-standard-errors-activity-7346062624972754944-3yJB",
        "title": "WOE, Log Odds, and Standard Errors",
        "relevance": "Alta",
        "theme": "WOE uncertainty and logistic standard errors",
        "summary_es": "Propone cuantificar incertidumbre de WOE conectando log-odds por bin con errores estandar de regresion logistica.",
        "tesis_use": "Libro capitulo 05; Paper 4: appendix/prototype de incertidumbre WOE si existe fuente verificable.",
        "attachment_type": "Image/link/GitHub",
        "external_links": "https://lnkd.in/dFgQAEWG | https://lnkd.in/dPYBzzYi",
        "short_snippet_under_25_words": "WOE uncertainty can be tied to familiar logistic-regression standard errors.",
        "why_relevant": "Refuerza WOE como evidencia con incertidumbre, no solo encoding.",
        "stop_condition": "Cerrar cuando se lea post/imagen/repos y se decida si queda como nota o prototipo futuro.",
    },
    {
        "activity_id": "7320341844326797319",
        "post_url": "https://www.linkedin.com/posts/denisburakov_build-explainable-scorecards-with-catboost-activity-7320341844326797319-2NcG",
        "title": "Build explainable scorecards with CatBoost",
        "relevance": "Alta",
        "theme": "CatBoost-to-scorecard extraction",
        "summary_es": "Anuncia soporte CatBoost en xBooster para extraer reglas interpretables de arboles y producir scorecards compatibles con auditoria.",
        "tesis_use": "Paper 4: related work/prototype parking; Libro capitulo 11.",
        "attachment_type": "External article/GitHub",
        "external_links": "https://lnkd.in/djgXGmnJ | https://lnkd.in/dySsE_Z6",
        "short_snippet_under_25_words": "CatBoost rules can be compressed into explainable scorecard-like structures.",
        "why_relevant": "Candidato natural para la linea scorecard-boosting del proyecto.",
        "stop_condition": "Cerrar cuando Medium/GitHub queden leidos y se decida prototype/park/archive.",
    },
    {
        "activity_id": "7155476795474030592",
        "post_url": "https://www.linkedin.com/posts/denisburakov_credit-risk-modeling-with-naive-bayes-activity-7155476795474030592-8HtC",
        "title": "Credit Risk Modeling with Naive Bayes",
        "relevance": "Alta",
        "theme": "WOE as log-likelihood ratio / Naive Bayes scorecarding",
        "summary_es": "Explica WOE como log-likelihood ratio bajo independencia Naive Bayes, con enlaces a boosting NB y pseudo-WOE.",
        "tesis_use": "Libro capitulo 05; Paper 4: nota teorica/appendix si ayuda a defender WOE.",
        "attachment_type": "Image/link/article",
        "external_links": "https://lnkd.in/dsbbHi-M | https://lnkd.in/dcTST2NY | https://linktr.ee/deburky",
        "short_snippet_under_25_words": "WOE can be read as a Naive Bayes log-likelihood ratio.",
        "why_relevant": "Da una lectura probabilistica mas profunda del WOE usado en scorecards.",
        "stop_condition": "Cerrar cuando post, comentarios de recursos y articulo asociado queden triageados.",
    },
    {
        "activity_id": "7366001503611924480",
        "post_url": "https://www.linkedin.com/posts/denisburakov_creditriskmodeling-datascience-machinelearning-activity-7366001503611924480-tg_i",
        "title": "Machine Learning in Internal Credit Risk Models",
        "relevance": "Alta",
        "theme": "ECB/IRB model governance and ML reproducibility",
        "summary_es": "Resume expectativas ECB sobre ML en modelos internos: seeds, orden de observaciones, explainability y conceptual soundness.",
        "tesis_use": "Libro capitulo 10/14/19; Paper Estrella: defensa de governance/reproducibilidad, no champion.",
        "attachment_type": "LinkedIn document/deck/link",
        "external_links": "https://lnkd.in/dbKAXYHU",
        "short_snippet_under_25_words": "ML changes the functional form, not the economic purpose or governance duties of risk models.",
        "why_relevant": "Aporta criterios regulatorios concretos para MLOps, reproducibilidad y validacion.",
        "stop_condition": "Cerrar cuando deck y guia ECB enlazada queden leidos/clasificados como oficial o contexto.",
    },
    {
        "activity_id": "7372524468570689537",
        "post_url": "https://www.linkedin.com/posts/denisburakov_information-theoretic-framework-for-credit-activity-7372524468570689537-eQlT",
        "title": "Information-Theoretic Framework for Credit Risk Modeling",
        "relevance": "Alta",
        "theme": "IV, PSI, Jeffreys divergence, fairness/performance tradeoff",
        "summary_es": "Post sobre paper SSRN que relaciona IV/PSI con divergencia de Jeffreys y errores estandar para governance/fairness.",
        "tesis_use": "Libro capitulo 05/10; Paper 4/Paper Estrella: solo si paper SSRN se etiqueta como preprint y se verifica.",
        "attachment_type": "Image/link/preprint",
        "external_links": "https://lnkd.in/dWnRG7Wu",
        "short_snippet_under_25_words": "IV and PSI are framed as Jeffreys divergence with uncertainty-aware tests.",
        "why_relevant": "Puede fortalecer la trazabilidad teorica de WOE/IV/PSI y fairness.",
        "stop_condition": "Cerrar cuando SSRN/preprint se lea o quede bloqueado con source-status explicito.",
    },
    {
        "activity_id": "7181176150688223232",
        "post_url": "https://www.linkedin.com/posts/denisburakov_python-logisticregression-logisticregression-activity-7181176150688223232-b8mn",
        "title": "Logistic Regression with Fisher Scoring",
        "relevance": "Media",
        "theme": "MLE, Fisher scoring, focal loss, confidence diagnostics",
        "summary_es": "Carousel de implementacion desde cero de logistic regression por Fisher scoring; incluye focal loss y diagnosticos de confianza.",
        "tesis_use": "Libro capitulo 06 o appendix pedagogico; Paper 4: no implementar salvo necesidad didactica.",
        "attachment_type": "Image/carousel/links",
        "external_links": "https://lnkd.in/ebHdg-fB | https://lnkd.in/dGSxaKAS | https://lnkd.in/dzg3g6Z9 | https://lnkd.in/dJfthBa7",
        "short_snippet_under_25_words": "Fisher scoring estimates logistic regression without a manually tuned learning rate.",
        "why_relevant": "Completa el arco MLE/logit que sostiene scorecards y calibracion.",
        "stop_condition": "Cerrar cuando slides y enlaces relacionados queden leidos o archivados como pedagogia.",
    },
    {
        "activity_id": "7173943545752383488",
        "post_url": "https://www.linkedin.com/posts/denisburakov_additive-logistic-regression-activity-7173943545752383488-mnVZ",
        "title": "Additive Logistic Regression",
        "relevance": "Media",
        "theme": "LogitBoost and additive logistic models",
        "summary_es": "Presenta LogitBoost como puente entre regresion logistica y gradient boosting para clasificacion crediticia.",
        "tesis_use": "Libro capitulo 06/11; Paper 4: related-work/parking para boosting scorecards.",
        "attachment_type": "Image/carousel/links",
        "external_links": "https://lnkd.in/edwQq2Rq | https://lnkd.in/ebHdg-fB | https://lnkd.in/dGSxaKAS",
        "short_snippet_under_25_words": "LogitBoost adds logistic components in a boosting framework for classification.",
        "why_relevant": "Conecta logistic regression con boosting sin saltar directo a caja negra.",
        "stop_condition": "Cerrar cuando slides y links queden leidos o archivados como puente conceptual.",
    },
    {
        "activity_id": "7368538277517156352",
        "post_url": "https://www.linkedin.com/posts/denisburakov_hidden-tricks-in-catboost-you-should-know-activity-7368538277517156352-udoZ",
        "title": "Hidden Tricks in CatBoost You Should Know",
        "relevance": "Media",
        "theme": "CatBoost explainability, feature stats, MLflow/SageMaker",
        "summary_es": "Post sobre practicas de CatBoost en riesgo: explainability, feature statistics, texto/embeddings y MLflow en SageMaker.",
        "tesis_use": "Libro capitulo 10/11; Paper 4: solo context/prototype parking si el articulo aporta detalles reproducibles.",
        "attachment_type": "External article/image",
        "external_links": "https://lnkd.in/dCnwZ3tj",
        "short_snippet_under_25_words": "CatBoost risk-modeling practices: explainability, feature statistics, text/embeddings, and MLflow deployment.",
        "why_relevant": "Complementa CatBoost del proyecto con notas de interpretabilidad y deployment.",
        "stop_condition": "Cerrar cuando articulo enlazado quede leido o archivado como no-credit-risk/no-evidence.",
    },
    {
        "activity_id": "7152948849597132801",
        "post_url": "https://www.linkedin.com/posts/denisburakov_logistic-regression-the-two-cultures-activity-7152948849597132801-P7P_?utm_source=share&utm_medium=member_desktop",
        "title": "Logistic Regression: The Two Cultures",
        "relevance": "Media",
        "theme": "Logistic regression cultures, GLM/MLE pedagogy",
        "summary_es": "Post hijo enlazado desde Fisher Scoring y Additive Logistic Regression; cierra la cadena MLE/logit/scorecard.",
        "tesis_use": "Libro capitulo 06 como pedagogia MLE/logit; no paper claim nuevo.",
        "attachment_type": "LinkedIn child post/deck",
        "external_links": "",
        "short_snippet_under_25_words": "Linked child source for the logistic-regression/MLE thread.",
        "why_relevant": "Completa el hilo logistico que conecta estimacion, scorecards y calibracion.",
        "stop_condition": "Cerrar cuando el post y deck queden leidos y clasificados como pedagogia o archivo.",
    },
    {
        "activity_id": "7168870006380740608",
        "post_url": "https://www.linkedin.com/posts/denisburakov_log-loss-activity-7168870006380740608-Qgbq?utm_source=share&utm_medium=member_desktop",
        "title": "Log Loss",
        "relevance": "Media",
        "theme": "Log loss, scoring rules, classification probability quality",
        "summary_es": "Post hijo enlazado desde Additive Logistic Regression; cierra la cadena log-loss/logitboost/calibracion.",
        "tesis_use": "Libro capitulo 06: log loss junto con Brier/ECE/AUC; no paper claim nuevo.",
        "attachment_type": "LinkedIn child post/deck",
        "external_links": "",
        "short_snippet_under_25_words": "Linked child source for log-loss and probability-quality pedagogy.",
        "why_relevant": "Refuerza que los modelos PD deben evaluar calidad probabilistica, no solo ranking.",
        "stop_condition": "Cerrar cuando el post y deck queden leidos y mapeados a metricas de capitulo 06.",
    },
]


ARTICLE_CANDIDATES = [
    {
        "article_id": "article_designing_ai_underwriters",
        "source_url": "https://www.linkedin.com/pulse/designing-ai-underwriters-denis-burakov-j5qgf",
        "title": "Designing AI Underwriters",
        "parent_activity_id": "",
        "theme": "AI underwriting, multi-agent decision support, human-in-the-loop governance",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Puede informar el marco de gobernanza y auditoria de decisiones asistidas por IA sin alterar el champion.",
        "stop_condition": "Cerrar tras lectura completa, separando idea de producto/contexto de evidencia academica.",
    },
    {
        "article_id": "article_measuring_calibration_accuracy",
        "source_url": "https://www.linkedin.com/pulse/measuring-calibration-accuracy-modern-pd-models-denis-burakov",
        "title": "Measuring Calibration Accuracy of Modern PD Models",
        "parent_activity_id": "",
        "theme": "PD calibration, ECE, Brier/log-loss, reliability diagrams",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Complementa los posts de Brier/ECE con una explicacion larga y referencias.",
        "stop_condition": "Cerrar tras capturar referencias utiles; no usar como fuente primaria academica.",
    },
    {
        "article_id": "article_designing_credit_scoring_systems_ml_components",
        "source_url": "https://www.linkedin.com/pulse/designing-credit-scoring-systems-ml-components-denis-burakov",
        "title": "Designing Credit Scoring Systems with ML Components",
        "parent_activity_id": "",
        "theme": "Modular credit scoring, submodels, calibration, monitoring",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Muy alineado con separacion score/calibrator/policy y modulos en el libro.",
        "stop_condition": "Cerrar cuando se mapee a gobierno/arquitectura o se archive como contexto.",
    },
    {
        "article_id": "article_understanding_lgd_risk",
        "source_url": "https://www.linkedin.com/pulse/understanding-lgd-risk-denis-burakov",
        "title": "Understanding LGD Risk",
        "parent_activity_id": "",
        "theme": "LGD models, workout LGD, direct/indirect approaches, LGD metrics",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Puede alimentar Paper 4 LGD appendix/probabilistic bins y separar PD/LGD/EAD scope.",
        "stop_condition": "Cerrar cuando los conceptos LGD queden resumidos con fuente-status y sin claim publico no verificado.",
    },
    {
        "article_id": "article_validating_tree_based_risk_models",
        "source_url": "https://www.linkedin.com/pulse/validating-tree-based-risk-models-denis-burakov-wpccf",
        "title": "Validating Tree-Based Risk Models",
        "parent_activity_id": "",
        "theme": "GBDT validation, monitoring, chi-square tests, model risk",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Aporta lenguaje de validacion y monitoreo para modelos tree-based regulados.",
        "stop_condition": "Cerrar cuando se clasifique como governance context o appendix; sin sustituir referencias oficiales.",
    },
    {
        "article_id": "article_validating_new_generation_credit_risk_models",
        "source_url": "https://www.linkedin.com/pulse/validating-new-generation-credit-risk-models-denis-burakov",
        "title": "Validating New Generation Credit Risk Models",
        "parent_activity_id": "",
        "theme": "Model validation, challengers, parsimony, complexity, Optuna",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Aporta checklist de validacion para Paper Estrella/Paper 4 governance.",
        "stop_condition": "Cerrar tras separar ideas de validacion de fuentes oficiales y papers revisados.",
    },
    {
        "article_id": "article_scorecarding_naive_bayes",
        "source_url": "https://www.linkedin.com/pulse/scorecarding-na%C3%AFve-bayes-denis-burakov-0aosf",
        "title": "Scorecarding with Naive Bayes",
        "parent_activity_id": "7155476795474030592",
        "theme": "WOE-Naive Bayes scorecarding",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Es el articulo largo asociado al post Naive Bayes/WOE.",
        "stop_condition": "Cerrar al mapear WOE/NB a capitulo 05 y decidir si appendix teorico vale la pena.",
    },
    {
        "article_id": "article_leveraging_profit_scoring",
        "source_url": "https://www.linkedin.com/pulse/leveraging-profit-scoring-digital-loan-underwriting-denis-burakov",
        "title": "Leveraging Profit Scoring in Digital Loan Underwriting",
        "parent_activity_id": "",
        "theme": "Profit scoring, ARR, Bondora, risk-adjusted returns",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Potencial para capitulo 09 sobre decisiones economicas, aunque dataset/profit target difiere.",
        "stop_condition": "Cerrar cuando se decida si queda como future-work/portfolio economics o archivo.",
    },
    {
        "article_id": "article_balancing_risk_and_profit",
        "source_url": "https://www.linkedin.com/pulse/balancing-risk-profit-denis-burakov-3sixe",
        "title": "Balancing Risk and Profit",
        "parent_activity_id": "",
        "theme": "Profit-based credit models, risk rating, OptBinning",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Puede fortalecer el lenguaje de trade-off riesgo/retorno del capitulo 09.",
        "stop_condition": "Cerrar como context/future-work salvo que aporte fuente academica reusable.",
    },
    {
        "article_id": "article_benchmarking_pd_models",
        "source_url": "https://www.linkedin.com/pulse/benchmarking-pd-models-denis-burakov",
        "title": "Benchmarking PD Models",
        "parent_activity_id": "",
        "theme": "PD benchmarking, Gini caveats, cutoffs, model comparison",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Directamente alineado con capitulo 06 y guardrails de comparacion.",
        "stop_condition": "Cerrar tras extraer solo lenguaje de context y referencias externas.",
    },
    {
        "article_id": "article_exploring_scorecard_boosting",
        "source_url": "https://www.linkedin.com/pulse/exploring-interpretable-scorecard-boosting-denis-burakov",
        "title": "Exploring Interpretable Scorecard Boosting",
        "parent_activity_id": "7320341844326797319",
        "theme": "Scorecard boosting and WOE LR challengers",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Articulo largo para la linea xBooster/CatBoost/scorecard boosting.",
        "stop_condition": "Cerrar al decidir si alimenta related work/prototype parking.",
    },
    {
        "article_id": "article_unlocking_lending_profitability",
        "source_url": "https://www.linkedin.com/pulse/unlocking-lending-profitability-risk-modeling-denis-burakov",
        "title": "Unlocking Lending Profitability with Risk Modeling",
        "parent_activity_id": "",
        "theme": "Risk-based pricing and lending profitability",
        "source_status": "linkedin_article_public_pending_capture",
        "why_relevant": "Contexto para decisiones economicas del capitulo 09.",
        "stop_condition": "Cerrar como context/future-work o fuente de referencias externas.",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def known_activity_ids() -> set[str]:
    ids: set[str] = set()
    for rel in ("data/posts_index.csv", "data/external_linkedin_child_post_backlog.csv"):
        for row in read_csv(ROOT_PACK / rel):
            if row.get("activity_id"):
                ids.add(row["activity_id"])
    return ids


def post_index_rows(known_ids: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, post in enumerate(
        (p for p in MISSING_POSTS if p["activity_id"] not in known_ids), start=1
    ):
        rows.append(
            {field: str(idx) if field == "n" else post.get(field, "") for field in POST_FIELDS}
        )
    return rows


def discovery_rows(known_ids: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for post in MISSING_POSTS:
        rows.append(
            {
                "candidate_kind": "linkedin_post",
                "activity_id": post["activity_id"],
                "source_url": post["post_url"],
                "title": post["title"],
                "theme": post["theme"],
                "discovery_source": "public_search_result_or_public_permalink_snippet_2026-05-21",
                "already_in_first_ingest": "yes" if post["activity_id"] in known_ids else "no",
                "capture_priority": post["relevance"],
                "why_relevant": post["why_relevant"],
                "stop_condition": post["stop_condition"],
            }
        )
    for article in ARTICLE_CANDIDATES:
        rows.append(
            {
                "candidate_kind": "linkedin_article",
                "activity_id": article["article_id"],
                "source_url": article["source_url"],
                "title": article["title"],
                "theme": article["theme"],
                "discovery_source": "public_search_result_or_post_external_link_2026-05-21",
                "already_in_first_ingest": "no",
                "capture_priority": "Alta"
                if "validation" in article["theme"].lower()
                or "calibration" in article["theme"].lower()
                else "Media",
                "why_relevant": article["why_relevant"],
                "stop_condition": article["stop_condition"],
            }
        )
    return rows


def manifest_rows(posts: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for post in posts:
        raw_links = [
            item.strip() for item in post.get("external_links", "").split("|") if item.strip()
        ]
        for link_idx, link in enumerate(raw_links, start=1):
            rows.append(
                {
                    "activity_id": post["activity_id"],
                    "asset_id": f"{post['activity_id']}_external_{link_idx:02d}",
                    "asset_type": "external_link",
                    "source_url": link,
                    "local_path": "",
                    "page_or_slide_count": "",
                    "ocr_status": "not_applicable_until_source_downloaded",
                    "checksum": "",
                    "text_extract_status": "pending_external_source_review",
                    "analytic_memo": "Resolve and read the canonical external source before promoting claims.",
                }
            )
    return rows


def inventory_rows(posts: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "activity_id": post["activity_id"],
            "post_url": post["post_url"],
            "date": "",
            "author": "Denis Burakov",
            "post_type": "linkedin_activity_post_second_ingest",
            "attachment_type": post["attachment_type"],
            "capture_status": "pending_public_permalink_capture",
            "completeness_status": "pending",
            "blocker_or_next_action": "Capture public permalink assets, resolve links, and write analytic decision.",
            "source_status": "linkedin_member_post_second_ingest_discovered_publicly",
        }
        for post in posts
    ]


def write_docs(post_count: int, article_count: int) -> None:
    docs_dir = ROOT_PACK / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    plan = f"""# Second LinkedIn Ingest Plan - 2026-05-21

## Scope

This second ingest preserves the closed first corpus and creates a separate
sub-pack at `reports/linkedin_credit_risk_denis_burakov/second_ingest/`.

- Missing public LinkedIn posts queued: {post_count}
- Public LinkedIn articles queued as associated sources: {article_count}
- Discovery basis: public search-result snippets, public LinkedIn permalink
  pages, and prior first-ingest blocker audit.

## Rules

- Public pages only; no fake accounts, captcha bypass, stealth, rate evasion, or
  browser-session extraction.
- LinkedIn-only material can inform project framing and implementation ideas but
  cannot promote paper/book claims by itself.
- Academic, official, preprint, GitHub, blog, Medium, and LinkedIn-only sources
  remain separated in the logs.
- Stop each row when post text, attachments, external links, and article/source
  status are either read or assigned a concrete blocker.

## Stop Condition

The second ingest closes when every queued post/article has a capture/read
status, every link has a resolved or blocked source row, and every concept is
mapped to one of: promote, append, prototype, park, archive, or blocked.
"""
    (docs_dir / "second_ingest_plan_2026-05-21.md").write_text(plan, encoding="utf-8")


def main() -> None:
    known_ids = known_activity_ids()
    posts = post_index_rows(known_ids)
    write_csv(
        ROOT_PACK / "data" / "second_ingest_discovery_candidates.csv",
        discovery_rows(known_ids),
        DISCOVERY_FIELDS,
    )
    write_csv(SECOND_PACK / "data" / "posts_index.csv", posts, POST_FIELDS)
    write_csv(
        SECOND_PACK / "data" / "attachment_manifest.csv", manifest_rows(posts), MANIFEST_FIELDS
    )
    write_csv(
        SECOND_PACK / "data" / "linkedin_corpus_inventory.csv",
        inventory_rows(posts),
        INVENTORY_FIELDS,
    )
    write_csv(SECOND_PACK / "data" / "article_candidates.csv", ARTICLE_CANDIDATES, ARTICLE_FIELDS)
    write_docs(len(posts), len(ARTICLE_CANDIDATES))
    print(
        f"Second ingest pack initialized: {len(posts)} missing posts, {len(ARTICLE_CANDIDATES)} article sources"
    )


if __name__ == "__main__":
    main()
