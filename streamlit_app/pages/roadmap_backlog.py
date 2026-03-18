"""Operational roadmap and backlog page for promoted champion + next sessions."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from streamlit_app.components.release_governance import render_release_governance
from streamlit_app.components.story_shell import render_key_takeaway, render_page_header
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.utils import PROJECT_ROOT, load_official_baseline_registry, try_load_json

DOCS_DIR = PROJECT_ROOT / "docs"
BACKLOG_PRIMARY = DOCS_DIR / "backlog-papers-unified.md"
BACKLOG_LEGACY = DOCS_DIR / "backlog-13-03.md"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _comparison_payload(run_tag: str) -> dict:
    path = PROJECT_ROOT / "reports" / "run_comparisons" / run_tag / "comparison.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _render_snapshot_cards() -> None:
    baseline = load_official_baseline_registry()
    official_tag = str(baseline.get("official_run_tag", "N/D") or "N/D")
    model_comp = try_load_json("model_comparison", directory="data", default={})
    fairness = try_load_json("fairness_audit_status", directory="models", default={})
    governance = try_load_json("governance_status", directory="models", default={})
    champion_policy = try_load_json("champion_portfolio_policy", directory="models", default={})
    comparison = _comparison_payload(official_tag) if official_tag != "N/D" else {}
    final_metrics = model_comp.get("final_test_metrics", {})
    policy = champion_policy.get("selected_policy", {})

    cols = st.columns(4)
    cols[0].metric("Baseline oficial", official_tag)
    cols[1].metric("AUC PD", f"{float(final_metrics.get('auc', 0.0)):.4f}")
    cols[2].metric(
        "Fairness",
        "PASS" if bool(fairness.get("overall_pass", False)) else "WARN",
    )
    cols[3].metric(
        "Gates globales",
        "PASS" if bool(comparison.get("overall_pass", False)) else "REVISION",
    )

    cols2 = st.columns(4)
    cols2[0].metric("Calibración", str(model_comp.get("best_calibration", "N/D")))
    cols2[1].metric("Trials HPO", str(model_comp.get("optuna_n_trials", "N/D")))
    cols2[2].metric("Threshold fairness", str(fairness.get("prediction_threshold", "N/D")))
    cols2[3].metric("Policy champion", str(policy.get("policy_mode", "N/D")))

    if comparison.get("conformal_statistical_warning", False):
        st.warning(
            "El champion oficial ya está promovido, pero conformal PD conserva warning estadístico "
            "en Kupiec/Christoffersen. Eso sigue abierto en el backlog de research."
        )
    if bool(governance.get("warnings", {}).get("warn_c2st", False)) or bool(
        governance.get("warnings", {}).get("warn_distribution_tests", False)
    ):
        st.info(
            "Governance pasa operativamente, pero la narrativa de drift/c2st todavía requiere "
            "trabajo metodológico y editorial."
        )


def _render_promotions() -> None:
    st.markdown("## Qué quedó montado y promovido")
    st.markdown(
        """
- `baseline oficial`: `champion-2026-03-12-mega-definitive`
- `PD core`: CatBoost tuned + calibrated
- `calibración oficial vigente`: Venn-Abers
- `portfolio champion`: risk_tolerance 0.18 + capped_blended_uncertainty
- `survival`: RSF mejorado y promovido
- `fairness`: 6/6 atributos pasan con threshold operativo 0.35
- `governance`: PASS operativo
- `LGD/EAD conformal`: promovido
- `PD conformal`: promovido operativamente con warning estadístico
"""
    )


st.title("🗂️ Roadmap y Backlog")
st.caption(
    "Vista de control para lo ya promovido en el champion actual y la agenda pendiente de "
    "trabajo por pipeline."
)

page_contract = get_page_contract("roadmap_backlog")
render_page_header(page_contract)
render_key_takeaway(
    "Esta página concentra el estado oficial del champion promovido y el backlog vivo; úsala "
    "como referencia única entre sesiones antes de reabrir cambios metodológicos."
)

with st.container(border=True):
    st.markdown("#### Diferencia frente a `Backlog Papers y Quarto`")
    st.markdown(
        """
- **Esta página (`roadmap_backlog`)** prioriza el trabajo técnico-operativo por pipeline:
  `champion_search`, `canonical_rebuild` e `insights_factory`.
- **La página `Backlog Papers y Quarto`** reorganiza esos pendientes en función de papers,
  paper estrella, journal/conference y migración editorial a Quarto.
- Regla práctica:
  si la pregunta es **“qué toca mejorar en el sistema”**, usa esta página;
  si la pregunta es **“qué bloquea cada paper o el libro”**, usa la página de papers.
"""
    )

render_release_governance(show_title=True)
_render_snapshot_cards()
_render_promotions()

primary_text = _read_text(BACKLOG_PRIMARY)

st.markdown("## Documentos de trabajo")
if primary_text:
    st.download_button(
        "Descargar backlog-papers-unified.md",
        data=primary_text,
        file_name="backlog-papers-unified.md",
        mime="text/markdown",
    )
    st.markdown(primary_text)
else:
    st.error("No se pudo cargar `docs/backlog-papers-unified.md`.")

with st.expander("Backlog histórico (backlog-13-03.md — DEPRECATED)", expanded=False):
    legacy_text = _read_text(BACKLOG_LEGACY)
    if legacy_text:
        st.markdown(legacy_text)
    else:
        st.info("Archivo histórico no encontrado.")

with st.expander("Cómo usar esta página entre sesiones"):
    st.markdown(
        """
1. Validar primero el baseline oficial y los gates de promoción.
2. Leer el bloque de promociones para no reabrir decisiones ya congeladas.
3. El backlog unificado (`backlog-papers-unified.md`) organiza los pendientes por paper y por Quarto.
4. El backlog histórico `backlog-13-03.md` está disponible como referencia pero está superseded.
"""
    )
