"""Active Streamlit companion surface for the local thesis lab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanionLabSpec:
    page_id: str
    path: str
    title: str
    icon: str
    default: bool = False


ACTIVE_COMPANION_LABS: tuple[CompanionLabSpec, ...] = (
    CompanionLabSpec(
        page_id="model_laboratory",
        path="pages/model_laboratory.py",
        title="PD & Uncertainty Lab",
        icon=":material/science:",
        default=True,
    ),
    CompanionLabSpec(
        page_id="model_interpretability",
        path="pages/model_interpretability.py",
        title="Explainability Explorer",
        icon=":material/psychology:",
    ),
    CompanionLabSpec(
        page_id="portfolio_optimizer",
        path="pages/portfolio_optimizer.py",
        title="Portfolio & IFRS9 Simulator",
        icon=":material/account_balance:",
    ),
    CompanionLabSpec(
        page_id="causal_intelligence",
        path="pages/causal_intelligence.py",
        title="Causal Policy Lab",
        icon=":material/genetics:",
    ),
    CompanionLabSpec(
        page_id="data_story",
        path="pages/data_story.py",
        title="Data Explorer",
        icon=":material/bar_chart:",
    ),
)


ACTIVE_COMPANION_PAGE_IDS = tuple(lab.page_id for lab in ACTIVE_COMPANION_LABS)
