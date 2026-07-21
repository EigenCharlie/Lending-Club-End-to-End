"""Cross-surface scientific and editorial boundary checks."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK = REPO_ROOT / "book"
EXTERNAL_CONTRACT = REPO_ROOT / "docs/research/crpto_external_contract_2026-07-20.yml"
PAPER2_CONTRACT = REPO_ROOT / "reports/paper_material/paper2/paper2_claim_contract.yml"
PAPER4_FINDINGS = (
    REPO_ROOT / "reports/paper_material/paper4/tables/paper4_current_official_findings.csv"
)
SCOPE_INCLUDE = "_scientific-scope-contract.qmd"
PAPER2_CONTRACT_LINK = "paper2_claim_contract.yml"


def _active_chapters() -> list[Path]:
    config = yaml.safe_load((BOOK / "_quarto.yml").read_text(encoding="utf-8"))

    def walk(items: list[object]) -> list[str]:
        paths: list[str] = []
        for item in items:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                nested = item.get("chapters")
                if isinstance(nested, list):
                    paths.extend(walk(nested))
        return paths

    return [BOOK / path for path in walk(config["book"]["chapters"]) if path.endswith(".qmd")]


def test_obsolete_core_book_entrypoints_are_removed() -> None:
    for path in (
        BOOK / "_quarto-core.yml",
        BOOK / "index-core.qmd",
        REPO_ROOT / "scripts/serve_book_core.py",
    ):
        assert not path.exists(), path

    for registry in (
        REPO_ROOT / "configs/pipeline_registry/script_role_registry.yaml",
        REPO_ROOT / "models/pipeline_registry/script_role_registry.json",
    ):
        assert "serve_book_core.py" not in registry.read_text(encoding="utf-8")


def test_book_build_is_dated_reproducibly_and_has_valid_figure_ids() -> None:
    config = (BOOK / "_quarto.yml").read_text(encoding="utf-8")
    assert "date: today" not in config.casefold()

    malformed: list[tuple[Path, str]] = []
    for path in _active_chapters():
        assert path.is_file(), path
        for match in re.findall(r"\{#fig[^}\s]*", path.read_text(encoding="utf-8")):
            if not re.fullmatch(r"\{#fig-[A-Za-z0-9_-]+", match):
                malformed.append((path, match))
    assert malformed == []


def test_high_risk_book_surfaces_import_the_scientific_scope_contract() -> None:
    high_risk = (
        "chapters/04-pipeline-overview/index.qmd",
        "chapters/05-feature-engineering/05c-feature-contract.qmd",
        "chapters/06-pd-modeling/06b-catboost-tuned.qmd",
        "chapters/06-pd-modeling/06c-calibration-selection.qmd",
        "chapters/06-pd-modeling/06d-model-comparison-champion.qmd",
        "chapters/07-conformal/07d-backtest-monitoring.qmd",
        "chapters/08-time-survival-causal/08a-survival-analysis.qmd",
        "chapters/10-ifrs9-governance/index.qmd",
    )
    for relative in high_risk:
        text = (BOOK / relative).read_text(encoding="utf-8")
        assert SCOPE_INCLUDE in text, relative

    scope = (BOOK / "includes/_scientific-scope-contract.qmd").read_text(encoding="utf-8")
    for phrase in (
        "resultado observado $Y$",
        "no** son intervalos de confianza para la PD",
        "Paper 2 queda `parked_ifrs9`",
        "Paper 4 sigue como living lab sin promoción",
    ):
        assert phrase in scope


def test_every_active_scientific_page_imports_a_current_claim_boundary() -> None:
    non_scientific_or_reference_only = {
        "chapters/00a-dedication.qmd",
        "chapters/A-notebook-atlas.qmd",
        "chapters/D-configuration-reference.qmd",
    }
    paper2_pages = {
        "chapters/10-ifrs9-governance/10a-ecl-calculation.qmd",
        "chapters/10-ifrs9-governance/10c-sicr-conformal-signal.qmd",
        "chapters/15-paper-ifrs9/index.qmd",
        "chapters/15-paper-ifrs9/15a-introduction.qmd",
        "chapters/15-paper-ifrs9/15b-methodology.qmd",
        "chapters/15-paper-ifrs9/15c-results.qmd",
        "chapters/15-paper-ifrs9/15d-discussion.qmd",
    }

    for path in _active_chapters():
        relative = path.relative_to(BOOK).as_posix()
        text = path.read_text(encoding="utf-8")
        if relative in non_scientific_or_reference_only:
            continue
        if relative in paper2_pages:
            assert PAPER2_CONTRACT_LINK in text, relative
        else:
            assert SCOPE_INCLUDE in text, relative


def test_crpto_paper2_and_paper4_statuses_cannot_collapse_into_one_claim() -> None:
    external = yaml.safe_load(EXTERNAL_CONTRACT.read_text(encoding="utf-8"))
    paper2 = yaml.safe_load(PAPER2_CONTRACT.read_text(encoding="utf-8"))
    findings = pd.read_csv(PAPER4_FINDINGS)
    f02 = findings.loc[findings["finding_id"].eq("F02")].iloc[0]
    f06 = findings.loc[findings["finding_id"].eq("F06")].iloc[0]
    f08 = findings.loc[findings["finding_id"].eq("F08")].iloc[0]
    f13 = findings.loc[findings["finding_id"].eq("F13")].iloc[0]

    assert external["scientific_contract"]["prediction_target"] == "observed_binary_outcome_Y"
    assert external["scientific_contract"]["selected_policy"] is None
    assert paper2["status"] == "parked_ifrs9"
    assert f02["evidence_artifact"] == "docs/research/crpto_external_contract_2026-07-20.yml"
    assert f02["status"] == "external_contract"
    assert f06["status"] == "diagnostic_only"
    assert "teacher-cost" in f06["finding"]
    assert "No realized-loss" in f06["claim_boundary"]
    assert f08["status"] == "diagnostic_negative_audit"
    assert "does not identify PD, ECL, SICR or staging" in f08["finding"]
    assert f13["status"] == "diagnostic_negative_audit"
    assert "may not reuse" in f13["official_claim"]


def test_high_authority_state_files_expose_the_july_supersession() -> None:
    session = (REPO_ROOT / "SESSION_STATE.md").read_text(encoding="utf-8")
    old_ledger = (
        REPO_ROOT / "docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md"
    ).read_text(encoding="utf-8")

    assert "Last updated: 2026-07-20" in session
    assert "no selected learner or policy" in session
    assert "Paper 2 | `parked_ifrs9`" in session
    assert "SUPERSEDED FOR SCIENTIFIC CLAIMS (2026-07-20)" in old_ledger


def test_current_facing_engineering_docs_separate_runtime_from_scientific_authority() -> None:
    expected_markers = {
        "docs/PROJECT_JUSTIFICATION.md": "execution graph",
        "docs/RUNBOOK.md": "RUNTIME REPRODUCIBILITY SCOPE (2026-07-20)",
        "docs/MODEL_RISK_MANAGEMENT.md": "HISTORICAL INTERNAL CONTROL DOCUMENT",
        "docs/conformal_prediction_README.md": "LEGACY ENGINEERING NOTE",
    }
    for relative, marker in expected_markers.items():
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert marker in text, relative

    rationale = (REPO_ROOT / "docs/PROJECT_JUSTIFICATION.md").read_text(encoding="utf-8")
    for phrase in (
        "observed binary outcome $Y$",
        "none is a current selected policy",
        "Paper 2 therefore remains",
        "bounded living lab without promotion",
    ):
        assert phrase in rationale


def test_runtime_conformal_api_states_the_binary_outcome_boundary() -> None:
    source = (REPO_ROOT / "src/models/conformal.py").read_text(encoding="utf-8")
    normalized = " ".join(source.split())
    assert "future binary outcome ``Y``" in normalized
    assert "not confidence coverage" in normalized
    assert "ECL, SICR status, or a selected downstream" in normalized
