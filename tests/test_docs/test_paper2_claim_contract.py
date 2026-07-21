"""Claim guardrails for the parked Paper 2 IFRS9-inspired note."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "reports/paper_material/paper2/paper2_claim_contract.yml"
CONTRACT_LINK = "../../../reports/paper_material/paper2/paper2_claim_contract.yml"
LEGACY_ARCHIVE = REPO_ROOT / "reports/paper_material/paper2/archive/legacy_non_citable"
LEGACY_NOTEBOOK = REPO_ROOT / "notebooks/11_paper2_ifrs9_e2e.ipynb"

PAPER2_SURFACES = [
    REPO_ROOT / "book/chapters/15-paper-ifrs9/index.qmd",
    REPO_ROOT / "book/chapters/15-paper-ifrs9/15a-introduction.qmd",
    REPO_ROOT / "book/chapters/15-paper-ifrs9/15b-methodology.qmd",
    REPO_ROOT / "book/chapters/15-paper-ifrs9/15c-results.qmd",
    REPO_ROOT / "book/chapters/15-paper-ifrs9/15d-discussion.qmd",
]
GOVERNANCE_SURFACES = [
    REPO_ROOT / "book/chapters/10-ifrs9-governance/10a-ecl-calculation.qmd",
    REPO_ROOT / "book/chapters/10-ifrs9-governance/10c-sicr-conformal-signal.qmd",
]
GOVERNED_SURFACES = PAPER2_SURFACES + GOVERNANCE_SURFACES
PAPER4_ACTIVE_SURFACES = [
    REPO_ROOT / f"book/chapters/19-paper-mega-extension/{name}"
    for name in (
        "index.qmd",
        "19a-proposal-and-scope.qmd",
        "19b-current-assets-and-gaps.qmd",
        "19c-integrated-architecture.qmd",
        "19f-sequential-decision-framework.qmd",
        "19h-mvp-evidence-pack.qmd",
        "19i-regret-auditability-frontier.qmd",
        "19n-online-mdcp-fairness.qmd",
        "19t-multi-period-solver.qmd",
        "19ca-v38-final-synthesis.qmd",
        "19cb-v38-appendix-registers.qmd",
        "19cc-v39-pyepo-real-suite.qmd",
    )
]

REQUIRED_LABELS = {
    "index.qmd": set(),
    "15a-introduction.qmd": {
        "sec-p2-intro",
        "sec-p2-regulatory",
        "sec-p2-industry",
        "sec-p2-gap",
        "sec-p2-questions",
        "sec-p2-contributions",
    },
    "15b-methodology.qmd": {
        "sec-p2-methodology",
        "tbl-p2-method",
        "tbl-p2-scenarios",
    },
    "15c-results.qmd": {
        "sec-p2-results",
        "tbl-p2-results",
        "tbl-p2-sicr",
    },
    "15d-discussion.qmd": {
        "sec-p2-discussion",
        "sec-p2-policy",
        "sec-p2-sicr-comparison",
        "sec-p2-limitations",
        "sec-p2-adoption",
        "sec-p2-future",
        "sec-p2-conclusion",
    },
    "10a-ecl-calculation.qmd": {
        "sec-ecl-calculation",
        "eq-ecl-formula",
        "tbl-staging-logic",
        "sec-cif-correction",
        "tbl-cif-impact",
        "sec-ecl-conformal-staging",
        "tbl-conformal-staging-adjustment",
    },
    "10c-sicr-conformal-signal.qmd": {
        "sec-sicr-signal",
        "eq-width-signal",
        "eq-sicr-trigger",
        "tbl-sicr-optimal",
        "eq-alpha-ecl",
    },
}

# These are retired result strings or promotional formulations, not generic
# IFRS9 vocabulary. Their reappearance would reactivate a claim that the
# contract explicitly parks.
RETIRED_CLAIM_PATTERNS = {
    "selected width threshold": re.compile(r"\bt\s*\^?\s*\*?\s*=\s*0[.,]30\b"),
    "historical recall": re.compile(r"\b75[.,]8\s*%"),
    "historical precision": re.compile(r"\b8[.,]9\s*%"),
    "historical SICR amount": re.compile(r"\b56[.,]6\s*m\b"),
    "retired ECL lower endpoint": re.compile(r"\b455[.,]4\s*m\b"),
    "retired ECL upper endpoint": re.compile(r"\b1[.,]563[.,]3\s*m\b"),
    "retired alpha percentage": re.compile(r"\b267\s*%"),
    "retired temporal amounts": re.compile(r"\b(?:336|47[.,]5|305)\s*m\b"),
    "retired stage cost": re.compile(r"\b97[.,]3\s*m\b"),
    "retired CIF reserve amount": re.compile(r"\b125[.,]8\s*m\b"),
    "BMA dominance": re.compile(r"\b(?:cp\s+domina|domina\s+a\s+bma)\b"),
    "formal ECL interval": re.compile(
        r"\b(?:intervalo\s+conformal\s+de\s+ecl|rango\s+ecl\s+conformal)\b"
    ),
    "prudence-cost promotion": re.compile(r"\bcosto\s+de\s+la\s+prudencia\b"),
    "regulatory lever promotion": re.compile(r"\bpalanca\s+de\s+politica\b"),
    "actionable policy promotion": re.compile(r"\binformacion\s+directamente\s+accionable\b"),
    "validated deterioration signal": re.compile(r"\bsenal\s+legitima\s+de\s+deterioro\b"),
    "practical adoption roadmap": re.compile(r"\bruta\s+de\s+adopcion\s+practica\b"),
    "reasonable buffer recommendation": re.compile(r"\bbuffer\s+razonable\b"),
}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents.lower())


def _load_contract() -> dict[str, object]:
    assert CONTRACT_PATH.exists(), f"Missing Paper 2 claim contract: {CONTRACT_PATH}"
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_paper2_contract_is_parked_and_complete() -> None:
    contract = _load_contract()

    assert contract["status"] == "parked_ifrs9"
    assert contract["decision"] == "park"

    for key, minimum in {
        "allowed_now": 6,
        "forbidden_now": 8,
        "unpark_requirements": 10,
    }.items():
        values = contract[key]
        assert isinstance(values, list)
        assert len(values) >= minimum
        assert all(isinstance(value, str) and value.strip() for value in values)

    allowed = _normalize(" ".join(contract["allowed_now"]))
    forbidden = _normalize(" ".join(contract["forbidden_now"]))
    requirements = _normalize(" ".join(contract["unpark_requirements"]))
    stop_rule = _normalize(str(contract["stop_rule"]))

    for phrase in ["resultado binario", "sensibilidades algebraicas", "prototipo retrospectivo"]:
        assert phrase in allowed
    for phrase in ["pd individual", "ecl", "sicr", "adopcion bancaria"]:
        assert phrase in forbidden
    for phrase in ["prestamo-fecha-de-reporte", "12 meses", "test final", "loan_status"]:
        assert phrase in requirements
    assert "no reestimar" in stop_rule


def test_legacy_paper2_outputs_are_quarantined_and_not_paper_facing() -> None:
    manifest = yaml.safe_load((LEGACY_ARCHIVE / "manifest.yml").read_text(encoding="utf-8"))
    assert manifest["claim_status"] == "historical_non_citable"
    assert manifest["estimand_status"] == "mechanical_proxy_not_ifrs9"
    assert len(manifest["files"]) == 8
    assert all((LEGACY_ARCHIVE / path).is_file() for path in manifest["files"])

    active_table_dir = REPO_ROOT / "reports/paper_material/paper2/tables"
    assert not active_table_dir.exists() or not any(active_table_dir.iterdir())

    notebook_text = LEGACY_NOTEBOOK.read_text(encoding="utf-8")
    assert "DO NOT EXECUTE FOR PAPER CLAIMS" in notebook_text
    assert "LEGACY_NON_CITABLE_DIR" in notebook_text
    assert "historical_non_citable" in notebook_text
    assert '"outputs": []' in notebook_text


def test_contract_governs_exact_requested_surfaces() -> None:
    contract = _load_contract()
    expected = {path.relative_to(REPO_ROOT).as_posix() for path in GOVERNED_SURFACES}
    assert set(contract["governed_surfaces"]) == expected

    for path in GOVERNED_SURFACES:
        text = path.read_text(encoding="utf-8")
        assert CONTRACT_LINK in text, f"Missing claim-contract link in {path}"


def test_parked_surfaces_do_not_execute_or_restore_retired_figures() -> None:
    retired_figures = {
        "diagrama-paper-2.png",
        "p2_fig4_sicr_grid.png",
        "p2_fig5_ecl_alpha_sensitivity.png",
        "p2_fig6_bma_vs_cp.png",
    }
    for path in GOVERNED_SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "```{python}" not in text, f"Parked surface executes Python: {path}"
        assert not any(name in text for name in retired_figures), (
            f"Retired Paper 2 figure restored in {path}"
        )


@pytest.mark.parametrize("path", GOVERNED_SURFACES, ids=lambda path: path.name)
def test_parked_surfaces_do_not_restore_retired_claims(path: Path) -> None:
    text = _normalize(path.read_text(encoding="utf-8"))
    violations = [
        label for label, pattern in RETIRED_CLAIM_PATTERNS.items() if pattern.search(text)
    ]
    assert violations == [], f"Retired Paper 2 claims found in {path}: {violations}"


def test_surfaces_state_the_estimand_boundary() -> None:
    combined = _normalize("\n".join(path.read_text(encoding="utf-8") for path in GOVERNED_SURFACES))

    for phrase in [
        "resultado binario",
        "no es una cota para pd",
        "no identifica sicr",
        "parked_ifrs9",
        "fecha de reporte",
        "test final",
    ]:
        assert phrase in combined

    ecl_page = _normalize(GOVERNANCE_SURFACES[0].read_text(encoding="utf-8"))
    sicr_page = _normalize(GOVERNANCE_SURFACES[1].read_text(encoding="utf-8"))
    assert "no son limites de la probabilidad condicional" in ecl_page
    assert "no esta validado como senal sicr" in sicr_page
    assert "default futuro no equivale a sicr" in sicr_page


def test_paper4_does_not_reimport_parked_paper2_claims() -> None:
    combined = _normalize(
        "\n".join(path.read_text(encoding="utf-8") for path in PAPER4_ACTIVE_SURFACES)
    )
    violations = [
        label for label, pattern in RETIRED_CLAIM_PATTERNS.items() if pattern.search(combined)
    ]
    assert violations == [], f"Paper 4 restored retired Paper 2 claims: {violations}"

    prohibited_promotions = (
        "propaga incertidumbre a ecl",
        "ecl/sicr proxy prudencial",
        "proxy prudencial",
        "absorbe el valor prudencial",
        "complementary sicr signal",
        "staging thresholds are economic decisions",
        "uncertainty-aware ecl proxy",
    )
    restored = [phrase for phrase in prohibited_promotions if phrase in combined]
    assert restored == []

    for required in (
        "resultado binario observado",
        "no es un intervalo para pd",
        "no identifica sicr",
        "no equivale a stage 2",
        "procedencia no citable",
    ):
        assert required in combined


@pytest.mark.parametrize("path", GOVERNED_SURFACES, ids=lambda path: path.name)
def test_required_cross_reference_labels_are_preserved(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    labels = set(re.findall(r"#((?:sec|tbl|fig|eq)-[A-Za-z0-9_-]+)", text))
    assert REQUIRED_LABELS[path.name] <= labels
