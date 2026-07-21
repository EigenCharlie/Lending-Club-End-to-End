"""Guardrails for the retired local CRPTO surface and external contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_CRPTO = Path("/mnt/c/Users/carlos/Documents/Paper_CRPTO")
EXTERNAL_CONTRACT = REPO_ROOT / "docs/research/crpto_external_contract_2026-07-20.yml"
RETIREMENT_MEMO = REPO_ROOT / "docs/research/crpto_retirement_and_paper4_role_2026-06-06.md"


def _contract() -> dict:
    return yaml.safe_load(EXTERNAL_CONTRACT.read_text(encoding="utf-8"))


def _external_crpto_or_skip() -> Path:
    configured = os.environ.get("CRPTO_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_EXTERNAL_CRPTO
    if not root.is_dir():
        pytest.skip("Set CRPTO_ROOT to run the optional cross-repository integration checks")
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_crpto_mini_book_surface_is_retired() -> None:
    retired_paths = [
        REPO_ROOT / "papers/paper_crpto_book",
        REPO_ROOT / "papers/paper1_estrella",
        REPO_ROOT / "book/chapters/14-paper-estrella",
    ]
    for path in retired_paths:
        assert not path.exists(), f"Retired CRPTO surface still exists: {path}"

    quarto_config = (REPO_ROOT / "book/_quarto.yml").read_text(encoding="utf-8")
    for retired in ("paper_crpto_book", "paper1_estrella", "14-paper-estrella"):
        assert retired not in quarto_config


def test_external_contract_pins_the_current_claim_boundary() -> None:
    contract = _contract()
    repository = contract["repository"]
    science = contract["scientific_contract"]

    assert contract["status"] == "observed_external_contract"
    assert repository["branch"] == "main"
    assert len(repository["commit"]) == 40
    assert science["paper_role"] == "retrospective_identification_audit"
    assert science["prediction_target"] == "observed_binary_outcome_Y"
    assert science["registered_dvc_pointers"] == 33
    assert science["active_claims"] == 20
    assert science["primary_oot_candidates"] == (
        science["primary_oot_resolved"] + science["primary_oot_unresolved"]
    )
    for key in (
        "selected_learner",
        "selected_residual_window",
        "selected_taxonomy",
        "selected_gamma",
        "selected_ruler",
        "selected_coordinate",
        "selected_cap",
        "selected_comparator",
        "selected_policy",
    ):
        assert science[key] is None


def test_optional_external_crpto_matches_pinned_hashes_and_registries() -> None:
    root = _external_crpto_or_skip()
    contract = _contract()

    for descriptor in contract["surfaces"].values():
        path = root / descriptor["path"]
        assert path.is_file(), f"Missing external CRPTO source: {path}"
        assert _sha256(path) == descriptor["sha256"], path

    source_registry = yaml.safe_load(
        (root / "configs/ijds_active_evidence_sources.yaml").read_text(encoding="utf-8")
    )
    claim_ledger = yaml.safe_load(
        (root / "configs/ijds_claim_ledger.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "reports/crpto/ijds_binary_geometry_frontier_v4_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    science = contract["scientific_contract"]

    assert len(source_registry["dvc_pointers"]) == science["registered_dvc_pointers"]
    assert len(claim_ledger["claims"]) == science["active_claims"]
    assert manifest["design"]["primary_oot_candidates"] == science["primary_oot_candidates"]
    assert manifest["decision_challenger"]["interpretation"]["policy_winner"] is None
    assert manifest["claim_boundary"]["selected_set_validity"] is False


def test_retirement_memo_sets_current_boundary_for_paper4() -> None:
    text = RETIREMENT_MEMO.read_text(encoding="utf-8")

    assert "69095e05beae282701b4ea38aa69da26a209106f" in text
    assert "autoridad\nautocontenida para CRPTO" in text
    assert "Paper 4 queda como living lab" in text
    assert "no selecciona learner ni policy" in text
    assert "No reconstruir `book/chapters/14-paper-estrella`" in text
