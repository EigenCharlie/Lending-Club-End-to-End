import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER4_ROOT = REPO_ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
SCRIPT_DIR = REPO_ROOT / "scripts" / "papers"
BOOK_DIR = REPO_ROOT / "book"
EXPORT_MEMO = REPO_ROOT / "docs" / "research" / "paper4_to_estrella_export_2026-05-17.md"
RETIREMENT_MEMO = REPO_ROOT / "docs/research/crpto_retirement_and_paper4_role_2026-06-06.md"
DEEP_CLEANUP_MANIFEST = TABLE_DIR / "paper4_deep_cleanup_manifest_2026-05-18.csv"

EXPECTED_CHAMPION_LABEL = "bound_aware_276k_economic_champion"
EXPECTED_CHAMPION_RETURN = 170464.5429284627
RETIRED_DATA_RANGES = (range(39, 467), range(479, 527))
RETIRED_SCRIPT_RANGES = (range(39, 527),)
CURATED_EXPORT_WAVES = range(467, 479)


def _read_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Missing CSV fixture: {path}"
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    assert path.exists(), f"Missing JSON fixture: {path}"
    return json.loads(path.read_text())


def _version_from_name(path: Path) -> int | None:
    match = re.search(r"(?:paper4|build_paper4)_v(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else None


def _in_any_range(version: int | None, ranges: tuple[range, ...]) -> bool:
    return version is not None and any(version in version_range for version_range in ranges)


def test_paper4_core_policy_and_champion_guardrails() -> None:
    registry = _read_csv(TABLE_DIR / "paper4_policy_class_registry.csv")
    paper1_champion = registry[registry["policy_id"].eq("paper1_economic_champion")]
    assert len(paper1_champion) == 1
    assert paper1_champion["policy_class"].iat[0] == "CFA"
    assert paper1_champion["is_paper1_champion"].eq(True).iat[0]

    promotion = _read_json(REPO_ROOT / "models" / "final_project_promotion.json")
    champion = promotion["final_champion"]
    assert champion["label"] == EXPECTED_CHAMPION_LABEL
    assert champion["champion_role"] == "economic_champion"
    assert champion["realized_total_return"] == pytest.approx(EXPECTED_CHAMPION_RETURN)
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_book_surface_inputs_exist() -> None:
    csv_literals: set[str] = set()

    for path in sorted((BOOK_DIR / "chapters" / "19-paper-mega-extension").glob("*.qmd")):
        text = path.read_text()
        csv_literals.update(re.findall(r'_csv\("([^"]*paper4_[A-Za-z0-9_-]+\.csv)"\)', text))

    bounded_export_surfaces = [EXPORT_MEMO, RETIREMENT_MEMO]
    for path in bounded_export_surfaces:
        csv_literals.update(re.findall(r"paper4_[A-Za-z0-9_-]+\.csv", path.read_text()))

    assert csv_literals, "Expected Paper 4 CSV references in book/export surfaces"
    missing = sorted(name for name in csv_literals if not (TABLE_DIR / name).exists())
    assert missing == []


def test_paper4_official_v31_v37_and_export_evidence_are_retained() -> None:
    retained_tables = [
        "paper4_v31_512_feasibility_guard.csv",
        "paper4_v31_champion_vs_cvar_stress_memo.csv",
        "paper4_v31_dynamic_policy_summary.csv",
        "paper4_v31_scale_convergence.csv",
        "paper4_v33_cvar_frontier_v3.csv",
        "paper4_v33_cvar_full_universe_feasibility_attempt.csv",
        "paper4_v33_cvar_infeasibility_certificate_v3.csv",
        "paper4_v35_external_macro_source_registry.csv",
        "paper4_v35_online_min_support_sensitivity.csv",
        "paper4_v35_online_temporal_holdout.csv",
        "paper4_v36_ifrs9_readiness_matrix.csv",
        "paper4_v37_cate_gate_report.csv",
        "paper4_v37_fairness_proxy_only_protocol.csv",
        "paper4_v467_cvar_frontier_probe.csv",
        "paper4_v468_tight_source_rankings.csv",
        "paper4_v470_online_monitoring_proxy_summary.csv",
        "paper4_v478_section_text_stubs.csv",
    ]
    for filename in retained_tables:
        assert (TABLE_DIR / filename).exists(), filename

    for version in CURATED_EXPORT_WAVES:
        assert (STATUS_DIR / f"paper4_v{version}_status.json").exists()
        assert list(NOTE_DIR.glob(f"paper4_v{version}_*.md")), version


def test_paper4_deep_cleanup_manifest_and_absence() -> None:
    manifest = _read_csv(DEEP_CLEANUP_MANIFEST)
    assert len(manifest) == 3534
    assert manifest["action"].eq("delete").all()
    assert manifest["kind"].value_counts().to_dict() == {
        "table_csv": 2416,
        "builder_py": 440,
        "status_json": 444,
        "pycache": 129,
        "note_md": 105,
    }
    assert not any((REPO_ROOT / path).exists() for path in manifest["path"])

    retained_data_dirs = (TABLE_DIR, STATUS_DIR, NOTE_DIR)
    retired_data = [
        path
        for directory in retained_data_dirs
        for path in directory.iterdir()
        if path.suffix in {".csv", ".json", ".md"}
        if _in_any_range(_version_from_name(path), RETIRED_DATA_RANGES)
    ]
    assert retired_data == []

    retired_scripts = [
        path
        for path in SCRIPT_DIR.glob("build_paper4_v*.py")
        if _in_any_range(_version_from_name(path), RETIRED_SCRIPT_RANGES)
    ]
    assert retired_scripts == []
    assert list(PAPER4_ROOT.rglob("*v527*")) == []


def test_paper4_current_boundaries_are_artifact_backed_after_cleanup() -> None:
    boundaries = _read_csv(TABLE_DIR / "paper4_current_claim_boundaries.csv")
    assert len(boundaries) == 92

    cleanup_claims = boundaries[boundaries["claim"].str.contains("deep cleanup", case=False)]
    assert len(cleanup_claims) == 2
    assert set(cleanup_claims["allowed"]) == {True}
    assert set(cleanup_claims["evidence_artifact"]) == {
        "reports/paper_material/paper4/tables/paper4_deep_cleanup_manifest_2026-05-18.csv"
    }

    for artifact in boundaries["evidence_artifact"].dropna():
        if artifact == "paper4_final_promotion.json remains absent":
            continue
        path = REPO_ROOT / artifact
        assert path.exists(), artifact

    lab4_claims = boundaries[boundaries["evidence_artifact"].str.contains("paper4_lab4_", na=False)]
    assert len(lab4_claims) == 9
    assert {"append", "park"}.issubset(
        set(_read_csv(TABLE_DIR / "paper4_lab4_all_lane_summary_2026-05-18.csv")["decision"])
    )

    paper2_absorption = boundaries[
        boundaries["evidence_artifact"].eq(
            "reports/paper_material/paper4/tables/paper4_paper2_absorption_anchors_2026-05-18.csv"
        )
    ]
    assert len(paper2_absorption) == 2
    assert set(paper2_absorption["allowed"]) == {True, False}
    anchors = _read_csv(TABLE_DIR / "paper4_paper2_absorption_anchors_2026-05-18.csv")
    assert len(anchors) == 7
    assert {"append", "append_strong", "context", "supersede_near_term"}.issubset(
        set(anchors["decision"])
    )

    metric_governance = boundaries[
        boundaries["evidence_artifact"].eq(
            "reports/paper_material/paper4/tables/paper4_frontier_metric_governance_decision_2026-05-19.csv"
        )
    ]
    assert len(metric_governance) == 1
    assert set(metric_governance["allowed"]) == {True}
    assert set(metric_governance["prohibited_claim_flag"]) == {False}

    prohibited = boundaries[boundaries["prohibited_claim_flag"].eq(True)]
    assert not prohibited.empty
    assert prohibited["allowed"].eq(False).all()
    assert (
        prohibited["claim"]
        .str.contains(
            "IFRS9|CATE|Fair-lending|fully deployable|final-paper-ready|replace",
            regex=True,
        )
        .any()
    )


def test_paper4_living_backlog_is_compact_after_cleanup() -> None:
    backlog = _read_csv(TABLE_DIR / "paper4_living_lab_backlog.csv")
    assert len(backlog) <= 10
    assert {"closed_rule", "completed", "active_rule", "parked"}.issubset(set(backlog["status"]))
    assert "cleanup_2026_05_18" in set(backlog["last_wave"])
    assert "v1_v38" in set(backlog["last_wave"])
    assert "v467_v478" in set(backlog["last_wave"])
    assert "lab4_2026_05_18" in set(backlog["last_wave"])


def test_paper4_loop_closure_acceptance_and_export_is_guarded() -> None:
    status = _read_json(STATUS_DIR / "paper4_loop_closure_status_2026-05-17.json")
    assert status["phase"] == "paper4_loop_closure_accept_and_export"
    assert status["outcome_decision"] == "accept"
    assert status["accepted_outcome_rows"] == 14
    assert status["patch_scope"] == "four_chapter_bounded_patch"
    assert status["paper4_final_promotion_created"] is False
    assert status["paper_estrella_champion_reopened"] is False

    outcomes = _read_csv(TABLE_DIR / "paper4_loop_closure_accept_outcomes_2026-05-17.csv")
    assert len(outcomes) == 14
    assert outcomes["outcome_decision"].eq("accept").all()
    assert outcomes["patch_allowed"].eq(True).all()
    assert outcomes["claim_boundary_ok"].eq(True).all()

    closure_manifest = _read_csv(TABLE_DIR / "paper4_loop_closure_cleanup_manifest_2026-05-17.csv")
    assert len(closure_manifest) == 326
    assert closure_manifest["decision"].eq("delete").all()

    memo_text = EXPORT_MEMO.read_text()
    for token in ["F03", "F04", "F05", "SDAM"]:
        assert token in memo_text
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v478_points_to_cleanup_not_v479() -> None:
    status = _read_json(STATUS_DIR / "paper4_v478_status.json")
    assert status["next_artifact_v478"] == "paper4_deep_cleanup_manifest_2026-05-18.csv"
    assert "v479-v489" in status["claim_boundary"]
    assert not (STATUS_DIR / "paper4_v479_status.json").exists()
    assert not list(NOTE_DIR.glob("paper4_v479_*.md"))
    assert not list(TABLE_DIR.glob("paper4_v479_*.csv"))
