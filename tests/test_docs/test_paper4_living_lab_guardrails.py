import json
import re
from pathlib import Path

import pandas as pd

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
SCIENTIFIC_SCOPE_INCLUDE = "{{< include ../../includes/_scientific-scope-contract.qmd >}}"

RETIRED_DATA_RANGES = (range(39, 467), range(479, 527))
RETIRED_SCRIPT_RANGES = (range(39, 527),)
CURATED_EXPORT_WAVES = range(467, 479)
ACTIVE_PAPER4_PAGES = (
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


def test_paper4_uses_current_external_crpto_identification_boundary() -> None:
    findings = _read_csv(TABLE_DIR / "paper4_current_official_findings.csv")
    f02 = findings[findings["finding_id"].eq("F02")]
    assert len(f02) == 1
    assert f02["status"].iat[0] == "external_contract"
    assert "retrospective identification audit" in f02["finding"].iat[0]
    assert "selects no learner" in f02["finding"].iat[0]
    assert f02["evidence_artifact"].iat[0] != (
        "reports/paper_material/paper4/tables/paper4_current_official_findings.csv"
    )
    assert (REPO_ROOT / f02["evidence_artifact"].iat[0]).exists()
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_active_paper4_pages_keep_crpto_and_pyepo_claims_bounded() -> None:
    chapter_dir = BOOK_DIR / "chapters" / "19-paper-mega-extension"
    active_text = "\n".join(
        (chapter_dir / page).read_text(encoding="utf-8") for page in ACTIVE_PAPER4_PAGES
    ).casefold()
    forbidden = {
        "external official champion surface",
        "official project champion",
        "pool93",
        "a35--a39",
        "spo+ gana regret",
        "crpto gana garantia",
        "p = 3.80e-163",
        "v35 strict holdouts",
        "f15 | pyepo",
        "propaga incertidumbre a ecl",
        "ecl/sicr proxy prudencial",
        "rango conformal ecl",
        "absorbe el valor prudencial",
        "ifrs9/sicr proxy | `append_strong`",
        "usd 870.3m",
        "usd 1.479b",
        "usd 432.1m",
        "usd 1.408b",
        "usd 56.6m",
        "usd 125.8m",
        "usd 97.3m",
    }
    violations = sorted(token for token in forbidden if token in active_text)
    assert not violations, f"Found stale Paper 4 claim language: {violations}"
    for required in (
        "retrospective identification audit",
        "post-selection slices",
        "teacher-cost benchmark",
        "no se transfiere al crpto ijds activo",
        "negative estimand audit",
        "no citable",
    ):
        assert required in active_text


def test_all_active_paper4_pages_import_the_scientific_scope_contract() -> None:
    chapter_dir = BOOK_DIR / "chapters" / "19-paper-mega-extension"
    for page in ACTIVE_PAPER4_PAGES:
        text = (chapter_dir / page).read_text(encoding="utf-8")
        assert SCIENTIFIC_SCOPE_INCLUDE in text, page


def test_paper4_status_and_registry_cover_all_twelve_official_pages() -> None:
    status = _read_json(STATUS_DIR / "paper4_quarto_restructure_status.json")
    assert status["official_quarto_page_count"] == len(ACTIVE_PAPER4_PAGES) == 12
    assert set(status["official_pages"]) == set(ACTIVE_PAPER4_PAGES)

    registry = _read_csv(TABLE_DIR / "paper4_quarto_page_registry.csv")
    official = registry[registry["rendered_in_quarto"].eq(True)]
    assert set(official["page"]) == set(ACTIVE_PAPER4_PAGES)
    assert set(official["role"]) == {"official_curated", "official_appendix"}
    assert official["path_exists"].eq(True).all()


def test_paper4_negative_ifrs9_audit_is_consistent_across_current_contracts() -> None:
    findings = _read_csv(TABLE_DIR / "paper4_current_official_findings.csv").set_index("finding_id")
    for finding_id in ("F08", "F13"):
        row = findings.loc[finding_id]
        assert row["status"] == "diagnostic_negative_audit"
        assert "negative" in f"{row['finding']} {row['official_claim']}".casefold()
    assert "No PD ECL SICR Stage 2" in findings.loc["F08", "claim_boundary"]
    assert "may not reuse" in findings.loc["F13", "official_claim"]

    assert findings.loc["F09", "evidence_artifact"] == (
        "reports/paper_material/paper4/tables/paper4_frontier_goal_summary_2026-05-18.csv"
    )

    architecture = _read_csv(TABLE_DIR / "paper4_one_page_architecture_2026-05-18.csv").set_index(
        "layer_id"
    )
    assert architecture.loc["A1", "primary_evidence"] == (
        "docs/research/crpto_external_contract_2026-07-20.yml"
    )
    assert "Negative IFRS9/SICR" in architecture.loc["A2", "layer"]
    assert architecture.loc["A4", "primary_evidence"].endswith(
        "paper4_v35_online_postselection_slices.csv"
    )

    anchors = _read_csv(TABLE_DIR / "paper4_paper2_absorption_anchors_2026-05-18.csv")
    assert set(anchors["anchor_id"]) == {f"P2A0{i}" for i in range(1, 8)}
    assert set(anchors["decision"]) == {
        "diagnostic_negative_audit",
        "parked_non_citable",
    }
    anchor_text = " ".join(anchors.astype(str).stack()).casefold()
    for retired in (
        "usd ",
        "870.3",
        "1.479",
        "432.1",
        "1.408",
        "56.6",
        "125.8",
        "97.3",
        "t*=",
        "pd_threshold",
    ):
        assert retired not in anchor_text

    strong = _read_csv(TABLE_DIR / "paper4_strong_appendix_register_2026-05-19.csv")
    ifrs9_strong = strong[strong["appendix_block"].str.contains("IFRS9", case=False)]
    assert len(ifrs9_strong) == 1
    assert ifrs9_strong["status_taxonomy"].iat[0] == "diagnostic_negative_audit"
    assert "non-citable" in ifrs9_strong["claim_boundary"].iat[0].casefold()

    body_split = _read_csv(TABLE_DIR / "paper4_future_paper_body_appendix_split_2026-05-18.csv")
    paper2_split = body_split[body_split["section"].str.contains("Paper 2", case=False)]
    assert len(paper2_split) == 1
    assert "Negative audit only" in paper2_split["rule"].iat[0]

    taxonomy = _read_csv(TABLE_DIR / "paper4_status_taxonomy_2026-05-18.csv")
    assert {"diagnostic_negative_audit", "parked_non_citable"}.issubset(set(taxonomy["status"]))

    frontier = _read_csv(TABLE_DIR / "paper4_frontier_goal_summary_2026-05-18.csv")
    assert frontier.loc[frontier["lane"].eq("ifrs9_sicr"), "decision"].iat[0] == (
        "diagnostic_negative_audit"
    )
    lab4 = _read_csv(TABLE_DIR / "paper4_lab4_all_lane_summary_2026-05-18.csv")
    assert lab4.loc[lab4["lane"].eq("lane7_ifrs9_proxy"), "decision"].iat[0] == (
        "diagnostic_negative_audit"
    )
    lab4_text = " ".join(lab4.astype(str).stack()).casefold()
    for retired in (
        "official champion",
        "protect_official_champion",
        "retain official champion",
        "economic champion",
        "strict holdout survival",
    ):
        assert retired not in lab4_text

    data_frontier = _read_csv(
        TABLE_DIR / "paper4_data_frontier_lane_decisions_2026-05-18.csv"
    ).set_index("lane")
    assert data_frontier.loc["ifrs9_sicr", "decision_after_data_audit"] == (
        "diagnostic_negative_audit"
    )
    for lane in ("online_conformal", "spo_dfl", "cvar_oce"):
        assert data_frontier.loc[lane, "decision_after_data_audit"] == "diagnostic_only"

    boundaries = _read_csv(TABLE_DIR / "paper4_current_claim_boundaries.csv")
    v472 = boundaries[
        boundaries["evidence_artifact"].eq(
            "reports/paper_material/paper4/tables/paper4_v472_ifrs9_proxy_boundary_summary.csv"
        )
    ]
    assert len(v472) == 1
    assert "negative estimand provenance" in v472["claim"].iat[0]
    assert "No positive PD ECL SICR" in v472["boundary"].iat[0]


def test_current_pyepo_evidence_is_descriptive_and_noninferential() -> None:
    findings = _read_csv(TABLE_DIR / "paper4_current_official_findings.csv").set_index("finding_id")
    f06_path = REPO_ROOT / findings.loc["F06", "evidence_artifact"]
    assert f06_path.name == "pyepo_real_suite_summary_descriptive_20260528.csv"

    for path in (
        f06_path,
        TABLE_DIR / "pyepo_real_suite_summary_temporal_descriptive_20260528.csv",
    ):
        summary = _read_csv(path)
        assert {
            "auditability_score",
            "wilcoxon_vs_two_stage_statistic",
            "wilcoxon_vs_two_stage_pvalue",
            "wilcoxon_alternative",
        }.isdisjoint(summary.columns)
        assert summary["inferential_valid"].eq(False).all()
        assert summary["observation_unit"].eq("overlapping_menu_seed_row").all()
        assert summary["dependence_boundary"].str.contains("not independent").all()

    pyepo_page = (
        BOOK_DIR / "chapters/19-paper-mega-extension/19cc-v39-pyepo-real-suite.qmd"
    ).read_text(encoding="utf-8")
    assert "pyepo_real_suite_summary_full_20260528.csv" not in pyepo_page
    assert "pyepo_real_suite_summary_temporal_20260528.csv" not in pyepo_page
    assert "inferential_valid=False" in pyepo_page


def test_v35_current_evidence_is_explicitly_postselection_not_holdout() -> None:
    findings = _read_csv(TABLE_DIR / "paper4_current_official_findings.csv").set_index("finding_id")
    f05_path = REPO_ROOT / findings.loc["F05", "evidence_artifact"]
    assert f05_path.name == "paper4_v35_online_postselection_slices.csv"
    slices = _read_csv(f05_path)
    assert slices["strict_holdout"].eq(False).all()
    assert not slices["validation_item"].str.contains("holdout", case=False).any()
    assert slices["claim_boundary_v35"].str.contains("not .*holdout|no refit", case=False).all()

    legacy = (
        (TABLE_DIR / "paper4_v35_online_temporal_holdout.csv")
        .read_text(encoding="utf-8")
        .casefold()
    )
    assert "strict temporal holdout diagnostic" not in legacy
    assert "leave_last_six_months_temporal_holdout" not in legacy


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
    findings = _read_csv(TABLE_DIR / "paper4_current_official_findings.csv")
    assert boundaries["claim"].is_unique
    assert findings["finding_id"].is_unique
    assert findings["finding"].is_unique
    assert "F15" not in set(findings["finding_id"])

    boundary_registry = "reports/paper_material/paper4/tables/paper4_current_claim_boundaries.csv"
    finding_registry = "reports/paper_material/paper4/tables/paper4_current_official_findings.csv"
    assert not boundaries["evidence_artifact"].eq(boundary_registry).any()
    assert not findings["evidence_artifact"].eq(finding_registry).any()

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
    assert len(lab4_claims) == 8
    assert {"append", "park"}.issubset(
        set(_read_csv(TABLE_DIR / "paper4_lab4_all_lane_summary_2026-05-18.csv")["decision"])
    )

    pyepo_claims = boundaries[boundaries["claim"].str.contains("PyEPO|DFL", case=False)]
    assert set(pyepo_claims["claim"]) == {
        "PyEPO is executable as an exploratory top-k teacher-cost benchmark in Paper 4.",
        "PyEPO establishes a current CRPTO policy result or selected-set conformal guarantee.",
    }
    allowed_pyepo = pyepo_claims[pyepo_claims["allowed"]]
    prohibited_pyepo = pyepo_claims[~pyepo_claims["allowed"]]
    assert len(allowed_pyepo) == 1
    assert "teacher-regret" in allowed_pyepo["boundary"].iat[0]
    assert "no realized-loss evaluation" in allowed_pyepo["boundary"].iat[0]
    assert len(prohibited_pyepo) == 1
    assert prohibited_pyepo["prohibited_claim_flag"].iat[0]
    assert "selected-set" in prohibited_pyepo["claim"].iat[0]

    paper2_absorption = boundaries[
        boundaries["evidence_artifact"].eq(
            "reports/paper_material/paper4/tables/paper4_paper2_absorption_anchors_2026-05-18.csv"
        )
    ]
    assert len(paper2_absorption) == 2
    assert set(paper2_absorption["allowed"]) == {True, False}
    anchors = _read_csv(TABLE_DIR / "paper4_paper2_absorption_anchors_2026-05-18.csv")
    assert len(anchors) == 7
    assert set(anchors["decision"]) == {
        "diagnostic_negative_audit",
        "parked_non_citable",
    }
    assert (
        not anchors["claim_boundary"]
        .str.contains(
            "complementary SICR signal|economic decisions|uncertainty diagnostic",
            case=False,
            regex=True,
        )
        .any()
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
    assert {
        "historical_superseded",
        "completed",
        "active_rule",
        "parked",
    }.issubset(set(backlog["status"]))
    assert "cleanup_2026_05_18" in set(backlog["last_wave"])
    assert "v1_v38" in set(backlog["last_wave"])
    assert "v467_v478" in set(backlog["last_wave"])
    assert "lab4_2026_05_18" in set(backlog["last_wave"])


def test_paper4_loop_closure_acceptance_and_superseded_export_are_guarded() -> None:
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
    assert "Status: `superseded`" in memo_text
    assert "was not transferred into the active standalone CRPTO IJDS paper" in memo_text
    assert "retrospective identification audit" in memo_text
    assert not (STATUS_DIR / "paper4_final_promotion.json").exists()


def test_paper4_v478_points_to_cleanup_not_v479() -> None:
    status = _read_json(STATUS_DIR / "paper4_v478_status.json")
    assert status["next_artifact_v478"] == "paper4_deep_cleanup_manifest_2026-05-18.csv"
    assert "v479-v489" in status["claim_boundary"]
    assert not (STATUS_DIR / "paper4_v479_status.json").exists()
    assert not list(NOTE_DIR.glob("paper4_v479_*.md"))
    assert not list(TABLE_DIR.glob("paper4_v479_*.csv"))
