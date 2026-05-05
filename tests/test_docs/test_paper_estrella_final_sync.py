"""Guardrails for the final Paper Estrella champion closure."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest
import yaml

EXPECTED_RUN_TAG = "paper-thesis-final-economic-2026-04-06"
EXPECTED_LABEL = "bound_aware_276k_economic_champion"
EXPECTED_RETURN = 170464.5429284627
EXPECTED_V = 0.03645
EXPECTED_GAMMA_CP = 0.18591
PAPER_ESTRELLA_DISCUSSION = Path("book/chapters/14-paper-estrella/14e-discussion-conclusions.qmd")
PAPER_ESTRELLA_EDITORIAL_GUIDE = Path(
    "book/chapters/14-paper-estrella/14f-editorial-claims-references.qmd"
)
PAPER_ESTRELLA_MANUSCRIPT_BLUEPRINT = Path(
    "book/chapters/14-paper-estrella/14g-manuscript-blueprint.qmd"
)
PAPER_ESTRELLA_JOURNAL_APPENDIX = Path(
    "book/chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd"
)
PAPER_ESTRELLA_INDEX = Path("book/chapters/14-paper-estrella/index.qmd")
PAPER_ESTRELLA_SUPPORT_PAGES = {
    "14i": Path("book/chapters/14-paper-estrella/14i-mondrian-ablation.qmd"),
    "14j": Path("book/chapters/14-paper-estrella/14j-spo-protocol-and-regret.qmd"),
    "14k": Path("book/chapters/14-paper-estrella/14k-fair-lending-checkpoint.qmd"),
    "14l": Path("book/chapters/14-paper-estrella/14l-governance-mrm-approval.qmd"),
    "14m": Path("book/chapters/14-paper-estrella/14m-funded-set-composition.qmd"),
    "14n": Path("book/chapters/14-paper-estrella/14n-artifact-traceability.qmd"),
    "14o": Path("book/chapters/14-paper-estrella/14o-extraction-release-manifest.qmd"),
}
PAPER_ESTRELLA_BACKLOG = Path("docs/research/paper_estrella_backlog_2026-05-04.md")
PAPER_ESTRELLA_QUARTO_EXPANSION = Path(
    "docs/research/paper_estrella_quarto_expansion_2026-05-04.md"
)
P1_EVIDENCE_STATUS = Path("models/paper1_p1_evidence_status.json")
P1_EVIDENCE_DOSSIER = Path("docs/research/paper_estrella_p1_evidence_2026-05-04.md")
P1_THEORY_APPENDIX = Path(
    "docs/research/paper_estrella_conditional_tightening_appendix_2026-05-04.md"
)
P1_JOURNAL_STATUS = Path("models/paper1_journal_package_status.json")
P1_JOURNAL_DOSSIER = Path("docs/research/paper_estrella_journal_package_2026-05-04.md")
P1_TABLES = {
    "nested": Path("reports/paper_material/paper1/tables/paper1_tableA3_nested_holdout.csv"),
    "segment": Path(
        "reports/paper_material/paper1/tables/paper1_tableA4_segment_period_sensitivity.csv"
    ),
    "selector": Path(
        "reports/paper_material/paper1/tables/paper1_tableA5_decision_aware_selector.csv"
    ),
    "shift": Path("reports/paper_material/paper1/tables/paper1_tableA6_synthetic_shift.csv"),
    "funded_loans": Path(
        "reports/paper_material/paper1/tables/paper1_tableA7_funded_set_loans.csv"
    ),
    "funded_composition": Path(
        "reports/paper_material/paper1/tables/paper1_tableA8_funded_set_composition.csv"
    ),
    "strict_holdout": Path(
        "reports/paper_material/paper1/tables/paper1_tableA9_strict_temporal_holdout.csv"
    ),
    "finalist_exact": Path(
        "reports/paper_material/paper1/tables/paper1_tableA10_conformal_finalist_exact_bound_eval.csv"
    ),
    "enhanced_shift": Path(
        "reports/paper_material/paper1/tables/paper1_tableA11_enhanced_synthetic_shift.csv"
    ),
}
P1_JOURNAL_TABLES = {
    "tail_risk": Path(
        "reports/paper_material/paper1/tables/paper1_tableA12_tail_risk_oce_cvar.csv"
    ),
    "satisficing": Path(
        "reports/paper_material/paper1/tables/paper1_tableA13_satisficing_margins.csv"
    ),
    "dependency": Path(
        "reports/paper_material/paper1/tables/paper1_tableA14_dependency_cluster_diagnostics.csv"
    ),
    "period_stress": Path(
        "reports/paper_material/paper1/tables/paper1_tableA15_leave_one_period_stress.csv"
    ),
    "bootstrap": Path(
        "reports/paper_material/paper1/tables/paper1_tableA16_bootstrap_funded_set_metrics.csv"
    ),
    "budget_lgd_cap": Path(
        "reports/paper_material/paper1/tables/paper1_tableA17_budget_cap_lgd_sensitivity.csv"
    ),
    "robust_region_family": Path(
        "reports/paper_material/paper1/tables/paper1_tableA18_robust_region_policy_family.csv"
    ),
}
P1_JOURNAL_FIGURES = {
    "crpto": Path(
        "reports/paper_material/figures_publication/estrella_fig12_crpto_conceptual_pipeline.png"
    ),
    "alpha_gamma": Path(
        "reports/paper_material/figures_publication/estrella_fig13_alpha_gamma_funded_set.png"
    ),
    "robust_region": Path(
        "reports/paper_material/figures_publication/estrella_fig14_robust_region_heatmap.png"
    ),
}


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_key_metric_table() -> dict[str, str]:
    with Path("reports/paper_material/paper1/tables/paper1_table0_key_metrics.csv").open(
        encoding="utf-8"
    ) as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_paper_estrella_champion_artifacts_agree() -> None:
    assert Path("data/processed/final_project_summary.parquet.dvc").exists()

    promotion = _load_json("models/final_project_promotion.json")
    policy = _load_json("models/champion_portfolio_policy.json")
    registry = _load_json("models/champion_registry.json")
    metrics = _load_json("reports/dvc/metrics_summary.json")
    table0 = _load_key_metric_table()

    champion = promotion["final_champion"]
    selected_policy = policy["selected_policy"]
    registry_portfolio = registry["portfolio"]

    assert promotion["run_tag"] == EXPECTED_RUN_TAG
    assert champion["label"] == EXPECTED_LABEL
    assert policy["run_tag"] == EXPECTED_RUN_TAG
    assert registry_portfolio["run_tag"] == EXPECTED_RUN_TAG
    assert registry_portfolio["selection_stage"] == "paper_thesis_final_economic_v1"

    for field in ("risk_tolerance", "gamma", "uncertainty_aversion"):
        assert selected_policy[field] == pytest.approx(champion[field])
        assert registry_portfolio["selected_policy"][field] == pytest.approx(champion[field])
    assert selected_policy["policy_mode"] == champion["policy_mode"]
    assert registry_portfolio["selected_policy"]["policy_mode"] == champion["policy_mode"]

    assert champion["realized_total_return"] == pytest.approx(EXPECTED_RETURN)
    assert champion["alpha01_exact_pass"] is True
    assert champion["alpha01_weighted_miscoverage_V"] == pytest.approx(EXPECTED_V)
    assert champion["alpha01_gamma_cp"] == pytest.approx(EXPECTED_GAMMA_CP)

    assert metrics["paper1.final.robust_return"] == pytest.approx(EXPECTED_RETURN)
    assert metrics["paper1.final.alpha01_exact_pass"] == 1.0
    assert metrics["paper1.final.alpha01_weighted_miscoverage_V"] == pytest.approx(EXPECTED_V)
    assert metrics["paper1.final.alpha01_gamma_cp"] == pytest.approx(EXPECTED_GAMMA_CP)
    assert metrics["paper1.final.robust_region_n_policies"] == 45.0
    assert metrics["paper1.final.robust_region_alpha01_pass_rate"] == 1.0

    assert table0["run_tag"] == EXPECTED_RUN_TAG
    assert table0["champion_label"] == EXPECTED_LABEL
    assert float(table0["robust_return"]) == pytest.approx(EXPECTED_RETURN)
    assert table0["alpha01_exact_pass"] == "True"
    assert float(table0["alpha01_weighted_miscoverage_V"]) == pytest.approx(EXPECTED_V)
    assert float(table0["alpha01_gamma_cp"]) == pytest.approx(EXPECTED_GAMMA_CP)


def test_search_registry_names_are_unique() -> None:
    registry = yaml.safe_load(Path("configs/pipeline_registry/search_registry.yaml").read_text())
    search_names = [entry["search"] for entry in registry["searches"]]

    assert len(search_names) == len(set(search_names))


def test_dvc_outputs_are_not_tracked_directly_by_git() -> None:
    dvc_config = yaml.safe_load(Path("dvc.yaml").read_text())
    tracked = set(subprocess.check_output(["git", "ls-files"], text=True).splitlines())

    duplicate_owned_outputs: list[str] = []
    for stage in dvc_config.get("stages", {}).values():
        for output in stage.get("outs", []) or []:
            output_path = output["path"] if isinstance(output, dict) else output
            if output_path in tracked:
                duplicate_owned_outputs.append(output_path)

    assert not duplicate_owned_outputs


def test_paper_estrella_journal_backlog_is_documented() -> None:
    assert PAPER_ESTRELLA_BACKLOG.exists()
    assert PAPER_ESTRELLA_INDEX.exists()
    assert PAPER_ESTRELLA_EDITORIAL_GUIDE.exists()
    assert PAPER_ESTRELLA_MANUSCRIPT_BLUEPRINT.exists()
    assert PAPER_ESTRELLA_JOURNAL_APPENDIX.exists()
    for page in PAPER_ESTRELLA_SUPPORT_PAGES.values():
        assert page.exists()
    assert PAPER_ESTRELLA_QUARTO_EXPANSION.exists()

    discussion = PAPER_ESTRELLA_DISCUSSION.read_text(encoding="utf-8")
    editorial_guide = PAPER_ESTRELLA_EDITORIAL_GUIDE.read_text(encoding="utf-8")
    manuscript_blueprint = PAPER_ESTRELLA_MANUSCRIPT_BLUEPRINT.read_text(encoding="utf-8")
    journal_appendix = PAPER_ESTRELLA_JOURNAL_APPENDIX.read_text(encoding="utf-8")
    backlog = PAPER_ESTRELLA_BACKLOG.read_text(encoding="utf-8")
    quarto_config = Path("book/_quarto.yml").read_text(encoding="utf-8")

    for token in (
        "tbl-p1-claim-artifact-test",
        "tbl-p1-journal-roadmap",
        "nested holdout",
        "CROMS",
        "OCE/CVaR",
        "paper1.final.robust_return",
        "tbl-p1-p1-strict-temporal-holdout",
        "tbl-p1-p1-finalist-exact-eval",
        "tbl-p1-p1-enhanced-synthetic-shift",
        "paper1_tableA12_tail_risk_oce_cvar.csv",
        "paper1_tableA18_robust_region_policy_family.csv",
        "test_paper_estrella_journal_package_artifacts_exist",
    ):
        assert token in discussion

    for token in (
        "Do Not Reopen Without Approval",
        EXPECTED_RUN_TAG,
        "Decision-aware conformal selector",
        "paper1_tableA10_conformal_finalist_exact_bound_eval.csv",
        "paper1_tableA11_enhanced_synthetic_shift.csv",
        "14f-editorial-claims-references.qmd",
        "14g-manuscript-blueprint.qmd",
        "14h-journal-appendix-robustness.qmd",
        "14i-mondrian-ablation.qmd",
        "14j-spo-protocol-and-regret.qmd",
        "14k-fair-lending-checkpoint.qmd",
        "14l-governance-mrm-approval.qmd",
        "14m-funded-set-composition.qmd",
        "14n-artifact-traceability.qmd",
        "14o-extraction-release-manifest.qmd",
        "Extraction/release manifest",
        "paper1_tableA12_tail_risk_oce_cvar.csv",
        "paper1_tableA18_robust_region_policy_family.csv",
        "scripts/build_paper1_journal_package.py",
        "OCE/CVaR funded-set conformal risk",
        "Online conformal recalibration",
    ):
        assert token in backlog

    for token in (
        "chapters/14-paper-estrella/14f-editorial-claims-references.qmd",
        "Guía Editorial, Claims y Referencias",
        "tbl-p1-claim-ladder",
        "tbl-p1-reviewer-audiences",
        "tbl-p1-paper-placement",
        "tbl-p1-journal-ready-package",
        "[1] Vovk",
        "[17] Powell",
    ):
        assert token in quarto_config + editorial_guide

    for token in (
        "chapters/14-paper-estrella/14g-manuscript-blueprint.qmd",
        "chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd",
        "Management Science",
        "C1",
        "tbl-p1-manuscript-figures",
        "tbl-p1-claim-artifact-test-location",
        "tbl-p1-notation-unified",
    ):
        assert token in quarto_config + manuscript_blueprint

    for token in (
        "tbl-p1-journal-tail-risk",
        "fig-p1-crpto-conceptual",
        "fig-p1-alpha-gamma-funded-set",
        "fig-p1-robust-region-heatmap",
        "paper1_tableA18_robust_region_policy_family.csv",
        "models/paper1_journal_package_status.json",
    ):
        assert token in journal_appendix


def test_paper_estrella_support_pages_are_synchronized() -> None:
    quarto_config = Path("book/_quarto.yml").read_text(encoding="utf-8")
    landing = PAPER_ESTRELLA_INDEX.read_text(encoding="utf-8")
    discussion = PAPER_ESTRELLA_DISCUSSION.read_text(encoding="utf-8")
    backlog = PAPER_ESTRELLA_BACKLOG.read_text(encoding="utf-8")
    traceability = PAPER_ESTRELLA_SUPPORT_PAGES["14n"].read_text(encoding="utf-8")
    manifest = PAPER_ESTRELLA_SUPPORT_PAGES["14o"].read_text(encoding="utf-8")
    governance = PAPER_ESTRELLA_SUPPORT_PAGES["14l"].read_text(encoding="utf-8")
    fairness = PAPER_ESTRELLA_SUPPORT_PAGES["14k"].read_text(encoding="utf-8")
    spo = PAPER_ESTRELLA_SUPPORT_PAGES["14j"].read_text(encoding="utf-8")

    for page in PAPER_ESTRELLA_SUPPORT_PAGES.values():
        chapter_path = page.relative_to(Path("book")).as_posix()
        assert f"chapters/{chapter_path.removeprefix('chapters/')}" in quarto_config
        assert page.name.replace(".qmd", ".html") in landing

    assert "Fairness proxy, no atributos protegidos directos" in discussion
    assert "3 atributos base y 3 cruces interseccionales proxy" in discussion
    assert "14e-future-directions" not in governance
    assert "@sec-estrella-future" in governance

    assert "proxies socioeconómicos" in fairness
    assert "no debe leerse como certificación legal" in fairness
    assert "models/spo_real_training_status.json" in spo
    assert "data/processed/crpto_vs_spo_stability.json" in spo
    assert "`n_items=100`" in spo
    assert "`n_items=50`" in spo

    for obsolete in (
        "Fairness por atributo, no interseccional",
        "scripts/build_paper1_figures.py",
        "scripts/run_conformal_reopen.py",
        "scripts/promote_paper_estrella_final.py",
        "tests/test_docs/test_paper1_p1_evidence_*.py",
        "tests/test_evaluation/test_bound_validation.py",
        "tests/test_models/test_pd_canonical.py",
    ):
        assert obsolete not in discussion + traceability + backlog

    for real_path in (
        "scripts/train_pd_model.py",
        "src/models/calibration.py",
        "tests/test_scripts/test_train_pd_model.py",
        "tests/test_evaluation/test_calibration_mapping.py",
        "scripts/search/run_conformal_reopen_search.py",
        "tests/test_scripts/test_run_conformal_reopen_search.py",
        "scripts/search/run_portfolio_bound_aware_search.py",
        "scripts/export_final_project_promotion.py",
        "scripts/generate_paper_figures.py",
        "scripts/run_crpto_vs_spo_stability.py",
        "scripts/validate_alpha_gamma_bound.py",
        "tests/test_docs/test_paper_estrella_final_sync.py",
    ):
        assert real_path in traceability
        assert Path(real_path).exists()

    for token in (
        "Quarto Expansion Snapshot - 2026-05-05",
        "Direct protected-attribute / temporal fairness validation",
        "proxy base + proxy-intersectional audit exists in `14k`",
        "14o-extraction-release-manifest.qmd",
        "OCE/CVaR optimization",
        "new method, guarantee or dataset",
        "execute.freeze: true",
    ):
        assert token in backlog

    for token in (
        "CRPTO post-hoc auditable con economic champion congelado",
        "tbl-p1-no-direction-change-filter",
        "tbl-p1-section-extraction-manifest",
        "tbl-p1-table-extraction-manifest",
        "tbl-p1-figure-extraction-manifest",
        "tbl-p1-venue-response-bank",
        "tbl-p1-release-checklist",
        "OCE/CVaR como constraint de optimización",
        "No implementar ahora",
        "todo lo que crea una nueva policy, una nueva garantía o un nuevo dataset",
    ):
        assert token in manifest


def test_paper_estrella_p1_evidence_artifacts_exist() -> None:
    assert P1_EVIDENCE_STATUS.exists()
    assert P1_EVIDENCE_DOSSIER.exists()
    assert P1_THEORY_APPENDIX.exists()
    for table in P1_TABLES.values():
        assert table.exists()

    status = _load_json(str(P1_EVIDENCE_STATUS))
    assert status["run_tag"] == EXPECTED_RUN_TAG
    assert status["champion_label"] == EXPECTED_LABEL

    nested = _read_csv_rows(P1_TABLES["nested"])
    assert len(nested) == 3
    final_nested = next(row for row in nested if row["stage"] == "bound_aware_276k")
    assert final_nested["alpha01_exact_pass"] == "True"
    assert final_nested["selected_matches_final_champion"] == "True"
    assert float(final_nested["realized_total_return"]) == pytest.approx(EXPECTED_RETURN)
    assert float(final_nested["alpha01_weighted_miscoverage_V"]) == pytest.approx(EXPECTED_V)
    assert float(final_nested["alpha01_gamma_cp"]) == pytest.approx(EXPECTED_GAMMA_CP)

    selector = _read_csv_rows(P1_TABLES["selector"])
    selected = [row for row in selector if row["decision_aware_selected"] == "True"]
    assert len(selected) == 1
    assert selected[0]["rank"] == "1"
    assert selected[0]["gate_pass"] == "True"
    assert selected[0]["exact_bound_available"] == "True"
    assert all(row["exact_bound_available"] == "True" for row in selector)
    assert status["decision_aware_selector"]["exact_bound_available_for_all_ranks"] is True

    segment = _read_csv_rows(P1_TABLES["segment"])
    assert len(segment) >= 20
    assert min(float(row["coverage_90"]) for row in segment) >= 0.90
    assert status["segment_period"]["flagged_segments"] == 0

    shift = _read_csv_rows(P1_TABLES["shift"])
    assert {row["scenario"] for row in shift} >= {"baseline", "high_pd_tail_3x"}
    assert all(row["coverage90_pass"] == "True" for row in shift)
    assert status["synthetic_shift"]["all_coverage90_pass"] is True

    funded_loans = _read_csv_rows(P1_TABLES["funded_loans"])
    assert len(funded_loans) >= 300
    total_exposure = sum(float(row["funded_exposure"]) for row in funded_loans)
    assert total_exposure == pytest.approx(1_000_000.0)
    assert status["funded_set_export"]["status"] == "implemented"
    assert status["funded_set_export"]["n_funded_loans"] == len(funded_loans)

    funded_composition = _read_csv_rows(P1_TABLES["funded_composition"])
    assert len(funded_composition) >= 20
    assert max(float(row["exposure_share"]) for row in funded_composition) < 0.25
    assert status["funded_set_composition"]["status"] == "implemented"

    strict_holdout = _read_csv_rows(P1_TABLES["strict_holdout"])
    assert len(strict_holdout) == 2
    assert all(row["alpha01_exact_pass"] == "True" for row in strict_holdout)
    assert all(float(row["alpha01_violation"]) == pytest.approx(0.0) for row in strict_holdout)
    assert status["strict_temporal_holdout"]["strict_disjoint_split"] is True
    assert status["strict_temporal_holdout"]["all_alpha01_pass"] is True

    finalist_exact = _read_csv_rows(P1_TABLES["finalist_exact"])
    assert {row["rank"] for row in finalist_exact} == {"1", "2", "3"}
    assert all(row["alpha01_exact_pass"] == "True" for row in finalist_exact)
    assert status["conformal_finalist_exact_eval"]["status"] == "implemented"
    assert status["conformal_finalist_exact_eval"]["alpha01_pass_ranks"] == [1, 2, 3]

    enhanced_shift = _read_csv_rows(P1_TABLES["enhanced_shift"])
    assert len(enhanced_shift) >= 5
    assert all(row["coverage90_pass"] == "True" for row in enhanced_shift)
    assert status["enhanced_synthetic_shift"]["status"] == "implemented"
    assert status["enhanced_synthetic_shift"]["all_coverage90_pass"] is True
    assert status["conditional_tightening"]["appendix_artifact"] == str(P1_THEORY_APPENDIX)


def test_paper_estrella_journal_package_artifacts_exist() -> None:
    assert P1_JOURNAL_STATUS.exists()
    assert P1_JOURNAL_DOSSIER.exists()
    for table in P1_JOURNAL_TABLES.values():
        assert table.exists()
        assert _read_csv_rows(table)
    for figure in P1_JOURNAL_FIGURES.values():
        assert figure.exists()
        assert figure.stat().st_size > 0

    status = _load_json(str(P1_JOURNAL_STATUS))
    assert status["run_tag"] == EXPECTED_RUN_TAG
    assert status["champion_label"] == EXPECTED_LABEL
    assert status["bootstrap_draws"] == 2000
    assert status["bootstrap_seed"] == 20260504

    generated = set(status["generated_artifacts"])
    for table in P1_JOURNAL_TABLES.values():
        assert str(table) in generated
    for figure in P1_JOURNAL_FIGURES.values():
        assert str(figure) in generated

    tail_risk = _read_csv_rows(P1_JOURNAL_TABLES["tail_risk"])
    assert {row["lgd"] for row in tail_risk} == {"0.35", "0.45", "0.6"}
    lgd45 = next(row for row in tail_risk if row["lgd"] == "0.45")
    assert float(lgd45["cvar_95_loss_rate"]) > 0
    assert "funded_set_repriced_return" in lgd45

    satisficing = _read_csv_rows(P1_JOURNAL_TABLES["satisficing"])
    assert {row["criterion"] for row in satisficing} >= {
        "return_beats_theorem_tight",
        "V_below_sqrt_alpha01",
        "gamma_cp_below_020",
        "violation_zero",
        "robust_region_all_pass",
    }
    assert all(row["pass"] == "True" for row in satisficing)

    dependency = _read_csv_rows(P1_JOURNAL_TABLES["dependency"])
    assert {"period", "grade", "period_grade"} <= {row["cluster_type"] for row in dependency}
    assert max(float(row["exposure_share"]) for row in dependency) < 0.5

    period_stress = _read_csv_rows(P1_JOURNAL_TABLES["period_stress"])
    assert "baseline" in {row["scenario"] for row in period_stress}
    assert any(row["scenario"].startswith("leave_out_") for row in period_stress)
    assert any(row["scenario"].startswith("overweight_2x_") for row in period_stress)
    assert all("funded_set_repriced_return_lgd45" in row for row in period_stress)

    bootstrap = _read_csv_rows(P1_JOURNAL_TABLES["bootstrap"])
    assert all(int(row["n_draws"]) == 2000 for row in bootstrap)
    assert {row["metric"] for row in bootstrap} >= {
        "funded_set_repriced_return_lgd45",
        "weighted_default_rate",
        "weighted_miscoverage_V",
    }

    budget_lgd_cap = _read_csv_rows(P1_JOURNAL_TABLES["budget_lgd_cap"])
    assert {row["sensitivity_type"] for row in budget_lgd_cap} == {
        "budget_scaling_diagnostic",
        "lgd_sensitivity",
        "segment_cap_diagnostic",
    }
    assert any(row["scenario"] == "lgd_0.60" for row in budget_lgd_cap)

    robust_region = _read_csv_rows(P1_JOURNAL_TABLES["robust_region_family"])
    assert len(robust_region) == 15
    assert all(row["all_alpha01_pass"] == "True" for row in robust_region)
    assert sum(int(row["n_policies"]) for row in robust_region) == 45
