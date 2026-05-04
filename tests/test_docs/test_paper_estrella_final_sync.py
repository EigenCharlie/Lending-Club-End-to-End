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
PAPER_ESTRELLA_BACKLOG = Path("docs/research/paper_estrella_backlog_2026-05-04.md")
P1_EVIDENCE_STATUS = Path("models/paper1_p1_evidence_status.json")
P1_EVIDENCE_DOSSIER = Path("docs/research/paper_estrella_p1_evidence_2026-05-04.md")
P1_TABLES = {
    "nested": Path("reports/paper_material/paper1/tables/paper1_tableA3_nested_holdout.csv"),
    "segment": Path(
        "reports/paper_material/paper1/tables/paper1_tableA4_segment_period_sensitivity.csv"
    ),
    "selector": Path(
        "reports/paper_material/paper1/tables/paper1_tableA5_decision_aware_selector.csv"
    ),
    "shift": Path("reports/paper_material/paper1/tables/paper1_tableA6_synthetic_shift.csv"),
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

    discussion = PAPER_ESTRELLA_DISCUSSION.read_text(encoding="utf-8")
    backlog = PAPER_ESTRELLA_BACKLOG.read_text(encoding="utf-8")

    for token in (
        "tbl-p1-claim-artifact-test",
        "tbl-p1-journal-roadmap",
        "nested holdout",
        "CROMS",
        "OCE/CVaR",
        "paper1.final.robust_return",
    ):
        assert token in discussion

    for token in (
        "Do Not Reopen Without Approval",
        EXPECTED_RUN_TAG,
        "Decision-aware conformal selector",
        "OCE/CVaR funded-set conformal risk",
        "Online conformal recalibration",
    ):
        assert token in backlog


def test_paper_estrella_p1_evidence_artifacts_exist() -> None:
    assert P1_EVIDENCE_STATUS.exists()
    assert P1_EVIDENCE_DOSSIER.exists()
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

    segment = _read_csv_rows(P1_TABLES["segment"])
    assert len(segment) >= 20
    assert min(float(row["coverage_90"]) for row in segment) >= 0.90
    assert status["segment_period"]["flagged_segments"] == 0

    shift = _read_csv_rows(P1_TABLES["shift"])
    assert {row["scenario"] for row in shift} >= {"baseline", "high_pd_tail_3x"}
    assert all(row["coverage90_pass"] == "True" for row in shift)
    assert status["synthetic_shift"]["all_coverage90_pass"] is True
