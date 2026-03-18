from __future__ import annotations

import json

import pandas as pd

from scripts import generate_paper_grade_protocol as proto_mod


def test_generate_paper_grade_protocol_writes_expected_sections(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(proto_mod, "ROOT", tmp_path)
    monkeypatch.setattr(proto_mod, "DATA", data_dir)
    monkeypatch.setattr(proto_mod, "MODELS", model_dir)

    (model_dir / "conformal_policy_status.json").write_text(
        json.dumps(
            {
                "strict_overall_pass": False,
                "methodological_justification_pass": True,
                "failing_non_statistical_checks": [],
                "failing_statistical_checks": ["kupiec_pvalue_90"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "conformal_variant_selection_status.json").write_text(
        json.dumps({"promotion_pass": False}),
        encoding="utf-8",
    )
    (model_dir / "time_series_status.json").write_text(
        json.dumps(
            {
                "final_interval_decision": {"status": "research_only"},
                "point_champion": {"model": "AutoARIMA"},
                "interval_champion": {"model": "AutoARIMA", "promotable": False},
                "adaptive_method_status": {"best_method": "enbpi"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "causal_policy_rule.json").write_text(
        json.dumps(
            {
                "selected_rule": "high_plus_medium_positive",
                "selection_reason": "best_value",
                "selected_metrics": {"net_value": 123.0},
                "role": "insights_only",
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "causal_policy_oot_status.json").write_text(
        json.dumps({"status": "diagnostic"}),
        encoding="utf-8",
    )
    (model_dir / "ab_simulation_status.json").write_text(
        json.dumps({"no_regression": {"passed": True}}),
        encoding="utf-8",
    )
    (model_dir / "ab_simulation_status_ambiguity_defer.json").write_text(
        json.dumps({"no_regression": {"passed": True}, "decision_scenario": "ambiguity_defer"}),
        encoding="utf-8",
    )
    (model_dir / "cate_portfolio_status.json").write_text(
        json.dumps(
            {
                "promotion_eligible": False,
                "role": "insights_only",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "governance_status.json").write_text(
        json.dumps(
            {
                "run_tag": "run-paper-grade",
                "overall_pass": True,
                "warnings": {"distribution_shift": True},
                "summary": {"max_psi": 0.06, "distribution_warning_ratio": 0.2, "c2st_auc": 0.61},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "segment": ["high_sensitivity", "medium_sensitivity"],
            "cate": [0.12, 0.08],
            "net_value": [100.0, 30.0],
        }
    ).to_parquet(data_dir / "causal_policy_simulation.parquet", index=False)
    pd.DataFrame(
        {
            "ambiguous": [0, 1],
        }
    ).to_parquet(data_dir / "pd_set_prediction_cases.parquet", index=False)
    pd.DataFrame(
        {
            "pd_low_90": [0.02, 0.05],
            "pd_high_90": [0.10, 0.20],
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

    monkeypatch.setattr(proto_mod, "resolve_official_baseline_run_tag", lambda: "run-official")
    monkeypatch.setenv("PAPER_GRADE_FINAL_RUN_TAG", "paper-grade-final")
    monkeypatch.setenv("PAPER_GRADE_FINAL_RUN_PROMOTED", "true")
    monkeypatch.setenv(
        "PAPER_GRADE_FINAL_RUN_COMPARISON_PATH",
        "reports/run_comparisons/paper-grade-final/comparison.json",
    )
    monkeypatch.setenv("PAPER_GRADE_FINAL_REBUILD_VERIFIED", "true")

    proto_mod.main()

    payload = json.loads(
        (model_dir / "paper_grade_protocol_status.json").read_text(encoding="utf-8")
    )
    assert payload["run_tag"] == "run-paper-grade"
    assert payload["pd_conformal"]["closed_for_paper_grade"] is True
    assert payload["time_series"]["decision"] == "research_only"
    assert payload["causal_cate"]["closed_for_paper_grade"] is True
    assert payload["causal_cate"]["role"] == "insights_only"
    assert payload["ab_evidence"]["ambiguity_defer_scenario_available"] is True
    assert payload["ab_evidence"]["ab_artifacts_present"] is True
    assert payload["ab_evidence"]["ab_no_regression_pass"] is True
    assert payload["governance"]["overall_pass"] is True
    assert payload["final_checks"]["protocol_frozen"] is True
    assert payload["final_checks"]["ab_no_regression"] is True
    assert payload["final_run_tag"] == "paper-grade-final"
    assert payload["final_run_promoted"] is True
    assert payload["final_rebuild_verified"] is True
