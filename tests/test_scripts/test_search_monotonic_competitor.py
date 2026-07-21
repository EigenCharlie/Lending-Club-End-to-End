"""Tests for the fairness-aware monotonic competitor search helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from scripts import run_fairness_audit as fairness_mod
from scripts import search_monotonic_competitor as search_mod


def test_fit_venn_abers_calibrator_honors_versioned_point_rule() -> None:
    scores = np.linspace(0.02, 0.98, 200)
    y_true = (scores > 0.55).astype(int)
    evaluation_scores = np.linspace(0.05, 0.95, 31)

    legacy = search_mod._fit_calibrator(
        "venn_abers",
        y_true,
        scores,
        point_rule=search_mod.VennAbersScoreCalibrator.LEGACY_POINT_RULE,
    )
    minimax = search_mod._fit_calibrator(
        "venn_abers",
        y_true,
        scores,
        point_rule=search_mod.VennAbersScoreCalibrator.LOG_LOSS_POINT_RULE,
    )
    legacy_point, p0, p1 = legacy.predict_with_bounds(evaluation_scores)
    minimax_point, q0, q1 = minimax.predict_with_bounds(evaluation_scores)

    np.testing.assert_allclose(legacy_point, (p0 + p1) / 2.0)
    np.testing.assert_allclose(minimax_point, q1 / (1.0 - q0 + q1))
    assert not np.allclose(legacy_point, minimax_point)


def test_prepare_official_fairness_context_matches_audit_semantics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()

    fairness_cfg = {
        "policy": {
            "dpd_threshold": 0.10,
            "eo_gap_threshold": 0.11,
            "dir_threshold": 0.80,
            "prediction_threshold": 0.35,
            "outcome_mode": "approval",
        },
        "intersectional": {"enabled": True, "max_order": 2, "min_group_size": 1},
        "attributes": [
            {"name": "home_ownership", "column": "home_ownership"},
            {"name": "annual_inc_quartile", "column": "annual_inc", "binning": "quartile"},
            {"name": "verification_status", "column": "verification_status"},
        ],
    }
    (tmp_path / "configs" / "fairness_policy.yaml").write_text(
        yaml.safe_dump(fairness_cfg),
        encoding="utf-8",
    )
    (tmp_path / "models" / "threshold_semantics.json").write_text(
        json.dumps({"fairness_primary_threshold": 0.35}),
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE"],
            "annual_inc": [40_000, 60_000, 80_000, 120_000],
            "verification_status": ["Verified", "Verified", "Not Verified", "Source Verified"],
        }
    )
    y_true = np.array([0, 1, 0, 1], dtype=float)
    y_prob = np.array([0.20, 0.80, 0.30, 0.90], dtype=float)

    ctx = search_mod._prepare_official_fairness_context(df, y_true=y_true, y_pred_proba=y_prob)

    base_groups = fairness_mod._build_groups_dict(df, fairness_cfg["attributes"])
    expected_groups = dict(base_groups)
    expected_groups.update(
        fairness_mod.build_intersectional_groups(
            base_groups,
            max_order=2,
            min_group_size=1,
        )
    )

    assert (
        ctx["outcome_mode"]
        == fairness_mod._resolve_outcome_mode(fairness_cfg["policy"])
        == "approval"
    )
    assert set(ctx["base_groups"]) == set(base_groups)
    assert set(ctx["all_groups"]) == set(expected_groups)
    assert len(ctx["base_groups"]) == 3
    assert len(ctx["all_groups"]) == 6
    np.testing.assert_allclose(ctx["y_true_eval"], 1.0 - y_true)
    np.testing.assert_allclose(ctx["y_pred_eval"], 1.0 - y_prob)
    assert ctx["threshold"] == 0.35


def test_report_summary_preserves_base_and_intersectional_counts() -> None:
    y_true = np.array([1, 1, 0, 0], dtype=float)
    y_prob = np.array([0.90, 0.70, 0.20, 0.10], dtype=float)
    groups_base = {
        "home_ownership": np.array(["RENT", "RENT", "OWN", "OWN"]),
        "annual_inc_quartile": np.array(["Q1", "Q2", "Q3", "Q4"]),
        "verification_status": np.array(
            ["Verified", "Not Verified", "Verified", "Not Verified"],
            dtype=object,
        ),
    }
    groups_all = dict(groups_base)
    groups_all.update(
        search_mod.build_intersectional_groups(groups_base, max_order=2, min_group_size=1)
    )

    report_all = search_mod.fairness_report(
        y_true=y_true,
        y_pred_proba=y_prob,
        groups_dict=groups_all,
        threshold=0.5,
        dpd_threshold=1.0,
        eo_gap_threshold=1.0,
        dir_threshold=0.0,
    )
    report_base = search_mod.fairness_report(
        y_true=y_true,
        y_pred_proba=y_prob,
        groups_dict=groups_base,
        threshold=0.5,
        dpd_threshold=1.0,
        eo_gap_threshold=1.0,
        dir_threshold=0.0,
    )

    summary_all = search_mod._report_summary(report_all)
    summary_base = search_mod._report_summary(report_base)

    assert summary_base["n_attributes"] == 3
    assert summary_all["n_attributes"] == 6
    assert summary_all["n_passed"] >= summary_base["n_passed"]
    assert isinstance(summary_all["rows"], list)
    assert isinstance(summary_base["rows"], list)
