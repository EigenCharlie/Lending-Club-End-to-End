from __future__ import annotations

from pathlib import Path

import yaml

from scripts import build_pd_hpo_local_config as mod


def test_build_pd_hpo_local_config_namespaces_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "_load_yaml",
        lambda _path: {
            "model": {"params": {"iterations": 3000}},
            "hpo": {"enabled": False},
            "validation": {"seed_replay": {"enabled": False}},
            "output": {"model_path": "models/pd_canonical.cbm"},
            "decision_threshold": {},
            "stable_core": {"enabled": True, "exclude_features": ["rev_utilization"]},
            "calibration": {"method": "auto", "candidates": ["venn_abers", "isotonic"]},
        },
    )
    monkeypatch.setattr(
        mod,
        "_load_best_variant",
        lambda _tag: {
            "profile": "blockwise_affordability",
            "variant_id": "deeper_regularized",
            "calibration_method": "venn_abers",
            "selected_threshold_mean": 0.25,
            "params_reference": {
                "iterations": 3450,
                "learning_rate": 0.03059,
                "depth": 9,
                "l2_leaf_reg": 125.7,
                "min_data_in_leaf": 185,
                "rsm": 0.6655,
                "random_strength": 1.7e-6,
                "border_count": 148,
                "bootstrap_type": "MVS",
                "subsample": 0.77,
                "grow_policy": "SymmetricTree",
                "leaf_estimation_iterations": 3,
                "monotone_constraints": "installment:1,annual_inc:-1,dti:1,loan_to_income:1",
            },
        },
    )
    monkeypatch.setattr(
        mod,
        "_load_valid_catboost_features",
        lambda _path, excluded_features=None: {
            feature
            for feature in {
                "loan_to_income",
                "annual_inc",
                "dti",
                "installment",
                "rev_utilization",
                "delinq_recency",
            }
            if feature not in (excluded_features or set())
        },
    )
    output_path = tmp_path / "pd_model_hpo_local.yaml"

    path = mod.build_pd_hpo_local_config(
        run_tag="pd-hpo-local-test",
        base_search_run_tag="base-search",
        output_path=str(output_path),
        n_trials=140,
    )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert path == output_path
    assert payload["model"]["params"]["iterations"] == 3450
    assert payload["calibration"]["method"] == "venn_abers"
    assert payload["hpo"]["enabled"] is True
    assert payload["hpo"]["n_trials"] == 140
    assert payload["hpo"]["search_space_mode"] == "local_refine"
    assert payload["hpo"]["constraints_policy"]["max_ece_delta"] == 0.0025
    penalties = payload["hpo"]["local_refine"]["first_feature_use_penalties"]
    assert "days_since_last_delinq" not in penalties
    assert "rev_utilization" not in penalties
    assert "delinq_recency" in penalties
    assert payload["output"]["model_path"].startswith("models/search_pd/pd-hpo-local-test/")
    assert payload["output"]["canonical_model_path"].startswith(
        "models/search_pd/pd-hpo-local-test/"
    )
    assert payload["output"]["threshold_semantics_path"].startswith(
        "models/search_pd/pd-hpo-local-test/"
    )
    assert payload["decision_threshold"]["output_path_v2"].startswith(
        "models/search_pd/pd-hpo-local-test/"
    )
