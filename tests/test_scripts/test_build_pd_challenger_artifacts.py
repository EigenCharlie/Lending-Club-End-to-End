"""Tests for challenger feature-selection artifact builder."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import yaml

from scripts import build_pd_challenger_artifacts as challenger_mod


def test_build_pd_challenger_artifacts_outputs(tmp_path) -> None:
    rng = np.random.RandomState(42)
    n = 800

    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "default_flag": rng.binomial(1, 0.2, size=n),
            "loan_amnt": rng.normal(12000, 3000, size=n),
            "annual_inc": rng.normal(70000, 20000, size=n),
            "loan_to_income": rng.uniform(0.05, 0.5, size=n),
            "dti": rng.uniform(1.0, 30.0, size=n),
            "rev_utilization": rng.uniform(0.0, 100.0, size=n),
            "num_delinq_2yrs": rng.poisson(0.5, size=n),
            "days_since_last_delinq": rng.uniform(0, 500, size=n),
            "int_rate": rng.uniform(5.0, 25.0, size=n),
            "installment": rng.uniform(100, 1000, size=n),
            "grade_woe": rng.normal(0.0, 1.0, size=n),
            "int_rate_bucket": rng.choice(["low", "mid", "high"], size=n),
            "term": rng.choice(["36", "60"], size=n),
        }
    )
    train_path = data_dir / "train_fe.parquet"
    df.to_parquet(train_path, index=False)

    feature_cfg = {
        "CATBOOST_FEATURES": [
            "loan_amnt",
            "annual_inc",
            "loan_to_income",
            "dti",
            "rev_utilization",
            "num_delinq_2yrs",
            "days_since_last_delinq",
            "int_rate",
            "installment",
            "grade_woe",
            "int_rate_bucket",
            "term",
        ],
        "CATEGORICAL_FEATURES": ["int_rate_bucket", "term"],
        "LOGREG_FEATURES": [
            "loan_amnt",
            "annual_inc",
            "loan_to_income",
            "dti",
            "rev_utilization",
            "num_delinq_2yrs",
            "days_since_last_delinq",
            "int_rate",
            "installment",
            "grade_woe",
        ],
    }
    with open(data_dir / "feature_config.pkl", "wb") as f:
        pickle.dump(feature_cfg, f)

    cfg = {
        "data": {"train_path": str(train_path)},
        "feature_source": {
            "mode": "feature_config",
            "feature_config_path": str(data_dir / "feature_config.pkl"),
        },
        "challenger_pipeline": {
            "enabled": True,
            "random_state": 42,
            "sample_rows": 500,
            "top_k": 5,
            "permutation_max_rows": 400,
            "boruta_n_estimators": 32,
            "no_smote": True,
            "feature_selection_output": str(data_dir / "challenger_feature_selection.parquet"),
            "spec_output": str(model_dir / "pd_challenger_spec.json"),
            "spec_output_v2": str(model_dir / "pd_challenger_spec_v2.json"),
            "monotonic_constraints": {"annual_inc": -1, "dti": 1, "int_rate": 1},
        },
    }

    cfg_path = tmp_path / "pd_model.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    challenger_mod.main(str(cfg_path))

    out = pd.read_parquet(data_dir / "challenger_feature_selection.parquet")
    spec = json.loads((model_dir / "pd_challenger_spec.json").read_text(encoding="utf-8"))
    spec_v2 = json.loads((model_dir / "pd_challenger_spec_v2.json").read_text(encoding="utf-8"))

    assert len(out) > 0
    assert "selected_topk" in out.columns
    assert int(out["selected_topk"].sum()) == 5
    assert spec["modeling_policies"]["no_smote"] is True
    assert len(spec["selected_features"]) == 5
    assert spec_v2["selected_features"] == spec["selected_features"]
