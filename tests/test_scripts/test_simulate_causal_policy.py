"""Tests for scripts/simulate_causal_policy.py."""

from __future__ import annotations

import json
import pickle

import pandas as pd

from scripts import simulate_causal_policy as simulate_mod


def test_simulate_causal_policy_persists_policy_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "cate": [0.02, 0.015, -0.005, 0.03],
            "int_rate": [10.0, 11.0, 12.0, 13.0],
            "loan_amnt": [5000, 7000, 8000, 9000],
            "grade": ["A", "B", "B", "C"],
            "default_flag": [0, 1, 0, 1],
        }
    ).to_parquet(data_dir / "cate_estimates.parquet", index=False)

    with open(model_dir / "causal_summary.pkl", "wb") as f:
        pickle.dump({"treatment": "int_rate", "run_tag": "run-causal-test"}, f)
    (model_dir / "causal_effect_status.json").write_text(
        json.dumps({"run_tag": "run-causal-test"}),
        encoding="utf-8",
    )

    simulate_mod.main()

    sim_df = pd.read_parquet(data_dir / "causal_policy_simulation.parquet")
    runtime_status = json.loads(
        (model_dir / "causal_policy_simulation_runtime_status.json").read_text(encoding="utf-8")
    )
    with open(model_dir / "causal_policy_summary.pkl", "rb") as f:
        payload = pickle.load(f)

    assert "recommended_action" in sim_df.columns
    assert runtime_status["state"] == "completed"
    assert payload["overall"]["run_tag"] == "run-causal-test"
    assert payload["metadata"]["policy_semantics"] == "local_cate_policy_simulation"
