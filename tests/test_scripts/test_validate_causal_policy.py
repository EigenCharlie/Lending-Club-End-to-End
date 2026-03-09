"""Tests for scripts/validate_causal_policy.py."""

from __future__ import annotations

import json

import pandas as pd

from scripts import validate_causal_policy as validate_mod


def test_validate_causal_policy_writes_traceable_status(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "cate": [0.03, 0.02, 0.01, 0.04, 0.025, 0.015],
            "segment": [
                "high_sensitivity",
                "high_sensitivity",
                "medium_sensitivity",
                "high_sensitivity",
                "medium_sensitivity",
                "low_sensitivity",
            ],
            "net_value": [100, 120, 80, 90, 75, 5],
            "expected_loss_reduction": [150, 170, 100, 120, 95, 8],
            "revenue_impact": [-50, -50, -20, -30, -20, -3],
            "grade": ["A", "A", "B", "B", "C", "C"],
        }
    ).to_parquet(data_dir / "causal_policy_simulation.parquet", index=False)
    (model_dir / "causal_effect_status.json").write_text(
        json.dumps({"run_tag": "run-causal-test"}),
        encoding="utf-8",
    )

    validate_mod.main(max_action_rate=1.0, bootstrap_samples=50)

    status = json.loads((model_dir / "causal_policy_rule.json").read_text(encoding="utf-8"))
    assert status["run_tag"] == "run-causal-test"
    assert status["schema_version"]
    assert status["generated_at_utc"]
    assert status["source_effect_status_path"] == "models/causal_effect_status.json"
    assert status["selected_rule"]
