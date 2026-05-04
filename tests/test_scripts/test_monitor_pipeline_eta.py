from __future__ import annotations

import json
from pathlib import Path

from scripts import monitor_pipeline_eta


def test_step_order_respects_pipeline_allowed_groups() -> None:
    run_info = {
        "allowed_step_groups": [
            "preflight",
            "core_data_pd",
            "diagnostics_governance",
        ],
        "forbidden_step_groups": [
            "core_conformal",
            "core_ts",
            "paper2_survival",
            "core_portfolio",
            "core_ifrs9",
            "publication_exports",
            "research_causal",
            "research_cate_portfolio",
            "research_rapids",
            "research_notebooks",
        ],
        "include_rapids": False,
        "include_notebooks": False,
    }

    assert monitor_pipeline_eta._step_order(run_info) == [
        "preflight",
        "core_data_pd",
        "diagnostics_governance",
    ]


def test_core_data_pd_uses_exhaustive_monotonic_eta(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    status_dir = run_dir / "status"
    status_dir.mkdir(parents=True)
    (status_dir / "core_data_pd.json").write_text(
        json.dumps(
            {
                "command": (
                    "uv run python -u scripts/train_pd_model.py --config configs/pd_model.champion.yaml "
                    "&& uv run python -u scripts/search_monotonic_competitor.py "
                    "--config configs/monotonic_competitor_blockwise_exhaustive.yaml"
                )
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(monitor_pipeline_eta, "_collect_completed_history", lambda *a, **k: [])

    est, source = monitor_pipeline_eta._step_estimate_seconds("core_data_pd", "run-tag", run_dir)

    assert est == 30.0 * 3600.0
    assert source == "heuristic(monotonic_blockwise_exhaustive)"
