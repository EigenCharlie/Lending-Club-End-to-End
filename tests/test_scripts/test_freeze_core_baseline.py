"""Tests for scripts/freeze_core_baseline.py."""

from __future__ import annotations

import json
import sys

from scripts import freeze_core_baseline as freeze_mod


def test_freeze_core_baseline_sets_current_and_history(tmp_path, monkeypatch) -> None:
    run_tag = "2026-03-04-C-core-balanced-cert2"
    comparisons = tmp_path / "reports" / "run_comparisons" / run_tag
    comparisons.mkdir(parents=True)
    snapshot = comparisons / "baseline_snapshot.json"
    snapshot.write_text(json.dumps({"run_tag": run_tag}), encoding="utf-8")

    registry_path = tmp_path / "configs" / "baselines" / "core_official_baseline.json"
    primary_registry_path = (
        tmp_path / "configs" / "baselines" / "canonical_operational_baseline.json"
    )

    monkeypatch.setattr(freeze_mod, "ROOT", tmp_path)
    monkeypatch.setattr(freeze_mod, "RUN_COMPARISONS", tmp_path / "reports" / "run_comparisons")
    monkeypatch.setattr(freeze_mod, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(freeze_mod, "PRIMARY_REGISTRY_PATH", primary_registry_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_core_baseline.py",
            "--run-tag",
            run_tag,
            "--set-current",
            "--notes",
            "test freeze",
        ],
    )

    freeze_mod.main()

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    primary_payload = json.loads(primary_registry_path.read_text(encoding="utf-8"))
    assert payload["official_run_tag"] == run_tag
    assert primary_payload["official_run_tag"] == run_tag
    assert payload["baseline_snapshot_path"].endswith(
        f"reports/run_comparisons/{run_tag}/baseline_snapshot.json"
    )
    assert payload["baseline_snapshot_sha256"]
    assert payload["frozen_at_utc"]
    assert isinstance(payload.get("history"), list)
    assert len(payload["history"]) == 1
    assert payload["history"][0]["run_tag"] == run_tag
