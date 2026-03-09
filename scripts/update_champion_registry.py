"""Assemble current canonical champion decisions into one registry artifact."""

from __future__ import annotations

import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

SCHEMA_VERSION = "2026-03-08.1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def main() -> None:
    train_rec = _load_pickle(MODELS / "pd_training_record.pkl")
    fairness_status = _load_json(MODELS / "fairness_audit_status.json")
    governance_status = _load_json(MODELS / "governance_status.json")
    champion_policy = _load_json(MODELS / "champion_portfolio_policy.json")
    cate_status = _load_json(MODELS / "cate_portfolio_status.json")
    time_series_status = _load_json(MODELS / "time_series_status.json")

    resolved_run_tag = (
        str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
        or str(governance_status.get("run_tag", "")).strip()
        or str(fairness_status.get("run_tag", "")).strip()
        or str(champion_policy.get("run_tag", "")).strip()
        or "untracked"
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "pd": {
            "training_regime": train_rec.get("training_regime", {}),
            "stable_core": train_rec.get("stable_core", {}),
            "decision_threshold": (
                train_rec.get("decision_threshold", {}) if isinstance(train_rec, dict) else {}
            ),
        },
        "portfolio": champion_policy,
        "fairness": {
            "overall_pass": bool(fairness_status.get("overall_pass", False)),
            "prediction_threshold": fairness_status.get("prediction_threshold"),
            "prediction_threshold_source": fairness_status.get("prediction_threshold_source"),
            "decision_policy": fairness_status.get("decision_policy", {}),
        },
        "cate": {
            "promotion_eligible": cate_status.get("promotion_eligible"),
            "cate_policy_mode": cate_status.get("cate_policy_mode"),
            "fallback_applied": cate_status.get("fallback_applied"),
        },
        "time_series": {
            "run_tag": time_series_status.get("run_tag"),
            "status": time_series_status.get("status"),
        },
        "governance": {
            "overall_pass": bool(governance_status.get("overall_pass", False)),
            "summary": governance_status.get("summary", {}),
        },
    }

    out_path = MODELS / "champion_registry.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[champion_registry] saved {out_path}")


if __name__ == "__main__":
    main()
