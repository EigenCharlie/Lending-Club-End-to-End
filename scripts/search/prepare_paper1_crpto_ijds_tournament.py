#!/usr/bin/env python3
"""Prepare predeclared CRPTO/IJDS tournament control files.

This script snapshots the search profile into run-root governance files. It
does not run PD, conformal, portfolio, or exact-eval jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = (
    ROOT / "configs" / "run_profiles" / "paper1_crpto_ijds_champion_tournament_2026_05_25.yaml"
)
DEFAULT_PROTOCOL = (
    ROOT / "docs" / "research" / "paper1_crpto_ijds_champion_tournament_protocol_2026-05-25.md"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _candidate_registry(profile: dict[str, Any]) -> dict[str, Any]:
    lanes = dict(profile.get("candidate_lanes") or {})
    external = dict(lanes.get("external_pd_finalists", {}).get("candidates") or {})
    hpo = dict(lanes.get("governance_aware_pd_hpo") or {})
    conformal = dict(profile.get("calibration_conformal") or {})
    portfolio = dict(profile.get("portfolio") or {})
    solver_policy = dict(profile.get("solver_policy") or {})
    exact_tournament = dict(profile.get("exact_tournament") or {})
    execution_mode = str(profile.get("execution_mode", "search_tournament"))

    pd_candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "incumbent_replay",
            "lane": "incumbent_replay",
            "role": "comparator",
            "can_replace_champion": False,
            "status": "predeclared_comparator",
        }
    ]
    for candidate_id, spec in sorted(external.items()):
        pd_candidates.append(
            {
                "candidate_id": candidate_id,
                "lane": "external_pd_finalists",
                "role": "main_challenger",
                "can_replace_champion": True,
                "pd_run_tag": spec.get("pd_run_tag"),
                "status": spec.get("status"),
            }
        )
    for candidate_id in hpo.get("base_candidates", []) or []:
        pd_candidates.append(
            {
                "candidate_id": f"{candidate_id}__governance_hpo",
                "lane": "governance_aware_pd_hpo",
                "role": "controlled_hpo_challenger",
                "can_replace_champion": True,
                "base_candidate": candidate_id,
                "initial_trials": hpo.get("initial_trials_per_lane"),
                "max_trials_without_downstream_signal": hpo.get(
                    "max_trials_per_lane_without_downstream_signal"
                ),
            }
        )

    return {
        "generated_at_utc": _utc_now(),
        "schema_version": profile.get("schema_version"),
        "profile_name": profile.get("profile_name"),
        "execution_mode": execution_mode,
        "execution_modes_supported": profile.get("execution_modes_supported", []),
        "solver_policy": solver_policy,
        "objective": profile.get("objective"),
        "champion": profile.get("champion", {}),
        "pd_candidates": pd_candidates,
        "conformal_candidates": {
            "partitions": conformal.get("partitions", {}),
            "calibrators": conformal.get("calibrators", []),
            "mapie_role": conformal.get("mapie_role", {}),
            "gates": conformal.get("gates", {}),
        },
        "portfolio_candidates": {
            "waves": portfolio.get("waves", {}),
            "grids": portfolio.get("grids", {}),
            "solver": {
                "solver_backend": portfolio.get("solver_backend"),
                "exact_solver_backend": portfolio.get("exact_solver_backend"),
                "cuopt": portfolio.get("cuopt", {}),
            },
            "execution_mode_behavior": portfolio.get("execution_mode_behavior", {}),
            "exact_tournament": exact_tournament,
        },
        "anti_cherry_pick": profile.get("anti_cherry_pick", {}),
    }


def _selection_rule(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": _utc_now(),
        "schema_version": profile.get("schema_version"),
        "profile_name": profile.get("profile_name"),
        "execution_mode": profile.get("execution_mode", "search_tournament"),
        "solver_policy": profile.get("solver_policy", {}),
        "exact_tournament": profile.get("exact_tournament", {}),
        "champion": profile.get("champion", {}),
        "selection_rules": profile.get("selection_rules", {}),
        "theory_and_robustness": profile.get("theory_and_robustness", {}),
        "outputs": profile.get("outputs", {}),
    }


def _phase_gate_status(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": _utc_now(),
        "schema_version": profile.get("schema_version"),
        "profile_name": profile.get("profile_name"),
        "execution_mode": profile.get("execution_mode", "search_tournament"),
        "solver_policy": profile.get("solver_policy", {}),
        "state": "predeclared",
        "phases": {
            "env_audit": {"state": "pending"},
            "candidate_registry": {"state": "predeclared"},
            "pd_phase": {
                "state": "pending",
                "gate": profile.get("selection_rules", {}).get("pd_phase", {}),
            },
            "conformal_phase": {
                "state": "pending",
                "gate": profile.get("calibration_conformal", {}).get("gates", {}),
            },
            "portfolio_phase": {
                "state": "pending",
                "gate": profile.get("selection_rules", {}).get("portfolio_phase", {}),
            },
            "exact_tournament_phase": {
                "state": "pending",
                "gate": profile.get("exact_tournament", {}),
            },
            "final_replacement_gate": {
                "state": "pending",
                "gate": profile.get("selection_rules", {}).get("final_replacement_gate", {}),
            },
        },
    }


def _write_negative_registry(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "logged_at_utc",
                "phase",
                "candidate_id",
                "artifact_path",
                "decision",
                "reason",
                "paper_sink",
            ]
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE))
    parser.add_argument("--protocol-path", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--log-dir", default=str(ROOT / "reports" / "run_logs"))
    args = parser.parse_args(argv)

    profile_path = Path(args.profile_path)
    if not profile_path.is_absolute():
        profile_path = (ROOT / profile_path).resolve()
    protocol_path = Path(args.protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = (ROOT / protocol_path).resolve()
    run_dir = Path(args.log_dir) / args.run_root
    run_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_yaml(profile_path)
    payload_common = {
        "profile_path": _rel(profile_path),
        "protocol_path": _rel(protocol_path),
        "run_root": args.run_root,
    }
    registry = _candidate_registry(profile)
    registry.update(payload_common)
    selection = _selection_rule(profile)
    selection.update(payload_common)
    gates = _phase_gate_status(profile)
    gates.update(payload_common)

    outputs = [
        run_dir / "predeclared_candidate_registry.json",
        run_dir / "selection_rule.json",
        run_dir / "phase_gate_status.json",
        run_dir / "negative_results_registry.csv",
    ]
    _write_json(outputs[0], registry)
    _write_json(outputs[1], selection)
    _write_json(outputs[2], gates)
    _write_negative_registry(outputs[3])

    if protocol_path.exists():
        shutil.copy2(protocol_path, run_dir / "protocol_snapshot.md")
        outputs.append(run_dir / "protocol_snapshot.md")
    shutil.copy2(profile_path, run_dir / "run_profile_snapshot.yaml")
    outputs.append(run_dir / "run_profile_snapshot.yaml")

    for path in outputs:
        if path.exists():
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
