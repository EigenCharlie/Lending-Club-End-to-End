"""Search and promote the best monotonic economic policy candidate.

This driver rebuilds the economic selector for a monotonic PD candidate and
evaluates a matrix of A/B scenarios / policy selectors before deciding whether
the challenger has earned a champion portfolio artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"
SCHEMA_VERSION = "2026-03-29.1"
VALID_POLICY_SELECTORS = {
    "promotion_first",
    "robustness_aware",
    "balanced_robustness",
    "guardrail_robustness",
    "actual_ab_guarded",
    "explicit_champion_only",
}
VALID_SCENARIOS = {"baseline", "ambiguity_defer", "selective_ambiguity_defer"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_json(
    src: Path,
    dst: Path,
    *,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _load_json(src)
    if transform is not None:
        payload = transform(copy.deepcopy(payload))
    _write_json(dst, payload)
    return payload


def _safe_tag_fragment(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _run_command(cmd: list[str]) -> None:
    logger.info("Running: {}", " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(shlex.quote(part) for part in cmd)}"
        )


def _selector_metrics_lookup(selector_payload: dict[str, Any], selector: str) -> dict[str, Any]:
    mapping = {
        "explicit_champion_only": selector_payload.get("economic_metrics", {}),
        "promotion_first": selector_payload.get("economic_metrics", {}),
        "robustness_aware": selector_payload.get("research_selection_metrics", {}),
        "balanced_robustness": selector_payload.get("balanced_selection_metrics", {}),
        "guardrail_robustness": selector_payload.get("guardrail_selection_metrics", {}),
    }
    return dict(mapping.get(selector, {}) or {})


def _selected_policy_from_selector(
    selector_payload: dict[str, Any],
    *,
    selector: str,
    ab_status: dict[str, Any],
) -> dict[str, Any]:
    if selector == "robustness_aware":
        return dict(
            selector_payload.get("research_alternatives", {}).get("robustness_aware")
            or selector_payload.get("selected_policy_robustness_aware")
            or selector_payload.get("selected_policy", {})
        )
    if selector == "balanced_robustness":
        return dict(
            selector_payload.get("research_alternatives", {}).get("balanced_robustness")
            or selector_payload.get("selected_policy_balanced_robustness")
            or selector_payload.get("selected_policy", {})
        )
    if selector == "guardrail_robustness":
        return dict(
            selector_payload.get("research_alternatives", {}).get("guardrail_robustness")
            or selector_payload.get("selected_policy_guardrail_robustness")
            or selector_payload.get("selected_policy", {})
        )
    if selector == "actual_ab_guarded":
        return dict(
            ab_status.get("robust_policy", {}) or selector_payload.get("selected_policy", {})
        )
    return dict(selector_payload.get("selected_policy", {}))


def _scenario_status_path(scenario: str) -> Path:
    if scenario == "baseline":
        return MODELS / "ab_simulation_status_baseline.json"
    return MODELS / f"ab_simulation_status_{scenario}.json"


def _status_output_path(selector: str, scenario: str) -> Path:
    return (
        MODELS
        / f"ab_simulation_status_{_safe_tag_fragment(selector)}_{_safe_tag_fragment(scenario)}.json"
    )


def _results_output_path(selector: str, scenario: str) -> Path:
    return (
        DATA
        / f"ab_simulation_results_{_safe_tag_fragment(selector)}_{_safe_tag_fragment(scenario)}.parquet"
    )


def _summary_output_path(selector: str, scenario: str) -> Path:
    return (
        DATA
        / f"ab_simulation_summary_{_safe_tag_fragment(selector)}_{_safe_tag_fragment(scenario)}.parquet"
    )


def _diff_total_return(status: dict[str, Any]) -> float:
    no_reg = (
        status.get("no_regression", {}) if isinstance(status.get("no_regression"), dict) else {}
    )
    if "diff_total_return" in no_reg:
        return float(no_reg.get("diff_total_return", 0.0) or 0.0)
    metrics_a = status.get("metrics_a", {}) or {}
    metrics_b = status.get("metrics_b", {}) or {}
    return float(metrics_b.get("total_return", 0.0) or 0.0) - float(
        metrics_a.get("total_return", 0.0) or 0.0
    )


def _funded_ratio(status: dict[str, Any]) -> float:
    metrics_a = status.get("metrics_a", {}) or {}
    metrics_b = status.get("metrics_b", {}) or {}
    funded_a = float(metrics_a.get("n_funded", 0.0) or 0.0)
    funded_b = float(metrics_b.get("n_funded", 0.0) or 0.0)
    if funded_a <= 0:
        return 0.0
    return funded_b / funded_a


def _avg_return_per_funded_delta(status: dict[str, Any]) -> float:
    metrics_a = status.get("metrics_a", {}) or {}
    metrics_b = status.get("metrics_b", {}) or {}
    return float(metrics_b.get("avg_return_per_funded", 0.0) or 0.0) - float(
        metrics_a.get("avg_return_per_funded", 0.0) or 0.0
    )


def _passes_no_regression(status: dict[str, Any]) -> bool:
    no_reg = (
        status.get("no_regression", {}) if isinstance(status.get("no_regression"), dict) else {}
    )
    return bool(no_reg.get("passed", False))


def _passes_cross_scenario(status: dict[str, Any]) -> bool:
    gate = (
        status.get("cross_scenario_gate", {})
        if isinstance(status.get("cross_scenario_gate"), dict)
        else {}
    )
    return bool(gate.get("passed", False))


def _candidate_record(
    *,
    selector: str,
    scenario: str,
    status_path: Path,
    status: dict[str, Any],
    selector_payload: dict[str, Any],
) -> dict[str, Any]:
    selector_metrics = _selector_metrics_lookup(selector_payload, selector)
    return {
        "selector": selector,
        "scenario": scenario,
        "status_path": str(status_path.relative_to(ROOT)),
        "passed_no_regression": _passes_no_regression(status),
        "diff_total_return": _diff_total_return(status),
        "funded_ratio": _funded_ratio(status),
        "avg_return_per_funded_delta": _avg_return_per_funded_delta(status),
        "selector_realized_total_return": float(
            selector_metrics.get("realized_total_return", float("-inf")) or float("-inf")
        ),
        "selector_n_funded": float(selector_metrics.get("n_funded", 0.0) or 0.0),
        "selector_price_of_robustness_pct": float(
            selector_metrics.get("price_of_robustness_pct", 0.0) or 0.0
        ),
        "decision_scenario": str(status.get("decision_scenario", scenario)),
        "policy_selector": str(status.get("policy_selector", selector)),
        "status": status,
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(candidate.get("diff_total_return", float("-inf"))),
        float(candidate.get("funded_ratio", 0.0)),
        float(candidate.get("selector_realized_total_return", float("-inf"))),
        float(candidate.get("selector_n_funded", 0.0)),
    )


def _best_candidate_for_scenario(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    passing = [candidate for candidate in candidates if bool(candidate.get("passed_no_regression"))]
    pool = passing or candidates
    return sorted(pool, key=_candidate_sort_key, reverse=True)[0]


def _build_cross_scenario_gate(
    *,
    baseline_status: dict[str, Any] | None,
    selective_status: dict[str, Any],
    tolerance_pct: float,
) -> dict[str, Any]:
    baseline_status = baseline_status or {}
    base_a = float((baseline_status.get("metrics_a", {}) or {}).get("total_return", 0.0) or 0.0)
    base_b = float((baseline_status.get("metrics_b", {}) or {}).get("total_return", 0.0) or 0.0)
    sel_a = float((selective_status.get("metrics_a", {}) or {}).get("total_return", 0.0) or 0.0)
    sel_b = float((selective_status.get("metrics_b", {}) or {}).get("total_return", 0.0) or 0.0)
    robust_diff = sel_b - base_b
    robust_tolerance = abs(base_b) * float(tolerance_pct)
    nonrobust_improvement = sel_a - base_a
    robust_within_tolerance = robust_diff >= -robust_tolerance
    nonrobust_improved = nonrobust_improvement > 0.0
    return {
        "gate": "cross_scenario_no_regression",
        "baseline_nonrobust_return": base_a,
        "baseline_robust_return": base_b,
        "selective_nonrobust_return": sel_a,
        "selective_robust_return": sel_b,
        "nonrobust_improvement": nonrobust_improvement,
        "nonrobust_improvement_pct": nonrobust_improvement / (abs(base_a) + 1e-6) * 100.0,
        "robust_diff": robust_diff,
        "robust_tolerance": robust_tolerance,
        "robust_tolerance_pct": float(tolerance_pct),
        "robust_within_tolerance": robust_within_tolerance,
        "nonrobust_improved": nonrobust_improved,
        "passed": bool(robust_within_tolerance and nonrobust_improved),
        "note": (
            "Cross-scenario gate: selective defer may replace baseline when non-robust improves "
            "and robust remains within tolerance."
        ),
    }


def _choose_overall_winner(
    *,
    scenario_best: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    baseline = scenario_best.get("baseline")
    ambiguity = scenario_best.get("ambiguity_defer")
    selective = scenario_best.get("selective_ambiguity_defer")

    if (
        selective is not None
        and (
            bool(selective.get("passed_no_regression"))
            or bool(selective.get("cross_scenario_gate", {}).get("passed", False))
        )
        and (baseline is None or _candidate_sort_key(selective) > _candidate_sort_key(baseline))
    ):
        return selective, "selective_ambiguity_defer", None

    if baseline is not None and bool(baseline.get("passed_no_regression")):
        return baseline, "baseline", None

    if ambiguity is not None and bool(ambiguity.get("passed_no_regression")):
        return ambiguity, "ambiguity_defer", None

    if selective is not None and bool(
        selective.get("cross_scenario_gate", {}).get("passed", False)
    ):
        return selective, "selective_ambiguity_defer", None

    fallback_pool = [
        candidate for candidate in [baseline, ambiguity, selective] if candidate is not None
    ]
    if not fallback_pool:
        return None, None, "economic_policy_failed"
    winner = sorted(fallback_pool, key=_candidate_sort_key, reverse=True)[0]
    return winner, str(winner.get("scenario", "")), "economic_policy_failed"


def _materialize_champion_policy(
    *,
    selector_payload: dict[str, Any],
    winning_candidate: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    status = dict(winning_candidate.get("status", {}) or {})
    selector = str(winning_candidate["selector"])
    scenario = str(winning_candidate["scenario"])
    selected_policy = _selected_policy_from_selector(
        selector_payload,
        selector=selector,
        ab_status=status,
    )
    payload = copy.deepcopy(selector_payload)
    payload["selection_stage"] = "monotonic_economic_promotion_v1"
    payload["selection_outcome"] = (
        "robust_selected"
        if (_passes_no_regression(status) or _passes_cross_scenario(status))
        else "fallback_nonpromoted"
    )
    payload["decision_scenario"] = scenario
    payload["selected_policy"] = selected_policy
    payload["selected_policy_source"] = selector
    payload["economic_metrics"] = {
        "diff_total_return": _diff_total_return(status),
        "passed_no_regression": _passes_no_regression(status),
        "passed_cross_scenario_gate": _passes_cross_scenario(status),
        "funded_ratio": _funded_ratio(status),
        "avg_return_per_funded_delta": _avg_return_per_funded_delta(status),
    }
    payload["ab_status_path"] = str(Path(winning_candidate["status_path"]))
    payload["promotion_decision"] = {
        "selector": selector,
        "scenario": scenario,
    }
    payload.update(
        build_artifact_metadata(
            schema_version=selector_payload.get("schema_version", SCHEMA_VERSION),
            run_tag=run_tag,
            require_explicit=True,
        )
    )
    return payload


def main(
    *,
    config_path: str = "configs/optimization.yaml",
    run_tag: str | None = None,
    frontier_path: str = "data/processed/portfolio_robustness_frontier.parquet",
    candidate_universe_path: str = "data/processed/champion_candidate_universe.parquet",
    champion_policy_path: str = "models/champion_portfolio_policy.json",
    status_path: str = "models/monotonic_economic_search_status.json",
    best_output_path: str = "models/monotonic_economic_best_policy.json",
    tradeoff_max_candidates: int = 80000,
    grid_profile: str = "balanced",
    max_portfolio_pd: float = 0.18,
    max_candidates: int = 150000,
    n_boot: int = 5000,
    seed: int = 42,
    no_regression_tolerance_pct: float = 0.05,
    policy_selector_default: str = "explicit_champion_only",
    policy_selector_candidates: str = "explicit_champion_only",
    decision_scenarios: str = "baseline,ambiguity_defer,selective_ambiguity_defer",
    actual_ab_top_k: int = 20,
    tradeoff_solver_backend: str = "highs",
    selector_solver_backend: str = "highs",
    ab_solver_backend: str = "highs",
) -> None:
    resolved_run_tag = resolve_run_tag(run_tag, require_explicit=True)
    selector_candidates = [
        value.strip().lower()
        for value in str(policy_selector_candidates).split(",")
        if value.strip()
    ]
    if not selector_candidates:
        selector_candidates = [policy_selector_default]
    invalid_selectors = [
        value for value in selector_candidates if value not in VALID_POLICY_SELECTORS
    ]
    if invalid_selectors:
        raise ValueError(f"Unsupported policy selectors: {invalid_selectors!r}")
    scenarios = [value.strip() for value in str(decision_scenarios).split(",") if value.strip()]
    if not scenarios:
        scenarios = ["baseline"]
    invalid_scenarios = [value for value in scenarios if value not in VALID_SCENARIOS]
    if invalid_scenarios:
        raise ValueError(f"Unsupported decision scenarios: {invalid_scenarios!r}")

    safe_run_tag = _safe_tag_fragment(resolved_run_tag)
    candidate_policy_path = MODELS / f"champion_portfolio_policy_candidate_{safe_run_tag}.json"
    candidate_selector_status_path = (
        MODELS / f"champion_policy_selection_status_candidate_{safe_run_tag}.json"
    )
    research_policy_path = MODELS / "portfolio_research_policy.json"

    _run_command(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.optimize_portfolio_tradeoff",
            "--config",
            config_path,
            "--max_candidates",
            str(int(tradeoff_max_candidates)),
            "--grid-profile",
            str(grid_profile),
            "--solver_backend",
            str(tradeoff_solver_backend),
        ]
    )
    _run_command(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.select_economic_portfolio_policy",
            "--config",
            config_path,
            "--run-tag",
            resolved_run_tag,
            "--solver_backend",
            str(selector_solver_backend),
            "--frontier_path",
            frontier_path,
            "--research_policy_path",
            str(research_policy_path.relative_to(ROOT)),
            "--champion_policy_path",
            str(candidate_policy_path.relative_to(ROOT)),
            "--status_path",
            str(candidate_selector_status_path.relative_to(ROOT)),
            "--candidate_universe_path",
            candidate_universe_path,
            "--decision-scenario",
            "baseline",
        ]
    )
    selector_payload = _load_json(candidate_policy_path)

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for selector in selector_candidates:
            status_out = _status_output_path(selector, scenario)
            results_out = _results_output_path(selector, scenario)
            summary_out = _summary_output_path(selector, scenario)
            _run_command(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.simulate_ab_test",
                    "--max_portfolio_pd",
                    str(float(max_portfolio_pd)),
                    "--max_candidates",
                    str(int(max_candidates)),
                    "--n_boot",
                    str(int(n_boot)),
                    "--seed",
                    str(int(seed)),
                    "--no_regression_tolerance_pct",
                    str(float(no_regression_tolerance_pct)),
                    "--champion_policy_path",
                    str(candidate_policy_path.relative_to(ROOT)),
                    "--candidate_universe_path",
                    candidate_universe_path,
                    "--results_path",
                    str(results_out.relative_to(ROOT)),
                    "--summary_path",
                    str(summary_out.relative_to(ROOT)),
                    "--status_path",
                    str(status_out.relative_to(ROOT)),
                    "--run-tag",
                    resolved_run_tag,
                    "--solver_backend",
                    str(ab_solver_backend),
                    "--policy_selector",
                    selector,
                    "--frontier_path",
                    frontier_path,
                    "--actual_ab_top_k",
                    str(int(actual_ab_top_k)),
                    "--decision-scenario",
                    scenario,
                ]
            )
            status_payload = _load_json(status_out)
            results.append(
                _candidate_record(
                    selector=selector,
                    scenario=scenario,
                    status_path=status_out,
                    status=status_payload,
                    selector_payload=selector_payload,
                )
            )

    scenario_best: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        best = _best_candidate_for_scenario(
            [row for row in results if str(row.get("scenario")) == scenario]
        )
        if best is None:
            continue
        scenario_best[scenario] = best

    baseline_best = scenario_best.get("baseline")
    selective_best = scenario_best.get("selective_ambiguity_defer")
    if selective_best is not None:
        selective_status = dict(selective_best.get("status", {}) or {})
        selective_status["cross_scenario_gate"] = _build_cross_scenario_gate(
            baseline_status=(baseline_best or {}).get("status", {}) if baseline_best else None,
            selective_status=selective_status,
            tolerance_pct=float(no_regression_tolerance_pct),
        )
        selective_best["status"] = selective_status
        _write_json(
            _status_output_path(str(selective_best["selector"]), "selective_ambiguity_defer"),
            selective_status,
        )

    scenario_artifacts: dict[str, dict[str, Any]] = {}
    for scenario, best in scenario_best.items():
        canonical_path = _scenario_status_path(scenario)
        scenario_payload = _copy_json(
            Path(best["status_path"]),
            canonical_path,
            transform=lambda payload, s=scenario, sel=str(best["selector"]): {
                **payload,
                "economic_search_selection": {
                    "scenario": s,
                    "selector": sel,
                    "selected_for_scenario": True,
                },
            },
        )
        scenario_artifacts[scenario] = scenario_payload

    overall_winner, selected_scenario, promotion_blocker = _choose_overall_winner(
        scenario_best=scenario_best
    )
    canonical_ab_status: dict[str, Any] = {}
    if overall_winner is not None:
        canonical_ab_status = _copy_json(
            Path(overall_winner["status_path"]),
            MODELS / "ab_simulation_status.json",
            transform=lambda payload: {
                **payload,
                "economic_search_selection": {
                    "selected_scenario": selected_scenario,
                    "selected_selector": str(overall_winner["selector"]),
                    "promotion_blocker": promotion_blocker,
                    "selected_for_canonical_ab": True,
                },
            },
        )

    promotion_eligible = False
    chosen_policy_payload: dict[str, Any] = {}
    if overall_winner is not None and (
        bool(overall_winner.get("passed_no_regression"))
        or bool(
            (overall_winner.get("status", {}) or {})
            .get("cross_scenario_gate", {})
            .get("passed", False)
        )
    ):
        chosen_policy_payload = _materialize_champion_policy(
            selector_payload=selector_payload,
            winning_candidate=overall_winner,
            run_tag=resolved_run_tag,
        )
        _write_json(ROOT / champion_policy_path, chosen_policy_payload)
        promotion_eligible = True

    status_payload = {
        "run_tag": resolved_run_tag,
        "policy_selector_default": policy_selector_default,
        "policy_selector_candidates": selector_candidates,
        "decision_scenarios": scenarios,
        "scenario_best": {
            scenario: {key: value for key, value in candidate.items() if key not in {"status"}}
            for scenario, candidate in scenario_best.items()
        },
        "selected_scenario": selected_scenario,
        "promotion_eligible": promotion_eligible,
        "promotion_blocker": promotion_blocker,
        "canonical_ab_status_path": "models/ab_simulation_status.json",
        "champion_policy_path": champion_policy_path,
        "candidate_selector_policy_path": str(candidate_policy_path.relative_to(ROOT)),
        "results": [
            {key: value for key, value in row.items() if key not in {"status"}} for row in results
        ],
    }
    status_payload.update(
        build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        )
    )
    _write_json(ROOT / status_path, status_payload)

    best_payload = {
        "selected_scenario": selected_scenario,
        "promotion_eligible": promotion_eligible,
        "promotion_blocker": promotion_blocker,
        "canonical_ab_status": canonical_ab_status,
        "champion_policy": chosen_policy_payload,
    }
    best_payload.update(
        build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        )
    )
    _write_json(ROOT / best_output_path, best_payload)
    logger.info(
        "Monotonic economic search finished: selected_scenario={} promotion_eligible={}",
        selected_scenario,
        promotion_eligible,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search monotonic economic promotion policy.")
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--frontier_path", default="data/processed/portfolio_robustness_frontier.parquet"
    )
    parser.add_argument(
        "--candidate_universe_path",
        default="data/processed/champion_candidate_universe.parquet",
    )
    parser.add_argument("--champion_policy_path", default="models/champion_portfolio_policy.json")
    parser.add_argument("--status_path", default="models/monotonic_economic_search_status.json")
    parser.add_argument("--best_output_path", default="models/monotonic_economic_best_policy.json")
    parser.add_argument("--tradeoff_max_candidates", type=int, default=80000)
    parser.add_argument("--grid_profile", default="balanced")
    parser.add_argument("--max_portfolio_pd", type=float, default=0.18)
    parser.add_argument("--max_candidates", type=int, default=150000)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_regression_tolerance_pct", type=float, default=0.05)
    parser.add_argument("--policy_selector_default", default="explicit_champion_only")
    parser.add_argument("--policy_selector_candidates", default="explicit_champion_only")
    parser.add_argument(
        "--decision_scenarios",
        default="baseline,ambiguity_defer,selective_ambiguity_defer",
    )
    parser.add_argument("--actual_ab_top_k", type=int, default=20)
    parser.add_argument("--tradeoff_solver_backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument("--selector_solver_backend", choices=["highs", "cuopt"], default="highs")
    parser.add_argument("--ab_solver_backend", choices=["highs", "cuopt"], default="highs")
    args = parser.parse_args()
    main(
        config_path=args.config,
        run_tag=args.run_tag,
        frontier_path=args.frontier_path,
        candidate_universe_path=args.candidate_universe_path,
        champion_policy_path=args.champion_policy_path,
        status_path=args.status_path,
        best_output_path=args.best_output_path,
        tradeoff_max_candidates=args.tradeoff_max_candidates,
        grid_profile=args.grid_profile,
        max_portfolio_pd=args.max_portfolio_pd,
        max_candidates=args.max_candidates,
        n_boot=args.n_boot,
        seed=args.seed,
        no_regression_tolerance_pct=args.no_regression_tolerance_pct,
        policy_selector_default=args.policy_selector_default,
        policy_selector_candidates=args.policy_selector_candidates,
        decision_scenarios=args.decision_scenarios,
        actual_ab_top_k=args.actual_ab_top_k,
        tradeoff_solver_backend=args.tradeoff_solver_backend,
        selector_solver_backend=args.selector_solver_backend,
        ab_solver_backend=args.ab_solver_backend,
    )
