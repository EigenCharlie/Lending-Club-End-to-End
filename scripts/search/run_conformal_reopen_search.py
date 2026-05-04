"""Reopen conformal search exhaustively over a fixed upstream PD candidate."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.generate_conformal_intervals import (
    _build_feature_matrix,
    _load_model,
    _resolve_features,
)
from src.models.calibration import (
    calibrate_beta,
    calibrate_isotonic,
    calibrate_platt,
    evaluate_calibration,
)
from src.models.conformal import apply_probability_calibrator
from src.models.venn_abers import VennAbersScoreCalibrator
from src.utils.pipeline_topology import load_profile_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_tag() -> str:
    return datetime.now(UTC).strftime("conformal-reopen-%Y-%m-%d-%H%M")


def _profile_cfg(profile_name: str) -> dict[str, Any]:
    profile = load_profile_config(profile_name)
    if not profile:
        raise FileNotFoundError(f"Missing conformal reopen profile: {profile_name}")
    return profile


def _phase1_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    return dict((profile.get("search_space", {}) or {}).get("phase1", {}) or {})


def _phase2_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    return dict((profile.get("search_space", {}) or {}).get("phase2", {}) or {})


def _sidecar_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    return dict(profile.get("sidecar", {}) or {})


def _validation_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    return dict(profile.get("validation", {}) or {})


def _run_python(script: str, args: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, script, *args]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def _namespace(run_tag: str, *parts: object) -> str:
    suffix = "__".join(str(part).strip().replace("/", "_") for part in parts if str(part).strip())
    return f"{run_tag}__{suffix}" if suffix else run_tag


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return dict(payload) if isinstance(payload, dict) else {"payload": payload}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_run_paths(namespace: str) -> dict[str, Path]:
    ns = str(namespace).strip().replace("/", "_")
    data_dir = REPO_ROOT / "data" / "processed" / "conformal_gap" / ns
    models_dir = REPO_ROOT / "models" / "conformal_gap" / ns
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "tuning": data_dir / "conformal_mondrian_tuning_90.parquet",
        "results": models_dir / "conformal_results_mondrian.pkl",
        "policy_status": models_dir / "conformal_policy_status.json",
        "selection_status": models_dir / "conformal_variant_selection_status.json",
        "set_status": models_dir / "pd_set_prediction_status.json",
    }


def _reopen_artifact_paths(run_tag: str) -> dict[str, Path]:
    run_paths = _resolve_run_paths(run_tag)
    data_dir = run_paths["data_dir"]
    models_dir = run_paths["models_dir"]
    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "inner_search": data_dir / "conformal_reopen_inner_search.parquet",
        "inner_aggregate": data_dir / "conformal_reopen_inner_aggregate.parquet",
        "phase1_shortlist": data_dir / "conformal_reopen_phase1_shortlist.parquet",
        "phase1_final_candidates": data_dir / "conformal_reopen_phase1_final_candidates.parquet",
        "phase2_search": data_dir / "conformal_reopen_phase2_search.parquet",
        "status": models_dir / "conformal_reopen_status.json",
    }


class _temporary_env:
    def __init__(self, updates: dict[str, str]) -> None:
        self._updates = updates
        self._previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self._updates.items():
            self._previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, previous in self._previous.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _normalize_design_row(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    raw = dict(row)
    return {
        "partition": str(raw["partition"]),
        "partition_probability_source": str(raw.get("partition_probability_source", "raw")),
        "n_score_bins": int(float(raw.get("n_score_bins", 10))),
        "fallback_mode": str(raw.get("fallback_mode", "grade_then_global")),
        "alpha_used_90": float(raw.get("alpha_used_90", raw.get("selected_alpha_used", 0.10))),
        "alpha_used_95": float(raw.get("alpha_used_95", 0.05)),
        "score_scale_family": str(raw.get("score_scale_family", "none")),
        "min_group_size": int(float(raw.get("min_group_size", 100))),
        "calibration_fraction": float(raw.get("calibration_fraction", 1.0)),
    }


def _semantic_candidate_key(row: dict[str, Any] | pd.Series) -> str:
    design = _normalize_design_row(row)
    partition = str(design["partition"]).strip().lower()
    payload: dict[str, Any] = {
        "partition": partition,
        "fallback_mode": str(design["fallback_mode"]),
        "alpha_used_90": float(design["alpha_used_90"]),
        "alpha_used_95": float(design["alpha_used_95"]),
        "score_scale_family": str(design["score_scale_family"]),
        "min_group_size": int(design["min_group_size"]),
        "calibration_fraction": float(design["calibration_fraction"]),
    }
    if partition != "grade":
        payload["partition_probability_source"] = str(design["partition_probability_source"])
        payload["n_score_bins"] = int(design["n_score_bins"])
    return json.dumps(payload, sort_keys=True)


def _candidate_key(row: pd.Series) -> str:
    return json.dumps(_normalize_design_row(row), sort_keys=True)


def _aggregate_inner_search(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise RuntimeError("Inner conformal search produced no rows.")
    work = rows.copy()
    if "alpha_used_95" not in work.columns:
        work["alpha_used_95"] = 0.05
    work["coverage_gap_abs"] = work["coverage_gap"].abs()
    work["candidate_key"] = work.apply(_candidate_key, axis=1)
    agg = (
        work.groupby(
            [
                "candidate_key",
                "partition",
                "partition_probability_source",
                "n_score_bins",
                "fallback_mode",
                "alpha_used_90",
                "alpha_used_95",
                "score_scale_family",
                "min_group_size",
                "calibration_fraction",
            ],
            dropna=False,
        )
        .agg(
            n_runs=("candidate_key", "size"),
            global_ok_rate=("global_ok", "mean"),
            group_ok_rate=("group_ok", "mean"),
            width_ok_rate=("width_ok", "mean"),
            pareto_rate=("is_pareto", "mean"),
            mean_coverage=("empirical_coverage", "mean"),
            mean_abs_coverage_gap=("coverage_gap_abs", "mean"),
            mean_avg_interval_width=("avg_interval_width", "mean"),
            mean_min_group_coverage=("min_group_coverage", "mean"),
            mean_winkler_90=("winkler_90", "mean"),
            mean_stability_over_time=("stability_over_time", "mean"),
            mean_max_monthly_gap=("max_monthly_gap", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values(
        by=[
            "global_ok_rate",
            "group_ok_rate",
            "width_ok_rate",
            "mean_abs_coverage_gap",
            "mean_min_group_coverage",
            "mean_winkler_90",
            "mean_avg_interval_width",
            "mean_stability_over_time",
        ],
        ascending=[False, False, False, True, False, True, True, True],
    ).reset_index(drop=True)
    agg["selection_rank"] = range(1, len(agg) + 1)
    return agg


def _dedupe_designs(rows: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows.to_dict(orient="records"):
        key = _semantic_candidate_key(row)
        if key in seen:
            continue
        seen.add(key)
        records.append(dict(row))
        if top_k is not None and len(records) >= int(top_k):
            break
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.reset_index(drop=True)
    out["selection_rank"] = range(1, len(out) + 1)
    return out


def _design_args(design: dict[str, Any]) -> list[str]:
    normalized = _normalize_design_row(design)
    return [
        "--partition",
        str(normalized["partition"]),
        "--partition_candidates",
        str(normalized["partition"]),
        "--partition_probability_sources",
        str(normalized["partition_probability_source"]),
        "--n_score_bins_candidates",
        str(int(normalized["n_score_bins"])),
        "--fallback_modes",
        str(normalized["fallback_mode"]),
        "--alpha_candidates_90",
        str(float(normalized["alpha_used_90"])),
        "--alpha_candidates_95",
        str(float(normalized.get("alpha_used_95", 0.05))),
        "--min_group_sizes",
        str(int(normalized["min_group_size"])),
        "--score_scale_families",
        str(normalized["score_scale_family"]),
        "--scaled_scores_options",
        "true" if str(normalized["score_scale_family"]).strip().lower() != "none" else "false",
        "--calibration_fraction",
        str(float(normalized["calibration_fraction"])),
    ]


def _benchmark_row_to_design(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    raw = dict(row)
    return _normalize_design_row(
        {
            "partition": raw["partition"],
            "partition_probability_source": raw.get("partition_probability_source", "raw"),
            "n_score_bins": raw.get("n_score_bins", 10),
            "fallback_mode": raw.get("fallback_mode", "grade_then_global"),
            "alpha_used_90": raw.get("selected_alpha_used", raw.get("alpha", 0.10)),
            "alpha_used_95": 0.05,
            "score_scale_family": raw.get("score_scale_family", "none"),
            "min_group_size": raw.get("min_group_size", 100),
            "calibration_fraction": raw.get("calibration_fraction", 1.0),
        }
    )


def _fit_calibrator(
    *,
    method: str,
    output_path: Path,
    upstream_run_tag: str,
) -> tuple[str, dict[str, float]]:
    env = os.environ.copy()
    env["UPSTREAM_CANONICAL_RUN_TAG"] = upstream_run_tag
    with _temporary_env(env):
        model, _ = _load_model()
        cal_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "calibration_fe.parquet")
        test_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "test_fe.parquet")
        features, categorical = _resolve_features(model, cal_df, test_df)
        X_cal = _build_feature_matrix(cal_df, features, categorical)
        X_test = _build_feature_matrix(test_df, features, categorical)
        y_cal = cal_df["default_flag"].astype(int)
        y_test = test_df["default_flag"].astype(int).to_numpy(dtype=int)
        y_prob_cal_raw = model.predict_proba(X_cal)[:, 1]
        y_prob_test_raw = model.predict_proba(X_test)[:, 1]

        method_key = str(method).strip().lower()
        if method_key == "venn_abers":
            calibrator = VennAbersScoreCalibrator().fit(y_prob_cal_raw, y_cal.to_numpy(dtype=int))
        elif method_key == "isotonic":
            calibrator = calibrate_isotonic(y_cal.to_numpy(dtype=int), y_prob_cal_raw)
        elif method_key in {"platt", "logit"}:
            calibrator = calibrate_platt(model, X_cal, y_cal)
            method_key = "platt"
        elif method_key == "beta":
            calibrator = calibrate_beta(y_cal.to_numpy(dtype=int), y_prob_cal_raw)
        else:
            raise ValueError(f"Unsupported calibrator method: {method}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump(calibrator, handle)

        calibrated_test = apply_probability_calibrator(calibrator, y_prob_test_raw)
        metrics = evaluate_calibration(y_test, calibrated_test, name=method_key)
        return method_key, {k: float(v) for k, v in metrics.items()}


def _acceptance_pass(policy_status: dict[str, Any], validation_cfg: dict[str, Any]) -> bool:
    acceptance = dict(validation_cfg.get("acceptance", {}) or {})
    coverage = float(policy_status.get("coverage_90", 0.0))
    min_group_coverage = float(policy_status.get("min_group_coverage_90", 0.0))
    avg_width = float(policy_status.get("avg_width_90", 10.0))
    coverage_gap = abs(coverage - 0.90)
    warning_alerts = int(policy_status.get("warning_alerts", 0))
    total_alerts = int(policy_status.get("total_alerts", 0))
    return bool(
        policy_status.get("overall_pass", False)
        and warning_alerts <= int(acceptance.get("warning_alerts_max", 5))
        and total_alerts <= int(acceptance.get("total_alerts_max", 5))
        and coverage_gap <= float(acceptance.get("coverage_deviation_90_max", 0.03))
        and min_group_coverage >= float(acceptance.get("min_group_coverage_90_min", 0.88))
        and avg_width <= float(acceptance.get("avg_width_90_max", 0.80))
    )


def _policy_reason_code(policy_status: dict[str, Any], acceptance_pass: bool) -> str:
    if acceptance_pass:
        return "accepted_policy_gate"
    if not bool(policy_status.get("overall_pass", False)):
        if not bool(policy_status.get("strict_overall_pass", False)) and not bool(
            policy_status.get("methodological_justification_pass", False)
        ):
            return "policy_overall_fail"
        return "policy_warning_gate"
    return "acceptance_gate_fail"


def _run_set_prediction_sidecar(
    *,
    namespace: str,
    sidecar_cfg: dict[str, Any],
    design: dict[str, Any],
    env: dict[str, str],
    calibrator_override_path: str | None = None,
) -> None:
    methods = sidecar_cfg.get("methods", ["lac", "margin"])
    partitions = sidecar_cfg.get(
        "partitions",
        ["global", "grade", "score_decile_mondrian", "grade_x_scoreband_mondrian"],
    )
    partition_probability_sources = list(
        sidecar_cfg.get("partition_probability_sources", ["calibrated"]) or ["calibrated"]
    )
    calibration_fractions = [
        float(x)
        for x in (sidecar_cfg.get("calibration_size_fractions", [0.25, 0.50, 0.75, 1.0]) or [])
    ]
    args = [
        "--artifact_namespace",
        namespace,
        "--methods",
        ",".join(str(x) for x in methods),
        "--partitions",
        ",".join(str(x) for x in partitions),
        "--partition_probability_source",
        str(partition_probability_sources[0]),
        "--n_score_bins",
        str(int(sidecar_cfg.get("n_score_bins", design["n_score_bins"]))),
        "--min_group_size",
        str(int(sidecar_cfg.get("min_group_size", design["min_group_size"]))),
        "--fallback_mode",
        str(sidecar_cfg.get("fallback_mode", design["fallback_mode"])),
        "--calibration-size-fractions",
        ",".join(str(float(x)) for x in calibration_fractions),
    ]
    if calibrator_override_path:
        args.extend(["--calibrator_override_path", str(calibrator_override_path)])
    _run_python("scripts/benchmark_pd_set_prediction.py", args, env)


def _run_phase1_oot_candidate(
    *,
    run_tag: str,
    rank: int,
    design: dict[str, Any],
    env: dict[str, str],
    alpha_candidates_95: list[float],
    partition_candidates: list[str],
    partition_probability_sources: list[str],
    n_score_bins_candidates: list[int],
    fallback_modes: list[str],
    score_scale_families: list[str],
    calibration_fractions: list[float],
    sidecar_cfg: dict[str, Any],
    calibrator_override_path: str | None = None,
    phase_prefix: str = "phase1",
) -> dict[str, Any]:
    design = _normalize_design_row(design)
    namespace = _namespace(run_tag, phase_prefix, "final", f"rank-{rank}")
    interval_args = [
        "--artifact_namespace",
        namespace,
        "--evaluation_scope",
        "test",
        "--alpha_candidates_95",
        ",".join(str(x) for x in alpha_candidates_95),
        *_design_args(design),
    ]
    if calibrator_override_path:
        interval_args.extend(["--calibrator_override_path", str(calibrator_override_path)])
    _run_python("scripts/generate_conformal_intervals.py", interval_args, env)
    benchmark_args = [
        "--artifact_namespace",
        namespace,
        "--selected_config_path",
        str(_resolve_run_paths(namespace)["results"]),
        "--partition_candidates",
        ",".join(str(x) for x in partition_candidates),
        "--partition_probability_sources",
        ",".join(str(x) for x in partition_probability_sources),
        "--n_score_bins_candidates",
        ",".join(str(int(x)) for x in n_score_bins_candidates),
        "--fallback_modes",
        ",".join(str(x) for x in fallback_modes),
        "--score_scale_families",
        ",".join(str(x) for x in score_scale_families),
        "--min_group_size_default",
        str(int(design["min_group_size"])),
        "--calibration_size_fractions",
        ",".join(str(float(x)) for x in calibration_fractions),
    ]
    if calibrator_override_path:
        benchmark_args.extend(["--calibrator_override_path", str(calibrator_override_path)])
    _run_python("scripts/benchmark_conformal_variants.py", benchmark_args, env)
    _run_python(
        "scripts/validate_conformal_experiment.py",
        ["--namespace", namespace, "--run-tag", run_tag],
        env,
    )
    _run_set_prediction_sidecar(
        namespace=namespace,
        sidecar_cfg=sidecar_cfg,
        design=design,
        env=env,
        calibrator_override_path=calibrator_override_path,
    )

    policy_status = _load_json(_resolve_run_paths(namespace)["policy_status"])
    set_status = _load_json(_resolve_run_paths(namespace)["set_status"])
    selection_status = _load_json(_resolve_run_paths(namespace)["selection_status"])
    return {
        **design,
        "namespace": namespace,
        "policy_status": policy_status,
        "set_status": set_status,
        "selection_status": selection_status,
    }


def _phase1_candidates_frame(
    candidates: list[dict[str, Any]],
    validation_cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        policy_status = dict(candidate["policy_status"])
        set_status = dict(candidate["set_status"])
        acceptance_pass = _acceptance_pass(policy_status, validation_cfg)
        selection_status = dict(candidate.get("selection_status", {}))
        rows.append(
            {
                **{
                    key: value
                    for key, value in candidate.items()
                    if key not in {"policy_status", "set_status", "selection_status"}
                },
                "policy_overall_pass": bool(policy_status.get("overall_pass", False)),
                "strict_overall_pass": bool(policy_status.get("strict_overall_pass", False)),
                "methodological_justification_pass": bool(
                    policy_status.get("methodological_justification_pass", False)
                ),
                "coverage_90": float(policy_status.get("coverage_90", 0.0)),
                "avg_width_90": float(policy_status.get("avg_width_90", 10.0)),
                "min_group_coverage_90": float(policy_status.get("min_group_coverage_90", 0.0)),
                "warning_alerts": int(policy_status.get("warning_alerts", 0)),
                "total_alerts": int(policy_status.get("total_alerts", 0)),
                "acceptance_pass": bool(acceptance_pass),
                "decision_reason_code": _policy_reason_code(policy_status, acceptance_pass),
                "sidecar_set_coverage": float(
                    (set_status.get("summary") or {}).get("set_coverage", 0.0)
                ),
                "sidecar_singleton_rate": float(
                    (set_status.get("summary") or {}).get("singleton_rate", 0.0)
                ),
                "local_variant_promotion_pass": bool(selection_status.get("promotion_pass", False)),
                "local_selected_variant": str(selection_status.get("selected_variant", "")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=[
            "acceptance_pass",
            "policy_overall_pass",
            "coverage_90",
            "avg_width_90",
            "min_group_coverage_90",
        ],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)


def _extract_rank1_local_winner(source_run_tag: str) -> dict[str, Any] | None:
    source_ns = _namespace(source_run_tag, "phase1", "final", "rank-1")
    selection_status_path = _resolve_run_paths(source_ns)["selection_status"]
    benchmark_path = (
        _resolve_run_paths(source_ns)["data_dir"] / "conformal_variant_benchmark.parquet"
    )
    if not selection_status_path.exists() or not benchmark_path.exists():
        return None
    selection_status = _load_json(selection_status_path)
    if not bool(selection_status.get("promotion_pass", False)):
        return None
    bench = pd.read_parquet(benchmark_path)
    if bench.empty:
        return None
    promotable = bench.loc[bench["promotion_pass"].fillna(False)].copy()
    if promotable.empty:
        promotable = bench.head(1).copy()
    winner = promotable.iloc[0].to_dict()
    return _benchmark_row_to_design(winner)


def _build_resume_shortlist(
    *,
    source_run_tag: str,
    top_k_inner: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_paths = _reopen_artifact_paths(source_run_tag)
    aggregate = pd.read_parquet(source_paths["inner_aggregate"])
    aggregate_unique = _dedupe_designs(aggregate)
    source_seed = _normalize_design_row(aggregate_unique.iloc[0].to_dict())
    source_local_winner = _extract_rank1_local_winner(source_run_tag)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    resume_meta = {
        "source_seed_design": source_seed,
        "source_local_winner_included": False,
        "source_local_winner_design": source_local_winner,
    }

    if source_local_winner is not None:
        local_key = _semantic_candidate_key(source_local_winner)
        if local_key != _semantic_candidate_key(source_seed):
            selected.append(source_local_winner)
            seen.add(local_key)
            resume_meta["source_local_winner_included"] = True

    for row in aggregate_unique.to_dict(orient="records"):
        design = _normalize_design_row(row)
        key = _semantic_candidate_key(design)
        if key in seen:
            continue
        seen.add(key)
        selected.append(design)
        if len(selected) >= int(top_k_inner):
            break

    shortlist = pd.DataFrame(selected).reset_index(drop=True)
    if not shortlist.empty:
        shortlist["selection_rank"] = range(1, len(shortlist) + 1)
    resume_meta["source_aggregate_unique_path"] = str(source_paths["inner_aggregate"])
    return shortlist, resume_meta


def _run_phase2_search(
    *,
    run_tag: str,
    upstream_run_tag: str,
    env: dict[str, str],
    aggregated: pd.DataFrame,
    alpha_candidates_95: list[float],
    tuning_holdout_ratios: list[float],
    inner_random_states: list[int],
    partition_candidates: list[str],
    partition_probability_sources: list[str],
    n_score_bins_candidates: list[int],
    fallback_modes: list[str],
    score_scale_families: list[str],
    calibration_fractions: list[float],
    phase2_cfg: dict[str, Any],
    sidecar_cfg: dict[str, Any],
    validation_cfg: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    paths = _reopen_artifact_paths(run_tag)
    models_dir = paths["models_dir"]
    calibrator_dir = models_dir / "phase2_calibrators"
    calibrator_rows: list[dict[str, Any]] = []
    top_designs = _dedupe_designs(aggregated).head(int(phase2_cfg.get("top_k_designs", 3)))
    baseline_metrics: dict[str, float] | None = None
    baseline_path = calibrator_dir / "venn_abers.pkl"
    try:
        _resolved, baseline_metrics = _fit_calibrator(
            method="venn_abers",
            output_path=baseline_path,
            upstream_run_tag=upstream_run_tag,
        )
    except Exception:
        baseline_metrics = None

    max_metric_degradation = dict(phase2_cfg.get("max_metric_degradation", {}) or {})
    for method_name in phase2_cfg.get("calibrators", ["venn_abers", "isotonic", "platt", "beta"]):
        calibrator_path = calibrator_dir / f"{str(method_name).strip().lower()}.pkl"
        resolved_method, calibration_metrics = _fit_calibrator(
            method=str(method_name),
            output_path=calibrator_path,
            upstream_run_tag=upstream_run_tag,
        )
        if baseline_metrics is not None:
            metric_blocked = any(
                float(calibration_metrics.get(metric_name, float("inf")))
                > float(baseline_metrics.get(metric_name, 0.0))
                + float(max_metric_degradation.get(metric_name, 0.0))
                for metric_name in baseline_metrics
            )
            if metric_blocked:
                continue

        for design in top_designs.to_dict(orient="records"):
            design_norm = _normalize_design_row(design)
            namespace = _namespace(
                run_tag,
                "phase2",
                resolved_method,
                f"rank-{int(design.get('selection_rank', 1))}",
            )
            _run_python(
                "scripts/generate_conformal_intervals.py",
                [
                    "--artifact_namespace",
                    namespace,
                    "--evaluation_scope",
                    "holdout",
                    "--calibrator_override_path",
                    str(calibrator_path),
                    "--tuning_holdout_ratio",
                    str(float(tuning_holdout_ratios[0])),
                    "--tuning_random_state",
                    str(int(inner_random_states[0])),
                    "--alpha_candidates_95",
                    ",".join(str(x) for x in alpha_candidates_95),
                    *_design_args(design_norm),
                ],
                env,
            )
            payload = _load_pickle(_resolve_run_paths(namespace)["results"])
            metrics_90 = dict(payload.get("metrics_90", {}) or {})
            calibrator_rows.append(
                {
                    "artifact_namespace": namespace,
                    "calibrator_method": resolved_method,
                    "selection_rank": int(design.get("selection_rank", 1)),
                    **design_norm,
                    "holdout_coverage": float(metrics_90.get("empirical_coverage", 0.0)),
                    "holdout_width": float(metrics_90.get("avg_interval_width", 1.0)),
                    "calibrator_ece": float(calibration_metrics.get("ece", float("inf"))),
                    "calibrator_brier": float(calibration_metrics.get("brier_score", float("inf"))),
                    "calibrator_log_loss": float(calibration_metrics.get("log_loss", float("inf"))),
                }
            )

    phase2_df = pd.DataFrame(calibrator_rows)
    if phase2_df.empty:
        return (
            "policy_review_candidate",
            {},
            {},
            {
                "search_path": str(paths["phase2_search"]),
                "best_candidate": {},
                "status": "no_noninferior_calibrator_candidate",
            },
        )

    phase2_df["coverage_gap_abs"] = (phase2_df["holdout_coverage"] - 0.90).abs()
    phase2_df = phase2_df.sort_values(
        by=[
            "coverage_gap_abs",
            "holdout_width",
            "calibrator_ece",
            "calibrator_brier",
            "selection_rank",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    phase2_df.to_parquet(paths["phase2_search"], index=False)
    phase2_best = phase2_df.iloc[0].to_dict()
    _namespace(run_tag, "phase2", "final")
    phase2_calibrator_path = calibrator_dir / f"{phase2_best['calibrator_method']}.pkl"
    final_candidate = _run_phase1_oot_candidate(
        run_tag=run_tag,
        rank=1,
        design=phase2_best,
        env=env,
        alpha_candidates_95=alpha_candidates_95,
        partition_candidates=partition_candidates,
        partition_probability_sources=partition_probability_sources,
        n_score_bins_candidates=n_score_bins_candidates,
        fallback_modes=fallback_modes,
        score_scale_families=score_scale_families,
        calibration_fractions=calibration_fractions,
        sidecar_cfg=sidecar_cfg,
        calibrator_override_path=str(phase2_calibrator_path),
        phase_prefix="phase2",
    )
    final_policy = dict(final_candidate["policy_status"])
    final_sets = dict(final_candidate["set_status"])
    final_decision = (
        "promotable_for_followup"
        if _acceptance_pass(final_policy, validation_cfg)
        else "policy_review_candidate"
    )
    phase2_summary = {
        "search_path": str(paths["phase2_search"]),
        "best_candidate": phase2_best,
        "final_namespace": final_candidate["namespace"],
    }
    return final_decision, final_policy, final_sets, phase2_summary


def _write_consolidated_status(
    *,
    run_tag: str,
    upstream_run_tag: str,
    pipeline_profile: str,
    mode: str,
    inner_search_path: str,
    aggregate_path: str,
    shortlist_path: str,
    phase1_final_path: str,
    inner_search_runs: list[dict[str, Any]],
    inner_search_winner: dict[str, Any],
    best_phase1_namespace: str,
    final_policy: dict[str, Any],
    final_sets: dict[str, Any],
    final_decision: str,
    final_namespace: str,
    phase2_summary: dict[str, Any] | None,
    resume_meta: dict[str, Any] | None,
) -> None:
    status = {
        "schema_version": "2026-04-05.1",
        "generated_at_utc": _utc_now(),
        "run_tag": run_tag,
        "mode": mode,
        "upstream_canonical_run_tag": upstream_run_tag,
        "pipeline_profile": pipeline_profile,
        "inner_search_winner": inner_search_winner,
        "inner_search_path": inner_search_path,
        "inner_search_aggregate_path": aggregate_path,
        "phase1_shortlist_path": shortlist_path,
        "phase1_final_candidates_path": phase1_final_path,
        "inner_search_runs": inner_search_runs,
        "phase1_oot_namespace": best_phase1_namespace,
        "oot_confirmation_result": final_policy,
        "sidecar_set_result": final_sets,
        "promotion_decision": final_decision,
        "policy_review_needed": bool(final_decision == "policy_review_candidate"),
        "final_namespace": final_namespace,
        "phase2": phase2_summary,
    }
    if resume_meta is not None:
        status["resume_meta"] = resume_meta
    _write_json(_reopen_artifact_paths(run_tag)["status"], status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=_default_run_tag())
    parser.add_argument("--pipeline-profile", default="search_conformal_reopen_exhaustive")
    parser.add_argument("--upstream-canonical-run-tag", default="pd-hpo-local-2026-04-03-1325")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--resume-from-run-tag", default=None)
    args = parser.parse_args(argv)

    run_tag = str(args.run_tag).strip()
    upstream_run_tag = str(args.upstream_canonical_run_tag).strip()
    resume_from_run_tag = str(args.resume_from_run_tag).strip() if args.resume_from_run_tag else ""
    mode = "derived_resume" if resume_from_run_tag else "fresh_search"

    profile = _profile_cfg(args.pipeline_profile)
    cfg = _phase1_cfg(profile)
    phase2_cfg = _phase2_cfg(profile)
    sidecar_cfg = _sidecar_cfg(profile)
    validation_cfg = _validation_cfg(profile)
    output_paths = _reopen_artifact_paths(run_tag)

    env = os.environ.copy()
    env["PIPELINE_RUN_TAG"] = run_tag
    env["UPSTREAM_CANONICAL_RUN_TAG"] = upstream_run_tag

    alpha_candidates_90 = cfg.get("alpha_candidates_90", [0.09, 0.095, 0.10, 0.105, 0.11, 0.12])
    alpha_candidates_95 = cfg.get("alpha_candidates_95", [0.045, 0.05, 0.055, 0.06])
    partition_candidates = cfg.get(
        "partition_candidates",
        ["grade", "score_decile_mondrian", "grade_x_scoreband_mondrian"],
    )
    partition_probability_sources = cfg.get("partition_probability_sources", ["calibrated", "raw"])
    n_score_bins_candidates = cfg.get("n_score_bins_candidates", [5, 10, 15, 20])
    min_group_sizes = cfg.get("min_group_sizes", [100, 150, 250, 500, 1000])
    fallback_modes = cfg.get("fallback_modes", ["grade_then_global", "global_only"])
    score_scale_families = cfg.get(
        "score_scale_families",
        ["none", "bernoulli_sqrt", "bernoulli_sqrt_clipped_0.02", "bernoulli_sqrt_clipped_0.05"],
    )
    calibration_fractions = cfg.get("calibration_fractions", [0.25, 0.50, 0.75, 1.00])
    tuning_holdout_ratios = cfg.get("tuning_holdout_ratios", [0.20, 0.30])
    inner_random_states = cfg.get("inner_random_states", [42, 314, 2026])
    top_k_inner = int(validation_cfg.get("top_k_inner", 3))

    resume_meta: dict[str, Any] | None = None
    inner_runs: list[dict[str, Any]] = []
    if resume_from_run_tag:
        source_paths = _reopen_artifact_paths(resume_from_run_tag)
        if not source_paths["inner_aggregate"].exists():
            raise FileNotFoundError(
                f"Resume source missing aggregate artifact: {source_paths['inner_aggregate']}"
            )
        shortlist, resume_meta = _build_resume_shortlist(
            source_run_tag=resume_from_run_tag,
            top_k_inner=top_k_inner,
        )
        shortlist.to_parquet(output_paths["phase1_shortlist"], index=False)
        aggregate_path = str(source_paths["inner_aggregate"])
        inner_search_path = str(source_paths["inner_search"])
        aggregated = pd.read_parquet(source_paths["inner_aggregate"])
        inner_search_winner = (
            _normalize_design_row(aggregated.iloc[0].to_dict()) if not aggregated.empty else {}
        )
    else:
        inner_frames: list[pd.DataFrame] = []
        for calibration_fraction in calibration_fractions:
            for holdout_ratio in tuning_holdout_ratios:
                for random_state in inner_random_states:
                    namespace = _namespace(
                        run_tag,
                        "phase1",
                        f"calfrac-{float(calibration_fraction):.2f}",
                        f"holdout-{float(holdout_ratio):.2f}",
                        f"seed-{int(random_state)}",
                    )
                    _run_python(
                        "scripts/generate_conformal_intervals.py",
                        [
                            "--artifact_namespace",
                            namespace,
                            "--evaluation_scope",
                            "holdout",
                            "--alpha_candidates_90",
                            ",".join(str(x) for x in alpha_candidates_90),
                            "--alpha_candidates_95",
                            ",".join(str(x) for x in alpha_candidates_95),
                            "--partition_candidates",
                            ",".join(str(x) for x in partition_candidates),
                            "--partition_probability_sources",
                            ",".join(str(x) for x in partition_probability_sources),
                            "--n_score_bins_candidates",
                            ",".join(str(int(x)) for x in n_score_bins_candidates),
                            "--fallback_modes",
                            ",".join(str(x) for x in fallback_modes),
                            "--min_group_sizes",
                            ",".join(str(int(x)) for x in min_group_sizes),
                            "--score_scale_families",
                            ",".join(str(x) for x in score_scale_families),
                            "--calibration_fraction",
                            str(float(calibration_fraction)),
                            "--tuning_holdout_ratio",
                            str(float(holdout_ratio)),
                            "--tuning_random_state",
                            str(int(random_state)),
                        ],
                        env,
                    )
                    tuning_df = pd.read_parquet(_resolve_run_paths(namespace)["tuning"])
                    tuning_df["calibration_fraction"] = float(calibration_fraction)
                    tuning_df["holdout_ratio"] = float(holdout_ratio)
                    tuning_df["random_state"] = int(random_state)
                    tuning_df["artifact_namespace"] = namespace
                    inner_frames.append(tuning_df)
                    result_payload = _load_pickle(_resolve_run_paths(namespace)["results"])
                    inner_runs.append(
                        {
                            "artifact_namespace": namespace,
                            "calibration_fraction": float(calibration_fraction),
                            "holdout_ratio": float(holdout_ratio),
                            "random_state": int(random_state),
                            "metrics_90": result_payload.get("metrics_90", {}),
                            "tuning_90_best": result_payload.get("tuning_90_best", {}),
                            "alpha_used_95": result_payload.get("alpha_used_95"),
                        }
                    )

        inner_df = pd.concat(inner_frames, ignore_index=True)
        inner_df.to_parquet(output_paths["inner_search"], index=False)
        aggregated = _aggregate_inner_search(inner_df)
        aggregated.to_parquet(output_paths["inner_aggregate"], index=False)
        shortlist = _dedupe_designs(aggregated, top_k_inner)
        shortlist.to_parquet(output_paths["phase1_shortlist"], index=False)
        aggregate_path = str(output_paths["inner_aggregate"])
        inner_search_path = str(output_paths["inner_search"])
        inner_search_winner = (
            _normalize_design_row(aggregated.iloc[0].to_dict()) if not aggregated.empty else {}
        )

    if shortlist.empty:
        raise RuntimeError("No phase1 shortlist candidates available for OOT confirmation.")

    phase1_candidates: list[dict[str, Any]] = []
    for rank, design in enumerate(shortlist.to_dict(orient="records"), start=1):
        phase1_candidates.append(
            _run_phase1_oot_candidate(
                run_tag=run_tag,
                rank=rank,
                design=design,
                env=env,
                alpha_candidates_95=alpha_candidates_95,
                partition_candidates=partition_candidates,
                partition_probability_sources=partition_probability_sources,
                n_score_bins_candidates=n_score_bins_candidates,
                fallback_modes=fallback_modes,
                score_scale_families=score_scale_families,
                calibration_fractions=calibration_fractions,
                sidecar_cfg=sidecar_cfg,
            )
        )

    phase1_df = _phase1_candidates_frame(phase1_candidates, validation_cfg)
    phase1_df.to_parquet(output_paths["phase1_final_candidates"], index=False)

    best_phase1_ns = str(phase1_df.iloc[0]["namespace"])
    best_phase1 = next(
        candidate for candidate in phase1_candidates if candidate["namespace"] == best_phase1_ns
    )
    final_namespace = best_phase1["namespace"]
    final_policy = dict(best_phase1["policy_status"])
    final_sets = dict(best_phase1["set_status"])
    final_decision = (
        "promotable_for_followup"
        if _acceptance_pass(final_policy, validation_cfg)
        else "keep_current_canonical"
    )
    phase2_summary: dict[str, Any] | None = None

    if (
        (not _acceptance_pass(final_policy, validation_cfg))
        and (not bool(args.phase1_only))
        and bool(phase2_cfg.get("enabled", True))
    ):
        final_decision, final_policy, final_sets, phase2_summary = _run_phase2_search(
            run_tag=run_tag,
            upstream_run_tag=upstream_run_tag,
            env=env,
            aggregated=aggregated,
            alpha_candidates_95=alpha_candidates_95,
            tuning_holdout_ratios=tuning_holdout_ratios,
            inner_random_states=inner_random_states,
            partition_candidates=partition_candidates,
            partition_probability_sources=partition_probability_sources,
            n_score_bins_candidates=n_score_bins_candidates,
            fallback_modes=fallback_modes,
            score_scale_families=score_scale_families,
            calibration_fractions=calibration_fractions,
            phase2_cfg=phase2_cfg,
            sidecar_cfg=sidecar_cfg,
            validation_cfg=validation_cfg,
        )
        if phase2_summary and phase2_summary.get("final_namespace"):
            final_namespace = str(phase2_summary["final_namespace"])

    _write_consolidated_status(
        run_tag=run_tag,
        upstream_run_tag=upstream_run_tag,
        pipeline_profile=str(args.pipeline_profile),
        mode=mode,
        inner_search_path=inner_search_path,
        aggregate_path=aggregate_path,
        shortlist_path=str(output_paths["phase1_shortlist"]),
        phase1_final_path=str(output_paths["phase1_final_candidates"]),
        inner_search_runs=inner_runs,
        inner_search_winner=inner_search_winner,
        best_phase1_namespace=best_phase1_ns,
        final_policy=final_policy,
        final_sets=final_sets,
        final_decision=final_decision,
        final_namespace=final_namespace,
        phase2_summary=phase2_summary,
        resume_meta=resume_meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
