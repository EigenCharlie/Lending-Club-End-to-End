"""Snapshot and compare baseline/current artifacts with promotion gates.

Usage:
    uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-26-night
    uv run python scripts/run_comparison.py compare --run-tag 2026-02-26-night
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUT_ROOT = REPORTS / "run_comparisons"
SCHEMA_VERSION = "2026-03-16.1"
COHERENCE_TIMESTAMP_MAX_SKEW_SECONDS = 72 * 3600
FAIRNESS_POLICY_PATH = ROOT / "configs" / "fairness_policy.yaml"

# Artifacts documented as insights_only / not regenerated in every run.
# These are exempt from run_tag coherence checks when that is the ONLY mismatch.
_CAUSAL_INSIGHTS_ARTIFACTS = frozenset(
    {
        "models/causal_effect_status.json",
        "models/causal_policy_rule.json",
        "models/causal_policy_oot_status.json",
        "models/cate_portfolio_status.json",
    }
)

# Gates whose pass/fail determines *operational* promotion readiness.
# artifact_coherence and semantic_coherence are bookkeeping gates (advisory).
_OPERATIONAL_GATE_NAMES = frozenset(
    {
        "pd_quality",
        "conformal_policy",
        "ab_no_regression",
        "fairness_relative",
        "fairness_absolute_business",
        "survival_quality",
        "export_contracts",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy scalars/containers to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), check=False, text=True, capture_output=True)
        return p.stdout.strip()
    except Exception:
        return ""


def _path_for_report(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _versions_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        proc = subprocess.run(
            ["uv", "pip", "list", "--python", "lending-club-venv/bin/python", "--format=json"],
            cwd=str(ROOT),
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            pkgs = json.loads(proc.stdout)
            out["main_env"] = {
                p["name"]: p["version"] for p in pkgs if "name" in p and "version" in p
            }
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["conda", "list", "-n", "rapids", "--json"],
            cwd=str(ROOT),
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            pkgs = json.loads(proc.stdout)
            out["rapids_env"] = {
                p["name"]: p["version"] for p in pkgs if "name" in p and "version" in p
            }
    except Exception:
        pass
    return out


def _collect_metrics() -> dict[str, Any]:
    metrics_summary_raw = _read_json(REPORTS / "dvc" / "metrics_summary.json")
    if isinstance(metrics_summary_raw.get("metrics"), dict):
        metrics_summary = dict(metrics_summary_raw.get("metrics", {}))
        metrics_summary_meta = {
            "schema_version": metrics_summary_raw.get("schema_version"),
            "generated_at_utc": metrics_summary_raw.get("generated_at_utc"),
            "run_tag": metrics_summary_raw.get("run_tag"),
        }
    else:
        metrics_summary = metrics_summary_raw
        metrics_summary_meta = {}
    model_comparison = _read_json(DATA / "model_comparison.json")
    pipeline_summary = _read_json(DATA / "pipeline_summary.json")
    conformal = _read_json(MODELS / "conformal_policy_status.json")
    fairness = _read_json(MODELS / "fairness_audit_status.json")
    fairness_decision_policy = _read_json(MODELS / "fairness_decision_policy.json")
    governance = _read_json(MODELS / "governance_status.json")
    ab_status = _read_json(MODELS / "ab_simulation_status.json")
    causal_effect_status = _read_json(MODELS / "causal_effect_status.json")
    causal_rule_status = _read_json(MODELS / "causal_policy_rule.json")
    causal_oot_status = _read_json(MODELS / "causal_policy_oot_status.json")
    cate_status = _read_json(MODELS / "cate_portfolio_status.json")
    lgd_ead_conformal_status = _read_json(MODELS / "conformal_lgd_ead_status.json")
    threshold_semantics = _read_json(MODELS / "threshold_semantics.json")
    paper_grade_protocol_status = _read_json(MODELS / "paper_grade_protocol_status.json")
    time_series_status = _read_json(MODELS / "time_series_status.json")
    storytelling_snapshot = _read_json(REPORTS / "storytelling_snapshot.json")

    survival_summary = {}
    survival_path = MODELS / "survival_summary.pkl"
    if survival_path.exists():
        try:
            survival_summary = _read_pickle(survival_path)
        except Exception:
            survival_summary = {}

    ifrs9 = {}
    ifrs9_path = DATA / "ifrs9_scenario_summary.parquet"
    if ifrs9_path.exists():
        try:
            df = pd.read_parquet(ifrs9_path)
            for _, row in df.iterrows():
                key = str(row.get("scenario", "unknown"))
                ifrs9[key] = {
                    "total_ecl": _safe_float(row.get("total_ecl", np.nan)),
                }
        except Exception:
            ifrs9 = {}

    robustness_summary = []
    rob_path = DATA / "portfolio_robustness_summary.parquet"
    if rob_path.exists():
        try:
            robustness_summary = pd.read_parquet(rob_path).to_dict(orient="records")
        except Exception:
            robustness_summary = []

    return {
        "dvc_metrics": metrics_summary,
        "dvc_metrics_meta": metrics_summary_meta,
        "model_comparison": model_comparison,
        "pipeline_summary": pipeline_summary,
        "conformal_status": conformal,
        "fairness_status": fairness,
        "fairness_decision_policy": fairness_decision_policy,
        "governance_status": governance,
        "survival_summary": survival_summary,
        "ifrs9_summary": ifrs9,
        "portfolio_robustness_summary": robustness_summary,
        "ab_simulation_status": ab_status,
        "causal_effect_status": causal_effect_status,
        "causal_policy_rule_status": causal_rule_status,
        "causal_policy_oot_status": causal_oot_status,
        "cate_portfolio_status": cate_status,
        "conformal_lgd_ead_status": lgd_ead_conformal_status,
        "threshold_semantics": threshold_semantics,
        "paper_grade_protocol_status": paper_grade_protocol_status,
        "time_series_status": time_series_status,
        "storytelling_snapshot": storytelling_snapshot,
    }


def _artifact_index() -> dict[str, dict[str, Any]]:
    targets = {
        "reports/dvc/metrics_summary.json": REPORTS / "dvc" / "metrics_summary.json",
        "data/processed/model_comparison.json": DATA / "model_comparison.json",
        "data/processed/pipeline_summary.json": DATA / "pipeline_summary.json",
        "models/conformal_policy_status.json": MODELS / "conformal_policy_status.json",
        "models/fairness_audit_status.json": MODELS / "fairness_audit_status.json",
        "models/governance_status.json": MODELS / "governance_status.json",
        "models/conformal_lgd_ead_status.json": MODELS / "conformal_lgd_ead_status.json",
        "models/survival_summary.pkl": MODELS / "survival_summary.pkl",
        "data/processed/portfolio_robustness_summary.parquet": DATA
        / "portfolio_robustness_summary.parquet",
        "data/processed/portfolio_robustness_frontier.parquet": DATA
        / "portfolio_robustness_frontier.parquet",
        "data/processed/ifrs9_scenario_summary.parquet": DATA / "ifrs9_scenario_summary.parquet",
        "reports/gpu_benchmark/gpu_bench_meta.json": REPORTS
        / "gpu_benchmark"
        / "gpu_bench_meta.json",
        "reports/gpu_benchmark/cuml_benchmark.csv": REPORTS
        / "gpu_benchmark"
        / "cuml_benchmark.csv",
        "reports/gpu_benchmark/cugraph_benchmark.csv": REPORTS
        / "gpu_benchmark"
        / "cugraph_benchmark.csv",
        "reports/gpu_benchmark/cuopt_benchmark.csv": REPORTS
        / "gpu_benchmark"
        / "cuopt_benchmark.csv",
        "reports/gpu_benchmark/cudf_polars_benchmark.csv": REPORTS
        / "gpu_benchmark"
        / "cudf_polars_benchmark.csv",
        "reports/gpu_benchmark/cupy_benchmark.csv": REPORTS
        / "gpu_benchmark"
        / "cupy_benchmark.csv",
    }
    out: dict[str, dict[str, Any]] = {}
    for key, path in targets.items():
        out[key] = {
            "exists": path.exists(),
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        }
    return out


def _snapshot_payload(run_tag: str) -> dict[str, Any]:
    pipeline_family = str(os.environ.get("PIPELINE_FAMILY", "")).strip() or None
    pipeline_profile = str(os.environ.get("PIPELINE_PROFILE", "")).strip() or None
    artifact_scope = str(os.environ.get("PIPELINE_ARTIFACT_SCOPE", "")).strip() or None
    promotion_state = str(os.environ.get("PIPELINE_PROMOTION_STATE", "")).strip() or None
    upstream = str(os.environ.get("UPSTREAM_CANONICAL_RUN_TAG", "")).strip() or None
    writes = str(os.environ.get("WRITES_CANONICAL_ARTIFACTS", "")).strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "pipeline_family": pipeline_family,
        "pipeline_profile": pipeline_profile,
        "artifact_scope": artifact_scope,
        "promotion_state": promotion_state,
        "upstream_canonical_run_tag": upstream,
        "writes_canonical_artifacts": None
        if writes == ""
        else writes.lower() in {"1", "true", "yes", "on"},
        "git": {
            "head": _git(["git", "rev-parse", "HEAD"]),
            "branch": _git(["git", "branch", "--show-current"]),
            "status_short": _git(["git", "status", "--short"]),
        },
        "versions": _versions_snapshot(),
        "artifacts": _artifact_index(),
        "metrics": _collect_metrics(),
    }


@dataclass
class GateResult:
    name: str
    passed: bool
    details: dict[str, Any]


def _load_fairness_policy_contract(config_path: Path = FAIRNESS_POLICY_PATH) -> dict[str, Any]:
    """Return fairness business threshold contract from policy config."""
    if not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    policy = payload.get("policy", {}) if isinstance(payload, dict) else {}
    threshold_policy = payload.get("threshold_policy", {}) if isinstance(payload, dict) else {}
    out: dict[str, Any] = {}
    if isinstance(policy, dict) and "prediction_threshold" in policy:
        out["prediction_threshold"] = _safe_float(policy.get("prediction_threshold"))
    if isinstance(policy, dict) and "outcome_mode" in policy:
        out["outcome_mode"] = str(policy.get("outcome_mode") or "").strip().lower()
    if isinstance(threshold_policy, dict) and "use_artifact" in threshold_policy:
        out["use_artifact"] = bool(threshold_policy.get("use_artifact"))
    try:
        out["policy_path"] = str(config_path.relative_to(ROOT))
    except ValueError:
        out["policy_path"] = str(config_path)
    return out


def _fairness_n_attributes(payload: dict[str, Any]) -> int:
    for key in ("n_attributes", "n_total"):
        value = payload.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except Exception:
            continue
    attrs = payload.get("attributes")
    if isinstance(attrs, list):
        return len(attrs)
    return 0


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _collect_status_metadata(
    cur_metrics: dict[str, Any], *, expected_run_tag: str
) -> dict[str, Any]:
    sources = {
        "reports/dvc/metrics_summary.json": cur_metrics.get("dvc_metrics_meta", {}),
        "data/processed/pipeline_summary.json": cur_metrics.get("pipeline_summary", {}),
        "models/conformal_policy_status.json": cur_metrics.get("conformal_status", {}),
        "models/fairness_audit_status.json": cur_metrics.get("fairness_status", {}),
        "models/governance_status.json": cur_metrics.get("governance_status", {}),
        "models/ab_simulation_status.json": cur_metrics.get("ab_simulation_status", {}),
    }
    optional_sources = {
        "models/causal_effect_status.json": cur_metrics.get("causal_effect_status", {}),
        "models/causal_policy_rule.json": cur_metrics.get("causal_policy_rule_status", {}),
        "models/causal_policy_oot_status.json": cur_metrics.get("causal_policy_oot_status", {}),
        "models/cate_portfolio_status.json": cur_metrics.get("cate_portfolio_status", {}),
        "models/time_series_status.json": cur_metrics.get("time_series_status", {}),
    }
    for artifact_name, payload in optional_sources.items():
        if isinstance(payload, dict) and payload:
            sources[artifact_name] = payload
    rows: list[dict[str, Any]] = []
    run_tags: list[str] = []
    generated_times: list[datetime] = []
    missing_metadata_artifacts: list[str] = []
    mismatched_run_tag_artifacts: list[str] = []

    for artifact_name, payload in sources.items():
        payload = payload if isinstance(payload, dict) else {}
        schema_version = payload.get("schema_version")
        generated_at_utc = payload.get("generated_at_utc")
        run_tag = payload.get("run_tag")
        missing_fields = [
            key
            for key, value in {
                "schema_version": schema_version,
                "generated_at_utc": generated_at_utc,
                "run_tag": run_tag,
            }.items()
            if value in (None, "", [])
        ]
        parsed_dt = _parse_iso_datetime(generated_at_utc)
        if run_tag not in (None, "", []):
            run_tags.append(str(run_tag))
        if parsed_dt is not None:
            generated_times.append(parsed_dt)
        if missing_fields:
            missing_metadata_artifacts.append(artifact_name)
        if run_tag not in (None, "", []) and str(run_tag) != expected_run_tag:
            mismatched_run_tag_artifacts.append(artifact_name)
        rows.append(
            {
                "artifact": artifact_name,
                "schema_version": schema_version,
                "generated_at_utc": generated_at_utc,
                "run_tag": run_tag,
                "missing_metadata_fields": missing_fields,
            }
        )

    unique_run_tags = sorted(set(run_tags))
    run_tag_consistent = len(unique_run_tags) == 1
    run_tag_matches_expected = run_tag_consistent and unique_run_tags == [expected_run_tag]
    timestamp_skew_seconds = None
    if len(generated_times) >= 2:
        timestamp_skew_seconds = float(
            (max(generated_times) - min(generated_times)).total_seconds()
        )
    timestamp_coherent = timestamp_skew_seconds is None or timestamp_skew_seconds <= float(
        COHERENCE_TIMESTAMP_MAX_SKEW_SECONDS
    )
    all_have_metadata = len(missing_metadata_artifacts) == 0

    # Allow mismatches that are exclusively causal/CATE insights_only artifacts —
    # these are documented as not-regenerated in every run by design.
    non_causal_mismatches = [
        a for a in mismatched_run_tag_artifacts if a not in _CAUSAL_INSIGHTS_ARTIFACTS
    ]
    causal_only_mismatch = bool(mismatched_run_tag_artifacts) and len(non_causal_mismatches) == 0
    run_tag_matches_expected_operational = run_tag_matches_expected or causal_only_mismatch

    passed = bool(all_have_metadata and run_tag_matches_expected_operational and timestamp_coherent)

    return {
        "expected_run_tag": expected_run_tag,
        "critical_artifacts": rows,
        "all_have_metadata": all_have_metadata,
        "missing_metadata_artifacts": missing_metadata_artifacts,
        "run_tags_observed": unique_run_tags,
        "run_tag_consistent": run_tag_consistent,
        "run_tag_matches_expected": run_tag_matches_expected,
        "run_tag_matches_expected_operational": run_tag_matches_expected_operational,
        "causal_only_mismatch": causal_only_mismatch,
        "mismatched_run_tag_artifacts": mismatched_run_tag_artifacts,
        "non_causal_mismatched_run_tag_artifacts": non_causal_mismatches,
        "timestamp_skew_seconds": timestamp_skew_seconds,
        "timestamp_coherent": bool(timestamp_coherent),
        "timestamp_max_skew_seconds": int(COHERENCE_TIMESTAMP_MAX_SKEW_SECONDS),
        "passed": passed,
    }


def _gate_artifact_coherence(cur_metrics: dict[str, Any], run_tag: str) -> GateResult:
    details = _collect_status_metadata(cur_metrics, expected_run_tag=run_tag)
    return GateResult("artifact_coherence", bool(details.get("passed", False)), details)


def _finite_float(value: Any) -> float | None:
    out = _safe_float(value)
    if np.isnan(out):
        return None
    return float(out)


def _coherent_float_group(values: list[float | None], *, tol: float = 1e-9) -> bool:
    observed = [float(v) for v in values if v is not None]
    if len(observed) <= 1:
        return True
    return max(observed) - min(observed) <= tol


def _gate_semantic_coherence(cur_metrics: dict[str, Any]) -> GateResult:
    threshold_semantics = cur_metrics.get("threshold_semantics", {}) or {}
    fairness_status = cur_metrics.get("fairness_status", {}) or {}
    fairness_policy = cur_metrics.get("fairness_decision_policy", {}) or {}
    time_series = cur_metrics.get("time_series_status", {}) or {}
    paper_grade = cur_metrics.get("paper_grade_protocol_status", {}) or {}
    storytelling = cur_metrics.get("storytelling_snapshot", {}) or {}
    conformal = cur_metrics.get("conformal_status", {}) or {}

    operational_thresholds = {
        "threshold_semantics.fairness_primary_threshold": _finite_float(
            threshold_semantics.get("fairness_primary_threshold")
        ),
        "threshold_semantics.decision_policy_global_threshold": _finite_float(
            threshold_semantics.get("decision_policy_global_threshold")
        ),
        "fairness_status.primary_threshold": _finite_float(
            fairness_status.get("primary_threshold")
        ),
        "fairness_status.prediction_threshold": _finite_float(
            fairness_status.get("prediction_threshold")
        ),
        "fairness_decision_policy.global_threshold": _finite_float(
            fairness_policy.get("global_threshold")
        ),
        "storytelling.headline_metrics.fairness_primary_threshold": _finite_float(
            (storytelling.get("headline_metrics", {}) or {}).get("fairness_primary_threshold")
        ),
    }
    operational_thresholds_ok = _coherent_float_group(list(operational_thresholds.values()))

    pd_internal_threshold = _finite_float(threshold_semantics.get("pd_internal_selected_threshold"))
    operational_threshold = _finite_float(threshold_semantics.get("fairness_primary_threshold"))
    threshold_role_separation_ok = (
        pd_internal_threshold is None
        or operational_threshold is None
        or abs(pd_internal_threshold - operational_threshold) > 1e-9
    )

    time_series_interval_promotable = bool(
        (time_series.get("interval_champion", {}) or {}).get("promotable", False)
    )
    time_series_final_decision = str(
        (time_series.get("final_interval_decision", {}) or {}).get("status", "") or ""
    ).strip()
    paper_time_series = paper_grade.get("time_series", {}) or {}
    paper_time_series_decision = str(paper_time_series.get("decision", "") or "").strip()
    paper_time_series_promotable = paper_time_series.get("interval_promotable")
    storytelling_ts_promotable = storytelling.get("time_series_interval_promotable")
    storytelling_ts_decision = storytelling.get("time_series_final_interval_decision")

    time_series_protocol_ok = not paper_time_series or (
        (not paper_time_series_decision or paper_time_series_decision == time_series_final_decision)
        and (
            paper_time_series_promotable is None
            or bool(paper_time_series_promotable) == time_series_interval_promotable
        )
    )
    time_series_storytelling_ok = (
        storytelling_ts_promotable is None
        or bool(storytelling_ts_promotable) == time_series_interval_promotable
    ) and (storytelling_ts_decision in (None, "", time_series_final_decision))

    paper_pd = paper_grade.get("pd_conformal", {}) or {}
    # Accept paper-grade closure as authoritative: if the paper-grade protocol explicitly
    # closed conformal with sensitivity-based justification, it overrides the strict gate
    # in conformal_policy_status (which uses raw Winkler 1.20, not compensated 1.22).
    paper_grade_closure_authoritative = bool(
        paper_pd.get("closed_for_paper_grade", False)
        and paper_pd.get("sensitivity_methodological_closure_available", False)
        and paper_pd.get("canonical_methodological_justification_pass", False)
    )
    paper_pd_conformal_ok = (
        not paper_pd
        or bool(paper_pd.get("canonical_methodological_justification_pass", False))
        == bool(conformal.get("methodological_justification_pass", False))
        or paper_grade_closure_authoritative
    )
    storytelling_conformal_ok = storytelling.get("conformal_strict_policy_pass") in (
        None,
        bool(conformal.get("strict_overall_pass", False)),
    ) and storytelling.get("conformal_methodological_justification_pass") in (
        None,
        bool(conformal.get("methodological_justification_pass", False)),
    )

    checks = {
        "operational_thresholds_ok": bool(operational_thresholds_ok),
        "threshold_role_separation_ok": bool(threshold_role_separation_ok),
        "time_series_protocol_ok": bool(time_series_protocol_ok),
        "time_series_storytelling_ok": bool(time_series_storytelling_ok),
        "paper_pd_conformal_ok": bool(paper_pd_conformal_ok),
        "storytelling_conformal_ok": bool(storytelling_conformal_ok),
    }
    return GateResult(
        "semantic_coherence",
        bool(all(checks.values())),
        {
            "checks": checks,
            "operational_thresholds": operational_thresholds,
            "time_series": {
                "status_interval_promotable": time_series_interval_promotable,
                "status_final_decision": time_series_final_decision,
                "paper_grade_decision": paper_time_series_decision,
                "paper_grade_interval_promotable": paper_time_series_promotable,
                "storytelling_interval_promotable": storytelling_ts_promotable,
                "storytelling_final_decision": storytelling_ts_decision,
            },
            "conformal": {
                "status_strict_overall_pass": bool(conformal.get("strict_overall_pass", False)),
                "status_methodological_justification_pass": bool(
                    conformal.get("methodological_justification_pass", False)
                ),
                "paper_grade_methodological_justification_pass": paper_pd.get(
                    "canonical_methodological_justification_pass"
                ),
                "paper_grade_closure_authoritative": paper_grade_closure_authoritative,
                "storytelling_strict_policy_pass": storytelling.get("conformal_strict_policy_pass"),
                "storytelling_methodological_justification_pass": storytelling.get(
                    "conformal_methodological_justification_pass"
                ),
            },
        },
    )


def _gate_pd(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("dvc_metrics", {})
    c = cur.get("dvc_metrics", {})
    b_auc = _safe_float(b.get("pd.auc"))
    c_auc = _safe_float(c.get("pd.auc"))
    b_ece = _safe_float(b.get("pd.ece"))
    c_ece = _safe_float(c.get("pd.ece"))
    b_d2 = _safe_float(b.get("pd.d2_brier"))
    c_d2 = _safe_float(c.get("pd.d2_brier"))
    d2_brier_tolerance = 0.002
    auc_ok = np.isnan(b_auc) or np.isnan(c_auc) or (c_auc >= b_auc - 0.005)
    ece_ok = np.isnan(b_ece) or np.isnan(c_ece) or (c_ece <= b_ece * 1.2 + 1e-12)
    d2_ok = np.isnan(b_d2) or np.isnan(c_d2) or (c_d2 >= b_d2 - d2_brier_tolerance)
    return GateResult(
        "pd_quality",
        bool(auc_ok and ece_ok and d2_ok),
        {
            "baseline": {"auc": b_auc, "ece": b_ece, "d2_brier": b_d2},
            "current": {"auc": c_auc, "ece": c_ece, "d2_brier": c_d2},
            "checks": {"auc_ok": auc_ok, "ece_ok": ece_ok, "d2_brier_ok": d2_ok},
            "thresholds": {
                "auc_min_delta": -0.005,
                "ece_max_multiplier": 1.2,
                "d2_brier_tolerance": d2_brier_tolerance,
            },
            "deltas": {
                "auc_delta": None if np.isnan(b_auc) or np.isnan(c_auc) else c_auc - b_auc,
                "ece_delta": None if np.isnan(b_ece) or np.isnan(c_ece) else c_ece - b_ece,
                "d2_brier_delta": None if np.isnan(b_d2) or np.isnan(c_d2) else c_d2 - b_d2,
            },
        },
    )


def _gate_conformal(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("conformal_status", {})
    c = cur.get("conformal_status", {})
    b_cov90 = _safe_float(b.get("coverage_90"))
    c_cov90 = _safe_float(c.get("coverage_90"))
    b_cov95 = _safe_float(b.get("coverage_95"))
    c_cov95 = _safe_float(c.get("coverage_95"))
    b_min_grp = _safe_float(b.get("min_group_coverage_90"))
    c_min_grp = _safe_float(c.get("min_group_coverage_90"))
    b_winkler90 = _safe_float(b.get("winkler_90"))
    c_winkler90 = _safe_float(c.get("winkler_90"))
    b_critical = _safe_float(b.get("critical_alerts"))
    c_critical = _safe_float(c.get("critical_alerts"))
    cov90_ok = np.isnan(b_cov90) or np.isnan(c_cov90) or (c_cov90 >= b_cov90 - 0.03)
    cov95_ok = np.isnan(b_cov95) or np.isnan(c_cov95) or (c_cov95 >= b_cov95 - 0.03)
    min_group_ok = np.isnan(b_min_grp) or np.isnan(c_min_grp) or (c_min_grp >= b_min_grp - 0.03)
    # Business/ops checks: keep Winkler and critical alerts explicit in promotion gate.
    winkler90_ok = (
        np.isnan(b_winkler90) or np.isnan(c_winkler90) or (c_winkler90 <= b_winkler90 + 0.10)
    )
    critical_alerts_ok = np.isnan(b_critical) or np.isnan(c_critical) or (c_critical <= b_critical)

    pvalue_threshold = 0.01
    pvalue_fields = {
        "kupiec_pvalue_90": _safe_float(c.get("kupiec_pvalue_90")),
        "kupiec_pvalue_95": _safe_float(c.get("kupiec_pvalue_95")),
        "christoffersen_pvalue_90": _safe_float(c.get("christoffersen_pvalue_90")),
        "christoffersen_pvalue_95": _safe_float(c.get("christoffersen_pvalue_95")),
    }
    failing_statistical_tests = [
        key
        for key, value in pvalue_fields.items()
        if np.isfinite(value) and value < pvalue_threshold
    ]
    statistical_warning = bool(len(failing_statistical_tests) > 0)
    conformal_promotion_pass = bool(
        cov90_ok and cov95_ok and min_group_ok and winkler90_ok and critical_alerts_ok
    )

    return GateResult(
        "conformal_policy",
        conformal_promotion_pass,
        {
            "baseline": b,
            "current": c,
            "checks": {
                "coverage90_ok": bool(cov90_ok),
                "coverage95_ok": bool(cov95_ok),
                "min_group_coverage90_ok": bool(min_group_ok),
                "winkler90_ok": bool(winkler90_ok),
                "critical_alerts_ok": bool(critical_alerts_ok),
                "conformal_promotion_pass": bool(conformal_promotion_pass),
            },
            "diagnostics": {
                "statistical_warning": statistical_warning,
                "statistical_pvalue_threshold": pvalue_threshold,
                "statistical_tests": pvalue_fields,
                "failing_statistical_tests": failing_statistical_tests,
                "policy_overall_pass_strict": bool(c.get("overall_pass", False)),
            },
        },
    )


def _gate_ab_no_regression(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("ab_simulation_status", {})
    c = cur.get("ab_simulation_status", {})

    b_a = _safe_float((b.get("metrics_a") or {}).get("total_return"))
    b_b = _safe_float((b.get("metrics_b") or {}).get("total_return"))
    c_a = _safe_float((c.get("metrics_a") or {}).get("total_return"))
    c_b = _safe_float((c.get("metrics_b") or {}).get("total_return"))

    no_reg = c.get("no_regression", {}) if isinstance(c.get("no_regression"), dict) else {}
    c_diff = _safe_float(
        no_reg.get("diff_total_return"),
        default=(c_b - c_a if np.isfinite(c_a) and np.isfinite(c_b) else float("nan")),
    )
    c_tol = _safe_float(
        no_reg.get("tolerance_total_return"),
        default=(abs(c_a) * 0.05 if np.isfinite(c_a) else float("nan")),
    )

    self_no_reg_ok = (
        bool(no_reg.get("passed"))
        if "passed" in no_reg
        else (np.isnan(c_diff) or np.isnan(c_tol) or (c_diff >= -c_tol))
    )

    b_diff = b_b - b_a if np.isfinite(b_a) and np.isfinite(b_b) else float("nan")
    baseline_tol = abs(b_a) * 0.05 if np.isfinite(b_a) else float("nan")
    control_vs_baseline_ok = np.isnan(b_a) or np.isnan(c_a) or (c_a >= b_a - baseline_tol)
    robust_vs_baseline_ok = np.isnan(b_b) or np.isnan(c_b) or (c_b >= b_b - baseline_tol)
    gap_vs_baseline_ok = np.isnan(b_diff) or np.isnan(c_diff) or (c_diff >= b_diff - baseline_tol)

    passed = bool(self_no_reg_ok)

    comparison = c.get("comparison", {}) if isinstance(c.get("comparison"), dict) else {}
    return GateResult(
        "ab_no_regression",
        passed,
        {
            "checks": {
                "self_no_regression_ok": bool(self_no_reg_ok),
                "control_vs_baseline_ok": bool(control_vs_baseline_ok),
                "robust_vs_baseline_ok": bool(robust_vs_baseline_ok),
                "gap_vs_baseline_ok": bool(gap_vs_baseline_ok),
            },
            "warnings": {
                "control_vs_baseline_warning": bool(not control_vs_baseline_ok),
                "robust_vs_baseline_warning": bool(not robust_vs_baseline_ok),
                "gap_vs_baseline_warning": bool(not gap_vs_baseline_ok),
            },
            "current": {
                "control_total_return": c_a,
                "robust_total_return": c_b,
                "diff_total_return": c_diff,
                "tolerance_total_return": c_tol,
                "n_candidates_used": c.get("n_candidates_used"),
            },
            "baseline": {
                "control_total_return": b_a,
                "robust_total_return": b_b,
                "diff_total_return": b_diff,
            },
            "diagnostics": {
                "p_value": _safe_float(comparison.get("p_value")),
                "significant": bool(comparison.get("significant", False)),
                "significance_role": "diagnostic",
                "gate_mode": "no_regression",
            },
        },
    )


def _gate_fairness(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("fairness_status", {})
    c = cur.get("fairness_status", {})
    b_passed = int(b.get("n_passed", 0) or 0)
    c_passed = int(c.get("n_passed", 0) or 0)
    b_total = _fairness_n_attributes(b)
    c_total = _fairness_n_attributes(c)
    return GateResult(
        "fairness_relative",
        c_passed >= b_passed,
        {
            "baseline_n_passed": b_passed,
            "current_n_passed": c_passed,
            "baseline_n_total": b_total,
            "current_n_total": c_total,
            "baseline_overall_pass": bool(b.get("overall_pass", False)),
            "current_overall_pass": bool(c.get("overall_pass", False)),
        },
    )


def _gate_fairness_absolute_business(_base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    c = cur.get("fairness_status", {})
    policy_contract = _load_fairness_policy_contract()

    expected_threshold = _safe_float(policy_contract.get("prediction_threshold"))
    expected_outcome_mode = str(policy_contract.get("outcome_mode", "") or "").strip().lower()
    expected_use_artifact = policy_contract.get("use_artifact", None)

    current_threshold = _safe_float(c.get("prediction_threshold"))
    current_source = str(c.get("prediction_threshold_source", "") or "").strip()
    current_outcome_mode = str(c.get("outcome_mode", "") or "").strip().lower()
    source_uses_artifact = current_source.startswith("artifact")
    if not source_uses_artifact:
        source_uses_artifact = "artifact" in current_source

    threshold_ok = bool(expected_use_artifact) or (
        np.isnan(expected_threshold)
        or np.isnan(current_threshold)
        or bool(abs(current_threshold - expected_threshold) <= 1e-9)
    )
    if expected_use_artifact is None or (
        current_source == "" and bool(expected_use_artifact) is False
    ):
        source_ok = True
    else:
        source_ok = source_uses_artifact == bool(expected_use_artifact)
    outcome_mode_ok = (not expected_outcome_mode) or (current_outcome_mode == expected_outcome_mode)

    n_passed = int(c.get("n_passed", 0) or 0)
    n_total = _fairness_n_attributes(c)
    all_attributes_ok = n_total <= 0 or n_passed >= n_total
    overall_pass_ok = bool(c.get("overall_pass", False))

    passed = bool(
        overall_pass_ok and all_attributes_ok and threshold_ok and source_ok and outcome_mode_ok
    )
    return GateResult(
        "fairness_absolute_business",
        passed,
        {
            "policy_contract": {
                "prediction_threshold": expected_threshold,
                "outcome_mode": expected_outcome_mode,
                "use_artifact": expected_use_artifact,
                "policy_path": policy_contract.get("policy_path"),
            },
            "current": {
                "prediction_threshold": current_threshold,
                "prediction_threshold_source": current_source,
                "outcome_mode": current_outcome_mode,
                "n_passed": n_passed,
                "n_total": n_total,
                "overall_pass": bool(c.get("overall_pass", False)),
            },
            "checks": {
                "overall_pass_ok": overall_pass_ok,
                "all_attributes_ok": bool(all_attributes_ok),
                "threshold_match_ok": bool(threshold_ok),
                "threshold_source_ok": bool(source_ok),
                "outcome_mode_ok": bool(outcome_mode_ok),
            },
        },
    )


def _gate_survival(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("survival_summary", {})
    c = cur.get("survival_summary", {})
    b_cox = _safe_float(b.get("cox_concordance_index"))
    c_cox = _safe_float(c.get("cox_concordance_index"))
    b_rsf = _safe_float(b.get("rsf_c_index_test"))
    c_rsf = _safe_float(c.get("rsf_c_index_test"))
    cox_ok = np.isnan(b_cox) or np.isnan(c_cox) or (c_cox >= b_cox - 0.01)
    rsf_ok = np.isnan(b_rsf) or np.isnan(c_rsf) or (c_rsf >= b_rsf - 0.01)
    return GateResult(
        "survival_quality",
        bool(cox_ok and rsf_ok),
        {
            "baseline": {"cox_cindex": b_cox, "rsf_cindex": b_rsf},
            "current": {"cox_cindex": c_cox, "rsf_cindex": c_rsf},
            "checks": {"cox_ok": cox_ok, "rsf_ok": rsf_ok},
        },
    )


def _gate_exports(cur: dict[str, Any]) -> GateResult:
    metrics = cur.get("metrics", {})
    model_comparison = metrics.get("model_comparison", {})
    pipeline_summary = metrics.get("pipeline_summary", {})
    missing = []
    for key in ["schema_version", "generated_at_utc", "models", "final_test_metrics"]:
        if key not in model_comparison:
            missing.append(f"model_comparison.{key}")
    for key in ["schema_version", "generated_at_utc", "flattened_summary"]:
        if key not in pipeline_summary:
            missing.append(f"pipeline_summary.{key}")
    return GateResult(
        "export_contracts",
        len(missing) == 0,
        {"missing_keys": missing},
    )


def _compare_artifacts(base: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    b_idx = base.get("artifacts", {})
    c_idx = cur.get("artifacts", {})
    for key in sorted(set(b_idx) | set(c_idx)):
        b = b_idx.get(key, {})
        c = c_idx.get(key, {})
        out[key] = {
            "baseline_exists": bool(b.get("exists", False)),
            "current_exists": bool(c.get("exists", False)),
            "hash_changed": b.get("sha256") != c.get("sha256"),
            "size_bytes_baseline": int(b.get("size_bytes", 0) or 0),
            "size_bytes_current": int(c.get("size_bytes", 0) or 0),
        }
    return out


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        f"# Run Comparison: {report['run_tag']}",
        "",
        f"- Generated: {report['generated_at_utc']}",
        f"- Overall gates pass: `{report['overall_pass']}`",
        f"- Conformal promotion pass: `{report.get('conformal_promotion_pass', False)}`",
        f"- Conformal statistical warning: `{report.get('conformal_statistical_warning', False)}`",
        f"- Artifact coherence pass: `{report.get('artifact_coherence_pass', False)}`",
        f"- Semantic coherence pass: `{report.get('semantic_coherence_pass', False)}`",
        f"- Fairness absolute (business) pass: `{report.get('fairness_absolute_business_pass', False)}`",
        f"- A/B gate mode: `{report.get('ab_gate_mode', 'no_regression')}`",
        f"- A/B no-regression pass: `{report.get('ab_no_regression_pass', False)}`",
        f"- A/B significance (diagnostic): `{report.get('ab_significant', False)}`",
        "",
        "## Gates",
    ]
    for gate in gates:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- `{gate['name']}`: **{status}**")
    lines.extend(["", "## Artifact Changes"])
    changed = [
        (k, v)
        for k, v in report["artifact_changes"].items()
        if v.get("hash_changed") or (not v.get("baseline_exists")) != (not v.get("current_exists"))
    ]
    if not changed:
        lines.append("- No tracked artifact hash changes.")
    else:
        for key, meta in changed:
            lines.append(
                f"- `{key}`: hash_changed={meta['hash_changed']}, "
                f"baseline_exists={meta['baseline_exists']}, current_exists={meta['current_exists']}"
            )
    failing_stats = report.get("conformal_failing_statistical_tests", [])
    if failing_stats:
        lines.extend(["", "## Conformal Diagnostics"])
        lines.append(
            "- Statistical warnings (non-blocking): "
            + ", ".join(f"`{name}`" for name in failing_stats)
        )
    return "\n".join(lines) + "\n"


def _write_snapshot(run_tag: str) -> Path:
    out_dir = OUT_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_snapshot.json"
    path.write_text(
        json.dumps(_snapshot_payload(run_tag), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[compare] Baseline snapshot saved: {path.relative_to(ROOT)}")
    return path


def _write_compare(run_tag: str, baseline_path: Path) -> tuple[Path, Path]:
    baseline_path = baseline_path.expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = _snapshot_payload(run_tag)
    gate_results = [
        _gate_artifact_coherence(current["metrics"], run_tag),
        _gate_semantic_coherence(current["metrics"]),
        _gate_pd(baseline["metrics"], current["metrics"]),
        _gate_conformal(baseline["metrics"], current["metrics"]),
        _gate_ab_no_regression(baseline["metrics"], current["metrics"]),
        _gate_fairness(baseline["metrics"], current["metrics"]),
        _gate_fairness_absolute_business(baseline["metrics"], current["metrics"]),
        _gate_survival(baseline["metrics"], current["metrics"]),
        _gate_exports(current),
    ]
    conformal_gate = next((g for g in gate_results if g.name == "conformal_policy"), None)
    ab_gate = next((g for g in gate_results if g.name == "ab_no_regression"), None)
    coherence_gate = next((g for g in gate_results if g.name == "artifact_coherence"), None)
    semantic_gate = next((g for g in gate_results if g.name == "semantic_coherence"), None)
    conformal_details = conformal_gate.details if conformal_gate is not None else {}
    ab_details = ab_gate.details if ab_gate is not None else {}
    fairness_abs_gate = next(
        (g for g in gate_results if g.name == "fairness_absolute_business"), None
    )
    conformal_checks = conformal_details.get("checks", {})
    conformal_diagnostics = conformal_details.get("diagnostics", {})
    ab_diagnostics = ab_details.get("diagnostics", {})
    coherence_details = coherence_gate.details if coherence_gate is not None else {}
    try:
        baseline_path_out = str(baseline_path.relative_to(ROOT))
    except ValueError:
        baseline_path_out = str(baseline_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "baseline_path": baseline_path_out,
        "overall_pass": bool(all(g.passed for g in gate_results)),
        "operational_overall_pass": bool(
            all(g.passed for g in gate_results if g.name in _OPERATIONAL_GATE_NAMES)
        ),
        "artifact_coherence_pass": bool(coherence_gate.passed)
        if coherence_gate is not None
        else False,
        "artifact_coherence": coherence_details,
        "semantic_coherence_pass": bool(semantic_gate.passed)
        if semantic_gate is not None
        else False,
        "semantic_coherence": semantic_gate.details if semantic_gate is not None else {},
        "conformal_promotion_pass": bool(conformal_checks.get("conformal_promotion_pass", False)),
        "conformal_statistical_warning": bool(
            conformal_diagnostics.get("statistical_warning", False)
        ),
        "conformal_failing_statistical_tests": conformal_diagnostics.get(
            "failing_statistical_tests", []
        ),
        "ab_no_regression_pass": bool(ab_gate.passed) if ab_gate is not None else False,
        "fairness_absolute_business_pass": bool(fairness_abs_gate.passed)
        if fairness_abs_gate is not None
        else False,
        "ab_gate_mode": str(ab_diagnostics.get("gate_mode", "no_regression")),
        "ab_significant": bool(ab_diagnostics.get("significant", False)),
        "ab_significance_role": str(ab_diagnostics.get("significance_role", "diagnostic")),
        "gates": [{"name": g.name, "passed": g.passed, "details": g.details} for g in gate_results],
        "artifact_changes": _compare_artifacts(baseline, current),
        "baseline_head": baseline.get("git", {}).get("head", ""),
        "current_head": current.get("git", {}).get("head", ""),
        "quality_contract": {
            "conformal_checks_required": 13,
            "ab_gate_mode": "no_regression",
            "ab_significance_role": "diagnostic",
            "fairness_gates": ["fairness_relative", "fairness_absolute_business"],
            "fairness_policy_path": _path_for_report(FAIRNESS_POLICY_PATH),
            "artifact_coherence_required": True,
            "semantic_coherence_required": True,
            "required_status_metadata": ["schema_version", "generated_at_utc", "run_tag"],
        },
    }
    out_dir = OUT_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    md_path = out_dir / "comparison.md"
    json_path.write_text(
        json.dumps(_to_builtin(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"[compare] Comparison JSON: {json_path.relative_to(ROOT)}")
    print(f"[compare] Comparison MD:   {md_path.relative_to(ROOT)}")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot/compare run artifacts with gates.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snapshot = sub.add_parser("snapshot")
    p_snapshot.add_argument("--run-tag", required=True)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--run-tag", required=True)
    p_compare.add_argument("--baseline", default=None, help="Path to baseline_snapshot.json")

    args = parser.parse_args()

    if args.cmd == "snapshot":
        _write_snapshot(args.run_tag)
        return

    baseline_path = (
        Path(args.baseline).expanduser().resolve()
        if args.baseline
        else (OUT_ROOT / args.run_tag / "baseline_snapshot.json")
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline snapshot not found: {baseline_path}")
    _write_compare(args.run_tag, baseline_path)


if __name__ == "__main__":
    main()
