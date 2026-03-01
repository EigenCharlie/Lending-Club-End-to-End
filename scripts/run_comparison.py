"""Snapshot and compare baseline/current artifacts with promotion gates.

Usage:
    uv run python scripts/run_comparison.py snapshot --run-tag 2026-02-26-night
    uv run python scripts/run_comparison.py compare --run-tag 2026-02-26-night
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUT_ROOT = REPORTS / "run_comparisons"
SCHEMA_VERSION = "2026-02-27.1"


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
    metrics_summary = _read_json(REPORTS / "dvc" / "metrics_summary.json")
    model_comparison = _read_json(DATA / "model_comparison.json")
    pipeline_summary = _read_json(DATA / "pipeline_summary.json")
    conformal = _read_json(MODELS / "conformal_policy_status.json")
    fairness = _read_json(MODELS / "fairness_audit_status.json")
    ab_status = _read_json(MODELS / "ab_simulation_status.json")
    cate_status = _read_json(MODELS / "cate_portfolio_status.json")

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
        "model_comparison": model_comparison,
        "pipeline_summary": pipeline_summary,
        "conformal_status": conformal,
        "fairness_status": fairness,
        "survival_summary": survival_summary,
        "ifrs9_summary": ifrs9,
        "portfolio_robustness_summary": robustness_summary,
        "ab_simulation_status": ab_status,
        "cate_portfolio_status": cate_status,
    }


def _artifact_index() -> dict[str, dict[str, Any]]:
    targets = {
        "reports/dvc/metrics_summary.json": REPORTS / "dvc" / "metrics_summary.json",
        "data/processed/model_comparison.json": DATA / "model_comparison.json",
        "data/processed/pipeline_summary.json": DATA / "pipeline_summary.json",
        "models/conformal_policy_status.json": MODELS / "conformal_policy_status.json",
        "models/fairness_audit_status.json": MODELS / "fairness_audit_status.json",
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
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
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


def _gate_pd(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("dvc_metrics", {})
    c = cur.get("dvc_metrics", {})
    b_auc = _safe_float(b.get("pd.auc"))
    c_auc = _safe_float(c.get("pd.auc"))
    b_ece = _safe_float(b.get("pd.ece"))
    c_ece = _safe_float(c.get("pd.ece"))
    b_d2 = _safe_float(b.get("pd.d2_brier"))
    c_d2 = _safe_float(c.get("pd.d2_brier"))
    auc_ok = np.isnan(b_auc) or np.isnan(c_auc) or (c_auc >= b_auc - 0.005)
    ece_ok = np.isnan(b_ece) or np.isnan(c_ece) or (c_ece <= b_ece * 1.2 + 1e-12)
    d2_ok = np.isnan(b_d2) or np.isnan(c_d2) or (c_d2 >= b_d2 - 1e-9)
    return GateResult(
        "pd_quality",
        bool(auc_ok and ece_ok and d2_ok),
        {
            "baseline": {"auc": b_auc, "ece": b_ece, "d2_brier": b_d2},
            "current": {"auc": c_auc, "ece": c_ece, "d2_brier": c_d2},
            "checks": {"auc_ok": auc_ok, "ece_ok": ece_ok, "d2_brier_ok": d2_ok},
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


def _gate_fairness(base: dict[str, Any], cur: dict[str, Any]) -> GateResult:
    b = base.get("fairness_status", {})
    c = cur.get("fairness_status", {})
    b_passed = int(b.get("n_passed", 0) or 0)
    c_passed = int(c.get("n_passed", 0) or 0)
    return GateResult(
        "fairness_relative",
        c_passed >= b_passed,
        {
            "baseline_n_passed": b_passed,
            "current_n_passed": c_passed,
            "baseline_overall_pass": bool(b.get("overall_pass", False)),
            "current_overall_pass": bool(c.get("overall_pass", False)),
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
        _gate_pd(baseline["metrics"], current["metrics"]),
        _gate_conformal(baseline["metrics"], current["metrics"]),
        _gate_fairness(baseline["metrics"], current["metrics"]),
        _gate_survival(baseline["metrics"], current["metrics"]),
        _gate_exports(current),
    ]
    conformal_gate = next((g for g in gate_results if g.name == "conformal_policy"), None)
    conformal_details = conformal_gate.details if conformal_gate is not None else {}
    conformal_checks = conformal_details.get("checks", {})
    conformal_diagnostics = conformal_details.get("diagnostics", {})
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
        "conformal_promotion_pass": bool(conformal_checks.get("conformal_promotion_pass", False)),
        "conformal_statistical_warning": bool(
            conformal_diagnostics.get("statistical_warning", False)
        ),
        "conformal_failing_statistical_tests": conformal_diagnostics.get(
            "failing_statistical_tests", []
        ),
        "gates": [{"name": g.name, "passed": g.passed, "details": g.details} for g in gate_results],
        "artifact_changes": _compare_artifacts(baseline, current),
        "baseline_head": baseline.get("git", {}).get("head", ""),
        "current_head": current.get("git", {}).get("head", ""),
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
