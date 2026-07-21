#!/usr/bin/env python3
"""Summarize CRPTO/IJDS champion-reopen outputs into one live status table."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = "paper1_crpto_reopen_ijds_2026_05_25"
CHAMPION = {
    "return": 170464.5429284627,
    "V": 0.03645,
    "Gamma_CP": 0.18591,
    "violation": 0.0,
    "coverage": 0.9433,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}"}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _decision(metrics: dict[str, Any]) -> str:
    ret = _safe_float(metrics.get("realized_total_return"))
    v = _safe_float(metrics.get("alpha01_weighted_miscoverage_V"))
    gamma = _safe_float(metrics.get("alpha01_gamma_cp"))
    violation = _safe_float(metrics.get("alpha01_violation"))
    coverage = _safe_float(metrics.get("alpha01_empirical_coverage_funded"))
    exact = bool(metrics.get("alpha01_exact_pass"))
    if not exact:
        return "no_alpha01_exact_pass"
    if violation is not None and violation > 1e-12:
        return "alpha01_violation"
    if ret is None or v is None or gamma is None:
        return "incomplete_metrics"
    coverage_ok = coverage is None or coverage >= CHAMPION["coverage"] - 0.01
    if (
        ret >= CHAMPION["return"]
        and v <= CHAMPION["V"]
        and gamma <= CHAMPION["Gamma_CP"]
        and coverage_ok
    ):
        return "promote_if_nested_full_confirmed"
    if ret >= CHAMPION["return"]:
        return "return_challenger_bound_worse"
    if v <= CHAMPION["V"] or gamma <= CHAMPION["Gamma_CP"]:
        return "bound_challenger_return_worse"
    return "append_or_park"


def _metric_row(
    *,
    run_label: str,
    artifact_kind: str,
    tier: str,
    metrics: dict[str, Any],
    path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ret = _safe_float(metrics.get("realized_total_return"))
    v = _safe_float(metrics.get("alpha01_weighted_miscoverage_V"))
    gamma = _safe_float(metrics.get("alpha01_gamma_cp"))
    coverage = _safe_float(metrics.get("alpha01_empirical_coverage_funded"))
    row = {
        "run_label": run_label,
        "artifact_kind": artifact_kind,
        "tier": tier,
        "decision_read": _decision(metrics),
        "alpha01_exact_pass": metrics.get("alpha01_exact_pass"),
        "realized_total_return": ret,
        "return_delta_vs_champion": None if ret is None else ret - CHAMPION["return"],
        "alpha01_weighted_miscoverage_V": v,
        "V_delta_vs_champion": None if v is None else v - CHAMPION["V"],
        "alpha01_gamma_cp": gamma,
        "Gamma_delta_vs_champion": None if gamma is None else gamma - CHAMPION["Gamma_CP"],
        "alpha01_violation": _safe_float(metrics.get("alpha01_violation")),
        "alpha01_empirical_coverage_funded": coverage,
        "coverage_delta_vs_champion": None if coverage is None else coverage - CHAMPION["coverage"],
        "n_funded": _safe_float(metrics.get("n_funded")),
        "policy_mode": metrics.get("policy_mode"),
        "risk_tolerance": _safe_float(metrics.get("risk_tolerance")),
        "gamma": _safe_float(metrics.get("gamma")),
        "uncertainty_aversion": _safe_float(metrics.get("uncertainty_aversion")),
        "candidate_rank": metrics.get("candidate_rank"),
        "shortlist_bucket": metrics.get("shortlist_bucket"),
        "artifact_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        if path.exists()
        else None,
    }
    if extra:
        row.update(extra)
    return row


def _selection_rows(run_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (ROOT / "models" / "portfolio_bound_aware").glob(
            f"{run_root}*/portfolio_bound_aware_selection.json"
        )
    ):
        payload = _load_json(path)
        metrics = dict(payload.get("selected_metrics", {}) or {})
        policy = dict(payload.get("selected_policy", {}) or {})
        for key, value in policy.items():
            metrics.setdefault(key, value)
        rows.append(
            _metric_row(
                run_label=str(payload.get("run_label") or path.parent.name),
                artifact_kind="portfolio_selection",
                tier="selected",
                metrics=metrics,
                path=path,
                extra={
                    "selection_reason": payload.get("selection_reason"),
                    "solver_agreement_status": payload.get("solver_agreement_status"),
                    "solver_agreement_report_path": payload.get("solver_agreement_report_path"),
                    "exact_tournament_mode": (payload.get("exact_tournament") or {}).get(
                        "mode", "pass1_pass2"
                    )
                    if isinstance(payload.get("exact_tournament"), dict)
                    else "pass1_pass2",
                },
            )
        )
    return rows


def _shortlist_rows(run_root: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(
        (ROOT / "data" / "processed" / "portfolio_bound_aware").glob(
            f"{run_root}*/portfolio_bound_aware_shortlist.parquet"
        )
    ):
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            rows.append(
                {
                    "run_label": path.parent.name,
                    "artifact_kind": "portfolio_shortlist",
                    "tier": "read_error",
                    "decision_read": f"{type(exc).__name__}: {exc}",
                    "artifact_path": str(path.relative_to(ROOT)),
                }
            )
            continue
        if df.empty or "alpha01_exact_pass" not in df.columns:
            rows.append(
                {
                    "run_label": path.parent.name,
                    "artifact_kind": "portfolio_shortlist",
                    "tier": "empty_or_no_alpha01",
                    "decision_read": "incomplete_metrics",
                    "artifact_path": str(path.relative_to(ROOT)),
                }
            )
            continue
        work = df[df["alpha01_exact_pass"].fillna(False)].copy()
        if work.empty:
            rows.append(
                {
                    "run_label": path.parent.name,
                    "artifact_kind": "portfolio_shortlist",
                    "tier": "no_alpha01_pass",
                    "decision_read": "no_alpha01_exact_pass",
                    "artifact_path": str(path.relative_to(ROOT)),
                }
            )
            continue
        tiers = {
            "best_return": work.sort_values(
                ["realized_total_return", "alpha01_weighted_miscoverage_V", "alpha01_gamma_cp"],
                ascending=[False, True, True],
            ).iloc[0],
            "best_V": work.sort_values(
                ["alpha01_weighted_miscoverage_V", "realized_total_return"],
                ascending=[True, False],
            ).iloc[0],
            "best_Gamma": work.sort_values(
                ["alpha01_gamma_cp", "realized_total_return"],
                ascending=[True, False],
            ).iloc[0],
        }
        for tier, series in tiers.items():
            rows.append(
                _metric_row(
                    run_label=path.parent.name,
                    artifact_kind="portfolio_shortlist",
                    tier=tier,
                    metrics=series.to_dict(),
                    path=path,
                )
            )
    return rows


def _conformal_rows(run_root: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(
        (ROOT / "data" / "processed" / "conformal_gap").glob(
            f"{run_root}*/conformal_intervals_mondrian.parquet"
        )
    ):
        try:
            df = pd.read_parquet(path, columns=["y_true", "y_pred", "pd_low_90", "pd_high_90"])
            width = df["pd_high_90"] - df["pd_low_90"]
            coverage = (df["y_true"] <= df["pd_high_90"]).mean()
            gamma_mean = (df["pd_high_90"] - df["y_pred"]).clip(lower=0, upper=1).mean()
            rows.append(
                {
                    "run_label": path.parent.name,
                    "artifact_kind": "conformal_intervals",
                    "tier": "interval_summary",
                    "decision_read": "needs_portfolio_gate",
                    "n_rows": int(len(df)),
                    "coverage90_upper_only": float(coverage),
                    "width90_mean": float(width.mean()),
                    "gamma_mean_proxy": float(gamma_mean),
                    "artifact_path": str(path.relative_to(ROOT)),
                    "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run_label": path.parent.name,
                    "artifact_kind": "conformal_intervals",
                    "tier": "read_error",
                    "decision_read": f"{type(exc).__name__}: {exc}",
                    "artifact_path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def _runtime_rows(run_root: str, since: timedelta | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cutoff = datetime.now(UTC) - since if since is not None else None
    paths = [
        *sorted(
            (ROOT / "models" / "portfolio_bound_aware").glob(
                f"{run_root}*/portfolio_bound_aware_runtime_status.json"
            )
        ),
        *sorted(
            (ROOT / "models" / "conformal_gap").glob(f"{run_root}*/conformal_reopen_status.json")
        ),
        *sorted((ROOT / "reports" / "run_logs" / run_root / "status").glob("*.json")),
    ]
    for path in paths:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if cutoff is not None and mtime < cutoff:
            continue
        payload = _load_json(path)
        rows.append(
            {
                "run_label": path.parent.name,
                "artifact_kind": "runtime_status",
                "tier": path.stem,
                "decision_read": str(payload.get("state") or payload.get("phase") or "observed"),
                "phase": payload.get("phase"),
                "state": payload.get("state"),
                "global_pct_complete": payload.get("global_pct_complete"),
                "frontier_pct_complete": payload.get("frontier_pct_complete"),
                "eta_sec": payload.get("eta_sec"),
                "artifact_path": str(path.relative_to(ROOT)),
                "mtime_utc": mtime.isoformat(),
            }
        )
    return rows


def _write_outputs(
    rows: list[dict[str, Any]], run_dir: Path, *, write_final: bool
) -> tuple[Path, Path, Path | None]:
    try:
        import pandas as pd
    except Exception:
        pd = None

    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "summary_current.json"
    csv_path = run_dir / "summary_current.csv"
    payload = {
        "generated_at_utc": _utc_now(),
        "champion": CHAMPION,
        "n_rows": len(rows),
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    if pd is not None:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    else:
        csv_path.write_text(json.dumps(rows, sort_keys=True, default=str) + "\n", encoding="utf-8")

    final_path: Path | None = None
    if write_final and pd is not None:
        final_path = (
            ROOT
            / "reports"
            / "paper_material"
            / "paper1"
            / "tables"
            / "paper1_crpto_ijds_reopen_final_summary_2026-05-25.csv"
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(final_path, index=False)
    return json_path, csv_path, final_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT", DEFAULT_RUN_ROOT))
    parser.add_argument("--log-dir", default=str(ROOT / "reports" / "run_logs"))
    parser.add_argument("--since-minutes", type=float, default=None)
    parser.add_argument("--write-final", action="store_true")
    args = parser.parse_args(argv)

    since = timedelta(minutes=args.since_minutes) if args.since_minutes is not None else None
    rows: list[dict[str, Any]] = []
    rows.extend(_selection_rows(args.run_root))
    rows.extend(_shortlist_rows(args.run_root))
    rows.extend(_conformal_rows(args.run_root))
    rows.extend(_runtime_rows(args.run_root, since))

    run_dir = Path(args.log_dir) / args.run_root
    json_path, csv_path, final_path = _write_outputs(rows, run_dir, write_final=args.write_final)
    print(json_path)
    print(csv_path)
    if final_path is not None:
        print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
