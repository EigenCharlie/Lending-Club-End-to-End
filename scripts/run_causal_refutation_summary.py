"""Causal refutation summary + OOT tail risk analysis.

Reads existing causal artifacts and produces a structured summary of:
- Refutation test status (placebo, random common cause, data subset)
- OOT CATE distribution with tail risk metrics
- Grade-level CATE heterogeneity

Outputs:
    models/causal_refutation_summary.json
    data/processed/causal_oot_tail_risk.parquet
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SCHEMA_VERSION = "2026-03-17.1"


def _interpret_refutation(test: str, result: str) -> str:
    if "unavailable" in result:
        return (
            "La refutacion no pudo ejecutarse en este snapshot; debe leerse como evidencia "
            "incompleta y no como validacion favorable."
        )
    if "passed" in result.lower():
        return "Passed — causal estimate is robust to this perturbation."
    return f"Check result: {result}"


def _extract_p_value(raw_result: str, explicit_value: object) -> float | None:
    if explicit_value is not None:
        try:
            return float(explicit_value)
        except Exception:
            pass
    match = re.search(
        r"p\s*value\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        str(raw_result),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _oot_tail_risk(cate_oot: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """OOT CATE distribution + tail risk metrics."""
    cate = cate_oot["cate"].dropna()

    # Tail risk: loans with large positive CATE = most benefit from treatment (rate reduction)
    # Loans with large negative CATE = harm from treatment
    p5 = float(np.percentile(cate, 5))
    p25 = float(np.percentile(cate, 25))
    p50 = float(np.percentile(cate, 50))
    p75 = float(np.percentile(cate, 75))
    p95 = float(np.percentile(cate, 95))

    n_positive = int((cate > 0).sum())
    n_negative = int((cate < 0).sum())

    summary = {
        "n_obs": len(cate),
        "ate_oot": round(float(cate.mean()), 6),
        "cate_std": round(float(cate.std()), 6),
        "percentiles": {
            "p5": round(p5, 6),
            "p25": round(p25, 6),
            "p50": round(p50, 6),
            "p75": round(p75, 6),
            "p95": round(p95, 6),
        },
        "n_positive_cate": n_positive,
        "n_negative_cate": n_negative,
        "pct_positive": round(n_positive / len(cate), 3),
        "tail_risk_p5": round(p5, 6),
        "tail_risk_interpretation": (
            f"Bottom 5% CATE = {p5:.4f}: these loans see the largest negative effect "
            f"(treatment increases default risk). Top 5% CATE ≥ {p95:.4f}: "
            f"strongest benefit from treatment. "
            f"{n_positive / len(cate):.1%} of OOT loans have positive CATE."
        ),
    }

    # Grade-level CATE with tail risk
    if "grade" in cate_oot.columns:
        grade_df = (
            cate_oot.groupby("grade")["cate"]
            .agg(
                n="count",
                mean="mean",
                std="std",
                p5=lambda x: np.percentile(x.dropna(), 5),
                p95=lambda x: np.percentile(x.dropna(), 95),
            )
            .reset_index()
        )
        for col in ["mean", "std", "p5", "p95"]:
            grade_df[col] = grade_df[col].round(5)
        return summary, grade_df
    return summary, pd.DataFrame()


def main() -> None:
    logger.info("Causal refutation summary + OOT tail risk")

    causal_path = MODELS_DIR / "causal_effect_status.json"
    if not causal_path.exists():
        logger.error("causal_effect_status.json not found.")
        return

    causal = json.loads(causal_path.read_text(encoding="utf-8"))
    run_tag = causal.get("run_tag", "untracked")

    # Refutation summary
    refs = causal.get("refutation_summary", [])
    refutation_detail = []
    for ref in refs:
        test = ref.get("test", "unknown")
        result = ref.get("result", "")
        refutation_detail.append(
            {
                "test": test,
                "status": "unavailable" if "unavailable" in result else "completed",
                "estimated_effect": ref.get("estimated_effect"),
                "new_effect": ref.get("new_effect"),
                "p_value": _extract_p_value(result, ref.get("p_value")),
                "result_raw": result,
                "interpretation": _interpret_refutation(test, result),
            }
        )
    logger.info(f"Refutation tests found: {len(refutation_detail)}")

    # OOT tail risk
    oot_path_str = causal.get("oot_cate_artifact_path", "")
    oot_summary: dict = {}
    tail_risk_df = pd.DataFrame()

    if oot_path_str:
        oot_path = Path(oot_path_str)
        if not oot_path.is_absolute():
            oot_path = PROJECT_ROOT / oot_path
        if oot_path.exists():
            cate_oot = pd.read_parquet(oot_path)
            logger.info(f"OOT CATE: {len(cate_oot):,} loans")
            oot_summary, tail_risk_df = _oot_tail_risk(cate_oot)
            logger.info(
                f"  ATE OOT: {oot_summary['ate_oot']:.5f} | "
                f"  P5: {oot_summary['percentiles']['p5']:.5f} | "
                f"  P95: {oot_summary['percentiles']['p95']:.5f}"
            )
        else:
            logger.warning(f"OOT CATE not found at {oot_path}")

    if not tail_risk_df.empty:
        out_df = DATA_PROC / "causal_oot_tail_risk.parquet"
        tail_risk_df.to_parquet(out_df, index=False)
        logger.success(f"Saved {out_df}")

    # OOT policy validation key metrics
    oot_val_path = MODELS_DIR / "causal_policy_oot_status.json"
    oot_val: dict = {}
    if oot_val_path.exists():
        oot_val = json.loads(oot_val_path.read_text(encoding="utf-8"))

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "run_tag": run_tag,
        "ate": causal.get("ate"),
        "ate_ci": causal.get("ate_ci"),
        "n_obs": causal.get("n_obs"),
        "identification_strategy": causal.get("identification_strategy"),
        "refutation_tests": refutation_detail,
        "refutation_verdict": (
            "DoWhy refutations are treated as an audit layer over the official DML estimate. "
            "The correct reading is not that refutations 'prove' causal validity, but that they "
            "either reinforce stability under perturbations or remain unavailable/inconclusive. "
            "This is consistent with the lane remaining research-grade by default."
        ),
        "oot_cate_summary": oot_summary,
        "oot_policy_validation": {
            "rule_name": oot_val.get("rule_name"),
            "n_months_evaluated": oot_val.get("n_months_evaluated"),
            "avg_action_rate": oot_val.get("avg_action_rate"),
            "total_net_value": oot_val.get("total_net_value"),
            "p05_monthly_net": oot_val.get("p05_monthly_net"),
            "worst_month": oot_val.get("worst_month"),
            "best_month": oot_val.get("best_month"),
        },
        "insights_only_note": causal.get(
            "insights_only_note",
            "The causal lane is research-grade by design and only escalates if overlap, sensitivity "
            "and policy-value gates all pass.",
        ),
        "tail_risk_artifact": str(DATA_PROC / "causal_oot_tail_risk.parquet")
        if not tail_risk_df.empty
        else None,
    }

    out = MODELS_DIR / "causal_refutation_summary.json"
    out.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    logger.success(f"Saved {out}")


if __name__ == "__main__":
    main()
