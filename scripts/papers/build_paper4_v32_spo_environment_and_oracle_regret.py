"""Build Paper 4 v32 SPO environment and oracle-regret artifacts."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import _safe_read_csv
from scripts.papers.build_paper4_v6_priority_resolution import (
    TABLE_DIR,
    _write_csv,
    _write_json,
    _write_note,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-15.32"


def _import_probe(package: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package)
    if spec is None:
        return {
            "package": package,
            "available_v32": False,
            "version_v32": "",
            "import_error_v32": "ModuleNotFoundError",
        }
    try:
        mod = importlib.import_module(package)
        return {
            "package": package,
            "available_v32": True,
            "version_v32": str(getattr(mod, "__version__", "installed")),
            "import_error_v32": "",
        }
    except Exception as exc:
        return {
            "package": package,
            "available_v32": False,
            "version_v32": "",
            "import_error_v32": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }


def _dependency_blockers() -> pd.DataFrame:
    packages = [
        "numpy",
        "cvxpy",
        "cvxpylayers",
        "torch",
        "pyomo",
        "highspy",
        "catboost",
        "sklearn",
        "pandas",
        "scipy",
    ]
    rows = [_import_probe(pkg) for pkg in packages]
    df = pd.DataFrame(rows)
    df["formal_differentiable_spo_claim_allowed"] = False
    df["decision_v32"] = np.select(
        [
            df["package"].isin(["cvxpy", "cvxpylayers", "torch"]) & ~df["available_v32"],
            df["package"].isin(["pyomo", "highspy", "catboost", "sklearn"]) & df["available_v32"],
        ],
        ["dependency_blocked_for_differentiable_spo", "usable_for_oracle_or_surrogate_route"],
        default="environment_context_only",
    )
    df["future_install_path"] = np.where(
        df["package"].isin(["cvxpy", "cvxpylayers", "torch"]),
        "create isolated env with NumPy<2 or NumPy-compatible cvxpy/cvxpylayers/torch pins; do not mutate main .venv until tested",
        "no action for v32",
    )
    return df


def _environment_spec() -> pd.DataFrame:
    py = sys.version.split()[0]
    rows = [
        {
            "env_name": "paper4_spo_isolated_candidate",
            "python": py,
            "dependency": "numpy",
            "suggested_pin": "numpy<2.0 if using old cvxpy wheels; otherwise test NumPy-2 compatible cvxpy",
            "reason": "current cvxpy ABI error references NumPy 1.x compiled extension",
        },
        {
            "env_name": "paper4_spo_isolated_candidate",
            "python": py,
            "dependency": "cvxpy",
            "suggested_pin": "latest compatible with chosen NumPy plus solver backend",
            "reason": "needed only for differentiable/SPO+ prototype, not for oracle-regret path",
        },
        {
            "env_name": "paper4_spo_isolated_candidate",
            "python": py,
            "dependency": "cvxpylayers torch",
            "suggested_pin": "install only in isolated env after cvxpy import is clean",
            "reason": "formal differentiable SPO+ claim remains false until validated",
        },
    ]
    return pd.DataFrame(rows)


def _oracle_regret_v3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regret = _safe_read_csv(TABLE_DIR / "paper4_v28_spo_temporal_oracle_regret.csv")
    report = _safe_read_csv(TABLE_DIR / "paper4_v28_spo_training_report.csv")
    candidates = _safe_read_csv(TABLE_DIR / "paper4_v30_candidate_registry.csv")
    if regret.empty:
        regret = _safe_read_csv(TABLE_DIR / "paper4_v24_spo_temporal_regret.csv")
    if report.empty:
        report = _safe_read_csv(TABLE_DIR / "paper4_v24_spo_training_report.csv")

    regret_v3 = regret.copy()
    if regret_v3.empty:
        regret_v3 = pd.DataFrame(
            [
                {
                    "split": "not_available",
                    "policy_id": "spo_oracle_regret_v3",
                    "oracle_regret": np.nan,
                    "claim_boundary_v32": "prior SPO artifacts missing",
                }
            ]
        )
    for col in ["oracle_regret", "mean_regret", "regret", "temporal_regret"]:
        if col in regret_v3.columns:
            regret_col = col
            break
    else:
        regret_col = ""
    if regret_col:
        regret_v3["regret_rank_v32"] = pd.to_numeric(regret_v3[regret_col], errors="coerce").rank(
            method="average", ascending=True
        )
    else:
        regret_v3["regret_rank_v32"] = np.nan
    regret_v3["version_v32"] = "spo_oracle_regret_v3_without_differentiable_layers"
    regret_v3["formal_spo_plus_claim_allowed"] = False
    regret_v3["claim_boundary_v32"] = (
        "decision-oracle/surrogate regret only; not formal differentiable SPO+"
    )

    candidate_rows = []
    if not candidates.empty:
        for _, row in candidates.head(12).iterrows():
            candidate_rows.append(
                {
                    "policy_id": row["policy_id"],
                    "source": "paper4_v30_candidate_registry",
                    "full_candidate_score_prior": row.get("full_candidate_score_v30", np.nan),
                    "paired_robustness_gate": row.get("paired_robustness_gate_v30", False),
                    "decision_scope_v32": "compare_against_oracle_regret_only",
                    "spo_candidate_status_v32": "serious_comparator"
                    if "spo" in str(row["policy_id"]).lower()
                    else "non_spo_reference",
                }
            )
    spo_candidates = pd.DataFrame(candidate_rows)

    report_v3 = report.copy()
    if report_v3.empty:
        report_v3 = pd.DataFrame(
            [
                {
                    "model": "oracle_regret_surrogate_v3",
                    "training_status": "reused_prior_artifacts_missing_or_not_material",
                    "formal_spo_plus_claim_allowed": False,
                }
            ]
        )
    report_v3["version_v32"] = "spo_oracle_regret_v3"
    report_v3["formal_spo_plus_claim_allowed"] = False
    report_v3["temporal_split_claim_v32"] = (
        "validated only to the extent prior v28/v24 temporal split artifacts support it"
    )
    return report_v3, regret_v3, spo_candidates


def build_v32() -> dict[str, Any]:
    start = time.time()
    deps = _dependency_blockers()
    env_spec = _environment_spec()
    report, regret, candidates = _oracle_regret_v3()

    _write_csv("paper4_v32_spo_dependency_blockers.csv", deps)
    _write_csv("paper4_v32_spo_isolated_env_plan.csv", env_spec)
    _write_csv("paper4_v32_spo_training_report_v3.csv", report)
    _write_csv("paper4_v32_spo_temporal_oracle_regret_v3.csv", regret)
    _write_csv("paper4_v32_spo_candidate_comparison_v3.csv", candidates)

    cvxpy_row = deps.loc[deps["package"].eq("cvxpy")]
    cvxpy_available = bool(cvxpy_row["available_v32"].iloc[0]) if not cvxpy_row.empty else False
    torch_available = (
        bool(deps.loc[deps["package"].eq("torch"), "available_v32"].iloc[0])
        if (deps["package"].eq("torch")).any()
        else False
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v32_spo_environment_and_oracle_regret",
        "cvxpy_import_clean_v32": cvxpy_available,
        "torch_available_v32": torch_available,
        "formal_differentiable_spo_claim_allowed": False,
        "oracle_regret_rows_v32": int(len(regret)),
        "spo_candidate_rows_v32": int(len(candidates)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "SPO oracle-regret route only; differentiable SPO+ dependency blocked unless isolated env passes",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v32_status.json", status)
    _write_note(
        "paper4_v32_spo_environment_and_oracle_regret.md",
        "\n".join(
            [
                "# Paper 4 v32 SPO Environment and Oracle-Regret",
                "",
                f"- cvxpy import clean: `{status['cvxpy_import_clean_v32']}`.",
                f"- torch available: `{status['torch_available_v32']}`.",
                "- Formal differentiable SPO+ remains blocked.",
                "- Oracle-regret artifacts remain usable as lab comparators.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_v32()


if __name__ == "__main__":
    main()
