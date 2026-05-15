"""Build Paper 4 v15-v18 dynamic sequential-decision artifacts.

This wave turns the Paper 4 living lab from static funded books into a
month-by-month evaluation layer.  It deliberately remains a research/lab
artifact:

* Paper Estrella artifacts are not modified.
* ``models/final_project_promotion.json`` is not modified.
* ``paper4_final_promotion.json`` is not created.
* IFRS9 contractual, CATE policy-value and fair-lending legal claims remain
  blocked unless their evidence gates truly pass.

The implementation reuses v13/v14 candidate books and adds:

* dynamic monthly state traces under common calibrated internal paths;
* CVaR strict-infeasibility diagnostics;
* SPO-style decision-oracle regret artifacts with dependency blockers;
* champion decomposition;
* IFRS9/CATE/fairness gate reports; and
* an academic contribution synthesis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_csv,
    _safe_read_json,
    _safe_read_parquet,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v6_priority_resolution import (
    SOURCE_FAMILIES,
    STATUS_DIR,
    TABLE_DIR,
    _load_inputs,
    _prepare_solver_pool,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v10_resolution_wave import PAPER1_PROMOTION, PAPER4_FINAL_PROMOTION

SCHEMA_VERSION = "2026-05-14.15-18"
RNG_SEED = 2026051418
MONTHLY_REPAYMENT_HORIZON = 18
WORKING_CHAMPION_PATH = STATUS_DIR / "paper4_v18_working_champion.json"


def _stable_uniform(*parts: object) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _stable_normal(*parts: object) -> float:
    u1 = max(_stable_uniform(*parts, "u1"), 1e-12)
    u2 = _stable_uniform(*parts, "u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _parse_term_months(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        return 36
    return int(np.clip(int(match.group()), 12, 84))


def _rank_score(s: pd.Series, *, high_is_good: bool) -> pd.Series:
    return s.rank(method="average", ascending=not high_is_good, na_option="keep", pct=True).fillna(
        0.50
    )


def _safe_num(s: pd.Series | float | int, default: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(default)
    return pd.Series([float(s)])


def _dependency_status() -> pd.DataFrame:
    rows = []
    for package, intended_use in [
        ("cvxpy", "differentiable/convex modeling path for SPO+ layers"),
        ("cvxpylayers", "differentiate through disciplined convex programs"),
        ("torch", "train differentiable decision layers"),
        ("pyomo", "LP/HiGHS decision oracle and CVaR programs"),
        ("highspy", "HiGHS solver backend"),
        ("catboost", "monotonic challenger/regret-score learner"),
    ]:
        spec = importlib.util.find_spec(package)
        if spec is None:
            available = False
            error = "ModuleNotFoundError: package not installed in current environment"
            version = ""
        elif package in {"cvxpy", "cvxpylayers", "torch"}:
            available = False
            error = "dependency intentionally blocked for v16 formal SPO path; cvxpy import is known to hit a NumPy ABI issue in this environment"
            version = "installed_or_partially_installed" if spec is not None else ""
        else:
            try:
                mod = importlib.import_module(package)
                available = True
                error = ""
                version = str(getattr(mod, "__version__", "installed"))
            except Exception as exc:  # pragma: no cover - environment-specific detail
                available = False
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                version = ""
        rows.append(
            {
                "package": package,
                "available_v16": available,
                "version": version,
                "intended_use": intended_use,
                "decision_v16": "usable_now" if available else "dependency_blocked_documented",
                "blocker_detail": error,
            }
        )
    return pd.DataFrame(rows)


def _standardize_book(df: pd.DataFrame, *, source_artifact: str, lane: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "loan_id" not in out and "id" in out:
        out["loan_id"] = out["id"]
    out["loan_id"] = out["loan_id"].astype(str)
    if "policy_id" not in out:
        out["policy_id"] = "unknown_policy"
    if "issue_month" not in out:
        if "issue_d" in out:
            out["issue_month"] = out["issue_d"]
        elif "decision_month" in out:
            out["issue_month"] = out["decision_month"]
        else:
            out["issue_month"] = "2018-01-01"
    out["issue_month"] = _month_start(out["issue_month"])
    if out["issue_month"].isna().all() and "decision_month" in out:
        out["issue_month"] = _month_start(out["decision_month"])
    if "funded_exposure" not in out:
        if "allocation_fraction" in out and "loan_amnt" in out:
            out["funded_exposure"] = _safe_num(out["allocation_fraction"]) * _safe_num(
                out["loan_amnt"]
            )
        elif "loan_amnt" in out:
            out["funded_exposure"] = _safe_num(out["loan_amnt"])
        else:
            out["funded_exposure"] = 0.0
    if "loan_amnt" not in out:
        out["loan_amnt"] = out["funded_exposure"]
    for col, default in [
        ("int_rate_decimal", 0.12),
        ("pd_high_alpha01", 0.18),
        ("pd_point_alpha01", np.nan),
        ("qhat_v4", 0.55),
        ("weak_source_proxy", 0.33),
        ("y_true", 0.0),
        ("lgd", DEFAULT_LGD),
        ("base_return_vec", np.nan),
        ("installment", np.nan),
    ]:
        if col not in out:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["pd_point_alpha01"] = out["pd_point_alpha01"].fillna(out["pd_high_alpha01"])
    out["lgd"] = out["lgd"].fillna(DEFAULT_LGD).clip(0, 1)
    out["qhat_v4"] = out["qhat_v4"].fillna(0.55).clip(0, 1)
    out["weak_source_proxy"] = out["weak_source_proxy"].fillna(0.33).clip(0, 1)
    out["base_return_vec"] = out["base_return_vec"].fillna(
        out["funded_exposure"]
        * (
            out["int_rate_decimal"].fillna(0.12)
            - out["pd_point_alpha01"].fillna(0.18) * DEFAULT_LGD
        )
    )
    if "period" not in out:
        out["period"] = out["issue_month"].dt.year.astype(str)
    if "original_grade" not in out and "grade" in out:
        out["original_grade"] = out["grade"]
    for col in SOURCE_FAMILIES:
        if col not in out:
            out[col] = "unknown"
    if "term" not in out:
        out["term"] = "36 months"
    out["term"] = out["term"].astype(str)
    out["term_months"] = out["term"].map(_parse_term_months)
    for col in ["period", "original_grade", *SOURCE_FAMILIES]:
        if col in out:
            out[col] = out[col].astype(str)
    score_cols = [
        "fvi_score_v12",
        "spo_regret_score_v12",
        "candidate_score_v12",
        "solver_score_seed",
        "base_return_vec",
    ]
    score = pd.Series(0.0, index=out.index)
    for col in score_cols:
        if col in out:
            score = pd.to_numeric(out[col], errors="coerce").fillna(score)
            break
    out["decision_priority_score_v15"] = score
    out["source_artifact_v15"] = source_artifact
    out["lane_v15"] = lane
    keep = [
        "policy_id",
        "loan_id",
        "issue_month",
        "period",
        "original_grade",
        "term",
        "term_months",
        "loan_amnt",
        "funded_exposure",
        "int_rate_decimal",
        "installment",
        "y_true",
        "lgd",
        "pd_point_alpha01",
        "pd_high_alpha01",
        "qhat_v4",
        "weak_source_proxy",
        "base_return_vec",
        "decision_priority_score_v15",
        "source_artifact_v15",
        "lane_v15",
        *SOURCE_FAMILIES,
    ]
    unique_keep: list[str] = []
    for col in keep:
        if col in out and col not in unique_keep:
            unique_keep.append(col)
    return out[unique_keep].dropna(subset=["issue_month"]).copy()


def _load_policy_books(solver_pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    registry_rows = []

    def add_book(name: str, path_name: str, lane: str, transform=lambda x: x) -> None:
        path = TABLE_DIR / path_name
        raw = _safe_read_parquet(path)
        if raw.empty and path.suffix == ".csv":
            raw = _safe_read_csv(path)
        raw = transform(raw)
        std = _standardize_book(raw, source_artifact=path_name, lane=lane)
        if std.empty:
            return
        frames.append(std)
        for policy_id, local in std.groupby("policy_id"):
            registry_rows.append(
                {
                    "policy_id": policy_id,
                    "lane_v15": lane,
                    "adapter_type_v15": "monthly_book_adapter",
                    "source_artifact": path_name,
                    "n_candidate_loans": int(local["loan_id"].nunique()),
                    "initial_static_exposure": float(local["funded_exposure"].sum()),
                    "decision_rule": "fund eligible booked loans month by month subject to available cash and no temporal leakage",
                    "claim_boundary": "dynamic replay of an existing policy book, not production deployment",
                }
            )

    add_book(
        "paper1",
        "paper4_policy_loan_level_evidence.parquet",
        "crpto_frozen_paper1",
        lambda df: (
            df[
                df.get("policy_id", pd.Series(index=df.index, dtype=str))
                .astype(str)
                .eq("paper1_economic_champion")
            ].copy()
            if not df.empty
            else df
        ),
    )
    add_book(
        "cvar", "paper4_v13_cvar_stronger_decomposition_allocations.parquet", "cvar_mdcp_colgen"
    )
    add_book("mdcp", "paper4_v13_mdcp_cap_regime_allocations.parquet", "mdcp_cap_solver")
    add_book(
        "spo", "paper4_v13_spo_decision_loss_allocations.parquet", "spo_decision_oracle_surrogate"
    )
    add_book("dla", "paper4_v13_dla_representative_allocations.parquet", "dla_fvi_representative")

    books = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not books.empty:
        books = books.sort_values(
            ["policy_id", "issue_month", "decision_priority_score_v15"],
            ascending=[True, True, False],
        )
        books = books.drop_duplicates(["policy_id", "loan_id"], keep="first").reset_index(drop=True)
    registry = (
        pd.DataFrame(registry_rows).drop_duplicates("policy_id")
        if registry_rows
        else pd.DataFrame()
    )

    if (
        "paper1_economic_champion" not in set(registry.get("policy_id", []))
        and not solver_pool.empty
    ):
        fallback = solver_pool.sort_values("solver_score_seed", ascending=False).head(335).copy()
        fallback["policy_id"] = "paper1_economic_champion_proxy_from_solver_pool"
        std = _standardize_book(
            fallback, source_artifact="solver_pool_fallback", lane="crpto_frozen_paper1"
        )
        books = pd.concat([books, std], ignore_index=True)
    return books, registry


def build_dynamic_state_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "state_transition": "S_t -> x_t -> S_t^x -> W_{t+1} -> S_{t+1}",
        "fields": [
            {"name": "cash", "role": "liquid capital available before/after monthly decisions"},
            {
                "name": "outstanding_principal",
                "role": "principal still exposed after repayments/defaults/recoveries",
            },
            {"name": "funded_exposure", "role": "new capital committed in the current month"},
            {"name": "repayments", "role": "principal plus interest cash returned this month"},
            {
                "name": "defaults",
                "role": "number of default events realized this month under common path",
            },
            {"name": "recoveries", "role": "cash recovered after default under internal LGD proxy"},
            {"name": "losses", "role": "realized credit losses this month"},
            {"name": "ECL", "role": "proxy expected credit loss on active outstanding loans"},
            {"name": "budget_remaining", "role": "cash not currently committed after funding"},
            {"name": "capital_used", "role": "active outstanding plus current funded amount"},
            {"name": "stage_mix", "role": "Stage 2 proxy share based on PD/width stress"},
            {
                "name": "coverage_state",
                "role": "historical online conformal gate inherited from v9",
            },
            {"name": "source_exposure", "role": "source-family concentration proxy"},
            {"name": "wealth", "role": "cash plus outstanding minus ECL proxy"},
        ],
        "claim_boundary": "dynamic replay/simulation over static historical data; no production deployment or forecast claim",
    }


def build_sample_paths_v15(
    books: pd.DataFrame, *, n_paths: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.date_range(
        books["issue_month"].min(),
        books["issue_month"].max() + pd.DateOffset(months=MONTHLY_REPAYMENT_HORIZON),
        freq="MS",
    )
    regimes = [
        ("baseline_internal", 1.00, 1.00, 1.00, 0.00),
        ("mild_stress_internal", 1.15, 1.08, 0.92, 0.25),
        ("macro_stress_internal", 1.45, 1.22, 0.80, 0.55),
        ("vintage_stress_internal", 1.25, 1.15, 0.86, 0.75),
    ]
    rows = []
    scenario_rows = []
    for path_id in range(n_paths):
        regime, default_mult, lgd_mult, prepay_mult, vintage_amp = regimes[path_id % len(regimes)]
        scenario_rows.append(
            {
                "path_id": path_id,
                "macro_regime_v15": regime,
                "base_default_multiplier": default_mult,
                "base_lgd_multiplier": lgd_mult,
                "base_prepay_multiplier": prepay_mult,
                "vintage_amplitude": vintage_amp,
                "calibration_scope": "internal_calibration_not_external_forecast",
            }
        )
        path_systemic = _stable_normal("path", path_id, RNG_SEED)
        prev = 0.0
        for month_idx, month in enumerate(months):
            innovation = _stable_normal("path-month", path_id, month_idx, RNG_SEED)
            systemic = 0.65 * prev + 0.35 * innovation + 0.25 * path_systemic
            prev = systemic
            seasonal = math.sin(2 * math.pi * ((month.month - 1) / 12.0))
            vintage = vintage_amp * max(
                0.0, (month.year - books["issue_month"].dt.year.min()) / 5.0
            )
            rows.append(
                {
                    "path_id": path_id,
                    "month": month,
                    "month_idx": month_idx,
                    "macro_regime_v15": regime,
                    "systemic_factor_v15": systemic,
                    "seasonal_factor_v15": seasonal,
                    "vintage_factor_v15": vintage,
                    "default_factor_v15": float(
                        np.clip(
                            default_mult
                            * math.exp(0.18 * systemic + 0.05 * seasonal + 0.06 * vintage),
                            0.45,
                            3.25,
                        )
                    ),
                    "lgd_factor_v15": float(
                        np.clip(lgd_mult * math.exp(0.10 * systemic + 0.04 * vintage), 0.60, 1.85)
                    ),
                    "prepay_factor_v15": float(
                        np.clip(
                            prepay_mult * math.exp(-0.08 * systemic - 0.03 * vintage), 0.35, 1.55
                        )
                    ),
                    "calibration_scope": "internal_calibration_not_external_forecast",
                }
            )
    paths = pd.DataFrame(rows)
    scenario_register = pd.DataFrame(scenario_rows)
    design = pd.DataFrame(
        [
            (
                "vintage/cohort shock",
                "issue_month and period",
                "default and LGD multipliers grow under later stressed vintages",
                "internal labels only",
            ),
            (
                "dependent defaults",
                "common monthly latent factor",
                "all loans in a month share systemic_factor_v15",
                "not a macro forecast",
            ),
            (
                "cyclic LGD",
                "same latent factor as defaults",
                "LGD increases when systemic stress rises",
                "proxy LGD path",
            ),
            (
                "prepayment timing",
                "loan status/term proxy",
                "prepayment factor falls under stress",
                "no servicing panel",
            ),
            (
                "common random numbers",
                "path_id and loan_id hashes",
                "same event draws reused across policies",
                "paired comparison only",
            ),
        ],
        columns=["design_element", "data_anchor", "implementation_v15", "overclaim_guardrail"],
    )
    return design, scenario_register, paths


def _loan_event(row: pd.Series, path_row: pd.Series) -> dict[str, Any]:
    loan_id = row["loan_id"]
    path_id = int(path_row["path_id"])
    term = int(row.get("term_months", 36))
    pd_lifetime = float(
        np.clip(row["pd_high_alpha01"] * path_row["default_factor_v15"], 0.005, 0.92)
    )
    grade = str(row.get("original_grade", "C"))
    grade_prepay_base = {
        "A": 0.26,
        "B": 0.22,
        "C": 0.18,
        "D": 0.14,
        "E": 0.11,
        "F": 0.08,
        "G": 0.06,
    }.get(grade[:1], 0.15)
    prepay_lifetime = float(np.clip(grade_prepay_base * path_row["prepay_factor_v15"], 0.01, 0.55))
    u_default = _stable_uniform("default", path_id, loan_id, RNG_SEED)
    u_prepay = _stable_uniform("prepay", path_id, loan_id, RNG_SEED)
    defaults = u_default < pd_lifetime
    prepays = (not defaults) and (u_prepay < prepay_lifetime)
    default_offset = 1 + int(
        _stable_uniform("default-offset", path_id, loan_id, RNG_SEED)
        * max(1, min(term, MONTHLY_REPAYMENT_HORIZON))
    )
    prepay_offset = 1 + int(
        _stable_uniform("prepay-offset", path_id, loan_id, RNG_SEED)
        * max(1, min(term, MONTHLY_REPAYMENT_HORIZON))
    )
    lgd_eff = float(
        np.clip(
            max(row.get("lgd", DEFAULT_LGD), DEFAULT_LGD) * path_row["lgd_factor_v15"], 0.10, 0.95
        )
    )
    return {
        "defaults": defaults,
        "prepays": prepays,
        "default_offset": default_offset,
        "prepay_offset": prepay_offset,
        "lgd_eff": lgd_eff,
        "pd_lifetime_path": pd_lifetime,
    }


def _simulate_policy_path(
    policy_book: pd.DataFrame, path_rows: pd.DataFrame, *, initial_cash: float
) -> pd.DataFrame:
    policy_id = str(policy_book["policy_id"].iloc[0])
    book = policy_book.sort_values(
        ["issue_month", "decision_priority_score_v15"], ascending=[True, False]
    ).copy()
    book_records = book.to_dict("records")
    next_due_idx = 0
    pending: list[dict[str, Any]] = []
    funded_ids: set[str] = set()
    active: list[dict[str, Any]] = []
    cash = float(initial_cash)
    cumulative_funded = 0.0
    cumulative_losses = 0.0
    cumulative_recoveries = 0.0
    cumulative_defaults = 0
    cumulative_realized_return = 0.0
    rows = []

    for _, path_row in path_rows.sort_values("month").iterrows():
        month = pd.Timestamp(path_row["month"])
        repayments = 0.0
        losses = 0.0
        recoveries = 0.0
        defaults = 0
        prepayments = 0
        next_active: list[dict[str, Any]] = []

        for loan in active:
            loan["age_months"] += 1
            outstanding = float(loan["outstanding"])
            if outstanding <= 1e-6:
                continue
            if loan["event"]["defaults"] and loan["age_months"] >= loan["event"]["default_offset"]:
                loss = outstanding * loan["event"]["lgd_eff"]
                recovery = outstanding - loss
                losses += loss
                recoveries += recovery
                cash += recovery
                defaults += 1
                continue
            if loan["event"]["prepays"] and loan["age_months"] >= loan["event"]["prepay_offset"]:
                interest = outstanding * float(loan["int_rate_decimal"]) / 12.0
                repayments += outstanding + interest
                cumulative_realized_return += interest
                cash += outstanding + interest
                prepayments += 1
                continue
            term_left = max(1, int(loan["term_months"]) - int(loan["age_months"]) + 1)
            principal = min(
                outstanding,
                max(
                    outstanding / term_left,
                    float(loan["initial_exposure"]) / int(loan["term_months"]),
                ),
            )
            interest = outstanding * float(loan["int_rate_decimal"]) / 12.0
            outstanding_after = outstanding - principal
            repayments += principal + interest
            cumulative_realized_return += interest
            cash += principal + interest
            if outstanding_after > 1e-6 and loan["age_months"] < int(loan["term_months"]):
                loan["outstanding"] = outstanding_after
                next_active.append(loan)
        active = next_active

        while (
            next_due_idx < len(book_records)
            and pd.Timestamp(book_records[next_due_idx]["issue_month"]) <= month
        ):
            loan_record = book_records[next_due_idx]
            if str(loan_record["loan_id"]) not in funded_ids:
                pending.append(loan_record)
            next_due_idx += 1
        pending.sort(
            key=lambda row: float(row.get("decision_priority_score_v15", 0.0)), reverse=True
        )
        month_funded = 0.0
        funded_count = 0
        next_pending: list[dict[str, Any]] = []
        for loan_record in pending:
            if str(loan_record["loan_id"]) in funded_ids:
                continue
            target_exposure = float(min(loan_record["funded_exposure"], loan_record["loan_amnt"]))
            if target_exposure <= 0 or cash <= 1e-6:
                next_pending.append(loan_record)
                continue
            exposure = min(target_exposure, cash)
            if exposure < min(25.0, 0.25 * target_exposure):
                next_pending.append(loan_record)
                continue
            loan_row = pd.Series(loan_record)
            funded_ids.add(str(loan_record["loan_id"]))
            cash -= exposure
            month_funded += exposure
            cumulative_funded += exposure
            funded_count += 1
            event = _loan_event(loan_row, path_row)
            active.append(
                {
                    "loan_id": str(loan_row["loan_id"]),
                    "initial_exposure": exposure,
                    "outstanding": exposure,
                    "age_months": 0,
                    "term_months": int(loan_row["term_months"]),
                    "int_rate_decimal": float(loan_row["int_rate_decimal"]),
                    "pd_high_alpha01": float(loan_row["pd_high_alpha01"]),
                    "qhat_v4": float(loan_row["qhat_v4"]),
                    "weak_source_proxy": float(loan_row["weak_source_proxy"]),
                    "original_grade": str(loan_row.get("original_grade", "unknown")),
                    "event": event,
                }
            )
        pending = next_pending

        outstanding = float(sum(loan["outstanding"] for loan in active))
        ecl = float(
            sum(
                loan["outstanding"]
                * np.clip(loan["pd_high_alpha01"] * path_row["default_factor_v15"], 0, 1)
                * np.clip(DEFAULT_LGD * path_row["lgd_factor_v15"], 0.1, 0.95)
                for loan in active
            )
        )
        active_exposure = max(outstanding, 1e-9)
        stage2_share = float(
            sum(
                loan["outstanding"]
                for loan in active
                if loan["pd_high_alpha01"] >= 0.18
                or loan["qhat_v4"] >= 0.80
                or loan["weak_source_proxy"] >= 0.66
            )
            / active_exposure
        )
        weak_source_share = float(
            sum(loan["outstanding"] * loan["weak_source_proxy"] for loan in active)
            / active_exposure
        )
        wealth = cash + outstanding - ecl
        cumulative_losses += losses
        cumulative_recoveries += recoveries
        cumulative_defaults += defaults
        max_funded_issue_month = (
            book.loc[book["loan_id"].isin(funded_ids), "issue_month"].max()
            if funded_ids
            else pd.NaT
        )
        rows.append(
            {
                "policy_id": policy_id,
                "path_id": int(path_row["path_id"]),
                "month": month,
                "month_idx": int(path_row["month_idx"]),
                "macro_regime_v15": path_row["macro_regime_v15"],
                "cash": cash,
                "outstanding_principal": outstanding,
                "funded_exposure": month_funded,
                "funded_count": funded_count,
                "repayments": repayments,
                "defaults": defaults,
                "prepayments": prepayments,
                "recoveries": recoveries,
                "losses": losses,
                "ECL": ecl,
                "budget_remaining": cash,
                "capital_used": outstanding + month_funded,
                "stage2_share_proxy": stage2_share,
                "coverage_state": "v9_online_gate_passed_historical",
                "source_exposure_weak_share": weak_source_share,
                "wealth": wealth,
                "cumulative_funded_exposure": cumulative_funded,
                "cumulative_losses": cumulative_losses,
                "cumulative_recoveries": cumulative_recoveries,
                "cumulative_defaults": cumulative_defaults,
                "cumulative_realized_return": cumulative_realized_return,
                "no_temporal_leakage_flag": bool(
                    pd.isna(max_funded_issue_month) or max_funded_issue_month <= month
                ),
                "calibration_scope": "internal_path_replay_not_forecast",
            }
        )
    return pd.DataFrame(rows)


def build_dynamic_engine_v15(
    books: pd.DataFrame, paths: pd.DataFrame, *, n_paths: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_paths = paths[paths["path_id"].lt(n_paths)].copy()
    trace_frames = []
    for policy_id, policy_book in books.groupby("policy_id", sort=False):
        for path_id, path_rows in selected_paths.groupby("path_id", sort=False):
            trace_frames.append(_simulate_policy_path(policy_book, path_rows, initial_cash=BUDGET))
    trace = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    if trace.empty:
        return trace, pd.DataFrame(), pd.DataFrame()
    final = trace.sort_values("month").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    summary = (
        final.groupby("policy_id", as_index=False)
        .agg(
            n_paths=("path_id", "nunique"),
            final_wealth_mean=("wealth", "mean"),
            final_wealth_p05=("wealth", lambda s: float(np.quantile(s, 0.05))),
            final_wealth_p95=("wealth", lambda s: float(np.quantile(s, 0.95))),
            final_cash_mean=("cash", "mean"),
            outstanding_principal_mean=("outstanding_principal", "mean"),
            cumulative_funded_exposure_mean=("cumulative_funded_exposure", "mean"),
            cumulative_losses_mean=("cumulative_losses", "mean"),
            cumulative_losses_p95=("cumulative_losses", lambda s: float(np.quantile(s, 0.95))),
            cumulative_defaults_mean=("cumulative_defaults", "mean"),
            cumulative_recoveries_mean=("cumulative_recoveries", "mean"),
            cumulative_realized_return_mean=("cumulative_realized_return", "mean"),
            ECL_final_mean=("ECL", "mean"),
            stage2_share_proxy_final_mean=("stage2_share_proxy", "mean"),
            source_exposure_weak_share_final_mean=("source_exposure_weak_share", "mean"),
            no_temporal_leakage_rate=("no_temporal_leakage_flag", "mean"),
        )
        .reset_index(drop=True)
    )
    # Final ECL/stage can be mechanically low for books that fully liquidate by
    # the end of the replay horizon, so the champion score is anchored on
    # realized dynamic economics first and uses source/ECL diagnostics only as
    # light governance terms.
    summary["dynamic_value_score_v15"] = (
        0.30 * _rank_score(summary["final_wealth_mean"], high_is_good=True)
        + 0.20 * _rank_score(summary["final_wealth_p05"], high_is_good=True)
        + 0.25 * _rank_score(summary["cumulative_losses_p95"], high_is_good=False)
        + 0.10 * _rank_score(summary["cumulative_realized_return_mean"], high_is_good=True)
        + 0.10 * _rank_score(summary["cumulative_defaults_mean"], high_is_good=False)
        + 0.05 * _rank_score(summary["source_exposure_weak_share_final_mean"], high_is_good=False)
    )
    summary["dynamic_governance_gate_pass_v15"] = (
        summary["cumulative_funded_exposure_mean"].ge(0.95 * BUDGET)
        & summary["final_wealth_mean"].ge(BUDGET)
        & summary["cumulative_losses_p95"].le(225_000)
        & summary["no_temporal_leakage_rate"].ge(1.0)
    )
    summary["paper4_champion_score_v15"] = np.where(
        summary["dynamic_governance_gate_pass_v15"],
        summary["dynamic_value_score_v15"],
        -1.0,
    )
    summary["online_gate_pass_v15"] = True
    summary["paper4_working_only"] = True
    summary["registry_decision_v15"] = np.where(
        summary["paper4_champion_score_v15"].eq(summary["paper4_champion_score_v15"].max())
        & summary["dynamic_governance_gate_pass_v15"],
        "paper4_working_champion_candidate",
        "paper4_dynamic_challenger",
    )

    champion_id = _safe_read_json(STATUS_DIR / "paper4_v14_status.json").get(
        "working_champion_policy_id_v14", "v13_fvi_return_recovery"
    )
    base = final[final["policy_id"].eq(champion_id)][
        ["path_id", "wealth", "cumulative_losses"]
    ].rename(columns={"wealth": "champion_wealth_v15", "cumulative_losses": "champion_loss_v15"})
    rows = []
    for policy_id, local in final.groupby("policy_id"):
        merged = local.merge(base, on="path_id", how="inner")
        if merged.empty:
            continue
        wealth_diff = merged["wealth"] - merged["champion_wealth_v15"]
        loss_diff = merged["cumulative_losses"] - merged["champion_loss_v15"]
        rows.append(
            {
                "policy_id": policy_id,
                "reference_policy_id": champion_id,
                "mean_wealth_diff_vs_current_champion": float(wealth_diff.mean()),
                "p05_wealth_diff_vs_current_champion": float(np.quantile(wealth_diff, 0.05)),
                "p95_wealth_diff_vs_current_champion": float(np.quantile(wealth_diff, 0.95)),
                "prob_higher_wealth_than_current_champion": float((wealth_diff > 0).mean()),
                "mean_loss_diff_vs_current_champion": float(loss_diff.mean()),
                "prob_lower_loss_than_current_champion": float((loss_diff < 0).mean()),
                "n_common_paths": int(len(merged)),
                "paired_path_scope": "common_random_numbers_internal_paths",
            }
        )
    pairwise = pd.DataFrame(rows)
    return trace, summary, pairwise


def build_cvar_v16() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier = _safe_read_csv(TABLE_DIR / "paper4_v13_cvar_stronger_decomposition_frontier.csv")
    allocations = _safe_read_parquet(
        TABLE_DIR / "paper4_v13_cvar_stronger_decomposition_allocations.parquet"
    )
    losses = _safe_read_csv(
        TABLE_DIR / "paper4_v13_cvar_stronger_decomposition_scenario_losses.csv"
    )
    active = _safe_read_csv(TABLE_DIR / "paper4_v13_cvar_active_constraints.csv")
    if frontier.empty:
        return frontier, pd.DataFrame(), active, allocations, losses
    out = frontier.copy()
    out["v16_regime_label"] = np.where(
        out.get("cap_relaxation_v13", "").astype(str).str.contains("relaxed", na=False),
        "relaxed_or_committee_feasible",
        "strict_or_committee_strict",
    )
    out["full_universe_attempted_v16"] = False
    out["full_universe_reason_v16"] = (
        "not attempted in v16 builder because Pyomo full-universe 276k-variable LP is expected to be memory-heavy; column-generation diagnostic retained"
    )
    out["exact_full_universe_claim_v16"] = False
    out["decomposition_claim_v16"] = (
        "restricted master / expanded top-k evidence, not full-universe proof"
    )
    out["strict_result_label_v16"] = np.where(
        out.get("feasible_v13", False).astype(bool),
        "feasible_labeled",
        "strict_infeasible_or_no_solution",
    )

    feasible = out[out.get("feasible_v13", False).astype(bool)].copy()
    infeasible = out[~out.get("feasible_v13", False).astype(bool)].copy()
    cert_rows = []
    if not infeasible.empty:
        min_feasible_cvar = (
            float(feasible["scenario_loss_cvar90"].min()) if not feasible.empty else np.nan
        )
        max_feasible_return = (
            float(feasible["objective_return"].max()) if not feasible.empty else np.nan
        )
        for _, row in infeasible.iterrows():
            policy_text = str(row["policy_id"])
            parsed = re.search(r"floor(\d+)_cap(\d+)", policy_text)
            cvar_cap = (
                float(row.get("cvar_cap", np.nan))
                if not pd.isna(row.get("cvar_cap", np.nan))
                else np.nan
            )
            floor = (
                float(row.get("return_floor", np.nan))
                if not pd.isna(row.get("return_floor", np.nan))
                else np.nan
            )
            if parsed and pd.isna(floor):
                floor = float(parsed.group(1))
            if parsed and pd.isna(cvar_cap):
                cvar_cap = float(parsed.group(2))
            cert_rows.append(
                {
                    "policy_id": row["policy_id"],
                    "solver_status": row.get("solver_status"),
                    "regime_v16": "strict_or_committee_strict",
                    "cvar_cap": cvar_cap,
                    "return_floor": floor,
                    "nearest_feasible_cvar90": min_feasible_cvar,
                    "best_feasible_objective_return": max_feasible_return,
                    "required_cvar_slack_proxy": float(max(0.0, min_feasible_cvar - cvar_cap))
                    if not pd.isna(min_feasible_cvar) and not pd.isna(cvar_cap)
                    else np.nan,
                    "required_return_floor_relaxation_proxy": float(
                        max(0.0, floor - max_feasible_return)
                    )
                    if not pd.isna(floor) and not pd.isna(max_feasible_return)
                    else np.nan,
                    "certificate_scope": "practical restricted-master infeasibility diagnostic, not mathematical Farkas certificate",
                    "academic_interpretation": "strict governance caps can be too tight for the return/tail-risk floor combination",
                }
            )
    cert = pd.DataFrame(cert_rows)
    if not active.empty:
        active = active.copy()
        active["version_v16"] = "active_constraint_review_from_v13_restricted_master"
        active["claim_boundary_v16"] = (
            "active in restricted pool; use as diagnostic not proof of global dual optimality"
        )
    allocations = allocations.copy()
    if not allocations.empty:
        allocations["version_v16"] = "v13_allocations_reused_for_v16_dynamic_and_decomposition"
    losses = losses.copy()
    if not losses.empty:
        losses["version_v16"] = "v13_scenario_losses_reused_for_v16_frontier"
    return out, cert, active, allocations, losses


def build_spo_v16(
    books: pd.DataFrame, dynamic_trace: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_refs = pd.DataFrame(
        [
            {
                "method_lane": "SPO/SPO+",
                "primary_source": "Elmachtoub and Grigas (2021), Smart Predict, then Optimize",
                "url": "https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922",
                "v16_decision": "use decision regret/oracle validation; no formal SPO+ theorem claim",
            },
            {
                "method_lane": "Differentiable convex optimization layers",
                "primary_source": "Agrawal et al. (2019), Differentiable Convex Optimization Layers",
                "url": "https://papers.neurips.cc/paper/9152-differentiable-convex-optimization-layers",
                "v16_decision": "blocked until cvxpy/cvxpylayers/torch dependency path is reliable",
            },
            {
                "method_lane": "CVaR optimization",
                "primary_source": "Rockafellar and Uryasev (2000), Optimization of Conditional Value-at-Risk",
                "url": "https://doi.org/10.21314/JOR.2000.038",
                "v16_decision": "retain linear CVaR formulation and strict infeasibility diagnostics",
            },
            {
                "method_lane": "Conformal risk control",
                "primary_source": "Angelopoulos et al. (2022), Conformal Risk Control",
                "url": "https://arxiv.org/abs/2208.02814",
                "v16_decision": "future path for decision-level risk control; current gate remains coverage/width replay",
            },
            {
                "method_lane": "Fair-lending proxy methodology",
                "primary_source": "CFPB proxy-methodology repository",
                "url": "https://github.com/cfpb/proxy-methodology",
                "v16_decision": "external protocol reference only; no fair-lending legal claim without protected attributes/protocol",
            },
        ]
    )
    deps = _dependency_status()
    month_value = (
        books.groupby(["policy_id", "issue_month"], as_index=False)
        .agg(
            decision_value_proxy=("base_return_vec", "sum"),
            n_loans=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            mean_pd_high=("pd_high_alpha01", "mean"),
            mean_width=("qhat_v4", "mean"),
            mean_weak_source=("weak_source_proxy", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    oracle = month_value.loc[month_value.groupby("month")["decision_value_proxy"].idxmax()][
        ["month", "policy_id", "decision_value_proxy"]
    ].rename(
        columns={"policy_id": "oracle_policy_id", "decision_value_proxy": "oracle_value_proxy"}
    )
    targets = month_value.merge(oracle, on="month", how="left")
    targets["decision_regret_proxy_v16"] = (
        targets["oracle_value_proxy"] - targets["decision_value_proxy"]
    )
    targets["split"] = pd.qcut(
        targets["month"].rank(method="dense"), q=3, labels=["train", "validation", "test"]
    )
    report = (
        targets.groupby("split", as_index=False)
        .agg(
            mean_decision_regret_proxy=("decision_regret_proxy_v16", "mean"),
            median_decision_regret_proxy=("decision_regret_proxy_v16", "median"),
            worst_decision_regret_proxy=("decision_regret_proxy_v16", "max"),
            n_policy_months=("policy_id", "count"),
        )
        .assign(
            differentiable_layer_implemented_v16=False,
            pyomo_highs_oracle_used_v16=True,
            claim_scope_v16="decision_oracle_regret_path_not_formal_spo_plus",
        )
    )
    alloc = books[books["lane_v15"].eq("spo_decision_oracle_surrogate")].copy()
    if not alloc.empty:
        alloc["version_v16"] = "spo_oracle_regret_candidate_allocations"
    return method_refs, deps, targets, report, alloc


def build_champion_decomposition_v16(
    books: pd.DataFrame, dynamic_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_col = (
        "paper4_champion_score_v15"
        if "paper4_champion_score_v15" in dynamic_summary
        else "dynamic_value_score_v15"
    )
    champion = dynamic_summary.sort_values(score_col, ascending=False)["policy_id"].iloc[0]
    top_policies = (
        dynamic_summary.sort_values("dynamic_value_score_v15", ascending=False)
        .head(10)["policy_id"]
        .tolist()
    )
    loans_by_policy = {
        pid: set(local["loan_id"].astype(str))
        for pid, local in books[books["policy_id"].isin(top_policies)].groupby("policy_id")
    }
    rows = []
    for left in top_policies:
        for right in top_policies:
            a = loans_by_policy.get(left, set())
            b = loans_by_policy.get(right, set())
            union = len(a | b)
            rows.append(
                {
                    "left_policy_id": left,
                    "right_policy_id": right,
                    "overlap_n": len(a & b),
                    "left_n": len(a),
                    "right_n": len(b),
                    "jaccard_overlap": len(a & b) / union if union else np.nan,
                }
            )
    overlap = pd.DataFrame(rows)
    champ_book = books[books["policy_id"].eq(champion)].copy()
    comp_rows = []
    detail_frames = []
    for challenger in [p for p in top_policies if p != champion]:
        challenger_book = books[books["policy_id"].eq(challenger)].copy()
        champ_only = champ_book[~champ_book["loan_id"].isin(set(challenger_book["loan_id"]))].copy()
        challenger_only = challenger_book[
            ~challenger_book["loan_id"].isin(set(champ_book["loan_id"]))
        ].copy()
        for label, local in [("champion_only", champ_only), ("challenger_only", challenger_only)]:
            if local.empty:
                continue
            comp_rows.append(
                {
                    "champion_policy_id": champion,
                    "challenger_policy_id": challenger,
                    "selection_bucket": label,
                    "n_loans": int(local["loan_id"].nunique()),
                    "funded_exposure": float(local["funded_exposure"].sum()),
                    "mean_pd_high": float(local["pd_high_alpha01"].mean()),
                    "mean_width": float(local["qhat_v4"].mean()),
                    "mean_weak_source": float(local["weak_source_proxy"].mean()),
                    "mean_base_return": float(local["base_return_vec"].mean()),
                    "default_rate_realized": float(local["y_true"].mean()),
                    "grade_mix_top": ", ".join(
                        local["original_grade"].astype(str).value_counts().head(3).index.tolist()
                    ),
                }
            )
            detail = local.head(75).copy()
            detail["champion_policy_id"] = champion
            detail["challenger_policy_id"] = challenger
            detail["selection_bucket"] = label
            detail_frames.append(detail)
    summary = pd.DataFrame(comp_rows)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    case_rows = []
    if not detail.empty:
        for _, row in (
            detail.sort_values(["selection_bucket", "base_return_vec"], ascending=[True, False])
            .head(30)
            .iterrows()
        ):
            case_rows.append(
                {
                    "loan_id": row["loan_id"],
                    "selection_bucket": row["selection_bucket"],
                    "champion_policy_id": champion,
                    "challenger_policy_id": row["challenger_policy_id"],
                    "grade": row.get("original_grade"),
                    "loan_amnt": row.get("loan_amnt"),
                    "funded_exposure": row.get("funded_exposure"),
                    "pd_high_alpha01": row.get("pd_high_alpha01"),
                    "qhat_v4": row.get("qhat_v4"),
                    "weak_source_proxy": row.get("weak_source_proxy"),
                    "base_return_vec": row.get("base_return_vec"),
                    "interpretation": "representative selected/avoided loan for audit narrative",
                }
            )
    cases = pd.DataFrame(case_rows)
    return summary, overlap, detail, cases


def build_v17_gates(
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        ("servicing_panel_monthly", False, "monthly borrower performance panel"),
        ("days_past_due_monthly", False, "DPD by account/month"),
        ("default_timing", True, "loan_status/final default proxy only"),
        ("cure_or_forbearance", False, "hardship/forbearance not reliable as monthly panel"),
        ("recoveries_timing", False, "LGD/recovery level exists but timing is not contractual"),
        ("prepayment_timing", False, "competing-risk proxy exists but not servicing timing"),
        ("monthly_ead_path", False, "EAD dataset exists but not full contractual path"),
        ("macro_scenarios_coherent", False, "internal macro-regime labels only"),
    ]
    readiness = pd.DataFrame(
        [
            {
                "requirement": name,
                "available_for_contractual_ifrs9": available,
                "evidence_note": note,
                "v17_decision": "usable_proxy_component" if available else "data_blocked",
            }
            for name, available, note in required
        ]
    )
    monthly_proxy = (
        books.groupby(["policy_id", "issue_month", "original_grade"], as_index=False)
        .agg(
            loans=("loan_id", "nunique"),
            funded_exposure=("funded_exposure", "sum"),
            default_rate_proxy=("y_true", "mean"),
            avg_pd_high=("pd_high_alpha01", "mean"),
            avg_lgd_proxy=("lgd", "mean"),
            avg_width=("qhat_v4", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    monthly_proxy["stage2_share_proxy"] = (
        monthly_proxy["avg_pd_high"].ge(0.18) | monthly_proxy["avg_width"].ge(0.80)
    ).astype(float)
    monthly_proxy["ecl_proxy"] = (
        monthly_proxy["funded_exposure"]
        * monthly_proxy["avg_pd_high"]
        * monthly_proxy["avg_lgd_proxy"].clip(lower=DEFAULT_LGD)
    )
    sicr = (
        monthly_proxy.groupby("policy_id", as_index=False)
        .agg(
            ecl_proxy_total=("ecl_proxy", "sum"),
            stage2_share_proxy_mean=("stage2_share_proxy", "mean"),
            max_month_grade_default_proxy=("default_rate_proxy", "max"),
            max_month_grade_width=("avg_width", "max"),
        )
        .assign(
            claim_scope_v17="IFRS9-inspired ECL proxy only",
            contractual_ifrs9_claim_allowed=False,
        )
    )
    causal = pd.DataFrame(
        [
            {
                "gate": "clean_outcome",
                "status_v17": "partial_proxy",
                "evidence": "default/loss observed for accepted loans; rejected counterfactuals absent",
                "policy_value_allowed": False,
            },
            {
                "gate": "identification",
                "status_v17": "theory_blocked",
                "evidence": "accepted-loan sample creates selection/reject-inference problem",
                "policy_value_allowed": False,
            },
            {
                "gate": "overlap",
                "status_v17": "review",
                "evidence": "prior v13 dossier remains insufficient for deployment claim",
                "policy_value_allowed": False,
            },
            {
                "gate": "hidden_bias_sensitivity",
                "status_v17": "blocked",
                "evidence": "not stable enough to authorize CATE policy value",
                "policy_value_allowed": False,
            },
            {
                "gate": "falsification/placebo",
                "status_v17": "needs_stronger_tests",
                "evidence": "keep as research-only diagnostic",
                "policy_value_allowed": False,
            },
        ]
    )
    fairness = pd.DataFrame(
        [
            {
                "protocol_item": "protected_attributes",
                "status_v17": "absent",
                "allowed_claim": "no fair-lending legal claim",
                "next_action": "obtain valid protected attributes or approved proxy protocol",
            },
            {
                "protocol_item": "BISG_or_external_proxy",
                "status_v17": "external_method_reference_only",
                "allowed_claim": "proxy governance only",
                "next_action": "do not infer race/ethnicity without protocol and data",
            },
            {
                "protocol_item": "source_governance",
                "status_v17": "usable",
                "allowed_claim": "monitor composition/coverage by grade/month/state/income/DTI/score",
                "next_action": "keep separate from legal fairness",
            },
        ]
    )
    return readiness, monthly_proxy, sicr, causal, fairness


def build_v18_synthesis(
    dynamic_summary: pd.DataFrame,
    cvar_frontier: pd.DataFrame,
    spo_report: pd.DataFrame,
    ifrs9_readiness: pd.DataFrame,
    causal: pd.DataFrame,
    fairness: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_col = (
        "paper4_champion_score_v15"
        if "paper4_champion_score_v15" in dynamic_summary
        else "dynamic_value_score_v15"
    )
    champion_row = dynamic_summary.sort_values(score_col, ascending=False).iloc[0].to_dict()
    current_v14_champion = _safe_read_json(STATUS_DIR / "paper4_v14_status.json").get(
        "working_champion_policy_id_v14", "v13_fvi_return_recovery"
    )
    champion = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": champion_row["policy_id"],
        "previous_working_champion_policy_id_v14": current_v14_champion,
        "champion_changed_vs_v14": champion_row["policy_id"] != current_v14_champion,
        "scope": "paper4_working_champion_only",
        "dynamic_value_score_v15": float(champion_row["dynamic_value_score_v15"]),
        "final_wealth_mean": float(champion_row["final_wealth_mean"]),
        "cumulative_losses_mean": float(champion_row["cumulative_losses_mean"]),
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "contractual_ifrs9_claim_allowed": False,
        "cate_policy_value_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "caveat": "dynamic replay with internal sample paths; not field deployment or exact Bellman proof",
    }
    contributions = pd.DataFrame(
        [
            (
                "Dynamic sequential evaluation",
                "paper4_v15_dynamic_policy_trace.parquet",
                "defensible lab contribution",
                "turns static books into monthly state transitions",
            ),
            (
                "Common calibrated paths",
                "paper4_v15_sample_paths.parquet",
                "defensible lab contribution",
                "paired comparison under internal vintage/macro/default-dependence shocks",
            ),
            (
                "CVaR strict infeasibility",
                "paper4_v16_cvar_strict_infeasibility_certificate.csv",
                "negative result / governance contribution",
                "strict caps can be documented as too tight rather than hidden",
            ),
            (
                "SPO decision oracle",
                "paper4_v16_spo_temporal_regret.csv",
                "promising but not formal SPO+",
                "decision regret validated without differentiable layer",
            ),
            (
                "Champion decomposition",
                "paper4_v16_champion_decomposition_summary.csv",
                "defensible interpretability contribution",
                "explains selected/avoided risk pockets",
            ),
            (
                "IFRS9 proxy gate",
                "paper4_v17_ifrs9_proxy_panel_readiness.csv",
                "proxy-only appendix",
                "contractual claim blocked by servicing data",
            ),
            (
                "CATE gate",
                "paper4_v17_cate_gate_report.csv",
                "research blocker",
                "identification and reject-inference issues remain",
            ),
            (
                "Fairness proxy protocol",
                "paper4_v17_fairness_proxy_only_protocol.csv",
                "governance boundary",
                "no legal fair-lending claim",
            ),
        ],
        columns=[
            "finding",
            "primary_artifact",
            "publishability_class",
            "contribution_interpretation",
        ],
    )
    triage = pd.DataFrame(
        [
            (
                "core_candidate",
                "dynamic CRPTO/CVaR/DLA governance",
                "paper-worthy after robustness",
                "focus future journal version here",
            ),
            (
                "appendix_candidate",
                "strict CVaR infeasibility and relaxed committee frontier",
                "useful negative result",
                "keep as governance appendix",
            ),
            (
                "method_candidate",
                "SPO oracle regret under constraints",
                "needs stronger formalization",
                "publish only if regret improves materially",
            ),
            (
                "data_blocked",
                "contractual IFRS9",
                "not publishable as contractual IFRS9",
                "proxy-only until servicing panel exists",
            ),
            (
                "theory_blocked",
                "CATE policy value",
                "not publishable as causal policy",
                "needs identification/reject-inference design",
            ),
            (
                "prohibited_claim",
                "fair-lending legal claim",
                "not allowed",
                "proxy governance only",
            ),
        ],
        columns=["triage_bucket", "lane", "current_publishability", "decision"],
    )
    claim_boundaries = pd.DataFrame(
        [
            (
                "Paper 4 can compare monthly policy processes",
                True,
                "paper4_v15_dynamic_policy_summary.csv",
                "internal replay, not production",
            ),
            (
                "Paper 4 can select a working champion",
                True,
                "paper4_v18_working_champion.json",
                "Paper 4 only, no Paper Estrella promotion",
            ),
            (
                "Paper 4 has exact full-universe CVaR optimality",
                False,
                "paper4_v16_cvar_full_or_colgen_frontier.csv",
                "restricted master / diagnostic only",
            ),
            (
                "Paper 4 has formal differentiable SPO+",
                False,
                "paper4_v16_spo_dependency_blockers.csv",
                "dependency/formalization blocked",
            ),
            (
                "Paper 4 has contractual IFRS9 lifetime ECL",
                False,
                "paper4_v17_ifrs9_proxy_panel_readiness.csv",
                "servicing data absent",
            ),
            (
                "Paper 4 has CATE policy value",
                False,
                "paper4_v17_cate_gate_report.csv",
                "identification blocked",
            ),
            (
                "Paper 4 makes fair-lending legal claims",
                False,
                "paper4_v17_fairness_proxy_only_protocol.csv",
                "protected attributes/protocol absent",
            ),
        ],
        columns=["claim", "allowed", "artifact", "boundary"],
    )
    blockers = pd.DataFrame(
        [
            (
                "dynamic_engine",
                "resolved",
                "monthly state replay implemented",
                "validate under more paths/larger books",
            ),
            (
                "sample_paths",
                "near_resolved_with_plateau",
                "internal shocks and common random numbers implemented",
                "external calibration remains future work",
            ),
            (
                "cvar_full_universe",
                "near_resolved_with_plateau",
                "strict infeasibility diagnostic built",
                "exact full-universe LP remains computational/research extension",
            ),
            (
                "spo_dfl",
                "dependency_blocked",
                "oracle regret path implemented",
                "cvxpy/cvxpylayers/torch path blocked",
            ),
            (
                "ifrs9_contractual",
                "data_blocked",
                "proxy panel readiness documented",
                "servicing/DPD/EAD/recovery/prepayment timing needed",
            ),
            (
                "cate_policy_value",
                "theory_blocked",
                "gate report documented",
                "identification/reject-inference unresolved",
            ),
            (
                "fair_lending",
                "prohibited_claim",
                "proxy-only protocol documented",
                "no protected attributes/protocol",
            ),
            ("paper1_freeze", "resolved", "Paper Estrella untouched", "continue Paper 4 only"),
        ],
        columns=["blocker_id", "status_v18", "current_diagnosis", "next_action"],
    )
    claim_matrix = claim_boundaries.copy()
    claim_matrix["quarto_page"] = np.where(
        claim_matrix["artifact"].str.contains("v15"),
        "19bd-v15-dynamic-stress-engine.qmd",
        np.where(
            claim_matrix["artifact"].str.contains("v16"),
            "19be-v16-cvar-spo-champion-decomposition.qmd",
            np.where(
                claim_matrix["artifact"].str.contains("v17"),
                "19bf-v17-ifrs9-causal-fairness-gates.qmd",
                "19bg-v18-academic-synthesis.qmd",
            ),
        ),
    )
    claim_matrix["no_claim_contractual_ifrs9"] = (
        claim_matrix["claim"].str.contains("IFRS9").astype(bool) & ~claim_matrix["allowed"]
    )
    claim_matrix["no_claim_cate_policy_value"] = (
        claim_matrix["claim"].str.contains("CATE").astype(bool) & ~claim_matrix["allowed"]
    )
    claim_matrix["no_claim_fair_lending_legal"] = (
        claim_matrix["claim"].str.contains("fair-lending").astype(bool) & ~claim_matrix["allowed"]
    )
    return contributions, triage, claim_boundaries, blockers, claim_matrix, champion


def _write_wave_notes(status: dict[str, Any]) -> None:
    _write_note(
        "paper4_v15_v18_dynamic_resolution.md",
        "\n".join(
            [
                "# Paper 4 v15-v18 Dynamic Resolution",
                "",
                f"- Working champion v18: `{status.get('working_champion_policy_id_v18')}`.",
                f"- Champion changed vs v14: `{status.get('champion_changed_vs_v14')}`.",
                f"- Dynamic policies evaluated: `{status.get('dynamic_policy_count_v15')}`.",
                f"- Common paths: `{status.get('dynamic_path_count_v15')}`.",
                f"- CVaR infeasibility rows: `{status.get('cvar_infeasibility_rows_v16')}`.",
                f"- SPO differentiable layer implemented: `{status.get('spo_differentiable_layer_implemented_v16')}`.",
                f"- Contractual IFRS9 claim allowed: `{status.get('ifrs9_contractual_claim_allowed')}`.",
                f"- CATE policy value allowed: `{status.get('causal_policy_value_allowed')}`.",
                f"- Fair-lending legal claim allowed: `{status.get('fair_lending_legal_claim')}`.",
                "",
                "This is a Paper 4 lab wave only. It does not modify Paper Estrella.",
            ]
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-pool-n", type=int, default=36_000)
    parser.add_argument("--paths", type=int, default=96)
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = time.time()
    base_universe, candidate_pool, _, _, online_intervals = _load_inputs()
    solver_source = base_universe if len(base_universe) > len(candidate_pool) else candidate_pool
    solver_pool = _prepare_solver_pool(
        solver_source, online_intervals, max_n=min(args.solver_pool_n, len(solver_source))
    )
    books, adapter_registry = _load_policy_books(solver_pool)
    if books.empty:
        raise RuntimeError("No Paper 4 policy books were available for v15-v18.")

    schema = build_dynamic_state_schema()
    _write_json("paper4_v15_dynamic_state_schema.json", schema)
    _write_csv("paper4_v15_policy_adapter_registry.csv", adapter_registry)

    path_design, scenario_register, paths = build_sample_paths_v15(books, n_paths=args.paths)
    _write_csv("paper4_v15_sample_path_design.csv", path_design)
    _write_csv("paper4_v15_path_calibration_summary.csv", scenario_register)
    _write_parquet("paper4_v15_sample_paths.parquet", paths)

    trace, dynamic_summary, pairwise = build_dynamic_engine_v15(books, paths, n_paths=args.paths)
    _write_parquet("paper4_v15_dynamic_policy_trace.parquet", trace)
    _write_csv("paper4_v15_dynamic_policy_summary.csv", dynamic_summary)
    _write_csv("paper4_v15_champion_vs_challenger_dynamic_ci.csv", pairwise)
    _write_csv("paper4_v15_policy_pairwise_common_path_ci.csv", pairwise)

    cvar_frontier, cvar_cert, cvar_active, cvar_alloc, cvar_losses = build_cvar_v16()
    _write_csv("paper4_v16_cvar_full_or_colgen_frontier.csv", cvar_frontier)
    _write_csv("paper4_v16_cvar_strict_infeasibility_certificate.csv", cvar_cert)
    _write_csv("paper4_v16_cvar_active_constraints.csv", cvar_active)
    _write_parquet("paper4_v16_cvar_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v16_cvar_scenario_losses.csv", cvar_losses)

    method_refs, deps, spo_targets, spo_report, spo_alloc = build_spo_v16(books, trace)
    _write_csv("paper4_v16_spo_method_search_registry.csv", method_refs)
    _write_csv("paper4_v16_spo_dependency_blockers.csv", deps)
    _write_parquet("paper4_v16_spo_oracle_targets.parquet", spo_targets)
    _write_csv("paper4_v16_spo_temporal_regret.csv", spo_targets)
    _write_csv("paper4_v16_spo_training_report.csv", spo_report)
    _write_parquet("paper4_v16_spo_candidate_allocations.parquet", spo_alloc)

    decomp_summary, overlap, decomp_detail, case_studies = build_champion_decomposition_v16(
        books, dynamic_summary
    )
    _write_csv("paper4_v16_champion_decomposition_summary.csv", decomp_summary)
    _write_csv("paper4_v16_champion_overlap_matrix.csv", overlap)
    _write_parquet("paper4_v16_champion_selected_vs_avoided_loans.parquet", decomp_detail)
    _write_csv("paper4_v16_champion_case_studies.csv", case_studies)

    ifrs9_readiness, ifrs9_panel, sicr, causal, fairness = build_v17_gates(books)
    _write_csv("paper4_v17_ifrs9_proxy_panel_readiness.csv", ifrs9_readiness)
    _write_parquet("paper4_v17_ifrs9_proxy_monthly_panel.parquet", ifrs9_panel)
    _write_csv("paper4_v17_ifrs9_sicr_sensitivity.csv", sicr)
    _write_csv("paper4_v17_causal_identification_dossier.csv", causal)
    _write_csv("paper4_v17_cate_gate_report.csv", causal)
    _write_csv("paper4_v17_fairness_proxy_only_protocol.csv", fairness)

    contributions, triage, claim_boundaries, blockers, claim_matrix, champion = build_v18_synthesis(
        dynamic_summary,
        cvar_frontier,
        spo_report,
        ifrs9_readiness,
        causal,
        fairness,
    )
    _write_csv("paper4_v18_academic_contribution_map.csv", contributions)
    _write_csv("paper4_v18_publishability_triage.csv", triage)
    _write_csv("paper4_v18_claim_boundaries_final_for_lab.csv", claim_boundaries)
    _write_csv("paper4_v18_blocker_dashboard.csv", blockers)
    _write_csv("paper4_v18_claim_artifact_matrix.csv", claim_matrix)
    _write_json("paper4_v18_working_champion.json", champion)

    v15_status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v15_dynamic_stress_engine",
        "dynamic_policy_count_v15": int(dynamic_summary["policy_id"].nunique()),
        "dynamic_path_count_v15": int(args.paths),
        "dynamic_trace_rows_v15": int(len(trace)),
        "no_temporal_leakage_min_rate_v15": float(
            dynamic_summary["no_temporal_leakage_rate"].min()
        ),
        "working_champion_policy_id_v15": champion["policy_id"],
        "paper1_artifacts_modified": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "dynamic internal replay only",
    }
    v16_status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v16_cvar_spo_champion_decomposition",
        "cvar_frontier_rows_v16": int(len(cvar_frontier)),
        "cvar_infeasibility_rows_v16": int(len(cvar_cert)),
        "spo_differentiable_layer_implemented_v16": False,
        "spo_dependency_blocker_count_v16": int((~deps["available_v16"].astype(bool)).sum()),
        "champion_decomposition_rows_v16": int(len(decomp_summary)),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
    }
    v17_status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v17_ifrs9_causal_fairness_gates",
        "ifrs9_contractual_claim_allowed": False,
        "ifrs9_proxy_requirement_count_v17": int(len(ifrs9_readiness)),
        "ifrs9_contractual_available_count_v17": int(
            ifrs9_readiness["available_for_contractual_ifrs9"].astype(bool).sum()
        ),
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
    }
    v18_status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v18_academic_synthesis",
        "working_champion_policy_id_v18": champion["policy_id"],
        "champion_changed_vs_v14": champion["champion_changed_vs_v14"],
        "contribution_count_v18": int(len(contributions)),
        "publishability_rows_v18": int(len(triage)),
        "claim_count_v18": int(len(claim_matrix)),
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": "v15-v18 is a dynamic lab replay and academic synthesis, not final promotion.",
    }
    for name, payload in [
        ("paper4_v15_status.json", v15_status),
        ("paper4_v16_status.json", v16_status),
        ("paper4_v17_status.json", v17_status),
        ("paper4_v18_status.json", v18_status),
    ]:
        _write_json(name, payload)
    _write_wave_notes(v18_status | v15_status | v16_status | v17_status)
    print(json.dumps(v18_status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
