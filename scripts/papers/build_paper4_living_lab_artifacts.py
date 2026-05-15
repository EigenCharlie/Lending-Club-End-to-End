"""Build Paper 4 living-lab artifacts.

This generator is deliberately conservative. It consumes frozen Paper Estrella,
IFRS9, conformal, SPO+, fairness, and research-only artifacts, then writes
Paper 4 diagnostic artifacts without promoting or overwriting any champion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
STATUS_DIR = OUT_ROOT / "status"
NOTE_DIR = OUT_ROOT / "notes"
SCHEMA_VERSION = "2026-05-12.1"
TOTAL_OOT_LOANS = 276_869
DEFAULT_LGD = 0.45


def _load_json(path: str) -> dict[str, Any]:
    full_path = ROOT / path
    if not full_path.exists():
        return {}
    try:
        return json.loads(full_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_csv(path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(full_path)
    except Exception:
        return pd.DataFrame()


def _load_parquet(path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(full_path)
    except Exception:
        return pd.DataFrame()


def _write_csv(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def _write_parquet(name: str, df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    df.to_parquet(path, index=False)
    return path


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_note(name: str, text: str) -> Path:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTE_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def _policy_id(row: pd.Series) -> str:
    bucket = str(row.get("shortlist_bucket", "unknown")).lower().replace(" ", "_")
    return (
        f"crpto_rt{float(row['risk_tolerance']):.3f}"
        f"_g{float(row['gamma']):.2f}"
        f"_u{float(row.get('uncertainty_aversion', 0.0)):.2f}"
        f"_alpha{float(row['alpha']):.2f}"
        f"_{bucket}"
        f"_rs{int(row.get('eval_random_state', 0))}"
    ).replace(".", "p")


def _selection_universe_path() -> str:
    policy = _load_json("models/champion_portfolio_policy.json")
    return policy.get(
        "selection_universe_path",
        "data/processed/portfolio_bound_aware/rank1_alpha01_bound_aware_276k_full_2026-04-05-1734/portfolio_bound_aware_bound_eval.parquet",
    )


def load_policy_universe() -> pd.DataFrame:
    source_path = _selection_universe_path()
    raw = _load_parquet(source_path)
    if raw.empty:
        return pd.DataFrame()
    robust = raw[
        (raw["alpha"].round(6) == 0.01)
        & (raw["all_bounds_hold"].astype(bool))
        & (raw["ab_pass_all"].astype(bool))
    ].copy()
    robust["policy_id"] = robust.apply(_policy_id, axis=1)
    robust["source_artifact"] = source_path
    robust["paper4_status"] = "frozen_policy_universe"
    robust["is_paper1_champion"] = False

    promotion = _load_json("models/final_project_promotion.json")
    champion = promotion.get("final_champion", {})
    if champion:
        mask = (
            np.isclose(robust["risk_tolerance"], float(champion.get("risk_tolerance", -1)))
            & np.isclose(robust["gamma"], float(champion.get("gamma", -1)))
            & np.isclose(
                robust["realized_total_return"],
                float(champion.get("realized_total_return", np.nan)),
            )
        )
        robust.loc[mask, "is_paper1_champion"] = True
        robust.loc[mask, "policy_id"] = "paper1_economic_champion"

    cols = [
        "policy_id",
        "risk_tolerance",
        "gamma",
        "alpha",
        "confidence",
        "policy_mode",
        "uncertainty_aversion",
        "shortlist_bucket",
        "n_funded",
        "total_allocated",
        "realized_total_return",
        "price_of_robustness_pct",
        "weighted_miscoverage_V",
        "gamma_cp",
        "violation",
        "weighted_pd_true",
        "weighted_pd_high",
        "weighted_pd_point",
        "all_bounds_hold",
        "ab_pass_all",
        "is_paper1_champion",
        "source_artifact",
        "paper4_status",
    ]
    return robust[cols].sort_values("realized_total_return", ascending=False).reset_index(drop=True)


def build_source_manifest() -> pd.DataFrame:
    rows = [
        {
            "artifact": "models/final_project_promotion.json",
            "source_paper": "Paper Estrella",
            "role": "frozen champion truth",
            "status": "implemented",
            "run_tag": _load_json("models/final_project_promotion.json").get("run_tag", ""),
            "caveat": "Paper 4 may consume this artifact but must not overwrite the champion.",
        },
        {
            "artifact": _selection_universe_path(),
            "source_paper": "Paper Estrella",
            "role": "frozen robust-region policy universe",
            "status": "implemented",
            "run_tag": _load_json("models/final_project_promotion.json").get("run_tag", ""),
            "caveat": "Only alpha01 exact/pass policies are used for MVP diagnostics.",
        },
        {
            "artifact": "reports/paper_material/paper1/tables/paper1_tableA12_tail_risk_oce_cvar.csv",
            "source_paper": "Paper Estrella",
            "role": "tail-risk diagnostic seed",
            "status": "implemented",
            "run_tag": "paper1_journal_package",
            "caveat": "Diagnostic repricing, not a solver objective.",
        },
        {
            "artifact": "reports/paper_material/paper1/tables/paper1_tableA13_satisficing_margins.csv",
            "source_paper": "Paper Estrella",
            "role": "satisficing threshold seed",
            "status": "implemented",
            "run_tag": "paper1_journal_package",
            "caveat": "Thresholds are diagnostic until a selector is declared.",
        },
        {
            "artifact": "data/processed/ecl_alpha_sensitivity.parquet",
            "source_paper": "Paper 2 / IFRS9",
            "role": "ECL alpha sensitivity proxy",
            "status": "implemented",
            "run_tag": "research_only",
            "caveat": "Global ECL proxy; Paper 4 scales it for MVP diagnostics.",
        },
        {
            "artifact": "data/processed/conformal_backtest_monthly.parquet",
            "source_paper": "Paper 3 / conformal monitoring",
            "role": "monthly coverage replay seed",
            "status": "implemented",
            "run_tag": "research_only",
            "caveat": "Historical replay, not online learning.",
        },
        {
            "artifact": "data/processed/crpto_vs_spo_stability_detail.parquet",
            "source_paper": "Paper Estrella / DFL comparator",
            "role": "regret-auditability frontier seed",
            "status": "implemented",
            "run_tag": "spo_v2",
            "caveat": "SPO+ is a comparator; it does not replace CRPTO.",
        },
        {
            "artifact": "data/processed/fairness_audit.parquet",
            "source_paper": "Paper Estrella governance",
            "role": "proxy fairness stress seed",
            "status": "implemented",
            "run_tag": "fairness_proxy",
            "caveat": "Proxy/intersectional audit, not legal protected-attribute proof.",
        },
    ]
    manifest = pd.DataFrame(rows)
    manifest["path_exists"] = manifest["artifact"].map(lambda p: (ROOT / p).exists())
    return manifest


def build_sdam_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "paper4_mode": "living_lab_not_publication_first",
        "elements": {
            "R_t": {
                "name": "resource_state",
                "description": "Committed and available credit resources.",
                "examples": [
                    "budget_remaining",
                    "capital_used",
                    "funded_exposure",
                    "stage_balances",
                ],
                "current_artifacts": ["paper4_policy_level_evidence.parquet"],
            },
            "I_t": {
                "name": "information_state",
                "description": "Observable data at decision time.",
                "examples": [
                    "loan_applications",
                    "pd_point",
                    "pd_high",
                    "grade",
                    "period",
                    "scenario",
                ],
                "current_artifacts": ["paper4_loan_level_policy_evidence.parquet"],
            },
            "B_t": {
                "name": "belief_state",
                "description": "Updatable uncertainty beliefs.",
                "examples": [
                    "conformal_quantile",
                    "macro_forecast",
                    "ecl_proxy",
                    "drift_status",
                    "cate_belief",
                ],
                "current_artifacts": ["online_conformal_coverage_regret.parquet"],
            },
            "x_t": {
                "name": "decision",
                "description": "Approval, funded-set, selector, threshold, or intervention.",
                "examples": ["funded_flag", "policy_id", "alpha_schedule", "ifrs9_selector"],
                "current_artifacts": ["paper4_policy_class_registry.csv"],
            },
            "W_t_plus_1": {
                "name": "exogenous_information",
                "description": "Information observed after the decision.",
                "examples": ["default", "repayment", "recovery", "miscoverage", "new_macro"],
                "current_artifacts": ["paper4_monthly_replay.parquet"],
            },
            "S_M": {
                "name": "transition_model",
                "description": "State update from decision and new information.",
                "examples": [
                    "stage_update",
                    "budget_update",
                    "conformal_recalibration",
                    "drift_update",
                ],
                "current_artifacts": ["paper4_post_decision_state_schema.json"],
            },
            "C_t": {
                "name": "contribution",
                "description": "Reward/cost contribution for evaluation.",
                "examples": [
                    "robust_return",
                    "ecl",
                    "cvar_loss",
                    "fairness_penalty",
                    "coverage_regret",
                ],
                "current_artifacts": ["paper4_table2_net_return_after_ecl.csv"],
            },
            "X_pi": {
                "name": "policy",
                "description": "Policy class with declared parameters.",
                "examples": ["PFA", "CFA", "VFA", "DLA"],
                "current_artifacts": ["paper4_policy_class_registry.csv"],
            },
        },
        "gate": "Every Paper 4 claim must map to at least one SDAM element or remain future work.",
    }


def build_policy_class_registry(policy_universe: pd.DataFrame) -> pd.DataFrame:
    current = policy_universe.copy()
    current["source"] = "paper1_robust_region"
    current["policy_class"] = "CFA"
    current["objective_type"] = "robust_return_subject_to_conformal_constraints"
    current["evaluation_mode"] = "offline_static"
    current["decision_scope"] = "one_shot_funded_set"
    current["status"] = "implemented_frozen"
    current["notes"] = "Frozen alpha01 robust-region policy consumed from Paper Estrella."
    registry = current[
        [
            "policy_id",
            "source",
            "policy_class",
            "objective_type",
            "evaluation_mode",
            "decision_scope",
            "status",
            "is_paper1_champion",
            "source_artifact",
            "notes",
        ]
    ].copy()
    planned = pd.DataFrame(
        [
            {
                "policy_id": "paper4_ifrs9_aware_selector",
                "source": "paper4_planned",
                "policy_class": "CFA",
                "objective_type": "net_return_after_ecl_selector",
                "evaluation_mode": "offline_static",
                "decision_scope": "rank_frozen_policies",
                "status": "planned",
                "is_paper1_champion": False,
                "source_artifact": "paper4_table2_net_return_after_ecl.csv",
                "notes": "May rank frozen policies; must not create a champion without promotion JSON.",
            },
            {
                "policy_id": "paper4_tail_satisficing_selector",
                "source": "paper4_planned",
                "policy_class": "PFA/CFA",
                "objective_type": "threshold_satisficing_and_tail_gate",
                "evaluation_mode": "offline_static",
                "decision_scope": "committee_gate",
                "status": "planned",
                "is_paper1_champion": False,
                "source_artifact": "paper4_table4_satisficing_screen.csv",
                "notes": "Turns diagnostic thresholds into an explicit selector only after gates are approved.",
            },
            {
                "policy_id": "paper4_online_conformal_proto",
                "source": "paper4_planned",
                "policy_class": "PFA",
                "objective_type": "coverage_regret_update_rule",
                "evaluation_mode": "forward_replay_required",
                "decision_scope": "belief_update",
                "status": "planned",
                "is_paper1_champion": False,
                "source_artifact": "online_conformal_coverage_regret.parquet",
                "notes": "Historical monthly replay is not online until updates are forward-only.",
            },
            {
                "policy_id": "paper4_multi_period_dla",
                "source": "paper4_planned",
                "policy_class": "DLA",
                "objective_type": "rolling_horizon_return_ecl_tail",
                "evaluation_mode": "sequential",
                "decision_scope": "multi_period_portfolio",
                "status": "planned",
                "is_paper1_champion": False,
                "source_artifact": "paper4_post_decision_state_schema.json",
                "notes": "Requires transition model and explicit guarantee by period/policy.",
            },
            {
                "policy_id": "paper4_causal_policy_value",
                "source": "paper4_planned",
                "policy_class": "VFA/CFA",
                "objective_type": "causal_policy_value_gated",
                "evaluation_mode": "offline_identification_first",
                "decision_scope": "intervention_layer",
                "status": "blocked_by_identification",
                "is_paper1_champion": False,
                "source_artifact": "causal_identification_report.md",
                "notes": "No central causal objective until overlap, sensitivity, and policy value pass.",
            },
        ]
    )
    return pd.concat([registry, planned], ignore_index=True)


def build_post_decision_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "state_name": "S_t^x",
        "description": "State immediately after funding/selection and before defaults, repayments, drift, and new applications.",
        "fields": [
            {"name": "policy_id", "type": "string", "source": "policy_class_registry"},
            {"name": "funded_exposure", "type": "float", "source": "funded_set_or_policy_eval"},
            {"name": "capital_used", "type": "float", "source": "policy_level_evidence"},
            {"name": "budget_remaining", "type": "float", "source": "policy_level_evidence"},
            {"name": "stage_mix_proxy", "type": "object", "source": "ifrs9_policy_eval"},
            {"name": "conformal_status", "type": "object", "source": "V/Gamma_CP/violation"},
            {"name": "tail_status", "type": "object", "source": "OCE/CVaR overlay"},
            {"name": "fairness_proxy_status", "type": "object", "source": "fairness audit"},
        ],
        "current_status": "schema_declared_for_living_lab",
    }


def build_ifrs9_policy_grid(policy_universe: pd.DataFrame) -> pd.DataFrame:
    ecl = _load_parquet("data/processed/ecl_alpha_sensitivity.parquet")
    if ecl.empty or policy_universe.empty:
        return pd.DataFrame()
    ecl_row = ecl.iloc[(ecl["alpha"] - 0.10).abs().argsort()[:1]].iloc[0]
    scenarios = [
        ("baseline", 1.0),
        ("adverse", 1.25),
        ("severe", 1.60),
    ]
    rows: list[dict[str, Any]] = []
    for _, policy in policy_universe.iterrows():
        policy_scale = float(policy["n_funded"]) / TOTAL_OOT_LOANS
        baseline_ecl = float(ecl_row["ecl_additional"]) * policy_scale
        for scenario, multiplier in scenarios:
            ecl_value = baseline_ecl * multiplier
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "scenario": scenario,
                    "alpha_proxy": float(ecl_row["alpha"]),
                    "ecl": ecl_value,
                    "provision": ecl_value,
                    "stage2_share": float(ecl_row["pct_loans_reclassified"]),
                    "adverse_cost": ecl_value - baseline_ecl,
                    "n_funded": int(policy["n_funded"]),
                    "source_artifact": "data/processed/ecl_alpha_sensitivity.parquet",
                    "status": "diagnostic_scaled_proxy_not_policy_level_ifrs9",
                }
            )
    return pd.DataFrame(rows)


def build_net_return_table(policy_universe: pd.DataFrame, ifrs9_grid: pd.DataFrame) -> pd.DataFrame:
    if policy_universe.empty or ifrs9_grid.empty:
        return pd.DataFrame()
    baseline = ifrs9_grid[ifrs9_grid["scenario"] == "baseline"][
        ["policy_id", "ecl", "provision", "stage2_share"]
    ]
    merged = policy_universe.merge(baseline, on="policy_id", how="left")
    merged["robust_return"] = merged["realized_total_return"]
    merged["net_return_after_ecl"] = merged["robust_return"] - merged["provision"]
    merged["robust_return_rank"] = (
        merged["robust_return"].rank(ascending=False, method="min").astype(int)
    )
    merged["net_return_rank"] = (
        merged["net_return_after_ecl"].rank(ascending=False, method="min").astype(int)
    )
    merged["ranking_delta"] = merged["net_return_rank"] - merged["robust_return_rank"]
    merged["status"] = "diagnostic_scaled_proxy"
    return merged[
        [
            "policy_id",
            "risk_tolerance",
            "gamma",
            "uncertainty_aversion",
            "shortlist_bucket",
            "n_funded",
            "robust_return",
            "ecl",
            "provision",
            "net_return_after_ecl",
            "robust_return_rank",
            "net_return_rank",
            "ranking_delta",
            "stage2_share",
            "is_paper1_champion",
            "status",
        ]
    ]


def build_tail_table(policy_universe: pd.DataFrame) -> pd.DataFrame:
    tail = _load_csv("reports/paper_material/paper1/tables/paper1_tableA12_tail_risk_oce_cvar.csv")
    if tail.empty or policy_universe.empty:
        return pd.DataFrame()
    rows = []
    champion_rows = policy_universe.loc[policy_universe["is_paper1_champion"]]
    champion_return = float(
        champion_rows["realized_total_return"].max()
        if not champion_rows.empty
        else policy_universe["realized_total_return"].max()
    )
    for _, policy in policy_universe.iterrows():
        return_scale = float(policy["realized_total_return"]) / champion_return
        for _, base in tail.iterrows():
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "lgd": float(base["lgd"]),
                    "mean_loss_rate": float(base["mean_loss_rate"]) * return_scale,
                    "entropic_oce_theta5": float(base["entropic_oce_theta5"]) * return_scale,
                    "cvar_90_loss_rate": float(base["cvar_90_loss_rate"]) * return_scale,
                    "cvar_95_loss_rate": float(base["cvar_95_loss_rate"]) * return_scale,
                    "cvar_99_loss_rate": float(base["cvar_99_loss_rate"]) * return_scale,
                    "funded_set_repriced_return": float(base["funded_set_repriced_return"])
                    * return_scale,
                    "weighted_default_rate": float(policy["weighted_pd_true"]),
                    "status": "diagnostic_scaled_from_paper1_A12",
                }
            )
    return pd.DataFrame(rows)


def build_satisficing_table(policy_universe: pd.DataFrame) -> pd.DataFrame:
    sat = _load_csv("reports/paper_material/paper1/tables/paper1_tableA13_satisficing_margins.csv")
    if sat.empty or policy_universe.empty:
        return pd.DataFrame()
    thresholds = {row["criterion"]: float(row["threshold"]) for _, row in sat.iterrows()}
    rows = []
    for _, policy in policy_universe.iterrows():
        observations = {
            "return_beats_theorem_tight": float(policy["realized_total_return"]),
            "V_below_sqrt_alpha01": float(policy["weighted_miscoverage_V"]),
            "gamma_cp_below_020": float(policy["gamma_cp"]),
            "zero_violation": float(policy["violation"]),
            "robust_region_pass": 1.0 if bool(policy["all_bounds_hold"]) else 0.0,
        }
        local_margins: list[float] = []
        for criterion, observed in observations.items():
            if criterion == "return_beats_theorem_tight":
                threshold = thresholds.get(criterion, 166_269.822319)
                margin = observed - threshold
                passed = margin >= 0
            elif criterion == "zero_violation":
                threshold = 0.0
                margin = -observed
                passed = observed <= 1e-12
            elif criterion == "robust_region_pass":
                threshold = 1.0
                margin = observed - threshold
                passed = observed >= 1.0
            else:
                threshold = thresholds.get(criterion, 0.0)
                margin = threshold - observed
                passed = observed <= threshold
            local_margins.append(float(margin))
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "criterion": criterion,
                    "observed": observed,
                    "threshold": threshold,
                    "margin": margin,
                    "pass": bool(passed),
                    "fragility_score": min(local_margins) if local_margins else margin,
                    "status": "diagnostic_threshold_screen",
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_differences(
    policy_universe: pd.DataFrame, net_return: pd.DataFrame
) -> pd.DataFrame:
    if policy_universe.empty or net_return.empty:
        return pd.DataFrame()
    champion_id = "paper1_economic_champion"
    champion = policy_universe[policy_universe["policy_id"] == champion_id].iloc[0]
    net_champ = net_return[net_return["policy_id"] == champion_id].iloc[0]
    rows = []
    for _, policy in policy_universe.iterrows():
        net_row = net_return[net_return["policy_id"] == policy["policy_id"]].iloc[0]
        rows.append(
            {
                "policy_id": policy["policy_id"],
                "baseline_policy_id": champion_id,
                "sample_path_id": "alpha01_robust_region_static",
                "return_diff_vs_champion": float(
                    policy["realized_total_return"] - champion["realized_total_return"]
                ),
                "net_return_after_ecl_diff_vs_champion": float(
                    net_row["net_return_after_ecl"] - net_champ["net_return_after_ecl"]
                ),
                "V_diff_vs_champion": float(
                    policy["weighted_miscoverage_V"] - champion["weighted_miscoverage_V"]
                ),
                "gamma_cp_diff_vs_champion": float(policy["gamma_cp"] - champion["gamma_cp"]),
                "n_funded_diff_vs_champion": int(policy["n_funded"] - champion["n_funded"]),
                "status": "paired_static_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def build_loan_policy_evidence() -> pd.DataFrame:
    funded = _load_csv("reports/paper_material/paper1/tables/paper1_tableA7_funded_set_loans.csv")
    if funded.empty:
        return pd.DataFrame()
    evidence = funded.copy()
    evidence["policy_id"] = "paper1_economic_champion"
    evidence["funded_flag"] = True
    evidence["pd_interval_low"] = evidence["pd_point"]
    evidence["pd_interval_high"] = evidence["pd_high_alpha01"]
    evidence["ecl_proxy_lgd45"] = (
        evidence["pd_interval_high"] * DEFAULT_LGD * evidence["funded_exposure"]
    )
    evidence["stage_proxy"] = np.where(
        evidence["pd_interval_high"] >= 0.30, "Stage2_proxy", "Stage1_proxy"
    )
    evidence["segment"] = (
        evidence["period"].astype(str) + "_" + evidence["original_grade"].astype(str)
    )
    evidence["outcome"] = evidence["y_true"]
    evidence["status"] = "champion_funded_loan_proxy_evidence"
    evidence = evidence.rename(columns={"id": "loan_id", "original_grade": "grade"})
    return evidence[
        [
            "loan_id",
            "issue_d",
            "period",
            "policy_id",
            "funded_flag",
            "pd_point",
            "pd_interval_low",
            "pd_interval_high",
            "ecl_proxy_lgd45",
            "stage_proxy",
            "grade",
            "segment",
            "outcome",
            "portfolio_weight",
            "funded_exposure",
            "status",
        ]
    ]


def build_policy_level_evidence(
    policy_universe: pd.DataFrame,
    net_return: pd.DataFrame,
    tail: pd.DataFrame,
    sat: pd.DataFrame,
) -> pd.DataFrame:
    if policy_universe.empty:
        return pd.DataFrame()
    severe_tail = tail[(tail["lgd"] == 0.45)] if not tail.empty else pd.DataFrame()
    if not severe_tail.empty:
        severe_tail = severe_tail[["policy_id", "cvar_95_loss_rate", "entropic_oce_theta5"]]
    sat_agg = (
        sat.groupby("policy_id", as_index=False).agg(
            satisficing_pass_rate=("pass", "mean"), min_satisficing_margin=("margin", "min")
        )
        if not sat.empty
        else pd.DataFrame()
    )
    merged = policy_universe.merge(
        net_return[["policy_id", "ecl", "net_return_after_ecl"]], on="policy_id", how="left"
    )
    if not severe_tail.empty:
        merged = merged.merge(severe_tail, on="policy_id", how="left")
    if not sat_agg.empty:
        merged = merged.merge(sat_agg, on="policy_id", how="left")
    fairness = _load_parquet("data/processed/fairness_audit.parquet")
    merged["fairness_proxy_pass"] = (
        bool(fairness["passed_all"].all()) if not fairness.empty else np.nan
    )
    merged["status"] = "diagnostic_policy_level_evidence"
    return merged


def build_monthly_replay() -> pd.DataFrame:
    funded = _load_csv("reports/paper_material/paper1/tables/paper1_tableA7_funded_set_loans.csv")
    monthly = _load_parquet("data/processed/conformal_backtest_monthly.parquet")
    if funded.empty:
        return pd.DataFrame()
    funded["policy_id"] = "paper1_economic_champion"
    funded["issue_month"] = pd.to_datetime(funded["issue_d"]).dt.to_period("M").dt.to_timestamp()
    funded["realized_return_proxy_lgd45"] = (
        funded["funded_exposure"] * (funded["int_rate"] / 100.0) * (1 - funded["y_true"])
        - funded["funded_exposure"] * DEFAULT_LGD * funded["y_true"]
    )
    funded["ecl_proxy_lgd45"] = funded["pd_high_alpha01"] * DEFAULT_LGD * funded["funded_exposure"]
    replay = (
        funded.groupby("issue_month", as_index=False)
        .agg(
            period=("period", "first"),
            policy_id=("policy_id", lambda _: "paper1_economic_champion"),
            funded_count=("id", "count"),
            funded_exposure=("funded_exposure", "sum"),
            realized_return_proxy_lgd45=("realized_return_proxy_lgd45", "sum"),
            ecl_proxy_lgd45=("ecl_proxy_lgd45", "sum"),
            observed_default_rate=("y_true", "mean"),
            weighted_miscoverage=("miscovered_alpha01", "mean"),
        )
        .rename(columns={"issue_month": "month"})
    )
    if not monthly.empty:
        monthly = monthly.copy()
        monthly["month"] = pd.to_datetime(monthly["month"])
        replay = replay.merge(
            monthly[["month", "coverage_90", "coverage_95", "avg_width_90", "gap_90"]],
            on="month",
            how="left",
        )
    replay["status"] = "historical_replay_not_online_learning"
    return replay


def build_online_conformal_regret() -> pd.DataFrame:
    monthly = _load_parquet("data/processed/conformal_backtest_monthly.parquet")
    if monthly.empty:
        return pd.DataFrame()
    out = monthly.copy()
    out["month"] = pd.to_datetime(out["month"])
    out["miscoverage_90"] = 1.0 - out["coverage_90"]
    out["target_miscoverage_90"] = 0.10
    out["excess_miscoverage_90"] = (out["miscoverage_90"] - out["target_miscoverage_90"]).clip(
        lower=0
    )
    out["coverage_regret_90_cum"] = out["excess_miscoverage_90"].cumsum()
    out["status"] = "offline_monthly_replay_proxy_not_true_online"
    return out


def build_mdcp_and_fairness() -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_grade = _load_parquet("data/processed/conformal_backtest_monthly_grade.parquet")
    if monthly_grade.empty:
        mdcp = pd.DataFrame()
    else:
        mdcp = (
            monthly_grade.groupby("grade", as_index=False)
            .agg(
                n_total=("n", "sum"),
                worst_month_coverage_90=("coverage_90", "min"),
                mean_coverage_90=("coverage_90", "mean"),
                mean_width_90=("avg_width_90", "mean"),
            )
            .assign(
                target_coverage_90=0.90,
                worst_source_gap=lambda df: df["worst_month_coverage_90"] - 0.90,
                status="candidate_mdcp_source_diagnostic",
            )
        )
    fairness = _load_parquet("data/processed/fairness_audit.parquet")
    if not fairness.empty:
        fairness = fairness.copy()
        fairness["policy_id"] = "paper1_economic_champion"
        fairness["status"] = "proxy_fairness_diagnostic_not_protected_attribute_proof"
    return mdcp, fairness


def build_regret_auditability_frontier() -> pd.DataFrame:
    stability = _load_parquet("data/processed/crpto_vs_spo_stability_detail.parquet")
    if stability.empty:
        return pd.DataFrame()
    rows = []
    for _, row in stability.iterrows():
        for method, regret_col, std_col, formal in [
            ("two_stage", "two_stage_mean_regret", "two_stage_std_regret", False),
            ("spo_plus", "spo_plus_mean_regret", "spo_plus_std_regret", False),
            (
                "crpto_conformal_robust",
                "conformal_robust_mean_regret",
                "conformal_robust_std_regret",
                True,
            ),
        ]:
            coverage = float(row["coverage_90"]) if formal else np.nan
            min_grade = float(row["min_grade_coverage_90"]) if formal else np.nan
            auditability_score = 0.15
            if method == "spo_plus":
                auditability_score = 0.30
            if formal:
                auditability_score = 0.20
                auditability_score += 0.25 if coverage >= 0.90 else 0.0
                auditability_score += 0.20 if min_grade >= 0.90 else 0.0
                auditability_score += 0.20
                auditability_score += 0.15
            rows.append(
                {
                    "period": row["period"],
                    "method": method,
                    "mean_regret": float(row[regret_col]),
                    "std_regret": float(row[std_col]),
                    "coverage_90": coverage,
                    "min_grade_coverage_90": min_grade,
                    "formal_conformal_guarantee": formal,
                    "auditability_score": auditability_score,
                    "status": "current_comparator",
                }
            )
    rows.append(
        {
            "period": "planned",
            "method": "spo_plus_conformal_hybrid",
            "mean_regret": np.nan,
            "std_regret": np.nan,
            "coverage_90": np.nan,
            "min_grade_coverage_90": np.nan,
            "formal_conformal_guarantee": True,
            "auditability_score": np.nan,
            "status": "planned_experiment",
        }
    )
    return pd.DataFrame(rows)


def write_regret_auditability_plot(frontier: pd.DataFrame) -> Path | None:
    if frontier.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    current = frontier[frontier["status"] == "current_comparator"].copy()
    agg = (
        current.groupby("method", as_index=False)
        .agg(mean_regret=("mean_regret", "mean"), auditability_score=("auditability_score", "mean"))
        .sort_values("mean_regret")
    )
    colors = {
        "two_stage": "#8a8f98",
        "spo_plus": "#c35f2d",
        "crpto_conformal_robust": "#1f77b4",
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in agg.iterrows():
        ax.scatter(
            row["mean_regret"],
            row["auditability_score"],
            s=120,
            color=colors.get(row["method"], "#333333"),
        )
        ax.annotate(
            row["method"].replace("_", " "),
            (row["mean_regret"], row["auditability_score"]),
            xytext=(8, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean decision regret (lower is better)")
    ax.set_ylabel("Auditability score (higher is better)")
    ax.set_title("Paper 4 seed: regret-auditability frontier")
    ax.grid(True, alpha=0.25)
    path = FIGURE_DIR / "paper4_fig4_regret_auditability_frontier.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_causal_identification_report() -> Path:
    causal = _load_json("models/causal_policy_rule.json")
    cate = _load_json("models/cate_portfolio_status.json")
    text = f"""# Causal Identification Sketch for Paper 4

This note keeps CATE/causal CRPTO gated. It is not a central claim and it does
not change the Paper Estrella champion.

## Current status

- Selected rule: `{causal.get("selected_rule", "N/D")}`
- Overlap pass: `{causal.get("overlap_pass", "N/D")}`
- Sensitivity pass: `{causal.get("sensitivity_pass", "N/D")}`
- CATE portfolio state: `{cate.get("promotion_state", "N/D")}`

## Required before promotion

1. Treatment definition: approval, funding, pricing, hardship or intervention.
2. Outcome definition: default, loss, repayment, recovery or net value.
3. Overlap and balance report.
4. Sensitivity analysis that passes predeclared thresholds.
5. Policy value estimator with uncertainty.
6. Decision link to `x_t`, `W_{{t+1}}(S_t, x_t)`, and `C_t`.

## Current decision

Keep causal signals as `B_t`/future-intervention hypotheses. Do not use them as
an objective or selector in the Paper 4 MVP.
"""
    return _write_note("causal_identification_report.md", text)


def build_artifact_registry(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "artifact": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "role": "paper4_living_lab_artifact",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": rows,
    }


def main() -> None:
    generated: list[Path] = []
    policy_universe = load_policy_universe()
    source_manifest = build_source_manifest()
    generated.append(_write_csv("paper4_table0_source_manifest.csv", source_manifest))
    generated.append(_write_csv("paper4_policy_universe.csv", policy_universe))

    generated.append(_write_json("paper4_sequential_decision_schema.json", build_sdam_schema()))
    generated.append(
        _write_json("paper4_post_decision_state_schema.json", build_post_decision_schema())
    )

    policy_registry = build_policy_class_registry(policy_universe)
    generated.append(_write_csv("paper4_table9_policy_class_registry.csv", policy_registry))
    generated.append(_write_csv("paper4_policy_class_registry.csv", policy_registry))

    ifrs9 = build_ifrs9_policy_grid(policy_universe)
    generated.append(_write_csv("paper4_table1_policy_ecl_scenario.csv", ifrs9))
    net = build_net_return_table(policy_universe, ifrs9)
    generated.append(_write_csv("paper4_table2_net_return_after_ecl.csv", net))
    tail = build_tail_table(policy_universe)
    generated.append(_write_csv("paper4_table3_tail_risk_oce_cvar_by_policy.csv", tail))
    satisficing = build_satisficing_table(policy_universe)
    generated.append(_write_csv("paper4_table4_satisficing_screen.csv", satisficing))
    pairwise = build_pairwise_differences(policy_universe, net)
    generated.append(_write_csv("paper4_table10_policy_pairwise_differences.csv", pairwise))

    loan_evidence = build_loan_policy_evidence()
    generated.append(_write_parquet("paper4_loan_level_policy_evidence.parquet", loan_evidence))
    policy_evidence = build_policy_level_evidence(policy_universe, net, tail, satisficing)
    generated.append(_write_parquet("paper4_policy_level_evidence.parquet", policy_evidence))
    monthly = build_monthly_replay()
    generated.append(_write_parquet("paper4_monthly_replay.parquet", monthly))
    generated.append(
        _write_csv(
            "paper4_table5_temporal_replay_summary.csv",
            monthly.drop(columns=["month"], errors="ignore").head(36),
        )
    )
    online = build_online_conformal_regret()
    generated.append(_write_parquet("online_conformal_coverage_regret.parquet", online))

    mdcp, fairness = build_mdcp_and_fairness()
    generated.append(_write_parquet("mdcp_source_coverage_report.parquet", mdcp))
    generated.append(_write_parquet("fairness_constrained_policy_eval.parquet", fairness))

    frontier = build_regret_auditability_frontier()
    generated.append(_write_csv("paper4_regret_auditability_frontier.csv", frontier))
    fig_path = write_regret_auditability_plot(frontier)
    if fig_path is not None:
        generated.append(fig_path)
    generated.append(build_causal_identification_report())

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "foundation_and_mvp_diagnostics",
        "mode": "living_lab_not_publication_first",
        "paper1_champion_protected": True,
        "paper4_final_promotion_created": False,
        "n_frozen_policies": int(len(policy_universe)),
        "mvp_tables": {
            "source_manifest": "reports/paper_material/paper4/tables/paper4_table0_source_manifest.csv",
            "policy_ecl_scenario": "reports/paper_material/paper4/tables/paper4_table1_policy_ecl_scenario.csv",
            "net_return_after_ecl": "reports/paper_material/paper4/tables/paper4_table2_net_return_after_ecl.csv",
            "tail_risk": "reports/paper_material/paper4/tables/paper4_table3_tail_risk_oce_cvar_by_policy.csv",
            "satisficing": "reports/paper_material/paper4/tables/paper4_table4_satisficing_screen.csv",
            "temporal_replay": "reports/paper_material/paper4/tables/paper4_table5_temporal_replay_summary.csv",
            "policy_class_registry": "reports/paper_material/paper4/tables/paper4_table9_policy_class_registry.csv",
            "pairwise_differences": "reports/paper_material/paper4/tables/paper4_table10_policy_pairwise_differences.csv",
        },
        "caveat": "IFRS9/tail overlays are diagnostic MVP artifacts unless a future selector or solver is explicitly opened.",
        "generated_artifacts": [p.relative_to(ROOT).as_posix() for p in generated],
    }
    generated.append(_write_json("paper4_mvp_status.json", status))
    registry_path = _write_json("paper4_artifact_registry.json", build_artifact_registry(generated))
    generated.append(registry_path)
    print(json.dumps({"generated": [p.relative_to(ROOT).as_posix() for p in generated]}, indent=2))


if __name__ == "__main__":
    main()
