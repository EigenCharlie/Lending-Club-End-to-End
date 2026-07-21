"""Build compact intake artifacts for the 2026-05-21 bound-improvement handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_EXTERNAL_ROOT = Path(
    "/mnt/d/crpto_experiments/regret_auditability/regret_auditability_20260513_v3_resource_tuned"
)
TABLE_DIR = Path("reports/paper_material/paper1/tables")
DOC_DIR = Path("docs/research")
DATE_TAG = "2026-05-21"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _pd_intake(root: Path) -> pd.DataFrame:
    selection = _read_json(root / "pd" / "_selection" / "pd-refine_selection.json")
    incumbent = selection["incumbent_baseline"]
    rows: list[dict[str, Any]] = [
        {
            "role": "incumbent_replay",
            "rank": 0,
            "lane_id": incumbent["lane_id"],
            "feature_profile": incumbent["feature_profile"],
            "monotonic_policy": incumbent["monotonic_policy"],
            "auc_roc": incumbent["auc_roc"],
            "brier_score": incumbent["brier_score"],
            "ece": incumbent["ece"],
            "hpo_best_validation_auc": incumbent["hpo_best_validation_auc"],
            "delta_auc_vs_incumbent": 0.0,
            "delta_brier_vs_incumbent": 0.0,
            "delta_ece_vs_incumbent": 0.0,
            "parent_decision": "frozen_reference",
        }
    ]
    for candidate in selection["selected"]:
        rows.append(
            {
                "role": "challenger",
                "rank": candidate["selection_rank"],
                "lane_id": candidate["lane_id"],
                "feature_profile": candidate["feature_profile"],
                "monotonic_policy": candidate["monotonic_policy"],
                "auc_roc": candidate["auc_roc"],
                "brier_score": candidate["brier_score"],
                "ece": candidate["ece"],
                "hpo_best_validation_auc": candidate["hpo_best_validation_auc"],
                "delta_auc_vs_incumbent": candidate["auc_roc"] - incumbent["auc_roc"],
                "delta_brier_vs_incumbent": candidate["brier_score"] - incumbent["brier_score"],
                "delta_ece_vs_incumbent": candidate["ece"] - incumbent["ece"],
                "parent_decision": "main_challenger"
                if candidate["selection_rank"] == 1
                else "sensitivity_baseline",
            }
        )
    return pd.DataFrame(rows)


def _conformal_intake(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = root / "conformal" / "regret_auditability_20260513_v3_resource_tuned" / "data"
    group = pd.read_parquet(data_dir / "conformal_group_metrics_mondrian.parquet")
    group = group.assign(
        cov90_gap_to_0p90=group["coverage_90"] - 0.90,
        cov95_gap_to_0p95=group["coverage_95"] - 0.95,
        grade_gate=lambda df: df.apply(
            lambda row: (
                "pass" if row["coverage_90"] >= 0.90 and row["coverage_95"] >= 0.95 else "watch"
            ),
            axis=1,
        ),
        parent_action=lambda df: df.apply(
            lambda row: (
                "targeted_followup"
                if row["group"] in {"E", "F", "G"} or row["coverage_90"] < 0.90
                else "retain_as_supporting_evidence"
            ),
            axis=1,
        ),
    )

    tuning = pd.read_parquet(data_dir / "conformal_mondrian_tuning_90.parquet")
    selected = tuning[
        (tuning["partition"].eq("grade"))
        & (tuning["partition_probability_source"].eq("raw"))
        & (tuning["n_score_bins"].eq(5))
        & (tuning["fallback_mode"].eq("grade_then_global"))
        & (tuning["alpha_used_90"].eq(0.075))
        & (tuning["scaled_scores"].eq(True))
        & (tuning["score_scale_family"].eq("bernoulli_sqrt"))
        & (tuning["min_group_size"].eq(100))
    ].copy()
    selected["candidate_role"] = "child_selected"

    alternatives = []
    filters = {
        "best_width_with_group_gate": tuning[tuning["global_ok"] & tuning["group_ok"]],
        "best_min_group_0p92": tuning[tuning["global_ok"] & tuning["min_group_coverage"].ge(0.92)],
        "best_grade_variant": tuning[
            tuning["global_ok"] & tuning["partition"].astype(str).str.contains("grade")
        ],
    }
    for role, frame in filters.items():
        if frame.empty:
            continue
        if role == "best_width_with_group_gate":
            row = frame.sort_values(["width_ok", "avg_interval_width"], ascending=[False, True])
        else:
            row = frame.sort_values(
                ["min_group_coverage", "avg_interval_width"], ascending=[False, True]
            )
        best = row.head(1).copy()
        best["candidate_role"] = role
        alternatives.append(best)

    config_cols = [
        "candidate_role",
        "partition",
        "partition_probability_source",
        "n_score_bins",
        "fallback_mode",
        "alpha_used_90",
        "scaled_scores",
        "score_scale_family",
        "min_group_size",
        "empirical_coverage",
        "avg_interval_width",
        "median_interval_width",
        "min_group_coverage",
        "winkler_90",
        "max_monthly_gap",
        "is_pareto",
        "global_ok",
        "group_ok",
        "width_ok",
    ]
    config = pd.concat([selected, *alternatives], ignore_index=True)[config_cols]
    return group, config.drop_duplicates()


def _portfolio_intake(root: Path) -> pd.DataFrame:
    path = (
        root
        / "portfolio_quick_close"
        / "regret_auditability_20260513_v3_resource_tuned_quick"
        / "data"
        / "portfolio_bound_aware_bound_eval.parquet"
    )
    bound = pd.read_parquet(path)
    alpha01 = bound[bound["alpha"].eq(0.01)].copy()
    cols = [
        "candidate_rank",
        "risk_tolerance",
        "policy_mode",
        "gamma",
        "uncertainty_aversion",
        "realized_total_return",
        "n_funded",
        "gamma_cp",
        "weighted_miscoverage_V",
        "violation",
        "empirical_coverage_funded",
        "all_bounds_hold",
        "shortlist_bucket",
    ]
    alpha01 = alpha01[cols].sort_values(
        ["all_bounds_hold", "realized_total_return"], ascending=[False, False]
    )
    alpha01["parent_decision"] = alpha01["all_bounds_hold"].map(
        {True: "directional_pass_only", False: "fails_alpha01_gate"}
    )
    return alpha01


def _parent_smoke_intake() -> pd.DataFrame:
    path = (
        Path("data/processed/portfolio_bound_aware/regret_auditability_parent_smoke_2026_05_21")
        / "portfolio_bound_aware_bound_eval.parquet"
    )
    if not path.exists():
        return pd.DataFrame()

    bound = pd.read_parquet(path)
    alpha01 = bound[bound["alpha"].eq(0.01)].copy()
    cols = [
        "candidate_rank",
        "risk_tolerance",
        "policy_mode",
        "gamma",
        "uncertainty_aversion",
        "realized_total_return",
        "n_funded",
        "gamma_cp",
        "weighted_miscoverage_V",
        "violation",
        "empirical_coverage_funded",
        "all_bounds_hold",
        "shortlist_bucket",
    ]
    alpha01 = alpha01[cols].sort_values(
        ["all_bounds_hold", "realized_total_return"], ascending=[False, False]
    )
    alpha01["parent_decision"] = alpha01["all_bounds_hold"].map(
        {True: "parent_smoke_pass", False: "parent_smoke_fails_alpha01"}
    )
    return alpha01


def _bound_fronts() -> pd.DataFrame:
    rows = [
        (
            "nested_prospective_confirmation",
            "Strongest path to attack post-selection risk.",
            "Run only after PD/conformal/portfolio selection is frozen.",
            "Strict temporal or prospectively sealed split keeps alpha01 pass with zero violation.",
            "Paper Estrella appendix/main robustness.",
        ),
        (
            "direct_crc_ltt_decision_loss",
            "Current proof derives from individual conformal coverage.",
            "Calibrate monotone loss L=max(0,sum w_i Y_i - tau) or V directly.",
            "Decision-loss gate passes without weakening return or coverage.",
            "Paper 4 strong appendix; Paper Estrella if clean.",
        ),
        (
            "dependency_aware_concentration",
            "Hoeffding/Bernstein tightening currently needs strong independence.",
            "Cluster by issue_month, grade, source/state and compare cluster-robust tail bounds.",
            "Cluster-aware bound is less vacuous than Markov and more credible than iid.",
            "Paper Estrella appendix if simple; Paper 4 if complex.",
        ),
        (
            "mondrian_funded_set_refinement",
            "Current corollary exists; needs funded-set weights by group.",
            "Compute sum_g W_g alpha_g for selected and challenger policies.",
            "Weighted group bound improves over nominal alpha without hidden subgroup failure.",
            "Paper Estrella appendix.",
        ),
        (
            "decision_aware_conformal_selector",
            "Current selector is CROMS-lite over finalists.",
            "Select by coverage, width, robust_return, V, gamma_cp, violation and group gates.",
            "Selector changes or confirms conformal choice under decision metrics.",
            "Paper Estrella reviewer defense; Paper 4 method appendix.",
        ),
        (
            "less_conservative_uncertainty_sets",
            "Current set is upper-box and audit-friendly but wide.",
            "Compare grade x scoreband or polyhedral/contextual conformal candidates.",
            "Gamma_CP falls while coverage and min-group gates hold.",
            "Paper 4 first; Paper Estrella only if transparent.",
        ),
        (
            "online_shift_aware_bound",
            "Current online/source evidence is caveat, not deployment claim.",
            "Use temporal replay or online/weighted conformal only as declared retrospective gate.",
            "Coverage and V remain stable under sealed temporal slices.",
            "Paper Estrella limitation; Paper 4 future gate.",
        ),
        (
            "richer_financial_target",
            "Default target is defensible but not full loss/ECL.",
            "Prototype LGD*default or ECL proxy loss if data quality is sufficient.",
            "Financial target improves interpretability without adding unsupported IFRS9 claims.",
            "Paper 4; maybe future thesis appendix.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["front", "current_state", "next_action", "promotion_gate", "artifact_sink"],
    )


def _write_memo(
    *,
    root: Path,
    pd_table: pd.DataFrame,
    conformal_group: pd.DataFrame,
    conformal_config: pd.DataFrame,
    portfolio: pd.DataFrame,
    parent_smoke: pd.DataFrame,
    fronts: pd.DataFrame,
) -> Path:
    pd_main = pd_table[pd_table["parent_decision"].eq("main_challenger")].iloc[0]
    port_pass = portfolio[portfolio["all_bounds_hold"].eq(True)].head(1)
    best_pass = port_pass.iloc[0] if not port_pass.empty else None
    smoke_pass = parent_smoke[parent_smoke["all_bounds_hold"].eq(True)].head(1)
    best_smoke = smoke_pass.iloc[0] if not smoke_pass.empty else None

    lines = [
        "# Paper Estrella Bound-Improvement Intake 2026-05-21",
        "",
        f"External artifact root: `{root}`.",
        "",
        "## Decision",
        "",
        "This intake does not replace the frozen champion. It records a credible PD/conformal",
        "challenger package and defines gated parent-project runs for the bound-improvement lane.",
        "",
        "## PD signal",
        "",
        (
            f"Main challenger: `{pd_main['lane_id']}` with AUC `{pd_main['auc_roc']:.6f}`, "
            f"Brier `{pd_main['brier_score']:.6f}`, ECE `{pd_main['ece']:.6f}`. "
            f"Delta AUC vs incumbent replay is `{pd_main['delta_auc_vs_incumbent']:.6f}`."
        ),
        "",
        "## Conformal signal",
        "",
        "The child-selected conformal configuration is usable, but not final: grade E fails",
        "the strict 90% group gate and E/G are weak at 95%. The next parent action is a",
        "focused conformal follow-up before any full cuOpt portfolio promotion.",
        "",
        "## Portfolio quick signal",
        "",
    ]
    if best_pass is not None:
        lines.append(
            f"The quick CPU run produced an alpha01 pass candidate with return "
            f"`{best_pass['realized_total_return']:.2f}`, `V={best_pass['weighted_miscoverage_V']:.6f}`, "
            f"`Gamma_CP={best_pass['gamma_cp']:.6f}` and zero violation, but it used only 25k candidates."
        )
    else:
        lines.append("The quick CPU run did not produce an alpha01 pass candidate.")
    lines += [
        "",
        "## Parent smoke",
        "",
    ]
    if best_smoke is not None:
        lines.append(
            f"The parent-project HiGHS smoke run confirmed local compatibility with the external "
            f"intervals and produced an alpha01 pass candidate: mode `{best_smoke['policy_mode']}`, "
            f"risk `{best_smoke['risk_tolerance']:.3f}`, gamma `{best_smoke['gamma']:.3f}`, return "
            f"`{best_smoke['realized_total_return']:.2f}`, `V={best_smoke['weighted_miscoverage_V']:.6f}`, "
            f"`Gamma_CP={best_smoke['gamma_cp']:.6f}` and zero violation. This is a compatibility "
            f"smoke, not a replacement for the full cuOpt search."
        )
    else:
        lines.append("No parent-project smoke run was available when this memo was generated.")
    lines += [
        "",
        "## Parent-project gates",
        "",
        "- Do not compare the quick 25k return directly with the frozen 276k champion.",
        "- Run final portfolio only after focused conformal follow-up.",
        "- Use cuOpt/proxy-first broad search plus exact rerank; CPU exact-all is out of scope.",
        "- Promote only if the challenger improves a declared metric without breaking coverage,",
        "  min-group coverage, exact alpha01 pass, zero violation, and source/temporal caveats.",
        "",
        "## Generated tables",
        "",
        f"- `{TABLE_DIR / f'paper1_bound_improvement_pd_intake_{DATE_TAG}.csv'}`",
        f"- `{TABLE_DIR / f'paper1_bound_improvement_conformal_group_diagnostics_{DATE_TAG}.csv'}`",
        f"- `{TABLE_DIR / f'paper1_bound_improvement_conformal_config_candidates_{DATE_TAG}.csv'}`",
        f"- `{TABLE_DIR / f'paper1_bound_improvement_portfolio_quick_alpha01_{DATE_TAG}.csv'}`",
        f"- `{TABLE_DIR / f'paper1_bound_improvement_parent_smoke_alpha01_{DATE_TAG}.csv'}`",
        f"- `{TABLE_DIR / f'paper1_bound_improvement_theory_fronts_{DATE_TAG}.csv'}`",
        "",
        "## Bound fronts",
        "",
    ]
    for row in fronts.itertuples(index=False):
        lines.append(f"- `{row.front}`: {row.next_action} Gate: {row.promotion_gate}")

    path = DOC_DIR / f"paper1_bound_improvement_intake_{DATE_TAG}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT))
    args = parser.parse_args()

    root = Path(args.external_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    pd_table = _pd_intake(root)
    conformal_group, conformal_config = _conformal_intake(root)
    portfolio = _portfolio_intake(root)
    parent_smoke = _parent_smoke_intake()
    fronts = _bound_fronts()

    _write_csv(pd_table, TABLE_DIR / f"paper1_bound_improvement_pd_intake_{DATE_TAG}.csv")
    _write_csv(
        conformal_group,
        TABLE_DIR / f"paper1_bound_improvement_conformal_group_diagnostics_{DATE_TAG}.csv",
    )
    _write_csv(
        conformal_config,
        TABLE_DIR / f"paper1_bound_improvement_conformal_config_candidates_{DATE_TAG}.csv",
    )
    _write_csv(
        portfolio,
        TABLE_DIR / f"paper1_bound_improvement_portfolio_quick_alpha01_{DATE_TAG}.csv",
    )
    if not parent_smoke.empty:
        _write_csv(
            parent_smoke,
            TABLE_DIR / f"paper1_bound_improvement_parent_smoke_alpha01_{DATE_TAG}.csv",
        )
    _write_csv(
        fronts,
        TABLE_DIR / f"paper1_bound_improvement_theory_fronts_{DATE_TAG}.csv",
    )
    memo = _write_memo(
        root=root,
        pd_table=pd_table,
        conformal_group=conformal_group,
        conformal_config=conformal_config,
        portfolio=portfolio,
        parent_smoke=parent_smoke,
        fronts=fronts,
    )
    print(f"Wrote bound-improvement intake memo: {memo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
