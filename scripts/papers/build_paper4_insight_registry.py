"""Build Paper 4 insight-factory integration registry.

This script is intentionally lightweight: it does not re-run research lanes or
promote any champion. It reads existing research-only and paper-grade artifacts
and materializes a small set of tables explaining how each insight can connect
to the Paper 4 end-to-end architecture.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = OUT_ROOT / "tables"
STATUS_DIR = OUT_ROOT / "status"
SCHEMA_VERSION = "2026-05-09.1"


def _load_json(path: str) -> dict[str, Any]:
    full_path = ROOT / path
    if not full_path.exists():
        return {}
    try:
        return json.loads(full_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_parquet(path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(full_path)
    except Exception:
        return pd.DataFrame()


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "N/D"


def _fmt_number(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "N/D"


def _fmt_money(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return "N/D"
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:,.{digits}f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:,.{digits}f}M"
    return f"${number / 1_000:,.{digits}f}K"


def _write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_signal_registry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    promotion = _load_json("models/final_project_promotion.json")
    uncertainty = _load_json("models/uncertainty_baselines_status.json")
    bma = _load_json("models/bma_comparison_status.json")
    sicr = _load_json("models/sicr_conformal_status.json")
    cif = _load_json("models/cif_ecl_impact_status.json")
    stage = _load_json("models/stage_misclassification_status.json")
    ts = _load_json("models/time_series_vnext_status.json")
    ts_ecl = _load_json("models/ts_ecl_intervals_status.json")
    causal_rule = _load_json("models/causal_policy_rule.json")
    cate_portfolio = _load_json("models/cate_portfolio_status.json")
    spo = _load_json("models/spo_real_training_status.json")
    stability = _load_json("data/processed/crpto_vs_spo_stability.json")
    gpu = _load_json("reports/gpu_insights/2026-03-10-rapids-insight-v4/artifacts/run_summary.json")

    final_champion = promotion.get("final_champion", {})
    bma_equal = bma.get("results", {}).get("equal", {})
    uncertainty_cp = uncertainty.get("results", {}).get("conformal_mondrian", {})
    ts_interval = ts.get("interval_champion", {})
    spo_results = spo.get("results", {})
    gpu_stages = {row.get("stage"): row for row in gpu.get("stages", [])}
    stability_periods = stability.get("per_period", {})
    min_stability_coverage = None
    if stability_periods:
        min_stability_coverage = min(
            float(row.get("coverage_90", 0.0)) for row in stability_periods.values()
        )

    rows = [
        {
            "signal": "CRPTO champion and robust region",
            "source_lane": "paper1_e2e / search_portfolio",
            "artifact": "models/final_project_promotion.json",
            "current_result": (
                f"return {_fmt_money(final_champion.get('realized_total_return'))}; "
                f"alpha01={final_champion.get('alpha01_exact_pass')}; "
                f"V={final_champion.get('alpha01_weighted_miscoverage_V', 'N/D')}; "
                f"Gamma_CP={final_champion.get('alpha01_gamma_cp', 'N/D')}"
            ),
            "paper4_role": "base policy universe",
            "end_to_end_connection": "starting funded-set and robust-policy candidates for all overlays",
            "promotion_gate": "must not be overwritten; Paper 4 needs its own promotion JSON",
            "immediate_action": "use as source manifest and compare overlays against frozen champion",
        },
        {
            "signal": "Uncertainty baselines and BMA",
            "source_lane": "insights_factory canonical/research",
            "artifact": "models/uncertainty_baselines_status.json; models/bma_comparison_status.json",
            "current_result": (
                f"CP coverage {_fmt_pct(uncertainty_cp.get('empirical_coverage'))}, "
                f"width {uncertainty_cp.get('avg_width', 'N/D')}; "
                f"BMA width reduction by CP {bma_equal.get('width_reduction_cp_vs_bma_pct', 'N/D')}%"
            ),
            "paper4_role": "uncertainty primitive justification",
            "end_to_end_connection": "chooses conformal intervals as the uncertainty source for IFRS9/CRPTO overlays",
            "promotion_gate": "diagnostic only unless a new conformal selector is opened",
            "immediate_action": "keep as evidence that Paper 4 should use conformal, not bootstrap/BMA, as base U_t",
        },
        {
            "signal": "Alpha and Gamma policy dial",
            "source_lane": "insights_factory research",
            "artifact": "data/processed/alpha_sweep_pareto_both.parquet; data/processed/ecl_alpha_sensitivity.parquet",
            "current_result": (
                f"SICR alpha ECL at 0.10 {_fmt_money(sicr.get('part_b', {}).get('ecl_at_alpha_010'))}; "
                f"alpha 0.05 {_fmt_money(sicr.get('part_b', {}).get('ecl_at_alpha_005'))}"
            ),
            "paper4_role": "policy dial",
            "end_to_end_connection": "propagates alpha into Gamma_CP, eligible funded set, SICR, ECL and capital overlays",
            "promotion_gate": "selector must declare alpha policy before ranking policies",
            "immediate_action": "add alpha -> Gamma_CP -> ECL -> funded-set table to the MVP package",
        },
        {
            "signal": "SICR and stage-cost overlays",
            "source_lane": "paper2_e2e / insights_factory research",
            "artifact": "models/sicr_conformal_status.json; models/stage_misclassification_status.json",
            "current_result": (
                f"SICR width threshold {sicr.get('part_a', {}).get('optimal_width_threshold', 'N/D')}; "
                f"stage cost {stage.get('part_b_combined', {}).get('total_cost_M', 'N/D')}M"
            ),
            "paper4_role": "IFRS9 accounting constraint",
            "end_to_end_connection": "turns conformal width into Stage 2 migration and provision cost for each policy",
            "promotion_gate": "must be evaluated policy-by-policy before it becomes a selector",
            "immediate_action": "build policy x ECL x stage grid over CRPTO robust region",
        },
        {
            "signal": "Competing-risk ECL correction",
            "source_lane": "paper2_e2e / insights_factory research",
            "artifact": "models/cif_ecl_impact_status.json",
            "current_result": (
                f"KM excess reserve {_fmt_money(cif.get('ecl_impact', {}).get('total_excess_reserve'))}; "
                f"excess {cif.get('ecl_impact', {}).get('excess_reserve_pct', 'N/D')}%"
            ),
            "paper4_role": "lifetime-risk correction",
            "end_to_end_connection": "prevents multi-period IFRS9 CRPTO from overcounting prepayment as censoring",
            "promotion_gate": "derive per-policy lifetime ECL before changing objective",
            "immediate_action": "add CIF-adjusted ECL as a diagnostic column in Paper 4 MVP",
        },
        {
            "signal": "TS -> ECL uncertainty and vNext intervals",
            "source_lane": "core_ts / research_only",
            "artifact": "models/time_series_vnext_status.json; models/ts_ecl_intervals_status.json",
            "current_result": (
                f"interval promotable={ts_interval.get('promotable')}; "
                f"coverage90={_fmt_pct(ts_interval.get('coverage_90'))}; "
                f"ECL band {_fmt_money(ts_ecl.get('ts_forecast', {}).get('ecl_band_width_90'))}"
            ),
            "paper4_role": "temporal state, not current selector",
            "end_to_end_connection": "feeds monthly state S_t, ECL stress and future online conformal gates",
            "promotion_gate": "coverage by horizon and forward coherence before online/multi-period claim",
            "immediate_action": "use TS as replay/stress state; keep interval champion research-only",
        },
        {
            "signal": "SPO+ and CRPTO stability",
            "source_lane": "insights_factory research",
            "artifact": "models/spo_real_training_status.json; data/processed/crpto_vs_spo_stability.json",
            "current_result": (
                f"SPO+ regret improvement {_fmt_number(spo_results.get('spo_improvement_vs_ts_pct'))}%; "
                f"min period coverage {_fmt_pct(min_stability_coverage)}"
            ),
            "paper4_role": "DFL/regret comparator",
            "end_to_end_connection": "benchmarks decision regret against auditability, coverage and robust feasibility",
            "promotion_gate": "online DFL needs repeated-decision coverage and IFRS9 net-return metrics",
            "immediate_action": "frame as regret-auditability frontier, not as replacement for CRPTO",
        },
        {
            "signal": "Causal rule and CATE portfolio gate",
            "source_lane": "research_causal / research_cate_portfolio",
            "artifact": "models/causal_policy_rule.json; models/cate_portfolio_status.json",
            "current_result": (
                f"rule {causal_rule.get('selected_rule', 'N/D')}; "
                f"overlap={causal_rule.get('overlap_pass')}; "
                f"sensitivity={causal_rule.get('sensitivity_pass')}; "
                f"CATE portfolio={cate_portfolio.get('promotion_state', 'N/D')}"
            ),
            "paper4_role": "intervention candidate, currently blocked",
            "end_to_end_connection": "could add policy_value_causal only after identification and sensitivity gates pass",
            "promotion_gate": "causal proof package plus policy_evaluation_consistent=true",
            "immediate_action": "include as explicit gate and future module; do not use as objective today",
        },
        {
            "signal": "Causal OOT economic signal",
            "source_lane": "research_causal",
            "artifact": "models/causal_policy_oot_status.json; data/processed/causal_policy_oot_backtest.parquet",
            "current_result": (
                f"rule net {_fmt_money(_load_json('models/causal_policy_oot_status.json').get('total_net_value'))}; "
                f"p05 monthly {_fmt_money(_load_json('models/causal_policy_oot_status.json').get('p05_monthly_net'), 3)}"
            ),
            "paper4_role": "causal stress-test evidence",
            "end_to_end_connection": "shows what a pricing/intervention layer would report once causal gates pass",
            "promotion_gate": "same causal gate; OOT economic value alone is insufficient",
            "immediate_action": "show as example of future causal readout, separated from champion selection",
        },
        {
            "signal": "RAPIDS insight factory segmentation and graph structure",
            "source_lane": "research_rapids",
            "artifact": "reports/gpu_insights/2026-03-10-rapids-insight-v4/artifacts/*",
            "current_result": (
                f"UMAP {_fmt_number(gpu_stages.get('cuml_umap', {}).get('speedup_gpu_vs_cpu'))}x; "
                f"HDBSCAN clusters {gpu_stages.get('cuml_hdbscan', {}).get('gpu_clusters', 'N/D')}; "
                f"Louvain communities {gpu_stages.get('cugraph_similarity', {}).get('gpu_louvain_communities', 'N/D')}"
            ),
            "paper4_role": "candidate regime/source discovery",
            "end_to_end_connection": "can propose segments for MDCP, fairness stress, drift monitoring and scenario slicing",
            "promotion_gate": "segments need stability, interpretability and policy impact before use",
            "immediate_action": "turn clusters/communities into candidate source labels for diagnostics only",
        },
        {
            "signal": "Notebook/storytelling atlas",
            "source_lane": "research_notebooks",
            "artifact": "reports/notebook_exec; reports/notebook_images; export_storytelling_snapshot",
            "current_result": "figures and exploratory narratives already feed Quarto/companion",
            "paper4_role": "evidence atlas",
            "end_to_end_connection": "keeps exploratory evidence linked to claim/artifact/test maps",
            "promotion_gate": "not a metric source unless backed by generated artifact",
            "immediate_action": "use for communication and appendix curation, not selection",
        },
    ]

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_signals": len(rows),
        "signals_ready_for_mvp": [
            "CRPTO champion and robust region",
            "Uncertainty baselines and BMA",
            "Alpha and Gamma policy dial",
            "SICR and stage-cost overlays",
            "Competing-risk ECL correction",
            "TS -> ECL uncertainty and vNext intervals",
            "SPO+ and CRPTO stability",
        ],
        "signals_diagnostic_or_blocked": [
            "Causal rule and CATE portfolio gate",
            "Causal OOT economic signal",
            "RAPIDS insight factory segmentation and graph structure",
            "Notebook/storytelling atlas",
        ],
        "primary_rule": (
            "Insights Factory can feed Paper 4 only when each insight is mapped to "
            "a policy variable, gate, state component or validation slice."
        ),
    }
    return rows, status


def build_readiness_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    readiness_rows = [
        {
            "block": "Uncertainty / conformal baselines",
            "usable_now_as": "base uncertainty justification",
            "requires_new_run": "no",
            "not_ready_for": "replacing score_decile_mondrian without selector",
            "paper4_next_artifact": "paper4_uncertainty_source_manifest.csv",
        },
        {
            "block": "IFRS9 advanced overlays",
            "usable_now_as": "ECL/stage/tail diagnostics",
            "requires_new_run": "derived policy grid",
            "not_ready_for": "new IFRS9-aware champion",
            "paper4_next_artifact": "policy_ifrs9_ecl_grid.parquet",
        },
        {
            "block": "Temporal / TS",
            "usable_now_as": "monthly replay and stress state",
            "requires_new_run": "yes for online conformal",
            "not_ready_for": "forward interval promotion",
            "paper4_next_artifact": "monthly_policy_replay.parquet",
        },
        {
            "block": "SPO+ / DFL",
            "usable_now_as": "regret-auditability comparator",
            "requires_new_run": "yes for online DFL",
            "not_ready_for": "replacing CRPTO champion",
            "paper4_next_artifact": "paper4_regret_auditability_frontier.csv",
        },
        {
            "block": "Causal / CATE",
            "usable_now_as": "intervention hypothesis and blocked gate",
            "requires_new_run": "yes",
            "not_ready_for": "central objective or champion selector",
            "paper4_next_artifact": "causal_identification_report.md",
        },
        {
            "block": "RAPIDS insight factory",
            "usable_now_as": "scale and candidate segment discovery",
            "requires_new_run": "only if segments enter gates",
            "not_ready_for": "automatic MDCP/fairness source labels",
            "paper4_next_artifact": "paper4_candidate_regime_labels.parquet",
        },
        {
            "block": "Notebook/storytelling outputs",
            "usable_now_as": "appendix and communication atlas",
            "requires_new_run": "no",
            "not_ready_for": "metric source without artifact",
            "paper4_next_artifact": "paper4_evidence_atlas.csv",
        },
    ]

    join_rows = [
        {
            "join_key": "loan_id / id",
            "connects": "PD, conformal interval, funded decision, ECL, CATE, fairness slices",
            "input_artifacts": "final_project_summary, conformal_intervals, cate_estimates_oot, IFRS9 tables",
            "paper4_output": "paper4_loan_level_policy_evidence.parquet",
            "status": "viable with existing artifacts",
        },
        {
            "join_key": "policy_id / policy parameters",
            "connects": "CRPTO robust region, ECL overlay, tail risk, satisficing, selector",
            "input_artifacts": "final_project_promotion, robust-region tables, A12/A13 artifacts",
            "paper4_output": "paper4_policy_level_evidence.parquet",
            "status": "needs derived policy grid",
        },
        {
            "join_key": "month / period",
            "connects": "TS state, ECL replay, coverage replay, causal OOT policy value",
            "input_artifacts": "time_series_vnext, ts_ecl_intervals, causal_policy_oot_backtest",
            "paper4_output": "paper4_monthly_replay.parquet",
            "status": "diagnostic now; online later",
        },
        {
            "join_key": "segment / source label",
            "connects": "Mondrian groups, fairness, MDCP, RAPIDS clusters, causal heterogeneity",
            "input_artifacts": "grade/score decile, fairness audit, gpu insight cluster/community profiles",
            "paper4_output": "paper4_segment_source_registry.parquet",
            "status": "candidate labels need stability gates",
        },
        {
            "join_key": "scenario",
            "connects": "baseline/adverse/severe IFRS9, CVaR/OCE, capital/provision limits",
            "input_artifacts": "ifrs9_scenario_summary, ecl_alpha_sensitivity, ts_ecl_intervals",
            "paper4_output": "paper4_scenario_policy_grid.parquet",
            "status": "viable as MVP diagnostic",
        },
    ]
    return readiness_rows, join_rows


def main() -> int:
    signal_rows, status = build_signal_registry()
    readiness_rows, join_rows = build_readiness_tables()

    signal_path = _write_csv("paper4_table6_insights_factory_signal_registry.csv", signal_rows)
    readiness_path = _write_csv("paper4_table7_research_lane_readiness.csv", readiness_rows)
    join_path = _write_csv("paper4_table8_insight_to_policy_join_contract.csv", join_rows)
    status_path = _write_json(
        "paper4_insights_factory_status.json",
        {
            **status,
            "tables": {
                "signal_registry": str(signal_path.relative_to(ROOT)),
                "research_lane_readiness": str(readiness_path.relative_to(ROOT)),
                "join_contract": str(join_path.relative_to(ROOT)),
            },
        },
    )
    print(f"Wrote {signal_path.relative_to(ROOT)}")
    print(f"Wrote {readiness_path.relative_to(ROOT)}")
    print(f"Wrote {join_path.relative_to(ROOT)}")
    print(f"Wrote {status_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
