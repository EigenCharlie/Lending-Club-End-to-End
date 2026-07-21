"""Run Paper 4 conformal candidate diagnostics from the Papers_tesis audit.

The script evaluates three bounded candidates over frozen Paper 4 v4 online
conformal replay artifacts:

1. group-weighted source replay;
2. utility-directed conformal replay;
3. localized score conformal replay.

The experiment is intentionally appendix-scoped. It uses a chronological split
inside the frozen replay table and does not retrain the CRPTO champion or create
any Paper 4 promotion artifact.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"

INPUT_INTERVALS = TABLE_DIR / "paper4_online_conformal_v4_intervals.parquet"
SUMMARY_PATH = TABLE_DIR / "paper4_papers_tesis_conformal_candidate_summary_2026-06-06.csv"
SOURCE_PATH = TABLE_DIR / "paper4_papers_tesis_conformal_candidate_by_source_2026-06-06.csv"
GATE_PATH = TABLE_DIR / "paper4_papers_tesis_conformal_candidate_gate_register_2026-06-06.csv"
STATUS_PATH = STATUS_DIR / "paper4_papers_tesis_conformal_candidates_status_2026-06-06.json"
NOTE_PATH = NOTE_DIR / "paper4_papers_tesis_conformal_candidates_2026-06-06.md"

ALPHA = 0.10
TARGET_COVERAGE = 1.0 - ALPHA
CALIBRATION_END = pd.Timestamp("2018-12-31")
SOURCE_MIN_SUPPORT = 100
SOURCE_FAMILIES = (
    "original_grade",
    "term",
    "score_decile",
    "income_band",
    "dti_band",
    "state_top20",
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finite_sample_quantile(scores: pd.Series | np.ndarray, alpha: float = ALPHA) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    level = min(1.0, np.ceil((values.size + 1) * (1.0 - alpha)) / values.size)
    return float(np.quantile(values, level, method="higher"))


def weighted_quantile(
    scores: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    alpha: float = ALPHA,
) -> float:
    values = np.asarray(scores, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weight_values) & (weight_values > 0)
    if not mask.any():
        return finite_sample_quantile(values, alpha=alpha)
    values = values[mask]
    weight_values = weight_values[mask]
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weight_values[order]
    cutoff = (1.0 - alpha) * sorted_weights.sum()
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")])


def _coverage_frame(eval_frame: pd.DataFrame, qhat: np.ndarray, variant: str) -> pd.DataFrame:
    q = np.clip(np.asarray(qhat, dtype=float), 0.0, 1.0)
    pred = pd.to_numeric(eval_frame["y_pred"], errors="coerce").to_numpy(dtype=float)
    low = np.clip(pred - q, 0.0, 1.0)
    high = np.clip(pred + q, 0.0, 1.0)
    y = pd.to_numeric(eval_frame["y_true"], errors="coerce").to_numpy(dtype=float)
    out = eval_frame.copy()
    out["variant"] = variant
    out["qhat"] = q
    out["pd_low_candidate"] = low
    out["pd_high_candidate"] = high
    out["covered_candidate"] = ((y >= low) & (y <= high)).astype(bool)
    out["interval_width_candidate"] = high - low
    return out


def _load_base_frame() -> pd.DataFrame:
    if not INPUT_INTERVALS.exists():
        raise FileNotFoundError(INPUT_INTERVALS)
    cols = [
        "loan_id",
        "issue_month",
        "period",
        "original_grade",
        "term",
        "score_decile",
        "state_top20",
        "income_band",
        "dti_band",
        "y_true",
        "y_pred",
        "score_abs",
        "online_method_v4",
        "qhat_v4",
        "pd_low_online_v4",
        "pd_high_online_v4",
        "covered_online_v4",
        "interval_width_online_v4",
    ]
    raw = pd.read_parquet(INPUT_INTERVALS, columns=cols)
    base = raw[raw["online_method_v4"].eq("source_aware_guarded")].copy()
    if base.empty:
        base = raw.drop_duplicates("loan_id").copy()
    base["issue_month"] = pd.to_datetime(base["issue_month"], errors="coerce")
    base["score_abs"] = pd.to_numeric(base["score_abs"], errors="coerce").fillna(
        (
            pd.to_numeric(base["y_true"], errors="coerce")
            - pd.to_numeric(base["y_pred"], errors="coerce")
        ).abs()
    )
    for col in SOURCE_FAMILIES:
        base[col] = base[col].fillna("UNKNOWN").astype(str)
    return base.dropna(subset=["issue_month", "y_true", "y_pred", "score_abs"]).reset_index(
        drop=True
    )


def _split_replay(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = base[base["issue_month"].le(CALIBRATION_END)].copy()
    holdout = base[base["issue_month"].gt(CALIBRATION_END)].copy()
    if calibration.empty or holdout.empty:
        raise ValueError("Expected non-empty chronological calibration and holdout frames")
    return calibration.reset_index(drop=True), holdout.reset_index(drop=True)


def _mondrian_grade(calibration: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    global_q = finite_sample_quantile(calibration["score_abs"])
    quantiles = (
        calibration.groupby("original_grade", observed=True)["score_abs"]
        .apply(lambda values: finite_sample_quantile(values) if len(values) >= 500 else global_q)
        .to_dict()
    )
    qhat = holdout["original_grade"].map(quantiles).fillna(global_q).to_numpy(dtype=float)
    return _coverage_frame(holdout, qhat, "mondrian_grade_temporal_replay")


def _group_weighted_source(calibration: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    scores = calibration["score_abs"].to_numpy(dtype=float)
    global_q = finite_sample_quantile(scores)
    family_value_q: dict[tuple[str, str], float] = {}
    for family in SOURCE_FAMILIES:
        for value in sorted(set(holdout[family].astype(str))):
            same = calibration[family].astype(str).eq(value).to_numpy(dtype=bool)
            if int(same.sum()) < SOURCE_MIN_SUPPORT:
                family_value_q[(family, value)] = global_q
                continue
            weights = np.ones(len(calibration), dtype=float)
            weights[same] = 4.0
            family_value_q[(family, value)] = weighted_quantile(scores, weights)

    per_family = []
    for family in SOURCE_FAMILIES:
        per_family.append(
            holdout[family]
            .astype(str)
            .map(
                lambda value, source_family=family: family_value_q.get(
                    (source_family, value), global_q
                )
            )
        )
    qhat = pd.concat(per_family, axis=1).max(axis=1).fillna(global_q).to_numpy(dtype=float)
    return _coverage_frame(holdout, qhat, "group_weighted_source_max_replay")


def _localized_score(calibration: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    cal_pred = pd.to_numeric(calibration["y_pred"], errors="coerce")
    bins = np.unique(np.quantile(cal_pred, np.linspace(0.0, 1.0, 21)))
    if bins.size < 3:
        qhat = np.repeat(finite_sample_quantile(calibration["score_abs"]), len(holdout))
        return _coverage_frame(holdout, qhat, "localized_score_replay")

    cal_bins = pd.cut(cal_pred, bins=bins, include_lowest=True, duplicates="drop")
    holdout_bins = pd.cut(
        pd.to_numeric(holdout["y_pred"], errors="coerce"),
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    )
    global_q = finite_sample_quantile(calibration["score_abs"])
    quantiles = (
        calibration.assign(_score_bin=cal_bins)
        .groupby("_score_bin", observed=True)["score_abs"]
        .apply(lambda values: finite_sample_quantile(values) if len(values) >= 500 else global_q)
        .to_dict()
    )
    qhat = pd.Series(holdout_bins).map(quantiles).fillna(global_q).to_numpy(dtype=float)
    return _coverage_frame(holdout, qhat, "localized_score_replay")


def _utility_directed(calibration: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    scores = pd.to_numeric(calibration["score_abs"], errors="coerce").to_numpy(dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0.80, 0.995, 80)))
    pred = pd.to_numeric(calibration["y_pred"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(calibration["y_true"], errors="coerce").to_numpy(dtype=float)
    best_q = finite_sample_quantile(scores)
    best_loss = float("inf")
    for q in candidates:
        low = np.clip(pred - q, 0.0, 1.0)
        high = np.clip(pred + q, 0.0, 1.0)
        covered = ((y >= low) & (y <= high)).astype(float)
        coverage = float(covered.mean())
        width = float((high - low).mean())
        tail_miss = float(((y == 1.0) & (high < 1.0)).mean())
        coverage_shortfall = max(0.0, TARGET_COVERAGE - coverage)
        loss = width + 8.0 * coverage_shortfall + 2.5 * tail_miss
        if loss < best_loss:
            best_loss = loss
            best_q = float(q)
    qhat = np.repeat(best_q, len(holdout))
    out = _coverage_frame(holdout, qhat, "utility_directed_loss_replay")
    out["calibration_utility_loss"] = best_loss
    return out


def _existing_v4_context(base: pd.DataFrame) -> pd.DataFrame:
    holdout = base[base["issue_month"].gt(CALIBRATION_END)].copy()
    context = holdout.copy()
    context["variant"] = "source_aware_guarded_v4_context"
    context["qhat"] = pd.to_numeric(context["qhat_v4"], errors="coerce")
    context["pd_low_candidate"] = pd.to_numeric(context["pd_low_online_v4"], errors="coerce")
    context["pd_high_candidate"] = pd.to_numeric(context["pd_high_online_v4"], errors="coerce")
    context["covered_candidate"] = context["covered_online_v4"].astype(bool)
    context["interval_width_candidate"] = pd.to_numeric(
        context["interval_width_online_v4"], errors="coerce"
    )
    return context


def _source_breakdown(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        variant = str(frame["variant"].iat[0])
        for family in SOURCE_FAMILIES:
            grouped = (
                frame.groupby(family, observed=True)
                .agg(
                    n=("covered_candidate", "size"),
                    coverage=("covered_candidate", "mean"),
                    avg_width=("interval_width_candidate", "mean"),
                )
                .reset_index()
                .rename(columns={family: "source_value"})
            )
            grouped["variant"] = variant
            grouped["source_family"] = family
            grouped["defended_cell"] = grouped["n"].ge(SOURCE_MIN_SUPPORT)
            rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows)[
        ["variant", "source_family", "source_value", "n", "defended_cell", "coverage", "avg_width"]
    ]


def _summary(frames: list[pd.DataFrame], source_breakdown: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame in frames:
        variant = str(frame["variant"].iat[0])
        defended = source_breakdown[
            source_breakdown["variant"].eq(variant) & source_breakdown["defended_cell"].astype(bool)
        ]
        coverage = float(frame["covered_candidate"].mean())
        avg_width = float(frame["interval_width_candidate"].mean())
        p95_width = float(frame["interval_width_candidate"].quantile(0.95))
        worst_source = float(defended["coverage"].min()) if not defended.empty else float("nan")
        absolute_gate = bool(
            coverage >= TARGET_COVERAGE and worst_source >= 0.80 and avg_width <= 0.98
        )
        rows.append(
            {
                "variant": variant,
                "n_holdout": int(len(frame)),
                "coverage": coverage,
                "target_coverage": TARGET_COVERAGE,
                "coverage_gap": coverage - TARGET_COVERAGE,
                "avg_width": avg_width,
                "p95_width": p95_width,
                "worst_defended_source_coverage": worst_source,
                "passes_absolute_appendix_gate": absolute_gate,
            }
        )
    summary = pd.DataFrame(rows)
    mondrian = summary[summary["variant"].eq("mondrian_grade_temporal_replay")].iloc[0]
    mondrian_width = float(mondrian["avg_width"])
    mondrian_coverage = float(mondrian["coverage"])
    mondrian_worst_source = float(mondrian["worst_defended_source_coverage"])
    summary["avg_width_delta_vs_mondrian"] = summary["avg_width"] - mondrian_width
    summary["coverage_delta_vs_mondrian"] = summary["coverage"] - mondrian_coverage
    summary["worst_source_delta_vs_mondrian"] = (
        summary["worst_defended_source_coverage"] - mondrian_worst_source
    )
    summary["claim_decision"] = summary.apply(_claim_decision, axis=1)
    return summary.sort_values(
        [
            "passes_absolute_appendix_gate",
            "coverage",
            "worst_defended_source_coverage",
            "avg_width",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _claim_decision(row: pd.Series) -> str:
    variant = str(row["variant"])
    if not bool(row["passes_absolute_appendix_gate"]):
        return "park_gate_fail"
    if variant == "source_aware_guarded_v4_context":
        return "append_context_reference"
    if variant == "mondrian_grade_temporal_replay":
        return "retain_baseline"
    if variant == "group_weighted_source_max_replay":
        if (
            float(row["worst_source_delta_vs_mondrian"]) > 0
            and float(row["coverage_delta_vs_mondrian"]) > 0
        ):
            return "append_positive_diagnostic"
        return "append_mixed_diagnostic_source_not_better"
    if variant == "localized_score_replay":
        if (
            float(row["coverage_delta_vs_mondrian"]) > 0
            and float(row["worst_source_delta_vs_mondrian"]) > 0
            and float(row["avg_width_delta_vs_mondrian"]) <= 0
        ):
            return "append_positive_diagnostic"
        return "append_mixed_diagnostic_wider_than_mondrian"
    if variant == "utility_directed_loss_replay":
        if float(row["avg_width_delta_vs_mondrian"]) <= 0:
            return "append_positive_diagnostic"
        return "park_width_fail"
    return "append_diagnostic"


def _gate_register(summary: pd.DataFrame) -> pd.DataFrame:
    lookup = summary.set_index("variant")
    rows = [
        {
            "paper": "Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction",
            "candidate_variant": "group_weighted_source_max_replay",
            "claim_target": "Improve source-family holdout coverage versus Mondrian without exceeding avg width 0.98.",
            "evidence_gate": "coverage>=0.90, worst defended source coverage>=0.80, avg_width<=0.98 on 2019-2020 holdout.",
            "artifact_sink": SOURCE_PATH.name,
            "stop_rule": "Append diagnostic only; no online/source deployment claim and no CRPTO champion change.",
        },
        {
            "paper": "Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction",
            "candidate_variant": "utility_directed_loss_replay",
            "claim_target": "Check whether a fixed width/violation utility loss can reduce width while retaining coverage.",
            "evidence_gate": "coverage>=0.90, worst defended source coverage>=0.80, avg_width lower than Mondrian.",
            "artifact_sink": SUMMARY_PATH.name,
            "stop_rule": "Park if it does not beat Mondrian width at the same empirical coverage gate.",
        },
        {
            "paper": "Guan 2023 - Localized Conformal Prediction",
            "candidate_variant": "localized_score_replay",
            "claim_target": "Test local score bins as a challenger to grade Mondrian calibration.",
            "evidence_gate": "coverage>=0.90, worst defended source coverage>=0.80, avg_width lower than Mondrian.",
            "artifact_sink": SOURCE_PATH.name,
            "stop_rule": "Append only as reviewer-facing diagnostic; no localized guarantee claim.",
        },
    ]
    for row in rows:
        variant = row["candidate_variant"]
        if variant in lookup.index:
            result = lookup.loc[variant]
            row.update(
                {
                    "observed_coverage": float(result["coverage"]),
                    "observed_avg_width": float(result["avg_width"]),
                    "observed_worst_defended_source_coverage": float(
                        result["worst_defended_source_coverage"]
                    ),
                    "coverage_delta_vs_mondrian": float(result["coverage_delta_vs_mondrian"]),
                    "avg_width_delta_vs_mondrian": float(result["avg_width_delta_vs_mondrian"]),
                    "worst_source_delta_vs_mondrian": float(
                        result["worst_source_delta_vs_mondrian"]
                    ),
                    "passes_absolute_appendix_gate": bool(result["passes_absolute_appendix_gate"]),
                    "decision": str(result["claim_decision"]),
                }
            )
    return pd.DataFrame(rows)


def _note(summary: pd.DataFrame, gate_register: pd.DataFrame) -> str:
    best = summary.iloc[0]
    rows = "\n".join(
        f"| {row.variant} | {row.coverage:.4f} | {row.avg_width:.4f} | "
        f"{row.worst_defended_source_coverage:.4f} | "
        f"{row.avg_width_delta_vs_mondrian:.4f} | {row.claim_decision} |"
        for row in summary.itertuples()
    )
    gates = "\n".join(
        f"| {row.paper} | {row.candidate_variant} | {row.decision} | "
        f"{bool(row.passes_absolute_appendix_gate)} |"
        for row in gate_register.itertuples()
    )
    return f"""# Paper 4 Papers_tesis conformal candidate diagnostics - 2026-06-06

## Protocol

- **Claim target:** test whether group-weighted, utility-directed or localized
  conformal candidates can improve Paper 4 source/coverage diagnostics.
- **Split:** frozen v4 replay table, 2018 rows for calibration and 2019--2020
  rows for holdout.
- **Gate:** holdout coverage >= 0.90, defended-source coverage >= 0.80 and
  average interval width <= 0.98.
- **Stop rule:** append diagnostic evidence only. Do not modify the Paper
  Estrella champion and do not make online deployment, legal fairness or exact
  conditional-coverage claims.

## Summary

| variant | coverage | avg_width | worst defended source coverage | width delta vs Mondrian | decision |
| --- | ---: | ---: | ---: | ---: | --- |
{rows}

Best diagnostic row by the predeclared sort is `{best["variant"]}`. The result
is appendix-scoped: it can support a Paper 4 source/shift conformal discussion,
but it does not replace the CRPTO champion protocol.

## Gate Register

| paper | candidate | decision | absolute gate pass |
| --- | --- | --- | --- |
{gates}
"""


def main() -> None:
    base = _load_base_frame()
    calibration, holdout = _split_replay(base)
    frames = [
        _existing_v4_context(base),
        _mondrian_grade(calibration, holdout),
        _group_weighted_source(calibration, holdout),
        _localized_score(calibration, holdout),
        _utility_directed(calibration, holdout),
    ]
    source_breakdown = _source_breakdown(frames)
    summary = _summary(frames, source_breakdown)
    gate_register = _gate_register(summary)

    _write_csv(SUMMARY_PATH, summary)
    _write_csv(SOURCE_PATH, source_breakdown)
    _write_csv(GATE_PATH, gate_register)
    _write_json(
        STATUS_PATH,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "schema_version": "2026-06-06.papers_tesis_candidates.v1",
            "input_artifact": str(INPUT_INTERVALS.relative_to(ROOT)),
            "calibration_end": CALIBRATION_END.date().isoformat(),
            "n_calibration": int(len(calibration)),
            "n_holdout": int(len(holdout)),
            "target_coverage": TARGET_COVERAGE,
            "source_min_support": SOURCE_MIN_SUPPORT,
            "outputs": {
                "summary": str(SUMMARY_PATH.relative_to(ROOT)),
                "by_source": str(SOURCE_PATH.relative_to(ROOT)),
                "gate_register": str(GATE_PATH.relative_to(ROOT)),
                "note": str(NOTE_PATH.relative_to(ROOT)),
            },
            "champion_modified": False,
            "paper4_final_promotion_created": False,
            "claim_boundary": "diagnostic appendix only; no source deployment or champion replacement claim",
        },
    )
    _write_note(NOTE_PATH, _note(summary, gate_register))
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {SOURCE_PATH}")
    print(f"Wrote {GATE_PATH}")
    print(f"Wrote {STATUS_PATH}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
