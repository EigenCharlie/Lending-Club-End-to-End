"""Build Paper 4 v37 CATE and fairness/source-governance protocol artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime

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

SCHEMA_VERSION = "2026-05-15.37"


def _causal_protocol() -> pd.DataFrame:
    gates = [
        (
            "treatment_definition",
            "high_rate_within_grade or better-defined loan term treatment",
            "diagnostic_available",
            False,
        ),
        ("primary_outcome", "realized default/loss", "available_proxy", False),
        (
            "accepted_loan_selection",
            "reject inference and application selection",
            "theory_blocked",
            False,
        ),
        ("overlap", "propensity/feature overlap after trimming", "needs_stronger_evidence", False),
        ("balance", "SMD <= 0.10 after trimming/IPW/DR", "partially_passed_prior", False),
        (
            "placebo_falsification",
            "placebo outcomes and pre-treatment tests",
            "diagnostic_available",
            False,
        ),
        (
            "hidden_bias_sensitivity",
            "robustness to unobserved confounding",
            "not_stable_enough",
            False,
        ),
        ("cate_intervals", "useful intervals for policy value", "blocked", False),
    ]
    return pd.DataFrame(
        gates,
        columns=["gate", "requirement_v37", "status_v37", "cate_policy_value_allowed"],
    ).assign(claim_boundary_v37="accepted-loan causal diagnostic only; no policy-value claim")


def _cate_gate_reaudit() -> pd.DataFrame:
    gate = _safe_read_csv(TABLE_DIR / "paper4_v29_cate_gate_report.csv")
    if gate.empty:
        return _causal_protocol()
    out = gate.copy()
    out["status_v37"] = out.get("status_v29", out.get("status_v25", "diagnostic_only"))
    out["cate_policy_value_allowed"] = False
    out["accepted_loan_selection_limit_v37"] = "reject inference unresolved"
    out["claim_boundary_v37"] = (
        "CATE policy value blocked unless identification, overlap, sensitivity, falsification and intervals pass"
    )
    return out


def _fairness_protocol() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = _safe_read_csv(TABLE_DIR / "paper4_v29_source_governance_diagnostics.csv")
    no_claim = _safe_read_csv(TABLE_DIR / "paper4_v29_no_legal_claim_flags.csv")
    if source.empty:
        source = pd.DataFrame()
    else:
        source = source.copy()
        source["version_v37"] = "source_governance_appendix_v3"
        if "loans" in source:
            source["support_band_v37"] = pd.cut(
                pd.to_numeric(source["loans"], errors="coerce").fillna(0),
                bins=[-1, 24, 49, 99, np.inf],
                labels=["small_pool_to_parent", "monitor", "standalone", "high_support"],
            ).astype(str)
        else:
            source["support_band_v37"] = "unknown"
        source["fair_lending_legal_claim_allowed"] = False
        source["claim_boundary_v37"] = (
            "observable source governance only; protected attributes are not inferred"
        )
    protocol = pd.DataFrame(
        [
            {
                "protocol_item": "protected_attribute_data",
                "status_v37": "missing",
                "allowed_claim": False,
                "decision": "do_not_infer_protected_attributes",
            },
            {
                "protocol_item": "external_proxy_protocol",
                "status_v37": "not_approved",
                "allowed_claim": False,
                "decision": "source governance only",
            },
            {
                "protocol_item": "observable_source_caps",
                "status_v37": "implemented_as_governance_diagnostic",
                "allowed_claim": True,
                "decision": "use grade/month/period/state/income/DTI/score support-aware sources",
            },
            {
                "protocol_item": "fair_lending_legal_claim",
                "status_v37": "prohibited_claim",
                "allowed_claim": False,
                "decision": "no legal claim without data and approved protocol",
            },
        ]
    )
    flags = no_claim.copy()
    if flags.empty:
        flags = pd.DataFrame(
            [
                {
                    "claim_or_requirement": "fair_lending_legal_claim",
                    "allowed_v37": False,
                    "status_v37": "prohibited_claim",
                },
                {
                    "claim_or_requirement": "cate_policy_value",
                    "allowed_v37": False,
                    "status_v37": "theory_blocked",
                },
            ]
        )
    else:
        flags["allowed_v37"] = False
        flags.loc[
            ~flags["claim_or_requirement"]
            .astype(str)
            .str.contains("fair_lending|cate", case=False, na=False),
            "allowed_v37",
        ] = flags.get("allowed_v29", False)
        flags["status_v37"] = np.where(
            flags["allowed_v37"].astype(bool), "allowed_proxy_claim", "blocked_or_prohibited"
        )
    return protocol, source, flags


def build_v37() -> dict:
    start = time.time()
    causal_protocol = _causal_protocol()
    cate_gate = _cate_gate_reaudit()
    fairness_protocol, source, flags = _fairness_protocol()
    _write_csv("paper4_v37_causal_identification_protocol.csv", causal_protocol)
    _write_csv("paper4_v37_cate_gate_report.csv", cate_gate)
    _write_csv("paper4_v37_fairness_proxy_only_protocol.csv", fairness_protocol)
    _write_csv("paper4_v37_source_governance_appendix.csv", source)
    _write_csv("paper4_v37_no_legal_claim_flags.csv", flags)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v37_cate_fairness_protocol",
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "causal_protocol_rows_v37": int(len(causal_protocol)),
        "source_governance_rows_v37": int(len(source)),
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "claim_boundary": "CATE/fairness remain governed diagnostics; no policy-value or fair-lending legal claim",
        "runtime_seconds": round(time.time() - start, 3),
    }
    _write_json("paper4_v37_status.json", status)
    _write_note(
        "paper4_v37_cate_fairness_protocol.md",
        "\n".join(
            [
                "# Paper 4 v37 CATE and Fairness Protocol",
                "",
                "- CATE policy value remains blocked.",
                "- Fair-lending legal claim remains prohibited.",
                "- Source governance remains available as observable proxy governance only.",
            ]
        ),
    )
    print(pd.Series(status).to_json(indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    build_v37()


if __name__ == "__main__":
    main()
