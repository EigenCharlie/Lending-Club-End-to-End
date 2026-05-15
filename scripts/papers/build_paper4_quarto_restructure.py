#!/usr/bin/env python3
"""Build the curated Paper 4 Quarto surface and living-lab notebook.

This script intentionally separates the official Paper 4 chapter from the
historical implementation waves. The Quarto chapter stays compact and
artifact-backed; the markdown notebook keeps the lab memory.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER4_ROOT = ROOT / "reports" / "paper_material" / "paper4"
TABLE_DIR = PAPER4_ROOT / "tables"
STATUS_DIR = PAPER4_ROOT / "status"
NOTE_DIR = PAPER4_ROOT / "notes"
GLOBAL_DIR = ROOT / "reports" / "paper_material" / "global"
BOOK_DIR = ROOT / "book" / "chapters" / "19-paper-mega-extension"

OFFICIAL_PAGES = [
    "index.qmd",
    "19a-proposal-and-scope.qmd",
    "19b-current-assets-and-gaps.qmd",
    "19c-integrated-architecture.qmd",
    "19f-sequential-decision-framework.qmd",
    "19h-mvp-evidence-pack.qmd",
    "19i-regret-auditability-frontier.qmd",
    "19n-online-mdcp-fairness.qmd",
    "19t-multi-period-solver.qmd",
    "19ca-v38-final-synthesis.qmd",
]


def read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def md_table(
    rows: Iterable[dict[str, object]], columns: list[str], max_rows: int | None = None
) -> str:
    rows = list(rows)
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return "\n_No rows available._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, "")).replace("\n", " ").replace("|", "\\|")
            values.append(value)
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def build_findings() -> list[dict[str, object]]:
    v31 = read_json(STATUS_DIR / "paper4_v31_status.json")
    v35 = read_json(STATUS_DIR / "paper4_v35_status.json")
    v38 = read_json(GLOBAL_DIR / "status" / "global_v38_status.json")
    cvar = first_row(TABLE_DIR / "paper4_v31_champion_vs_cvar_stress_memo.csv")

    return [
        {
            "finding_id": "F01",
            "finding": "Paper 4 is now a reproducible sequential-decision living lab, not a final promotion protocol.",
            "evidence_artifact": "reports/paper_material/global/status/global_v38_status.json",
            "official_claim": "The lab can compare policies under governed claim boundaries.",
            "claim_boundary": "Working/lab evidence only; no Paper Estrella promotion from Paper 4.",
            "quarto_page": "index.qmd",
            "status": "official",
        },
        {
            "finding_id": "F02",
            "finding": "Paper Estrella remains the official project champion after the v31-v38 audit.",
            "evidence_artifact": "reports/paper_material/global/tables/global_v38_promotion_decisions.csv",
            "official_claim": "No challenger currently justifies Paper Estrella promotion.",
            "claim_boundary": "Promotion requires named candidate, gate pass, artifact sync, and tests.",
            "quarto_page": "19ca-v38-final-synthesis.qmd",
            "status": "official",
        },
        {
            "finding_id": "F03",
            "finding": f"Dynamic stress replay reached {v31.get('dynamic_path_count_v31', 'NA')} common paths and {v31.get('dynamic_trace_rows_v31', 'NA')} trace rows.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v31_dynamic_policy_trace.parquet",
            "official_claim": "Serious candidates can be compared as monthly processes under common random numbers.",
            "claim_boundary": "Internal replay, not external forecast or live deployment.",
            "quarto_page": "19t-multi-period-solver.qmd",
            "status": "official",
        },
        {
            "finding_id": "F04",
            "finding": f"The strongest CVaR challenger reduced tail loss but did not beat paired wealth robustly; prob beat reference = {cvar.get('prob_challenger_beats_reference', 'NA')}.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v31_champion_vs_cvar_stress_memo.csv",
            "official_claim": "CVaR/OCE is a serious tail-risk challenger and committee profile, not the current economic champion.",
            "claim_boundary": "No exact full-universe CVaR optimality claim.",
            "quarto_page": "19i-regret-auditability-frontier.qmd",
            "status": "official",
        },
        {
            "finding_id": "F05",
            "finding": f"Online conformal passed the nominal v10 gate but v35 strict holdouts did not survive universally; online_gate_survives_v35 = {v35.get('online_gate_survives_v35', 'NA')}.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v35_online_temporal_holdout.csv",
            "official_claim": "Online coverage is promising but not universally robust by source-family holdout.",
            "claim_boundary": "Selection-replay/holdout evidence only; not live deployment.",
            "quarto_page": "19n-online-mdcp-fairness.qmd",
            "status": "official_with_caveat",
        },
        {
            "finding_id": "F06",
            "finding": "SPO/DFL has an oracle-regret/surrogate lane, while differentiable SPO remains dependency-blocked.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v32_spo_dependency_blockers.csv",
            "official_claim": "Decision-loss comparison can be studied through oracle regret today.",
            "claim_boundary": "No formal differentiable SPO+ claim until cvxpy/cvxpylayers/torch path is implemented and validated.",
            "quarto_page": "19t-multi-period-solver.qmd",
            "status": "dependency_blocked",
        },
        {
            "finding_id": "F07",
            "finding": "DLA/ADP has a formal state/transition framing and failure analysis, but not exact Bellman optimality.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v34_dla_state_transition_schema.csv",
            "official_claim": "Paper 4 can explain dynamic policy behavior using SDAM state variables.",
            "claim_boundary": "Representative ADP/rollout evidence; Bellman exact claim is false.",
            "quarto_page": "19t-multi-period-solver.qmd",
            "status": "near_resolved_with_plateau",
        },
        {
            "finding_id": "F08",
            "finding": "IFRS9 is supported as an ECL/SICR-inspired proxy lane, not as contractual IFRS9.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v36_ifrs9_readiness_matrix.csv",
            "official_claim": "ECL proxy can stress rankings and policy claims.",
            "claim_boundary": "No contractual IFRS9 without servicing panel, DPD, cure, recovery/prepayment timing, EAD paths and coherent macro scenarios.",
            "quarto_page": "19b-current-assets-and-gaps.qmd",
            "status": "data_blocked",
        },
        {
            "finding_id": "F09",
            "finding": "CATE policy value remains blocked by identification, overlap, hidden-bias and reject-inference limits.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v37_cate_gate_report.csv",
            "official_claim": "Causal diagnostics are documented as a gate, not as policy value.",
            "claim_boundary": "No CATE policy-value claim.",
            "quarto_page": "19b-current-assets-and-gaps.qmd",
            "status": "theory_blocked",
        },
        {
            "finding_id": "F10",
            "finding": "Fairness remains proxy/source governance only because protected attributes or approved proxy protocol are absent.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v37_fairness_proxy_only_protocol.csv",
            "official_claim": "Source governance can be audited; fair-lending legal claims cannot.",
            "claim_boundary": "No fair-lending legal claim.",
            "quarto_page": "19n-online-mdcp-fairness.qmd",
            "status": "data_blocked",
        },
        {
            "finding_id": "F11",
            "finding": "Powell/SDAM is useful as a governance and framing contract, not as an automatic optimizer.",
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v14_powell_framing_audit.csv",
            "official_claim": "The paper can separate metrics, decisions, uncertainty, policy class, evidence, and claim scope.",
            "claim_boundary": "Framing does not prove optimality.",
            "quarto_page": "19f-sequential-decision-framework.qmd",
            "status": "official",
        },
    ]


def build_claims() -> list[dict[str, object]]:
    return [
        {
            "claim": "Paper 4 defines a governed sequential decision analytics lab.",
            "allowed": True,
            "evidence_artifact": "reports/paper_material/paper4/status/paper4_quarto_restructure_status.json",
            "boundary": "Architecture and artifact-backed lab claim only.",
            "prohibited_claim_flag": False,
            "current_quarto_page": "index.qmd",
        },
        {
            "claim": "Paper Estrella official champion remains retained after v38.",
            "allowed": True,
            "evidence_artifact": "reports/paper_material/global/tables/global_v38_promotion_decisions.csv",
            "boundary": "No new promotion unless future promotion protocol passes.",
            "prohibited_claim_flag": False,
            "current_quarto_page": "19ca-v38-final-synthesis.qmd",
        },
        {
            "claim": "Dynamic stress replay compares policies as monthly processes.",
            "allowed": True,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v31_dynamic_policy_summary.csv",
            "boundary": "Internal replay and common sample paths, not forecast validity.",
            "prohibited_claim_flag": False,
            "current_quarto_page": "19t-multi-period-solver.qmd",
        },
        {
            "claim": "CVaR/OCE is a serious tail-risk challenger.",
            "allowed": True,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v31_champion_vs_cvar_stress_memo.csv",
            "boundary": "Tail-governance challenger, not economic champion and not exact full-universe optimum.",
            "prohibited_claim_flag": False,
            "current_quarto_page": "19i-regret-auditability-frontier.qmd",
        },
        {
            "claim": "Online conformal coverage is fully deployable under all temporal/source holdouts.",
            "allowed": False,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v35_online_temporal_holdout.csv",
            "boundary": "Nominal gate passed historically, strict holdouts remain fragile.",
            "prohibited_claim_flag": True,
            "current_quarto_page": "19n-online-mdcp-fairness.qmd",
        },
        {
            "claim": "Contractual IFRS9 lifetime ECL is implemented.",
            "allowed": False,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v36_ifrs9_readiness_matrix.csv",
            "boundary": "Only IFRS9-inspired proxy ECL/SICR is allowed.",
            "prohibited_claim_flag": True,
            "current_quarto_page": "19b-current-assets-and-gaps.qmd",
        },
        {
            "claim": "Formal differentiable SPO+ is implemented.",
            "allowed": False,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v32_spo_dependency_blockers.csv",
            "boundary": "Oracle-regret/surrogate only until dependencies and validation are resolved.",
            "prohibited_claim_flag": True,
            "current_quarto_page": "19t-multi-period-solver.qmd",
        },
        {
            "claim": "CATE policy value is ready for decision selection.",
            "allowed": False,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v37_cate_gate_report.csv",
            "boundary": "Causal diagnostics only; no policy-value claim.",
            "prohibited_claim_flag": True,
            "current_quarto_page": "19b-current-assets-and-gaps.qmd",
        },
        {
            "claim": "Fair-lending legal claim is supported.",
            "allowed": False,
            "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v37_fairness_proxy_only_protocol.csv",
            "boundary": "Proxy/source governance only; no protected-attribute legal claim.",
            "prohibited_claim_flag": True,
            "current_quarto_page": "19n-online-mdcp-fairness.qmd",
        },
    ]


def build_backlog() -> list[dict[str, object]]:
    return [
        {
            "horizon": "immediate",
            "lane": "Quarto governance",
            "executable_item": "Keep chapter 19 curated and move each new wave summary to the living notebook first.",
            "status": "active_rule",
            "next_artifact": "paper4_living_lab_notebook.md",
            "success_condition": "No new wave page is registered unless it becomes official evidence.",
        },
        {
            "horizon": "immediate",
            "lane": "Dynamic stress",
            "executable_item": "Run focused champion-vs-tail challenger stress only when a new candidate changes paired robustness.",
            "status": "resolved_but_rerunnable",
            "next_artifact": "paper4_v39_dynamic_candidate_stress.csv",
            "success_condition": "Candidate improves paired wealth and governance score without claim violations.",
        },
        {
            "horizon": "immediate",
            "lane": "Online conformal",
            "executable_item": "Target the v35 weak source-family holdouts with min-support pooling and source-family calibration.",
            "status": "near_resolved_with_plateau",
            "next_artifact": "paper4_v39_online_source_family_holdout.csv",
            "success_condition": "Source-family holdouts pass without width inflation beyond accepted gate.",
        },
        {
            "horizon": "immediate",
            "lane": "CVaR/OCE",
            "executable_item": "Document strict infeasibility and rerun column generation only for new caps or return floors.",
            "status": "near_resolved_with_plateau",
            "next_artifact": "paper4_v39_cvar_certificate_delta.csv",
            "success_condition": "Clear strict/committee/relaxed label and no false full-universe exact claim.",
        },
        {
            "horizon": "short",
            "lane": "SPO/DFL",
            "executable_item": "Build an isolated dependency environment for cvxpy/cvxpylayers/torch without breaking the main repo.",
            "status": "dependency_blocked",
            "next_artifact": "paper4_spo_isolated_env_repro.md",
            "success_condition": "Minimal differentiable example runs or blocker report is exact and reproducible.",
        },
        {
            "horizon": "short",
            "lane": "DLA/ADP",
            "executable_item": "Improve ADP rollout features and compare only against common-path dynamic baselines.",
            "status": "near_resolved_with_plateau",
            "next_artifact": "paper4_v39_dla_rollout_reaudit.csv",
            "success_condition": "Explains whether gains are state/value driven or only adapter artifacts.",
        },
        {
            "horizon": "short",
            "lane": "Sample paths",
            "executable_item": "Separate external macro context from internal calibration labels and keep forecast claim false.",
            "status": "near_resolved_with_plateau",
            "next_artifact": "paper4_sample_path_claim_boundary.md",
            "success_condition": "Every path family has calibration source, transform and claim boundary.",
        },
        {
            "horizon": "short",
            "lane": "IFRS9/SICR",
            "executable_item": "Improve proxy SICR sensitivity and data blocker register without calling it contractual IFRS9.",
            "status": "data_blocked",
            "next_artifact": "paper4_v39_ifrs9_sicr_proxy_update.csv",
            "success_condition": "Stage 2 behavior is explainable and no contractual claim appears.",
        },
        {
            "horizon": "medium",
            "lane": "CATE",
            "executable_item": "Only continue causal policy value if a cleaner treatment/outcome and overlap gate are found.",
            "status": "theory_blocked",
            "next_artifact": "paper4_cate_identification_reaudit.md",
            "success_condition": "Identification, overlap, sensitivity and falsification all pass.",
        },
        {
            "horizon": "medium",
            "lane": "Fairness",
            "executable_item": "Keep proxy governance unless protected attributes or approved external proxy protocol exist.",
            "status": "data_blocked",
            "next_artifact": "paper4_fairness_protocol_update.md",
            "success_condition": "No fair-lending legal claim without valid data/protocol.",
        },
        {
            "horizon": "medium",
            "lane": "Academic synthesis",
            "executable_item": "Decide whether Paper 4 should become a governance paper or remain a mega lab appendix.",
            "status": "editorial_decision_pending",
            "next_artifact": "paper4_publishability_focus_memo.md",
            "success_condition": "A future paper outline has one primary contribution and bounded appendices.",
        },
        {
            "horizon": "long",
            "lane": "Data expansion",
            "executable_item": "Seek servicing panel, monthly DPD, recovery/cure/prepayment timing and protected attributes only if a real publication route emerges.",
            "status": "data_blocked",
            "next_artifact": "paper4_external_data_requirements.md",
            "success_condition": "External data unlocks contractual IFRS9/fairness claims under explicit protocol.",
        },
    ]


def build_page_registry() -> list[dict[str, object]]:
    pages = sorted(path.name for path in BOOK_DIR.glob("*.qmd"))
    registry = []
    for page in pages:
        rendered = page in OFFICIAL_PAGES
        registry.append(
            {
                "page": page,
                "rendered_in_quarto": rendered,
                "role": "official_curated" if rendered else "historical_archive",
                "target_surface": "quarto_official_chapter"
                if rendered
                else "living_lab_notebook_reference",
                "archive_reason": ""
                if rendered
                else "implementation wave, roadmap, blocker, or intermediate result moved out of main render",
                "path": f"book/chapters/19-paper-mega-extension/{page}",
                "path_exists": (BOOK_DIR / page).exists(),
            }
        )
    return registry


def write_artifacts() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    findings = build_findings()
    claims = build_claims()
    backlog = build_backlog()
    page_registry = build_page_registry()

    write_csv(
        TABLE_DIR / "paper4_current_official_findings.csv",
        findings,
        [
            "finding_id",
            "finding",
            "evidence_artifact",
            "official_claim",
            "claim_boundary",
            "quarto_page",
            "status",
        ],
    )
    write_csv(
        TABLE_DIR / "paper4_current_claim_boundaries.csv",
        claims,
        [
            "claim",
            "allowed",
            "evidence_artifact",
            "boundary",
            "prohibited_claim_flag",
            "current_quarto_page",
        ],
    )
    write_csv(
        TABLE_DIR / "paper4_living_lab_backlog.csv",
        backlog,
        ["horizon", "lane", "executable_item", "status", "next_artifact", "success_condition"],
    )
    write_csv(
        TABLE_DIR / "paper4_quarto_page_registry.csv",
        page_registry,
        [
            "page",
            "rendered_in_quarto",
            "role",
            "target_surface",
            "archive_reason",
            "path",
            "path_exists",
        ],
    )

    status = {
        "schema_version": "2026-05-15.quarto-restructure",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "paper4_quarto_restructure_official_chapter_plus_living_notebook",
        "official_quarto_page_count": len(OFFICIAL_PAGES),
        "historical_archive_page_count": sum(
            1 for row in page_registry if not row["rendered_in_quarto"]
        ),
        "living_notebook_path": "reports/paper_material/paper4/notes/paper4_living_lab_notebook.md",
        "paper4_final_promotion_created": False,
        "paper1_artifacts_modified": False,
        "contractual_ifrs9_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "official_pages": OFFICIAL_PAGES,
        "claim_boundary": "Quarto contains curated official evidence; implementation waves remain in the markdown living notebook.",
    }
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / "paper4_quarto_restructure_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return findings, claims, backlog, page_registry


def write_notebook(
    findings: list[dict[str, object]],
    claims: list[dict[str, object]],
    backlog: list[dict[str, object]],
    page_registry: list[dict[str, object]],
) -> None:
    rendered = [row for row in page_registry if row["rendered_in_quarto"]]
    archived = [row for row in page_registry if not row["rendered_in_quarto"]]
    notebook = f"""# Paper 4 Living Lab Notebook

Generated by `scripts/papers/build_paper4_quarto_restructure.py`.

This is the canonical working notebook for Paper 4. The Quarto chapter now
keeps only official, artifact-backed and thesis/paper-grade material. Iteration
waves, failed experiments, blocker analysis, goal prompts and speculative
roadmaps belong here first.

## Current State

- Paper 4 is a living sequential-decision laboratory, not a final promotion
  protocol.
- Paper Estrella remains the official champion source unless a future promotion
  memo passes its own gates.
- The current Paper 4 official Quarto surface has {len(rendered)} rendered
  pages.
- {len(archived)} historical pages remain on disk as archive material and are
  intentionally not rendered.
- The forbidden final promotion JSON remains absent:
  `reports/paper_material/paper4/status/paper4_final_promotion.json`.

## What Enters Quarto

An item enters the official Quarto chapter only when it is backed by an
artifact, has a clear claim boundary, is stable enough for a paper/thesis
reader, and is not merely a prompt, backlog or transient wave result.

{md_table(rendered, ["page", "role", "target_surface"])}

## What Stays Here

An item stays in this notebook if it is a roadmap, open experiment, failure,
blocked lane, dependency note, goal prompt, or useful evidence that is not yet
ready to become an official Paper 4 contribution.

## Consolidated Findings

{md_table(findings, ["finding_id", "finding", "status", "quarto_page"])}

## Claim Boundaries

{md_table(claims, ["claim", "allowed", "boundary", "current_quarto_page"])}

## Wave Memory, Condensed By Result

| Era | What mattered | What became official | What stayed lab-only |
|---|---|---|---|
| Foundation / MVP | Manifest, SDAM schema, policy registry, evidence tables | Artifact-backed lab foundation | Early page-by-page roadmap |
| v3-v8 | Full policy exact replay, IFRS9 proxy, online CP, CVaR, MDCP, causal/fairness gates | Evidence that the lab can evaluate many policy families | Per-wave implementation details |
| v9-v14 | Online gate, selector governance, Powell framing | Powell/SDAM framing and claim governance | Most intermediate selector/search pages |
| v15-v18 | Dynamic stress engine, sample paths, CVaR/SPO/DLA consolidation | Sequential evaluation thesis | Wave mechanics and failed candidates |
| v19-v30 | Scale-up, registries, champion stress, IFRS9/CATE/fairness gates | Candidate registry and claim discipline | Long-form iteration notes |
| v31-v38 | 512-path stress, SPO dependency audit, CVaR certificate, DLA failure analysis, global synthesis | Official final synthesis and current boundaries | Future implementation prompts |

## Living Lanes

### Dynamic / DLA

Dynamic replay is official as an internal common-path evaluation engine. DLA/ADP
is still representative rather than Bellman-exact. Future work should focus on
state features, rollout depth and explaining underperformance, not on claiming
optimality.

### CVaR / OCE

CVaR is a serious tail-risk challenger. The strongest CVaR candidate improves
tail loss but does not beat the economic champion in paired wealth robustness.
Strict infeasibility and column-generation diagnostics are valid results, but
restricted-master evidence is not full-universe exact optimality.

### SPO / DFL

The defensible lane today is oracle-regret/surrogate training. Formal
differentiable SPO+ remains blocked until the dependency stack is isolated and
validated.

### Online Conformal / MDCP / Source Governance

The nominal online gate was resolved historically, but v35 source-family
holdouts show fragility. This lane is valuable but should remain caveated until
source-family robustness survives stricter validation.

### IFRS9 / SICR

Paper 4 supports IFRS9-inspired ECL/SICR proxy analysis. It does not support a
contractual IFRS9 claim without servicing, DPD, cure, recovery, prepayment, EAD
paths and coherent macro scenarios.

### CATE

CATE is a gated research lane. No policy-value claim should be made until
identification, overlap, sensitivity, falsification and intervals pass.

### Fairness

Fairness is proxy/source governance only. No protected-attribute or fair-lending
legal claim exists.

### Powell / SDAM Governance

Powell is worth keeping because it turns the project into a declared sequential
decision problem: metrics, decisions, uncertainty, policy classes, artifacts and
claims are no longer mixed together.

## Implementable Backlog

{md_table(backlog, ["horizon", "lane", "status", "executable_item", "success_condition"])}

## Historical Page Archive

The following pages remain on disk but are not part of the official Quarto
render. They are provenance, not the current source of truth.

{md_table(archived, ["page", "archive_reason"], max_rows=None)}

## Template For Future Waves

Use this template for new iterations before editing Quarto:

```markdown
## Wave vXX: Short Name

- Goal:
- Scripts:
- New artifacts:
- Result:
- Interpretation:
- Negative result or blocker:
- Claim impact:
- Quarto promotion decision: keep in notebook / promote to official chapter
- Next implementable step:
```
"""
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    (NOTE_DIR / "paper4_living_lab_notebook.md").write_text(notebook, encoding="utf-8")


def frontmatter(title: str) -> str:
    return f'---\ntitle: "{title}"\npage-layout: article\n---\n\n'


def write_pages(
    findings: list[dict[str, object]],
    claims: list[dict[str, object]],
    backlog: list[dict[str, object]],
    page_registry: list[dict[str, object]],
) -> None:
    official_table = md_table(
        [row for row in page_registry if row["rendered_in_quarto"]],
        ["page", "role", "target_surface"],
    )
    findings_table = md_table(findings, ["finding_id", "finding", "status", "quarto_page"])
    claim_table = md_table(claims, ["claim", "allowed", "boundary", "current_quarto_page"])
    backlog_table = md_table(backlog, ["horizon", "lane", "status", "executable_item"], max_rows=12)

    pages = {
        "index.qmd": frontmatter("Paper 4: Sequential Credit Decision Analytics Living Lab")
        + f"""{{{{< include ../../includes/_paper-autocontenido.qmd >}}}}

## Papel actual del Paper 4 {{#sec-paper4-mega-extension}}

Paper 4 queda reestructurado como un **laboratorio vivo** con una superficie
oficial compacta. El libro Quarto ya no intenta renderizar cada ola de
implementacion; conserva solo los resultados que son tangibles, auditables y
defendibles para un futuro paper o tesis.

La bitacora canonica del laboratorio es
`reports/paper_material/paper4/notes/paper4_living_lab_notebook.md`. Ahi viven
los goals, fracasos, bloqueos, prompts largos, decisiones intermedias y nuevas
iteraciones. Quarto queda para lo que ya puede sostener una lectura academica.

::: {{.callout-note}}
## Regla editorial

Si un resultado no tiene artifact, claim boundary y estabilidad suficiente,
primero va al notebook vivo. Solo se promueve al capitulo cuando aporta a la
tesis oficial de Paper 4.
:::

## Superficie oficial renderizada

{official_table}

## Hallazgos oficiales actuales

{findings_table}

## Como leer ahora el capitulo

1. **Propuesta y alcance** define la tesis academica.
2. **Activos y gaps** separa resultados oficiales de lanes bloqueadas.
3. **Arquitectura integrada** explica CRPTO, IFRS9 proxy, SDAM y governance.
4. **Marco SDAM** declara estado, decision, informacion exogena y politica.
5. **Evidence pack** lista la evidencia tangible.
6. **Regret-auditability** muestra la tension retorno, cola, cobertura y auditabilidad.
7. **Online/MDCP/fairness** fija cobertura y source governance con limites claros.
8. **Multi-period solver** resume dynamic stress, DLA/ADP, SPO/DFL y sample paths.
9. **Sintesis v38** deja la verdad actual y los claims permitidos.
""",
        "19a-proposal-and-scope.qmd": frontmatter("Paper 4 Proposal And Scope")
        + """## Idea central

Paper 4 estudia si una decision crediticia puede gobernarse como un problema
secuencial: no basta con predecir PD o maximizar retorno en un libro estatico.
La pregunta es como elegir, evaluar y auditar politicas bajo incertidumbre,
cola, cobertura conformal, provision proxy, fuentes debiles y decisiones
repetidas.

El Paper Estrella sigue siendo la base solida y la referencia oficial. Paper 4
consume esa base, pero no la reemplaza automaticamente. Su valor academico esta
en convertir CRPTO en un laboratorio de decision analytics: policies, states,
uncertainty, sample paths, gates y claim boundaries.

## Tesis defendible hoy

La tesis defendible no es que Paper 4 ya tiene un nuevo champion final. La tesis
defendible es que el proyecto ya tiene una arquitectura reproducible para
comparar familias de politicas bajo gobernanza secuencial:

- CRPTO/Paper Estrella como referencia economica.
- CVaR/OCE como challenger de cola.
- Online conformal y MDCP como gobernanza de cobertura.
- IFRS9/SICR como proxy prudencial, no contractual.
- DLA/ADP como representacion dinamica, no Bellman exacto.
- SPO/DFL como oracle-regret/surrogate, no diferenciable formal.
- CATE/fairness como gates bloqueados con evidencia.

## Lo que queda fuera del claim oficial

No se hacen claims de IFRS9 contractual, CATE policy value, fair-lending legal,
forecast externo, Bellman optimality ni SPO+ diferenciable formal. Esos carriles
siguen vivos en el notebook, pero no entran como contribucion oficial todavia.
""",
        "19b-current-assets-and-gaps.qmd": frontmatter("Current Assets And Gaps")
        + f"""## Activos oficiales

{findings_table}

## Boundaries de claims

{claim_table}

## Gaps reales

Los gaps no son fracasos editoriales: son limites que hacen al paper mas serio.
IFRS9 contractual esta bloqueado por datos de servicing; CATE esta bloqueado por
identificacion y sensibilidad; fairness legal esta bloqueado por ausencia de
atributos protegidos o protocolo aprobado; SPO diferenciable esta bloqueado por
dependencias; CVaR exacto full-universe no debe reclamarse sin prueba exacta.

## Backlog resumido

{backlog_table}
""",
        "19c-integrated-architecture.qmd": frontmatter("Integrated Architecture")
        + """## Arquitectura oficial

Paper 4 se organiza como una arquitectura de evidencia, no como una sola corrida
monolitica.

| Capa | Rol | Claim permitido |
|---|---|---|
| Paper Estrella / CRPTO | Referencia economica robusta | Champion oficial conservado salvo promocion futura con gates |
| Conformal / online | Cobertura y ancho bajo replay | Prometedor, pero holdouts source-family son caveat |
| CVaR / OCE | Control de cola y committee profile | Serious tail challenger |
| IFRS9 proxy | ECL/SICR inspirado en IFRS9 | Prudential proxy, no contractual IFRS9 |
| MDCP/source | Cobertura por familia/fuente | Governance de fuentes, no fairness legal |
| DLA/ADP | Decision mensual y estado secuencial | Representative dynamic evaluation |
| SPO/DFL | Oracle regret y decision loss | Surrogate/oracle-regret only |
| CATE | Identificacion causal futura | Diagnostic gate only |
| Powell/SDAM | Contrato de framing | Separacion de metricas, decisiones, incertidumbre y claims |

## Flujo

1. Se toma una familia de policies o candidate books.
2. Se evalua retorno, ECL proxy, cola, cobertura, source governance y auditabilidad.
3. Se repite la comparacion como replay mensual con sample paths comunes cuando
   la policy lo permite.
4. Se decide si el resultado es official evidence, serious challenger, lab-only,
   data-blocked, theory-blocked o prohibited claim.

Esta arquitectura evita que un resultado con buen retorno sea presentado como
paper-grade si rompe cobertura, auditabilidad, claim safety o datos necesarios.
""",
        "19f-sequential-decision-framework.qmd": frontmatter("Sequential Decision Framework")
        + """## Por que Powell/SDAM si el dataset es estatico

Powell/SDAM no se usa aqui para fingir produccion. Se usa como contrato de
framing academico: declara que incluso con dataset historico, las comparaciones
de policies deben diferenciar estado, decision, informacion exogena,
post-decision state, transicion y politica.

## Elementos SDAM

| Elemento | Interpretacion en Paper 4 |
|---|---|
| `S_t` | cash, outstanding, ECL proxy, stage mix, coverage state, source exposure |
| `x_t` | financiar, no financiar, seleccionar policy, ajustar caps o thresholds |
| `S_t^x` | exposicion financiada, capital usado, budget restante, composicion |
| `W_{t+1}` | defaults, prepayments, recoveries, macro/internal shocks, drift |
| `S_{t+1}` | estado actualizado despues de pagos, perdidas, recuperaciones y ECL |
| `X^pi` | clase de politica: CRPTO, CVaR, DLA, SPO surrogate, static reference |

## Contribucion del framework

El framework vale porque fuerza a separar lo que antes se mezclaba: metricas
base, metricas de riesgo, decisiones, incertidumbre, evidencia y claims. En un
paper academico futuro, esto puede ser una contribucion de gobernanza
secuencial incluso si el dataset no es productivo.
""",
        "19h-mvp-evidence-pack.qmd": frontmatter("MVP Evidence Pack")
        + f"""## Evidencia tangible

El evidence pack oficial ya no es una lista de todas las olas. Es el conjunto
minimo que permite defender que Paper 4 existe como laboratorio reproducible.

{findings_table}

## Artefactos livianos de esta reestructura

| Artifact | Rol |
|---|---|
| `paper4_quarto_restructure_status.json` | Estado de la separacion Quarto/notebook |
| `paper4_current_official_findings.csv` | Hallazgos oficiales actuales |
| `paper4_current_claim_boundaries.csv` | Claims permitidos y prohibidos |
| `paper4_living_lab_backlog.csv` | Implementables por horizonte |
| `paper4_quarto_page_registry.csv` | Paginas oficiales vs archivo historico |
| `paper4_living_lab_notebook.md` | Bitacora canonica de laboratorio |

## Regla de uso

Cuando una nueva ola produzca resultados, primero se agrega al notebook. Solo si
se vuelve estable, artifact-backed y claim-safe se promueve a una pagina oficial.
""",
        "19i-regret-auditability-frontier.qmd": frontmatter("Regret Auditability Frontier")
        + """## Lectura oficial de la frontera

La frontera regret-auditability organiza una tension central: una policy puede
ganar en retorno o cola, pero perder en auditabilidad, cobertura o claim safety.
Por eso Paper 4 no promueve candidates solo por una metrica.

## Resultado consolidado

CVaR/OCE aparece como el challenger de cola mas serio: reduce perdida de cola y
puede ganar bajo un comite tail-first. Sin embargo, el stress v31 muestra que no
supera de forma robusta el wealth pareado del champion economico. Esa es una
contribucion util: Paper 4 puede mostrar como cambia la decision bajo perfiles
de gobernanza distintos.

## Claims permitidos

- CVaR/OCE es un serious challenger de tail-risk.
- La decision cambia si el comite prioriza cola sobre retorno medio.
- La frontera es una herramienta de auditoria, no una promocion final.

## Claims no permitidos

- No hay full-universe exact CVaR optimality.
- No hay champion economico nuevo por CVaR.
- No se debe mezclar relaxed/committee caps con strict feasibility.
""",
        "19n-online-mdcp-fairness.qmd": frontmatter("Online Conformal MDCP And Source Governance")
        + """## Online conformal

El resultado historico importante es doble. Primero, el gate nominal se resolvio
en la ola online: source-month, policy-month y ancho quedaron en zona operativa.
Segundo, la validacion v35 mostro que el resultado no sobrevive de forma
universal a holdouts source-family. Por eso el claim oficial debe ser cuidadoso.

## MDCP y source governance

MDCP/source coverage sirve como gobernanza de cobertura por familias: grade,
month, period, state, income, DTI, score decile e intersecciones con soporte
suficiente. Esto es evidencia de robustez operacional, no evidencia legal de
fair lending.

## Fairness

La fairness del Paper 4 queda como proxy governance only. Sin atributos
protegidos o protocolo externo aprobado, no hay fair-lending legal claim.

## Claim oficial

Paper 4 puede defender que mide y gobierna fuentes debiles de cobertura. No
puede afirmar cumplimiento legal de fair lending ni despliegue online real.
""",
        "19t-multi-period-solver.qmd": frontmatter("Multi Period Solver And Dynamic Stress")
        + """## Dynamic stress engine

La contribucion tangible mas importante del laboratorio es que las policies
dejaron de compararse solo como funded books estaticos. El stress engine v31
evalua candidates como procesos mensuales bajo common random numbers, con cash,
outstanding, defaults, prepayments, recoveries, ECL, coverage state y wealth.

## DLA/ADP

DLA/ADP ya tiene lenguaje SDAM: estado, decision, post-decision state,
transicion y reward. Pero la evidencia actual sigue siendo representativa. No se
reclama Bellman optimality.

## SPO/DFL

SPO/DFL queda como oracle-regret/surrogate. Es util para comparar decision loss
contra oracles restringidos, pero no debe llamarse differentiable SPO+ hasta que
el stack de dependencias se resuelva y exista validacion.

## Sample paths

Los sample paths son internos y sirven para comparacion pareada, sensibilidad y
stress. No son forecast externo. Cuando se usa macro externo, debe quedar
etiquetado como contexto o calibracion parcial.
""",
        "19ca-v38-final-synthesis.qmd": frontmatter("Current Paper 4 Synthesis")
        + f"""## Verdad actual

La sintesis vigente despues de v38 y de esta reestructura es simple:

- Paper Estrella conserva el champion oficial.
- Paper 4 conserva un laboratorio secuencial con working candidates y serious
  challengers.
- No existe promocion final de Paper 4.
- El capitulo Quarto contiene solo evidencia oficial y claim-safe.
- El notebook vivo contiene las olas, pendientes, bloqueos e iteraciones.

## Hallazgos oficiales

{findings_table}

## Claims y prohibiciones

{claim_table}

## Pendientes ejecutables

{backlog_table}

## Decision editorial

El Paper 4 vale la pena si se presenta como arquitectura academica de decision
secuencial gobernada. No debe presentarse todavia como un nuevo champion final
ni como sistema IFRS9/fairness/causal completo. Su aporte real es mostrar que el
proceso de decidir bajo incertidumbre puede auditarse por multiples lentes:
retorno, regret, cola, cobertura, source governance, ECL proxy, dinamica y claim
safety.
""",
    }

    for page, content in pages.items():
        (BOOK_DIR / page).write_text(content, encoding="utf-8")


def main() -> None:
    findings, claims, backlog, page_registry = write_artifacts()
    write_notebook(findings, claims, backlog, page_registry)
    write_pages(findings, claims, backlog, page_registry)
    print("Generated Paper 4 restructure artifacts, living notebook, and curated Quarto pages.")


if __name__ == "__main__":
    main()
