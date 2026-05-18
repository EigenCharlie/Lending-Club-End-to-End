# Paper 4 Goal Loop Final Audit - 2026-05-18

## Decision

The Paper 4 goal loop is closed.

Paper Estrella remains the official publication and champion surface. Paper 4
remains a rendered long-horizon living lab and provenance chapter, not a final
promotion protocol. Future Paper 4 work must start with a target claim, an
evidence gate, an artifact sink, a budget, and a stop rule before any new
experiment, builder, table, or test is created.

No follow-up should recreate `paper4_final_promotion.json`, assert retired
artifacts in guardrail tests, or open another unbounded `paper4_v###` wave
without new data, a new proof, or a concrete reviewer requirement.

## What The Loop Produced

- Closure/export commit: `f45b810`.
- Deep cleanup commit: `229ccb0`.
- Rendered Quarto book: 122 HTML outputs.
- Active Paper Estrella book surface: `14a` through `14o` plus index.
- Active Paper 4 book surface: `index`, `19a`, `19b`, `19c`, `19f`, `19h`,
  `19i`, `19n`, `19t`, and `19ca`.
- Paper 4 source archive: 80 `.qmd` files remain, but only the 10 files listed
  above are active in `book/_quarto.yml`.
- Cleanup manifest:
  `reports/paper_material/paper4/tables/paper4_deep_cleanup_manifest_2026-05-18.csv`
  with 3,534 retired paths.

Current retained Paper 4 generated surface:

| Surface | Count | Role |
|---|---:|---|
| `reports/paper_material/paper4/tables/*.csv` | 561 | Retained evidence, manifests, and compact provenance |
| `scripts/papers/*.py` | 41 | Retained source builders only |
| `reports/paper_material/paper4/status/*.json` | 74 | Status and schema records |
| `reports/paper_material/paper4/notes/*.md` | 50 | Human-readable provenance |

## What Serves Paper Estrella

| Finding | Destination | Source |
|---|---|---|
| F03 dynamic stress replay reached 512 common paths and 494,592 trace rows. | Robustness appendix and discussion. | `paper4_v31_champion_vs_cvar_stress_memo.csv`; `14h-journal-appendix-robustness.qmd` |
| F04 CVaR/OCE challenger reduced tail loss but did not beat paired wealth robustly (`prob_beats = 47.65625%`). | Justifies retaining the economic champion instead of switching to CVaR/OCE. | `paper4_v31_champion_vs_cvar_stress_memo.csv`; `paper4_v467_cvar_frontier_probe.csv`; `14h-journal-appendix-robustness.qmd` |
| F05 online conformal passed a nominal gate but failed strict source-style holdout expectations. | Limitations and future work, not a deployment claim. | `paper4_v470_online_monitoring_proxy_summary.csv`; `14h-journal-appendix-robustness.qmd` |
| Powell/SDAM gives the right language for decision analytics and controlled policy search. | Theory positioning in Paper Estrella. | `14b-theoretical-framework.qmd`; `19f-sequential-decision-framework.qmd`; `paper4_sequential_decision_schema.json` |

Paper Estrella absorbed these pieces in `14b`, `14e`, and `14h`. It did not
absorb Paper 4 as a new official champion, and it should not inherit Paper 4's
long experimental tail.

## What Serves Paper 4

Paper 4 is useful as a governed research lab when it stays bounded:

- `19a`, `19b`, and `19c` define proposal, assets, gaps, and architecture.
- `19f` gives the SDAM/sequential-decision framework.
- `19h` is the compact evidence pack.
- `19i` records the regret/auditability frontier and CVaR/OCE boundary.
- `19n` records online conformal, MDCP, and fairness as gated lanes.
- `19t` records the dynamic stress/multi-period solver lane.
- `19ca` closes the v38 synthesis and states what is official.

The unrendered Paper 4 `.qmd` files are source history. They can be useful for
forensics, but they are not the active book surface and should not drive new
tests or new waves.

## What Was Retired

The loop became wasteful when generated artifacts were preserved because tests
mentioned them rather than because they supported a claim.

The cleanup retired:

- v39-v466 and v479-v489 generated Paper 4 tables, status JSONs, and notes.
- v39-v489 generated builder scripts under `scripts/papers/`.
- The v490-v526 procedural follow-up loop.
- Stale Python cache files from retired builders.
- The old oversized guardrail body that treated loop artifacts as permanent
  requirements.

The deletion manifest records:

| Retired kind | Count |
|---|---:|
| table CSV | 2,416 |
| builder script | 440 |
| status JSON | 444 |
| note Markdown | 105 |
| Python cache | 129 |

## What Can Still Advance

Advance only the lanes below, and only with explicit gates.

| Lane | Worth doing if | Output cap |
|---|---|---|
| Paper Estrella journal package | The manuscript needs final appendix prose, table citations, and reviewer-facing caveats. | One appendix pass; no new Paper 4 wave. |
| External/source holdout for online conformal | There is real external data or a defensible source-family holdout protocol. | One table and one limitations update. |
| CVaR/OCE challenger | A new cap, return floor, formal infeasibility certificate, or reviewer request changes the claim. | One challenger table and one interpretation note. |
| IFRS9/SICR | Real contractual servicing, DPD, recovery, prepayment, EAD, and macro scenario data are available. | One bounded evidence pack; otherwise keep proxy caveat. |
| Fairness/CATE | Protected attributes, validated proxy protocol, or credible causal identification/overlap evidence exists. | One governance/identification memo; no policy-value claim without it. |
| Paper 4 governance paper | The goal is a separate methodology paper about research-program governance. | One outline plus one claim-evidence map. |

Post-audit update: `paper4_data_frontier_research_proposal_2026-05-18.md`
audits the cleaned parquet, raw servicing fields, and Lending Club dictionary.
It moves IFRS9/SICR, online conformal, CATE, and CVaR/OCE into bounded
experiment menus, while keeping fair-lending legal claims, exact Bellman
optimality, and integrated differentiable SPO+ parked until missing data or
dependencies appear.

## Stop Rules

- Do not create a new `paper4_v###` wave for polish, curiosity, or a failing
  test alone.
- Do not add tests that require retired artifacts.
- Do not promote Paper 4 into Paper Estrella unless a claim passes the
  promote/append/park/archive/delete gate.
- Do not generate more than one table per accepted follow-up claim unless the
  claim explicitly needs multiple evidence levels.
- Archive or delete negative results after they are summarized; do not keep
  every failed iteration as a permanent paper artifact.
