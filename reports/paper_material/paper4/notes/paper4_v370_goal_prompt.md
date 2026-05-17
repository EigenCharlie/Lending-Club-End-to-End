# Paper 4 Living-Lab Goal Prompt v370

Goal: continue Paper 4 living-lab execution after v365-v369 without time limit, prioritizing executable diagnostics that improve publishable Paper 4 claims while preserving all claim boundaries.

Current hard facts:
- Latest committed wave before this prompt: v370 backlog refresh after v369 proxy/live gate separation.
- v366 chunk 0001 screened 1710000 ordered one-swaps and found 0 source-exact rows.
- v367 selected `bounded_claim_scope_update`.
- v368 allowed 4 bounded publishable claim rows and kept champion/final/global claims blocked.
- v369 found 2 of 10 proxy/live/final gate requirements met.
- Strict live deployability remains `False`.
- Final promotion remains `False`.

Non-negotiable constraints:
- Before every run, commit or push, verify `reports/paper_material/paper4/status/paper4_final_promotion.json` does not exist.
- Do not create Paper 4 final promotion, working champion, Paper Estrella replacement, live deployment, contractual IFRS9, fairness legal or full-universe optimality claims.
- Keep new evidence in the living notebook and claim-boundary tables unless a later explicit promotion gate is built and approved.
- Use `--no-verify` for commits and pushes in this branch.

Immediate executable order:
1. v371_source_governance_blocker_diagnostic: build `paper4_v371_source_governance_blocker_diagnostic.csv`. Success: rank source families/block ids causing the zero-source-exact outcome.
2. v372_paper4_claim_language_section_draft: build `paper4_v372_claim_language_section_draft.md`. Success: draft abstract/results/limitations wording with citations to v361-v369.
3. v373_full_v55_chunk_002_or_stop_rule: build `paper4_v373_full_v55_chunk_002_or_stop_rule.csv`. Success: choose chunk 002, targeted chunk sampling or a documented stop rule.
4. v374_live_gate_data_contract: build `paper4_v374_live_gate_data_contract.csv`. Success: list exact external holdout, IFRS9, monitoring and approval inputs.
5. v375_quarto_integration_decision: build `paper4_v375_quarto_integration_decision.csv`. Success: decide living notebook only versus registered curated chapter page.
6. v376_guardrail_debt_register: build `paper4_v376_guardrail_debt_register.csv`. Success: separate living-lab wave guardrails from historical Quarto registry debt.

Definition of done for the next iteration:
- At least one new wave is executed and committed.
- New evidence has status JSON, CSV/MD artifacts, living notebook block, claim boundaries, backlog row and pytest guardrail.
- The final-promotion artifact remains absent.
- The final answer reports useful evidence, remaining blockers and next executable artifact.
