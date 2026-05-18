# Paper 4 Data Frontier Research Proposal - 2026-05-18

## Purpose

This memo re-audits the seven Paper 4 lanes that were parked or blocked after
loop closure. The audit uses `data/interim/lending_club_cleaned.parquet`, the
raw Lending Club CSV header and selected raw servicing fields, and
`docs/LCDataDictionary.xlsx`.

The goal is not to reopen the old wave loop. The goal is to decide which lanes
can support one bounded experiment and which remain blocked with the current
data.

## Dataset Surface

| metric | value |
|---|---:|
| cleaned rows | 1860764 |
| cleaned columns | 113 |
| distinct loans | 1860764 |
| issue months | 160 |
| issue range | 2007-06-01 00:00:00 to 2020-09-01 00:00:00 |
| states | 51 |
| zip3 prefixes | 949 |
| default rate | 19.51% |
| mean LGD | 12.06% |
| mean LGD mature 24m | 11.78% |

Top loan statuses:

| loan_status | n |
| --- | --- |
| Fully Paid | 1497783 |
| Charged Off | 362548 |
| Default | 433 |

Application types:

| application_type | n |
| --- | --- |
| Individual | 1791229 |
| Joint App | 69535 |

## Lane Decisions After Data Audit

| lane | decision | cleaned_missing | raw_adds | hard_blockers |
| --- | --- | --- | --- | --- |
| ifrs9_sicr | bounded_experiment | none | funded_amnt;out_prncp;total_pymnt;total_rec_prncp;total_rec_int;recoveries;collection_recovery_fee;last_pymnt_d;last_pymnt_amnt;last_credit_pull_d;hardship_flag;hardship_status;hardship_dpd;hardship_loan_status;debt_settlement_flag | No monthly account performance panel; No contractual days-past-due history before default; No borrower-level macro scenario path |
| fair_lending_proxy | proxy_only | none | zip_code;addr_state | No race, ethnicity, sex or age; No surname for BISG; No full address or Census tract |
| cate_policy_value | diagnostic_only | none | funded_amnt;total_pymnt;recoveries | Only accepted/funded loans are visible; No randomized pricing or approval instrument; No rejected-applicant counterfactuals in the retained project data |
| online_conformal | bounded_experiment | none | last_credit_pull_d;hardship_flag | No true external source distribution; No production feedback loop; Only historical retrospective issue-month evaluation |
| spo_dfl | isolated_prototype | none | total_pymnt;recoveries | Differentiable optimization dependency and scaling risk; No reason to disturb the main CRPTO pipeline |
| dla_adp | rollout_only | none | last_pymnt_d;last_credit_pull_d;out_prncp;hardship_flag | No monthly borrower state trajectory; No actual sequential decision logs; No realized action policy history beyond accepted loans |
| cvar_oce | tail_challenger_only | none | total_pymnt;total_rec_prncp;total_rec_int;recoveries | Existing paired replay did not beat economic champion; No new cap or return floor has changed the objective |

## Raw Servicing Fields Worth Knowing

| variable | non_null | missing_rate | top_values_sample |
| --- | --- | --- | --- |
| debt_settlement_flag | 2925492 | 0.00% | N:298; Y:2 |
| hardship_dpd | 143637 | 95.09% |  |
| hardship_flag | 2887057 | 1.31% | N:285; Y:15 |
| hardship_status | 143635 | 95.09% | ACTIVE:184; COMPLETED:64; COMPLETE:34; BROKEN:18 |
| last_pymnt_d | 2920571 | 0.17% | May-2020:102; Apr-2018:9; Sep-2015:8; Jan-2015:7; Nov-2018:7 |
| recoveries | 2925492 | 0.00% |  |
| total_pymnt | 2925492 | 0.00% |  |

## External Research Triangulation

| Lane | Source | Implication For Paper 4 |
|---|---|---|
| IFRS9/SICR | IFRS Foundation IFRS 9 official page: https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/ | IFRS9 is the right accounting reference for expected credit losses, but Paper 4 still lacks contractual servicing and macro scenario infrastructure. |
| IFRS9/SICR | Competing-risks survival for lifetime ECL: https://www.sciencedirect.com/science/article/pii/S095741742400472X | The best bounded upgrade is a lifetime/default-timing diagnostic, not a full accounting-compliance claim. |
| Fair lending | CFPB BISG proxy methodology: https://github.com/cfpb/proxy-methodology | BISG needs surname plus geocoding. Lending Club exposes zip3/state but no surname or tract, so legal fair-lending stays false. |
| Fair lending | Zhang, "Assessing Fair Lending Risks Using Race/Ethnicity Proxies": https://pubsonline.informs.org/doi/10.1287/mnsc.2016.2579 | Proxy-based disparity estimation is a real research lane, but Paper 4 has insufficient protected-attribute proxy inputs. |
| CATE | DoWhy assumptions paper: https://www.microsoft.com/en-us/research/publication/dowhy-addressing-challenges-in-expressing-and-validating-causal-assumptions/ | Causal estimates need explicit assumptions and refutations; prediction-style validation is not enough. |
| CATE | EconML CausalForestDML docs: https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html | A high-rate-within-grade CATE screen is technically possible, but only as sensitivity/diagnostic evidence. |
| Online conformal | Adaptive conformal inference: https://arxiv.org/abs/2106.00170 | ACI is relevant for distribution shift and online coverage, but Paper 4 still has retrospective history rather than production feedback. |
| Online conformal | Multi-Distribution Robust Conformal Prediction: https://arxiv.org/abs/2601.02998 | MDCP-style uniform source coverage is the right direction for source-family holdouts. |
| SPO/DFL | Smart "Predict, then Optimize": https://arxiv.org/abs/1710.08005 | SPO+ is relevant because CRPTO has an optimization decision downstream of predictions. |
| SPO/DFL | PyEPO SPOPlus docs: https://khalil-research.github.io/PyEPO/build/html/content/examples/function.html | Use PyEPO only in an isolated prototype to avoid disturbing the main pipeline. |
| SPO/DFL | CVXPYlayers: https://github.com/cvxpy/cvxpylayers | Differentiable convex layers are feasible, but dependency/scaling risk keeps this out of the official champion. |
| CVaR/OCE | Rockafellar and Uryasev CVaR optimization: https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf | CVaR can be optimized with scenario/LP methods; this supports challenger analysis, not champion replacement by itself. |
| CVaR/OCE | Riskfolio-Lib: https://github.com/dcajasn/Riskfolio-Lib | Mature CVaR tooling exists, so future work should reuse tooling or a compact CVXPY LP rather than more generated waves. |

## Bounded Implementation Proposal

1. `ifrs9_sicr`: run one raw-enriched lifetime-ECL/SICR diagnostic using
   `total_pymnt`, `recoveries`, `last_pymnt_d`, hardship flags, and the existing
   LGD fields. Claim remains IFRS9-inspired, not contractual IFRS9.
2. `online_conformal`: run one source-family holdout redesign using
   issue-month, grade, state, income/DTI bins, and zip3. Evaluate MDCP-style
   max-p/union or defended-source pooling. Claim remains retrospective source
   governance.
3. `cate_policy_value`: run only a high-rate-within-grade observational
   sensitivity screen. The output is a causal-identification memo and placebo
   diagnostics, not a policy-value claim.
4. `cvar_oce`: run at most one raw-cashflow repricing check to see whether
   recovery-aware losses materially change the tail challenger. The champion
   cannot change unless paired wealth beats the current economic champion.
5. `fair_lending_proxy`: keep legal fair-lending false. Optionally add a
   geography/source governance appendix. BISG is not feasible without surnames
   and finer geocoding.
6. `spo_dfl`: keep isolated. A toy PyEPO or cvxpylayers prototype may be useful,
   but it should not enter the main CRPTO pipeline.
7. `dla_adp`: keep exact Bellman optimality false. Improve only the rollout
   simulator language and feature list.

## Files Written

- `reports/paper_material/paper4/tables/paper4_data_frontier_variable_inventory_2026-05-18.csv`
- `reports/paper_material/paper4/tables/paper4_data_frontier_lane_decisions_2026-05-18.csv`
- `reports/paper_material/paper4/tables/paper4_data_frontier_raw_servicing_profile_2026-05-18.csv`
