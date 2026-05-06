# Time Series vNext Decision

Date: 2026-04-02

## Purpose

This document closes the `time_series_vnext` redesign lane as an executed research program. It records what was implemented, what was tested, what should remain in the project, and what should not be promoted yet.

Artifacts of record:

- `models/time_series_vnext_status.json`
- `models/time_series_policy_review.json`
- `data/processed/time_series_vnext.parquet`
- `data/processed/time_series_panel_vnext.parquet`
- `data/processed/ts_backtest_metrics_vnext.parquet`
- `data/processed/ts_interval_eval_vnext.parquet`
- `data/processed/ts_joint_path_eval_vnext.parquet`
- `data/processed/ts_ifrs9_scenarios_vnext.parquet`
- `data/processed/ts_ecl_intervals_vnext.parquet`
- `data/processed/time_series_policy_review_matrix.parquet`

## What Was Implemented

- Enriched internal-only monthly vintage contract in `src/data/build_datasets.py`
- Parallel targets:
  - `raw_rate`
  - `logit_rate`
- Point benchmarking in `src/models/time_series_vnext.py`
- Marginal interval benchmarking with governed comparison against adaptive candidates
- Joint uncertainty via:
  - `gaussian_copula`
  - `schaake_shuffle`
- TS -> IFRS9 / ECL translation in `scripts/run_time_series_vnext.py`
- Policy review artifact and recommendation matrix

## Executed Outcome

Selected target:
- `raw_rate`

Point layer:
- champion: `AutoARIMA`
- promotable: `true`
- interpretation: operationally usable, but not materially better than the canonical point lane

Interval layer:
- best backtest challenger: `MAPIE_ENBPI`
- promotable: `false`
- key reason: improved the 90% coverage gap versus the canonical interval layer, but still failed the governed promotion threshold

Operational forward interval generation:
- backtest winner `MAPIE_ENBPI` is not a forward-producing operational model in the current implementation
- forward interval artifact therefore falls back to `AutoARIMA`

Joint uncertainty:
- both `gaussian_copula` and `schaake_shuffle` produced usable path artifacts
- they are informative for accumulated-risk and ECL-width analysis
- they are not yet promoted to official operational policy

TS -> ECL:
- vNext ECL translation was numerically stable
- this makes it worth keeping as research support
- it is not yet a canonical downstream dependency

Overall recommendation:
- `maintain_canonical_keep_vnext_research`

## Keep / Research / Do Not Promote

### Keep

- Canonical point forecast lane
- Canonical IFRS9 temporal overlay
- Canonical publication of diagnostic intervals
- vNext TS -> ECL artifact as research support

### Keep As Research Only

- Enriched internal-only time-series contract
- `MAPIE_ENBPI` / adaptive interval challengers
- `gaussian_copula` sample paths
- `schaake_shuffle` sample paths
- `time_series_vnext` orchestration lane and outputs

### Do Not Promote Yet

- Replacing canonical `time_series` inputs with the enriched vNext contract
- Replacing canonical interval semantics with adaptive intervals
- Treating sample-path uncertainty as an official IFRS9 policy layer
- Rewiring canonical `run_ifrs9_sensitivity.py` to consume vNext artifacts by default

## Why The Canonical Lane Stays

- The redesign did not produce a material point-forecast improvement.
- The interval challenger improved diagnostics but still did not pass the governed promotion threshold.
- The strongest new value came from prudential interpretation, not from a clean operational replacement.

That means the right decision is not to discard the work, but to contain it as a reproducible research lane.

## Relation To Papers And Mega Extension

- **Paper 2**: the time-series lane supports IFRS9 scenario sensitivity and
  TS->ECL interpretation. The point forecast can remain as overlay input, while
  interval outputs should be described as analytical support rather than an
  official monthly provisioning policy.
- **Paper Estrella**: the current CRPTO champion is one-shot/funded-set and does
  not consume time-series intervals. TS becomes relevant only for a future
  multi-period or sequential CRPTO extension.
- **Paper 3 / Mondrian**: the failed promotion of temporal intervals motivates
  online/adaptive conformal coverage as future work, not as current evidence.
- **Mega extension IFRS9 + CATE + CRPTO**: vNext sample paths and TS->ECL
  artifacts are natural inputs to a future state variable `S_t` with macro
  forecasts, stage mix and ECL by policy, but they require new selector
  artifacts before becoming central to a champion.

## Practical Repo Policy

- `scripts/forecast_default_rates.py` remains the canonical producer.
- `scripts/run_time_series_vnext.py` remains a namespaced research producer.
- `models/time_series_status.json` remains the only canonical status contract for operational consumption.
- `models/time_series_vnext_status.json` and `models/time_series_policy_review.json` remain research governance artifacts.

## Next Promotion Condition

Promotion should be reconsidered only if a future rerun demonstrates all of the following:

- no material deterioration in point forecasting
- interval coverage clears the governed threshold
- forward interval generation is operationally coherent, not only backtest-strong
- the added complexity is justified by downstream IFRS9 or governance value
