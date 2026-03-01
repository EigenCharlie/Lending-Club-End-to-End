# Run Comparison: 2026-03-01-C-smart-v2

- Generated: 2026-03-01T18:37:58.571495+00:00
- Overall gates pass: `True`
- Conformal promotion pass: `True`
- Conformal statistical warning: `True`
- Artifact coherence pass: `True`
- A/B gate mode: `no_regression`
- A/B no-regression pass: `True`
- A/B significance (diagnostic): `False`

## Gates
- `artifact_coherence`: **PASS**
- `pd_quality`: **PASS**
- `conformal_policy`: **PASS**
- `ab_no_regression`: **PASS**
- `fairness_relative`: **PASS**
- `survival_quality`: **PASS**
- `export_contracts`: **PASS**

## Artifact Changes
- `data/processed/ifrs9_scenario_summary.parquet`: hash_changed=True, baseline_exists=True, current_exists=True
- `data/processed/model_comparison.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `data/processed/pipeline_summary.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/conformal_lgd_ead_status.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/conformal_policy_status.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/fairness_audit_status.json`: hash_changed=True, baseline_exists=False, current_exists=True
- `models/survival_summary.pkl`: hash_changed=True, baseline_exists=True, current_exists=True
- `reports/dvc/metrics_summary.json`: hash_changed=True, baseline_exists=False, current_exists=True
- `reports/gpu_benchmark/cudf_polars_benchmark.csv`: hash_changed=True, baseline_exists=True, current_exists=True
- `reports/gpu_benchmark/cugraph_benchmark.csv`: hash_changed=True, baseline_exists=True, current_exists=True
- `reports/gpu_benchmark/cuml_benchmark.csv`: hash_changed=True, baseline_exists=True, current_exists=True
- `reports/gpu_benchmark/cuopt_benchmark.csv`: hash_changed=True, baseline_exists=True, current_exists=True

## Conformal Diagnostics
- Statistical warnings (non-blocking): `kupiec_pvalue_90`, `kupiec_pvalue_95`, `christoffersen_pvalue_90`, `christoffersen_pvalue_95`
