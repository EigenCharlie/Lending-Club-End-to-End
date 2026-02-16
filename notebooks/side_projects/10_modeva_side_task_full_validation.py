# %% [markdown]
# # Side Task: Evaluacion Integral con Modeva (Lending Club Risk)
#
# Este notebook es una tarea lateral (`side task`) para evaluar metodos y tecnicas de `modeva`
# sobre el caso de riesgo de credito del proyecto.
#
# Alcance:
# - Ejecutar metodos utiles/relevantes para datos tabulares de default.
# - Comparar resultados con el baseline core del proyecto.
# - Producir insights tecnicos y recomendacion de adopcion.
#
# Nota:
# - No reemplaza el pipeline canonico actual (PD + calibracion + conformal + optimizacion).
# - Debe ejecutarse en el entorno principal del proyecto (no en `Legacy`).

# %%
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

warnings.filterwarnings("ignore")

import modeva
import modeva.auth as modeva_auth
from modeva import models

# Evita prompt interactivo de auth en ejecucion batch.
modeva_auth.Authenticator.run = lambda self, auth_code=None: None

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

print("modeva version:", modeva.__version__)
print("project root:", PROJECT_ROOT)

# %% [markdown]
# ## 1) Contexto baseline del proyecto core

# %%
with open(MODEL_DIR / "pd_training_record.pkl", "rb") as f:
    pd_record = pickle.load(f)

conformal_status = json.loads((MODEL_DIR / "conformal_policy_status.json").read_text())
causal_rule = json.loads((MODEL_DIR / "causal_policy_rule.json").read_text())
ifrs9_summary = pd.read_parquet(DATA_DIR / "ifrs9_scenario_summary.parquet")

core_snapshot = pd.DataFrame(
    {
        "metric": [
            "core_pd_auc_test",
            "core_pd_brier_test",
            "core_conformal_cov90",
            "core_conformal_cov95",
            "core_conformal_min_group_cov90",
            "core_causal_selected_rule",
            "core_ifrs9_baseline_ecl",
            "core_ifrs9_severe_ecl",
        ],
        "value": [
            pd_record["final_test_metrics"]["auc_roc"],
            pd_record["final_test_metrics"]["brier_score"],
            conformal_status["coverage_90"],
            conformal_status["coverage_95"],
            conformal_status["min_group_coverage_90"],
            causal_rule["selected_rule"],
            float(ifrs9_summary.loc[ifrs9_summary["scenario"] == "baseline", "total_ecl"].iloc[0]),
            float(ifrs9_summary.loc[ifrs9_summary["scenario"] == "severe", "total_ecl"].iloc[0]),
        ],
    }
)
core_snapshot

# %% [markdown]
# ## 2) Carga de datos y construccion de datasets

# %%
base_features = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "int_rate",
    "installment",
    "loan_to_income",
    "rev_utilization",
    "open_acc",
    "total_acc",
    "fico_score",
    "grade",
    "home_ownership",
    "purpose",
    "verification_status",
    "term",
]
target_col = "default_flag"

train_raw_full = pd.read_parquet(DATA_DIR / "train_fe.parquet")
test_raw_full = pd.read_parquet(DATA_DIR / "test_fe.parquet")

raw_cols = base_features + [target_col]
train_raw = train_raw_full.sample(6000, random_state=RANDOM_STATE).reset_index(drop=True)[raw_cols].copy()
test_raw = test_raw_full.sample(3000, random_state=RANDOM_STATE).reset_index(drop=True)[raw_cols].copy()

protected_data = pd.concat(
    [train_raw[["home_ownership"]], test_raw[["home_ownership"]]],
    axis=0,
).reset_index(drop=True)

combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
combined_num = pd.get_dummies(
    combined,
    columns=["grade", "home_ownership", "purpose", "verification_status"],
    drop_first=True,
)
combined_num = combined_num.fillna(combined_num.median(numeric_only=True))

train_num = combined_num.iloc[: len(train_raw)].reset_index(drop=True)
test_num = combined_num.iloc[len(train_raw) :].reset_index(drop=True)

print("train_raw:", train_raw.shape, "test_raw:", test_raw.shape)
print("train_num:", train_num.shape, "test_num:", test_num.shape)
print("train default rate:", round(train_raw[target_col].mean(), 6))
print("test default rate:", round(test_raw[target_col].mean(), 6))

# %% [markdown]
# ## 3) Setup Modeva + helpers

# %%
ds_raw = modeva.DataSet(name="lc_modeva_raw_side_task")
ds_raw.load_dataframe_train_test(train_raw, test_raw)
ds_raw.set_target(target_col)

ds = modeva.DataSet(name="lc_modeva_numeric_side_task")
ds.load_dataframe_train_test(train_num, test_num)
ds.set_target(target_col)
ds.set_protected_data(protected_data)

results = {}
run_log = []


def run_test(name, fn):
    try:
        out = fn()
        results[name] = out
        run_log.append({"test": name, "status": "ok", "error": ""})
        table = getattr(out, "table", None)
        if isinstance(table, pd.DataFrame):
            print(f"[OK] {name}: table_shape={table.shape}")
        elif isinstance(table, dict):
            print(f"[OK] {name}: table_keys={list(table.keys())}")
        else:
            print(f"[OK] {name}: no_table")
        return out
    except Exception as e:
        run_log.append({"test": name, "status": "error", "error": repr(e)})
        print(f"[FAIL] {name}: {repr(e)}")
        return None


def result_table(name):
    out = results.get(name)
    if out is None:
        return None
    table = getattr(out, "table", None)
    return table if isinstance(table, pd.DataFrame) else None


def result_value(name):
    out = results.get(name)
    if out is None:
        return None
    return getattr(out, "value", None)

# %% [markdown]
# ## 4) Preprocessing nativo de Modeva

# %%
_ = ds_raw.impute_missing(add_indicators=True)
_ = ds_raw.encode_categorical(method="onehot")
_ = ds_raw.scale_numerical(method="standardize")
_ = ds_raw.bin_numerical(features=("int_rate", "dti"), method="quantile", bins=5)
ds_raw.preprocess()

preprocess_summary = {
    "raw_shape": tuple(ds_raw.to_df(raw_data=True).shape),
    "preprocessed_shape": tuple(ds_raw.to_df(raw_data=False).shape),
}

x_small = ds_raw.get_data("train")[:10, :-1]
try:
    _ = ds_raw.transform(x_small)
    preprocess_summary["transform_status"] = "ok"
except Exception as e:
    preprocess_summary["transform_status"] = f"error: {repr(e)}"

try:
    _ = ds_raw.inverse_transform(x_small)
    preprocess_summary["inverse_transform_status"] = "ok"
except Exception as e:
    preprocess_summary["inverse_transform_status"] = f"error: {repr(e)}"

pd.DataFrame([preprocess_summary])

# %% [markdown]
# ## 5) Data diagnostics: EDA, drift, outliers, feature selection

# %%
run_test("summary", lambda: ds.summary())
run_test("eda_1d", lambda: ds.eda_1d(feature="int_rate", dataset="train", sample_size=2000))
run_test(
    "eda_2d",
    lambda: ds.eda_2d(
        feature_x="int_rate",
        feature_y="loan_to_income",
        feature_color=target_col,
        dataset="train",
        sample_size=2000,
    ),
)
run_test(
    "eda_3d",
    lambda: ds.eda_3d(
        feature_x="int_rate",
        feature_y="loan_to_income",
        feature_z="fico_score",
        feature_color=target_col,
        dataset="train",
        sample_size=1000,
    ),
)
run_test(
    "eda_correlation",
    lambda: ds.eda_correlation(
        features=("int_rate", "loan_to_income", "fico_score", "dti", target_col),
        dataset="train",
    ),
)
run_test("eda_pca", lambda: ds.eda_pca(dataset="train", sample_size=2000))
run_test(
    "eda_umap",
    lambda: ds.eda_umap(dataset="train", sample_size=1000, n_neighbors=15, random_state=RANDOM_STATE),
)

run_test(
    "drift_psi",
    lambda: ds.data_drift_test(dataset1="train", dataset2="test", distance_metric="PSI", psi_bins=10),
)
run_test("drift_wd1", lambda: ds.data_drift_test(dataset1="train", dataset2="test", distance_metric="WD1"))
run_test("drift_ks", lambda: ds.data_drift_test(dataset1="train", dataset2="test", distance_metric="KS"))

run_test(
    "outlier_iforest",
    lambda: ds.detect_outlier_isolation_forest(dataset="train", threshold=0.995, n_estimators=200),
)
run_test(
    "outlier_pca",
    lambda: ds.detect_outlier_pca(dataset="train", threshold=0.995, method="mahalanobis"),
)
run_test(
    "outlier_cblof",
    lambda: ds.detect_outlier_cblof(dataset="train", threshold=0.995, n_clusters=12, method="kmeans"),
)

run_test("fs_corr", lambda: ds.feature_select_corr(dataset="train", threshold=0.80))
run_test("fs_xgbpfi", lambda: ds.feature_select_xgbpfi(dataset="train", threshold=0.01))
run_test(
    "fs_rcit",
    lambda: ds.feature_select_rcit(
        dataset="train", threshold=1e-5, n_fourier=10, n_fourier2=3, n_forwards=1, random_state=RANDOM_STATE
    ),
)
run_test(
    "subsample_random",
    lambda: ds.subsample_random(dataset="train", sample_size=0.25, stratify=target_col, random_state=RANDOM_STATE),
)

# %%
display(Markdown("### Top drift (PSI)"))
psi_table = result_table("drift_psi")
if psi_table is not None:
    display(psi_table.sort_values("Distance_Scores", ascending=False).head(10))

display(Markdown("### Feature selection (XGB-PFI top)"))
fs_xgb = result_table("fs_xgbpfi")
if fs_xgb is not None:
    display(fs_xgb.sort_values("Importance", ascending=False).head(10))

# %% [markdown]
# ## 6) ModelZoo: entrenamiento y tuning

# %%
mz = modeva.ModelZoo(dataset=ds, name="lc_modeva_side_task_zoo", random_state=RANDOM_STATE)
mz.add_model(models.MoLogisticRegression(max_iter=500), name="MoLogReg", replace=True)
mz.add_model(models.MoLGBMClassifier(n_estimators=300, max_depth=6, verbose=-1), name="MoLGBM", replace=True)
mz.add_model(
    models.MoXGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, verbosity=0),
    name="MoXGB",
    replace=True,
)
mz.train_all(silent=True)
leaderboard = mz.leaderboard(order_by="test AUC", ascending=False)
leaderboard

# %%
tuner = models.ModelTuneRandomSearch(ds, models.MoLogisticRegression(max_iter=500))
run_test(
    "tune_random_logreg",
    lambda: tuner.run(
        param_distributions={"C": (0.01, 5.0)},
        dataset="train",
        n_iter=5,
        metric="AUC",
        cv=3,
        random_state=RANDOM_STATE,
    ),
)
result_table("tune_random_logreg")

# %% [markdown]
# ## 7) TestSuite: calidad, robustez, reliability, fairness y slicing

# %%
model_primary = mz.get_model("MoLGBM")
model_list = [mz.get_model(n) for n in mz.list_model_names()]
ts = modeva.TestSuite(dataset=ds, model=model_primary, models=model_list, name="lc_modeva_side_task_suite")

group_cfg = {
    "rent_vs_mortgage": {
        "feature": "home_ownership",
        "protected": "RENT",
        "reference": "MORTGAGE",
    }
}

run_test("diag_accuracy", lambda: ts.diagnose_accuracy_table())
run_test("cmp_accuracy", lambda: ts.compare_accuracy_table())
run_test("diag_reliability", lambda: ts.diagnose_reliability(train_dataset="train", test_dataset="test", alpha=0.1, width_threshold=0.15))
run_test("cmp_reliability", lambda: ts.compare_reliability(train_dataset="train", test_dataset="test", alpha=0.1))
run_test("diag_robustness", lambda: ts.diagnose_robustness(dataset="test", perturb_features=("int_rate", "dti"), noise_levels=(0.02, 0.05, 0.1), n_repeats=5))
run_test("cmp_robustness", lambda: ts.compare_robustness(dataset="test", perturb_features=("int_rate", "dti"), noise_levels=(0.02, 0.05, 0.1), n_repeats=5))
run_test("diag_resilience", lambda: ts.diagnose_resilience(dataset="test", method="worst-sample", alphas=(0.1, 0.2, 0.3)))
run_test("cmp_resilience", lambda: ts.compare_resilience(dataset="test", method="worst-sample", alphas=(0.1, 0.2, 0.3)))
run_test("diag_residual_analysis", lambda: ts.diagnose_residual_analysis(features="int_rate", dataset="test", sample_size=1500))
run_test("diag_residual_cluster", lambda: ts.diagnose_residual_cluster(dataset="test", n_clusters=8, sample_size=1500))
run_test("cmp_residual_cluster", lambda: ts.compare_residual_cluster(dataset="test", n_clusters=8, sample_size=1500))
run_test("diag_residual_interpret", lambda: ts.diagnose_residual_interpret(dataset="test", n_estimators=80, max_depth=2))
run_test("diag_fairness", lambda: ts.diagnose_fairness(group_config=group_cfg, dataset="test", metric="AIR"))
run_test("cmp_fairness", lambda: ts.compare_fairness(group_config=group_cfg, dataset="test", metric="AIR"))

run_test("diag_slicing_accuracy", lambda: ts.diagnose_slicing_accuracy(features=("int_rate", "fico_score"), dataset="test", bins=5))
run_test("cmp_slicing_accuracy", lambda: ts.compare_slicing_accuracy(features="int_rate", dataset="test", bins=5))
run_test("diag_slicing_overfit", lambda: ts.diagnose_slicing_overfit(features=("int_rate", "fico_score"), train_dataset="train", test_dataset="test", bins=5))
run_test("cmp_slicing_overfit", lambda: ts.compare_slicing_overfit(features="int_rate", train_dataset="train", test_dataset="test", bins=5))
run_test("diag_slicing_robustness", lambda: ts.diagnose_slicing_robustness(features=("int_rate", "fico_score"), dataset="test", bins=5, perturb_features=("int_rate", "dti"), noise_levels=0.05, n_repeats=3))
run_test("cmp_slicing_robustness", lambda: ts.compare_slicing_robustness(features="int_rate", dataset="test", bins=5, perturb_features=("int_rate", "dti"), noise_levels=0.05, n_repeats=3))
run_test("diag_slicing_reliability", lambda: ts.diagnose_slicing_reliability(features=("int_rate", "fico_score"), train_dataset="train", test_dataset="test", bins=5, alpha=0.1))
run_test("cmp_slicing_reliability", lambda: ts.compare_slicing_reliability(features="int_rate", train_dataset="train", test_dataset="test", bins=5, alpha=0.1))
run_test("diag_slicing_fairness", lambda: ts.diagnose_slicing_fairness(group_config=group_cfg, features=("int_rate", "fico_score"), dataset="test", bins=5, metric="AIR"))
run_test("cmp_slicing_fairness", lambda: ts.compare_slicing_fairness(group_config=group_cfg, features="int_rate", dataset="test", bins=5, metric="AIR"))

run_test("diag_mitigate_unfair_thresholding", lambda: ts.diagnose_mitigate_unfair_thresholding(group_config=group_cfg, dataset="test", metric="AIR", performance_metric="AUC", proba_cutoff=(0.30, 0.35, 0.40, 0.45)))
run_test("diag_mitigate_unfair_binning", lambda: ts.diagnose_mitigate_unfair_binning(group_config=group_cfg, dataset="test", metric="AIR", performance_metric="AUC", binning_features=("int_rate", "fico_score"), bins=5))

# %%
display(Markdown("### Compare Accuracy"))
display(result_table("cmp_accuracy"))
display(Markdown("### Compare Reliability"))
display(result_table("cmp_reliability"))
display(Markdown("### Compare Fairness (AIR)"))
display(result_table("cmp_fairness"))

# %% [markdown]
# ## 8) Explainability e interpretabilidad

# %%
run_test("explain_pfi", lambda: ts.explain_pfi(dataset="test", sample_size=2000, n_repeats=5))
run_test("explain_shap", lambda: ts.explain_shap(dataset="test", sample_index=25, baseline_sample_size=300))
run_test("explain_lime", lambda: ts.explain_lime(dataset="test", sample_index=25))
run_test("explain_pdp", lambda: ts.explain_pdp(features=("int_rate", "loan_to_income"), dataset="test", sample_size=2000))
run_test("explain_ale", lambda: ts.explain_ale(features=("int_rate", "loan_to_income"), dataset="test", sample_size=2000))
run_test("explain_hstat", lambda: ts.explain_hstatistic(features=("int_rate", "fico_score", "loan_to_income"), dataset="test", sample_size=1500))

ts_logreg = modeva.TestSuite(dataset=ds, model=mz.get_model("MoLogReg"), models=model_list, name="lc_modeva_side_task_suite_logreg")
run_test("interpret_coef", lambda: ts_logreg.interpret_coef())
run_test("interpret_effects", lambda: ts_logreg.interpret_effects())
run_test("interpret_ei", lambda: ts_logreg.interpret_ei(dataset="test"))
run_test("interpret_local_ei", lambda: ts_logreg.interpret_local_ei(dataset="test", sample_index=25))
run_test("interpret_fi", lambda: ts_logreg.interpret_fi(dataset="test"))
run_test("interpret_local_fi", lambda: ts_logreg.interpret_local_fi(dataset="test", sample_index=25))
run_test("interpret_local_linear_fi", lambda: ts_logreg.interpret_local_linear_fi(dataset="test", sample_index=25))

mz_tree = modeva.ModelZoo(dataset=ds, name="lc_modeva_side_task_tree", random_state=RANDOM_STATE)
mz_tree.add_model(models.MoDecisionTreeClassifier(max_depth=4), name="MoTree", replace=True)
mz_tree.train("MoTree")
ts_tree = modeva.TestSuite(dataset=ds, model=mz_tree.get_model("MoTree"), name="lc_modeva_side_task_suite_tree")
run_test("interpret_global_tree", lambda: ts_tree.interpret_global_tree())
run_test("interpret_local_tree", lambda: ts_tree.interpret_local_tree(dataset="test", sample_index=25))

# %%
display(Markdown("### PFI top"))
pfi_table = result_table("explain_pfi")
if pfi_table is not None:
    display(pfi_table.sort_values("Importance", ascending=False).head(15))

display(Markdown("### SHAP local (sample 25) top |effect|"))
shap_table = result_table("explain_shap")
if shap_table is not None:
    display(shap_table.assign(abs_effect=shap_table["Effect"].abs()).sort_values("abs_effect", ascending=False).head(15))

display(Markdown("### Coeficientes MoLogReg"))
coef_table = result_table("interpret_coef")
if coef_table is not None:
    display(coef_table.sort_values("Coefficients", ascending=False).head(15))

# %% [markdown]
# ## 9) Export reporte Modeva + auditoria de ejecucion

# %%
report_path = REPORT_DIR / "modeva_side_task_report.html"
run_test("export_report", lambda: ts.export_report(path=str(report_path)))

run_df = pd.DataFrame(run_log)
print("status counts")
display(run_df["status"].value_counts())
print("report path:", report_path)
run_df.sort_values(["status", "test"]).reset_index(drop=True)

# %% [markdown]
# ## 10) Insights de resultados y aporte potencial

# %%
# Performance por modelo en test
cmp_accuracy = result_table("cmp_accuracy")
model_perf_rows = []
if cmp_accuracy is not None:
    model_names = cmp_accuracy.columns.get_level_values(0).unique()
    for model_name in model_names:
        model_perf_rows.append(
            {
                "model": model_name,
                "test_auc": float(cmp_accuracy.loc["test", (model_name, "AUC")]),
                "test_brier": float(cmp_accuracy.loc["test", (model_name, "Brier")]),
                "gap_auc_test_minus_train": float(cmp_accuracy.loc["GAP", (model_name, "AUC")]),
            }
        )
model_perf = pd.DataFrame(model_perf_rows).sort_values("test_auc", ascending=False)
display(Markdown("### Performance comparada (Modeva sample)"))
display(model_perf)

# Drift top
drift_top = None
psi_table = result_table("drift_psi")
if psi_table is not None:
    drift_top = psi_table.sort_values("Distance_Scores", ascending=False).head(10)
display(Markdown("### Top drift features (PSI)"))
display(drift_top)

# Outlier rates
def outlier_rate(result_name):
    out = results.get(result_name)
    if out is None:
        return np.nan
    table = getattr(out, "table", None)
    if not isinstance(table, dict):
        return np.nan
    n_out = len(table.get("outliers", []))
    n_non = len(table.get("non-outliers", []))
    return n_out / (n_out + n_non) if (n_out + n_non) > 0 else np.nan


outlier_summary = pd.DataFrame(
    {
        "method": ["IsolationForest", "PCA-Mahalanobis", "CBLOF"],
        "outlier_rate": [
            outlier_rate("outlier_iforest"),
            outlier_rate("outlier_pca"),
            outlier_rate("outlier_cblof"),
        ],
    }
)
display(Markdown("### Outlier rate por metodo"))
display(outlier_summary)

# Feature selection consensus
fs_corr = result_table("fs_corr")
fs_xgb = result_table("fs_xgbpfi")
fs_rcit_value = result_value("fs_rcit")
fs_corr_sel = set(fs_corr.loc[fs_corr["Selected"] == True, "Name"].tolist()) if fs_corr is not None else set()
fs_xgb_sel = set(fs_xgb.loc[fs_xgb["Selected"] == True, "Name"].tolist()) if fs_xgb is not None else set()
fs_rcit_sel = set(fs_rcit_value.get("selected", [])) if isinstance(fs_rcit_value, dict) else set()
fs_consensus = sorted((fs_corr_sel & fs_xgb_sel) | fs_rcit_sel)

fs_summary = pd.DataFrame(
    {
        "selected_by_corr": [len(fs_corr_sel)],
        "selected_by_xgbpfi": [len(fs_xgb_sel)],
        "selected_by_rcit": [len(fs_rcit_sel)],
        "consensus_plus_rcit": [", ".join(fs_consensus[:20])],
    }
)
display(Markdown("### Feature selection resumen"))
display(fs_summary)

# Reliability / fairness / robustness summaries
reliability_table = result_table("cmp_reliability")
reliability_rows = []
if reliability_table is not None:
    for model_name in reliability_table.columns.get_level_values(0).unique():
        reliability_rows.append(
            {
                "model": model_name,
                "avg_width": float(reliability_table.loc[0, (model_name, "Avg.Width")]),
                "avg_coverage": float(reliability_table.loc[0, (model_name, "Avg.Coverage")]),
            }
        )
reliability_summary = pd.DataFrame(reliability_rows)
display(Markdown("### Reliability resumen"))
display(reliability_summary)

fairness_table = result_table("cmp_fairness")
fairness_rows = []
if fairness_table is not None:
    for model_name in fairness_table.columns.get_level_values(0).unique():
        fairness_rows.append(
            {
                "model": model_name,
                "AIR_rent_vs_mortgage": float(fairness_table.loc["AIR", (model_name, "rent_vs_mortgage")]),
            }
        )
fairness_summary = pd.DataFrame(fairness_rows)
display(Markdown("### Fairness resumen (AIR)"))
display(fairness_summary)

robustness_table = result_table("cmp_robustness")
robustness_rows = []
if robustness_table is not None:
    robust_grouped = robustness_table.groupby(level=[0, 1], axis=1).mean()
    noise_levels = sorted(set(robust_grouped.columns.get_level_values(1).tolist()))
    max_noise = max(noise_levels)
    for model_name in robust_grouped.columns.get_level_values(0).unique():
        base = float(robust_grouped[(model_name, 0.0)].mean())
        high_noise = float(robust_grouped[(model_name, max_noise)].mean())
        robustness_rows.append(
            {
                "model": model_name,
                "metric_at_noise_0.0": base,
                f"metric_at_noise_{max_noise}": high_noise,
                "delta_high_noise_minus_base": high_noise - base,
            }
        )
robustness_summary = pd.DataFrame(robustness_rows)
display(Markdown("### Robustness resumen"))
display(robustness_summary)

# Compare best Modeva AUC vs core PD AUC
core_auc = float(pd_record["final_test_metrics"]["auc_roc"])
best_modeva_auc = float(model_perf["test_auc"].max()) if not model_perf.empty else np.nan
auc_gap_vs_core = best_modeva_auc - core_auc if not np.isnan(best_modeva_auc) else np.nan

comparison_vs_core = pd.DataFrame(
    [{"core_pd_auc": core_auc, "best_modeva_auc_sample": best_modeva_auc, "modeva_minus_core_auc": auc_gap_vs_core}]
)
display(Markdown("### Comparacion AUC vs baseline core"))
display(comparison_vs_core)

# %%
failures = run_df.loc[run_df["status"] == "error", ["test", "error"]].copy()

lines = []
lines.append("## Conclusiones de la side task (Modeva)")
lines.append("")
lines.append(f"- Metodos OK: {int((run_df['status'] == 'ok').sum())}")
lines.append(f"- Metodos con error controlado: {int((run_df['status'] == 'error').sum())}")
if not model_perf.empty:
    best = model_perf.iloc[0]
    lines.append(f"- Mejor AUC en muestra Modeva: {best['model']} = {best['test_auc']:.4f}")
    lines.append(f"- Brecha vs AUC core ({core_auc:.4f}): {auc_gap_vs_core:+.4f}")
if drift_top is not None and not drift_top.empty:
    lines.append(f"- Mayor drift PSI: {drift_top.index[0]} ({float(drift_top.iloc[0]['Distance_Scores']):.4f})")
if not fairness_summary.empty:
    lines.append(
        f"- AIR rent_vs_mortgage (min/max): {fairness_summary['AIR_rent_vs_mortgage'].min():.4f} / {fairness_summary['AIR_rent_vs_mortgage'].max():.4f}"
    )
if not failures.empty:
    lines.append("- Limites observados en modeva:")
    for _, row in failures.iterrows():
        lines.append(f"  - {row['test']}: {row['error']}")

lines.append("")
lines.append("Aporte potencial al proyecto:")
lines.append("- Modeva aporta una bateria muy amplia de validaciones de gobernanza (drift, slicing, fairness, robustez, reliability, interpretabilidad).")
lines.append("- En esta muestra no supera el baseline de AUC del modelo canonico; no se recomienda reemplazo del core en este estado.")
lines.append("- Se recomienda uso complementario para auditoria/monitoring del modelo core.")
lines.append("")
lines.append("Esta entrega se considera side task y no parte del core, salvo que en futuras pruebas muestre mejoras materiales.")

display(Markdown("\\n".join(lines)))

# %% [markdown]
# ## 11) Recomendacion explicita: que llevar al core vs que descartar

# %%
method_decisions = [
    {
        "method_group": "data_drift_test (PSI/KS)",
        "status": "RECOMENDADO",
        "decision": "Incluir como monitoreo periodico",
        "justification": "Entrega alerta temprana de cambio distribucional en variables de riesgo.",
    },
    {
        "method_group": "diagnose_fairness / diagnose_slicing_fairness",
        "status": "RECOMENDADO",
        "decision": "Incluir como control de gobernanza",
        "justification": "Detecta sesgos por subgrupo y soporta discusiones de politica de credito.",
    },
    {
        "method_group": "diagnose_robustness / diagnose_slicing_robustness",
        "status": "RECOMENDADO",
        "decision": "Incluir como prueba de estres",
        "justification": "Cuantifica degradacion ante ruido en variables sensibles.",
    },
    {
        "method_group": "diagnose_reliability / slicing_reliability",
        "status": "RECOMENDADO",
        "decision": "Incluir como control complementario",
        "justification": "Aporta una vista adicional de estabilidad de prediccion por segmentos.",
    },
    {
        "method_group": "diagnose_slicing_accuracy / overfit",
        "status": "RECOMENDADO",
        "decision": "Incluir en reporte de validacion",
        "justification": "Hace visible donde el modelo rinde peor por cortes de negocio.",
    },
    {
        "method_group": "detect_outlier_isolation_forest",
        "status": "RECOMENDADO",
        "decision": "Usar como indicador de calidad de entrada",
        "justification": "Ayuda a monitorear estabilidad de poblacion y datos anomales.",
    },
    {
        "method_group": "explain_pfi / explain_shap / explain_lime",
        "status": "RECOMENDADO",
        "decision": "Mantener para auditoria y explicabilidad",
        "justification": "Facilita interpretacion de drivers globales y casos individuales.",
    },
    {
        "method_group": "ModelZoo para reentrenar modelos alternos",
        "status": "DESCARTAR (por ahora)",
        "decision": "No usar para reemplazar pipeline core",
        "justification": "En este benchmark no supero al modelo canonico en AUC.",
    },
    {
        "method_group": "interpret_local_ei (logistic wrapper)",
        "status": "DESCARTAR (bug de libreria)",
        "decision": "No usar hasta correccion upstream",
        "justification": "Falla por atributo faltante (`modeva_intercept_`).",
    },
    {
        "method_group": "transform / inverse_transform con binning",
        "status": "DESCARTAR (inestable)",
        "decision": "Evitar en flujo productivo",
        "justification": "Se observaron errores de shape tras pipeline de binning.",
    },
    {
        "method_group": "eda_umap / visualizaciones pesadas",
        "status": "DESCARTAR (core)",
        "decision": "Usar solo para analisis exploratorio",
        "justification": "Valor operativo bajo para gate de produccion.",
    },
]

decision_df = pd.DataFrame(method_decisions)
display(decision_df)

core_recommended = decision_df[decision_df["status"].str.contains("RECOMENDADO")]
core_discarded = decision_df[~decision_df["status"].str.contains("RECOMENDADO")]

display(Markdown("### Lista corta para el proyecto principal"))
display(core_recommended[["method_group", "decision"]])

display(Markdown("### Lista a descartar/no priorizar en el core"))
display(core_discarded[["method_group", "decision"]])
