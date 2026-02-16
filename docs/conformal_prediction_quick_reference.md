# Conformal Prediction Quick Reference - MAPIE 1.3.0

**Project:** Lending Club Risk Analysis
**Last Updated:** 2026-02-07

## Nota de estado (2026-02-16)

- Este archivo es una guía rápida de implementación MAPIE, no una foto exacta del serving actual.
- En este proyecto, los niveles conformal operativos vigentes son:
  - `alpha=0.10` (90%)
  - `alpha=0.05` (95%)
- La referencia canónica para resultados actuales es:
  - `models/conformal_results_mondrian.pkl`
  - `data/processed/conformal_intervals_mondrian.parquet`

---

## Installation

```bash
# Already installed in your project
uv sync  # MAPIE 1.3.0 is in pyproject.toml
```

---

## Basic Workflow

### 1. PD Regression Intervals (Your Main Use Case)

```python
from mapie.regression import SplitConformalRegressor
from src.models.conformal import ProbabilityRegressor

# Step 1: Wrap calibrated classifier
prob_reg = ProbabilityRegressor(catboost_model)

# Step 2: Create conformal predictor
mapie = SplitConformalRegressor(
    estimator=prob_reg,
    confidence_level=0.90,  # 1 - alpha
    conformity_score='absolute',  # or 'gamma', 'residual_normalized'
    prefit=True,
    n_jobs=-1
)

# Step 3: Fit (no-op when prefit=True)
mapie.fit(X_cal, y_cal.astype(float))

# Step 4: Conformalize (compute calibration scores)
mapie.conformalize(X_cal, y_cal.astype(float))

# Step 5: Predict intervals
y_pred = mapie.predict(X_test)
y_intervals = mapie.predict_interval(X_test)

# Shape: y_intervals is (n_samples, 2)
# y_intervals[:, 0] = lower bound
# y_intervals[:, 1] = upper bound

# Clip to [0, 1] for probabilities
y_intervals = np.clip(y_intervals, 0, 1)
```

---

### 2. Classification Sets (Binary: Default/Non-Default)

```python
from mapie.classification import SplitConformalClassifier

mapie_clf = SplitConformalClassifier(
    estimator=catboost_model,
    confidence_level=0.90,
    conformity_score='lac',  # or 'aps', 'raps'
    prefit=True
)

mapie_clf.fit(X_cal, y_cal)
mapie_clf.conformalize(X_cal, y_cal)

y_pred = mapie_clf.predict(X_test)
y_sets = mapie_clf.predict_set(X_test)

# Shape: y_sets is (n_samples, 2) for binary classification
# y_sets[i, 0] = 1 if class 0 is in prediction set
# y_sets[i, 1] = 1 if class 1 is in prediction set

# Singleton rate (confident predictions)
singleton_rate = (y_sets.sum(axis=1).squeeze() == 1).mean()
```

---

### 3. Multiple Alpha Levels

```python
# MAPIE 1.3 allows multiple confidence levels at once
mapie = SplitConformalRegressor(
    estimator=prob_reg,
    confidence_level=[0.80, 0.90, 0.95],  # Multiple levels
    prefit=True
)

mapie.fit(X_cal, y_cal)
mapie.conformalize(X_cal, y_cal)

y_intervals_multi = mapie.predict_interval(X_test)

# Shape: (n_samples, n_confidence_levels, 2)
# y_intervals_multi[:, 0, 0] = lower bound for 80% CI
# y_intervals_multi[:, 0, 1] = upper bound for 80% CI
# y_intervals_multi[:, 1, 0] = lower bound for 90% CI
# y_intervals_multi[:, 1, 1] = upper bound for 90% CI
# y_intervals_multi[:, 2, 0] = lower bound for 95% CI
# y_intervals_multi[:, 2, 1] = upper bound for 95% CI
```

---

## Conformity Scores

### Regression

| Score | When to Use | Formula |
|-------|-------------|---------|
| `'absolute'` | Default, homoscedastic | \|y - ŷ\| |
| `'gamma'` | Normalized residuals | \|y - ŷ\| / σ̂ |
| `'residual_normalized'` | Heteroscedastic | \|y - ŷ\| / ŝ(x) |

```python
# Example: Residual normalized for heteroscedastic PD
mapie = SplitConformalRegressor(
    estimator=prob_reg,
    conformity_score='residual_normalized',
    confidence_level=0.90,
    prefit=True
)
```

### Classification

| Score | Best For | Description |
|-------|----------|-------------|
| `'lac'` | Binary, well-calibrated | Least Ambiguous Criterion |
| `'aps'` | Multi-class (3+ classes) | Adaptive Prediction Sets |
| `'raps'` | Multi-class with imbalance | Regularized APS |

```python
# Example: LAC for binary PD classification
mapie_clf = SplitConformalClassifier(
    estimator=catboost_model,
    conformity_score='lac',
    confidence_level=0.90,
    prefit=True
)
```

---

## Validation Metrics

### Coverage

```python
def validate_coverage(y_true, y_intervals, alpha):
    low, high = y_intervals[:, 0], y_intervals[:, 1]
    covered = ((y_true >= low) & (y_true <= high)).mean()
    target = 1 - alpha
    gap = abs(covered - target)

    # Coverage should be ≈ target (within ±2%)
    return {
        'empirical_coverage': covered,
        'target_coverage': target,
        'coverage_gap': gap,
        'pass': gap < 0.02
    }
```

### Efficiency

```python
def efficiency_metrics(y_intervals):
    widths = y_intervals[:, 1] - y_intervals[:, 0]

    return {
        'avg_width': widths.mean(),
        'median_width': np.median(widths),
        'width_std': widths.std(),
        'width_iqr': np.percentile(widths, 75) - np.percentile(widths, 25),
        'efficiency_score': 1 / (1 + widths.mean())  # Higher = better
    }

# Target: avg_width < 0.30 for production
```

### Conditional Coverage (By Subgroup)

```python
def coverage_by_group(y_true, y_intervals, X_test, feature='grade'):
    groups = X_test[feature].unique()
    results = {}

    for group in groups:
        mask = (X_test[feature] == group)
        low, high = y_intervals[mask, 0], y_intervals[mask, 1]
        coverage = ((y_true[mask] >= low) & (y_true[mask] <= high)).mean()
        width = (high - low).mean()

        results[group] = {
            'n': mask.sum(),
            'coverage': coverage,
            'avg_width': width
        }

    return pd.DataFrame(results).T
```

---

## Common Patterns

### Pattern 1: Standard Split Conformal

```python
from src.models.conformal import create_pd_intervals, validate_coverage

# Generate intervals
y_pred, y_intervals = create_pd_intervals(
    classifier=model,
    X_cal=X_cal,
    y_cal=y_cal,
    X_test=X_test,
    alpha=0.10
)

# Validate
metrics = validate_coverage(y_test, y_intervals, alpha=0.10)
print(f"Coverage: {metrics['empirical_coverage']:.2%} (target: {metrics['target_coverage']:.0%})")
print(f"Avg Width: {metrics['avg_interval_width']:.4f}")
```

### Pattern 2: Cross-Conformal (K-Fold)

```python
from mapie.regression import CrossConformalRegressor

# When you don't want to sacrifice calibration data
mapie_cv = CrossConformalRegressor(
    estimator=ProbabilityRegressor(catboost_model),
    confidence_level=0.90,
    cv=5,  # 5-fold cross-validation
    prefit=False  # Will train 5 models
)

mapie_cv.fit(X_train, y_train)
y_intervals_cv = mapie_cv.predict_interval(X_test)

# Trade-off: More data, but 5x slower
```

### Pattern 3: Conformalized Quantile Regression

```python
from mapie.regression import ConformalizedQuantileRegressor
from lightgbm import LGBMRegressor

# For heteroscedastic data (varying uncertainty)
qr_low = LGBMRegressor(objective='quantile', alpha=0.05)
qr_high = LGBMRegressor(objective='quantile', alpha=0.95)

qr_low.fit(X_train, y_train)
qr_high.fit(X_train, y_train)

# Conformalize
mapie_cqr = ConformalizedQuantileRegressor(
    estimator=qr_low,  # Or qr_high
    confidence_level=0.90
)
mapie_cqr.fit(X_cal, y_cal)
y_intervals_cqr = mapie_cqr.predict_interval(X_test)
```

### Pattern 4: Mondrian Conformal (Group-Conditional)

```python
# Separate calibration per risk grade
grades = X_cal['grade'].unique()
intervals_by_grade = {}

for grade in grades:
    # Filter by grade
    mask_cal = (X_cal['grade'] == grade)
    mask_test = (X_test['grade'] == grade)

    # Create grade-specific predictor
    mapie_grade = SplitConformalRegressor(
        estimator=ProbabilityRegressor(model),
        confidence_level=0.90,
        prefit=True
    )

    # Calibrate on grade subset
    X_cal_grade = X_cal[mask_cal].drop(columns=['grade'])
    y_cal_grade = y_cal[mask_cal]

    mapie_grade.fit(X_cal_grade, y_cal_grade)
    mapie_grade.conformalize(X_cal_grade, y_cal_grade)

    # Predict on grade subset
    X_test_grade = X_test[mask_test].drop(columns=['grade'])
    intervals_by_grade[grade] = mapie_grade.predict_interval(X_test_grade)

# Combine results (sorted by test index)
final_intervals = np.vstack([intervals_by_grade[g] for g in grades])
```

---

## Integration with Optimization

### Using Intervals as Uncertainty Sets

```python
from src.optimization.robust_opt import build_box_uncertainty_set, worst_case_expected_loss

# Build uncertainty set
uncertainty = build_box_uncertainty_set(
    pd_low=y_intervals[:, 0],
    pd_high=y_intervals[:, 1]
)

# Worst-case loss
wcl = worst_case_expected_loss(
    allocation=x_opt,  # From Pyomo solution
    loan_amounts=loan_amounts,
    pd_high=uncertainty['pd_high'],
    lgd_point=lgd_estimates
)

print(f"Worst-case expected loss: ${wcl:,.2f}")
```

---

## Visualization

### Interval Width Distribution

```python
import matplotlib.pyplot as plt

widths = y_intervals[:, 1] - y_intervals[:, 0]

plt.figure(figsize=(10, 6))
plt.hist(widths, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(widths.mean(), color='red', linestyle='--',
            label=f'Mean: {widths.mean():.3f}')
plt.axvline(widths.median(), color='blue', linestyle='--',
            label=f'Median: {widths.median():.3f}')
plt.xlabel('Interval Width')
plt.ylabel('Frequency')
plt.title('Conformal Interval Width Distribution')
plt.legend()
plt.show()
```

### Coverage by Grade

```python
import seaborn as sns

coverage_df = coverage_by_group(y_test, y_intervals, X_test, feature='grade')

fig, ax = plt.subplots(figsize=(10, 6))
coverage_df['coverage'].plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
ax.axhline(0.90, color='red', linestyle='--', label='Target (90%)')
ax.set_xlabel('Grade')
ax.set_ylabel('Empirical Coverage')
ax.set_title('Coverage by Risk Grade')
ax.legend()
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

### Prediction Intervals Plot

```python
def plot_intervals(y_true, y_pred, y_intervals, n=100):
    idx = np.argsort(y_pred)[:n]
    low, high = y_intervals[idx, 0], y_intervals[idx, 1]

    plt.figure(figsize=(14, 6))
    plt.fill_between(range(n), low, high, alpha=0.3, label='90% PI')
    plt.scatter(range(n), y_true[idx], s=10, c='red', label='True', zorder=5)
    plt.plot(range(n), y_pred[idx], 'b-', linewidth=1, label='Predicted')
    plt.xlabel('Sample (sorted by prediction)')
    plt.ylabel('Probability of Default')
    plt.title('Conformal Prediction Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_intervals(y_test, y_pred, y_intervals, n=200)
```

---

## Troubleshooting

### Issue 1: Coverage Too Low

```python
# Check calibration set size
print(f"Calibration set: {len(X_cal)} samples")
# Need at least 100+ samples for alpha=0.10

# Check for distribution shift
from scipy.stats import ks_2samp
ks_stat, p_value = ks_2samp(y_cal, y_test)
print(f"KS test: stat={ks_stat:.4f}, p={p_value:.4f}")
# If p < 0.05, distributions differ (conformal may not hold)
```

### Issue 2: Intervals Too Wide

```python
# 1. Check base model calibration
from src.models.calibration import expected_calibration_error
ece = expected_calibration_error(y_cal, model.predict_proba(X_cal)[:, 1])
print(f"ECE: {ece:.4f}")  # Should be < 0.05

# 2. Try CQR for heteroscedastic data
# 3. Use Mondrian CP for subgroup-specific widths
# 4. Consider more calibration data
```

### Issue 3: Shape Mismatch

```python
# MAPIE 1.3 returns (n, 2), not (n, 2, 1)
# If you get shape errors, check:
print(y_intervals.shape)  # Should be (n_samples, 2)

# Extract bounds
low = y_intervals[:, 0]
high = y_intervals[:, 1]

# NOT y_intervals[:, 0, 0] (this is for multiple confidence levels)
```

---

## Performance Tips

### Tip 1: Use prefit=True

```python
# Faster: model already trained
mapie = SplitConformalRegressor(estimator=model, prefit=True)

# vs slower: MAPIE will train the model
mapie = SplitConformalRegressor(estimator=model, prefit=False)
```

### Tip 2: Parallelize with n_jobs

```python
# Use all cores for conformalization
mapie = SplitConformalRegressor(
    estimator=model,
    prefit=True,
    n_jobs=-1  # All cores
)
```

### Tip 3: Cache Calibration Results

```python
import joblib

# After conformalization, save
joblib.dump(mapie, 'models/mapie_pd_calibrated.pkl')

# Load and predict
mapie = joblib.load('models/mapie_pd_calibrated.pkl')
y_intervals = mapie.predict_interval(X_new)
```

---

## Decision Tree: Which Method to Use?

```
START
|
├─ Binary classification (default/non-default)?
│  ├─ YES → Want prediction sets?
│  │       ├─ YES → SplitConformalClassifier(conformity_score='lac')
│  │       └─ NO  → Want probability intervals?
│  │               └─ YES → SplitConformalRegressor + ProbabilityRegressor
│  │
│  └─ NO → Multi-class (3+ classes)?
│          └─ YES → SplitConformalClassifier(conformity_score='aps' or 'raps')
|
├─ Regression on continuous target (LGD, EAD)?
│  ├─ Homoscedastic (constant variance)?
│  │   └─ YES → SplitConformalRegressor(conformity_score='absolute')
│  │
│  └─ Heteroscedastic (varying variance)?
│      └─ YES → ConformalizedQuantileRegressor OR
│              SplitConformalRegressor(conformity_score='residual_normalized')
|
└─ Limited calibration data?
   └─ YES → CrossConformalRegressor (k-fold) instead of Split
```

---

## References

- **MAPIE Docs:** https://mapie.readthedocs.io/en/stable/
- **Your Implementation:** `src/models/conformal.py`
- **Metrics:** `src/evaluation/metrics.py`
- **Optimization:** `src/optimization/robust_opt.py`

---

**Quick Start Checklist:**

- [ ] Import `SplitConformalRegressor` or `SplitConformalClassifier`
- [ ] Wrap classifier with `ProbabilityRegressor` (if PD intervals)
- [ ] Set `confidence_level = 1 - alpha` in `__init__`
- [ ] Set `prefit=True` if model already trained
- [ ] Call `.fit()` then `.conformalize()` on calibration set
- [ ] Call `.predict_interval()` or `.predict_set()` on test set
- [ ] Validate coverage: `empirical ≈ target ± 2%`
- [ ] Check efficiency: `avg_width < 0.30`
- [ ] Visualize: interval widths, coverage by grade
- [ ] Save intervals for optimization

---

**End of Quick Reference**
