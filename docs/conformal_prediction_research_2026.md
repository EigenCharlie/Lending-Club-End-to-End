# Conformal Prediction for Credit Risk: Research Summary 2024-2026

**Generated:** 2026-02-07
**MAPIE Version Installed:** 1.3.0
**Project:** Lending Club Risk - Master's Thesis

## Nota de vigencia (2026-02-16)

Este documento preserva investigación y contexto técnico de febrero 2026.  
Para métricas y decisiones operativas del proyecto actual, usar como fuente de verdad:

- `models/conformal_results_mondrian.pkl`
- `data/processed/conformal_intervals_mondrian.parquet`
- `models/conformal_policy_status.json`

---

## Executive Summary

This document consolidates recent best practices for conformal prediction in credit risk modeling, specifically for PD (Probability of Default) prediction intervals. Since web search was unavailable, this is based on:
1. MAPIE 1.3.0 installed API inspection
2. Your project's current implementation in `src/models/conformal.py`
3. Industry knowledge as of January 2025

---

## 1. MAPIE 1.3.0 API - Current State

### Key Classes (Verified in Your Environment)

#### **Regression Conformal Predictors**
```python
from mapie.regression import SplitConformalRegressor
from mapie.regression import CrossConformalRegressor
from mapie.regression import ConformalizedQuantileRegressor
```

**`SplitConformalRegressor` Signature:**
```python
SplitConformalRegressor(
    estimator: RegressorMixin = LinearRegression(),
    confidence_level: Union[float, Iterable[float]] = 0.9,
    conformity_score: Union[str, BaseRegressionScore] = 'absolute',
    prefit: bool = True,
    n_jobs: Optional[int] = None,
    verbose: int = 0
)
```

**Methods:**
- `.fit(X, y)` - Fit the estimator (no-op if prefit=True)
- `.conformalize(X_cal, y_cal)` - Compute nonconformity scores on calibration set
- `.predict(X)` - Point predictions
- `.predict_interval(X)` - Returns shape (n_samples, 2) array: [:, 0] = lower, [:, 1] = upper

#### **Classification Conformal Predictors**
```python
from mapie.classification import SplitConformalClassifier
from mapie.classification import CrossConformalClassifier
```

**`SplitConformalClassifier` Signature:**
```python
SplitConformalClassifier(
    estimator: ClassifierMixin = LogisticRegression(),
    confidence_level: Union[float, Iterable[float]] = 0.9,
    conformity_score: Union[str, BaseClassificationScore] = 'lac',
    prefit: bool = True,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    random_state: Optional[Union[int, np.random.RandomState]] = None
)
```

**Methods:**
- `.fit(X, y)` - Fit the estimator
- `.conformalize(X_cal, y_cal)` - Compute calibration scores
- `.predict(X)` - Point predictions
- `.predict_set(X)` - Returns boolean array (n_samples, n_classes) indicating set membership

### Conformity Score Functions Available

**Classification (verified):**
```python
from mapie.conformity_scores import LACConformityScore  # Least Ambiguous Criterion
from mapie.conformity_scores import APSConformityScore  # Adaptive Prediction Sets
from mapie.conformity_scores import RAPSConformityScore # Regularized APS
from mapie.conformity_scores import TopKConformityScore
```

**Regression (verified):**
```python
from mapie.conformity_scores import AbsoluteConformityScore      # |y - ŷ|
from mapie.conformity_scores import GammaConformityScore         # Normalized residuals
from mapie.conformity_scores import ResidualNormalisedScore      # Heteroscedastic
```

### API Migration Notes (1.2 → 1.3)

Your `src/models/conformal.py` correctly documents the migration:

| Old API (< 1.3) | New API (1.3+) |
|-----------------|----------------|
| `MapieRegressor` | `SplitConformalRegressor` |
| `MapieClassifier` | `SplitConformalClassifier` |
| `.fit().predict(alpha=0.1)` | `.fit().conformalize().predict_interval()` |
| `alpha` parameter in predict | `confidence_level = 1 - alpha` in `__init__` |
| `cv="prefit"` | `prefit=True` |

**Your implementation is CORRECT for MAPIE 1.3.0.**

---

## 2. Conformal Prediction for PD Modeling - Best Practices

### Why Conformal Prediction for Credit Risk?

Traditional approaches have critical weaknesses:
- **Point estimates**: No uncertainty quantification → fragile portfolios
- **Bootstrap intervals**: No finite-sample guarantees, computationally expensive
- **Bayesian intervals**: Require strong distributional assumptions
- **Conformal intervals**:
  - Distribution-free
  - Finite-sample coverage guarantees: P(y ∈ [ŷ_low, ŷ_high]) ≥ 1 - α
  - Computational efficiency

### Your Thesis Innovation

```
CatBoost PD Model → Isotonic Calibration → MAPIE Conformal Intervals →
  Box Uncertainty Sets → Robust Portfolio Optimization (Pyomo + HiGHS)
```

**Key differentiator**: Using conformal intervals as uncertainty sets in optimization (predict-then-optimize framework).

### ProbabilityRegressor Pattern (Your Implementation)

Since MAPIE expects a regressor but CatBoost outputs probabilities via `.predict_proba()`, you use a wrapper:

```python
class ProbabilityRegressor(BaseEstimator, RegressorMixin):
    """Turns classifier predict_proba into regressor predict."""
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        return self  # Already fitted

    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1]  # P(default)
```

**This is the standard pattern for conformal prediction on probabilities.**

---

## 3. Key Techniques for PD Conformal Prediction

### Split Conformal (Current Implementation)

**Your code in `src/models/conformal.py` lines 46-91:**
```python
def create_pd_intervals(classifier, X_cal, y_cal, X_test, alpha=0.1):
    prob_reg = ProbabilityRegressor(classifier)
    mapie = SplitConformalRegressor(
        estimator=prob_reg,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.fit(X_cal, y_cal.astype(float))
    mapie.conformalize(X_cal, y_cal.astype(float))

    y_pred = mapie.predict(X_test)
    y_intervals = mapie.predict_interval(X_test)

    # Clip to [0, 1] for probabilities
    y_intervals = np.clip(y_intervals, 0, 1)
    return y_pred, y_intervals
```

**Strengths:**
- Simple, efficient (single calibration set)
- Marginal coverage guarantee: covers 1-α of test points
- Fast inference

**Weaknesses:**
- Requires separate calibration set (reduces training data)
- Only marginal coverage (not conditional on features)

### Cross-Conformal (Alternative)

Use when you don't want to sacrifice calibration data:

```python
from mapie.regression import CrossConformalRegressor

mapie = CrossConformalRegressor(
    estimator=prob_reg,
    confidence_level=0.9,
    prefit=False,  # Will do k-fold CV internally
    cv=5  # 5-fold cross-validation
)
mapie.fit(X_train, y_train)
y_intervals = mapie.predict_interval(X_test)
```

**Trade-off:** More training data usage but 5x slower (need to train 5 models).

### Conformalized Quantile Regression (CQR)

For **heteroscedastic** data (when interval width should vary with features):

```python
from mapie.regression import ConformalizedQuantileRegressor
from lightgbm import LGBMRegressor

# Train quantile regressor
qr = LGBMRegressor(objective='quantile', alpha=0.05)  # Lower quantile
qr.fit(X_train, y_train)

mapie_cqr = ConformalizedQuantileRegressor(
    estimator=qr,
    confidence_level=0.9
)
mapie_cqr.fit(X_cal, y_cal)
y_intervals = mapie_cqr.predict_interval(X_test)
```

**When to use:** If you notice interval widths should be wider for high-risk segments (e.g., low-grade loans) and narrower for low-risk.

**Your current dataset:** Start with `SplitConformalRegressor` (simpler, interpretable). Consider CQR if you see evidence of heteroscedasticity.

---

## 4. Classification Sets (LAC, APS, RAPS)

### When to Use Each

Your implementation supports all three via `create_classification_sets()`:

```python
def create_classification_sets(classifier, X_cal, y_cal, X_test, alpha=0.1, method='lac'):
    mapie = SplitConformalClassifier(
        estimator=classifier,
        confidence_level=1-alpha,
        conformity_score=method,  # 'lac', 'aps', or 'raps'
        prefit=True
    )
    mapie.conformalize(X_cal, y_cal)
    y_sets = mapie.predict_set(X_test)  # Shape (n, n_classes)
    return y_pred, y_sets
```

#### **LAC (Least Ambiguous Criterion)** - Your Default
- **Score:** 1 - P(y_true)
- **Set:** Include all classes with P(class) ≥ P(y_true) - threshold
- **Best for:** Binary classification with well-calibrated probabilities
- **Singleton rate:** Typically high (70-90%) for binary tasks

#### **APS (Adaptive Prediction Sets)**
- **Score:** Rank-based cumulative probability
- **Set:** Include classes until cumulative prob > 1 - α
- **Best for:** Multi-class problems (3+ classes)
- **Guarantees smallest average set size**

#### **RAPS (Regularized APS)**
- **Score:** APS + regularization penalty
- **Set:** APS with penalty for large sets
- **Best for:** Multi-class with class imbalance
- **Hyperparameter:** Penalty weight (tune on validation set)

**For your binary PD task:** LAC is the right choice. APS/RAPS are overkill for 2 classes.

### Interpretation for Binary Classification

```python
y_sets = mapie.predict_set(X_test)
# y_sets[:, 0] = 1 if class 0 (non-default) is in set
# y_sets[:, 1] = 1 if class 1 (default) is in set

# Singleton: exactly one class
singleton_mask = y_sets.sum(axis=1) == 1
singleton_rate = singleton_mask.mean()

# Empty set (should be rare if well-calibrated)
empty_mask = y_sets.sum(axis=1) == 0

# Full set (uncertain predictions)
full_mask = y_sets.sum(axis=1) == 2
```

**Singleton rate** is the key metric: % of predictions where you're confident about the single class.

---

## 5. Mondrian Conformal Prediction (Group-Conditional Coverage)

**Problem with marginal coverage:** Conformal prediction guarantees 90% coverage *overall*, but not within each risk grade.

**Mondrian CP solution:** Compute separate thresholds for each subgroup.

```python
# Pseudo-code for Mondrian CP (not in MAPIE 1.3 directly)
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
intervals_by_grade = {}

for grade in grades:
    mask_cal = (X_cal['grade'] == grade)
    mask_test = (X_test['grade'] == grade)

    mapie = SplitConformalRegressor(
        estimator=ProbabilityRegressor(classifier),
        confidence_level=0.9,
        prefit=True
    )
    mapie.fit(X_cal[mask_cal], y_cal[mask_cal])
    mapie.conformalize(X_cal[mask_cal], y_cal[mask_cal])

    intervals_by_grade[grade] = mapie.predict_interval(X_test[mask_test])

# Combine intervals
final_intervals = np.concatenate([intervals_by_grade[g] for g in grades])
```

**Trade-off:**
- **Pros:** Conditional coverage within each grade (fairness)
- **Cons:** Requires enough calibration samples per grade; more complex

**Recommendation for thesis:**
1. Start with marginal coverage (simpler, standard)
2. Add Mondrian as "extension" if you want to show group-wise fairness analysis

---

## 6. Validation & Efficiency Metrics

### Coverage Validation (Your Implementation)

`src/models/conformal.py` lines 154-179:
```python
def validate_coverage(y_true, y_intervals, alpha):
    low, high = y_intervals[:, 0], y_intervals[:, 1]
    covered = ((y_true >= low) & (y_true <= high)).mean()
    target = 1 - alpha

    return {
        "empirical_coverage": covered,
        "target_coverage": target,
        "coverage_gap": abs(covered - target),
        "avg_interval_width": (high - low).mean(),
        "median_interval_width": float(np.median(high - low)),
    }
```

**This is correct.** Additional metrics to consider:

```python
def extended_validation(y_true, y_intervals, alpha, X_test, feature_name='grade'):
    """Extended coverage analysis by subgroup."""
    metrics = validate_coverage(y_true, y_intervals, alpha)

    # Coverage by subgroup (conditional coverage check)
    groups = X_test[feature_name].unique()
    coverage_by_group = {}
    for group in groups:
        mask = (X_test[feature_name] == group)
        if mask.sum() > 0:
            low, high = y_intervals[mask, 0], y_intervals[mask, 1]
            coverage_by_group[group] = ((y_true[mask] >= low) & (y_true[mask] <= high)).mean()

    metrics['coverage_by_grade'] = coverage_by_group
    metrics['min_group_coverage'] = min(coverage_by_group.values())
    metrics['max_group_coverage'] = max(coverage_by_group.values())

    # Efficiency: width distribution
    widths = y_intervals[:, 1] - y_intervals[:, 0]
    metrics['width_5th_pct'] = np.percentile(widths, 5)
    metrics['width_95th_pct'] = np.percentile(widths, 95)
    metrics['width_iqr'] = np.percentile(widths, 75) - np.percentile(widths, 25)

    return metrics
```

### Efficiency Analysis

**Efficiency** = How narrow are the intervals while maintaining coverage?

```python
# Efficiency metrics (add to src/evaluation/metrics.py)
def conformal_efficiency(y_intervals, alpha):
    """Quantify prediction interval efficiency."""
    widths = y_intervals[:, 1] - y_intervals[:, 0]

    return {
        'avg_width': widths.mean(),
        'median_width': np.median(widths),
        'std_width': widths.std(),
        'min_width': widths.min(),
        'max_width': widths.max(),
        'width_90th_pct': np.percentile(widths, 90),
        'efficiency_score': 1 / (1 + widths.mean()),  # Higher = better
        'relative_efficiency': widths.std() / widths.mean(),  # Lower = more uniform
    }
```

**Interpretation:**
- **avg_width < 0.2:** Excellent (very informative intervals)
- **avg_width 0.2-0.4:** Good
- **avg_width 0.4-0.6:** Moderate (still useful but wide)
- **avg_width > 0.6:** Poor (intervals too wide, low information)

Your legacy project had `avg_width = 0.808` which was too wide. Target: **< 0.3 for production use**.

---

## 7. Visualization Best Practices

### Interval Width Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_interval_widths(y_intervals, alpha=0.1, feature=None, X_test=None):
    """Plot interval width distribution."""
    widths = y_intervals[:, 1] - y_intervals[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(widths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(widths.mean(), color='red', linestyle='--',
                    label=f'Mean: {widths.mean():.3f}')
    axes[0].axvline(widths.median(), color='blue', linestyle='--',
                    label=f'Median: {widths.median():.3f}')
    axes[0].set_xlabel('Interval Width')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Interval Width Distribution (α={alpha})')
    axes[0].legend()

    # Box plot by feature (if provided)
    if feature is not None and X_test is not None:
        data = pd.DataFrame({
            'width': widths,
            'group': X_test[feature].values
        })
        sns.boxplot(data=data, x='group', y='width', ax=axes[1])
        axes[1].set_title(f'Width by {feature}')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        # CDF
        sorted_widths = np.sort(widths)
        cdf = np.arange(1, len(sorted_widths)+1) / len(sorted_widths)
        axes[1].plot(sorted_widths, cdf)
        axes[1].set_xlabel('Interval Width')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('CDF of Interval Widths')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

### Coverage by Subgroup

```python
def plot_coverage_by_subgroup(y_true, y_intervals, X_test, feature='grade', alpha=0.1):
    """Plot empirical coverage by risk segment."""
    groups = sorted(X_test[feature].unique())
    coverages = []
    counts = []

    for group in groups:
        mask = (X_test[feature] == group)
        low, high = y_intervals[mask, 0], y_intervals[mask, 1]
        coverage = ((y_true[mask] >= low) & (y_true[mask] <= high)).mean()
        coverages.append(coverage)
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(groups))

    bars = ax.bar(x_pos, coverages, alpha=0.7, edgecolor='black')
    ax.axhline(1-alpha, color='red', linestyle='--',
               label=f'Target Coverage ({1-alpha:.0%})')

    # Color bars by coverage gap
    for i, (bar, cov) in enumerate(zip(bars, coverages)):
        if abs(cov - (1-alpha)) > 0.05:
            bar.set_color('orange')
        else:
            bar.set_color('green')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'Coverage by {feature.capitalize()} (α={alpha})')
    ax.set_ylim(0.8, 1.0)
    ax.legend()

    # Add sample counts
    for i, (x, count) in enumerate(zip(x_pos, counts)):
        ax.text(x, 0.82, f'n={count}', ha='center', fontsize=9)

    plt.tight_layout()
    return fig
```

### Prediction Interval Plot (Enhanced)

Your `src/utils/visualization.py` has `plot_conformal_intervals()` but it assumes shape `(n, 2, 1)`. MAPIE 1.3 returns `(n, 2)`. Update:

```python
def plot_conformal_intervals_v2(
    y_true, y_pred, y_intervals, n_samples=100,
    sort_by='prediction', title='Conformal Prediction Intervals'
):
    """Plot intervals with true values (MAPIE 1.3 compatible)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if sort_by == 'prediction':
        idx = np.argsort(y_pred)[:n_samples]
    elif sort_by == 'width':
        widths = y_intervals[:, 1] - y_intervals[:, 0]
        idx = np.argsort(widths)[-n_samples:]  # Widest intervals
    else:
        idx = np.arange(n_samples)

    low = y_intervals[idx, 0]
    high = y_intervals[idx, 1]

    # Coverage indicator
    covered = (y_true[idx] >= low) & (y_true[idx] <= high)

    ax.fill_between(range(len(idx)), low, high, alpha=0.3,
                     label=f'90% PI', color='lightblue')
    ax.scatter(range(len(idx)), y_true[idx], s=15,
               c=['green' if c else 'red' for c in covered],
               label='True (green=covered)', zorder=5)
    ax.plot(range(len(idx)), y_pred[idx], 'b-', linewidth=1.2,
            label='Predicted', alpha=0.8)

    ax.set_xlabel(f'Sample (sorted by {sort_by})')
    ax.set_ylabel('Probability of Default')
    ax.set_title(f'{title} | Coverage: {covered.mean():.1%}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

### Calibration + Conformal Uncertainty

```python
def plot_calibration_with_intervals(y_true, y_pred, y_intervals, n_bins=10):
    """Calibration curve with interval widths."""
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    axes[0].plot(mean_pred, frac_pos, 's-', label='Model')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Interval width vs prediction
    widths = y_intervals[:, 1] - y_intervals[:, 0]
    axes[1].scatter(y_pred, widths, alpha=0.3, s=5)
    axes[1].set_xlabel('Predicted PD')
    axes[1].set_ylabel('Interval Width')
    axes[1].set_title('Uncertainty vs Prediction')
    axes[1].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(y_pred, widths, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    axes[1].plot(x_trend, p(x_trend), 'r--', alpha=0.7, label='Trend')
    axes[1].legend()

    plt.tight_layout()
    return fig
```

---

## 8. Alternative Libraries: CREPES vs MAPIE

### CREPES (Installed: 0.9.0)

**Pros:**
- More flexible API (works with any predictor)
- Supports Venn-ABERS (probability intervals)
- Mondrian CP built-in
- Conformal Predictive Systems (full distributions)

**Cons:**
- Less popular (fewer examples)
- Steeper learning curve
- No scikit-learn integration

**Example:**
```python
from crepes import ConformalRegressor

# Wrap fitted model
cr = ConformalRegressor()
cr.fit(residuals_cal)  # Just needs calibration residuals!

# Predict intervals
y_pred, y_intervals = cr.predict(X_test, confidence=0.9, y_hat=model.predict(X_test))
```

**When to use CREPES:**
- If you want Venn-ABERS for probability intervals
- If you need Mondrian CP without manual implementation
- If MAPIE doesn't support your use case

**For your thesis:** Stick with MAPIE (industry standard, better docs, sklearn integration).

### Nonconformist (Legacy)

**Status:** Not actively maintained (last update 2020). Use MAPIE instead.

---

## 9. Code Patterns for Your Notebooks

### Notebook 04: Conformal Prediction (Implementation Template)

```python
# Cell 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from src.models.conformal import (
    create_pd_intervals,
    validate_coverage,
    create_classification_sets
)
from src.evaluation.metrics import conformal_metrics
from src.utils.visualization import plot_conformal_intervals

# Cell 2: Load data and model
train = pd.read_parquet('data/processed/train.parquet')
test = pd.read_parquet('data/processed/test.parquet')
calibration = pd.read_parquet('data/processed/calibration.parquet')

import joblib
model = joblib.load('models/catboost_pd_calibrated.pkl')

# Cell 3: Split features/target
X_cal = calibration.drop(columns=['default_flag'])
y_cal = calibration['default_flag']
X_test = test.drop(columns=['default_flag'])
y_test = test['default_flag']

# Cell 4: Generate conformal intervals (alpha = 0.10 = 90% coverage)
y_pred, y_intervals_90 = create_pd_intervals(
    classifier=model,
    X_cal=X_cal,
    y_cal=y_cal,
    X_test=X_test,
    alpha=0.10
)

# Cell 5: Validate coverage
metrics_90 = validate_coverage(y_test, y_intervals_90, alpha=0.10)
print(f"Empirical Coverage: {metrics_90['empirical_coverage']:.4f}")
print(f"Target Coverage: {metrics_90['target_coverage']:.4f}")
print(f"Avg Interval Width: {metrics_90['avg_interval_width']:.4f}")

# Cell 6: Try multiple alpha levels
alphas = [0.01, 0.05, 0.10, 0.20]
results = []

for alpha in alphas:
    _, intervals = create_pd_intervals(model, X_cal, y_cal, X_test, alpha)
    metrics = conformal_metrics(y_test, intervals, alpha)
    metrics['alpha'] = alpha
    results.append(metrics)

results_df = pd.DataFrame(results)
print(results_df)

# Cell 7: Efficiency analysis
widths = y_intervals_90[:, 1] - y_intervals_90[:, 0]
print(f"Width distribution:")
print(f"  Min: {widths.min():.4f}")
print(f"  25th: {np.percentile(widths, 25):.4f}")
print(f"  Median: {widths.median():.4f}")
print(f"  75th: {np.percentile(widths, 75):.4f}")
print(f"  95th: {np.percentile(widths, 95):.4f}")
print(f"  Max: {widths.max():.4f}")

# Cell 8: Coverage by risk grade
grades = X_test['grade'].unique()
coverage_by_grade = {}

for grade in sorted(grades):
    mask = (X_test['grade'] == grade)
    low, high = y_intervals_90[mask, 0], y_intervals_90[mask, 1]
    coverage = ((y_test[mask] >= low) & (y_test[mask] <= high)).mean()
    width = (high - low).mean()
    coverage_by_grade[grade] = {
        'n_samples': mask.sum(),
        'coverage': coverage,
        'avg_width': width
    }

coverage_df = pd.DataFrame(coverage_by_grade).T
print("\nCoverage by Grade:")
print(coverage_df)

# Cell 9: Visualizations
fig1 = plot_interval_widths(y_intervals_90, alpha=0.10,
                             feature='grade', X_test=X_test)
plt.show()

fig2 = plot_coverage_by_subgroup(y_test, y_intervals_90, X_test,
                                  feature='grade', alpha=0.10)
plt.show()

fig3 = plot_conformal_intervals_v2(y_test, y_pred, y_intervals_90,
                                    n_samples=200, sort_by='prediction')
plt.show()

# Cell 10: Classification sets (LAC vs APS vs RAPS)
y_pred_lac, y_sets_lac = create_classification_sets(
    model, X_cal, y_cal, X_test, alpha=0.10, method='lac'
)
y_pred_aps, y_sets_aps = create_classification_sets(
    model, X_cal, y_cal, X_test, alpha=0.10, method='aps'
)

singleton_lac = (y_sets_lac.sum(axis=1).squeeze() == 1).mean()
singleton_aps = (y_sets_aps.sum(axis=1).squeeze() == 1).mean()

print(f"\nClassification Sets:")
print(f"  LAC singleton rate: {singleton_lac:.2%}")
print(f"  APS singleton rate: {singleton_aps:.2%}")

# Cell 11: Save intervals for optimization
output = pd.DataFrame({
    'loan_id': test.index,
    'pd_point': y_pred,
    'pd_low': y_intervals_90[:, 0],
    'pd_high': y_intervals_90[:, 1],
    'pd_width': y_intervals_90[:, 1] - y_intervals_90[:, 0],
    'covered': (y_test >= y_intervals_90[:, 0]) & (y_test <= y_intervals_90[:, 1])
})
output.to_parquet('data/processed/conformal_intervals_mondrian.parquet')
logger.info(f"Saved {len(output)} conformal intervals")
```

---

## 10. Integration with Robust Optimization

### Using Conformal Intervals as Uncertainty Sets

Your `src/optimization/robust_opt.py` already implements this:

```python
from src.optimization.robust_opt import build_box_uncertainty_set

# Build uncertainty set from conformal intervals
uncertainty_set = build_box_uncertainty_set(
    pd_low=y_intervals[:, 0],
    pd_high=y_intervals[:, 1]
)

# Use in robust optimization
from src.optimization.portfolio_model import build_portfolio_model

model = build_portfolio_model(
    loan_amounts=loan_amounts,
    pd_estimates=uncertainty_set['pd_high'],  # Worst-case
    lgd_estimates=lgd_point,
    budget=1_000_000,
    risk_tolerance=0.05
)
```

**Key insight:** The conformal intervals provide **distribution-free uncertainty sets** with coverage guarantees → robust optimization uses upper bounds for conservative allocations.

---

## 11. Key Takeaways for Thesis

### What Makes Your Work Novel

1. **First application of MAPIE 1.3+ to credit risk PD modeling** (to our knowledge)
2. **Conformal-enhanced IFRS9:** Using interval widths for SICR detection
3. **Predict-then-optimize with conformal uncertainty:** Direct integration into Pyomo optimization
4. **Comparison with traditional methods:** Bootstrap vs Bayesian vs Conformal
5. **Efficiency analysis by risk segment:** Demonstrating practical applicability

### Recommended Experiments for Notebook 04

1. **Coverage validation:** α ∈ {0.01, 0.05, 0.10, 0.20}
2. **Efficiency benchmarking:** Compare MAPIE interval widths with bootstrap (1000 samples)
3. **Conditional coverage:** By grade, by loan amount quartile, by year
4. **Calibration impact:** Uncalibrated vs Isotonic calibrated CatBoost on conformal intervals
5. **Mondrian CP:** Split by grade, show fairness improvement
6. **Time complexity:** Measure conformalize() time vs bootstrap resampling

### Metrics to Report

| Metric | Formula | Target |
|--------|---------|--------|
| Empirical Coverage | mean(y ∈ [ŷ_low, ŷ_high]) | ≈ 1 - α |
| Coverage Gap | \|empirical - target\| | < 0.02 |
| Avg Width | mean(ŷ_high - ŷ_low) | < 0.30 |
| Width IQR | P75(width) - P25(width) | Low = uniform |
| Singleton Rate | mean(set_size == 1) | > 0.80 |
| Min Group Coverage | min over grades | > 0.85 |

---

## 12. Open Questions / Future Extensions

1. **Adaptive Conformal Prediction:** Update intervals online as new loans default (sequential learning)
2. **Conformal Survival Analysis:** Extend to time-to-default prediction (via lifelines)
3. **Conformal CATE:** Uncertainty on causal effects (econML + MAPIE)
4. **Conformal Time Series:** EnbPI for forecasting with MAPIE (not implemented in 1.3 for regression)
5. **Deep Learning + Conformal:** Neural networks for PD + MAPIE wrappers

---

## 13. References & Documentation

### Primary Resources
- **MAPIE Documentation:** https://mapie.readthedocs.io/en/stable/
- **MAPIE GitHub:** https://github.com/scikit-learn-contrib/MAPIE
- **CREPES:** https://github.com/henrikbostrom/crepes

### Key Papers (As of Jan 2025 Knowledge)
1. **Split Conformal Prediction:** Vovk et al. (2005) - foundational theory
2. **CQR:** Romano et al. (2019) - Conformalized Quantile Regression
3. **LAC/APS:** Sadinle et al. (2019) - Classification sets
4. **RAPS:** Angelopoulos et al. (2020) - Regularized APS
5. **Predict-then-Optimize:** Elmachtoub & Grigas (2022) - SPO+ framework

### Your Implementation Files (Cross-Reference)
- **Conformal logic:** `src/models/conformal.py`
- **Metrics:** `src/evaluation/metrics.py` (lines 59-84)
- **Robust optimization:** `src/optimization/robust_opt.py`
- **Calibration:** `src/models/calibration.py`
- **Visualization:** `src/utils/visualization.py` (needs update for MAPIE 1.3 shape)

---

## Summary

**Your current implementation is production-ready for MAPIE 1.3.0.** Key strengths:
- Correct API usage (SplitConformalRegressor with prefit=True)
- Proper ProbabilityRegressor wrapper
- Comprehensive metrics (coverage, width, efficiency)
- Integration with robust optimization

**Next steps for Notebook 04:**
1. Run experiments with multiple alpha levels
2. Add conditional coverage by grade
3. Compare efficiency with bootstrap
4. Visualize intervals (use updated plotting functions)
5. Save intervals for downstream optimization (Notebook 08)

**For thesis contribution:**
- Emphasize distribution-free guarantees vs Bayesian assumptions
- Show efficiency gains vs bootstrap (computational + statistical)
- Demonstrate fairness via Mondrian CP (if time permits)
- Integrate into full pipeline (NB09: end-to-end)

---

**Generated for:** Lending Club Risk Project
**Author:** Claude Code Analysis
**Date:** 2026-02-07
**Version:** MAPIE 1.3.0
