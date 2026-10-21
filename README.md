# Sampling Toolbox

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)]()

> **Quantifying the Winner's Curse, estimating statistical noise thresholds, and de-biasing automated feature engineering.**

---

## 📌 Overview

Automated Machine Learning (AutoML) and automated feature engineering engines explore massive hypothesis spaces—evaluating thousands to millions of candidate transformations, numerical cuts, n-grams, and context joins.

Selecting features that maximize performance metrics on a training split introduces **selection bias (The Winner's Curse)**:
1. **Spurious Features Pass as Genuine:** Random noise features will inevitably achieve high sample metric scores purely by chance.
2. **Optimistic Overestimation:** Genuine features exhibit inflated scores on training data compared to test and production data:
   $$\text{Bias} = \text{Metric}_{\text{train}} - \text{Metric}_{\text{unseen}} > 0$$
3. **Compounded by Large Search Budgets:** As the search space ($M$) expands (e.g., adding external datasets like OpenStreetMap), the noise floor and overestimation bias increase systematically.

**Sampling Toolbox** provides an information-theoretic framework to:
* **Instantly calculate closed-form noise thresholds** given dataset size ($N$), target distribution ($p$), and search space size ($M$).
* **Run vectorized Monte-Carlo permutation tests** (100x faster than traditional dataframe shuffling).
* **De-bias feature importances** and classify features into $1\sigma$ and $2\sigma$ statistical confidence tiers via bootstrap resampling.

---

## 📐 Mathematical Foundations

### 1. Relative Information Gain (RIG) & Wilks' Theorem
Relative Information Gain (normalized mutual information) measures the reduction in target entropy:
$$\text{RIG}(X, Y) = \frac{I(X; Y)}{H(Y)} = \frac{H(Y) - H(Y \mid X)}{H(Y)}$$

Under the null hypothesis $H_0$ that feature $X$ and target $Y$ are independent, Wilks' Theorem establishes that the deviance statistic asymptotically follows a Chi-squared distribution:
$$2 N \ln(2) \cdot I(X; Y) \xrightarrow{d} \chi^2_{df}$$
where $df = (|X| - 1)(|Y| - 1) = 1$ for boolean features and binary targets.

Dividing by the target prior entropy $H(Y)$ gives the asymptotic distribution of null RIG:
$$\text{RIG}_{\text{null}} \sim \frac{\chi^2_1}{2 N \ln(2) \cdot H(Y)}$$

```
Key Theoretical Insights:
• Scaling: Expected null RIG scales as O(1/N).
• Target Imbalance: Lower minority class prevalence p reduces H(Y), which raises the noise floor!
• Invariance to Feature Marginals: Null RIG mean is invariant to feature support P(X=1).
```

### 2. Search-Space-Aware Noise Gates ($M$ Evaluated Features)
When an AutoML engine evaluates $M$ candidate features:

* **Bonferroni / Šidák Significance Threshold ($\alpha = 0.05$):**
  $$\tau_{\text{significance}}(N, p, M, \alpha) = \frac{F_{\chi^2_1}^{-1}\left(1 - \frac{\alpha}{M}\right)}{2 N \ln(2) \cdot H(p)}$$

* **Extreme Value Theory (EVT) Expected Maximum:**
  $$\mathbb{E}[\max_{1 \le i \le M} \chi^2_1] \approx 2 \ln M - \ln \ln M - \ln \pi + 2\gamma \quad (\gamma \approx 0.5772)$$
  $$\tau_{\text{EVT}}(N, p, M) \approx \frac{2 \ln M - \ln \ln M - 0.012}{2 N \ln(2) \cdot H(p)}$$

---

## 🚀 Quickstart

### Installation
```bash
git clone https://github.com/peter-eldridge/sampling_toolbox.git
cd sampling_toolbox
pip install -e .
```

### 1. Instant Noise Threshold Calculation
Calculate how high a feature's RIG must be to be statistically distinguished from noise after evaluating $M=20,000$ candidates on a dataset of 5,000 rows with a 15% minority class:

```python
from sampling_toolbox import estimate_noise_threshold, FeatureNoiseThreshold

# Instant lookup (< 1 millisecond)
tau = estimate_noise_threshold(nrows=5000, minority_frac=0.15, n_features_evaluated=20000, alpha=0.05)
print(f"Noise threshold: RIG must exceed {tau:.5f}")

# Full search-space sensitivity table
estimator = FeatureNoiseThreshold(nrows=5000, target_distribution=0.15)
summary_df = estimator.evaluate_search_space([100, 1000, 10000, 100000])
print(summary_df)
```

### 2. Fast Monte-Carlo Permutation Test
```python
from sampling_toolbox import PermutationObject

perm = PermutationObject()
perm.set_data_parameters(nrows=2000, class_weights=[0.25])
perm.feature_fractions = [0.10]
perm.create_synthetic_data(seed=42)

# Runs 5,000 permutations in seconds using vectorized NumPy bincount
null_df = perm.calculate_null_rigs(permutations=5000)
perm.null_rigs_comparison(threshold=0.005)
```

### 3. Feature De-Biasing & Stability Classification
```python
from sampling_toolbox import FeatureDebiaser

# 1. Compute holdout bias: Bias = Train_RIG - Unseen_RIG
df_bias = FeatureDebiaser.compute_holdout_bias(df_train, df_test, target='target')
median_bias = df_bias['test_bias'].median()

# 2. De-bias and classify stability via Bootstrap 1-sigma & 2-sigma testing
df_stability = FeatureDebiaser.bootstrap_stability_analysis(
    df=df_train,
    target='target',
    estimated_bias=median_bias,
    n_bootstraps=100
)

# Features with corrected_rig - 2*sigma > 0 are robust to selection bias
stable_features = df_stability[df_stability['is_2_sigma']]
print(stable_features[['feature', 'train_rig', 'corrected_rig', 'stability_class']])
```

---

## 📊 End-to-End Walkthrough Notebooks

| Notebook | Description |
| :--- | :--- |
| **[`01_feature_noise_thresholds.ipynb`](notebooks/01_feature_noise_thresholds.ipynb)** | Mathematical proofs, asymptotic $\chi^2$ vs Monte-Carlo permutations, parameter sweeps ($N$, minority class $p$), and Bonferroni/EVT noise gates. |
| **[`02_selection_bias_and_debiasing.ipynb`](notebooks/02_selection_bias_and_debiasing.ipynb)** | Quantifying the Winner's Curse on Lending Club loan defaults, measuring search complexity vs. bias, and executing $1\sigma / 2\sigma$ bootstrap de-biasing. |

---

## 📂 Repository Structure

```
sampling_toolbox/
├── sampling_toolbox/             # Core library
│   ├── __init__.py               # Public API exports
│   ├── metrics.py                # Fast entropy, mutual info, RIG calculations
│   ├── threshold.py              # Analytical Chi-squared & EVT noise estimators
│   ├── permutation.py            # Vectorized Monte-Carlo permutation tester
│   └── debias.py                 # Out-of-Fold (OOF) & Bootstrap de-biasing engine
├── notebooks/                    # Self-contained walkthroughs
│   ├── 01_feature_noise_thresholds.ipynb
│   └── 02_selection_bias_and_debiasing.ipynb
├── tests/                        # Automated unit tests (15 test suites)
│   ├── test_metrics.py
│   ├── test_threshold.py
│   ├── test_permutation.py
│   └── test_debias.py
├── Data/                         # Benchmark datasets (King County, Lending Club)
├── integrations/                 # Optional connectors for AutoML log parsing
├── archive/                      # Historical notebooks preserved for reference
├── pyproject.toml                # Modern package configuration
└── README.md
```

---

## 🧪 Testing

Run the test suite using standard Python `unittest`:

```bash
python -m unittest discover -s tests -v
```

All 15 tests run in $< 1$ second.

---

## 📜 Citation & License

This project is licensed under the MIT License - see the LICENSE file for details.
