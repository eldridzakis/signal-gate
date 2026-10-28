"""Analytical and asymptotic noise thresholds for feature evaluation.

Provides closed-form mathematical estimators for Relative Information Gain (RIG)
under the null hypothesis of independence, leveraging Wilks' Theorem,
Bonferroni/Sidak corrections, and Extreme Value Theory (EVT).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import special, stats


def binary_entropy(p: float) -> float:
    """Shannon entropy in bits for a binary distribution with parameter p."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def target_entropy_from_weights(
    class_weights: Union[float, Sequence[float], np.ndarray, pd.Series]
) -> float:
    """Computes Shannon entropy for arbitrary binary or multiclass distributions."""
    if isinstance(class_weights, (int, float)):
        return binary_entropy(float(class_weights))
    weights = np.asarray(class_weights, dtype=float)
    if weights.size == 0:
        return 0.0
    if weights.size == 1:
        p = float(weights.flat[0])
        return binary_entropy(p)
    total = weights.sum()
    if total <= 0.0:
        return 0.0
    p_norm = weights / total
    nonzero = p_norm > 0.0
    return float(-np.sum(p_norm[nonzero] * np.log2(p_norm[nonzero])))


class FeatureNoiseThreshold:
    """Closed-form estimator for feature noise floors and significance thresholds.

    Under the null hypothesis H0 (feature X and target Y are independent),
    Wilks' Theorem states that twice the mutual information scaled by sample size
    converges asymptotically to a Chi-squared distribution:

        G = 2 * N * ln(2) * I(X; Y) ~ Chi2(df)

    where df = (|X| - 1) * (|Y| - 1).

    Since RIG(X, Y) = I(X; Y) / H(Y), the null RIG follows:

        RIG_null ~ Chi2(df) / [2 * N * ln(2) * H(Y)]
    """

    def __init__(
        self,
        nrows: int,
        target_distribution: Optional[Union[float, Sequence[float], np.ndarray, pd.Series]] = None,
        cardinality_x: Optional[int] = None,
        cardinality_y: Optional[int] = None,
        n_feature_categories: int = 2,
        n_classes: int = 2,
        df: Optional[int] = None,
        minority_frac: Optional[Union[float, Sequence[float], np.ndarray, pd.Series]] = None,
    ):
        """Initializes the threshold estimator.

        Parameters
        ----------
        nrows : int
            Number of observations (sample size N).
        target_distribution : float or Sequence[float], optional
            Minority class fraction (for binary targets, e.g. 0.2) or full
            class probability distribution / counts (e.g. [0.7, 0.2, 0.1]).
        cardinality_x : int, optional
            Number of distinct values in the candidate feature (R). Alias for n_feature_categories.
        cardinality_y : int, optional
            Number of distinct values in the target (C). Alias for n_classes.
        n_feature_categories : int, default 2
            Number of distinct categories in the candidate feature (R).
        n_classes : int, default 2
            Number of distinct classes in the target (C).
        df : int, optional
            Explicit degrees of freedom k = (R-1)*(C-1). Overrides cardinality parameters.
        minority_frac : float or Sequence[float], optional
            Alias for target_distribution for backward compatibility.
        """
        if nrows <= 0:
            raise ValueError("nrows must be positive.")

        if target_distribution is None:
            if minority_frac is not None:
                target_distribution = minority_frac
            else:
                raise ValueError("Either target_distribution or minority_frac must be provided.")

        self.nrows = nrows
        if isinstance(target_distribution, (int, float)):
            self.class_weights = [float(target_distribution)]
        else:
            self.class_weights = list(np.asarray(target_distribution, dtype=float))

        # Determine R and C
        r = cardinality_x if cardinality_x is not None else n_feature_categories
        c = cardinality_y if cardinality_y is not None else n_classes

        # Auto-infer C if target_distribution has length > 2 and C was not explicitly overridden
        if len(self.class_weights) > 2 and cardinality_y is None and n_classes == 2:
            c = len(self.class_weights)

        self.cardinality_x = int(r)
        self.cardinality_y = int(c)
        self.n_feature_categories = self.cardinality_x
        self.n_classes = self.cardinality_y

        if df is not None:
            self.df = int(df)
        else:
            self.df = max(1, (self.cardinality_x - 1) * (self.cardinality_y - 1))

        self.target_entropy = target_entropy_from_weights(self.class_weights)
        if self.target_entropy <= 0.0:
            raise ValueError("Target entropy must be greater than zero.")

        # Scaling factor converting Chi2 statistic to RIG:
        # RIG = Chi2 / scaling_factor
        self.scaling_factor = 2.0 * self.nrows * np.log(2.0) * self.target_entropy

    @property
    def expected_null_mean(self) -> float:
        """Expected value (mean) of the null RIG distribution: E[Chi2(df)] / C."""
        return float(self.df / self.scaling_factor)

    @property
    def expected_null_median(self) -> float:
        """Theoretical median of the null RIG distribution."""
        chi2_median = stats.chi2.ppf(0.5, df=self.df)
        return float(chi2_median / self.scaling_factor)

    def quantile_threshold(self, quantile: float) -> float:
        """Computes the null RIG at a given quantile (e.g. 0.999 for 1-in-1,000).

        Parameters
        ----------
        quantile : float
            Cumulative probability in (0, 1).

        Returns
        -------
        float
            RIG cutoff corresponding to this quantile in the null distribution.
        """
        if not (0.0 < quantile < 1.0):
            raise ValueError("Quantile must be strictly between 0 and 1.")
        chi2_val = stats.chi2.ppf(quantile, df=self.df)
        return float(chi2_val / self.scaling_factor)

    def tail_probability(self, rig_threshold: float) -> float:
        """Computes the probability of observing a null RIG >= rig_threshold.

        P(RIG_null >= tau) = 1 - F_Chi2(tau * scaling_factor).
        """
        chi2_stat = rig_threshold * self.scaling_factor
        return float(stats.chi2.sf(chi2_stat, df=self.df))

    def analytical_threshold(
        self,
        n_features_evaluated: int = 1,
        alpha: float = 0.05,
        method: str = "bonferroni",
        effective_features_evaluated: Optional[float] = None,
        candidate_df: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> float:
        """Calculates the feature significance threshold controlling Family-Wise Error Rate (FWER).

        Parameters
        ----------
        n_features_evaluated : int, default 1
            Total number of candidate features evaluated (M_raw).
        alpha : float, default 0.05
            Overall significance tolerance (e.g. 5% chance of any false discovery).
        method : {'bonferroni', 'sidak'}, default 'bonferroni'
            Multiple comparison correction method.
        effective_features_evaluated : float, optional
            Effective number of independent features (M_eff <= M_raw).
        candidate_df : pd.DataFrame or np.ndarray, optional
            Evaluated features matrix from which M_eff will be estimated automatically.

        Returns
        -------
        float
            Significance threshold tau. Features with train RIG > tau are statistically
            significant after adjusting for the tests.
        """
        if candidate_df is not None:
            eff_res = estimate_effective_tests(candidate_df)
            m = float(eff_res["M_eff"])
        elif effective_features_evaluated is not None:
            m = float(effective_features_evaluated)
        else:
            m = float(n_features_evaluated)

        if m < 1.0:
            raise ValueError("n_features_evaluated / M_eff must be >= 1.0.")

        if method == "sidak":
            alpha_per_test = 1.0 - (1.0 - alpha) ** (1.0 / m)
        else:  # bonferroni
            alpha_per_test = alpha / m

        quantile = 1.0 - alpha_per_test
        chi2_crit = stats.chi2.ppf(quantile, df=self.df)
        return float(chi2_crit / self.scaling_factor)

    def evt_expected_max(
        self,
        n_features_evaluated: int = 1,
        effective_features_evaluated: Optional[float] = None,
        candidate_df: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> float:
        """Expected maximum null RIG among M evaluated features using Extreme Value Theory.

        For M independent null features with k degrees of freedom, the asymptotic Gumbel
        expected maximum follows:
            E[max(Chi2_k)] ~= 2*ln(M) + (k - 2)*ln(ln(M)) - 2*ln(Gamma(k/2)) + 2*gamma
        where gamma ~= 0.5772156649 (Euler-Mascheroni constant) and ln(Gamma(k/2)) is
        computed via scipy.special.gammaln(k / 2.0).

        When k = 1, since Gamma(1/2) = sqrt(pi) and 2*ln(Gamma(1/2)) = ln(pi), this
        formula reduces identically to the binary formula:
            2*ln(M) - ln(ln(M)) - ln(pi) + 2*gamma.

        Parameters
        ----------
        n_features_evaluated : int, default 1
            Number of candidate features evaluated (M_raw).
        effective_features_evaluated : float, optional
            Effective number of independent features (M_eff <= M_raw).
        candidate_df : pd.DataFrame or np.ndarray, optional
            Evaluated features matrix from which M_eff will be estimated automatically.

        Returns
        -------
        float
            Expected maximum null RIG. Represents the noise ceiling under H0.
        """
        if candidate_df is not None:
            eff_res = estimate_effective_tests(candidate_df)
            m = float(eff_res["M_eff"])
        elif effective_features_evaluated is not None:
            m = float(effective_features_evaluated)
        else:
            m = float(n_features_evaluated)

        if m <= 1.0:
            return self.expected_null_mean

        k = float(self.df)
        gamma = 0.5772156649015328

        if m <= 2.0:
            # For small M <= 2, guard against ln(ln(M)) domain issues using quantile bound
            expected_chi2_max = max(k, float(stats.chi2.ppf(1.0 - 1.0 / (2.0 * m), df=self.df)))
        else:
            ln_m = np.log(m)
            ln_ln_m = np.log(ln_m)
            log_gamma_k2 = float(special.gammaln(k / 2.0))
            expected_chi2_max = 2.0 * ln_m + (k - 2.0) * ln_ln_m - 2.0 * log_gamma_k2 + 2.0 * gamma
            # Ensure expected max is at least the mean of 1 variable
            expected_chi2_max = max(k, expected_chi2_max)

        return float(max(0.0, expected_chi2_max / self.scaling_factor))

    def evaluate_search_space(
        self,
        feature_counts: Sequence[int] = (10, 100, 1000, 10000, 100000, 1000000),
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """Generates a summary table across various feature search space sizes."""
        rows = []
        for m in feature_counts:
            rows.append({
                "Features Evaluated (M)": m,
                "Expected Mean Null RIG": self.expected_null_mean,
                "1-in-M Null RIG": self.quantile_threshold(1.0 - 1.0 / m if m > 1 else 0.5),
                "EVT Expected Max Noise RIG": self.evt_expected_max(m),
                f"Significance Threshold (alpha={alpha})": self.analytical_threshold(m, alpha=alpha)
            })
        return pd.DataFrame(rows)


def estimate_effective_tests(
    X_or_df: Union[pd.DataFrame, np.ndarray, Sequence],
    max_m_corr: int = 2000
) -> Dict[str, Any]:
    """Estimates the effective number of independent hypothesis tests (M_eff <= M_raw).

    Uses the linear algebra Frobenius-norm trace identity:
        sum_{i=1}^M lambda_i^2 = Tr(R^2) = ||R||_F^2
    avoiding costly O(M^3) eigendecompositions. If M > N, applies the dual Gram
    matrix trick G = Z @ Z.T (N x N) which shares the exact same non-zero eigenvalues.

    Applies the Cheverud-Nyholt / Li-Ji eigenvalue variance formula:
        Var(lambda) = (sum_{i=1}^M lambda_i^2 - M) / (M * (M - 1))
        M_eff = 1 + (M - 1) * (1 - Var(lambda))
    bounded strictly in [1.0, M_raw].

    Parameters
    ----------
    X_or_df : pd.DataFrame, np.ndarray, or sequence
        N x M matrix of candidate feature values evaluated by the agent/pipeline.
    max_m_corr : int, default 2000
        Maximum M for computing exact mean absolute off-diagonal correlation.

    Returns
    -------
    dict
        M_raw : int, total candidate features evaluated
        M_eff : float, effective number of independent tests
        redundancy_ratio : float, M_eff / M_raw in (0, 1]
        mean_abs_correlation : float, average off-diagonal |r_ij|
    """
    if isinstance(X_or_df, pd.DataFrame):
        arr = X_or_df.to_numpy(dtype=float, copy=True)
    else:
        arr = np.array(X_or_df, dtype=float, copy=True)
        if arr.ndim == 1:
            arr = arr[:, None]

    if arr.size == 0 or arr.ndim < 2 or arr.shape[1] == 0:
        return {"M_raw": 0, "M_eff": 0.0, "redundancy_ratio": 1.0, "mean_abs_correlation": 0.0}

    n_samples, m_raw = arr.shape
    if m_raw == 1:
        return {"M_raw": 1, "M_eff": 1.0, "redundancy_ratio": 1.0, "mean_abs_correlation": 0.0}
    if n_samples < 2:
        return {"M_raw": m_raw, "M_eff": float(m_raw), "redundancy_ratio": 1.0, "mean_abs_correlation": 0.0}

    # Handle NaNs if any
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    # Standardize non-constant columns to zero mean and unit L2 norm in-place
    means = np.mean(arr, axis=0)
    arr -= means
    norms = np.sqrt(np.einsum("ij,ij->j", arr, arr))
    valid_cols = norms > 1e-12
    n_valid = int(np.sum(valid_cols))

    if n_valid <= 1:
        return {
            "M_raw": int(m_raw),
            "M_eff": 1.0,
            "redundancy_ratio": float(1.0 / m_raw),
            "mean_abs_correlation": 0.0
        }

    if n_valid == m_raw:
        z = arr
        z /= norms
    else:
        z = arr[:, valid_cols]
        z /= norms[valid_cols]
    m_valid = n_valid

    # Frobenius-norm trace identity: Tr(R^2) = ||R||_F^2
    if m_valid <= n_samples:
        # Primal: M x M
        gram = z.T @ z
        sum_lambda_sq = float(np.einsum("ij,ij->", gram, gram))
        if m_valid <= max_m_corr:
            abs_gram = np.abs(gram)
            mean_abs_corr = float((np.sum(abs_gram) - m_valid) / (m_valid * (m_valid - 1)))
        else:
            mean_abs_corr = 0.0
    else:
        # Dual trick: N x N
        gram_dual = z @ z.T
        sum_lambda_sq = float(np.einsum("ij,ij->", gram_dual, gram_dual))
        if m_valid <= max_m_corr:
            r = z.T @ z
            abs_r = np.abs(r)
            mean_abs_corr = float((np.sum(abs_r) - m_valid) / (m_valid * (m_valid - 1)))
        else:
            mean_abs_corr = 0.0

    var_ratio = (sum_lambda_sq - m_valid) / (m_valid * (m_valid - 1))
    var_ratio = float(np.clip(var_ratio, 0.0, 1.0))

    m_eff = 1.0 + (m_valid - 1.0) * (1.0 - var_ratio)
    m_eff = float(np.clip(m_eff, 1.0, float(m_raw)))
    redundancy_ratio = float(m_eff / m_raw)

    return {
        "M_raw": int(m_raw),
        "M_eff": float(m_eff),
        "redundancy_ratio": float(redundancy_ratio),
        "mean_abs_correlation": float(mean_abs_corr),
    }


def estimate_noise_threshold(
    nrows: int,
    minority_frac: Optional[Union[float, Sequence[float], np.ndarray, pd.Series]] = None,
    n_features_evaluated: int = 1,
    alpha: float = 0.05,
    target_distribution: Optional[Union[float, Sequence[float], np.ndarray, pd.Series]] = None,
    n_classes: int = 2,
    n_feature_categories: int = 2,
    df: Optional[int] = None,
    effective_features_evaluated: Optional[float] = None,
    candidate_df: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> float:
    """Convenience function to quickly estimate the RIG noise threshold.

    Parameters
    ----------
    nrows : int
        Sample size (rows in training dataset).
    minority_frac : float or Sequence[float], optional
        Minority class proportion (e.g. 0.20 for binary) or class distribution.
    n_features_evaluated : int, default 1
        Total number of candidate features evaluated (M_raw).
    alpha : float, default 0.05
        Family-wise error rate tolerance.
    target_distribution : float or Sequence[float], optional
        Class probability distribution or counts (alias for minority_frac).
    n_classes : int, default 2
        Number of target classes (C).
    n_feature_categories : int, default 2
        Number of distinct categories in the candidate feature (R).
    df : int, optional
        Explicit degrees of freedom k = (R-1)*(C-1). Overrides cardinality parameters.
    effective_features_evaluated : float, optional
        Effective number of independent features (M_eff <= M_raw).
    candidate_df : pd.DataFrame or np.ndarray, optional
        Evaluated features matrix from which M_eff will be estimated automatically.

    Returns
    -------
    float
        RIG threshold above which a feature is statistically distinguishable from noise.
    """
    dist = target_distribution if target_distribution is not None else minority_frac
    if dist is None:
        dist = 0.5

    estimator = FeatureNoiseThreshold(
        nrows=nrows,
        target_distribution=dist,
        n_classes=n_classes,
        n_feature_categories=n_feature_categories,
        df=df
    )
    return estimator.analytical_threshold(
        n_features_evaluated=n_features_evaluated,
        alpha=alpha,
        effective_features_evaluated=effective_features_evaluated,
        candidate_df=candidate_df
    )
