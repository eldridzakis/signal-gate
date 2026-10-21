"""Analytical and asymptotic noise thresholds for feature evaluation.

Provides closed-form mathematical estimators for Relative Information Gain (RIG)
under the null hypothesis of independence, leveraging Wilks' Theorem,
Bonferroni/Sidak corrections, and Extreme Value Theory (EVT).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def binary_entropy(p: float) -> float:
    """Shannon entropy in bits for a binary distribution with parameter p."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def target_entropy_from_weights(class_weights: Sequence[float]) -> float:
    """Computes Shannon entropy for arbitrary class distributions."""
    weights = np.asarray(class_weights, dtype=float)
    if len(weights) == 1:
        p = float(weights[0])
        return binary_entropy(p)
    weights = weights / weights.sum()
    nonzero = weights > 0.0
    return float(-np.sum(weights[nonzero] * np.log2(weights[nonzero])))


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
        target_distribution: Union[float, Sequence[float]],
        cardinality_x: int = 2,
        cardinality_y: int = 2
    ):
        """Initializes the threshold estimator.

        Parameters
        ----------
        nrows : int
            Number of observations (sample size N).
        target_distribution : float or Sequence[float]
            Minority class fraction (for binary targets, e.g. 0.2) or full
            class probability distribution (e.g. [0.7, 0.2, 0.1]).
        cardinality_x : int, default 2
            Number of distinct values in the candidate feature (2 for boolean).
        cardinality_y : int, default 2
            Number of distinct values in the target (2 for binary classification).
        """
        if nrows <= 0:
            raise ValueError("nrows must be positive.")

        self.nrows = nrows
        if isinstance(target_distribution, (int, float)):
            self.class_weights = [float(target_distribution)]
        else:
            self.class_weights = list(target_distribution)

        self.cardinality_x = cardinality_x
        self.cardinality_y = cardinality_y
        self.df = (cardinality_x - 1) * (cardinality_y - 1)

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
        n_features_evaluated: int,
        alpha: float = 0.05,
        method: str = "bonferroni"
    ) -> float:
        """Calculates the feature significance threshold controlling Family-Wise Error Rate (FWER).

        Parameters
        ----------
        n_features_evaluated : int
            Total number of candidate features evaluated (M).
        alpha : float, default 0.05
            Overall significance tolerance (e.g. 5% chance of any false discovery).
        method : {'bonferroni', 'sidak'}, default 'bonferroni'
            Multiple comparison correction method.

        Returns
        -------
        float
            Significance threshold tau. Features with train RIG > tau are statistically
            significant after adjusting for the M tests.
        """
        if n_features_evaluated < 1:
            raise ValueError("n_features_evaluated must be >= 1.")

        if method == "sidak":
            alpha_per_test = 1.0 - (1.0 - alpha) ** (1.0 / n_features_evaluated)
        else:  # bonferroni
            alpha_per_test = alpha / n_features_evaluated

        quantile = 1.0 - alpha_per_test
        chi2_crit = stats.chi2.ppf(quantile, df=self.df)
        return float(chi2_crit / self.scaling_factor)

    def evt_expected_max(self, n_features_evaluated: int) -> float:
        """Expected maximum null RIG among M evaluated features using Extreme Value Theory.

        For M independent null features, the distribution of max(Chi2_1) follows a Gumbel-type
        extreme value distribution with:
            E[max(Chi2_1)] ~= 2*ln(M) - ln(ln(M)) - ln(pi) + 2*gamma
        where gamma ~= 0.57721566 (Euler-Mascheroni constant).

        Parameters
        ----------
        n_features_evaluated : int
            Number of candidate features evaluated (M).

        Returns
        -------
        float
            Expected maximum null RIG. Represents the noise ceiling under H0.
        """
        m = float(n_features_evaluated)
        if m <= 1:
            return self.expected_null_mean

        if self.df == 1:
            gamma = 0.5772156649015328
            ln_m = np.log(m)
            ln_ln_m = np.log(max(ln_m, 1e-12))
            expected_chi2_max = 2.0 * ln_m - ln_ln_m - np.log(np.pi) + 2.0 * gamma
        else:
            # General degrees of freedom quantile approximation at 1 - 1/M
            expected_chi2_max = stats.chi2.ppf(1.0 - 1.0 / m, df=self.df)

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


def estimate_noise_threshold(
    nrows: int,
    minority_frac: float,
    n_features_evaluated: int,
    alpha: float = 0.05
) -> float:
    """Convenience function to quickly estimate the RIG noise threshold.

    Parameters
    ----------
    nrows : int
        Sample size (rows in training dataset).
    minority_frac : float
        Minority class proportion (e.g. 0.20 for a 20% default rate).
    n_features_evaluated : int
        Total number of candidate features evaluated.
    alpha : float, default 0.05
        Family-wise error rate tolerance.

    Returns
    -------
    float
        RIG threshold above which a feature is statistically distinguishable from noise.
    """
    estimator = FeatureNoiseThreshold(nrows=nrows, target_distribution=minority_frac)
    return estimator.analytical_threshold(n_features_evaluated=n_features_evaluated, alpha=alpha)
