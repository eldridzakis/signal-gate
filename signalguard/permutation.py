"""Permutation testing engine for empirical null distribution analysis.

Provides high-performance vectorized Monte-Carlo permutation testing
for Relative Information Gain (RIG) and generic user-defined metrics.
"""

from typing import Callable, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from .metrics import fast_contingency_matrix, fast_entropy_from_marginals, fast_mutual_info_from_contingency, rig, rig_all_columns


def _compute_rig_vectorized_binary(
    features_matrix: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """Computes RIG for multiple boolean/binary features against a target array."""
    n_samples, n_features = features_matrix.shape
    if n_samples == 0 or n_features == 0:
        return np.zeros(n_features)

    # Prior entropy of target
    y_codes, _ = pd.factorize(target)
    n_y = int(y_codes.max() + 1) if len(y_codes) > 0 else 0
    if n_y <= 1:
        return np.zeros(n_features)

    counts_y = np.bincount(y_codes, minlength=n_y)
    p_y = counts_y / n_samples
    h_y = fast_entropy_from_marginals(p_y)
    if h_y <= 1e-12:
        return np.zeros(n_features)

    rigs = np.empty(n_features, dtype=float)

    # For each feature column, compute contingency via bincount
    for col_idx in range(n_features):
        x_col = features_matrix[:, col_idx]
        x_codes, _ = pd.factorize(x_col)
        n_x = int(x_codes.max() + 1) if len(x_codes) > 0 else 0
        if n_x <= 1:
            rigs[col_idx] = 0.0
            continue

        flat_indices = x_codes * n_y + y_codes
        contingency = np.bincount(flat_indices, minlength=n_x * n_y).reshape(n_x, n_y)
        mi = fast_mutual_info_from_contingency(contingency)
        rigs[col_idx] = mi / h_y

    return np.clip(rigs, 0.0, 1.0)


class PermutationObject:
    """High-performance Monte-Carlo permutation tester supporting RIG and custom metrics."""

    def __init__(self):
        # Default Feature Fractions (feature support)
        self.feature_fractions: List[float] = [0.1]

        # Default reference information gain level
        self.gain_threshold: float = 0.0005

        # Configuration and state
        self.nrows: int = 0
        self.class_weights: List[float] = []
        self.data: pd.DataFrame = pd.DataFrame()
        self.null_rigs: pd.DataFrame = pd.DataFrame()
        self.null_rig_values: pd.Series = pd.Series(dtype=float)
        self.permutations: int = 0

        # Generic metric aliases
        self.df_nulls: pd.DataFrame = pd.DataFrame()
        self.null_values: pd.Series = pd.Series(dtype=float)

    def set_data_parameters(
        self,
        nrows: int,
        class_weights: Optional[Union[float, Sequence[float]]] = None,
        minority_class: Optional[float] = None
    ):
        """Sets dataset size and class balance parameters.

        Supports both class_weights (list) and minority_class (scalar).
        """
        self.nrows = int(nrows)
        if class_weights is not None:
            if isinstance(class_weights, (int, float)):
                weights = [float(class_weights)]
            else:
                weights = list(class_weights)
        elif minority_class is not None:
            weights = [float(minority_class)]
        else:
            raise ValueError("Either class_weights or minority_class must be provided.")

        if len(weights) > 1:
            total = sum(weights)
            if not np.isclose(total, 1.0, atol=1e-4):
                raise ValueError(f"Class weights should sum to 1.0, got {total}")
        self.class_weights = weights

    def create_synthetic_data(self, seed: Optional[int] = 42) -> pd.DataFrame:
        """Generates synthetic data with configured row count, target distribution, and features."""
        rng = np.random.default_rng(seed)

        # 1. Target generation
        if len(self.class_weights) == 1:
            minority_fraction = self.class_weights[0]
            n_minority = int(round(self.nrows * minority_fraction))
            n_majority = self.nrows - n_minority
            target = np.array([0] * n_majority + [1] * n_minority, dtype=int)
            rng.shuffle(target)
        else:
            weights = np.array(self.class_weights) / sum(self.class_weights)
            counts = np.floor(weights * self.nrows).astype(int)
            remainder = self.nrows - counts.sum()
            if remainder > 0:
                counts[:remainder] += 1
            values = []
            for class_idx, count in enumerate(counts):
                values.extend([class_idx] * count)
            target = np.array(values, dtype=int)
            rng.shuffle(target)

        df = pd.DataFrame({'target': target})

        # 2. Synthetic boolean or multi-category features
        for feat_idx, feature_fraction in enumerate(self.feature_fractions):
            if isinstance(feature_fraction, (int, float)):
                col_name = f'feature_{round(float(feature_fraction), 3)}'
                n_ones = int(round(self.nrows * float(feature_fraction)))
                n_zeros = self.nrows - n_ones
                feat_vals = np.array([0] * n_zeros + [1] * n_ones, dtype=int)
                rng.shuffle(feat_vals)
                df[col_name] = feat_vals
            else:
                # Multi-category feature fractions summing to 1.0
                weights = np.asarray(feature_fraction, dtype=float)
                weights = weights / weights.sum()
                counts = np.floor(weights * self.nrows).astype(int)
                remainder = self.nrows - counts.sum()
                if remainder > 0:
                    counts[:remainder] += 1
                values = []
                for cat_idx, count in enumerate(counts):
                    values.extend([cat_idx] * count)
                feat_vals = np.array(values, dtype=int)
                rng.shuffle(feat_vals)
                col_name = f'feature_cat_{len(weights)}_{feat_idx}'
                df[col_name] = feat_vals

        self.data = df
        return self.data

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        gain_threshold: float = 0.0005
    ) -> "PermutationObject":
        """Alternative constructor initializing from an existing DataFrame."""
        obj = cls()
        obj.nrows = len(df)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target]
        obj.data = df[[target] + list(feature_cols)].copy()
        obj.gain_threshold = gain_threshold
        return obj

    def calculate_null_distribution(
        self,
        permutations: int,
        metric_fn: Optional[Callable[[pd.Series, pd.Series], float]] = None,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Runs Monte-Carlo permutation test by shuffling the target repeatedly.

        Parameters
        ----------
        permutations : int
            Number of target shuffle iterations.
        metric_fn : Callable, optional
            Function (feature, target) -> float. If None, uses fast vectorized RIG.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool, default True
            Whether to print summary quantiles upon completion.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape (permutations, num_features) with null metric values.
        """
        self.permutations = int(permutations)
        rng = np.random.default_rng(seed)

        target_col = 'target'
        feature_cols = [c for c in self.data.columns if c != target_col]
        n_features = len(feature_cols)

        if metric_fn is None:
            # Fast vectorized multiclass & multi-category RIG path
            features_df = self.data[feature_cols]
            n_features = len(feature_cols)
            n_samples = len(self.data)

            target_vals = self.data[target_col].to_numpy()
            y_codes, _ = pd.factorize(target_vals)
            n_y = int(y_codes.max() + 1) if len(y_codes) > 0 else 0
            if n_y <= 1:
                null_matrix = np.zeros((self.permutations, n_features), dtype=float)
            else:
                counts_y = np.bincount(y_codes, minlength=n_y)
                p_y = counts_y / n_samples
                h_y = fast_entropy_from_marginals(p_y)

                if h_y <= 1e-12:
                    null_matrix = np.zeros((self.permutations, n_features), dtype=float)
                else:
                    # Pre-factorize features once
                    x_codes_list = []
                    n_x_list = []
                    for col in feature_cols:
                        xc, _ = pd.factorize(features_df[col].to_numpy())
                        nx = int(xc.max() + 1) if len(xc) > 0 else 0
                        x_codes_list.append(xc)
                        n_x_list.append(nx)

                    null_matrix = np.empty((self.permutations, n_features), dtype=float)
                    y_shuffled = y_codes.copy()

                    for i in range(self.permutations):
                        rng.shuffle(y_shuffled)
                        for col_idx in range(n_features):
                            nx = n_x_list[col_idx]
                            if nx <= 1:
                                null_matrix[i, col_idx] = 0.0
                                continue
                            flat = x_codes_list[col_idx] * n_y + y_shuffled
                            contingency = np.bincount(flat, minlength=nx * n_y).reshape(nx, n_y)
                            mi = fast_mutual_info_from_contingency(contingency)
                            null_matrix[i, col_idx] = mi / h_y

            null_matrix = np.clip(null_matrix, 0.0, 1.0)
        else:
            # Generic user-defined metric function path
            null_matrix = np.empty((self.permutations, n_features), dtype=float)
            target_s = self.data[target_col].copy()
            features_df = self.data[feature_cols]

            for i in range(self.permutations):
                shuffled_y = pd.Series(rng.permutation(target_s.values), index=target_s.index)
                for j, col in enumerate(feature_cols):
                    null_matrix[i, j] = float(metric_fn(features_df[col], shuffled_y))

        self.null_rigs = pd.DataFrame(null_matrix, columns=feature_cols)
        self.null_rig_values = pd.Series(null_matrix.flatten(), dtype=float)

        # Aliases
        self.df_nulls = self.null_rigs
        self.null_values = self.null_rig_values

        if verbose:
            n_greater = 1
            quantile = max(0.0, 1.0 - (n_greater / self.permutations))
            print(f'Max null value \t\t= {self.null_rig_values.max():.6f}')
            print(f'Median null value \t= {self.null_rig_values.median():.6f}')
            print(f'{n_greater} in {self.permutations} null value \t= {self.null_rig_values.quantile(quantile):.6f}')
            print(f'Threshold \t\t= {self.gain_threshold:.6f}')

        return self.null_rigs

    # Alias for backward compatibility
    calculate_null_rigs = calculate_null_distribution

    def null_rigs_comparison(self, threshold: Optional[float] = None) -> int:
        """Counts how many null values exceed the threshold."""
        thresh = self.gain_threshold if threshold is None else threshold
        exceed_count = int((self.null_rig_values > thresh).sum())
        total = len(self.null_rig_values)
        pct = (exceed_count / total * 100) if total > 0 else 0.0
        print(f'{exceed_count} in {total} null values ({pct:.2f}%) greater than {thresh} threshold')
        return exceed_count

    # Alias
    nulls_comparison = null_rigs_comparison
