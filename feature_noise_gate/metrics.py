"""Information-theoretic metrics for feature evaluation.

Provides robust, high-performance implementations of Shannon entropy,
Mutual Information, and Relative Information Gain (RIG).
"""

from typing import List, Sequence, Union
import numpy as np
import pandas as pd


def entropy(values: Sequence[float]) -> float:
    """Calculates Shannon entropy in bits (base-2 logarithm).

    Parameters
    ----------
    values : Sequence[float]
        Sequence of probability values or normalized counts summing to 1.0.

    Returns
    -------
    float
        Shannon entropy in bits: -sum(p * log2(p)).
    """
    total = 0.0
    for v in values:
        if v > 0.0 and v < 1.0:
            total -= v * np.log2(v)
    return float(total)


def fast_contingency_matrix(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """Computes a 2D contingency matrix between two categorical/discrete arrays.

    Uses integer factorization and np.bincount for high performance.
    """
    # Factorize to contiguous integers [0, K-1]
    if isinstance(x, pd.Series):
        x_codes, _ = pd.factorize(x.values)
    else:
        x_codes, _ = pd.factorize(np.asarray(x))

    if isinstance(y, pd.Series):
        y_codes, _ = pd.factorize(y.values)
    else:
        y_codes, _ = pd.factorize(np.asarray(y))

    # Mask valid (non-null) pairs
    valid = (x_codes >= 0) & (y_codes >= 0)
    if not np.all(valid):
        x_codes = x_codes[valid]
        y_codes = y_codes[valid]

    n_x = int(x_codes.max() + 1) if len(x_codes) > 0 else 0
    n_y = int(y_codes.max() + 1) if len(y_codes) > 0 else 0

    if n_x == 0 or n_y == 0:
        return np.zeros((0, 0), dtype=np.int64)

    # 2D bin count via 1D index: x * n_y + y
    flat_indices = x_codes * n_y + y_codes
    counts = np.bincount(flat_indices, minlength=n_x * n_y)
    return counts.reshape(n_x, n_y)


def fast_mutual_info_from_contingency(contingency: np.ndarray) -> float:
    """Computes Mutual Information I(X; Y) in bits from a 2D contingency table."""
    total_n = contingency.sum()
    if total_n == 0:
        return 0.0

    p_xy = contingency / total_n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    # Filter non-zero cells
    nonzero = p_xy > 0
    ratio = p_xy[nonzero] / (p_x @ p_y)[nonzero]
    mi = np.sum(p_xy[nonzero] * np.log2(ratio))
    return float(max(0.0, mi))


def fast_entropy_from_marginals(p: np.ndarray) -> float:
    """Computes Shannon entropy from marginal probabilities."""
    nonzero = p > 0.0
    return float(-np.sum(p[nonzero] * np.log2(p[nonzero])))


def rig(x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]) -> float:
    """Calculates Relative Information Gain (RIG), also known as normalized mutual info.

    RIG(X, Y) = (H(Y) - H(Y | X)) / H(Y) = I(X; Y) / H(Y)

    Parameters
    ----------
    x : array-like
        Feature values (discrete/categorical or boolean).
    y : array-like
        Target variable (discrete/categorical or binned).

    Returns
    -------
    float
        Relative information gain in [0, 1]. Returns 0.0 if target has zero entropy.
    """
    contingency = fast_contingency_matrix(x, y)
    if contingency.size == 0 or contingency.shape[1] <= 1:
        return 0.0

    total_n = contingency.sum()
    if total_n == 0:
        return 0.0

    p_xy = contingency / total_n
    p_y = p_xy.sum(axis=0)

    h_y = fast_entropy_from_marginals(p_y)
    if h_y <= 1e-12:
        return 0.0

    mi = fast_mutual_info_from_contingency(contingency)
    rig_value = mi / h_y
    return float(np.clip(rig_value, 0.0, 1.0))


def rig_all_columns(
    df: pd.DataFrame,
    target: str,
    feature_cols: Sequence[str] = None
) -> List[float]:
    """Calculates RIG for each feature column against a target column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and target.
    target : str
        Name of target column.
    feature_cols : Sequence[str], optional
        List of feature column names. If None, all columns except target are used.

    Returns
    -------
    List[float]
        List of RIG values in the same order as feature_cols.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target]

    y_series = df[target]
    rigs = []
    for col in feature_cols:
        rigs.append(rig(df[col], y_series))
    return rigs
