"""Information-theoretic metrics for feature evaluation.

Provides robust, high-performance implementations of Shannon entropy,
Mutual Information, and Relative Information Gain (RIG).
"""

from typing import List, Sequence, Union
import numpy as np
import pandas as pd


def entropy(values: Union[float, Sequence[float], np.ndarray, pd.Series]) -> float:
    """Calculates Shannon entropy in bits (base-2 logarithm).

    Parameters
    ----------
    values : float or Sequence[float]
        If a single float p in [0, 1], computes binary entropy:
            H(p) = -p*log2(p) - (1-p)*log2(1-p).
        If a sequence or 1D array of class probabilities or counts, computes:
            H(P) = -sum(p_i * log2(p_i)).

    Returns
    -------
    float
        Shannon entropy in bits: -sum(p * log2(p)).
    """
    if isinstance(values, (int, float)):
        p = float(values)
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        p = float(arr.flat[0])
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    total = arr.sum()
    if total <= 0.0:
        return 0.0
    p_norm = arr / total
    nonzero = p_norm > 0.0
    return float(-np.sum(p_norm[nonzero] * np.log2(p_norm[nonzero])))


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


def anova_rig_from_group_stats(
    counts: Sequence[Union[int, float]],
    means: Sequence[float],
    stds: Sequence[float],
    ddof: int = 1
) -> dict:
    """Calculates continuous Relative Information Gain (RIG) and ANOVA F-statistic
    directly from SQL group summary statistics: COUNT(*), AVG(y), and STDDEV(y).

    Based on the exact Gaussian mutual information identity:
        SS_between = sum(n_r * (mean_r - grand_mean)^2)
        SS_within  = sum((n_r - ddof) * std_r^2)
        eta^2 = SS_between / (SS_between + SS_within)
        I(X; Y) = -0.5 * log2(1 - eta^2)
        Deviance = 2 * N * ln(2) * I(X; Y) = -N * ln(1 - eta^2) ~ Chi2_{R-1}

    Parameters
    ----------
    counts : Sequence of int or float
        Sample counts (n_r) in each group r in {1, ..., R}.
    means : Sequence of float
        Group means (bar{Y}_r) in each group r.
    stds : Sequence of float
        Group sample standard deviations (s_r) in each group r.
    ddof : int, default 1
        Delta degrees of freedom for sample standard deviation (1 for sample std, 0 for population std).

    Returns
    -------
    dict
        Dictionary containing:
          - 'observed_rig': float in [0, 1]
          - 'eta_squared': float in [0, 1]
          - 'f_statistic': float (one-way ANOVA F)
          - 'p_value': float (exact F-distribution p-value)
          - 'deviance_chi2': float (asymptotic Wilks deviance)
          - 'df_between': int (R - 1)
          - 'df_within': int (N - R)
          - 'total_nrows': int (N)
          - 'grand_mean': float (bar{Y})
          - 'min_cell_count': int
    """
    from scipy import stats

    counts_arr = np.asarray(counts, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)

    valid = (counts_arr > 0) & np.isfinite(means_arr) & np.isfinite(stds_arr)
    counts_arr = counts_arr[valid]
    means_arr = means_arr[valid]
    stds_arr = stds_arr[valid]

    r = len(counts_arr)
    if r <= 1:
        return {
            "observed_rig": 0.0,
            "eta_squared": 0.0,
            "f_statistic": 0.0,
            "p_value": 1.0,
            "deviance_chi2": 0.0,
            "df_between": 0,
            "df_within": 0,
            "total_nrows": int(counts_arr.sum()) if r == 1 else 0,
            "grand_mean": float(means_arr[0]) if r == 1 else 0.0,
            "min_cell_count": int(counts_arr[0]) if r == 1 else 0,
        }

    total_n = float(counts_arr.sum())
    grand_mean = float(np.sum(counts_arr * means_arr) / total_n)

    ss_between = float(np.sum(counts_arr * (means_arr - grand_mean) ** 2))
    df_within_per_group = np.maximum(0.0, counts_arr - float(ddof))
    ss_within = float(np.sum(df_within_per_group * (stds_arr ** 2)))
    ss_total = ss_between + ss_within

    if ss_total <= 1e-15:
        return {
            "observed_rig": 0.0,
            "eta_squared": 0.0,
            "f_statistic": 0.0,
            "p_value": 1.0,
            "deviance_chi2": 0.0,
            "df_between": r - 1,
            "df_within": int(total_n - r),
            "total_nrows": int(total_n),
            "grand_mean": grand_mean,
            "min_cell_count": int(np.min(counts_arr)),
        }

    eta2 = float(np.clip(ss_between / ss_total, 0.0, 0.999999))
    df1 = r - 1
    df2 = max(1, int(total_n - r))

    ms_between = ss_between / float(df1)
    ms_within = ss_within / float(df2) if ss_within > 1e-15 else 1e-15
    f_stat = float(ms_between / ms_within)

    p_val = float(stats.f.sf(f_stat, df1, df2))

    # Exact Gaussian mutual information: I(X; Y) = -0.5 * log2(1 - eta^2)
    # Relative Information Gain for continuous targets is normalized by Gaussian target entropy:
    # RIG = 1 - (1 - eta^2) = eta^2 (asymptotic first-order), or exact normalized MI:
    # observed_rig = -0.5 * log2(1 - eta^2)
    obs_rig = float(np.clip(-0.5 * np.log2(max(1e-12, 1.0 - eta2)), 0.0, 1.0))
    deviance = float(-total_n * np.log(max(1e-12, 1.0 - eta2)))

    return {
        "observed_rig": obs_rig,
        "eta_squared": eta2,
        "f_statistic": f_stat,
        "p_value": p_val,
        "deviance_chi2": deviance,
        "df_between": df1,
        "df_within": df2,
        "total_nrows": int(total_n),
        "grand_mean": grand_mean,
        "min_cell_count": int(np.min(counts_arr)),
    }


def fast_continuous_rig(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series]
) -> dict:
    """Calculates continuous Relative Information Gain between a categorical feature X
    and a continuous target Y directly from raw series without binning Y.
    """
    df_temp = pd.DataFrame({"x": x, "y": y}).dropna()
    grouped = df_temp.groupby("x")["y"].agg(["count", "mean", "std"]).reset_index()
    return anova_rig_from_group_stats(
        counts=grouped["count"].values,
        means=grouped["mean"].values,
        stds=grouped["std"].fillna(0.0).values
    )

