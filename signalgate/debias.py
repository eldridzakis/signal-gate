"""Feature de-biasing and stability classification engine.

Implements selection bias estimation (The Winner's Curse),
Out-of-Fold (OOF) cross-validation, and Bootstrap 1-sigma / 2-sigma
stability classification for automated feature engineering across arbitrary metrics.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .metrics import rig, rig_all_columns


class FeatureDebiaser:
    """Engine for estimating and correcting feature selection bias.

    When automated feature engineering selects features that maximize train-set performance,
    observed train performance is systematically inflated above its true population value.
    This module quantifies:
        Bias = Metric_train - Metric_unseen
    and corrects feature rankings via:
        Metric_corrected = Metric_train - Bias
    followed by bootstrap confidence interval testing. Supports arbitrary metric functions.
    """

    @staticmethod
    def compute_holdout_bias(
        df_train: pd.DataFrame,
        df_eval: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        eval_label: str = "test",
        metric_fn: Callable[[pd.Series, pd.Series], float] = rig,
        metric_name: str = "rig"
    ) -> pd.DataFrame:
        """Computes selection bias between training data and holdout/population data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training set used for feature discovery/selection.
        df_eval : pd.DataFrame
            Evaluation set (test set or unseen population sample).
        target : str
            Name of target column.
        feature_cols : Sequence[str], optional
            Columns to evaluate. If None, all common non-target columns are used.
        eval_label : str, default 'test'
            Prefix label for evaluation metrics ('test' or 'population').
        metric_fn : Callable[[pd.Series, pd.Series], float], default rig
            Function computing the performance metric between feature and target.
        metric_name : str, default 'rig'
            Name label for the metric column.

        Returns
        -------
        pd.DataFrame
            DataFrame with train_metric, eval_metric, bias, and overfitting ratio.
        """
        if feature_cols is None:
            common = set(df_train.columns).intersection(set(df_eval.columns))
            feature_cols = [c for c in df_train.columns if c in common and c != target]

        records = []
        for feat in feature_cols:
            train_val = float(metric_fn(df_train[feat], df_train[target]))
            eval_val = float(metric_fn(df_eval[feat], df_eval[target]))
            bias = train_val - eval_val
            overfitting_ratio = (bias / train_val) if train_val > 1e-12 else 0.0

            rec = {
                "feature": feat,
                f"train_{metric_name}": train_val,
                f"{eval_label}_{metric_name}": eval_val,
                f"{eval_label}_bias": bias,
                f"overfitting_ratio_{eval_label}": overfitting_ratio
            }
            # Also keep test_bias / train_rig aliases if metric_name is rig
            records.append(rec)

        df_metrics = pd.DataFrame(records)
        return df_metrics.sort_values(by=f"train_{metric_name}", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_oof_metric(
        df: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        n_splits: int = 5,
        metric_fn: Callable[[pd.Series, pd.Series], float] = rig,
        metric_name: str = "rig",
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Computes Out-of-Fold (OOF) cross-validated metric to eliminate selection leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset.
        target : str
            Target column name.
        feature_cols : Sequence[str], optional
            Columns to evaluate.
        n_splits : int, default 5
            Number of cross-validation folds.
        metric_fn : Callable, default rig
            Metric function to evaluate.
        metric_name : str, default 'rig'
            Metric label.
        seed : int, optional
            Random state for fold assignment.

        Returns
        -------
        pd.DataFrame
            DataFrame with train_metric, mean_oof_metric, and oof_bias.
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target]

        rng = np.random.default_rng(seed)
        n = len(df)
        indices = np.arange(n)
        rng.shuffle(indices)
        fold_assignments = np.empty(n, dtype=int)
        for i, idx in enumerate(indices):
            fold_assignments[idx] = i % n_splits

        records = []
        for feat in feature_cols:
            train_val = float(metric_fn(df[feat], df[target]))
            oof_vals = []
            for fold in range(n_splits):
                val_mask = fold_assignments == fold
                val_feat = df[feat].iloc[val_mask]
                val_target = df[target].iloc[val_mask]
                oof_vals.append(float(metric_fn(val_feat, val_target)))

            mean_oof_val = float(np.mean(oof_vals))
            oof_bias = train_val - mean_oof_val
            records.append({
                "feature": feat,
                f"train_{metric_name}": train_val,
                f"oof_{metric_name}": mean_oof_val,
                "oof_bias": oof_bias,
                "overfitting_ratio_oof": (oof_bias / train_val) if train_val > 1e-12 else 0.0
            })

        df_res = pd.DataFrame(records)
        return df_res.sort_values(by=f"train_{metric_name}", ascending=False).reset_index(drop=True)

    # Alias for backward compatibility
    compute_oof_rig = compute_oof_metric

    @staticmethod
    def bootstrap_stability_analysis(
        df: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        estimated_bias: Optional[Union[float, Dict[str, float]]] = None,
        metric_fn: Callable[[pd.Series, pd.Series], float] = rig,
        metric_name: str = "rig",
        n_bootstraps: int = 100,
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Performs bootstrap estimation of standard errors and classifies feature stability.

        Implements the 1-sigma and 2-sigma confidence criteria:
            Metric_corrected = Metric_train - Bias
            2-Sigma Stable: Metric_corrected - 2 * sigma_boot > 0
            1-Sigma Stable: Metric_corrected - sigma_boot > 0

        Parameters
        ----------
        df : pd.DataFrame
            Dataset (training sample).
        target : str
            Target column.
        feature_cols : Sequence[str], optional
            Features to evaluate.
        estimated_bias : float or Dict[str, float], optional
            Estimated selection bias to subtract.
        metric_fn : Callable, default rig
            Metric function to evaluate.
        metric_name : str, default 'rig'
            Name of metric for output columns.
        n_bootstraps : int, default 100
            Number of bootstrap iterations.
        seed : int, optional
            Random seed.

        Returns
        -------
        pd.DataFrame
            DataFrame with train_metric, estimated_bias, corrected_metric, sigma_boot,
            is_1_sigma, is_2_sigma, and stability_class.
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target]

        n = len(df)
        rng = np.random.default_rng(seed)

        # Baseline train metric
        train_vals = {feat: float(metric_fn(df[feat], df[target])) for feat in feature_cols}

        # Bootstrap distributions
        boot_vals = {feat: [] for feat in feature_cols}
        for _ in range(n_bootstraps):
            boot_idx = rng.choice(n, size=n, replace=True)
            df_boot = df.iloc[boot_idx]
            for feat in feature_cols:
                boot_vals[feat].append(float(metric_fn(df_boot[feat], df_boot[target])))

        records = []
        for feat in feature_cols:
            tr_val = train_vals[feat]
            sigma = float(np.std(boot_vals[feat], ddof=1))

            if estimated_bias is None:
                bias_val = 0.0
            elif isinstance(estimated_bias, dict):
                bias_val = float(estimated_bias.get(feat, 0.0))
            else:
                bias_val = float(estimated_bias)

            corrected_val = max(0.0, tr_val - bias_val)
            is_1_sigma = (corrected_val - sigma) > 0.0
            is_2_sigma = (corrected_val - 2.0 * sigma) > 0.0

            if is_2_sigma:
                status = "Highly Stable (2-Sigma)"
            elif is_1_sigma:
                status = "Moderately Stable (1-Sigma)"
            else:
                status = "Unstable / Noise"

            records.append({
                "feature": feat,
                f"train_{metric_name}": tr_val,
                "estimated_bias": bias_val,
                f"corrected_{metric_name}": corrected_val,
                "sigma_boot": sigma,
                "is_1_sigma": is_1_sigma,
                "is_2_sigma": is_2_sigma,
                "stability_class": status
            })

        df_stab = pd.DataFrame(records)
        return df_stab.sort_values(by=f"corrected_{metric_name}", ascending=False).reset_index(drop=True)
