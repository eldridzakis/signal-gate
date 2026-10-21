"""Feature de-biasing and stability classification engine.

Implements selection bias estimation (The Winner's Curse),
Out-of-Fold (OOF) cross-validation RIG, and Bootstrap 1-sigma / 2-sigma
stability classification for automated feature engineering.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .metrics import rig, rig_all_columns


class FeatureDebiaser:
    """Engine for estimating and correcting feature selection bias.

    When automated feature engineering selects features that maximize train-set RIG,
    observed train performance is systematically inflated above its true population value.
    This module quantifies:
        Bias = RIG_train - RIG_unseen
    and corrects feature rankings via:
        RIG_corrected = RIG_train - Bias
    followed by bootstrap confidence interval testing.
    """

    @staticmethod
    def compute_holdout_bias(
        df_train: pd.DataFrame,
        df_eval: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        eval_label: str = "test"
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

        Returns
        -------
        pd.DataFrame
            DataFrame with train_rig, eval_rig, bias, and overfitting ratio.
        """
        if feature_cols is None:
            common = set(df_train.columns).intersection(set(df_eval.columns))
            feature_cols = [c for c in df_train.columns if c in common and c != target]

        records = []
        for feat in feature_cols:
            train_r = rig(df_train[feat], df_train[target])
            eval_r = rig(df_eval[feat], df_eval[target])
            bias = train_r - eval_r
            overfitting_ratio = (bias / train_r) if train_r > 1e-12 else 0.0

            records.append({
                "feature": feat,
                "train_rig": train_r,
                f"{eval_label}_rig": eval_r,
                f"{eval_label}_bias": bias,
                f"overfitting_ratio_{eval_label}": overfitting_ratio
            })

        df_metrics = pd.DataFrame(records)
        return df_metrics.sort_values(by="train_rig", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_oof_rig(
        df: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        n_splits: int = 5,
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Computes Out-of-Fold (OOF) cross-validated RIG to eliminate selection leakage.

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
        seed : int, optional
            Random state for fold assignment.

        Returns
        -------
        pd.DataFrame
            DataFrame with train_rig, mean_oof_rig, and oof_bias.
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
            train_r = rig(df[feat], df[target])
            oof_rigs = []
            for fold in range(n_splits):
                val_mask = fold_assignments == fold
                val_feat = df[feat].iloc[val_mask]
                val_target = df[target].iloc[val_mask]
                oof_rigs.append(rig(val_feat, val_target))

            mean_oof_r = float(np.mean(oof_rigs))
            oof_bias = train_r - mean_oof_r
            records.append({
                "feature": feat,
                "train_rig": train_r,
                "oof_rig": mean_oof_r,
                "oof_bias": oof_bias,
                "overfitting_ratio_oof": (oof_bias / train_r) if train_r > 1e-12 else 0.0
            })

        df_res = pd.DataFrame(records)
        return df_res.sort_values(by="train_rig", ascending=False).reset_index(drop=True)

    @staticmethod
    def bootstrap_stability_analysis(
        df: pd.DataFrame,
        target: str,
        feature_cols: Optional[Sequence[str]] = None,
        estimated_bias: Optional[Union[float, Dict[str, float]]] = None,
        n_bootstraps: int = 100,
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Performs bootstrap estimation of RIG standard errors and classifies feature stability.

        Implements the 1-sigma and 2-sigma confidence criteria:
            RIG_corrected = RIG_train - Bias
            2-Sigma Stable: RIG_corrected - 2 * sigma_boot > 0
            1-Sigma Stable: RIG_corrected - sigma_boot > 0

        Parameters
        ----------
        df : pd.DataFrame
            Dataset (training sample).
        target : str
            Target column.
        feature_cols : Sequence[str], optional
            Features to evaluate.
        estimated_bias : float or Dict[str, float], optional
            Estimated selection bias to subtract. Can be a single float (e.g. median holdout bias
            or EVT noise floor) or a dictionary mapping feature names to individual bias values.
            If None, defaults to 0.0.
        n_bootstraps : int, default 100
            Number of bootstrap iterations.
        seed : int, optional
            Random seed.

        Returns
        -------
        pd.DataFrame
            DataFrame with train_rig, estimated_bias, corrected_rig, sigma_boot,
            is_1_sigma, is_2_sigma, and stability_class.
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target]

        n = len(df)
        rng = np.random.default_rng(seed)

        # Baseline train RIG
        train_rigs = {feat: rig(df[feat], df[target]) for feat in feature_cols}

        # Bootstrap distributions
        boot_rigs = {feat: [] for feat in feature_cols}
        for _ in range(n_bootstraps):
            boot_idx = rng.choice(n, size=n, replace=True)
            df_boot = df.iloc[boot_idx]
            for feat in feature_cols:
                boot_rigs[feat].append(rig(df_boot[feat], df_boot[target]))

        records = []
        for feat in feature_cols:
            tr_rig = train_rigs[feat]
            sigma = float(np.std(boot_rigs[feat], ddof=1))

            if estimated_bias is None:
                bias_val = 0.0
            elif isinstance(estimated_bias, dict):
                bias_val = float(estimated_bias.get(feat, 0.0))
            else:
                bias_val = float(estimated_bias)

            corrected_rig = max(0.0, tr_rig - bias_val)
            is_1_sigma = (corrected_rig - sigma) > 0.0
            is_2_sigma = (corrected_rig - 2.0 * sigma) > 0.0

            if is_2_sigma:
                status = "Highly Stable (2-Sigma)"
            elif is_1_sigma:
                status = "Moderately Stable (1-Sigma)"
            else:
                status = "Unstable / Noise"

            records.append({
                "feature": feat,
                "train_rig": tr_rig,
                "estimated_bias": bias_val,
                "corrected_rig": corrected_rig,
                "sigma_boot": sigma,
                "is_1_sigma": is_1_sigma,
                "is_2_sigma": is_2_sigma,
                "stability_class": status
            })

        df_stab = pd.DataFrame(records)
        return df_stab.sort_values(by="corrected_rig", ascending=False).reset_index(drop=True)
