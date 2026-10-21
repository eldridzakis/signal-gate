"""Unit tests for feature de-biasing and stability classification."""

import unittest
import numpy as np
import pandas as pd
from feature_noise_gate.debias import FeatureDebiaser


class TestDebias(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 500
        # Target
        y = rng.integers(0, 2, size=n)
        # Strong genuine feature
        x_genuine = y.copy()
        flip_mask = rng.random(n) < 0.2
        x_genuine[flip_mask] = 1 - x_genuine[flip_mask]
        # Pure noise feature
        x_noise = rng.integers(0, 2, size=n)

        self.df_train = pd.DataFrame({'genuine': x_genuine, 'noise': x_noise, 'target': y})

        # Separate holdout
        n_eval = 500
        y_eval = rng.integers(0, 2, size=n_eval)
        x_gen_eval = y_eval.copy()
        flip_mask_eval = rng.random(n_eval) < 0.2
        x_gen_eval[flip_mask_eval] = 1 - x_gen_eval[flip_mask_eval]
        x_noise_eval = rng.integers(0, 2, size=n_eval)

        self.df_eval = pd.DataFrame({'genuine': x_gen_eval, 'noise': x_noise_eval, 'target': y_eval})

    def test_holdout_bias(self):
        df_bias = FeatureDebiaser.compute_holdout_bias(
            self.df_train, self.df_eval, target='target'
        )
        self.assertEqual(len(df_bias), 2)
        self.assertIn('test_bias', df_bias.columns)
        # Genuine feature should have higher RIG than noise
        gen_rig = df_bias.loc[df_bias['feature'] == 'genuine', 'train_rig'].values[0]
        noise_rig = df_bias.loc[df_bias['feature'] == 'noise', 'train_rig'].values[0]
        self.assertGreater(gen_rig, noise_rig)

    def test_oof_rig(self):
        df_oof = FeatureDebiaser.compute_oof_rig(
            self.df_train, target='target', n_splits=5, seed=42
        )
        self.assertEqual(len(df_oof), 2)
        self.assertIn('oof_rig', df_oof.columns)
        self.assertIn('oof_bias', df_oof.columns)

    def test_bootstrap_stability(self):
        df_stab = FeatureDebiaser.bootstrap_stability_analysis(
            self.df_train, target='target', estimated_bias=0.005, n_bootstraps=50, seed=42
        )
        self.assertEqual(len(df_stab), 2)
        self.assertIn('is_1_sigma', df_stab.columns)
        self.assertIn('is_2_sigma', df_stab.columns)
        # Genuine feature should be 2-sigma stable
        gen_status = df_stab.loc[df_stab['feature'] == 'genuine', 'is_2_sigma'].values[0]
        self.assertTrue(gen_status)


if __name__ == '__main__':
    unittest.main()
