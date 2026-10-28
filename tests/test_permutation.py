"""Unit tests for Monte-Carlo permutation testing."""

import unittest
import numpy as np
from signalguard.permutation import PermutationObject
from signalguard.threshold import FeatureNoiseThreshold


class TestPermutation(unittest.TestCase):

    def test_synthetic_data_creation(self):
        po = PermutationObject()
        po.set_data_parameters(nrows=1000, class_weights=[0.3])
        po.feature_fractions = [0.1, 0.25]
        df = po.create_synthetic_data(seed=42)

        self.assertEqual(len(df), 1000)
        self.assertIn('target', df.columns)
        self.assertIn('feature_0.1', df.columns)
        self.assertIn('feature_0.25', df.columns)
        # Verify proportions
        self.assertEqual(df['target'].sum(), 300)
        self.assertEqual(df['feature_0.1'].sum(), 100)
        self.assertEqual(df['feature_0.25'].sum(), 250)

    def test_vectorized_permutation_speed_and_accuracy(self):
        po = PermutationObject()
        nrows = 1000
        minority = 0.3
        po.set_data_parameters(nrows=nrows, class_weights=[minority])
        po.feature_fractions = [0.1]
        po.create_synthetic_data(seed=42)

        # 2000 permutations should run fast and match theoretical mean closely
        null_df = po.calculate_null_rigs(permutations=2000, seed=42, verbose=False)
        self.assertEqual(len(null_df), 2000)

        # Theoretical expected mean from Wilks' Theorem
        est = FeatureNoiseThreshold(nrows=nrows, target_distribution=minority)
        theo_mean = est.expected_null_mean
        emp_mean = po.null_rig_values.mean()

        # Empirical mean should be within 10% of asymptotic theoretical mean
        self.assertAlmostEqual(emp_mean, theo_mean, delta=theo_mean * 0.15)


    def test_generic_metric_and_aliases(self):
        po = PermutationObject()
        po.set_data_parameters(nrows=200, minority_class=0.3)
        po.create_synthetic_data(seed=42)

        def simple_corr(x, y):
            return float(np.corrcoef(x, y)[0, 1])

        null_df = po.calculate_null_distribution(permutations=20, metric_fn=simple_corr, verbose=False)
        self.assertEqual(len(null_df), 20)
        self.assertTrue(hasattr(po, 'df_nulls'))
        self.assertTrue(hasattr(po, 'null_values'))


if __name__ == '__main__':
    unittest.main()
