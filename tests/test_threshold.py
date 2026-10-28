"""Unit tests for analytical noise thresholds."""

import unittest
from signalguard.threshold import FeatureNoiseThreshold, estimate_noise_threshold, binary_entropy


class TestThreshold(unittest.TestCase):

    def test_binary_entropy(self):
        self.assertAlmostEqual(binary_entropy(0.5), 1.0, places=6)
        self.assertAlmostEqual(binary_entropy(0.0), 0.0, places=6)
        self.assertAlmostEqual(binary_entropy(1.0), 0.0, places=6)
        self.assertAlmostEqual(binary_entropy(0.1), 0.4689955935892812, places=5)

    def test_sample_size_scaling(self):
        # Expected null mean must scale inversely with sample size N: O(1/N)
        est_1000 = FeatureNoiseThreshold(nrows=1000, target_distribution=0.5)
        est_2000 = FeatureNoiseThreshold(nrows=2000, target_distribution=0.5)
        self.assertAlmostEqual(est_1000.expected_null_mean, 2.0 * est_2000.expected_null_mean, places=6)

    def test_search_space_penalty(self):
        # As evaluated features M grows, threshold must strictly increase
        est = FeatureNoiseThreshold(nrows=2000, target_distribution=0.3)
        tau_10 = est.analytical_threshold(n_features_evaluated=10, alpha=0.05)
        tau_100 = est.analytical_threshold(n_features_evaluated=100, alpha=0.05)
        tau_1000 = est.analytical_threshold(n_features_evaluated=1000, alpha=0.05)

        self.assertLess(tau_10, tau_100)
        self.assertLess(tau_100, tau_1000)

    def test_evt_max_growth(self):
        est = FeatureNoiseThreshold(nrows=5000, target_distribution=0.2)
        evt_100 = est.evt_expected_max(100)
        evt_10000 = est.evt_expected_max(10000)
        self.assertLess(evt_100, evt_10000)

    def test_convenience_function(self):
        tau = estimate_noise_threshold(nrows=2000, minority_frac=0.3, n_features_evaluated=500)
        self.assertGreater(tau, 0.0)
        self.assertLess(tau, 1.0)


if __name__ == '__main__':
    unittest.main()
