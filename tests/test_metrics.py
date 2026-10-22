"""Unit tests for information-theoretic metrics."""

import unittest
import numpy as np
import pandas as pd
from signalgate.metrics import entropy, rig, rig_all_columns, fast_contingency_matrix
import signalgate
import feature_noise_gate
import sampling_toolbox


class TestMetrics(unittest.TestCase):

    def test_entropy_properties(self):
        # Deterministic
        self.assertAlmostEqual(entropy([1.0, 0.0]), 0.0, places=6)
        self.assertAlmostEqual(entropy([0.0, 1.0]), 0.0, places=6)
        # Uniform binary
        self.assertAlmostEqual(entropy([0.5, 0.5]), 1.0, places=6)
        # Empty or single
        self.assertAlmostEqual(entropy([1.0]), 0.0, places=6)

    def test_rig_perfect_correlation(self):
        # Feature identical to target
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        x = y.copy()
        self.assertAlmostEqual(rig(x, y), 1.0, places=5)

    def test_rig_zero_entropy_target(self):
        # Constant target should return 0.0 without division by zero error
        y = np.array([1, 1, 1, 1, 1])
        x = np.array([0, 1, 0, 1, 0])
        self.assertEqual(rig(x, y), 0.0)

    def test_rig_independent(self):
        # With large sample, independent feature should have RIG close to 0
        rng = np.random.default_rng(123)
        n = 10000
        x = rng.integers(0, 2, size=n)
        y = rng.integers(0, 2, size=n)
        r = rig(x, y)
        self.assertLess(r, 0.01)

    def test_rig_all_columns(self):
        df = pd.DataFrame({
            'feat1': [0, 1, 0, 1],
            'feat2': [1, 1, 0, 0],
            'target': [0, 1, 0, 1]
        })
        rigs = rig_all_columns(df, target='target')
        self.assertEqual(len(rigs), 2)
        self.assertAlmostEqual(rigs[0], 1.0, places=5)


    def test_backward_compatibility(self):
        # signalgate
        self.assertTrue(hasattr(signalgate, 'rig'))
        self.assertTrue(hasattr(signalgate, 'PermutationObject'))
        self.assertTrue(hasattr(signalgate, 'FeatureNoiseThreshold'))
        self.assertTrue(hasattr(signalgate, 'FeatureDebiaser'))
        self.assertTrue(hasattr(signalgate, 'NoiseGateSessionLedger'))
        self.assertTrue(hasattr(signalgate, 'SignalGateSessionLedger'))

        # feature_noise_gate alias
        self.assertTrue(hasattr(feature_noise_gate, 'rig'))
        self.assertTrue(hasattr(feature_noise_gate, 'PermutationObject'))
        self.assertTrue(hasattr(feature_noise_gate, 'FeatureNoiseThreshold'))
        self.assertTrue(hasattr(feature_noise_gate, 'FeatureDebiaser'))

        # sampling_toolbox alias
        self.assertTrue(hasattr(sampling_toolbox, 'rig'))
        self.assertTrue(hasattr(sampling_toolbox, 'PermutationObject'))
        self.assertTrue(hasattr(sampling_toolbox, 'FeatureNoiseThreshold'))
        self.assertTrue(hasattr(sampling_toolbox, 'FeatureDebiaser'))


if __name__ == '__main__':
    unittest.main()
