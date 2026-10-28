"""Unit tests for Multiclass Classification Targets and Multi-Category Discrete Features."""

import unittest
import numpy as np
import pandas as pd

from signalguard.metrics import entropy, rig, fast_contingency_matrix
from signalguard.threshold import FeatureNoiseThreshold, estimate_noise_threshold, target_entropy_from_weights
from signalguard.permutation import PermutationObject
from signalguard.mcp_server import NoiseGateSessionLedger
from signalguard.agent_sandbox import InstrumentedAgentSandbox


class TestMulticlass(unittest.TestCase):
    """Test suite validating multiclass (C >= 2) and multi-category (R >= 2) generalizations."""

    def test_binary_reduction(self):
        """Test 1: Binary reduction (k=1) matches scalar minority_frac to 10^-12."""
        n = 2500
        p = 0.22
        m_values = [1, 5, 20, 100]

        # Binary estimator with scalar minority_frac
        est_binary = FeatureNoiseThreshold(nrows=n, minority_frac=p)

        # Multiclass estimator with C=2 probability vector [1-p, p] and R=2 categories (k=1)
        est_multiclass = FeatureNoiseThreshold(
            nrows=n,
            target_distribution=[1.0 - p, p],
            n_classes=2,
            n_feature_categories=2,
            df=1
        )

        # 1. Target entropy check
        h_bin = target_entropy_from_weights(p)
        h_multi = target_entropy_from_weights([1.0 - p, p])
        np.testing.assert_allclose(h_bin, h_multi, atol=1e-12)

        # 2. Closed-form Bonferroni / Šidák gates across multiple hypothesis budgets M
        for m in m_values:
            tau_bin = est_binary.analytical_threshold(n_features_evaluated=m, alpha=0.05)
            tau_multi = est_multiclass.analytical_threshold(n_features_evaluated=m, alpha=0.05)
            np.testing.assert_allclose(tau_bin, tau_multi, atol=1e-12)

        # 3. EVT Expected Max across M
        for m in m_values:
            evt_bin = est_binary.evt_expected_max(n_features_evaluated=m)
            evt_multi = est_multiclass.evt_expected_max(n_features_evaluated=m)
            np.testing.assert_allclose(evt_bin, evt_multi, atol=1e-12)

    def test_asymptotic_chi2_calibration_vs_monte_carlo(self):
        """Test 2: Calibration vs Monte Carlo: N=4000, C=4, R=3 -> k=6 matches analytical mean and p95."""
        n = 4000
        class_weights = [0.4, 0.3, 0.2, 0.1]
        feature_fractions = [[0.5, 0.3, 0.2]]
        n_permutations = 2000
        seed = 42

        # C = 4 classes, R = 3 categories -> k = (3-1)*(4-1) = 6 degrees of freedom
        k_df = 6
        h_y = target_entropy_from_weights(class_weights)
        expected_mean = k_df / (2.0 * n * np.log(2.0) * h_y)

        estimator = FeatureNoiseThreshold(
            nrows=n,
            target_distribution=class_weights,
            n_classes=4,
            n_feature_categories=3,
            df=k_df
        )
        expected_p95 = estimator.analytical_threshold(n_features_evaluated=1, alpha=0.05)

        po = PermutationObject()
        po.set_data_parameters(nrows=n, class_weights=class_weights)
        po.feature_fractions = feature_fractions
        po.create_synthetic_data(seed=seed)

        df_nulls = po.calculate_null_distribution(permutations=n_permutations, seed=seed, verbose=False)
        null_vals = df_nulls.iloc[:, 0].to_numpy()

        actual_mean = float(np.mean(null_vals))
        actual_p95 = float(np.percentile(null_vals, 95))

        # Monte-Carlo estimates must be within 3% of closed-form Wilks' theorem asymptotic values
        mean_relative_error = abs(actual_mean - expected_mean) / expected_mean
        p95_relative_error = abs(actual_p95 - expected_p95) / expected_p95

        self.assertLess(mean_relative_error, 0.03, f"Mean error {mean_relative_error:.4f} exceeds 3%")
        self.assertLess(p95_relative_error, 0.03, f"P95 error {p95_relative_error:.4f} exceeds 3%")

    def test_high_cardinality_penalty(self):
        """Test 3: High-cardinality penalty: C=4, R=10 (k=27) has higher thresholds than R=2 (k=3)."""
        n = 3000
        target_dist = [0.25, 0.25, 0.25, 0.25]  # C = 4 classes

        # Low cardinality: binary feature R=2 -> k = (2-1)*(4-1) = 3
        est_binary_feat = FeatureNoiseThreshold(
            nrows=n,
            target_distribution=target_dist,
            n_classes=4,
            n_feature_categories=2,
            df=3
        )

        # High cardinality: 10-category feature R=10 -> k = (10-1)*(4-1) = 27
        est_high_feat = FeatureNoiseThreshold(
            nrows=n,
            target_distribution=target_dist,
            n_classes=4,
            n_feature_categories=10,
            df=27
        )

        for m in [1, 10, 50]:
            tau_sig_bin = est_binary_feat.analytical_threshold(n_features_evaluated=m)
            tau_sig_high = est_high_feat.analytical_threshold(n_features_evaluated=m)
            self.assertGreater(
                tau_sig_high,
                tau_sig_bin,
                f"High-cardinality gate ({tau_sig_high}) must exceed binary gate ({tau_sig_bin}) for M={m}"
            )

            tau_evt_bin = est_binary_feat.evt_expected_max(n_features_evaluated=m)
            tau_evt_high = est_high_feat.evt_expected_max(n_features_evaluated=m)
            self.assertGreater(
                tau_evt_high,
                tau_evt_bin,
                f"High-cardinality EVT ceiling ({tau_evt_high}) must exceed binary ceiling ({tau_evt_bin}) for M={m}"
            )

    def test_agent_sandbox_multiclass(self):
        """Test 4: Agent sandbox multiclass run on 4-class synthetic dataset."""
        rng = np.random.default_rng(42)
        n = 1200

        # 4-class target
        target = rng.choice([0, 1, 2, 3], size=n, p=[0.40, 0.30, 0.20, 0.10])

        # Feature 1: Pure noise binary
        feat_noise_bin = rng.choice([0, 1], size=n, p=[0.70, 0.30])

        # Feature 2: Pure noise 5-category
        feat_noise_cat5 = rng.choice([0, 1, 2, 3, 4], size=n)

        # Feature 3: Injected strong signal (aligned with target class 0 and 1)
        feat_signal = target.copy()
        flip_mask = rng.random(n) < 0.20
        feat_signal[flip_mask] = rng.choice([0, 1, 2, 3], size=int(flip_mask.sum()))

        df = pd.DataFrame({
            "target": target,
            "feat_noise_bin": feat_noise_bin,
            "feat_noise_cat5": feat_noise_cat5,
            "feat_signal": feat_signal,
        })

        sandbox = InstrumentedAgentSandbox(df=df, target_col="target", guardrail_enabled=True, session_id="multi_test")

        # 1. Scan single binary noise feature
        res_bin = sandbox.scan_single_feature("feat_noise_bin")
        self.assertEqual(res_bin["turn"], 1)
        self.assertEqual(res_bin["verdict"], "REJECT_SPURIOUS_NOISE")

        # 2. Scan single high-cardinality noise feature
        res_cat = sandbox.scan_single_feature("feat_noise_cat5")
        self.assertEqual(res_cat["turn"], 2)
        self.assertEqual(res_cat["verdict"], "REJECT_SPURIOUS_NOISE")

        # 3. Scan genuine signal feature
        res_sig = sandbox.scan_single_feature("feat_signal")
        self.assertEqual(res_sig["turn"], 3)
        self.assertEqual(res_sig["verdict"], "VERIFIED_SIGNAL")

        # Check telemetry columns and degrees of freedom
        telemetry = sandbox.get_telemetry_df()
        self.assertEqual(len(telemetry), 3)

        # Row 0: binary feature (R=2, C=4 -> k=3)
        self.assertEqual(telemetry.iloc[0]["degrees_of_freedom_k"], 3)
        self.assertEqual(telemetry.iloc[0]["n_classes"], 4)
        self.assertEqual(telemetry.iloc[0]["n_feature_categories"], 2)

        # Row 1: 5-category feature (R=5, C=4 -> k=12)
        self.assertEqual(telemetry.iloc[1]["degrees_of_freedom_k"], 12)
        self.assertEqual(telemetry.iloc[1]["n_classes"], 4)
        self.assertEqual(telemetry.iloc[1]["n_feature_categories"], 5)

        # Verify audit_dataframe also works with multiclass
        ledger = NoiseGateSessionLedger()
        audit_res = ledger.audit_dataframe(
            df_train=df,
            target_col="target",
            candidate_cols=["feat_noise_bin", "feat_noise_cat5", "feat_signal"],
            n_bootstraps=30,
            session_id="audit_multi"
        )
        self.assertEqual(audit_res["n_classes"], 4)
        self.assertIn("feat_signal", audit_res["promoted_2_sigma_features"])
        self.assertIn("feat_noise_bin", audit_res["rejected_noise_features"])
        self.assertIn("feat_noise_cat5", audit_res["rejected_noise_features"])

    def test_cochran_sparse_multiclass_warning(self):
        """Test 5: Verify Cochran's rule small-cell warning on sparse multiclass contingency table."""
        ledger = NoiseGateSessionLedger()

        # Case A: Sparse 3 x 4 contingency table with very small counts triggering Cochran's rule
        # Table shape: 3 rows (categories), 4 columns (classes)
        sparse_table = [
            [2, 1, 0, 1],
            [1, 0, 2, 0],
            [40, 35, 25, 20]
        ]
        res_sparse = ledger.evaluate_feature_or_slice(
            contingency_counts=sparse_table,
            session_id="sparse_session"
        )
        self.assertTrue(res_sparse["small_cell_warning"])
        self.assertIn("small-cell", res_sparse["agent_directive"].lower())
        self.assertEqual(res_sparse["degrees_of_freedom_k"], (3 - 1) * (4 - 1))
        self.assertEqual(res_sparse["n_classes"], 4)
        self.assertEqual(res_sparse["n_feature_categories"], 3)

        # Case B: Dense 3 x 4 contingency table where all expected cell counts >= 5
        dense_table = [
            [100, 100, 100, 100],
            [100, 100, 100, 100],
            [100, 100, 100, 100]
        ]
        res_dense = ledger.evaluate_feature_or_slice(
            contingency_counts=dense_table,
            session_id="dense_session"
        )
        self.assertFalse(res_dense["small_cell_warning"])
        self.assertNotIn("WARNING", res_dense["agent_directive"])
        self.assertEqual(res_dense["degrees_of_freedom_k"], 6)


if __name__ == "__main__":
    unittest.main()
