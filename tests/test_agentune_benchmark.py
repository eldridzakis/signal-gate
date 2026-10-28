"""Unit tests for the Agentune AutoML Overfitting & SignalGate Benchmark."""

import unittest
import numpy as np
import pandas as pd

from signalgate.agentune_benchmark import (
    AgentuneFeatureSynthesizer,
    calculate_holdout_replication_metrics,
    run_agentune_overfitting_benchmark,
)


class TestAgentuneBenchmark(unittest.TestCase):
    """Test suite for Agentune feature synthesis and overfitting benchmarks."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        n_tr, n_te = 200, 100
        self.train_df = pd.DataFrame({
            "cat_a": self.rng.choice(["low", "med", "high"], size=n_tr),
            "cat_b": self.rng.choice(["type_1", "type_2"], size=n_tr),
            "num_bin": self.rng.choice(["bin_0", "bin_1", "bin_2"], size=n_tr),
            "target": self.rng.choice([0, 1], size=n_tr, p=[0.7, 0.3])
        })
        self.test_df = pd.DataFrame({
            "cat_a": self.rng.choice(["low", "med", "high"], size=n_te),
            "cat_b": self.rng.choice(["type_1", "type_2"], size=n_te),
            "num_bin": self.rng.choice(["bin_0", "bin_1", "bin_2"], size=n_te),
            "target": self.rng.choice([0, 1], size=n_te, p=[0.7, 0.3])
        })

    def test_synthesizer_generates_diverse_candidates(self):
        synth = AgentuneFeatureSynthesizer(seed=42)
        cands = synth.synthesize_candidates(
            train_df=self.train_df,
            test_df=self.test_df,
            candidate_cols=["cat_a", "cat_b", "num_bin"],
            max_candidates=50,
            noise_feature_ratio=0.20
        )
        self.assertGreater(len(cands), 0)
        types = {c.feature_type for c in cands}
        self.assertTrue("univariate_slice" in types)
        for c in cands:
            self.assertEqual(len(c.train_vector), len(self.train_df))
            self.assertEqual(len(c.test_vector), len(self.test_df))

    def test_calculate_holdout_replication_metrics(self):
        y_test = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        # Perfect signal
        perfect_feat = y_test.copy()
        res_perf = calculate_holdout_replication_metrics(y_test, perfect_feat, train_rig=0.5)
        self.assertGreater(res_perf["holdout_rig"], 0.4)
        self.assertLess(res_perf["decay_pct"], 20.0)

        # Pure noise feature
        noise_feat = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        res_noise = calculate_holdout_replication_metrics(y_test, noise_feat, train_rig=0.1)
        self.assertIn("generalization_bias", res_noise)
        self.assertIn("holdout_p_value", res_noise)

    def test_run_agentune_overfitting_benchmark_execution(self):
        res = run_agentune_overfitting_benchmark(
            dataset_name="lending_club",
            max_candidates=40,
            top_k=5,
            sample_train=300,
            sample_test=150,
            seed=42
        )
        self.assertIn("summary_df", res)
        self.assertIn("downstream_df", res)
        self.assertIn("unassisted_verified_df", res)
        self.assertIn("signalgate_verified_df", res)

        summary = res["summary_df"]
        self.assertEqual(len(summary), 2)
        self.assertTrue("Holdout_Replication_Rate (%)" in summary.columns)
        self.assertTrue("Real_FDR (%)" in summary.columns)

        downstream = res["downstream_df"]
        self.assertEqual(len(downstream), 3)
        self.assertTrue("Train_AUC" in downstream.columns)
        self.assertTrue("Holdout_Test_AUC" in downstream.columns)


if __name__ == "__main__":
    unittest.main()
