"""Unit tests for Effective Number of Hypotheses (M_eff) and Collinearity Guardrail."""

import time
import unittest
import numpy as np
import pandas as pd

from signalgate.threshold import (
    estimate_effective_tests,
    estimate_noise_threshold,
    FeatureNoiseThreshold,
)
from signalgate.agent_sandbox import (
    generate_benchmark_datasets,
    InstrumentedAgentSandbox,
    simulate_agent_archetypes,
)


class TestEffectiveTests(unittest.TestCase):
    """Test suite for Frobenius-norm trace M_eff estimation and correlation adjustments."""

    def test_identity_cases(self):
        """Test 1: Orthogonal columns have M_eff ~= M_raw, identical columns have M_eff ~= 1.0."""
        rng = np.random.default_rng(42)
        n = 500
        m_orth = 20

        # Case A: M mutually orthogonal columns
        # Generate random orthonormal matrix via QR decomposition
        h = rng.normal(size=(n, m_orth))
        q, _ = np.linalg.qr(h)
        res_orth = estimate_effective_tests(q)

        self.assertEqual(res_orth["M_raw"], m_orth)
        self.assertAlmostEqual(res_orth["M_eff"], float(m_orth), delta=0.2)
        self.assertAlmostEqual(res_orth["redundancy_ratio"], 1.0, delta=0.02)
        self.assertAlmostEqual(res_orth["mean_abs_correlation"], 0.0, delta=0.05)

        # Case B: M = 50 identical columns
        m_ident = 50
        single_vec = rng.normal(size=(n, 1))
        x_ident = np.repeat(single_vec, m_ident, axis=1)
        res_ident = estimate_effective_tests(x_ident)

        self.assertEqual(res_ident["M_raw"], m_ident)
        self.assertAlmostEqual(res_ident["M_eff"], 1.0, delta=1e-6)
        self.assertAlmostEqual(res_ident["redundancy_ratio"], 1.0 / m_ident, delta=1e-6)
        self.assertAlmostEqual(res_ident["mean_abs_correlation"], 1.0, delta=1e-6)

        # Verify estimate_noise_threshold with M_eff ~= 1.0 equals M=1 threshold
        tau_single = estimate_noise_threshold(nrows=n, minority_frac=0.2, n_features_evaluated=1)
        tau_adjusted = estimate_noise_threshold(
            nrows=n,
            minority_frac=0.2,
            n_features_evaluated=m_ident,
            effective_features_evaluated=res_ident["M_eff"]
        )
        tau_inflated = estimate_noise_threshold(nrows=n, minority_frac=0.2, n_features_evaluated=m_ident)

        self.assertAlmostEqual(tau_adjusted, tau_single, places=5)
        self.assertGreater(tau_inflated, tau_adjusted * 1.5)

    def test_dual_trick_equivalence(self):
        """Test 2: Dual Gram matrix (N x N) produces exact same sum_lambda_sq and M_eff when M > N."""
        rng = np.random.default_rng(123)
        n = 50
        m = 200

        x = rng.normal(size=(n, m))
        # Compute using library estimator (which internally selects dual trick since M > N)
        res_dual = estimate_effective_tests(x)

        # Compute primal manually
        xc = x - np.mean(x, axis=0)
        norms = np.linalg.norm(xc, axis=0)
        z = xc / norms
        r_primal = z.T @ z
        sum_sq_primal = float(np.sum(r_primal * r_primal))
        var_ratio_primal = (sum_sq_primal - m) / (m * (m - 1))
        m_eff_primal = 1.0 + (m - 1) * (1.0 - var_ratio_primal)

        self.assertAlmostEqual(res_dual["M_eff"], m_eff_primal, delta=1e-8)
        self.assertEqual(res_dual["M_raw"], m)

    def test_speed_benchmark(self):
        """Test 3: Computing estimate_effective_tests on N=3000, M=500 finishes promptly without O(M^3) blowup."""
        rng = np.random.default_rng(42)
        n = 3000
        m = 500
        x = rng.normal(size=(n, m))

        t0 = time.perf_counter()
        res = estimate_effective_tests(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(res["M_raw"], m)
        self.assertGreater(res["M_eff"], 1.0)
        # Generous sanity upper bound to prevent CI/thermal throttling flakiness while guarding against O(M^3) eigendecompositions
        self.assertLess(elapsed_ms, 2500.0, f"Computation took {elapsed_ms:.2f} ms, exceeding 2500 ms limit.")

    def test_sandbox_feature_logging_and_effective_telemetry(self):
        """Test 4: Sandbox logs all evaluated drivers and tracks cumulative_M_eff."""
        df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=500, seed=42)
        sb = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True, session_id="test_m_eff")

        # 1. Scan single feature
        res1 = sb.scan_single_feature("is_international_ip")
        self.assertEqual(res1["turn"], 1)
        self.assertEqual(res1["cumulative_M"], 1)
        self.assertAlmostEqual(res1["cumulative_M_eff"], 1.0, delta=1e-5)
        self.assertIn("bonferroni_gate_raw", res1)

        # 2. Scan batch of 5 features
        cols = ["weekend_tx_flag", "failed_login_burst", "device_os_ios", "new_beneficiary_added", "crypto_offramp_flag"]
        res2 = sb.scan_feature_batch(cols)
        self.assertEqual(res2["turn"], 2)
        self.assertEqual(res2["cumulative_M"], 6)
        self.assertGreater(res2["cumulative_M_eff"], 1.0)
        self.assertLessEqual(res2["cumulative_M_eff"], 6.0)

        # Verify evaluated DataFrame
        eval_df = sb.get_evaluated_features_df()
        self.assertEqual(len(eval_df), 500)
        self.assertEqual(len(eval_df.columns), 6)

        # Verify telemetry columns
        telem = sb.get_telemetry_df()
        self.assertIn("cumulative_M_eff", telem.columns)
        self.assertIn("redundancy_ratio", telem.columns)
        self.assertIn("bonferroni_gate_eff", telem.columns)
        self.assertIn("bonferroni_gate_raw", telem.columns)

    def test_simulate_agent_archetypes(self):
        """Test 5: Archetype simulation runs 3 distinct profiles evaluating M_raw = 200 features."""
        df, _ = generate_benchmark_datasets(mode="needle_in_haystack", nrows=1000, seed=42)
        res = simulate_agent_archetypes(df=df, target_col="is_fraud", seed=42)

        self.assertIn("broad_orthogonal", res)
        self.assertIn("combinatorial_grid", res)
        self.assertIn("threshold_tweaker", res)

        for arch_key, arch_dict in res.items():
            self.assertEqual(arch_dict["M_raw"], 200)
            self.assertGreater(arch_dict["M_eff"], 1.0)
            self.assertLessEqual(arch_dict["M_eff"], 200.0)
            self.assertGreater(arch_dict["redundancy_ratio"], 0.0)
            self.assertLessEqual(arch_dict["redundancy_ratio"], 1.0)
            self.assertEqual(len(arch_dict["evaluated_df"].columns), 200)
            self.assertEqual(len(arch_dict["evaluated_df"]), 1000)

        # Verify archetype ordering in redundancy: Orthogonal > Combinatorial > Tweaker
        self.assertGreater(res["broad_orthogonal"]["redundancy_ratio"], res["combinatorial_grid"]["redundancy_ratio"])
        self.assertGreater(res["combinatorial_grid"]["redundancy_ratio"], res["threshold_tweaker"]["redundancy_ratio"])


if __name__ == "__main__":
    unittest.main()
