"""Unit tests for InstrumentedAgentSandbox and P-Hacking Benchmark Simulator."""

import unittest
import numpy as np
import pandas as pd

from signalgate.agent_sandbox import (
    BASE_DOMAIN_COLUMNS,
    generate_benchmark_datasets,
    InstrumentedAgentSandbox,
    run_exploratory_agent_simulation,
)
import feature_noise_gate.agent_sandbox as fng_sandbox
import sampling_toolbox.agent_sandbox as stb_sandbox


class TestAgentSandbox(unittest.TestCase):
    """Test suite for agent sandbox telemetry, simulation, and benchmark generation."""

    def test_generate_benchmark_datasets(self):
        """Verifies synthetic dataset shapes and ground truth metadata."""
        # 1. Null Honeypot
        df_null, gt_null = generate_benchmark_datasets(mode="null_honeypot", nrows=500, minority_frac=0.10, seed=42)
        self.assertEqual(len(df_null), 500)
        self.assertIn("is_fraud", df_null.columns)
        self.assertEqual(len(df_null.columns), 31)  # 30 domain + 1 target
        self.assertEqual(len(gt_null), 30)
        for col, meta in gt_null.items():
            self.assertFalse(meta["is_genuine"])
            self.assertEqual(meta["population_rig"], 0.0)

        # 2. Needle in Haystack
        df_needle, gt_needle = generate_benchmark_datasets(mode="needle_in_haystack", nrows=500, minority_frac=0.10, seed=42)
        self.assertEqual(len(df_needle), 500)
        self.assertIn("is_fraud", df_needle.columns)
        self.assertEqual(len(df_needle.columns), 51)  # 47 noise + 3 needles + 1 target
        self.assertEqual(len(gt_needle), 50)

        needles = ["verified_velocity_spike", "compromised_asn_match", "high_risk_mcc_group"]
        for needle in needles:
            self.assertIn(needle, gt_needle)
            self.assertTrue(gt_needle[needle]["is_genuine"])
            self.assertGreater(gt_needle[needle]["target_rig"], 0.0)

    def test_sandbox_single_and_batch_scan(self):
        """Verifies hypothesis tracking and telemetry logging for single and batch scans."""
        df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=500, seed=42)
        sandbox = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True, session_id="test_scan")

        # Single scan (M += 1)
        res1 = sandbox.scan_single_feature("is_international_ip")
        self.assertEqual(res1["turn"], 1)
        self.assertEqual(res1["cumulative_M"], 1)
        self.assertIn("verdict", res1)
        self.assertIn(res1["verdict"], ["VERIFIED_SIGNAL", "BORDERLINE_WARNING", "REJECT_SPURIOUS_NOISE"])

        # Batch scan of 5 features (M += 5 -> cumulative M = 6)
        cols_batch = ["weekend_tx_flag", "failed_login_burst", "device_os_ios", "new_beneficiary_added", "crypto_offramp_flag"]
        res2 = sandbox.scan_feature_batch(cols_batch)
        self.assertEqual(res2["turn"], 2)
        self.assertEqual(res2["cumulative_M"], 6)

        telemetry = sandbox.get_telemetry_df()
        self.assertEqual(len(telemetry), 2)
        self.assertEqual(telemetry.iloc[1]["cumulative_M"], 6)
        self.assertEqual(telemetry.iloc[1]["batch_hypotheses_k"], 5)

    def test_sandbox_interaction_grid(self):
        """Verifies interaction grid evaluation and hypothesis inflation accounting."""
        df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=500, seed=42)
        sandbox = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True, session_id="test_grid")

        cols_a = ["is_international_ip", "weekend_tx_flag"]
        cols_b = ["failed_login_burst", "device_os_ios", "new_beneficiary_added"]
        # 2 x 3 = 6 pairs
        res = sandbox.test_interaction_grid(cols_a, cols_b, op="and")
        self.assertEqual(res["turn"], 1)
        self.assertEqual(res["cumulative_M"], 6)

        telemetry = sandbox.get_telemetry_df()
        self.assertEqual(telemetry.iloc[0]["batch_hypotheses_k"], 6)
        self.assertEqual(telemetry.iloc[0]["action_type"], "interaction_grid")

    def test_sandbox_drill_down_slice(self):
        """Verifies subgroup slicing, sample size shrinkage, and small-cell warning."""
        df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=1000, minority_frac=0.10, seed=42)
        sandbox = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True, session_id="test_slice")

        # Slice on rare feature (support ~ 0.05 -> ~50 rows)
        candidate_cols = ["weekend_tx_flag", "device_os_ios", "crypto_offramp_flag"]
        res = sandbox.drill_down_slice(filter_col="failed_login_burst", filter_val=1, candidate_cols=candidate_cols)

        self.assertEqual(res["turn"], 1)
        self.assertEqual(res["cumulative_M"], 3)

        telemetry = sandbox.get_telemetry_df()
        t_row = telemetry.iloc[0]
        self.assertLess(t_row["n_effective"], 1000)
        self.assertGreater(t_row["n_effective"], 10)
        # Check small cell condition calculation
        self.assertIn("small_cell_warning", res)

    def test_guardrail_modes(self):
        """Verifies baseline un-guardrailed mode vs SignalGate mode."""
        df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=500, seed=42)

        # Baseline mode
        sb_base = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=False, session_id="base")
        res_base = sb_base.scan_single_feature("is_international_ip")
        self.assertIn("naive_p_value", res_base)
        self.assertIn("naive_verdict", res_base)
        self.assertNotIn("bonferroni_gate", res_base)

        # Guardrailed mode
        sb_gate = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True, session_id="gate")
        res_gate = sb_gate.scan_single_feature("is_international_ip")
        self.assertIn("bonferroni_gate", res_gate)
        self.assertIn("evt_expected_max", res_gate)
        self.assertIn("verdict", res_gate)

    def test_exploratory_simulation_hypothesis_inflation(self):
        """Tests that a 15-turn exploratory agent evaluates > 500 hypotheses and SignalGate protects against noise."""
        df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=1000, minority_frac=0.12, seed=42)
        sandbox = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True, session_id="sim_test")

        telemetry = run_exploratory_agent_simulation(sandbox, seed=42)
        self.assertEqual(len(telemetry), 15)

        # 15 turns must evaluate > 500 hypotheses due to interaction grids and slices
        final_m = telemetry.iloc[-1]["cumulative_M"]
        self.assertGreater(final_m, 500)

        # On pure noise, SignalGate must not verify any feature as VERIFIED_SIGNAL
        verified_count = (telemetry["verdict"] == "VERIFIED_SIGNAL").sum()
        self.assertEqual(verified_count, 0)

    def test_backward_compatibility_imports(self):
        """Verifies that the sandbox classes can be imported from feature_noise_gate and sampling_toolbox."""
        self.assertTrue(hasattr(fng_sandbox, "InstrumentedAgentSandbox"))
        self.assertTrue(hasattr(fng_sandbox, "generate_benchmark_datasets"))
        self.assertTrue(hasattr(stb_sandbox, "InstrumentedAgentSandbox"))
        self.assertTrue(hasattr(stb_sandbox, "generate_benchmark_datasets"))


if __name__ == "__main__":
    unittest.main()
