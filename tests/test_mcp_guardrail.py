"""Unit tests for the stateful MCP statistical guardrail engine and agent tools."""

import unittest
import numpy as np
import pandas as pd

from signalguard.mcp_server import NoiseGateSessionLedger
from signalguard.agent_tools import (
    create_agent_tools,
    get_openai_tools,
    get_anthropic_tools,
)


class TestMCPGuardrail(unittest.TestCase):

    def setUp(self):
        self.ledger = NoiseGateSessionLedger()

    def test_monotonic_hypothesis_budget_and_gate_ratchet(self):
        """Tests that cumulative M increases and statistical gates ratchet upwards."""
        session_id = "test_ratchet"
        n = 5000
        p = 0.20

        # Step 1: Evaluate 1st hypothesis
        res1 = self.ledger.evaluate_feature_or_slice(
            observed_rig=0.010,
            nrows=n,
            minority_frac=p,
            hypotheses_in_batch=1,
            session_id=session_id
        )
        self.assertEqual(res1["cumulative_session_tests_M"], 1)
        tau_evt_1 = res1["evt_expected_noise_max"]
        tau_sig_1 = res1["bonferroni_significance_gate"]

        # Step 2: Evaluate 50 more hypotheses
        res2 = self.ledger.evaluate_feature_or_slice(
            observed_rig=0.010,
            nrows=n,
            minority_frac=p,
            hypotheses_in_batch=50,
            session_id=session_id
        )
        self.assertEqual(res2["cumulative_session_tests_M"], 51)
        tau_evt_51 = res2["evt_expected_noise_max"]
        tau_sig_51 = res2["bonferroni_significance_gate"]

        # Gates must strictly increase
        self.assertGreater(tau_evt_51, tau_evt_1)
        self.assertGreater(tau_sig_51, tau_sig_1)

        # Step 3: Check budget status
        status = self.ledger.get_search_budget_status(
            session_id=session_id,
            nrows=n,
            minority_frac=p
        )
        self.assertEqual(status["cumulative_hypotheses_M"], 51)
        self.assertGreater(status["gate_inflation_factor"], 1.0)

        # Step 4: Reset budget
        reset_res = self.ledger.reset_session_budget(session_id=session_id)
        self.assertEqual(reset_res["cumulative_hypotheses_M"], 0)
        status_after = self.ledger.get_search_budget_status(session_id=session_id)
        self.assertEqual(status_after["cumulative_hypotheses_M"], 0)

    def test_three_verdict_states(self):
        """Verifies VERIFIED_SIGNAL, BORDERLINE_WARNING, and REJECT_SPURIOUS_NOISE."""
        session_id = "test_verdicts"
        n = 5000
        p = 0.20

        # Evaluate at M=10
        status = self.ledger.get_search_budget_status(
            session_id=session_id, nrows=n, minority_frac=p
        )

        # Thresholds for n=5000, p=0.2, M=1: tau_sig ~ 0.00078, tau_evt ~ 0.00010
        # 1. High RIG -> VERIFIED_SIGNAL
        res_sig = self.ledger.evaluate_feature_or_slice(
            observed_rig=0.020,
            nrows=n,
            minority_frac=p,
            session_id=session_id
        )
        self.assertEqual(res_sig["verdict"], "VERIFIED_SIGNAL")
        self.assertIn("PROCEED", res_sig["agent_directive"])

        # 2. Intermediate RIG between tau_evt and tau_sig -> BORDERLINE_WARNING
        tau_evt = res_sig["evt_expected_noise_max"]
        tau_sig = res_sig["bonferroni_significance_gate"]
        mid_rig = (tau_evt + tau_sig) / 2.0

        res_border = self.ledger.evaluate_feature_or_slice(
            observed_rig=mid_rig,
            nrows=n,
            minority_frac=p,
            session_id=session_id
        )
        self.assertEqual(res_border["verdict"], "BORDERLINE_WARNING")
        self.assertIn("BORDERLINE", res_border["agent_directive"])

        # 3. Negligible RIG -> REJECT_SPURIOUS_NOISE
        res_reject = self.ledger.evaluate_feature_or_slice(
            observed_rig=0.00001,
            nrows=n,
            minority_frac=p,
            session_id=session_id
        )
        self.assertEqual(res_reject["verdict"], "REJECT_SPURIOUS_NOISE")
        self.assertIn("DISCARD", res_reject["agent_directive"])

    def test_contingency_counts_and_small_cell_warning(self):
        """Verifies 2x2 contingency count derivation and small-cell warning."""
        session_id = "test_contingency"

        # Case A: Adequate cell counts (n00=3000, n01=1000, n10=600, n11=400)
        # N = 5000, Y minority count = 1400 (28%), X support = 1000/5000 = 0.20
        # E11 = 5000 * 0.28 * 0.20 = 280 >> 5
        res_adequate = self.ledger.evaluate_feature_or_slice(
            contingency_counts=[3000, 1000, 600, 400],
            session_id=session_id
        )
        self.assertFalse(res_adequate["small_cell_warning"])
        self.assertGreater(res_adequate["observed_rig"], 0.0)

        # Case B: Tiny cell counts triggering Cochran's rule (E11 < 5)
        # N = 1000, Y minority count = 20 (p=0.02), feature support = 10/1000 = 0.01
        # E11 = 1000 * 0.02 * 0.01 = 0.2 < 5
        res_small = self.ledger.evaluate_feature_or_slice(
            contingency_counts=[972, 18, 8, 2],
            session_id=session_id
        )
        self.assertTrue(res_small["small_cell_warning"])
        self.assertIn("WARNING: Expected co-occurrence E11", res_small["agent_directive"])

    def test_audit_dataframe_workflow(self):
        """Tests the audit_dataframe method end-to-end with synthetic data."""
        session_id = "test_audit"
        rng = np.random.default_rng(42)
        n = 600

        # Target: binary 30% minority
        y = rng.choice([0, 1], size=n, p=[0.70, 0.30])

        # Feature 1: Genuine strong feature
        x_genuine = y.copy()
        flip = rng.random(n) < 0.15
        x_genuine[flip] = 1 - x_genuine[flip]

        # Feature 2: Weak/borderline feature
        x_weak = y.copy()
        flip2 = rng.random(n) < 0.40
        x_weak[flip2] = 1 - x_weak[flip2]

        # Feature 3: Pure random noise
        x_noise = rng.integers(0, 2, size=n)

        df_train = pd.DataFrame({
            "target": y,
            "feat_genuine": x_genuine,
            "feat_weak": x_weak,
            "feat_noise": x_noise
        })

        summary = self.ledger.audit_dataframe(
            df_train=df_train,
            target_col="target",
            candidate_cols=["feat_genuine", "feat_weak", "feat_noise"],
            n_bootstraps=50,
            session_id=session_id
        )

        self.assertEqual(summary["features_audited_in_batch"], 3)
        self.assertEqual(summary["cumulative_session_tests_M"], 3)
        self.assertIn("feat_genuine", summary["promoted_2_sigma_features"])
        self.assertIn("feat_noise", summary["rejected_noise_features"])
        self.assertIn("AUDIT COMPLETE", summary["agent_directive"])

    def test_agent_tools_wrapper_and_schemas(self):
        """Verifies OpenAI and Anthropic tool schema definitions and execution."""
        openai_tools = get_openai_tools()
        self.assertIsInstance(openai_tools, list)
        self.assertGreaterEqual(len(openai_tools), 3)

        anthropic_tools = get_anthropic_tools()
        self.assertIsInstance(anthropic_tools, list)
        self.assertGreaterEqual(len(anthropic_tools), 3)

        agent_tools = create_agent_tools(self.ledger)
        dispatch = agent_tools.to_callable_dict()
        self.assertIn("evaluate_feature_or_slice", dispatch)
        self.assertIn("audit_dataframe_candidates", dispatch)
        self.assertIn("get_search_budget_status", dispatch)
        self.assertIn("reset_session_budget", dispatch)

        # Execute through dispatch
        eval_res = dispatch["evaluate_feature_or_slice"](
            observed_rig=0.015,
            nrows=3000,
            minority_frac=0.25,
            session_id="agent_dispatch_test"
        )
        self.assertIn("verdict", eval_res)

    def test_frob_effective_m_in_ledger_and_audit(self):
        """Verifies Frobenius M_eff tracking inside NoiseGateSessionLedger."""
        rng = np.random.default_rng(101)
        n = 400
        base_vec = rng.integers(0, 2, size=n).astype(float)
        # 20 identical columns -> M_raw = 20, M_eff ~= 1.0
        collinear_mat = np.column_stack([base_vec for _ in range(20)])

        res = self.ledger.evaluate_feature_or_slice(
            observed_rig=0.008,
            nrows=n,
            minority_frac=0.20,
            hypotheses_in_batch=20,
            candidate_df=collinear_mat,
            session_id="test_ledger_meff"
        )
        self.assertEqual(res["cumulative_session_tests_M"], 20)
        self.assertAlmostEqual(res["cumulative_effective_tests_M_eff"], 1.0, places=4)
        self.assertLess(res["bonferroni_significance_gate"], res["bonferroni_significance_gate_raw"])

        status = self.ledger.get_search_budget_status(
            session_id="test_ledger_meff", nrows=n, minority_frac=0.20
        )
        self.assertAlmostEqual(status["cumulative_effective_M"], 1.0, places=4)
        self.assertAlmostEqual(status["redundancy_ratio"], 1.0 / 20.0, places=4)


if __name__ == "__main__":
    unittest.main()
