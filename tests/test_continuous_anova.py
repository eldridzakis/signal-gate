"""Unit tests for closed-form continuous ANOVA F -> RIG and SQL group aggregation gating."""

import unittest
import numpy as np
import pandas as pd
from scipy import stats

from signalguard.metrics import anova_rig_from_group_stats, fast_continuous_rig
from signalguard.mcp_server import NoiseGateSessionLedger


class TestContinuousANOVA(unittest.TestCase):
    """Test suite for continuous ANOVA F -> RIG identity and SQL group aggregation gates."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_anova_rig_matches_scipy_f_oneway(self):
        # Generate 3 groups with different means
        g1 = self.rng.normal(loc=10.0, scale=2.0, size=150)
        g2 = self.rng.normal(loc=12.5, scale=2.0, size=200)
        g3 = self.rng.normal(loc=15.0, scale=2.0, size=120)

        # Scipy baseline
        f_expected, p_expected = stats.f_oneway(g1, g2, g3)

        # SignalGuard summary stats
        counts = [len(g1), len(g2), len(g3)]
        means = [float(np.mean(g1)), float(np.mean(g2)), float(np.mean(g3))]
        stds = [float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1)), float(np.std(g3, ddof=1))]

        res = anova_rig_from_group_stats(counts, means, stds, ddof=1)

        # F-statistic and p-value must match Scipy exactly
        self.assertAlmostEqual(res["f_statistic"], f_expected, places=4)
        self.assertAlmostEqual(res["p_value"], p_expected, places=6)
        self.assertGreater(res["eta_squared"], 0.0)
        self.assertGreater(res["observed_rig"], 0.0)
        self.assertEqual(res["total_nrows"], sum(counts))
        self.assertEqual(res["df_between"], 2)

    def test_fast_continuous_rig_on_raw_series(self):
        df = pd.DataFrame({
            "category": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
            "target_y": np.concatenate([
                self.rng.normal(50.0, 5.0, 100),
                self.rng.normal(55.0, 5.0, 100),
                self.rng.normal(60.0, 5.0, 100)
            ])
        })
        res = fast_continuous_rig(df["category"], df["target_y"])
        self.assertGreater(res["observed_rig"], 0.1)
        self.assertLess(res["p_value"], 1e-10)

    def test_session_ledger_evaluates_sql_group_aggregation(self):
        ledger = NoiseGateSessionLedger()
        session_id = "test_sql_continuous_session"
        ledger.reset_session_budget(session_id)

        # 1. Genuine strong signal (e.g. house price by grade)
        res_sig = ledger.evaluate_continuous_group_aggregation(
            group_counts=[500, 1200, 800],
            group_means=[250000.0, 480000.0, 850000.0],
            group_stds=[40000.0, 65000.0, 120000.0],
            hypotheses_in_batch=1,
            session_id=session_id,
            feature_name="house_price_by_grade"
        )
        self.assertEqual(res_sig["verdict"], "VERIFIED_SIGNAL")
        self.assertIn("anova_details", res_sig)
        self.assertGreater(res_sig["observed_rig"], 0.05)

        # 2. Pure noise grouping (identical means)
        res_noise = ledger.evaluate_continuous_group_aggregation(
            group_counts=[500, 500],
            group_means=[500000.0, 500050.0],
            group_stds=[50000.0, 50000.0],
            hypotheses_in_batch=1,
            session_id=session_id,
            feature_name="pure_noise_group"
        )
        self.assertEqual(res_noise["verdict"], "REJECT_SPURIOUS_NOISE")

        # 3. Small-cell warning test (tiny group sample size)
        res_small = ledger.evaluate_continuous_group_aggregation(
            group_counts=[12, 500],
            group_means=[950000.0, 500000.0],
            group_stds=[80000.0, 50000.0],
            hypotheses_in_batch=1,
            session_id=session_id,
            feature_name="tiny_vip_slice"
        )
        self.assertTrue(res_small["small_cell_warning"])


if __name__ == "__main__":
    unittest.main()
