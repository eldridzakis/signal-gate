"""Unit tests for the plain-holdout criterion, signal-recall metric, and cached-output re-scoring."""

import unittest

from signalguard.real_world_benchmark import plain_holdout_signal_recall, plain_holdout_summary
from signalguard.sql_semantic_benchmark import (
    RealWorldSQLSemanticWarehouse,
    evaluate_sql_holdout_replication,
    run_sql_semantic_frontier_eval,
)


class TestPlainHoldoutMetrics(unittest.TestCase):
    def test_plain_summary_ignores_small_cell_rule(self):
        rows = [
            {"feature": "a", "holdout_p_value": 0.001, "replicated_on_holdout": False},  # small-cell rejected
            {"feature": "b", "holdout_p_value": 0.2, "replicated_on_holdout": False},
            {"feature": "c", "holdout_p_value": 1e-9, "replicated_on_holdout": True},
        ]
        s = plain_holdout_summary(rows)
        self.assertEqual(s["plain_replicated_count"], 2)
        self.assertAlmostEqual(s["plain_fdr"], 1 / 3)
        self.assertEqual(plain_holdout_summary([])["plain_fdr"], 0.0)

    def test_signal_recall_uses_bonferroni_pool(self):
        pool = [
            {"feature": "x", "holdout_p_value": 1e-12},
            {"feature": "y", "holdout_p_value": 1e-8},
            {"feature": "z", "holdout_p_value": 0.03},  # nominal only; fails alpha/3
        ]
        out = plain_holdout_signal_recall([{"feature": "x"}], pool)
        self.assertEqual(out["pool_size"], 3)
        self.assertEqual(out["real_pool_size"], 2)
        self.assertEqual(out["recovered"], 1)
        self.assertAlmostEqual(out["signal_recall"], 0.5)
        self.assertIsNone(plain_holdout_signal_recall([], [])["signal_recall"])

    def test_sql_audit_exposes_plain_fields(self):
        wh = RealWorldSQLSemanticWarehouse(mart_name="king_county_continuous", sample_train=1500, sample_test=1500, seed=42)
        audit = evaluate_sql_holdout_replication(["grade", "nonexistent_sql_dim"], wh)
        self.assertEqual(audit["plain_replicated_count"], 1)
        self.assertAlmostEqual(audit["plain_fdr"], 0.5)

    def test_cached_output_rescoring_makes_no_api_call(self):
        report = 'Findings.\n```json\n{"reported_discoveries": ["grade"], "self_reported_M": 5, "no_genuine_signal": false}\n```'
        res = run_sql_semantic_frontier_eval(
            mart_name="king_county_continuous", persona="statistician_unassisted", model_name="claude-opus-5-5",
            seed=42, sample_train=1500, cached_output=report,
        )
        self.assertTrue(res["cache_hit"])
        self.assertEqual(res["reported_discoveries"], ["grade"])
        self.assertIn("plain_fdr", res["holdout_audit"])
        self.assertIn("signal_recall", res["holdout_audit"])


if __name__ == "__main__":
    unittest.main()
