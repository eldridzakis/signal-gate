"""Unit Tests for Local SQLite + Governed Semantic Layer Benchmark (`sql_semantic_benchmark.py`)."""

import unittest
from signalguard.sql_semantic_benchmark import (
    RealWorldSQLSemanticWarehouse,
    populate_sql_semantic_trajectory,
    evaluate_sql_holdout_replication,
    run_sql_semantic_frontier_eval,
)


class TestSQLSemanticBenchmark(unittest.TestCase):
    """Verifies local SQLite warehouse creation, continuous/binary SQL GROUP BY gating, and holdout auditing."""

    def test_continuous_sql_warehouse_king_county(self):
        """Verifies continuous SQL GROUP BY AVG/STDDEV ANOVA F -> RIG gating on King County."""
        wh = RealWorldSQLSemanticWarehouse(
            mart_name="king_county_continuous",
            sample_train=1500,
            sample_test=1500,
            seed=42,
            guardrail_enabled=True,
        )
        res = wh.query_dimension("grade")
        self.assertGreater(res["observed_rig"], 0.15)
        self.assertEqual(res["verdict"], "VERIFIED_SIGNAL")

        transcript = populate_sql_semantic_trajectory(wh)
        self.assertIn("king_county_continuous", transcript)
        self.assertIn("VERIFIED_SIGNAL", transcript)

        audit = evaluate_sql_holdout_replication(
            reported_discoveries=["grade", "sqft_living_bin", "nonexistent_sql_dim"],
            warehouse=wh,
            self_reported_M=3,
            true_M_raw=38,
        )
        self.assertEqual(audit["n_reported"], 3)
        self.assertEqual(audit["replicated_count"], 2)
        self.assertEqual(audit["failed_count"], 1)

    def test_binary_sql_warehouse_dry_run(self):
        """Verifies dry-run SQL semantic evaluation on Lending Club."""
        for p in ["business_analyst", "statistician_unassisted", "statistician_signalguard"]:
            res = run_sql_semantic_frontier_eval(
                mart_name="lending_club",
                persona=p,
                dry_run=True,
                sample_train=500,
                sample_test=250,
                seed=42,
            )
            self.assertIn("holdout_audit", res)
            self.assertGreaterEqual(res["holdout_audit"]["replication_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
