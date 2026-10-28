"""Automated Unit Tests for SignalGate Real-World Empirical Benchmark Engine."""

import unittest
import numpy as np
import pandas as pd

from signalgate.real_world_benchmark import (
    load_real_world_benchmark_dataset,
    RealWorldAgentSandbox,
    populate_real_world_agent_trajectory,
    evaluate_real_world_holdout_replication,
    run_real_world_frontier_eval,
)


class TestRealWorldBenchmark(unittest.TestCase):
    """Test suite verifying real-world dataset loading, sandbox telemetry, and holdout auditing."""

    def test_load_real_world_datasets(self):
        """Verifies loading and preprocessing for Lending Club and King County."""
        # Lending Club
        ds_lc = load_real_world_benchmark_dataset(
            dataset_name="lending_club",
            sample_train=500,
            sample_test=250,
            seed=42
        )
        self.assertEqual(ds_lc["dataset_name"], "lending_club")
        self.assertEqual(ds_lc["target_col"], "bad_loans")
        self.assertEqual(len(ds_lc["train_df"]), 500)
        self.assertEqual(len(ds_lc["test_df"]), 250)
        self.assertTrue(len(ds_lc["candidate_features"]) >= 10)

        # King County
        ds_kc = load_real_world_benchmark_dataset(
            dataset_name="king_county",
            sample_train=500,
            sample_test=250,
            seed=42
        )
        self.assertEqual(ds_kc["dataset_name"], "king_county")
        self.assertEqual(ds_kc["target_col"], "high_value")
        self.assertEqual(len(ds_kc["train_df"]), 500)
        self.assertEqual(len(ds_kc["test_df"]), 250)

    def test_real_world_sandbox_and_trajectory(self):
        """Verifies agent sandbox tool telemetry on real-world training data."""
        ds = load_real_world_benchmark_dataset(
            dataset_name="lending_club",
            sample_train=300,
            sample_test=150,
            seed=42
        )
        sandbox = RealWorldAgentSandbox(
            train_df=ds["train_df"],
            target_col=ds["target_col"],
            candidate_features=ds["candidate_features"],
            guardrail_enabled=True,
            session_id="test_rw_sb"
        )
        transcript = populate_real_world_agent_trajectory(sandbox, blind_unassisted_counts=True)

        self.assertIn("Real-World Dataset Summary", transcript)
        self.assertTrue(len(sandbox.telemetry_history) >= 8)
        self.assertTrue(sandbox.telemetry_history[-1]["cumulative_M"] > 20)

    def test_holdout_replication_audit(self):
        """Verifies true out-of-sample holdout replication calculations."""
        ds = load_real_world_benchmark_dataset(
            dataset_name="lending_club",
            sample_train=1000,
            sample_test=500,
            seed=42
        )
        train_df = ds["train_df"]
        test_df = ds["test_df"]
        target = ds["target_col"]

        # Known strong predictors in Lending Club
        discoveries = ["grade", "sub_grade", "term", "nonexistent_hallucinated_column"]
        audit = evaluate_real_world_holdout_replication(
            reported_discoveries=discoveries,
            train_df=train_df,
            test_df=test_df,
            target_col=target,
            self_reported_M=4,
            true_M_raw=40
        )

        self.assertEqual(audit["n_reported"], 4)
        self.assertEqual(audit["replicated_count"], 3)  # grade, sub_grade, term replicate
        self.assertEqual(audit["failed_count"], 1)      # nonexistent fails
        self.assertAlmostEqual(audit["replication_rate"], 0.75, places=2)
        self.assertAlmostEqual(audit["real_fdr"], 0.25, places=2)
        self.assertEqual(audit["under_reporting_ratio"], 10.0)

    def test_frontier_eval_dry_run(self):
        """Verifies dry-run frontier evaluation across all personas."""
        for p in ["business_analyst", "statistician_unassisted", "statistician_signalgate"]:
            res = run_real_world_frontier_eval(
                dataset_name="lending_club",
                persona=p,
                dry_run=True,
                sample_train=500,
                sample_test=250,
                seed=42
            )
            self.assertIn("holdout_audit", res)
            self.assertIn("replication_rate", res["holdout_audit"])
            self.assertTrue(res["true_M_raw"] > 10)


if __name__ == "__main__":
    unittest.main()
