"""Unit tests for Phase 2 Fair Agentic Benchmark Engine (signalguard/phase2_benchmark.py)."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from signalguard.phase2_benchmark import (
    PHASE2_TASKS,
    Phase2SQLWarehouse,
    StatisticalArmAuditor,
    Phase2InteractiveAgentRunner,
    DecoupledHoldoutAuditor,
    Phase2TraceCache,
)


class TestPhase2BenchmarkEngine(unittest.TestCase):
    """Verifies all Phase 2 components: SQL warehouse, arms, interactive loop, and pure holdout audit."""

    def test_tasks_catalog_integrity(self):
        """All 4 tasks must be registered with required metadata."""
        expected = ["king_county_subgroup", "lending_club_subgroup", "bank_marketing_null_ab", "bank_marketing_seeded_hte"]
        for t in expected:
            self.assertIn(t, PHASE2_TASKS)
            cfg = PHASE2_TASKS[t]
            self.assertIn("target_col", cfg)
            self.assertIn("dimensions", cfg)

    def test_warehouse_creation_and_isolation(self):
        """Phase2SQLWarehouse must cleanly separate train_mart from holdout_vault."""
        wh = Phase2SQLWarehouse(task_name="bank_marketing_null_ab", seed=42)
        self.assertIsNotNone(wh.conn)
        self.assertGreater(len(wh.train_df), 1000)
        self.assertGreater(len(wh.holdout_df), 1000)

        # Query train_mart
        df_tr = wh.run_query("SELECT variant, COUNT(*) AS n_k, SUM(subscribed) AS sum_y FROM train_mart GROUP BY variant", on_holdout=False)
        self.assertEqual(len(df_tr), 2)
        self.assertIn("n_k", df_tr.columns)

        # Query holdout_vault
        df_te = wh.run_query("SELECT variant, COUNT(*) AS n_k, SUM(subscribed) AS sum_y FROM holdout_vault GROUP BY variant", on_holdout=True)
        self.assertEqual(len(df_te), 2)

    def test_seeded_hte_ground_truth(self):
        """Seeded HTE task must exhibit genuine positive treatment lift in seeded segments."""
        wh = Phase2SQLWarehouse(task_name="bank_marketing_seeded_hte", seed=42)
        # Check retired & housing=no treatment vs control
        sql = """
            SELECT variant, COUNT(*) AS n_k, AVG(subscribed) AS mean_y
            FROM holdout_vault
            WHERE job = 'retired' AND housing = 'no'
            GROUP BY variant
        """
        df = wh.run_query(sql, on_holdout=True)
        self.assertEqual(len(df), 2)
        t_rate = df[df["variant"] == "treatment"]["mean_y"].values[0]
        c_rate = df[df["variant"] == "control"]["mean_y"].values[0]
        self.assertGreater(t_rate, c_rate, "Seeded responder segment should exhibit positive treatment lift")

    def test_statistical_arms_audits(self):
        """Verifies unassisted, bonferroni, benjamini_hochberg, and signalguard arm diagnostics."""
        wh = Phase2SQLWarehouse(task_name="lending_club_subgroup", seed=42)
        df_res = wh.run_query("SELECT term, COUNT(*) AS n_k, SUM(bad_loans) AS sum_y FROM train_mart GROUP BY term", on_holdout=False)

        # 1. Unassisted
        aud_un = StatisticalArmAuditor(arm="unassisted", target_type="binary")
        r_un = aud_un.audit_query_result("SELECT term...", df_res, "bad_loans", len(wh.train_df))
        self.assertIn("verdict", r_un)

        # 2. Bonferroni
        aud_bonf = StatisticalArmAuditor(arm="bonferroni", target_type="binary")
        r_bonf = aud_bonf.audit_query_result("SELECT term...", df_res, "bad_loans", len(wh.train_df))
        self.assertIn("bonferroni_threshold", r_bonf)

        # 3. Benjamini-Hochberg
        aud_bh = StatisticalArmAuditor(arm="benjamini_hochberg", target_type="binary")
        r_bh = aud_bh.audit_query_result("SELECT term...", df_res, "bad_loans", len(wh.train_df))
        self.assertIn("bh_q_value", r_bh)

        # 4. SignalGuard
        aud_sg = StatisticalArmAuditor(arm="signalguard", target_type="binary")
        r_sg = aud_sg.audit_query_result("SELECT term...", df_res, "bad_loans", len(wh.train_df))
        self.assertIn("tau_sig", r_sg)
        self.assertIn("M_eff", r_sg)

    def test_dry_run_agent_session_and_holdout_audit(self):
        """Verifies full end-to-end multi-turn session in dry-run mode and pure holdout evaluation."""
        wh = Phase2SQLWarehouse(task_name="king_county_subgroup", seed=42)
        runner = Phase2InteractiveAgentRunner(warehouse=wh, arm="signalguard", model_name="gemini-3.5-flash-lite", max_turns=4)

        session_res = runner.run_session(dry_run=True)
        self.assertEqual(session_res["task"], "king_county_subgroup")
        self.assertGreater(session_res["turns_executed"], 1)
        self.assertGreater(len(session_res["reported_findings"]), 0)

        # Evaluate against holdout_vault
        evaluator = DecoupledHoldoutAuditor(warehouse=wh, alpha=0.05)
        audit_res = evaluator.evaluate_reported_findings(session_res["reported_findings"])

        self.assertIn("empirical_fdr", audit_res)
        self.assertIn("signal_recall", audit_res)
        self.assertIn("discovery_f1", audit_res)
        self.assertIn("audit_details", audit_res)
        self.assertGreaterEqual(audit_res["empirical_fdr"], 0.0)
        self.assertLessEqual(audit_res["empirical_fdr"], 100.0)


if __name__ == "__main__":
    unittest.main()
