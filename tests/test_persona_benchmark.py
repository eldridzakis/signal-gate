"""Unit tests for Persona Benchmark, The Hawthorne Effect, and Data Modality Engine."""

import tempfile
import os
import unittest
import numpy as np
import pandas as pd

from signalguard.persona_benchmark import (
    PERSONA_PROMPTS,
    TRACKING_INSTRUCTIONS,
    RawCSVSandbox,
    SemanticLayerSandbox,
    build_governed_semantic_dataset,
    simulate_persona_run,
    run_persona_modality_benchmark,
    build_langchain_agent_runner,
    parse_agent_discovery_report,
    evaluate_agent_run_metrics,
    LLMEvalTraceCache,
    format_prewarmed_trajectory_prompt,
    run_prewarmed_frontier_eval,
    run_prewarmed_frontier_benchmark,
    DailyBudgetExhausted,
)
from signalguard.agent_sandbox import generate_benchmark_datasets


class TestPersonaBenchmark(unittest.TestCase):
    """Test suite for agent personas, self-tracking ledgers, and semantic layer modality."""

    def setUp(self):
        self.df_null, self.gt_null = generate_benchmark_datasets(mode="null_honeypot", nrows=500, minority_frac=0.12, seed=42)
        self.df_needle, self.gt_needle = generate_benchmark_datasets(mode="needle_in_haystack", nrows=500, minority_frac=0.12, seed=42)

    def test_persona_and_tracking_prompts(self):
        """Test 1: Verify persona registry and tracking prompt structures."""
        self.assertIn("business_analyst", PERSONA_PROMPTS)
        self.assertIn("statistician_unassisted", PERSONA_PROMPTS)
        self.assertIn("statistician_signalguard", PERSONA_PROMPTS)

        self.assertIn("untracked", TRACKING_INSTRUCTIONS)
        self.assertIn("self_tracked_ledger", TRACKING_INSTRUCTIONS)

        self.assertIn("3–5 concrete drivers", PERSONA_PROMPTS["business_analyst"])
        self.assertIn("multiple hypothesis testing", PERSONA_PROMPTS["statistician_unassisted"])
        self.assertIn("SignalGuard Statistical Guardrail", PERSONA_PROMPTS["statistician_signalguard"])
        self.assertIn("CRITICAL: You must log every single feature", TRACKING_INSTRUCTIONS["self_tracked_ledger"])

    def test_raw_csv_sandbox_and_batch_blindness(self):
        """Test 2: Verify RawCSVSandbox tracking and batch blindness under-reporting ratio."""
        sb = RawCSVSandbox(self.df_null, target_col="is_fraud", session_id="test_raw")
        cand_cols = [c for c in self.df_null.columns if c != "is_fraud"]

        # Turn 1: 1 batch scan of 8 features
        sb.scan_feature_batch(cand_cols[:8])
        self.assertEqual(sb.agent_self_reported_M, 1)
        self.assertEqual(sb.telemetry_history[-1]["cumulative_M"], 8)

        # Turn 2: 1 grid test of 3x3 = 9 combinations
        sb.test_interaction_grid(cand_cols[:3], cand_cols[3:6], op="and")
        self.assertEqual(sb.agent_self_reported_M, 2)
        self.assertEqual(sb.telemetry_history[-1]["cumulative_M"], 17)

        # Verify under-reporting ratio
        ratio = sb.get_under_reporting_ratio()
        self.assertAlmostEqual(ratio, 17.0 / 2.0, delta=1e-5)
        self.assertGreater(ratio, 5.0)

    def test_semantic_layer_sandbox_and_slice_variance(self):
        """Test 3: Verify SemanticLayerSandbox governed dimensions, catalog, and query execution."""
        sem_df, catalog = build_governed_semantic_dataset(self.df_null, target_col="is_fraud")
        self.assertIn("geo_tier", catalog)
        self.assertIn("device_platform", catalog)
        self.assertIn("payment_channel", catalog)
        self.assertIn("auth_security", catalog)

        sb = SemanticLayerSandbox(self.df_null, target_col="is_fraud", session_id="test_sem")
        cat = sb.get_catalog()
        self.assertIn("dimensions", cat)
        self.assertIn("metrics", cat)
        self.assertEqual(cat["base_sample_size"], 500)

        # Execute single dimensional query
        res = sb.query_metric(metric="fraud_rate", dimensions=["geo_tier"])
        self.assertIn("results_table", res)
        self.assertGreater(len(res["results_table"]), 0)

        # Execute multi-filter deep slice with small sample size
        res_deep = sb.query_metric(
            metric="fraud_rate",
            dimensions=["payment_channel"],
            filters={"auth_security": "Failed_Burst"}
        )
        self.assertIn("results_table", res_deep)
        slice_telemetry = sb.get_slice_variance_telemetry()
        self.assertIsInstance(slice_telemetry, pd.DataFrame)
        self.assertGreater(len(slice_telemetry), 0)

    def test_simulate_persona_run(self):
        """Test 4: Verify simulate_persona_run on null honeypot and needle in haystack."""
        # On Null Honeypot (N=3000, p=0.12):
        res_ba = simulate_persona_run(persona="business_analyst", dataset_mode="null_honeypot", nrows=3000, seed=42)
        self.assertEqual(res_ba["false_discovery_rate"], 1.0)
        self.assertGreater(res_ba["false_discoveries"], 0)

        res_sg = simulate_persona_run(persona="statistician_signalguard", dataset_mode="null_honeypot", nrows=3000, seed=42)
        self.assertEqual(res_sg["false_discovery_rate"], 0.0)
        self.assertEqual(res_sg["false_discoveries"], 0)

        # On Needle in Haystack (N=3000, p=0.12):
        res_needle_sg = simulate_persona_run(persona="statistician_signalguard", dataset_mode="needle_in_haystack", nrows=3000, seed=42)
        self.assertEqual(res_needle_sg["false_discovery_rate"], 0.0)
        self.assertGreaterEqual(res_needle_sg["recall"], 0.66)

    def test_run_persona_modality_benchmark(self):
        """Test 5: Verify multi-seed benchmark runner returns all required summary DataFrames."""
        summary = run_persona_modality_benchmark(n_seeds=2, nrows=500, minority_frac=0.12, base_seed=42)
        self.assertIn("persona_df", summary)
        self.assertIn("hawthorne_df", summary)
        self.assertIn("modality_df", summary)
        self.assertIn("all_runs", summary)

        self.assertEqual(len(summary["persona_df"]), 6)  # 3 personas x 2 dataset modes
        self.assertEqual(len(summary["hawthorne_df"]), 4)  # 2 tracking modes x 2 personas
        self.assertEqual(len(summary["modality_df"]), 2)   # 2 modalities

    def test_langchain_agent_runner_fallback(self):
        """Test 6: Verify build_langchain_agent_runner dry-run fallback and structured evaluation."""
        sb = RawCSVSandbox(self.df_null, session_id="test_runner")
        runner = build_langchain_agent_runner(sb, persona="business_analyst", dry_run=True)
        res = runner.invoke({"input": "Find top fraud drivers"})
        self.assertIn("output", res)
        self.assertEqual(res["persona"], "business_analyst")
        self.assertGreater(res["sandbox_true_M_raw"], 0)
        self.assertIn("evaluation", res)
        self.assertIn("directive_compliance_rate", res)
        self.assertEqual(res["false_discovery_rate"], 1.0)
        self.assertEqual(res["directive_compliance_rate"], 0.0)

        # SignalGuard persona must achieve 100% directive compliance and 0% FDR on null honeypot
        sb_sg = RawCSVSandbox(self.df_null, guardrail_enabled=True, session_id="test_runner_sg")
        runner_sg = build_langchain_agent_runner(sb_sg, persona="statistician_signalguard", dry_run=True)
        res_sg = runner_sg.invoke({"input": "Find top fraud drivers"})
        self.assertEqual(res_sg["false_discovery_rate"], 0.0)
        self.assertEqual(res_sg["directive_compliance_rate"], 1.0)
        self.assertEqual(len(res_sg["reported_discoveries"]), 0)

    def test_parse_and_evaluate_agent_run_metrics(self):
        """Test 7: Verify structured JSON and natural language parsing of LLM discovery reports."""
        sb = RawCSVSandbox(self.df_needle, guardrail_enabled=True, session_id="test_parse")
        sb.scan_feature_batch(["verified_velocity_spike", "is_international_ip", "weekend_tx_flag"])

        llm_json_reply = (
            "After testing the features, only verified_velocity_spike cleared the gate.\n"
            '```json\n{"reported_discoveries": ["verified_velocity_spike"], "self_reported_M": 1, "no_genuine_signal": false}\n```'
        )
        metrics = evaluate_agent_run_metrics(
            output_text=llm_json_reply,
            sandbox=sb,
            persona="statistician_signalguard",
            dataset_mode="needle_in_haystack"
        )
        self.assertEqual(metrics["reported_discoveries"], ["verified_velocity_spike"])
        self.assertEqual(metrics["true_discoveries"], 1)
        self.assertEqual(metrics["false_discoveries"], 0)
        self.assertEqual(metrics["false_discovery_rate"], 0.0)
        self.assertEqual(metrics["under_reporting_ratio"], 3.0)

    def test_prewarmed_frontier_eval_and_jsonl_cache(self):
        """Test 8: Verify pre-warmed frontier evaluation and persistent JSONL trace caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_llm_cache.jsonl")

            # 1st call: populates cache on miss
            r1 = run_prewarmed_frontier_eval(
                persona="statistician_signalguard",
                dataset_mode="needle_in_haystack",
                nrows=3000,
                seed=42,
                dry_run=True,
                use_cache=True,
                cache_path=cache_file,
            )
            self.assertFalse(r1["cache_hit"])
            self.assertTrue(os.path.exists(cache_file))
            self.assertEqual(r1["false_discovery_rate"], 0.0)
            self.assertEqual(r1["directive_compliance_rate"], 1.0)
            self.assertGreaterEqual(r1["recall"], 0.66)
            self.assertIn("SignalGuard(M_raw=", r1["prompt_transcript"])

            # 2nd call with identical parameters: must hit JSONL disk cache
            r2 = run_prewarmed_frontier_eval(
                persona="statistician_signalguard",
                dataset_mode="needle_in_haystack",
                nrows=3000,
                seed=42,
                dry_run=True,
                use_cache=True,
                cache_path=cache_file,
            )
            self.assertTrue(r2["cache_hit"])
            self.assertEqual(r1["cache_key"], r2["cache_key"])
            self.assertEqual(r1["reported_discoveries"], r2["reported_discoveries"])

            # Multi-seed prewarmed benchmark
            bench = run_prewarmed_frontier_benchmark(
                personas=("business_analyst", "statistician_signalguard"),
                dataset_modes=("null_honeypot",),
                n_seeds=2,
                nrows=500,
                dry_run=True,
                use_cache=True,
                cache_path=cache_file,
            )
            self.assertIn("benchmark_df", bench)
            self.assertEqual(len(bench["benchmark_df"]), 2)

            # Test daily budget enforcement: when daily limit is 0 on un-cached run
            fresh_cache_file = os.path.join(tmpdir, "empty_cache.jsonl")
            with self.assertRaises(DailyBudgetExhausted):
                run_prewarmed_frontier_eval(
                    persona="business_analyst",
                    dataset_mode="null_honeypot",
                    nrows=200,
                    seed=999,
                    dry_run=False,
                    use_cache=True,
                    cache_path=fresh_cache_file,
                    max_live_calls_remaining=0,
                )

            # Test multi-day benchmark pausing on budget limit
            paused_bench = run_prewarmed_frontier_benchmark(
                personas=("business_analyst", "statistician_signalguard"),
                dataset_modes=("null_honeypot",),
                n_seeds=2,
                nrows=200,
                dry_run=False,
                use_cache=True,
                cache_path=fresh_cache_file,
                daily_call_limit=0,
            )
            self.assertTrue(paused_bench["paused_for_daily_budget"])
            self.assertEqual(paused_bench["total_completed"], 0)
            self.assertEqual(paused_bench["total_remaining"], 4)
            self.assertEqual(paused_bench["total_live_api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
