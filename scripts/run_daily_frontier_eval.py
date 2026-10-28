"""Daily Frontier Model Evaluation Runner for SignalGate.

Enforces strict daily API quota budgets (e.g. Gemini free tier 15 calls/day)
using persistent JSONL disk caching (`Data/llm_eval_cache.jsonl`).
Can be run once per day manually or automated via `/schedule`.

Example Usage:
    # Day 1: Run up to 12 live API calls (leaves 3 calls safety margin)
    python scripts/run_daily_frontier_eval.py --model-name gemini-3.5-flash-lite --daily-limit 12

    # Check progress without consuming quota:
    python scripts/run_daily_frontier_eval.py --status-only

    # Dry-run test (0 API calls, mock persona synthesis):
    python scripts/run_daily_frontier_eval.py --dry-run
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Ensure signalgate package is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signalgate.persona_benchmark import (
    run_prewarmed_frontier_benchmark,
    LLMEvalTraceCache,
    DEFAULT_LLM_CACHE_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-day frontier model evaluations under strict daily API call budgets."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Frontier LLM model identifier (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--daily-limit",
        type=int,
        default=12,
        help="Maximum live API calls to make today (default: 12, safe for 15 calls/day quota)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds per persona/dataset cell (default: 3 seeds -> 18 total runs)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=DEFAULT_LLM_CACHE_PATH,
        help="Path to persistent JSONL trace cache file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deterministic mock synthesis instead of live LLM (consumes 0 API quota)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Inspect disk cache and report progress without making any live calls",
    )

    args = parser.parse_args()

    # Route dry-run traces to a dedicated file to keep production cache pristine
    actual_cache_path = (
        str(Path(args.cache_path).parent / "llm_eval_dry_run_cache.jsonl")
        if args.dry_run
        else args.cache_path
    )

    cache = LLMEvalTraceCache(cache_path=actual_cache_path)
    personas = ("business_analyst", "statistician_unassisted", "statistician_signalgate")
    dataset_modes = ("null_honeypot", "needle_in_haystack")
    total_planned = len(personas) * len(dataset_modes) * args.n_seeds

    print("=" * 72)
    print("  SignalGate Frontier Benchmark -- Daily Incremental Runner")
    print("=" * 72)
    print(f"  Target Model      : {args.model_name}")
    print(f"  Personas          : {', '.join(personas)}")
    print(f"  Dataset Modes     : {', '.join(dataset_modes)}")
    print(f"  Seeds per Cell    : {args.n_seeds} ({total_planned} total planned runs)")
    print(f"  Daily Call Limit  : {args.daily_limit} calls")
    print(f"  Cache File        : {actual_cache_path}")
    print(f"  Cached Traces     : {len(cache._store)} entries on disk")
    print("=" * 72)

    if args.status_only:
        print("\n[Status Only] Inspecting completed runs from cache...")
        res = run_prewarmed_frontier_benchmark(
            model_name=args.model_name,
            personas=personas,
            dataset_modes=dataset_modes,
            n_seeds=args.n_seeds,
            use_cache=True,
            cache_path=actual_cache_path,
            daily_call_limit=0,
        )
        print(f"Completed Runs in Cache : {res['total_completed']} / {total_planned}")
        print(f"Remaining Runs to Run   : {res['total_remaining']}")
        if not res["benchmark_df"].empty:
            print("\nCurrent Intermediate Results:")
            print(res["benchmark_df"].to_string(index=False))
        return

    print("\nStarting evaluation session...")
    results = run_prewarmed_frontier_benchmark(
        model_name=args.model_name,
        personas=personas,
        dataset_modes=dataset_modes,
        n_seeds=args.n_seeds,
        dry_run=args.dry_run,
        use_cache=True,
        cache_path=actual_cache_path,
        daily_call_limit=args.daily_limit,
    )

    print("\n" + "-" * 72)
    print("  Session Run Summary")
    print("-" * 72)
    print(f"  Total Planned Runs   : {results['total_planned']}")
    print(f"  Completed Runs       : {results['total_completed']} / {results['total_planned']}")
    print(f"  Cache Hits (0 cost)  : {results['total_cache_hits']}")
    print(f"  Live Calls Made Today: {results['total_live_api_calls']}")
    print(f"  Remaining Runs       : {results['total_remaining']}")

    if results["paused_for_daily_budget"]:
        print("\n" + "!" * 72)
        print(f"  [PAUSED] Daily budget limit reached ({results['total_live_api_calls']} calls made today).")
        print(f"  {results['total_remaining']} runs remaining to complete the benchmark.")
        print(f"  All {results['total_completed']} traces are securely cached in: {args.cache_path}")
        print("  Re-run this script tomorrow to complete the remaining runs!")
        print("!" * 72)
    elif results["total_remaining"] == 0:
        print("\n" + "*" * 72)
        print("  [SUCCESS] All planned runs have completed across all days!")
        print("*" * 72)

    if not results["benchmark_df"].empty:
        print("\nAggregated Benchmark Results:")
        print(results["benchmark_df"].to_string(index=False))

        output_csv = Path(args.cache_path).parent / f"frontier_eval_{args.model_name}_summary.csv"
        try:
            results["benchmark_df"].to_csv(output_csv, index=False)
            print(f"\nSaved summary table to: {output_csv}")
        except Exception as e:
            pass


if __name__ == "__main__":
    main()
