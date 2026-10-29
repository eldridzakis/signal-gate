"""CLI Runner for Phase 2 Stage 1 Scout Benchmark.

Evaluates everyday, sub-cent models (Gemini 3.5 Flash Lite, GPT-6 Luna, Claude Haiku 5.5)
across the fair 5-pillar benchmark:
  1. Free Agentic SQL Search (LLM writes its own queries in an interactive loop)
  2. Decoupled Pure Holdout Validation (p_holdout < 0.05, no shared small-cell rule)
  3. Two-Sided Evaluation (Empirical FDR, Signal Recall, Discovery F1)
  4. Fair Statistical Baselines (unassisted, bonferroni, benjamini_hochberg, signalguard)
  5. Cross-Validation across 4 realistic tasks (King County, Lending Club, Null A/B, Seeded HTE)

Usage:
  # Dry-run test (0 API calls, deterministic mock agents):
  python scripts/run_phase2_stage1.py --dry-run

  # Live run on Gemini 3.5 Flash Lite (Free-tier safe, 500 RPD):
  python scripts/run_phase2_stage1.py --model-name gemini-3.5-flash-lite

  # Live run on GPT-6 Luna (~$0.15 total):
  python scripts/run_phase2_stage1.py --model-name gpt-6-luna

  # Live run on Claude Haiku 5.5 (~$0.15 total):
  python scripts/run_phase2_stage1.py --model-name claude-haiku-5-5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from signalguard.phase2_benchmark import (
    PHASE2_TASKS,
    Phase2SQLWarehouse,
    Phase2InteractiveAgentRunner,
    DecoupledHoldoutAuditor,
    Phase2TraceCache,
    DEFAULT_PHASE2_CACHE_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Stage 1 Scout Benchmark Runner")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Model ID (e.g. gemini-3.5-flash-lite, gpt-6-luna, claude-haiku-5-5)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["king_county_subgroup", "lending_club_subgroup", "bank_marketing_null_ab", "bank_marketing_seeded_hte"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["unassisted", "bonferroni", "benjamini_hochberg", "signalguard"],
        help="Statistical arms to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 59, 76],
        help="Random seeds for cross-validation splits",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum SQL exploration turns per session (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deterministic mock agents (0 API calls, $0 spend)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=DEFAULT_PHASE2_CACHE_PATH,
        help="Path to persistent JSONL trace cache",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(REPO_ROOT / "Data" / "phase2_stage1_summary.csv"),
        help="Path to save compiled summary CSV",
    )

    args = parser.parse_args()

    actual_cache_path = (
        str(Path(args.cache_path).parent / "phase2_dry_run_cache.jsonl")
        if args.dry_run
        else args.cache_path
    )
    cache = Phase2TraceCache(cache_path=actual_cache_path)

    total_runs = len(args.tasks) * len(args.arms) * len(args.seeds)

    print("=" * 76)
    print("  SignalGuard Phase 2 (Stage 1: Scout Benchmark)")
    print("=" * 76)
    print(f"  Target Model      : {args.model_name}")
    print(f"  Tasks ({len(args.tasks)})        : {', '.join(args.tasks)}")
    print(f"  Arms ({len(args.arms)})         : {', '.join(args.arms)}")
    print(f"  Seeds ({len(args.seeds)})        : {args.seeds}")
    print(f"  Total Run Matrix  : {total_runs} session evaluations")
    print(f"  Max Turns / Run   : {args.max_turns} SQL queries")
    print(f"  Mode              : {'DRY RUN (Mock Agents, $0 API Spend)' if args.dry_run else 'LIVE MODEL EXECUTION'}")
    print(f"  Cache File        : {actual_cache_path}")
    print("=" * 76)

    completed_records: List[Dict[str, Any]] = []
    run_idx = 0

    for task in args.tasks:
        for arm in args.arms:
            for seed in args.seeds:
                run_idx += 1
                prefix = f"[{run_idx:02d}/{total_runs:02d}] {task[:18]} | {arm[:12]} | seed={seed}"

                # Check cache first
                cached = cache.get(task=task, arm=arm, model=args.model_name, seed=seed)
                if cached is not None and not args.dry_run:
                    print(f"  {prefix} -> CACHED (FDR={cached.get('empirical_fdr', 0):.1f}%, F1={cached.get('discovery_f1', 0):.2f})")
                    completed_records.append(cached)
                    continue

                # Materialize warehouse for this task and seed
                wh = Phase2SQLWarehouse(task_name=task, seed=seed)
                runner = Phase2InteractiveAgentRunner(
                    warehouse=wh,
                    arm=arm,
                    model_name=args.model_name,
                    max_turns=args.max_turns,
                )

                try:
                    session_res = runner.run_session(dry_run=args.dry_run)
                    evaluator = DecoupledHoldoutAuditor(warehouse=wh, alpha=0.05)
                    audit = evaluator.evaluate_reported_findings(session_res["reported_findings"])

                    record = {
                        "task": task,
                        "arm": arm,
                        "model": args.model_name,
                        "seed": seed,
                        "queries_executed": session_res["queries_executed"],
                        "n_reported": audit["n_reported"],
                        "n_replicated": audit["n_replicated"],
                        "n_failed": audit["n_failed"],
                        "empirical_fdr": audit["empirical_fdr"],
                        "precision": audit["precision"],
                        "signal_recall": audit["signal_recall"],
                        "discovery_f1": audit["discovery_f1"],
                        "reported_findings": session_res["reported_findings"],
                        "audit_details": audit["audit_details"],
                        "query_history": session_res.get("query_history", []),
                        "messages": session_res.get("messages", []),
                    }

                    cache.put(record)
                    completed_records.append(record)

                    print(
                        f"  {prefix} -> Queries: {record['queries_executed']} | "
                        f"Reported: {record['n_reported']} | "
                        f"Replicated: {record['n_replicated']} | "
                        f"FDR: {record['empirical_fdr']:.1f}% | "
                        f"Recall: {record['signal_recall']:.1f}% | "
                        f"F1: {record['discovery_f1']:.2f}"
                    )
                except Exception as e:
                    print(f"  {prefix} -> ERROR: {str(e)}")

    # Compile into summary table
    if completed_records:
        df_all = pd.DataFrame(completed_records)
        summary = df_all.groupby(["task", "arm"], as_index=False).agg({
            "queries_executed": "mean",
            "n_reported": "mean",
            "n_replicated": "mean",
            "empirical_fdr": "mean",
            "signal_recall": "mean",
            "discovery_f1": "mean",
            "seed": "count",
        }).rename(columns={"seed": "completed_splits"})

        print("\n" + "=" * 76)
        print(f"  Stage 1 Summary Results: {args.model_name.upper()}")
        print("=" * 76)
        print(summary.to_string(index=False))

        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved summary table to: {summary_path}")


if __name__ == "__main__":
    main()
