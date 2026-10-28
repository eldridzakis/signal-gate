"""CLI Runner for Real-World Empirical Frontier Model Evaluations.

Evaluates autonomous LLM data agents on real-world production datasets:
  - Lending Club Loan Defaults (54,655 real loans, target: 'bad_loans')
  - King County Housing Sales (21,610 transactions, target: 'high_value')

Audits agent discoveries directly against untouched holdout test partitions.

Usage:
  python scripts/run_real_world_frontier_eval.py --dataset lending_club --model-name gemini-3.5-flash-lite
  python scripts/run_real_world_frontier_eval.py --dataset king_county --model-name gemini-3.5-flash-lite
  python scripts/run_real_world_frontier_eval.py --dry-run
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from signalgate.real_world_benchmark import (
    run_real_world_frontier_eval,
    DEFAULT_REAL_WORLD_CACHE_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run real-world frontier LLM evaluations with out-of-sample holdout auditing."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lending_club",
        choices=["lending_club", "king_county", "bank_marketing"],
        help="Real-world dataset to benchmark: 'lending_club', 'king_county', or 'bank_marketing' (default: lending_club)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Frontier LLM model identifier (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=["business_analyst", "statistician_unassisted", "statistician_signalgate"],
        help="Personas to evaluate",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds per persona (default: 3)",
    )
    parser.add_argument(
        "--sample-train",
        type=int,
        default=5000,
        help="Sample size for training partition (default: 5000)",
    )
    parser.add_argument(
        "--sample-test",
        type=int,
        default=None,
        help="Sample size for holdout test partition (default: None = full untouched holdout test partition)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute deterministic mock evaluation without calling external LLM APIs (0 cost)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass disk cache and force live API execution",
    )

    args = parser.parse_args()

    print("=" * 76)
    print("  SignalGate Real-World Empirical Frontier Benchmark")
    print("=" * 76)
    print(f"  Dataset           : {args.dataset.upper()} (Real-World Train vs. Holdout)")
    print(f"  Target Model      : {args.model_name}")
    print(f"  Personas          : {', '.join(args.personas)}")
    print(f"  Seeds per Persona : {args.n_seeds} ({len(args.personas) * args.n_seeds} total runs)")
    print(f"  Train Sample      : N = {args.sample_train} records")
    print(f"  Holdout Test      : N = {args.sample_test} records")
    print(f"  Cache File        : {DEFAULT_REAL_WORLD_CACHE_PATH}")
    print(f"  Mode              : {'DRY RUN (Mock Synthesis)' if args.dry_run else 'LIVE FRONTIER MODEL'}")
    print("=" * 76)

    seeds = [42 + i * 17 for i in range(args.n_seeds)]
    all_results = []

    for p in args.personas:
        print(f"\n>>> Running Persona: {p}...")
        for s in seeds:
            res = run_real_world_frontier_eval(
                dataset_name=args.dataset,
                persona=p,
                model_name=args.model_name,
                seed=s,
                sample_train=args.sample_train,
                sample_test=args.sample_test,
                dry_run=args.dry_run,
                use_cache=not args.no_cache,
            )
            audit = res["holdout_audit"]
            hit_str = "[CACHE HIT]" if res["cache_hit"] else "[LIVE API]"
            print(
                f"  {hit_str} Seed {s}: {len(res['reported_discoveries'])} drivers promoted | "
                f"Holdout Replicated: {audit['replicated_count']}/{audit['n_reported']} ({audit['replication_rate']*100:.0f}%) | "
                f"Bias: {audit['mean_bias']:.5f} | Under-Reporting: {audit['under_reporting_ratio']:.2f}x"
            )
            all_results.append({
                "Dataset": args.dataset,
                "Model": args.model_name,
                "Persona": p,
                "Seed": s,
                "Reported_Drivers": audit["n_reported"],
                "Replicated_Drivers": audit["replicated_count"],
                "Failed_Drivers": audit["failed_count"],
                "Holdout_Replication_Rate (%)": audit["replication_rate"] * 100.0,
                "Real_FDR (%)": audit["real_fdr"] * 100.0,
                "Mean_Train_RIG": audit["mean_train_rig"],
                "Mean_Holdout_RIG": audit["mean_holdout_rig"],
                "Generalization_Bias": audit["mean_bias"],
                "Decay (%)": audit["mean_decay_pct"],
                "Under_Reporting_Ratio": audit["under_reporting_ratio"],
                "Cache_Hit": res["cache_hit"],
            })

    df_res = pd.DataFrame(all_results)
    summary_df = df_res.groupby(["Dataset", "Model", "Persona"]).agg({
        "Holdout_Replication_Rate (%)": "mean",
        "Real_FDR (%)": "mean",
        "Mean_Train_RIG": "mean",
        "Mean_Holdout_RIG": "mean",
        "Generalization_Bias": "mean",
        "Decay (%)": "mean",
        "Under_Reporting_Ratio": "mean",
        "Seed": "count",
    }).reset_index().rename(columns={"Seed": "Completed_Seeds"})

    print("\n" + "=" * 76)
    print("  Aggregated Real-World Benchmark Results (Out-of-Sample Holdout Audit)")
    print("=" * 76)
    print(summary_df.to_string(index=False))
    print("=" * 76)

    out_csv = REPO_ROOT / "Data" / f"real_world_eval_{args.dataset}_{args.model_name}_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary table to: {out_csv}")


if __name__ == "__main__":
    main()
