#!/usr/bin/env python3
"""CLI script to run the Agentune AutoML Overfitting & SignalGate Benchmark."""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from signalgate.agentune_benchmark import run_agentune_overfitting_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run Agentune AutoML Overfitting & SignalGate Benchmark.")
    parser.add_argument("--dataset", type=str, default="lending_club", choices=["lending_club", "king_county", "bank_marketing"])
    parser.add_argument("--max-candidates", type=int, default=1000, help="Number of synthetic candidate features M")
    parser.add_argument("--top-k", type=int, default=15, help="Top K features to promote")
    parser.add_argument("--sample-train", type=int, default=5000)
    parser.add_argument("--sample-test", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "Data"))

    args = parser.parse_args()

    print("=" * 76)
    print("  Agentune-Style AutoML Overfitting vs SignalGate Benchmark")
    print("=" * 76)
    print(f"  Dataset          : {args.dataset.upper()}")
    print(f"  Candidates (M)   : {args.max_candidates}")
    print(f"  Top K Features   : {args.top_k}")
    print(f"  Sample Sizes     : Train N={args.sample_train}, Holdout Test N={args.sample_test}")
    print(f"  Random Seed      : {args.seed}")
    print("=" * 76)

    res = run_agentune_overfitting_benchmark(
        dataset_name=args.dataset,
        max_candidates=args.max_candidates,
        top_k=args.top_k,
        sample_train=args.sample_train,
        sample_test=args.sample_test,
        seed=args.seed
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"agentune_benchmark_{args.dataset}_summary.csv"
    downstream_path = out_dir / f"agentune_benchmark_{args.dataset}_downstream.csv"
    unassisted_path = out_dir / f"agentune_benchmark_{args.dataset}_unassisted_topk.csv"
    signalgate_path = out_dir / f"agentune_benchmark_{args.dataset}_signalgate_topk.csv"

    res["summary_df"].to_csv(summary_path, index=False)
    res["downstream_df"].to_csv(downstream_path, index=False)
    res["unassisted_verified_df"].to_csv(unassisted_path, index=False)
    res["signalgate_verified_df"].to_csv(signalgate_path, index=False)

    print("\n>>> SUMMARY METRICS (Out-of-Sample Holdout Replication):")
    print(res["summary_df"].to_string(index=False))

    print("\n>>> DOWNSTREAM ML MODEL EVALUATION (Overfitting Gap):")
    print(res["downstream_df"].to_string(index=False))

    print(f"\nSaved benchmark outputs to {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
