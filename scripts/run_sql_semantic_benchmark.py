"""CLI Runner for the Local SQLite + Governed Semantic Layer End-to-End Benchmark.

Evaluates autonomous LLM agents exploring real-world production marts via SQL GROUP BY:
  - king_county_continuous: Continuous log sale price (ANOVA F -> RIG gate)
  - lending_club: Binary loan default rate (Wilks Contingency RIG gate)
  - bank_marketing: Low-SNR deposit subscription rate (Wilks Contingency RIG gate)

Usage:
  python scripts/run_sql_semantic_benchmark.py --mart king_county_continuous --model-name claude-opus-5-5
  python scripts/run_sql_semantic_benchmark.py --dry-run
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from signalgate.sql_semantic_benchmark import (
    SEMANTIC_CATALOGS,
    run_sql_semantic_frontier_eval,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Local SQLite + Governed Semantic Layer Frontier LLM Evaluations."
    )
    parser.add_argument(
        "--mart",
        type=str,
        default="king_county_continuous",
        choices=list(SEMANTIC_CATALOGS.keys()),
        help="Semantic SQL mart to evaluate (default: king_county_continuous)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="claude-opus-5-5",
        help="Frontier LLM model identifier (default: claude-opus-5-5)",
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
        help="Training mart row count (default: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deterministic mock evaluation without calling external LLM APIs",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("  SignalGate End-to-End SQLite Warehouse + Semantic Layer Benchmark")
    print("=" * 78)
    print(f"  Semantic Mart     : {args.mart.upper()} ({SEMANTIC_CATALOGS[args.mart]['target_type']} metric)")
    print(f"  Target Model      : {args.model_name}")
    print(f"  Personas          : {', '.join(args.personas)}")
    print(f"  Seeds per Persona : {args.n_seeds}")
    print("=" * 78)

    seeds = [42 + i * 17 for i in range(args.n_seeds)]
    rows = []

    for p in args.personas:
        print(f"\n>>> Evaluating Persona: {p}...")
        for s in seeds:
            res = run_sql_semantic_frontier_eval(
                mart_name=args.mart,
                persona=p,
                model_name=args.model_name,
                seed=s,
                sample_train=args.sample_train,
                dry_run=args.dry_run,
            )
            audit = res["holdout_audit"]
            hit = "[CACHE HIT]" if res["cache_hit"] else "[LIVE API]"
            print(
                f"  {hit} Seed {s}: {audit['n_reported']} SQL drivers | "
                f"Holdout Replicated: {audit['replicated_count']}/{audit['n_reported']} ({audit['replication_rate']*100:.0f}%) | "
                f"Holdout RIG: {audit['mean_holdout_rig']:.4f} | Under-Reporting: {audit['under_reporting_ratio']:.2f}x"
            )
            rows.append({
                "Mart": args.mart,
                "Metric_Type": SEMANTIC_CATALOGS[args.mart]["target_type"],
                "Model": args.model_name,
                "Persona": p,
                "Seed": s,
                "Holdout_Replication_Rate (%)": audit["replication_rate"] * 100.0,
                "Real_FDR (%)": audit["real_fdr"] * 100.0,
                "Mean_Train_RIG": audit["mean_train_rig"],
                "Mean_Holdout_RIG": audit["mean_holdout_rig"],
                "Generalization_Bias": audit["mean_bias"],
                "Decay (%)": audit["mean_decay_pct"],
                "Under_Reporting_Ratio": audit["under_reporting_ratio"],
            })

    df_res = pd.DataFrame(rows)
    summary_df = df_res.groupby(["Mart", "Metric_Type", "Model", "Persona"], as_index=False).agg({
        "Holdout_Replication_Rate (%)": "mean",
        "Real_FDR (%)": "mean",
        "Mean_Train_RIG": "mean",
        "Mean_Holdout_RIG": "mean",
        "Generalization_Bias": "mean",
        "Decay (%)": "mean",
        "Under_Reporting_Ratio": "mean",
        "Seed": "count",
    }).rename(columns={"Seed": "Completed_Seeds"})

    print("\n" + "=" * 78)
    print("  SQLite + Semantic Layer Holdout Audit Summary")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print("=" * 78)

    out_csv = REPO_ROOT / "Data" / f"sql_semantic_eval_{args.mart}_{args.model_name}_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary table to: {out_csv}")


if __name__ == "__main__":
    main()
