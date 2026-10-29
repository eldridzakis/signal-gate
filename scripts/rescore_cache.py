"""
Rescores all cached Stage 1 runs using the updated DecoupledHoldoutAuditor.
Zero LLM API calls are made ($0.00 spend).
"""

import json
from pathlib import Path
import pandas as pd
from signalguard.phase2_benchmark import (
    Phase2SQLWarehouse,
    DecoupledHoldoutAuditor,
    DEFAULT_PHASE2_CACHE_PATH,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "Data" / "phase2_stage1_cache.jsonl"
SUMMARY_PATH = REPO_ROOT / "Data" / "phase2_stage1_summary.csv"

def rescore_all():
    if not CACHE_PATH.exists():
        print(f"No cache file found at {CACHE_PATH}")
        return

    records = []
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} cached session runs from {CACHE_PATH}")

    warehouses = {}
    rescored_records = []

    for i, rec in enumerate(records, 1):
        task = rec["task"]
        arm = rec["arm"]
        seed = rec["seed"]
        model = rec.get("model", "models/gemini-2.5-flash-lite")
        findings = rec.get("reported_findings", [])

        wh_key = (task, seed)
        if wh_key not in warehouses:
            warehouses[wh_key] = Phase2SQLWarehouse(task_name=task, seed=seed)
        wh = warehouses[wh_key]

        auditor = DecoupledHoldoutAuditor(warehouse=wh, alpha=0.05)
        audit = auditor.evaluate_reported_findings(findings)

        rec["n_reported"] = audit["n_reported"]
        rec["n_replicated"] = audit["n_replicated"]
        rec["n_failed"] = audit["n_failed"]
        rec["empirical_fdr"] = audit["empirical_fdr"]
        rec["precision"] = audit["precision"]
        rec["signal_recall"] = audit["signal_recall"]
        rec["discovery_f1"] = audit["discovery_f1"]
        rec["audit_details"] = audit["audit_details"]

        rescored_records.append(rec)
        print(
            f"[{i:02d}/{len(records):02d}] {task[:18]:<18} | {arm[:12]:<12} | seed={seed} -> "
            f"Rep: {audit['n_replicated']}/{audit['n_reported']} | "
            f"FDR: {audit['empirical_fdr']:.1f}% | "
            f"Recall: {audit['signal_recall']:.1f}% | "
            f"F1: {audit['discovery_f1']:.2f}"
        )

    # Save updated cache
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for r in rescored_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(rescored_records)} rescored records to {CACHE_PATH}")

    # Generate summary CSV
    df = pd.DataFrame(rescored_records)
    summary = df.groupby(["task", "arm"], as_index=False).agg({
        "queries_executed": "mean",
        "n_reported": "mean",
        "n_replicated": "mean",
        "empirical_fdr": "mean",
        "precision": "mean",
        "signal_recall": "mean",
        "discovery_f1": "mean",
        "seed": "count",
    }).rename(columns={"seed": "completed_splits"})

    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved aggregated summary to {SUMMARY_PATH}\n")

    # Print summary table
    print("=" * 86)
    print("  STAGE 1 RESCORED BENCHMARK SUMMARY (gemini-3.5-flash-lite)")
    print("=" * 86)
    print(summary.to_string(index=False))
    print("=" * 86)

    # Print overall cross-task rollup by arm
    rollup = df.groupby("arm", as_index=False).agg({
        "queries_executed": "mean",
        "n_reported": "sum",
        "n_replicated": "sum",
        "empirical_fdr": "mean",
        "precision": "mean",
        "signal_recall": "mean",
        "discovery_f1": "mean",
    })
    print("\n" + "=" * 86)
    print("  OVERALL CROSS-TASK ROLLUP BY STATISTICAL ARM")
    print("=" * 86)
    print(rollup.to_string(index=False))
    print("=" * 86)

if __name__ == "__main__":
    rescore_all()
