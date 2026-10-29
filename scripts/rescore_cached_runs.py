"""Re-score cached frontier-LLM reports under a plain holdout criterion (NO API calls).

For every cached LLM report (tabular + SQL semantic benchmarks) this script re-runs the deterministic exploration
transcript and holdout audit, but takes the model output from `Data/real_world_llm_eval_cache.jsonl` instead of calling
an API. It reports, per run:

  * original criterion  : holdout p < 0.05 AND RIG > eps AND no small-cell warning   (as in REPORT.md Tables 1-2)
  * plain criterion     : holdout p < 0.05 only                                      (independent of SignalGuard's gate)
  * signal recall       : share of 'real-signal pool' candidates (Bonferroni-significant on holdout) the agent reported

Usage:
  python scripts/rescore_cached_runs.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from signalguard.real_world_benchmark import run_real_world_frontier_eval  # noqa: E402
from signalguard.sql_semantic_benchmark import run_sql_semantic_frontier_eval  # noqa: E402

CACHE = REPO_ROOT / "Data" / "real_world_llm_eval_cache.jsonl"
OUT_RUNS = REPO_ROOT / "Data" / "rescored_plain_holdout_per_run.csv"
OUT_SUMMARY = REPO_ROOT / "Data" / "rescored_plain_holdout_summary.csv"
SEEDS = [42, 59, 76]
PERSONA_ALIASES = {"statistician_signalgate": "statistician_signalguard"}
PERSONAS = ["business_analyst", "statistician_unassisted", "statistician_signalguard"]


def load_cache():
    tab, sql = {}, {}
    for line in open(CACHE, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        v = d.get("value", d)
        if v.get("dry_run"):
            continue
        persona = PERSONA_ALIASES.get(v["persona"], v["persona"])
        if "mart_name" in v:
            sql[(v["mart_name"], v["model_name"], persona, int(v["seed"]))] = v["output"]
        else:
            tab[(v["dataset_name"], v["model_name"], persona, int(v["seed"]))] = v["output"]
    return tab, sql


def main():
    tab, sql = load_cache()
    print(f"Cached live reports: {len(tab)} tabular, {len(sql)} SQL")

    rows = []
    for kind, store, fn, name_kw in (
        ("tabular", tab, run_real_world_frontier_eval, "dataset_name"),
        ("sql", sql, run_sql_semantic_frontier_eval, "mart_name"),
    ):
        keys = sorted(k for k in store if k[3] in SEEDS)
        for i, (name, model, persona, seed) in enumerate(keys, 1):
            res = fn(**{name_kw: name}, persona=persona, model_name=model, seed=seed,
                     cached_output=store[(name, model, persona, seed)])
            a = res["holdout_audit"]
            rec = a.get("signal_recall", {})
            rows.append({
                "Benchmark": kind, "Dataset": name, "Model": model, "Persona": persona, "Seed": seed,
                "N_Reported": a["n_reported"],
                "Orig_Failed": a["failed_count"],
                "Plain_Failed": a["plain_failed_count"],
                "Orig_FDR (%)": a["real_fdr"] * 100.0 if a["n_reported"] else 0.0,
                "Plain_FDR (%)": a["plain_fdr"] * 100.0 if a["n_reported"] else 0.0,
                "Pool_Size": rec.get("pool_size"),
                "Real_Pool_Size": rec.get("real_pool_size"),
                "Recovered": rec.get("recovered"),
            })
            if i % 20 == 0:
                print(f"  [{kind}] {i}/{len(keys)}")

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_RUNS, index=False)

    def agg(g):
        n = g["N_Reported"].sum()
        real = g["Real_Pool_Size"].sum()
        return pd.Series({
            "Runs": len(g),
            "Findings_Reported": int(n),
            "Orig_FDR_pooled (%)": 100.0 * g["Orig_Failed"].sum() / n if n else 0.0,
            "Plain_FDR_pooled (%)": 100.0 * g["Plain_Failed"].sum() / n if n else 0.0,
            "Real_Pool_Findings": int(real),
            "Signal_Recall_pooled (%)": 100.0 * g["Recovered"].sum() / real if real else np.nan,
        })

    summary = runs.groupby(["Benchmark", "Dataset", "Persona"], as_index=False).apply(agg, include_groups=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== By benchmark / dataset / persona (pooled over models x seeds) ===")
    print(summary.round(1).to_string(index=False))
    overall = runs.groupby(["Benchmark", "Persona"], as_index=False).apply(agg, include_groups=False)
    print("\n=== By benchmark / persona ===")
    print(overall.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
