"""Regenerate master and per-model summary CSVs directly from the cached LLM outputs.

This ensures 100% data lineage and reproducibility:
`real_world_llm_eval_cache.jsonl` -> per-run evaluation -> summary CSVs.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(r"c:\Users\eldri\Code\sampling_toolbox")
sys.path.insert(0, str(REPO_ROOT))

from signalguard.real_world_benchmark import run_real_world_frontier_eval
from signalguard.sql_semantic_benchmark import run_sql_semantic_frontier_eval

CACHE_PATH = REPO_ROOT / "Data" / "real_world_llm_eval_cache.jsonl"
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.5-flash-lite"]
SEEDS = [42, 59, 76]
PERSONAS = ["business_analyst", "statistician_unassisted", "statistician_signalguard"]
PERSONA_ALIASES = {"statistician_signalgate": "statistician_signalguard"}

def load_cache():
    tab, sql = {}, {}
    for line in open(CACHE_PATH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        v = d.get("value", d)
        if v.get("dry_run"):
            continue
        p = PERSONA_ALIASES.get(v["persona"], v["persona"])
        m = v["model_name"]
        s = int(v["seed"])
        if m not in MODELS or s not in SEEDS or p not in PERSONAS:
            continue
        if "mart_name" in v:
            sql[(v["mart_name"], m, p, s)] = v["output"]
        else:
            tab[(v["dataset_name"], m, p, s)] = v["output"]
    return tab, sql

def main():
    tab_cache, sql_cache = load_cache()
    print(f"Loaded cache entries: {len(tab_cache)} tabular, {len(sql_cache)} SQL")

    # 1. Process Tabular Real-World runs
    tab_runs = []
    datasets = ["lending_club", "king_county", "bank_marketing"]
    for ds in datasets:
        for model in MODELS:
            model_runs = []
            for persona in PERSONAS:
                for seed in SEEDS:
                    key = (ds, model, persona, seed)
                    if key not in tab_cache:
                        print(f"MISSING tabular cache: {key}")
                        continue
                    res = run_real_world_frontier_eval(
                        dataset_name=ds,
                        persona=persona,
                        model_name=model,
                        seed=seed,
                        cached_output=tab_cache[key]
                    )
                    a = res["holdout_audit"]
                    rec = a.get("signal_recall", {})
                    row = {
                        "Dataset": ds,
                        "Model": model,
                        "Persona": persona,
                        "Seed": seed,
                        "Holdout_Replication_Rate (%)": a["replication_rate"] * 100.0,
                        "Real_FDR (%)": a["real_fdr"] * 100.0,
                        "Mean_Train_RIG": a["mean_train_rig"],
                        "Mean_Holdout_RIG": a["mean_holdout_rig"],
                        "Generalization_Bias": a["mean_bias"],
                        "Decay (%)": a["mean_decay_pct"],
                        "Under_Reporting_Ratio": a["under_reporting_ratio"],
                        "Plain_Replication_Rate (%)": a.get("plain_replication_rate", 1.0) * 100.0,
                        "Plain_FDR (%)": a.get("plain_fdr", 0.0) * 100.0,
                        "Signal_Recall (%)": rec.get("signal_recall") * 100.0 if rec.get("signal_recall") is not None else np.nan,
                    }
                    tab_runs.append(row)
                    model_runs.append(row)

            # Per-model summary
            df_m = pd.DataFrame(model_runs)
            m_summary = df_m.groupby(["Dataset", "Model", "Persona"], as_index=False).agg({
                "Holdout_Replication_Rate (%)": "mean",
                "Real_FDR (%)": "mean",
                "Mean_Train_RIG": "mean",
                "Mean_Holdout_RIG": "mean",
                "Generalization_Bias": "mean",
                "Decay (%)": "mean",
                "Under_Reporting_Ratio": "mean",
                "Seed": "count",
            }).rename(columns={"Seed": "Completed_Seeds"})
            out_csv = REPO_ROOT / "Data" / f"real_world_eval_{ds}_{model}_summary.csv"
            m_summary.to_csv(out_csv, index=False)

    df_tab_all = pd.DataFrame(tab_runs)
    tab_master = df_tab_all.groupby(["Dataset", "Model", "Persona"], as_index=False).agg({
        "Holdout_Replication_Rate (%)": "mean",
        "Real_FDR (%)": "mean",
        "Mean_Train_RIG": "mean",
        "Mean_Holdout_RIG": "mean",
        "Generalization_Bias": "mean",
        "Decay (%)": "mean",
        "Under_Reporting_Ratio": "mean",
        "Seed": "count",
    }).rename(columns={"Seed": "Completed_Seeds"})
    tab_master.to_csv(REPO_ROOT / "Data" / "real_world_eval_all_models_summary.csv", index=False)
    print("Updated real_world_eval_all_models_summary.csv")

    # 2. Process SQL Semantic Warehouse runs
    sql_runs = []
    marts = [
        ("king_county_continuous", "continuous"),
        ("lending_club", "binary_classification"),
        ("bank_marketing", "binary_classification"),
    ]
    for mart, m_type in marts:
        for model in MODELS:
            model_runs = []
            for persona in PERSONAS:
                for seed in SEEDS:
                    key = (mart, model, persona, seed)
                    if key not in sql_cache:
                        print(f"MISSING sql cache: {key}")
                        continue
                    res = run_sql_semantic_frontier_eval(
                        mart_name=mart,
                        persona=persona,
                        model_name=model,
                        seed=seed,
                        cached_output=sql_cache[key]
                    )
                    a = res["holdout_audit"]
                    rec = a.get("signal_recall", {})
                    row = {
                        "Mart": mart,
                        "Metric_Type": m_type,
                        "Model": model,
                        "Persona": persona,
                        "Seed": seed,
                        "Holdout_Replication_Rate (%)": a["replication_rate"] * 100.0,
                        "Real_FDR (%)": a["real_fdr"] * 100.0,
                        "Mean_Train_RIG": a["mean_train_rig"],
                        "Mean_Holdout_RIG": a["mean_holdout_rig"],
                        "Generalization_Bias": a["mean_bias"],
                        "Decay (%)": a["mean_decay_pct"],
                        "Under_Reporting_Ratio": a["under_reporting_ratio"],
                        "Plain_Replication_Rate (%)": a.get("plain_replication_rate", 1.0) * 100.0,
                        "Plain_FDR (%)": a.get("plain_fdr", 0.0) * 100.0,
                        "Signal_Recall (%)": rec.get("signal_recall") * 100.0 if rec.get("signal_recall") is not None else np.nan,
                    }
                    sql_runs.append(row)
                    model_runs.append(row)

            # Per-model summary
            df_m = pd.DataFrame(model_runs)
            m_summary = df_m.groupby(["Mart", "Metric_Type", "Model", "Persona"], as_index=False).agg({
                "Holdout_Replication_Rate (%)": "mean",
                "Real_FDR (%)": "mean",
                "Mean_Train_RIG": "mean",
                "Mean_Holdout_RIG": "mean",
                "Generalization_Bias": "mean",
                "Decay (%)": "mean",
                "Under_Reporting_Ratio": "mean",
                "Seed": "count",
            }).rename(columns={"Seed": "Completed_Seeds"})
            out_csv = REPO_ROOT / "Data" / f"sql_semantic_eval_{mart}_{model}_summary.csv"
            m_summary.to_csv(out_csv, index=False)

    df_sql_all = pd.DataFrame(sql_runs)
    sql_master = df_sql_all.groupby(["Mart", "Metric_Type", "Model", "Persona"], as_index=False).agg({
        "Holdout_Replication_Rate (%)": "mean",
        "Real_FDR (%)": "mean",
        "Mean_Train_RIG": "mean",
        "Mean_Holdout_RIG": "mean",
        "Generalization_Bias": "mean",
        "Decay (%)": "mean",
        "Under_Reporting_Ratio": "mean",
        "Seed": "count",
    }).rename(columns={"Seed": "Completed_Seeds"})
    sql_master.to_csv(REPO_ROOT / "Data" / "sql_semantic_eval_all_models_summary.csv", index=False)
    print("Updated sql_semantic_eval_all_models_summary.csv")

if __name__ == "__main__":
    main()
