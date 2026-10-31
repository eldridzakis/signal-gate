import json
import re
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from signalguard.phase2_benchmark import (
    Phase2SQLWarehouse,
    Phase2InteractiveAgentRunner,
    DecoupledHoldoutAuditor,
    StatisticalArmAuditor,
)

class CardinalityAwarePhase2Runner(Phase2InteractiveAgentRunner):
    """
    Subclass of Phase2InteractiveAgentRunner that adds a cardinality-aware preview:
    When df_res has > 25 rows and contains 'n_k', sort by n_k descending so the largest
    slices are shown, and include a brief guidance hint.
    """
    def run_session(self, dry_run: bool = False):
        cfg = self.warehouse.config
        from signalguard.phase2_benchmark import SYSTEM_PROMPT_TEMPLATE
        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=cfg["description"],
            target_col=self.warehouse.target_col,
            target_type=self.warehouse.target_type,
            dimensions=", ".join(self.warehouse.dimensions),
            arm_name=self.arm.upper(),
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    "Begin your exploration by issuing your first SQL query using the format:\n"
                    "```sql\nSELECT ... FROM train_mart ...\n```\n"
                    "Or conclude your investigation when ready using:\n"
                    "```json\n{\n  \"action\": \"finish_analysis\",\n  \"findings\": [...]\n}\n```"
                ),
            },
        ]
        turns_executed = 0
        reported_findings = []

        while turns_executed < self.max_turns:
            turns_executed += 1
            if dry_run:
                response_text = '```json\n{"action": "finish_analysis", "findings": []}\n```'
            else:
                response_text = self._call_llm(messages)

            messages.append({"role": "assistant", "content": response_text})

            # Check if agent declared finish_analysis in a json block
            json_matches = re.findall(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            concluded = False
            for j_str in json_matches:
                try:
                    payload = json.loads(j_str)
                    if isinstance(payload, dict) and payload.get("action") == "finish_analysis":
                        reported_findings = payload.get("findings", [])
                        concluded = True
                        break
                except Exception:
                    pass

            if not concluded and "finish_analysis" in response_text:
                raw_json_match = re.search(r"(\{\s*\"action\"\s*:\s*\"finish_analysis\".*?\})", response_text, re.DOTALL)
                if raw_json_match:
                    try:
                        payload = json.loads(raw_json_match.group(1))
                        reported_findings = payload.get("findings", [])
                        concluded = True
                    except Exception:
                        pass

            if concluded:
                for f in reported_findings:
                    sf = f.get("subgroup_filter")
                    if isinstance(sf, dict):
                        f["subgroup_filter"] = " AND ".join([f"{k} = '{v}'" for k, v in sf.items()])
                break

            sql_match = re.search(r"```sql\s*(SELECT.*?;?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
            if not sql_match:
                messages.append({
                    "role": "user",
                    "content": (
                        "Turn note: No valid ```sql SELECT ... ``` or ```json {\"action\": \"finish_analysis\", \"findings\": [...]} ``` block detected. "
                        "Please write your SQL query in a ```sql ... ``` block or conclude."
                    ),
                })
                continue

            query = sql_match.group(1).strip().rstrip(";")
            try:
                df_res = self.warehouse.run_query(query, on_holdout=False)
                audit_res = self.auditor.audit_query_result(
                    query=query,
                    df_result=df_res,
                    target_col=self.warehouse.target_col,
                    n_table_rows=len(self.warehouse.train_df),
                )

                # CARDINALITY-AWARE PREVIEW
                display_df = df_res.copy()
                hint_msg = ""
                if len(df_res) > 25:
                    if "n_k" in display_df.columns:
                        display_df = display_df.sort_values(by="n_k", ascending=False)
                    hint_msg = f"\n[Notice: {len(df_res)} rows returned. Showing top 25 by sample size n_k. Consider filtering with WHERE or reducing GROUP BY dimensions.]"

                preview_str = display_df.head(25).to_string(index=False)
                if len(df_res) > 25:
                    preview_str += f"\n... [{len(df_res) - 25} more rows omitted]"
                if hint_msg:
                    preview_str += hint_msg

                tool_msg = (
                    f"**SQL Execution Result (Turn {turns_executed}/{self.max_turns})**:\n\n"
                    f"```\n{preview_str}\n```\n\n"
                    f"**Statistical Audit ({self.arm.upper()})**:\n"
                    f"- Verdict: `{audit_res['verdict']}`\n"
                    f"- Details: {audit_res['guidance']}\n"
                    f"- Degrees of freedom: {audit_res.get('degrees_of_freedom', 1)}, Sample N: {audit_res.get('sample_size', 0)}\n\n"
                    "Proceed with your next query or finalize your findings with ```json finish_analysis ```."
                )
            except Exception as e:
                tool_msg = f"SQL Execution Error: {str(e)}. Please correct your query syntax."

            messages.append({"role": "user", "content": tool_msg})

        # If turn budget reached
        if not reported_findings and not dry_run:
            messages.append({
                "role": "user",
                "content": (
                    "Turn budget reached. Please review the findings that cleared statistical verification across your exploration, and finalize your report in this JSON format:\n"
                    "```json\n{\n  \"action\": \"finish_analysis\",\n  \"findings\": [...]\n}\n```\n"
                    "If no findings cleared statistical verification, output {\"action\": \"finish_analysis\", \"findings\": []}."
                ),
            })
            try:
                final_resp = self._call_llm(messages)
                messages.append({"role": "assistant", "content": final_resp})
                json_matches = re.findall(r"```json\s*(.*?)\s*```", final_resp, re.DOTALL)
                for j_str in json_matches:
                    try:
                        payload = json.loads(j_str)
                        if isinstance(payload, dict):
                            reported_findings = payload.get("findings", [])
                            if reported_findings:
                                break
                    except Exception:
                        pass
            except Exception:
                pass

        return {
            "turns_executed": turns_executed,
            "queries_executed": len(self.auditor.history),
            "reported_findings": reported_findings,
            "query_history": self.auditor.history,
            "messages": messages,
        }

def run_ablation():
    model = "gpt-6-luna"
    task = "lending_club_subgroup"
    arms = ["unassisted", "signalguard"]
    seeds = [42, 59, 76]

    print(f"=== RUNNING TARGETED ABLATION: {model} on {task} (Cardinality-Aware Harness) ===")
    results = []

    out_file = REPO_ROOT / "Data" / "luna_ablation_results.jsonl"
    if out_file.exists():
        out_file.unlink()

    for arm in arms:
        for seed in seeds:
            print(f"\n--- Arm: {arm} | Seed: {seed} ---", flush=True)
            warehouse = Phase2SQLWarehouse(task_name=task, seed=seed)
            session = CardinalityAwarePhase2Runner(
                warehouse=warehouse,
                arm=arm,
                model_name=model,
                max_turns=8,
            )
            session_res = session.run_session()

            holdout_auditor = DecoupledHoldoutAuditor(warehouse=warehouse)
            audit_eval = holdout_auditor.evaluate_reported_findings(session_res["reported_findings"])

            rec = {
                "model": model,
                "arm": arm,
                "task": task,
                "seed": seed,
                "queries_executed": session_res["turns_executed"],
                "n_reported": audit_eval["n_reported"],
                "n_replicated": audit_eval["n_replicated"],
                "n_failed": audit_eval["n_failed"],
                "empirical_fdr": audit_eval["empirical_fdr"],
                "precision": audit_eval["precision"],
                "signal_recall": audit_eval["signal_recall"],
                "discovery_f1": audit_eval["discovery_f1"],
                "reported_findings": session_res["reported_findings"],
            }
            results.append(rec)
            print(f"  Result: Queries={rec['queries_executed']} | Reported={rec['n_reported']} | Replicated={rec['n_replicated']} | FDR={rec['empirical_fdr']:.1f}% | F1={rec['discovery_f1']:.3f}", flush=True)

            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    df_res = pd.DataFrame(results)
    print("\n=== ABLATION SUMMARY: LUNA on LENDING CLUB (Cardinality-Aware Preview) ===")
    summary = df_res.groupby("arm").agg(
        avg_q=("queries_executed", "mean"),
        tot_reported=("n_reported", "sum"),
        tot_replicated=("n_replicated", "sum"),
        tot_failed=("n_failed", "sum"),
        mean_fdr=("empirical_fdr", "mean"),
        mean_recall=("signal_recall", "mean"),
        mean_f1=("discovery_f1", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_ablation()
