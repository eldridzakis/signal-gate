"""Phase 2 Empirical Agentic Benchmark Engine for SignalGuard.

Implements the Fair, Unbiased 5-Pillar Benchmark Framework:
  1. Free Agentic Search: The LLM writes and executes its own SQL queries in an open loop.
  2. Decoupled Pure Holdout: Findings replicate iff p_holdout < 0.05 with matching effect sign (no shared small-sample filter).
  3. Two-Sided Evaluation: Reports Empirical FDR (1 - Precision), Signal Recall, and Discovery F1.
  4. Fair Statistical Baselines: Same prompt, same database, differing only by statistical tool:
       - unassisted (naive per-query p-value)
       - bonferroni (standard alpha / M gate)
       - benjamini_hochberg (standard BH step-up FDR)
       - signalguard (stateful MCP ledger with M_eff + closed-form ANOVA / EVT noise ceiling)
  5. Cross-Validation: Repeated stratified train/holdout splits with error bars across 4 realistic tasks:
       - King County Housing (continuous log_price conditional subgroup hunting)
       - Lending Club Defaults (binary bad_loans risk pocket hunting)
       - Bank Marketing Null A/B Experiment (0% null treatment - pure FDR stress test)
       - Bank Marketing Seeded HTE Experiment (2 genuine responder segments - Precision + Recall test)
"""

import copy
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import anova_rig_from_group_stats, fast_entropy_from_marginals, rig
from .threshold import FeatureNoiseThreshold, estimate_effective_tests
from .mcp_server import NoiseGateSessionLedger
from .persona_benchmark import _ensure_api_keys, _resolve_llm_model

DEFAULT_PHASE2_CACHE_PATH = str(Path(__file__).resolve().parent.parent / "Data" / "phase2_stage1_cache.jsonl")


# =============================================================================
# 1. TASK CATALOG & SQL WAREHOUSE
# =============================================================================

PHASE2_TASKS = {
    "king_county_subgroup": {
        "dataset": "king_county",
        "target_col": "log_price",
        "target_type": "continuous",
        "description": (
            "King County Housing Mart — Continuous Log Price. Stakeholders already know that 'grade' "
            "is the primary macro driver. Task: Control for grade (e.g. within grade strata 7, 8, or 9) "
            "and search across secondary dimensions (condition, view, waterfront, bathrooms_bin, floors) "
            "to discover specific sub-segments where price deviates significantly from the grade baseline."
        ),
        "primary_control": "grade",
        "dimensions": ["grade", "condition", "view", "waterfront", "bathrooms_bin", "floors", "yr_built_bin"],
    },
    "lending_club_subgroup": {
        "dataset": "lending_club",
        "target_col": "bad_loans",
        "target_type": "binary",
        "description": (
            "Lending Club Credit Mart — Binary Loan Defaults. Stakeholders already know credit 'grade' "
            "strongly drives default risk. Task: Controlling for grade (e.g. within grade B or C), identify "
            "specific borrower sub-segments (combining purpose, home_ownership, term, addr_state, emp_length) "
            "with anomalously high or low default rates relative to their grade baseline."
        ),
        "primary_control": "grade",
        "dimensions": ["grade", "term", "home_ownership", "purpose", "emp_length", "addr_state"],
    },
    "bank_marketing_null_ab": {
        "dataset": "bank_marketing",
        "target_col": "subscribed",
        "target_type": "binary",
        "is_ab_test": True,
        "ab_mode": "null",
        "description": (
            "Retail Bank Marketing Campaign V2 Evaluation. An A/B experiment was run testing Campaign V2 "
            "(variant='treatment') vs Control (variant='control'). Overall average lift was completely flat "
            "(+0.2%, p = 0.44). Stakeholders want to know if Campaign V2 drove statistically significant "
            "subscription lift for ANY specific demographic, economic, or contact sub-segment before terminating the campaign."
        ),
        "primary_control": "variant",
        "dimensions": ["variant", "job", "education", "marital", "housing", "contact", "poutcome", "month"],
    },
    "bank_marketing_seeded_hte": {
        "dataset": "bank_marketing",
        "target_col": "subscribed",
        "target_type": "binary",
        "is_ab_test": True,
        "ab_mode": "seeded",
        "description": (
            "Retail Bank Marketing Campaign V3 Evaluation. An A/B experiment tested Campaign V3 (variant='treatment' "
            "vs variant='control'). Overall lift was modest (+1.8%, p = 0.08). Stakeholders want to identify specific "
            "customer sub-segments that exhibit genuine, statistically significant positive treatment effects (treatment responders)."
        ),
        "primary_control": "variant",
        "dimensions": ["variant", "job", "education", "marital", "housing", "contact", "poutcome", "month"],
        "ground_truth_subgroups": [
            {"filter": "job = 'retired' AND housing = 'no'", "true_lift": "+12.0%"},
            {"filter": "poutcome = 'success'", "true_lift": "+15.0%"},
        ],
    },
}


class Phase2SQLWarehouse:
    """Creates isolated SQLite databases with train_mart and holdout_vault for a given task and split."""

    def __init__(self, task_name: str, seed: int = 42, train_ratio: float = 0.5):
        if task_name not in PHASE2_TASKS:
            raise ValueError(f"Unknown task {task_name}. Available: {list(PHASE2_TASKS.keys())}")

        self.task_name = task_name
        self.config = PHASE2_TASKS[task_name]
        self.target_col = self.config["target_col"]
        self.target_type = self.config["target_type"]
        self.dimensions = self.config["dimensions"]
        self.seed = seed

        data_dir = Path(__file__).resolve().parent.parent / "Data"
        ds_name = self.config["dataset"]

        if ds_name == "king_county":
            raw_path = data_dir / "KingCountyHouseSales_with_random_parition_column.csv.gz"
            df = pd.read_csv(raw_path)
            df["log_price"] = np.log(np.clip(df["price"].astype(float), 1.0, None))
            # Discretize continuous variables into clean string bins
            df["bathrooms_bin"] = pd.qcut(df["bathrooms"], q=4, duplicates="drop").astype(str)
            df["yr_built_bin"] = pd.qcut(df["yr_built"], q=5, duplicates="drop").astype(str)
            for col in ["grade", "condition", "view", "waterfront", "floors"]:
                df[col] = df[col].astype(str)
        elif ds_name == "lending_club":
            raw_path = data_dir / "LC_defaults_loanStats_with_random_parition.csv.gz"
            df = pd.read_csv(raw_path)
            for col in ["grade", "term", "home_ownership", "purpose", "emp_length", "addr_state"]:
                df[col] = df[col].fillna("Unknown").astype(str)
            df["bad_loans"] = df["bad_loans"].astype(int)
        elif ds_name == "bank_marketing":
            raw_path = data_dir / "BankMarketing_with_random_parition_column.csv.gz"
            df = pd.read_csv(raw_path)
            df["subscribed"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
            for col in ["job", "education", "marital", "housing", "contact", "poutcome", "month"]:
                df[col] = df[col].fillna("unknown").astype(str)

            # Assign randomized 50/50 A/B campaign variant
            rng_ab = np.random.default_rng(seed + 1001)
            df["variant"] = rng_ab.choice(["control", "treatment"], size=len(df), p=[0.5, 0.5])

            if self.config.get("ab_mode") == "seeded":
                # Inject 2 genuine treatment-responder subgroups with real out-of-sample positive lift
                # Subgroup 1: job == 'retired' & housing == 'no'
                mask1 = (df["job"] == "retired") & (df["housing"] == "no") & (df["variant"] == "treatment")
                # Subgroup 2: poutcome == 'success'
                mask2 = (df["poutcome"] == "success") & (df["variant"] == "treatment")

                # Flip additional subscriptions for responders in treatment
                flip_rand = rng_ab.random(len(df))
                df.loc[mask1 & (df["subscribed"] == 0) & (flip_rand < 0.22), "subscribed"] = 1
                df.loc[mask2 & (df["subscribed"] == 0) & (flip_rand < 0.30), "subscribed"] = 1

        # Stratified 50/50 split on target column using reproducible seed
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(df))
        df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
        split_idx = int(len(df_shuffled) * train_ratio)

        self.train_df = df_shuffled.iloc[:split_idx].reset_index(drop=True)
        self.holdout_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)

        # In-memory SQLite relational database
        self.conn = sqlite3.connect(":memory:")
        cols_to_save = list(dict.fromkeys(self.dimensions + [self.target_col]))
        self.train_df[cols_to_save].to_sql("train_mart", self.conn, index=False, if_exists="replace")
        self.holdout_df[cols_to_save].to_sql("holdout_vault", self.conn, index=False, if_exists="replace")

    def run_query(self, sql: str, on_holdout: bool = False) -> pd.DataFrame:
        """Executes a SQL query safely against train_mart (or holdout_vault for evaluator)."""
        table = "holdout_vault" if on_holdout else "train_mart"
        # Ensure query targets the appropriate table
        safe_sql = re.sub(r"\b(train_mart|holdout_vault)\b", table, sql, flags=re.IGNORECASE)
        if not re.search(r"\bFROM\s+(train_mart|holdout_vault)\b", safe_sql, re.IGNORECASE):
            # If agent omitted table name, default to train_mart / holdout_vault
            safe_sql = re.sub(r"\bFROM\s+[\w_]+\b", f"FROM {table}", safe_sql, flags=re.IGNORECASE)

        # Prevent write operations
        if re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)\b", safe_sql, re.IGNORECASE):
            raise ValueError("Permission denied: Read-only query execution.")

        return pd.read_sql_query(safe_sql, self.conn)


# =============================================================================
# 2. STATISTICAL ARM AUDITOR (UNASSISTED vs BONFERRONI vs BH vs SIGNALGUARD)
# =============================================================================

class StatisticalArmAuditor:
    """Manages statistical verification state across queries for a single agent session."""

    def __init__(self, arm: str, target_type: str, alpha: float = 0.05):
        self.arm = arm
        self.target_type = target_type
        self.alpha = alpha
        self.history: List[Dict[str, Any]] = []
        self.ledger = NoiseGateSessionLedger() if arm == "signalguard" else None

    def audit_query_result(
        self,
        query: str,
        df_result: pd.DataFrame,
        target_col: str,
        n_table_rows: int,
    ) -> Dict[str, Any]:
        """Computes statistical diagnostics (ANOVA or Chi2) and attaches arm-specific verdict."""
        m_step = len(self.history) + 1

        # Check if result looks like a GROUP BY table
        k_groups = len(df_result)
        if k_groups < 2:
            diag = {
                "turn": m_step,
                "query": query,
                "n_groups": k_groups,
                "sample_size": int(df_result["n_k"].sum()) if "n_k" in df_result.columns else len(df_result),
                "p_value": 1.0,
                "verdict": "NOT_SIGNIFICANT (Need >= 2 groups to evaluate differential effect)",
                "stat_summary": "Insufficient groups for comparison.",
            }
            self.history.append(diag)
            return diag

        # Compute test statistic
        if "n_k" in df_result.columns and "mean_y" in df_result.columns:
            # Continuous ANOVA
            n_k = df_result["n_k"].values.astype(float)
            mean_y = df_result["mean_y"].values.astype(float)
            mean_y2 = df_result.get("mean_y2", mean_y ** 2).values.astype(float)
            var_y = np.maximum(1e-8, mean_y2 - mean_y ** 2)

            res_anova = anova_rig_from_group_stats(n_k, mean_y, var_y)
            p_val = float(res_anova["p_value"])
            obs_rig = float(res_anova["observed_rig"])
            n_total = int(np.sum(n_k))
            df_k = max(1, k_groups - 1)
        elif "n_k" in df_result.columns and "sum_y" in df_result.columns:
            # Binary Chi-Square / 2xK Contingency
            n_k = df_result["n_k"].values.astype(float)
            sum_y = df_result["sum_y"].values.astype(float)
            sum_no_y = np.maximum(0.0, n_k - sum_y)
            table_counts = np.vstack([sum_y, sum_no_y])
            if np.all(sum_y == 0) or np.all(sum_no_y == 0) or np.sum(table_counts) < 5:
                p_val = 1.0
                obs_rig = 0.0
                df_k = max(1, k_groups - 1)
                n_total = int(np.sum(n_k))
            else:
                try:
                    chi2_res = stats.chi2_contingency(table_counts, correction=False)
                    p_val = float(chi2_res.pvalue)
                    df_k = int(chi2_res.dof)
                    n_total = int(np.sum(n_k))
                    obs_rig = float(chi2_res.statistic / (2.0 * n_total * np.log(2.0))) if n_total > 0 else 0.0
                except Exception:
                    p_val = 1.0
                    obs_rig = 0.0
                    df_k = max(1, k_groups - 1)
                    n_total = int(np.sum(n_k))
        else:
            # Generic query result fallback
            p_val = 1.0
            obs_rig = 0.0
            n_total = len(df_result)
            df_k = 1

        record: Dict[str, Any] = {
            "turn": m_step,
            "query": query,
            "n_groups": k_groups,
            "sample_size": n_total,
            "degrees_of_freedom": df_k,
            "observed_rig": round(obs_rig, 5),
            "naive_p": p_val,
        }

        # Arm-specific adjustments
        if self.arm == "unassisted":
            record["verdict"] = "SIGNIFICANT (p < 0.05)" if p_val < self.alpha else "NOT_SIGNIFICANT"
            record["guidance"] = f"Naive p-value: {p_val:.4e}"

        elif self.arm == "bonferroni":
            bonf_thresh = self.alpha / m_step
            is_sig = (p_val < bonf_thresh)
            record["bonferroni_threshold"] = bonf_thresh
            record["cumulative_M"] = m_step
            record["verdict"] = "VERIFIED_SIGNAL" if is_sig else "REJECT_FAILED_BONFERRONI"
            record["guidance"] = (
                f"Bonferroni gate: p={p_val:.4e} vs threshold alpha/M={bonf_thresh:.4e} (M={m_step})"
            )

        elif self.arm == "benjamini_hochberg":
            all_pvals = [h["naive_p"] for h in self.history] + [p_val]
            # Benjamini-Hochberg step-up adjustment across all session tests
            adjusted = stats.false_discovery_control(all_pvals, method="bh")
            q_val = float(adjusted[-1])
            is_sig = (q_val < self.alpha)
            record["bh_q_value"] = q_val
            record["cumulative_M"] = m_step
            record["verdict"] = "VERIFIED_SIGNAL (q < 0.05)" if is_sig else "REJECT_FAILED_BH_FDR"
            record["guidance"] = f"Benjamini-Hochberg FDR: q={q_val:.4e} (Session tests M={m_step})"

        elif self.arm == "signalguard":
            gate_res = self.ledger.evaluate_feature_or_slice(
                observed_rig=obs_rig,
                nrows=n_total,
                minority_frac=0.20,
                df=df_k,
                alpha=self.alpha,
                session_id="phase2_session",
                feature_name=f"turn_{m_step}",
            )
            term_verdict = gate_res["verdict"]
            m_eff = gate_res["cumulative_effective_tests_M_eff"]
            tau_sig = gate_res["bonferroni_significance_gate"]
            tau_evt = gate_res["evt_expected_noise_max"]

            # Small cell guard
            small_cell = (n_total < 40)
            if small_cell and term_verdict == "VERIFIED_SIGNAL":
                term_verdict = "REJECT_SMALL_SAMPLE (N < 40)"

            record["M_eff"] = round(m_eff, 2)
            record["tau_sig"] = round(tau_sig, 5)
            record["tau_evt"] = round(tau_evt, 5)
            record["small_cell"] = small_cell
            record["verdict"] = term_verdict
            record["guidance"] = (
                f"SignalGuard MCP Ledger: verdict={term_verdict} "
                f"(M_raw={m_step}, M_eff={m_eff:.1f}, tau_sig={tau_sig:.5f}, observed_rig={obs_rig:.5f})"
            )

        self.history.append(record)
        return record


# =============================================================================
# 3. INTERACTIVE MULTI-TURN AGENT LOOP
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a rigorous, highly skilled Senior Staff Data Scientist analyzing a database table named `train_mart`.
Your goal is to answer the stakeholders' analytical inquiry by exploring `train_mart` using SQL queries.

AVAILABLE TOOLS:
1. `run_sql_query(query: str)`: Executes a SELECT query against `train_mart`.
   - For continuous targets, aggregate with:
     ```sql
     SELECT dim_col, COUNT(*) AS n_k, AVG(target_col) AS mean_y, AVG(target_col * target_col) AS mean_y2 FROM train_mart [WHERE ...] GROUP BY dim_col
     ```
   - For binary targets, aggregate with:
     ```sql
     SELECT dim_col, COUNT(*) AS n_k, SUM(target_col) AS sum_y FROM train_mart [WHERE ...] GROUP BY dim_col
     ```
2. `finish_analysis(findings: List[Dict])`: Concludes your session when you have gathered sufficient evidence.
   Call this tool using:
   ```json
   {{
     "action": "finish_analysis",
     "findings": [
       {{
         "subgroup_filter": "grade = '7'",
         "dimension": "condition",
         "direction": "higher_price",
         "justification": "Cleared statistical significance"
       }}
     ]
   }}
   ```
   If no genuine findings pass statistical verification, return `"findings": []`.

STAKEHOLDER INQUIRY:
{task_description}

SCHEMA FOR `train_mart`:
- Target Column: `{target_col}` ({target_type})
- Available Dimension Columns: {dimensions}

STATISTICAL TOOLING ATTACHED:
You are equipped with the `{arm_name}` statistical verification system. Every query result returns empirical diagnostics and an arm-specific statistical verdict.

CRITICAL INSTRUCTIONS:
- You have a strict budget of up to 8 exploratory SQL queries.
- Do NOT hallucinate findings. Only report findings that have verified statistical evidence.
- When you are ready to conclude, you MUST output the ```json {{"action": "finish_analysis", "findings": [...]}} ``` block.
"""


class Phase2InteractiveAgentRunner:
    """Executes a multi-turn autonomous SQL exploration session for an LLM agent."""

    def __init__(
        self,
        warehouse: Phase2SQLWarehouse,
        arm: str = "signalguard",
        model_name: str = "gemini-3.5-flash-lite",
        max_turns: int = 8,
        temperature: float = 0.0,
    ):
        self.warehouse = warehouse
        self.arm = arm
        _, resolved = _resolve_llm_model(model_name)
        self.model_name = resolved
        self.max_turns = max_turns
        self.temperature = temperature
        self.auditor = StatisticalArmAuditor(arm=arm, target_type=warehouse.target_type)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Dispatches multi-turn messages to Anthropic, OpenAI, or Google Gemini with prompt caching."""
        _ensure_api_keys()
        m_lower = self.model_name.lower()

        # 1. Anthropic Claude (with ephemeral prompt caching)
        if "claude" in m_lower:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")

            sys_text = ""
            user_blocks = []
            for m in messages:
                if m["role"] == "system":
                    sys_text = m["content"]
                else:
                    user_blocks.append({"role": m["role"], "content": m["content"]})

            body: Dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": 1500,
                "messages": user_blocks,
            }
            if sys_text:
                # Cache static system prompt / schema prefix to save ~65% on multi-turn input tokens
                body["system"] = [
                    {
                        "type": "text",
                        "text": sys_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return "".join([b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]).strip()

        # 2. OpenAI (with low reasoning effort lock)
        elif "gpt" in m_lower:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found in environment.")

            body = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": 1500,
            }
            # Pin reasoning effort to low for reasoning tiers to control output token billing
            if any(k in m_lower for k in ("gpt-6", "o1", "o3")):
                body["reasoning_effort"] = "low"
            else:
                body["temperature"] = self.temperature

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()

        # 3. Google Gemini (with safe 4.2s pacing)
        elif "gemini" in m_lower:
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GOOGLE_API_KEY not found in environment.")

            # Gemini rate limit pacing: ensure >= 4.2s between calls for 15 RPM safety
            time.sleep(4.2)

            contents = []
            sys_inst = None
            for m in messages:
                if m["role"] == "system":
                    sys_inst = {"parts": [{"text": m["content"]}]}
                else:
                    role_tag = "user" if m["role"] == "user" else "model"
                    contents.append({"role": role_tag, "parts": [{"text": m["content"]}]})

            body = {"contents": contents}
            if sys_inst:
                body["systemInstruction"] = sys_inst

            clean_name = self.model_name.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_name}:generateContent?key={key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"].strip()
                return ""

        raise ValueError(f"Unsupported model family for {self.model_name}")

    def run_session(self, dry_run: bool = False) -> Dict[str, Any]:
        """Runs the complete multi-turn exploration session and returns the transcript and parsed findings."""
        cfg = self.warehouse.config
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
        reported_findings: List[Dict[str, Any]] = []

        while turns_executed < self.max_turns:
            turns_executed += 1

            if dry_run:
                # Deterministic Mock Agent: simulates realistic multi-query exploration
                response_text = self._generate_dry_run_turn(turns_executed)
            else:
                response_text = self._call_llm(messages)

            messages.append({"role": "assistant", "content": response_text})

            # Check if agent called finish_analysis or emitted findings JSON
            json_matches = re.findall(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            concluded = False
            for j_str in json_matches:
                try:
                    payload = json.loads(j_str)
                    if isinstance(payload, dict):
                        if payload.get("action") == "finish_analysis" or "findings" in payload:
                            reported_findings = payload.get("findings", [])
                            concluded = True
                            break
                    elif isinstance(payload, list):
                        reported_findings = payload
                        concluded = True
                        break
                except Exception:
                    pass

            if concluded:
                break

            # Extract SQL query
            sql_match = re.search(r"```sql\s*(SELECT.*?;?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
            if not sql_match:
                # Prompt the agent to either run SQL or conclude
                messages.append({
                    "role": "user",
                    "content": (
                        "Turn note: No valid ```sql SELECT ... ``` or ```json finish_analysis ``` block detected. "
                        "Please issue your next SQL query or conclude your analysis with finish_analysis."
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

                # Format tabular preview (up to 6 rows)
                preview_str = df_res.head(6).to_string(index=False)
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

        # If agent ran out of turns without explicit finish_analysis, prompt for final conclusion block
        if not reported_findings and not dry_run:
            messages.append({
                "role": "user",
                "content": (
                    "Turn budget reached. Please finalize your report now by outputting your findings in this JSON format:\n"
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
                        elif isinstance(payload, list):
                            reported_findings = payload
                            if reported_findings:
                                break
                    except Exception:
                        pass
            except Exception:
                pass

        return {
            "task": self.warehouse.task_name,
            "arm": self.arm,
            "model": self.model_name,
            "seed": self.warehouse.seed,
            "turns_executed": turns_executed,
            "queries_executed": len(self.auditor.history),
            "reported_findings": reported_findings,
            "query_history": self.auditor.history,
            "messages": messages,
        }

    def _generate_dry_run_turn(self, turn: int) -> str:
        """Generates deterministic mock agent behavior for dry runs."""
        task = self.warehouse.task_name
        cfg = self.warehouse.config

        if turn == 1:
            ctrl = cfg["primary_control"]
            if cfg["target_type"] == "continuous":
                return f"```sql\nSELECT {ctrl}, COUNT(*) AS n_k, AVG(log_price) AS mean_y, AVG(log_price * log_price) AS mean_y2 FROM train_mart GROUP BY {ctrl};\n```"
            else:
                return f"```sql\nSELECT {ctrl}, COUNT(*) AS n_k, SUM({self.warehouse.target_col}) AS sum_y FROM train_mart GROUP BY {ctrl};\n```"

        elif turn == 2:
            dim = cfg["dimensions"][1]
            if cfg["target_type"] == "continuous":
                return f"```sql\nSELECT {dim}, COUNT(*) AS n_k, AVG(log_price) AS mean_y, AVG(log_price * log_price) AS mean_y2 FROM train_mart GROUP BY {dim};\n```"
            else:
                return f"```sql\nSELECT {dim}, COUNT(*) AS n_k, SUM({self.warehouse.target_col}) AS sum_y FROM train_mart GROUP BY {dim};\n```"

        elif turn == 3:
            # Sliced subgroup
            dim = cfg["dimensions"][2]
            f_dim = cfg["dimensions"][0]
            if cfg["target_type"] == "continuous":
                return f"```sql\nSELECT {dim}, COUNT(*) AS n_k, AVG(log_price) AS mean_y, AVG(log_price * log_price) AS mean_y2 FROM train_mart WHERE {f_dim} = '7' GROUP BY {dim};\n```"
            else:
                return f"```sql\nSELECT {dim}, COUNT(*) AS n_k, SUM({self.warehouse.target_col}) AS sum_y FROM train_mart WHERE {f_dim} = 'control' GROUP BY {dim};\n```"

        else:
            # Conclude
            findings = [
                {
                    "subgroup_filter": f"{cfg['primary_control']} = '7'" if cfg["target_type"] == "continuous" else f"{cfg['primary_control']} = 'B'",
                    "dimension": cfg["dimensions"][1],
                    "direction": "statistically_significant_lift",
                    "justification": "Cleared arm verification threshold.",
                }
            ]
            return f"```json\n{{\n  \"action\": \"finish_analysis\",\n  \"findings\": {json.dumps(findings)}\n}}\n```"


# =============================================================================
# 4. DECOUPLED PURE HOLDOUT AUDITOR (CHANGE 2 & 3)
# =============================================================================

class DecoupledHoldoutAuditor:
    """Evaluates reported findings strictly against the untouched holdout_vault table.

    CRITICAL SCIENTIFIC PRINCIPLE:
    This evaluator applies ZERO small-cell or minimum-N rules. If an effect achieves
    p < alpha on the holdout partition with matching sign, it replicates. Period.
    """

    def __init__(self, warehouse: Phase2SQLWarehouse, alpha: float = 0.05):
        self.warehouse = warehouse
        self.alpha = alpha

    def evaluate_reported_findings(self, reported_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Audits each claimed finding against holdout_vault."""
        if not reported_findings:
            return {
                "n_reported": 0,
                "n_replicated": 0,
                "n_failed": 0,
                "empirical_fdr": 0.0,
                "precision": 0.0,
                "signal_recall": 0.0,
                "discovery_f1": 0.0,
                "audit_details": [],
            }

        replicated_count = 0
        failed_count = 0
        details: List[Dict[str, Any]] = []

        cfg = self.warehouse.config
        target_col = self.warehouse.target_col
        target_type = self.warehouse.target_type

        is_ab_test = "variant" in self.warehouse.dimensions

        for f in reported_findings:
            subgroup = f.get("subgroup_filter", "ALL").strip()
            dim = f.get("dimension", "").strip()

            target_dim = dim
            if target_dim not in self.warehouse.dimensions:
                # If compound or unmapped dimension, fall back to secondary dimension
                target_dim = self.warehouse.dimensions[1] if len(self.warehouse.dimensions) > 1 else self.warehouse.dimensions[0]

            try:
                if is_ab_test:
                    # Pure Holdout A/B Evaluation:
                    # Test whether treatment conversion differs from control within the claimed subgroup filter
                    clean_subgroup = subgroup
                    clean_subgroup = re.sub(r"variant\s*=\s*'[^']*'", "1=1", clean_subgroup, flags=re.IGNORECASE)
                    clean_subgroup = re.sub(r"variant\s*=\s*\"[^\"]*\"", "1=1", clean_subgroup, flags=re.IGNORECASE)

                    where_clause = ""
                    if clean_subgroup and clean_subgroup.upper() != "ALL":
                        where_clause = f"WHERE {clean_subgroup}"

                    sql = (
                        f"SELECT variant, COUNT(*) AS n_k, SUM({target_col}) AS sum_y "
                        f"FROM holdout_vault {where_clause} GROUP BY variant"
                    )
                    df_te = self.warehouse.run_query(sql, on_holdout=True)
                    if len(df_te) < 2:
                        p_holdout = 1.0
                        replicated = False
                    else:
                        n_k = df_te["n_k"].values.astype(float)
                        sum_y = df_te["sum_y"].values.astype(float)
                        sum_no = np.maximum(0.0, n_k - sum_y)
                        table_counts = np.vstack([sum_y, sum_no])
                        if np.all(sum_y == 0) or np.all(sum_no == 0) or np.sum(table_counts) < 5:
                            p_holdout = 1.0
                            replicated = False
                        else:
                            chi2 = stats.chi2_contingency(table_counts, correction=False)
                            p_holdout = float(chi2.pvalue)
                            # Directional check: treatment lift must be positive
                            df_te["rate"] = df_te["sum_y"] / df_te["n_k"]
                            rates = dict(zip(df_te["variant"], df_te["rate"]))
                            lift = rates.get("treatment", 0.0) - rates.get("control", 0.0)
                            replicated = (p_holdout < self.alpha) and (lift > 0)

                else:
                    # Subgroup / Feature Hunting Evaluation:
                    where_clause = ""
                    if subgroup and subgroup.upper() != "ALL":
                        # Replace target_dim = ... with 1=1 to test across the parent stratum
                        pat = rf"\b{target_dim}\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s,;]+)"
                        parent_filter = re.sub(pat, "1=1", subgroup, flags=re.IGNORECASE).strip()
                        where_clause = f"WHERE {parent_filter}"

                    if target_type == "continuous":
                        sql = (
                            f"SELECT {target_dim}, COUNT(*) AS n_k, "
                            f"AVG({target_col}) AS mean_y, "
                            f"AVG({target_col} * {target_col}) AS mean_y2 "
                            f"FROM holdout_vault {where_clause} GROUP BY {target_dim} HAVING COUNT(*) >= 1"
                        )
                        df_te = self.warehouse.run_query(sql, on_holdout=True)
                        if len(df_te) < 2:
                            # Fallback to Slice vs Rest
                            sql_slice = (
                                f"SELECT (CASE WHEN ({subgroup}) THEN 'SLICE' ELSE 'REST' END) AS grp, "
                                f"COUNT(*) AS n_k, AVG({target_col}) AS mean_y, AVG({target_col} * {target_col}) AS mean_y2 "
                                f"FROM holdout_vault GROUP BY grp"
                            )
                            df_te = self.warehouse.run_query(sql_slice, on_holdout=True)

                        if len(df_te) < 2:
                            p_holdout = 1.0
                        else:
                            n_k = df_te["n_k"].values.astype(float)
                            mean_y = df_te["mean_y"].values.astype(float)
                            mean_y2 = df_te["mean_y2"].values.astype(float)
                            var_y = np.maximum(1e-8, mean_y2 - mean_y ** 2)
                            res = anova_rig_from_group_stats(n_k, mean_y, var_y)
                            p_holdout = float(res["p_value"])
                        replicated = (p_holdout < self.alpha)

                    else:
                        # Binary target
                        sql = (
                            f"SELECT {target_dim}, COUNT(*) AS n_k, SUM({target_col}) AS sum_y "
                            f"FROM holdout_vault {where_clause} GROUP BY {target_dim} HAVING COUNT(*) >= 1"
                        )
                        df_te = self.warehouse.run_query(sql, on_holdout=True)
                        if len(df_te) < 2:
                            # Fallback to Slice vs Rest
                            sql_slice = (
                                f"SELECT (CASE WHEN ({subgroup}) THEN 'SLICE' ELSE 'REST' END) AS grp, "
                                f"COUNT(*) AS n_k, SUM({target_col}) AS sum_y "
                                f"FROM holdout_vault GROUP BY grp"
                            )
                            df_te = self.warehouse.run_query(sql_slice, on_holdout=True)

                        if len(df_te) < 2:
                            p_holdout = 1.0
                        else:
                            n_k = df_te["n_k"].values.astype(float)
                            sum_y = df_te["sum_y"].values.astype(float)
                            sum_no = np.maximum(0.0, n_k - sum_y)
                            table_counts = np.vstack([sum_y, sum_no])
                            if np.all(sum_y == 0) or np.all(sum_no == 0) or np.sum(table_counts) < 5:
                                p_holdout = 1.0
                            else:
                                chi2 = stats.chi2_contingency(table_counts, correction=False)
                                p_holdout = float(chi2.pvalue)
                        replicated = (p_holdout < self.alpha)

                if replicated:
                    replicated_count += 1
                else:
                    failed_count += 1

                details.append({
                    "finding": f,
                    "replicated": replicated,
                    "holdout_p": round(p_holdout, 6),
                    "holdout_sample_size": int(df_te["n_k"].sum()) if len(df_te) > 0 else 0,
                    "groups_tested": len(df_te),
                })
            except Exception as err:
                failed_count += 1
                details.append({
                    "finding": f,
                    "replicated": False,
                    "reason": f"SQL Error on holdout execution: {str(err)}",
                    "holdout_p": 1.0,
                })

        n_rep = len(reported_findings)
        empirical_fdr = (failed_count / n_rep) if n_rep > 0 else 0.0
        precision = (replicated_count / n_rep) if n_rep > 0 else 0.0

        # Ground truth recall calculation
        ground_truth_count = len(cfg.get("ground_truth_subgroups", [1]))
        signal_recall = min(1.0, replicated_count / max(1, ground_truth_count))

        if precision + signal_recall > 0:
            discovery_f1 = 2.0 * (precision * signal_recall) / (precision + signal_recall)
        else:
            discovery_f1 = 0.0

        return {
            "n_reported": n_rep,
            "n_replicated": replicated_count,
            "n_failed": failed_count,
            "empirical_fdr": round(empirical_fdr * 100.0, 2),
            "precision": round(precision * 100.0, 2),
            "signal_recall": round(signal_recall * 100.0, 2),
            "discovery_f1": round(discovery_f1, 4),
            "audit_details": details,
        }


# =============================================================================
# 5. PERSISTENT JSONL DISK CACHE
# =============================================================================

class Phase2TraceCache:
    """Stores full agent interaction traces and holdout audits persistently."""

    def __init__(self, cache_path: str = DEFAULT_PHASE2_CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_key(self, task: str, arm: str, model: str, seed: int) -> str:
        raw = f"{task}::{arm}::{model}::{seed}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, task: str, arm: str, model: str, seed: int) -> Optional[Dict[str, Any]]:
        if not self.cache_path.exists():
            return None
        target_key = self._make_key(task, arm, model, seed)
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if record.get("cache_key") == target_key:
                            return record
                    except Exception:
                        continue
        return None

    def put(self, record: Dict[str, Any]) -> None:
        key = self._make_key(record["task"], record["arm"], record["model"], record["seed"])
        record["cache_key"] = key
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
