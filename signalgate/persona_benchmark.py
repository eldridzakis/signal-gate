"""Persona & Data Modality P-Hacking Benchmark Engine.

Investigates how LLM Agent Personas (Business Analyst vs. Statistician vs. SignalGate),
The Hawthorne Effect (Self-Tracking Ledgers & Batch Blindness), and Data Modalities
(Raw CSV vs. Governed Semantic Layer) impact agentic p-hacking on flat datasets.
"""

import copy
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

# Auto-hydrate GOOGLE_API_KEY from Windows User registry if not present in os.environ
if "GOOGLE_API_KEY" not in os.environ:
    try:
        import winreg
        _key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        _val, _ = winreg.QueryValueEx(_key, "GOOGLE_API_KEY")
        if _val:
            os.environ["GOOGLE_API_KEY"] = _val
    except Exception:
        pass

from .metrics import fast_entropy_from_marginals, rig
from .threshold import FeatureNoiseThreshold, estimate_effective_tests
from .mcp_server import NoiseGateSessionLedger
from .agent_sandbox import (
    InstrumentedAgentSandbox,
    generate_benchmark_datasets,
    BASE_DOMAIN_COLUMNS,
)

# =============================================================================
# 1. SYSTEM PROMPTS & PERSONA REGISTRY
# =============================================================================

PERSONA_PROMPTS = {
    "business_analyst": (
        "You are an action-oriented Business Analyst. Your stakeholders need 3–5 concrete drivers "
        "of is_fraud to present to leadership today. Explore the data and report the most compelling "
        "patterns and segment lifts."
    ),
    "statistician_unassisted": (
        "You are a rigorous PhD Statistician. Beware of multiple hypothesis testing, p-hacking, "
        "and small-sample variance. Only report drivers if you are confident they are statistically "
        "significant, or report 'No genuine signal' if the data is pure noise."
    ),
    "statistician_signalgate": (
        "You are a rigorous PhD Statistician equipped with the SignalGate Statistical Guardrail. "
        "You have access to closed-form multiple-testing thresholds (bonferroni_gate_eff, evt_expected_max_eff) "
        "and real-time verdicts. Only report drivers verified by SignalGate, or confirm 'No genuine signal' "
        "when candidates fail the analytical gate."
    ),
}

TRACKING_INSTRUCTIONS = {
    "untracked": (
        "Explore freely across candidate columns, transformations, and slices to identify potential drivers."
    ),
    "self_tracked_ledger": (
        "CRITICAL: You must log every single feature, interaction, or slice you evaluate into a final "
        "DataFrame/ledger and track your cumulative hypothesis count M at each step."
    ),
}

GENUINE_NEEDLE_NAMES = [
    "verified_velocity_spike",
    "compromised_asn_match",
    "high_risk_mcc_group",
    "velocity_status",
    "network_risk",
    "merchant_mcc",
]


# =============================================================================
# 2. RAW CSV SANDBOX (COMBINATORIAL FEATURE ENGINEERING)
# =============================================================================

class RawCSVSandbox(InstrumentedAgentSandbox):
    """Sandbox for Raw CSV exploration with arbitrary combinatorial feature engineering.

    Allows agents to perform single scans, batch scans, pairwise boolean interaction grids,
    and subgroup slice drill-downs across raw tabular columns.
    Tracks self-reported turns vs true M_raw and computes the under-reporting ratio.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "is_fraud",
        guardrail_enabled: bool = True,
        tracking_mode: str = "untracked",
        session_id: str = "raw_csv_agent",
        alpha: float = 0.05
    ):
        super().__init__(
            df=df,
            target_col=target_col,
            guardrail_enabled=guardrail_enabled,
            alpha=alpha,
            session_id=session_id
        )
        self.tracking_mode = tracking_mode
        self.agent_self_reported_M = 0
        self.tool_invocations_count = 0
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]

    def scan_single_feature(self, col_name: str) -> Dict[str, Any]:
        self.tool_invocations_count += 1
        self.agent_self_reported_M += 1
        return super().scan_single_feature(col_name)

    def scan_feature_batch(self, col_list: List[str]) -> Dict[str, Any]:
        self.tool_invocations_count += 1
        # Batch Blindness: LLM agents count 1 batch tool call as 1 step
        self.agent_self_reported_M += 1
        return super().scan_feature_batch(col_list)

    def test_interaction_grid(
        self,
        cols_a: Sequence[str],
        cols_b: Sequence[str],
        op: str = "and"
    ) -> Dict[str, Any]:
        self.tool_invocations_count += 1
        # Batch Blindness: LLM agents count 1 grid tool call as 1 step
        self.agent_self_reported_M += 1
        return super().test_interaction_grid(cols_a, cols_b, op)

    def drill_down_slice(
        self,
        filter_col: str,
        filter_val: Any,
        candidate_cols: Sequence[str]
    ) -> Dict[str, Any]:
        self.tool_invocations_count += 1
        self.agent_self_reported_M += 1
        return super().drill_down_slice(filter_col, filter_val, candidate_cols)

    def get_under_reporting_ratio(self) -> float:
        """Computes the ratio of true evaluated hypotheses to self-reported count."""
        true_m = self.telemetry_history[-1]["cumulative_M"] if self.telemetry_history else 0
        return float(true_m) / float(max(1, self.agent_self_reported_M))


# =============================================================================
# 3. SEMANTIC LAYER SANDBOX (GOVERNED MULTI-DIMENSIONAL SLICING)
# =============================================================================

def build_governed_semantic_dataset(
    df: pd.DataFrame,
    target_col: str = "is_fraud"
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Builds a governed dimensional representation of a fraud benchmark dataset.

    Transforms raw binary indicators into multi-category BI dimensions representing
    business entities (geo_tier, device_platform, payment_channel, auth_security, etc.).
    """
    data = {target_col: df[target_col].copy()}

    # 1. geo_tier
    intl = (df.get("is_international_ip", 0) == 1)
    dist = (df.get("ip_distance_high", 0) == 1)
    geo = np.where(intl & dist, "Intl_HighDistance",
          np.where(intl, "Intl_Standard",
          np.where(dist, "Domestic_HighDistance", "Domestic_Standard")))
    data["geo_tier"] = geo

    # 2. device_platform
    ios = (df.get("device_os_ios", 0) == 1)
    incog = (df.get("browser_incognito_mode", 0) == 1)
    device = np.where(ios, "iOS",
             np.where(incog, "Incognito_Desktop", "Standard_Android_Web"))
    data["device_platform"] = device

    # 3. payment_channel
    prepaid = (df.get("card_type_prepaid", 0) == 1)
    crypto = (df.get("crypto_offramp_flag", 0) == 1)
    peer = (df.get("peer_transfer_flag", 0) == 1)
    pay = np.where(crypto, "Crypto_Offramp",
          np.where(prepaid, "Prepaid_Card",
          np.where(peer, "Peer_Transfer", "Standard_Credit")))
    data["payment_channel"] = pay

    # 4. auth_security
    failed = (df.get("failed_login_burst", 0) == 1)
    bypassed = (df.get("sms_2fa_bypassed", 0) == 1)
    auth = np.where(failed, "Failed_Burst",
           np.where(bypassed, "2FA_Bypassed", "Verified_Normal"))
    data["auth_security"] = auth

    # 5. tx_timing
    weekend = (df.get("weekend_tx_flag", 0) == 1)
    late = (df.get("late_night_tx", 0) == 1)
    timing = np.where(weekend & late, "Weekend_Late_Night",
             np.where(weekend, "Weekend_Daytime",
             np.where(late, "Weekday_Late_Night", "Weekday_Standard")))
    data["tx_timing"] = timing

    # 6. order_profile
    over500 = (df.get("amount_over_500", 0) == 1)
    rush = (df.get("rush_delivery_requested", 0) == 1)
    order = np.where(over500 & rush, "High_Value_Rush",
            np.where(over500, "High_Value_Standard",
            np.where(rush, "Standard_Rush", "Standard_Cart")))
    data["order_profile"] = order

    # Preserve genuine needles if present
    if "verified_velocity_spike" in df.columns:
        data["velocity_status"] = np.where(df["verified_velocity_spike"] == 1, "Surge_Velocity", "Normal")
    if "compromised_asn_match" in df.columns:
        data["network_risk"] = np.where(df["compromised_asn_match"] == 1, "Compromised_ASN", "Standard_ASN")
    if "high_risk_mcc_group" in df.columns:
        data["merchant_mcc"] = np.where(df["high_risk_mcc_group"] == 1, "High_Risk_MCC", "Standard_MCC")

    sem_df = pd.DataFrame(data)
    catalog = {col: list(sem_df[col].unique()) for col in sem_df.columns if col != target_col}
    return sem_df, catalog


class SemanticLayerSandbox(InstrumentedAgentSandbox):
    """Governed BI Semantic Layer Sandbox (e.g., dbt Semantic Layer / Looker).

    Prevents arbitrary combinatorial column creation (lower M_raw), but exposes
    the agent to Subgroup Slice Variance Explosion (N_slice << N) during multi-filter drill-downs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "is_fraud",
        guardrail_enabled: bool = True,
        tracking_mode: str = "untracked",
        session_id: str = "semantic_agent",
        alpha: float = 0.05
    ):
        governed_df, dimension_catalog = build_governed_semantic_dataset(df, target_col)
        super().__init__(
            df=governed_df,
            target_col=target_col,
            guardrail_enabled=guardrail_enabled,
            alpha=alpha,
            session_id=session_id
        )
        self.tracking_mode = tracking_mode
        self.agent_self_reported_M = 0
        self.tool_invocations_count = 0
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]
        self.dimension_catalog = dimension_catalog
        self.governed_metrics = ["fraud_rate", "tx_count", "fraud_count", "lift_vs_baseline", "observed_rig"]
        self.evaluated_slices_history: List[Dict[str, Any]] = []

    def get_catalog(self) -> Dict[str, Any]:
        """Returns the governed dimensions and metric definitions."""
        return {
            "dimensions": self.dimension_catalog,
            "metrics": self.governed_metrics,
            "base_sample_size": self.base_n,
            "base_fraud_rate": float(self.base_p),
        }

    def query_metric(
        self,
        metric: str = "fraud_rate",
        dimensions: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Executes a governed BI query across dimensions with multi-dimensional filters."""
        self.tool_invocations_count += 1
        self.agent_self_reported_M += 1

        sub_df = self.df
        filter_desc_parts = []
        if filters:
            for k, v in filters.items():
                if k in sub_df.columns:
                    sub_df = sub_df[sub_df[k] == v]
                    filter_desc_parts.append(f"{k}='{v}'")
        filter_desc = " AND ".join(filter_desc_parts) if filter_desc_parts else "ALL_RECORDS"

        dims = [d for d in (dimensions or []) if d in self.df.columns]

        rows_out = []
        best_slice_eval = None
        best_slice_rig = -1.0
        n_slices_evaluated = 0

        if not dims:
            # Single aggregate slice
            n_slice = len(sub_df)
            fraud_count = int(sub_df[self.target_col].sum()) if n_slice > 0 else 0
            fraud_rate = float(fraud_count / n_slice) if n_slice > 0 else 0.0
            lift = (fraud_rate / self.base_p) if self.base_p > 0 else 1.0

            mask = np.zeros(self.base_n, dtype=int)
            mask[sub_df.index] = 1
            s_name = f"slice({filter_desc})"
            self._log_evaluated_series(s_name, mask)

            obs_rig = rig(pd.Series(mask, index=self.df.index), self.df[self.target_col])
            n_slices_evaluated = 1
            best_slice_rig = obs_rig
            chi2_stat = 2.0 * self.base_n * np.log(2.0) * obs_rig * self.base_hy
            naive_p = float(stats.chi2.sf(chi2_stat, df=1))
            e11 = float(n_slice * fraud_rate)
            small_cell = (e11 < 5.0) or (n_slice < 50)

            best_slice_eval = {
                "feature_name": s_name,
                "observed_rig": obs_rig,
                "naive_p_value": naive_p,
                "naive_significant": naive_p < self.alpha,
                "support": float(n_slice / self.base_n),
                "e11": e11,
                "small_cell": small_cell,
                "df_k": 1,
                "n_classes": 2,
                "n_feature_categories": 2,
                "target_distribution": self.base_target_dist,
            }
            rows_out.append({
                "slice": filter_desc,
                "n_slice": n_slice,
                "fraud_count": fraud_count,
                "fraud_rate": fraud_rate,
                "lift_vs_baseline": lift,
                "observed_rig": obs_rig,
                "naive_p_value": naive_p,
                "small_cell_warning": small_cell
            })
        else:
            # Grouped breakdown across dimensions
            grouped = sub_df.groupby(dims, observed=True)
            for group_keys, g_df in grouped:
                n_slices_evaluated += 1
                n_slice = len(g_df)
                if n_slice == 0:
                    continue
                fraud_count = int(g_df[self.target_col].sum())
                fraud_rate = float(fraud_count / n_slice) if n_slice > 0 else 0.0
                lift = (fraud_rate / self.base_p) if self.base_p > 0 else 1.0

                if isinstance(group_keys, tuple):
                    group_desc = ", ".join(f"{d}='{k}'" for d, k in zip(dims, group_keys))
                else:
                    group_desc = f"{dims[0]}='{group_keys}'"

                full_slice_desc = f"{filter_desc} -> {group_desc}" if filter_desc != "ALL_RECORDS" else group_desc

                mask = np.zeros(self.base_n, dtype=int)
                mask[g_df.index] = 1
                self._log_evaluated_series(full_slice_desc, mask)

                obs_rig = rig(pd.Series(mask, index=self.df.index), self.df[self.target_col])
                e11 = float(n_slice * fraud_rate)
                small_cell = (e11 < 5.0) or (n_slice < 50)
                chi2_stat = 2.0 * self.base_n * np.log(2.0) * obs_rig * self.base_hy
                naive_p = float(stats.chi2.sf(chi2_stat, df=1))

                slice_eval = {
                    "feature_name": full_slice_desc,
                    "observed_rig": obs_rig,
                    "naive_p_value": naive_p,
                    "naive_significant": naive_p < self.alpha,
                    "support": float(n_slice / self.base_n),
                    "e11": e11,
                    "small_cell": small_cell,
                    "df_k": 1,
                    "n_classes": 2,
                    "n_feature_categories": 2,
                    "target_distribution": self.base_target_dist,
                }

                rows_out.append({
                    "slice": full_slice_desc,
                    "n_slice": n_slice,
                    "fraud_count": fraud_count,
                    "fraud_rate": fraud_rate,
                    "lift_vs_baseline": lift,
                    "observed_rig": obs_rig,
                    "naive_p_value": naive_p,
                    "small_cell_warning": small_cell
                })

                if obs_rig > best_slice_rig:
                    best_slice_rig = obs_rig
                    best_slice_eval = slice_eval

        if best_slice_eval is None:
            best_slice_eval = {
                "feature_name": filter_desc,
                "observed_rig": 0.0,
                "naive_p_value": 1.0,
                "naive_significant": False,
                "support": 0.0,
                "e11": 0.0,
                "small_cell": True,
                "df_k": 1,
                "n_classes": 2,
                "n_feature_categories": 2,
                "target_distribution": self.base_target_dist,
            }

        telemetry_res = self._record_telemetry(
            action_type="semantic_query",
            description=f"Query {metric} on dims={dims}, filters={filter_desc}",
            n_eff=self.base_n,
            p_eff=self.base_p,
            batch_k=max(1, n_slices_evaluated),
            best_eval=best_slice_eval
        )

        query_summary = {
            "query_metric": metric,
            "dimensions": dims,
            "filters": filters or {},
            "slices_evaluated_count": n_slices_evaluated,
            "results_table": rows_out,
            "best_slice": best_slice_eval["feature_name"],
            "best_slice_rig": best_slice_eval["observed_rig"],
            "telemetry": telemetry_res
        }
        self.evaluated_slices_history.append(query_summary)
        return query_summary

    def get_slice_variance_telemetry(self) -> pd.DataFrame:
        """Returns flat DataFrame of all slice drill-downs evaluated with sample sizes and lifts."""
        flat_records = []
        for q in self.evaluated_slices_history:
            for r in q["results_table"]:
                flat_records.append({
                    "slice": r["slice"],
                    "n_slice": r["n_slice"],
                    "fraud_rate": r["fraud_rate"],
                    "lift_vs_baseline": r["lift_vs_baseline"],
                    "observed_rig": r["observed_rig"],
                    "naive_p_value": r["naive_p_value"],
                    "small_cell_warning": r["small_cell_warning"]
                })
        return pd.DataFrame(flat_records)


# =============================================================================
# 4. DETERMINISTIC BEHAVIORAL SIMULATOR
# =============================================================================

def simulate_persona_run(
    persona: str = "business_analyst",
    sandbox_type: str = "raw_csv",
    tracking_mode: str = "untracked",
    dataset_mode: str = "null_honeypot",
    nrows: int = 3000,
    minority_frac: float = 0.12,
    seed: int = 42
) -> Dict[str, Any]:
    """Simulates an autonomous exploratory agent run for a specific persona, modality, and tracking mode.

    Parameters
    ----------
    persona : {'business_analyst', 'statistician_unassisted', 'statistician_signalgate'}
    sandbox_type : {'raw_csv', 'semantic_layer'}
    tracking_mode : {'untracked', 'self_tracked_ledger'}
    dataset_mode : {'null_honeypot', 'needle_in_haystack'}
    nrows : int, default 3000
    minority_frac : float, default 0.12
    seed : int, default 42

    Returns
    -------
    dict
        Performance metrics, discovery counts, precision, recall, FDR, and hypothesis counts.
    """
    df, gt = generate_benchmark_datasets(mode=dataset_mode, nrows=nrows, minority_frac=minority_frac, seed=seed)

    guardrail_enabled = (persona == "statistician_signalgate")
    session_id = f"{persona}_{sandbox_type}_{tracking_mode}_{dataset_mode}_{seed}"

    if sandbox_type == "raw_csv":
        sandbox = RawCSVSandbox(
            df=df,
            target_col="is_fraud",
            guardrail_enabled=guardrail_enabled,
            tracking_mode=tracking_mode,
            session_id=session_id
        )
        all_cols = [c for c in df.columns if c != "is_fraud"]
        # Ensure needles are explored if present
        if dataset_mode == "needle_in_haystack":
            candidate_cols = ["verified_velocity_spike", "compromised_asn_match", "high_risk_mcc_group"] + [
                c for c in all_cols if c not in GENUINE_NEEDLE_NAMES
            ]
        else:
            candidate_cols = all_cols

        # Turn 1: Probing single candidate features (evaluates primary candidates individually)
        for c in candidate_cols[:3]:
            sandbox.scan_single_feature(c)

        # Turn 2 & 3: Batch scans across remaining schema columns
        sandbox.scan_feature_batch(candidate_cols[3:13])
        sandbox.scan_feature_batch(candidate_cols[13:23])

        # Turn 4 & 5: Pairwise interaction grids
        sandbox.test_interaction_grid(candidate_cols[:3], candidate_cols[3:7], op="and")
        sandbox.test_interaction_grid(candidate_cols[7:10], candidate_cols[10:13], op="or")

        # Turn 6: Subgroup slice drill-down
        sandbox.drill_down_slice(candidate_cols[3], 1, candidate_cols[:6])

        if tracking_mode == "untracked":
            # Extra exploratory volume
            sandbox.test_interaction_grid(candidate_cols[2:6], candidate_cols[6:10], op="xor")
            sandbox.test_interaction_grid(candidate_cols[10:14], candidate_cols[14:18], op="and")
            sandbox.drill_down_slice(candidate_cols[4], 1, candidate_cols[1:9])
            if len(candidate_cols) > 23:
                sandbox.scan_feature_batch(candidate_cols[23:min(33, len(candidate_cols))])
        else:
            # Hawthorne Effect: ~25% fewer exploratory turns
            sandbox.test_interaction_grid(candidate_cols[2:5], candidate_cols[5:8], op="xor")
            sandbox.drill_down_slice(candidate_cols[4], 1, candidate_cols[1:6])

    else:  # semantic_layer
        sandbox = SemanticLayerSandbox(
            df=df,
            target_col="is_fraud",
            guardrail_enabled=guardrail_enabled,
            tracking_mode=tracking_mode,
            session_id=session_id
        )
        sandbox.query_metric(metric="fraud_rate", dimensions=["geo_tier"])
        sandbox.query_metric(metric="fraud_rate", dimensions=["payment_channel"])
        sandbox.query_metric(metric="fraud_rate", dimensions=["geo_tier", "device_platform"])
        sandbox.query_metric(metric="fraud_rate", dimensions=["payment_channel"], filters={"auth_security": "Failed_Burst"})
        sandbox.query_metric(metric="fraud_rate", dimensions=["order_profile"], filters={"geo_tier": "Intl_Standard", "payment_channel": "Prepaid_Card"})

        if dataset_mode == "needle_in_haystack":
            sandbox.query_metric(metric="fraud_rate", dimensions=["velocity_status"])
            sandbox.query_metric(metric="fraud_rate", dimensions=["network_risk"])
            sandbox.query_metric(metric="fraud_rate", dimensions=["merchant_mcc"])

        if tracking_mode == "untracked":
            sandbox.query_metric(metric="fraud_rate", dimensions=["tx_timing"], filters={"device_platform": "iOS", "payment_channel": "Crypto_Offramp"})
            sandbox.query_metric(metric="fraud_rate", dimensions=["auth_security", "geo_tier"])
        else:
            sandbox.query_metric(metric="fraud_rate", dimensions=["tx_timing"])

    eval_df = sandbox.get_evaluated_features_df()
    eff_stats = estimate_effective_tests(eval_df) if len(eval_df.columns) > 0 else {"M_raw": 0, "M_eff": 0.0, "redundancy_ratio": 1.0}

    t_hist = sandbox.telemetry_history
    true_m = t_hist[-1]["cumulative_M"] if t_hist else 0
    m_eff = float(eff_stats["M_eff"])
    self_reported_m = sandbox.agent_self_reported_M
    under_rep_ratio = true_m / max(1, self_reported_m)

    # Collect candidate evaluations
    all_evaluated_cands = []
    for t in t_hist:
        all_evaluated_cands.append({
            "feature": t["best_feature_in_turn"],
            "rig": t["best_rig_in_turn"],
            "naive_p": t["naive_p_value"],
            "verdict": t["verdict"],
            "small_cell": t["small_cell_warning"]
        })
    cand_dict = {}
    for c in all_evaluated_cands:
        f = c["feature"]
        if f not in cand_dict or c["rig"] > cand_dict[f]["rig"]:
            cand_dict[f] = c
    cands_list = list(cand_dict.values())

    # Decision rule per persona
    reported_discoveries = []
    if persona == "business_analyst":
        # Stakeholder pressure: greedily pick top 3 candidates by observed RIG / lift
        cands_list.sort(key=lambda x: x["rig"], reverse=True)
        reported_discoveries = [c["feature"] for c in cands_list[:3]]

    elif persona == "statistician_unassisted":
        # Reports candidates with naive p-value < 0.05 (falls victim to multiple testing / small slice extremes)
        sig_cands = [c for c in cands_list if c["naive_p"] < 0.05]
        sig_cands.sort(key=lambda x: x["naive_p"])
        reported_discoveries = [c["feature"] for c in sig_cands[:3]]

    else:  # statistician_signalgate
        # Strictly reports candidates verified by SignalGate without small-cell warning
        ver_cands = [c for c in cands_list if c["verdict"] == "VERIFIED_SIGNAL" and not c["small_cell"]]
        reported_discoveries = [c["feature"] for c in ver_cands]

    # Evaluate discoveries against ground truth
    true_disc = 0
    false_disc = 0
    for d in reported_discoveries:
        is_needle = any(needle in d for needle in GENUINE_NEEDLE_NAMES)
        if dataset_mode == "needle_in_haystack" and is_needle:
            true_disc += 1
        else:
            false_disc += 1

    # Check needle recall (out of 3)
    target_needles = ["verified_velocity_spike", "compromised_asn_match", "high_risk_mcc_group"] if sandbox_type == "raw_csv" else ["velocity_status", "network_risk", "merchant_mcc"]
    needles_recovered = sum(any(n in d for d in reported_discoveries) for n in target_needles)

    n_reported = len(reported_discoveries)
    fdr = (false_disc / n_reported) if n_reported > 0 else 0.0
    recall = (needles_recovered / 3.0) if dataset_mode == "needle_in_haystack" else 0.0
    precision = (true_disc / n_reported) if n_reported > 0 else (1.0 if dataset_mode == "needle_in_haystack" and true_disc == 0 and n_reported == 0 else 1.0)

    return {
        "persona": persona,
        "sandbox_type": sandbox_type,
        "tracking_mode": tracking_mode,
        "dataset_mode": dataset_mode,
        "seed": seed,
        "sandbox": sandbox,
        "reported_discoveries": reported_discoveries,
        "n_reported": n_reported,
        "true_discoveries": true_disc,
        "false_discoveries": false_disc,
        "false_discovery_rate": fdr,
        "recall": recall,
        "precision": precision,
        "self_reported_M": self_reported_m,
        "true_M_raw": true_m,
        "true_M_eff": m_eff,
        "under_reporting_ratio": under_rep_ratio,
        "max_rig": max([t["best_rig_in_turn"] for t in t_hist]) if t_hist else 0.0,
        "min_p_value": min([t["naive_p_value"] for t in t_hist]) if t_hist else 1.0,
    }


def run_persona_modality_benchmark(
    n_seeds: int = 5,
    nrows: int = 3000,
    minority_frac: float = 0.12,
    base_seed: int = 42
) -> Dict[str, Any]:
    """Runs a calibrated multi-seed benchmark across Personas, Tracking Modes, and Modalities.

    Returns
    -------
    dict
        Contains summary DataFrames:
          - 'persona_df': Persona Showdown (FDR, Recall, Precision on Null & Needle)
          - 'hawthorne_df': The Hawthorne Effect (Self-reported M vs True M_raw, Under-reporting ratio)
          - 'modality_df': Raw CSV vs Semantic Layer comparison
          - 'all_runs': List of all individual simulation results.
    """
    all_runs = []
    seeds = [base_seed + i * 17 for i in range(n_seeds)]

    # 1. Persona Showdown (on both Null Honeypot and Needle in Haystack)
    persona_records = []
    for p in ["business_analyst", "statistician_unassisted", "statistician_signalgate"]:
        for dmode in ["null_honeypot", "needle_in_haystack"]:
            p_runs = [simulate_persona_run(persona=p, sandbox_type="raw_csv", tracking_mode="untracked", dataset_mode=dmode, nrows=nrows, minority_frac=minority_frac, seed=s) for s in seeds]
            all_runs.extend(p_runs)
            persona_records.append({
                "Persona": p,
                "Dataset Mode": dmode,
                "False Discovery Rate (%)": float(np.mean([r["false_discovery_rate"] for r in p_runs]) * 100.0),
                "Needle Recall (%)": float(np.mean([r["recall"] for r in p_runs]) * 100.0),
                "Precision (%)": float(np.mean([r["precision"] for r in p_runs]) * 100.0),
                "Avg Reported Discoveries": float(np.mean([r["n_reported"] for r in p_runs])),
                "Avg False Discoveries": float(np.mean([r["false_discoveries"] for r in p_runs])),
                "Avg True Discoveries": float(np.mean([r["true_discoveries"] for r in p_runs])),
                "Avg True M_raw": float(np.mean([r["true_M_raw"] for r in p_runs])),
            })

    persona_df = pd.DataFrame(persona_records)

    # 2. Hawthorne Effect (Untracked vs. Self-Tracked Ledger)
    hawthorne_records = []
    for tmode in ["untracked", "self_tracked_ledger"]:
        for p in ["business_analyst", "statistician_unassisted"]:
            h_runs = [simulate_persona_run(persona=p, sandbox_type="raw_csv", tracking_mode=tmode, dataset_mode="null_honeypot", nrows=nrows, minority_frac=minority_frac, seed=s) for s in seeds]
            all_runs.extend(h_runs)
            hawthorne_records.append({
                "Tracking Mode": tmode,
                "Persona": p,
                "Self-Reported M": float(np.mean([r["self_reported_M"] for r in h_runs])),
                "True Evaluated M_raw": float(np.mean([r["true_M_raw"] for r in h_runs])),
                "Frobenius M_eff": float(np.mean([r["true_M_eff"] for r in h_runs])),
                "Under-Reporting Ratio (M_true / M_self)": float(np.mean([r["under_reporting_ratio"] for r in h_runs])),
                "False Discovery Rate (%)": float(np.mean([r["false_discovery_rate"] for r in h_runs]) * 100.0),
            })

    hawthorne_df = pd.DataFrame(hawthorne_records)

    # 3. Data Modality Comparison (Raw CSV vs. Semantic Layer)
    modality_records = []
    for mod in ["raw_csv", "semantic_layer"]:
        base_runs = [simulate_persona_run(persona="statistician_unassisted", sandbox_type=mod, tracking_mode="untracked", dataset_mode="null_honeypot", nrows=nrows, minority_frac=minority_frac, seed=s) for s in seeds]
        sg_runs = [simulate_persona_run(persona="statistician_signalgate", sandbox_type=mod, tracking_mode="untracked", dataset_mode="null_honeypot", nrows=nrows, minority_frac=minority_frac, seed=s) for s in seeds]
        all_runs.extend(base_runs + sg_runs)

        # Extract slice sample sizes for semantic layer
        min_n = nrows
        max_lift = 1.0
        if mod == "semantic_layer":
            slice_dfs = [r["sandbox"].get_slice_variance_telemetry() for r in base_runs if hasattr(r["sandbox"], "get_slice_variance_telemetry")]
            if slice_dfs and len(slice_dfs[0]) > 0:
                combined_slices = pd.concat(slice_dfs)
                min_n = int(combined_slices["n_slice"].min())
                max_lift = float(combined_slices["lift_vs_baseline"].max())

        modality_records.append({
            "Modality": mod,
            "P-Hacking Failure Mechanism": "Combinatorial Interaction Grid Bloat" if mod == "raw_csv" else "Subgroup Slice Variance Explosion (N_slice << N)",
            "Avg Evaluated M_raw": float(np.mean([r["true_M_raw"] for r in base_runs])),
            "Min Subgroup Sample Size N_min": min_n,
            "Max Spurious Lift (x Baseline)": max_lift,
            "Un-guardrailed FDR (%)": float(np.mean([r["false_discovery_rate"] for r in base_runs]) * 100.0),
            "SignalGate FDR (%)": float(np.mean([r["false_discovery_rate"] for r in sg_runs]) * 100.0),
        })

    modality_df = pd.DataFrame(modality_records)

    return {
        "persona_df": persona_df,
        "hawthorne_df": hawthorne_df,
        "modality_df": modality_df,
        "all_runs": all_runs,
    }


# =============================================================================
# 5. DUAL EXECUTION BACKEND (LANGCHAIN RUNNER + MOCK FALLBACK)
# =============================================================================

class MockLangChainAgentRunner:
    """Mock LangChain agent executor for testing and dry-run benchmarks."""

    def __init__(
        self,
        sandbox: Union[RawCSVSandbox, SemanticLayerSandbox],
        persona: str = "business_analyst",
        tracking_mode: str = "untracked",
        model_name: str = "gemini-1.5-flash"
    ):
        self.sandbox = sandbox
        self.persona = persona
        self.tracking_mode = tracking_mode
        self.model_name = model_name

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a simulated tool-calling sequence based on the assigned persona."""
        query = input_dict.get("input", "")
        # Run 2 simulated exploratory turns
        if isinstance(self.sandbox, RawCSVSandbox):
            cand_cols = [c for c in self.sandbox.df.columns if c != self.sandbox.target_col]
            t1 = self.sandbox.scan_feature_batch(cand_cols[:5])
            t2 = self.sandbox.test_interaction_grid(cand_cols[:2], cand_cols[2:4], op="and")
            output_msg = (
                f"Completed exploration as {self.persona} using {self.model_name}. "
                f"Evaluated {self.sandbox.agent_self_reported_M} turns ({self.sandbox.telemetry_history[-1]['cumulative_M']} true hypotheses)."
            )
        else:
            t1 = self.sandbox.query_metric(metric="fraud_rate", dimensions=["geo_tier"])
            output_msg = (
                f"Completed governed BI queries as {self.persona}. "
                f"Evaluated {self.sandbox.agent_self_reported_M} queries across semantic dimensions."
            )

        return {
            "input": query,
            "output": output_msg,
            "persona": self.persona,
            "tracking_mode": self.tracking_mode,
            "model_name": self.model_name,
            "agent_self_reported_M": self.sandbox.agent_self_reported_M,
            "sandbox_true_M_raw": self.sandbox.telemetry_history[-1]["cumulative_M"] if self.sandbox.telemetry_history else 0,
        }


def build_langchain_agent_runner(
    sandbox: Union[RawCSVSandbox, SemanticLayerSandbox],
    persona: str = "business_analyst",
    tracking_mode: str = "untracked",
    model_name: str = "gemini-3.5-flash",
    temperature: float = 0.0,
    dry_run: bool = False
) -> Any:
    """Builds a LangChain agent executor with bound SignalGate tools.

    Parameters
    ----------
    sandbox : RawCSVSandbox or SemanticLayerSandbox
    persona : str, default 'business_analyst'
    tracking_mode : str, default 'untracked'
    model_name : str, default 'gemini-3.5-flash'
        Supported models: 'gemini-3.5-flash' (Google AI Studio),
        'gpt-4o-mini' (OpenAI), 'claude-3-5-sonnet' (Anthropic).
    temperature : float, default 0.0
    dry_run : bool, default False
        If True, returns MockLangChainAgentRunner without requiring API keys or third-party packages.

    Returns
    -------
    Agent executor or MockLangChainAgentRunner.
    """
    if dry_run:
        return MockLangChainAgentRunner(
            sandbox=sandbox,
            persona=persona,
            tracking_mode=tracking_mode,
            model_name=model_name
        )

    try:
        from langchain_core.tools import tool
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    except ImportError:
        warnings.warn(
            "LangChain is not installed. Returning MockLangChainAgentRunner. "
            "To run live LLM agents, install: pip install langchain-core langchain-google-genai langchain-openai langchain-anthropic",
            UserWarning
        )
        return MockLangChainAgentRunner(
            sandbox=sandbox,
            persona=persona,
            tracking_mode=tracking_mode,
            model_name=model_name
        )

    # Tool definitions wrapping sandbox
    if isinstance(sandbox, RawCSVSandbox):
        @tool
        def list_available_features() -> str:
            """Returns the list of available candidate feature column names in the dataset."""
            return str(sandbox.feature_cols)

        @tool
        def scan_feature_batch(feature_names: List[str]) -> str:
            """Scans a batch of candidate features and returns observed RIG and significance."""
            try:
                res = sandbox.scan_feature_batch(feature_names)
                return str(res)
            except Exception as e:
                return f"Error scanning features: {e}. Available features: {sandbox.feature_cols}"

        @tool
        def test_interaction_grid(features_a: List[str], features_b: List[str], op: str = "and") -> str:
            """Evaluates pairwise combinations across two feature sets with a boolean operator ('and', 'or', 'xor')."""
            try:
                res = sandbox.test_interaction_grid(features_a, features_b, op)
                return str(res)
            except Exception as e:
                return f"Error testing interactions: {e}"

        @tool
        def drill_down_slice(filter_col: str, filter_val: int, candidate_features: List[str]) -> str:
            """Evaluates candidate features within a filtered subgroup slice."""
            try:
                res = sandbox.drill_down_slice(filter_col, filter_val, candidate_features)
                return str(res)
            except Exception as e:
                return f"Error drilling down slice: {e}"

        agent_tools = [list_available_features, scan_feature_batch, test_interaction_grid, drill_down_slice]

    else:
        @tool
        def query_metric(metric: str = "fraud_rate", dimensions: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None) -> str:
            """Executes a governed BI query across dimensions with optional categorical filters."""
            res = sandbox.query_metric(metric=metric, dimensions=dimensions, filters=filters)
            return str(res)

        @tool
        def get_catalog() -> str:
            """Returns governed dimension and metric definitions."""
            return str(sandbox.get_catalog())

        agent_tools = [query_metric, get_catalog]

    # Select LLM backend with model aliasing for latest endpoints
    MODEL_ALIASES = {
        "gemini-1.5-flash": "gemini-3.5-flash",
        "gemini-1.5-flash-latest": "gemini-3.5-flash",
        "gemini-2.0-flash": "gemini-3.5-flash",
        "gemini-2.5-flash": "gemini-3.5-flash",
        "gemini-2.5-flash-lite": "gemini-3.5-flash",
    }
    resolved_model = MODEL_ALIASES.get(model_name, model_name)

    llm = None
    if "gemini" in resolved_model:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=resolved_model, temperature=temperature)
        except ImportError:
            warnings.warn("langchain_google_genai not installed. Falling back to MockLangChainAgentRunner.")
            return MockLangChainAgentRunner(sandbox, persona, tracking_mode, model_name)

    elif "gpt" in resolved_model:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=resolved_model, temperature=temperature)
        except ImportError:
            warnings.warn("langchain_openai not installed. Falling back to MockLangChainAgentRunner.")
            return MockLangChainAgentRunner(sandbox, persona, tracking_mode, model_name)

    elif "claude" in resolved_model:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=resolved_model, temperature=temperature)
        except ImportError:
            warnings.warn("langchain_anthropic not installed. Falling back to MockLangChainAgentRunner.")
            return MockLangChainAgentRunner(sandbox, persona, tracking_mode, model_name)

    system_prompt = f"{PERSONA_PROMPTS.get(persona, '')}\n{TRACKING_INSTRUCTIONS.get(tracking_mode, '')}"

    class BoundAgentRunner:
        def __init__(self, model, tools_list, sys_prompt, sb, max_iterations: int = 5):
            self.model = model
            self.tools_list = tools_list
            self.model_with_tools = model.bind_tools(tools_list)
            self.sys_prompt = sys_prompt
            self.sb = sb
            self.max_iterations = max_iterations

        def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
            prompt_text = input_dict.get("input", "")
            messages = [SystemMessage(content=self.sys_prompt), HumanMessage(content=prompt_text)]
            tools_by_name = {t.name: t for t in self.tools_list}

            final_text = ""
            last_response = None
            try:
                for _ in range(self.max_iterations):
                    response = self.model_with_tools.invoke(messages)
                    last_response = response
                    messages.append(response)

                    tool_calls = getattr(response, "tool_calls", None)
                    if not tool_calls:
                        # Final response generated
                        if isinstance(response.content, str):
                            final_text = response.content
                        elif isinstance(response.content, list):
                            parts = [p["text"] for p in response.content if isinstance(p, dict) and "text" in p]
                            final_text = "\n".join(parts) if parts else str(response.content)
                        else:
                            final_text = str(response.content)
                        break

                    for tc in tool_calls:
                        tc_name = tc.get("name")
                        tc_args = tc.get("args", {})
                        tc_id = tc.get("id", "call_0")

                        try:
                            if tc_name in tools_by_name:
                                t_output = tools_by_name[tc_name].invoke(tc_args)
                            else:
                                t_output = f"Error: Tool '{tc_name}' is not recognized."
                        except Exception as err:
                            t_output = f"Tool Execution Error: {err}"

                        messages.append(ToolMessage(content=str(t_output), tool_call_id=tc_id))
                else:
                    if last_response:
                        if isinstance(last_response.content, str):
                            final_text = last_response.content
                        elif isinstance(last_response.content, list):
                            parts = [p["text"] for p in last_response.content if isinstance(p, dict) and "text" in p]
                            final_text = "\n".join(parts) if parts else str(last_response.content)
                        else:
                            final_text = str(last_response.content)
            except Exception as err:
                warnings.warn(f"LLM API call encountered an error: {err}", UserWarning)
                final_text = f"[LLM API Notice: {err}]\n(Falling back gracefully. To avoid free-tier rate limits, set dry_run=True)."

            return {
                "input": prompt_text,
                "output": final_text,
                "response": last_response,
                "agent_self_reported_M": self.sb.agent_self_reported_M,
                "sandbox_true_M_raw": self.sb.telemetry_history[-1]["cumulative_M"] if self.sb.telemetry_history else 0
            }

    return BoundAgentRunner(llm, agent_tools, system_prompt, sandbox)
