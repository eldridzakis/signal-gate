"""Instrumented Agent Sandbox & P-Hacking Benchmark Simulator.

Provides synthetic fraud/risk benchmark datasets (null_honeypot and needle_in_haystack)
and an instrumented telemetry sandbox to simulate and benchmark autonomous exploratory
data agents, quantifying false discovery rates and showing how SignalGate prevents
agentic p-hacking in real time.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from .metrics import fast_entropy_from_marginals, fast_mutual_info_from_contingency, rig
from .threshold import FeatureNoiseThreshold, estimate_effective_tests
from .mcp_server import NoiseGateSessionLedger


# Realistic fraud domain column names for synthetic generation
BASE_DOMAIN_COLUMNS = [
    ("is_international_ip", 0.15),
    ("weekend_tx_flag", 0.28),
    ("failed_login_burst", 0.05),
    ("device_os_ios", 0.45),
    ("new_beneficiary_added", 0.08),
    ("crypto_offramp_flag", 0.03),
    ("vpn_proxy_detected", 0.07),
    ("fast_checkout_used", 0.35),
    ("email_domain_disposable", 0.02),
    ("billing_shipping_mismatch", 0.12),
    ("card_type_prepaid", 0.06),
    ("amount_over_500", 0.10),
    ("late_night_tx", 0.09),
    ("browser_incognito_mode", 0.04),
    ("multiple_cards_used", 0.05),
    ("avs_check_failed", 0.08),
    ("cvv_check_failed", 0.04),
    ("device_velocity_high", 0.06),
    ("ip_distance_high", 0.14),
    ("user_age_under_25", 0.22),
    ("first_time_shopper", 0.30),
    ("rush_delivery_requested", 0.11),
    ("gift_card_purchased", 0.07),
    ("account_tenure_low", 0.18),
    ("sms_2fa_bypassed", 0.05),
    ("peer_transfer_flag", 0.09),
    ("biometric_auth_failed", 0.03),
    ("foreign_card_used", 0.06),
    ("order_frequency_spike", 0.08),
    ("high_risk_browser_timezone", 0.10),
    ("dormant_account_reactivated", 0.04),
    ("address_change_recent", 0.05),
    ("bulk_coupon_applied", 0.12),
    ("phone_carrier_voip", 0.08),
    ("multiple_shipping_addresses", 0.06),
    ("tor_exit_node_match", 0.01),
    ("virtual_machine_detected", 0.03),
    ("session_duration_under_10s", 0.15),
    ("card_expiry_under_30d", 0.05),
    ("currency_conversion_spike", 0.07),
    ("high_entropy_email_handle", 0.09),
    ("paste_used_in_card_input", 0.14),
    ("refund_history_high", 0.06),
    ("device_language_mismatch", 0.08),
    ("chargeback_dispute_flag", 0.04),
    ("sim_swap_detected", 0.02),
    ("micro_deposit_verification_failed", 0.03),
]


def generate_benchmark_datasets(
    mode: str = "null_honeypot",
    nrows: int = 3000,
    minority_frac: float = 0.12,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Generates synthetic benchmark datasets for agent evaluation.

    Parameters
    ----------
    mode : {'null_honeypot', 'needle_in_haystack'}
        'null_honeypot': Generates 30 pure-noise domain columns 100% independent of target.
        'needle_in_haystack': Generates 47 pure-noise domain columns + 3 genuine signal columns.
    nrows : int, default 3000
        Number of transaction records (N).
    minority_frac : float, default 0.12
        Fraud rate / minority class proportion (p).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        df: Generated DataFrame with features and target 'is_fraud'.
        ground_truth: Dictionary mapping each column name to ground truth metadata.
    """
    rng = np.random.default_rng(seed)

    # 1. Target generation
    n_fraud = int(round(nrows * minority_frac))
    n_clean = nrows - n_fraud
    target_vals = np.array([0] * n_clean + [1] * n_fraud, dtype=int)
    rng.shuffle(target_vals)

    data = {"is_fraud": target_vals}
    ground_truth = {}

    idx_pos = np.where(target_vals == 1)[0]
    idx_neg = np.where(target_vals == 0)[0]

    if mode == "null_honeypot":
        # 30 pure-noise domain columns (independent of target)
        cols_to_use = BASE_DOMAIN_COLUMNS[:30]
        for col_name, support in cols_to_use:
            feat_vals = (rng.random(nrows) < support).astype(int)
            data[col_name] = feat_vals
            ground_truth[col_name] = {
                "is_genuine": False,
                "population_rig": 0.0,
                "target_support": support,
            }

    elif mode == "needle_in_haystack":
        # 47 pure-noise domain columns
        for col_name, support in BASE_DOMAIN_COLUMNS:
            feat_vals = (rng.random(nrows) < support).astype(int)
            data[col_name] = feat_vals
            ground_truth[col_name] = {
                "is_genuine": False,
                "population_rig": 0.0,
                "target_support": support,
            }

        # 3 Genuine Signal Needles with known population RIGs
        # Needle 1: verified_velocity_spike (RIG ~ 0.008)
        x1 = np.zeros(nrows, dtype=int)
        x1[rng.choice(idx_pos, size=int(round(0.170 * len(idx_pos))), replace=False)] = 1
        x1[rng.choice(idx_neg, size=int(round(0.093 * len(idx_neg))), replace=False)] = 1
        data["verified_velocity_spike"] = x1
        ground_truth["verified_velocity_spike"] = {
            "is_genuine": True,
            "target_rig": 0.008,
            "description": "Short-window velocity surge",
        }

        # Needle 2: compromised_asn_match (RIG ~ 0.015)
        x2 = np.zeros(nrows, dtype=int)
        x2[rng.choice(idx_pos, size=int(round(0.190 * len(idx_pos))), replace=False)] = 1
        x2[rng.choice(idx_neg, size=int(round(0.085 * len(idx_neg))), replace=False)] = 1
        data["compromised_asn_match"] = x2
        ground_truth["compromised_asn_match"] = {
            "is_genuine": True,
            "target_rig": 0.015,
            "description": "Known high-abuse autonomous system network",
        }

        # Needle 3: high_risk_mcc_group (RIG ~ 0.030)
        x3 = np.zeros(nrows, dtype=int)
        x3[rng.choice(idx_pos, size=int(round(0.228 * len(idx_pos))), replace=False)] = 1
        x3[rng.choice(idx_neg, size=int(round(0.076 * len(idx_neg))), replace=False)] = 1
        data["high_risk_mcc_group"] = x3
        ground_truth["high_risk_mcc_group"] = {
            "is_genuine": True,
            "target_rig": 0.030,
            "description": "High-risk merchant category code group",
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'null_honeypot' or 'needle_in_haystack'.")

    df = pd.DataFrame(data)
    return df, ground_truth


class InstrumentedAgentSandbox:
    """Instrumented sandbox wrapper tracking autonomous agent exploratory testing."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "is_fraud",
        guardrail_enabled: bool = True,
        alpha: float = 0.05,
        session_id: str = "sandbox_agent"
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.guardrail_enabled = guardrail_enabled
        self.alpha = alpha
        self.session_id = session_id

        # Internal stateful hypothesis tracker
        self.ledger = NoiseGateSessionLedger()
        self.ledger.get_or_create_session(session_id=self.session_id, target_name=self.target_col)

        # Baseline sample parameters
        self.base_n = len(self.df)
        val_counts = self.df[self.target_col].value_counts(normalize=True)
        self.n_classes = len(val_counts)
        self.base_target_dist = val_counts.values
        self.base_p = float(val_counts.min()) if len(val_counts) > 1 else 0.5
        self.base_hy = fast_entropy_from_marginals(self.base_target_dist)

        # Telemetry records
        self.turn_counter = 0
        self.cumulative_best_rig = 0.0
        self.cumulative_best_feature = "none"
        self.telemetry_history: List[Dict[str, Any]] = []
        self.evaluated_series_log: Dict[str, np.ndarray] = {}

    def _log_evaluated_series(self, feature_name: str, series_vals: Union[np.ndarray, pd.Series]):
        """Logs an evaluated candidate feature series into the session driver store."""
        arr = np.asarray(series_vals, dtype=float)
        if len(arr) != self.base_n:
            padded = np.zeros(self.base_n, dtype=float)
            padded[:min(len(arr), self.base_n)] = arr[:min(len(arr), self.base_n)]
            arr = padded
        col_key = feature_name
        if col_key in self.evaluated_series_log:
            col_key = f"{feature_name}_turn{self.turn_counter}_{len(self.evaluated_series_log)}"
        self.evaluated_series_log[col_key] = arr

    def get_evaluated_features_df(self) -> pd.DataFrame:
        """Returns the full N x M_raw DataFrame of all candidate drivers evaluated during the session."""
        if not self.evaluated_series_log:
            return pd.DataFrame(index=self.df.index)
        return pd.DataFrame(self.evaluated_series_log, index=self.df.index)

    def _evaluate_series(
        self,
        series: pd.Series,
        target: pd.Series,
        n_eff: int,
        p_eff: float,
        feature_name: str
    ) -> Dict[str, Any]:
        """Calculates RIG, naive p-value, and checks SignalGate gates."""
        obs_rig = rig(series, target)

        r_cats = max(2, int(series.nunique()))
        val_counts = target.value_counts(normalize=True)
        c_classes = max(2, len(val_counts))
        k_df = (r_cats - 1) * (c_classes - 1)

        p_y = val_counts.values
        hy = fast_entropy_from_marginals(p_y)
        mi = obs_rig * hy
        deviance_stat = 2.0 * n_eff * np.log(2.0) * mi
        naive_p = float(stats.chi2.sf(deviance_stat, df=k_df))

        # Support calculation
        if r_cats == 2 and set(series.dropna().unique()).issubset({0, 1}):
            support = float(series.mean())
            e11 = float(n_eff * p_eff * support)
            small_cell = e11 < 5.0
        else:
            support = float((series.value_counts(normalize=True)).min())
            mat = pd.crosstab(series, target).values
            n_tot = mat.sum()
            p_x_vec = mat.sum(axis=1) / n_tot
            p_y_vec = mat.sum(axis=0) / n_tot
            e_matrix = n_tot * (p_x_vec[:, None] @ p_y_vec[None, :])
            min_e = float(e_matrix.min())
            frac_lt_5 = float((e_matrix < 5.0).mean())
            small_cell = (min_e < 1.0) or (frac_lt_5 > 0.20)
            e11 = min_e

        return {
            "feature_name": feature_name,
            "observed_rig": obs_rig,
            "naive_p_value": naive_p,
            "naive_significant": naive_p < self.alpha,
            "support": support,
            "e11": e11,
            "small_cell": small_cell,
            "df_k": k_df,
            "n_classes": c_classes,
            "n_feature_categories": r_cats,
            "target_distribution": p_y,
        }

    def _record_telemetry(
        self,
        action_type: str,
        description: str,
        n_eff: int,
        p_eff: float,
        batch_k: int,
        best_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Updates ledger, computes noise gates, and logs turn telemetry."""
        self.turn_counter += 1

        # Effective tests M_eff estimation from all evaluated candidate features
        eval_df = self.get_evaluated_features_df()
        if len(eval_df.columns) > 0:
            eff_stats = estimate_effective_tests(eval_df)
            m_eff = float(eff_stats["M_eff"])
            redundancy_ratio = float(eff_stats["redundancy_ratio"])
            mean_abs_corr = float(eff_stats["mean_abs_correlation"])
        else:
            m_eff = None
            redundancy_ratio = 1.0
            mean_abs_corr = 0.0

        # Evaluate against stateful ledger (increments M by batch_k and records M_eff)
        guard_res = self.ledger.evaluate_feature_or_slice(
            observed_rig=best_eval["observed_rig"],
            nrows=n_eff,
            minority_frac=p_eff,
            target_distribution=best_eval.get("target_distribution"),
            feature_support=best_eval["support"],
            n_classes=best_eval.get("n_classes", 2),
            n_feature_categories=best_eval.get("n_feature_categories", 2),
            df=best_eval.get("df_k", 1),
            hypotheses_in_batch=batch_k,
            alpha=self.alpha,
            session_id=self.session_id,
            feature_name=best_eval["feature_name"],
            slice_description=description,
            effective_M=m_eff
        )

        cum_m = guard_res["cumulative_session_tests_M"]
        if m_eff is None:
            m_eff = float(cum_m)

        # Calculate noise thresholds for raw M and effective M
        estimator = FeatureNoiseThreshold(
            nrows=n_eff,
            target_distribution=best_eval.get("target_distribution", self.base_target_dist),
            n_classes=best_eval.get("n_classes", self.n_classes),
            n_feature_categories=best_eval.get("n_feature_categories", 2),
            df=best_eval.get("df_k", 1)
        )
        tau_sig_raw = estimator.analytical_threshold(n_features_evaluated=cum_m, alpha=self.alpha)
        tau_evt_raw = estimator.evt_expected_max(n_features_evaluated=cum_m)
        tau_sig_eff = estimator.analytical_threshold(effective_features_evaluated=m_eff, alpha=self.alpha)
        tau_evt_eff = estimator.evt_expected_max(effective_features_evaluated=m_eff)

        # Verdict using correlation-adjusted M_eff
        obs_rig = best_eval["observed_rig"]
        if obs_rig >= tau_sig_eff:
            verdict = "VERIFIED_SIGNAL"
            agent_directive = (
                f"PROCEED: Feature/slice observed RIG ({obs_rig:.5f}) clears the correlation-adjusted "
                f"significance gate (tau={tau_sig_eff:.5f}, df={best_eval.get('df_k', 1)}) at effective M={m_eff:.1f} "
                f"(raw M={cum_m}, redundancy ratio={redundancy_ratio:.2f}) and alpha={self.alpha}. "
                "Statistically confirmed to be genuine signal beyond random search noise."
            )
        elif obs_rig >= tau_evt_eff:
            verdict = "BORDERLINE_WARNING"
            agent_directive = (
                f"CAUTION: Feature/slice observed RIG ({obs_rig:.5f}) clears the correlation-adjusted EVT noise ceiling "
                f"({tau_evt_eff:.5f}) at effective M={m_eff:.1f} (raw M={cum_m}), but falls short of Bonferroni/Sidak significance "
                f"(tau={tau_sig_eff:.5f}). High risk of selection bias / Winner's Curse. "
                "Do NOT promote without independent holdout cross-validation."
            )
        else:
            verdict = "REJECT_SPURIOUS_NOISE"
            agent_directive = (
                f"DISCARD: Feature/slice observed RIG ({obs_rig:.5f}) falls below the correlation-adjusted EVT expected noise ceiling "
                f"({tau_evt_eff:.5f}) at effective M={m_eff:.1f} (raw M={cum_m}). "
                "Expected from multiple testing of random noise."
            )

        if best_eval["small_cell"]:
            agent_directive += (
                f" [WARNING: Expected co-occurrence E11 / Cochran's small-cell condition triggered (min expected cell count = {best_eval['e11']:.2f} < 5). "
                "Sparse cell counts make asymptotic Chi-squared anti-conservative; run PermutationObject to verify with exact Monte-Carlo resamples.]"
            )

        if best_eval["observed_rig"] > self.cumulative_best_rig:
            self.cumulative_best_rig = best_eval["observed_rig"]
            self.cumulative_best_feature = best_eval["feature_name"]

        telemetry_entry = {
            "turn_id": self.turn_counter,
            "action_type": action_type,
            "description": description,
            "n_effective": n_eff,
            "p_effective": p_eff,
            "batch_hypotheses_k": batch_k,
            "cumulative_M": cum_m,
            "cumulative_M_eff": m_eff,
            "redundancy_ratio": redundancy_ratio,
            "mean_abs_correlation": mean_abs_corr,
            "degrees_of_freedom_k": guard_res.get("degrees_of_freedom_k", best_eval.get("df_k", 1)),
            "n_classes": guard_res.get("n_classes", 2),
            "n_feature_categories": guard_res.get("n_feature_categories", 2),
            "best_feature_in_turn": best_eval["feature_name"],
            "best_rig_in_turn": best_eval["observed_rig"],
            "cumulative_best_rig": self.cumulative_best_rig,
            "cumulative_best_feature": self.cumulative_best_feature,
            "naive_p_value": best_eval["naive_p_value"],
            "naive_significant": best_eval["naive_significant"],
            "evt_expected_max_rig": tau_evt_eff,
            "evt_expected_max_raw": tau_evt_raw,
            "bonferroni_gate_rig": tau_sig_eff,
            "bonferroni_gate_raw": tau_sig_raw,
            "bonferroni_gate_eff": tau_sig_eff,
            "small_cell_warning": best_eval["small_cell"],
            "expected_e11": best_eval["e11"],
            "verdict": verdict,
            "agent_directive": agent_directive,
        }

        self.telemetry_history.append(telemetry_entry)

        # Format user response based on operating mode
        if not self.guardrail_enabled:
            return {
                "turn": self.turn_counter,
                "best_feature": best_eval["feature_name"],
                "observed_rig": best_eval["observed_rig"],
                "naive_p_value": best_eval["naive_p_value"],
                "naive_verdict": "SIGNIFICANT (p < 0.05)" if best_eval["naive_significant"] else "NOT_SIGNIFICANT",
                "message": (
                    f"Candidate '{best_eval['feature_name']}' has RIG={best_eval['observed_rig']:.5f} "
                    f"(naive p={best_eval['naive_p_value']:.4e})."
                )
            }
        else:
            return {
                "turn": self.turn_counter,
                "best_feature": best_eval["feature_name"],
                "observed_rig": best_eval["observed_rig"],
                "cumulative_M": cum_m,
                "cumulative_M_eff": m_eff,
                "redundancy_ratio": redundancy_ratio,
                "evt_expected_max": tau_evt_eff,
                "bonferroni_gate": tau_sig_eff,
                "bonferroni_gate_raw": tau_sig_raw,
                "verdict": verdict,
                "small_cell_warning": best_eval["small_cell"],
                "agent_directive": agent_directive
            }

    def scan_single_feature(self, col_name: str) -> Dict[str, Any]:
        """Scans a single candidate feature (M += 1)."""
        self._log_evaluated_series(col_name, self.df[col_name].to_numpy())
        ev = self._evaluate_series(
            series=self.df[col_name],
            target=self.df[self.target_col],
            n_eff=self.base_n,
            p_eff=self.base_p,
            feature_name=col_name
        )
        return self._record_telemetry(
            action_type="scan_single",
            description=f"Scan single feature: {col_name}",
            n_eff=self.base_n,
            p_eff=self.base_p,
            batch_k=1,
            best_eval=ev
        )

    def scan_feature_batch(self, col_list: Sequence[str]) -> Dict[str, Any]:
        """Scans a batch of candidate features (M += len(col_list))."""
        col_list = [c for c in col_list if c in self.df.columns and c != self.target_col]
        if not col_list:
            raise ValueError("No valid columns in col_list.")

        for col in col_list:
            self._log_evaluated_series(col, self.df[col].to_numpy())

        evals = [
            self._evaluate_series(
                series=self.df[col],
                target=self.df[self.target_col],
                n_eff=self.base_n,
                p_eff=self.base_p,
                feature_name=col
            )
            for col in col_list
        ]
        best_ev = max(evals, key=lambda x: x["observed_rig"])

        return self._record_telemetry(
            action_type="scan_batch",
            description=f"Scan batch of {len(col_list)} columns",
            n_eff=self.base_n,
            p_eff=self.base_p,
            batch_k=len(col_list),
            best_eval=best_ev
        )

    def test_interaction_grid(
        self,
        cols_a: Sequence[str],
        cols_b: Sequence[str],
        op: str = "and"
    ) -> Dict[str, Any]:
        """Evaluates pairwise boolean interactions between two column sets (M += |cols_a| * |cols_b|)."""
        pairs = []
        for a in cols_a:
            for b in cols_b:
                if a != b and (b, a) not in pairs:
                    pairs.append((a, b))

        if not pairs:
            raise ValueError("No valid pairs to test.")

        evals = []
        for a, b in pairs:
            name = f"({a} {op} {b})"
            if op == "and":
                inter = (self.df[a] == 1) & (self.df[b] == 1)
            elif op == "or":
                inter = (self.df[a] == 1) | (self.df[b] == 1)
            elif op == "xor":
                inter = (self.df[a] == 1) ^ (self.df[b] == 1)
            else:
                raise ValueError(f"Unsupported op: {op}")

            self._log_evaluated_series(name, inter.astype(int).to_numpy())
            ev = self._evaluate_series(
                series=inter.astype(int),
                target=self.df[self.target_col],
                n_eff=self.base_n,
                p_eff=self.base_p,
                feature_name=name
            )
            evals.append(ev)

        best_ev = max(evals, key=lambda x: x["observed_rig"])

        return self._record_telemetry(
            action_type="interaction_grid",
            description=f"Pairwise {op} interaction grid ({len(pairs)} pairs)",
            n_eff=self.base_n,
            p_eff=self.base_p,
            batch_k=len(pairs),
            best_eval=best_ev
        )

    def drill_down_slice(
        self,
        filter_col: str,
        filter_val: Any,
        candidate_cols: Sequence[str]
    ) -> Dict[str, Any]:
        """Filters dataset to a subgroup slice and evaluates candidate features on the slice."""
        slice_mask = self.df[filter_col] == filter_val
        df_slice = self.df[slice_mask]
        n_slice = len(df_slice)

        if n_slice < 10:
            raise ValueError(f"Slice '{filter_col} == {filter_val}' contains fewer than 10 records ({n_slice}).")

        val_counts = df_slice[self.target_col].value_counts(normalize=True)
        p_slice = float(val_counts.min()) if len(val_counts) > 1 else 0.5

        candidate_cols = [c for c in candidate_cols if c in df_slice.columns and c not in [self.target_col, filter_col]]
        for col in candidate_cols:
            full_series = np.where(slice_mask, self.df[col].to_numpy(), 0)
            self._log_evaluated_series(f"[{filter_col}=={filter_val}] {col}", full_series)

        evals = [
            self._evaluate_series(
                series=df_slice[col],
                target=df_slice[self.target_col],
                n_eff=n_slice,
                p_eff=p_slice,
                feature_name=f"[{filter_col}=={filter_val}] {col}"
            )
            for col in candidate_cols
        ]
        best_ev = max(evals, key=lambda x: x["observed_rig"])

        return self._record_telemetry(
            action_type="drill_down_slice",
            description=f"Subgroup slice ({filter_col}=={filter_val}, N={n_slice}) testing {len(candidate_cols)} cols",
            n_eff=n_slice,
            p_eff=p_slice,
            batch_k=len(candidate_cols),
            best_eval=best_ev
        )

    def get_telemetry_df(self) -> pd.DataFrame:
        """Returns the full turn-by-turn telemetry log as a DataFrame."""
        return pd.DataFrame(self.telemetry_history)


def run_exploratory_agent_simulation(
    sandbox: InstrumentedAgentSandbox,
    seed: Optional[int] = 42,
    include_slices: bool = False
) -> pd.DataFrame:
    """Simulates a realistic 15-turn autonomous data science exploratory agent workflow.

    Parameters
    ----------
    sandbox : InstrumentedAgentSandbox
        Sandbox instance to execute agent actions against.
    seed : int, optional
        Random seed for simulation consistency.
    include_slices : bool, default False
        If False (default), turns 1-15 explore full-population feature engineering
        (base scans, pairwise interaction grids, and high-order mutations on N=base_n),
        producing a clean, unclipped trajectory. If True, turns 9-12 perform subgroup
        slice drilldowns where N_slice collapses.
    """
    cols = [c for c in sandbox.df.columns if c != sandbox.target_col]
    rng = np.random.default_rng(seed)

    # Turns 1–3: Initial base column scans (30 cols, M=30)
    sandbox.scan_feature_batch(cols[:10])
    sandbox.scan_feature_batch(cols[10:20])
    sandbox.scan_feature_batch(cols[20:30])

    # Turns 4–8: Pairwise interaction grids on candidate columns (185 pairs, M=215)
    sandbox.test_interaction_grid(cols[:3], cols[3:7], op="and")     # 12 pairs
    sandbox.test_interaction_grid(cols[:3], cols[3:7], op="or")      # 12 pairs
    sandbox.test_interaction_grid(cols[:5], cols[5:10], op="and")    # 25 pairs
    sandbox.test_interaction_grid(cols[:6], cols[6:12], op="xor")    # 36 pairs
    sandbox.test_interaction_grid(cols[:10], cols[10:20], op="and")  # 100 pairs

    if include_slices:
        # Turns 9–12: Multi-dimensional subgroup slice drilldowns (N_slice collapse)
        slice_col1 = cols[0]
        sandbox.drill_down_slice(filter_col=slice_col1, filter_val=1, candidate_cols=cols[1:16])  # 15 cols
        slice_col2 = cols[1]
        sandbox.drill_down_slice(filter_col=slice_col2, filter_val=1, candidate_cols=cols[2:17])  # 15 cols

        # Compounded slice (filter by slice_col1, then slice_col2)
        slice_mask = (sandbox.df[slice_col1] == 1) & (sandbox.df[cols[2]] == 1)
        if slice_mask.sum() >= 15:
            sandbox.drill_down_slice(filter_col=cols[2], filter_val=1, candidate_cols=cols[3:18])  # 15 cols
        else:
            sandbox.drill_down_slice(filter_col=cols[3], filter_val=1, candidate_cols=cols[4:19])  # 15 cols

        slice_col3 = cols[4]
        sandbox.drill_down_slice(filter_col=slice_col3, filter_val=1, candidate_cols=cols[5:25])  # 20 cols

        # Turns 13–15: Deep feature mutations and candidate sweeps
        sandbox.test_interaction_grid(cols[:10], cols[10:20], op="and")   # 100 pairs
        sandbox.test_interaction_grid(cols[:10], cols[10:20], op="or")    # 100 pairs
        sandbox.scan_feature_batch(cols[:30])                             # 30 cols
    else:
        # Turns 9–12: Feature transformations and expanded interaction grids on population (N=base_n)
        sandbox.test_interaction_grid(cols[10:15], cols[15:25], op="and") # 50 pairs
        sandbox.test_interaction_grid(cols[10:15], cols[15:25], op="or")  # 50 pairs
        sandbox.test_interaction_grid(cols[5:12], cols[12:20], op="xor")  # 56 pairs
        sandbox.test_interaction_grid(cols[:10], cols[20:28], op="and")   # 80 pairs

        # Turns 13–15: Deep candidate sweeps and batch sweeps
        sandbox.test_interaction_grid(cols[2:6], cols[20:28], op="or")    # 32 pairs
        sandbox.scan_feature_batch(cols[:20])                             # 20 cols
        sandbox.scan_feature_batch(cols[10:30])                           # 20 cols

    return sandbox.get_telemetry_df()


def run_monte_carlo_benchmark(
    n_simulations: int = 50,
    nrows: int = 3000,
    minority_frac: float = 0.12,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """Runs a Monte-Carlo benchmark comparing Baseline Agent vs SignalGate over multiple runs."""
    rng = np.random.default_rng(seed)

    # 1. Null Honeypot Benchmark (Pure Noise)
    baseline_false_alarms = 0
    signalgate_false_alarms = 0
    total_null_runs = n_simulations

    for sim_idx in range(n_simulations):
        sim_seed = int(rng.integers(1, 1000000))
        df_null, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=nrows, minority_frac=minority_frac, seed=sim_seed)

        # Baseline run
        sb_base = InstrumentedAgentSandbox(df=df_null, target_col="is_fraud", guardrail_enabled=False, session_id=f"sim_base_{sim_idx}")
        run_exploratory_agent_simulation(sb_base, seed=sim_seed)
        t_base = sb_base.get_telemetry_df()
        # Did baseline agent claim significance (p < 0.05)?
        if (t_base["naive_p_value"] < 0.05).any():
            baseline_false_alarms += 1

        # SignalGate run
        sb_gate = InstrumentedAgentSandbox(df=df_null, target_col="is_fraud", guardrail_enabled=True, session_id=f"sim_gate_{sim_idx}")
        run_exploratory_agent_simulation(sb_gate, seed=sim_seed)
        t_gate = sb_gate.get_telemetry_df()
        # Did SignalGate verify any noise as verified signal?
        if (t_gate["verdict"] == "VERIFIED_SIGNAL").any():
            signalgate_false_alarms += 1

    # 2. Needle in Haystack Benchmark
    needles = ["verified_velocity_spike", "compromised_asn_match", "high_risk_mcc_group"]
    baseline_needles_captured = {needle: 0 for needle in needles}
    signalgate_needles_captured = {needle: 0 for needle in needles}
    baseline_noise_promoted = 0
    signalgate_noise_promoted = 0

    for sim_idx in range(n_simulations):
        sim_seed = int(rng.integers(1, 1000000))
        df_needle, _ = generate_benchmark_datasets(mode="needle_in_haystack", nrows=nrows, minority_frac=minority_frac, seed=sim_seed)

        # Scan the needles and noise features
        candidate_cols = needles + [c[0] for c in BASE_DOMAIN_COLUMNS[:15]]

        # Baseline evaluation
        sb_base = InstrumentedAgentSandbox(df=df_needle, target_col="is_fraud", guardrail_enabled=False, session_id=f"needle_base_{sim_idx}")
        sb_base.scan_feature_batch(candidate_cols)
        t_base = sb_base.get_telemetry_df()

        # SignalGate evaluation with bootstrap audit
        sb_gate = InstrumentedAgentSandbox(df=df_needle, target_col="is_fraud", guardrail_enabled=True, session_id=f"needle_gate_{sim_idx}")
        audit_res = sb_gate.ledger.audit_dataframe(
            df_train=df_needle,
            target_col="is_fraud",
            candidate_cols=candidate_cols,
            n_bootstraps=30,
            seed=sim_seed
        )

        promoted_gate = audit_res["promoted_2_sigma_features"]

        for needle in needles:
            if needle in promoted_gate:
                signalgate_needles_captured[needle] += 1
            # Baseline uses raw train score > naive threshold
            r_val = rig(df_needle[needle], df_needle["is_fraud"])
            if r_val > 0.00174:
                baseline_needles_captured[needle] += 1

        # Check noise promoted
        for col in candidate_cols:
            if col not in needles:
                if col in promoted_gate:
                    signalgate_noise_promoted += 1
                if rig(df_needle[col], df_needle["is_fraud"]) > 0.00174:
                    baseline_noise_promoted += 1

    fdr_baseline = baseline_false_alarms / total_null_runs
    fdr_signalgate = signalgate_false_alarms / total_null_runs

    summary = {
        "n_simulations": n_simulations,
        "sample_size_N": nrows,
        "minority_frac": minority_frac,
        "null_honeypot": {
            "baseline_false_discovery_rate": fdr_baseline,
            "signalgate_false_discovery_rate": fdr_signalgate,
        },
        "needle_in_haystack": {
            "needles": needles,
            "baseline_recall": {k: v / n_simulations for k, v in baseline_needles_captured.items()},
            "signalgate_recall": {k: v / n_simulations for k, v in signalgate_needles_captured.items()},
            "baseline_noise_promotions_per_run": baseline_noise_promoted / n_simulations,
            "signalgate_noise_promotions_per_run": signalgate_noise_promoted / n_simulations,
        }
    }
    return summary


def simulate_agent_archetypes(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    seed: int = 42
) -> Dict[str, Any]:
    """Simulates 3 distinct autonomous agent exploratory archetypes on the same dataset.

    Each archetype evaluates exactly M_raw = 200 candidate features and logs all
    evaluated drivers in an internal DataFrame store:
      - Archetype 1: 'Broad Schema & Orthogonal Explorer' (Low collinearity, high rho)
      - Archetype 2: 'Combinatorial Interaction Grid Agent' (Moderate collinearity, medium rho)
      - Archetype 3: 'Local Threshold & Window Tweaker Agent' (Extreme collinearity, low rho)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing candidate features and target column.
    target_col : str, default 'is_fraud'
        Binary or multiclass target column name.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping archetype names ('broad_orthogonal', 'combinatorial_grid', 'threshold_tweaker')
        to performance and correlation statistics, evaluated drivers DataFrame, and sandbox.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    candidate_cols = [c for c in df.columns if c != target_col]
    if len(candidate_cols) < 5:
        raise ValueError(f"Need at least 5 candidate columns in df, got {len(candidate_cols)}.")

    target_m = 200
    results = {}

    # =========================================================================
    # Archetype 1: Broad Schema & Orthogonal Explorer
    # =========================================================================
    sb1 = InstrumentedAgentSandbox(df=df, target_col=target_col, guardrail_enabled=True, session_id=f"arch1_{seed}")
    n_base1 = min(len(candidate_cols), 40)
    cols_base1 = candidate_cols[:n_base1]
    sb1.scan_feature_batch(cols_base1)

    needed1 = target_m - n_base1
    new_cols1 = {}
    pool1 = list(candidate_cols)
    while len(new_cols1) < needed1:
        c1, c2 = rng.choice(pool1, size=2, replace=False)
        op = rng.choice(["and", "or", "xor"])
        if op == "and":
            inter = (df[c1] == 1) & (df[c2] == 1)
        elif op == "or":
            inter = (df[c1] == 1) | (df[c2] == 1)
        else:
            inter = (df[c1] == 1) ^ (df[c2] == 1)
        cname = f"({c1}_{op}_{c2})_cand{len(new_cols1)}"
        new_cols1[cname] = inter.astype(int)

    df_new1 = pd.DataFrame(new_cols1, index=sb1.df.index)
    sb1.df = pd.concat([sb1.df, df_new1], axis=1)
    sb1.scan_feature_batch(list(new_cols1.keys()))
    sb1.df.drop(columns=list(new_cols1.keys()), inplace=True)

    eval1 = sb1.get_evaluated_features_df()
    stats1 = estimate_effective_tests(eval1)
    t1 = sb1.get_telemetry_df()
    last_t1 = t1.iloc[-1]
    raw_gate1 = float(last_t1["bonferroni_gate_raw"])
    eff_gate1 = float(last_t1["bonferroni_gate_eff"])
    red1 = float(100.0 * (raw_gate1 - eff_gate1) / raw_gate1) if raw_gate1 > 0 else 0.0

    results["broad_orthogonal"] = {
        "name": "Broad Schema & Orthogonal Explorer",
        "description": "Explores diverse base schema columns and independent cross-feature combinations.",
        "sandbox": sb1,
        "evaluated_df": eval1,
        "telemetry": t1,
        "M_raw": int(stats1["M_raw"]),
        "M_eff": float(stats1["M_eff"]),
        "redundancy_ratio": float(stats1["redundancy_ratio"]),
        "mean_abs_corr": float(stats1["mean_abs_correlation"]),
        "raw_bonferroni_gate": raw_gate1,
        "eff_bonferroni_gate": eff_gate1,
        "threshold_reduction_pct": red1,
        "best_feature": str(last_t1["cumulative_best_feature"]),
        "best_rig": float(last_t1["cumulative_best_rig"]),
        "verdict": str(last_t1["verdict"]),
    }

    # =========================================================================
    # Archetype 2: Combinatorial Interaction Grid Agent
    # =========================================================================
    sb2 = InstrumentedAgentSandbox(df=df, target_col=target_col, guardrail_enabled=True, session_id=f"arch2_{seed}")
    n_parents = min(6, len(candidate_cols))
    parents2 = candidate_cols[:n_parents]
    sb2.scan_feature_batch(parents2)

    needed2 = target_m - len(sb2.evaluated_series_log)
    new_cols2 = {}
    while len(new_cols2) < needed2:
        p1, p2 = rng.choice(parents2, size=2, replace=False)
        op = rng.choice(["and", "or", "xor"])
        if op == "and":
            v = (df[p1] == 1) & (df[p2] == 1)
        elif op == "or":
            v = (df[p1] == 1) | (df[p2] == 1)
        else:
            v = (df[p1] == 1) ^ (df[p2] == 1)
        p3 = rng.choice(parents2)
        v2 = (v & (df[p3] == 1)).astype(int)
        cname = f"({p1}_{op}_{p2}&{p3})_grid{len(new_cols2)}"
        new_cols2[cname] = v2

    df_new2 = pd.DataFrame(new_cols2, index=sb2.df.index)
    sb2.df = pd.concat([sb2.df, df_new2], axis=1)
    sb2.scan_feature_batch(list(new_cols2.keys()))
    sb2.df.drop(columns=list(new_cols2.keys()), inplace=True)

    eval2 = sb2.get_evaluated_features_df()
    stats2 = estimate_effective_tests(eval2)
    t2 = sb2.get_telemetry_df()
    last_t2 = t2.iloc[-1]
    raw_gate2 = float(last_t2["bonferroni_gate_raw"])
    eff_gate2 = float(last_t2["bonferroni_gate_eff"])
    red2 = float(100.0 * (raw_gate2 - eff_gate2) / raw_gate2) if raw_gate2 > 0 else 0.0

    results["combinatorial_grid"] = {
        "name": "Combinatorial Interaction Grid Agent",
        "description": "Greedily takes top base features and tests boolean combinations sharing common parent columns.",
        "sandbox": sb2,
        "evaluated_df": eval2,
        "telemetry": t2,
        "M_raw": int(stats2["M_raw"]),
        "M_eff": float(stats2["M_eff"]),
        "redundancy_ratio": float(stats2["redundancy_ratio"]),
        "mean_abs_corr": float(stats2["mean_abs_correlation"]),
        "raw_bonferroni_gate": raw_gate2,
        "eff_bonferroni_gate": eff_gate2,
        "threshold_reduction_pct": red2,
        "best_feature": str(last_t2["cumulative_best_feature"]),
        "best_rig": float(last_t2["cumulative_best_rig"]),
        "verdict": str(last_t2["verdict"]),
    }

    # =========================================================================
    # Archetype 3: Local Threshold & Window Tweaker Agent
    # =========================================================================
    sb3 = InstrumentedAgentSandbox(df=df, target_col=target_col, guardrail_enabled=True, session_id=f"arch3_{seed}")
    score_cols = candidate_cols[:min(5, len(candidate_cols))]
    w1 = rng.uniform(0.5, 1.5, size=len(score_cols))
    w2 = rng.uniform(0.5, 1.5, size=len(score_cols))
    s1 = df[score_cols].to_numpy().dot(w1) + rng.normal(0, 0.4, size=n)
    s2 = df[score_cols].to_numpy().dot(w2) + rng.normal(0, 0.4, size=n)

    cuts1 = np.linspace(0.35, 0.65, 100)
    cuts2 = np.linspace(0.35, 0.65, 100)
    new_cols3 = {}
    for q in cuts1:
        v = (s1 > np.percentile(s1, q * 100)).astype(int)
        new_cols3[f"score1_q_{q:.3f}"] = v
    for q in cuts2:
        v = (s2 > np.percentile(s2, q * 100)).astype(int)
        new_cols3[f"score2_q_{q:.3f}"] = v

    df_new3 = pd.DataFrame(new_cols3, index=sb3.df.index)
    sb3.df = pd.concat([sb3.df, df_new3], axis=1)
    sb3.scan_feature_batch(list(new_cols3.keys()))
    sb3.df.drop(columns=list(new_cols3.keys()), inplace=True)

    eval3 = sb3.get_evaluated_features_df()
    stats3 = estimate_effective_tests(eval3)
    t3 = sb3.get_telemetry_df()
    last_t3 = t3.iloc[-1]
    raw_gate3 = float(last_t3["bonferroni_gate_raw"])
    eff_gate3 = float(last_t3["bonferroni_gate_eff"])
    red3 = float(100.0 * (raw_gate3 - eff_gate3) / raw_gate3) if raw_gate3 > 0 else 0.0

    results["threshold_tweaker"] = {
        "name": "Local Threshold & Window Tweaker Agent",
        "description": "Fixates on continuous score variants and sweeps 100+ fine-grained, overlapping threshold cuts.",
        "sandbox": sb3,
        "evaluated_df": eval3,
        "telemetry": t3,
        "M_raw": int(stats3["M_raw"]),
        "M_eff": float(stats3["M_eff"]),
        "redundancy_ratio": float(stats3["redundancy_ratio"]),
        "mean_abs_corr": float(stats3["mean_abs_correlation"]),
        "raw_bonferroni_gate": raw_gate3,
        "eff_bonferroni_gate": eff_gate3,
        "threshold_reduction_pct": red3,
        "best_feature": str(last_t3["cumulative_best_feature"]),
        "best_rig": float(last_t3["cumulative_best_rig"]),
        "verdict": str(last_t3["verdict"]),
    }

    return results
