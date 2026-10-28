"""Real-World Empirical Agentic Benchmark Engine for SignalGate.

Evaluates autonomous LLM data agents on large-scale real-world production datasets:
  1. Lending Club Loan Defaults (54,655 real loans, target: 'bad_loans')
  2. King County Housing Sales (21,610 transactions, target: 'high_value')

Ground truth is determined not by synthetic injection, but by true Out-of-Sample (OOS)
Holdout Generalization on untouched test partitions:
  - Generalization Bias (The Winner's Curse): Train_Metric - Holdout_Metric
  - Holdout Replication Rate: Fraction of promoted drivers that achieve p < 0.05 on unseen test data
  - Real False Discovery Rate (FDR): Fraction of promoted drivers that fail out-of-sample
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import fast_contingency_matrix, fast_entropy_from_marginals, fast_mutual_info_from_contingency, rig
from .threshold import FeatureNoiseThreshold, estimate_effective_tests
from .mcp_server import NoiseGateSessionLedger
from .persona_benchmark import (
    PERSONA_PROMPTS,
    STRUCTURED_REPORT_INSTRUCTION,
    TRACKING_INSTRUCTIONS,
    LLMEvalTraceCache,
    NativeGeminiModel,
    NativeClaudeModel,
    NativeOpenAIModel,
    _ensure_api_keys,
    parse_agent_discovery_report,
)

DEFAULT_REAL_WORLD_CACHE_PATH = str(Path(__file__).resolve().parent.parent / "Data" / "real_world_llm_eval_cache.jsonl")


# =============================================================================
# 1. REAL-WORLD DATASET LOADER & PREPROCESSING
# =============================================================================

def load_real_world_benchmark_dataset(
    dataset_name: str = "lending_club",
    data_dir: Optional[Union[str, Path]] = None,
    n_bins: int = 5,
    sample_train: Optional[int] = None,
    sample_test: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads and preprocesses real-world train and test partitions for agentic evaluation.

    Parameters
    ----------
    dataset_name : {'lending_club', 'king_county'}, default 'lending_club'
    data_dir : str or Path, optional
        Directory containing the dataset files. Defaults to repository Data/ folder.
    n_bins : int, default 5
        Number of quantile bins used to discretize continuous features for well-conditioned
        information-theoretic mutual information and Chi-squared degrees of freedom.
    sample_train : int, optional
        Subsample size for training set (e.g. 5000 for high-velocity simulation).
    sample_test : int, optional
        Subsample size for holdout test set.
    seed : int, default 42
        Random seed for subsampling reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
          - 'train_df': pd.DataFrame (exploration laboratory)
          - 'test_df': pd.DataFrame (untouched holdout evaluation)
          - 'target_col': str (target variable name)
          - 'candidate_features': List[str] (feature column names)
          - 'dataset_name': str
          - 'base_rate': float (prevalence of target class)
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "Data"
    else:
        data_dir = Path(data_dir)

    rng = np.random.default_rng(seed)

    if dataset_name == "lending_club":
        file_path = data_dir / "LC_defaults_loanStats_with_random_parition.csv.gz"
        if not file_path.exists():
            raise FileNotFoundError(f"Lending Club dataset not found at {file_path}")

        raw_df = pd.read_csv(file_path)
        target_col = "bad_loans"
        split_col = "splitColumn"

        # Select clean, relevant financial features
        raw_feature_cols = [
            "grade", "sub_grade", "term", "home_ownership", "purpose", "addr_state",
            "int_rate", "annual_inc", "dti", "installment", "loan_amnt", "delinq_2yrs",
            "inq_last_6mths", "open_acc", "revol_util", "revol_bal", "total_acc"
        ]

    elif dataset_name == "king_county":
        file_path = data_dir / "KingCountyHouseSales_with_random_parition_column.csv.gz"
        if not file_path.exists():
            raise FileNotFoundError(f"King County dataset not found at {file_path}")

        raw_df = pd.read_csv(file_path)
        target_col = "high_value"
        # Binarize house price: 1 if price > $650,000 (top ~25% premium tier), else 0
        raw_df[target_col] = (raw_df["price"] > 650000.0).astype(int)
        split_col = "splitColumn"

        raw_feature_cols = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_above",
            "sqft_basement", "yr_built", "yr_renovated", "zipcode"
        ]

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Choose 'lending_club' or 'king_county'.")

    # Filter columns
    cols_to_keep = [c for c in raw_feature_cols if c in raw_df.columns]
    df_clean = raw_df[[target_col, split_col] + cols_to_keep].copy()

    # Preprocess & discretize continuous features into quantile bins
    processed_feature_cols = []
    for c in cols_to_keep:
        series = df_clean[c]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
            try:
                # Quantile binning to prevent high-cardinality inflation
                binned = pd.qcut(series, q=n_bins, duplicates="drop")
                bin_col = f"{c}_bin"
                df_clean[bin_col] = binned.astype(str)
                processed_feature_cols.append(bin_col)
            except Exception:
                # Fallback to string categories if qcut fails
                df_clean[c] = series.fillna("Missing").astype(str)
                processed_feature_cols.append(c)
        else:
            df_clean[c] = series.fillna("Missing").astype(str)
            processed_feature_cols.append(c)

    # Split into Train and Test partitions using the dataset's native partition column
    train_df = df_clean[df_clean[split_col] == "Train"].drop(columns=[split_col]).reset_index(drop=True)
    test_df = df_clean[df_clean[split_col] == "Test"].drop(columns=[split_col]).reset_index(drop=True)

    # Optional subsampling for test velocity
    if sample_train is not None and len(train_df) > sample_train:
        idx_tr = rng.choice(len(train_df), size=sample_train, replace=False)
        train_df = train_df.iloc[idx_tr].reset_index(drop=True)

    if sample_test is not None and len(test_df) > sample_test:
        idx_te = rng.choice(len(test_df), size=sample_test, replace=False)
        test_df = test_df.iloc[idx_te].reset_index(drop=True)

    base_rate = float(train_df[target_col].mean())

    return {
        "train_df": train_df,
        "test_df": test_df,
        "target_col": target_col,
        "candidate_features": processed_feature_cols,
        "dataset_name": dataset_name,
        "base_rate": base_rate,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


# =============================================================================
# 2. REAL-WORLD AGENT SANDBOX (TRAINING EXPLORATION ONLY)
# =============================================================================

class RealWorldAgentSandbox:
    """Instrumented exploratory laboratory wrapping real-world training partitions.

    Guarantees strict data isolation: the agent only explores df_train.
    Holdout evaluation occurs post-hoc on df_test.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        candidate_features: Sequence[str],
        guardrail_enabled: bool = False,
        tracking_mode: str = "untracked",
        session_id: str = "rw_sandbox",
        alpha: float = 0.05,
    ):
        self.df = train_df.copy()
        self.target_col = target_col
        self.feature_cols = list(candidate_features)
        self.guardrail_enabled = guardrail_enabled
        self.tracking_mode = tracking_mode
        self.session_id = session_id
        self.alpha = alpha

        self.ledger = NoiseGateSessionLedger() if guardrail_enabled else None
        self.agent_self_reported_M = 0
        self.telemetry_history: List[Dict[str, Any]] = []
        self._evaluated_feature_series: Dict[str, pd.Series] = {}

    def _evaluate_candidate_series(self, name: str, series: pd.Series) -> Dict[str, Any]:
        """Calculates observed Relative Information Gain (RIG), Wilks Chi2 p-value, and SignalGate verdict."""
        self._evaluated_feature_series[name] = series
        n = len(series)
        y = self.df.loc[series.index, self.target_col]

        observed_rig = float(rig(series, y))

        # Contingency & degrees of freedom
        contingency = fast_contingency_matrix(series.values, y.values)
        r, c = contingency.shape
        df_k = max(1, (r - 1) * (c - 1))

        # Asymptotic Chi-squared G-statistic under Wilks' theorem
        g_stat = 2.0 * n * np.log(2.0) * (observed_rig * fast_entropy_from_marginals(y.value_counts(normalize=True).values))
        naive_p = float(stats.chi2.sf(max(0.0, g_stat), df=df_k))

        # Expected marginal cells check
        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / float(n)
        min_expected = float(expected.min()) if expected.size > 0 else 0.0
        small_cell = min_expected < 5.0

        res: Dict[str, Any] = {
            "feature": name,
            "observed_rig": round(observed_rig, 5),
            "naive_p": naive_p,
            "degrees_of_freedom": df_k,
            "small_cell_warning": small_cell,
            "sample_size": n,
        }

        if self.guardrail_enabled and self.ledger is not None:
            # Audit against stateful ledger with Frobenius M_eff tracking
            base_rate = float(self.df[self.target_col].mean())
            gate_res = self.ledger.evaluate_feature_or_slice(
                observed_rig=observed_rig,
                nrows=n,
                minority_frac=base_rate,
                n_feature_categories=r,
                n_classes=c,
                alpha=self.alpha,
                session_id=self.session_id,
                feature_name=name,
            )
            res.update({
                "M_raw": gate_res["cumulative_session_tests_M"],
                "M_eff": gate_res["cumulative_effective_tests_M_eff"],
                "tau_evt": gate_res["evt_expected_noise_max"],
                "tau_sig": gate_res["bonferroni_significance_gate"],
                "verdict": gate_res["verdict"],
            })
        else:
            res.update({
                "verdict": "SIGNIFICANT (p < 0.05)" if naive_p < 0.05 else "NOT_SIGNIFICANT"
            })

        return res

    def scan_single_feature(self, col: str) -> Dict[str, Any]:
        """Scans an individual column in the dataset."""
        self.agent_self_reported_M += 1
        res = self._evaluate_candidate_series(col, self.df[col])
        res["turn_type"] = "scan_single"
        res["turn_label"] = f"Scan single feature: {col}"
        res["best_feature_in_turn"] = col
        res["cumulative_M"] = res.get("M_raw", len(self._evaluated_feature_series))
        self.telemetry_history.append(res)
        return res

    def scan_feature_batch(self, cols: Sequence[str]) -> Dict[str, Any]:
        """Evaluates a batch of candidate features and logs each tested hypothesis."""
        self.agent_self_reported_M += 1
        batch_results = []
        for c in cols:
            if c in self.df.columns:
                batch_results.append(self._evaluate_candidate_series(c, self.df[c]))

        batch_results.sort(key=lambda x: x["observed_rig"], reverse=True)
        best = batch_results[0] if batch_results else {"feature": "none", "observed_rig": 0.0, "naive_p": 1.0, "verdict": "NOT_SIGNIFICANT"}

        record = {
            "turn_type": "scan_batch",
            "turn_label": f"Scan feature batch ({len(cols)} columns)",
            "blind_turn_label": "Scan feature batch",
            "batch_size": len(cols),
            "best_feature_in_turn": best["feature"],
            "observed_rig": best["observed_rig"],
            "naive_p": best["naive_p"],
            "verdict": best["verdict"],
            "cumulative_M": best.get("M_raw", len(self._evaluated_feature_series)),
            "M_eff": best.get("M_eff", len(self._evaluated_feature_series)),
            "tau_sig": best.get("tau_sig", None),
            "tau_evt": best.get("tau_evt", None),
            "sample_size": len(self.df),
            "top_candidates": [b["feature"] for b in batch_results[:3]],
        }
        self.telemetry_history.append(record)
        return record

    def test_interaction_grid(self, cols_a: Sequence[str], cols_b: Sequence[str], op: str = "and") -> Dict[str, Any]:
        """Evaluates pairwise combinations across two feature sets with a logical operator ('and', 'or', 'xor')."""
        self.agent_self_reported_M += 1
        grid_results = []
        n_pairs = 0

        for ca in cols_a:
            for cb in cols_b:
                if ca in self.df.columns and cb in self.df.columns and ca != cb:
                    n_pairs += 1
                    s_a = self.df[ca].astype(str)
                    s_b = self.df[cb].astype(str)
                    comb_name = f"({ca} {op} {cb})"
                    comb_series = s_a + f"_{op}_" + s_b
                    grid_results.append(self._evaluate_candidate_series(comb_name, comb_series))

        grid_results.sort(key=lambda x: x["observed_rig"], reverse=True)
        best = grid_results[0] if grid_results else {"feature": "none", "observed_rig": 0.0, "naive_p": 1.0, "verdict": "NOT_SIGNIFICANT"}

        record = {
            "turn_type": "interaction_grid",
            "turn_label": f"Pairwise {op} interaction grid ({n_pairs} pairs)",
            "blind_turn_label": f"Pairwise {op} interaction grid",
            "grid_size": n_pairs,
            "best_feature_in_turn": best["feature"],
            "observed_rig": best["observed_rig"],
            "naive_p": best["naive_p"],
            "verdict": best["verdict"],
            "cumulative_M": best.get("M_raw", len(self._evaluated_feature_series)),
            "M_eff": best.get("M_eff", len(self._evaluated_feature_series)),
            "tau_sig": best.get("tau_sig", None),
            "tau_evt": best.get("tau_evt", None),
            "sample_size": len(self.df),
            "top_candidates": [g["feature"] for g in grid_results[:3]],
        }
        self.telemetry_history.append(record)
        return record

    def drill_down_slice(self, filter_col: str, filter_val: Any, candidate_features: Sequence[str]) -> Dict[str, Any]:
        """Evaluates candidate features within a filtered subgroup slice."""
        self.agent_self_reported_M += 1
        mask = self.df[filter_col].astype(str) == str(filter_val)
        slice_df = self.df[mask]
        n_slice = len(slice_df)

        slice_results = []
        if n_slice >= 10:
            for c in candidate_features:
                if c in slice_df.columns and c != filter_col:
                    s_name = f"[{filter_col}=={filter_val}] {c}"
                    slice_results.append(self._evaluate_candidate_series(s_name, slice_df[c]))

        slice_results.sort(key=lambda x: x["observed_rig"], reverse=True)
        best = slice_results[0] if slice_results else {"feature": f"[{filter_col}=={filter_val}] none", "observed_rig": 0.0, "naive_p": 1.0, "verdict": "NOT_SIGNIFICANT"}

        record = {
            "turn_type": "slice_drill_down",
            "turn_label": f"Subgroup slice ({filter_col}=={filter_val}, N={n_slice}) testing {len(candidate_features)} cols",
            "blind_turn_label": f"Subgroup slice ({filter_col}=={filter_val}) drill-down",
            "best_feature_in_turn": best["feature"],
            "observed_rig": best["observed_rig"],
            "naive_p": best["naive_p"],
            "verdict": best["verdict"],
            "cumulative_M": best.get("M_raw", len(self._evaluated_feature_series)),
            "M_eff": best.get("M_eff", len(self._evaluated_feature_series)),
            "tau_sig": best.get("tau_sig", None),
            "tau_evt": best.get("tau_evt", None),
            "sample_size": n_slice,
        }
        self.telemetry_history.append(record)
        return record

    def get_evaluated_features_df(self) -> pd.DataFrame:
        """Returns the full candidate feature matrix evaluated during exploration."""
        if not self._evaluated_feature_series:
            return pd.DataFrame(index=self.df.index)
        return pd.DataFrame(self._evaluated_feature_series, index=self.df.index)


# =============================================================================
# 3. TRAJECTORY GENERATOR (POPULATING REALISTIC AGENT LOGS)
# =============================================================================

def populate_real_world_agent_trajectory(
    sandbox: RealWorldAgentSandbox,
    blind_unassisted_counts: bool = True,
) -> str:
    """Populates a realistic exploratory data science trajectory on the training dataset.

    Executes:
      - 3 single feature scans
      - 2 batch scans (~20 candidate columns)
      - 2 interaction grids (~24 feature pairs)
      - 2 subgroup slice drill-downs (e.g. within specific risk grades or states)
    """
    cands = sandbox.feature_cols
    if len(cands) < 10:
        raise ValueError(f"Need at least 10 candidate features, got {len(cands)}.")

    # 1. Single feature scans
    sandbox.scan_single_feature(cands[0])
    sandbox.scan_single_feature(cands[1])
    sandbox.scan_single_feature(cands[2])

    # 2. Batch feature scans
    sandbox.scan_feature_batch(cands[3:min(11, len(cands))])
    sandbox.scan_feature_batch(cands[min(11, len(cands)):min(19, len(cands))])

    # 3. Interaction grids
    sandbox.test_interaction_grid(cands[:2], cands[2:6], op="and")
    sandbox.test_interaction_grid(cands[6:9], cands[9:12], op="or")

    # 4. Subgroup slice drill-downs
    sample_val = str(sandbox.df[cands[0]].iloc[0])
    sandbox.drill_down_slice(cands[0], sample_val, cands[1:6])
    sample_val_2 = str(sandbox.df[cands[1]].iloc[0])
    sandbox.drill_down_slice(cands[1], sample_val_2, cands[2:7])

    # Build prompt transcript
    is_unassisted = not sandbox.guardrail_enabled
    lines = [
        f"Real-World Dataset Summary: N={len(sandbox.df)} training records, target='{sandbox.target_col}'.",
        f"Completed Exploration Log ({len(sandbox.telemetry_history)} tool turns executed):"
    ]

    for idx, t in enumerate(sandbox.telemetry_history, 1):
        if is_unassisted and blind_unassisted_counts:
            label = t.get("blind_turn_label", t["turn_label"])
        else:
            label = t["turn_label"]

        line = (
            f"  Turn {idx} [{label} | N={t['sample_size']}]: "
            f"best_candidate='{t['best_feature_in_turn']}', "
            f"observed_rig={t['observed_rig']:.5f}, naive_p={t['naive_p']:.4e}"
        )

        if not is_unassisted:
            # Show SignalGate stateful ledger metrics
            m_raw = t.get("cumulative_M", 1)
            m_eff = float(t.get("M_eff") or 1.0)
            tau_evt = float(t.get("tau_evt") or 0.0)
            tau_sig = float(t.get("tau_sig") or 0.0)
            v = t.get("verdict", "REJECT_SPURIOUS_NOISE")
            line += f" | SignalGate(M_raw={m_raw}, M_eff={m_eff:.1f}, tau_evt={tau_evt:.5f}, tau_sig={tau_sig:.5f}) -> verdict={v}"
        else:
            line += f" ({t['verdict']})"

        lines.append(line)

    lines.append(
        "\nBased on the exploration log above, provide your final analysis of the drivers of "
        f"'{sandbox.target_col}' and conclude with the required ```json ... ``` summary block."
    )
    return "\n".join(lines)


# =============================================================================
# 4. HOLDOUT REPLICATION AUDIT (TRUE GENERALIZATION BENCHMARK)
# =============================================================================

def evaluate_real_world_holdout_replication(
    reported_discoveries: Sequence[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    self_reported_M: Optional[int] = None,
    true_M_raw: int = 1,
) -> Dict[str, Any]:
    """Audits candidate discoveries against the untouched holdout test partition.

    Quantifies:
      - Train RIG vs Holdout RIG
      - Generalization Bias (The Winner's Curse): Train_RIG - Holdout_RIG
      - Holdout Statistical Replication Rate: Fraction of features with p_test < 0.05
      - Real False Discovery Rate (FDR): Fraction of promoted features that fail on test
      - Metacognitive Under-Reporting Ratio: True_M_raw / Self_Reported_M
    """
    if not reported_discoveries:
        under_rep = float(true_M_raw) / float(max(1, self_reported_M or 1))
        return {
            "n_reported": 0,
            "replicated_count": 0,
            "failed_count": 0,
            "replication_rate": 1.0,  # Conservatism / correct rejection of pure noise
            "real_fdr": 0.0,
            "mean_train_rig": 0.0,
            "mean_holdout_rig": 0.0,
            "mean_bias": 0.0,
            "mean_decay_pct": 0.0,
            "under_reporting_ratio": under_rep,
            "audit_table": [],
        }

    audit_table = []
    n_replicated = 0
    n_failed = 0

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    for feat in reported_discoveries:
        # Reconstruct series from training and test frames (handles single, interaction, or slices)
        clean_name = feat.strip()
        s_train = None
        s_test = None
        y_tr_sub = None
        y_te_sub = None

        if " and " in clean_name or " or " in clean_name or " xor " in clean_name:
            # Pairwise interaction
            m = re.match(r"\((.*?)\s+(and|or|xor)\s+(.*?)\)", clean_name)
            if m:
                ca, op, cb = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                if ca in train_df.columns and cb in train_df.columns and ca in test_df.columns and cb in test_df.columns:
                    s_train = train_df[ca].astype(str) + f"_{op}_" + train_df[cb].astype(str)
                    s_test = test_df[ca].astype(str) + f"_{op}_" + test_df[cb].astype(str)
        elif clean_name.startswith("[") and "==" in clean_name and "]" in clean_name:
            # Subgroup slice (supports interval brackets e.g. [col==(a, b]] sub_feat)
            m = re.match(r"^\[(.*?)\s*==\s*(.*)\]\s*([\w_]+)$", clean_name)
            if not m:
                m = re.match(r"\[(.*?)\s*==\s*(.*?)\]\s*(.*)", clean_name)
            if m:
                f_col, f_val, sub_feat = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                if f_col in train_df.columns and sub_feat in train_df.columns and f_col in test_df.columns and sub_feat in test_df.columns:
                    mask_tr = train_df[f_col].astype(str) == str(f_val)
                    mask_te = test_df[f_col].astype(str) == str(f_val)
                    if mask_te.sum() >= 5 and mask_tr.sum() >= 5:
                        s_train = train_df.loc[mask_tr, sub_feat]
                        s_test = test_df.loc[mask_te, sub_feat]
                        y_tr_sub = train_df.loc[mask_tr, target_col]
                        y_te_sub = test_df.loc[mask_te, target_col]
        else:
            # Single column
            # Strip potential comments or descriptions appended by LLM (e.g. "int_rate: strong predictor")
            pure_col = clean_name.split(":")[0].split(" ")[0].strip("'`\"")
            if pure_col in train_df.columns and pure_col in test_df.columns:
                s_train = train_df[pure_col]
                s_test = test_df[pure_col]

        if s_train is not None and s_test is not None:
            tr_y = y_tr_sub if y_tr_sub is not None else y_train
            te_y = y_te_sub if y_te_sub is not None else y_test

            tr_rig = float(rig(s_train, tr_y))
            te_rig = float(rig(s_test, te_y))

            # Test-set p-value
            cont_te = fast_contingency_matrix(s_test.values, te_y.values)
            r, c = cont_te.shape
            df_k = max(1, (r - 1) * (c - 1))
            g_stat = 2.0 * len(s_test) * np.log(2.0) * (te_rig * fast_entropy_from_marginals(te_y.value_counts(normalize=True).values))
            test_p = float(stats.chi2.sf(max(0.0, g_stat), df=df_k))

            # Check holdout replication
            bias = tr_rig - te_rig
            decay_pct = (bias / tr_rig) * 100.0 if tr_rig > 1e-6 else 0.0
            replicated = (test_p < 0.05) and (te_rig > 0.001)

            if replicated:
                n_replicated += 1
            else:
                n_failed += 1

            audit_table.append({
                "feature": clean_name,
                "train_rig": round(tr_rig, 5),
                "holdout_rig": round(te_rig, 5),
                "bias": round(bias, 5),
                "decay_pct": round(decay_pct, 1),
                "holdout_p_value": test_p,
                "replicated_on_holdout": replicated,
            })
        else:
            # Hallucinated or non-matching feature
            n_failed += 1
            audit_table.append({
                "feature": clean_name,
                "train_rig": 0.0,
                "holdout_rig": 0.0,
                "bias": 0.0,
                "decay_pct": 100.0,
                "holdout_p_value": 1.0,
                "replicated_on_holdout": False,
            })

    total_promoted = len(audit_table)
    rep_rate = (n_replicated / total_promoted) if total_promoted > 0 else 0.0
    real_fdr = (n_failed / total_promoted) if total_promoted > 0 else 0.0

    mean_tr = float(np.mean([a["train_rig"] for a in audit_table])) if audit_table else 0.0
    mean_te = float(np.mean([a["holdout_rig"] for a in audit_table])) if audit_table else 0.0
    mean_bias = float(np.mean([a["bias"] for a in audit_table])) if audit_table else 0.0
    mean_decay = float(np.mean([a["decay_pct"] for a in audit_table])) if audit_table else 0.0

    under_rep = float(true_M_raw) / float(max(1, self_reported_M or 1))

    return {
        "n_reported": total_promoted,
        "replicated_count": n_replicated,
        "failed_count": n_failed,
        "replication_rate": rep_rate,
        "real_fdr": real_fdr,
        "mean_train_rig": mean_tr,
        "mean_holdout_rig": mean_te,
        "mean_bias": mean_bias,
        "mean_decay_pct": mean_decay,
        "under_reporting_ratio": under_rep,
        "audit_table": audit_table,
    }


# =============================================================================
# 5. FRONTIER LLM EVALUATOR (LIVE & CACHED)
# =============================================================================

def run_real_world_frontier_eval(
    dataset_name: str = "lending_club",
    persona: str = "business_analyst",
    model_name: str = "gemini-3.5-flash-lite",
    seed: int = 42,
    sample_train: int = 5000,
    sample_test: int = 2500,
    temperature: float = 0.0,
    dry_run: bool = False,
    use_cache: bool = True,
    cache_path: str = DEFAULT_REAL_WORLD_CACHE_PATH,
    blind_unassisted_counts: bool = True,
) -> Dict[str, Any]:
    """Runs an empirical frontier LLM evaluation on real-world datasets with holdout auditing."""
    _ensure_api_keys()

    # Load dataset
    ds = load_real_world_benchmark_dataset(
        dataset_name=dataset_name,
        sample_train=sample_train,
        sample_test=sample_test,
        seed=seed,
    )
    train_df = ds["train_df"]
    test_df = ds["test_df"]
    target_col = ds["target_col"]
    candidate_features = ds["candidate_features"]

    # Initialize sandbox
    guardrail_enabled = (persona == "statistician_signalgate")
    tracking_mode = "signalgate_ledger" if guardrail_enabled else "untracked"

    sandbox = RealWorldAgentSandbox(
        train_df=train_df,
        target_col=target_col,
        candidate_features=candidate_features,
        guardrail_enabled=guardrail_enabled,
        tracking_mode=tracking_mode,
        session_id=f"rw_{dataset_name}_{seed}",
    )

    prompt_transcript = populate_real_world_agent_trajectory(
        sandbox=sandbox,
        blind_unassisted_counts=blind_unassisted_counts,
    )

    system_prompt = (
        f"{PERSONA_PROMPTS.get(persona, '')}\n"
        f"{TRACKING_INSTRUCTIONS.get(tracking_mode, '')}"
        f"{STRUCTURED_REPORT_INSTRUCTION}"
    )

    trace_cache = LLMEvalTraceCache(cache_path=cache_path) if use_cache else None
    cache_key = (
        trace_cache.compute_key(
            model_name=model_name,
            temperature=temperature,
            system_prompt=system_prompt,
            messages_payload=prompt_transcript,
        )
        if trace_cache is not None
        else None
    )

    output_text = ""
    cache_hit = False

    # 1. Check cache
    if trace_cache is not None and cache_key is not None:
        cached = trace_cache.get(cache_key)
        if cached is not None and (dry_run or not cached.get("dry_run", False)):
            output_text = cached.get("output", "")
            cache_hit = True

    # 2. Execute live LLM call if not cached
    if not cache_hit:
        if dry_run:
            # Deterministic mock report for unit testing
            if persona == "business_analyst":
                best_cands = [t["best_feature_in_turn"] for t in sandbox.telemetry_history[:3]]
                output_text = f"Top drivers: {best_cands}\n```json\n{{\"reported_discoveries\": {json.dumps(best_cands)}, \"self_reported_M\": 3}}\n```"
            elif persona == "statistician_unassisted":
                best_cands = [t["best_feature_in_turn"] for t in sandbox.telemetry_history if t.get("naive_p", 1.0) < 0.05][:2]
                output_text = f"Significant drivers: {best_cands}\n```json\n{{\"reported_discoveries\": {json.dumps(best_cands)}, \"self_reported_M\": 8}}\n```"
            else:
                ver_cands = [t["best_feature_in_turn"] for t in sandbox.telemetry_history if t.get("verdict") == "VERIFIED_SIGNAL"]
                output_text = f"SignalGate verified drivers: {ver_cands}\n```json\n{{\"reported_discoveries\": {json.dumps(ver_cands)}, \"self_reported_M\": {sandbox.telemetry_history[-1].get('cumulative_M', 50)}}}\n```"
        else:
            # Live Frontier LLM call
            msgs_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_transcript},
            ]

            if "gemini" in model_name.lower():
                llm = NativeGeminiModel(model_name=model_name, temperature=temperature)
            elif "claude" in model_name.lower():
                llm = NativeClaudeModel(model_name=model_name, temperature=temperature)
            else:
                llm = NativeOpenAIModel(model_name=model_name, temperature=temperature)

            resp = llm.invoke(msgs_payload)
            output_text = getattr(resp, "content", str(resp))

            if trace_cache is not None and cache_key is not None:
                trace_cache.put(cache_key, {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "persona": persona,
                    "seed": seed,
                    "dry_run": False,
                    "output": output_text,
                })

    # 3. Parse report
    parsed = parse_agent_discovery_report(output_text, sandbox)
    reported = parsed["reported_discoveries"]
    self_m = parsed["parsed_self_reported_M"] or sandbox.agent_self_reported_M
    true_m = sandbox.telemetry_history[-1].get("cumulative_M", 50) if sandbox.telemetry_history else 50

    # 4. Audit against unseen holdout test set!
    holdout_audit = evaluate_real_world_holdout_replication(
        reported_discoveries=reported,
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        self_reported_M=self_m,
        true_M_raw=true_m,
    )

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "persona": persona,
        "seed": seed,
        "cache_hit": cache_hit,
        "output_text": output_text,
        "reported_discoveries": reported,
        "self_reported_M": self_m,
        "true_M_raw": true_m,
        "holdout_audit": holdout_audit,
    }
