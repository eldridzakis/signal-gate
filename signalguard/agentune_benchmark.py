"""Agentune-Style Automated Feature Engineering & Overfitting Benchmark.

Benchmarks unassisted automated feature generation and driver selection (e.g., SparkBeyond Agentune)
against SignalGuard-governed selection. Demonstrates how unassisted greedy selection overfits
high-dimensional interaction candidates out-of-sample, and how SignalGuard eliminates spurious
drivers through stateful Extreme Value Theory (EVT) noise ceilings and small-cell gating.
"""

from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from .metrics import rig
from .mcp_server import NoiseGateSessionLedger
from .real_world_benchmark import load_real_world_benchmark_dataset


@dataclass
class GeneratedFeature:
    """Represents a synthetically generated interaction or slice feature."""
    name: str
    feature_type: str  # 'univariate_slice', 'conjunction_and', 'disjunction_or', 'ratio', 'noise_control'
    train_vector: np.ndarray
    test_vector: np.ndarray
    expression: str


class AgentuneFeatureSynthesizer:
    """Synthesizes high-dimensional candidate features mimicking SparkBeyond Agentune."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def synthesize_candidates(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidate_cols: List[str],
        max_candidates: int = 1500,
        noise_feature_ratio: float = 0.20
    ) -> List[GeneratedFeature]:
        """Synthesizes candidate features across univariate slices, boolean combinations, and noise controls."""
        candidates: List[GeneratedFeature] = []

        # 1. Univariate categorical / binned slice indicators
        univariate_preds: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
        for col in candidate_cols:
            if col not in train_df.columns:
                continue
            unique_vals = train_df[col].dropna().unique()
            # If high cardinality, pick most frequent
            if len(unique_vals) > 8:
                top_vals = train_df[col].value_counts().nlargest(8).index.tolist()
            else:
                top_vals = unique_vals.tolist()

            for val in top_vals:
                tr_mask = (train_df[col] == val).astype(np.int8).to_numpy()
                te_mask = (test_df[col] == val).astype(np.int8).to_numpy()
                name = f"[{col} == '{val}']"
                univariate_preds.append((col, str(val), tr_mask, te_mask))
                candidates.append(
                    GeneratedFeature(
                        name=name,
                        feature_type="univariate_slice",
                        train_vector=tr_mask,
                        test_vector=te_mask,
                        expression=name
                    )
                )

        # 2. Pairwise Boolean Conjunctions: (colA == valA) & (colB == valB)
        # Standard Agentune / SparkBeyond interaction generator
        num_preds = len(univariate_preds)
        indices = list(range(num_preds))
        self.rng.shuffle(indices)

        pair_count = 0
        target_pairs = int(max_candidates * (1.0 - noise_feature_ratio) * 0.70)
        for i, j in itertools.combinations(indices, 2):
            col_a, val_a, tr_a, te_a = univariate_preds[i]
            col_b, val_b, tr_b, te_b = univariate_preds[j]
            if col_a == col_b:
                continue  # Skip same column combinations

            # AND conjunction
            tr_and = (tr_a & tr_b).astype(np.int8)
            te_and = (te_a & te_b).astype(np.int8)
            # Only include if there is non-trivial support
            if tr_and.sum() >= 5:
                name = f"({col_a}=='{val_a}') & ({col_b}=='{val_b}')"
                candidates.append(
                    GeneratedFeature(
                        name=name,
                        feature_type="conjunction_and",
                        train_vector=tr_and,
                        test_vector=te_and,
                        expression=name
                    )
                )
                pair_count += 1
                if pair_count >= target_pairs:
                    break

        # 3. Pairwise Disjunctions: (colA == valA) | (colB == valB)
        or_count = 0
        target_or = int(max_candidates * (1.0 - noise_feature_ratio) * 0.20)
        for i, j in itertools.combinations(reversed(indices), 2):
            col_a, val_a, tr_a, te_a = univariate_preds[i]
            col_b, val_b, tr_b, te_b = univariate_preds[j]
            if col_a == col_b:
                continue

            tr_or = (tr_a | tr_b).astype(np.int8)
            te_or = (te_a | te_b).astype(np.int8)
            if tr_or.sum() >= 10 and tr_or.sum() <= len(tr_or) - 10:
                name = f"({col_a}=='{val_a}') | ({col_b}=='{val_b}')"
                candidates.append(
                    GeneratedFeature(
                        name=name,
                        feature_type="disjunction_or",
                        train_vector=tr_or,
                        test_vector=te_or,
                        expression=name
                    )
                )
                or_count += 1
                if or_count >= target_or:
                    break

        # 4. Injected Noise Features (Null Controls / Spurious Random Transformations)
        # Real-world AutoML pipelines evaluate weak/irrelevant features and spurious random splits
        target_noise = max_candidates - len(candidates)
        n_tr = len(train_df)
        n_te = len(test_df)
        for k in range(max(10, target_noise)):
            p_active = float(self.rng.uniform(0.01, 0.40))
            tr_noise = (self.rng.uniform(0, 1, size=n_tr) < p_active).astype(np.int8)
            te_noise = (self.rng.uniform(0, 1, size=n_te) < p_active).astype(np.int8)
            name = f"synth_interaction_noise_{k:04d}"
            candidates.append(
                GeneratedFeature(
                    name=name,
                    feature_type="noise_control",
                    train_vector=tr_noise,
                    test_vector=te_noise,
                    expression=f"RandomInteraction({p_active:.3f})"
                )
            )
            if len(candidates) >= max_candidates:
                break

        return candidates[:max_candidates]


def calculate_holdout_replication_metrics(
    y_test: np.ndarray,
    test_feature: np.ndarray,
    train_rig: float
) -> Dict[str, Any]:
    """Calculates holdout RIG, generalization bias, decay percentage, and permutation p-value."""
    test_rig = float(rig(y_test, test_feature))
    bias = float(train_rig - test_rig)

    if train_rig > 1e-9:
        decay_pct = float(np.clip((1.0 - (test_rig / train_rig)) * 100.0, 0.0, 100.0))
    else:
        decay_pct = 0.0

    # Chi-square / Fisher exact test for holdout replication
    table = pd.crosstab(y_test, test_feature)
    if table.shape == (2, 2) and (table.values >= 5).all():
        chi2, p_val, _, _ = stats.chi2_contingency(table, correction=False)
    elif table.shape == (2, 2):
        _, p_val = stats.fisher_exact(table)
    else:
        p_val = 1.0

    replicated = bool(p_val < 0.05 and test_rig > 0.0001)

    return {
        "holdout_rig": test_rig,
        "generalization_bias": bias,
        "decay_pct": decay_pct,
        "holdout_p_value": float(p_val),
        "replicated": replicated
    }


def run_agentune_overfitting_benchmark(
    dataset_name: str = "lending_club",
    max_candidates: int = 1500,
    top_k: int = 15,
    min_cell_size: int = 25,
    alpha: float = 0.05,
    sample_train: int = 5000,
    sample_test: int = 2500,
    seed: int = 42
) -> Dict[str, Any]:
    """Runs a complete empirical benchmark comparing unassisted Agentune vs SignalGuard-governed selection.

    Parameters
    ----------
    dataset_name : str
        'lending_club' or 'king_county'
    max_candidates : int
        Number of synthetic candidate features to generate (M)
    top_k : int
        Number of top features to select for holdout evaluation and downstream models
    min_cell_size : int
        Minimum contingency cell count required by SignalGuard to reject small-cell artifacts
    alpha : float
        Statistical significance level for multiple testing
    sample_train : int
        Training sample size
    sample_test : int
        Holdout test sample size
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Structured benchmark results including candidate ledger, top-k tables, summary comparisons,
        and downstream ML model evaluation.
    """
    # 1. Load Real-World Dataset
    data = load_real_world_benchmark_dataset(
        dataset_name=dataset_name,
        sample_train=sample_train,
        sample_test=sample_test,
        seed=seed
    )
    train_df = data["train_df"]
    test_df = data["test_df"]
    target_col = data["target_col"]
    candidate_cols = data["candidate_features"]

    y_train = train_df[target_col].to_numpy().astype(int)
    y_test = test_df[target_col].to_numpy().astype(int)
    n_train = len(y_train)
    minority_frac = float(y_train.mean())

    # 2. Synthesize High-Dimensional Interaction Candidates
    synthesizer = AgentuneFeatureSynthesizer(seed=seed)
    candidates = synthesizer.synthesize_candidates(
        train_df=train_df,
        test_df=test_df,
        candidate_cols=candidate_cols,
        max_candidates=max_candidates,
        noise_feature_ratio=0.25
    )

    # 3. Evaluate Candidates in SignalGuard Session Ledger
    ledger = NoiseGateSessionLedger()
    session_id = f"agentune_eval_{dataset_name}_{seed}"
    ledger.reset_session_budget(session_id)

    evaluated_records: List[Dict[str, Any]] = []

    for idx, cand in enumerate(candidates):
        tr_vec = cand.train_vector
        obs_rig = float(rig(y_train, tr_vec))

        # Check contingency cell support
        t_active = int(tr_vec.sum())
        n11 = int((tr_vec & y_train).sum())
        n10 = t_active - n11
        n01 = int((((1 - tr_vec) & y_train)).sum())
        n00 = (n_train - t_active) - n01

        min_cell = min(n00, n01, n10, n11)
        small_cell_artifact = bool(min_cell < min_cell_size)

        # SignalGuard stateful multi-testing ledger evaluation
        gate_res = ledger.evaluate_feature_or_slice(
            observed_rig=obs_rig,
            nrows=n_train,
            minority_frac=minority_frac,
            feature_support=float(t_active / n_train),
            hypotheses_in_batch=1,
            alpha=alpha,
            session_id=session_id,
            feature_name=cand.name
        )

        tau_sig = float(gate_res.get("tau_sig") or 0.0)
        tau_evt = float(gate_res.get("tau_evt") or 0.0)
        bonf_passed = bool(gate_res.get("bonferroni_significance_gate", False))
        fdr_passed = bool(gate_res.get("fdr_significance_gate", False))
        passed_noise_gate = bool(bonf_passed or (obs_rig > tau_sig and not small_cell_artifact))

        evaluated_records.append({
            "candidate_idx": idx,
            "feature_name": cand.name,
            "feature_type": cand.feature_type,
            "expression": cand.expression,
            "train_rig": obs_rig,
            "min_cell": min_cell,
            "small_cell_artifact": small_cell_artifact,
            "cumulative_M": gate_res.get("cumulative_session_tests_M", idx + 1),
            "tau_sig": tau_sig,
            "tau_evt": tau_evt,
            "bonf_passed": bonf_passed,
            "fdr_passed": fdr_passed,
            "passed_noise_gate": passed_noise_gate,
            "cand_obj": cand
        })

    eval_df = pd.DataFrame(evaluated_records)

    # 4. Strategy 1: Unassisted Agentic AutoML (Agentune Baseline)
    # Greedily ranks all M generated features by train_rig (Search Space Amnesia)
    unassisted_top = eval_df.sort_values(by="train_rig", ascending=False).head(top_k).copy()

    # Strategy 2: SignalGuard-Governed Selection
    # Requires clearing dynamic noise ceiling tau_sig AND passing small-cell guardrails
    signalguard_pool = eval_df[
        (eval_df["passed_noise_gate"]) & (~eval_df["small_cell_artifact"])
    ].copy()

    if len(signalguard_pool) > 0:
        signalguard_top = signalguard_pool.sort_values(by="train_rig", ascending=False).head(top_k).copy()
    else:
        # Fallback to features beating tau_evt if strict tau_sig yielded fewer than top_k
        signalguard_fallback = eval_df[
            (eval_df["train_rig"] > eval_df["tau_evt"]) & (~eval_df["small_cell_artifact"])
        ].copy()
        signalguard_top = signalguard_fallback.sort_values(by="train_rig", ascending=False).head(top_k).copy()

    # 5. Out-of-Sample Holdout Verification
    def _evaluate_top_set(df_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        rows = []
        for _, row in df_subset.iterrows():
            cand: GeneratedFeature = row["cand_obj"]
            metrics = calculate_holdout_replication_metrics(
                y_test=y_test,
                test_feature=cand.test_vector,
                train_rig=row["train_rig"]
            )
            rec = {
                "feature_name": row["feature_name"],
                "feature_type": row["feature_type"],
                "train_rig": row["train_rig"],
                "min_cell": row["min_cell"],
                "small_cell_artifact": row["small_cell_artifact"],
                "tau_sig": row["tau_sig"],
                "holdout_rig": metrics["holdout_rig"],
                "generalization_bias": metrics["generalization_bias"],
                "decay_pct": metrics["decay_pct"],
                "holdout_p_value": metrics["holdout_p_value"],
                "replicated": metrics["replicated"]
            }
            rows.append(rec)

        res_df = pd.DataFrame(rows)
        if len(res_df) > 0:
            rep_rate = float(res_df["replicated"].mean() * 100.0)
            real_fdr = float(100.0 - rep_rate)
            mean_train_rig = float(res_df["train_rig"].mean())
            mean_holdout_rig = float(res_df["holdout_rig"].mean())
            mean_bias = float(res_df["generalization_bias"].mean())
            mean_decay = float(res_df["decay_pct"].mean())
        else:
            rep_rate = 0.0
            real_fdr = 100.0
            mean_train_rig = 0.0
            mean_holdout_rig = 0.0
            mean_bias = 0.0
            mean_decay = 0.0

        summary = {
            "selected_features_count": len(res_df),
            "holdout_replication_rate_pct": rep_rate,
            "real_fdr_pct": real_fdr,
            "mean_train_rig": mean_train_rig,
            "mean_holdout_rig": mean_holdout_rig,
            "generalization_bias": mean_bias,
            "mean_decay_pct": mean_decay
        }
        return res_df, summary

    unassisted_verified_df, unassisted_summary = _evaluate_top_set(unassisted_top)
    signalguard_verified_df, signalguard_summary = _evaluate_top_set(signalguard_top)

    # 6. Downstream ML Model Evaluation (Overfitting Gap)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Construct baseline feature matrix via pd.get_dummies for categorical stability
    top_base_cols = [c for c in candidate_cols[:10] if c in train_df.columns]
    combined_base = pd.concat([train_df[top_base_cols], test_df[top_base_cols]], axis=0)
    combined_dummies = pd.get_dummies(combined_base, drop_first=True, dtype=float)
    X_base_tr = combined_dummies.iloc[:n_train].to_numpy()
    X_base_te = combined_dummies.iloc[n_train:].to_numpy()

    # Construct Unassisted Augmented Matrix
    unassisted_tr_feats = [r["cand_obj"].train_vector for _, r in unassisted_top.iterrows()]
    unassisted_te_feats = [r["cand_obj"].test_vector for _, r in unassisted_top.iterrows()]
    if unassisted_tr_feats:
        X_unassisted_tr = np.column_stack([X_base_tr] + unassisted_tr_feats)
        X_unassisted_te = np.column_stack([X_base_te] + unassisted_te_feats)
    else:
        X_unassisted_tr, X_unassisted_te = X_base_tr, X_base_te

    # Construct SignalGuard Augmented Matrix
    signalguard_tr_feats = [r["cand_obj"].train_vector for _, r in signalguard_top.iterrows()]
    signalguard_te_feats = [r["cand_obj"].test_vector for _, r in signalguard_top.iterrows()]
    if signalguard_tr_feats:
        X_signalguard_tr = np.column_stack([X_base_tr] + signalguard_tr_feats)
        X_signalguard_te = np.column_stack([X_base_te] + signalguard_te_feats)
    else:
        X_signalguard_tr, X_signalguard_te = X_base_tr, X_base_te

    def _fit_eval_model(X_tr: np.ndarray, X_te: np.ndarray) -> Dict[str, float]:
        clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
        clf.fit(X_tr, y_train)
        pred_tr = clf.predict_proba(X_tr)[:, 1]
        pred_te = clf.predict_proba(X_te)[:, 1]
        auc_tr = float(roc_auc_score(y_train, pred_tr))
        auc_te = float(roc_auc_score(y_test, pred_te))
        gap = float(auc_tr - auc_te)
        return {
            "train_auc": auc_tr,
            "holdout_test_auc": auc_te,
            "overfitting_gap": gap
        }

    model_base = _fit_eval_model(X_base_tr, X_base_te)
    model_unassisted = _fit_eval_model(X_unassisted_tr, X_unassisted_te)
    model_signalguard = _fit_eval_model(X_signalguard_tr, X_signalguard_te)

    downstream_df = pd.DataFrame([
        {
            "Pipeline": "Baseline Raw Features",
            "N_Features": X_base_tr.shape[1],
            "Train_AUC": model_base["train_auc"],
            "Holdout_Test_AUC": model_base["holdout_test_auc"],
            "Overfitting_Gap": model_base["overfitting_gap"]
        },
        {
            "Pipeline": "Baseline + Unassisted Agentune Top-K",
            "N_Features": X_unassisted_tr.shape[1],
            "Train_AUC": model_unassisted["train_auc"],
            "Holdout_Test_AUC": model_unassisted["holdout_test_auc"],
            "Overfitting_Gap": model_unassisted["overfitting_gap"]
        },
        {
            "Pipeline": "Baseline + SignalGuard-Governed Top-K",
            "N_Features": X_signalguard_tr.shape[1],
            "Train_AUC": model_signalguard["train_auc"],
            "Holdout_Test_AUC": model_signalguard["holdout_test_auc"],
            "Overfitting_Gap": model_signalguard["overfitting_gap"]
        }
    ])

    summary_df = pd.DataFrame([
        {
            "Approach": "Unassisted Agentune (Greedy AutoML)",
            "Total_Generated_M": max_candidates,
            "Selected_Features": unassisted_summary["selected_features_count"],
            "Holdout_Replication_Rate (%)": unassisted_summary["holdout_replication_rate_pct"],
            "Real_FDR (%)": unassisted_summary["real_fdr_pct"],
            "Mean_Train_RIG": unassisted_summary["mean_train_rig"],
            "Mean_Holdout_RIG": unassisted_summary["mean_holdout_rig"],
            "Generalization_Bias": unassisted_summary["generalization_bias"],
            "Decay (%)": unassisted_summary["mean_decay_pct"]
        },
        {
            "Approach": "SignalGuard-Governed Selection",
            "Total_Generated_M": max_candidates,
            "Selected_Features": signalguard_summary["selected_features_count"],
            "Holdout_Replication_Rate (%)": signalguard_summary["holdout_replication_rate_pct"],
            "Real_FDR (%)": signalguard_summary["real_fdr_pct"],
            "Mean_Train_RIG": signalguard_summary["mean_train_rig"],
            "Mean_Holdout_RIG": signalguard_summary["mean_holdout_rig"],
            "Generalization_Bias": signalguard_summary["generalization_bias"],
            "Decay (%)": signalguard_summary["mean_decay_pct"]
        }
    ])

    return {
        "dataset_name": dataset_name,
        "n_train": n_train,
        "n_test": len(y_test),
        "total_generated_candidates_M": max_candidates,
        "unassisted_verified_df": unassisted_verified_df,
        "signalguard_verified_df": signalguard_verified_df,
        "summary_df": summary_df,
        "downstream_df": downstream_df
    }
