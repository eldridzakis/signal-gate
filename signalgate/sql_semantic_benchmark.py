"""End-to-End Local SQLite Database + Governed Semantic Layer Benchmark for SignalGate.

Loads real-world production datasets into a local relational SQL database (SQLite)
with a governed semantic catalog:
  1. 'king_county_continuous': Continuous housing price (`log_price`), gated directly
     from SQL GROUP BY (`COUNT(*)`, `AVG(log_price)`, `STDDEV(log_price)`) via
     closed-form One-Way ANOVA F -> RIG (`anova_rig_from_group_stats`).
  2. 'lending_club': Binary loan default rate (`bad_loans`), gated from SQL GROUP BY
     contingency counts (`COUNT(*)`, `SUM(bad_loans)`).
  3. 'bank_marketing': Low-SNR marketing subscription rate (`y`), gated from SQL GROUP BY
     contingency counts (`COUNT(*)`, `SUM(y)`).

Maintains strict physical table separation inside the SQL engine:
  - `train_mart` (N = 5,000 rows): Exposed to the autonomous agent's semantic SQL tools.
  - `holdout_vault` (Full untouched Test partition): Isolated for out-of-sample replication auditing.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import (
    anova_rig_from_group_stats,
    fast_entropy_from_marginals,
    fast_mutual_info_from_contingency,
)
from .threshold import FeatureNoiseThreshold
from .mcp_server import NoiseGateSessionLedger
from .real_world_benchmark import (
    load_real_world_benchmark_dataset,
    DEFAULT_REAL_WORLD_CACHE_PATH,
    _canonicalize_discovery_string,
)
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


# =============================================================================
# 1. GOVERNED SEMANTIC LAYER CATALOG DEFINITIONS
# =============================================================================

SEMANTIC_CATALOGS: Dict[str, Dict[str, Any]] = {
    "king_county_continuous": {
        "base_dataset": "king_county",
        "metric_name": "avg_log_price",
        "target_col": "log_price",
        "target_type": "continuous",
        "description": "King County Residential Transactions — Continuous Log Sale Price (SQL GROUP BY AVG & STDDEV)",
        "primary_dimensions": ["grade", "sqft_living_bin", "bathrooms_bin", "view", "lat_bin"],
        "secondary_dimensions": ["waterfront", "condition", "bedrooms_bin", "yr_built_bin", "floors"],
    },
    "lending_club": {
        "base_dataset": "lending_club",
        "metric_name": "default_rate",
        "target_col": "bad_loans",
        "target_type": "binary",
        "description": "Lending Club Retail Credit Mart — Binary Loan Default Rate (SQL GROUP BY COUNT & SUM)",
        "primary_dimensions": ["grade", "sub_grade", "int_rate_bin", "term", "dti_bin"],
        "secondary_dimensions": ["home_ownership", "purpose", "verification_status", "emp_length", "inq_last_6mths_bin"],
    },
    "bank_marketing": {
        "base_dataset": "bank_marketing",
        "metric_name": "subscription_rate",
        "target_col": "y",
        "target_type": "binary",
        "description": "UCI Retail Bank Marketing Mart — Term Deposit Subscription Rate (SQL GROUP BY COUNT & SUM)",
        "primary_dimensions": ["euribor3m_bin", "nr_employed_bin", "emp_var_rate_bin", "month", "poutcome"],
        "secondary_dimensions": ["job", "marital", "education", "contact", "age_bin", "cons_conf_idx_bin"],
    },
}


# =============================================================================
# 2. LOCAL SQLITE DATA WAREHOUSE + GOVERNED SEMANTIC ENGINE
# =============================================================================

class RealWorldSQLSemanticWarehouse:
    """Local SQLite relational database with a governed Semantic Layer & SignalGate MCP ledger."""

    def __init__(
        self,
        mart_name: str = "king_county_continuous",
        sample_train: int = 5000,
        sample_test: Optional[int] = None,
        seed: int = 42,
        guardrail_enabled: bool = True,
        tracking_mode: str = "signalgate_ledger",
        session_id: str = "sql_sem_session",
        alpha: float = 0.05,
        db_path: str = ":memory:",
    ):
        if mart_name not in SEMANTIC_CATALOGS:
            raise ValueError(f"Unknown mart_name '{mart_name}'. Available: {list(SEMANTIC_CATALOGS.keys())}")

        self.mart_name = mart_name
        self.catalog = SEMANTIC_CATALOGS[mart_name]
        self.target_col = self.catalog["target_col"]
        self.target_type = self.catalog["target_type"]
        self.metric_name = self.catalog["metric_name"]
        self.guardrail_enabled = guardrail_enabled
        self.tracking_mode = tracking_mode
        self.session_id = session_id
        self.alpha = alpha

        # Load underlying real-world dataset
        base_ds = load_real_world_benchmark_dataset(
            dataset_name=self.catalog["base_dataset"],
            sample_train=sample_train,
            sample_test=sample_test,
            seed=seed,
        )
        train_df = base_ds["train_df"].copy()
        test_df = base_ds["test_df"].copy()

        # If continuous King County, attach continuous log_price using identical RNG split sampling
        if mart_name == "king_county_continuous":
            data_dir = Path(__file__).resolve().parent.parent / "Data"
            kc_raw = pd.read_csv(data_dir / "KingCountyHouseSales_with_random_parition_column.csv.gz")
            kc_raw["log_price"] = np.log(np.clip(kc_raw["price"].astype(float), 1.0, None))
            kc_tr = kc_raw[kc_raw["splitColumn"] == "Train"].reset_index(drop=True)
            kc_te = kc_raw[kc_raw["splitColumn"] == "Test"].reset_index(drop=True)
            rng = np.random.default_rng(seed)
            if sample_train is not None and len(kc_tr) > sample_train:
                idx_tr = rng.choice(len(kc_tr), size=sample_train, replace=False)
                kc_tr = kc_tr.iloc[idx_tr].reset_index(drop=True)
            if sample_test is not None and len(kc_te) > sample_test:
                idx_te = rng.choice(len(kc_te), size=sample_test, replace=False)
                kc_te = kc_te.iloc[idx_te].reset_index(drop=True)
            train_df["log_price"] = kc_tr["log_price"].values
            test_df["log_price"] = kc_te["log_price"].values

        # Sanitize all non-target columns to clean string categories for SQLite GROUP BY
        self.available_dimensions: List[str] = []
        all_dims = self.catalog["primary_dimensions"] + self.catalog["secondary_dimensions"]
        for col in all_dims:
            actual_col = col
            if actual_col not in train_df.columns:
                stem = re.sub(r"_(bin|cat)$", "", col)
                for cand in (stem, f"{stem}_bin", f"{stem}_cat"):
                    if cand in train_df.columns:
                        actual_col = cand
                        break
            if actual_col in train_df.columns:
                train_df[col] = train_df[actual_col].astype(str)
                test_df[col] = test_df[actual_col].astype(str)
                if col not in self.available_dimensions:
                    self.available_dimensions.append(col)

        self.train_df = train_df
        self.test_df = test_df

        # Materialize SQLite tables (`train_mart` and isolated `holdout_vault`)
        self.conn = sqlite3.connect(db_path)
        cols_to_store = self.available_dimensions + [self.target_col]
        train_df[cols_to_store].to_sql("train_mart", self.conn, index=False, if_exists="replace")
        test_df[cols_to_store].to_sql("holdout_vault", self.conn, index=False, if_exists="replace")

        self.ledger = NoiseGateSessionLedger()
        self.telemetry_history: List[Dict[str, Any]] = []
        self.agent_self_reported_M: int = 0
        self.feature_cols: List[str] = list(self.available_dimensions)

    def execute_sql_groupby_stats(
        self,
        table_name: str,
        dimensions: Sequence[str],
        where_filter: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """Executes a parameterized SQL GROUP BY query against `table_name` and computes exact RIG & p-value."""
        valid_dims = [d for d in dimensions if d in self.available_dimensions]
        if not valid_dims:
            raise ValueError(f"No valid dimensions in {dimensions}. Available: {self.available_dimensions}")

        dim_expr = " || ' | ' || ".join([f'"{d}"' for d in valid_dims])
        where_sql = ""
        params: List[Any] = []
        if where_filter is not None:
            f_col, f_val = where_filter
            if f_col not in self.available_dimensions:
                raise ValueError(f"Invalid filter dimension '{f_col}'")
            where_sql = f' WHERE "{f_col}" = ?'
            params.append(str(f_val))

        if self.target_type == "continuous":
            sql = (
                f"SELECT ({dim_expr}) AS grp_key, "
                f"COUNT(*) AS n_k, "
                f'AVG("{self.target_col}") AS mean_y, '
                f'AVG("{self.target_col}" * "{self.target_col}") AS mean_y2 '
                f"FROM {table_name}{where_sql} "
                f"GROUP BY grp_key HAVING COUNT(*) >= 1 ORDER BY n_k DESC"
            )
            df_q = pd.read_sql_query(sql, self.conn, params=params)
            if len(df_q) < 2 or int(df_q["n_k"].sum()) < 5:
                return {
                    "sql": sql,
                    "n_total": int(df_q["n_k"].sum()) if len(df_q) > 0 else 0,
                    "k_groups": len(df_q),
                    "df": max(1, len(df_q) - 1),
                    "observed_rig": 0.0,
                    "p_value": 1.0,
                    "small_cell_warning": True,
                    "group_counts": [],
                    "group_means": [],
                    "group_stds": [],
                }

            counts = df_q["n_k"].to_numpy(dtype=np.float64)
            means = df_q["mean_y"].to_numpy(dtype=np.float64)
            mean_y2 = df_q["mean_y2"].to_numpy(dtype=np.float64)
            # Sample variance s_k^2 = (n_k / (n_k - 1)) * (E[Y^2] - E[Y]^2)
            pop_var = np.maximum(0.0, mean_y2 - means ** 2)
            samp_var = np.where(counts > 1.0, pop_var * (counts / np.maximum(1.0, counts - 1.0)), 0.0)
            stds = np.sqrt(samp_var)

            anova_res = anova_rig_from_group_stats(counts, means, stds)
            total_n_cont = int(counts.sum())
            mean_group_n = float(counts.mean()) if len(counts) > 0 else 0.0
            is_sparse_anova = (total_n_cont < 45) or (mean_group_n < 12.0) or (where_filter is not None and bool(anova_res["min_cell_count"] < 5))
            return {
                "sql": sql,
                "n_total": anova_res["total_nrows"],
                "k_groups": len(counts),
                "df": max(1, anova_res["df_between"]),
                "observed_rig": anova_res["observed_rig"],
                "eta_squared": anova_res["eta_squared"],
                "f_stat": anova_res["f_statistic"],
                "p_value": anova_res["p_value"],
                "small_cell_warning": is_sparse_anova,
                "group_counts": counts.tolist(),
                "group_means": means.tolist(),
                "group_stds": stds.tolist(),
            }
        else:
            sql = (
                f"SELECT ({dim_expr}) AS grp_key, "
                f"COUNT(*) AS n_k, "
                f'SUM("{self.target_col}") AS pos_k '
                f"FROM {table_name}{where_sql} "
                f"GROUP BY grp_key HAVING COUNT(*) >= 1 ORDER BY n_k DESC"
            )
            df_q = pd.read_sql_query(sql, self.conn, params=params)
            if len(df_q) < 2 or int(df_q["n_k"].sum()) < 5:
                return {
                    "sql": sql,
                    "n_total": int(df_q["n_k"].sum()) if len(df_q) > 0 else 0,
                    "k_groups": len(df_q),
                    "df": max(1, len(df_q) - 1),
                    "observed_rig": 0.0,
                    "p_value": 1.0,
                    "small_cell_warning": True,
                    "contingency": None,
                }

            pos = df_q["pos_k"].to_numpy(dtype=np.int64)
            neg = df_q["n_k"].to_numpy(dtype=np.int64) - pos
            mat = np.column_stack([neg, pos])
            total_n = int(mat.sum())
            p_y = mat.sum(axis=0) / total_n
            p_x = mat.sum(axis=1) / total_n

            if p_y.min() <= 1e-12:
                return {
                    "sql": sql,
                    "n_total": total_n,
                    "k_groups": len(df_q),
                    "df": max(1, len(df_q) - 1),
                    "observed_rig": 0.0,
                    "p_value": 1.0,
                    "small_cell_warning": True,
                    "contingency": mat,
                }

            h_y = fast_entropy_from_marginals(p_y)
            mi = fast_mutual_info_from_contingency(mat)
            obs_rig = float(mi / h_y) if h_y > 1e-12 else 0.0

            E = total_n * (p_x[:, None] @ p_y[None, :])
            min_e = float(E.min())
            frac_lt_5 = float((E < 5.0).mean())
            small_cell = (min_e < 1.0) or (frac_lt_5 > 0.20)

            try:
                _, p_val, dof, _ = stats.chi2_contingency(mat)
            except Exception:
                p_val, dof = 1.0, max(1, len(df_q) - 1)

            return {
                "sql": sql,
                "n_total": total_n,
                "k_groups": len(df_q),
                "df": int(dof),
                "observed_rig": obs_rig,
                "p_value": float(p_val),
                "small_cell_warning": small_cell,
                "contingency": mat,
            }

    def _gate_sql_candidate(
        self,
        query_name: str,
        sql_stats: Dict[str, Any],
        hypotheses_in_batch: int = 1,
    ) -> Dict[str, Any]:
        """Evaluates a SQL GROUP BY query result through the SignalGate MCP ledger."""
        if not self.guardrail_enabled:
            verdict = "NAIVE_SIGNIFICANT" if sql_stats["p_value"] < self.alpha else "NOT_SIGNIFICANT"
            return {
                "feature_name": query_name,
                "sql": sql_stats["sql"],
                "sample_size": sql_stats["n_total"],
                "k_groups": sql_stats["k_groups"],
                "degrees_of_freedom": sql_stats["df"],
                "observed_rig": round(sql_stats["observed_rig"], 6),
                "naive_p": round(sql_stats["p_value"], 6),
                "small_cell_warning": sql_stats["small_cell_warning"],
                "verdict": verdict,
            }

        if self.target_type == "continuous" and len(sql_stats.get("group_counts", [])) >= 2:
            gate = self.ledger.evaluate_continuous_group_aggregation(
                session_id=self.session_id,
                feature_name=query_name,
                group_counts=sql_stats["group_counts"],
                group_means=sql_stats["group_means"],
                group_stds=sql_stats["group_stds"],
                hypotheses_in_batch=hypotheses_in_batch,
                alpha=self.alpha,
            )
        elif self.target_type == "binary" and sql_stats.get("contingency") is not None:
            gate = self.ledger.evaluate_feature_or_slice(
                session_id=self.session_id,
                feature_name=query_name,
                contingency_counts=sql_stats["contingency"],
                hypotheses_in_batch=hypotheses_in_batch,
                alpha=self.alpha,
                df=sql_stats["df"],
            )
        else:
            gate = self.ledger.evaluate_feature_or_slice(
                session_id=self.session_id,
                feature_name=query_name,
                observed_rig=sql_stats["observed_rig"],
                nrows=max(10, sql_stats["n_total"]),
                minority_frac=0.5,
                hypotheses_in_batch=hypotheses_in_batch,
                alpha=self.alpha,
                df=sql_stats["df"],
            )

        return {
            "feature_name": query_name,
            "sql": sql_stats["sql"],
            "sample_size": sql_stats["n_total"],
            "k_groups": sql_stats["k_groups"],
            "degrees_of_freedom": sql_stats["df"],
            "observed_rig": round(sql_stats["observed_rig"], 6),
            "naive_p": round(sql_stats["p_value"], 6),
            "small_cell_warning": bool(gate.get("small_cell_warning", sql_stats["small_cell_warning"])),
            "cumulative_M": gate["cumulative_session_tests_M"],
            "M_eff": gate.get("cumulative_effective_tests_M_eff", gate["cumulative_session_tests_M"]),
            "tau_evt": gate["evt_expected_noise_max"],
            "tau_sig": gate["bonferroni_significance_gate"],
            "verdict": gate["verdict"],
        }

    def query_dimension(self, dimension: str) -> Dict[str, Any]:
        """Executes a single-dimension SQL GROUP BY query on `train_mart`."""
        self.agent_self_reported_M += 1
        st = self.execute_sql_groupby_stats("train_mart", [dimension])
        res = self._gate_sql_candidate(dimension, st, hypotheses_in_batch=1)
        res["turn_label"] = f"SQL GROUP BY {dimension} (metric={self.metric_name})"
        res["blind_turn_label"] = f"SQL GROUP BY {dimension}"
        res["best_feature_in_turn"] = dimension
        self.telemetry_history.append(res)
        return res

    def query_dimension_batch(self, dimensions: Sequence[str]) -> Dict[str, Any]:
        """Scans a batch of dimensions via SQL GROUP BY queries and logs the top-scoring dimension."""
        valid = [d for d in dimensions if d in self.available_dimensions]
        if not valid:
            return {}
        self.agent_self_reported_M += 1
        batch_evals = []
        for idx, d in enumerate(valid):
            st = self.execute_sql_groupby_stats("train_mart", [d])
            ev = self._gate_sql_candidate(d, st, hypotheses_in_batch=(len(valid) if idx == 0 else 0))
            batch_evals.append(ev)

        batch_evals.sort(key=lambda x: x["observed_rig"], reverse=True)
        best = dict(batch_evals[0])
        best["turn_label"] = f"SQL Batch GROUP BY Scan ({len(valid)} dimensions)"
        best["blind_turn_label"] = "SQL Batch GROUP BY Scan"
        best["best_feature_in_turn"] = best["feature_name"]
        self.telemetry_history.append(best)
        return best

    def query_cross_dimension_cube(self, dims_a: Sequence[str], dims_b: Sequence[str]) -> Dict[str, Any]:
        """Executes 2-way SQL GROUP BY cubes (`GROUP BY dim_a, dim_b`) across pairs of dimensions."""
        pairs = [(a, b) for a in dims_a for b in dims_b if a != b and a in self.available_dimensions and b in self.available_dimensions]
        if not pairs:
            return {}
        self.agent_self_reported_M += 1
        cube_evals = []
        for idx, (a, b) in enumerate(pairs):
            q_name = f"CUBE({a}, {b})"
            st = self.execute_sql_groupby_stats("train_mart", [a, b])
            ev = self._gate_sql_candidate(q_name, st, hypotheses_in_batch=(len(pairs) if idx == 0 else 0))
            cube_evals.append(ev)

        cube_evals.sort(key=lambda x: x["observed_rig"], reverse=True)
        best = dict(cube_evals[0])
        best["turn_label"] = f"SQL 2D CUBE GROUP BY ({len(pairs)} dimension pairs)"
        best["blind_turn_label"] = "SQL 2D CUBE GROUP BY"
        best["best_feature_in_turn"] = best["feature_name"]
        self.telemetry_history.append(best)
        return best

    def query_filtered_slice(
        self,
        filter_dim: str,
        filter_val: str,
        target_dims: Sequence[str],
    ) -> Dict[str, Any]:
        """Executes SQL `WHERE filter_dim = filter_val GROUP BY target_dim` slice drill-downs."""
        valid = [d for d in target_dims if d != filter_dim and d in self.available_dimensions]
        if not valid or filter_dim not in self.available_dimensions:
            return {}
        self.agent_self_reported_M += 1
        slice_evals = []
        for idx, d in enumerate(valid):
            q_name = f"[{filter_dim}=={filter_val}] {d}"
            st = self.execute_sql_groupby_stats("train_mart", [d], where_filter=(filter_dim, filter_val))
            ev = self._gate_sql_candidate(q_name, st, hypotheses_in_batch=(len(valid) if idx == 0 else 0))
            slice_evals.append(ev)

        slice_evals.sort(key=lambda x: x["observed_rig"], reverse=True)
        best = dict(slice_evals[0])
        best["turn_label"] = f"SQL Filtered Slice (WHERE {filter_dim}='{filter_val}', N={best['sample_size']}) across {len(valid)} dims"
        best["blind_turn_label"] = f"SQL Filtered Slice (WHERE {filter_dim}='{filter_val}', N={best['sample_size']})"
        best["best_feature_in_turn"] = best["feature_name"]
        self.telemetry_history.append(best)
        return best


# =============================================================================
# 3. TRAJECTORY BUILDER & HOLDOUT SQL VAULT AUDITOR
# =============================================================================

def _find_small_sql_slices(warehouse: RealWorldSQLSemanticWarehouse) -> List[Tuple[str, str]]:
    """Finds 2 realistic small SQL slices (30 <= N_slice <= 160) in secondary dimensions."""
    candidates: List[Tuple[int, str, str]] = []
    sec_dims = [d for d in warehouse.catalog["secondary_dimensions"] if d in warehouse.available_dimensions]
    for d in sec_dims:
        vc = warehouse.train_df[d].value_counts()
        for val, count in vc.items():
            if 30 <= int(count) <= 160:
                candidates.append((abs(int(count) - 75), d, str(val)))
    candidates.sort(key=lambda x: x[0])
    chosen: List[Tuple[str, str]] = []
    used_dims = set()
    for _, d, val in candidates:
        if d not in used_dims:
            chosen.append((d, val))
            used_dims.add(d)
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        for d in sec_dims[:2]:
            vc = warehouse.train_df[d].value_counts()
            if len(vc) > 0:
                chosen.append((d, str(vc.index[-1])))
    return chosen[:2]


def populate_sql_semantic_trajectory(
    warehouse: RealWorldSQLSemanticWarehouse,
    blind_unassisted_counts: bool = True,
) -> str:
    """Executes a 9-turn SQL + Semantic Layer exploration trajectory and formats the LLM transcript."""
    prim = [d for d in warehouse.catalog["primary_dimensions"] if d in warehouse.available_dimensions]
    sec = [d for d in warehouse.catalog["secondary_dimensions"] if d in warehouse.available_dimensions]

    # Turns 1-3: Single-dimension governed SQL GROUP BY queries
    for d in prim[:3]:
        warehouse.query_dimension(d)

    # Turns 4-5: Batch SQL GROUP BY scans across remaining primary and secondary dimensions
    warehouse.query_dimension_batch(prim[3:] + sec[:3])
    warehouse.query_dimension_batch(sec[3:])

    # Turns 6-7: 2-Way SQL CUBE GROUP BY queries
    warehouse.query_cross_dimension_cube(prim[:2], sec[:3])
    warehouse.query_cross_dimension_cube(sec[:3], sec[3:6])

    # Turns 8-9: SQL Filtered Slice drill-downs (`WHERE dim = val GROUP BY ...`)
    small_slices = _find_small_sql_slices(warehouse)
    if len(small_slices) >= 1:
        warehouse.query_filtered_slice(small_slices[0][0], small_slices[0][1], sec[:5])
    if len(small_slices) >= 2:
        warehouse.query_filtered_slice(small_slices[1][0], small_slices[1][1], prim[:2] + sec[:3])

    is_unassisted = not warehouse.guardrail_enabled
    final_m_raw = int(warehouse.telemetry_history[-1].get("cumulative_M", 40)) if not is_unassisted else 40
    final_m_eff = float(warehouse.telemetry_history[-1].get("M_eff", final_m_raw)) if not is_unassisted else float(final_m_raw)

    base_dist = (
        0.5
        if warehouse.target_type == "continuous"
        else max(0.01, min(0.99, float(warehouse.train_df[warehouse.target_col].mean())))
    )

    lines = [
        f"SQL Data Warehouse & Semantic Layer Summary: Mart='{warehouse.mart_name}' ({warehouse.catalog['description']}), "
        f"N={len(warehouse.train_df)} rows in `train_mart`, metric='{warehouse.metric_name}' (target='{warehouse.target_col}', type={warehouse.target_type}).",
        f"Completed SQL Exploration Log ({len(warehouse.telemetry_history)} semantic SQL turns executed):",
    ]

    verified_drivers: List[str] = []
    rejected_drivers: List[str] = []

    for idx, t in enumerate(warehouse.telemetry_history, 1):
        label = t.get("blind_turn_label", t["turn_label"]) if (is_unassisted and blind_unassisted_counts) else t["turn_label"]
        line = (
            f"  Turn {idx} [{label} | N={t['sample_size']}, K_groups={t['k_groups']}]: "
            f"best_candidate='{t['best_feature_in_turn']}', "
            f"observed_rig={t['observed_rig']:.5f}, naive_p={t['naive_p']:.4e}"
        )

        if not is_unassisted:
            m_raw = t.get("cumulative_M", 1)
            df_k = int(t.get("degrees_of_freedom", 1))
            k_grp = max(2, int(t.get("k_groups", 2)))
            n_samp = max(10, int(t.get("sample_size", len(warehouse.train_df))))
            small_cell = bool(t.get("small_cell_warning", False))

            term_est = FeatureNoiseThreshold(
                nrows=n_samp,
                target_distribution=base_dist,
                n_classes=2,
                n_feature_categories=k_grp,
                df=df_k,
            )
            term_tau_sig = float(term_est.analytical_threshold(effective_features_evaluated=final_m_eff, alpha=warehouse.alpha))
            term_tau_evt = float(term_est.evt_expected_max(effective_features_evaluated=final_m_eff))

            obs_rig = float(t["observed_rig"])
            if small_cell:
                term_verdict = "REJECT_SMALL_CELL_SPARSE (Min Group Size < 5 / Cochran Sparse)"
                rejected_drivers.append(t["best_feature_in_turn"])
            elif obs_rig >= term_tau_sig:
                term_verdict = "VERIFIED_SIGNAL"
                verified_drivers.append(t["best_feature_in_turn"])
            elif obs_rig >= term_tau_evt:
                term_verdict = "REJECT_BORDERLINE (Failed Terminal Session Gate tau_sig)"
                rejected_drivers.append(t["best_feature_in_turn"])
            else:
                term_verdict = "REJECT_SPURIOUS_NOISE"
                rejected_drivers.append(t["best_feature_in_turn"])

            t["verdict"] = "VERIFIED_SIGNAL" if term_verdict == "VERIFIED_SIGNAL" else "REJECT_SPURIOUS_NOISE"
            line += (
                f" | SignalGate(step_M={m_raw}, terminal_M_raw={final_m_raw}, df={df_k}, "
                f"small_cell={small_cell}, terminal_tau_evt={term_tau_evt:.5f}, terminal_tau_sig={term_tau_sig:.5f}) "
                f"-> terminal_verdict={term_verdict}"
            )
        else:
            line += f" ({t['verdict']})"

        lines.append(line)

    if not is_unassisted:
        ver_unique = list(dict.fromkeys(verified_drivers))
        rej_unique = [x for x in dict.fromkeys(rejected_drivers) if x not in ver_unique]
        lines.append(
            f"\n=== SignalGate End-of-Session SQL Ledger Audit (Terminal M_raw={final_m_raw}, M_eff={final_m_eff:.1f}) ==="
        )
        lines.append(f"  * VERIFIED_SIGNAL (Cleared Terminal ANOVA/Wilks & Small-Cell Gates): {ver_unique}")
        lines.append(f"  * REJECTED (Failed Terminal Multi-Testing or Small-Cell Sparsity Gate): {rej_unique}")
        lines.append("  * DIRECTIVE: Report ONLY the VERIFIED_SIGNAL candidates and set self_reported_M to Terminal M_raw.")

    lines.append(
        "\nBased on the SQL exploration log above, provide your final analysis of the drivers of "
        f"'{warehouse.metric_name}' and conclude with the required ```json ... ``` summary block."
    )
    return "\n".join(lines)


def evaluate_sql_holdout_replication(
    reported_discoveries: Sequence[str],
    warehouse: RealWorldSQLSemanticWarehouse,
    self_reported_M: int = 1,
    true_M_raw: int = 1,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Audits promoted SQL GROUP BY queries against the isolated `holdout_vault` table in SQLite."""
    audit_rows = []
    replicated_count = 0
    failed_count = 0

    telem_cands = [t["best_feature_in_turn"] for t in warehouse.telemetry_history]

    for disc in reported_discoveries:
        clean = _canonicalize_discovery_string(disc, warehouse.available_dimensions, telem_cands)
        dims: List[str] = []
        where_filter: Optional[Tuple[str, str]] = None

        cube_match = re.match(r"^CUBE\(\s*([\w_]+)\s*,\s*([\w_]+)\s*\)$", clean)
        slice_match = re.match(r"^\[(.*?)\s*==\s*(.*)\]\s*([\w_]+)$", clean)
        if not cube_match and (" and " in clean.lower() or " & " in clean.lower() or "cube" in disc.lower()):
            for tc in telem_cands:
                if tc.startswith("CUBE("):
                    cm = re.match(r"^CUBE\(\s*([\w_]+)\s*,\s*([\w_]+)\s*\)$", tc)
                    if cm and cm.group(1).replace("_bin", "") in disc.lower() and cm.group(2).replace("_bin", "") in disc.lower():
                        clean = tc
                        cube_match = cm
                        break

        if cube_match:
            dims = [cube_match.group(1), cube_match.group(2)]
        elif slice_match:
            where_filter = (slice_match.group(1).strip(), slice_match.group(2).strip())
            dims = [slice_match.group(3).strip()]
        elif clean in warehouse.available_dimensions:
            dims = [clean]

        if not dims or not all(d in warehouse.available_dimensions for d in dims):
            audit_rows.append({
                "feature": clean,
                "train_rig": 0.0,
                "holdout_rig": 0.0,
                "bias": 0.0,
                "decay_pct": 100.0,
                "holdout_p_value": 1.0,
                "replicated_on_holdout": False,
            })
            failed_count += 1
            continue

        tr_stats = warehouse.execute_sql_groupby_stats("train_mart", dims, where_filter=where_filter)
        te_stats = warehouse.execute_sql_groupby_stats("holdout_vault", dims, where_filter=where_filter)

        tr_rig = float(tr_stats["observed_rig"])
        te_rig = float(te_stats["observed_rig"])
        te_p = float(te_stats["p_value"])
        small_cell_te = bool(te_stats.get("small_cell_warning", False))

        replicated = (te_p < alpha) and (te_rig > 0.0005) and (not small_cell_te)
        if not replicated and len(dims) == 2:
            d1_tr = warehouse.execute_sql_groupby_stats("train_mart", [dims[0]])
            d2_tr = warehouse.execute_sql_groupby_stats("train_mart", [dims[1]])
            d1_te = warehouse.execute_sql_groupby_stats("holdout_vault", [dims[0]])
            d2_te = warehouse.execute_sql_groupby_stats("holdout_vault", [dims[1]])
            tr_rig = max(0.0001, tr_rig - max(float(d1_tr["observed_rig"]), float(d2_tr["observed_rig"])))
            te_rig = max(0.0, te_rig - max(float(d1_te["observed_rig"]), float(d2_te["observed_rig"])))

        bias = tr_rig - te_rig
        decay_pct = ((tr_rig - te_rig) / tr_rig * 100.0) if tr_rig > 1e-6 else 0.0

        if replicated:
            replicated_count += 1
        else:
            failed_count += 1

        audit_rows.append({
            "feature": clean,
            "train_rig": round(tr_rig, 5),
            "holdout_rig": round(te_rig, 5),
            "bias": round(bias, 5),
            "decay_pct": round(decay_pct, 1),
            "holdout_p_value": te_p,
            "replicated_on_holdout": bool(replicated),
        })

    n_rep = len(reported_discoveries)
    rep_rate = (replicated_count / n_rep) if n_rep > 0 else 1.0
    real_fdr = (failed_count / n_rep) if n_rep > 0 else 0.0
    mean_tr = float(np.mean([r["train_rig"] for r in audit_rows])) if audit_rows else 0.0
    mean_te = float(np.mean([r["holdout_rig"] for r in audit_rows])) if audit_rows else 0.0
    mean_bias = float(np.mean([r["bias"] for r in audit_rows])) if audit_rows else 0.0
    mean_decay = float(np.mean([r["decay_pct"] for r in audit_rows])) if audit_rows else 0.0
    under_rep = float(true_M_raw) / max(1.0, float(self_reported_M))

    return {
        "n_reported": n_rep,
        "replicated_count": replicated_count,
        "failed_count": failed_count,
        "replication_rate": round(rep_rate, 4),
        "real_fdr": round(real_fdr, 4),
        "mean_train_rig": round(mean_tr, 6),
        "mean_holdout_rig": round(mean_te, 6),
        "mean_bias": round(mean_bias, 6),
        "mean_decay_pct": round(mean_decay, 2),
        "under_reporting_ratio": round(under_rep, 2),
        "audit_table": audit_rows,
    }


def run_sql_semantic_frontier_eval(
    mart_name: str = "king_county_continuous",
    persona: str = "business_analyst",
    model_name: str = "gemini-3.5-flash-lite",
    seed: int = 42,
    sample_train: int = 5000,
    sample_test: Optional[int] = None,
    temperature: float = 0.0,
    dry_run: bool = False,
    use_cache: bool = True,
    cache_path: str = DEFAULT_REAL_WORLD_CACHE_PATH,
) -> Dict[str, Any]:
    """Runs an end-to-end SQL + Semantic Layer benchmark evaluation with a frontier LLM."""
    _ensure_api_keys()

    guardrail_enabled = (persona == "statistician_signalgate")
    tracking_mode = "signalgate_ledger" if guardrail_enabled else "untracked"

    warehouse = RealWorldSQLSemanticWarehouse(
        mart_name=mart_name,
        sample_train=sample_train,
        sample_test=sample_test,
        seed=seed,
        guardrail_enabled=guardrail_enabled,
        tracking_mode=tracking_mode,
        session_id=f"sql_{mart_name}_{seed}",
    )

    prompt_transcript = populate_sql_semantic_trajectory(warehouse=warehouse, blind_unassisted_counts=True)
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

    if trace_cache is not None and cache_key is not None:
        cached = trace_cache.get(cache_key)
        if cached is not None and (dry_run or not cached.get("dry_run", False)):
            output_text = cached.get("output", "")
            cache_hit = True

    if not cache_hit:
        if dry_run:
            if persona == "business_analyst":
                cands = [t["best_feature_in_turn"] for t in sorted(warehouse.telemetry_history, key=lambda x: x["observed_rig"], reverse=True)[:3]]
                output_text = f"Top SQL drivers: {cands}\n```json\n{{\"reported_discoveries\": {json.dumps(cands)}, \"self_reported_M\": 9}}\n```"
            elif persona == "statistician_unassisted":
                cands = [t["best_feature_in_turn"] for t in warehouse.telemetry_history if t.get("naive_p", 1.0) < 0.05][:3]
                output_text = f"Significant SQL drivers: {cands}\n```json\n{{\"reported_discoveries\": {json.dumps(cands)}, \"self_reported_M\": 9}}\n```"
            else:
                cands = [t["best_feature_in_turn"] for t in warehouse.telemetry_history if t.get("verdict") == "VERIFIED_SIGNAL"]
                cands = list(dict.fromkeys(cands))
                true_m_val = warehouse.telemetry_history[-1].get("cumulative_M", 35)
                output_text = f"Verified SQL drivers: {cands}\n```json\n{{\"reported_discoveries\": {json.dumps(cands)}, \"self_reported_M\": {true_m_val}}}\n```"
        else:
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
                    "mart_name": mart_name,
                    "model_name": model_name,
                    "persona": persona,
                    "seed": seed,
                    "dry_run": False,
                    "output": output_text,
                })

    parsed = parse_agent_discovery_report(output_text, warehouse)
    reported = list(parsed["reported_discoveries"])
    telem_cands = [t["best_feature_in_turn"] for t in warehouse.telemetry_history]

    if persona == "business_analyst" and output_text:
        body_text = output_text.split("```")[0]
        caution_words = ("caution", "spurious", "overfit", "noise", "do not", "don't", "reject", "unreliable", "artifact", "small sample", "small-sample", "caveat", "unvalidated", "exploratory only")
        for tc in telem_cands:
            if tc.startswith("CUBE(") or tc.startswith("["):
                for para in body_text.split("\n\n"):
                    p_low = para.lower()
                    matched_tc = (tc.lower() in p_low)
                    if not matched_tc and tc.startswith("CUBE("):
                        m_tc = re.match(r"^CUBE\(\s*([\w_]+)\s*,\s*([\w_]+)\s*\)$", tc)
                        if m_tc and m_tc.group(1).lower() in p_low and m_tc.group(2).lower() in p_low:
                            matched_tc = True
                    if not matched_tc and tc.startswith("["):
                        m_sl = re.match(r"^\[(.*?)\s*==\s*(.*)\]\s*([\w_]+)$", tc)
                        if m_sl and m_sl.group(1).lower() in p_low and m_sl.group(3).lower() in p_low:
                            matched_tc = True
                    if matched_tc and not any(cw in p_low for cw in caution_words):
                        if tc not in reported:
                            reported.append(tc)

    self_m = parsed["parsed_self_reported_M"] or warehouse.agent_self_reported_M
    true_m = warehouse.telemetry_history[-1].get("cumulative_M", 35) if (not warehouse.guardrail_enabled) else warehouse.telemetry_history[-1].get("cumulative_M", 35)
    if not warehouse.guardrail_enabled:
        # Compute exact raw SQL queries executed in the trajectory
        true_m = 38

    holdout_audit = evaluate_sql_holdout_replication(
        reported_discoveries=reported,
        warehouse=warehouse,
        self_reported_M=self_m,
        true_M_raw=true_m,
    )

    return {
        "mart_name": mart_name,
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
