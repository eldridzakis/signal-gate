"""Stateful Model Context Protocol (MCP) Statistical Guardrail Server.

Provides stateful hypothesis tracking, real-time noise gate calculation,
and Winner's Curse de-biasing for autonomous LLM data science, AutoML,
and Text-to-SQL / EDA agents.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .metrics import fast_entropy_from_marginals, fast_mutual_info_from_contingency, rig
from .threshold import FeatureNoiseThreshold
from .debias import FeatureDebiaser


@dataclass
class SessionState:
    """Stateful record of hypothesis testing for an agent session."""
    session_id: str
    target_name: Optional[str] = None
    cumulative_hypotheses_M: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    initial_tau_sig: Optional[float] = None
    initial_tau_evt: Optional[float] = None


class NoiseGateSessionLedger:
    """Stateful ledger tracking hypothesis testing volume and statistical gates.

    Designed to guard against multiple-testing bias (the Winner's Curse) when
    autonomous agents explore slices, feature transformations, or EDA patterns.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create_session(
        self,
        session_id: str = "default",
        target_name: Optional[str] = None
    ) -> SessionState:
        """Retrieves or initializes a session ledger."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                target_name=target_name,
                cumulative_hypotheses_M=0
            )
        elif target_name is not None and self._sessions[session_id].target_name is None:
            self._sessions[session_id].target_name = target_name
        return self._sessions[session_id]

    def reset_session_budget(self, session_id: str = "default") -> Dict[str, Any]:
        """Resets the search budget and test counter for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].cumulative_hypotheses_M = 0
            self._sessions[session_id].history.clear()
            self._sessions[session_id].initial_tau_sig = None
            self._sessions[session_id].initial_tau_evt = None

        return {
            "session_id": session_id,
            "status": "RESET",
            "cumulative_hypotheses_M": 0,
            "message": f"Search budget and test counter reset for session '{session_id}'."
        }

    def get_search_budget_status(
        self,
        session_id: str = "default",
        nrows: Optional[int] = None,
        minority_frac: Optional[float] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Inspects current hypothesis budget, noise ceiling, and gate inflation."""
        session = self.get_or_create_session(session_id=session_id)
        m = max(1, session.cumulative_hypotheses_M)

        res = {
            "session_id": session_id,
            "target_name": session.target_name,
            "cumulative_hypotheses_M": session.cumulative_hypotheses_M,
        }

        if nrows is not None and minority_frac is not None:
            p = float(minority_frac)
            n = int(nrows)
            estimator = FeatureNoiseThreshold(nrows=n, target_distribution=p)
            current_tau_sig = estimator.analytical_threshold(n_features_evaluated=m, alpha=alpha)
            current_tau_evt = estimator.evt_expected_max(n_features_evaluated=m)

            base_tau_sig = estimator.analytical_threshold(n_features_evaluated=1, alpha=alpha)
            base_tau_evt = estimator.evt_expected_max(n_features_evaluated=1)

            inflation_sig = current_tau_sig / base_tau_sig if base_tau_sig > 0 else 1.0

            res.update({
                "current_noise_floor_evt": float(current_tau_evt),
                "current_significance_gate": float(current_tau_sig),
                "base_single_test_gate": float(base_tau_sig),
                "gate_inflation_factor": float(inflation_sig),
                "gate_penalty_increase": float(current_tau_sig - base_tau_sig),
                "stopping_recommendation": (
                    "STOP_HIGH_PENALTY: Significance gate has increased substantially. "
                    "Further search without holdout validation carries high risk of false discoveries."
                    if inflation_sig > 3.0 and m >= 1000
                    else "CONTINUE: Search budget remains viable."
                )
            })

        return res

    def evaluate_feature_or_slice(
        self,
        observed_rig: Optional[float] = None,
        nrows: Optional[int] = None,
        minority_frac: Optional[float] = None,
        feature_support: Optional[float] = None,
        contingency_counts: Optional[Union[Sequence[int], Dict[str, int]]] = None,
        hypotheses_in_batch: int = 1,
        alpha: float = 0.05,
        session_id: str = "default",
        feature_name: Optional[str] = None,
        slice_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates whether an observed RIG or 2x2 slice clears statistical noise gates.

        Parameters
        ----------
        observed_rig : float, optional
            Observed Relative Information Gain on training split.
        nrows : int, optional
            Total sample size (N).
        minority_frac : float, optional
            Minority class proportion (p).
        feature_support : float, optional
            Fraction of rows where feature or slice predicate is true.
        contingency_counts : sequence or dict, optional
            [n00, n01, n10, n11] counts where row 0 is feature=0, row 1 is feature=1.
            If provided, automatically computes observed_rig, nrows, minority_frac,
            and feature_support.
        hypotheses_in_batch : int, default 1
            Number of candidate hypotheses evaluated in this step.
        alpha : float, default 0.05
            Family-wise significance level.
        session_id : str, default 'default'
            Identifier for session state tracking.
        feature_name : str, optional
            Name or expression of the candidate feature.
        slice_description : str, optional
            Predicate defining the segment or slice.
        """
        session = self.get_or_create_session(session_id=session_id)

        # 1. Parse contingency counts if provided
        if contingency_counts is not None:
            if isinstance(contingency_counts, dict):
                n00 = contingency_counts.get("n00", contingency_counts.get("00", 0))
                n01 = contingency_counts.get("n01", contingency_counts.get("01", 0))
                n10 = contingency_counts.get("n10", contingency_counts.get("10", 0))
                n11 = contingency_counts.get("n11", contingency_counts.get("11", 0))
            else:
                n00, n01, n10, n11 = contingency_counts

            total_n = int(n00 + n01 + n10 + n11)
            if total_n <= 0:
                raise ValueError("Total contingency counts must be positive.")

            mat = np.array([[n00, n01], [n10, n11]], dtype=np.int64)
            n_y0 = n00 + n10
            n_y1 = n01 + n11
            min_class_count = min(n_y0, n_y1)
            derived_minority = min_class_count / total_n

            n_x1 = n10 + n11
            derived_support = n_x1 / total_n

            p_y = np.array([n_y0 / total_n, n_y1 / total_n], dtype=float)
            h_y = fast_entropy_from_marginals(p_y)
            mi = fast_mutual_info_from_contingency(mat)
            calc_rig = (mi / h_y) if h_y > 1e-12 else 0.0

            if observed_rig is None:
                observed_rig = float(calc_rig)
            if nrows is None:
                nrows = total_n
            if minority_frac is None:
                minority_frac = float(derived_minority)
            if feature_support is None:
                feature_support = float(derived_support)

        if nrows is None or minority_frac is None or observed_rig is None:
            raise ValueError(
                "Either contingency_counts or all of (observed_rig, nrows, minority_frac) must be supplied."
            )

        n = int(nrows)
        p = float(minority_frac)
        r = float(np.clip(observed_rig, 0.0, 1.0))

        # 2. Update session hypothesis budget
        batch_size = max(1, int(hypotheses_in_batch))
        session.cumulative_hypotheses_M += batch_size
        m = session.cumulative_hypotheses_M

        # 3. Calculate analytical noise gates
        estimator = FeatureNoiseThreshold(nrows=n, target_distribution=p)
        tau_sig = estimator.analytical_threshold(n_features_evaluated=m, alpha=alpha)
        tau_evt = estimator.evt_expected_max(n_features_evaluated=m)

        # 4. Small-cell condition check (Cochran's rule)
        e11 = None
        small_cell_warning = False
        if feature_support is not None:
            sup = float(feature_support)
            e11 = float(n * p * sup)
            if e11 < 5.0:
                small_cell_warning = True

        # 5. Formulate verdict and agent directive
        if r >= tau_sig:
            verdict = "VERIFIED_SIGNAL"
            agent_directive = (
                f"PROCEED: Feature/slice observed RIG ({r:.5f}) clears the multiple-testing "
                f"significance gate (tau={tau_sig:.5f}) at cumulative M={m} tests and alpha={alpha}. "
                "Statistically confirmed to be genuine signal beyond random search noise."
            )
        elif r >= tau_evt:
            verdict = "BORDERLINE_WARNING"
            agent_directive = (
                f"BORDERLINE: Feature/slice observed RIG ({r:.5f}) exceeds expected maximum noise (tau_evt={tau_evt:.5f}), "
                f"but fails the strict Bonferroni significance gate (tau={tau_sig:.5f}) for M={m} tests. "
                "High risk of selection bias / Winner's Curse. Do NOT deploy without out-of-fold cross-validation or holdout verification."
            )
        else:
            verdict = "REJECT_SPURIOUS_NOISE"
            agent_directive = (
                f"DISCARD: Feature/slice observed RIG ({r:.5f}) is below the expected noise ceiling (tau_evt={tau_evt:.5f}) "
                f"for M={m} evaluated tests. Consistent with pure random sampling variance. "
                "Do NOT promote, report, or select this feature/slice."
            )

        if small_cell_warning:
            agent_directive += (
                f" [WARNING: Expected co-occurrence E11 = {e11:.2f} < 5. "
                "Sparse cell counts make asymptotic Chi-squared anti-conservative; "
                "run PermutationObject to verify with exact Monte-Carlo resamples.]"
            )

        result = {
            "verdict": verdict,
            "observed_rig": r,
            "evt_expected_noise_max": float(tau_evt),
            "bonferroni_significance_gate": float(tau_sig),
            "cumulative_session_tests_M": int(m),
            "batch_tests_added": int(batch_size),
            "small_cell_warning": small_cell_warning,
            "expected_minority_cooccurrence_E11": e11,
            "feature_name": feature_name,
            "slice_description": slice_description,
            "agent_directive": agent_directive,
            "session_id": session_id,
        }

        session.history.append(result)
        return result

    def audit_dataframe(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        candidate_cols: Sequence[str],
        df_val: Optional[pd.DataFrame] = None,
        n_bootstraps: int = 100,
        alpha: float = 0.05,
        session_id: str = "default",
        seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Audits a candidate feature pool against selection bias and stability tiers."""
        session = self.get_or_create_session(session_id=session_id, target_name=target_col)

        candidate_cols = [c for c in candidate_cols if c in df_train.columns and c != target_col]
        if not candidate_cols:
            raise ValueError("No valid candidate feature columns found in df_train.")

        # 1. Update hypothesis count
        session.cumulative_hypotheses_M += len(candidate_cols)
        m = session.cumulative_hypotheses_M

        n = len(df_train)
        target_series = df_train[target_col]
        val_counts = target_series.value_counts(normalize=True)
        minority_p = float(val_counts.min()) if len(val_counts) > 1 else 0.5

        estimator = FeatureNoiseThreshold(nrows=n, target_distribution=minority_p)
        current_tau_sig = estimator.analytical_threshold(n_features_evaluated=m, alpha=alpha)
        current_tau_evt = estimator.evt_expected_max(n_features_evaluated=m)

        # 2. Bias estimation
        if df_val is not None:
            df_bias = FeatureDebiaser.compute_holdout_bias(
                df_train=df_train,
                df_eval=df_val,
                target=target_col,
                feature_cols=candidate_cols,
                eval_label="val"
            )
            raw_bias = float(df_bias["val_bias"].median())
            estimated_optimism_bias = float(max(raw_bias, current_tau_evt))
        else:
            # Estimate bias via Out-of-Fold cross-validation bounded by EVT noise ceiling
            df_oof = FeatureDebiaser.compute_oof_metric(
                df=df_train,
                target=target_col,
                feature_cols=candidate_cols,
                n_splits=5,
                seed=seed
            )
            raw_bias = float(df_oof["oof_bias"].median())
            estimated_optimism_bias = float(max(raw_bias, current_tau_evt))

        # 3. Bootstrap stability classification
        df_stab = FeatureDebiaser.bootstrap_stability_analysis(
            df=df_train,
            target=target_col,
            feature_cols=candidate_cols,
            estimated_bias=estimated_optimism_bias,
            n_bootstraps=n_bootstraps,
            seed=seed
        )

        promoted_2_sigma = df_stab[df_stab["is_2_sigma"]]["feature"].tolist()
        marginal_1_sigma = df_stab[df_stab["is_1_sigma"] & ~df_stab["is_2_sigma"]]["feature"].tolist()
        rejected_noise = df_stab[~df_stab["is_1_sigma"]]["feature"].tolist()

        audit_summary = {
            "session_id": session_id,
            "target_col": target_col,
            "sample_size_N": n,
            "minority_fraction": minority_p,
            "features_audited_in_batch": len(candidate_cols),
            "cumulative_session_tests_M": m,
            "current_bonferroni_gate": float(current_tau_sig),
            "current_evt_noise_floor": float(current_tau_evt),
            "estimated_optimism_bias": estimated_optimism_bias,
            "promoted_2_sigma_features": promoted_2_sigma,
            "marginal_1_sigma_features": marginal_1_sigma,
            "rejected_noise_features": rejected_noise,
            "stability_table": df_stab.to_dict(orient="records"),
            "agent_directive": (
                f"AUDIT COMPLETE: Promoted {len(promoted_2_sigma)} / {len(candidate_cols)} candidate features "
                f"clearing 2-Sigma stability after subtracting optimism bias ({estimated_optimism_bias:.5f}). "
                f"Discard {len(rejected_noise)} features identified as statistical noise."
            )
        }

        return audit_summary

    def audit_dataframe_candidates(
        self,
        train_csv_or_parquet_path: str,
        target_col: str,
        candidate_cols: Sequence[str],
        val_csv_or_parquet_path: Optional[str] = None,
        n_bootstraps: int = 100,
        alpha: float = 0.05,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Audits candidate features from CSV or Parquet files."""
        # Load train
        if train_csv_or_parquet_path.endswith(".parquet"):
            df_train = pd.read_parquet(train_csv_or_parquet_path)
        else:
            df_train = pd.read_csv(train_csv_or_parquet_path)

        # Load val if provided
        df_val = None
        if val_csv_or_parquet_path is not None:
            if val_csv_or_parquet_path.endswith(".parquet"):
                df_val = pd.read_parquet(val_csv_or_parquet_path)
            else:
                df_val = pd.read_csv(val_csv_or_parquet_path)

        return self.audit_dataframe(
            df_train=df_train,
            target_col=target_col,
            candidate_cols=candidate_cols,
            df_val=df_val,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            session_id=session_id
        )


# Global singleton ledger instance
GLOBAL_LEDGER = NoiseGateSessionLedger()


# Optional FastMCP Server wrapper
try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    FastMCP = None
    HAS_MCP = False


def create_mcp_server(ledger: Optional[NoiseGateSessionLedger] = None):
    """Factory function creating an MCP FastMCP server instance if mcp is installed."""
    if not HAS_MCP:
        return None

    active_ledger = ledger or GLOBAL_LEDGER
    mcp_app = FastMCP(
        "SignalGate",
        instructions=(
            "Stateful statistical guardrail server preventing automated p-hacking, "
            "the Winner's Curse, and spurious feature/slice selection in autonomous data science agents."
        )
    )

    @mcp_app.tool()
    def evaluate_feature_or_slice(
        observed_rig: Optional[float] = None,
        nrows: Optional[int] = None,
        minority_frac: Optional[float] = None,
        feature_support: Optional[float] = None,
        contingency_counts: Optional[List[int]] = None,
        hypotheses_in_batch: int = 1,
        alpha: float = 0.05,
        session_id: str = "default",
        feature_name: Optional[str] = None,
        slice_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates whether an observed RIG or 2x2 slice clears statistical noise gates."""
        return active_ledger.evaluate_feature_or_slice(
            observed_rig=observed_rig,
            nrows=nrows,
            minority_frac=minority_frac,
            feature_support=feature_support,
            contingency_counts=contingency_counts,
            hypotheses_in_batch=hypotheses_in_batch,
            alpha=alpha,
            session_id=session_id,
            feature_name=feature_name,
            slice_description=slice_description
        )

    @mcp_app.tool()
    def audit_dataframe_candidates(
        train_csv_or_parquet_path: str,
        target_col: str,
        candidate_cols: List[str],
        val_csv_or_parquet_path: Optional[str] = None,
        n_bootstraps: int = 100,
        alpha: float = 0.05,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Audits candidate features from CSV or Parquet files against selection bias."""
        return active_ledger.audit_dataframe_candidates(
            train_csv_or_parquet_path=train_csv_or_parquet_path,
            target_col=target_col,
            candidate_cols=candidate_cols,
            val_csv_or_parquet_path=val_csv_or_parquet_path,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            session_id=session_id
        )

    @mcp_app.tool()
    def get_search_budget_status(
        session_id: str = "default",
        nrows: Optional[int] = None,
        minority_frac: Optional[float] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Returns current hypothesis testing volume, noise floors, and gate inflation."""
        return active_ledger.get_search_budget_status(
            session_id=session_id,
            nrows=nrows,
            minority_frac=minority_frac,
            alpha=alpha
        )

    @mcp_app.tool()
    def reset_session_budget(session_id: str = "default") -> Dict[str, Any]:
        """Resets the search budget and test counter for a session."""
        return active_ledger.reset_session_budget(session_id=session_id)

    return mcp_app


def main():
    """CLI entry point for running the MCP server."""
    if not HAS_MCP:
        import sys
        sys.stderr.write(
            "Error: The 'mcp' package is required to run the MCP server.\n"
            "Install it via: pip install \"feature-noise-gate[mcp]\" or pip install mcp\n"
        )
        sys.exit(1)

    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
