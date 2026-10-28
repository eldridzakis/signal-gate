"""Framework-agnostic agent tool integration for LLM pipelines.

Provides ready-to-use tool definitions, JSON schemas (OpenAI / Anthropic),
and callable Python wrappers compatible with LangGraph, CrewAI, AutoGen,
or custom autonomous agent loops.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from .mcp_server import GLOBAL_LEDGER, NoiseGateSessionLedger


def get_openai_tools() -> List[Dict[str, Any]]:
    """Returns OpenAI function-calling tool schemas for statistical guardrails."""
    return [
        {
            "type": "function",
            "function": {
                "name": "evaluate_feature_or_slice",
                "description": (
                    "Evaluates whether a candidate feature, transformation, or Text-to-SQL segment/slice "
                    "is statistically significant or represents spurious random noise. "
                    "Maintains stateful multiple-testing tracking to guard against the Winner's Curse."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "observed_rig": {
                            "type": "number",
                            "description": "Relative Information Gain (RIG) of feature or slice in [0, 1]. Optional if contingency_counts provided."
                        },
                        "nrows": {
                            "type": "integer",
                            "description": "Sample size (total rows in training split). Optional if contingency_counts provided."
                        },
                        "minority_frac": {
                            "type": "number",
                            "description": "Minority class proportion (e.g. 0.20 for 20% positive rate). Optional if contingency_counts provided."
                        },
                        "feature_support": {
                            "type": "number",
                            "description": "Fraction of rows where feature/slice predicate is true. Used to check small-cell validity."
                        },
                        "contingency_counts": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional [n00, n01, n10, n11] counts where row 0 is feature=0, row 1 is feature=1."
                        },
                        "hypotheses_in_batch": {
                            "type": "integer",
                            "default": 1,
                            "description": "Number of hypotheses tested in this step (increments search budget M)."
                        },
                        "alpha": {
                            "type": "number",
                            "default": 0.05,
                            "description": "Family-wise significance level (default 0.05)."
                        },
                        "session_id": {
                            "type": "string",
                            "default": "default",
                            "description": "Session identifier for stateful tracking."
                        },
                        "feature_name": {
                            "type": "string",
                            "description": "Name or expression of the candidate feature."
                        },
                        "slice_description": {
                            "type": "string",
                            "description": "Predicate defining the slice (e.g., 'country == US and tenure > 12')."
                        },
                        "effective_M": {
                            "type": "number",
                            "description": "Optional effective number of independent hypotheses (M_eff <= M_raw) when testing collinear features."
                        },
                        "feature_values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Optional evaluated feature vector to automatically compute Frobenius collinearity-adjusted M_eff."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_search_budget_status",
                "description": "Returns current hypothesis testing volume (M), current noise floors, and stopping criteria recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "default": "default"
                        },
                        "nrows": {
                            "type": "integer",
                            "description": "Sample size N to calculate current noise gates."
                        },
                        "minority_frac": {
                            "type": "number",
                            "description": "Target minority fraction to calculate current noise gates."
                        },
                        "alpha": {
                            "type": "number",
                            "default": 0.05
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reset_session_budget",
                "description": "Resets the cumulative hypothesis counter for a session (e.g., when switching to a completely new dataset).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "default": "default"
                        }
                    }
                }
            }
        }
    ]


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """Returns Anthropic Claude tool schemas for statistical guardrails."""
    openai_tools = get_openai_tools()
    anthropic_tools = []
    for tool in openai_tools:
        fn = tool["function"]
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"]
        })
    return anthropic_tools


class AgentGuardrailTools:
    """Callable tool wrapper bundle backed by a stateful NoiseGateSessionLedger."""

    def __init__(self, ledger: Optional[NoiseGateSessionLedger] = None):
        self.ledger = ledger or GLOBAL_LEDGER

    def evaluate_feature_or_slice(self, **kwargs) -> Dict[str, Any]:
        """Evaluates a candidate feature/slice against multiple-testing noise gates."""
        return self.ledger.evaluate_feature_or_slice(**kwargs)

    def audit_dataframe_candidates(self, **kwargs) -> Dict[str, Any]:
        """Audits candidate features from files or DataFrames against selection bias."""
        return self.ledger.audit_dataframe_candidates(**kwargs)

    def get_search_budget_status(self, **kwargs) -> Dict[str, Any]:
        """Inspects current search budget, noise ceiling, and gate inflation."""
        return self.ledger.get_search_budget_status(**kwargs)

    def reset_session_budget(self, session_id: str = "default") -> Dict[str, Any]:
        """Resets the search budget and test counter for a session."""
        return self.ledger.reset_session_budget(session_id=session_id)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a tool call by name with provided arguments."""
        callables = self.to_callable_dict()
        if tool_name not in callables:
            raise ValueError(f"Unknown tool: {tool_name}. Choose from {list(callables.keys())}")
        return callables[tool_name](**arguments)

    def to_callable_dict(self) -> Dict[str, Callable]:
        """Returns a mapping of tool names to callable methods for agent dispatchers."""
        return {
            "evaluate_feature_or_slice": self.evaluate_feature_or_slice,
            "audit_dataframe_candidates": self.audit_dataframe_candidates,
            "get_search_budget_status": self.get_search_budget_status,
            "reset_session_budget": self.reset_session_budget,
        }


def create_agent_tools(ledger: Optional[NoiseGateSessionLedger] = None) -> AgentGuardrailTools:
    """Factory creating an AgentGuardrailTools instance."""
    return AgentGuardrailTools(ledger=ledger)
