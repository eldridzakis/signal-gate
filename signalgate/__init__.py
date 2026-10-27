"""SignalGate: Stateful Statistical Guardrails for Autonomous LLM Agents & AutoML.

Guarding against multiple-testing bias (The Winner's Curse), estimating closed-form
noise thresholds, and de-biasing agent insights, feature engineering, and Text-to-SQL slices.
"""

from .metrics import (
    entropy,
    rig,
    rig_all_columns,
    fast_contingency_matrix,
    fast_mutual_info_from_contingency,
)
from .threshold import (
    FeatureNoiseThreshold,
    estimate_noise_threshold,
    estimate_effective_tests,
    binary_entropy,
    target_entropy_from_weights,
)
from .permutation import (
    PermutationObject,
)
from .debias import (
    FeatureDebiaser,
)
from .mcp_server import (
    NoiseGateSessionLedger,
    create_mcp_server,
    GLOBAL_LEDGER,
)
from .agent_tools import (
    get_openai_tools,
    get_anthropic_tools,
    create_agent_tools,
    AgentGuardrailTools,
)
from .agent_sandbox import (
    BASE_DOMAIN_COLUMNS,
    generate_benchmark_datasets,
    InstrumentedAgentSandbox,
    run_exploratory_agent_simulation,
    run_monte_carlo_benchmark,
    simulate_agent_archetypes,
)
from .persona_benchmark import (
    PERSONA_PROMPTS,
    TRACKING_INSTRUCTIONS,
    RawCSVSandbox,
    SemanticLayerSandbox,
    build_governed_semantic_dataset,
    simulate_persona_run,
    run_persona_modality_benchmark,
    build_langchain_agent_runner,
)

# Convenient top-level aliases
SignalGateSessionLedger = NoiseGateSessionLedger
SignalDebiaser = FeatureDebiaser
SignalNoiseThreshold = FeatureNoiseThreshold

__version__ = "1.2.0"
__all__ = [
    "entropy",
    "rig",
    "rig_all_columns",
    "fast_contingency_matrix",
    "fast_mutual_info_from_contingency",
    "FeatureNoiseThreshold",
    "SignalNoiseThreshold",
    "estimate_noise_threshold",
    "estimate_effective_tests",
    "binary_entropy",
    "target_entropy_from_weights",
    "PermutationObject",
    "FeatureDebiaser",
    "SignalDebiaser",
    "NoiseGateSessionLedger",
    "SignalGateSessionLedger",
    "create_mcp_server",
    "GLOBAL_LEDGER",
    "get_openai_tools",
    "get_anthropic_tools",
    "create_agent_tools",
    "AgentGuardrailTools",
    "BASE_DOMAIN_COLUMNS",
    "generate_benchmark_datasets",
    "InstrumentedAgentSandbox",
    "run_exploratory_agent_simulation",
    "run_monte_carlo_benchmark",
    "simulate_agent_archetypes",
    "PERSONA_PROMPTS",
    "TRACKING_INSTRUCTIONS",
    "RawCSVSandbox",
    "SemanticLayerSandbox",
    "build_governed_semantic_dataset",
    "simulate_persona_run",
    "run_persona_modality_benchmark",
    "build_langchain_agent_runner",
]
