"""Backward compatibility forwarding for agent_sandbox."""
from signalgate.agent_sandbox import (
    BASE_DOMAIN_COLUMNS,
    generate_benchmark_datasets,
    InstrumentedAgentSandbox,
    run_exploratory_agent_simulation,
    run_monte_carlo_benchmark,
)

__all__ = [
    "BASE_DOMAIN_COLUMNS",
    "generate_benchmark_datasets",
    "InstrumentedAgentSandbox",
    "run_exploratory_agent_simulation",
    "run_monte_carlo_benchmark",
]
