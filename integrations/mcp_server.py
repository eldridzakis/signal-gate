"""Re-export of SignalGate MCP server for integrations module."""

from signalgate.mcp_server import (
    NoiseGateSessionLedger,
    SessionState,
    GLOBAL_LEDGER,
    create_mcp_server,
    main,
    HAS_MCP,
)

__all__ = [
    "NoiseGateSessionLedger",
    "SessionState",
    "GLOBAL_LEDGER",
    "create_mcp_server",
    "main",
    "HAS_MCP",
]


if __name__ == "__main__":
    main()
