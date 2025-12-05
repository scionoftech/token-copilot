"""LangChain integration for token_copilot."""

from .callbacks import TokenCoPilotCallback

# Backward compatibility alias
TokenPilotCallback = TokenCoPilotCallback

__all__ = ["TokenCoPilotCallback", "TokenPilotCallback"]
