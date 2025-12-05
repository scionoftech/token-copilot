"""LlamaIndex integration for token_copilot."""

from .callbacks import TokenCoPilotCallbackHandler

# Backward compatibility alias
TokenPilotCallbackHandler = TokenCoPilotCallbackHandler

__all__ = ["TokenCoPilotCallbackHandler", "TokenPilotCallbackHandler"]
