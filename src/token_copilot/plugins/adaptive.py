"""Adaptive plugin for budget-aware LLM operations."""

from typing import Any, Dict
from ..core.plugin import Plugin


class AdaptivePlugin(Plugin):
    """Plugin for adaptive token operations based on budget tier.

    Automatically adjusts LLM parameters based on remaining budget:
    - ABUNDANT (>75%): Premium settings
    - COMFORTABLE (50-75%): Balanced approach
    - MODERATE (25-50%): Start optimizing
    - LOW (10-25%): Aggressive optimization
    - CRITICAL (<10%): Minimal token usage

    Example:
        >>> from token_copilot import TokenCoPilot
        >>> from token_copilot.plugins import AdaptivePlugin
        >>>
        >>> copilot = TokenCoPilot(budget_limit=100.00)
        >>> copilot.add_plugin(AdaptivePlugin())
        >>>
        >>> # Get adaptive operations
        >>> adaptive = copilot.get_plugins(AdaptivePlugin)[0]
        >>> adaptive_ops = adaptive.operations
        >>>
        >>> # Use adaptive generate
        >>> result = adaptive_ops.generate(llm, "Explain AI")

    Example (Builder):
        >>> copilot = (TokenCoPilot(budget_limit=100.00)
        ...     .with_adaptive()
        ... )
    """

    def __init__(self, enable_logging: bool = True):
        """Initialize adaptive plugin.

        Args:
            enable_logging: Log adaptive parameter decisions
        """
        super().__init__()
        self.enable_logging = enable_logging
        self._operations = None

    def on_attach(self):
        """Initialize adaptive operations when attached."""
        try:
            from ..adaptive import TokenAwareOperations

            # Create adapter to use TokenCoPilot with TokenAwareOperations
            # TokenAwareOperations expects TokenCoPilotCallback, so we create a simple wrapper
            class CopilotAdapter:
                """Adapter to make TokenCoPilot compatible with TokenAwareOperations."""
                def __init__(self, copilot):
                    self.copilot = copilot

                @property
                def budget_limit(self):
                    return self.copilot.budget_limit

                def get_total_cost(self):
                    return self.copilot.get_total_cost()

                @property
                def tracker(self):
                    return self.copilot.tracker

            adapter = CopilotAdapter(self.copilot)
            self._operations = TokenAwareOperations(adapter, enable_logging=self.enable_logging)

        except ImportError as e:
            import logging
            logging.warning(
                f"Adaptive plugin requires LangChain: {e}. "
                "Install with: pip install langchain"
            )

    @property
    def operations(self):
        """Get TokenAwareOperations instance.

        Returns:
            TokenAwareOperations for adaptive LLM calls

        Example:
            >>> adaptive = copilot.get_plugins(AdaptivePlugin)[0]
            >>> ops = adaptive.operations
            >>> result = ops.generate(llm, "prompt")
        """
        return self._operations

    def get_current_tier(self):
        """Get current budget tier.

        Returns:
            BudgetTier enum value

        Example:
            >>> adaptive = copilot.get_plugins(AdaptivePlugin)[0]
            >>> tier = adaptive.get_current_tier()
            >>> print(f"Current tier: {tier.value}")
        """
        if not self._operations:
            return None
        return self._operations.get_current_tier()

    def get_tier_info(self) -> Dict[str, Any]:
        """Get detailed budget tier information.

        Returns:
            Dictionary with tier details and budget stats

        Example:
            >>> adaptive = copilot.get_plugins(AdaptivePlugin)[0]
            >>> info = adaptive.get_tier_info()
            >>> print(f"Tier: {info['tier_name']}")
            >>> print(f"Remaining: ${info['remaining']:.2f}")
        """
        if not self._operations:
            return {}
        return self._operations.get_tier_info()
