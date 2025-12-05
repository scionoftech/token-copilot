"""Routing plugin for intelligent model selection."""

from typing import Any, Dict, List, Optional
from ..core.plugin import Plugin


class RoutingPlugin(Plugin):
    """Plugin for intelligent model routing based on complexity and cost.

    Automatically suggests the optimal model based on:
    - Task complexity
    - Budget constraints
    - Quality requirements
    - Historical performance

    Strategies:
    - CHEAPEST_FIRST: Always use cheapest model
    - QUALITY_FIRST: Always use highest quality model
    - BALANCED: Balance cost and quality
    - COST_THRESHOLD: Use cheapest above threshold
    - LEARNED: Learn from historical quality scores

    Example:
        >>> from token_copilot import TokenCoPilot
        >>> from token_copilot.plugins import RoutingPlugin
        >>> from token_copilot.utils import ModelConfig
        >>>
        >>> models = [
        ...     ModelConfig("gpt-4o-mini", quality=0.7, input_cost=0.15, output_cost=0.60),
        ...     ModelConfig("gpt-4o", quality=0.9, input_cost=5.0, output_cost=15.0),
        ... ]
        >>>
        >>> copilot = TokenCoPilot(budget_limit=100.00)
        >>> copilot.add_plugin(RoutingPlugin(models=models, strategy="balanced"))

    Example (Builder):
        >>> copilot = (TokenCoPilot(budget_limit=100.00)
        ...     .with_routing(models=models, strategy="balanced")
        ... )
    """

    def __init__(
        self,
        models: Optional[List] = None,
        strategy: str = "balanced",
        enable_learning: bool = False,
    ):
        """Initialize routing plugin.

        Args:
            models: List of ModelConfig objects defining available models
            strategy: Routing strategy ("cheapest_first", "quality_first", "balanced", etc.)
            enable_learning: Enable learning from quality feedback
        """
        super().__init__()
        self.models = models or []
        self.strategy = strategy
        self.enable_learning = enable_learning
        self._router = None

    def on_attach(self):
        """Initialize router when attached."""
        if not self.models:
            import logging
            logging.warning("RoutingPlugin: No models configured")
            return

        try:
            from ..routing import ModelRouter, RoutingStrategy

            # Convert string to enum
            if isinstance(self.strategy, str):
                strategy_enum = RoutingStrategy(self.strategy.upper())
            else:
                strategy_enum = self.strategy

            self._router = ModelRouter(
                models=self.models,
                strategy=strategy_enum
            )

        except ImportError as e:
            import logging
            logging.warning(
                f"Routing plugin requires additional dependencies: {e}. "
                "Install with: pip install token-copilot[analytics]"
            )

    # Public API methods

    def suggest_model(self, prompt: str, estimated_tokens: Optional[int] = None):
        """Suggest optimal model for a given prompt.

        Args:
            prompt: The prompt text
            estimated_tokens: Estimated token count (optional)

        Returns:
            RoutingDecision object with suggested model and reasoning

        Example:
            >>> routing = copilot.get_plugins(RoutingPlugin)[0]
            >>> decision = routing.suggest_model("Write a simple email", estimated_tokens=100)
            >>> print(f"Use model: {decision.selected_model}")
            >>> print(f"Reason: {decision.reasoning}")
        """
        if not self._router:
            return None

        return self._router.route(prompt, estimated_tokens)

    def record_quality(self, model: str, quality_score: float):
        """Record quality feedback for learned routing.

        Args:
            model: Model name
            quality_score: Quality score (0.0 to 1.0)

        Example:
            >>> routing = copilot.get_plugins(RoutingPlugin)[0]
            >>> # After using a model, record how well it performed
            >>> routing.record_quality("gpt-4o-mini", quality_score=0.85)
        """
        if not self._router or not self.enable_learning:
            return

        self._router.record_quality(model, quality_score)
