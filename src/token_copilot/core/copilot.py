"""Core TokenCoPilot class - minimal LLM cost tracking callback."""

from typing import Any, Dict, List, Optional
import logging

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        BaseCallbackHandler = object

from ..tracking import MultiTenantTracker, BudgetEnforcer
from .plugin import Plugin, PluginManager

logger = logging.getLogger(__name__)


class TokenCoPilot(BaseCallbackHandler):
    """Minimal LLM cost tracking with plugin-based extensibility.

    TokenCoPilot provides core cost tracking and budget enforcement,
    with a plugin system for adding advanced features.

    Core Features:
        - Automatic token and cost tracking
        - Budget enforcement with hard limits
        - Multi-tenant support (track by user, org, session, etc.)
        - DataFrame export for analytics

    Advanced Features (via Plugins):
        - Real-time streaming (StreamingPlugin)
        - Anomaly detection (AnalyticsPlugin)
        - Model routing (RoutingPlugin)
        - Adaptive operations (AdaptivePlugin)
        - Budget forecasting (ForecastingPlugin)

    Example (Basic):
        >>> from token_copilot import TokenCoPilot
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> copilot = TokenCoPilot(budget_limit=10.00)
        >>> llm = ChatOpenAI(callbacks=[copilot])
        >>> result = llm.invoke("Hello")
        >>>
        >>> print(f"Cost: ${copilot.cost:.4f}")
        >>> print(f"Tokens: {copilot.tokens}")

    Example (With Plugins):
        >>> from token_copilot import TokenCoPilot
        >>> from token_copilot.plugins import StreamingPlugin, AnalyticsPlugin
        >>>
        >>> copilot = TokenCoPilot(budget_limit=100.00)
        >>> copilot.add_plugin(StreamingPlugin(webhook_url="..."))
        >>> copilot.add_plugin(AnalyticsPlugin(detect_anomalies=True))
        >>>
        >>> llm = ChatOpenAI(callbacks=[copilot])

    Example (Builder Pattern):
        >>> copilot = (TokenCoPilot(budget_limit=100.00)
        ...     .with_streaming(webhook_url="...")
        ...     .with_analytics(detect_anomalies=True)
        ...     .with_adaptive()
        ...     .build()
        ... )
    """

    def __init__(
        self,
        budget_limit: Optional[float] = None,
        budget_period: str = "total",
        on_budget_exceeded: str = "raise",
        **kwargs
    ):
        """Initialize TokenCoPilot.

        Args:
            budget_limit: Budget limit in USD (None = no limit)
            budget_period: Budget period ("total", "daily", "monthly", "per_user", "per_org")
            on_budget_exceeded: Action when budget exceeded ("raise", "warn", "ignore")
            **kwargs: Additional callback handler arguments
        """
        super().__init__(**kwargs)

        # Core tracking
        self.tracker = MultiTenantTracker()
        self.budget_enforcer = BudgetEnforcer(
            limit=budget_limit,
            period=budget_period,
            on_exceeded=on_budget_exceeded,
        )

        # Plugin system
        self._plugin_manager = PluginManager()

        # Metadata for current request
        self._current_metadata: Dict[str, Any] = {}

        # Track run IDs for token counting
        self._run_prompts: Dict[str, List[str]] = {}

    # =========================================================================
    # Plugin Management
    # =========================================================================

    def add_plugin(self, plugin: Plugin) -> 'TokenCoPilot':
        """Add a plugin to extend functionality.

        Args:
            plugin: Plugin instance to add

        Returns:
            Self for method chaining

        Example:
            >>> copilot = TokenCoPilot()
            >>> copilot.add_plugin(StreamingPlugin(webhook_url="..."))
            >>> copilot.add_plugin(AnalyticsPlugin())
        """
        self._plugin_manager.add(plugin, self)
        return self

    def remove_plugin(self, plugin: Plugin) -> 'TokenCoPilot':
        """Remove a plugin.

        Args:
            plugin: Plugin instance to remove

        Returns:
            Self for method chaining
        """
        self._plugin_manager.remove(plugin)
        return self

    def get_plugins(self, plugin_type: type = None) -> List[Plugin]:
        """Get all plugins or plugins of a specific type.

        Args:
            plugin_type: Optional plugin type to filter by

        Returns:
            List of plugins
        """
        return self._plugin_manager.get_plugins(plugin_type)

    # =========================================================================
    # Builder Pattern Methods
    # =========================================================================

    def with_streaming(self, **kwargs) -> 'TokenCoPilot':
        """Add streaming plugin (fluent interface).

        Args:
            **kwargs: StreamingPlugin arguments (webhook_url, kafka_brokers, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> copilot = (TokenCoPilot(budget_limit=100.00)
            ...     .with_streaming(webhook_url="https://example.com/webhook")
            ... )
        """
        from ..plugins.streaming import StreamingPlugin
        self.add_plugin(StreamingPlugin(**kwargs))
        return self

    def with_analytics(self, **kwargs) -> 'TokenCoPilot':
        """Add analytics plugin (fluent interface).

        Args:
            **kwargs: AnalyticsPlugin arguments

        Returns:
            Self for method chaining

        Example:
            >>> copilot = (TokenCoPilot(budget_limit=100.00)
            ...     .with_analytics(detect_anomalies=True)
            ... )
        """
        from ..plugins.analytics import AnalyticsPlugin
        self.add_plugin(AnalyticsPlugin(**kwargs))
        return self

    def with_routing(self, **kwargs) -> 'TokenCoPilot':
        """Add routing plugin (fluent interface).

        Args:
            **kwargs: RoutingPlugin arguments

        Returns:
            Self for method chaining

        Example:
            >>> copilot = (TokenCoPilot(budget_limit=100.00)
            ...     .with_routing(models=[...], strategy="balanced")
            ... )
        """
        from ..plugins.routing import RoutingPlugin
        self.add_plugin(RoutingPlugin(**kwargs))
        return self

    def with_adaptive(self, **kwargs) -> 'TokenCoPilot':
        """Add adaptive operations plugin (fluent interface).

        Args:
            **kwargs: AdaptivePlugin arguments

        Returns:
            Self for method chaining

        Example:
            >>> copilot = (TokenCoPilot(budget_limit=100.00)
            ...     .with_adaptive()
            ... )
        """
        from ..plugins.adaptive import AdaptivePlugin
        self.add_plugin(AdaptivePlugin(**kwargs))
        return self

    def with_forecasting(self, **kwargs) -> 'TokenCoPilot':
        """Add forecasting plugin (fluent interface).

        Args:
            **kwargs: ForecastingPlugin arguments

        Returns:
            Self for method chaining
        """
        from ..plugins.forecasting import ForecastingPlugin
        self.add_plugin(ForecastingPlugin(**kwargs))
        return self

    def with_persistence(self, **kwargs) -> 'TokenCoPilot':
        """Add persistence plugin (fluent interface).

        Args:
            **kwargs: PersistencePlugin arguments
                backend: PersistenceBackend to use (SQLiteBackend or JSONBackend)
                session_id: Optional session identifier
                user_id: Optional user identifier
                auto_flush: Whether to save immediately (default: True)

        Returns:
            Self for method chaining

        Example:
            >>> from token_copilot.plugins.persistence import SQLiteBackend
            >>> copilot = (TokenCoPilot(budget_limit=100.00)
            ...     .with_persistence(backend=SQLiteBackend("costs.db"))
            ... )
        """
        from ..plugins.persistence import PersistencePlugin
        self.add_plugin(PersistencePlugin(**kwargs))
        return self

    def build(self) -> 'TokenCoPilot':
        """Finalize builder (optional, for clarity).

        Returns:
            Self

        Example:
            >>> copilot = (TokenCoPilot(budget_limit=100.00)
            ...     .with_streaming(webhook_url="...")
            ...     .with_analytics()
            ...     .build()  # Optional but makes intent clear
            ... )
        """
        return self

    # =========================================================================
    # LangChain Callback Handlers
    # =========================================================================

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Any,
        **kwargs: Any
    ):
        """Called when LLM starts."""
        # Extract metadata
        metadata = kwargs.get("metadata", {})
        self._current_metadata = metadata

        # Store prompts for token counting
        self._run_prompts[str(run_id)] = prompts

        # Notify plugins
        self._plugin_manager.call_on_llm_start(serialized, prompts, run_id, **kwargs)

    def on_llm_end(
        self,
        response: Any,
        run_id: Any,
        **kwargs: Any
    ):
        """Called when LLM ends successfully."""
        try:
            # Extract token usage and model
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})

            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
            model = llm_output.get("model_name", "unknown")

            # Debug: Log model name for troubleshooting
            if model == "unknown":
                logger.debug(f"Unknown model detected. llm_output keys: {list(llm_output.keys())}")
                logger.debug(f"Full llm_output: {llm_output}")

            # Track cost
            entry = self.tracker.track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=self._current_metadata,
            )

            # Add cost to budget
            self.budget_enforcer.add(entry.cost, self._current_metadata)

            # Enforce budget
            self.budget_enforcer.check(self._current_metadata)

            # Notify plugins
            self._plugin_manager.call_on_llm_end(response, run_id, **kwargs)
            self._plugin_manager.call_on_cost_tracked(
                model=entry.model,
                input_tokens=entry.input_tokens,
                output_tokens=entry.output_tokens,
                cost=entry.cost,
                metadata=self._current_metadata,
            )

        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
            raise
        finally:
            # Cleanup
            self._run_prompts.pop(str(run_id), None)
            self._current_metadata = {}

    def on_llm_error(
        self,
        error: Exception,
        run_id: Any,
        **kwargs: Any
    ):
        """Called when LLM encounters an error."""
        # Notify plugins
        self._plugin_manager.call_on_llm_error(error, run_id, **kwargs)

        # Cleanup
        self._run_prompts.pop(str(run_id), None)
        self._current_metadata = {}

    # =========================================================================
    # Public API - Core Metrics
    # =========================================================================

    @property
    def cost(self) -> float:
        """Total cost in USD."""
        return self.get_total_cost()

    @property
    def tokens(self) -> int:
        """Total tokens used."""
        return self.get_total_tokens()

    @property
    def budget_limit(self) -> Optional[float]:
        """Budget limit in USD."""
        return self.budget_enforcer.limit

    def get_total_cost(self) -> float:
        """Get total cost across all LLM calls."""
        return self.tracker.get_total_cost()

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return self.tracker.get_total_tokens()

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with cost, token, and call statistics
        """
        return self.tracker.get_stats()

    def get_remaining_budget(self, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Get remaining budget.

        Args:
            metadata: Optional metadata for per-user/per-org budgets

        Returns:
            Remaining budget in USD (float('inf') if no limit)
        """
        return self.budget_enforcer.get_remaining(metadata)

    def get_costs_by(self, dimension: str) -> Dict[str, float]:
        """Get costs grouped by dimension.

        Args:
            dimension: Dimension to group by ("user_id", "org_id", "model", etc.)

        Returns:
            Dictionary mapping dimension values to costs
        """
        costs = {}
        for entry in self.tracker.entries:
            key = getattr(entry, dimension, None) or entry.metadata.get(dimension, "unknown")
            costs[key] = costs.get(key, 0.0) + entry.cost
        return costs

    def to_dataframe(self):
        """Export to pandas DataFrame.

        Returns:
            DataFrame with all cost entries

        Raises:
            ImportError: If pandas is not installed
        """
        return self.tracker.to_dataframe()

    def reset(self):
        """Reset all tracking data."""
        self.tracker.reset()
