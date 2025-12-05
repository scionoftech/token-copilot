"""Base plugin system for TokenCoPilot."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .copilot import TokenCoPilot


class Plugin(ABC):
    """Base class for all TokenCoPilot plugins.

    Plugins extend TokenCoPilot functionality in a modular way. Each plugin
    can hook into the LLM lifecycle events and add specific capabilities.

    Example:
        >>> class CustomPlugin(Plugin):
        ...     def on_llm_end(self, llm_output, run_id, **kwargs):
        ...         print(f"LLM call completed: {llm_output}")
        ...
        >>> copilot = TokenCoPilot()
        >>> copilot.add_plugin(CustomPlugin())
    """

    def __init__(self):
        """Initialize plugin."""
        self.copilot: Optional['TokenCoPilot'] = None
        self.enabled = True

    def attach(self, copilot: 'TokenCoPilot'):
        """Attach plugin to a TokenCoPilot instance.

        Args:
            copilot: TokenCoPilot instance to attach to
        """
        self.copilot = copilot
        self.on_attach()

    def on_attach(self):
        """Called when plugin is attached to copilot.

        Override this to perform initialization that requires access
        to the copilot instance.
        """
        pass

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Any,
        **kwargs: Any
    ):
        """Called when LLM starts.

        Args:
            serialized: Serialized LLM
            prompts: List of prompts
            run_id: Run identifier
            **kwargs: Additional keyword arguments (includes metadata)
        """
        pass

    def on_llm_end(
        self,
        response: Any,
        run_id: Any,
        **kwargs: Any
    ):
        """Called when LLM ends successfully.

        Args:
            response: LLM response
            run_id: Run identifier
            **kwargs: Additional keyword arguments (includes metadata)
        """
        pass

    def on_llm_error(
        self,
        error: Exception,
        run_id: Any,
        **kwargs: Any
    ):
        """Called when LLM encounters an error.

        Args:
            error: Exception that occurred
            run_id: Run identifier
            **kwargs: Additional keyword arguments
        """
        pass

    def on_cost_tracked(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        metadata: Dict[str, Any]
    ):
        """Called after cost is tracked.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            metadata: Request metadata
        """
        pass

    def enable(self):
        """Enable this plugin."""
        self.enabled = True

    def disable(self):
        """Disable this plugin."""
        self.enabled = False


class PluginManager:
    """Manages plugins for TokenCoPilot."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: List[Plugin] = []

    def add(self, plugin: Plugin, copilot: 'TokenCoPilot'):
        """Add a plugin.

        Args:
            plugin: Plugin to add
            copilot: TokenCoPilot instance
        """
        plugin.attach(copilot)
        self.plugins.append(plugin)

    def remove(self, plugin: Plugin):
        """Remove a plugin.

        Args:
            plugin: Plugin to remove
        """
        if plugin in self.plugins:
            self.plugins.remove(plugin)

    def get_plugins(self, plugin_type: type = None) -> List[Plugin]:
        """Get all plugins or plugins of a specific type.

        Args:
            plugin_type: Optional plugin type to filter by

        Returns:
            List of plugins
        """
        if plugin_type is None:
            return self.plugins
        return [p for p in self.plugins if isinstance(p, plugin_type)]

    def call_on_llm_start(self, serialized, prompts, run_id, **kwargs):
        """Call on_llm_start on all enabled plugins."""
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    plugin.on_llm_start(serialized, prompts, run_id, **kwargs)
                except Exception as e:
                    # Log but don't fail
                    import logging
                    logging.warning(f"Plugin {type(plugin).__name__} failed on_llm_start: {e}")

    def call_on_llm_end(self, response, run_id, **kwargs):
        """Call on_llm_end on all enabled plugins."""
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    plugin.on_llm_end(response, run_id, **kwargs)
                except Exception as e:
                    import logging
                    logging.warning(f"Plugin {type(plugin).__name__} failed on_llm_end: {e}")

    def call_on_llm_error(self, error, run_id, **kwargs):
        """Call on_llm_error on all enabled plugins."""
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    plugin.on_llm_error(error, run_id, **kwargs)
                except Exception as e:
                    import logging
                    logging.warning(f"Plugin {type(plugin).__name__} failed on_llm_error: {e}")

    def call_on_cost_tracked(self, model, input_tokens, output_tokens, cost, metadata):
        """Call on_cost_tracked on all enabled plugins."""
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    plugin.on_cost_tracked(model, input_tokens, output_tokens, cost, metadata)
                except Exception as e:
                    import logging
                    logging.warning(f"Plugin {type(plugin).__name__} failed on_cost_tracked: {e}")
