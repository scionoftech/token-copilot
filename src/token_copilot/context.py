"""Context managers for convenient TokenCoPilot usage."""

from contextlib import contextmanager
from typing import Optional
from .core import TokenCoPilot


@contextmanager
def track_costs(
    budget_limit: Optional[float] = None,
    on_budget_exceeded: str = "raise",
    **plugin_kwargs
):
    """Context manager for tracking LLM costs.

    Automatically creates a TokenCoPilot instance and provides it within the context.
    Useful for tracking costs in a specific code block.

    Args:
        budget_limit: Optional budget limit in USD
        on_budget_exceeded: Action when budget exceeded ("raise", "warn", "ignore")
        **plugin_kwargs: Additional plugin configuration

    Yields:
        TokenCoPilot instance

    Example:
        >>> from token_copilot import track_costs
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> with track_costs(budget_limit=5.00) as copilot:
        ...     llm = ChatOpenAI(callbacks=[copilot])
        ...     result = llm.invoke("Hello")
        ...     print(f"Cost: ${copilot.cost:.4f}")

    Example (with plugins):
        >>> with track_costs(
        ...     budget_limit=10.00,
        ...     webhook_url="https://example.com/webhook"
        ... ) as copilot:
        ...     llm = ChatOpenAI(callbacks=[copilot])
        ...     result = llm.invoke("Hello")
    """
    copilot = TokenCoPilot(
        budget_limit=budget_limit,
        on_budget_exceeded=on_budget_exceeded
    )

    # Configure plugins from kwargs
    if plugin_kwargs.get('webhook_url') or plugin_kwargs.get('kafka_brokers'):
        streaming_kwargs = {
            k: v for k, v in plugin_kwargs.items()
            if k in ['webhook_url', 'kafka_brokers', 'kafka_topic', 'syslog_host', 'logstash_host', 'otlp_endpoint']
        }
        if streaming_kwargs:
            copilot.with_streaming(**streaming_kwargs)

    if plugin_kwargs.get('detect_anomalies'):
        analytics_kwargs = {
            k: v for k, v in plugin_kwargs.items()
            if k in ['detect_anomalies', 'anomaly_sensitivity', 'alert_handlers']
        }
        copilot.with_analytics(**analytics_kwargs)

    if plugin_kwargs.get('enable_adaptive'):
        copilot.with_adaptive()

    try:
        yield copilot
    finally:
        # Optional: log summary on exit
        if hasattr(copilot, 'get_stats'):
            import logging
            logger = logging.getLogger(__name__)
            stats = copilot.get_stats()
            logger.debug(
                f"Track costs context exited. "
                f"Total cost: ${stats.get('total_cost', 0):.4f}, "
                f"Total calls: {stats.get('total_calls', 0)}"
            )


@contextmanager
def with_budget(
    limit: float,
    on_exceeded: str = "raise",
    warn_at: Optional[float] = None,
):
    """Context manager focused on budget enforcement.

    Simpler than track_costs, focused specifically on budget limits.

    Args:
        limit: Budget limit in USD
        on_exceeded: Action when budget exceeded ("raise", "warn", "ignore")
        warn_at: Optional threshold to warn at (e.g., 0.8 for 80%)

    Yields:
        TokenCoPilot instance

    Example:
        >>> from token_copilot import with_budget
        >>>
        >>> with with_budget(limit=10.00, warn_at=0.8) as budget:
        ...     llm = ChatOpenAI(callbacks=[budget])
        ...
        ...     for task in tasks:
        ...         if budget.get_remaining_budget() > 0:
        ...             result = llm.invoke(task)
        ...         else:
        ...             print("Budget exhausted!")
        ...             break
    """
    copilot = TokenCoPilot(
        budget_limit=limit,
        on_budget_exceeded=on_exceeded
    )

    # Set up warning threshold if specified
    if warn_at is not None:
        warn_amount = limit * warn_at
        original_tracker = copilot.tracker

        class WarningTracker:
            """Wrapper to add budget warnings."""
            def __init__(self, tracker, warn_at_cost):
                self._tracker = tracker
                self._warn_at_cost = warn_at_cost
                self._warned = False

            def __getattr__(self, name):
                return getattr(self._tracker, name)

            def track_cost(self, *args, **kwargs):
                entry = self._tracker.track(*args, **kwargs)
                if not self._warned and self._tracker.get_total_cost() >= self._warn_at_cost:
                    import logging
                    logging.warning(
                        f"Budget warning: ${self._tracker.get_total_cost():.2f} / ${limit:.2f} "
                        f"({self._tracker.get_total_cost()/limit*100:.0f}%) used"
                    )
                    self._warned = True
                return entry

        copilot.tracker = WarningTracker(original_tracker, warn_amount)

    try:
        yield copilot
    finally:
        remaining = copilot.get_remaining_budget()
        used = limit - remaining
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Budget context exited. "
            f"Used: ${used:.4f} / ${limit:.2f} ({used/limit*100:.1f}%)"
        )


@contextmanager
def monitored(
    budget_limit: Optional[float] = None,
    name: str = "llm_operation",
    log_on_exit: bool = True,
):
    """Context manager for monitored LLM operations with automatic logging.

    Tracks costs for a named operation and logs results on exit.

    Args:
        budget_limit: Optional budget limit in USD
        name: Name of the operation for logging
        log_on_exit: Log summary when exiting context (default: True)

    Yields:
        TokenCoPilot instance

    Example:
        >>> from token_copilot import monitored
        >>>
        >>> with monitored(budget_limit=5.00, name="document_processing") as copilot:
        ...     llm = ChatOpenAI(callbacks=[copilot])
        ...     for doc in documents:
        ...         result = llm.invoke(f"Summarize: {doc}")
        ... # Logs: "Operation [document_processing]: Cost=$2.34, Tokens=1500, Calls=10"
    """
    import logging
    import time

    logger = logging.getLogger(__name__)

    copilot = TokenCoPilot(budget_limit=budget_limit)
    start_time = time.time()

    logger.info(f"Starting monitored operation: {name}")

    try:
        yield copilot
    finally:
        duration = time.time() - start_time
        stats = copilot.get_stats()

        if log_on_exit:
            logger.info(
                f"Operation [{name}] completed: "
                f"Cost=${stats.get('total_cost', 0):.4f}, "
                f"Tokens={stats.get('total_tokens', 0)}, "
                f"Calls={stats.get('total_calls', 0)}, "
                f"Duration={duration:.2f}s"
            )
