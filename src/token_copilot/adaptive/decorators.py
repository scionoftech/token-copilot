"""Decorators for token-aware operations."""

import functools
import logging
from typing import Any, Callable, Optional
from .tiers import BudgetTier, classify_budget_tier

try:
    from ..langchain import TokenCoPilotCallback
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    TokenCoPilotCallback = None  # type: ignore

logger = logging.getLogger(__name__)


def token_aware(
    callback: Optional['TokenCoPilotCallback'] = None,
    operation: str = 'generate',
    enable_logging: bool = True
) -> Callable:
    """Decorator that makes a function token-aware and adaptive to budget tiers.

    The decorated function will have its behavior adjusted based on the current
    budget tier. Use with the adaptive_context manager to set the callback.

    Args:
        callback: TokenCoPilotCallback instance (can be set via context manager)
        operation: Type of operation ('generate', 'search', 'retry')
        enable_logging: Whether to log adaptive behavior

    Returns:
        Decorated function that adapts based on budget tier

    Example:
        >>> from token_copilot.adaptive import token_aware, adaptive_context
        >>>
        >>> @token_aware(operation='generate')
        >>> def my_generation_task(llm, prompt, max_tokens=None):
        ...     return llm.invoke(prompt, max_tokens=max_tokens)
        >>>
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> with adaptive_context(callback):
        ...     result = my_generation_task(llm, "Explain AI")
        ...     # Automatically uses tier-appropriate max_tokens
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get callback from parameter or context
            from .context import get_current_callback

            active_callback = callback or get_current_callback()

            if active_callback is None:
                if enable_logging:
                    logger.warning(
                        f"@token_aware decorator on {func.__name__} called without "
                        "callback. Use adaptive_context() or pass callback parameter. "
                        "Running without adaptive behavior."
                    )
                return func(*args, **kwargs)

            # Get current tier
            tier = classify_budget_tier(active_callback)

            # Get adaptive parameters based on tier and operation
            from .operations import TokenAwareOperations
            ops = TokenAwareOperations(active_callback, enable_logging=enable_logging)
            adaptive_params = ops._get_adaptive_params(tier, operation)

            # Merge adaptive params with user kwargs (user always wins)
            final_kwargs = adaptive_params.copy()
            user_overrides = []

            for key, value in kwargs.items():
                if key in final_kwargs:
                    user_overrides.append(key)
                final_kwargs[key] = value

            # Log if enabled
            if enable_logging:
                ops._log_adaptive_params(
                    f"{func.__name__}",
                    tier,
                    final_kwargs,
                    user_overrides
                )

            # Call original function with merged params
            return func(*args, **final_kwargs)

        return wrapper
    return decorator


def budget_gate(
    callback: Optional['TokenCoPilotCallback'] = None,
    min_tier: BudgetTier = BudgetTier.MODERATE,
    skip_on_insufficient: bool = True,
    raise_on_insufficient: bool = False,
    enable_logging: bool = True
) -> Callable:
    """Decorator that gates function execution based on minimum budget tier.

    Prevents expensive operations from running when budget is too low.

    Args:
        callback: TokenCoPilotCallback instance (can be set via context manager)
        min_tier: Minimum budget tier required to execute
        skip_on_insufficient: Return None if budget insufficient (default: True)
        raise_on_insufficient: Raise exception if budget insufficient (default: False)
        enable_logging: Whether to log gating decisions

    Returns:
        Decorated function that only runs if budget tier is sufficient

    Raises:
        RuntimeError: If raise_on_insufficient=True and budget is insufficient

    Example:
        >>> @budget_gate(min_tier=BudgetTier.COMFORTABLE)
        >>> def expensive_operation(llm, prompt):
        ...     return llm.invoke(prompt, max_tokens=5000)
        >>>
        >>> with adaptive_context(callback):
        ...     result = expensive_operation(llm, "prompt")
        ...     # Skipped if budget tier is LOW or CRITICAL
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from .context import get_current_callback

            active_callback = callback or get_current_callback()

            if active_callback is None:
                if enable_logging:
                    logger.warning(
                        f"@budget_gate on {func.__name__} called without callback. "
                        "Running without budget check."
                    )
                return func(*args, **kwargs)

            # Check current tier
            current_tier = classify_budget_tier(active_callback)

            # Define tier hierarchy
            tier_hierarchy = [
                BudgetTier.CRITICAL,
                BudgetTier.LOW,
                BudgetTier.MODERATE,
                BudgetTier.COMFORTABLE,
                BudgetTier.ABUNDANT,
            ]

            current_level = tier_hierarchy.index(current_tier)
            required_level = tier_hierarchy.index(min_tier)

            if current_level >= required_level:
                # Budget is sufficient
                if enable_logging:
                    logger.debug(
                        f"Budget gate passed for {func.__name__}: "
                        f"current={current_tier.value}, required={min_tier.value}"
                    )
                return func(*args, **kwargs)
            else:
                # Budget is insufficient
                if enable_logging:
                    logger.warning(
                        f"Budget gate blocked {func.__name__}: "
                        f"current={current_tier.value}, required={min_tier.value}"
                    )

                if raise_on_insufficient:
                    raise RuntimeError(
                        f"Insufficient budget to execute {func.__name__}. "
                        f"Current tier: {current_tier.value}, "
                        f"required: {min_tier.value}"
                    )

                if skip_on_insufficient:
                    return None

                # Fallback: run anyway (this shouldn't happen with defaults)
                return func(*args, **kwargs)

        return wrapper
    return decorator


def track_efficiency(
    callback: Optional['TokenCoPilotCallback'] = None,
    metric_name: Optional[str] = None,
    enable_logging: bool = True
) -> Callable:
    """Decorator that tracks efficiency metrics for a function.

    Records the cost and tokens used by a function for efficiency analysis.

    Args:
        callback: TokenCoPilotCallback instance (can be set via context manager)
        metric_name: Custom metric name (defaults to function name)
        enable_logging: Whether to log efficiency data

    Returns:
        Decorated function that tracks its token efficiency

    Example:
        >>> @track_efficiency(metric_name="document_summary")
        >>> def summarize_doc(llm, doc):
        ...     return llm.invoke(f"Summarize: {doc}")
        >>>
        >>> with adaptive_context(callback):
        ...     result = summarize_doc(llm, document)
        ...     # Efficiency automatically tracked
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from .context import get_current_callback

            active_callback = callback or get_current_callback()

            if active_callback is None:
                if enable_logging:
                    logger.warning(
                        f"@track_efficiency on {func.__name__} called without "
                        "callback. Running without tracking."
                    )
                return func(*args, **kwargs)

            # Record state before
            cost_before = active_callback.get_total_cost()
            tokens_before = sum(
                entry.input_tokens + entry.output_tokens
                for entry in active_callback.tracker.entries
            )

            # Execute function
            result = func(*args, **kwargs)

            # Record state after
            cost_after = active_callback.get_total_cost()
            tokens_after = sum(
                entry.input_tokens + entry.output_tokens
                for entry in active_callback.tracker.entries
            )

            # Calculate delta
            cost_delta = cost_after - cost_before
            tokens_delta = tokens_after - tokens_before

            # Log efficiency
            if enable_logging:
                name = metric_name or func.__name__
                logger.info(
                    f"Efficiency [{name}]: "
                    f"tokens={tokens_delta}, cost=${cost_delta:.6f}"
                )

            return result

        return wrapper
    return decorator
