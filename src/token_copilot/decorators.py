"""Decorators for automatic cost tracking and budget enforcement."""

import functools
from typing import Optional, Callable
from .core import TokenCoPilot


def track_cost(
    budget_limit: Optional[float] = None,
    on_budget_exceeded: str = "raise",
    attach_to: str = "copilot",
):
    """Decorator to automatically track costs for a function.

    Creates a TokenCoPilot instance and attaches it to the decorated function.
    Useful for tracking costs of specific operations.

    Args:
        budget_limit: Optional budget limit in USD
        on_budget_exceeded: Action when budget exceeded
        attach_to: Attribute name to attach copilot to (default: "copilot")

    Returns:
        Decorated function with copilot attached

    Example:
        >>> from token_copilot.decorators import track_cost
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> @track_cost(budget_limit=5.00)
        ... def process_document(doc):
        ...     llm = ChatOpenAI(callbacks=[process_document.copilot])
        ...     return llm.invoke(f"Summarize: {doc}")
        >>>
        >>> result = process_document("My document...")
        >>> print(f"Cost: ${process_document.copilot.cost:.4f}")

    Example (multiple calls):
        >>> @track_cost(budget_limit=10.00)
        ... def analyze_text(text):
        ...     llm = ChatOpenAI(callbacks=[analyze_text.copilot])
        ...     return llm.invoke(f"Analyze: {text}")
        >>>
        >>> # Each call adds to the same copilot instance
        >>> analyze_text("First text")
        >>> analyze_text("Second text")
        >>> print(f"Total cost: ${analyze_text.copilot.cost:.4f}")
    """
    def decorator(func: Callable) -> Callable:
        # Create copilot instance once for the function
        copilot = TokenCoPilot(
            budget_limit=budget_limit,
            on_budget_exceeded=on_budget_exceeded
        )

        # Attach to function
        setattr(func, attach_to, copilot)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Just call the function - copilot is available via func.copilot
            return func(*args, **kwargs)

        # Also attach to wrapper for easy access
        setattr(wrapper, attach_to, copilot)

        return wrapper

    return decorator


def enforce_budget(
    limit: float,
    on_exceeded: str = "raise",
    reset_per_call: bool = False,
):
    """Decorator to enforce budget limits on a function.

    Creates a TokenCoPilot with strict budget enforcement and automatically
    passes it to the decorated function.

    Args:
        limit: Budget limit in USD
        on_exceeded: Action when budget exceeded ("raise", "warn", "ignore")
        reset_per_call: Reset budget for each call (default: False)

    Returns:
        Decorated function with budget enforcement

    Example:
        >>> from token_copilot.decorators import enforce_budget
        >>>
        >>> @enforce_budget(limit=1.00, on_exceeded="raise")
        ... def expensive_operation(copilot):
        ...     llm = ChatOpenAI(callbacks=[copilot])
        ...     return llm.invoke("Generate a long essay...")
        >>>
        >>> try:
        ...     result = expensive_operation()
        ... except BudgetExceededError:
        ...     print("Budget limit reached!")

    Example (reset per call):
        >>> @enforce_budget(limit=0.50, reset_per_call=True)
        ... def per_call_budget(copilot):
        ...     llm = ChatOpenAI(callbacks=[copilot])
        ...     return llm.invoke("Quick task")
        >>>
        >>> # Each call gets fresh $0.50 budget
        >>> result1 = per_call_budget()
        >>> result2 = per_call_budget()
    """
    def decorator(func: Callable) -> Callable:
        if reset_per_call:
            # Create new copilot for each call
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                copilot = TokenCoPilot(
                    budget_limit=limit,
                    on_budget_exceeded=on_exceeded
                )
                # Inject copilot as first argument if function expects it
                if func.__code__.co_varnames and func.__code__.co_varnames[0] == 'copilot':
                    return func(copilot, *args, **kwargs)
                else:
                    # Store in kwargs
                    kwargs['copilot'] = copilot
                    return func(*args, **kwargs)
        else:
            # Create copilot once, shared across calls
            copilot = TokenCoPilot(
                budget_limit=limit,
                on_budget_exceeded=on_exceeded
            )

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Inject copilot
                if func.__code__.co_varnames and func.__code__.co_varnames[0] == 'copilot':
                    return func(copilot, *args, **kwargs)
                else:
                    kwargs['copilot'] = copilot
                    return func(*args, **kwargs)

            # Attach copilot to wrapper
            wrapper.copilot = copilot

        return wrapper

    return decorator


def monitored(
    name: Optional[str] = None,
    budget_limit: Optional[float] = None,
    log_result: bool = True,
):
    """Decorator to monitor and log cost metrics for a function.

    Automatically tracks costs and logs results after each call.

    Args:
        name: Operation name for logging (default: function name)
        budget_limit: Optional budget limit
        log_result: Log cost metrics after each call (default: True)

    Returns:
        Decorated function with monitoring

    Example:
        >>> from token_copilot.decorators import monitored
        >>>
        >>> @monitored(name="document_analysis", budget_limit=10.00)
        ... def analyze_document(doc, copilot):
        ...     llm = ChatOpenAI(callbacks=[copilot])
        ...     return llm.invoke(f"Analyze: {doc}")
        >>>
        >>> result = analyze_document("My document")
        ... # Logs: "Function [document_analysis]: Cost=$0.12, Tokens=150"
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or func.__name__
        copilot = TokenCoPilot(budget_limit=budget_limit)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import logging
            logger = logging.getLogger(__name__)

            # Track initial state
            initial_cost = copilot.get_total_cost()
            initial_tokens = copilot.get_total_tokens()

            # Inject copilot
            if func.__code__.co_varnames and 'copilot' in func.__code__.co_varnames:
                kwargs['copilot'] = copilot

            # Call function
            try:
                result = func(*args, **kwargs)

                # Log metrics if enabled
                if log_result:
                    cost_delta = copilot.get_total_cost() - initial_cost
                    tokens_delta = copilot.get_total_tokens() - initial_tokens
                    logger.info(
                        f"Function [{operation_name}]: "
                        f"Cost=${cost_delta:.4f}, "
                        f"Tokens={tokens_delta}"
                    )

                return result

            except Exception as e:
                logger.error(f"Function [{operation_name}] failed: {e}")
                raise

        # Attach copilot for access
        wrapper.copilot = copilot

        return wrapper

    return decorator
