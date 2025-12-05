"""Context management for adaptive token operations."""

import threading
from contextlib import contextmanager
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..langchain import TokenCoPilotCallback

# Thread-local storage for callback context
_thread_local = threading.local()


def get_current_callback() -> Optional['TokenCoPilotCallback']:
    """Get the currently active TokenCoPilotCallback from thread-local context.

    Returns:
        Active callback instance or None if no context is set

    Example:
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> with adaptive_context(callback):
        ...     current = get_current_callback()
        ...     assert current is callback
    """
    return getattr(_thread_local, 'callback', None)


def set_current_callback(callback: Optional['TokenCoPilotCallback']) -> None:
    """Set the current TokenCoPilotCallback in thread-local context.

    Args:
        callback: Callback instance to set (or None to clear)

    Example:
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> set_current_callback(callback)
        >>> current = get_current_callback()
        >>> assert current is callback
    """
    _thread_local.callback = callback


@contextmanager
def adaptive_context(callback: 'TokenCoPilotCallback'):
    """Context manager that sets a TokenCoPilotCallback for adaptive operations.

    This allows decorators and adaptive functions to access the callback
    without explicitly passing it as a parameter.

    Args:
        callback: TokenCoPilotCallback instance to use within this context

    Yields:
        The callback instance

    Example:
        >>> from token_copilot import TokenCoPilotCallback
        >>> from token_copilot.adaptive import adaptive_context, token_aware
        >>>
        >>> @token_aware(operation='generate')
        >>> def generate_text(llm, prompt):
        ...     return llm.invoke(prompt)
        >>>
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> with adaptive_context(callback):
        ...     result = generate_text(llm, "Explain quantum computing")
        ...     # Decorator automatically uses callback from context

    Example with nested contexts:
        >>> callback1 = TokenCoPilotCallback(budget_limit=100.0)
        >>> callback2 = TokenCoPilotCallback(budget_limit=50.0)
        >>>
        >>> with adaptive_context(callback1):
        ...     # Uses callback1
        ...     result1 = generate_text(llm, "prompt1")
        ...
        ...     with adaptive_context(callback2):
        ...         # Uses callback2 (inner context)
        ...         result2 = generate_text(llm, "prompt2")
        ...
        ...     # Back to callback1
        ...     result3 = generate_text(llm, "prompt3")
    """
    # Save previous callback (for nested contexts)
    previous_callback = get_current_callback()

    try:
        # Set new callback
        set_current_callback(callback)
        yield callback
    finally:
        # Restore previous callback
        set_current_callback(previous_callback)


@contextmanager
def budget_aware_section(
    callback: 'TokenCoPilotCallback',
    section_name: str,
    log_summary: bool = True
):
    """Context manager for tracking a specific section of code with budget awareness.

    Tracks costs and tokens for a named section and logs summary on exit.

    Args:
        callback: TokenCoPilotCallback instance
        section_name: Name of the section for logging
        log_summary: Whether to log cost/token summary on exit (default: True)

    Yields:
        Dictionary containing section tracking info

    Example:
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> with budget_aware_section(callback, "data_processing") as section:
        ...     result1 = llm.invoke("process data 1", callbacks=[callback])
        ...     result2 = llm.invoke("process data 2", callbacks=[callback])
        ... # Logs: "Section [data_processing]: tokens=X, cost=$Y"

    Example with manual tracking:
        >>> with budget_aware_section(callback, "retrieval", log_summary=False) as section:
        ...     docs = retriever.get_relevant_documents(query)
        ...     print(f"Section cost so far: ${section['cost_delta']:.4f}")
    """
    import logging

    logger = logging.getLogger(__name__)

    # Record initial state
    initial_cost = callback.get_total_cost()
    initial_tokens = sum(
        entry.input_tokens + entry.output_tokens
        for entry in callback.tracker.entries
    )

    # Create section tracking dict
    section_info = {
        'name': section_name,
        'initial_cost': initial_cost,
        'initial_tokens': initial_tokens,
        'cost_delta': 0.0,
        'tokens_delta': 0
    }

    # Set context for decorators
    previous_callback = get_current_callback()
    set_current_callback(callback)

    try:
        yield section_info
    finally:
        # Calculate deltas
        final_cost = callback.get_total_cost()
        final_tokens = sum(
            entry.input_tokens + entry.output_tokens
            for entry in callback.tracker.entries
        )

        section_info['cost_delta'] = final_cost - initial_cost
        section_info['tokens_delta'] = final_tokens - initial_tokens

        # Log summary
        if log_summary:
            logger.info(
                f"Section [{section_name}]: "
                f"tokens={section_info['tokens_delta']}, "
                f"cost=${section_info['cost_delta']:.6f}"
            )

        # Restore previous context
        set_current_callback(previous_callback)
