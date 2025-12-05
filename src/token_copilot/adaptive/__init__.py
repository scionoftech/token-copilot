"""Adaptive token operations that adjust based on budget availability.

This module provides budget-aware operations that automatically optimize their
behavior based on the current budget tier. All features use composition pattern
and are completely optional.

Key Features:
- TokenAwareOperations: Main class for adaptive LLM operations
- Budget tier classification (ABUNDANT, COMFORTABLE, MODERATE, LOW, CRITICAL)
- Decorators for token-aware functions (@token_aware, @budget_gate, @track_efficiency)
- Context managers for scoped adaptive behavior
- Automatic parameter adjustment based on remaining budget
- User parameters always override adaptive defaults

Quick Start:
    ```python
    from token_copilot import TokenCoPilotCallback
    from token_copilot.adaptive import TokenAwareOperations

    callback = TokenCoPilotCallback(budget_limit=100.0)
    adaptive = TokenAwareOperations(callback)

    # Automatically adjusts max_tokens and temperature based on budget
    result = adaptive.generate(llm, "Explain quantum computing")

    # User params always override
    result = adaptive.generate(llm, "prompt", max_tokens=500)
    ```

Using Decorators:
    ```python
    from token_copilot.adaptive import token_aware, adaptive_context

    @token_aware(operation='generate')
    def my_task(llm, prompt):
        return llm.invoke(prompt)

    callback = TokenCoPilotCallback(budget_limit=100.0)
    with adaptive_context(callback):
        result = my_task(llm, "Explain AI")
        # Automatically uses tier-appropriate parameters
    ```

Budget Gating:
    ```python
    from token_copilot.adaptive import budget_gate, BudgetTier

    @budget_gate(min_tier=BudgetTier.COMFORTABLE)
    def expensive_operation(llm, prompt):
        return llm.invoke(prompt, max_tokens=5000)

    # Skipped if budget tier is too low
    ```
"""

# Core classes
from .tiers import BudgetTier, classify_budget_tier, get_tier_description
from .operations import TokenAwareOperations

# Decorators
from .decorators import (
    token_aware,
    budget_gate,
    track_efficiency,
)

# Context management
from .context import (
    adaptive_context,
    budget_aware_section,
    get_current_callback,
    set_current_callback,
)

__all__ = [
    # Budget tiers
    "BudgetTier",
    "classify_budget_tier",
    "get_tier_description",
    # Main operations class
    "TokenAwareOperations",
    # Decorators
    "token_aware",
    "budget_gate",
    "track_efficiency",
    # Context management
    "adaptive_context",
    "budget_aware_section",
    "get_current_callback",
    "set_current_callback",
]
