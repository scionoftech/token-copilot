"""Budget tier classification for adaptive operations."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..langchain import TokenCoPilotCallback


class BudgetTier(Enum):
    """Budget tier classification based on remaining budget percentage.

    Each tier represents a different level of budget availability and triggers
    different adaptive behaviors in token-aware operations.
    """

    ABUNDANT = "abundant"      # >75% budget remaining - use premium settings
    COMFORTABLE = "comfortable"  # 50-75% remaining - balanced approach
    MODERATE = "moderate"       # 25-50% remaining - start optimizing
    LOW = "low"                # 10-25% remaining - aggressive optimization
    CRITICAL = "critical"       # <10% remaining - minimal token usage


def classify_budget_tier(callback: 'TokenCoPilotCallback') -> BudgetTier:
    """Classify current budget tier based on callback's budget status.

    Args:
        callback: TokenCoPilotCallback instance with budget tracking

    Returns:
        BudgetTier enum value based on remaining budget percentage

    Examples:
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> tier = classify_budget_tier(callback)
        >>> tier == BudgetTier.ABUNDANT
        True

        >>> # After spending $80
        >>> tier = classify_budget_tier(callback)
        >>> tier == BudgetTier.LOW
        True
    """
    budget_limit = callback.budget_limit
    if budget_limit is None or budget_limit <= 0:
        # No budget limit set - always abundant
        return BudgetTier.ABUNDANT

    total_cost = callback.get_total_cost()
    remaining = budget_limit - total_cost
    percentage = (remaining / budget_limit) * 100

    if percentage > 75:
        return BudgetTier.ABUNDANT
    elif percentage > 50:
        return BudgetTier.COMFORTABLE
    elif percentage > 25:
        return BudgetTier.MODERATE
    elif percentage > 10:
        return BudgetTier.LOW
    else:
        return BudgetTier.CRITICAL


def get_tier_description(tier: BudgetTier) -> str:
    """Get human-readable description of a budget tier.

    Args:
        tier: BudgetTier enum value

    Returns:
        Description string explaining the tier's meaning
    """
    descriptions = {
        BudgetTier.ABUNDANT: "Abundant budget (>75% remaining) - using premium settings",
        BudgetTier.COMFORTABLE: "Comfortable budget (50-75% remaining) - balanced approach",
        BudgetTier.MODERATE: "Moderate budget (25-50% remaining) - starting optimization",
        BudgetTier.LOW: "Low budget (10-25% remaining) - aggressive optimization",
        BudgetTier.CRITICAL: "Critical budget (<10% remaining) - minimal token usage",
    }
    return descriptions.get(tier, "Unknown tier")
