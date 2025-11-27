"""Budget enforcement for cost control."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional
from ..utils.exceptions import BudgetExceededError


class BudgetEnforcer:
    """
    Budget enforcement with configurable behavior.

    Supports global, per-user, and per-org budget limits.
    Can raise exceptions or log warnings when exceeded.

    Example:
        >>> enforcer = BudgetEnforcer(limit=10.00, on_exceeded="raise")
        >>> enforcer.check()  # OK if under budget
        >>> enforcer.add(5.00)
        >>> enforcer.add(6.00)  # Raises BudgetExceededError
    """

    def __init__(
        self,
        limit: Optional[float] = None,
        period: str = "total",
        on_exceeded: str = "raise",
    ):
        """
        Initialize budget enforcer.

        Args:
            limit: Budget limit in USD (None = no limit)
            period: Budget period:
                - "total": Total across all time
                - "daily": Reset daily
                - "monthly": Reset monthly
                - "per_user": Separate budget per user_id
                - "per_org": Separate budget per org_id
            on_exceeded: Behavior when exceeded:
                - "raise": Raise BudgetExceededError
                - "warn": Log warning but continue
                - "ignore": No action

        Example:
            >>> # Global budget
            >>> enforcer = BudgetEnforcer(limit=100.00)
            >>> # Per-user budget
            >>> enforcer = BudgetEnforcer(limit=10.00, period="per_user")
            >>> # Daily budget
            >>> enforcer = BudgetEnforcer(limit=50.00, period="daily")
        """
        self.limit = limit
        self.period = period
        self.on_exceeded = on_exceeded
        self._spent: Dict[str, float] = defaultdict(float)

    def check(self, metadata: Optional[Dict] = None) -> bool:
        """
        Check if budget allows operation.

        Args:
            metadata: Metadata dict (for per-user/per-org budgets)

        Returns:
            True if under budget

        Raises:
            BudgetExceededError: If over budget and on_exceeded="raise"

        Example:
            >>> enforcer.check()  # Global check
            >>> enforcer.check({"user_id": "user_123"})  # Per-user check
        """
        if self.limit is None:
            return True

        key = self._get_key(metadata)
        spent = self._spent[key]

        if spent >= self.limit:
            if self.on_exceeded == "raise":
                raise BudgetExceededError(
                    message=f"Budget limit ${self.limit:.2f} exceeded (spent: ${spent:.2f})",
                    current=spent,
                    limit=self.limit,
                )
            elif self.on_exceeded == "warn":
                import logging
                logging.warning(
                    f"Budget limit exceeded: ${spent:.2f} / ${self.limit:.2f}"
                )
            # else: ignore

        return spent < self.limit

    def add(self, cost: float, metadata: Optional[Dict] = None):
        """
        Add cost to budget tracking.

        Args:
            cost: Cost to add in USD
            metadata: Metadata dict (for per-user/per-org budgets)

        Example:
            >>> enforcer.add(1.50)
            >>> enforcer.add(0.75, {"user_id": "user_123"})
        """
        key = self._get_key(metadata)
        self._spent[key] += cost

    def get_spent(self, metadata: Optional[Dict] = None) -> float:
        """
        Get amount spent.

        Args:
            metadata: Metadata dict (for per-user/per-org budgets)

        Returns:
            Amount spent in USD

        Example:
            >>> total = enforcer.get_spent()
            >>> user_spent = enforcer.get_spent({"user_id": "user_123"})
        """
        if self.limit is None:
            return 0.0

        key = self._get_key(metadata)
        return self._spent[key]

    def get_remaining(self, metadata: Optional[Dict] = None) -> float:
        """
        Get remaining budget.

        Args:
            metadata: Metadata dict (for per-user/per-org budgets)

        Returns:
            Remaining budget in USD (inf if no limit)

        Example:
            >>> remaining = enforcer.get_remaining()
            >>> print(f"${remaining:.2f} left")
        """
        if self.limit is None:
            return float("inf")

        key = self._get_key(metadata)
        return max(0, self.limit - self._spent[key])

    def reset(self, metadata: Optional[Dict] = None):
        """
        Reset budget tracking.

        Args:
            metadata: Metadata dict (for per-user/per-org budgets)
                     If None, resets all budgets

        Example:
            >>> enforcer.reset()  # Reset all
            >>> enforcer.reset({"user_id": "user_123"})  # Reset user
        """
        if metadata is None:
            self._spent.clear()
        else:
            key = self._get_key(metadata)
            self._spent[key] = 0.0

    def _get_key(self, metadata: Optional[Dict] = None) -> str:
        """Get budget key based on period."""
        if self.period == "total":
            return "total"
        elif self.period == "daily":
            return datetime.now().strftime("%Y-%m-%d")
        elif self.period == "monthly":
            return datetime.now().strftime("%Y-%m")
        elif self.period == "per_user":
            if metadata and "user_id" in metadata:
                return f"user:{metadata['user_id']}"
            return "user:unknown"
        elif self.period == "per_org":
            if metadata and "org_id" in metadata:
                return f"org:{metadata['org_id']}"
            return "org:unknown"
        else:
            return "total"
