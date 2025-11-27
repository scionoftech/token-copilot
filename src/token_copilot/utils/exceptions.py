"""Custom exceptions for token_copilot package."""


class TokenPilotError(Exception):
    """Base exception for all token_copilot errors."""

    pass


class BudgetExceededError(TokenPilotError):
    """Raised when operation would exceed budget limit."""

    def __init__(
        self,
        message: str = None,
        current: float = None,
        limit: float = None,
        additional: float = None,
    ):
        """
        Initialize budget exceeded error.

        Args:
            message: Custom error message (optional)
            current: Current spent amount
            limit: Budget limit
            additional: Additional cost that would exceed limit
        """
        self.current = current
        self.limit = limit
        self.additional = additional

        if message:
            super().__init__(message)
        elif all(x is not None for x in [current, limit, additional]):
            super().__init__(
                f"Budget exceeded: ${current:.4f} + ${additional:.4f} = "
                f"${current + additional:.4f} (limit: ${limit:.4f})"
            )
        else:
            super().__init__("Budget limit exceeded")


class TrackingError(TokenPilotError):
    """Raised when cost tracking fails."""

    def __init__(self, message: str):
        """Initialize tracking error."""
        super().__init__(f"Tracking error: {message}")


class AlertError(TokenPilotError):
    """Raised when alert delivery fails."""

    def __init__(self, message: str, alert_type: str = None):
        """Initialize alert error."""
        self.alert_type = alert_type
        msg = f"Alert error: {message}"
        if alert_type:
            msg += f" (type: {alert_type})"
        super().__init__(msg)
