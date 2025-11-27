"""Multi-tenant cost tracker with analytics."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..utils.pricing import calculate_cost


@dataclass
class CostEntry:
    """Single cost tracking entry."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    session_id: Optional[str] = None
    feature: Optional[str] = None
    endpoint: Optional[str] = None
    environment: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


class MultiTenantTracker:
    """
    Enhanced cost tracker with multi-tenant support.

    Tracks LLM costs with metadata for multi-tenant applications.
    Supports grouping by user, organization, session, feature, etc.

    Example:
        >>> tracker = MultiTenantTracker()
        >>> tracker.track(
        ...     model="gpt-4",
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     metadata={"user_id": "user_123", "org_id": "org_456"}
        ... )
        >>> df = tracker.to_dataframe()
        >>> print(df.groupby('user_id')['cost'].sum())
    """

    def __init__(self):
        """Initialize tracker."""
        self._entries: List[CostEntry] = []
        self._last_cost: float = 0.0

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostEntry:
        """
        Track a completion with metadata.

        Args:
            model: Model ID (e.g., "gpt-4", "claude-3-opus")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional metadata dict with keys:
                - user_id: User identifier
                - org_id: Organization identifier
                - session_id: Session identifier
                - feature: Feature name (e.g., "chat", "summarize")
                - endpoint: API endpoint (e.g., "/api/chat")
                - environment: Environment (e.g., "prod", "staging")
                - tags: Dict of custom key-value pairs

        Returns:
            CostEntry object with tracking data

        Example:
            >>> entry = tracker.track(
            ...     model="gpt-3.5-turbo",
            ...     input_tokens=100,
            ...     output_tokens=50,
            ...     metadata={
            ...         "user_id": "user_123",
            ...         "org_id": "org_456",
            ...         "feature": "chat"
            ...     }
            ... )
            >>> print(f"Cost: ${entry.cost:.4f}")
        """
        metadata = metadata or {}
        cost = calculate_cost(model, input_tokens, output_tokens)
        self._last_cost = cost

        entry = CostEntry(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            user_id=metadata.get("user_id"),
            org_id=metadata.get("org_id"),
            session_id=metadata.get("session_id"),
            feature=metadata.get("feature"),
            endpoint=metadata.get("endpoint"),
            environment=metadata.get("environment"),
            tags=metadata.get("tags", {}),
        )

        self._entries.append(entry)
        return entry

    def get_last_cost(self) -> float:
        """Get cost of last tracked completion."""
        return self._last_cost

    def get_total_cost(self) -> float:
        """Get total cost across all tracked completions."""
        return sum(entry.cost for entry in self._entries)

    def get_total_tokens(self) -> int:
        """Get total tokens across all tracked completions."""
        return sum(
            entry.input_tokens + entry.output_tokens for entry in self._entries
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dict with total_cost, total_tokens, num_calls, avg_cost, etc.
        """
        if not self._entries:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_calls": 0,
                "avg_cost_per_call": 0.0,
                "avg_tokens_per_call": 0.0,
            }

        total_cost = self.get_total_cost()
        total_tokens = self.get_total_tokens()
        num_calls = len(self._entries)

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_calls": num_calls,
            "avg_cost_per_call": total_cost / num_calls,
            "avg_tokens_per_call": total_tokens / num_calls,
        }

    def to_dataframe(self):
        """
        Export tracking data to pandas DataFrame.

        Returns:
            pandas.DataFrame with columns:
                - timestamp (index)
                - model
                - input_tokens
                - output_tokens
                - total_tokens
                - cost
                - user_id
                - org_id
                - session_id
                - feature
                - endpoint
                - environment
                - any custom tags

        Example:
            >>> df = tracker.to_dataframe()
            >>> # Group by user
            >>> print(df.groupby('user_id')['cost'].sum())
            >>> # Group by organization
            >>> print(df.groupby('org_id')['cost'].sum())
            >>> # Filter by feature
            >>> chat_df = df[df['feature'] == 'chat']
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame export. "
                "Install with: pip install pandas"
            )

        if not self._entries:
            return pd.DataFrame()

        data = []
        for entry in self._entries:
            row = {
                "timestamp": entry.timestamp,
                "model": entry.model,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "total_tokens": entry.input_tokens + entry.output_tokens,
                "cost": entry.cost,
                "user_id": entry.user_id,
                "org_id": entry.org_id,
                "session_id": entry.session_id,
                "feature": entry.feature,
                "endpoint": entry.endpoint,
                "environment": entry.environment,
            }
            # Add custom tags as columns
            row.update(entry.tags)
            data.append(row)

        df = pd.DataFrame(data)
        if len(df) > 0:
            df.set_index("timestamp", inplace=True)
        return df

    def get_costs_by(self, dimension: str) -> Dict[str, float]:
        """
        Get costs grouped by a dimension.

        Args:
            dimension: Column to group by (e.g., "user_id", "org_id", "model")

        Returns:
            Dict mapping dimension values to total costs

        Example:
            >>> costs_by_user = tracker.get_costs_by("user_id")
            >>> costs_by_org = tracker.get_costs_by("org_id")
            >>> costs_by_model = tracker.get_costs_by("model")
        """
        df = self.to_dataframe()
        if df.empty:
            return {}
        return df.groupby(dimension)["cost"].sum().to_dict()

    def clear(self):
        """Clear all tracking data."""
        self._entries.clear()
        self._last_cost = 0.0

    def __len__(self) -> int:
        """Return number of tracked entries."""
        return len(self._entries)
