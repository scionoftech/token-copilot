"""Base classes for streaming integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
import json


@dataclass
class StreamEvent:
    """
    Structured event for streaming cost data.

    This is the standard format streamed to all external systems.
    """
    timestamp: datetime
    event_type: str = "llm_cost"
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    session_id: Optional[str] = None
    feature: Optional[str] = None
    endpoint: Optional[str] = None
    environment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class BaseStreamer(ABC):
    """
    Abstract base class for all streamers.

    Subclasses must implement the `send` method to stream events
    to their specific backend.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize streamer.

        Args:
            enabled: Whether streaming is enabled (default: True)
        """
        self.enabled = enabled

    @abstractmethod
    def send(self, event: StreamEvent) -> bool:
        """
        Send event to streaming backend.

        Args:
            event: StreamEvent to send

        Returns:
            True if sent successfully, False otherwise
        """
        pass

    def send_if_enabled(self, event: StreamEvent) -> bool:
        """
        Send event only if streaming is enabled.

        Args:
            event: StreamEvent to send

        Returns:
            True if sent successfully or disabled, False on error
        """
        if not self.enabled:
            return True
        return self.send(event)

    def close(self) -> None:
        """
        Close streamer and cleanup resources.

        Override this method if your streamer needs cleanup.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
