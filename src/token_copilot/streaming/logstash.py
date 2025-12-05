"""Logstash streaming for Elasticsearch/ELK stack."""

import logging
import socket
import json
from typing import Optional

from .base import BaseStreamer, StreamEvent

logger = logging.getLogger(__name__)


class LogstashStreamer(BaseStreamer):
    """
    Stream cost events to Logstash for Elasticsearch indexing.

    Sends JSON events over TCP to Logstash JSON input plugin.

    Example:
        ```python
        from token_copilot import TokenPilotCallback
        from token_copilot.streaming import LogstashStreamer

        # Basic Logstash streaming
        streamer = LogstashStreamer(
            host="logstash.company.com",
            port=5000
        )

        # With index and tags
        streamer = LogstashStreamer(
            host="logstash.company.com",
            port=5000,
            index="llm-costs",
            tags=["production", "api"],
            extra_fields={"environment": "prod", "region": "us-east-1"}
        )

        callback = TokenPilotCallback(streamer=streamer)
        ```

    Logstash Configuration:
        ```
        input {
          tcp {
            port => 5000
            codec => json_lines
          }
        }

        filter {
          # Add any transformations here
        }

        output {
          elasticsearch {
            hosts => ["elasticsearch:9200"]
            index => "llm-costs-%{+YYYY.MM.dd}"
          }
        }
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        index: Optional[str] = None,
        tags: Optional[list] = None,
        extra_fields: Optional[dict] = None,
        reconnect_attempts: int = 3,
        enabled: bool = True,
    ):
        """
        Initialize Logstash streamer.

        Args:
            host: Logstash server hostname (default: "localhost")
            port: Logstash TCP input port (default: 5000)
            index: Elasticsearch index name (optional)
            tags: List of tags to add to events (optional)
            extra_fields: Dict of extra fields to add to events (optional)
            reconnect_attempts: Number of reconnection attempts (default: 3)
            enabled: Whether streaming is enabled (default: True)
        """
        super().__init__(enabled=enabled)
        self.host = host
        self.port = port
        self.index = index
        self.tags = tags or []
        self.extra_fields = extra_fields or {}
        self.reconnect_attempts = reconnect_attempts

        # Connection state
        self._socket: Optional[socket.socket] = None
        self._connect()

    def _connect(self):
        """Connect to Logstash server."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.connect((self.host, self.port))
            logger.debug(f"Connected to Logstash at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Logstash: {e}")
            self._socket = None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to Logstash."""
        for attempt in range(self.reconnect_attempts):
            try:
                logger.info(f"Reconnecting to Logstash (attempt {attempt + 1})")
                self._connect()
                if self._socket:
                    return True
            except:
                pass
        return False

    def _format_event(self, event: StreamEvent) -> dict:
        """Format event for Logstash/Elasticsearch."""
        data = event.to_dict()

        # Add index if specified
        if self.index:
            data['_index'] = self.index

        # Add tags
        if self.tags:
            data['tags'] = self.tags

        # Add extra fields
        data.update(self.extra_fields)

        # Add @timestamp for Elasticsearch
        data['@timestamp'] = event.timestamp.isoformat()

        # Add document type
        data['@type'] = 'llm_cost'

        return data

    def send(self, event: StreamEvent) -> bool:
        """
        Send event to Logstash.

        Args:
            event: StreamEvent to send

        Returns:
            True if sent successfully, False on error
        """
        if not self._socket:
            if not self._reconnect():
                return False

        try:
            # Format as JSON
            data = self._format_event(event)
            message = json.dumps(data) + '\n'
            message_bytes = message.encode('utf-8')

            # Send to Logstash
            self._socket.sendall(message_bytes)
            logger.debug(f"Sent event to Logstash: {len(message_bytes)} bytes")
            return True

        except (socket.error, BrokenPipeError) as e:
            logger.error(f"Logstash connection error: {e}")
            self._socket = None

            # Try to reconnect and resend
            if self._reconnect():
                try:
                    data = self._format_event(event)
                    message = json.dumps(data) + '\n'
                    self._socket.sendall(message.encode('utf-8'))
                    return True
                except:
                    return False
            return False

        except Exception as e:
            logger.error(f"Failed to send to Logstash: {e}")
            return False

    def close(self):
        """Close Logstash connection."""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
