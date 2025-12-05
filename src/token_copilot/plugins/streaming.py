"""Streaming plugin for real-time cost event streaming."""

from typing import Any, Dict, Optional
from ..core.plugin import Plugin


class StreamingPlugin(Plugin):
    """Plugin for real-time cost event streaming to external systems.

    Supports multiple streaming backends:
    - Webhook (HTTP POST)
    - Kafka
    - Syslog
    - Logstash
    - OpenTelemetry (OTLP)

    Example (Webhook):
        >>> from token_copilot import TokenCoPilot
        >>> from token_copilot.plugins import StreamingPlugin
        >>>
        >>> copilot = TokenCoPilot(budget_limit=100.00)
        >>> copilot.add_plugin(StreamingPlugin(webhook_url="https://example.com/webhook"))

    Example (Builder):
        >>> copilot = (TokenCoPilot(budget_limit=100.00)
        ...     .with_streaming(webhook_url="https://example.com/webhook")
        ... )

    Example (Kafka):
        >>> copilot.add_plugin(StreamingPlugin(
        ...     kafka_brokers=["localhost:9092"],
        ...     kafka_topic="llm_costs"
        ... ))
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        kafka_brokers: Optional[list] = None,
        kafka_topic: str = "llm_costs",
        syslog_host: Optional[str] = None,
        syslog_port: int = 514,
        logstash_host: Optional[str] = None,
        logstash_port: int = 5000,
        otlp_endpoint: Optional[str] = None,
    ):
        """Initialize streaming plugin.

        Args:
            webhook_url: HTTP webhook URL for POST requests
            kafka_brokers: List of Kafka broker addresses
            kafka_topic: Kafka topic name (default: "llm_costs")
            syslog_host: Syslog server host
            syslog_port: Syslog server port (default: 514)
            logstash_host: Logstash server host
            logstash_port: Logstash server port (default: 5000)
            otlp_endpoint: OpenTelemetry OTLP endpoint
        """
        super().__init__()
        self.webhook_url = webhook_url
        self.kafka_brokers = kafka_brokers
        self.kafka_topic = kafka_topic
        self.syslog_host = syslog_host
        self.syslog_port = syslog_port
        self.logstash_host = logstash_host
        self.logstash_port = logstash_port
        self.otlp_endpoint = otlp_endpoint

        self._streamers = []

    def on_attach(self):
        """Initialize streamers when attached to copilot."""
        # Import streamers lazily
        try:
            if self.webhook_url:
                from ..streaming import WebhookStreamer
                streamer = WebhookStreamer(self.webhook_url)
                self._streamers.append(streamer)

            if self.kafka_brokers:
                from ..streaming import KafkaStreamer
                streamer = KafkaStreamer(
                    bootstrap_servers=self.kafka_brokers,
                    topic=self.kafka_topic
                )
                self._streamers.append(streamer)

            if self.syslog_host:
                from ..streaming import SyslogStreamer
                streamer = SyslogStreamer(
                    host=self.syslog_host,
                    port=self.syslog_port
                )
                self._streamers.append(streamer)

            if self.logstash_host:
                from ..streaming import LogstashStreamer
                streamer = LogstashStreamer(
                    host=self.logstash_host,
                    port=self.logstash_port
                )
                self._streamers.append(streamer)

            if self.otlp_endpoint:
                from ..streaming import OpenTelemetryStreamer
                streamer = OpenTelemetryStreamer(endpoint=self.otlp_endpoint)
                self._streamers.append(streamer)

        except ImportError as e:
            import logging
            logging.warning(
                f"Streaming plugin requires additional dependencies: {e}. "
                "Install with: pip install token-copilot[streaming]"
            )

    def on_cost_tracked(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        metadata: Dict[str, Any]
    ):
        """Stream cost event to configured backends."""
        if not self._streamers:
            return

        # Create event
        from ..streaming.base import StreamEvent
        from datetime import datetime

        event = StreamEvent(
            timestamp=datetime.utcnow(),
            event_type="llm_cost",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            user_id=metadata.get("user_id"),
            org_id=metadata.get("org_id"),
            session_id=metadata.get("session_id"),
            request_id=metadata.get("request_id"),
            metadata=metadata,
        )

        # Send to all configured streamers
        for streamer in self._streamers:
            try:
                streamer.send_if_enabled(event)
            except Exception as e:
                import logging
                logging.warning(f"Streamer {type(streamer).__name__} failed: {e}")
