"""
Real-time streaming integrations for cost tracking.

Stream cost events to external systems for real-time analytics, persistence,
and integration with existing observability stacks.

Features:
- WebhookStreamer: POST events to HTTP endpoints
- SyslogStreamer: Stream to syslog servers (RFC 5424)
- LogstashStreamer: Stream to Logstash/Elasticsearch
- KafkaStreamer: Stream to Apache Kafka
- OpenTelemetryStreamer: Export via OTLP protocol

All streamers are optional and require additional dependencies.

Example:
    ```python
    from token_copilot import TokenPilotCallback
    from token_copilot.streaming import WebhookStreamer

    streamer = WebhookStreamer(url="https://analytics.company.com/ingest")
    callback = TokenPilotCallback(
        budget_limit=100.00,
        streamer=streamer
    )
    ```
"""

from .base import BaseStreamer, StreamEvent
from .webhook import WebhookStreamer
from .syslog import SyslogStreamer
from .logstash import LogstashStreamer
from .kafka_stream import KafkaStreamer
from .opentelemetry import OpenTelemetryStreamer

__all__ = [
    "BaseStreamer",
    "StreamEvent",
    "WebhookStreamer",
    "SyslogStreamer",
    "LogstashStreamer",
    "KafkaStreamer",
    "OpenTelemetryStreamer",
]
