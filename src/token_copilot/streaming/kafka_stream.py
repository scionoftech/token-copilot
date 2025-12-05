"""Kafka streaming for Apache Kafka message queue."""

import logging
import json
from typing import Optional, Dict, Any

from .base import BaseStreamer, StreamEvent

logger = logging.getLogger(__name__)


class KafkaStreamer(BaseStreamer):
    """
    Stream cost events to Apache Kafka.

    Sends events as JSON messages to Kafka topics with support for
    partitioning, compression, and delivery guarantees.

    Requires: pip install kafka-python

    Example:
        ```python
        from token_copilot import TokenPilotCallback
        from token_copilot.streaming import KafkaStreamer

        # Basic Kafka streaming
        streamer = KafkaStreamer(
            bootstrap_servers=["kafka:9092"],
            topic="llm-costs"
        )

        # With advanced options
        streamer = KafkaStreamer(
            bootstrap_servers=["kafka1:9092", "kafka2:9092"],
            topic="llm-costs",
            compression_type="gzip",
            acks="all",
            partition_key="user_id"  # Partition by user_id
        )

        # With SASL authentication
        streamer = KafkaStreamer(
            bootstrap_servers=["kafka:9092"],
            topic="llm-costs",
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="password"
        )

        callback = TokenPilotCallback(streamer=streamer)
        ```
    """

    def __init__(
        self,
        bootstrap_servers: list,
        topic: str,
        partition_key: Optional[str] = None,
        compression_type: Optional[str] = None,
        acks: str = "1",
        retries: int = 3,
        batch_size: int = 16384,
        linger_ms: int = 10,
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: Optional[str] = None,
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
        ssl_cafile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        producer_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize Kafka streamer.

        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic: Kafka topic to publish to
            partition_key: Event field to use as partition key (e.g., "user_id")
            compression_type: Compression type: None, "gzip", "snappy", "lz4", "zstd"
            acks: Acknowledgment mode: "0", "1", or "all"
            retries: Number of retries (default: 3)
            batch_size: Batch size in bytes (default: 16384)
            linger_ms: Milliseconds to wait before sending batch (default: 10)
            security_protocol: "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"
            sasl_mechanism: SASL mechanism: "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"
            sasl_username: SASL username (if using SASL)
            sasl_password: SASL password (if using SASL)
            ssl_cafile: Path to CA certificate file (if using SSL)
            ssl_certfile: Path to client certificate file (if using SSL)
            ssl_keyfile: Path to client key file (if using SSL)
            producer_config: Additional producer config dict (optional)
            enabled: Whether streaming is enabled (default: True)
        """
        super().__init__(enabled=enabled)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.partition_key = partition_key

        # Build producer config
        config = {
            'bootstrap_servers': bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'acks': acks,
            'retries': retries,
            'batch_size': batch_size,
            'linger_ms': linger_ms,
            'security_protocol': security_protocol,
        }

        if compression_type:
            config['compression_type'] = compression_type

        # SASL config
        if sasl_mechanism:
            config['sasl_mechanism'] = sasl_mechanism
        if sasl_username:
            config['sasl_plain_username'] = sasl_username
        if sasl_password:
            config['sasl_plain_password'] = sasl_password

        # SSL config
        if ssl_cafile:
            config['ssl_cafile'] = ssl_cafile
        if ssl_certfile:
            config['ssl_certfile'] = ssl_certfile
        if ssl_keyfile:
            config['ssl_keyfile'] = ssl_keyfile

        # Merge additional config
        if producer_config:
            config.update(producer_config)

        # Create producer
        self._producer = None
        self._init_producer(config)

    def _init_producer(self, config: dict):
        """Initialize Kafka producer."""
        try:
            from kafka import KafkaProducer
        except ImportError:
            logger.warning(
                "kafka-python library required for Kafka streaming. "
                "Install with: pip install token-copilot[streaming] "
                "or pip install kafka-python"
            )
            return

        try:
            self._producer = KafkaProducer(**config)
            logger.debug(f"Kafka producer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self._producer = None

    def _get_partition_key(self, event: StreamEvent) -> Optional[bytes]:
        """Get partition key from event."""
        if not self.partition_key:
            return None

        # Get value from event
        value = None
        if self.partition_key == "user_id":
            value = event.user_id
        elif self.partition_key == "org_id":
            value = event.org_id
        elif self.partition_key == "session_id":
            value = event.session_id
        elif self.partition_key == "model":
            value = event.model
        elif event.metadata and self.partition_key in event.metadata:
            value = event.metadata[self.partition_key]

        return value.encode('utf-8') if value else None

    def send(self, event: StreamEvent) -> bool:
        """
        Send event to Kafka.

        Args:
            event: StreamEvent to send

        Returns:
            True if sent successfully, False on error
        """
        if not self._producer:
            return False

        try:
            # Convert to dict
            data = event.to_dict()

            # Get partition key
            key = self._get_partition_key(event)

            # Send to Kafka (async)
            future = self._producer.send(
                self.topic,
                value=data,
                key=key
            )

            # Optional: wait for send to complete
            # future.get(timeout=10)

            logger.debug(f"Sent event to Kafka topic: {self.topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
            return False

    def flush(self):
        """Flush pending messages."""
        if self._producer:
            try:
                self._producer.flush(timeout=10)
            except Exception as e:
                logger.error(f"Kafka flush error: {e}")

    def close(self):
        """Close Kafka producer."""
        if self._producer:
            try:
                self._producer.flush(timeout=10)
                self._producer.close(timeout=10)
            except Exception as e:
                logger.error(f"Kafka close error: {e}")
            self._producer = None
