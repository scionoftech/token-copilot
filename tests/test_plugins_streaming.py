"""Tests for StreamingPlugin."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.plugins.streaming import StreamingPlugin
from token_copilot.core.copilot import TokenCoPilot


class TestStreamingPlugin:
    """Tests for StreamingPlugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.copilot = MagicMock(spec=TokenCoPilot)

    def test_init_with_webhook(self):
        """Test initialization with webhook URL."""
        plugin = StreamingPlugin(webhook_url="https://example.com/webhook")

        assert plugin.webhook_url == "https://example.com/webhook"

    def test_init_with_kafka(self):
        """Test initialization with Kafka brokers."""
        plugin = StreamingPlugin(
            kafka_brokers=["kafka1:9092", "kafka2:9092"],
            kafka_topic="llm-costs"
        )

        assert plugin.kafka_brokers == ["kafka1:9092", "kafka2:9092"]
        assert plugin.kafka_topic == "llm-costs"

    def test_init_with_syslog(self):
        """Test initialization with Syslog."""
        plugin = StreamingPlugin(
            syslog_host="syslog.example.com",
            syslog_port=514
        )

        assert plugin.syslog_host == "syslog.example.com"
        assert plugin.syslog_port == 514

    def test_init_with_logstash(self):
        """Test initialization with Logstash."""
        plugin = StreamingPlugin(
            logstash_host="logstash.example.com",
            logstash_port=5000
        )

        assert plugin.logstash_host == "logstash.example.com"
        assert plugin.logstash_port == 5000

    def test_init_with_otlp(self):
        """Test initialization with OTLP endpoint."""
        plugin = StreamingPlugin(
            otlp_endpoint="http://collector:4318"
        )

        assert plugin.otlp_endpoint == "http://collector:4318"

    def test_on_attach_creates_webhook_streamer(self):
        """Test that on_attach creates WebhookStreamer when webhook_url provided."""
        with patch('token_copilot.streaming.WebhookStreamer') as MockStreamer:
            plugin = StreamingPlugin(webhook_url="https://example.com/webhook")
            plugin.attach(self.copilot)

            MockStreamer.assert_called_once_with("https://example.com/webhook")

    def test_on_attach_creates_kafka_streamer(self):
        """Test that on_attach creates KafkaStreamer when kafka_brokers provided."""
        with patch('token_copilot.streaming.KafkaStreamer') as MockStreamer:
            plugin = StreamingPlugin(
                kafka_brokers=["kafka1:9092"],
                kafka_topic="llm-costs"
            )
            plugin.attach(self.copilot)

            MockStreamer.assert_called()

    def test_on_attach_creates_multiple_streamers(self):
        """Test that on_attach creates multiple streamers when multiple configs provided."""
        with patch('token_copilot.streaming.WebhookStreamer') as MockWebhook, \
             patch('token_copilot.streaming.SyslogStreamer') as MockSyslog:

            plugin = StreamingPlugin(
                webhook_url="https://example.com/webhook",
                syslog_host="syslog.example.com"
            )
            plugin.attach(self.copilot)

            MockWebhook.assert_called()
            MockSyslog.assert_called()

    def test_on_cost_tracked_streams_to_all(self):
        """Test that on_cost_tracked streams to all configured streamers."""
        mock_streamer1 = MagicMock()
        mock_streamer2 = MagicMock()

        plugin = StreamingPlugin()
        plugin._streamers = [mock_streamer1, mock_streamer2]
        plugin.attach(self.copilot)

        plugin.on_cost_tracked("gpt-4", 100, 50, 0.01, {"user_id": "123"})

        mock_streamer1.stream.assert_called_once()
        mock_streamer2.stream.assert_called_once()

    def test_on_detach_clears_streamers(self):
        """Test that on_detach clears streamers."""
        plugin = StreamingPlugin(webhook_url="https://example.com/webhook")
        plugin._streamers = [MagicMock(), MagicMock()]

        plugin.detach()

        assert len(plugin._streamers) == 0

    def test_no_streamers_when_no_config(self):
        """Test that no streamers are created when no config provided."""
        plugin = StreamingPlugin()
        plugin.attach(self.copilot)

        assert len(plugin._streamers) == 0
