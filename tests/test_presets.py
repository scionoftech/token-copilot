"""Tests for factory presets."""
import pytest
from unittest.mock import patch
from token_copilot.presets import basic, development, production, enterprise, quick


class TestPresets:
    """Tests for factory preset functions."""

    def test_basic_preset(self):
        """Test basic() preset."""
        copilot = basic(budget_limit=10.00)

        assert copilot.budget_limit == 10.00
        assert copilot.on_budget_exceeded == "raise"
        # Basic should have no plugins
        assert len(copilot._plugin_manager.get_all_plugins()) == 0

    def test_development_preset(self):
        """Test development() preset."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin') as MockAnalytics:
            copilot = development(budget_limit=50.00)

            assert copilot.budget_limit == 50.00
            assert copilot.on_budget_exceeded == "warn"
            # Development should have analytics
            MockAnalytics.assert_called()

    def test_development_with_anomaly_detection(self):
        """Test development() preset with anomaly detection."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin') as MockAnalytics:
            copilot = development(
                budget_limit=50.00,
                detect_anomalies=True,
                anomaly_sensitivity=2.5
            )

            MockAnalytics.assert_called_with(
                detect_anomalies=True,
                anomaly_sensitivity=2.5
            )

    def test_production_preset_minimal(self):
        """Test production() preset with minimal config."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):
            copilot = production(budget_limit=1000.00)

            assert copilot.budget_limit == 1000.00
            assert copilot.on_budget_exceeded == "raise"

    def test_production_with_webhook(self):
        """Test production() preset with webhook."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'):

            copilot = production(
                budget_limit=1000.00,
                webhook_url="https://example.com/webhook"
            )

            MockStreaming.assert_called()

    def test_production_with_slack(self):
        """Test production() preset with Slack webhook."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'):

            copilot = production(
                budget_limit=1000.00,
                slack_webhook="https://hooks.slack.com/services/XXX"
            )

            MockStreaming.assert_called()

    def test_production_with_forecasting(self):
        """Test production() preset with forecasting."""
        with patch('token_copilot.plugins.forecasting.ForecastingPlugin') as MockForecasting, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'):

            copilot = production(
                budget_limit=1000.00,
                enable_forecasting=True,
                forecast_hours=48
            )

            MockForecasting.assert_called()

    def test_enterprise_preset_minimal(self):
        """Test enterprise() preset with minimal config."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'), \
             patch('token_copilot.plugins.adaptive.AdaptivePlugin'):

            copilot = enterprise(budget_limit=10000.00)

            assert copilot.budget_limit == 10000.00
            assert copilot.on_budget_exceeded == "raise"

    def test_enterprise_with_kafka(self):
        """Test enterprise() preset with Kafka."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'), \
             patch('token_copilot.plugins.adaptive.AdaptivePlugin'):

            copilot = enterprise(
                budget_limit=10000.00,
                kafka_brokers=["kafka1:9092", "kafka2:9092"],
                kafka_topic="llm-costs"
            )

            MockStreaming.assert_called()

    def test_enterprise_with_otlp(self):
        """Test enterprise() preset with OTLP."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'), \
             patch('token_copilot.plugins.adaptive.AdaptivePlugin'):

            copilot = enterprise(
                budget_limit=10000.00,
                otlp_endpoint="http://collector:4318"
            )

            MockStreaming.assert_called()

    def test_quick_preset(self):
        """Test quick() preset (alias for basic)."""
        copilot = quick(budget_limit=5.00)

        assert copilot.budget_limit == 5.00
        assert len(copilot._plugin_manager.get_all_plugins()) == 0

    def test_preset_returns_copilot_instance(self):
        """Test that all presets return TokenCoPilot instance."""
        from token_copilot import TokenCoPilot

        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'), \
             patch('token_copilot.plugins.adaptive.AdaptivePlugin'):

            assert isinstance(basic(budget_limit=10.00), TokenCoPilot)
            assert isinstance(development(budget_limit=50.00), TokenCoPilot)
            assert isinstance(production(budget_limit=1000.00), TokenCoPilot)
            assert isinstance(enterprise(budget_limit=10000.00), TokenCoPilot)
            assert isinstance(quick(budget_limit=5.00), TokenCoPilot)
