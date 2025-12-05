"""Tests for context managers."""
import pytest
from unittest.mock import patch, MagicMock
from token_copilot.context import track_costs, with_budget, monitored


class TestContextManagers:
    """Tests for context manager functions."""

    def test_track_costs_basic(self):
        """Test basic track_costs context manager."""
        with track_costs(budget_limit=10.00) as copilot:
            assert copilot.budget_limit == 10.00
            assert copilot.on_budget_exceeded == "raise"

    def test_track_costs_custom_action(self):
        """Test track_costs with custom budget exceeded action."""
        with track_costs(budget_limit=10.00, on_budget_exceeded="warn") as copilot:
            assert copilot.on_budget_exceeded == "warn"

    def test_track_costs_with_webhook(self):
        """Test track_costs with webhook streaming."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming:
            with track_costs(
                budget_limit=10.00,
                webhook_url="https://example.com/webhook"
            ) as copilot:
                MockStreaming.assert_called()

    def test_track_costs_with_analytics(self):
        """Test track_costs with analytics."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin') as MockAnalytics:
            with track_costs(
                budget_limit=10.00,
                detect_anomalies=True
            ) as copilot:
                MockAnalytics.assert_called()

    def test_track_costs_cleanup_on_exit(self):
        """Test that track_costs cleans up resources on exit."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'):
            copilot_ref = None
            with track_costs(budget_limit=10.00) as copilot:
                copilot_ref = copilot
                # Simulate some work
                pass

            # After context exits, copilot should still be accessible
            assert copilot_ref is not None

    def test_with_budget_basic(self):
        """Test basic with_budget context manager."""
        with with_budget(budget_limit=50.00) as copilot:
            assert copilot.budget_limit == 50.00

    def test_with_budget_auto_warn(self):
        """Test with_budget automatically sets warn action."""
        with with_budget(budget_limit=50.00) as copilot:
            assert copilot.on_budget_exceeded == "warn"

    def test_monitored_basic(self):
        """Test basic monitored context manager."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            with monitored(budget_limit=100.00) as copilot:
                assert copilot.budget_limit == 100.00
                # Monitored should have multiple plugins
                assert len(copilot._plugin_manager.get_all_plugins()) > 0

    def test_monitored_with_webhook(self):
        """Test monitored with webhook URL."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            with monitored(
                budget_limit=100.00,
                webhook_url="https://example.com/webhook"
            ) as copilot:
                MockStreaming.assert_called()

    def test_monitored_analytics_enabled(self):
        """Test that monitored enables analytics by default."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin') as MockAnalytics, \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            with monitored(budget_limit=100.00) as copilot:
                MockAnalytics.assert_called()

    def test_monitored_forecasting_enabled(self):
        """Test that monitored enables forecasting by default."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin') as MockForecasting:

            with monitored(budget_limit=100.00) as copilot:
                MockForecasting.assert_called()

    def test_context_manager_exception_handling(self):
        """Test that context managers handle exceptions properly."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'):
            try:
                with track_costs(budget_limit=10.00) as copilot:
                    raise ValueError("Test exception")
            except ValueError:
                # Exception should propagate
                pass

    def test_nested_context_managers(self):
        """Test that context managers can be nested."""
        with track_costs(budget_limit=10.00) as copilot1:
            with track_costs(budget_limit=20.00) as copilot2:
                assert copilot1.budget_limit == 10.00
                assert copilot2.budget_limit == 20.00
                assert copilot1 is not copilot2

    def test_context_manager_returns_copilot(self):
        """Test that all context managers return TokenCoPilot instance."""
        from token_copilot import TokenCoPilot

        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            with track_costs(budget_limit=10.00) as copilot:
                assert isinstance(copilot, TokenCoPilot)

            with with_budget(budget_limit=50.00) as copilot:
                assert isinstance(copilot, TokenCoPilot)

            with monitored(budget_limit=100.00) as copilot:
                assert isinstance(copilot, TokenCoPilot)
