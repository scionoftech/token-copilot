"""Tests for core TokenCoPilot class."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.core.copilot import TokenCoPilot
from token_copilot.core.plugin import Plugin


class TestTokenCoPilot:
    """Tests for TokenCoPilot core class."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        copilot = TokenCoPilot(budget_limit=10.00)

        assert copilot.budget_limit == 10.00
        assert copilot.budget_period == "total"
        assert copilot.on_budget_exceeded == "raise"

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        copilot = TokenCoPilot(
            budget_limit=100.00,
            budget_period="daily",
            on_budget_exceeded="warn"
        )

        assert copilot.budget_limit == 100.00
        assert copilot.budget_period == "daily"
        assert copilot.on_budget_exceeded == "warn"

    def test_cost_property(self):
        """Test that .cost property works."""
        copilot = TokenCoPilot(budget_limit=10.00)

        # Mock get_total_cost
        with patch.object(copilot, 'get_total_cost', return_value=5.50):
            assert copilot.cost == 5.50

    def test_tokens_property(self):
        """Test that .tokens property works."""
        copilot = TokenCoPilot(budget_limit=10.00)

        # Mock get_total_tokens
        with patch.object(copilot, 'get_total_tokens', return_value=1000):
            assert copilot.tokens == 1000

    def test_add_plugin(self):
        """Test adding a plugin."""

        class MockPlugin(Plugin):
            pass

        copilot = TokenCoPilot(budget_limit=10.00)
        plugin = MockPlugin()

        copilot.add_plugin(plugin)

        plugins = copilot._plugin_manager.get_all_plugins()
        assert len(plugins) == 1
        assert plugin in plugins

    def test_remove_plugin(self):
        """Test removing a plugin."""

        class MockPlugin(Plugin):
            pass

        copilot = TokenCoPilot(budget_limit=10.00)
        plugin = MockPlugin()

        copilot.add_plugin(plugin)
        copilot.remove_plugin(plugin)

        plugins = copilot._plugin_manager.get_all_plugins()
        assert len(plugins) == 0

    def test_with_streaming_returns_self(self):
        """Test that with_streaming returns self for chaining."""
        copilot = TokenCoPilot(budget_limit=10.00)

        with patch('token_copilot.plugins.streaming.StreamingPlugin'):
            result = copilot.with_streaming(webhook_url="https://example.com")
            assert result is copilot

    def test_with_analytics_returns_self(self):
        """Test that with_analytics returns self for chaining."""
        copilot = TokenCoPilot(budget_limit=10.00)

        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):
            result = copilot.with_analytics()
            assert result is copilot

    def test_with_routing_returns_self(self):
        """Test that with_routing returns self for chaining."""
        copilot = TokenCoPilot(budget_limit=10.00)

        with patch('token_copilot.plugins.routing.RoutingPlugin'):
            result = copilot.with_routing()
            assert result is copilot

    def test_with_adaptive_returns_self(self):
        """Test that with_adaptive returns self for chaining."""
        copilot = TokenCoPilot(budget_limit=10.00)

        with patch('token_copilot.plugins.adaptive.AdaptivePlugin'):
            result = copilot.with_adaptive()
            assert result is copilot

    def test_with_forecasting_returns_self(self):
        """Test that with_forecasting returns self for chaining."""
        copilot = TokenCoPilot(budget_limit=10.00)

        with patch('token_copilot.plugins.forecasting.ForecastingPlugin'):
            result = copilot.with_forecasting()
            assert result is copilot

    def test_builder_pattern_chaining(self):
        """Test that builder methods can be chained."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            copilot = (TokenCoPilot(budget_limit=100.00)
                .with_streaming(webhook_url="https://example.com")
                .with_analytics(detect_anomalies=True)
                .with_forecasting()
            )

            # Should have 3 plugins added
            assert len(copilot._plugin_manager.get_all_plugins()) == 3

    def test_build_returns_self(self):
        """Test that build() returns self."""
        copilot = TokenCoPilot(budget_limit=10.00)
        result = copilot.build()

        assert result is copilot

    def test_reset_clears_tracking(self):
        """Test that reset() clears tracking data."""
        copilot = TokenCoPilot(budget_limit=10.00)

        # Mock tracker reset
        copilot.tracker = MagicMock()
        copilot.reset()

        copilot.tracker.reset.assert_called_once()

    def test_get_stats_returns_dict(self):
        """Test that get_stats() returns a dictionary."""
        copilot = TokenCoPilot(budget_limit=10.00)

        # Mock tracker
        copilot.tracker = MagicMock()
        copilot.tracker.get_stats.return_value = {"total_cost": 5.00}

        stats = copilot.get_stats()

        assert isinstance(stats, dict)
        assert "total_cost" in stats

    def test_to_dataframe(self):
        """Test that to_dataframe() returns DataFrame."""
        copilot = TokenCoPilot(budget_limit=10.00)

        # Mock tracker
        copilot.tracker = MagicMock()
        mock_df = MagicMock()
        copilot.tracker.to_dataframe.return_value = mock_df

        result = copilot.to_dataframe()

        assert result is mock_df
        copilot.tracker.to_dataframe.assert_called_once()

    def test_get_remaining_budget(self):
        """Test get_remaining_budget calculation."""
        copilot = TokenCoPilot(budget_limit=10.00)

        with patch.object(copilot, 'get_total_cost', return_value=3.50):
            remaining = copilot.get_remaining_budget()
            assert remaining == 6.50

    def test_get_remaining_budget_no_limit(self):
        """Test get_remaining_budget with no budget limit."""
        copilot = TokenCoPilot()

        remaining = copilot.get_remaining_budget()
        assert remaining == float('inf')
