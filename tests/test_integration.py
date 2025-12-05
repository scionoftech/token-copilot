"""Integration tests for all v2.0 usage patterns."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot import (
    TokenCoPilot,
    basic,
    development,
    production,
    enterprise,
    track_costs,
    with_budget,
    monitored,
)
from token_copilot.decorators import track_cost, enforce_budget


class TestMinimalPattern:
    """Integration tests for minimal usage pattern."""

    def test_minimal_usage(self):
        """Test minimal pattern - simplest usage."""
        copilot = TokenCoPilot(budget_limit=10.00)

        assert copilot.budget_limit == 10.00
        assert isinstance(copilot, TokenCoPilot)

    def test_minimal_with_langchain(self):
        """Test minimal pattern with LangChain mock."""
        copilot = TokenCoPilot(budget_limit=10.00)

        # Mock LangChain LLM
        mock_llm = MagicMock()
        mock_llm.callbacks = [copilot]

        assert copilot in mock_llm.callbacks


class TestBuilderPattern:
    """Integration tests for builder pattern."""

    def test_builder_basic(self):
        """Test basic builder pattern."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'):

            copilot = (TokenCoPilot(budget_limit=100.00)
                .with_streaming(webhook_url="https://example.com")
                .with_analytics(detect_anomalies=True)
            )

            assert copilot.budget_limit == 100.00
            assert len(copilot._plugin_manager.get_all_plugins()) == 2

    def test_builder_full(self):
        """Test builder pattern with all plugins."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.routing.RoutingPlugin'), \
             patch('token_copilot.plugins.adaptive.AdaptivePlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            copilot = (TokenCoPilot(budget_limit=100.00)
                .with_streaming(webhook_url="https://example.com")
                .with_analytics(detect_anomalies=True)
                .with_routing(models=[], strategy="balanced")
                .with_adaptive()
                .with_forecasting()
                .build()
            )

            assert copilot.budget_limit == 100.00
            # Should have 5 plugins
            assert len(copilot._plugin_manager.get_all_plugins()) == 5

    def test_builder_chaining_returns_self(self):
        """Test that builder methods return self for chaining."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'):
            copilot = TokenCoPilot(budget_limit=100.00)
            result = copilot.with_streaming(webhook_url="https://example.com")

            assert result is copilot


class TestFactoryPattern:
    """Integration tests for factory presets."""

    def test_basic_preset(self):
        """Test basic factory preset."""
        copilot = basic(budget_limit=10.00)

        assert copilot.budget_limit == 10.00
        assert len(copilot._plugin_manager.get_all_plugins()) == 0

    def test_development_preset(self):
        """Test development factory preset."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):
            copilot = development(budget_limit=50.00)

            assert copilot.budget_limit == 50.00
            assert len(copilot._plugin_manager.get_all_plugins()) >= 1

    def test_production_preset(self):
        """Test production factory preset."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            copilot = production(
                budget_limit=1000.00,
                webhook_url="https://example.com"
            )

            assert copilot.budget_limit == 1000.00
            assert len(copilot._plugin_manager.get_all_plugins()) >= 2

    def test_enterprise_preset(self):
        """Test enterprise factory preset."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'), \
             patch('token_copilot.plugins.adaptive.AdaptivePlugin'):

            copilot = enterprise(budget_limit=10000.00)

            assert copilot.budget_limit == 10000.00
            # Enterprise should have multiple plugins
            assert len(copilot._plugin_manager.get_all_plugins()) >= 3


class TestContextManagerPattern:
    """Integration tests for context managers."""

    def test_track_costs_context(self):
        """Test track_costs context manager."""
        with track_costs(budget_limit=10.00) as copilot:
            assert copilot.budget_limit == 10.00
            assert isinstance(copilot, TokenCoPilot)

    def test_with_budget_context(self):
        """Test with_budget context manager."""
        with with_budget(budget_limit=50.00) as copilot:
            assert copilot.budget_limit == 50.00
            assert copilot.on_budget_exceeded == "warn"

    def test_monitored_context(self):
        """Test monitored context manager."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            with monitored(budget_limit=100.00) as copilot:
                assert copilot.budget_limit == 100.00
                # Should have multiple plugins
                assert len(copilot._plugin_manager.get_all_plugins()) >= 2

    def test_nested_contexts(self):
        """Test nested context managers."""
        with track_costs(budget_limit=10.00) as copilot1:
            with track_costs(budget_limit=20.00) as copilot2:
                assert copilot1.budget_limit == 10.00
                assert copilot2.budget_limit == 20.00
                assert copilot1 is not copilot2


class TestDecoratorPattern:
    """Integration tests for decorators."""

    def test_track_cost_decorator(self):
        """Test @track_cost decorator."""

        @track_cost(budget_limit=5.00)
        def process_text(text):
            return f"Processed: {text}"

        assert hasattr(process_text, 'copilot')
        assert process_text.copilot.budget_limit == 5.00

        result = process_text("hello")
        assert result == "Processed: hello"

    def test_enforce_budget_decorator(self):
        """Test @enforce_budget decorator."""

        @enforce_budget(budget_limit=10.00)
        def process_text(text):
            return f"Processed: {text}"

        assert hasattr(process_text, 'copilot')
        assert process_text.copilot.budget_limit == 10.00
        assert process_text.copilot.on_budget_exceeded == "raise"

    def test_monitored_decorator(self):
        """Test @monitored decorator."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            @monitored(budget_limit=20.00)
            def process_text(text):
                return f"Processed: {text}"

            assert hasattr(process_text, 'copilot')
            assert len(process_text.copilot._plugin_manager.get_all_plugins()) >= 2


class TestPluginIntegration:
    """Integration tests for plugin system."""

    def test_multiple_plugins_coexist(self):
        """Test that multiple plugins can coexist."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            copilot = (TokenCoPilot(budget_limit=100.00)
                .with_streaming(webhook_url="https://example.com")
                .with_analytics(detect_anomalies=True)
                .with_forecasting()
            )

            plugins = copilot._plugin_manager.get_all_plugins()
            assert len(plugins) == 3

    def test_plugins_receive_lifecycle_events(self):
        """Test that plugins receive lifecycle events."""
        from token_copilot.core.plugin import Plugin

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_attach_called = False
                self.on_cost_tracked_called = False

            def on_attach(self):
                self.on_attach_called = True

            def on_cost_tracked(self, model, input_tokens, output_tokens, cost, metadata):
                self.on_cost_tracked_called = True

        copilot = TokenCoPilot(budget_limit=100.00)
        plugin = MockPlugin()
        copilot.add_plugin(plugin)

        assert plugin.on_attach_called

        # Trigger cost tracking
        copilot._plugin_manager.trigger("on_cost_tracked", "gpt-4", 100, 50, 0.01, {})
        assert plugin.on_cost_tracked_called

    def test_plugin_detachment(self):
        """Test plugin detachment."""
        from token_copilot.core.plugin import Plugin

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_detach_called = False

            def on_detach(self):
                self.on_detach_called = True

        copilot = TokenCoPilot(budget_limit=100.00)
        plugin = MockPlugin()

        copilot.add_plugin(plugin)
        assert len(copilot._plugin_manager.get_all_plugins()) == 1

        copilot.remove_plugin(plugin)
        assert len(copilot._plugin_manager.get_all_plugins()) == 0
        assert plugin.on_detach_called


class TestCrossPatternCompatibility:
    """Test that different patterns work together."""

    def test_builder_to_context(self):
        """Test using builder pattern inside context manager."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'):
            with track_costs(budget_limit=100.00) as copilot:
                copilot.with_streaming(webhook_url="https://example.com")

                assert copilot.budget_limit == 100.00
                assert len(copilot._plugin_manager.get_all_plugins()) >= 1

    def test_factory_in_decorator(self):
        """Test using factory preset in decorator context."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):

            @track_cost(budget_limit=50.00)
            def process_text(text):
                # Can still create additional copilots inside
                copilot = development(budget_limit=100.00)
                return f"Processed: {text}"

            assert hasattr(process_text, 'copilot')
            result = process_text("hello")
            assert result == "Processed: hello"

    def test_multiple_patterns_same_scope(self):
        """Test using multiple patterns in same scope."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'):

            # Pattern 1: Minimal
            copilot1 = TokenCoPilot(budget_limit=10.00)

            # Pattern 2: Builder
            copilot2 = (TokenCoPilot(budget_limit=20.00)
                .with_streaming(webhook_url="https://example.com")
            )

            # Pattern 3: Factory
            copilot3 = development(budget_limit=30.00)

            # All should be independent instances
            assert copilot1 is not copilot2
            assert copilot2 is not copilot3
            assert copilot1 is not copilot3

            # Each should have correct config
            assert copilot1.budget_limit == 10.00
            assert copilot2.budget_limit == 20.00
            assert copilot3.budget_limit == 30.00


class TestRealWorldScenarios:
    """Integration tests for real-world scenarios."""

    def test_progressive_enhancement(self):
        """Test progressive enhancement - start simple, add features."""
        # Start minimal
        copilot = TokenCoPilot(budget_limit=100.00)
        assert len(copilot._plugin_manager.get_all_plugins()) == 0

        # Add analytics
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):
            copilot.with_analytics(detect_anomalies=True)
            assert len(copilot._plugin_manager.get_all_plugins()) == 1

        # Add streaming
        with patch('token_copilot.plugins.streaming.StreamingPlugin'):
            copilot.with_streaming(webhook_url="https://example.com")
            assert len(copilot._plugin_manager.get_all_plugins()) == 2

    def test_production_deployment(self):
        """Test production deployment scenario."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            # Production setup with all monitoring
            copilot = production(
                budget_limit=1000.00,
                webhook_url="https://monitoring.example.com",
                detect_anomalies=True,
                enable_forecasting=True
            )

            # Should have comprehensive monitoring
            assert copilot.budget_limit == 1000.00
            assert copilot.on_budget_exceeded == "raise"
            assert len(copilot._plugin_manager.get_all_plugins()) >= 2

    def test_development_workflow(self):
        """Test development workflow with debugging."""
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):
            # Development with analytics for debugging
            copilot = development(
                budget_limit=50.00,
                detect_anomalies=True,
                anomaly_sensitivity=2.0
            )

            # Should warn instead of raise
            assert copilot.on_budget_exceeded == "warn"
            assert copilot.budget_limit == 50.00

    def test_quick_prototype(self):
        """Test quick prototyping scenario."""
        # Quick start - no frills
        copilot = basic(budget_limit=5.00)

        assert copilot.budget_limit == 5.00
        # No plugins for simplicity
        assert len(copilot._plugin_manager.get_all_plugins()) == 0
