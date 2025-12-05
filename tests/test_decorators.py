"""Tests for decorators."""
import pytest
from unittest.mock import patch, MagicMock
from token_copilot.decorators import track_cost, enforce_budget, monitored


class TestDecorators:
    """Tests for decorator functions."""

    def test_track_cost_basic(self):
        """Test basic @track_cost decorator."""

        @track_cost(budget_limit=5.00)
        def process_text(text):
            return f"Processed: {text}"

        # Function should have copilot attribute
        assert hasattr(process_text, 'copilot')
        assert process_text.copilot.budget_limit == 5.00

        # Function should still work
        result = process_text("hello")
        assert result == "Processed: hello"

    def test_track_cost_custom_attribute(self):
        """Test @track_cost with custom attribute name."""

        @track_cost(budget_limit=5.00, attach_to="tracker")
        def process_text(text):
            return f"Processed: {text}"

        assert hasattr(process_text, 'tracker')
        assert process_text.tracker.budget_limit == 5.00

    def test_track_cost_preserves_function_metadata(self):
        """Test that @track_cost preserves function metadata."""

        @track_cost(budget_limit=5.00)
        def process_text(text):
            """Process text and return result."""
            return f"Processed: {text}"

        assert process_text.__name__ == "process_text"
        assert process_text.__doc__ == "Process text and return result."

    def test_track_cost_with_args_and_kwargs(self):
        """Test @track_cost with function that uses args and kwargs."""

        @track_cost(budget_limit=5.00)
        def process_text(text, prefix="", suffix=""):
            return f"{prefix}{text}{suffix}"

        result = process_text("hello", prefix="[", suffix="]")
        assert result == "[hello]"

    def test_enforce_budget_basic(self):
        """Test basic @enforce_budget decorator."""

        @enforce_budget(budget_limit=10.00)
        def process_text(text):
            return f"Processed: {text}"

        assert hasattr(process_text, 'copilot')
        assert process_text.copilot.budget_limit == 10.00
        assert process_text.copilot.on_budget_exceeded == "raise"

    def test_enforce_budget_raises_on_exceed(self):
        """Test that @enforce_budget uses 'raise' action."""

        @enforce_budget(budget_limit=10.00)
        def process_text(text):
            return f"Processed: {text}"

        # Should enforce budget strictly
        assert process_text.copilot.on_budget_exceeded == "raise"

    def test_monitored_decorator_basic(self):
        """Test basic @monitored decorator."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin'), \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            @monitored(budget_limit=20.00)
            def process_text(text):
                return f"Processed: {text}"

            assert hasattr(process_text, 'copilot')
            assert process_text.copilot.budget_limit == 20.00
            # Should have plugins
            assert len(process_text.copilot._plugin_manager.get_all_plugins()) > 0

    def test_monitored_with_webhook(self):
        """Test @monitored decorator with webhook."""
        with patch('token_copilot.plugins.streaming.StreamingPlugin') as MockStreaming, \
             patch('token_copilot.plugins.analytics.AnalyticsPlugin'), \
             patch('token_copilot.plugins.forecasting.ForecastingPlugin'):

            @monitored(
                budget_limit=20.00,
                webhook_url="https://example.com/webhook"
            )
            def process_text(text):
                return f"Processed: {text}"

            MockStreaming.assert_called()

    def test_multiple_decorators_same_function(self):
        """Test that multiple decorators can't be applied (last wins)."""

        @track_cost(budget_limit=5.00)
        @enforce_budget(budget_limit=10.00)
        def process_text(text):
            return f"Processed: {text}"

        # The outer decorator (track_cost) should win
        assert process_text.copilot.budget_limit == 5.00

    def test_decorator_on_class_method(self):
        """Test decorator on class methods."""

        class TextProcessor:
            @track_cost(budget_limit=5.00)
            def process(self, text):
                return f"Processed: {text}"

        processor = TextProcessor()
        result = processor.process("hello")
        assert result == "Processed: hello"

        # Copilot should be on the method
        assert hasattr(processor.process, 'copilot')

    def test_decorator_with_async_function(self):
        """Test decorator with async function."""
        import asyncio

        @track_cost(budget_limit=5.00)
        async def async_process(text):
            await asyncio.sleep(0.01)
            return f"Processed: {text}"

        # Should have copilot
        assert hasattr(async_process, 'copilot')

        # Function should work
        result = asyncio.run(async_process("hello"))
        assert result == "Processed: hello"

    def test_decorator_returns_copilot_instance(self):
        """Test that decorators attach TokenCoPilot instance."""
        from token_copilot import TokenCoPilot

        @track_cost(budget_limit=5.00)
        def process_text(text):
            return f"Processed: {text}"

        assert isinstance(process_text.copilot, TokenCoPilot)

    def test_decorated_function_can_access_copilot(self):
        """Test that decorated function can access copilot inside."""

        @track_cost(budget_limit=5.00)
        def process_text(text):
            # Access copilot through function attribute
            return f"Budget: ${process_text.copilot.budget_limit}"

        result = process_text("hello")
        assert "5.0" in result or "5.00" in result
