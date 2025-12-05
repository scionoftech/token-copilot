"""Tests for AdaptivePlugin."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.plugins.adaptive import AdaptivePlugin
from token_copilot.core.copilot import TokenCoPilot


class TestAdaptivePlugin:
    """Tests for AdaptivePlugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.copilot = MagicMock(spec=TokenCoPilot)
        self.copilot.budget_limit = 100.00

    def test_init_default(self):
        """Test default initialization."""
        plugin = AdaptivePlugin()

        # Should have default tier thresholds
        assert plugin.tier_thresholds is not None

    def test_init_custom_thresholds(self):
        """Test initialization with custom tier thresholds."""
        thresholds = {
            "low": {"max_budget": 10.0},
            "medium": {"max_budget": 100.0},
            "high": {"max_budget": float('inf')}
        }
        plugin = AdaptivePlugin(tier_thresholds=thresholds)

        assert plugin.tier_thresholds == thresholds

    def test_on_attach_creates_adaptive_ops(self):
        """Test that on_attach creates AdaptiveOperations."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            MockOps.assert_called()

    def test_get_tier_low_budget(self):
        """Test get_tier returns 'low' for low budget."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.get_tier.return_value = "low"
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            tier = plugin.get_tier()

            assert tier == "low"

    def test_get_tier_medium_budget(self):
        """Test get_tier returns 'medium' for medium budget."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.get_tier.return_value = "medium"
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            tier = plugin.get_tier()

            assert tier == "medium"

    def test_get_tier_high_budget(self):
        """Test get_tier returns 'high' for high budget."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.get_tier.return_value = "high"
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            tier = plugin.get_tier()

            assert tier == "high"

    def test_get_recommended_params_low_tier(self):
        """Test get_recommended_params for low tier."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.get_recommended_params.return_value = {
                "max_tokens": 512,
                "temperature": 0.3
            }
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            params = plugin.get_recommended_params()

            assert params["max_tokens"] == 512
            assert params["temperature"] == 0.3

    def test_get_recommended_params_high_tier(self):
        """Test get_recommended_params for high tier."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.get_recommended_params.return_value = {
                "max_tokens": 4096,
                "temperature": 0.7
            }
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            params = plugin.get_recommended_params()

            assert params["max_tokens"] == 4096
            assert params["temperature"] == 0.7

    def test_should_use_streaming(self):
        """Test should_use_streaming based on tier."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.should_use_streaming.return_value = True
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            result = plugin.should_use_streaming()

            assert result is True

    def test_should_use_caching(self):
        """Test should_use_caching based on tier."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            mock_ops.should_use_caching.return_value = False
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            result = plugin.should_use_caching()

            assert result is False

    def test_on_cost_tracked_updates_tier(self):
        """Test that on_cost_tracked can trigger tier updates."""
        with patch('token_copilot.adaptive.AdaptiveOperations') as MockOps:
            mock_ops = MagicMock()
            MockOps.return_value = mock_ops

            plugin = AdaptivePlugin()
            plugin.attach(self.copilot)

            # Simulate cost tracking
            plugin.on_cost_tracked("gpt-4", 100, 50, 0.01, {})

            # Tier should be checked/updated
            mock_ops.get_tier.assert_called()

    def test_on_detach_clears_adaptive_ops(self):
        """Test that on_detach clears adaptive operations."""
        plugin = AdaptivePlugin()
        plugin._adaptive_ops = MagicMock()

        plugin.detach()

        assert plugin._adaptive_ops is None
