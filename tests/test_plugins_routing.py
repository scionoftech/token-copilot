"""Tests for RoutingPlugin."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.plugins.routing import RoutingPlugin
from token_copilot.core.copilot import TokenCoPilot
from token_copilot.utils import ModelConfig


class TestRoutingPlugin:
    """Tests for RoutingPlugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.copilot = MagicMock(spec=TokenCoPilot)
        self.models = [
            ModelConfig("gpt-4o-mini", 0.7, 0.15, 0.60, 128000),
            ModelConfig("gpt-4o", 0.9, 5.0, 15.0, 128000),
        ]

    def test_init_with_models(self):
        """Test initialization with model configs."""
        plugin = RoutingPlugin(models=self.models)

        assert len(plugin.models) == 2
        assert plugin.models[0].name == "gpt-4o-mini"

    def test_init_with_strategy(self):
        """Test initialization with routing strategy."""
        plugin = RoutingPlugin(
            models=self.models,
            strategy="balanced"
        )

        assert plugin.strategy == "balanced"

    def test_init_default_strategy(self):
        """Test default strategy is balanced."""
        plugin = RoutingPlugin(models=self.models)

        assert plugin.strategy == "balanced"

    def test_on_attach_creates_router(self):
        """Test that on_attach creates ModelRouter."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            plugin = RoutingPlugin(models=self.models)
            plugin.attach(self.copilot)

            MockRouter.assert_called()

    def test_select_model_cheapest(self):
        """Test select_model with cheapest strategy."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            mock_router.select_model.return_value = self.models[0]
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models, strategy="cheapest")
            plugin.attach(self.copilot)

            result = plugin.select_model("Test prompt")

            assert result.name == "gpt-4o-mini"
            mock_router.select_model.assert_called()

    def test_select_model_quality(self):
        """Test select_model with quality strategy."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            mock_router.select_model.return_value = self.models[1]
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models, strategy="quality")
            plugin.attach(self.copilot)

            result = plugin.select_model("Complex task")

            assert result.name == "gpt-4o"
            mock_router.select_model.assert_called()

    def test_select_model_balanced(self):
        """Test select_model with balanced strategy."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            mock_router.select_model.return_value = self.models[0]
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models, strategy="balanced")
            plugin.attach(self.copilot)

            result = plugin.select_model("Medium task")

            mock_router.select_model.assert_called()

    def test_select_model_adaptive(self):
        """Test select_model with adaptive strategy."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            mock_router.select_model.return_value = self.models[0]
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models, strategy="adaptive")
            plugin.attach(self.copilot)

            result = plugin.select_model("Task")

            mock_router.select_model.assert_called()

    def test_select_model_quality_based(self):
        """Test select_model with quality_based strategy."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            mock_router.select_model.return_value = self.models[1]
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models, strategy="quality_based")
            plugin.attach(self.copilot)

            result = plugin.select_model("Task", quality_threshold=0.8)

            mock_router.select_model.assert_called()

    def test_on_llm_end_updates_quality_scores(self):
        """Test that on_llm_end updates quality scores for quality_based strategy."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models, strategy="quality_based")
            plugin.attach(self.copilot)

            mock_response = MagicMock()
            plugin.on_llm_end(mock_response, "run-123", model_name="gpt-4o")

            # Should update quality scores
            mock_router.update_quality_score.assert_called()

    def test_get_routing_stats(self):
        """Test get_routing_stats method."""
        with patch('token_copilot.routing.ModelRouter') as MockRouter:
            mock_router = MagicMock()
            mock_router.get_stats.return_value = {
                "total_selections": 10,
                "model_distribution": {"gpt-4o-mini": 7, "gpt-4o": 3}
            }
            MockRouter.return_value = mock_router

            plugin = RoutingPlugin(models=self.models)
            plugin.attach(self.copilot)

            stats = plugin.get_routing_stats()

            assert stats["total_selections"] == 10
            assert stats["model_distribution"]["gpt-4o-mini"] == 7

    def test_on_detach_clears_router(self):
        """Test that on_detach clears router."""
        plugin = RoutingPlugin(models=self.models)
        plugin._router = MagicMock()

        plugin.detach()

        assert plugin._router is None
