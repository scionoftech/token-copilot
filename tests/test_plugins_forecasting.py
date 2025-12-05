"""Tests for ForecastingPlugin."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.plugins.forecasting import ForecastingPlugin
from token_copilot.core.copilot import TokenCoPilot


class TestForecastingPlugin:
    """Tests for ForecastingPlugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.copilot = MagicMock(spec=TokenCoPilot)
        self.copilot.budget_limit = 100.00
        self.copilot.tracker = MagicMock()

    def test_init_default(self):
        """Test default initialization."""
        plugin = ForecastingPlugin()

        assert plugin.forecast_hours == 24
        assert plugin.alert_threshold == 0.8

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        plugin = ForecastingPlugin(
            forecast_hours=48,
            alert_threshold=0.9
        )

        assert plugin.forecast_hours == 48
        assert plugin.alert_threshold == 0.9

    def test_on_attach_creates_predictor(self):
        """Test that on_attach creates BudgetPredictor."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor:
            plugin = ForecastingPlugin()
            plugin.attach(self.copilot)

            MockPredictor.assert_called()

    def test_on_attach_creates_alert_manager(self):
        """Test that on_attach creates AlertManager."""
        with patch('token_copilot.forecasting.BudgetPredictor'), \
             patch('token_copilot.analytics.AlertManager') as MockManager:

            plugin = ForecastingPlugin()
            plugin.attach(self.copilot)

            MockManager.assert_called()

    def test_predict_exhaustion(self):
        """Test predict_exhaustion method."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor:
            mock_predictor = MagicMock()
            mock_predictor.predict_exhaustion.return_value = 12.5
            MockPredictor.return_value = mock_predictor

            plugin = ForecastingPlugin()
            plugin.attach(self.copilot)

            hours = plugin.predict_exhaustion()

            assert hours == 12.5
            mock_predictor.predict_exhaustion.assert_called()

    def test_predict_exhaustion_no_data(self):
        """Test predict_exhaustion with no data."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor:
            mock_predictor = MagicMock()
            mock_predictor.predict_exhaustion.return_value = None
            MockPredictor.return_value = mock_predictor

            plugin = ForecastingPlugin()
            plugin.attach(self.copilot)

            hours = plugin.predict_exhaustion()

            assert hours is None

    def test_get_forecast(self):
        """Test get_forecast method."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor:
            mock_predictor = MagicMock()
            mock_predictor.get_forecast.return_value = {
                "predicted_cost_24h": 50.00,
                "predicted_exhaustion_hours": 48.0,
                "confidence": 0.85
            }
            MockPredictor.return_value = mock_predictor

            plugin = ForecastingPlugin(forecast_hours=24)
            plugin.attach(self.copilot)

            forecast = plugin.get_forecast()

            assert forecast["predicted_cost_24h"] == 50.00
            assert forecast["predicted_exhaustion_hours"] == 48.0
            assert forecast["confidence"] == 0.85

    def test_on_cost_tracked_updates_forecast(self):
        """Test that on_cost_tracked updates forecast data."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor:
            mock_predictor = MagicMock()
            MockPredictor.return_value = mock_predictor

            plugin = ForecastingPlugin()
            plugin.attach(self.copilot)

            plugin.on_cost_tracked("gpt-4", 100, 50, 0.01, {})

            # Predictor should receive new data
            mock_predictor.add_data_point.assert_called()

    def test_on_cost_tracked_triggers_alert(self):
        """Test that on_cost_tracked triggers alert when threshold exceeded."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor, \
             patch('token_copilot.analytics.AlertManager') as MockManager:

            mock_predictor = MagicMock()
            mock_predictor.predict_exhaustion.return_value = 2.0  # 2 hours
            MockPredictor.return_value = mock_predictor

            mock_manager = MagicMock()
            MockManager.return_value = mock_manager

            plugin = ForecastingPlugin(alert_threshold=0.8)
            plugin.attach(self.copilot)

            # Simulate cost that would trigger alert
            self.copilot.cost = 85.00  # 85% of budget
            plugin.on_cost_tracked("gpt-4", 100, 50, 5.00, {})

            # Alert should be triggered
            mock_manager.send_alert.assert_called()

    def test_on_cost_tracked_no_alert_below_threshold(self):
        """Test that on_cost_tracked doesn't trigger alert below threshold."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor, \
             patch('token_copilot.analytics.AlertManager') as MockManager:

            mock_predictor = MagicMock()
            mock_predictor.predict_exhaustion.return_value = 48.0  # 48 hours
            MockPredictor.return_value = mock_predictor

            mock_manager = MagicMock()
            MockManager.return_value = mock_manager

            plugin = ForecastingPlugin(alert_threshold=0.8)
            plugin.attach(self.copilot)

            # Simulate cost below threshold
            self.copilot.cost = 50.00  # 50% of budget
            plugin.on_cost_tracked("gpt-4", 100, 50, 1.00, {})

            # Alert should NOT be triggered
            mock_manager.send_alert.assert_not_called()

    def test_get_forecast_confidence(self):
        """Test get_forecast returns confidence score."""
        with patch('token_copilot.forecasting.BudgetPredictor') as MockPredictor:
            mock_predictor = MagicMock()
            mock_predictor.get_confidence.return_value = 0.92
            MockPredictor.return_value = mock_predictor

            plugin = ForecastingPlugin()
            plugin.attach(self.copilot)

            confidence = plugin.get_forecast_confidence()

            assert confidence == 0.92

    def test_on_detach_clears_components(self):
        """Test that on_detach clears forecasting components."""
        plugin = ForecastingPlugin()
        plugin._predictor = MagicMock()
        plugin._alert_manager = MagicMock()

        plugin.detach()

        assert plugin._predictor is None
        assert plugin._alert_manager is None
