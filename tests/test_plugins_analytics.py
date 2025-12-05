"""Tests for AnalyticsPlugin."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.plugins.analytics import AnalyticsPlugin
from token_copilot.core.copilot import TokenCoPilot


class TestAnalyticsPlugin:
    """Tests for AnalyticsPlugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.copilot = MagicMock(spec=TokenCoPilot)
        self.copilot.tracker = MagicMock()

    def test_init_with_anomaly_detection(self):
        """Test initialization with anomaly detection."""
        plugin = AnalyticsPlugin(
            detect_anomalies=True,
            anomaly_sensitivity=2.5
        )

        assert plugin.detect_anomalies is True
        assert plugin.anomaly_sensitivity == 2.5

    def test_init_with_waste_detection(self):
        """Test initialization with waste detection."""
        plugin = AnalyticsPlugin(detect_waste=True)

        assert plugin.detect_waste is True

    def test_init_with_efficiency_scoring(self):
        """Test initialization with efficiency scoring."""
        plugin = AnalyticsPlugin(track_efficiency=True)

        assert plugin.track_efficiency is True

    def test_init_with_alert_handlers(self):
        """Test initialization with alert handlers."""
        handler = MagicMock()
        plugin = AnalyticsPlugin(alert_handlers=[handler])

        assert handler in plugin.alert_handlers

    def test_on_attach_creates_waste_analyzer(self):
        """Test that on_attach creates WasteAnalyzer when enabled."""
        with patch('token_copilot.analytics.WasteAnalyzer') as MockAnalyzer:
            plugin = AnalyticsPlugin(detect_waste=True)
            plugin.attach(self.copilot)

            MockAnalyzer.assert_called()

    def test_on_attach_creates_anomaly_detector(self):
        """Test that on_attach creates AnomalyDetector when enabled."""
        with patch('token_copilot.analytics.AnomalyDetector') as MockDetector:
            plugin = AnalyticsPlugin(
                detect_anomalies=True,
                anomaly_sensitivity=3.0
            )
            plugin.attach(self.copilot)

            MockDetector.assert_called()

    def test_on_attach_creates_efficiency_scorer(self):
        """Test that on_attach creates EfficiencyScorer when enabled."""
        with patch('token_copilot.analytics.EfficiencyScorer') as MockScorer:
            plugin = AnalyticsPlugin(track_efficiency=True)
            plugin.attach(self.copilot)

            MockScorer.assert_called()

    def test_on_attach_creates_alert_manager(self):
        """Test that on_attach creates AlertManager when handlers provided."""
        with patch('token_copilot.analytics.AlertManager') as MockManager:
            handler = MagicMock()
            plugin = AnalyticsPlugin(alert_handlers=[handler])
            plugin.attach(self.copilot)

            MockManager.assert_called()

    def test_analyze_waste(self):
        """Test analyze_waste method."""
        with patch('token_copilot.analytics.WasteAnalyzer') as MockAnalyzer:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = {"wasted_tokens": 100}
            MockAnalyzer.return_value = mock_instance

            plugin = AnalyticsPlugin(detect_waste=True)
            plugin.attach(self.copilot)

            result = plugin.analyze_waste()

            assert result == {"wasted_tokens": 100}
            mock_instance.analyze.assert_called_once()

    def test_get_efficiency_score(self):
        """Test get_efficiency_score method."""
        with patch('token_copilot.analytics.EfficiencyScorer') as MockScorer:
            mock_instance = MagicMock()
            mock_instance.get_score.return_value = 0.85
            MockScorer.return_value = mock_instance

            plugin = AnalyticsPlugin(track_efficiency=True)
            plugin.attach(self.copilot)

            score = plugin.get_efficiency_score()

            assert score == 0.85
            mock_instance.get_score.assert_called_once()

    def test_on_cost_tracked_detects_anomalies(self):
        """Test that on_cost_tracked detects anomalies."""
        with patch('token_copilot.analytics.AnomalyDetector') as MockDetector:
            mock_instance = MagicMock()
            mock_instance.detect.return_value = True
            MockDetector.return_value = mock_instance

            plugin = AnalyticsPlugin(detect_anomalies=True)
            plugin.attach(self.copilot)

            plugin.on_cost_tracked("gpt-4", 100, 50, 10.00, {})

            mock_instance.detect.assert_called()

    def test_on_cost_tracked_triggers_alerts(self):
        """Test that on_cost_tracked triggers alerts when anomaly detected."""
        with patch('token_copilot.analytics.AnomalyDetector') as MockDetector, \
             patch('token_copilot.analytics.AlertManager') as MockManager:

            mock_detector = MagicMock()
            mock_detector.detect.return_value = True
            MockDetector.return_value = mock_detector

            mock_manager = MagicMock()
            MockManager.return_value = mock_manager

            handler = MagicMock()
            plugin = AnalyticsPlugin(
                detect_anomalies=True,
                alert_handlers=[handler]
            )
            plugin.attach(self.copilot)

            plugin.on_cost_tracked("gpt-4", 100, 50, 10.00, {})

            # Alert manager should be triggered
            mock_manager.send_alert.assert_called()

    def test_on_detach_clears_components(self):
        """Test that on_detach clears analytics components."""
        plugin = AnalyticsPlugin(detect_waste=True, detect_anomalies=True)
        plugin._waste_analyzer = MagicMock()
        plugin._anomaly_detector = MagicMock()

        plugin.detach()

        assert plugin._waste_analyzer is None
        assert plugin._anomaly_detector is None
