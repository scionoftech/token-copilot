"""Analytics plugin for waste detection, efficiency scoring, and anomaly detection."""

from typing import Any, Dict, List, Optional, Callable
from ..core.plugin import Plugin


class AnalyticsPlugin(Plugin):
    """Plugin for advanced analytics and monitoring.

    Features:
    - Token waste detection (repeated prompts, excessive context)
    - Efficiency scoring and leaderboards
    - Anomaly detection (cost/token/frequency spikes)
    - Custom alert handlers

    Example:
        >>> from token_copilot import TokenCoPilot
        >>> from token_copilot.plugins import AnalyticsPlugin
        >>> from token_copilot.analytics import log_alert, slack_alert
        >>>
        >>> copilot = TokenCoPilot(budget_limit=100.00)
        >>> copilot.add_plugin(AnalyticsPlugin(
        ...     detect_anomalies=True,
        ...     alert_handlers=[log_alert, slack_alert]
        ... ))

    Example (Builder):
        >>> copilot = (TokenCoPilot(budget_limit=100.00)
        ...     .with_analytics(detect_anomalies=True)
        ... )
    """

    def __init__(
        self,
        detect_anomalies: bool = True,
        anomaly_sensitivity: float = 3.0,
        alert_handlers: Optional[List[Callable]] = None,
        track_waste: bool = True,
        track_efficiency: bool = True,
    ):
        """Initialize analytics plugin.

        Args:
            detect_anomalies: Enable real-time anomaly detection
            anomaly_sensitivity: Std dev threshold for anomalies (default: 3.0)
            alert_handlers: List of alert handler callables
            track_waste: Enable waste detection and analysis
            track_efficiency: Enable efficiency scoring
        """
        super().__init__()
        self.detect_anomalies = detect_anomalies
        self.anomaly_sensitivity = anomaly_sensitivity
        self.alert_handlers = alert_handlers or []
        self.track_waste = track_waste
        self.track_efficiency = track_efficiency

        self._waste_analyzer = None
        self._efficiency_scorer = None
        self._anomaly_detector = None

    def on_attach(self):
        """Initialize analytics components when attached."""
        try:
            if self.track_waste:
                from ..analytics import WasteAnalyzer
                self._waste_analyzer = WasteAnalyzer()

            if self.track_efficiency:
                from ..analytics import EfficiencyScorer
                self._efficiency_scorer = EfficiencyScorer()

            if self.detect_anomalies:
                from ..analytics import AnomalyDetector
                self._anomaly_detector = AnomalyDetector(
                    threshold=self.anomaly_sensitivity,
                    alert_handlers=self.alert_handlers
                )

        except ImportError as e:
            import logging
            logging.warning(
                f"Analytics plugin requires additional dependencies: {e}. "
                "Install with: pip install token-copilot[analytics]"
            )

    def on_cost_tracked(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        metadata: Dict[str, Any]
    ):
        """Run analytics on tracked cost."""
        # Anomaly detection
        if self._anomaly_detector and self.copilot:
            self._anomaly_detector.check(self.copilot.tracker)

        # Waste analysis (done on-demand via analyze_waste())
        # Efficiency tracking (done on-demand via get_efficiency_score())

    # Public API methods

    def analyze_waste(self) -> Dict[str, Any]:
        """Analyze token waste patterns.

        Returns:
            Dictionary with waste analysis results

        Example:
            >>> analytics = copilot.get_plugins(AnalyticsPlugin)[0]
            >>> waste_report = analytics.analyze_waste()
            >>> print(f"Potential savings: ${waste_report['total_potential_savings']:.2f}")
        """
        if not self._waste_analyzer or not self.copilot:
            return {}
        return self._waste_analyzer.analyze(self.copilot.tracker)

    def get_efficiency_score(self, entity_type: str, entity_id: str):
        """Get efficiency score for a user or org.

        Args:
            entity_type: "user_id" or "org_id"
            entity_id: ID of the entity

        Returns:
            EfficiencyMetrics object

        Example:
            >>> analytics = copilot.get_plugins(AnalyticsPlugin)[0]
            >>> metrics = analytics.get_efficiency_score("user_id", "user_123")
            >>> print(f"Efficiency: {metrics.score:.2f}")
        """
        if not self._efficiency_scorer or not self.copilot:
            return None
        return self._efficiency_scorer.get_efficiency(
            self.copilot.tracker,
            entity_type,
            entity_id
        )

    def get_leaderboard(self, entity_type: str = "user_id", top_n: int = 10) -> List[Dict]:
        """Get efficiency leaderboard.

        Args:
            entity_type: "user_id" or "org_id"
            top_n: Number of top performers to return

        Returns:
            List of dictionaries with rankings

        Example:
            >>> analytics = copilot.get_plugins(AnalyticsPlugin)[0]
            >>> leaderboard = analytics.get_leaderboard("user_id", top_n=5)
            >>> for rank in leaderboard:
            ...     print(f"{rank['entity_id']}: {rank['efficiency_score']:.2f}")
        """
        if not self._efficiency_scorer or not self.copilot:
            return []
        return self._efficiency_scorer.get_leaderboard(
            self.copilot.tracker,
            entity_type,
            top_n
        )

    def get_anomalies(self, minutes: int = 60, min_severity: str = "medium") -> List:
        """Get recent anomalies.

        Args:
            minutes: Look back this many minutes
            min_severity: Minimum severity ("low", "medium", "high", "critical")

        Returns:
            List of Anomaly objects

        Example:
            >>> analytics = copilot.get_plugins(AnalyticsPlugin)[0]
            >>> anomalies = analytics.get_anomalies(minutes=30, min_severity="high")
            >>> for anomaly in anomalies:
            ...     print(f"{anomaly.anomaly_type}: {anomaly.message}")
        """
        if not self._anomaly_detector:
            return []
        return self._anomaly_detector.get_recent_anomalies(minutes, min_severity)
