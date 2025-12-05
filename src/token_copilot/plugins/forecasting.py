"""Forecasting plugin for budget prediction and alerts."""

from typing import Any, Dict, Optional
from ..core.plugin import Plugin


class ForecastingPlugin(Plugin):
    """Plugin for budget forecasting and predictive alerts.

    Features:
    - Linear regression budget forecasting
    - Burn rate analysis
    - Predictive budget exhaustion alerts
    - Custom alert rules

    Example:
        >>> from token_copilot import TokenCoPilot
        >>> from token_copilot.plugins import ForecastingPlugin
        >>>
        >>> copilot = TokenCoPilot(budget_limit=100.00)
        >>> copilot.add_plugin(ForecastingPlugin(forecast_hours=24))

    Example (Builder):
        >>> copilot = (TokenCoPilot(budget_limit=100.00)
        ...     .with_forecasting(forecast_hours=48)
        ... )
    """

    def __init__(
        self,
        forecast_hours: int = 24,
        enable_alerts: bool = True,
    ):
        """Initialize forecasting plugin.

        Args:
            forecast_hours: Hours of history to use for forecasting
            enable_alerts: Enable predictive budget alerts
        """
        super().__init__()
        self.forecast_hours = forecast_hours
        self.enable_alerts = enable_alerts
        self._predictor = None
        self._alert_manager = None

    def on_attach(self):
        """Initialize forecasting components when attached."""
        try:
            from ..forecasting import BudgetPredictor, AlertManager

            self._predictor = BudgetPredictor()

            if self.enable_alerts:
                self._alert_manager = AlertManager()

        except ImportError as e:
            import logging
            logging.warning(
                f"Forecasting plugin requires additional dependencies: {e}. "
                "Install with: pip install token-copilot[analytics]"
            )

    # Public API methods

    def get_forecast(self, forecast_hours: Optional[int] = None):
        """Get budget forecast.

        Args:
            forecast_hours: Hours to forecast (default: plugin setting)

        Returns:
            BudgetForecast object with predictions

        Example:
            >>> forecasting = copilot.get_plugins(ForecastingPlugin)[0]
            >>> forecast = forecasting.get_forecast(hours=48)
            >>> print(f"Burn rate: ${forecast.burn_rate_per_hour:.4f}/hr")
            >>> print(f"Budget exhaustion: {forecast.hours_until_exhaustion:.1f} hours")
        """
        if not self._predictor or not self.copilot:
            return None

        hours = forecast_hours or self.forecast_hours
        return self._predictor.predict(
            self.copilot.tracker,
            self.copilot.budget_limit,
            forecast_hours=hours
        )

    def add_alert_rule(self, rule):
        """Add a custom alert rule.

        Args:
            rule: AlertRule object

        Example:
            >>> from token_copilot.forecasting import AlertRule
            >>> forecasting = copilot.get_plugins(ForecastingPlugin)[0]
            >>>
            >>> rule = AlertRule(
            ...     name="high_burn_rate",
            ...     condition=lambda f: f.burn_rate_per_hour > 10.0,
            ...     message="High burn rate detected!"
            ... )
            >>> forecasting.add_alert_rule(rule)
        """
        if not self._alert_manager:
            return

        self._alert_manager.add_rule(rule)
