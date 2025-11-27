"""Alert management for budget forecasting."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional
import threading
import time
from .predictor import BudgetForecast


@dataclass
class AlertRule:
    """Alert rule definition."""

    name: str
    condition: Callable[[BudgetForecast], bool]
    handler: Callable[[BudgetForecast, str], None]
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None


class AlertManager:
    """
    Manages alert rules for budget forecasting.

    Supports:
    - Custom alert rules with conditions
    - Pre-defined rules (critical, warning, daily)
    - Cooldown periods to prevent spam
    - Background monitoring thread

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> def my_handler(forecast, rule_name):
        ...     print(f"Alert: {rule_name} - {forecast.hours_until_exhausted}h remaining")
        >>> callback = TokenPilotCallback(
        ...     budget_limit=100.00,
        ...     predictive_alerts=True
        ... )
        >>> # Custom alert rule
        >>> callback.alert_manager.add_rule(
        ...     name="custom_threshold",
        ...     condition=lambda f: f.remaining_budget < 20.0,
        ...     handler=my_handler,
        ...     cooldown_minutes=30
        ... )
    """

    def __init__(self):
        """Initialize AlertManager."""
        self._rules: Dict[str, AlertRule] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Add default rules
        self._add_default_rules()

    def add_rule(
        self,
        name: str,
        condition: Callable[[BudgetForecast], bool],
        handler: Callable[[BudgetForecast, str], None],
        cooldown_minutes: int = 60,
    ):
        """
        Add alert rule.

        Args:
            name: Rule name (unique identifier)
            condition: Callable that returns True if alert should trigger
            handler: Callable to invoke when alert triggers (receives forecast and rule_name)
            cooldown_minutes: Minutes to wait before re-triggering same rule

        Example:
            >>> manager.add_rule(
            ...     name="high_burn_rate",
            ...     condition=lambda f: f.burn_rate_per_hour > 5.0,
            ...     handler=lambda f, n: print(f"High burn: ${f.burn_rate_per_hour:.2f}/h"),
            ...     cooldown_minutes=30
            ... )
        """
        self._rules[name] = AlertRule(
            name=name,
            condition=condition,
            handler=handler,
            cooldown_minutes=cooldown_minutes,
            last_triggered=None,
        )

    def remove_rule(self, name: str):
        """
        Remove alert rule.

        Args:
            name: Rule name to remove

        Example:
            >>> manager.remove_rule("custom_threshold")
        """
        if name in self._rules:
            del self._rules[name]

    def check_alerts(self, forecast: BudgetForecast):
        """
        Check all alert rules and trigger handlers.

        Args:
            forecast: Current budget forecast

        Example:
            >>> forecast = predictor.forecast(100.00, 75.00)
            >>> manager.check_alerts(forecast)
        """
        now = datetime.now()

        for rule in self._rules.values():
            # Check cooldown
            if rule.last_triggered:
                cooldown = timedelta(minutes=rule.cooldown_minutes)
                if now - rule.last_triggered < cooldown:
                    continue  # Still in cooldown

            # Check condition
            try:
                if rule.condition(forecast):
                    # Trigger handler
                    try:
                        rule.handler(forecast, rule.name)
                        rule.last_triggered = now
                    except Exception as e:
                        import logging
                        logging.error(f"Alert handler '{rule.name}' failed: {e}")
            except Exception as e:
                import logging
                logging.error(f"Alert condition '{rule.name}' failed: {e}")

    def start_monitoring(
        self,
        predictor,  # BudgetPredictor instance
        budget_limit: float,
        get_current_cost: Callable[[], float],
        interval_minutes: int = 5,
    ):
        """
        Start background monitoring thread.

        Periodically checks forecast and triggers alerts.

        Args:
            predictor: BudgetPredictor instance
            budget_limit: Budget limit
            get_current_cost: Callable that returns current total cost
            interval_minutes: Check interval in minutes

        Example:
            >>> manager.start_monitoring(
            ...     predictor=predictor,
            ...     budget_limit=100.00,
            ...     get_current_cost=lambda: callback.get_total_cost(),
            ...     interval_minutes=5
            ... )
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return  # Already monitoring

        def monitor():
            while not self._stop_monitoring.is_set():
                try:
                    current_cost = get_current_cost()
                    forecast = predictor.forecast(budget_limit, current_cost)
                    self.check_alerts(forecast)
                except Exception as e:
                    import logging
                    logging.error(f"Monitoring error: {e}")

                # Wait for interval
                self._stop_monitoring.wait(interval_minutes * 60)

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """
        Stop background monitoring thread.

        Example:
            >>> manager.stop_monitoring()
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)

    def _add_default_rules(self):
        """Add pre-defined alert rules."""
        # Critical: <1 hour remaining
        self.add_rule(
            name="critical_budget",
            condition=lambda f: (
                f.hours_until_exhausted is not None
                and f.hours_until_exhausted < 1.0
            ),
            handler=self._default_handler,
            cooldown_minutes=15,
        )

        # Warning: <4 hours remaining
        self.add_rule(
            name="warning_budget",
            condition=lambda f: (
                f.hours_until_exhausted is not None
                and f.hours_until_exhausted < 4.0
            ),
            handler=self._default_handler,
            cooldown_minutes=30,
        )

        # Daily: <24 hours remaining
        self.add_rule(
            name="daily_budget",
            condition=lambda f: (
                f.hours_until_exhausted is not None
                and f.hours_until_exhausted < 24.0
            ),
            handler=self._default_handler,
            cooldown_minutes=60,
        )

    def _default_handler(self, forecast: BudgetForecast, rule_name: str):
        """Default handler that logs to console."""
        import logging

        logger = logging.getLogger('token_copilot.budget')

        if rule_name == "critical_budget":
            level = logging.CRITICAL
            emoji = "🔥"
        elif rule_name == "warning_budget":
            level = logging.WARNING
            emoji = "⚠️"
        else:
            level = logging.INFO
            emoji = "ℹ️"

        message = (
            f"{emoji} Budget Alert: {rule_name} - "
            f"${forecast.remaining_budget:.2f} remaining"
        )

        if forecast.hours_until_exhausted:
            message += f" (~{forecast.hours_until_exhausted:.1f}h until exhausted)"

        logger.log(level, message)
