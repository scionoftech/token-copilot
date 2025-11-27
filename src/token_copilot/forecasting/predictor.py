"""Predictive budget forecasting using time series analysis."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from collections import deque
from enum import Enum
import numpy as np


class Trend(str, Enum):
    """Budget trend classification."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class BudgetForecast:
    """Budget forecast result."""

    timestamp: datetime
    current_cost: float
    budget_limit: float
    remaining_budget: float
    burn_rate_per_hour: float
    hours_until_exhausted: Optional[float]
    days_until_exhausted: Optional[float]
    trend: str
    confidence: float  # 0-1
    projected_cost_24h: float
    projected_cost_7d: float
    projected_cost_30d: float
    recommendations: List[str]


class BudgetPredictor:
    """
    Predicts budget exhaustion using time series analysis.

    Analyzes recent spending patterns using linear regression to forecast
    when budget will be exhausted. Provides confidence scores based on
    data quality and pattern consistency.

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> callback = TokenPilotCallback(
        ...     budget_limit=100.00,
        ...     predictive_alerts=True,
        ...     forecast_window_hours=24
        ... )
        >>> # ... make LLM calls ...
        >>> forecast = callback.get_forecast()
        >>> if forecast.hours_until_exhausted and forecast.hours_until_exhausted < 4:
        ...     print(f"WARNING: Budget exhausts in {forecast.hours_until_exhausted:.1f}h")
    """

    def __init__(
        self,
        window_hours: int = 24,
        min_data_points: int = 10,
    ):
        """
        Initialize BudgetPredictor.

        Args:
            window_hours: Hours of history to analyze for trend
            min_data_points: Minimum data points needed for prediction
        """
        self.window_hours = window_hours
        self.min_data_points = min_data_points

        # History: (timestamp, cost) pairs
        self._history: deque = deque(maxlen=1000)

    def record(self, cost: float, timestamp: Optional[datetime] = None):
        """
        Record a cost event.

        Args:
            cost: Cost of the event
            timestamp: Event timestamp (default: now)

        Example:
            >>> predictor.record(0.05)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self._history.append((timestamp, cost))

    def forecast(
        self,
        budget_limit: float,
        current_cost: float,
        forecast_hours: int = 24,
    ) -> BudgetForecast:
        """
        Generate budget forecast.

        Args:
            budget_limit: Total budget limit
            current_cost: Current total cost
            forecast_hours: Hours to forecast ahead

        Returns:
            BudgetForecast with predictions and recommendations

        Example:
            >>> forecast = predictor.forecast(
            ...     budget_limit=100.00,
            ...     current_cost=75.00,
            ...     forecast_hours=24
            ... )
            >>> print(f"Remaining: ${forecast.remaining_budget:.2f}")
            >>> print(f"Hours until exhausted: {forecast.hours_until_exhausted:.1f}")
        """
        timestamp = datetime.now()
        remaining_budget = max(0, budget_limit - current_cost)

        # Analyze trend
        burn_rate, trend, confidence = self._analyze_trend()

        # Calculate hours until exhausted
        if burn_rate > 0 and remaining_budget > 0:
            hours_until_exhausted = remaining_budget / burn_rate
            days_until_exhausted = hours_until_exhausted / 24
        else:
            hours_until_exhausted = None
            days_until_exhausted = None

        # Project future costs
        projected_24h = current_cost + (burn_rate * 24)
        projected_7d = current_cost + (burn_rate * 24 * 7)
        projected_30d = current_cost + (burn_rate * 24 * 30)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            hours_until_exhausted=hours_until_exhausted,
            trend=trend,
            burn_rate=burn_rate,
            confidence=confidence,
        )

        return BudgetForecast(
            timestamp=timestamp,
            current_cost=current_cost,
            budget_limit=budget_limit,
            remaining_budget=remaining_budget,
            burn_rate_per_hour=burn_rate,
            hours_until_exhausted=hours_until_exhausted,
            days_until_exhausted=days_until_exhausted,
            trend=trend.value,
            confidence=confidence,
            projected_cost_24h=projected_24h,
            projected_cost_7d=projected_7d,
            projected_cost_30d=projected_30d,
            recommendations=recommendations,
        )

    def _analyze_trend(self) -> Tuple[float, Trend, float]:
        """
        Analyze spending trend using linear regression.

        Returns:
            Tuple of (burn_rate_per_hour, trend, confidence)
        """
        if len(self._history) < self.min_data_points:
            return 0.0, Trend.STABLE, 0.0

        # Filter to recent window
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        recent = [(t, c) for t, c in self._history if t >= cutoff]

        if len(recent) < self.min_data_points:
            return 0.0, Trend.STABLE, 0.0

        # Convert to hours since first event
        start_time = recent[0][0]
        hours = [(t - start_time).total_seconds() / 3600 for t, c in recent]
        cumsum_costs = np.cumsum([c for t, c in recent])

        # Linear regression: cumsum_cost = slope * hours + intercept
        try:
            coefficients = np.polyfit(hours, cumsum_costs, 1)
            slope = coefficients[0]  # Cost per hour
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, Trend.STABLE, 0.0

        # Classify trend
        if slope > 0.001:  # Increasing
            trend = Trend.INCREASING
        elif slope < -0.001:  # Decreasing
            trend = Trend.DECREASING
        else:  # Stable
            trend = Trend.STABLE

        # Calculate confidence
        confidence = self._calculate_confidence(
            hours=hours,
            costs=cumsum_costs,
            slope=slope,
        )

        return slope, trend, confidence

    def _calculate_confidence(
        self,
        hours: List[float],
        costs: np.ndarray,
        slope: float,
    ) -> float:
        """
        Calculate forecast confidence (0-1).

        Based on:
        - Data amount (more data = higher confidence)
        - Pattern consistency (lower variance = higher confidence)
        """
        # Data amount factor (0-1)
        data_factor = min(1.0, len(hours) / 100)

        # Pattern consistency (coefficient of variation)
        if len(costs) > 1 and np.mean(costs) > 0:
            cv = np.std(costs) / np.mean(costs)
            # Lower CV = higher confidence
            consistency_factor = 1.0 / (1.0 + cv)
        else:
            consistency_factor = 0.5

        # Combined confidence
        confidence = (data_factor + consistency_factor) / 2

        return min(1.0, max(0.0, confidence))

    def _generate_recommendations(
        self,
        hours_until_exhausted: Optional[float],
        trend: Trend,
        burn_rate: float,
        confidence: float,
    ) -> List[str]:
        """Generate recommendations based on forecast."""
        recommendations = []

        # Budget exhaustion warnings
        if hours_until_exhausted is not None:
            if hours_until_exhausted < 1:
                recommendations.append(
                    "🔥 CRITICAL: Budget will be exhausted in less than 1 hour! "
                    "Immediate action required."
                )
            elif hours_until_exhausted < 4:
                recommendations.append(
                    "⚠️ WARNING: Budget will be exhausted in less than 4 hours. "
                    "Review spending immediately."
                )
            elif hours_until_exhausted < 24:
                recommendations.append(
                    "ℹ️ INFO: Budget will be exhausted within 24 hours. "
                    "Plan accordingly."
                )

        # Trend-based recommendations
        if trend == Trend.INCREASING:
            if burn_rate > 1.0:  # $1/hour
                recommendations.append(
                    f"Spending is increasing rapidly (${burn_rate:.2f}/hour). "
                    "Consider using cheaper models or reducing usage."
                )
            else:
                recommendations.append(
                    "Spending trend is increasing. Monitor closely."
                )
        elif trend == Trend.DECREASING:
            recommendations.append(
                "Spending trend is decreasing. Recent optimizations are working."
            )

        # Confidence warnings
        if confidence < 0.3:
            recommendations.append(
                "⚠️ Low forecast confidence due to limited data. "
                "Predictions may be unreliable."
            )

        # General advice
        if not recommendations:
            recommendations.append(
                "Budget is within normal range. Continue monitoring."
            )

        return recommendations
