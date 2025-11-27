"""Anomaly detection for cost and usage spikes."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum
import statistics


class Severity(str, Enum):
    """Anomaly severity levels."""

    LOW = "low"  # 1.5-2.0 standard deviations
    MEDIUM = "medium"  # 2.0-3.0 standard deviations
    HIGH = "high"  # 3.0-4.0 standard deviations
    CRITICAL = "critical"  # 4.0+ standard deviations


@dataclass
class Anomaly:
    """Detected anomaly."""

    timestamp: datetime
    anomaly_type: str  # 'cost_spike', 'token_spike', 'frequency_spike', 'unusual_model'
    severity: str  # Severity enum value
    value: float
    mean: float
    std_dev: float
    z_score: float
    message: str
    metadata: Dict[str, Any]


class AnomalyDetector:
    """
    Detects anomalies in LLM usage using rolling statistics.

    Monitors:
    - Cost spikes (per-call cost)
    - Token spikes (per-call tokens)
    - Frequency spikes (calls per minute)
    - Unusual model usage (models used <5% of recent calls)

    Uses standard deviation thresholds:
    - Low: 1.5σ
    - Medium: 2.0σ
    - High: 3.0σ (default)
    - Critical: 4.0σ+

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> from token_copilot.analytics import log_alert
        >>> callback = TokenPilotCallback(
        ...     anomaly_detection=True,
        ...     anomaly_sensitivity=3.0,
        ...     alert_handlers=[log_alert]
        ... )
        >>> # ... make LLM calls ...
        >>> anomalies = callback.get_anomalies(minutes=60)
        >>> for anomaly in anomalies:
        ...     print(f"{anomaly.severity}: {anomaly.message}")
    """

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 3.0,
        alert_handlers: Optional[List[Callable]] = None,
    ):
        """
        Initialize AnomalyDetector.

        Args:
            window_size: Number of recent data points to maintain
            threshold: Standard deviation threshold for anomaly detection
            alert_handlers: List of callables to invoke when anomaly detected
                           Each handler receives Anomaly object as argument
        """
        self.window_size = window_size
        self.threshold = threshold
        self.alert_handlers = alert_handlers or []

        # Rolling windows (deque for efficient append/pop)
        self._costs: deque = deque(maxlen=window_size)
        self._tokens: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        self._models: deque = deque(maxlen=window_size)

        # Detected anomalies
        self._anomalies: List[Anomaly] = []

    def check(
        self,
        cost: float,
        tokens: int,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Anomaly]:
        """
        Check for anomalies in current data point.

        Args:
            cost: Current call cost
            tokens: Current call tokens
            model: Model used
            metadata: Optional metadata dict

        Returns:
            Anomaly object if detected, None otherwise

        Example:
            >>> anomaly = detector.check(cost=5.00, tokens=50000, model="gpt-4")
            >>> if anomaly:
            ...     print(f"{anomaly.severity}: {anomaly.message}")
        """
        timestamp = datetime.now()
        metadata = metadata or {}

        anomaly = None

        # Check cost spike
        if len(self._costs) >= 10:  # Need minimum data for statistics
            cost_anomaly = self._check_spike(
                value=cost,
                history=list(self._costs),
                anomaly_type='cost_spike',
                unit='USD',
                metadata=metadata,
            )
            if cost_anomaly:
                anomaly = cost_anomaly

        # Check token spike
        if len(self._tokens) >= 10:
            token_anomaly = self._check_spike(
                value=float(tokens),
                history=list(self._tokens),
                anomaly_type='token_spike',
                unit='tokens',
                metadata=metadata,
            )
            if token_anomaly and (not anomaly or token_anomaly.z_score > anomaly.z_score):
                anomaly = token_anomaly

        # Check frequency spike
        if len(self._timestamps) >= 10:
            frequency_anomaly = self._check_frequency_spike(
                timestamp=timestamp,
                metadata=metadata,
            )
            if frequency_anomaly and (not anomaly or frequency_anomaly.z_score > anomaly.z_score):
                anomaly = frequency_anomaly

        # Check unusual model usage
        if len(self._models) >= 20:
            model_anomaly = self._check_unusual_model(
                model=model,
                metadata=metadata,
            )
            if model_anomaly and not anomaly:  # Lower priority
                anomaly = model_anomaly

        # Update rolling windows
        self._costs.append(cost)
        self._tokens.append(float(tokens))
        self._timestamps.append(timestamp)
        self._models.append(model)

        # Store and trigger alerts if anomaly detected
        if anomaly:
            self._anomalies.append(anomaly)
            self._trigger_alerts(anomaly)

        return anomaly

    def get_recent_anomalies(
        self,
        minutes: int = 60,
        min_severity: Optional[str] = None,
    ) -> List[Anomaly]:
        """
        Get recent anomalies.

        Args:
            minutes: Look back this many minutes
            min_severity: Minimum severity to include ('low', 'medium', 'high', 'critical')

        Returns:
            List of Anomaly objects

        Example:
            >>> recent = detector.get_recent_anomalies(minutes=30, min_severity='high')
            >>> print(f"Found {len(recent)} high+ severity anomalies")
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        anomalies = [a for a in self._anomalies if a.timestamp >= cutoff]

        if min_severity:
            severity_order = ['low', 'medium', 'high', 'critical']
            min_index = severity_order.index(min_severity)
            anomalies = [
                a for a in anomalies
                if severity_order.index(a.severity) >= min_index
            ]

        return anomalies

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get anomaly statistics.

        Returns:
            Dict with total count, counts by type, counts by severity

        Example:
            >>> stats = detector.get_statistics()
            >>> print(f"Total anomalies: {stats['total']}")
            >>> print(f"Critical: {stats['by_severity']['critical']}")
        """
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for anomaly in self._anomalies:
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1

        return {
            'total': len(self._anomalies),
            'by_type': by_type,
            'by_severity': by_severity,
            'window_size': self.window_size,
            'threshold': self.threshold,
        }

    def _check_spike(
        self,
        value: float,
        history: List[float],
        anomaly_type: str,
        unit: str,
        metadata: Dict[str, Any],
    ) -> Optional[Anomaly]:
        """Check if value is a spike compared to history."""
        if len(history) < 2:
            return None

        mean = statistics.mean(history)
        try:
            std_dev = statistics.stdev(history)
        except statistics.StatisticsError:
            # All values are the same
            std_dev = 0.0

        if std_dev == 0:
            # No variation, any different value is anomalous
            if value != mean:
                z_score = 10.0  # Arbitrary high value
            else:
                return None
        else:
            z_score = (value - mean) / std_dev

        # Check if exceeds threshold
        if abs(z_score) < self.threshold:
            return None

        # Classify severity
        severity = self._classify_severity(abs(z_score))

        message = (
            f"{anomaly_type.replace('_', ' ').title()}: "
            f"{value:.4f} {unit} "
            f"({z_score:+.2f}σ from mean {mean:.4f} {unit})"
        )

        return Anomaly(
            timestamp=datetime.now(),
            anomaly_type=anomaly_type,
            severity=severity.value,
            value=value,
            mean=mean,
            std_dev=std_dev,
            z_score=z_score,
            message=message,
            metadata=metadata,
        )

    def _check_frequency_spike(
        self,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[Anomaly]:
        """Check for frequency spike (calls per minute)."""
        # Calculate recent call rate
        recent_window = timedelta(minutes=1)
        recent_times = [
            t for t in self._timestamps
            if timestamp - t <= recent_window
        ]

        calls_per_minute = len(recent_times)

        # Calculate historical average
        if len(self._timestamps) < 10:
            return None

        # Calculate calls per minute for each minute in history
        historical_rates = []
        times_list = list(self._timestamps)
        for i, t in enumerate(times_list):
            count = sum(1 for other_t in times_list if abs((other_t - t).total_seconds()) <= 60)
            historical_rates.append(count)

        if not historical_rates:
            return None

        mean = statistics.mean(historical_rates)
        try:
            std_dev = statistics.stdev(historical_rates)
        except statistics.StatisticsError:
            std_dev = 0.0

        if std_dev == 0:
            if calls_per_minute != mean:
                z_score = 10.0
            else:
                return None
        else:
            z_score = (calls_per_minute - mean) / std_dev

        if abs(z_score) < self.threshold:
            return None

        severity = self._classify_severity(abs(z_score))

        message = (
            f"Frequency Spike: {calls_per_minute} calls/min "
            f"({z_score:+.2f}σ from mean {mean:.1f} calls/min)"
        )

        return Anomaly(
            timestamp=timestamp,
            anomaly_type='frequency_spike',
            severity=severity.value,
            value=float(calls_per_minute),
            mean=mean,
            std_dev=std_dev,
            z_score=z_score,
            message=message,
            metadata=metadata,
        )

    def _check_unusual_model(
        self,
        model: str,
        metadata: Dict[str, Any],
    ) -> Optional[Anomaly]:
        """Check for unusual model usage."""
        # Count model usage in recent history
        model_counts: Dict[str, int] = {}
        for m in self._models:
            model_counts[m] = model_counts.get(m, 0) + 1

        total = len(self._models)
        current_count = model_counts.get(model, 0)
        percentage = (current_count / total * 100) if total > 0 else 0

        # Flag if model is used <5% of the time
        if percentage >= 5.0:
            return None

        message = (
            f"Unusual Model: '{model}' used only {percentage:.1f}% "
            f"of recent calls ({current_count}/{total})"
        )

        return Anomaly(
            timestamp=datetime.now(),
            anomaly_type='unusual_model',
            severity=Severity.LOW.value,  # Always low severity
            value=percentage,
            mean=100.0 / len(model_counts) if model_counts else 0.0,
            std_dev=0.0,
            z_score=0.0,  # Not applicable
            message=message,
            metadata={**metadata, 'model': model},
        )

    def _classify_severity(self, z_score: float) -> Severity:
        """Classify anomaly severity based on z-score."""
        if z_score >= 4.0:
            return Severity.CRITICAL
        elif z_score >= 3.0:
            return Severity.HIGH
        elif z_score >= 2.0:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _trigger_alerts(self, anomaly: Anomaly):
        """Trigger registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(anomaly)
            except Exception as e:
                # Don't let handler errors break detection
                import logging
                logging.error(f"Alert handler failed: {e}")
