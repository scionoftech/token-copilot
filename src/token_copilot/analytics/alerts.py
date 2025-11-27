"""Alert handlers for anomaly detection."""

import logging
from typing import Callable
from .anomaly_detector import Anomaly


def log_alert(anomaly: Anomaly) -> None:
    """
    Log anomaly to console.

    Simple alert handler that logs anomalies using Python logging.

    Args:
        anomaly: Detected anomaly

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> from token_copilot.analytics import log_alert
        >>> callback = TokenPilotCallback(
        ...     anomaly_detection=True,
        ...     alert_handlers=[log_alert]
        ... )
    """
    logger = logging.getLogger('token_copilot.anomaly')

    level_map = {
        'low': logging.INFO,
        'medium': logging.WARNING,
        'high': logging.ERROR,
        'critical': logging.CRITICAL,
    }

    level = level_map.get(anomaly.severity, logging.WARNING)
    logger.log(
        level,
        f"[{anomaly.severity.upper()}] {anomaly.message}"
    )


def webhook_alert(url: str) -> Callable[[Anomaly], None]:
    """
    Create webhook alert handler.

    Returns a callable that POSTs anomaly data to specified URL.

    Args:
        url: Webhook URL to POST to

    Returns:
        Callable that accepts Anomaly and sends to webhook

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> from token_copilot.analytics import webhook_alert
        >>> callback = TokenPilotCallback(
        ...     anomaly_detection=True,
        ...     alert_handlers=[webhook_alert('https://example.com/webhook')]
        ... )
    """
    def handler(anomaly: Anomaly) -> None:
        """Send anomaly to webhook."""
        try:
            import json
            try:
                import requests
            except ImportError:
                logging.warning(
                    "requests library required for webhook alerts. "
                    "Install with: pip install requests"
                )
                return

            payload = {
                'timestamp': anomaly.timestamp.isoformat(),
                'type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'message': anomaly.message,
                'value': anomaly.value,
                'mean': anomaly.mean,
                'std_dev': anomaly.std_dev,
                'z_score': anomaly.z_score,
                'metadata': anomaly.metadata,
            }

            response = requests.post(
                url,
                json=payload,
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

        except Exception as e:
            logging.error(f"Webhook alert failed: {e}")

    return handler


def slack_alert(webhook_url: str) -> Callable[[Anomaly], None]:
    """
    Create Slack alert handler.

    Returns a callable that posts anomaly to Slack via webhook.

    Args:
        webhook_url: Slack webhook URL

    Returns:
        Callable that accepts Anomaly and sends to Slack

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> from token_copilot.analytics import slack_alert
        >>> callback = TokenPilotCallback(
        ...     anomaly_detection=True,
        ...     alert_handlers=[
        ...         slack_alert('https://hooks.slack.com/services/YOUR/WEBHOOK/URL')
        ...     ]
        ... )
    """
    def handler(anomaly: Anomaly) -> None:
        """Send anomaly to Slack."""
        try:
            import json
            try:
                import requests
            except ImportError:
                logging.warning(
                    "requests library required for Slack alerts. "
                    "Install with: pip install requests"
                )
                return

            # Color code by severity
            color_map = {
                'low': '#36a64f',  # Green
                'medium': '#ff9900',  # Orange
                'high': '#ff0000',  # Red
                'critical': '#8b0000',  # Dark red
            }
            color = color_map.get(anomaly.severity, '#808080')

            # Icon by severity
            icon_map = {
                'low': ':information_source:',
                'medium': ':warning:',
                'high': ':rotating_light:',
                'critical': ':fire:',
            }
            icon = icon_map.get(anomaly.severity, ':bell:')

            payload = {
                'attachments': [
                    {
                        'color': color,
                        'title': f"{icon} TokenPilot Anomaly Detected",
                        'text': anomaly.message,
                        'fields': [
                            {
                                'title': 'Severity',
                                'value': anomaly.severity.upper(),
                                'short': True,
                            },
                            {
                                'title': 'Type',
                                'value': anomaly.anomaly_type.replace('_', ' ').title(),
                                'short': True,
                            },
                            {
                                'title': 'Value',
                                'value': f"{anomaly.value:.4f}",
                                'short': True,
                            },
                            {
                                'title': 'Mean',
                                'value': f"{anomaly.mean:.4f}",
                                'short': True,
                            },
                        ],
                        'footer': 'TokenPilot',
                        'ts': int(anomaly.timestamp.timestamp()),
                    }
                ]
            }

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

        except Exception as e:
            logging.error(f"Slack alert failed: {e}")

    return handler
