"""Factory presets for common TokenCoPilot configurations.

Presets provide pre-configured TokenCoPilot instances for common use cases:
- basic(): Minimal tracking only
- development(): Local development with logging
- production(): Production-ready with monitoring
- enterprise(): Full featured with all plugins
"""

from typing import Optional, List
from .core import TokenCoPilot


def basic(budget_limit: Optional[float] = None) -> TokenCoPilot:
    """Create a basic TokenCoPilot with minimal configuration.

    Just tracks costs and tokens. Perfect for getting started.

    Args:
        budget_limit: Optional budget limit in USD

    Returns:
        TokenCoPilot instance

    Example:
        >>> from token_copilot.presets import basic
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> copilot = basic(budget_limit=10.00)
        >>> llm = ChatOpenAI(callbacks=[copilot])
        >>> result = llm.invoke("Hello")
        >>> print(f"Cost: ${copilot.cost:.4f}")
    """
    return TokenCoPilot(budget_limit=budget_limit)


def development(
    budget_limit: Optional[float] = None,
    detect_anomalies: bool = False,
) -> TokenCoPilot:
    """Create a TokenCoPilot configured for local development.

    Includes:
    - Cost tracking
    - Optional anomaly detection
    - Logging alerts (no external services)

    Args:
        budget_limit: Optional budget limit in USD
        detect_anomalies: Enable anomaly detection with log alerts

    Returns:
        TokenCoPilot instance

    Example:
        >>> from token_copilot.presets import development
        >>>
        >>> copilot = development(budget_limit=50.00, detect_anomalies=True)
        >>> llm = ChatOpenAI(callbacks=[copilot])
    """
    copilot = TokenCoPilot(budget_limit=budget_limit)

    if detect_anomalies:
        try:
            from .analytics import log_alert
            copilot.with_analytics(
                detect_anomalies=True,
                alert_handlers=[log_alert]
            )
        except ImportError:
            import logging
            logging.warning(
                "Analytics features require additional dependencies. "
                "Install with: pip install token-copilot[analytics]"
            )

    return copilot


def production(
    budget_limit: float,
    webhook_url: Optional[str] = None,
    slack_webhook: Optional[str] = None,
    detect_anomalies: bool = True,
    enable_forecasting: bool = True,
) -> TokenCoPilot:
    """Create a TokenCoPilot configured for production use.

    Includes:
    - Cost tracking and budget enforcement
    - Real-time streaming (webhook/Slack)
    - Anomaly detection with alerts
    - Budget forecasting

    Args:
        budget_limit: Budget limit in USD (required)
        webhook_url: Optional webhook URL for cost event streaming
        slack_webhook: Optional Slack webhook URL for alerts
        detect_anomalies: Enable anomaly detection (default: True)
        enable_forecasting: Enable budget forecasting (default: True)

    Returns:
        TokenCoPilot instance

    Example:
        >>> from token_copilot.presets import production
        >>>
        >>> copilot = production(
        ...     budget_limit=1000.00,
        ...     webhook_url="https://monitoring.example.com/webhook",
        ...     slack_webhook="https://hooks.slack.com/...",
        ... )
        >>> llm = ChatOpenAI(callbacks=[copilot])
    """
    copilot = TokenCoPilot(
        budget_limit=budget_limit,
        on_budget_exceeded="raise"  # Strict enforcement in production
    )

    # Add streaming if configured
    if webhook_url or slack_webhook:
        kwargs = {}
        if webhook_url:
            kwargs['webhook_url'] = webhook_url
        copilot.with_streaming(**kwargs)

    # Add analytics with alerts
    if detect_anomalies:
        alert_handlers = []

        # Add log alert (always)
        try:
            from .analytics import log_alert
            alert_handlers.append(log_alert)
        except ImportError:
            pass

        # Add Slack alert if configured
        if slack_webhook:
            try:
                from .analytics import slack_alert
                # Create Slack alert with webhook
                def slack_alert_handler(anomaly):
                    slack_alert(anomaly, webhook_url=slack_webhook)
                alert_handlers.append(slack_alert_handler)
            except ImportError:
                import logging
                logging.warning("Slack alerts require additional dependencies")

        if alert_handlers:
            copilot.with_analytics(
                detect_anomalies=True,
                alert_handlers=alert_handlers
            )

    # Add forecasting
    if enable_forecasting:
        copilot.with_forecasting(forecast_hours=24)

    return copilot


def enterprise(
    budget_limit: float,
    kafka_brokers: Optional[List[str]] = None,
    kafka_topic: str = "llm_costs",
    otlp_endpoint: Optional[str] = None,
    enable_all: bool = True,
) -> TokenCoPilot:
    """Create a TokenCoPilot configured for enterprise deployment.

    Includes ALL features:
    - Cost tracking and budget enforcement
    - Kafka/OTLP streaming for observability
    - Advanced analytics and anomaly detection
    - Model routing for cost optimization
    - Adaptive operations for budget awareness
    - Budget forecasting and alerts

    Args:
        budget_limit: Budget limit in USD (required)
        kafka_brokers: Optional list of Kafka broker addresses
        kafka_topic: Kafka topic name (default: "llm_costs")
        otlp_endpoint: Optional OpenTelemetry OTLP endpoint
        enable_all: Enable all features (default: True)

    Returns:
        TokenCoPilot instance

    Example:
        >>> from token_copilot.presets import enterprise
        >>>
        >>> copilot = enterprise(
        ...     budget_limit=10000.00,
        ...     kafka_brokers=["kafka1:9092", "kafka2:9092"],
        ...     otlp_endpoint="http://collector:4318",
        ... )
        >>> llm = ChatOpenAI(callbacks=[copilot])
    """
    copilot = TokenCoPilot(
        budget_limit=budget_limit,
        budget_period="daily",  # Daily budget for enterprise
        on_budget_exceeded="raise"
    )

    if not enable_all:
        return copilot

    # Streaming to enterprise observability platforms
    streaming_kwargs = {}
    if kafka_brokers:
        streaming_kwargs['kafka_brokers'] = kafka_brokers
        streaming_kwargs['kafka_topic'] = kafka_topic
    if otlp_endpoint:
        streaming_kwargs['otlp_endpoint'] = otlp_endpoint

    if streaming_kwargs:
        copilot.with_streaming(**streaming_kwargs)

    # Analytics with all features
    try:
        from .analytics import log_alert
        copilot.with_analytics(
            detect_anomalies=True,
            track_waste=True,
            track_efficiency=True,
            alert_handlers=[log_alert]
        )
    except ImportError:
        import logging
        logging.warning("Analytics features require: pip install token-copilot[analytics]")

    # Adaptive operations for budget-aware behavior
    copilot.with_adaptive()

    # Forecasting for proactive budget management
    copilot.with_forecasting(forecast_hours=48)

    return copilot


# Convenience alias
def quick(budget_limit: Optional[float] = None) -> TokenCoPilot:
    """Alias for basic() preset.

    Args:
        budget_limit: Optional budget limit in USD

    Returns:
        TokenCoPilot instance
    """
    return basic(budget_limit)
