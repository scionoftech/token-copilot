"""OpenTelemetry streaming for OTLP protocol."""

import logging
from typing import Optional, Dict, Any

from .base import BaseStreamer, StreamEvent

logger = logging.getLogger(__name__)


class OpenTelemetryStreamer(BaseStreamer):
    """
    Stream cost events via OpenTelemetry (OTLP).

    Exports cost data as OpenTelemetry spans with attributes,
    enabling integration with modern observability platforms like
    Jaeger, Zipkin, Honeycomb, Datadog, etc.

    Requires: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

    Example:
        ```python
        from token_copilot import TokenPilotCallback
        from token_copilot.streaming import OpenTelemetryStreamer

        # Basic OTLP streaming
        streamer = OpenTelemetryStreamer(
            endpoint="http://otel-collector:4318",
            service_name="llm-service"
        )

        # With gRPC endpoint
        streamer = OpenTelemetryStreamer(
            endpoint="otel-collector:4317",
            protocol="grpc",
            service_name="llm-service",
            insecure=True
        )

        # With authentication
        streamer = OpenTelemetryStreamer(
            endpoint="https://api.honeycomb.io",
            service_name="llm-service",
            headers={
                "x-honeycomb-team": "your-api-key",
                "x-honeycomb-dataset": "llm-costs"
            }
        )

        callback = TokenPilotCallback(streamer=streamer)
        ```
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        protocol: str = "http",
        service_name: str = "token-copilot",
        service_version: Optional[str] = None,
        deployment_environment: Optional[str] = None,
        insecure: bool = False,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        compression: Optional[str] = None,
        export_as: str = "span",  # "span" or "log"
        enabled: bool = True,
    ):
        """
        Initialize OpenTelemetry streamer.

        Args:
            endpoint: OTLP endpoint URL
                HTTP: http://host:4318 (default)
                gRPC: host:4317
            protocol: "http" or "grpc" (default: "http")
            service_name: Service name for telemetry (default: "token-copilot")
            service_version: Service version (optional)
            deployment_environment: Deployment environment (e.g., "prod", "staging")
            insecure: Allow insecure connection (default: False)
            headers: Additional HTTP headers (e.g., for auth)
            timeout: Request timeout in seconds (default: 10)
            compression: Compression type: None, "gzip" (default: None)
            export_as: Export as "span" or "log" (default: "span")
            enabled: Whether streaming is enabled (default: True)
        """
        super().__init__(enabled=enabled)
        self.endpoint = endpoint
        self.protocol = protocol
        self.service_name = service_name
        self.service_version = service_version
        self.deployment_environment = deployment_environment
        self.insecure = insecure
        self.headers = headers or {}
        self.timeout = timeout
        self.compression = compression
        self.export_as = export_as

        # Initialize OpenTelemetry
        self._tracer = None
        self._logger = None
        self._init_opentelemetry()

    def _init_opentelemetry(self):
        """Initialize OpenTelemetry components."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCSpanExporter

            # Build resource attributes
            resource_attrs = {SERVICE_NAME: self.service_name}
            if self.service_version:
                resource_attrs[SERVICE_VERSION] = self.service_version
            if self.deployment_environment:
                resource_attrs[DEPLOYMENT_ENVIRONMENT] = self.deployment_environment

            resource = Resource.create(resource_attrs)

            # Create trace provider
            provider = TracerProvider(resource=resource)

            # Create exporter
            exporter_kwargs = {
                'endpoint': self.endpoint if self.protocol == "http" else f"http://{self.endpoint}",
                'timeout': self.timeout,
            }

            if self.headers:
                exporter_kwargs['headers'] = self.headers
            if self.compression:
                exporter_kwargs['compression'] = self.compression

            if self.protocol == "grpc":
                exporter = GRPCSpanExporter(**exporter_kwargs)
            else:
                exporter = HTTPSpanExporter(**exporter_kwargs)

            # Add span processor
            provider.add_span_processor(BatchSpanProcessor(exporter))

            # Get tracer
            self._tracer = provider.get_tracer(
                instrumenting_module_name="token_copilot",
                instrumenting_library_version="1.0.0"
            )

            logger.debug(f"OpenTelemetry initialized with {self.protocol} exporter")

        except ImportError:
            logger.warning(
                "OpenTelemetry libraries required for OTLP streaming. "
                "Install with: pip install token-copilot[streaming] "
                "or pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")

    def _create_span_attributes(self, event: StreamEvent) -> Dict[str, Any]:
        """Create span attributes from event."""
        attrs = {
            'llm.event_type': event.event_type,
            'llm.timestamp': event.timestamp.isoformat(),
        }

        if event.model:
            attrs['llm.model'] = event.model
        if event.input_tokens is not None:
            attrs['llm.input_tokens'] = event.input_tokens
        if event.output_tokens is not None:
            attrs['llm.output_tokens'] = event.output_tokens
        if event.total_tokens is not None:
            attrs['llm.total_tokens'] = event.total_tokens
        if event.cost is not None:
            attrs['llm.cost'] = event.cost
        if event.user_id:
            attrs['llm.user_id'] = event.user_id
        if event.org_id:
            attrs['llm.org_id'] = event.org_id
        if event.session_id:
            attrs['llm.session_id'] = event.session_id
        if event.feature:
            attrs['llm.feature'] = event.feature
        if event.endpoint:
            attrs['llm.endpoint'] = event.endpoint
        if event.environment:
            attrs['llm.environment'] = event.environment

        # Add metadata
        if event.metadata:
            for key, value in event.metadata.items():
                attrs[f'llm.metadata.{key}'] = str(value)

        return attrs

    def send(self, event: StreamEvent) -> bool:
        """
        Send event via OpenTelemetry.

        Args:
            event: StreamEvent to send

        Returns:
            True if sent successfully, False on error
        """
        if not self._tracer:
            return False

        try:
            # Create span
            span_name = f"LLM Cost: {event.model or 'unknown'}"
            with self._tracer.start_as_current_span(span_name) as span:
                # Add attributes
                attrs = self._create_span_attributes(event)
                for key, value in attrs.items():
                    span.set_attribute(key, value)

                # Add event
                span.add_event(
                    name="llm_cost_event",
                    attributes={
                        'cost': event.cost or 0.0,
                        'model': event.model or 'unknown',
                    }
                )

            logger.debug(f"Sent span to OpenTelemetry: {span_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send to OpenTelemetry: {e}")
            return False

    def close(self):
        """Close OpenTelemetry exporter."""
        # Spans are flushed automatically by BatchSpanProcessor
        pass
