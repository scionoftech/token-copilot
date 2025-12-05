"""LlamaIndex callback for automatic cost tracking and budget enforcement."""

from typing import Any, Dict, List, Optional, Callable

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType
    EventType = CBEventType  # Alias for backward compatibility
except ImportError:
    try:
        # Fallback for older LlamaIndex versions
        from llama_index.callbacks.base_handler import BaseCallbackHandler
        from llama_index.callbacks.schema import CBEventType
        EventType = CBEventType
    except ImportError:
        try:
            # Even older versions
            from llama_index.callbacks.base import BaseCallbackHandler
            from llama_index.callbacks.schema import CBEventType
            EventType = CBEventType
        except ImportError:
            BaseCallbackHandler = object
            CBEventType = None
            EventType = None

from ..tracking import MultiTenantTracker, BudgetEnforcer

# Analytics imports (lazy loaded)
try:
    from ..analytics import WasteAnalyzer, EfficiencyScorer, AnomalyDetector, Anomaly
except ImportError:
    WasteAnalyzer = None
    EfficiencyScorer = None
    AnomalyDetector = None
    Anomaly = None

# Routing imports (lazy loaded)
try:
    from ..routing import ModelRouter, RoutingStrategy, RoutingDecision, ModelConfig as RouterModelConfig
except ImportError:
    ModelRouter = None
    RoutingStrategy = None
    RoutingDecision = None
    RouterModelConfig = None

# Forecasting imports (lazy loaded)
try:
    from ..forecasting import BudgetPredictor, BudgetForecast, AlertManager
except ImportError:
    BudgetPredictor = None
    BudgetForecast = None
    AlertManager = None

# Queuing imports (lazy loaded)
try:
    from ..queuing import QueueManager, QueueMode, Priority
except ImportError:
    QueueManager = None
    QueueMode = None
    Priority = None


class TokenCoPilotCallbackHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler for automatic cost tracking and budget enforcement.

    Primary interface for token_copilot with LlamaIndex. Automatically tracks costs, enforces
    budgets, and provides analytics for LlamaIndex applications.

    Features:
        - Automatic cost tracking per LLM call
        - Multi-tenant support (track by user_id, org_id, etc.)
        - Budget enforcement (hard stop at limit)
        - Pandas DataFrame export for analytics
        - Token waste analysis
        - Efficiency scoring with leaderboards
        - Anomaly detection with alerts
        - Cross-model routing
        - Predictive budget alerts
        - Smart request queuing

    Example:
        >>> from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        >>> from llama_index.core.callbacks import CallbackManager
        >>> from token_copilot.llamaindex import TokenCoPilotCallbackHandler
        >>>
        >>> # Create callback with budget limit
        >>> callback = TokenCoPilotCallbackHandler(budget_limit=10.00)
        >>> callback_manager = CallbackManager([callback])
        >>>
        >>> # Use with LlamaIndex
        >>> documents = SimpleDirectoryReader("data").load_data()
        >>> index = VectorStoreIndex.from_documents(
        ...     documents,
        ...     callback_manager=callback_manager
        ... )
        >>>
        >>> # Query with tracking
        >>> query_engine = index.as_query_engine()
        >>> response = query_engine.query("What is this about?")
        >>>
        >>> # Get analytics
        >>> print(f"Total cost: ${callback.get_total_cost():.4f}")
        >>> df = callback.to_dataframe()
        >>> print(df.groupby('user_id')['cost'].sum())
    """

    def __init__(
        self,
        # Core params
        budget_limit: Optional[float] = None,
        budget_period: str = "total",
        on_budget_exceeded: str = "raise",

        # Streaming params
        streamer: Optional[Any] = None,  # BaseStreamer

        # Analytics params
        anomaly_detection: bool = False,
        anomaly_sensitivity: float = 3.0,
        alert_handlers: Optional[List[Callable]] = None,

        # Routing params
        auto_routing: bool = False,
        routing_models: Optional[List] = None,  # List[RouterModelConfig]
        routing_strategy: Optional[str] = None,  # RoutingStrategy

        # Forecasting params
        predictive_alerts: bool = False,
        forecast_window_hours: int = 24,

        # Queuing params
        queue_mode: Optional[str] = None,  # QueueMode
        max_queue_size: int = 1000,

        # LlamaIndex specific
        event_starts_to_ignore: Optional[List] = None,
        event_ends_to_ignore: Optional[List] = None,
    ):
        """
        Initialize TokenCoPilotCallbackHandler.

        Args:
            budget_limit: Budget limit in USD (None = no limit)
            budget_period: Budget period (total, daily, monthly, per_user, per_org)
            on_budget_exceeded: Behavior when exceeded (raise, warn, ignore)

            streamer: Optional BaseStreamer for real-time cost event streaming
                (WebhookStreamer, SyslogStreamer, LogstashStreamer, etc.)

            anomaly_detection: Enable anomaly detection
            anomaly_sensitivity: Standard deviation threshold (default 3.0)
            alert_handlers: List of alert handler callables

            auto_routing: Enable automatic model routing
            routing_models: List of ModelConfig objects for routing
            routing_strategy: Routing strategy

            predictive_alerts: Enable budget forecasting alerts
            forecast_window_hours: Hours of history for forecasting

            queue_mode: Queue mode (DISABLED, SOFT, HARD, SMART)
            max_queue_size: Maximum queue size

            event_starts_to_ignore: List of event types to ignore on start
            event_ends_to_ignore: List of event types to ignore on end

        Example:
            >>> from token_copilot.llamaindex import TokenCoPilotCallbackHandler
            >>> from token_copilot.analytics import log_alert
            >>> callback = TokenCoPilotCallbackHandler(
            ...     budget_limit=100.00,
            ...     anomaly_detection=True,
            ...     alert_handlers=[log_alert]
            ... )
        """
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

        # Initialize tracker and budget enforcer
        self.tracker = MultiTenantTracker()
        self.budget_enforcer = BudgetEnforcer(
            limit=budget_limit,
            period=budget_period,
            on_exceeded=on_budget_exceeded,
        )
        # Keep backward compatibility
        self.budget = self.budget_enforcer

        # Store streaming component
        self.streamer = streamer

        # Store event data
        self._event_data: Dict[str, Dict] = {}

        # Analytics components (lazy init)
        self.waste_analyzer = None
        self.efficiency_scorer = None
        self.anomaly_detector = None

        if anomaly_detection and AnomalyDetector is not None:
            self.anomaly_detector = AnomalyDetector(
                threshold=anomaly_sensitivity,
                alert_handlers=alert_handlers or [],
            )

        # Routing components (lazy init)
        self.router = None

        if auto_routing and ModelRouter is not None and routing_models:
            # Convert routing_strategy string to enum if needed
            if routing_strategy and isinstance(routing_strategy, str):
                routing_strategy = RoutingStrategy(routing_strategy)
            elif routing_strategy is None:
                routing_strategy = RoutingStrategy.BALANCED if RoutingStrategy else None

            self.router = ModelRouter(
                models=routing_models,
                strategy=routing_strategy,
            )

        # Forecasting components (lazy init)
        self.predictor = None
        self.alert_manager = None

        if predictive_alerts and BudgetPredictor is not None:
            self.predictor = BudgetPredictor(
                window_hours=forecast_window_hours,
            )
            self.alert_manager = AlertManager()

        # Queuing components (lazy init)
        self.queue_manager = None

        if queue_mode and queue_mode != "disabled" and QueueManager is not None:
            # Convert queue_mode string to enum if needed
            if isinstance(queue_mode, str):
                queue_mode = QueueMode(queue_mode)

            self.queue_manager = QueueManager(
                mode=queue_mode,
                max_size=max_queue_size,
            )

    def on_event_start(
        self,
        event_type: "CBEventType",
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Called when an event starts.

        Args:
            event_type: Type of event (LLM, EMBEDDING, etc.)
            payload: Event payload data
            event_id: Unique event identifier
            parent_id: Parent event identifier
            **kwargs: Additional arguments

        Returns:
            Event ID
        """
        # Check budget before LLM call
        if EventType and event_type == EventType.LLM:
            # Extract metadata if available
            metadata = {}
            if payload:
                metadata = payload.get("metadata", {})

            self.budget_enforcer.check(metadata)

            # Store event data for later use
            self._event_data[event_id] = {
                "metadata": metadata,
                "payload": payload,
            }

        return event_id

    def on_event_end(
        self,
        event_type: "CBEventType",
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Called when an event ends.

        Tracks cost and updates budget for LLM events.

        Args:
            event_type: Type of event
            payload: Event payload data
            event_id: Event identifier
            **kwargs: Additional arguments
        """
        # Only track LLM events
        if not EventType or event_type != EventType.LLM:
            return

        if not payload:
            return

        # Extract token usage from payload
        response = payload.get("response")
        if not response:
            return

        # Get token counts (structure varies by LlamaIndex version)
        input_tokens = 0
        output_tokens = 0
        model = "unknown"

        # Try to extract from raw response
        if hasattr(response, "raw"):
            raw = response.raw
            if hasattr(raw, "usage"):
                usage = raw.usage
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
            if hasattr(raw, "model"):
                model = raw.model

        # Fallback: try additional_kwargs
        if hasattr(response, "additional_kwargs"):
            additional = response.additional_kwargs
            if "usage" in additional:
                usage = additional["usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            if "model" in additional:
                model = additional["model"]

        # Get metadata from event start
        event_data = self._event_data.get(event_id, {})
        metadata = event_data.get("metadata", {})

        if input_tokens == 0 and output_tokens == 0:
            # No token data available, clean up and return
            if event_id in self._event_data:
                del self._event_data[event_id]
            return

        total_tokens = input_tokens + output_tokens

        # Track cost with metadata
        entry = self.tracker.track(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata,
        )

        # Update budget
        self.budget_enforcer.add(entry.cost, metadata)

        # Stream event to external system if configured
        if self.streamer:
            try:
                from ..streaming.base import StreamEvent
                stream_event = StreamEvent(
                    timestamp=entry.timestamp,
                    event_type="llm_cost",
                    model=entry.model,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    total_tokens=entry.input_tokens + entry.output_tokens,
                    cost=entry.cost,
                    user_id=entry.user_id,
                    org_id=entry.org_id,
                    session_id=entry.session_id,
                    feature=entry.feature,
                    endpoint=entry.endpoint,
                    environment=entry.environment,
                    metadata=entry.tags,
                )
                self.streamer.send_if_enabled(stream_event)
            except Exception as e:
                # Don't fail tracking if streaming fails
                import logging
                logging.warning(f"Failed to stream event: {e}")

        # Check for anomalies
        if self.anomaly_detector:
            anomaly = self.anomaly_detector.check(
                cost=entry.cost,
                tokens=total_tokens,
                model=model,
                metadata=metadata,
            )
            # Alerts are triggered automatically by detector

        # Record for prediction
        if self.predictor:
            self.predictor.record(entry.cost)

            # Check alerts if manager exists
            if self.alert_manager:
                forecast = self.get_forecast()
                self.alert_manager.check_alerts(forecast)

        # Clean up event data
        if event_id in self._event_data:
            del self._event_data[event_id]

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Called when a trace starts."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Called when a trace ends."""
        pass

    # Analytics methods (same as LangChain version)

    def get_total_cost(self) -> float:
        """Get total cost across all tracked calls."""
        return self.tracker.get_total_cost()

    def get_total_tokens(self) -> int:
        """Get total tokens across all tracked calls."""
        return self.tracker.get_total_tokens()

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.tracker.get_stats()

    def get_remaining_budget(self, metadata: Optional[Dict] = None) -> float:
        """Get remaining budget."""
        return self.budget_enforcer.get_remaining(metadata)

    def to_dataframe(self):
        """Export tracking data to pandas DataFrame."""
        return self.tracker.to_dataframe()

    def get_costs_by_user(self) -> Dict[str, float]:
        """Get costs grouped by user."""
        return self.tracker.get_costs_by("user_id")

    def get_costs_by_org(self) -> Dict[str, float]:
        """Get costs grouped by organization."""
        return self.tracker.get_costs_by("org_id")

    def get_costs_by_model(self) -> Dict[str, float]:
        """Get costs grouped by model."""
        return self.tracker.get_costs_by("model")

    def reset(self):
        """Reset all tracking and budget data."""
        self.tracker.clear()
        self.budget_enforcer.reset()

    # Analytics Methods

    def analyze_waste(self) -> Dict[str, Any]:
        """Run waste analysis on tracked data."""
        if WasteAnalyzer is None:
            raise ImportError("Analytics module not available")

        if not self.waste_analyzer:
            self.waste_analyzer = WasteAnalyzer()

        df = self.to_dataframe()
        return self.waste_analyzer.get_report(df)

    def get_efficiency_score(
        self,
        entity_type: str = 'user_id',
        entity_id: Optional[str] = None,
    ):
        """Get efficiency score(s)."""
        if EfficiencyScorer is None:
            raise ImportError("Analytics module not available")

        if not self.efficiency_scorer:
            self.efficiency_scorer = EfficiencyScorer()

        df = self.to_dataframe()

        if entity_id:
            return self.efficiency_scorer.score_entity(df, entity_type, entity_id)
        else:
            return self.efficiency_scorer.score_all(df, entity_type)

    def get_leaderboard(
        self,
        entity_type: str = 'user_id',
        top_n: int = 10,
    ) -> List[Dict]:
        """Get efficiency leaderboard."""
        if EfficiencyScorer is None:
            raise ImportError("Analytics module not available")

        if not self.efficiency_scorer:
            self.efficiency_scorer = EfficiencyScorer()

        df = self.to_dataframe()
        return self.efficiency_scorer.get_leaderboard(df, entity_type, top_n)

    def get_anomalies(
        self,
        minutes: int = 60,
        min_severity: Optional[str] = None,
    ) -> List:
        """Get recent anomalies."""
        if not self.anomaly_detector:
            return []

        return self.anomaly_detector.get_recent_anomalies(minutes, min_severity)

    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly statistics."""
        if not self.anomaly_detector:
            return {}

        return self.anomaly_detector.get_statistics()

    # Routing Methods

    def suggest_model(
        self,
        prompt: str,
        estimated_tokens: int = 1000,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Get model recommendation."""
        if not self.router:
            raise ValueError("Auto-routing not enabled. Set auto_routing=True in __init__")

        return self.router.route(prompt, estimated_tokens, metadata)

    def record_model_quality(self, model: str, quality_score: float):
        """Record quality score for a model."""
        if not self.router:
            raise ValueError("Auto-routing not enabled")

        self.router.record_quality(model, quality_score)

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get routing statistics per model."""
        if not self.router:
            return {}

        return self.router.get_model_stats()

    # Forecasting Methods

    def get_forecast(self, forecast_hours: int = 24):
        """Get budget forecast."""
        if not self.predictor:
            raise ValueError("Predictive alerts not enabled. Set predictive_alerts=True in __init__")

        budget_limit = self.budget_enforcer.limit if self.budget_enforcer.limit else float('inf')
        current_cost = self.get_total_cost()

        return self.predictor.forecast(
            budget_limit=budget_limit,
            current_cost=current_cost,
            forecast_hours=forecast_hours,
        )

    # Queuing Methods

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        if not self.queue_manager:
            return {}

        return self.queue_manager.get_stats()
