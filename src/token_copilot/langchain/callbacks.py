"""LangChain callback for automatic cost tracking and budget enforcement."""

from typing import Any, Dict, List, Optional, Callable

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    # Fallback for older LangChain versions
    from langchain.callbacks.base import BaseCallbackHandler

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


class TokenCoPilotCallback(BaseCallbackHandler):
    """
    LangChain/LangGraph callback for automatic cost tracking and budget enforcement.

    Primary interface for token_copilot. Automatically tracks costs, enforces
    budgets, and provides analytics for LangChain and LangGraph applications.

    Features:
        - Works with LangChain chains and LangGraph workflows
        - Automatic cost tracking per LLM call
        - Multi-tenant support (track by user_id, org_id, etc.)
        - Budget enforcement (hard stop at limit)
        - Pandas DataFrame export for analytics
        - Zero configuration required

    Example (LangChain):
        >>> from langchain import ChatOpenAI, LLMChain
        >>> from token_copilot.langchain import TokenCoPilotCallback
        >>>
        >>> # Create callback with budget limit
        >>> callback = TokenCoPilotCallback(budget_limit=10.00)
        >>>
        >>> # Use with any LangChain LLM/chain
        >>> llm = ChatOpenAI(callbacks=[callback])
        >>> chain = LLMChain(llm=llm, prompt=prompt)
        >>>
        >>> # Track per user
        >>> result = chain.run(
        ...     "question",
        ...     metadata={"user_id": "user_123", "org_id": "org_456"}
        ... )
        >>>
        >>> # Get analytics
        >>> print(f"Total cost: ${callback.get_total_cost():.4f}")
        >>> print(f"Remaining budget: ${callback.get_remaining_budget():.2f}")
        >>>
        >>> # Export to pandas
        >>> df = callback.to_dataframe()
        >>> print(df.groupby('user_id')['cost'].sum())

    Example (LangGraph):
        >>> from langgraph.graph import StateGraph, START, END
        >>> from langchain_openai import ChatOpenAI
        >>> from token_copilot import TokenCoPilotCallback
        >>>
        >>> callback = TokenCoPilotCallback(budget_limit=10.00)
        >>>
        >>> # Create graph
        >>> builder = StateGraph(State)
        >>> builder.add_node("agent", agent_node)
        >>> builder.add_edge(START, "agent")
        >>> graph = builder.compile()
        >>>
        >>> # Run with cost tracking
        >>> result = graph.invoke(
        ...     {"messages": [("user", "Hello")]},
        ...     config={"callbacks": [callback]}
        ... )
        >>>
        >>> print(f"Total cost: ${callback.get_total_cost():.4f}")
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

        **kwargs
    ):
        """
        Initialize TokenCoPilotCallback.

        Args:
            budget_limit: Budget limit in USD (None = no limit)
            budget_period: Budget period:
                - "total": Total across all time (default)
                - "daily": Reset daily
                - "monthly": Reset monthly
                - "per_user": Separate budget per user_id
                - "per_org": Separate budget per org_id
            on_budget_exceeded: Behavior when budget exceeded:
                - "raise": Raise BudgetExceededError (default)
                - "warn": Log warning but continue
                - "ignore": No action

            streamer: Optional BaseStreamer for real-time cost event streaming
                (WebhookStreamer, SyslogStreamer, LogstashStreamer, etc.)

            anomaly_detection: Enable anomaly detection
            anomaly_sensitivity: Standard deviation threshold (default 3.0)
            alert_handlers: List of alert handler callables

            auto_routing: Enable automatic model routing
            routing_models: List of ModelConfig objects for routing
            routing_strategy: Routing strategy (CHEAPEST_FIRST, QUALITY_FIRST, BALANCED, etc.)

            predictive_alerts: Enable budget forecasting alerts
            forecast_window_hours: Hours of history for forecasting (default 24)

            queue_mode: Queue mode (DISABLED, SOFT, HARD, SMART)
            max_queue_size: Maximum queue size (default 1000)

            **kwargs: Additional callback arguments

        Example:
            >>> # Basic usage
            >>> callback = TokenCoPilotCallback(budget_limit=100.00)
            >>>
            >>> # With anomaly detection
            >>> from token_copilot.analytics import log_alert
            >>> callback = TokenCoPilotCallback(
            ...     budget_limit=100.00,
            ...     anomaly_detection=True,
            ...     anomaly_sensitivity=3.0,
            ...     alert_handlers=[log_alert]
            ... )
            >>>
            >>> # Full features
            >>> from token_copilot.routing import ModelConfig, RoutingStrategy
            >>> from token_copilot.queuing import QueueMode
            >>> models = [
            ...     ModelConfig("gpt-4o-mini", 0.7, 0.15, 0.60, 128000),
            ...     ModelConfig("gpt-4o", 0.9, 5.0, 15.0, 128000),
            ... ]
            >>> callback = TokenCoPilotCallback(
            ...     budget_limit=100.00,
            ...     auto_routing=True,
            ...     routing_models=models,
            ...     routing_strategy=RoutingStrategy.BALANCED,
            ...     predictive_alerts=True,
            ...     queue_mode=QueueMode.SMART
            ... )
        """
        super().__init__(**kwargs)

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

        # Store current metadata for tracking
        self._current_metadata: Dict[str, Any] = {}

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

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """
        Called when LLM starts.

        Extracts metadata and checks budget before making the call.

        Args:
            serialized: Serialized LLM
            prompts: Input prompts
            **kwargs: Additional arguments (may contain metadata)
        """
        # Extract metadata from kwargs
        self._current_metadata = kwargs.get("metadata", {})

        # Check budget before call
        self.budget_enforcer.check(self._current_metadata)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Called when LLM ends successfully.

        Tracks cost and updates budget. Integrates advanced features:
        - Anomaly detection
        - Budget forecasting
        - Alert management

        Args:
            response: LLM response
            **kwargs: Additional arguments
        """
        # Extract token usage from response
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            model = response.llm_output.get("model_name", "unknown")

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = input_tokens + output_tokens

            # Track cost with metadata
            entry = self.tracker.track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=self._current_metadata,
            )

            # Update budget
            self.budget_enforcer.add(entry.cost, self._current_metadata)

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
                    metadata=self._current_metadata,
                )
                # Alerts are triggered automatically by detector

            # Record for prediction
            if self.predictor:
                self.predictor.record(entry.cost)

                # Check alerts if manager exists
                if self.alert_manager:
                    forecast = self.get_forecast()
                    self.alert_manager.check_alerts(forecast)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """
        Called when LLM errors.

        Args:
            error: The error that occurred
            **kwargs: Additional arguments
        """
        # Don't track cost on errors
        pass

    # Analytics methods

    def get_total_cost(self) -> float:
        """
        Get total cost across all tracked calls.

        Returns:
            Total cost in USD

        Example:
            >>> cost = callback.get_total_cost()
            >>> print(f"Total spent: ${cost:.4f}")
        """
        return self.tracker.get_total_cost()

    def get_total_tokens(self) -> int:
        """
        Get total tokens across all tracked calls.

        Returns:
            Total token count

        Example:
            >>> tokens = callback.get_total_tokens()
            >>> print(f"Total tokens: {tokens:,}")
        """
        return self.tracker.get_total_tokens()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dict with total_cost, total_tokens, num_calls, averages, etc.

        Example:
            >>> stats = callback.get_stats()
            >>> print(f"Calls: {stats['total_calls']}")
            >>> print(f"Avg cost: ${stats['avg_cost_per_call']:.4f}")
        """
        return self.tracker.get_stats()

    def get_remaining_budget(self, metadata: Optional[Dict] = None) -> float:
        """
        Get remaining budget.

        Args:
            metadata: Metadata dict (for per-user/per-org budgets)

        Returns:
            Remaining budget in USD (inf if no limit)

        Example:
            >>> remaining = callback.get_remaining_budget()
            >>> print(f"${remaining:.2f} remaining")
            >>>
            >>> # Per-user remaining
            >>> user_remaining = callback.get_remaining_budget(
            ...     {"user_id": "user_123"}
            ... )
        """
        return self.budget.get_remaining(metadata)

    def to_dataframe(self):
        """
        Export tracking data to pandas DataFrame.

        Returns:
            pandas.DataFrame with columns:
                timestamp, model, input_tokens, output_tokens,
                total_tokens, cost, user_id, org_id, session_id,
                feature, endpoint, environment, custom tags

        Example:
            >>> df = callback.to_dataframe()
            >>>
            >>> # Group by user
            >>> print(df.groupby('user_id')['cost'].sum())
            >>>
            >>> # Group by organization
            >>> print(df.groupby('org_id')['cost'].sum())
            >>>
            >>> # Group by model
            >>> print(df.groupby('model')['cost'].sum())
            >>>
            >>> # Filter by feature
            >>> chat_df = df[df['feature'] == 'chat']
        """
        return self.tracker.to_dataframe()

    def get_costs_by_user(self) -> Dict[str, float]:
        """
        Get costs grouped by user.

        Returns:
            Dict mapping user_id to total cost

        Example:
            >>> costs = callback.get_costs_by_user()
            >>> for user_id, cost in costs.items():
            ...     print(f"{user_id}: ${cost:.2f}")
        """
        return self.tracker.get_costs_by("user_id")

    def get_costs_by_org(self) -> Dict[str, float]:
        """
        Get costs grouped by organization.

        Returns:
            Dict mapping org_id to total cost

        Example:
            >>> costs = callback.get_costs_by_org()
            >>> for org_id, cost in costs.items():
            ...     print(f"{org_id}: ${cost:.2f}")
        """
        return self.tracker.get_costs_by("org_id")

    def get_costs_by_model(self) -> Dict[str, float]:
        """
        Get costs grouped by model.

        Returns:
            Dict mapping model to total cost

        Example:
            >>> costs = callback.get_costs_by_model()
            >>> for model, cost in costs.items():
            ...     print(f"{model}: ${cost:.2f}")
        """
        return self.tracker.get_costs_by("model")

    def reset(self):
        """
        Reset all tracking and budget data.

        Example:
            >>> callback.reset()  # Start fresh
        """
        self.tracker.clear()
        self.budget_enforcer.reset()

    # Analytics Methods

    def analyze_waste(self) -> Dict[str, Any]:
        """
        Run waste analysis on tracked data.

        Returns:
            Dict with summary, categories, recommendations, potential_savings

        Example:
            >>> report = callback.analyze_waste()
            >>> print(f"Total waste: ${report['summary']['total_waste_cost']:.2f}")
            >>> for rec in report['recommendations']:
            ...     print(f"  - {rec}")
        """
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
        """
        Get efficiency score(s).

        Args:
            entity_type: Entity type ('user_id', 'org_id', 'model')
            entity_id: Specific entity ID (None = all entities)

        Returns:
            EfficiencyMetrics if entity_id provided, else Dict[str, EfficiencyMetrics]

        Example:
            >>> # Single user
            >>> score = callback.get_efficiency_score('user_id', 'user_123')
            >>> print(f"Overall: {score.overall_score:.2f}")
            >>>
            >>> # All users
            >>> all_scores = callback.get_efficiency_score('user_id')
            >>> for user_id, score in all_scores.items():
            ...     print(f"{user_id}: {score.overall_score:.2f}")
        """
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
        """
        Get efficiency leaderboard.

        Args:
            entity_type: Entity type ('user_id', 'org_id', 'model')
            top_n: Number of top performers to return

        Returns:
            List of dicts with rank, entity_id, and scores

        Example:
            >>> leaderboard = callback.get_leaderboard('user_id', top_n=5)
            >>> for entry in leaderboard:
            ...     print(f"{entry['rank']}. {entry['entity_id']}: {entry['overall_score']:.2f}")
        """
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
        """
        Get recent anomalies.

        Args:
            minutes: Look back this many minutes
            min_severity: Minimum severity ('low', 'medium', 'high', 'critical')

        Returns:
            List of Anomaly objects

        Example:
            >>> anomalies = callback.get_anomalies(minutes=60, min_severity='high')
            >>> for anomaly in anomalies:
            ...     print(f"{anomaly.severity}: {anomaly.message}")
        """
        if not self.anomaly_detector:
            return []

        return self.anomaly_detector.get_recent_anomalies(minutes, min_severity)

    def get_anomaly_stats(self) -> Dict[str, Any]:
        """
        Get anomaly statistics.

        Returns:
            Dict with total count, counts by type, counts by severity

        Example:
            >>> stats = callback.get_anomaly_stats()
            >>> print(f"Total: {stats['total']}")
            >>> print(f"Critical: {stats['by_severity'].get('critical', 0)}")
        """
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
        """
        Get model recommendation.

        Args:
            prompt: Input prompt text
            estimated_tokens: Estimated total tokens
            metadata: Optional metadata

        Returns:
            RoutingDecision with selected model and rationale

        Example:
            >>> decision = callback.suggest_model("What is Python?", 500)
            >>> print(f"Use {decision.selected_model}")
            >>> print(f"Cost: ${decision.estimated_cost:.4f}")
            >>> print(f"Reason: {decision.reason}")
        """
        if not self.router:
            raise ValueError("Auto-routing not enabled. Set auto_routing=True in __init__")

        return self.router.route(prompt, estimated_tokens, metadata)

    def record_model_quality(self, model: str, quality_score: float):
        """
        Record quality score for a model (for LEARNED routing strategy).

        Args:
            model: Model name
            quality_score: Quality score (0-1)

        Example:
            >>> callback.record_model_quality("gpt-4o", 0.95)
        """
        if not self.router:
            raise ValueError("Auto-routing not enabled")

        self.router.record_quality(model, quality_score)

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get routing statistics per model.

        Returns:
            Dict mapping model name to stats

        Example:
            >>> stats = callback.get_model_stats()
            >>> for model, data in stats.items():
            ...     print(f"{model}: {data['avg_quality']:.2f} quality")
        """
        if not self.router:
            return {}

        return self.router.get_model_stats()

    # Forecasting Methods

    def get_forecast(self, forecast_hours: int = 24):
        """
        Get budget forecast.

        Args:
            forecast_hours: Hours to forecast ahead

        Returns:
            BudgetForecast with predictions and recommendations

        Example:
            >>> forecast = callback.get_forecast()
            >>> print(f"Remaining: ${forecast.remaining_budget:.2f}")
            >>> if forecast.hours_until_exhausted:
            ...     print(f"Exhausts in {forecast.hours_until_exhausted:.1f}h")
            >>> for rec in forecast.recommendations:
            ...     print(f"  - {rec}")
        """
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
        """
        Get queue statistics.

        Returns:
            Dict with current_size, total_queued, total_processed, etc.

        Example:
            >>> stats = callback.get_queue_stats()
            >>> print(f"Queue size: {stats['current_size']}")
            >>> print(f"Avg wait: {stats['avg_wait_seconds']:.1f}s")
        """
        if not self.queue_manager:
            return {}

        return self.queue_manager.get_stats()
