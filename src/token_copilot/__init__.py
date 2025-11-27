"""
token_copilot - Your AI copilot for LLM costs.

Multi-tenant cost tracking for LangChain, LangGraph, and LlamaIndex applications.
A comprehensive library for tracking, analyzing, and optimizing LLM costs in production.

Features:
- **Framework Support**: Works with LangChain, LangGraph, and LlamaIndex
- Automatic cost tracking via callbacks
- Multi-tenant support (track by user, organization, session, etc.)
- Budget enforcement with hard stops
- Pandas DataFrame export for analytics
- Token waste analysis (detect repeated prompts, excessive context, verbose outputs)
- Efficiency scoring with leaderboards
- Anomaly detection with alerts (cost/token/frequency spikes)
- Customizable alert handlers (log, webhook, Slack)
- Cross-model routing (route to cheapest suitable model)
- Predictive budget alerts (forecast exhaustion)
- Smart request queuing with priorities
- Multiple routing strategies (CHEAPEST_FIRST, QUALITY_FIRST, BALANCED, etc.)
- Zero configuration required

Quick Start (LangChain):
    ```python
    from langchain import ChatOpenAI
    from token_copilot import TokenPilotCallback

    callback = TokenPilotCallback(budget_limit=100.00)
    llm = ChatOpenAI(callbacks=[callback])

    # Get analytics
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    ```

Quick Start (LangGraph):
    ```python
    from langgraph.graph import StateGraph
    from langchain_openai import ChatOpenAI
    from token_copilot import TokenPilotCallback

    callback = TokenPilotCallback(budget_limit=100.00)

    # Create graph with callback
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    graph = builder.compile()

    # Run with cost tracking
    result = graph.invoke(
        {"messages": [("user", "Hello")]},
        config={"callbacks": [callback]}
    )

    # Get analytics
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    ```

Quick Start (LlamaIndex):
    ```python
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.callbacks import CallbackManager
    from token_copilot import TokenPilotCallbackHandler

    callback = TokenPilotCallbackHandler(budget_limit=100.00)
    callback_manager = CallbackManager([callback])

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager
    )

    # Get analytics
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    ```

Advanced Features:
    ```python
    from token_copilot.analytics import log_alert
    from token_copilot.routing import ModelConfig, RoutingStrategy
    from token_copilot.queuing import QueueMode

    # Full optimization (works with both frameworks)
    models = [
        ModelConfig("gpt-4o-mini", 0.7, 0.15, 0.60, 128000),
        ModelConfig("gpt-4o", 0.9, 5.0, 15.0, 128000),
    ]
    callback = TokenPilotCallback(  # or TokenPilotCallbackHandler
        budget_limit=100.00,
        anomaly_detection=True,
        alert_handlers=[log_alert],
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED,
        predictive_alerts=True,
        queue_mode=QueueMode.SMART
    )

    # Analytics
    waste_report = callback.analyze_waste()
    efficiency = callback.get_efficiency_score('user_id', 'user_123')
    forecast = callback.get_forecast()
    ```

Learn more: https://github.com/scionoftech/token-copilot
"""

__version__ = "1.0.0"
__author__ = "scionoftech"

# Core tracking (always available)
from .tracking import MultiTenantTracker, CostEntry, BudgetEnforcer

# Primary API - LangChain integration (optional)
try:
    from .langchain import TokenPilotCallback
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    TokenPilotCallback = None  # type: ignore

# LlamaIndex integration (optional)
try:
    from .llamaindex import TokenPilotCallbackHandler
    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False
    TokenPilotCallbackHandler = None  # type: ignore

# Utilities
from .utils import (
    # Exceptions
    TokenPilotError,
    BudgetExceededError,
    TrackingError,
    AlertError,
    # Pricing
    ModelConfig,
    MODEL_PRICING,
    MODEL_ALIASES,
    get_model_config,
    calculate_cost,
    list_models,
    list_providers,
)

# Analytics module (optional)
try:
    from .analytics import (
        WasteAnalyzer,
        WasteCategory,
        EfficiencyScorer,
        EfficiencyMetrics,
        AnomalyDetector,
        Anomaly,
        Severity,
        log_alert,
        webhook_alert,
        slack_alert,
    )
    _ANALYTICS_AVAILABLE = True
except ImportError:
    _ANALYTICS_AVAILABLE = False
    WasteAnalyzer = None  # type: ignore
    WasteCategory = None  # type: ignore
    EfficiencyScorer = None  # type: ignore
    EfficiencyMetrics = None  # type: ignore
    AnomalyDetector = None  # type: ignore
    Anomaly = None  # type: ignore
    Severity = None  # type: ignore
    log_alert = None  # type: ignore
    webhook_alert = None  # type: ignore
    slack_alert = None  # type: ignore

# Routing module (optional)
try:
    from .routing import (
        ModelRouter,
        RoutingStrategy,
        RoutingDecision,
        ModelConfig as RouterModelConfig,
    )
    _ROUTING_AVAILABLE = True
except ImportError:
    _ROUTING_AVAILABLE = False
    ModelRouter = None  # type: ignore
    RoutingStrategy = None  # type: ignore
    RoutingDecision = None  # type: ignore
    RouterModelConfig = None  # type: ignore

# Forecasting module (optional)
try:
    from .forecasting import (
        BudgetPredictor,
        BudgetForecast,
        Trend,
        AlertManager,
        AlertRule,
    )
    _FORECASTING_AVAILABLE = True
except ImportError:
    _FORECASTING_AVAILABLE = False
    BudgetPredictor = None  # type: ignore
    BudgetForecast = None  # type: ignore
    Trend = None  # type: ignore
    AlertManager = None  # type: ignore
    AlertRule = None  # type: ignore

# Queuing module (optional)
try:
    from .queuing import (
        QueueManager,
        QueueMode,
        Priority,
        QueuedRequest,
    )
    _QUEUING_AVAILABLE = True
except ImportError:
    _QUEUING_AVAILABLE = False
    QueueManager = None  # type: ignore
    QueueMode = None  # type: ignore
    Priority = None  # type: ignore
    QueuedRequest = None  # type: ignore

__all__ = [
    # Primary API
    "TokenPilotCallback",  # LangChain
    "TokenPilotCallbackHandler",  # LlamaIndex
    # Core (advanced)
    "MultiTenantTracker",
    "CostEntry",
    "BudgetEnforcer",
    # Exceptions
    "TokenPilotError",
    "BudgetExceededError",
    "TrackingError",
    "AlertError",
    # Pricing utilities
    "ModelConfig",
    "MODEL_PRICING",
    "MODEL_ALIASES",
    "get_model_config",
    "calculate_cost",
    "list_models",
    "list_providers",
    # Analytics
    "WasteAnalyzer",
    "WasteCategory",
    "EfficiencyScorer",
    "EfficiencyMetrics",
    "AnomalyDetector",
    "Anomaly",
    "Severity",
    "log_alert",
    "webhook_alert",
    "slack_alert",
    # Routing
    "ModelRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "RouterModelConfig",
    # Forecasting
    "BudgetPredictor",
    "BudgetForecast",
    "Trend",
    "AlertManager",
    "AlertRule",
    # Queuing
    "QueueManager",
    "QueueMode",
    "Priority",
    "QueuedRequest",
]
