"""
token_copilot - Your AI copilot for LLM costs.

A modern, plugin-based library for tracking, analyzing, and optimizing LLM costs
in production. Works seamlessly with LangChain, LangGraph, and LlamaIndex.

Quick Start (Minimal):
    ```python
    from token_copilot import TokenCoPilot
    from langchain_openai import ChatOpenAI

    copilot = TokenCoPilot(budget_limit=10.00)
    llm = ChatOpenAI(callbacks=[copilot])

    result = llm.invoke("Hello!")
    print(f"Cost: ${copilot.cost:.4f}")
    ```

Quick Start (Builder Pattern):
    ```python
    copilot = (TokenCoPilot(budget_limit=100.00)
        .with_streaming(webhook_url="https://example.com/webhook")
        .with_analytics(detect_anomalies=True)
        .with_adaptive()
        .build()
    )
    llm = ChatOpenAI(callbacks=[copilot])
    ```

Quick Start (Factory Preset):
    ```python
    from token_copilot.presets import production

    copilot = production(
        budget_limit=1000.00,
        webhook_url="https://monitoring.example.com",
    )
    llm = ChatOpenAI(callbacks=[copilot])
    ```

Quick Start (Context Manager):
    ```python
    from token_copilot import track_costs

    with track_costs(budget_limit=5.00) as copilot:
        llm = ChatOpenAI(callbacks=[copilot])
        result = llm.invoke("Hello!")
        print(f"Cost: ${copilot.cost:.4f}")
    ```

Quick Start (Decorator):
    ```python
    from token_copilot.decorators import track_cost

    @track_cost(budget_limit=5.00)
    def process_text(text):
        llm = ChatOpenAI(callbacks=[process_text.copilot])
        return llm.invoke(f"Process: {text}")

    result = process_text("my text")
    print(f"Cost: ${process_text.copilot.cost:.4f}")
    ```

Features:
- **Zero Config**: Start tracking with one line
- **Plugin-Based**: Add features as needed
- **Multi-Pattern**: Builder, factory, context managers, decorators
- **Multi-Tenant**: Track by user, organization, session
- **Budget Enforcement**: Hard stops at limits
- **Real-Time Streaming**: Webhook, Kafka, syslog, OTLP
- **Advanced Analytics**: Waste detection, anomaly alerts
- **Adaptive Operations**: Auto-adjust parameters by budget
- **Model Routing**: Intelligent model selection
- **Forecasting**: Predict budget exhaustion

Learn more: https://github.com/scionoftech/token-copilot
"""

__version__ = "1.0.2"
__author__ = "Sai Kumar Yava"

# Core
from .core import TokenCoPilot, Plugin

# Core tracking (always available)
from .tracking import MultiTenantTracker, CostEntry, BudgetEnforcer

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

# Plugins (lazy-loaded)
from . import plugins

# Factory presets
from . import presets
from .presets import basic, development, production, enterprise, quick

# Context managers
from .context import track_costs, with_budget, monitored

# Decorators
from . import decorators

# Analytics (optional)
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

# Routing (optional)
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

# Forecasting (optional)
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

# Adaptive operations (optional)
try:
    from .adaptive import (
        TokenAwareOperations,
        BudgetTier,
        classify_budget_tier,
        get_tier_description,
        token_aware,
        budget_gate,
        track_efficiency,
        adaptive_context,
        budget_aware_section,
        get_current_callback,
        set_current_callback,
    )
    _ADAPTIVE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_AVAILABLE = False
    TokenAwareOperations = None  # type: ignore
    BudgetTier = None  # type: ignore
    classify_budget_tier = None  # type: ignore
    get_tier_description = None  # type: ignore
    token_aware = None  # type: ignore
    budget_gate = None  # type: ignore
    track_efficiency = None  # type: ignore
    adaptive_context = None  # type: ignore
    budget_aware_section = None  # type: ignore
    get_current_callback = None  # type: ignore
    set_current_callback = None  # type: ignore

__all__ = [
    # Core
    "TokenCoPilot",
    "Plugin",
    # Core tracking (advanced)
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
    # Modules
    "plugins",
    "presets",
    "decorators",
    # Factory presets
    "basic",
    "development",
    "production",
    "enterprise",
    "quick",
    # Context managers
    "track_costs",
    "with_budget",
    "monitored",
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
    # Adaptive operations
    "TokenAwareOperations",
    "BudgetTier",
    "classify_budget_tier",
    "get_tier_description",
    "token_aware",
    "budget_gate",
    "track_efficiency",
    "adaptive_context",
    "budget_aware_section",
    "get_current_callback",
    "set_current_callback",
]
