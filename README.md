# token-copilot

> **Your AI copilot for LLM costs** - Modern, plugin-based cost tracking for LangChain, LangGraph, and LlamaIndex

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/token-copilot.svg)](https://badge.fury.io/py/token-copilot)
[![Version](https://img.shields.io/badge/version-1.0.2-green.svg)](https://github.com/scionoftech/token-copilot)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](./TEST_RESULTS.md)

---

## 🚀 What is token-copilot?

A **lightweight, production-ready library** for tracking and optimizing LLM costs. Track costs in real-time, enforce budgets, analyze usage patterns, and export data for reporting - all with minimal configuration.

**Key Benefits:**
- 🎯 **Zero Config** - Start tracking with 2 lines of code
- 💰 **Budget Control** - Automatic budget enforcement and alerts
- 📊 **Multi-Tenant** - Track costs by user, org, session, or any dimension
- 🔌 **Plugin-Based** - Add features only when needed
- 🌐 **Framework Agnostic** - Works with LangChain, LangGraph, LlamaIndex
- ☁️ **Azure OpenAI** - Full support with automatic cost calculation

**Verified Working:**
- ✅ 12/12 core tests passing with Azure OpenAI
- ✅ Production-tested with real workloads
- ✅ 19+ supported LLM models (OpenAI, Anthropic, Ollama)

---

## ⚡ Quick Start

```bash
pip install token-copilot
```

```python
from token_copilot import TokenCoPilot
from langchain_openai import ChatOpenAI

# Create copilot with budget
copilot = TokenCoPilot(budget_limit=10.00)
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

# Use normally
result = llm.invoke("What is Python?")

# Get metrics
print(f"Cost: ${copilot.cost:.4f}")
print(f"Tokens: {copilot.tokens:,}")
print(f"Remaining: ${copilot.get_remaining_budget():.2f}")
```

**That's it!** You're now tracking costs and enforcing budgets.

---

## 📦 Installation

### Basic (Core Features)
```bash
pip install token-copilot
```

### With Analytics
```bash
pip install token-copilot[analytics]
```

### With Streaming
```bash
pip install token-copilot[streaming]
```

### All Features
```bash
pip install token-copilot[all]
```

---

## ✨ Features

### Core Features
| Feature | Description |
|---------|-------------|
| **Cost Tracking** | Automatic token and cost tracking for all LLM calls |
| **Budget Enforcement** | Hard stops at budget limits with configurable actions |
| **Multi-Tenant** | Track costs by user, organization, session, or custom dimensions |
| **DataFrame Export** | Export to pandas for advanced analytics and reporting |
| **19+ Models** | Support for OpenAI, Anthropic, Azure OpenAI, and Ollama models |
| **Real-time Stats** | Get total cost, tokens, averages, and remaining budget instantly |

### Optional Plugins
| Plugin | Description |
|--------|-------------|
| **Persistence** | Save cost history to SQLite or JSON for long-term tracking |
| **Analytics** | Detect waste, anomalies, and efficiency issues automatically |
| **Streaming** | Stream events to Webhook, Kafka, Syslog, or OpenTelemetry |
| **Routing** | Intelligent model selection based on cost and quality |
| **Adaptive** | Auto-adjust parameters based on remaining budget |
| **Forecasting** | Predict when budget will be exhausted |

### Framework Support
- ✅ **LangChain** - Full support via callbacks
- ✅ **LangGraph** - Works with graph-based workflows
- ✅ **LlamaIndex** - Dedicated callback handler
- ✅ **Azure OpenAI** - Automatic model detection and cost calculation

---

## 🎯 Usage Patterns

Choose the style that fits your needs:

### 1. Minimal (Simplest)
```python
from token_copilot import TokenCoPilot
from langchain_openai import ChatOpenAI

copilot = TokenCoPilot(budget_limit=10.00)
llm = ChatOpenAI(callbacks=[copilot])
result = llm.invoke("Hello!")
print(f"Cost: ${copilot.cost:.4f}")
```

### 2. Builder (Fluent API)
```python
copilot = (TokenCoPilot(budget_limit=100.00)
    .with_persistence(backend=SQLiteBackend("costs.db"))
    .with_analytics(detect_anomalies=True)
    .with_adaptive()
    .build()
)
llm = ChatOpenAI(callbacks=[copilot])
```

### 3. Factory Presets
```python
from token_copilot.presets import production

copilot = production(
    budget_limit=1000.00,
    webhook_url="https://monitoring.example.com",
    detect_anomalies=True
)
llm = ChatOpenAI(callbacks=[copilot])
```

### 4. Context Manager
```python
from token_copilot import track_costs

with track_costs(budget_limit=5.00) as copilot:
    llm = ChatOpenAI(callbacks=[copilot])
    result = llm.invoke("Hello!")
    print(f"Cost: ${copilot.cost:.4f}")
# Automatic summary on exit
```

### 5. Decorator
```python
from token_copilot.decorators import track_cost

@track_cost(budget_limit=5.00)
def process_text(text):
    llm = ChatOpenAI(callbacks=[process_text.copilot])
    return llm.invoke(f"Process: {text}")

result = process_text("my text")
print(f"Cost: ${process_text.copilot.cost:.4f}")
```

---

## 📖 Core Examples

### Budget Enforcement
```python
from token_copilot import TokenCoPilot, BudgetExceededError

copilot = TokenCoPilot(
    budget_limit=1.00,
    on_budget_exceeded="raise"  # Options: "raise", "warn", "ignore"
)

llm = ChatOpenAI(callbacks=[copilot])

try:
    result = llm.invoke("Expensive task...")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
```

### Multi-Tenant Tracking
```python
copilot = TokenCoPilot(budget_limit=100.00)
llm = ChatOpenAI(callbacks=[copilot])

# Track per user
result = llm.invoke(
    "Hello",
    config={
        "metadata": {
            "user_id": "user_123",
            "org_id": "org_456",
            "session_id": "session_789"
        }
    }
)

# Get costs by dimension
user_costs = copilot.tracker.get_costs_by("user_id")
org_costs = copilot.tracker.get_costs_by("org_id")

print(f"User user_123: ${user_costs['user_123']:.4f}")
print(f"Org org_456: ${org_costs['org_456']:.4f}")
```

### DataFrame Export & Analytics
```python
import pandas as pd

copilot = TokenCoPilot()
llm = ChatOpenAI(callbacks=[copilot])

# Make calls...
for i in range(100):
    result = llm.invoke(f"Task {i}")

# Export to DataFrame
df = copilot.to_dataframe()

# Analyze
print(df.groupby('user_id')['cost'].sum())
print(df.groupby('model')['cost'].mean())

# Time series
hourly_costs = df.resample('H')['cost'].sum()

# Save reports
df.to_csv('llm_costs.csv')
df.to_excel('llm_costs.xlsx')
```

### Statistics & Metrics
```python
copilot = TokenCoPilot(budget_limit=50.00)
llm = ChatOpenAI(callbacks=[copilot])

# Make some calls...

# Get statistics
stats = copilot.get_stats()
print(f"Total Calls: {stats['total_calls']}")
print(f"Total Cost: ${stats['total_cost']:.4f}")
print(f"Total Tokens: {stats['total_tokens']:,}")
print(f"Avg Cost/Call: ${stats['avg_cost_per_call']:.4f}")
print(f"Avg Tokens/Call: {stats['avg_tokens_per_call']:.1f}")

# Check remaining budget
remaining = copilot.get_remaining_budget()
print(f"Remaining: ${remaining:.2f}")
```

---

## 🔌 Plugin Examples

### Persistence (Save History)
```python
from token_copilot import TokenCoPilot
from token_copilot.plugins import SQLiteBackend

# SQLite backend (production-ready)
backend = SQLiteBackend(db_path="costs.db")
copilot = (TokenCoPilot(budget_limit=100.00)
    .with_persistence(backend=backend, session_id="session_123")
)

llm = ChatOpenAI(callbacks=[copilot])
response = llm.invoke("Hello!")

# Query historical data
plugin = copilot._plugin_manager.get_plugins()[0]
summary = plugin.get_summary()
print(f"Total cost: ${summary['total_cost']:.2f}")
print(f"Total calls: {summary['total_calls']}")

# Get recent events
from datetime import datetime, timedelta
events = plugin.get_events(
    start_time=datetime.now() - timedelta(hours=24)
)
```

### Analytics (Detect Issues)
```python
copilot = (TokenCoPilot(budget_limit=100.00)
    .with_analytics(detect_anomalies=True, track_waste=True)
)

llm = ChatOpenAI(callbacks=[copilot])

# Make calls...
for i in range(50):
    result = llm.invoke(f"Task {i}")

# Get analytics
from token_copilot.plugins.analytics import AnalyticsPlugin
analytics = copilot._plugin_manager.get_plugins(AnalyticsPlugin)[0]

# Check for anomalies
anomalies = analytics.get_anomalies(minutes=60)
for anomaly in anomalies:
    print(f"[{anomaly.severity}] {anomaly.message}")
```

### Adaptive (Budget-Based Adjustments)
```python
copilot = (TokenCoPilot(budget_limit=100.00)
    .with_adaptive()
)

from token_copilot.plugins.adaptive import AdaptivePlugin
adaptive = copilot._plugin_manager.get_plugins(AdaptivePlugin)[0]

# Get current budget tier
tier_info = adaptive.get_tier_info()
print(f"Budget tier: {tier_info['tier_name']}")  # abundant, comfortable, constrained, critical
print(f"Remaining: ${tier_info['remaining']:.2f}")

# Operations automatically adjust based on tier
# - abundant: high quality, max tokens
# - comfortable: balanced
# - constrained: conservative
# - critical: minimal usage
```

---

## 🌐 Azure OpenAI Support

Full support with automatic cost calculation:

```python
from token_copilot import TokenCoPilot
from langchain_openai import AzureChatOpenAI
import os

# Configure Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Use with token-copilot (costs tracked automatically)
copilot = TokenCoPilot(budget_limit=10.00)
response = llm.invoke("Hello!", config={"callbacks": [copilot]})

print(f"Cost: ${copilot.cost:.6f}")
print(f"Tokens: {copilot.tokens:,}")
```

**Environment Setup (.env file):**
```bash
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

**Supported Models:**
- ✅ gpt-4o-mini (all versions)
- ✅ gpt-4o (all versions)
- ✅ gpt-4-turbo (all versions)
- ✅ gpt-3.5-turbo (all versions)

---

## 🎯 LangGraph Support

Works seamlessly with LangGraph workflows:

```python
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilot

copilot = TokenCoPilot(budget_limit=10.00)

# Create graph
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
graph = builder.compile()

# Run with cost tracking
result = graph.invoke(
    {"messages": [("user", "Hello")]},
    config={"callbacks": [copilot]}
)

print(f"Total cost: ${copilot.cost:.4f}")
print(f"Total tokens: {copilot.tokens:,}")
```

---

## 📚 API Reference

### TokenCoPilot Class

```python
copilot = TokenCoPilot(
    budget_limit=100.00,           # Optional: Budget limit in USD
    budget_period="total",         # "total", "daily", "monthly", "per_user", "per_org"
    on_budget_exceeded="raise"     # "raise", "warn", "ignore"
)
```

**Properties:**
- `copilot.cost` - Total cost in USD
- `copilot.tokens` - Total tokens used
- `copilot.budget_limit` - Current budget limit

**Core Methods:**
- `get_total_cost()` - Get total cost
- `get_total_tokens()` - Get total tokens
- `get_stats()` - Get summary statistics (dict)
- `get_remaining_budget(metadata=None)` - Get remaining budget
- `to_dataframe()` - Export to pandas DataFrame

**Builder Methods:**
- `.with_persistence(backend, session_id)` - Add persistence
- `.with_analytics(detect_anomalies=True)` - Add analytics
- `.with_streaming(webhook_url=...)` - Add streaming
- `.with_adaptive()` - Add adaptive operations
- `.with_forecasting(forecast_hours=48)` - Add forecasting
- `.build()` - Finalize (optional)

### Pricing Utilities

```python
from token_copilot import get_model_config, calculate_cost, list_models

# Get model configuration
config = get_model_config("gpt-4o-mini")
print(config.input_cost_per_1m)   # Cost per 1M input tokens
print(config.output_cost_per_1m)  # Cost per 1M output tokens

# Calculate cost
cost = calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
print(f"Cost: ${cost:.6f}")

# List all supported models
models = list_models()  # Returns 19+ model IDs
```

### Direct Tracker Usage (Without LangChain)

```python
from token_copilot.tracking import MultiTenantTracker

tracker = MultiTenantTracker()
entry = tracker.track(
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=50,
    metadata={"user_id": "user_123"}
)

print(f"Cost: ${entry.cost:.6f}")
print(f"Total: ${tracker.get_total_cost():.6f}")
```

---

## 🏭 Factory Presets

Pre-configured setups for common scenarios:

```python
from token_copilot.presets import basic, development, production, enterprise

# Basic - Just cost tracking
copilot = basic(budget_limit=10.00)

# Development - With logging and anomaly detection
copilot = development(budget_limit=50.00, detect_anomalies=True)

# Production - Full monitoring with alerts
copilot = production(
    budget_limit=1000.00,
    webhook_url="https://monitoring.example.com",
    detect_anomalies=True,
    enable_forecasting=True
)

# Enterprise - All features enabled
copilot = enterprise(
    budget_limit=10000.00,
    kafka_brokers=["kafka:9092"],
    otlp_endpoint="http://collector:4318",
    enable_all=True
)
```

---

## 🔍 Real-World Example

Complete chatbot with cost tracking:

```python
from token_copilot import TokenCoPilot, BudgetExceededError
from langchain_openai import ChatOpenAI

def chatbot():
    copilot = TokenCoPilot(budget_limit=5.00, on_budget_exceeded="warn")
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

    print("Chatbot started! (type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        # Check budget
        if copilot.get_remaining_budget() <= 0:
            print("Budget exhausted!")
            break

        try:
            response = llm.invoke(user_input)
            print(f"Bot: {response.content}")
            print(f"Cost this turn: ${copilot.tracker.get_last_cost():.6f}")
        except BudgetExceededError:
            print("Budget limit reached!")
            break

    # Final stats
    stats = copilot.get_stats()
    print(f"\n📊 Session Summary:")
    print(f"  Total turns: {stats['total_calls']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    print(f"  Avg cost/turn: ${stats['avg_cost_per_call']:.4f}")

if __name__ == "__main__":
    chatbot()
```

---

## ❓ FAQ

**Q: Does this work with streaming responses?**
A: Currently tracks costs after completion. Streaming support coming in v1.0.3.

**Q: Can I use without LangChain?**
A: Yes! Use `MultiTenantTracker` directly (see API Reference above).

**Q: How accurate is the cost tracking?**
A: Uses official pricing from OpenAI and Anthropic. Updated regularly. 100% accurate for supported models.

**Q: Which usage pattern should I use?**
A:
- **Getting started**: Minimal or Factory presets
- **Production**: Builder or Production preset
- **Reusable code**: Decorators or Context managers

**Q: Can I create custom plugins?**
A: Yes! Extend the `Plugin` base class:

```python
from token_copilot.core import Plugin

class MyPlugin(Plugin):
    def on_cost_tracked(self, model, tokens, cost, metadata):
        print(f"Custom logic: ${cost:.6f}")

copilot = TokenCoPilot()
copilot.add_plugin(MyPlugin())
```

---

## 📚 Documentation

### Full Documentation
**[📖 Complete Documentation (Single Page HTML)](./DOCUMENTATION.html)** - Open in your browser for comprehensive guide with all features, examples, and API reference.

---

## 🤝 Contributing

Contributions are welcome! We appreciate all contributions, from bug reports to new features.

### How to Contribute

1. **Report Bugs** - Open an [issue](https://github.com/scionoftech/token-copilot/issues) with details
2. **Suggest Features** - Share your ideas in [discussions](https://github.com/scionoftech/token-copilot/discussions)
3. **Submit PRs** - Fork, create a branch, and submit a pull request
4. **Improve Docs** - Help make documentation better
5. **Share Examples** - Contribute real-world usage examples

### Development Setup

```bash
# Clone the repository
git clone https://github.com/scionoftech/token-copilot.git
cd token-copilot

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python test_core.py
```

### Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Follow existing code style
- Be respectful and constructive

All contributions, big or small, are appreciated!
---

## 🔗 Links

- **GitHub**: https://github.com/scionoftech/token-copilot
- **PyPI**: https://pypi.org/project/token-copilot/
- **Issues**: https://github.com/scionoftech/token-copilot/issues
- **Documentation**: [DOCUMENTATION.html](./DOCUMENTATION.html)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

Built with ❤️ by [Sai Kumar Yava](https://github.com/scionoftech)

If you find this useful, please ⭐ star the repo!

---

## 🚀 Quick Start Checklist

- [ ] Install: `pip install token-copilot`
- [ ] Import: `from token_copilot import TokenCoPilot`
- [ ] Create: `copilot = TokenCoPilot(budget_limit=10.00)`
- [ ] Use: `llm = ChatOpenAI(callbacks=[copilot])`
- [ ] Track: `print(f"Cost: ${copilot.cost:.4f}")`
- [ ] **You're done!** 🎉

**Need help?** Open an [issue](https://github.com/scionoftech/token-copilot/issues) or check the [documentation](./DOCUMENTATION.html).
