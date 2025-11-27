# token-copilot

> **Your AI copilot for LLM costs**

**Multi-tenant cost tracking and budget enforcement for LangChain, LangGraph, and LlamaIndex**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/token-copilot.svg)](https://badge.fury.io/py/token-copilot)

---

## What is token-copilot?

`token-copilot` is a comprehensive library for tracking, analyzing, and optimizing LLM costs in production. It works seamlessly with **LangChain**, **LangGraph**, and **LlamaIndex** applications, providing automatic cost tracking, multi-tenant support, intelligent routing, and budget enforcement.

### Why token-copilot?

- **🚀 Zero Config**: One-line integration with LangChain, LangGraph, and LlamaIndex
- **👥 Multi-Tenant**: Track costs by user, organization, session, or any dimension
- **💰 Budget Enforcement**: Hard stops when budget limits reached
- **📊 Advanced Analytics**: Waste analysis, efficiency scoring, anomaly detection
- **🧭 Intelligent Routing**: Auto-select optimal models based on complexity
- **📈 Forecasting**: Predict budget exhaustion with confidence scores
- **⚡ Request Queuing**: Priority-based request management
- **📉 Cost Optimization**: Identify and eliminate waste in real-time

---

## Installation

```bash
pip install token-copilot
```

**With all features (analytics, forecasting, routing):**
```bash
pip install token-copilot[analytics]
```

**For development:**
```bash
pip install token-copilot[dev]
```

---

## Quick Start

### Basic Usage

```python
from langchain import ChatOpenAI, LLMChain, PromptTemplate
from token_copilot import TokenPilotCallback

# Create callback with budget limit
callback = TokenPilotCallback(budget_limit=10.00)

# Use with any LangChain LLM
llm = ChatOpenAI(callbacks=[callback])
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer: {question}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Make calls
result = chain.run("What is Python?")

# Get stats
print(f"Total cost: ${callback.get_total_cost():.4f}")
print(f"Remaining budget: ${callback.get_remaining_budget():.2f}")
```

### Multi-Tenant Tracking

```python
from token_copilot import TokenPilotCallback

callback = TokenPilotCallback()

llm = ChatOpenAI(callbacks=[callback])
chain = LLMChain(llm=llm, prompt=prompt)

# Track per user/organization
result = chain.run(
    "question",
    metadata={
        "user_id": "user_123",
        "org_id": "org_456",
        "feature": "chat"
    }
)

# Get costs by user
costs_by_user = callback.get_costs_by('user_id')
print(costs_by_user)
# {'user_123': 0.0015, 'user_456': 0.0032, ...}

# Get costs by organization
costs_by_org = callback.get_costs_by('org_id')
print(costs_by_org)
# {'org_456': 0.0047, ...}
```

### Analytics with Pandas

```python
import pandas as pd
from token_copilot import TokenPilotCallback

callback = TokenPilotCallback()

# ... make LLM calls ...

# Export to DataFrame
df = callback.to_dataframe()

# Analyze costs
print(df.groupby('user_id')['cost'].sum())
print(df.groupby('org_id')['cost'].sum())
print(df.groupby('model')['cost'].sum())

# Filter and analyze
chat_costs = df[df['feature'] == 'chat']['cost'].sum()
summary_costs = df[df['feature'] == 'summarize']['cost'].sum()
```

### Budget Enforcement

```python
from token_copilot import TokenPilotCallback, BudgetExceededError

# Option 1: Global budget
callback = TokenPilotCallback(
    budget_limit=100.00,           # $100 total
    on_budget_exceeded="raise"     # Raise exception (default)
)

# Option 2: Daily budget
callback = TokenPilotCallback(
    budget_limit=50.00,
    budget_period="daily"          # Reset daily
)

# Option 3: Per-user budget
callback = TokenPilotCallback(
    budget_limit=10.00,
    budget_period="per_user"       # $10 per user
)

# Option 4: Per-organization budget
callback = TokenPilotCallback(
    budget_limit=100.00,
    budget_period="per_org"        # $100 per org
)

try:
    result = chain.run("question", metadata={"user_id": "user_123"})
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
    # Handle gracefully
```

---

## Features

### ✨ Core Features

- ✅ **LangChain Integration**: Simple callback interface (`TokenPilotCallback`)
- ✅ **LangGraph Integration**: Works with StateGraph workflows
- ✅ **LlamaIndex Integration**: Full support via `TokenPilotCallbackHandler`
- ✅ **Multi-Tenant Tracking**: Track by user, org, session, feature, endpoint, etc.
- ✅ **Budget Enforcement**: Total, daily, monthly, per-user, per-org budgets
- ✅ **Pandas Export**: DataFrame export for advanced analytics
- ✅ **Model Pricing**: Built-in pricing for 19+ OpenAI and Anthropic models

### 📊 Analytics & Optimization

- ✅ **Waste Analysis**: Detect repeated prompts, excessive context, verbose outputs
- ✅ **Efficiency Scoring**: Score users/orgs with leaderboards
- ✅ **Anomaly Detection**: Real-time cost/token/frequency spike detection
- ✅ **Alert Handlers**: Log, webhook, and Slack integrations

### 🧭 Intelligent Routing

- ✅ **Model Router**: Auto-select optimal models based on complexity
- ✅ **5 Routing Strategies**: CHEAPEST_FIRST, QUALITY_FIRST, BALANCED, COST_THRESHOLD, LEARNED
- ✅ **Quality Feedback**: Learn from historical quality scores

### 📈 Forecasting & Monitoring

- ✅ **Budget Predictor**: Linear regression forecasting
- ✅ **Burn Rate Analysis**: Hours until budget exhaustion
- ✅ **Predictive Alerts**: Custom alert rules with cooldown periods
- ✅ **Background Monitoring**: Automated budget monitoring threads

### ⚡ Request Management

- ✅ **Smart Queuing**: Priority-based request queuing (4 modes)
- ✅ **Priority Levels**: CRITICAL, HIGH, NORMAL, LOW
- ✅ **Budget-Aware**: Automatic queuing based on budget thresholds

---

## API Reference

### TokenPilotCallback

Primary interface for cost tracking.

```python
from token_copilot import TokenPilotCallback

callback = TokenPilotCallback(
    budget_limit=100.00,           # Optional budget limit in USD
    budget_period="total",         # "total", "daily", "monthly", "per_user", "per_org"
    on_budget_exceeded="raise"     # "raise", "warn", "ignore"
)
```

**Core Methods:**

- `get_total_cost()` → `float`: Total cost across all calls
- `get_total_tokens()` → `int`: Total tokens used
- `get_stats()` → `dict`: Summary statistics
- `get_remaining_budget(metadata=None)` → `float`: Remaining budget
- `to_dataframe()` → `pd.DataFrame`: Export to pandas
- `get_costs_by(dimension)` → `dict`: Costs grouped by dimension ('user_id', 'org_id', 'model')
- `reset()`: Reset all tracking data

**Analytics Methods** (requires `pip install token-copilot[analytics]`):

- `analyze_waste()` → `dict`: Detect token waste and calculate savings
- `get_efficiency_score(entity_type, entity_id)` → `EfficiencyMetrics`: Score efficiency
- `get_leaderboard(entity_type, top_n)` → `List[dict]`: Get top performers
- `get_anomalies(minutes, min_severity)` → `List[Anomaly]`: Get recent anomalies

**Routing Methods:**

- `suggest_model(prompt, estimated_tokens)` → `RoutingDecision`: Get model suggestion
- `record_model_quality(model, quality_score)`: Record quality for learned routing

**Forecasting Methods:**

- `get_forecast(forecast_hours)` → `BudgetForecast`: Get budget forecast
- `get_queue_stats()` → `dict`: Get queue statistics

### Metadata Fields

Pass metadata to track costs by dimension:

```python
metadata = {
    "user_id": "user_123",        # User identifier
    "org_id": "org_456",          # Organization identifier
    "session_id": "session_789",  # Session identifier
    "feature": "chat",            # Feature name
    "endpoint": "/api/chat",      # API endpoint
    "environment": "prod",        # Environment
    "tags": {"key": "value"}      # Custom tags
}

result = chain.run("question", metadata=metadata)
```

---

## Examples

See [examples/basic_usage.py](examples/basic_usage.py) for complete examples:

- Basic cost tracking
- Budget enforcement
- Multi-tenant tracking
- Pandas analytics

---

## Production Usage

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException, Header
from langchain import ChatOpenAI, LLMChain
from token_copilot import TokenPilotCallback, BudgetExceededError

app = FastAPI()

# Global callback with daily budget
callback = TokenPilotCallback(
    budget_limit=100.00,
    budget_period="daily"
)

llm = ChatOpenAI(callbacks=[callback])
chain = LLMChain(llm=llm, prompt=prompt)

@app.post("/chat")
async def chat(
    message: str,
    user_id: str = Header(...),
    org_id: str = Header(...)
):
    try:
        result = chain.run(
            message,
            metadata={
                "user_id": user_id,
                "org_id": org_id,
                "feature": "chat",
                "endpoint": "/chat"
            }
        )

        return {
            "response": result,
            "cost": callback.tracker.get_last_cost(),
            "budget_remaining": callback.get_remaining_budget()
        }

    except BudgetExceededError:
        raise HTTPException(status_code=429, detail="Daily budget exceeded")


@app.get("/analytics")
async def analytics(org_id: str = Header(...)):
    df = callback.to_dataframe()
    org_df = df[df['org_id'] == org_id]

    return {
        "total_cost": float(org_df['cost'].sum()),
        "total_tokens": int(org_df['total_tokens'].sum()),
        "num_requests": len(org_df),
        "cost_by_user": org_df.groupby('user_id')['cost'].sum().to_dict()
    }
```

---

## Supported Models

Built-in pricing for:

**OpenAI:**
- gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini

**Anthropic:**
- claude-2.0, claude-2.1, claude-3-opus, claude-3-sonnet, claude-3-haiku

See [model pricing database](src/token-copilot/utils/pricing.py) for complete list.

---

## FAQ

**Q: Does this work with streaming?**
A: v1.0 tracks costs after completion. Streaming support coming in v1.1.

**Q: Can I use this without LangChain?**
A: Yes! Use `MultiTenantTracker` directly:

```python
from token_copilot import MultiTenantTracker

tracker = MultiTenantTracker()
tracker.track(
    model="gpt-4",
    input_tokens=1000,
    output_tokens=500,
    metadata={"user_id": "user_123"}
)
```

**Q: How accurate is the cost calculation?**
A: Costs are calculated using official provider pricing. Accuracy depends on correct token counts from LangChain.

**Q: Does this require API keys?**
A: No! token-copilot only tracks costs, it doesn't make API calls. Your LangChain LLM handles API calls.

---

## Contributing

Contributions welcome! Please open an issue or PR.

### Development Setup

```bash
git clone https://github.com/scionoftech/token-copilot.git
cd token-copilot
pip install -e ".[dev]"
pytest
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Support

- **Issues**: https://github.com/scionoftech/token-copilot/issues
- **Discussions**: https://github.com/scionoftech/token-copilot/discussions

---

**Made with ❤️ for the LangChain community**
