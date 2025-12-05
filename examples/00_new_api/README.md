# New API Examples (v2.0)

This directory showcases token-copilot's redesigned API with multiple usage patterns.

## 🎯 Design Philosophy

token-copilot v2.0 follows a **hybrid approach** offering multiple patterns:

1. **Minimal** - Zero config, just track costs
2. **Builder** - Fluent API for adding features
3. **Factory** - Pre-configured presets
4. **Context Managers** - Pythonic scoped tracking
5. **Decorators** - Function-level automation

Choose the pattern that fits your style!

## 📁 Examples

### 01_minimal.py - Simplest Usage
```python
from token_copilot import TokenCoPilot

copilot = TokenCoPilot(budget_limit=1.00)
llm = ChatOpenAI(callbacks=[copilot])
print(f"Cost: ${copilot.cost:.4f}")
```

**When to use**: Getting started, simple scripts

---

### 02_builder_pattern.py - Fluent API
```python
copilot = (TokenCoPilot(budget_limit=100.00)
    .with_streaming(webhook_url="...")
    .with_analytics(detect_anomalies=True)
    .with_adaptive()
    .build()
)
```

**When to use**: Need multiple features, prefer explicit configuration

---

### 03_factory_presets.py - Instant Configuration
```python
from token_copilot.presets import production

copilot = production(
    budget_limit=1000.00,
    webhook_url="https://monitoring.example.com",
)
```

**When to use**: Common use cases, quick setup

**Available presets**:
- `basic()` - Just tracking
- `development()` - Local dev with logging
- `production()` - Monitoring + alerts
- `enterprise()` - All features enabled

---

### 04_context_managers.py - Pythonic Tracking
```python
from token_copilot import track_costs

with track_costs(budget_limit=5.00) as copilot:
    llm = ChatOpenAI(callbacks=[copilot])
    result = llm.invoke("Hello")
```

**When to use**: Scoped operations, automatic cleanup

**Available contexts**:
- `track_costs()` - General purpose
- `with_budget()` - Budget-focused
- `monitored()` - Auto-logging

---

### 05_decorators.py - Function-Level Tracking
```python
from token_copilot.decorators import track_cost

@track_cost(budget_limit=5.00)
def process_text(text):
    llm = ChatOpenAI(callbacks=[process_text.copilot])
    return llm.invoke(f"Process: {text}")

result = process_text("my text")
print(f"Cost: ${process_text.copilot.cost:.4f}")
```

**When to use**: Reusable functions, clean separation

**Available decorators**:
- `@track_cost` - Attach copilot to function
- `@enforce_budget` - Strict budget limits
- `@monitored` - Auto-logging

---

## 🔌 Plugin System

All patterns support plugins:

```python
# Method 1: Explicit plugin addition
from token_copilot.plugins import StreamingPlugin

copilot = TokenCoPilot()
copilot.add_plugin(StreamingPlugin(webhook_url="..."))

# Method 2: Builder pattern
copilot = TokenCoPilot().with_streaming(webhook_url="...")

# Method 3: Factory preset
copilot = production(webhook_url="...")
```

**Available plugins**:
- `StreamingPlugin` - Real-time event streaming
- `AnalyticsPlugin` - Waste detection, anomalies
- `RoutingPlugin` - Intelligent model selection
- `AdaptivePlugin` - Budget-aware operations
- `ForecastingPlugin` - Budget prediction

---

## 🚀 Quick Comparison

| Pattern | Code Lines | Setup Time | Flexibility |
|---------|-----------|------------|-------------|
| Minimal | 2 | Instant | Low |
| Builder | 5 | Fast | High |
| Factory | 3 | Instant | Medium |
| Context | 4 | Fast | Medium |
| Decorator | 6 | Medium | High |

---

## 💡 Recommendations

**For beginners**: Start with `minimal` or `factory presets`

**For production**: Use `builder pattern` or `production preset`

**For enterprise**: Use `enterprise preset` with custom config

**For reusable code**: Use `decorators` or `context managers`

**For experiments**: Use `minimal` or `track_costs()` context

---

## 📚 Learn More

- [Redesign Proposal](../../REDESIGN_PROPOSAL.md) - Architecture details
- [API Documentation](../../README.md) - Full API reference
- [Migration Guide](../../MIGRATION_GUIDE.md) - Upgrade from v1.x
