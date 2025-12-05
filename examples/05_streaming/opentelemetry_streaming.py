"""
Example: Streaming cost events via OpenTelemetry (OTLP).

This example demonstrates how to export LLM cost events as OpenTelemetry
spans for integration with modern observability platforms.

Supported platforms:
- Jaeger
- Zipkin
- Honeycomb
- Datadog
- New Relic
- Grafana Tempo
- AWS X-Ray
- Google Cloud Trace

Prerequisites:
    - pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
    - OpenTelemetry Collector running (or direct platform endpoint)
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from token_copilot import TokenCoPilotCallback
from token_copilot.streaming import OpenTelemetryStreamer

# Option 1: Local OpenTelemetry Collector
streamer = OpenTelemetryStreamer(
    endpoint="http://localhost:4318",  # HTTP endpoint
    protocol="http",
    service_name="llm-service",
    service_version="1.0.0",
    deployment_environment="production"
)

# Option 2: Honeycomb
# streamer = OpenTelemetryStreamer(
#     endpoint="https://api.honeycomb.io",
#     service_name="llm-service",
#     headers={
#         "x-honeycomb-team": "your-api-key",
#         "x-honeycomb-dataset": "llm-costs"
#     }
# )

# Option 3: Datadog
# streamer = OpenTelemetryStreamer(
#     endpoint="https://trace.agent.datadoghq.com",
#     service_name="llm-service",
#     headers={
#         "DD-API-KEY": "your-datadog-api-key"
#     }
# )

# Create callback with streamer
callback = TokenCoPilotCallback(
    budget_limit=100.00,
    streamer=streamer
)

# Create LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[callback]
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# Create chain
chain = prompt | llm

# Make traced calls
print("Exporting LLM cost spans via OpenTelemetry...")
print(f"Service: {streamer.service_name}")
print(f"Endpoint: {streamer.endpoint}\n")

scenarios = [
    {
        "user": "alice",
        "org": "engineering",
        "question": "How do I optimize a database query?",
        "feature": "code_assistance"
    },
    {
        "user": "bob",
        "org": "engineering",
        "question": "Explain microservices architecture",
        "feature": "documentation"
    },
    {
        "user": "charlie",
        "org": "sales",
        "question": "Write a product description",
        "feature": "content_generation"
    },
]

for scenario in scenarios:
    response = chain.invoke(
        {"question": scenario["question"]},
        config={
            "metadata": {
                "user_id": scenario["user"],
                "org_id": scenario["org"],
                "feature": scenario["feature"],
                "environment": "production"
            }
        }
    )
    print(f"✓ Span exported: {scenario['user']} - {scenario['feature']}")

# Analytics
print(f"\n{'='*60}")
print("Cost Summary")
print(f"{'='*60}")

stats = callback.get_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Total cost: ${stats['total_cost']:.4f}")
print(f"Avg cost/call: ${stats['avg_cost_per_call']:.4f}")

print(f"\nCosts by Feature:")
df = callback.to_dataframe()
if not df.empty:
    feature_costs = df.groupby('feature')['cost'].sum()
    for feature, cost in feature_costs.items():
        print(f"  {feature}: ${cost:.4f}")

print(f"\nAll spans exported to OpenTelemetry!")
print(f"View traces in your observability platform.")

# Close streamer (flushes spans)
streamer.close()
