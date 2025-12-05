"""
Example: Streaming cost events to a webhook endpoint.

This example demonstrates how to stream LLM cost events to an HTTP webhook
for real-time analytics and monitoring.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from token_copilot import TokenCoPilotCallback
from token_copilot.streaming import WebhookStreamer

# Create webhook streamer
streamer = WebhookStreamer(
    url="https://your-analytics-service.com/api/events",
    batch_size=5,  # Batch 5 events together for efficiency
    flush_interval=10.0,  # Flush every 10 seconds
    headers={
        "Authorization": "Bearer your-api-key-here",
        "X-Environment": "production"
    },
    async_mode=True,  # Send in background thread
)

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

# Make some calls with metadata
print("Making LLM calls with webhook streaming...")

response1 = chain.invoke(
    {"question": "What is Python?"},
    config={
        "metadata": {
            "user_id": "user_123",
            "org_id": "org_456",
            "feature": "chat"
        }
    }
)
print(f"Response 1: {response1.content[:100]}...")

response2 = chain.invoke(
    {"question": "Explain machine learning in simple terms"},
    config={
        "metadata": {
            "user_id": "user_789",
            "org_id": "org_456",
            "feature": "chat"
        }
    }
)
print(f"Response 2: {response2.content[:100]}...")

# Get stats
print(f"\nTotal cost: ${callback.get_total_cost():.4f}")
print(f"Total tokens: {callback.get_total_tokens():,}")

# Force flush remaining events
streamer.flush()

print("\nAll events streamed to webhook!")
print(f"Webhook URL: {streamer.url}")

# Close streamer (flushes and cleans up)
streamer.close()
