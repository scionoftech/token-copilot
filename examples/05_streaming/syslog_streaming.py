"""
Example: Streaming cost events to syslog server.

This example demonstrates how to stream LLM cost events to a syslog server
using RFC 5424 structured data format.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from token_copilot import TokenCoPilotCallback
from token_copilot.streaming import SyslogStreamer

# Create syslog streamer
streamer = SyslogStreamer(
    host="logs.company.com",  # or "localhost" for testing
    port=514,
    protocol="tcp",  # or "udp"
    facility=16,  # LOCAL0
    severity=6,   # INFORMATIONAL
    app_name="token-copilot",
    format="rfc5424"  # or "json"
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
print("Making LLM calls with syslog streaming...")

questions = [
    "What is Python?",
    "Explain machine learning",
    "What are neural networks?",
]

for i, question in enumerate(questions, 1):
    response = chain.invoke(
        {"question": question},
        config={
            "metadata": {
                "user_id": f"user_{i}",
                "org_id": "org_456",
                "feature": "qa_system",
                "environment": "production"
            }
        }
    )
    print(f"Question {i}: {question}")
    print(f"Response: {response.content[:80]}...")
    print()

# Get stats
stats = callback.get_stats()
print(f"\nStatistics:")
print(f"  Total cost: ${stats['total_cost']:.4f}")
print(f"  Total tokens: {stats['total_tokens']:,}")
print(f"  Total calls: {stats['total_calls']}")
print(f"  Avg cost/call: ${stats['avg_cost_per_call']:.4f}")

print(f"\nAll events streamed to syslog server at {streamer.host}:{streamer.port}")

# Close streamer
streamer.close()
