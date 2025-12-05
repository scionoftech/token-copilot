"""
Example: Streaming cost events to Apache Kafka.

This example demonstrates how to stream LLM cost events to Kafka
for event-driven architectures and downstream processing.

Prerequisites:
    - Apache Kafka running (or Confluent Cloud, AWS MSK, etc.)
    - pip install kafka-python
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from token_copilot import TokenCoPilotCallback
from token_copilot.streaming import KafkaStreamer

# Create Kafka streamer
streamer = KafkaStreamer(
    bootstrap_servers=["kafka:9092"],  # or ["localhost:9092"] for local
    topic="llm-costs",
    partition_key="user_id",  # Partition by user_id for ordering
    compression_type="gzip",  # Compress for efficiency
    acks="all",  # Wait for all replicas
)

# For Confluent Cloud or other SASL authentication:
# streamer = KafkaStreamer(
#     bootstrap_servers=["pkc-xxx.us-east-1.aws.confluent.cloud:9092"],
#     topic="llm-costs",
#     security_protocol="SASL_SSL",
#     sasl_mechanism="PLAIN",
#     sasl_username="your-api-key",
#     sasl_password="your-api-secret",
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

# Simulate high-volume event streaming
print("Streaming cost events to Kafka...")
print(f"Topic: {streamer.topic}")
print(f"Partition key: {streamer.partition_key}\n")

# Generate events from multiple users
for i in range(10):
    user_id = f"user_{i % 3}"  # 3 different users
    org_id = f"org_{i % 2}"    # 2 different orgs

    response = chain.invoke(
        {"question": f"Question {i+1}: Tell me a fact"},
        config={
            "metadata": {
                "user_id": user_id,
                "org_id": org_id,
                "session_id": f"session_{i}",
                "feature": "fact_generator"
            }
        }
    )
    print(f"✓ Event {i+1} → Kafka (user: {user_id}, org: {org_id})")

# Flush pending messages
streamer.flush()

# Analytics
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"Total events: {len(callback.tracker._entries)}")
print(f"Total cost: ${callback.get_total_cost():.4f}")
print(f"Total tokens: {callback.get_total_tokens():,}")

print(f"\nCosts by User:")
for user_id, cost in callback.get_costs_by_user().items():
    print(f"  {user_id}: ${cost:.4f}")

print(f"\nAll events streamed to Kafka topic: {streamer.topic}")
print("Downstream consumers can now process these events!")

# Close streamer
streamer.close()
