"""
Example: Streaming cost events to Logstash for Elasticsearch.

This example demonstrates how to stream LLM cost events to Logstash
for indexing in Elasticsearch (ELK stack).

Prerequisites:
    - Logstash running with TCP JSON input
    - Example Logstash config:
      ```
      input {
        tcp {
          port => 5000
          codec => json_lines
        }
      }
      output {
        elasticsearch {
          hosts => ["elasticsearch:9200"]
          index => "llm-costs-%{+YYYY.MM.dd}"
        }
      }
      ```
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from token_copilot import TokenCoPilotCallback
from token_copilot.streaming import LogstashStreamer

# Create Logstash streamer
streamer = LogstashStreamer(
    host="logstash.company.com",  # or "localhost" for testing
    port=5000,
    index="llm-costs",  # Elasticsearch index name
    tags=["production", "api", "ml-service"],
    extra_fields={
        "environment": "production",
        "region": "us-east-1",
        "service": "chat-api"
    }
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

# Simulate multi-user, multi-org scenario
print("Streaming cost events to Logstash/Elasticsearch...")

users = [
    ("user_alice", "org_acme", "Support chat"),
    ("user_bob", "org_acme", "Code generation"),
    ("user_charlie", "org_tech", "Documentation"),
]

for user_id, org_id, use_case in users:
    response = chain.invoke(
        {"question": f"Help me with {use_case.lower()}"},
        config={
            "metadata": {
                "user_id": user_id,
                "org_id": org_id,
                "feature": use_case,
                "endpoint": "/api/chat"
            }
        }
    )
    print(f"✓ {user_id} ({org_id}): {use_case}")

# Analytics
print(f"\n{'='*60}")
print("Cost Analysis")
print(f"{'='*60}")

costs_by_user = callback.get_costs_by_user()
for user_id, cost in costs_by_user.items():
    print(f"  {user_id}: ${cost:.4f}")

print(f"\nCosts by Organization:")
costs_by_org = callback.get_costs_by_org()
for org_id, cost in costs_by_org.items():
    print(f"  {org_id}: ${cost:.4f}")

print(f"\nTotal cost: ${callback.get_total_cost():.4f}")
print(f"\nAll events sent to Logstash → Elasticsearch")
print(f"Index: {streamer.index}")

# Close streamer
streamer.close()
