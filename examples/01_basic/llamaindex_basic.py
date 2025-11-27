"""
Basic LlamaIndex Example - Getting Started with Token Copilot

This example demonstrates cost tracking in LlamaIndex applications:
- Vector store indexing
- Query engines
- Chat engines
- Multi-step reasoning
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI
from token_copilot import TokenPilotCallbackHandler


def simple_indexing():
    """Example 1: Basic indexing with cost tracking."""
    print("=" * 60)
    print("Example 1: Simple Indexing")
    print("=" * 60)

    # Create callback
    callback = TokenPilotCallbackHandler(budget_limit=1.00)
    callback_manager = CallbackManager([callback])

    # Create sample documents
    documents = [
        Document(text="Python is a high-level programming language."),
        Document(text="JavaScript is used for web development."),
        Document(text="Rust is a systems programming language."),
    ]

    # Build index with tracking
    print("\nBuilding index...")
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager
    )

    print(f"✅ Index built")
    print(f"Cost: ${callback.get_total_cost():.4f}")
    print(f"Tokens: {callback.get_total_tokens():,}")


def query_engine():
    """Example 2: Query engine with cost tracking."""
    print("\n" + "=" * 60)
    print("Example 2: Query Engine")
    print("=" * 60)

    callback = TokenPilotCallbackHandler(budget_limit=5.00)
    callback_manager = CallbackManager([callback])

    # Create documents
    documents = [
        Document(text="LangChain is a framework for building LLM applications."),
        Document(text="LlamaIndex is a data framework for LLM applications."),
        Document(text="Both frameworks help developers build with LLMs."),
    ]

    # Build index
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager
    )

    # Create query engine
    query_engine = index.as_query_engine()

    # Make queries
    queries = [
        "What is LangChain?",
        "What is LlamaIndex?",
        "How do these frameworks help?"
    ]

    print("\nMaking queries...")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")
        print(f"Cost so far: ${callback.get_total_cost():.4f}")

    # Final stats
    print("\n" + "-" * 60)
    print(f"Total queries: {len(queries)}")
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    print(f"Avg cost/query: ${callback.get_total_cost()/len(queries):.4f}")


def chat_engine():
    """Example 3: Chat engine with conversation tracking."""
    print("\n" + "=" * 60)
    print("Example 3: Chat Engine")
    print("=" * 60)

    callback = TokenPilotCallbackHandler(budget_limit=5.00)
    callback_manager = CallbackManager([callback])

    # Create documents
    documents = [
        Document(text="The Eiffel Tower is in Paris, France."),
        Document(text="Paris is the capital of France."),
        Document(text="The Eiffel Tower was built in 1889."),
    ]

    # Build index
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager
    )

    # Create chat engine
    chat_engine = index.as_chat_engine()

    # Have a conversation
    conversation = [
        "Where is the Eiffel Tower?",
        "When was it built?",
        "Tell me more about Paris",
    ]

    print("\nHaving a conversation...")
    for i, message in enumerate(conversation, 1):
        print(f"\n👤 User: {message}")
        response = chat_engine.chat(message)
        print(f"🤖 Assistant: {response}")
        print(f"Cost: ${callback.get_total_cost():.4f}")

    print("\n" + "-" * 60)
    print(f"Total messages: {len(conversation)}")
    print(f"Total cost: ${callback.get_total_cost():.4f}")


def multi_user_tracking():
    """Example 4: Multi-user cost tracking."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-User Tracking")
    print("=" * 60)

    callback = TokenPilotCallbackHandler(
        budget_limit=2.00,
        budget_period="per_user"
    )
    callback_manager = CallbackManager([callback])

    # Create documents
    documents = [
        Document(text="Machine learning is a subset of AI."),
        Document(text="Deep learning uses neural networks."),
    ]

    # Build index
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager
    )

    query_engine = index.as_query_engine()

    # Simulate multiple users
    users = ["alice", "bob", "charlie"]

    print("\nSimulating multi-user queries...")
    for user in users:
        print(f"\n👤 User: {user}")

        # Query with user metadata
        # Note: LlamaIndex metadata passing varies by version
        # This is a conceptual example
        response = query_engine.query("What is machine learning?")

        print(f"Response: {response}")

    # Costs by user
    print("\n" + "-" * 60)
    print("Costs by user:")
    costs_by_user = callback.get_costs_by_user()
    for user, cost in costs_by_user.items():
        print(f"  {user}: ${cost:.4f}")


def with_analytics():
    """Example 5: Full analytics integration."""
    print("\n" + "=" * 60)
    print("Example 5: With Analytics")
    print("=" * 60)

    callback = TokenPilotCallbackHandler(
        budget_limit=10.00,
        anomaly_detection=True,
        predictive_alerts=True
    )
    callback_manager = CallbackManager([callback])

    # Create documents
    documents = [
        Document(text=f"Document {i} contains information.")
        for i in range(5)
    ]

    # Build index
    print("\nBuilding index...")
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager
    )

    # Make queries
    query_engine = index.as_query_engine()

    print("\nMaking queries...")
    for i in range(10):
        response = query_engine.query(f"Query {i}")

    # Analytics
    print("\n" + "-" * 60)
    print("📊 ANALYTICS")

    # Basic stats
    stats = callback.get_stats()
    print(f"\nTotal calls: {stats['total_calls']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"Total tokens: {stats['total_tokens']:,}")

    # Anomalies
    anomalies = callback.get_anomalies(minutes=60)
    if anomalies:
        print(f"\n⚠️ Anomalies detected: {len(anomalies)}")
    else:
        print("\n✅ No anomalies detected")

    # Forecast
    try:
        forecast = callback.get_forecast()
        print(f"\n📈 Forecast:")
        print(f"  Remaining budget: ${forecast.remaining_budget:.2f}")
        if forecast.hours_until_exhausted:
            print(f"  Hours until exhausted: {forecast.hours_until_exhausted:.1f}")
    except Exception as e:
        print(f"\nForecast: {e}")

    # Export
    df = callback.to_dataframe()
    print(f"\n📄 Exported to DataFrame: {df.shape}")


def custom_llm_settings():
    """Example 6: Custom LLM settings."""
    print("\n" + "=" * 60)
    print("Example 6: Custom LLM Settings")
    print("=" * 60)

    callback = TokenPilotCallbackHandler(budget_limit=5.00)
    callback_manager = CallbackManager([callback])

    # Custom LLM
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=512
    )

    # Create documents
    documents = [
        Document(text="Custom LLM settings example."),
    ]

    # Build index with custom LLM
    index = VectorStoreIndex.from_documents(
        documents,
        callback_manager=callback_manager,
        llm=llm
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("Test query")

    print(f"\n✅ Query completed")
    print(f"Cost: ${callback.get_total_cost():.4f}")
    print(f"Model: gpt-4o-mini")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "TOKENPILOT - BASIC LLAMAINDEX EXAMPLES" + " " * 9 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run all examples
    simple_indexing()
    query_engine()
    chat_engine()
    multi_user_tracking()
    with_analytics()
    custom_llm_settings()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nKey Points:")
    print("  • Use TokenPilotCallbackHandler for LlamaIndex")
    print("  • Create CallbackManager with the callback")
    print("  • Pass callback_manager to index/query/chat engines")
    print("  • All analytics features work the same")
    print("\nNext Steps:")
    print("  1. Explore advanced LlamaIndex features")
    print("  2. Try waste analysis with your data")
    print("  3. Implement budget forecasting")
    print()


if __name__ == "__main__":
    main()
