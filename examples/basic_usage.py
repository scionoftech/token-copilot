"""Basic usage example for token_copilot with LangChain."""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from token_copilot import TokenCoPilotCallback

# Make sure you have OPENAI_API_KEY set
if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)


def example_basic():
    """Basic cost tracking."""
    print("=== Example 1: Basic Cost Tracking ===\n")

    # Create callback
    callback = TokenCoPilotCallback()

    # Create LangChain components using LCEL
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question briefly: {question}"
    )
    chain = prompt | llm

    # Make some calls
    questions = [
        "What is Python?",
        "What is LangChain?",
        "What is machine learning?"
    ]

    for q in questions:
        result = chain.invoke({"question": q})
        content = result.content if hasattr(result, 'content') else str(result)
        print(f"Q: {q}")
        print(f"A: {content}\n")

    # Get stats
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    print(f"Total tokens: {callback.get_total_tokens():,}")
    print(f"Total calls: {callback.get_stats()['total_calls']}\n")


def example_budget_enforcement():
    """Budget enforcement example."""
    print("=== Example 2: Budget Enforcement ===\n")

    # Create callback with $0.01 budget limit
    callback = TokenCoPilotCallback(budget_limit=0.01)

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    chain = prompt | llm

    try:
        # This should work
        result = chain.invoke({"question": "Say hello"})
        content = result.content if hasattr(result, 'content') else str(result)
        print(f"First call: {content}")
        print(f"Remaining: ${callback.get_remaining_budget():.4f}\n")

        # This might exceed budget
        result = chain.invoke({"question": "Write a long story about AI" * 100})
        content = result.content if hasattr(result, 'content') else str(result)
        print(f"Second call: {content}")

    except Exception as e:
        print(f"Budget exceeded: {e}\n")
        print(f"Total spent: ${callback.get_total_cost():.4f}")


def example_multi_tenant():
    """Multi-tenant cost tracking."""
    print("=== Example 3: Multi-Tenant Tracking ===\n")

    callback = TokenCoPilotCallback()

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    chain = prompt | llm

    # Track calls for different users
    users = [
        {"user_id": "alice", "org_id": "acme_corp"},
        {"user_id": "bob", "org_id": "acme_corp"},
        {"user_id": "charlie", "org_id": "beta_inc"},
    ]

    for user in users:
        result = chain.invoke(
            {"question": "Hello, how are you?"},
            config={"metadata": user}
        )
        content = result.content if hasattr(result, 'content') else str(result)
        print(f"{user['user_id']}: {content[:50]}...")

    # Get analytics
    print("\nCosts by user:")
    for user_id, cost in callback.get_costs_by_user().items():
        print(f"  {user_id}: ${cost:.4f}")

    print("\nCosts by organization:")
    for org_id, cost in callback.get_costs_by_org().items():
        print(f"  {org_id}: ${cost:.4f}")


def example_pandas_analytics():
    """Pandas DataFrame analytics."""
    print("=== Example 4: Pandas Analytics ===\n")

    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")
        return

    callback = TokenCoPilotCallback()

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    chain = prompt | llm

    # Make calls with metadata
    for i in range(5):
        chain.invoke(
            {"question": f"Question {i}"},
            config={"metadata": {
                "user_id": f"user_{i % 2}",  # 2 users
                "feature": "chat" if i % 2 == 0 else "summarize"
            }}
        )

    # Export to DataFrame
    df = callback.to_dataframe()

    print("DataFrame shape:", df.shape)
    print("\nDataFrame columns:", list(df.columns))
    print("\nCost by user:")
    print(df.groupby('user_id')['cost'].sum())
    print("\nCost by feature:")
    print(df.groupby('feature')['cost'].sum())
    print("\nTotal tokens by user:")
    print(df.groupby('user_id')['total_tokens'].sum())


if __name__ == "__main__":
    print("\nToken Copilot Examples\n" + "="*60 + "\n")

    # Run examples (comment out ones you don't want to run)
    example_basic()
    # example_budget_enforcement()  # Uncomment to test budget limits
    # example_multi_tenant()  # Uncomment for multi-tenant demo
    # example_pandas_analytics()  # Uncomment for pandas analytics

    print("\nAll examples completed!")
