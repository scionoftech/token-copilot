"""
Basic LangChain Example - Getting Started with Token Copilot

This example demonstrates the simplest way to start tracking LLM costs
in a LangChain application.
"""

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from token_copilot import TokenPilotCallback


def simple_tracking():
    """Most basic example - just track costs."""
    print("=" * 60)
    print("Example 1: Simple Cost Tracking")
    print("=" * 60)

    # Create callback - that's it!
    callback = TokenPilotCallback()

    # Use with any LangChain LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        callbacks=[callback]
    )

    # Make some calls
    response1 = llm.invoke("What is Python?")
    response2 = llm.invoke("What is machine learning?")

    # Get total cost
    print(f"\n✅ Total cost: ${callback.get_total_cost():.4f}")
    print(f"✅ Total tokens: {callback.get_total_tokens():,}")
    print(f"✅ Number of calls: {len(callback.tracker._entries)}")


def with_budget_limit():
    """Example with budget limit."""
    print("\n" + "=" * 60)
    print("Example 2: Budget Limit")
    print("=" * 60)

    # Set a budget limit
    callback = TokenPilotCallback(budget_limit=0.10)  # $0.10 limit

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        callbacks=[callback]
    )

    # Make calls
    response1 = llm.invoke("Explain quantum computing")
    print(f"\nAfter first call:")
    print(f"  Cost: ${callback.get_total_cost():.4f}")
    print(f"  Remaining: ${callback.get_remaining_budget():.4f}")

    response2 = llm.invoke("Explain neural networks")
    print(f"\nAfter second call:")
    print(f"  Cost: ${callback.get_total_cost():.4f}")
    print(f"  Remaining: ${callback.get_remaining_budget():.4f}")

    # Try to exceed budget (will raise error by default)
    try:
        for i in range(100):
            llm.invoke(f"Question {i}")
    except Exception as e:
        print(f"\n⚠️ Budget exceeded: {e}")


def with_chain():
    """Example using LangChain chains."""
    print("\n" + "=" * 60)
    print("Example 3: With LangChain Chains")
    print("=" * 60)

    callback = TokenPilotCallback(budget_limit=1.00)

    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in one sentence."
    )

    # Create chain
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain multiple times
    topics = ["Python", "JavaScript", "Rust", "Go", "TypeScript"]

    for topic in topics:
        result = chain.invoke({"topic": topic})
        print(f"\n{topic}: {result['text']}")

    # Get statistics
    print("\n" + "-" * 60)
    stats = callback.get_stats()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"Average cost per call: ${stats['avg_cost_per_call']:.4f}")
    print(f"Average tokens per call: {stats['avg_tokens_per_call']:.1f}")


def with_different_models():
    """Track costs across different models."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Models")
    print("=" * 60)

    # Single callback tracks all models
    callback = TokenPilotCallback(budget_limit=5.00)

    # Create different model instances
    gpt4_mini = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    gpt4 = ChatOpenAI(model="gpt-4o", callbacks=[callback])

    # Use different models
    print("\nUsing GPT-4o-mini (cheaper):")
    response1 = gpt4_mini.invoke("What is 2+2?")
    cost1 = callback.get_total_cost()
    print(f"  Cost: ${cost1:.4f}")

    print("\nUsing GPT-4o (more expensive):")
    response2 = gpt4.invoke("What is 2+2?")
    cost2 = callback.get_total_cost() - cost1
    print(f"  Cost: ${cost2:.4f}")

    # Compare costs by model
    print("\n" + "-" * 60)
    print("Costs by model:")
    costs_by_model = callback.get_costs_by_model()
    for model, cost in costs_by_model.items():
        print(f"  {model}: ${cost:.4f}")


def export_to_dataframe():
    """Export tracking data to pandas DataFrame."""
    print("\n" + "=" * 60)
    print("Example 5: Export to DataFrame")
    print("=" * 60)

    callback = TokenPilotCallback()

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Make several calls
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]

    for prompt in prompts:
        llm.invoke(prompt)

    # Export to DataFrame
    df = callback.to_dataframe()

    print("\nDataFrame shape:", df.shape)
    print("\nDataFrame columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df[['model', 'input_tokens', 'output_tokens', 'cost']].head())

    print("\nSummary statistics:")
    print(df[['input_tokens', 'output_tokens', 'cost']].describe())


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "TOKEN COPILOT - BASIC LANGCHAIN EXAMPLES" + " " * 9 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run all examples
    simple_tracking()
    with_budget_limit()
    with_chain()
    with_different_models()
    export_to_dataframe()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Try budget_enforcement.py for budget management")
    print("  2. Try multi_tenant.py for per-user tracking")
    print("  3. Explore analytics examples for deeper insights")
    print()


if __name__ == "__main__":
    main()
