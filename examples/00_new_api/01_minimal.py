"""Minimal example - simplest way to track LLM costs."""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilot


def main():
    """Demonstrate minimal usage."""
    print("=" * 60)
    print("Minimal Example - Just Track Costs")
    print("=" * 60)

    # Create copilot - that's it!
    copilot = TokenCoPilot(budget_limit=1.00)

    # Use with any LangChain LLM
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

    # Make calls
    print("\nMaking LLM calls...")
    result1 = llm.invoke("Say hello in 5 words")
    print(f"Response: {result1.content}")

    result2 = llm.invoke("What is 2+2?")
    print(f"Response: {result2.content}")

    # Get stats - clean, simple properties
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total cost: ${copilot.cost:.4f}")
    print(f"Total tokens: {copilot.tokens}")
    print(f"Remaining budget: ${copilot.get_remaining_budget():.2f}")

    # Get detailed stats
    stats = copilot.get_stats()
    print(f"\nTotal calls: {stats['total_calls']}")
    print(f"Average cost per call: ${stats['avg_cost_per_call']:.4f}")


if __name__ == "__main__":
    main()
