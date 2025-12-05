"""Basic example of adaptive token operations.

This example demonstrates how TokenAwareOperations automatically adjusts
LLM parameters based on remaining budget.
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
from token_copilot.adaptive import TokenAwareOperations


def main():
    """Demonstrate basic adaptive operations."""
    # Create callback with budget
    callback = TokenCoPilotCallback(budget_limit=10.00)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create adaptive operations wrapper
    adaptive = TokenAwareOperations(callback)

    print("=" * 60)
    print("Adaptive Token Operations - Basic Example")
    print("=" * 60)

    # Phase 1: Abundant budget (>75%)
    print("\n[Phase 1: ABUNDANT Budget - Premium Settings]")
    tier_info = adaptive.get_tier_info()
    print(f"Budget: ${tier_info['remaining']:.2f} / ${tier_info['budget_limit']:.2f}")
    print(f"Tier: {tier_info['tier_name'].upper()}")
    print(f"Description: {tier_info['description']}\n")

    response1 = adaptive.generate(
        llm,
        "Explain quantum computing in simple terms"
    )
    print(f"Response 1: {response1.content[:100]}...\n")

    # Phase 2: Simulate spending to reach MODERATE tier
    print("\n[Phase 2: Simulating Budget Usage...]")
    # Manually track some cost to demonstrate tier changes
    callback.tracker.track_cost(
        model="gpt-4o-mini",
        input_tokens=20000,
        output_tokens=15000,
        cost=5.50  # Spent $5.50, now at ~45% remaining
    )

    tier_info = adaptive.get_tier_info()
    print(f"Budget: ${tier_info['remaining']:.2f} / ${tier_info['budget_limit']:.2f}")
    print(f"Tier: {tier_info['tier_name'].upper()}")
    print(f"Description: {tier_info['description']}\n")

    response2 = adaptive.generate(
        llm,
        "What is machine learning?"
    )
    print(f"Response 2: {response2.content[:100]}...\n")

    # Phase 3: Simulate spending to reach LOW tier
    print("\n[Phase 3: Further Budget Usage...]")
    callback.tracker.track_cost(
        model="gpt-4o-mini",
        input_tokens=30000,
        output_tokens=20000,
        cost=3.00  # Total spent: $8.50, now at ~15% remaining
    )

    tier_info = adaptive.get_tier_info()
    print(f"Budget: ${tier_info['remaining']:.2f} / ${tier_info['budget_limit']:.2f}")
    print(f"Tier: {tier_info['tier_name'].upper()}")
    print(f"Description: {tier_info['description']}\n")

    response3 = adaptive.generate(
        llm,
        "Define AI"
    )
    print(f"Response 3: {response3.content[:100]}...\n")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    final_cost = callback.get_total_cost()
    print(f"Total Cost: ${final_cost:.4f}")
    print(f"Budget Used: {(final_cost / tier_info['budget_limit'] * 100):.1f}%")
    print(f"Remaining: ${tier_info['budget_limit'] - final_cost:.4f}")

    print("\nKey Observations:")
    print("- ABUNDANT tier: Used premium settings (max_tokens=2000, temp=0.7)")
    print("- MODERATE tier: Started optimizing (max_tokens=1000, temp=0.5)")
    print("- LOW tier: Aggressive optimization (max_tokens=500, temp=0.3)")
    print("- Parameters automatically adjusted based on budget!")


if __name__ == "__main__":
    main()
