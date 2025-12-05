"""Example using @budget_gate decorator to prevent expensive operations.

This example demonstrates how to gate function execution based on minimum
budget tier requirements, preventing expensive operations when budget is low.
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
from token_copilot.adaptive import (
    budget_gate,
    adaptive_context,
    BudgetTier,
    classify_budget_tier,
)


@budget_gate(min_tier=BudgetTier.COMFORTABLE, skip_on_insufficient=True)
def deep_analysis(llm, text):
    """Perform deep analysis - only runs with COMFORTABLE or better budget.

    This is an expensive operation that uses many tokens. It should only
    run when we have sufficient budget remaining.
    """
    print("  [DEEP ANALYSIS] Running expensive deep analysis...")
    prompt = f"""
    Provide a comprehensive, detailed analysis of the following text.
    Include: themes, tone, key points, implications, and recommendations.

    Text: {text}
    """
    response = llm.invoke(prompt, max_tokens=2000)
    return response.content


@budget_gate(min_tier=BudgetTier.MODERATE, skip_on_insufficient=True)
def moderate_task(llm, text):
    """Moderate cost task - requires MODERATE tier or better."""
    print("  [MODERATE TASK] Running moderate complexity task...")
    response = llm.invoke(f"Summarize: {text}", max_tokens=500)
    return response.content


@budget_gate(min_tier=BudgetTier.LOW, skip_on_insufficient=True)
def quick_task(llm, text):
    """Low cost task - runs even with LOW budget."""
    print("  [QUICK TASK] Running quick, low-cost task...")
    response = llm.invoke(f"Extract main topic from: {text}", max_tokens=50)
    return response.content


def always_runs(llm, text):
    """Basic task that always runs regardless of budget."""
    print("  [BASIC TASK] Running basic task (no budget gate)...")
    response = llm.invoke(f"One word topic: {text}", max_tokens=10)
    return response.content


@budget_gate(
    min_tier=BudgetTier.COMFORTABLE,
    skip_on_insufficient=False,
    raise_on_insufficient=True
)
def critical_expensive_task(llm, text):
    """Critical task that raises exception if budget insufficient.

    Use this pattern when you need to know if the task was skipped due to budget.
    """
    print("  [CRITICAL TASK] Running critical expensive task...")
    response = llm.invoke(f"Detailed analysis: {text}", max_tokens=1000)
    return response.content


def run_pipeline(llm, text, tier_name):
    """Run processing pipeline - tasks automatically gated by budget."""
    print(f"\nRunning pipeline at {tier_name} tier:")
    print("-" * 40)

    results = {}

    # Try each task - gates will allow/block based on tier
    results['deep'] = deep_analysis(llm, text)
    results['moderate'] = moderate_task(llm, text)
    results['quick'] = quick_task(llm, text)
    results['basic'] = always_runs(llm, text)

    # Show what ran
    print("\nResults:")
    for task, result in results.items():
        if result is None:
            print(f"  {task}: SKIPPED (insufficient budget)")
        else:
            print(f"  {task}: {result[:60]}...")

    return results


def main():
    """Demonstrate budget gating."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    sample_text = "Climate change is one of the most pressing challenges facing humanity."

    print("=" * 60)
    print("Budget Gating Example")
    print("=" * 60)
    print("\nDemonstrating how @budget_gate prevents expensive operations")
    print("when budget is insufficient.\n")

    # Scenario 1: ABUNDANT budget - everything runs
    print("\n" + "=" * 60)
    print("Scenario 1: ABUNDANT Budget (>75% remaining)")
    print("=" * 60)
    callback1 = TokenCoPilotCallback(budget_limit=10.00)
    with adaptive_context(callback1):
        tier = classify_budget_tier(callback1)
        print(f"Budget: ${callback1.budget_limit:.2f}, Tier: {tier.value.upper()}")
        run_pipeline(llm, sample_text, tier.value)

    # Scenario 2: MODERATE budget - deep analysis blocked
    print("\n" + "=" * 60)
    print("Scenario 2: MODERATE Budget (25-50% remaining)")
    print("=" * 60)
    callback2 = TokenCoPilotCallback(budget_limit=10.00)
    # Simulate spending to reach MODERATE tier
    callback2.tracker.track_cost("gpt-4o-mini", 30000, 20000, 6.00)  # ~40% remaining
    with adaptive_context(callback2):
        tier = classify_budget_tier(callback2)
        remaining = callback2.budget_limit - callback2.get_total_cost()
        print(f"Budget: ${remaining:.2f} / ${callback2.budget_limit:.2f}, Tier: {tier.value.upper()}")
        run_pipeline(llm, sample_text, tier.value)

    # Scenario 3: CRITICAL budget - only basic task runs
    print("\n" + "=" * 60)
    print("Scenario 3: CRITICAL Budget (<10% remaining)")
    print("=" * 60)
    callback3 = TokenCoPilotCallback(budget_limit=10.00)
    # Simulate heavy spending to reach CRITICAL tier
    callback3.tracker.track_cost("gpt-4o-mini", 50000, 40000, 9.50)  # ~5% remaining
    with adaptive_context(callback3):
        tier = classify_budget_tier(callback3)
        remaining = callback3.budget_limit - callback3.get_total_cost()
        print(f"Budget: ${remaining:.2f} / ${callback3.budget_limit:.2f}, Tier: {tier.value.upper()}")
        run_pipeline(llm, sample_text, tier.value)

    # Scenario 4: Exception on insufficient budget
    print("\n" + "=" * 60)
    print("Scenario 4: Raising Exception on Insufficient Budget")
    print("=" * 60)
    callback4 = TokenCoPilotCallback(budget_limit=10.00)
    callback4.tracker.track_cost("gpt-4o-mini", 50000, 40000, 9.50)
    with adaptive_context(callback4):
        tier = classify_budget_tier(callback4)
        print(f"Tier: {tier.value.upper()}")
        print("Attempting critical task that requires COMFORTABLE tier...")
        try:
            critical_expensive_task(llm, sample_text)
            print("Task completed successfully")
        except RuntimeError as e:
            print(f"Task blocked: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nBudget gates provide automatic cost control:")
    print("- Expensive operations blocked when budget is low")
    print("- Configure minimum tier per function")
    print("- Choose: skip silently OR raise exception")
    print("- Prevents budget overruns from expensive operations")
    print("- No manual budget checking required!")


if __name__ == "__main__":
    main()
