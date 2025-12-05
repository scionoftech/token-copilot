"""
Waste Analysis Example - Identify Cost Waste

This example demonstrates how to identify token waste patterns:
- Repeated prompts (system prompts sent multiple times)
- Excessive context windows (top 10% of input tokens)
- Verbose outputs (top 15% of output tokens)
- Actionable recommendations with savings estimates
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
import time


def simulate_usage_with_waste():
    """Simulate LLM usage with various waste patterns."""
    callback = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    print("Simulating LLM usage with waste patterns...")
    print("(This will make actual API calls)\n")

    # Pattern 1: Repeated prompts (waste!)
    print("1. Repeated system prompts:")
    repeated_prompt = "You are a helpful assistant. Please be concise and accurate."
    for i in range(5):
        llm.invoke(f"{repeated_prompt}\n\nUser: What is {i}?")
        print(f"   Call {i+1}: Same system prompt sent again")

    # Pattern 2: Excessive context
    print("\n2. Excessive context windows:")
    long_context = "Context: " + " ".join([f"Info{i}" for i in range(500)])
    for i in range(3):
        llm.invoke(f"{long_context}\n\nQuestion: What is {i}?")
        print(f"   Call {i+1}: Large context window")

    # Pattern 3: Verbose outputs
    print("\n3. Requesting verbose outputs:")
    for i in range(3):
        llm.invoke(
            f"Explain quantum computing in great detail with many examples and "
            f"analogies. Be very thorough and comprehensive. Question {i}."
        )
        print(f"   Call {i+1}: Requesting verbose response")

    # Some efficient calls for comparison
    print("\n4. Efficient calls (for comparison):")
    for i in range(5):
        llm.invoke(f"What is {i} + {i}?")
        print(f"   Call {i+1}: Simple, efficient prompt")

    return callback


def analyze_waste_patterns(callback):
    """Analyze and display waste patterns."""
    print("\n" + "=" * 60)
    print("WASTE ANALYSIS REPORT")
    print("=" * 60)

    # Run waste analysis
    report = callback.analyze_waste()

    # Summary
    summary = report['summary']
    print("\n📊 SUMMARY")
    print("-" * 60)
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Total waste: ${summary['total_waste_cost']:.4f} ({summary['waste_percentage']:.1f}%)")
    print(f"Potential monthly savings: ${summary['monthly_savings']:.2f}")

    # Detailed breakdown
    print("\n🔍 WASTE BREAKDOWN BY CATEGORY")
    print("-" * 60)

    categories = report['categories']

    if 'repeated_prompts' in categories:
        cat = categories['repeated_prompts']
        print(f"\n1. Repeated Prompts:")
        print(f"   • Waste cost: ${cat['waste_cost']:.4f}")
        print(f"   • Waste tokens: {cat['waste_tokens']:,}")
        print(f"   • Instances: {cat['instances']}")
        print(f"   • Percentage: {cat['percentage']:.1f}%")
        print(f"   • Monthly savings: ${cat['monthly_savings']:.2f}")
        print(f"   💡 {cat['recommendation']}")

    if 'excessive_context' in categories:
        cat = categories['excessive_context']
        print(f"\n2. Excessive Context Windows:")
        print(f"   • Waste cost: ${cat['waste_cost']:.4f}")
        print(f"   • Waste tokens: {cat['waste_tokens']:,}")
        print(f"   • Instances: {cat['instances']}")
        print(f"   • Percentage: {cat['percentage']:.1f}%")
        print(f"   • Monthly savings: ${cat['monthly_savings']:.2f}")
        print(f"   💡 {cat['recommendation']}")

    if 'verbose_outputs' in categories:
        cat = categories['verbose_outputs']
        print(f"\n3. Verbose Outputs:")
        print(f"   • Waste cost: ${cat['waste_cost']:.4f}")
        print(f"   • Waste tokens: {cat['waste_tokens']:,}")
        print(f"   • Instances: {cat['instances']}")
        print(f"   • Percentage: {cat['percentage']:.1f}%")
        print(f"   • Monthly savings: ${cat['monthly_savings']:.2f}")
        print(f"   💡 {cat['recommendation']}")

    # Recommendations
    print("\n💡 ACTIONABLE RECOMMENDATIONS")
    print("-" * 60)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")

    # Savings projection
    print("\n💰 POTENTIAL SAVINGS")
    print("-" * 60)
    print(f"Monthly: ${summary['monthly_savings']:.2f}")
    print(f"Yearly: ${summary['monthly_savings'] * 12:.2f}")


def optimization_examples():
    """Show how to optimize based on waste analysis."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION EXAMPLES")
    print("=" * 60)

    # Example 1: Cache system prompts
    print("\n1. BEFORE: Repeated system prompt")
    callback_before = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_before])

    system_prompt = "You are a helpful assistant."
    for i in range(3):
        llm.invoke(f"{system_prompt}\n\nUser: Question {i}")

    cost_before = callback_before.get_total_cost()
    print(f"   Cost: ${cost_before:.4f}")

    # Example 1: AFTER - Using conversation memory (more efficient)
    print("\n   AFTER: Using message history")
    callback_after = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_after])

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [SystemMessage(content=system_prompt)]
    for i in range(3):
        messages.append(HumanMessage(content=f"Question {i}"))
        llm.invoke(messages)

    cost_after = callback_after.get_total_cost()
    savings = cost_before - cost_after
    print(f"   Cost: ${cost_after:.4f}")
    print(f"   Savings: ${savings:.4f} ({savings/cost_before*100:.1f}%)")

    # Example 2: Reduce context
    print("\n2. BEFORE: Excessive context")
    callback_before2 = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_before2])

    long_context = "Context: " + " ".join([f"Info{i}" for i in range(500)])
    llm.invoke(f"{long_context}\n\nQuestion: What is 2+2?")

    cost_before2 = callback_before2.get_total_cost()
    print(f"   Cost: ${cost_before2:.4f}")

    print("\n   AFTER: Concise context")
    callback_after2 = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_after2])

    short_context = "Context: Relevant info only"
    llm.invoke(f"{short_context}\n\nQuestion: What is 2+2?")

    cost_after2 = callback_after2.get_total_cost()
    savings2 = cost_before2 - cost_after2
    print(f"   Cost: ${cost_after2:.4f}")
    print(f"   Savings: ${savings2:.4f} ({savings2/cost_before2*100:.1f}%)")


def continuous_monitoring():
    """Example of continuous waste monitoring."""
    print("\n" + "=" * 60)
    print("CONTINUOUS MONITORING")
    print("=" * 60)

    callback = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Simulate periodic usage
    print("\nMaking LLM calls...")

    for batch in range(3):
        print(f"\nBatch {batch + 1}:")

        # Make some calls
        for i in range(5):
            llm.invoke(f"Question {batch * 5 + i}")

        # Check waste periodically
        report = callback.analyze_waste()
        total_waste = report['summary']['total_waste_cost']
        waste_pct = report['summary']['waste_percentage']

        print(f"  Total waste: ${total_waste:.4f} ({waste_pct:.1f}%)")

        # Alert if waste exceeds threshold
        if waste_pct > 15:
            print(f"  ⚠️ WARNING: Waste exceeds 15%!")
            print(f"  Top recommendation: {report['recommendations'][0]}")


def export_waste_data():
    """Export waste data for further analysis."""
    print("\n" + "=" * 60)
    print("EXPORT WASTE DATA")
    print("=" * 60)

    callback = TokenCoPilotCallback()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Generate usage data
    for i in range(10):
        llm.invoke(f"Question {i}")

    # Export to DataFrame
    df = callback.to_dataframe()

    # Analyze in pandas
    print("\n📊 DataFrame Analysis:")
    print(f"Total calls: {len(df)}")
    print(f"\nTop 5 by input tokens (potential excessive context):")
    print(df.nlargest(5, 'input_tokens')[['input_tokens', 'output_tokens', 'cost']])

    print(f"\nTop 5 by output tokens (potential verbose outputs):")
    print(df.nlargest(5, 'output_tokens')[['input_tokens', 'output_tokens', 'cost']])

    # Save to CSV for further analysis
    # df.to_csv('waste_analysis.csv', index=False)
    # print("\n💾 Exported to waste_analysis.csv")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 16 + "WASTE ANALYSIS EXAMPLES" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Simulate usage and analyze
    callback = simulate_usage_with_waste()
    analyze_waste_patterns(callback)
    optimization_examples()
    continuous_monitoring()
    export_waste_data()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nKey Insights:")
    print("  • Repeated prompts: Cache or use conversation memory")
    print("  • Excessive context: Summarize or filter relevant info")
    print("  • Verbose outputs: Use concise instructions")
    print("  • Regular monitoring: Run analyze_waste() periodically")
    print("\nPotential Savings:")
    print("  • Reducing waste by 20% can save significant costs")
    print("  • Monthly savings compound over time")
    print()


if __name__ == "__main__":
    main()
