"""Complete workflow example combining all adaptive features.

This example demonstrates a realistic workflow using:
- TokenAwareOperations for adaptive generation
- @token_aware decorator for custom functions
- @budget_gate for expensive operations
- budget_aware_section for tracking
- Adaptive retry logic
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
from token_copilot.adaptive import (
    TokenAwareOperations,
    token_aware,
    budget_gate,
    track_efficiency,
    adaptive_context,
    budget_aware_section,
    BudgetTier,
)


@token_aware(operation='generate')
def extract_entities(llm, text, max_tokens=None):
    """Extract named entities from text with adaptive parameters."""
    prompt = f"Extract all named entities (people, places, organizations) from: {text}"
    response = llm.invoke(prompt, max_tokens=max_tokens)
    return response.content


@budget_gate(min_tier=BudgetTier.COMFORTABLE)
def sentiment_analysis(llm, text):
    """Analyze sentiment - only runs with sufficient budget."""
    response = llm.invoke(
        f"Analyze sentiment (positive/negative/neutral): {text}",
        max_tokens=100
    )
    return response.content


@track_efficiency(metric_name="topic_modeling")
def identify_topics(llm, text):
    """Identify main topics with efficiency tracking."""
    response = llm.invoke(f"Identify 3 main topics from: {text}", max_tokens=150)
    return response.content


def process_document_workflow(callback, llm, document):
    """Complete document processing workflow with adaptive behavior."""
    ops = TokenAwareOperations(callback)

    print("\n" + "=" * 60)
    print("Document Processing Workflow")
    print("=" * 60)

    # Section 1: Entity Extraction
    with budget_aware_section(callback, "entity_extraction") as section:
        print("\n[1/4] Extracting entities...")
        tier_info = ops.get_tier_info()
        print(f"  Budget tier: {tier_info['tier_name'].upper()}")
        print(f"  Remaining: ${tier_info['remaining']:.2f}")

        entities = extract_entities(llm, document)
        print(f"  Entities: {entities[:80]}...")
        print(f"  Section cost: ${section['cost_delta']:.4f}")

    # Section 2: Topic Identification
    with budget_aware_section(callback, "topic_identification") as section:
        print("\n[2/4] Identifying topics...")
        tier_info = ops.get_tier_info()
        print(f"  Budget tier: {tier_info['tier_name'].upper()}")
        print(f"  Remaining: ${tier_info['remaining']:.2f}")

        topics = identify_topics(llm, document)
        print(f"  Topics: {topics[:80]}...")
        print(f"  Section cost: ${section['cost_delta']:.4f}")

    # Section 3: Sentiment Analysis (may be gated)
    with budget_aware_section(callback, "sentiment_analysis") as section:
        print("\n[3/4] Analyzing sentiment...")
        tier_info = ops.get_tier_info()
        print(f"  Budget tier: {tier_info['tier_name'].upper()}")
        print(f"  Remaining: ${tier_info['remaining']:.2f}")

        sentiment = sentiment_analysis(llm, document)
        if sentiment is None:
            print("  Sentiment: SKIPPED (insufficient budget)")
        else:
            print(f"  Sentiment: {sentiment}")
            print(f"  Section cost: ${section['cost_delta']:.4f}")

    # Section 4: Summary with adaptive retry
    with budget_aware_section(callback, "summary_generation") as section:
        print("\n[4/4] Generating summary...")
        tier_info = ops.get_tier_info()
        print(f"  Budget tier: {tier_info['tier_name'].upper()}")
        print(f"  Remaining: ${tier_info['remaining']:.2f}")

        # Use adaptive retry - fewer retries when budget is low
        def generate_summary():
            return ops.generate(llm, f"Summarize concisely: {document}")

        summary = ops.retry(generate_summary)
        print(f"  Summary: {summary.content[:80]}...")
        print(f"  Section cost: ${section['cost_delta']:.4f}")

    # Return results
    return {
        'entities': entities,
        'topics': topics,
        'sentiment': sentiment,
        'summary': summary.content if hasattr(summary, 'content') else str(summary),
    }


def main():
    """Run complete adaptive workflow."""
    # Sample document
    document = """
    Microsoft Corporation announced today that it has successfully completed
    the acquisition of GitHub, the world's leading software development platform.
    CEO Satya Nadella expressed excitement about the merger, stating that it
    represents a significant milestone for both companies. The tech giant plans
    to invest heavily in GitHub's infrastructure and expand its capabilities.
    Industry analysts view this acquisition positively, predicting increased
    innovation in the developer tools space.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    print("=" * 60)
    print("Complete Adaptive Workflow Example")
    print("=" * 60)
    print("\nDemonstrating realistic workflow with multiple adaptive features")

    # Scenario 1: Abundant budget - all features run
    print("\n" + "=" * 60)
    print("SCENARIO 1: Abundant Budget ($10.00)")
    print("=" * 60)
    callback1 = TokenCoPilotCallback(budget_limit=10.00)
    with adaptive_context(callback1):
        results1 = process_document_workflow(callback1, llm, document)

    print("\n--- Scenario 1 Summary ---")
    print(f"Total cost: ${callback1.get_total_cost():.4f}")
    print(f"Budget used: {(callback1.get_total_cost() / 10.00 * 100):.1f}%")
    print("All operations completed successfully!")

    # Scenario 2: Limited budget - some features gated
    print("\n\n" + "=" * 60)
    print("SCENARIO 2: Limited Budget ($2.00)")
    print("=" * 60)
    callback2 = TokenCoPilotCallback(budget_limit=2.00)
    # Pre-spend to get to MODERATE tier
    callback2.tracker.track_cost("gpt-4o-mini", 10000, 8000, 1.00)

    with adaptive_context(callback2):
        results2 = process_document_workflow(callback2, llm, document)

    print("\n--- Scenario 2 Summary ---")
    print(f"Total cost: ${callback2.get_total_cost():.4f}")
    print(f"Budget used: {(callback2.get_total_cost() / 2.00 * 100):.1f}%")
    print("Adaptive behavior:")
    print("  - Lower max_tokens used automatically")
    print("  - Sentiment analysis may be skipped due to budget gate")
    print("  - Fewer retry attempts to conserve budget")

    # Scenario 3: Critical budget - minimal operations
    print("\n\n" + "=" * 60)
    print("SCENARIO 3: Critical Budget ($1.00, mostly spent)")
    print("=" * 60)
    callback3 = TokenCoPilotCallback(budget_limit=1.00)
    # Pre-spend to get to CRITICAL tier
    callback3.tracker.track_cost("gpt-4o-mini", 15000, 10000, 0.92)

    with adaptive_context(callback3):
        results3 = process_document_workflow(callback3, llm, document)

    print("\n--- Scenario 3 Summary ---")
    print(f"Total cost: ${callback3.get_total_cost():.4f}")
    print(f"Budget used: {(callback3.get_total_cost() / 1.00 * 100):.1f}%")
    print("Critical budget behavior:")
    print("  - Minimal max_tokens (200) used")
    print("  - Low temperature (0.1) for determinism")
    print("  - No retries (max_retries=0)")
    print("  - Expensive operations gated/skipped")

    # Final Summary
    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("1. TokenAwareOperations - adaptive generation parameters")
    print("2. @token_aware decorator - declarative adaptive functions")
    print("3. @budget_gate - prevent expensive ops when budget low")
    print("4. @track_efficiency - automatic efficiency logging")
    print("5. budget_aware_section - track cost by workflow section")
    print("6. Adaptive retry - fewer retries when budget constrained")
    print("\nBenefits:")
    print("- Automatic optimization based on budget")
    print("- No manual parameter tuning required")
    print("- Prevents budget overruns")
    print("- Transparent logging of adaptive decisions")
    print("- User parameters always override defaults")


if __name__ == "__main__":
    main()
