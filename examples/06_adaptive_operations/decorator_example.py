"""Example using @token_aware decorator for adaptive behavior.

This example shows how to use decorators to make your functions automatically
adapt to budget constraints without explicit parameter passing.
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
from token_copilot.adaptive import token_aware, adaptive_context, track_efficiency


# Decorated function - automatically receives adaptive parameters
@token_aware(operation='generate')
def summarize_text(llm, text, max_tokens=None, temperature=None):
    """Summarize text with adaptive parameters.

    The @token_aware decorator automatically injects appropriate max_tokens
    and temperature based on current budget tier.
    """
    prompt = f"Summarize the following text concisely:\n\n{text}"
    response = llm.invoke(prompt, max_tokens=max_tokens, temperature=temperature)
    return response.content


@token_aware(operation='generate')
def answer_question(llm, question, max_tokens=None):
    """Answer a question with adaptive parameters."""
    response = llm.invoke(question, max_tokens=max_tokens)
    return response.content


@track_efficiency(metric_name="document_processing")
def process_document(llm, document):
    """Process a document and track efficiency metrics."""
    # Decorated with @track_efficiency to automatically log tokens and cost
    summary = llm.invoke(f"Summarize: {document}")
    keywords = llm.invoke(f"Extract keywords from: {document}")
    return summary.content, keywords.content


def main():
    """Demonstrate decorator-based adaptive operations."""
    # Create callback and LLM
    callback = TokenCoPilotCallback(budget_limit=5.00)
    llm = ChatOpenAI(model="gpt-4o-mini")

    print("=" * 60)
    print("Adaptive Operations - Decorator Example")
    print("=" * 60)

    sample_text = """
    Artificial Intelligence (AI) has revolutionized many industries.
    Machine learning, a subset of AI, enables computers to learn from data
    without being explicitly programmed. Deep learning, using neural networks,
    has achieved remarkable results in image recognition, natural language
    processing, and game playing. The future of AI holds immense potential
    for solving complex problems in healthcare, climate change, and beyond.
    """

    # Use adaptive_context to make callback available to decorators
    with adaptive_context(callback):
        print("\n[Phase 1: ABUNDANT Budget]")
        print(f"Remaining: ${callback.budget_limit - callback.get_total_cost():.2f}")

        # Call decorated function - automatically uses ABUNDANT tier params
        summary1 = summarize_text(llm, sample_text)
        print(f"Summary: {summary1[:80]}...")
        print(f"Cost so far: ${callback.get_total_cost():.4f}\n")

        # Simulate spending
        callback.tracker.track_cost("gpt-4o-mini", 15000, 10000, 3.00)

        print("\n[Phase 2: MODERATE Budget]")
        print(f"Remaining: ${callback.budget_limit - callback.get_total_cost():.2f}")

        # Same function call - now uses MODERATE tier params automatically!
        summary2 = summarize_text(llm, sample_text)
        print(f"Summary: {summary2[:80]}...")
        print(f"Cost so far: ${callback.get_total_cost():.4f}\n")

        # User can still override adaptive params
        print("\n[User Override Example]")
        summary3 = summarize_text(
            llm,
            sample_text,
            max_tokens=100,  # User override - always respected
            temperature=0.9
        )
        print(f"Summary (user override): {summary3[:80]}...")
        print(f"Cost so far: ${callback.get_total_cost():.4f}\n")

        # Track efficiency example
        print("\n[Efficiency Tracking Example]")
        doc = "Short document for processing."
        summary, keywords = process_document(llm, doc)
        print(f"Summary: {summary[:60]}...")
        print(f"Keywords: {keywords[:60]}...")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Cost: ${callback.get_total_cost():.4f}")
    print(f"Budget Remaining: ${callback.budget_limit - callback.get_total_cost():.4f}")

    print("\nKey Observations:")
    print("- Decorators automatically adapt to budget tier")
    print("- Same function call, different parameters based on budget")
    print("- User parameters always override adaptive defaults")
    print("- @track_efficiency logs cost/token metrics automatically")
    print("- Clean, declarative code without explicit parameter management")


if __name__ == "__main__":
    main()
