"""Decorator examples - automatic cost tracking for functions."""

from langchain_openai import ChatOpenAI
from token_copilot.decorators import track_cost, enforce_budget, monitored


@track_cost(budget_limit=2.00)
def summarize_text(text):
    """Function with automatic cost tracking."""
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[summarize_text.copilot])
    response = llm.invoke(f"Summarize in one sentence: {text}")
    return response.content


@enforce_budget(limit=0.50, on_exceeded="warn")
def quick_task(copilot):
    """Function with strict budget enforcement."""
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])
    response = llm.invoke("Say hello in 5 words")
    return response.content


@monitored(name="document_analysis", budget_limit=5.00)
def analyze_document(doc, copilot):
    """Function with automatic monitoring and logging."""
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])
    response = llm.invoke(f"Analyze this document: {doc}")
    return response.content


def main():
    """Demonstrate decorators."""
    print("=" * 60)
    print("Decorators - Automatic Cost Tracking")
    print("=" * 60)

    # Demo @track_cost
    print("\n" + "=" * 60)
    print("@track_cost Decorator")
    print("=" * 60)

    text1 = "Artificial intelligence is transforming industries worldwide."
    text2 = "Machine learning enables computers to learn from data."

    print("\nCalling summarize_text multiple times...")
    summary1 = summarize_text(text1)
    print(f"Summary 1: {summary1}")

    summary2 = summarize_text(text2)
    print(f"Summary 2: {summary2}")

    # Access copilot attached to function
    print(f"\nTotal cost: ${summarize_text.copilot.cost:.4f}")
    print(f"Total tokens: {summarize_text.copilot.tokens}")

    # Demo @enforce_budget
    print("\n" + "=" * 60)
    print("@enforce_budget Decorator")
    print("=" * 60)

    print("\nCalling quick_task (budget: $0.50)...")
    result = quick_task()
    print(f"Result: {result}")
    print(f"Cost: ${quick_task.copilot.cost:.4f}")

    # Demo @monitored
    print("\n" + "=" * 60)
    print("@monitored Decorator")
    print("=" * 60)

    docs = [
        "Document about AI and machine learning",
        "Document about natural language processing",
    ]

    print("\nAnalyzing documents...")
    for doc in docs:
        analysis = analyze_document(doc)
        print(f"Analysis: {analysis[:60]}...")

    print(f"\nTotal cost: ${analyze_document.copilot.cost:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nDecorators provide function-level cost tracking:")
    print("  - @track_cost: Attach copilot to function")
    print("  - @enforce_budget: Strict budget limits")
    print("  - @monitored: Automatic logging")
    print("\nBenefits:")
    print("  - Clean separation of concerns")
    print("  - Reusable across function calls")
    print("  - No boilerplate in function body")


if __name__ == "__main__":
    main()
