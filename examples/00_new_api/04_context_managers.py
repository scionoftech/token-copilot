"""Context manager examples - Pythonic cost tracking."""

from langchain_openai import ChatOpenAI
from token_copilot import track_costs, with_budget, monitored


def demo_track_costs():
    """track_costs - simple context manager."""
    print("\n" + "=" * 60)
    print("track_costs() - Simple Tracking")
    print("=" * 60)

    with track_costs(budget_limit=2.00) as copilot:
        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

        print("Making LLM calls...")
        result1 = llm.invoke("Say hello")
        result2 = llm.invoke("What is 2+2?")

        print(f"\nInside context:")
        print(f"  Cost: ${copilot.cost:.4f}")
        print(f"  Tokens: {copilot.tokens}")

    print("\nContext exited - summary logged")


def demo_with_budget():
    """with_budget - budget-focused context."""
    print("\n" + "=" * 60)
    print("with_budget() - Budget Enforcement")
    print("=" * 60)

    tasks = [
        "Hello",
        "Explain AI",
        "What is Python?",
        "Tell me about space",
    ]

    with with_budget(limit=1.00, warn_at=0.5) as budget:
        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[budget])

        print("Processing tasks with budget limit...")
        for i, task in enumerate(tasks, 1):
            remaining = budget.get_remaining_budget()

            if remaining > 0:
                print(f"\nTask {i}: {task[:30]}...")
                result = llm.invoke(task)
                print(f"  Remaining: ${remaining:.4f}")
            else:
                print(f"\nTask {i}: SKIPPED - budget exhausted")
                break

    print("\nBudget context exited")


def demo_monitored():
    """monitored - automatic logging."""
    print("\n" + "=" * 60)
    print("monitored() - Automatic Logging")
    print("=" * 60)

    with monitored(budget_limit=3.00, name="data_processing") as copilot:
        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

        documents = [
            "Document about AI",
            "Document about Python",
            "Document about LLMs",
        ]

        print("Processing documents...")
        for doc in documents:
            result = llm.invoke(f"Summarize: {doc}")
            print(f"  Processed: {doc}")

    print("\nMonitored context exited - metrics logged")


def main():
    """Run all context manager demos."""
    print("=" * 60)
    print("Context Managers - Pythonic Cost Tracking")
    print("=" * 60)

    demo_track_costs()
    demo_with_budget()
    demo_monitored()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nContext managers provide clean, Pythonic API:")
    print("  - track_costs(): General purpose tracking")
    print("  - with_budget(): Budget-focused enforcement")
    print("  - monitored(): Automatic operation logging")


if __name__ == "__main__":
    main()
