"""
Budget Enforcement Example - Control LLM Spending

This example demonstrates different budget enforcement strategies:
- Total budget limits
- Daily/monthly budgets
- Per-user/per-org budgets
- Different behaviors when exceeded
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenPilotCallback
import time


def total_budget():
    """Example: Total budget across all time."""
    print("=" * 60)
    print("Example 1: Total Budget Limit")
    print("=" * 60)

    callback = TokenPilotCallback(
        budget_limit=0.05,  # $0.05 total
        budget_period="total",
        on_budget_exceeded="raise"  # Raise error when exceeded
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Make calls until budget exhausted
    try:
        for i in range(100):
            llm.invoke(f"What is {i} + {i}?")
            print(f"Call {i+1}: ${callback.get_total_cost():.4f} / ${callback.budget_enforcer.limit:.2f}")
    except Exception as e:
        print(f"\n⚠️ Budget exhausted: {e}")
        print(f"Final cost: ${callback.get_total_cost():.4f}")


def daily_budget():
    """Example: Daily budget that resets."""
    print("\n" + "=" * 60)
    print("Example 2: Daily Budget")
    print("=" * 60)

    callback = TokenPilotCallback(
        budget_limit=0.10,  # $0.10 per day
        budget_period="daily",
        on_budget_exceeded="warn"  # Just warn, don't stop
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Make calls
    for i in range(5):
        llm.invoke(f"Question {i}")
        print(f"Call {i+1}: ${callback.get_total_cost():.4f} spent today")

    print("\n💡 Budget resets daily at midnight")
    print(f"Current daily spend: ${callback.budget_enforcer.get_spent():.4f}")
    print(f"Remaining today: ${callback.get_remaining_budget():.4f}")


def per_user_budget():
    """Example: Separate budget per user."""
    print("\n" + "=" * 60)
    print("Example 3: Per-User Budget")
    print("=" * 60)

    callback = TokenPilotCallback(
        budget_limit=0.02,  # $0.02 per user
        budget_period="per_user",
        on_budget_exceeded="warn"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Simulate multiple users
    users = ["alice", "bob", "charlie"]

    for user in users:
        print(f"\n--- User: {user} ---")

        for i in range(3):
            # Include user metadata
            llm.invoke(
                f"Hello from {user}, question {i}",
                config={"metadata": {"user_id": user}}
            )

        # Check user's spending
        user_remaining = callback.get_remaining_budget({"user_id": user})
        print(f"{user}'s remaining budget: ${user_remaining:.4f}")

    # Overall costs by user
    print("\n" + "-" * 60)
    print("Total costs by user:")
    costs_by_user = callback.get_costs_by_user()
    for user, cost in costs_by_user.items():
        print(f"  {user}: ${cost:.4f}")


def per_org_budget():
    """Example: Separate budget per organization."""
    print("\n" + "=" * 60)
    print("Example 4: Per-Organization Budget")
    print("=" * 60)

    callback = TokenPilotCallback(
        budget_limit=0.05,  # $0.05 per org
        budget_period="per_org",
        on_budget_exceeded="warn"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Simulate multiple organizations
    orgs = {
        "acme_corp": ["user1", "user2"],
        "tech_startup": ["user3", "user4"],
    }

    for org_id, users in orgs.items():
        print(f"\n--- Organization: {org_id} ---")

        for user in users:
            llm.invoke(
                f"Request from {user}",
                config={"metadata": {"org_id": org_id, "user_id": user}}
            )
            print(f"  {user} made a request")

        # Check org's spending
        org_remaining = callback.get_remaining_budget({"org_id": org_id})
        print(f"{org_id}'s remaining budget: ${org_remaining:.4f}")

    # Overall costs by org
    print("\n" + "-" * 60)
    print("Total costs by organization:")
    costs_by_org = callback.get_costs_by_org()
    for org, cost in costs_by_org.items():
        print(f"  {org}: ${cost:.4f}")


def different_behaviors():
    """Example: Different behaviors when budget exceeded."""
    print("\n" + "=" * 60)
    print("Example 5: Different Exceeded Behaviors")
    print("=" * 60)

    # Behavior 1: Raise error (default)
    print("\n1. RAISE: Stop execution")
    callback_raise = TokenPilotCallback(
        budget_limit=0.01,
        on_budget_exceeded="raise"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_raise])

    try:
        for i in range(100):
            llm.invoke(f"Question {i}")
    except Exception as e:
        print(f"   ❌ Stopped: {type(e).__name__}")

    # Behavior 2: Warn but continue
    print("\n2. WARN: Log warning and continue")
    callback_warn = TokenPilotCallback(
        budget_limit=0.01,
        on_budget_exceeded="warn"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_warn])

    for i in range(3):
        llm.invoke(f"Question {i}")
    print(f"   ⚠️ Continued despite budget exceeded")
    print(f"   Total: ${callback_warn.get_total_cost():.4f}")

    # Behavior 3: Ignore (no action)
    print("\n3. IGNORE: No action")
    callback_ignore = TokenPilotCallback(
        budget_limit=0.01,
        on_budget_exceeded="ignore"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback_ignore])

    for i in range(3):
        llm.invoke(f"Question {i}")
    print(f"   ✅ Continued silently")
    print(f"   Total: ${callback_ignore.get_total_cost():.4f}")


def monthly_budget():
    """Example: Monthly budget."""
    print("\n" + "=" * 60)
    print("Example 6: Monthly Budget")
    print("=" * 60)

    callback = TokenPilotCallback(
        budget_limit=10.00,  # $10 per month
        budget_period="monthly",
        on_budget_exceeded="warn"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Simulate some usage
    for i in range(5):
        llm.invoke(f"Question {i}")

    print(f"\nMonthly spending: ${callback.budget_enforcer.get_spent():.4f}")
    print(f"Remaining this month: ${callback.get_remaining_budget():.4f}")
    print("\n💡 Budget resets on the 1st of each month")


def check_before_call():
    """Example: Check budget before making calls."""
    print("\n" + "=" * 60)
    print("Example 7: Check Budget Before Calls")
    print("=" * 60)

    callback = TokenPilotCallback(
        budget_limit=0.05,
        on_budget_exceeded="raise"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Check remaining budget before making calls
    for i in range(10):
        remaining = callback.get_remaining_budget()

        if remaining < 0.005:  # Less than half a cent
            print(f"\n⚠️ Budget low: ${remaining:.4f} remaining")
            print("Stopping to avoid exceeding budget")
            break

        llm.invoke(f"Question {i}")
        print(f"Call {i+1}: ${remaining:.4f} → ${callback.get_remaining_budget():.4f}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "BUDGET ENFORCEMENT EXAMPLES" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run all examples
    total_budget()
    daily_budget()
    per_user_budget()
    per_org_budget()
    different_behaviors()
    monthly_budget()
    check_before_call()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Use 'total' for overall budget limits")
    print("  • Use 'daily/monthly' for time-based limits")
    print("  • Use 'per_user/per_org' for multi-tenant apps")
    print("  • Choose 'raise/warn/ignore' based on your needs")
    print()


if __name__ == "__main__":
    main()
