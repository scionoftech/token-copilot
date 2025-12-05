"""Builder pattern example - fluent API for adding features."""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilot


def main():
    """Demonstrate builder pattern."""
    print("=" * 60)
    print("Builder Pattern Example - Fluent API")
    print("=" * 60)

    # Build copilot with fluent API
    copilot = (TokenCoPilot(budget_limit=5.00)
        .with_streaming(webhook_url="https://example.com/webhook")
        .with_analytics(detect_anomalies=True)
        .with_adaptive()
        .build()  # Optional but makes intent clear
    )

    print("\nConfigured plugins:")
    for plugin in copilot.get_plugins():
        print(f"  - {type(plugin).__name__}")

    # Use normally
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

    print("\nMaking LLM calls...")
    result = llm.invoke("Explain AI in simple terms")
    print(f"Response: {result.content[:100]}...")

    # Access plugin features
    print("\n" + "=" * 60)
    print("Plugin Features")
    print("=" * 60)

    # Get adaptive plugin
    from token_copilot.plugins import AdaptivePlugin
    adaptive_plugins = copilot.get_plugins(AdaptivePlugin)
    if adaptive_plugins:
        adaptive = adaptive_plugins[0]
        tier_info = adaptive.get_tier_info()
        print(f"\nBudget Tier: {tier_info['tier_name'].upper()}")
        print(f"Remaining: ${tier_info['remaining']:.2f}")
        print(f"Description: {tier_info['description']}")

    # Get analytics plugin
    from token_copilot.plugins import AnalyticsPlugin
    analytics_plugins = copilot.get_plugins(AnalyticsPlugin)
    if analytics_plugins:
        analytics = analytics_plugins[0]
        print(f"\nAnom detection enabled: {analytics.detect_anomalies}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total cost: ${copilot.cost:.4f}")
    print(f"Total tokens: {copilot.tokens}")


if __name__ == "__main__":
    main()
