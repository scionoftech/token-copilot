"""Factory preset examples - pre-configured for common use cases."""

from langchain_openai import ChatOpenAI
from token_copilot.presets import basic, development, production, enterprise


def demo_basic():
    """Basic preset - just tracking."""
    print("\n" + "=" * 60)
    print("BASIC Preset - Minimal Configuration")
    print("=" * 60)

    copilot = basic(budget_limit=1.00)
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

    result = llm.invoke("Hello!")
    print(f"Response: {result.content}")
    print(f"Cost: ${copilot.cost:.4f}")


def demo_development():
    """Development preset - for local development."""
    print("\n" + "=" * 60)
    print("DEVELOPMENT Preset - Local Development")
    print("=" * 60)

    copilot = development(budget_limit=5.00, detect_anomalies=True)
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[copilot])

    result = llm.invoke("What is Python?")
    print(f"Response: {result.content[:80]}...")
    print(f"Cost: ${copilot.cost:.4f}")
    print(f"Plugins: {[type(p).__name__ for p in copilot.get_plugins()]}")


def demo_production():
    """Production preset - ready for deployment."""
    print("\n" + "=" * 60)
    print("PRODUCTION Preset - Production Ready")
    print("=" * 60)

    copilot = production(
        budget_limit=100.00,
        webhook_url="https://monitoring.example.com/webhook",
        detect_anomalies=True,
        enable_forecasting=True,
    )

    print(f"Configured plugins: {[type(p).__name__ for p in copilot.get_plugins()]}")
    print(f"Budget limit: ${copilot.budget_limit:.2f}")
    print("Ready for production deployment!")


def demo_enterprise():
    """Enterprise preset - all features enabled."""
    print("\n" + "=" * 60)
    print("ENTERPRISE Preset - Full Featured")
    print("=" * 60)

    copilot = enterprise(
        budget_limit=10000.00,
        kafka_brokers=["kafka1:9092", "kafka2:9092"],
        otlp_endpoint="http://collector:4318",
        enable_all=True,
    )

    print(f"Configured plugins: {[type(p).__name__ for p in copilot.get_plugins()]}")
    print(f"Budget limit: ${copilot.budget_limit:.2f}")
    print("Enterprise features enabled:")
    print("  - Kafka streaming")
    print("  - OTLP export")
    print("  - Analytics & anomaly detection")
    print("  - Adaptive operations")
    print("  - Budget forecasting")


def main():
    """Run all preset demos."""
    print("=" * 60)
    print("Factory Presets - Choose Your Configuration")
    print("=" * 60)

    demo_basic()
    demo_development()
    demo_production()
    demo_enterprise()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nFactory presets provide instant configuration:")
    print("  - basic(): Just cost tracking")
    print("  - development(): Local dev with logging")
    print("  - production(): Monitoring + alerts")
    print("  - enterprise(): All features enabled")


if __name__ == "__main__":
    main()
