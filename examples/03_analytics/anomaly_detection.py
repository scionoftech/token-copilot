"""
Anomaly Detection Example - Detect Cost & Usage Spikes

This example demonstrates real-time anomaly detection:
- Cost spikes (unusually high per-call costs)
- Token spikes (unusually high token usage)
- Frequency spikes (unusually high call rates)
- Unusual model usage
- Custom alert handlers
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
from token_copilot.analytics import log_alert, webhook_alert
import time
import random


def basic_anomaly_detection():
    """Example 1: Basic anomaly detection."""
    print("=" * 60)
    print("Example 1: Basic Anomaly Detection")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        budget_limit=10.00,
        anomaly_detection=True,
        anomaly_sensitivity=3.0,  # 3 standard deviations
        alert_handlers=[log_alert]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Normal usage pattern
    print("\nPhase 1: Normal usage (building baseline)")
    for i in range(10):
        llm.invoke("Short question")
        time.sleep(0.1)

    print(f"Baseline cost: ${callback.get_total_cost():.4f}")

    # Introduce anomaly: Very long prompt
    print("\nPhase 2: Introducing cost spike")
    very_long_prompt = "Explain in great detail " * 100
    llm.invoke(very_long_prompt)

    # Check for anomalies
    anomalies = callback.get_anomalies(minutes=5)
    if anomalies:
        print(f"\n🚨 Detected {len(anomalies)} anomaly(ies):")
        for anomaly in anomalies:
            print(f"  [{anomaly.severity}] {anomaly.message}")
    else:
        print("\n✅ No anomalies detected")


def sensitivity_levels():
    """Example 2: Different sensitivity levels."""
    print("\n" + "=" * 60)
    print("Example 2: Sensitivity Levels")
    print("=" * 60)

    sensitivities = [
        (1.5, "Very sensitive (1.5σ)"),
        (2.0, "Sensitive (2.0σ)"),
        (3.0, "Normal (3.0σ)"),
        (4.0, "Tolerant (4.0σ)"),
    ]

    for sensitivity, label in sensitivities:
        print(f"\n{label}:")

        callback = TokenCoPilotCallback(
            anomaly_detection=True,
            anomaly_sensitivity=sensitivity,
        )

        llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

        # Build baseline
        for i in range(10):
            llm.invoke("Short")

        # Introduce moderate spike
        llm.invoke("Medium length prompt " * 20)

        # Check detection
        anomalies = callback.get_anomalies(minutes=5)
        if anomalies:
            print(f"  ✅ Detected: {len(anomalies)} anomaly(ies)")
        else:
            print(f"  ❌ Not detected")


def cost_spike_detection():
    """Example 3: Cost spike detection."""
    print("\n" + "=" * 60)
    print("Example 3: Cost Spike Detection")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        anomaly_sensitivity=2.0,
        alert_handlers=[log_alert]
    )

    gpt4_mini = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    gpt4 = ChatOpenAI(model="gpt-4o", callbacks=[callback])

    # Normal usage with cheap model
    print("\nNormal usage with GPT-4o-mini:")
    for i in range(10):
        gpt4_mini.invoke("Question")

    avg_cost = callback.get_total_cost() / 10
    print(f"Average cost: ${avg_cost:.4f}/call")

    # Spike: Switch to expensive model
    print("\nCost spike: Switching to GPT-4o:")
    gpt4.invoke("Question")

    # Check anomalies
    anomalies = callback.get_anomalies(minutes=5, min_severity='high')
    if anomalies:
        print(f"\n🚨 Cost spike detected!")
        for anomaly in anomalies:
            print(f"  {anomaly.message}")
            print(f"  Value: ${anomaly.value:.4f}")
            print(f"  Mean: ${anomaly.mean:.4f}")
            print(f"  Z-score: {anomaly.z_score:.2f}σ")


def token_spike_detection():
    """Example 4: Token spike detection."""
    print("\n" + "=" * 60)
    print("Example 4: Token Spike Detection")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        anomaly_sensitivity=2.0,
        alert_handlers=[log_alert]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Normal usage
    print("\nNormal usage with short prompts:")
    for i in range(10):
        llm.invoke(f"Question {i}")

    avg_tokens = callback.get_total_tokens() / 10
    print(f"Average tokens: {avg_tokens:.0f}/call")

    # Spike: Very long prompt
    print("\nToken spike: Very long prompt:")
    long_prompt = "Explain in detail " * 200
    llm.invoke(long_prompt)

    # Check anomalies
    anomalies = callback.get_anomalies(minutes=5)
    token_anomalies = [a for a in anomalies if a.anomaly_type == 'token_spike']

    if token_anomalies:
        print(f"\n🚨 Token spike detected!")
        for anomaly in token_anomalies:
            print(f"  {anomaly.message}")


def frequency_spike_detection():
    """Example 5: Frequency spike detection."""
    print("\n" + "=" * 60)
    print("Example 5: Frequency Spike Detection")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        anomaly_sensitivity=2.0,
        alert_handlers=[log_alert]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Normal usage: Slow rate
    print("\nNormal usage: Slow rate (1 call/second)")
    for i in range(10):
        llm.invoke("Question")
        time.sleep(1)

    # Spike: Burst of calls
    print("\nFrequency spike: Burst of calls")
    for i in range(20):
        llm.invoke("Question")
        time.sleep(0.05)  # Very fast

    # Check anomalies
    anomalies = callback.get_anomalies(minutes=5)
    freq_anomalies = [a for a in anomalies if a.anomaly_type == 'frequency_spike']

    if freq_anomalies:
        print(f"\n🚨 Frequency spike detected!")
        for anomaly in freq_anomalies:
            print(f"  {anomaly.message}")


def unusual_model_detection():
    """Example 6: Unusual model usage detection."""
    print("\n" + "=" * 60)
    print("Example 6: Unusual Model Usage")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        alert_handlers=[log_alert]
    )

    gpt4_mini = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])
    gpt4 = ChatOpenAI(model="gpt-4o", callbacks=[callback])

    # Mostly use one model
    print("\nMostly using GPT-4o-mini:")
    for i in range(20):
        gpt4_mini.invoke("Question")

    # Suddenly use different model
    print("\nSuddenly using GPT-4o:")
    gpt4.invoke("Question")

    # Check anomalies
    anomalies = callback.get_anomalies(minutes=5)
    model_anomalies = [a for a in anomalies if a.anomaly_type == 'unusual_model']

    if model_anomalies:
        print(f"\n⚠️ Unusual model detected!")
        for anomaly in model_anomalies:
            print(f"  {anomaly.message}")


def custom_alert_handlers():
    """Example 7: Custom alert handlers."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Alert Handlers")
    print("=" * 60)

    # Custom handler
    def custom_handler(anomaly):
        """Custom handler that prints formatted alert."""
        print(f"\n🔔 CUSTOM ALERT:")
        print(f"   Type: {anomaly.anomaly_type}")
        print(f"   Severity: {anomaly.severity}")
        print(f"   Message: {anomaly.message}")
        print(f"   Time: {anomaly.timestamp.strftime('%H:%M:%S')}")

    # Create callback with multiple handlers
    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        anomaly_sensitivity=2.0,
        alert_handlers=[log_alert, custom_handler]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Normal usage
    print("\nGenerating baseline...")
    for i in range(10):
        llm.invoke("Short")

    # Trigger anomaly
    print("\nTriggering anomaly...")
    llm.invoke("Very long prompt " * 100)


def anomaly_statistics():
    """Example 8: Anomaly statistics."""
    print("\n" + "=" * 60)
    print("Example 8: Anomaly Statistics")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        anomaly_sensitivity=2.0,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Generate various anomalies
    print("\nGenerating mixed usage patterns...")

    # Normal
    for i in range(10):
        llm.invoke("Normal")

    # Cost spike
    llm.invoke("Long " * 100)

    # Token spike
    llm.invoke("Very long " * 150)

    # Frequency spike
    for i in range(10):
        llm.invoke("Burst")
        time.sleep(0.05)

    # Get statistics
    stats = callback.get_anomaly_stats()

    print("\n📊 Anomaly Statistics:")
    print(f"Total anomalies: {stats['total']}")

    if stats['by_type']:
        print("\nBy type:")
        for atype, count in stats['by_type'].items():
            print(f"  {atype}: {count}")

    if stats['by_severity']:
        print("\nBy severity:")
        for severity, count in stats['by_severity'].items():
            print(f"  {severity}: {count}")


def real_time_monitoring():
    """Example 9: Real-time anomaly monitoring."""
    print("\n" + "=" * 60)
    print("Example 9: Real-Time Monitoring")
    print("=" * 60)

    callback = TokenCoPilotCallback(
        anomaly_detection=True,
        anomaly_sensitivity=2.5,
        alert_handlers=[log_alert]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    print("\nMonitoring for anomalies (10 iterations)...")

    for iteration in range(10):
        # Randomly introduce anomalies
        if random.random() < 0.3:  # 30% chance of anomaly
            prompt = "Anomaly prompt " * random.randint(50, 150)
        else:
            prompt = "Normal prompt"

        llm.invoke(prompt)

        # Check for recent anomalies
        recent = callback.get_anomalies(minutes=1, min_severity='medium')

        print(f"\nIteration {iteration + 1}:")
        print(f"  Cost: ${callback.get_total_cost():.4f}")
        if recent:
            print(f"  🚨 {len(recent)} anomaly(ies) detected")
        else:
            print(f"  ✅ No anomalies")

        time.sleep(0.5)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 14 + "ANOMALY DETECTION EXAMPLES" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run all examples
    basic_anomaly_detection()
    sensitivity_levels()
    cost_spike_detection()
    token_spike_detection()
    frequency_spike_detection()
    unusual_model_detection()
    custom_alert_handlers()
    anomaly_statistics()
    real_time_monitoring()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nKey Insights:")
    print("  • Set sensitivity based on your tolerance")
    print("  • Use 3.0σ (default) for normal applications")
    print("  • Use 2.0σ for stricter monitoring")
    print("  • Use 1.5σ for very sensitive detection")
    print("\nAnomaly Types:")
    print("  • Cost spikes: Unusually expensive calls")
    print("  • Token spikes: Unusually long inputs/outputs")
    print("  • Frequency spikes: Sudden burst of calls")
    print("  • Unusual models: Rarely-used models")
    print("\nBest Practices:")
    print("  • Build baseline with normal usage first")
    print("  • Monitor anomaly stats regularly")
    print("  • Set up alert handlers for production")
    print("  • Adjust sensitivity based on false positives")
    print()


if __name__ == "__main__":
    main()
