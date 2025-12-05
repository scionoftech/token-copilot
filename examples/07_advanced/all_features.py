"""
All Features Example - Complete Integration

This example demonstrates ALL token_copilot features in one complete application:

✅ Budget enforcement
✅ Multi-tenant tracking
✅ Waste analysis
✅ Efficiency scoring
✅ Anomaly detection with alerts
✅ Model routing
✅ Budget forecasting
✅ Request queuing
✅ DataFrame export and analysis

This serves as a reference for production-ready integration.
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback
from token_copilot.routing import ModelConfig, RoutingStrategy
from token_copilot.queuing import QueueMode
from token_copilot.analytics import log_alert, webhook_alert
import time


def production_setup():
    """Create a production-ready token_copilot setup."""
    print("=" * 60)
    print("PRODUCTION-READY SETUP")
    print("=" * 60)

    # Define available models
    models = [
        ModelConfig(
            name="gpt-4o-mini",
            quality_score=0.7,
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            max_tokens=128000,
            supports_functions=True,
            supports_vision=True
        ),
        ModelConfig(
            name="gpt-4o",
            quality_score=0.9,
            cost_per_1k_input=5.0,
            cost_per_1k_output=15.0,
            max_tokens=128000,
            supports_functions=True,
            supports_vision=True
        ),
    ]

    # Create callback with ALL features enabled
    callback = TokenCoPilotCallback(
        # Budget enforcement
        budget_limit=100.00,
        budget_period="daily",
        on_budget_exceeded="warn",

        # Anomaly detection
        anomaly_detection=True,
        anomaly_sensitivity=3.0,
        alert_handlers=[log_alert],

        # Model routing
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED,

        # Budget forecasting
        predictive_alerts=True,
        forecast_window_hours=24,

        # Request queuing
        queue_mode=QueueMode.SMART,
        max_queue_size=1000,
    )

    print("\n✅ Configured:")
    print("  • Budget: $100/day with warnings")
    print("  • Anomaly detection: Enabled (3σ)")
    print("  • Model routing: BALANCED strategy")
    print("  • Forecasting: 24h window")
    print("  • Queuing: SMART mode")

    return callback


def simulate_multi_tenant_usage(callback):
    """Simulate multi-tenant application usage."""
    print("\n" + "=" * 60)
    print("MULTI-TENANT USAGE SIMULATION")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Simulate different users and organizations
    users = {
        "alice": {"org": "acme_corp", "role": "developer"},
        "bob": {"org": "acme_corp", "role": "manager"},
        "charlie": {"org": "tech_startup", "role": "founder"},
    }

    print("\nSimulating usage from multiple users...")

    for user_id, info in users.items():
        print(f"\n👤 User: {user_id} ({info['role']} @ {info['org']})")

        # Get routing suggestion
        decision = callback.suggest_model(
            f"Task from {user_id}",
            estimated_tokens=500
        )
        print(f"  Routed to: {decision.selected_model}")

        # Make LLM call with metadata
        for i in range(2):
            llm.invoke(
                f"Hello from {user_id}, request {i}",
                config={
                    "metadata": {
                        "user_id": user_id,
                        "org_id": info["org"],
                        "role": info["role"],
                        "feature": "chat"
                    }
                }
            )

        # Check user's budget
        user_meta = {"user_id": user_id}
        if callback.budget_enforcer.period == "per_user":
            remaining = callback.get_remaining_budget(user_meta)
            print(f"  Remaining budget: ${remaining:.4f}")


def analyze_performance(callback):
    """Analyze performance across all dimensions."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    # 1. Basic statistics
    print("\n📊 BASIC STATISTICS")
    print("-" * 60)
    stats = callback.get_stats()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg cost/call: ${stats['avg_cost_per_call']:.4f}")
    print(f"Avg tokens/call: {stats['avg_tokens_per_call']:.1f}")

    # 2. Costs by dimension
    print("\n💰 COSTS BY DIMENSION")
    print("-" * 60)

    print("\nBy User:")
    costs_by_user = callback.get_costs_by_user()
    for user, cost in sorted(costs_by_user.items(), key=lambda x: x[1], reverse=True):
        print(f"  {user}: ${cost:.4f}")

    print("\nBy Organization:")
    costs_by_org = callback.get_costs_by_org()
    for org, cost in sorted(costs_by_org.items(), key=lambda x: x[1], reverse=True):
        print(f"  {org}: ${cost:.4f}")

    print("\nBy Model:")
    costs_by_model = callback.get_costs_by_model()
    for model, cost in sorted(costs_by_model.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: ${cost:.4f}")

    # 3. Waste analysis
    print("\n🗑️ WASTE ANALYSIS")
    print("-" * 60)
    try:
        waste_report = callback.analyze_waste()
        summary = waste_report['summary']
        print(f"Total waste: ${summary['total_waste_cost']:.4f} ({summary['waste_percentage']:.1f}%)")
        print(f"Potential savings: ${summary['monthly_savings']:.2f}/month")

        if waste_report['recommendations']:
            print("\nTop recommendations:")
            for i, rec in enumerate(waste_report['recommendations'][:3], 1):
                print(f"  {i}. {rec}")
    except Exception as e:
        print(f"Insufficient data for waste analysis: {e}")

    # 4. Efficiency scoring
    print("\n⭐ EFFICIENCY SCORES")
    print("-" * 60)
    try:
        # Score all users
        all_scores = callback.get_efficiency_score('user_id')

        for user_id, score in sorted(all_scores.items(),
                                     key=lambda x: x[1].overall_score,
                                     reverse=True):
            print(f"\n{user_id}:")
            print(f"  Overall: {score.overall_score:.2f}")
            print(f"  Token efficiency: {score.token_efficiency:.2f}")
            print(f"  Cost efficiency: {score.cost_efficiency:.2f}")
            print(f"  Quality estimate: {score.quality_estimate:.2f}")

        # Leaderboard
        print("\n🏆 EFFICIENCY LEADERBOARD")
        leaderboard = callback.get_leaderboard('user_id', top_n=3)
        for entry in leaderboard:
            print(f"  {entry['rank']}. {entry['entity_id']}: {entry['overall_score']:.2f}")
    except Exception as e:
        print(f"Insufficient data for efficiency scoring: {e}")

    # 5. Anomalies
    print("\n🚨 ANOMALY DETECTION")
    print("-" * 60)
    anomalies = callback.get_anomalies(minutes=60, min_severity='medium')

    if anomalies:
        print(f"Found {len(anomalies)} anomalies:")
        for anomaly in anomalies:
            print(f"  [{anomaly.severity.upper()}] {anomaly.message}")
    else:
        print("✅ No anomalies detected")

    anomaly_stats = callback.get_anomaly_stats()
    if anomaly_stats:
        print(f"\nTotal anomalies: {anomaly_stats['total']}")
        if anomaly_stats['by_severity']:
            print("By severity:")
            for severity, count in anomaly_stats['by_severity'].items():
                print(f"  {severity}: {count}")

    # 6. Budget forecast
    print("\n📈 BUDGET FORECAST")
    print("-" * 60)
    try:
        forecast = callback.get_forecast(forecast_hours=24)

        print(f"Current cost: ${forecast.current_cost:.4f}")
        print(f"Remaining budget: ${forecast.remaining_budget:.2f}")
        print(f"Burn rate: ${forecast.burn_rate_per_hour:.4f}/hour")

        if forecast.hours_until_exhausted:
            print(f"⚠️ Budget exhausts in: {forecast.hours_until_exhausted:.1f} hours")
        else:
            print("✅ Budget not expected to exhaust")

        print(f"\nProjections:")
        print(f"  24h: ${forecast.projected_cost_24h:.2f}")
        print(f"  7d: ${forecast.projected_cost_7d:.2f}")
        print(f"  30d: ${forecast.projected_cost_30d:.2f}")

        print(f"\nConfidence: {forecast.confidence:.2%}")
        print(f"Trend: {forecast.trend}")

        if forecast.recommendations:
            print("\nRecommendations:")
            for rec in forecast.recommendations:
                print(f"  • {rec}")
    except Exception as e:
        print(f"Forecasting error: {e}")

    # 7. Queue statistics
    print("\n📥 QUEUE STATISTICS")
    print("-" * 60)
    queue_stats = callback.get_queue_stats()

    if queue_stats:
        print(f"Current queue size: {queue_stats['current_size']}")
        print(f"Total queued: {queue_stats['total_queued']}")
        print(f"Total processed: {queue_stats['total_processed']}")
        print(f"Total dropped: {queue_stats['total_dropped']}")
        print(f"Avg wait time: {queue_stats['avg_wait_seconds']:.1f}s")
    else:
        print("Queue not configured")


def export_and_analyze(callback):
    """Export data for deeper analysis."""
    print("\n" + "=" * 60)
    print("DATA EXPORT & ANALYSIS")
    print("=" * 60)

    # Export to DataFrame
    df = callback.to_dataframe()

    print(f"\n📊 DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")

    # Custom analysis
    print("\n🔍 CUSTOM ANALYSIS")
    print("-" * 60)

    # Cost distribution
    print("\nCost distribution:")
    print(df['cost'].describe())

    # Token distribution
    print("\nToken distribution:")
    print(df[['input_tokens', 'output_tokens', 'total_tokens']].describe())

    # Peak usage times
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    print("\nUsage by hour:")
    hourly = df_copy.groupby('hour')['cost'].sum().sort_values(ascending=False)
    print(hourly.head())

    # Feature usage
    if 'feature' in df.columns:
        print("\nUsage by feature:")
        feature_usage = df.groupby('feature').agg({
            'cost': 'sum',
            'total_tokens': 'sum'
        }).sort_values('cost', ascending=False)
        print(feature_usage)

    # Export to CSV (commented out)
    # df.to_csv('token_copilot_export.csv')
    # print("\n💾 Exported to token_copilot_export.csv")


def routing_optimization(callback):
    """Demonstrate routing optimization."""
    print("\n" + "=" * 60)
    print("ROUTING OPTIMIZATION")
    print("=" * 60)

    prompts = [
        ("Simple", "Hi"),
        ("Medium", "Explain transformers"),
        ("Complex", "Design a distributed system architecture")
    ]

    print("\nRouting decisions:")
    for complexity, prompt in prompts:
        decision = callback.suggest_model(prompt, estimated_tokens=1000)

        print(f"\n[{complexity}] {prompt[:40]}...")
        print(f"  Model: {decision.selected_model}")
        print(f"  Cost: ${decision.estimated_cost:.4f}")
        print(f"  Quality: {decision.quality_score:.2f}")
        print(f"  Reason: {decision.reason}")

    # Model statistics
    print("\n📊 Model Performance:")
    model_stats = callback.get_model_stats()
    for model, stats in model_stats.items():
        print(f"\n{model}:")
        print(f"  Avg quality: {stats['avg_quality']:.2f}")
        print(f"  Calls: {stats['calls']}")


def main():
    """Run complete example with all features."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "ALL FEATURES - COMPLETE INTEGRATION" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Setup
    callback = production_setup()

    # Simulate usage
    simulate_multi_tenant_usage(callback)

    # Analyze everything
    analyze_performance(callback)

    # Export data
    export_and_analyze(callback)

    # Routing optimization
    routing_optimization(callback)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ COMPLETE FEATURE DEMO FINISHED")
    print("=" * 60)

    print("\n📋 Feature Checklist:")
    print("  ✅ Budget enforcement (daily limits)")
    print("  ✅ Multi-tenant tracking (users/orgs)")
    print("  ✅ Waste analysis (savings opportunities)")
    print("  ✅ Efficiency scoring (performance metrics)")
    print("  ✅ Anomaly detection (cost spikes)")
    print("  ✅ Model routing (auto-optimization)")
    print("  ✅ Budget forecasting (predictions)")
    print("  ✅ Request queuing (priority management)")
    print("  ✅ Data export (pandas integration)")

    print("\n💡 Next Steps:")
    print("  1. Integrate into your application")
    print("  2. Configure for your specific needs")
    print("  3. Monitor and optimize based on analytics")
    print("  4. Set up alerting for production")

    print("\n🎯 Production Checklist:")
    print("  • Set appropriate budget limits")
    print("  • Configure alert handlers (Slack/webhooks)")
    print("  • Enable anomaly detection")
    print("  • Use learned routing for optimization")
    print("  • Regular monitoring of forecasts")
    print("  • Export data for reporting")

    print()


if __name__ == "__main__":
    main()
