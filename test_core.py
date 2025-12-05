"""Core Functionality Test - Simplified and Reliable

Tests the main functionalities of token-copilot with Azure OpenAI.
"""

import sys
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add examples directory to path
examples_path = os.path.join(os.path.dirname(__file__), "examples")
if examples_path not in sys.path:
    sys.path.insert(0, examples_path)

from azure_config import print_config_status, get_azure_langchain_llm


def print_test_header(test_name):
    """Print test header."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)


def print_success(message="Test passed"):
    """Print success message."""
    print(f"\n[PASS] {message}")


def print_error(message):
    """Print error message."""
    print(f"\n[FAIL] {message}")


def test_1_basic_tracking():
    """Test 1: Basic cost tracking."""
    print_test_header("1. Basic Cost Tracking")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        print("Making API call...")
        response = llm.invoke(
            "Say hello in one word",
            config={"callbacks": [copilot]}
        )

        print(f"Response: {response.content}")
        print(f"Cost: ${copilot.cost:.6f}")
        print(f"Tokens: {copilot.tokens:,}")
        print(f"Remaining: ${copilot.get_remaining_budget():.2f}")

        assert copilot.cost > 0, "Cost should be tracked"
        assert copilot.tokens > 0, "Tokens should be tracked"

        print_success("Basic tracking works!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_multi_tenant():
    """Test 2: Multi-tenant tracking."""
    print_test_header("2. Multi-Tenant Tracking")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        # Make calls with different user metadata
        users = ["alice", "bob", "charlie"]

        print("Making calls for multiple users...")
        for user in users:
            response = llm.invoke(
                f"Say hi to {user}",
                config={
                    "callbacks": [copilot],
                    "metadata": {"user_id": user}
                }
            )
            print(f"  {user}: done")

        # Get costs by user using tracker's method
        user_costs = copilot.tracker.get_costs_by("user_id")

        print("\nCosts by user:")
        for user, cost in user_costs.items():
            print(f"  {user}: ${cost:.6f}")

        assert len(user_costs) == 3, "Should track 3 users"

        print_success("Multi-tenant tracking works!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_dataframe_export():
    """Test 3: DataFrame export."""
    print_test_header("3. DataFrame Export")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot()
        llm = get_azure_langchain_llm()

        # Make some calls
        print("Making test calls...")
        for i in range(3):
            response = llm.invoke(
                f"Count to {i+1}",
                config={"callbacks": [copilot]}
            )

        # Export to DataFrame
        df = copilot.to_dataframe()

        print(f"\nDataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample:")
        print(df[['model', 'input_tokens', 'output_tokens', 'cost']].head())

        assert len(df) == 3, "Should have 3 rows"
        assert "cost" in df.columns, "Should have cost column"

        print_success("DataFrame export works!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_builder_pattern():
    """Test 4: Builder pattern with plugins."""
    print_test_header("4. Builder Pattern")

    try:
        from token_copilot import TokenCoPilot

        # Build with adaptive plugin
        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_adaptive()
            .build()
        )

        print("Built copilot with adaptive plugin")

        llm = get_azure_langchain_llm()
        response = llm.invoke("Hello!", config={"callbacks": [copilot]})

        print(f"Cost: ${copilot.cost:.6f}")
        print(f"Plugins: {len(copilot._plugin_manager.get_plugins())}")

        assert copilot.cost > 0
        assert len(copilot._plugin_manager.get_plugins()) > 0

        print_success("Builder pattern works!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_factory_presets():
    """Test 5: Factory presets."""
    print_test_header("5. Factory Presets")

    try:
        from token_copilot.presets import basic, development

        copilot1 = basic(budget_limit=10.00)
        print("[OK] Basic preset created")

        copilot2 = development(budget_limit=10.00)
        print("[OK] Development preset created")

        # Test with one
        llm = get_azure_langchain_llm()
        response = llm.invoke("Test", config={"callbacks": [copilot1]})

        print(f"Cost: ${copilot1.cost:.6f}")

        print_success("Factory presets work!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_context_managers():
    """Test 6: Context managers."""
    print_test_header("6. Context Managers")

    try:
        from token_copilot import track_costs, with_budget

        # Test track_costs
        print("\n1. Testing track_costs...")
        with track_costs(budget_limit=10.00) as copilot:
            llm = get_azure_langchain_llm()
            result = llm.invoke("Hello!", config={"callbacks": [copilot]})
            cost1 = copilot.cost
            print(f"   Cost: ${cost1:.6f}")

        # Test with_budget
        print("\n2. Testing with_budget...")
        with with_budget(limit=10.00, warn_at=0.8) as budget:
            llm = get_azure_langchain_llm()
            result = llm.invoke("Hello!", config={"callbacks": [budget]})
            cost2 = budget.cost
            print(f"   Cost: ${cost2:.6f}")

        assert cost1 > 0 and cost2 > 0

        print_success("Context managers work!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_decorators():
    """Test 7: Decorators."""
    print_test_header("7. Decorators")

    try:
        from token_copilot.decorators import track_cost

        # Define decorated function
        @track_cost(budget_limit=10.00)
        def process_text(text):
            llm = get_azure_langchain_llm()
            return llm.invoke(f"Process: {text}", config={"callbacks": [process_text.copilot]})

        result = process_text("hello")
        cost = process_text.copilot.cost
        print(f"Cost: ${cost:.6f}")

        assert cost > 0

        print_success("Decorators work!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_persistence():
    """Test 8: Persistence plugin."""
    print_test_header("8. Persistence Plugin")

    try:
        from token_copilot import TokenCoPilot
        from token_copilot.plugins import SQLiteBackend

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            backend = SQLiteBackend(db_path)
            copilot = (TokenCoPilot(budget_limit=10.00)
                .with_persistence(backend=backend, session_id="test")
            )

            print(f"Using database: {db_path}")

            llm = get_azure_langchain_llm()

            # Make calls
            print("Making test calls...")
            for i in range(2):
                response = llm.invoke(f"Say {i}", config={"callbacks": [copilot]})

            # Query persistence
            plugin = copilot._plugin_manager.get_plugins()[0]
            events = plugin.get_events()
            summary = plugin.get_summary()

            print(f"\nEvents saved: {len(events)}")
            print(f"Total cost: ${summary['total_cost']:.6f}")
            print(f"Total calls: {summary['total_calls']}")

            backend.close()

            print_success("Persistence works!")
            return True

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_9_analytics():
    """Test 9: Analytics plugin."""
    print_test_header("9. Analytics Plugin")

    try:
        from token_copilot import TokenCoPilot

        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_analytics(detect_anomalies=True)
        )

        print("Analytics plugin added")

        llm = get_azure_langchain_llm()

        # Make calls
        print("Generating data...")
        for i in range(5):
            response = llm.invoke(f"Count to {i+1}", config={"callbacks": [copilot]})

        # Get analytics
        from token_copilot.plugins.analytics import AnalyticsPlugin
        plugins = copilot._plugin_manager.get_plugins(AnalyticsPlugin)

        if plugins:
            analytics = plugins[0]
            print("[OK] Analytics plugin found")

            anomalies = analytics.get_anomalies(minutes=60)
            print(f"[OK] Anomalies detected: {len(anomalies)}")

        print_success("Analytics works!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_adaptive():
    """Test 10: Adaptive plugin."""
    print_test_header("10. Adaptive Plugin")

    try:
        from token_copilot import TokenCoPilot

        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_adaptive()
        )

        print("Adaptive plugin added")

        # Get adaptive plugin
        from token_copilot.plugins.adaptive import AdaptivePlugin
        plugins = copilot._plugin_manager.get_plugins(AdaptivePlugin)

        if plugins:
            adaptive = plugins[0]
            tier_info = adaptive.get_tier_info()
            print(f"[OK] Budget tier: {tier_info['tier_name']}")
            print(f"[OK] Remaining: ${tier_info['remaining']:.2f}")

        print_success("Adaptive works!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_11_pricing_utils():
    """Test 11: Pricing utilities."""
    print_test_header("11. Pricing Utilities")

    try:
        from token_copilot import get_model_config, calculate_cost, list_models, list_providers

        # Get model config
        config = get_model_config("gpt-4o-mini")
        print(f"[OK] Model: {config.model_id}")
        print(f"  Provider: {config.provider}")
        print(f"  Input: ${config.input_cost_per_1m:.2f}/1M tokens")
        print(f"  Output: ${config.output_cost_per_1m:.2f}/1M tokens")

        # Calculate cost
        cost = calculate_cost("gpt-4o-mini", 1000, 500)
        print(f"\n[OK] Cost for 1000 input + 500 output: ${cost:.6f}")

        # List models
        models = list_models()
        print(f"\n[OK] Available models: {len(models)}")

        # List providers
        providers = list_providers()
        print(f"[OK] Providers: {', '.join(providers)}")

        print_success("Pricing utilities work!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_12_statistics():
    """Test 12: Statistics and metrics."""
    print_test_header("12. Statistics and Metrics")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        # Make calls
        print("Making test calls...")
        for i in range(5):
            response = llm.invoke(f"Count to {i+1}", config={"callbacks": [copilot]})

        # Get statistics
        stats = copilot.get_stats()

        print(f"\n[OK] Total calls: {stats['total_calls']}")
        print(f"[OK] Total cost: ${stats['total_cost']:.6f}")
        print(f"[OK] Total tokens: {stats['total_tokens']:,}")
        print(f"[OK] Avg cost/call: ${stats['avg_cost_per_call']:.6f}")
        print(f"[OK] Avg tokens/call: {stats['avg_tokens_per_call']:.1f}")

        assert stats['total_calls'] == 5

        print_success("Statistics work!")
        return True

    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TOKEN-COPILOT CORE FUNCTIONALITY TESTS")
    print("=" * 80)

    # Check configuration
    if not print_config_status():
        print("\n❌ Azure OpenAI not configured properly")
        return 1

    # Run all tests
    tests = [
        ("Basic Tracking", test_1_basic_tracking),
        ("Multi-Tenant", test_2_multi_tenant),
        ("DataFrame Export", test_3_dataframe_export),
        ("Builder Pattern", test_4_builder_pattern),
        ("Factory Presets", test_5_factory_presets),
        ("Context Managers", test_6_context_managers),
        ("Decorators", test_7_decorators),
        ("Persistence", test_8_persistence),
        ("Analytics", test_9_analytics),
        ("Adaptive", test_10_adaptive),
        ("Pricing Utils", test_11_pricing_utils),
        ("Statistics", test_12_statistics),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\nTests interrupted")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
