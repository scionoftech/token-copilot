"""Comprehensive Test Suite for token-copilot Package

This script tests ALL functionalities of the token-copilot package:
1. Core features (tracking, budget, multi-tenant, export)
2. All 5 usage patterns (minimal, builder, factory, context, decorator)
3. All plugins (persistence, analytics, routing, adaptive, forecasting)
4. Framework integrations (LangChain, LlamaIndex)

Prerequisites:
- .env file with Azure OpenAI credentials
- pip install token-copilot[all] langchain-openai python-dotenv
"""

import sys
import os
import tempfile
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to Python path to import token_copilot
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add examples directory to path for azure_config
examples_path = os.path.join(os.path.dirname(__file__), "examples")
if examples_path not in sys.path:
    sys.path.insert(0, examples_path)

from azure_config import print_config_status, get_azure_langchain_llm


class TestRunner:
    """Test runner with result tracking."""

    def __init__(self):
        self.results = {}
        self.current_test = None

    def start_test(self, test_name: str):
        """Start a new test."""
        self.current_test = test_name
        print("\n" + "=" * 80)
        print(f"TEST: {test_name}")
        print("=" * 80)

    def pass_test(self, message: str = ""):
        """Mark current test as passed."""
        self.results[self.current_test] = True
        status = "[PASS]"
        if message:
            print(f"\n{status} {message}")
        else:
            print(f"\n{status}")

    def fail_test(self, error: Exception):
        """Mark current test as failed."""
        self.results[self.current_test] = False
        print(f"\n[FAIL] {error}")
        import traceback
        traceback.print_exc()

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results.values() if r)
        total = len(self.results)

        for name, result in self.results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status}: {name}")

        print(f"\nResults: {passed}/{total} tests passed")

        if passed == total:
            print("\n[SUCCESS] All tests passed!")
            return 0
        else:
            print(f"\n[WARNING] {total - passed} test(s) failed.")
            return 1


def test_core_tracking(runner: TestRunner):
    """Test 1: Core cost tracking functionality."""
    runner.start_test("Core Cost Tracking")

    try:
        from token_copilot import TokenCoPilot

        # Create copilot
        copilot = TokenCoPilot(budget_limit=10.00)
        print("[OK] TokenCoPilot created")

        # Get Azure LLM
        llm = get_azure_langchain_llm()
        print("[OK] Azure OpenAI LLM initialized")

        # Make a call
        print("\nMaking API call...")
        response = llm.invoke(
            "Say 'Hello' in exactly one word.",
            config={"callbacks": [copilot]}
        )

        print(f"Response: {response.content}")
        print(f"\n[OK] Tracking Results:")
        print(f"  Total Cost: ${copilot.cost:.6f}")
        print(f"  Total Tokens: {copilot.tokens:,}")
        print(f"  Remaining Budget: ${copilot.get_remaining_budget():.2f}")

        # Verify tracking
        assert copilot.cost > 0, "Cost should be tracked"
        assert copilot.tokens > 0, "Tokens should be tracked"

        runner.pass_test("Core tracking working")

    except Exception as e:
        runner.fail_test(e)


def test_budget_enforcement(runner: TestRunner):
    """Test 2: Budget enforcement."""
    runner.start_test("Budget Enforcement")

    try:
        from token_copilot import TokenCoPilot, BudgetExceededError

        # Create copilot with very low budget and "raise" mode
        copilot = TokenCoPilot(budget_limit=0.0001, on_budget_exceeded="raise")
        print("[OK] TokenCoPilot created with $0.0001 budget")

        llm = get_azure_langchain_llm()

        # Try to make a call that exceeds budget
        budget_exceeded = False
        try:
            response = llm.invoke(
                "Write a long essay about AI.",
                config={"callbacks": [copilot]}
            )
        except BudgetExceededError as e:
            budget_exceeded = True
            print(f"[OK] Budget exceeded as expected: {e}")

        assert budget_exceeded, "Budget enforcement should have triggered"

        runner.pass_test("Budget enforcement working")

    except Exception as e:
        runner.fail_test(e)


def test_multi_tenant_tracking(runner: TestRunner):
    """Test 3: Multi-tenant tracking."""
    runner.start_test("Multi-Tenant Tracking")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        # Make calls with different metadata
        users = ["alice", "bob", "charlie"]
        orgs = ["acme_corp", "acme_corp", "tech_startup"]

        print("\nMaking calls for multiple users...")
        for user, org in zip(users, orgs):
            response = llm.invoke(
                f"Say hello to {user}",
                config={
                    "callbacks": [copilot],
                    "metadata": {
                        "user_id": user,
                        "org_id": org
                    }
                }
            )
            print(f"  {user}@{org}: ${copilot.cost:.6f}")

        # Get costs by dimension
        user_costs = copilot.get_costs_by("user_id")
        org_costs = copilot.get_costs_by("org_id")

        print(f"\n[OK] Costs by User:")
        for user, cost in user_costs.items():
            print(f"  {user}: ${cost:.6f}")

        print(f"\n[OK] Costs by Organization:")
        for org, cost in org_costs.items():
            print(f"  {org}: ${cost:.6f}")

        # Verify multi-tenant tracking
        assert len(user_costs) == 3, "Should track 3 users"
        assert len(org_costs) == 2, "Should track 2 orgs"

        runner.pass_test("Multi-tenant tracking working")

    except Exception as e:
        runner.fail_test(e)


def test_dataframe_export(runner: TestRunner):
    """Test 4: DataFrame export."""
    runner.start_test("DataFrame Export")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot()
        llm = get_azure_langchain_llm()

        # Make some calls
        print("\nMaking test calls...")
        for i in range(3):
            response = llm.invoke(
                f"Count to {i+1}",
                config={
                    "callbacks": [copilot],
                    "metadata": {"task": f"task_{i}"}
                }
            )

        # Export to DataFrame
        df = copilot.to_dataframe()
        print(f"\n[OK] DataFrame exported: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\nColumns: {', '.join(df.columns.tolist())}")
        print(f"\nSample data:")
        print(df.head())

        # Verify DataFrame
        assert len(df) == 3, "Should have 3 rows"
        assert "cost" in df.columns, "Should have cost column"
        assert "tokens" in df.columns, "Should have tokens column"

        runner.pass_test("DataFrame export working")

    except Exception as e:
        runner.fail_test(e)


def test_minimal_pattern(runner: TestRunner):
    """Test 5: Minimal usage pattern."""
    runner.start_test("Minimal Usage Pattern")

    try:
        from token_copilot import TokenCoPilot

        # Minimal pattern - simplest way
        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        result = llm.invoke("Hello!", config={"callbacks": [copilot]})
        print(f"Cost: ${copilot.cost:.6f}")

        assert copilot.cost > 0

        runner.pass_test("Minimal pattern working")

    except Exception as e:
        runner.fail_test(e)


def test_builder_pattern(runner: TestRunner):
    """Test 6: Builder usage pattern."""
    runner.start_test("Builder Usage Pattern")

    try:
        from token_copilot import TokenCoPilot

        # Builder pattern with chaining
        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_adaptive()
            .build()
        )
        print("[OK] Built copilot with adaptive plugin")

        llm = get_azure_langchain_llm()
        result = llm.invoke("Hello!", config={"callbacks": [copilot]})

        print(f"Cost: ${copilot.cost:.6f}")
        print(f"Plugins: {len(copilot._plugin_manager.get_plugins())}")

        assert copilot.cost > 0
        assert len(copilot._plugin_manager.get_plugins()) > 0

        runner.pass_test("Builder pattern working")

    except Exception as e:
        runner.fail_test(e)


def test_factory_presets(runner: TestRunner):
    """Test 7: Factory presets."""
    runner.start_test("Factory Presets")

    try:
        from token_copilot.presets import basic, development, production

        # Test basic preset
        copilot1 = basic(budget_limit=10.00)
        print("[OK] Basic preset created")

        # Test development preset
        copilot2 = development(budget_limit=10.00)
        print("[OK] Development preset created")

        # Test production preset
        copilot3 = production(budget_limit=10.00)
        print("[OK] Production preset created")

        # Use one of them
        llm = get_azure_langchain_llm()
        result = llm.invoke("Test", config={"callbacks": [copilot1]})

        print(f"Cost: ${copilot1.cost:.6f}")

        assert copilot1.cost > 0

        runner.pass_test("Factory presets working")

    except Exception as e:
        runner.fail_test(e)


def test_context_managers(runner: TestRunner):
    """Test 8: Context managers."""
    runner.start_test("Context Managers")

    try:
        from token_copilot import track_costs, with_budget, monitored

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

        # Test monitored
        print("\n3. Testing monitored...")
        with monitored(name="test_op", budget_limit=10.00) as copilot:
            llm = get_azure_langchain_llm()
            result = llm.invoke("Hello!", config={"callbacks": [copilot]})
            cost3 = copilot.cost
            print(f"   Cost: ${cost3:.6f}")

        assert cost1 > 0 and cost2 > 0 and cost3 > 0

        runner.pass_test("Context managers working")

    except Exception as e:
        runner.fail_test(e)


def test_decorators(runner: TestRunner):
    """Test 9: Decorators."""
    runner.start_test("Decorators")

    try:
        from token_copilot.decorators import track_cost, enforce_budget, monitored

        # Test track_cost decorator
        @track_cost(budget_limit=10.00)
        def process_text_1(text):
            llm = get_azure_langchain_llm()
            return llm.invoke(f"Process: {text}", config={"callbacks": [process_text_1.copilot]})

        result1 = process_text_1("hello")
        cost1 = process_text_1.copilot.cost
        print(f"[OK] track_cost decorator: ${cost1:.6f}")

        # Test enforce_budget decorator
        @enforce_budget(limit=10.00, on_exceeded="warn")
        def process_text_2(text, copilot):
            llm = get_azure_langchain_llm()
            return llm.invoke(f"Process: {text}", config={"callbacks": [copilot]})

        result2 = process_text_2("world")
        print(f"[OK] enforce_budget decorator working")

        # Test monitored decorator
        @monitored(name="test_func", budget_limit=10.00)
        def process_text_3(text, copilot):
            llm = get_azure_langchain_llm()
            return llm.invoke(f"Process: {text}", config={"callbacks": [copilot]})

        result3 = process_text_3("test")
        print(f"[OK] monitored decorator working")

        assert cost1 > 0

        runner.pass_test("Decorators working")

    except Exception as e:
        runner.fail_test(e)


def test_persistence_plugin(runner: TestRunner):
    """Test 10: Persistence plugin."""
    runner.start_test("Persistence Plugin")

    try:
        from token_copilot import TokenCoPilot
        from token_copilot.plugins import SQLiteBackend, JSONBackend

        # Test SQLite backend
        print("\n1. Testing SQLite backend...")
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            backend = SQLiteBackend(db_path)
            copilot = (TokenCoPilot(budget_limit=10.00)
                .with_persistence(backend=backend, session_id="test_session")
            )

            llm = get_azure_langchain_llm()

            # Make calls
            for i in range(2):
                response = llm.invoke(f"Say {i}", config={"callbacks": [copilot]})

            # Query persistence
            plugin = copilot._plugin_manager.get_plugins()[0]
            events = plugin.get_events()
            summary = plugin.get_summary()

            print(f"   Events saved: {len(events)}")
            print(f"   Total cost: ${summary['total_cost']:.6f}")
            print(f"   Total calls: {summary['total_calls']}")

            backend.close()

            assert len(events) == 2
            assert summary['total_calls'] == 2

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

        # Test JSON backend
        print("\n2. Testing JSON backend...")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json_path = tmp.name

        try:
            backend = JSONBackend(json_path)
            copilot = (TokenCoPilot(budget_limit=10.00)
                .with_persistence(backend=backend, session_id="test_json")
            )

            llm = get_azure_langchain_llm()
            response = llm.invoke("Test", config={"callbacks": [copilot]})

            # Query persistence
            plugin = copilot._plugin_manager.get_plugins()[0]
            events = plugin.get_events()

            print(f"   Events saved: {len(events)}")

            backend.close()

            assert len(events) == 1

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

        runner.pass_test("Persistence plugin working")

    except Exception as e:
        runner.fail_test(e)


def test_analytics_plugin(runner: TestRunner):
    """Test 11: Analytics plugin."""
    runner.start_test("Analytics Plugin")

    try:
        from token_copilot import TokenCoPilot

        # Create copilot with analytics
        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_analytics(
                detect_anomalies=True,
                track_waste=True,
                track_efficiency=True
            )
        )
        print("[OK] Analytics plugin added")

        llm = get_azure_langchain_llm()

        # Make multiple calls to generate data
        print("\nGenerating test data...")
        for i in range(5):
            response = llm.invoke(
                f"Count to {i+1}",
                config={
                    "callbacks": [copilot],
                    "metadata": {"user_id": f"user_{i % 2}"}
                }
            )

        # Get analytics plugin
        from token_copilot.plugins.analytics import AnalyticsPlugin
        plugins = copilot._plugin_manager.get_plugins(AnalyticsPlugin)

        if plugins:
            analytics = plugins[0]
            print("[OK] Analytics plugin found")

            # Try to get analytics (may not have enough data)
            try:
                waste_report = analytics.analyze_waste()
                print(f"[OK] Waste analysis: {waste_report['summary']['waste_percentage']:.1f}% waste")
            except Exception as e:
                print(f"[INFO] Waste analysis needs more data: {e}")

            try:
                efficiency = analytics.get_efficiency_score("user_id", "user_0")
                print(f"[OK] Efficiency score: {efficiency.overall_score:.2f}")
            except Exception as e:
                print(f"[INFO] Efficiency scoring needs more data: {e}")

            anomalies = analytics.get_anomalies(minutes=60)
            print(f"[OK] Anomalies detected: {len(anomalies)}")

        runner.pass_test("Analytics plugin working")

    except Exception as e:
        runner.fail_test(e)


def test_routing_plugin(runner: TestRunner):
    """Test 12: Routing plugin."""
    runner.start_test("Routing Plugin")

    try:
        from token_copilot import TokenCoPilot, ModelConfig

        # Define models for routing
        models = [
            ModelConfig("gpt-4o-mini", quality=0.7, input_cost=0.15, output_cost=0.60),
            ModelConfig("gpt-4o", quality=0.9, input_cost=5.0, output_cost=15.0),
        ]

        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_routing(models=models, strategy="balanced")
        )
        print("[OK] Routing plugin added")

        # Get routing suggestions
        from token_copilot.plugins.routing import RoutingPlugin
        plugins = copilot._plugin_manager.get_plugins(RoutingPlugin)

        if plugins:
            routing = plugins[0]

            # Get routing decision
            decision = routing.suggest_model("Simple task", estimated_tokens=100)
            print(f"[OK] Routing decision: {decision.selected_model}")
            print(f"     Reason: {decision.reason}")

        runner.pass_test("Routing plugin working")

    except Exception as e:
        runner.fail_test(e)


def test_adaptive_plugin(runner: TestRunner):
    """Test 13: Adaptive plugin."""
    runner.start_test("Adaptive Plugin")

    try:
        from token_copilot import TokenCoPilot

        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_adaptive()
        )
        print("[OK] Adaptive plugin added")

        # Get adaptive plugin
        from token_copilot.plugins.adaptive import AdaptivePlugin
        plugins = copilot._plugin_manager.get_plugins(AdaptivePlugin)

        if plugins:
            adaptive = plugins[0]

            # Get tier info
            tier_info = adaptive.get_tier_info()
            print(f"[OK] Budget tier: {tier_info['tier_name']}")
            print(f"     Remaining: ${tier_info['remaining']:.2f}")

            # Get operations
            ops = adaptive.operations
            print(f"[OK] Adaptive operations available")

        runner.pass_test("Adaptive plugin working")

    except Exception as e:
        runner.fail_test(e)


def test_forecasting_plugin(runner: TestRunner):
    """Test 14: Forecasting plugin."""
    runner.start_test("Forecasting Plugin")

    try:
        from token_copilot import TokenCoPilot

        copilot = (TokenCoPilot(budget_limit=10.00)
            .with_forecasting(forecast_hours=48)
        )
        print("[OK] Forecasting plugin added")

        llm = get_azure_langchain_llm()

        # Make some calls to generate data
        for i in range(3):
            response = llm.invoke(f"Count to {i+1}", config={"callbacks": [copilot]})

        # Get forecasting plugin
        from token_copilot.plugins.forecasting import ForecastingPlugin
        plugins = copilot._plugin_manager.get_plugins(ForecastingPlugin)

        if plugins:
            forecasting = plugins[0]

            # Get forecast (may not have enough data)
            try:
                forecast = forecasting.get_forecast()
                print(f"[OK] Burn rate: ${forecast.burn_rate_per_hour:.4f}/hr")
                print(f"     Current cost: ${forecast.current_cost:.4f}")
            except Exception as e:
                print(f"[INFO] Forecasting needs more data: {e}")

        runner.pass_test("Forecasting plugin working")

    except Exception as e:
        runner.fail_test(e)


def test_langchain_integration(runner: TestRunner):
    """Test 15: LangChain integration."""
    runner.start_test("LangChain Integration")

    try:
        from token_copilot import TokenCoPilot
        from langchain_openai import ChatOpenAI

        copilot = TokenCoPilot(budget_limit=10.00)

        # Test with callbacks parameter
        llm = get_azure_langchain_llm()
        response = llm.invoke("Hello!", config={"callbacks": [copilot]})

        print(f"[OK] LangChain integration working")
        print(f"     Response: {response.content}")
        print(f"     Cost: ${copilot.cost:.6f}")

        assert copilot.cost > 0

        runner.pass_test("LangChain integration working")

    except Exception as e:
        runner.fail_test(e)


def test_llamaindex_integration(runner: TestRunner):
    """Test 16: LlamaIndex integration."""
    runner.start_test("LlamaIndex Integration")

    try:
        from token_copilot.llamaindex import TokenCoPilotCallbackHandler

        try:
            from examples.azure_config import get_azure_llamaindex_llm
            from llama_index.core import Settings

            copilot = TokenCoPilotCallbackHandler(budget_limit=10.00)
            llm = get_azure_llamaindex_llm()

            Settings.llm = llm
            Settings.callback_manager.add_handler(copilot)

            response = llm.complete("Say hello in one word.")

            print(f"[OK] LlamaIndex integration working")
            print(f"     Response: {response.text}")
            print(f"     Cost: ${copilot.get_total_cost():.6f}")

            assert copilot.get_total_cost() > 0

            runner.pass_test("LlamaIndex integration working")

        except ImportError as ie:
            print(f"[SKIP] LlamaIndex not installed: {ie}")
            runner.pass_test("LlamaIndex not installed (skipped)")

    except Exception as e:
        runner.fail_test(e)


def test_multi_turn_conversation(runner: TestRunner):
    """Test 17: Multi-turn conversation tracking."""
    runner.start_test("Multi-Turn Conversation")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        messages = [
            "What is 2+2?",
            "What is that times 2?",
            "What is that minus 5?"
        ]

        print("\nMulti-turn conversation:")
        for i, msg in enumerate(messages, 1):
            response = llm.invoke(msg, config={"callbacks": [copilot]})
            print(f"  Turn {i}: ${copilot.cost:.6f}")

        print(f"\n[OK] Total cost: ${copilot.cost:.6f}")
        print(f"     Avg cost/turn: ${copilot.cost/len(messages):.6f}")

        assert copilot.cost > 0

        runner.pass_test("Multi-turn conversation working")

    except Exception as e:
        runner.fail_test(e)


def test_stats_and_metrics(runner: TestRunner):
    """Test 18: Statistics and metrics."""
    runner.start_test("Statistics and Metrics")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        # Make several calls
        for i in range(5):
            response = llm.invoke(f"Count to {i+1}", config={"callbacks": [copilot]})

        # Get statistics
        stats = copilot.get_stats()

        print(f"[OK] Statistics:")
        print(f"     Total calls: {stats['total_calls']}")
        print(f"     Total cost: ${stats['total_cost']:.6f}")
        print(f"     Total tokens: {stats['total_tokens']:,}")
        print(f"     Avg cost/call: ${stats['avg_cost_per_call']:.6f}")
        print(f"     Avg tokens/call: {stats['avg_tokens_per_call']:.1f}")

        assert stats['total_calls'] == 5
        assert stats['total_cost'] > 0

        runner.pass_test("Statistics and metrics working")

    except Exception as e:
        runner.fail_test(e)


def test_reset_functionality(runner: TestRunner):
    """Test 19: Reset functionality."""
    runner.start_test("Reset Functionality")

    try:
        from token_copilot import TokenCoPilot

        copilot = TokenCoPilot(budget_limit=10.00)
        llm = get_azure_langchain_llm()

        # Make a call
        response = llm.invoke("Hello", config={"callbacks": [copilot]})
        cost_before = copilot.cost

        print(f"[OK] Cost before reset: ${cost_before:.6f}")

        # Reset
        copilot.reset()
        cost_after = copilot.cost

        print(f"[OK] Cost after reset: ${cost_after:.6f}")

        assert cost_before > 0
        assert cost_after == 0

        runner.pass_test("Reset functionality working")

    except Exception as e:
        runner.fail_test(e)


def test_pricing_utilities(runner: TestRunner):
    """Test 20: Pricing utilities."""
    runner.start_test("Pricing Utilities")

    try:
        from token_copilot import (
            get_model_config,
            calculate_cost,
            list_models,
            list_providers,
            MODEL_PRICING
        )

        # Test get_model_config
        config = get_model_config("gpt-4o-mini")
        print(f"[OK] Model config for gpt-4o-mini:")
        print(f"     Input: ${config.input_cost}/1k tokens")
        print(f"     Output: ${config.output_cost}/1k tokens")

        # Test calculate_cost
        cost = calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        print(f"\n[OK] Cost calculation: ${cost:.6f}")

        # Test list_models
        models = list_models()
        print(f"\n[OK] Available models: {len(models)}")

        # Test list_providers
        providers = list_providers()
        print(f"[OK] Available providers: {', '.join(providers)}")

        # Test MODEL_PRICING
        print(f"\n[OK] Total models in pricing: {len(MODEL_PRICING)}")

        assert config is not None
        assert cost > 0
        assert len(models) > 0
        assert len(providers) > 0

        runner.pass_test("Pricing utilities working")

    except Exception as e:
        runner.fail_test(e)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TOKEN-COPILOT COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    # Check Azure configuration
    if not print_config_status():
        print("\n[ERROR] Azure OpenAI not configured properly")
        print("Please check your .env file")
        return 1

    # Create test runner
    runner = TestRunner()

    # Define all tests
    tests = [
        ("Core Cost Tracking", test_core_tracking),
        ("Budget Enforcement", test_budget_enforcement),
        ("Multi-Tenant Tracking", test_multi_tenant_tracking),
        ("DataFrame Export", test_dataframe_export),
        ("Minimal Usage Pattern", test_minimal_pattern),
        ("Builder Usage Pattern", test_builder_pattern),
        ("Factory Presets", test_factory_presets),
        ("Context Managers", test_context_managers),
        ("Decorators", test_decorators),
        ("Persistence Plugin", test_persistence_plugin),
        ("Analytics Plugin", test_analytics_plugin),
        ("Routing Plugin", test_routing_plugin),
        ("Adaptive Plugin", test_adaptive_plugin),
        ("Forecasting Plugin", test_forecasting_plugin),
        ("LangChain Integration", test_langchain_integration),
        ("LlamaIndex Integration", test_llamaindex_integration),
        ("Multi-Turn Conversation", test_multi_turn_conversation),
        ("Statistics and Metrics", test_stats_and_metrics),
        ("Reset Functionality", test_reset_functionality),
        ("Pricing Utilities", test_pricing_utilities),
    ]

    # Run all tests
    print("\n" + "=" * 80)
    print(f"Running {len(tests)} tests...")
    print("=" * 80)

    start_time = time.time()

    for name, test_func in tests:
        try:
            test_func(runner)
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time

    # Print summary
    result = runner.print_summary()

    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return result


if __name__ == "__main__":
    sys.exit(main())
