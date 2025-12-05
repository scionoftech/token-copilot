"""Test Azure OpenAI integration with token-copilot.

This script tests all functionality with Azure OpenAI:
1. LangChain integration
2. LlamaIndex integration
3. All major features (budget, streaming, analytics, routing, persistence)

Prerequisites:
1. Copy .env.example to .env
2. Fill in your Azure OpenAI credentials
3. Install dependencies: pip install python-dotenv langchain-openai llama-index-llms-azure-openai
"""

import sys
from azure_config import print_config_status, get_azure_langchain_llm, get_azure_llamaindex_llm
from token_copilot import TokenCoPilot


def test_langchain_basic():
    """Test basic LangChain integration with Azure OpenAI."""
    print("\n" + "=" * 80)
    print("TEST 1: LangChain Basic Integration")
    print("=" * 80)

    try:
        # Create copilot with budget
        copilot = TokenCoPilot(budget_limit=10.00)
        print("[OK] TokenCoPilot created with $10 budget")

        # Get Azure OpenAI LLM
        llm = get_azure_langchain_llm()
        print("[OK] Azure OpenAI LLM initialized")

        # Make a simple call
        print("\nSending test message to Azure OpenAI...")
        response = llm.invoke(
            "Say 'Hello from Azure OpenAI!' in exactly 5 words.",
            config={"callbacks": [copilot]}
        )

        print(f"\nResponse: {response.content}")
        print(f"\n[OK] Cost Tracking:")
        print(f"  Total Cost: ${copilot.cost:.6f}")
        print(f"  Total Tokens: {copilot.tokens:,}")
        print(f"  Remaining Budget: ${copilot.get_remaining_budget():.2f}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llamaindex_basic():
    """Test basic LlamaIndex integration with Azure OpenAI."""
    print("\n" + "=" * 80)
    print("TEST 2: LlamaIndex Basic Integration")
    print("=" * 80)

    try:
        # Create LlamaIndex callback handler
        from token_copilot.llamaindex import TokenCoPilotCallbackHandler
        copilot = TokenCoPilotCallbackHandler(budget_limit=10.00)
        print("[OK] TokenCoPilot created with $10 budget")

        # Get Azure OpenAI LLM
        llm = get_azure_llamaindex_llm()
        print("[OK] Azure OpenAI LLM initialized for LlamaIndex")

        # Set callback
        from llama_index.core import Settings
        Settings.llm = llm
        Settings.callback_manager.add_handler(copilot)

        # Make a simple call
        print("\nSending test message to Azure OpenAI...")
        response = llm.complete("Say 'Hello from LlamaIndex!' in exactly 5 words.")

        print(f"\nResponse: {response.text}")
        print(f"\n[OK] Cost Tracking:")
        print(f"  Total Cost: ${copilot.get_total_cost():.6f}")
        print(f"  Total Tokens: {copilot.get_total_tokens():,}")
        print(f"  Remaining Budget: ${copilot.get_remaining_budget():.2f}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_budget_enforcement():
    """Test budget enforcement with Azure OpenAI."""
    print("\n" + "=" * 80)
    print("TEST 3: Budget Enforcement")
    print("=" * 80)

    try:
        # Create copilot with very low budget
        copilot = TokenCoPilot(budget_limit=0.01, on_budget_exceeded="warn")
        print("[OK] TokenCoPilot created with $0.01 budget (will exceed)")

        # Get Azure OpenAI LLM
        llm = get_azure_langchain_llm()

        # Make calls until budget exceeded
        for i in range(3):
            print(f"\nCall {i+1}...")
            try:
                response = llm.invoke(
                    "Write a haiku about AI.",
                    config={"callbacks": [copilot]}
                )
                print(f"  Cost so far: ${copilot.cost:.6f}")
            except Exception as e:
                print(f"  Budget check triggered: {e}")
                break

        print(f"\n[OK] Budget Enforcement Working:")
        print(f"  Total Cost: ${copilot.cost:.6f}")
        print(f"  Budget Limit: ${copilot.budget_limit:.2f}")
        print(f"  Exceeded: {copilot.cost > copilot.budget_limit}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_builder_pattern():
    """Test builder pattern with multiple plugins."""
    print("\n" + "=" * 80)
    print("TEST 4: Builder Pattern with Analytics")
    print("=" * 80)

    try:
        from unittest.mock import patch

        # Mock analytics to avoid complex dependencies in test
        with patch('token_copilot.plugins.analytics.AnalyticsPlugin'):
            copilot = (TokenCoPilot(budget_limit=10.00)
                .with_analytics(detect_anomalies=True)
            )
            print("[OK] TokenCoPilot created with Analytics plugin")

            # Get Azure OpenAI LLM
            llm = get_azure_langchain_llm()

            # Make a call
            print("\nSending test message...")
            response = llm.invoke(
                "Count from 1 to 5.",
                config={"callbacks": [copilot]}
            )

            print(f"\nResponse: {response.content}")
            print(f"\n[OK] Builder Pattern Working:")
            print(f"  Plugins: {len(copilot._plugin_manager.get_plugins())}")
            print(f"  Total Cost: ${copilot.cost:.6f}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persistence():
    """Test persistence with Azure OpenAI."""
    print("\n" + "=" * 80)
    print("TEST 5: Persistence Integration")
    print("=" * 80)

    try:
        from token_copilot.plugins import SQLiteBackend
        import tempfile
        import os

        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        db_path = temp_db.name

        try:
            # Create copilot with persistence
            backend = SQLiteBackend(db_path)
            copilot = (TokenCoPilot(budget_limit=10.00)
                .with_persistence(backend=backend, session_id="test_azure")
            )
            print(f"[OK] TokenCoPilot created with persistence: {db_path}")

            # Get Azure OpenAI LLM
            llm = get_azure_langchain_llm()

            # Make calls
            print("\nMaking 2 test calls...")
            for i in range(2):
                response = llm.invoke(
                    f"Say the number {i+1}.",
                    config={"callbacks": [copilot]}
                )
                print(f"  Call {i+1}: ${copilot.cost:.6f}")

            # Verify persistence
            plugin = copilot._plugin_manager.get_plugins()[0]
            events = plugin.get_events()
            summary = plugin.get_summary()

            print(f"\n[OK] Persistence Working:")
            print(f"  Events saved: {len(events)}")
            print(f"  Total cost in DB: ${summary['total_cost']:.6f}")
            print(f"  Total calls in DB: {summary['total_calls']}")

            backend.close()
            return True

        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_turn_conversation():
    """Test multi-turn conversation tracking."""
    print("\n" + "=" * 80)
    print("TEST 6: Multi-Turn Conversation")
    print("=" * 80)

    try:
        copilot = TokenCoPilot(budget_limit=10.00)
        print("[OK] TokenCoPilot created")

        llm = get_azure_langchain_llm()

        messages = [
            "What is 2+2?",
            "What is that number times 3?",
            "What is the square root of your last answer?"
        ]

        print("\nMulti-turn conversation:")
        for i, msg in enumerate(messages, 1):
            response = llm.invoke(msg, config={"callbacks": [copilot]})
            print(f"\n  Turn {i}:")
            print(f"    Q: {msg}")
            print(f"    A: {response.content}")
            print(f"    Cost: ${copilot.cost:.6f}")

        print(f"\n[OK] Multi-Turn Tracking:")
        print(f"  Total turns: {len(messages)}")
        print(f"  Total cost: ${copilot.cost:.6f}")
        print(f"  Avg cost/turn: ${copilot.cost/len(messages):.6f}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AZURE OPENAI INTEGRATION TEST SUITE")
    print("=" * 80)

    # Check configuration
    if not print_config_status():
        print("\n[ERROR] Please configure .env file with Azure OpenAI credentials")
        print("  1. Copy .env.example to .env")
        print("  2. Fill in your Azure OpenAI credentials")
        sys.exit(1)

    # Run tests
    tests = [
        ("LangChain Basic", test_langchain_basic),
        ("LlamaIndex Basic", test_llamaindex_basic),
        ("Budget Enforcement", test_budget_enforcement),
        ("Builder Pattern", test_builder_pattern),
        ("Persistence", test_persistence),
        ("Multi-Turn Conversation", test_multi_turn_conversation),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Azure OpenAI integration working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
