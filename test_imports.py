"""
Test script to verify all import fixes for langchain, langgraph, and llamaindex.
This script tests that all the updated imports work correctly without errors.
"""

import sys

def test_langchain_imports():
    """Test LangChain callback imports."""
    print("Testing LangChain imports...")
    try:
        from src.token_copilot.langchain.callbacks import TokenPilotCallback
        print("  + TokenPilotCallback (langchain) imported successfully")

        from langchain_core.callbacks.base import BaseCallbackHandler
        print("  ✓ BaseCallbackHandler from langchain_core imported successfully")

        from langchain_openai import ChatOpenAI
        print("  ✓ ChatOpenAI from langchain_openai imported successfully")

        from langchain_core.prompts import PromptTemplate
        print("  ✓ PromptTemplate from langchain_core imported successfully")

        return True
    except ImportError as e:
        print(f"  X LangChain import error: {e}")
        return False


def test_llamaindex_imports():
    """Test LlamaIndex callback imports."""
    print("\nTesting LlamaIndex imports...")
    try:
        from src.token_copilot.llamaindex.callbacks import TokenPilotCallbackHandler
        print("  ✓ TokenPilotCallbackHandler (llamaindex) imported successfully")

        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
        print("  ✓ BaseCallbackHandler from llama_index.core imported successfully")

        from llama_index.core.callbacks.schema import CBEventType
        print("  ✓ CBEventType from llama_index.core imported successfully")

        from llama_index.core import VectorStoreIndex, Document
        print("  ✓ VectorStoreIndex and Document imported successfully")

        return True
    except ImportError as e:
        print(f"  ✗ LlamaIndex import error: {e}")
        return False


def test_langgraph_imports():
    """Test LangGraph imports."""
    print("\nTesting LangGraph imports...")
    try:
        from langgraph.graph import StateGraph, START, END
        print("  ✓ StateGraph, START, END from langgraph imported successfully")

        from langgraph.graph.message import add_messages
        print("  ✓ add_messages from langgraph imported successfully")

        return True
    except ImportError as e:
        print(f"  ✗ LangGraph import error: {e}")
        return False


def test_token_copilot_integration():
    """Test token_copilot integration imports."""
    print("\nTesting token_copilot integration...")
    try:
        from src.token_copilot import TokenPilotCallback, TokenPilotCallbackHandler
        print("  ✓ Main callbacks imported successfully")

        from src.token_copilot.routing import ModelConfig, RoutingStrategy
        print("  ✓ Routing components imported successfully")

        from src.token_copilot.queuing import QueueMode
        print("  ✓ Queuing components imported successfully")

        from src.token_copilot.analytics import log_alert
        print("  ✓ Analytics components imported successfully")

        return True
    except ImportError as e:
        print(f"  ✗ token_copilot import error: {e}")
        return False


def main():
    """Run all import tests."""
    print("=" * 70)
    print("IMPORT VERIFICATION TEST")
    print("=" * 70)
    print("\nThis script verifies that all import fixes are working correctly.\n")

    results = {
        "LangChain": test_langchain_imports(),
        "LlamaIndex": test_llamaindex_imports(),
        "LangGraph": test_langgraph_imports(),
        "token_copilot": test_token_copilot_integration()
    }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    all_passed = True
    for package, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{package:20s} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("✓ All import tests passed!")
        print("\nKey fixes applied:")
        print("  • LangChain: Using langchain_core.callbacks.base")
        print("  • LlamaIndex: Using llama_index.core.callbacks.base_handler")
        print("  • LangGraph: Using langgraph.graph (already correct)")
        print("  • Examples: Updated to use LCEL (LangChain Expression Language)")
        print("    instead of deprecated LLMChain")
        return 0
    else:
        print("✗ Some import tests failed. Please check the errors above.")
        print("\nNote: If packages are not installed, install them with:")
        print("  pip install langchain-core langchain-openai")
        print("  pip install llama-index-core")
        print("  pip install langgraph")
        return 1


if __name__ == "__main__":
    sys.exit(main())
