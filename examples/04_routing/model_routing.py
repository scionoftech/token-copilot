"""
Model Routing Example - Auto-Select Optimal Models

This example demonstrates automatic model selection based on:
- Request complexity (simple, medium, complex)
- Cost optimization
- Quality requirements
- Different routing strategies
"""

from langchain_openai import ChatOpenAI
from token_copilot import TokenPilotCallback
from token_copilot.routing import ModelConfig, RoutingStrategy


def setup_models():
    """Define available models with their characteristics."""
    return [
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
        ModelConfig(
            name="gpt-3.5-turbo",
            quality_score=0.6,
            cost_per_1k_input=0.50,
            cost_per_1k_output=1.50,
            max_tokens=16385,
            supports_functions=True,
            supports_vision=False
        ),
    ]


def basic_routing():
    """Example: Basic model routing."""
    print("=" * 60)
    print("Example 1: Basic Model Routing")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        budget_limit=10.00,
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED
    )

    # Test different complexity prompts
    prompts = {
        "simple": "What is 2+2?",
        "medium": "Explain machine learning in simple terms.",
        "complex": "Design a distributed system architecture for a high-traffic "
                   "e-commerce application with microservices."
    }

    print("\nModel Suggestions:")
    print("-" * 60)

    for complexity, prompt in prompts.items():
        decision = callback.suggest_model(prompt, estimated_tokens=1000)

        print(f"\n{complexity.upper()} Query:")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  ✅ Suggested model: {decision.selected_model}")
        print(f"  💰 Estimated cost: ${decision.estimated_cost:.4f}")
        print(f"  ⭐ Quality score: {decision.quality_score:.2f}")
        print(f"  📝 Reason: {decision.reason}")

        if decision.alternatives:
            print(f"  🔄 Alternatives:")
            for alt in decision.alternatives:
                print(f"     - {alt['model']}: ${alt['cost']:.4f} (quality: {alt['quality']:.2f})")


def cheapest_first_strategy():
    """Example: Always use cheapest model."""
    print("\n" + "=" * 60)
    print("Example 2: CHEAPEST_FIRST Strategy")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.CHEAPEST_FIRST
    )

    prompts = [
        "What is Python?",
        "Explain quantum computing in detail.",
        "Write a complex algorithm for graph traversal."
    ]

    print("\nAlways routes to cheapest model:")
    print("-" * 60)

    for prompt in prompts:
        decision = callback.suggest_model(prompt, estimated_tokens=500)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Model: {decision.selected_model}")
        print(f"  Cost: ${decision.estimated_cost:.4f}")


def quality_first_strategy():
    """Example: Always use highest quality model."""
    print("\n" + "=" * 60)
    print("Example 3: QUALITY_FIRST Strategy")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.QUALITY_FIRST
    )

    prompts = [
        "What is 2+2?",
        "Explain machine learning.",
        "Design a database schema."
    ]

    print("\nAlways routes to highest quality model:")
    print("-" * 60)

    for prompt in prompts:
        decision = callback.suggest_model(prompt, estimated_tokens=500)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Model: {decision.selected_model}")
        print(f"  Quality: {decision.quality_score:.2f}")
        print(f"  Cost: ${decision.estimated_cost:.4f}")


def balanced_strategy():
    """Example: Balance cost and quality by complexity."""
    print("\n" + "=" * 60)
    print("Example 4: BALANCED Strategy (Default)")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED
    )

    # Different complexity levels
    test_cases = [
        ("Simple", "Hi", 100),
        ("Simple", "What is the capital of France?", 200),
        ("Medium", "Explain how transformers work in NLP.", 500),
        ("Complex", "Design a scalable microservices architecture with detailed "
                   "component breakdown and data flow diagrams.", 2000),
    ]

    print("\nBalanced routing (cost/quality by complexity):")
    print("-" * 60)

    for complexity, prompt, tokens in test_cases:
        decision = callback.suggest_model(prompt, estimated_tokens=tokens)
        print(f"\n[{complexity}] {prompt[:40]}...")
        print(f"  Model: {decision.selected_model}")
        print(f"  Cost: ${decision.estimated_cost:.4f}, Quality: {decision.quality_score:.2f}")
        print(f"  Reason: {decision.reason}")


def cost_threshold_strategy():
    """Example: Use cheapest under threshold, else best quality."""
    print("\n" + "=" * 60)
    print("Example 5: COST_THRESHOLD Strategy")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.COST_THRESHOLD,
        # Note: cost_threshold is set in router, default is $0.01
    )

    test_cases = [
        (100, "Small request"),
        (1000, "Medium request"),
        (5000, "Large request"),
    ]

    print("\nUses cheapest if under threshold, else best quality:")
    print("-" * 60)

    for tokens, desc in test_cases:
        decision = callback.suggest_model(desc, estimated_tokens=tokens)
        print(f"\n{desc} ({tokens} tokens):")
        print(f"  Model: {decision.selected_model}")
        print(f"  Cost: ${decision.estimated_cost:.4f}")
        print(f"  Reason: {decision.reason}")


def learned_routing():
    """Example: Learn from quality feedback."""
    print("\n" + "=" * 60)
    print("Example 6: LEARNED Strategy (Quality-Based)")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.LEARNED
    )

    # Simulate usage with quality feedback
    print("\nSimulating quality feedback...")
    print("-" * 60)

    # Record quality scores for different models
    feedback = [
        ("gpt-4o-mini", 0.75),
        ("gpt-4o-mini", 0.70),
        ("gpt-4o-mini", 0.72),
        ("gpt-4o", 0.92),
        ("gpt-4o", 0.95),
        ("gpt-4o", 0.93),
        ("gpt-3.5-turbo", 0.60),
        ("gpt-3.5-turbo", 0.58),
    ]

    for model, quality in feedback:
        callback.record_model_quality(model, quality)
        print(f"  Recorded: {model} → {quality:.2f}")

    # Get model stats
    print("\n📊 Model Statistics:")
    stats = callback.get_model_stats()
    for model, data in stats.items():
        print(f"\n  {model}:")
        print(f"    Avg quality: {data['avg_quality']:.2f}")
        print(f"    Calls: {data['calls']}")

    # Now routing uses learned quality data
    print("\n🎯 Routing with learned quality:")
    decision = callback.suggest_model("Medium complexity task", estimated_tokens=1000)
    print(f"  Model: {decision.selected_model}")
    print(f"  Reason: {decision.reason}")


def capability_filtering():
    """Example: Filter by required capabilities."""
    print("\n" + "=" * 60)
    print("Example 7: Capability-Based Filtering")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED
    )

    # Request with function calling required
    print("\n1. Requiring function calling:")
    decision = callback.suggest_model(
        "Call a function",
        estimated_tokens=500,
        metadata={'required_capabilities': {'functions': True}}
    )
    print(f"  Model: {decision.selected_model}")
    print(f"  Supports functions: Yes")

    # Request with vision required
    print("\n2. Requiring vision:")
    decision = callback.suggest_model(
        "Analyze an image",
        estimated_tokens=500,
        metadata={'required_capabilities': {'vision': True}}
    )
    print(f"  Model: {decision.selected_model}")
    print(f"  Supports vision: Yes")


def actual_usage():
    """Example: Using routing suggestions in actual LLM calls."""
    print("\n" + "=" * 60)
    print("Example 8: Actual Usage with Routing")
    print("=" * 60)

    models = setup_models()

    callback = TokenPilotCallback(
        budget_limit=1.00,
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED
    )

    prompts = [
        "What is 2+2?",
        "Explain neural networks.",
    ]

    print("\nMaking actual LLM calls with routed models:")
    print("-" * 60)

    for prompt in prompts:
        # Get suggestion
        decision = callback.suggest_model(prompt, estimated_tokens=500)

        print(f"\nPrompt: {prompt}")
        print(f"  Routing to: {decision.selected_model}")
        print(f"  Estimated: ${decision.estimated_cost:.4f}")

        # Use suggested model
        llm = ChatOpenAI(model=decision.selected_model, callbacks=[callback])
        response = llm.invoke(prompt)

        # Actual cost
        print(f"  Actual cost: ${callback.get_total_cost():.4f}")

        # Record quality (simulated)
        quality_score = 0.8  # In practice, measure from user feedback
        callback.record_model_quality(decision.selected_model, quality_score)
        print(f"  Quality recorded: {quality_score:.2f}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 16 + "MODEL ROUTING EXAMPLES" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run all examples
    basic_routing()
    cheapest_first_strategy()
    quality_first_strategy()
    balanced_strategy()
    cost_threshold_strategy()
    learned_routing()
    capability_filtering()
    actual_usage()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nRouting Strategies Summary:")
    print("  • CHEAPEST_FIRST: Minimize costs")
    print("  • QUALITY_FIRST: Maximize quality")
    print("  • BALANCED: Adapt to complexity")
    print("  • COST_THRESHOLD: Budget-aware routing")
    print("  • LEARNED: Use historical quality data")
    print("\nBest Practices:")
    print("  • Use BALANCED for general applications")
    print("  • Use LEARNED with quality feedback loop")
    print("  • Record quality scores for continuous improvement")
    print()


if __name__ == "__main__":
    main()
