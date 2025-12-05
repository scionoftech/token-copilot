"""
Example usage of token_copilot with LangGraph.

This example demonstrates how to track costs in LangGraph applications,
including StateGraph, agent workflows, and multi-step reasoning.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from token_copilot import TokenCoPilotCallback


# Example 1: Simple StateGraph with cost tracking
def simple_graph_example():
    """Basic LangGraph example with cost tracking."""

    # Define state
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Create callback
    callback = TokenCoPilotCallback(
        budget_limit=10.00,
        budget_period="total"
    )

    # Create LLM with callback
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Define node
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    # Build graph
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    # Run with tracking
    result = graph.invoke(
        {"messages": [("user", "What is LangGraph?")]},
        config={"callbacks": [callback]}
    )

    # Get analytics
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    print(f"Total tokens: {callback.get_total_tokens():,}")
    print(f"Remaining budget: ${callback.get_remaining_budget():.2f}")

    return result


# Example 2: Multi-agent workflow with per-user tracking
def multi_agent_example():
    """LangGraph multi-agent workflow with user-specific tracking."""

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        next: str

    # Create callback with per-user budget
    callback = TokenCoPilotCallback(
        budget_limit=5.00,
        budget_period="per_user",  # Separate budget per user
        anomaly_detection=True
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # Agent nodes
    def researcher(state: State):
        response = llm.invoke([
            ("system", "You are a researcher. Provide factual information."),
            *state["messages"]
        ])
        return {"messages": [response]}

    def writer(state: State):
        response = llm.invoke([
            ("system", "You are a writer. Make information engaging."),
            *state["messages"]
        ])
        return {"messages": [response]}

    def router(state: State):
        # Simple routing logic
        last_message = state["messages"][-1].content.lower()
        if "research" in last_message:
            return "researcher"
        return "writer"

    # Build graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_conditional_edges(START, router)
    builder.add_edge("researcher", END)
    builder.add_edge("writer", END)
    graph = builder.compile()

    # Run for multiple users
    users = ["user_alice", "user_bob"]

    for user_id in users:
        print(f"\n--- Processing for {user_id} ---")

        result = graph.invoke(
            {"messages": [("user", "Research the benefits of LangGraph")]},
            config={
                "callbacks": [callback],
                "metadata": {"user_id": user_id}
            }
        )

        # Per-user analytics
        costs_by_user = callback.get_costs_by_user()
        print(f"{user_id} spent: ${costs_by_user.get(user_id, 0):.4f}")

    # Overall analytics
    print(f"\n--- Overall Analytics ---")
    print(f"Total cost: ${callback.get_total_cost():.4f}")
    df = callback.to_dataframe()
    print("\nCosts by user:")
    print(df.groupby('user_id')['cost'].sum())


# Example 3: Agentic RAG with budget forecasting
def agentic_rag_example():
    """LangGraph agentic RAG with predictive budget alerts."""

    class State(TypedDict):
        question: str
        context: str
        answer: str
        steps: list

    # Create callback with forecasting
    callback = TokenCoPilotCallback(
        budget_limit=100.00,
        predictive_alerts=True,
        forecast_window_hours=24
    )

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

    # RAG nodes
    def retrieve(state: State):
        # Simulate retrieval (in real app, query vector DB)
        return {
            "context": "LangGraph is a framework for building stateful agents...",
            "steps": state["steps"] + ["retrieved"]
        }

    def generate(state: State):
        response = llm.invoke([
            ("system", f"Answer based on this context: {state['context']}"),
            ("user", state["question"])
        ])
        return {
            "answer": response.content,
            "steps": state["steps"] + ["generated"]
        }

    def should_continue(state: State):
        # Check if we need more context
        if len(state["steps"]) < 2:
            return "retrieve"
        return "end"

    # Build graph
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges("retrieve", should_continue, {
        "retrieve": "retrieve",
        "end": "generate"
    })
    builder.add_edge("generate", END)
    graph = builder.compile()

    # Run queries
    questions = [
        "What is LangGraph?",
        "How does LangGraph differ from LangChain?",
        "What are the key features of LangGraph?"
    ]

    for question in questions:
        result = graph.invoke(
            {"question": question, "context": "", "answer": "", "steps": []},
            config={"callbacks": [callback]}
        )
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")

    # Budget forecast
    forecast = callback.get_forecast()
    print(f"\n--- Budget Forecast ---")
    print(f"Current cost: ${forecast.current_cost:.4f}")
    print(f"Remaining: ${forecast.remaining_budget:.2f}")
    if forecast.hours_until_exhausted:
        print(f"Budget exhausts in: {forecast.hours_until_exhausted:.1f} hours")
    print(f"Projected 24h cost: ${forecast.projected_cost_24h:.2f}")

    for rec in forecast.recommendations:
        print(f"  - {rec}")


# Example 4: Advanced features (routing + queuing)
def advanced_features_example():
    """LangGraph with model routing and request queuing."""

    from token_copilot.routing import ModelConfig, RoutingStrategy
    from token_copilot.queuing import QueueMode

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        complexity: str

    # Define available models
    models = [
        ModelConfig("gpt-4o-mini", quality_score=0.7, cost_per_1k_input=0.15,
                   cost_per_1k_output=0.60, max_tokens=128000),
        ModelConfig("gpt-4o", quality_score=0.9, cost_per_1k_input=5.0,
                   cost_per_1k_output=15.0, max_tokens=128000),
    ]

    # Create callback with all features
    callback = TokenCoPilotCallback(
        budget_limit=50.00,
        auto_routing=True,
        routing_models=models,
        routing_strategy=RoutingStrategy.BALANCED,
        queue_mode=QueueMode.SMART,
        anomaly_detection=True
    )

    # Get model suggestions for different complexities
    prompts = {
        "simple": "What is 2+2?",
        "medium": "Explain machine learning in simple terms.",
        "complex": "Design a distributed system architecture for a high-traffic application."
    }

    print("--- Model Routing Suggestions ---")
    for complexity, prompt in prompts.items():
        decision = callback.suggest_model(prompt, estimated_tokens=1000)
        print(f"\n{complexity.capitalize()} query:")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Suggested model: {decision.selected_model}")
        print(f"  Estimated cost: ${decision.estimated_cost:.4f}")
        print(f"  Reason: {decision.reason}")

    # Queue stats
    queue_stats = callback.get_queue_stats()
    if queue_stats:
        print(f"\n--- Queue Statistics ---")
        print(f"Current queue size: {queue_stats['current_size']}")
        print(f"Total processed: {queue_stats['total_processed']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Simple LangGraph with Cost Tracking")
    print("=" * 60)
    simple_graph_example()

    print("\n" + "=" * 60)
    print("Example 2: Multi-Agent Workflow with Per-User Tracking")
    print("=" * 60)
    multi_agent_example()

    print("\n" + "=" * 60)
    print("Example 3: Agentic RAG with Budget Forecasting")
    print("=" * 60)
    agentic_rag_example()

    print("\n" + "=" * 60)
    print("Example 4: Advanced Features (Routing + Queuing)")
    print("=" * 60)
    advanced_features_example()
