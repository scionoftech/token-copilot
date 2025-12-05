"""Example: Persisting cost tracking data with PersistencePlugin.

This example demonstrates how to save cost tracking data to persistent storage
for historical analysis, cross-session budgets, and reporting.
"""

from datetime import datetime, timedelta
from token_copilot import TokenCoPilot
from token_copilot.plugins.persistence import (
    PersistencePlugin,
    SQLiteBackend,
    JSONBackend,
)


# =============================================================================
# Example 1: SQLite Persistence (Recommended for Production)
# =============================================================================

def example_sqlite_basic():
    """Basic SQLite persistence example."""
    print("=== Example 1: Basic SQLite Persistence ===\n")

    # Create copilot with SQLite persistence
    backend = SQLiteBackend(db_path="token_costs.db")
    copilot = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend)
    )

    # Simulate some LLM calls
    print("Simulating LLM calls...")
    copilot.tracker.track_cost("gpt-4o-mini", 1000, 500, 0.15, {"user": "alice"})
    copilot.tracker.track_cost("gpt-4o", 500, 250, 5.00, {"user": "bob"})
    copilot.tracker.track_cost("gpt-4o-mini", 800, 400, 0.12, {"user": "alice"})

    print(f"Total cost: ${copilot.cost:.4f}\n")

    # Query historical data
    persistence_plugin = copilot._plugin_manager.get_all_plugins()[0]
    events = persistence_plugin.get_events()
    print(f"Stored {len(events)} events in database")

    # Get summary
    summary = persistence_plugin.get_summary()
    print(f"Summary: {summary}\n")


# =============================================================================
# Example 2: Multi-Session Budget Tracking
# =============================================================================

def example_multi_session():
    """Track budget across multiple sessions."""
    print("=== Example 2: Multi-Session Budget Tracking ===\n")

    backend = SQLiteBackend(db_path="multi_session.db")

    # Session 1
    print("Session 1:")
    session1 = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend, session_id="session_1")
    )
    session1.tracker.track_cost("gpt-4o-mini", 1000, 500, 0.15)
    print(f"Session 1 cost: ${session1.cost:.4f}")

    # Session 2 (can see previous session data)
    print("\nSession 2:")
    backend2 = SQLiteBackend(db_path="multi_session.db")
    session2 = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend2, session_id="session_2")
    )
    session2.tracker.track_cost("gpt-4o", 500, 250, 5.00)
    print(f"Session 2 cost: ${session2.cost:.4f}")

    # Query all sessions
    persistence_plugin = session2._plugin_manager.get_all_plugins()[0]
    all_events = persistence_plugin.get_events()
    print(f"\nTotal events across all sessions: {len(all_events)}")

    summary = persistence_plugin.get_summary()
    print(f"Total cost across all sessions: ${summary['total_cost']:.4f}\n")


# =============================================================================
# Example 3: Multi-Tenant Tracking
# =============================================================================

def example_multi_tenant():
    """Track costs per user/organization."""
    print("=== Example 3: Multi-Tenant Tracking ===\n")

    backend = SQLiteBackend(db_path="multi_tenant.db")

    # Track costs for different users
    users = ["alice", "bob", "charlie"]
    for user in users:
        copilot = (
            TokenCoPilot(budget_limit=50.00)
            .with_persistence(backend=SQLiteBackend("multi_tenant.db"), user_id=user)
        )

        # Simulate different usage patterns
        if user == "alice":
            copilot.tracker.track_cost("gpt-4o-mini", 2000, 1000, 0.30)
        elif user == "bob":
            copilot.tracker.track_cost("gpt-4o", 1000, 500, 10.00)
        else:
            copilot.tracker.track_cost("gpt-4o-mini", 500, 250, 0.08)

        print(f"{user}: ${copilot.cost:.4f}")

    # Query per-user costs (would need to extend backend with user filtering)
    print("\nTotal: Multi-tenant tracking enabled\n")


# =============================================================================
# Example 4: JSON Persistence (Simple File-Based)
# =============================================================================

def example_json_persistence():
    """Use JSON file for simple persistence."""
    print("=== Example 4: JSON File Persistence ===\n")

    # Create copilot with JSON persistence
    backend = JSONBackend(file_path="token_costs.json")
    copilot = (
        TokenCoPilot(budget_limit=50.00)
        .with_persistence(backend=backend)
    )

    # Simulate some LLM calls
    print("Simulating LLM calls...")
    copilot.tracker.track_cost("gpt-4o-mini", 1000, 500, 0.15)
    copilot.tracker.track_cost("gpt-4o-mini", 800, 400, 0.12)

    print(f"Total cost: ${copilot.cost:.4f}")
    print("Events saved to token_costs.json\n")


# =============================================================================
# Example 5: Time-Based Queries
# =============================================================================

def example_time_queries():
    """Query costs by time period."""
    print("=== Example 5: Time-Based Queries ===\n")

    backend = SQLiteBackend(db_path="time_tracking.db")
    copilot = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend)
    )

    # Simulate some LLM calls
    print("Simulating LLM calls...")
    copilot.tracker.track_cost("gpt-4o-mini", 1000, 500, 0.15)
    copilot.tracker.track_cost("gpt-4o", 500, 250, 5.00)
    copilot.tracker.track_cost("gpt-4o-mini", 800, 400, 0.12)

    # Query last hour
    persistence_plugin = copilot._plugin_manager.get_all_plugins()[0]
    now = datetime.now()
    last_hour = now - timedelta(hours=1)

    recent_events = persistence_plugin.get_events(start_time=last_hour)
    recent_summary = persistence_plugin.get_summary(start_time=last_hour)

    print(f"Events in last hour: {len(recent_events)}")
    print(f"Cost in last hour: ${recent_summary['total_cost']:.4f}\n")


# =============================================================================
# Example 6: Batch Flushing for Performance
# =============================================================================

def example_batch_flushing():
    """Use batch flushing for better performance."""
    print("=== Example 6: Batch Flushing ===\n")

    backend = SQLiteBackend(db_path="batch_tracking.db")
    copilot = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend, auto_flush=False)
    )

    # Simulate many LLM calls (events are buffered)
    print("Simulating 100 LLM calls...")
    for i in range(100):
        copilot.tracker.track_cost("gpt-4o-mini", 100, 50, 0.015)

    print(f"Total cost: ${copilot.cost:.4f}")

    # Flush all at once (more efficient than per-event writes)
    persistence_plugin = copilot._plugin_manager.get_all_plugins()[0]
    persistence_plugin.flush()
    print("Batch flushed to database\n")


# =============================================================================
# Example 7: Builder Pattern with Multiple Plugins
# =============================================================================

def example_builder_with_persistence():
    """Use persistence with other plugins."""
    print("=== Example 7: Persistence with Other Plugins ===\n")

    from token_copilot.plugins import AnalyticsPlugin

    backend = SQLiteBackend(db_path="full_tracking.db")
    copilot = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend, session_id="prod_session")
        .with_analytics(detect_anomalies=True)
    )

    # Simulate LLM calls
    print("Simulating LLM calls with analytics + persistence...")
    copilot.tracker.track_cost("gpt-4o-mini", 1000, 500, 0.15)
    copilot.tracker.track_cost("gpt-4o", 500, 250, 5.00)

    print(f"Total cost: ${copilot.cost:.4f}")
    print("Events saved to database + analytics enabled\n")


# =============================================================================
# Example 8: Generating Reports
# =============================================================================

def example_generate_report():
    """Generate cost reports from persisted data."""
    print("=== Example 8: Generate Cost Report ===\n")

    backend = SQLiteBackend(db_path="reports.db")
    copilot = (
        TokenCoPilot(budget_limit=100.00)
        .with_persistence(backend=backend)
    )

    # Simulate some LLM calls
    for _ in range(10):
        copilot.tracker.track_cost("gpt-4o-mini", 1000, 500, 0.15)
    for _ in range(5):
        copilot.tracker.track_cost("gpt-4o", 500, 250, 5.00)

    # Generate report
    persistence_plugin = copilot._plugin_manager.get_all_plugins()[0]
    summary = persistence_plugin.get_summary()

    print("=== COST REPORT ===")
    print(f"Total API Calls: {summary['total_calls']}")
    print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
    print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
    print(f"Total Cost: ${summary['total_cost']:.2f}")
    print(f"Average Cost per Call: ${summary['avg_cost']:.4f}")
    print(f"Min Cost: ${summary['min_cost']:.4f}")
    print(f"Max Cost: ${summary['max_cost']:.4f}")
    print()


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    example_sqlite_basic()
    example_multi_session()
    example_multi_tenant()
    example_json_persistence()
    example_time_queries()
    example_batch_flushing()
    example_builder_with_persistence()
    example_generate_report()

    print("✅ All persistence examples completed!")
    print("\nDatabase files created:")
    print("  - token_costs.db")
    print("  - multi_session.db")
    print("  - multi_tenant.db")
    print("  - time_tracking.db")
    print("  - batch_tracking.db")
    print("  - full_tracking.db")
    print("  - reports.db")
    print("\nJSON file created:")
    print("  - token_costs.json")
