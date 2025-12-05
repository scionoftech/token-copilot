"""Tests for PersistencePlugin."""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from token_copilot.core.copilot import TokenCoPilot
from token_copilot.plugins.persistence import (
    PersistencePlugin,
    PersistenceBackend,
    SQLiteBackend,
    JSONBackend,
)


class TestPersistenceBackend:
    """Test PersistenceBackend abstract class."""

    def test_backend_is_abstract(self):
        """Test that PersistenceBackend cannot be instantiated directly."""
        backend = PersistenceBackend()

        with pytest.raises(NotImplementedError):
            backend.save_event({})

        with pytest.raises(NotImplementedError):
            backend.get_events()

        with pytest.raises(NotImplementedError):
            backend.get_summary()


class TestSQLiteBackend:
    """Test SQLiteBackend."""

    def setup_method(self):
        """Set up test database."""
        # Use temporary file
        self.temp_db = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        )
        self.temp_db.close()
        self.db_path = self.temp_db.name

    def teardown_method(self):
        """Clean up test database."""
        # Give Windows time to release file locks
        import time
        import gc
        gc.collect()  # Force garbage collection to close any open connections
        time.sleep(0.1)  # Brief wait for file handles to release

        if os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except PermissionError:
                # File still locked, skip cleanup (will be overwritten next run)
                pass

    def test_initialization(self):
        """Test SQLiteBackend initialization."""
        backend = SQLiteBackend(self.db_path)
        assert backend.db_path == Path(self.db_path)
        assert backend.conn is not None
        backend.close()

    def test_table_creation(self):
        """Test that tables are created on initialization."""
        backend = SQLiteBackend(self.db_path)

        cursor = backend.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cost_events'"
        )
        result = cursor.fetchone()

        assert result is not None
        assert result[0] == "cost_events"
        backend.close()

    def test_save_event(self):
        """Test saving an event."""
        backend = SQLiteBackend(self.db_path)

        event = {
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.015,
            "metadata": {"key": "value"},
            "session_id": "session_1",
            "user_id": "alice",
        }

        backend.save_event(event)

        # Verify event was saved
        cursor = backend.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cost_events")
        count = cursor.fetchone()[0]
        assert count == 1

        backend.close()

    def test_get_events_all(self):
        """Test getting all events."""
        backend = SQLiteBackend(self.db_path)

        # Save multiple events
        for i in range(3):
            event = {
                "timestamp": datetime.now().isoformat(),
                "model": f"model_{i}",
                "input_tokens": 100 * i,
                "output_tokens": 50 * i,
                "cost": 0.01 * i,
                "metadata": {},
                "session_id": None,
                "user_id": None,
            }
            backend.save_event(event)

        events = backend.get_events()
        assert len(events) == 3
        backend.close()

    def test_get_events_filtered_by_time(self):
        """Test getting events filtered by time."""
        backend = SQLiteBackend(self.db_path)

        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        # Save events with different timestamps
        old_event = {
            "timestamp": yesterday.isoformat(),
            "model": "old_model",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.01,
            "metadata": {},
            "session_id": None,
            "user_id": None,
        }
        new_event = {
            "timestamp": now.isoformat(),
            "model": "new_model",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.01,
            "metadata": {},
            "session_id": None,
            "user_id": None,
        }

        backend.save_event(old_event)
        backend.save_event(new_event)

        # Filter by start time
        recent_events = backend.get_events(start_time=now - timedelta(hours=1))
        assert len(recent_events) == 1
        assert recent_events[0]["model"] == "new_model"

        backend.close()

    def test_get_events_filtered_by_session(self):
        """Test getting events filtered by session ID."""
        backend = SQLiteBackend(self.db_path)

        # Save events for different sessions
        for i in range(2):
            event = {
                "timestamp": datetime.now().isoformat(),
                "model": f"model_{i}",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.01,
                "metadata": {},
                "session_id": f"session_{i}",
                "user_id": None,
            }
            backend.save_event(event)

        # Filter by session
        session_events = backend.get_events(session_id="session_0")
        assert len(session_events) == 1
        assert session_events[0]["session_id"] == "session_0"

        backend.close()

    def test_get_events_filtered_by_user(self):
        """Test getting events filtered by user ID."""
        backend = SQLiteBackend(self.db_path)

        # Save events for different users
        for user in ["alice", "bob"]:
            event = {
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.01,
                "metadata": {},
                "session_id": None,
                "user_id": user,
            }
            backend.save_event(event)

        # Filter by user
        alice_events = backend.get_events(user_id="alice")
        assert len(alice_events) == 1
        assert alice_events[0]["user_id"] == "alice"

        backend.close()

    def test_get_summary(self):
        """Test getting summary statistics."""
        backend = SQLiteBackend(self.db_path)

        # Save multiple events
        events_data = [
            {"cost": 0.01, "input_tokens": 100, "output_tokens": 50},
            {"cost": 0.02, "input_tokens": 200, "output_tokens": 100},
            {"cost": 0.03, "input_tokens": 300, "output_tokens": 150},
        ]

        for data in events_data:
            event = {
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
                "cost": data["cost"],
                "metadata": {},
                "session_id": None,
                "user_id": None,
            }
            backend.save_event(event)

        summary = backend.get_summary()

        assert summary["total_calls"] == 3
        assert summary["total_cost"] == 0.06
        assert summary["total_input_tokens"] == 600
        assert summary["total_output_tokens"] == 300
        assert summary["avg_cost"] == 0.02
        assert summary["min_cost"] == 0.01
        assert summary["max_cost"] == 0.03

        backend.close()

    def test_get_summary_empty(self):
        """Test getting summary when no events exist."""
        backend = SQLiteBackend(self.db_path)

        summary = backend.get_summary()

        assert summary["total_calls"] == 0
        assert summary["total_cost"] == 0.0
        assert summary["avg_cost"] == 0.0

        backend.close()

    def test_close(self):
        """Test closing backend connection."""
        backend = SQLiteBackend(self.db_path)
        backend.close()

        # Connection should be closed
        with pytest.raises(Exception):
            cursor = backend.conn.cursor()


class TestJSONBackend:
    """Test JSONBackend."""

    def setup_method(self):
        """Set up test JSON file."""
        self.temp_json = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        )
        self.temp_json.close()
        self.json_path = self.temp_json.name

    def teardown_method(self):
        """Clean up test JSON file."""
        if os.path.exists(self.json_path):
            os.unlink(self.json_path)

    def test_initialization(self):
        """Test JSONBackend initialization."""
        backend = JSONBackend(self.json_path)
        assert backend.file_path == Path(self.json_path)
        assert backend.events == []

    def test_save_event(self):
        """Test saving an event."""
        backend = JSONBackend(self.json_path)

        event = {
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.015,
            "metadata": {"key": "value"},
        }

        backend.save_event(event)

        # Verify event was saved
        assert len(backend.events) == 1
        assert backend.events[0]["model"] == "gpt-4o-mini"

    def test_get_events_all(self):
        """Test getting all events."""
        backend = JSONBackend(self.json_path)

        # Save multiple events
        for i in range(3):
            event = {
                "timestamp": datetime.now().isoformat(),
                "model": f"model_{i}",
                "input_tokens": 100 * i,
                "output_tokens": 50 * i,
                "cost": 0.01 * i,
            }
            backend.save_event(event)

        events = backend.get_events()
        assert len(events) == 3

    def test_get_events_filtered_by_time(self):
        """Test getting events filtered by time."""
        backend = JSONBackend(self.json_path)

        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Save events with different timestamps
        old_event = {
            "timestamp": yesterday.isoformat(),
            "model": "old_model",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.01,
        }
        new_event = {
            "timestamp": now.isoformat(),
            "model": "new_model",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.01,
        }

        backend.save_event(old_event)
        backend.save_event(new_event)

        # Filter by start time
        recent_events = backend.get_events(start_time=now - timedelta(hours=1))
        assert len(recent_events) == 1
        assert recent_events[0]["model"] == "new_model"

    def test_get_summary(self):
        """Test getting summary statistics."""
        backend = JSONBackend(self.json_path)

        # Save multiple events
        events_data = [
            {"cost": 0.01, "input_tokens": 100, "output_tokens": 50},
            {"cost": 0.02, "input_tokens": 200, "output_tokens": 100},
            {"cost": 0.03, "input_tokens": 300, "output_tokens": 150},
        ]

        for data in events_data:
            event = {
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
                "cost": data["cost"],
            }
            backend.save_event(event)

        summary = backend.get_summary()

        assert summary["total_calls"] == 3
        assert summary["total_cost"] == 0.06
        assert summary["total_input_tokens"] == 600
        assert summary["total_output_tokens"] == 300
        assert summary["avg_cost"] == 0.02
        assert summary["min_cost"] == 0.01
        assert summary["max_cost"] == 0.03

    def test_persistence_across_instances(self):
        """Test that events persist across backend instances."""
        # Save events with first instance
        backend1 = JSONBackend(self.json_path)
        event = {
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.015,
        }
        backend1.save_event(event)

        # Create new instance and verify events are loaded
        backend2 = JSONBackend(self.json_path)
        assert len(backend2.events) == 1
        assert backend2.events[0]["model"] == "gpt-4o-mini"


class TestPersistencePlugin:
    """Test PersistencePlugin."""

    def setup_method(self):
        """Set up test."""
        self.mock_backend = MagicMock(spec=PersistenceBackend)
        self.copilot = MagicMock(spec=TokenCoPilot)

    def test_initialization(self):
        """Test PersistencePlugin initialization."""
        plugin = PersistencePlugin(
            backend=self.mock_backend,
            session_id="session_1",
            user_id="alice",
            auto_flush=False,
        )

        assert plugin.backend is self.mock_backend
        assert plugin.session_id == "session_1"
        assert plugin.user_id == "alice"
        assert plugin.auto_flush is False

    def test_on_attach(self):
        """Test on_attach lifecycle method."""
        plugin = PersistencePlugin(backend=self.mock_backend)
        plugin.attach(self.copilot)

        assert plugin.copilot is self.copilot

    def test_on_detach_flushes_pending(self):
        """Test that on_detach flushes pending events."""
        plugin = PersistencePlugin(
            backend=self.mock_backend,
            auto_flush=False
        )
        plugin.attach(self.copilot)

        # Add pending event
        plugin.on_cost_tracked("gpt-4o-mini", 100, 50, 0.015, {})

        # on_detach should flush
        plugin.on_detach()

        assert self.mock_backend.save_event.called
        assert self.mock_backend.close.called

    def test_on_cost_tracked_auto_flush(self):
        """Test on_cost_tracked with auto_flush=True."""
        plugin = PersistencePlugin(
            backend=self.mock_backend,
            auto_flush=True
        )
        plugin.attach(self.copilot)

        plugin.on_cost_tracked("gpt-4o-mini", 100, 50, 0.015, {"key": "value"})

        # Should save immediately
        assert self.mock_backend.save_event.called
        call_args = self.mock_backend.save_event.call_args[0][0]
        assert call_args["model"] == "gpt-4o-mini"
        assert call_args["input_tokens"] == 100
        assert call_args["output_tokens"] == 50
        assert call_args["cost"] == 0.015

    def test_on_cost_tracked_no_auto_flush(self):
        """Test on_cost_tracked with auto_flush=False."""
        plugin = PersistencePlugin(
            backend=self.mock_backend,
            auto_flush=False
        )
        plugin.attach(self.copilot)

        plugin.on_cost_tracked("gpt-4o-mini", 100, 50, 0.015, {})

        # Should buffer, not save
        assert not self.mock_backend.save_event.called
        assert len(plugin.pending_events) == 1

    def test_flush(self):
        """Test manual flush."""
        plugin = PersistencePlugin(
            backend=self.mock_backend,
            auto_flush=False
        )
        plugin.attach(self.copilot)

        # Add pending events
        plugin.on_cost_tracked("gpt-4o-mini", 100, 50, 0.015, {})
        plugin.on_cost_tracked("gpt-4o", 500, 250, 5.0, {})

        assert len(plugin.pending_events) == 2

        # Flush
        plugin.flush()

        # All events should be saved
        assert self.mock_backend.save_event.call_count == 2
        assert len(plugin.pending_events) == 0

    def test_get_events(self):
        """Test get_events."""
        plugin = PersistencePlugin(backend=self.mock_backend)
        plugin.attach(self.copilot)

        self.mock_backend.get_events.return_value = [
            {"model": "gpt-4o-mini", "cost": 0.015}
        ]

        events = plugin.get_events()

        assert len(events) == 1
        assert events[0]["model"] == "gpt-4o-mini"
        self.mock_backend.get_events.assert_called_once()

    def test_get_summary(self):
        """Test get_summary."""
        plugin = PersistencePlugin(backend=self.mock_backend)
        plugin.attach(self.copilot)

        self.mock_backend.get_summary.return_value = {
            "total_cost": 10.50,
            "total_calls": 100,
        }

        summary = plugin.get_summary()

        assert summary["total_cost"] == 10.50
        assert summary["total_calls"] == 100
        self.mock_backend.get_summary.assert_called_once()


class TestPersistenceIntegration:
    """Integration tests for persistence plugin."""

    def setup_method(self):
        """Set up integration test."""
        self.temp_db = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        )
        self.temp_db.close()
        self.db_path = self.temp_db.name

    def teardown_method(self):
        """Clean up test database."""
        # Give Windows time to release file locks
        import time
        import gc
        gc.collect()  # Force garbage collection to close any open connections
        time.sleep(0.1)  # Brief wait for file handles to release

        if os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except PermissionError:
                # File still locked, skip cleanup (will be overwritten next run)
                pass

    def test_with_persistence_builder(self):
        """Test using .with_persistence() builder method."""
        backend = SQLiteBackend(self.db_path)
        copilot = (
            TokenCoPilot(budget_limit=100.00)
            .with_persistence(backend=backend)
        )

        # Verify plugin was added
        plugins = copilot._plugin_manager.get_plugins()
        assert len(plugins) == 1
        assert isinstance(plugins[0], PersistencePlugin)

        backend.close()

    def test_end_to_end_sqlite(self):
        """Test end-to-end SQLite persistence."""
        backend = SQLiteBackend(self.db_path)
        copilot = (
            TokenCoPilot(budget_limit=100.00)
            .with_persistence(backend=backend, session_id="test_session")
        )

        # Get persistence plugin
        plugin = copilot._plugin_manager.get_plugins()[0]

        # Manually trigger cost tracking events (simulating LLM callbacks)
        plugin.on_cost_tracked("gpt-4o-mini", 100, 50, 0.015, {})
        plugin.on_cost_tracked("gpt-4o", 500, 250, 5.0, {})

        # Verify events were saved
        events = plugin.get_events()
        assert len(events) == 2

        # Verify summary
        summary = plugin.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_cost"] > 0

        backend.close()

    def test_session_filtering(self):
        """Test filtering by session ID."""
        backend = SQLiteBackend(self.db_path)

        # Create two copilots with different sessions
        copilot1 = (
            TokenCoPilot(budget_limit=100.00)
            .with_persistence(backend=SQLiteBackend(self.db_path), session_id="session_1")
        )
        copilot2 = (
            TokenCoPilot(budget_limit=100.00)
            .with_persistence(backend=SQLiteBackend(self.db_path), session_id="session_2")
        )

        # Get plugins and manually trigger cost tracking events
        plugin1 = copilot1._plugin_manager.get_plugins()[0]
        plugin2 = copilot2._plugin_manager.get_plugins()[0]

        plugin1.on_cost_tracked("gpt-4o-mini", 100, 50, 0.015, {})
        plugin2.on_cost_tracked("gpt-4o", 500, 250, 5.0, {})

        # Query by session
        backend_query = SQLiteBackend(self.db_path)
        session1_events = backend_query.get_events(session_id="session_1")
        session2_events = backend_query.get_events(session_id="session_2")

        assert len(session1_events) == 1
        assert len(session2_events) == 1
        assert session1_events[0]["session_id"] == "session_1"
        assert session2_events[0]["session_id"] == "session_2"

        backend.close()
        backend_query.close()
