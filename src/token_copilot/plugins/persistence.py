"""Persistence plugin for saving cost tracking data."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import sqlite3
from pathlib import Path

from ..core.plugin import Plugin


class PersistenceBackend:
    """Base class for persistence backends."""

    def save_event(self, event: Dict[str, Any]) -> None:
        """Save a tracking event."""
        raise NotImplementedError

    def get_events(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve tracking events."""
        raise NotImplementedError

    def get_summary(self, period: str = "total") -> Dict[str, Any]:
        """Get cost summary for a period."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the backend connection."""
        pass


class SQLiteBackend(PersistenceBackend):
    """SQLite-based persistence backend."""

    def __init__(self, db_path: str = "token_copilot.db"):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                metadata TEXT,
                session_id TEXT,
                user_id TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON cost_events(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session ON cost_events(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user ON cost_events(user_id)
        """)
        self.conn.commit()

    def save_event(self, event: Dict[str, Any]) -> None:
        """Save a tracking event to SQLite."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO cost_events
            (timestamp, model, input_tokens, output_tokens, cost, metadata, session_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["timestamp"],
                event["model"],
                event["input_tokens"],
                event["output_tokens"],
                event["cost"],
                json.dumps(event.get("metadata", {})),
                event.get("session_id"),
                event.get("user_id"),
            ),
        )
        self.conn.commit()

    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve tracking events from SQLite."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM cost_events WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            events.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "model": row[2],
                    "input_tokens": row[3],
                    "output_tokens": row[4],
                    "cost": row[5],
                    "metadata": json.loads(row[6]) if row[6] else {},
                    "session_id": row[7],
                    "user_id": row[8],
                }
            )
        return events

    def get_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost summary from SQLite."""
        cursor = self.conn.cursor()
        query = """
            SELECT
                COUNT(*) as total_calls,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cost) as total_cost,
                AVG(cost) as avg_cost,
                MIN(cost) as min_cost,
                MAX(cost) as max_cost
            FROM cost_events
            WHERE 1=1
        """
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        cursor.execute(query, params)
        row = cursor.fetchone()

        return {
            "total_calls": row[0] or 0,
            "total_input_tokens": row[1] or 0,
            "total_output_tokens": row[2] or 0,
            "total_cost": row[3] or 0.0,
            "avg_cost": row[4] or 0.0,
            "min_cost": row[5] or 0.0,
            "max_cost": row[6] or 0.0,
        }

    def close(self) -> None:
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()


class JSONBackend(PersistenceBackend):
    """JSON file-based persistence backend."""

    def __init__(self, file_path: str = "token_copilot_events.json"):
        """Initialize JSON backend.

        Args:
            file_path: Path to JSON file
        """
        self.file_path = Path(file_path)
        self.events = []
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            try:
                with open(self.file_path, "r") as f:
                    self.events = json.load(f)
            except json.JSONDecodeError:
                # Empty or invalid JSON file, start fresh
                self.events = []

    def save_event(self, event: Dict[str, Any]) -> None:
        """Save a tracking event to JSON."""
        self.events.append(event)
        with open(self.file_path, "w") as f:
            json.dump(self.events, f, indent=2)

    def get_events(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve tracking events from JSON."""
        filtered = self.events

        if start_time:
            filtered = [
                e for e in filtered if datetime.fromisoformat(e["timestamp"]) >= start_time
            ]
        if end_time:
            filtered = [
                e for e in filtered if datetime.fromisoformat(e["timestamp"]) <= end_time
            ]

        return filtered

    def get_summary(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost summary from JSON."""
        events = self.get_events(start_time, end_time)

        if not events:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "avg_cost": 0.0,
                "min_cost": 0.0,
                "max_cost": 0.0,
            }

        costs = [e["cost"] for e in events]
        return {
            "total_calls": len(events),
            "total_input_tokens": sum(e["input_tokens"] for e in events),
            "total_output_tokens": sum(e["output_tokens"] for e in events),
            "total_cost": sum(costs),
            "avg_cost": sum(costs) / len(costs),
            "min_cost": min(costs),
            "max_cost": max(costs),
        }


class PersistencePlugin(Plugin):
    """Plugin for persisting cost tracking data."""

    def __init__(
        self,
        backend: PersistenceBackend,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        auto_flush: bool = True,
    ):
        """Initialize persistence plugin.

        Args:
            backend: Persistence backend to use
            session_id: Optional session identifier for tracking
            user_id: Optional user identifier for multi-tenant tracking
            auto_flush: Whether to save events immediately (default: True)
        """
        self.backend = backend
        self.session_id = session_id
        self.user_id = user_id
        self.auto_flush = auto_flush
        self.pending_events = []

    def on_attach(self) -> None:
        """Called when plugin is attached to TokenCoPilot."""
        pass

    def on_detach(self) -> None:
        """Called when plugin is detached from TokenCoPilot."""
        if self.pending_events:
            self.flush()
        self.backend.close()

    def on_cost_tracked(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Called when costs are tracked.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in dollars
            metadata: Additional metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "metadata": metadata,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }

        if self.auto_flush:
            self.backend.save_event(event)
        else:
            self.pending_events.append(event)

    def flush(self) -> None:
        """Flush pending events to backend."""
        for event in self.pending_events:
            self.backend.save_event(event)
        self.pending_events.clear()

    def get_events(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve tracking events.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            List of tracking events
        """
        return self.backend.get_events(start_time, end_time)

    def get_summary(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost summary.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            Summary statistics
        """
        return self.backend.get_summary(start_time, end_time)
