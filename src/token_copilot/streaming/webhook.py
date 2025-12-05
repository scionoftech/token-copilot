"""Webhook streaming for HTTP endpoints."""

import logging
import time
import threading
from typing import List, Optional, Dict, Any
from queue import Queue, Empty

from .base import BaseStreamer, StreamEvent

logger = logging.getLogger(__name__)


class WebhookStreamer(BaseStreamer):
    """
    Stream cost events to HTTP webhook endpoints.

    Supports batching, async sending, and automatic retries.

    Example:
        ```python
        from token_copilot import TokenPilotCallback
        from token_copilot.streaming import WebhookStreamer

        # Simple webhook
        streamer = WebhookStreamer(url="https://analytics.company.com/ingest")

        # With batching for efficiency
        streamer = WebhookStreamer(
            url="https://analytics.company.com/ingest",
            batch_size=10,
            flush_interval=5.0,
            headers={"Authorization": "Bearer token123"}
        )

        callback = TokenPilotCallback(streamer=streamer)
        ```
    """

    def __init__(
        self,
        url: str,
        batch_size: int = 1,
        flush_interval: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        async_mode: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize webhook streamer.

        Args:
            url: Webhook URL to POST events to
            batch_size: Number of events to batch before sending (default: 1)
            flush_interval: Seconds to wait before flushing batch (default: 5.0)
            headers: Optional HTTP headers dict
            timeout: Request timeout in seconds (default: 10.0)
            max_retries: Maximum retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            async_mode: Send in background thread (default: True)
            enabled: Whether streaming is enabled (default: True)
        """
        super().__init__(enabled=enabled)
        self.url = url
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.async_mode = async_mode

        self._batch: List[StreamEvent] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()

        # Background thread for async mode
        self._queue: Optional[Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        if self.async_mode:
            self._start_worker()

    def _start_worker(self):
        """Start background worker thread."""
        self._queue = Queue(maxsize=1000)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="webhook-streamer-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self):
        """Background worker loop."""
        while not self._shutdown.is_set():
            try:
                # Get event with timeout
                event = self._queue.get(timeout=self.flush_interval)
                self._add_to_batch(event)
            except Empty:
                # Timeout - flush batch if needed
                self._flush_if_needed()
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

        # Final flush on shutdown
        self._flush_batch()

    def _add_to_batch(self, event: StreamEvent):
        """Add event to batch and flush if needed."""
        with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self.batch_size:
                self._flush_batch()

    def _flush_if_needed(self):
        """Flush batch if interval exceeded."""
        with self._lock:
            if self._batch and (time.time() - self._last_flush) >= self.flush_interval:
                self._flush_batch()

    def _flush_batch(self):
        """Flush current batch to webhook (must hold lock)."""
        if not self._batch:
            return

        batch_to_send = self._batch[:]
        self._batch.clear()
        self._last_flush = time.time()

        # Send outside lock
        self._send_batch(batch_to_send)

    def _send_batch(self, events: List[StreamEvent]):
        """Send batch of events to webhook."""
        try:
            import requests
        except ImportError:
            logger.warning(
                "requests library required for webhook streaming. "
                "Install with: pip install token-copilot[streaming] "
                "or pip install requests"
            )
            return

        # Convert to JSON payload
        if len(events) == 1:
            payload = events[0].to_dict()
        else:
            payload = {"events": [e.to_dict() for e in events]}

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                logger.debug(f"Sent {len(events)} event(s) to webhook")
                return
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Webhook send failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Webhook send failed after {self.max_retries} attempts: {e}")

    def send(self, event: StreamEvent) -> bool:
        """
        Send event to webhook.

        Args:
            event: StreamEvent to send

        Returns:
            True if queued/sent successfully, False on error
        """
        try:
            if self.async_mode and self._queue:
                # Queue for background sending
                try:
                    self._queue.put_nowait(event)
                    return True
                except:
                    logger.warning("Webhook queue full, dropping event")
                    return False
            else:
                # Synchronous sending
                self._add_to_batch(event)
                self._flush_if_needed()
                return True
        except Exception as e:
            logger.error(f"Webhook send error: {e}")
            return False

    def flush(self):
        """Force flush current batch."""
        with self._lock:
            self._flush_batch()

    def close(self):
        """Close streamer and flush remaining events."""
        if self.async_mode and self._worker_thread:
            self._shutdown.set()
            self._worker_thread.join(timeout=5.0)
        else:
            self.flush()
