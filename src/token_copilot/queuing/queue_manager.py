"""Smart request queuing with priority management."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import heapq
import threading


class QueueMode(str, Enum):
    """Queue mode options."""

    DISABLED = "disabled"  # No queuing
    SOFT = "soft"  # Queue at 80% budget
    HARD = "hard"  # Queue at 90% budget
    SMART = "smart"  # Adaptive by priority


class Priority(int, Enum):
    """Request priority levels."""

    CRITICAL = 1  # Never queues
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass(order=True)
class QueuedRequest:
    """Queued request with priority."""

    priority: int = field(compare=True)
    timestamp: datetime = field(compare=False)
    request_id: str = field(compare=False)
    callback: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    estimated_cost: float = field(default=0.0, compare=False)


class QueueManager:
    """
    Manages priority queue for LLM requests.

    Queue modes:
    - DISABLED: No queuing
    - SOFT: Queue at 80% budget (except CRITICAL)
    - HARD: Queue at 90% budget (except CRITICAL)
    - SMART: Adaptive by priority (CRITICAL never, HIGH at 90%, NORMAL at 80%, LOW at 70%)

    Priority levels:
    - CRITICAL (1): Never queues
    - HIGH (2): Queues only at high thresholds
    - NORMAL (3): Default behavior
    - LOW (4): Queues earliest

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> from token_copilot.queuing import QueueMode
        >>> callback = TokenPilotCallback(
        ...     budget_limit=100.00,
        ...     queue_mode=QueueMode.SMART,
        ...     max_queue_size=1000
        ... )
        >>> # Queue a request
        >>> callback.queue_manager.enqueue(
        ...     request_id="req_123",
        ...     callback=my_function,
        ...     args=(arg1, arg2),
        ...     priority=Priority.HIGH,
        ...     estimated_cost=0.05
        ... )
    """

    def __init__(
        self,
        mode: QueueMode = QueueMode.DISABLED,
        max_size: int = 1000,
        max_wait_seconds: int = 300,
    ):
        """
        Initialize QueueManager.

        Args:
            mode: Queue mode
            max_size: Maximum queue size
            max_wait_seconds: Maximum wait time before dropping request
        """
        self.mode = mode
        self.max_size = max_size
        self.max_wait_seconds = max_wait_seconds

        # Priority queue (heapq)
        self._queue: List[QueuedRequest] = []
        self._lock = threading.Lock()

        # Statistics
        self._total_queued = 0
        self._total_processed = 0
        self._total_dropped = 0
        self._wait_times: List[float] = []

    def should_queue(
        self,
        current_cost: float,
        budget_limit: float,
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """
        Check if request should be queued.

        Args:
            current_cost: Current total cost
            budget_limit: Budget limit
            priority: Request priority

        Returns:
            True if request should be queued

        Example:
            >>> should_queue = manager.should_queue(
            ...     current_cost=75.00,
            ...     budget_limit=100.00,
            ...     priority=Priority.NORMAL
            ... )
        """
        if self.mode == QueueMode.DISABLED:
            return False

        if budget_limit == 0:
            return False

        usage_pct = (current_cost / budget_limit) * 100

        # CRITICAL never queues
        if priority == Priority.CRITICAL:
            return False

        # Mode-based thresholds
        if self.mode == QueueMode.SOFT:
            return usage_pct >= 80.0

        elif self.mode == QueueMode.HARD:
            return usage_pct >= 90.0

        elif self.mode == QueueMode.SMART:
            # Adaptive by priority
            thresholds = {
                Priority.HIGH: 90.0,
                Priority.NORMAL: 80.0,
                Priority.LOW: 70.0,
            }
            threshold = thresholds.get(priority, 80.0)
            return usage_pct >= threshold

        return False

    def enqueue(
        self,
        request_id: str,
        callback: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        estimated_cost: float = 0.0,
    ) -> bool:
        """
        Add request to queue.

        Args:
            request_id: Unique request identifier
            callback: Callable to execute when dequeued
            args: Positional arguments for callback
            kwargs: Keyword arguments for callback
            priority: Request priority
            metadata: Optional metadata
            estimated_cost: Estimated cost of request

        Returns:
            True if enqueued, False if queue full

        Example:
            >>> success = manager.enqueue(
            ...     request_id="req_123",
            ...     callback=process_request,
            ...     args=("prompt",),
            ...     priority=Priority.HIGH,
            ...     estimated_cost=0.05
            ... )
        """
        with self._lock:
            # Check queue size
            if len(self._queue) >= self.max_size:
                self._total_dropped += 1
                return False

            # Create request
            request = QueuedRequest(
                priority=priority.value,
                timestamp=datetime.now(),
                request_id=request_id,
                callback=callback,
                args=args,
                kwargs=kwargs or {},
                metadata=metadata or {},
                estimated_cost=estimated_cost,
            )

            # Add to heap
            heapq.heappush(self._queue, request)
            self._total_queued += 1

            return True

    def dequeue(self) -> Optional[QueuedRequest]:
        """
        Remove and return highest priority request.

        Drops expired requests (exceeded max_wait_seconds).

        Returns:
            QueuedRequest or None if queue empty

        Example:
            >>> request = manager.dequeue()
            >>> if request:
            ...     result = request.callback(*request.args, **request.kwargs)
        """
        with self._lock:
            # Remove expired requests
            now = datetime.now()
            while self._queue:
                # Peek at highest priority
                request = self._queue[0]

                # Check if expired
                wait_time = (now - request.timestamp).total_seconds()
                if wait_time > self.max_wait_seconds:
                    # Drop expired request
                    heapq.heappop(self._queue)
                    self._total_dropped += 1
                    continue

                # Valid request, dequeue it
                request = heapq.heappop(self._queue)
                self._total_processed += 1

                # Record wait time
                self._wait_times.append(wait_time)
                if len(self._wait_times) > 1000:
                    self._wait_times = self._wait_times[-1000:]

                return request

            # Queue empty
            return None

    def process_queue(
        self,
        can_process: Callable[[QueuedRequest], bool],
        max_batch: int = 10,
    ) -> List[Any]:
        """
        Process queued requests in batch.

        Args:
            can_process: Callable that returns True if request can be processed
                        (e.g., check budget availability)
            max_batch: Maximum requests to process

        Returns:
            List of results from processed requests

        Example:
            >>> def can_process(req):
            ...     return current_cost + req.estimated_cost <= budget_limit
            >>> results = manager.process_queue(can_process, max_batch=5)
        """
        results = []
        processed = 0

        while processed < max_batch:
            # Peek at next request
            with self._lock:
                if not self._queue:
                    break

                request = self._queue[0]

                # Check if can process
                if not can_process(request):
                    break

                # Dequeue (will check expiry)
                request = self.dequeue()

            if not request:
                break

            # Process request
            try:
                result = request.callback(*request.args, **request.kwargs)
                results.append(result)
                processed += 1
            except Exception as e:
                # Log error but continue processing
                import logging
                logging.error(f"Queue processing error for {request.request_id}: {e}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dict with current size, totals, avg wait time, etc.

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Queue size: {stats['current_size']}")
            >>> print(f"Avg wait: {stats['avg_wait_seconds']:.1f}s")
        """
        with self._lock:
            current_size = len(self._queue)
            avg_wait = (
                sum(self._wait_times) / len(self._wait_times)
                if self._wait_times else 0.0
            )

            return {
                'current_size': current_size,
                'max_size': self.max_size,
                'mode': self.mode.value,
                'total_queued': self._total_queued,
                'total_processed': self._total_processed,
                'total_dropped': self._total_dropped,
                'avg_wait_seconds': avg_wait,
            }

    def clear(self):
        """
        Clear all queued requests.

        Example:
            >>> manager.clear()
        """
        with self._lock:
            self._queue.clear()

    def __len__(self) -> int:
        """Return current queue size."""
        with self._lock:
            return len(self._queue)
