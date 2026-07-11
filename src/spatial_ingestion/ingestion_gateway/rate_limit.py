from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int


class RateLimiter:
    """Small in-memory stub with the same interface a distributed limiter would expose."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    async def check(self, subject: str) -> RateLimitResult:
        now = monotonic()
        events = self._events[subject]
        while events and now - events[0] > self._window_seconds:
            events.popleft()

        if len(events) >= self._max_requests:
            return RateLimitResult(allowed=False, remaining=0)

        events.append(now)
        return RateLimitResult(allowed=True, remaining=self._max_requests - len(events))

