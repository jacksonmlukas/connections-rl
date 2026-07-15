"""In-memory request monitor: latency percentiles + counters for /metrics."""

from __future__ import annotations

import threading
from collections import Counter, deque


class RequestMonitor:
    def __init__(self, buffer_size: int = 5000):
        self._latencies: deque[float] = deque(maxlen=buffer_size)
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(self, endpoint: str, latency_ms: float, ok: bool = True) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            self._counts[endpoint] += 1
            if not ok:
                self._counts[f"{endpoint}:errors"] += 1

    def summary(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            counts = dict(self._counts)

        def pct(q: float) -> float:
            if not lat:
                return 0.0
            return lat[min(len(lat) - 1, int(q * len(lat)))]

        return {
            "requests": counts,
            "latency_ms": {
                "n": len(lat),
                "p50": pct(0.50),
                "p95": pct(0.95),
                "p99": pct(0.99),
                "max": lat[-1] if lat else 0.0,
            },
        }
