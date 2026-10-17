from __future__ import annotations

import copy
import os
import threading
import time
from typing import Any


def cache_ttl_seconds() -> float:
    """Read the cache TTL from `KINESIS_CACHE_TTL_SECONDS` (0 or negative disables caching)."""
    raw = os.getenv("KINESIS_CACHE_TTL_SECONDS", "").strip()
    if not raw:
        return 60.0
    try:
        return float(raw)
    except ValueError:
        return 60.0


class TTLCache:
    """Small thread-safe TTL cache for read-only API responses.

    Repeated tool calls and multi-turn `chat` sessions frequently re-request the same
    intervals.icu or PubMed data. Caching those reads for a short window removes the
    network round trip without serving stale data for long. Caching is disabled when the
    effective TTL is <= 0.
    """

    def __init__(self, ttl: float | None = None, max_entries: int = 512) -> None:
        self.ttl = cache_ttl_seconds() if ttl is None else ttl
        self.max_entries = max_entries
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def get(self, key: Any) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            with self._lock:
                self._store.pop(key, None)
            return None
        # Deep-copy so callers can freely mutate results without poisoning the cache.
        return copy.deepcopy(value)

    def set(self, key: Any, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            if len(self._store) >= self.max_entries:
                self._store.pop(next(iter(self._store)), None)
            self._store[key] = (time.monotonic() + self.ttl, copy.deepcopy(value))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
