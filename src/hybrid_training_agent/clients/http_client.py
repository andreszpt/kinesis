from __future__ import annotations

import threading

import httpx

_shared_client: httpx.Client | None = None
_shared_client_lock = threading.Lock()


def get_shared_http_client(timeout: float = 30.0) -> httpx.Client:
    """Return a process-wide `httpx.Client` so TCP/TLS connections are reused.

    Previously every API request opened a brand new client (and therefore a new TLS
    handshake) to intervals.icu or NCBI. Reusing one pooled client across tool calls and
    sub-agents removes those handshakes from the critical path.
    """
    global _shared_client
    with _shared_client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            )
        return _shared_client


def reset_shared_http_client() -> None:
    """Close and drop the shared client (used by tests)."""
    global _shared_client
    with _shared_client_lock:
        if _shared_client is not None and not _shared_client.is_closed:
            _shared_client.close()
        _shared_client = None
