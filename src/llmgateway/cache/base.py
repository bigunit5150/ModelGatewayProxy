"""Cache primitives: CacheEntry dataclass and the CacheBackend Protocol.

All async cache backends must implement :class:`CacheBackend`.  The
:class:`CacheEntry` dataclass is the unit of storage passed between the
manager and the backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class CacheEntry:
    """A single cached response entry.

    Attributes:
        key:        SHA-256 hex digest used as the cache key.
        value:      JSON-serialised response payload (str).
        created_at: Unix timestamp of when the entry was created.
        ttl:        Time-to-live in seconds; the backend enforces expiry.
        metadata:   Arbitrary key/value pairs (e.g. ``{"model": "gpt-4o"}``).
    """

    key: str
    value: str  # JSON-serialised response payload
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return ``True`` if the entry has passed its TTL."""
        return (time.time() - self.created_at) >= self.ttl


@runtime_checkable
class CacheBackend(Protocol):
    """Async cache backend protocol.

    All methods must be safe to call concurrently.  Implementations should
    swallow transient errors and return ``None`` / ``False`` rather than
    raising, so that callers can treat cache failures as cache misses
    (fail-open semantics).
    """

    async def get(self, key: str) -> CacheEntry | None:
        """Return the entry for *key*, or ``None`` if absent / expired."""
        ...

    async def set(self, entry: CacheEntry) -> None:
        """Store *entry*, overwriting any existing value for the same key."""
        ...

    async def delete(self, key: str) -> None:
        """Remove the entry for *key* (no-op if absent)."""
        ...

    async def exists(self, key: str) -> bool:
        """Return ``True`` if a non-expired entry exists for *key*."""
        ...
