"""Endpoint pool for OpenAI-compatible local/API backends."""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence

from streamweave.config import BackendConfig
from streamweave.schemas import BackendResult, ContentItem

from .base import BaseBackend


class RoundRobinEndpointSelector:
    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Endpoint selector requires at least one endpoint.")
        self.size = size
        self._next = 0
        self._lock = threading.Lock()

    def order(self) -> list[int]:
        with self._lock:
            start = self._next
            self._next = (self._next + 1) % self.size
        return [(start + offset) % self.size for offset in range(self.size)]


class BackendPool(BaseBackend):
    """Round-robin over endpoint-specific backends in a single process."""

    def __init__(self, backends: Sequence[BaseBackend], *, config: BackendConfig | None = None) -> None:
        if not backends:
            raise ValueError("BackendPool requires at least one backend.")
        self.backends = list(backends)
        self.config = config
        self.selector = RoundRobinEndpointSelector(len(self.backends))

    def generate(
        self,
        content: list[ContentItem],
        *,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        started = time.time()
        errors: list[str] = []
        for index in self.selector.order():
            backend = self.backends[index]
            try:
                result = backend.generate(content, generate_kwargs=generate_kwargs)
                if errors:
                    result.retry_errors = errors + list(result.retry_errors)
                    result.attempt_count += len(errors)
                    result.latency_seconds = time.time() - started
                return result
            except Exception as exc:
                error = repr(exc)
                errors.append(f"endpoint_index={index}: {error}")
                if self._is_non_retryable(error):
                    raise
        raise RuntimeError("All backend endpoints failed: " + " | ".join(errors))

    def _is_non_retryable(self, error: str) -> bool:
        if self.config is None:
            return False
        return any(pattern and pattern in error for pattern in self.config.non_retryable_error_patterns)
