"""Backend retry helper."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    retryable_error_patterns: list[str] | None = None
    non_retryable_error_patterns: list[str] | None = None


def run_with_retry(fn: Callable[[], T], policy: RetryPolicy) -> tuple[T, int, list[str]]:
    attempts = max(0, policy.max_retries) + 1
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt, errors
        except Exception as exc:
            if not is_retryable_exception(exc, policy):
                raise
            if attempt >= attempts:
                raise
            errors.append(repr(exc))
            sleep_seconds = max(0.0, policy.backoff_seconds) * (max(1.0, policy.backoff_multiplier) ** (attempt - 1))
            if sleep_seconds:
                time.sleep(sleep_seconds)
    raise RuntimeError("unreachable retry state")


def is_retryable_exception(exc: Exception, policy: RetryPolicy) -> bool:
    text = _exception_text(exc)
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code is not None:
        try:
            status_int = int(status_code)
        except (TypeError, ValueError):
            status_int = 0
        if status_int in {400, 401, 403}:
            return False
        if status_int in {429, 500, 502, 503, 504}:
            return True
    for pattern in policy.non_retryable_error_patterns or []:
        if pattern and pattern in text:
            return False
    for pattern in policy.retryable_error_patterns or []:
        if pattern and pattern in text:
            return True
    return False


def _exception_text(exc: Exception) -> str:
    parts = [repr(exc), str(exc), type(exc).__name__]
    response = getattr(exc, "response", None)
    if response is not None:
        parts.append(str(getattr(response, "status_code", "")))
        parts.append(str(getattr(response, "text", "")))
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    return "\n".join(parts)
