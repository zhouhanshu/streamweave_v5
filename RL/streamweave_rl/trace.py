"""Small helpers for opt-in StreamWeave RL debug tracing."""

from __future__ import annotations

import os
from typing import Any


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, *, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def shorten(text: str, *, limit_env: str = "STREAMWEAVE_TRACE_MAX_CHARS", default_limit: int = 2000) -> str:
    limit = max(env_int(limit_env, default=default_limit), 1)
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value) - limit} chars>"


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


def trace_print(message: str) -> None:
    print(message, flush=True)
