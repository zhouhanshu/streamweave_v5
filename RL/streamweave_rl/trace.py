"""Small helpers for opt-in StreamWeave RL debug tracing."""

from __future__ import annotations

import os
import zlib
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


def env_text(name: str, *, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def trace_rollout_allowed(*, group_idx: Any, traj_idx: int, sample_id: str, sample_index: Any = None) -> bool:
    if not env_flag("STREAMWEAVE_TRACE_FIRST_ROLLOUT", default=False):
        return False
    if int(traj_idx) != env_int("STREAMWEAVE_TRACE_TRAJ_INDEX", default=0):
        return False

    sample_filter = env_text("STREAMWEAVE_TRACE_SAMPLE_ID")
    if sample_filter and str(sample_id) != sample_filter:
        return False
    group_filter = env_text("STREAMWEAVE_TRACE_GROUP_IDX")
    if group_filter and str(group_idx) != group_filter:
        return False

    return _sampled_for_trace(sample_index if sample_index is not None else sample_id)


def trace_group_allowed(group_idx: Any) -> bool:
    group_filter = env_text("STREAMWEAVE_TRACE_GROUP_IDX")
    if group_filter:
        return str(group_idx) == group_filter
    return _sampled_for_trace(group_idx)


def _sampled_for_trace(value: Any) -> bool:
    every = env_int("STREAMWEAVE_TRACE_SAMPLE_EVERY", default=1)
    if every <= 1:
        return True
    offset = env_int("STREAMWEAVE_TRACE_SAMPLE_OFFSET", default=0) % every
    bucket = _trace_bucket(value)
    return bucket % every == offset


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


def _trace_bucket(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return zlib.crc32(str(value).encode("utf-8"))
