"""Rollout-level metrics for per-sample results and batch summaries."""

from __future__ import annotations

from typing import Any


REWARD_KEYS = (
    "format_reward",
    "opentail_bridge_reward",
    "note_bridge_timing_reward",
    "operation_order_reward",
)


def rollout_metrics_from_trace(trace: Any) -> dict[str, Any]:
    transitions = list(getattr(trace, "transitions", []) or [])
    reward_sums = {key: 0 for key in REWARD_KEYS}
    repair_types: dict[str, int] = {}
    backend_endpoints: set[str] = set()
    backend_retry_errors: list[str] = []
    total_latency_seconds = 0.0
    model_call_count = 0
    backend_attempt_count = 0
    backend_retry_count = 0
    num_notes = 0
    num_bridges = 0
    repair_count = 0

    for transition in transitions:
        backend_results = _transition_backend_results(transition)
        model_call_count += len(backend_results)
        for backend_result in backend_results:
            total_latency_seconds += float(getattr(backend_result, "latency_seconds", 0.0) or 0.0)
            attempts = int(getattr(backend_result, "attempt_count", 1) or 1)
            backend_attempt_count += attempts
            backend_retry_count += max(attempts - 1, 0)
            endpoint = str(getattr(backend_result, "endpoint_id", "") or "")
            if endpoint:
                backend_endpoints.add(endpoint)
            backend_retry_errors.extend(str(error) for error in (getattr(backend_result, "retry_errors", []) or []))

        applied = getattr(transition, "applied", None)
        if applied is not None:
            num_notes += len(getattr(applied, "notes", []) or [])
            num_bridges += len(getattr(applied, "bridges", []) or [])
            count = int(getattr(applied, "repair_count", 0) or 0)
            repair_count += count
            for name in getattr(applied, "repair_types", []) or []:
                repair_types[str(name)] = repair_types.get(str(name), 0) + 1

        rewards = getattr(getattr(transition, "quality", None), "rewards", None)
        for key in REWARD_KEYS:
            reward_sums[key] += int(getattr(rewards, key, 0) or 0)

    num_steps = len(transitions)
    return {
        "num_steps": num_steps,
        "num_notes": num_notes,
        "num_bridges": num_bridges,
        "total_latency_seconds": total_latency_seconds,
        "model_call_count": model_call_count,
        "backend_attempt_count": backend_attempt_count,
        "backend_retry_count": backend_retry_count,
        "backend_retry_error_count": len(backend_retry_errors),
        "backend_retry_errors": backend_retry_errors,
        "backend_endpoints": sorted(backend_endpoints),
        "repair_count": repair_count,
        "repair_types": dict(sorted(repair_types.items())),
        "reward_sums": reward_sums,
        "reward_means": _reward_means(reward_sums, num_steps),
    }


def summarize_rollout_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    reward_sums = {key: 0 for key in REWARD_KEYS}
    repair_types: dict[str, int] = {}
    backend_endpoints: set[str] = set()
    backend_retry_error_examples: list[str] = []

    num_steps = 0
    num_notes = 0
    num_bridges = 0
    total_latency_seconds = 0.0
    model_call_count = 0
    backend_attempt_count = 0
    backend_retry_count = 0
    backend_retry_error_count = 0
    repair_count = 0

    for row in results:
        steps = int(row.get("num_steps", 0) or 0)
        num_steps += steps
        num_notes += int(row.get("num_notes", 0) or 0)
        num_bridges += int(row.get("num_bridges", 0) or 0)
        total_latency_seconds += float(row.get("total_latency_seconds", 0.0) or 0.0)
        model_call_count += int(row.get("model_call_count", 0) or 0)
        backend_attempt_count += int(row.get("backend_attempt_count", 0) or 0)
        backend_retry_count += int(row.get("backend_retry_count", 0) or 0)
        backend_retry_error_count += int(row.get("backend_retry_error_count", 0) or 0)
        repair_count += int(row.get("repair_count", 0) or 0)

        for endpoint in row.get("backend_endpoints", []) or []:
            endpoint_text = str(endpoint)
            if endpoint_text:
                backend_endpoints.add(endpoint_text)
        for error in row.get("backend_retry_errors", []) or []:
            if len(backend_retry_error_examples) < 20:
                backend_retry_error_examples.append(str(error))

        for name, count in (row.get("repair_types", {}) or {}).items():
            repair_types[str(name)] = repair_types.get(str(name), 0) + int(count or 0)
        row_reward_sums = row.get("reward_sums", {}) or {}
        for key in REWARD_KEYS:
            reward_sums[key] += int(row_reward_sums.get(key, 0) or 0)

    return {
        "num_samples": len(results),
        "num_steps": num_steps,
        "num_notes": num_notes,
        "num_bridges": num_bridges,
        "total_latency_seconds": total_latency_seconds,
        "avg_latency_seconds_per_call": (total_latency_seconds / model_call_count) if model_call_count else None,
        "model_call_count": model_call_count,
        "backend_attempt_count": backend_attempt_count,
        "backend_retry_count": backend_retry_count,
        "backend_retry_error_count": backend_retry_error_count,
        "backend_retry_error_examples": backend_retry_error_examples,
        "backend_endpoints": sorted(backend_endpoints),
        "repair_count": repair_count,
        "repair_types": dict(sorted(repair_types.items())),
        "reward_sums": reward_sums,
        "reward_means": _reward_means(reward_sums, num_steps),
    }


def _transition_backend_results(transition: Any) -> list[Any]:
    attempts = list(getattr(transition, "attempts", []) or [])
    results = [getattr(attempt, "backend_result", None) for attempt in attempts]
    results = [result for result in results if result is not None]
    if results:
        return results
    backend_result = getattr(transition, "backend_result", None)
    return [backend_result] if backend_result is not None else []


def _reward_means(reward_sums: dict[str, int], num_steps: int) -> dict[str, float | None]:
    if not num_steps:
        return {key: None for key in REWARD_KEYS}
    return {key: reward_sums[key] / num_steps for key in REWARD_KEYS}
