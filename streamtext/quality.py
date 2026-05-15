"""Text-memory XML quality checks for StreamText."""

from __future__ import annotations

from dataclasses import dataclass

from streamweave.parser import strict_validate_raw_output
from streamweave.schemas import FrameRef, ModelAction, QualityReport, RewardFeatures, ValidationIssue


INTERVAL_TOLERANCE = 0.11


@dataclass(slots=True)
class TextQualityContext:
    frames: list[FrameRef]
    step_start: float
    step_end: float


def score_raw_output(raw: str, context: TextQualityContext, reward_config: object | None = None) -> tuple[ModelAction, QualityReport]:
    parsed = strict_validate_raw_output(raw)
    action = parsed.action
    issues = list(parsed.issues)

    format_reward = int(parsed.parser_ok)
    timing_reward = 1
    order_reward = 1
    opentail_reward = 1

    notes = [event for event in action.events if event.kind == "note"]
    deltas = [event for event in action.events if event.kind == "bridge"]
    if notes:
        issues.append(ValidationIssue("anchor_not_allowed", "StreamText memory is text-only; <anchor> is not allowed."))
        format_reward = 0
    if len(deltas) != 1:
        issues.append(
            ValidationIssue(
                "delta_count",
                f"StreamText requires exactly one <delta> per step, got {len(deltas)}.",
            )
        )
        format_reward = 0

    for delta in deltas:
        if not delta.text.strip():
            issues.append(ValidationIssue("empty_delta", "StreamText <delta> text must be non-empty."))
            timing_reward = 0
        if not _same_interval(delta.start_time, delta.end_time, context.step_start, context.step_end):
            issues.append(
                ValidationIssue(
                    "delta_time_mismatch",
                    (
                        f'StreamText <delta> must cover the current window '
                        f'{context.step_start:.1f}-{context.step_end:.1f}.'
                    ),
                )
            )
            timing_reward = 0
            order_reward = 0

    format_enabled = _enabled(reward_config, "enable_format_reward")
    timing_enabled = _enabled(reward_config, "enable_timing_reward")
    open_tail_enabled = _enabled(reward_config, "enable_open_tail_reward")
    rewards = RewardFeatures(
        format_reward=format_reward if format_enabled else 1,
        opentail_bridge_reward=opentail_reward if open_tail_enabled else 1,
        note_bridge_timing_reward=timing_reward if timing_enabled else 1,
        operation_order_reward=order_reward if timing_enabled else 1,
    )
    valid = all(
        value == 1
        for enabled, value in (
            (format_enabled, format_reward),
            (open_tail_enabled, opentail_reward),
            (timing_enabled, timing_reward),
            (timing_enabled, order_reward),
        )
        if enabled
    )
    metrics = {
        "state_length": len(action.state.strip()),
        "state_present": action.state_present,
        "num_events": len(action.events),
        "num_notes": len(notes),
        "num_bridges": len(deltas),
        "expected_delta_interval": [context.step_start, context.step_end],
    }
    return action, QualityReport(valid=valid, parser_ok=parsed.parser_ok, issues=issues, rewards=rewards, metrics=metrics)


def _same_interval(start: float, end: float, expected_start: float, expected_end: float) -> bool:
    return abs(start - expected_start) <= INTERVAL_TOLERANCE and abs(end - expected_end) <= INTERVAL_TOLERANCE


def _enabled(config: object | None, name: str) -> bool:
    if config is None:
        return True
    return bool(getattr(config, name, True))

