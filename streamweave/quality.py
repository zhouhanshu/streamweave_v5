"""Raw output quality and reward features."""

from __future__ import annotations

from dataclasses import dataclass

from .parser import strict_validate_raw_output
from .schemas import BridgeRecord, FrameRef, ModelAction, ModelEvent, QualityReport, RewardFeatures, ValidationIssue

TIME_TOLERANCE = 0.51
BRIDGE_GAP_TOLERANCE = 1e-6
BRIDGE_INTERVAL_TOLERANCE = 0.11


@dataclass(slots=True)
class QualityContext:
    frames: list[FrameRef]
    step_start: float
    step_end: float
    open_tail_bridge: BridgeRecord | None = None


def score_raw_output(raw: str, context: QualityContext, reward_config: object | None = None) -> tuple[ModelAction, QualityReport]:
    result = strict_validate_raw_output(raw)
    action = result.action
    issues = list(result.issues)
    format_reward = int(result.parser_ok)
    timing_reward = 1
    order_reward = 1
    opentail_reward = 1

    previous_end = context.step_start
    for event_index, event in enumerate(action.events):
        extends_open_tail = False
        if event.kind == "note":
            if event.frame_index is None or not (0 <= event.frame_index < len(context.frames)):
                issues.append(ValidationIssue("note_frame_oob", "Note frame index is outside current frames."))
                timing_reward = 0
                continue
            frame = context.frames[event.frame_index]
            if abs(event.start_time - frame.start_time) > TIME_TOLERANCE or abs(event.end_time - frame.end_time) > TIME_TOLERANCE:
                issues.append(ValidationIssue("note_time_mismatch", "Note time does not match referenced frame."))
                timing_reward = 0
        elif event.kind == "bridge":
            if not event.text.strip():
                issues.append(ValidationIssue("empty_bridge", "Bridge text is empty."))
                timing_reward = 0
            if context.open_tail_bridge is not None and event_index == 0:
                extends_open_tail = abs(event.start_time - context.open_tail_bridge.start_time) <= TIME_TOLERANCE
                if not extends_open_tail:
                    issues.append(ValidationIssue("open_tail_start_mismatch", "First bridge does not inherit open-tail start."))
                    opentail_reward = 0
            if not extends_open_tail:
                if event.start_time < context.step_start - TIME_TOLERANCE or event.end_time > context.step_end + TIME_TOLERANCE:
                    issues.append(ValidationIssue("bridge_time_oob", "Bridge time is outside current step."))
                    timing_reward = 0
            elif event.end_time > context.step_end + TIME_TOLERANCE:
                issues.append(ValidationIssue("open_tail_end_oob", "Open-tail bridge end is outside current step."))
                opentail_reward = 0

        if not extends_open_tail and event.start_time < previous_end - TIME_TOLERANCE:
            issues.append(ValidationIssue("event_overlap", "Event overlaps or is out of chronological order."))
            order_reward = 0
            timing_reward = 0
        previous_end = max(previous_end, event.end_time)

    if context.open_tail_bridge is not None and action.events and action.events[0].kind != "bridge":
        issues.append(ValidationIssue("missing_open_tail_bridge", "Open-tail memory requires first event to be a bridge."))
        opentail_reward = 0

    gap_issues, gap_metrics = _bridge_gap_issues(action, context)
    if gap_issues:
        issues.extend(gap_issues)
        format_reward = 0

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
        "num_events": len(action.events),
        "num_notes": sum(1 for event in action.events if event.kind == "note"),
        "num_bridges": sum(1 for event in action.events if event.kind == "bridge"),
        **gap_metrics,
    }
    return action, QualityReport(valid=valid, parser_ok=result.parser_ok, issues=issues, rewards=rewards, metrics=metrics)


def _bridge_gap_issues(action: ModelAction, context: QualityContext) -> tuple[list[ValidationIssue], dict[str, object]]:
    notes = sorted((event for event in action.events if event.kind == "note"), key=lambda item: (item.start_time, item.end_time))
    bridges = [event for event in action.events if event.kind == "bridge"]
    expected = _expected_bridge_gaps(notes, context)
    issues: list[ValidationIssue] = []
    matched_bridge_indexes: set[int] = set()
    duplicate_indexes: set[int] = set()

    for gap in expected:
        matches = [
            bridge_index
            for bridge_index, bridge in enumerate(bridges)
            if _same_interval(bridge.start_time, bridge.end_time, gap[0], gap[1])
        ]
        if not matches:
            issues.append(
                ValidationIssue(
                    "missing_bridge_gap",
                    f"Missing bridge for required gap {gap[0]:.1f}-{gap[1]:.1f}.",
                )
            )
            continue
        matched_bridge_indexes.add(matches[0])
        duplicate_indexes.update(matches[1:])

    for bridge_index, bridge in enumerate(bridges):
        if bridge_index in matched_bridge_indexes:
            continue
        if bridge_index in duplicate_indexes:
            issues.append(
                ValidationIssue(
                    "duplicate_bridge_gap",
                    f"Multiple bridges cover required gap {bridge.start_time:.1f}-{bridge.end_time:.1f}.",
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    "invalid_bridge_gap",
                    f"Bridge interval {bridge.start_time:.1f}-{bridge.end_time:.1f} does not match any required note/window gap.",
                )
            )

    metrics = {
        "num_expected_bridges": len(expected),
        "expected_bridge_gaps": [[start, end] for start, end in expected],
        "output_bridge_gaps": [[bridge.start_time, bridge.end_time] for bridge in bridges],
    }
    return issues, metrics


def _expected_bridge_gaps(notes: list[ModelEvent], context: QualityContext) -> list[tuple[float, float]]:
    if not notes:
        start = context.open_tail_bridge.start_time if context.open_tail_bridge is not None else context.step_start
        return [(start, context.step_end)] if context.step_end > start + BRIDGE_GAP_TOLERANCE else []

    gaps: list[tuple[float, float]] = []
    first_note = notes[0]
    if context.open_tail_bridge is not None:
        if first_note.start_time > context.open_tail_bridge.start_time + BRIDGE_GAP_TOLERANCE:
            gaps.append((context.open_tail_bridge.start_time, first_note.start_time))
    elif first_note.start_time > context.step_start + BRIDGE_GAP_TOLERANCE:
        gaps.append((context.step_start, first_note.start_time))

    previous_end = first_note.end_time
    for note in notes[1:]:
        if note.start_time > previous_end + BRIDGE_GAP_TOLERANCE:
            gaps.append((previous_end, note.start_time))
        previous_end = note.end_time

    if context.step_end > previous_end + BRIDGE_GAP_TOLERANCE:
        gaps.append((previous_end, context.step_end))
    return gaps


def _same_interval(start: float, end: float, expected_start: float, expected_end: float) -> bool:
    return (
        abs(start - expected_start) <= BRIDGE_INTERVAL_TOLERANCE
        and abs(end - expected_end) <= BRIDGE_INTERVAL_TOLERANCE
    )


def _enabled(config: object | None, name: str) -> bool:
    if config is None:
        return True
    return bool(getattr(config, name, True))
