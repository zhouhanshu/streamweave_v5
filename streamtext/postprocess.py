"""Deterministic execution repair for StreamText outputs."""

from __future__ import annotations

from streamweave.parser import parse_for_repair
from streamweave.schemas import AppliedAction, BridgeRecord, ModelAction, ModelEvent, QARecord

from .quality import TextQualityContext


FALLBACK_DELTA_TEXT = "The current frame window was observed, but no reliable text description was recovered."


def repair_for_execution(raw: str, context: TextQualityContext) -> AppliedAction:
    parsed = parse_for_repair(raw)
    action = parsed.action
    repair_types: list[str] = []
    source_deltas = [event for event in action.events if event.kind == "bridge" and event.text.strip()]
    if not source_deltas:
        text = FALLBACK_DELTA_TEXT
        repair_types.append("add_missing_delta")
    else:
        text = " ".join(delta.text.strip() for delta in source_deltas)
        if len(source_deltas) != 1:
            repair_types.append("merge_multiple_deltas")
        if any(
            abs(delta.start_time - context.step_start) > 0.11 or abs(delta.end_time - context.step_end) > 0.11
            for delta in source_deltas
        ):
            repair_types.append("normalize_delta_interval")
    if any(event.kind == "note" for event in action.events):
        repair_types.append("drop_anchor_not_allowed")

    state_text = action.state.strip()
    state_present = action.state_present
    if not state_text:
        state_text = "The current frame window is summarized into text memory."
        state_present = True
        repair_types.append("fill_empty_state")

    answer_text = action.answer.strip()
    event = ModelEvent(kind="bridge", start_time=context.step_start, end_time=context.step_end, text=text)
    bridge = BridgeRecord(start_time=context.step_start, end_time=context.step_end, text=text)
    answer = QARecord(timestamp=context.step_end, text=answer_text, role="a") if answer_text else None
    repaired_action = ModelAction(
        state=state_text,
        answer=answer_text,
        events=[event],
        raw=raw,
        state_present=state_present,
        answer_present=action.answer_present,
    )
    return AppliedAction(
        action=repaired_action,
        notes=[],
        bridges=[bridge],
        answer=answer,
        replace_open_tail=False,
        repair_count=len(repair_types),
        repair_types=repair_types,
    )


def apply_raw_action(action: ModelAction, context: TextQualityContext) -> AppliedAction:
    delta = next(event for event in action.events if event.kind == "bridge")
    text = delta.text.strip()
    bridge = BridgeRecord(start_time=context.step_start, end_time=context.step_end, text=text)
    normalized_event = ModelEvent(kind="bridge", start_time=context.step_start, end_time=context.step_end, text=text)
    normalized_action = ModelAction(
        state=action.state,
        answer=action.answer,
        events=[normalized_event],
        raw=action.raw,
        state_present=action.state_present,
        answer_present=action.answer_present,
    )
    answer_text = action.answer.strip()
    answer = QARecord(timestamp=context.step_end, text=answer_text, role="a") if answer_text else None
    return AppliedAction(action=normalized_action, notes=[], bridges=[bridge], answer=answer)
